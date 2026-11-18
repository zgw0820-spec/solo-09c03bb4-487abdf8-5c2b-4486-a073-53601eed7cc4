"""Job and checkpoint state machine.

Every state change is one ``BEGIN IMMEDIATE`` transaction in the shared
database.  A batch commit writes, together and atomically:

* the checkpoint row: step, weights, momentum, batch loss, dataset loss,
  commit time, worker and lease epoch;
* the job row: current weights/momentum/loss, ``completed_steps`` and status;
* the lease heartbeat, proving the committer is still the live owner;
* the audit event for a completion.

The append-only checkpoint stream is the source of truth.  A restart always
reloads the newest checkpoint, and progress is only advanced by one inside the
write lock, so no reader sees a step whose weights and progress disagree.
"""
from __future__ import annotations

import json
import math
import time
import uuid
from typing import Any

import numpy as np

from .db import reader, transaction
from .features import (
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    MIN_EPOCHS,
    MOMENTUM,
    InputError,
    normalise,
)
from .lease import Lease, LeaseLost, owner_ok
from .model import dataset_loss, sgd_step

JOB_STATUSES = ("queued", "running", "completed")
ERASED_STATUS = "erased"


class JobNotFound(LookupError):
    pass


class JobErased(LookupError):
    """The job was erased: its training data is gone and must stay gone."""


class StateConflict(RuntimeError):
    """The database changed in a way that should be impossible."""


def _dumps(values: np.ndarray) -> str:
    return json.dumps([float(v) for v in values], allow_nan=False)


def _loads(raw: str) -> np.ndarray:
    return np.asarray(json.loads(raw), dtype=np.float64)


def _job_row(conn, job_id: str):
    return conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()


def _event(conn, job_id: str, event: str, detail: dict[str, Any] | None = None) -> None:
    conn.execute(
        "INSERT INTO job_events(job_id, event, detail, created_at) VALUES(?, ?, ?, ?)",
        (job_id, event, json.dumps(detail or {}), time.time()),
    )


def _tombstone(conn, job_id: str):
    return conn.execute(
        "SELECT job_id, status, erased_at FROM erased_jobs WHERE job_id=?", (job_id,)
    ).fetchone()


def _tombstone_dict(row) -> dict[str, Any]:
    return {
        "job_id": row["job_id"],
        "status": row["status"],
        "erased_at": float(row["erased_at"]),
    }


def get_tombstone(db_path, job_id: str) -> dict[str, Any] | None:
    """Return the erasure record for ``job_id``, or ``None`` when not erased."""
    with reader(db_path) as conn:
        row = _tombstone(conn, job_id)
        return _tombstone_dict(row) if row is not None else None


def erase_job(db_path, job_id: str) -> dict[str, Any]:
    """Erase every piece of training data for one job, atomically.

    The job's samples (``features_json``/``labels_json``), current weights and
    momentum, all checkpoints, all audit events and its lease are deleted in a
    single ``BEGIN IMMEDIATE`` transaction and replaced by one tombstone row
    holding only the job id and the erasure status.  Any error rolls the whole
    transaction back, so a failed erase can neither report success nor leave a
    half-deleted job.

    The operation is idempotent: erasing an already-erased job returns the same
    tombstone.  The monotonic ``lease_epochs`` counter is deliberately retained
    - it carries no training data, and keeping it means a worker that still
    holds a stale lease can never be mistaken for the owner of a freshly
    recreated id.
    """
    now = time.time()
    with transaction(db_path) as conn:
        existing = _tombstone(conn, job_id)
        if existing is not None:
            return _tombstone_dict(existing)
        if _job_row(conn, job_id) is None:
            raise JobNotFound(job_id)
        # Order matters with foreign_keys=ON: checkpoints reference the job row,
        # so they must be removed before the job itself.
        conn.execute("DELETE FROM checkpoints WHERE job_id=?", (job_id,))
        conn.execute("DELETE FROM job_events WHERE job_id=?", (job_id,))
        conn.execute("DELETE FROM leases WHERE job_id=?", (job_id,))
        conn.execute("DELETE FROM jobs WHERE job_id=?", (job_id,))
        conn.execute(
            "INSERT INTO erased_jobs(job_id, status, erased_at) VALUES(?, ?, ?)",
            (job_id, ERASED_STATUS, now),
        )
    return {"job_id": job_id, "status": ERASED_STATUS, "erased_at": now}


def create_job(
    db_path,
    features: list[list[float]],
    labels: list[int],
    epochs: int,
) -> dict[str, Any]:
    """Validate, normalise and enqueue a training job."""
    if not (MIN_EPOCHS <= epochs <= MAX_EPOCHS):
        raise InputError(f"epochs must be between {MIN_EPOCHS} and {MAX_EPOCHS}")
    design, targets = normalise(features, labels)
    feature_dim = int(design.shape[1])
    per_epoch = math.ceil(design.shape[0] / BATCH_SIZE)
    total_steps = per_epoch * epochs
    weights = np.zeros(feature_dim, dtype=np.float64)
    velocity = np.zeros(feature_dim, dtype=np.float64)
    initial_loss = dataset_loss(weights, design, targets)

    job_id = uuid.uuid4().hex
    now = time.time()
    with transaction(db_path) as conn:
        conn.execute(
            """
            INSERT INTO jobs(
                job_id, status, feature_dim, epochs, batch_size, record_count,
                total_batches, learning_rate, momentum, features_json, labels_json,
                weights_json, velocity_json, loss, completed_steps, created_at, updated_at
            ) VALUES(?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (
                job_id,
                feature_dim,
                epochs,
                BATCH_SIZE,
                int(design.shape[0]),
                per_epoch,
                LEARNING_RATE,
                MOMENTUM,
                json.dumps(design.tolist()),
                json.dumps([float(v) for v in targets], allow_nan=False),
                _dumps(weights),
                _dumps(velocity),
                initial_loss,
                now,
                now,
            ),
        )
        _event(
            conn,
            job_id,
            "job_created",
            {
                "records": int(design.shape[0]),
                "epochs": epochs,
                "batches_per_epoch": per_epoch,
                "total_steps": total_steps,
            },
        )
    job = get_job(db_path, job_id)
    assert job is not None
    return job


def get_job(db_path, job_id: str) -> dict[str, Any] | None:
    """Read a consistent snapshot of the latest confirmed state.

    The job row and its newest checkpoint are read in one transaction.  Their
    step and weights must agree; a mismatch would mean a torn commit.

    An erased job has no training data left: only the tombstone (job id and
    erasure status) is returned, so samples, weights and events can never be
    read again through this path.
    """
    with reader(db_path) as conn:
        tombstone = _tombstone(conn, job_id)
        if tombstone is not None:
            return _tombstone_dict(tombstone)
        row = _job_row(conn, job_id)
        if row is None:
            return None
        checkpoint = conn.execute(
            "SELECT step, weights_json, velocity_json, loss FROM checkpoints "
            "WHERE job_id=? ORDER BY step DESC LIMIT 1",
            (job_id,),
        ).fetchone()
        if int(row["completed_steps"]) != 0:
            if checkpoint is None or int(checkpoint["step"]) != int(row["completed_steps"]):
                raise StateConflict("latest checkpoint does not match confirmed progress")
            if checkpoint["weights_json"] != row["weights_json"]:
                raise StateConflict("latest checkpoint weights do not match job weights")
            if checkpoint["velocity_json"] != row["velocity_json"]:
                raise StateConflict("latest checkpoint momentum does not match job momentum")
        design = np.asarray(json.loads(row["features_json"]), dtype=np.float64)
        targets = np.asarray(json.loads(row["labels_json"]), dtype=np.float64)
        evaluation_loss = dataset_loss(_loads(row["weights_json"]), design, targets)

    completed = int(row["completed_steps"])
    total = int(row["total_batches"]) * int(row["epochs"])
    latest_step = int(checkpoint["step"]) if checkpoint is not None else 0
    return {
        "job_id": row["job_id"],
        "status": row["status"],
        "feature_dim": int(row["feature_dim"]),
        "record_count": int(row["record_count"]),
        "epochs": int(row["epochs"]),
        "batch_size": int(row["batch_size"]),
        "batches_per_epoch": int(row["total_batches"]),
        "total_steps": total,
        "completed_steps": completed,
        "latest_checkpoint_step": latest_step,
        "progress": round(completed / total, 6) if total else 0.0,
        "learning_rate": float(row["learning_rate"]),
        "momentum": float(row["momentum"]),
        "loss": float(row["loss"]),
        "evaluation_loss": evaluation_loss,
        "weights": _loads(row["weights_json"]).tolist(),
        "momentum_velocity": _loads(row["velocity_json"]).tolist(),
        "final_weights": (
            _loads(row["weights_json"]).tolist() if row["status"] == "completed" else None
        ),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "finished_at": row["finished_at"],
    }


def list_jobs(db_path, limit: int = 50) -> list[dict[str, Any]]:
    with reader(db_path) as conn:
        rows = conn.execute(
            """
            SELECT job_id, created_at AS sort_time FROM jobs
            UNION ALL
            SELECT job_id, erased_at AS sort_time FROM erased_jobs
            ORDER BY sort_time DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    jobs = []
    for row in rows:
        job = get_job(db_path, row["job_id"])
        if job is not None:
            jobs.append(job)
    return jobs


def list_checkpoints(
    db_path, job_id: str, after_step: int = 0, limit: int = 500
) -> list[dict[str, Any]]:
    with reader(db_path) as conn:
        if _tombstone(conn, job_id) is not None:
            raise JobErased(job_id)
        job = _job_row(conn, job_id)
        if job is None:
            raise JobNotFound(job_id)
        rows = conn.execute(
            "SELECT * FROM checkpoints WHERE job_id=? AND step>? ORDER BY step LIMIT ?",
            (job_id, after_step, limit),
        ).fetchall()
        per_epoch = max(1, int(job["total_batches"]))
        total_steps = per_epoch * int(job["epochs"])
    return [
        {
            "step": int(row["step"]),
            "total_steps": total_steps,
            "progress": round(int(row["step"]) / total_steps, 6),
            "training_epoch": (int(row["step"]) - 1) // per_epoch + 1,
            "batch_in_epoch": (int(row["step"]) - 1) % per_epoch + 1,
            "batch_start_index": ((int(row["step"]) - 1) % per_epoch) * BATCH_SIZE,
            "weights": _loads(row["weights_json"]).tolist(),
            "momentum_velocity": _loads(row["velocity_json"]).tolist(),
            "batch_loss": float(row["batch_loss"]),
            "loss": float(row["loss"]),
            "committed_at": row["committed_at"],
            "worker_id": row["worker_id"],
            "lease_epoch": int(row["lease_epoch"]),
        }
        for row in rows
    ]


def list_events(db_path, job_id: str, limit: int = 200) -> list[dict[str, Any]]:
    with reader(db_path) as conn:
        if _tombstone(conn, job_id) is not None:
            raise JobErased(job_id)
        if _job_row(conn, job_id) is None:
            raise JobNotFound(job_id)
        rows = conn.execute(
            "SELECT id, event, detail, created_at FROM job_events "
            "WHERE job_id=? ORDER BY id LIMIT ?",
            (job_id, limit),
        ).fetchall()
    return [
        {
            "seq": int(row["id"]),
            "event": row["event"],
            "detail": json.loads(row["detail"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def claimable_jobs(db_path, limit: int = 20) -> list[str]:
    with reader(db_path) as conn:
        rows = conn.execute(
            """
            SELECT job_id FROM jobs
             WHERE status IN ('queued','running')
               AND completed_steps < total_batches * epochs
             ORDER BY created_at LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [row["job_id"] for row in rows]


def advance_job(db_path, lease: Lease, batch_delay: float = 0.0) -> dict[str, Any]:
    """Commit exactly one batch for ``lease``'s job.

    The newest committed checkpoint is the only source of truth.  The write is
    conditional on the fence token, so a worker that lost its lease - or that
    wakes up late after being suspended - raises :class:`LeaseLost` instead of
    writing, and can never move progress backwards.
    """
    if batch_delay > 0:
        time.sleep(batch_delay)
    with transaction(db_path) as conn:
        if not owner_ok(conn, lease):
            raise LeaseLost(
                f"job {lease.job_id} is no longer owned by {lease.worker_id}@{lease.epoch}"
            )
        # Erasure deletes the lease (so owner_ok above already fences a stale
        # holder); the tombstone check makes the refusal explicit and covers a
        # race where a lease row was recreated after erasure.
        if _tombstone(conn, lease.job_id) is not None:
            raise JobErased(lease.job_id)

        row = _job_row(conn, lease.job_id)
        if row is None:
            raise JobNotFound(lease.job_id)
        if row["status"] == "completed":
            return {
                "status": "completed",
                "job_id": lease.job_id,
                "step": int(row["completed_steps"]),
                "total_steps": int(row["total_batches"]) * int(row["epochs"]),
                "worker_id": lease.worker_id,
                "lease_epoch": lease.epoch,
            }

        completed = int(row["completed_steps"])
        per_epoch = int(row["total_batches"])
        total_steps = per_epoch * int(row["epochs"])
        if completed >= total_steps:
            raise StateConflict("job has no remaining steps but is not marked completed")

        design = np.asarray(json.loads(row["features_json"]), dtype=np.float64)
        targets = np.asarray(json.loads(row["labels_json"]), dtype=np.float64)
        weights = _loads(row["weights_json"])
        velocity = _loads(row["velocity_json"])

        batch_index = completed % per_epoch
        start = batch_index * BATCH_SIZE
        end = min(start + BATCH_SIZE, int(row["record_count"]))
        new_weights, new_velocity, batch_loss = sgd_step(
            weights,
            velocity,
            design[start:end],
            targets[start:end],
            float(row["learning_rate"]),
            float(row["momentum"]),
        )
        new_completed = completed + 1
        final = new_completed == total_steps
        evaluation = dataset_loss(new_weights, design, targets)
        now = time.time()

        updated = conn.execute(
            """
            UPDATE jobs
               SET weights_json=?, velocity_json=?, loss=?, completed_steps=?,
                   status=?, updated_at=?, finished_at=?
             WHERE job_id=? AND completed_steps=? AND status IN ('queued','running')
            """,
            (
                _dumps(new_weights),
                _dumps(new_velocity),
                evaluation,
                new_completed,
                "completed" if final else "running",
                now,
                now if final else None,
                lease.job_id,
                completed,
            ),
        )
        if updated.rowcount != 1:
            raise StateConflict("job progress changed during an exclusive transaction")
        conn.execute(
            """
            INSERT INTO checkpoints(
                job_id, step, weights_json, velocity_json, loss, batch_loss,
                committed_at, worker_id, lease_epoch
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                lease.job_id,
                new_completed,
                _dumps(new_weights),
                _dumps(new_velocity),
                evaluation,
                batch_loss,
                now,
                lease.worker_id,
                lease.epoch,
            ),
        )
        conn.execute(
            "UPDATE leases SET heartbeat_at=? WHERE job_id=?", (now, lease.job_id)
        )
        if final:
            _event(
                conn,
                lease.job_id,
                "job_completed",
                {"completed_steps": new_completed, "worker_id": lease.worker_id},
            )
    return {
        "status": "completed" if final else "running",
        "job_id": lease.job_id,
        "step": new_completed,
        "total_steps": total_steps,
        "batch_loss": batch_loss,
        "loss": evaluation,
        "worker_id": lease.worker_id,
        "lease_epoch": lease.epoch,
    }


def record_event(db_path, job_id: str, event: str, detail: dict[str, Any] | None = None) -> None:
    """Append an audit event, unless the job is gone or was erased.

    A worker that discovers its job vanished mid-training calls this on the
    way out; writing would either fail on the foreign key or resurrect an
    erased job's event stream, so it is silently skipped.
    """
    with transaction(db_path) as conn:
        if _tombstone(conn, job_id) is not None or _job_row(conn, job_id) is None:
            return
        _event(conn, job_id, event, detail)
