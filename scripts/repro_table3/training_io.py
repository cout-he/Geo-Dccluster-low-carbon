"""Durable local training state and CSV logs."""

from contextlib import contextmanager
import csv
import os
from pathlib import Path
import pickle
import shutil
import uuid

import torch

from scripts.repro_table3.common import sha256


COMPONENTS = (
    "energy_price", "carbon_emissions", "transmission_cost", "sla_penalty", "transmission_emissions",
)
LOG_COLUMNS = {
    "train_episodes.csv": (
        "episode", "start_step", "end_step", "total_reward", "mean_reward", "mean_reward_last10",
        "episode_length", "completed_at_utc", "reset_seed", "start_time_utc",
        "workload_start_time_utc", "dc_source_indices",
    ),
    "train_steps.csv": (
        "global_step", "interval_steps", "mean_reward", "actor_loss", "critic_loss", "entropy",
        "critic_updates", "actor_updates", "interval_critic_updates", "interval_actor_updates",
        "num_overflow_steps", "max_tasks_seen", "mean_total_cost_usd", "mean_total_carbon_kg",
        "sla_violated", "tasks_finished", "sla_violation_pct_finished",
    ),
    "train_reward_components.csv": ("global_step", "interval_steps", *COMPONENTS),
    "checkpoints_manifest.csv": ("global_step", "path", "sha256", "is_best", "kind"),
}


@contextmanager
def writer_lock(output):
    """Use a process lock released even when the training process is killed."""
    with (output / ".training.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            if not handle.tell():
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f"Another training process holds the output lock in {output}") from exc
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Another training process holds the output lock in {output}") from exc
        yield


def sync_directory(directory):
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write(path, write_data):
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("xb") as stream:
            write_data(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        sync_directory(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def replace_with_copy(source, destination):
    def copy_to(stream):
        with source.open("rb") as original:
            shutil.copyfileobj(original, stream)

    atomic_write(destination, copy_to)


class TrainingStorage:
    def __init__(self, output, resume_offsets=None):
        self.output = output
        self.models = output / "checkpoints"
        self.models.mkdir(exist_ok=True)
        self.resume_path = self.models / "resume_state.pkl"
        self.handles = {}
        self.writers = {}
        if resume_offsets is None:
            occupied = [name for name in LOG_COLUMNS if (output / name).exists()]
            occupied += [item.name for item in self.models.iterdir()]
            if occupied:
                raise FileExistsError(f"Training artifacts already exist in {output}: {', '.join(occupied)}")
        else:
            self.validate_resume_logs(resume_offsets)

        try:
            for name, fields in LOG_COLUMNS.items():
                path = output / name
                if resume_offsets is not None:
                    size = resume_offsets[name]
                    with path.open("r+b") as stream:
                        stream.truncate(size)
                handle = path.open("x" if resume_offsets is None else "a", newline="", encoding="utf-8")
                self.handles[name] = handle
                writer = csv.DictWriter(handle, fieldnames=fields)
                self.writers[name] = writer
                if resume_offsets is None:
                    writer.writeheader()
        except BaseException:
            self.close()
            raise

    def validate_resume_logs(self, resume_offsets):
        if set(resume_offsets) != set(LOG_COLUMNS):
            raise ValueError("Resume state has invalid CSV offsets")
        for name, fields in LOG_COLUMNS.items():
            path = self.output / name
            size = resume_offsets[name]
            if size <= 0 or not path.is_file() or path.stat().st_size < size:
                raise RuntimeError(f"Missing or truncated training CSV: {path}")
            with path.open(encoding="utf-8", newline="") as stream:
                if tuple(next(csv.reader(stream), ())) != fields:
                    raise RuntimeError(f"Invalid training CSV schema: {path}")

    def record(self, name, **values):
        self.writers[name].writerow(values)

    def publish_model(self, checkpoint, filename):
        path = self.models / filename
        atomic_write(path, lambda stream: torch.save(checkpoint, stream))
        digest = sha256(path)
        self.record(
            "checkpoints_manifest.csv",
            global_step=checkpoint["global_step"],
            path=path.relative_to(self.output).as_posix(),
            sha256=digest,
            is_best=checkpoint["kind"] == "best",
            kind=checkpoint["kind"],
        )
        return path, digest

    def final_model(self, global_step):
        manifest = self.output / "checkpoints_manifest.csv"
        with manifest.open(encoding="utf-8", newline="") as stream:
            rows = [row for row in csv.DictReader(stream) if row["kind"] == "final"]
        path = self.models / "final.pt"
        if (
            len(rows) != 1 or int(rows[0]["global_step"]) != global_step
            or rows[0]["path"] != "checkpoints/final.pt" or not path.exists()
            or sha256(path) != rows[0]["sha256"]
        ):
            raise RuntimeError(f"Completed resume state does not match {path}")
        return path, rows[0]["sha256"]

    def repair_best_alias(self, filename):
        alias = self.models / "best.pt"
        if filename is None:
            alias.unlink(missing_ok=True)
        else:
            replace_with_copy(self.models / Path(filename).name, alias)

    def save_resume(self, state):
        offsets = {}
        for name, stream in self.handles.items():
            stream.flush()
            os.fsync(stream.fileno())
            offsets[name] = (self.output / name).stat().st_size
        state = {**state, "csv_offsets": offsets}
        # Stream arrays directly. torch.save would buffer pickled NumPy data in RAM.
        atomic_write(self.resume_path, lambda stream: pickle.dump(state, stream, protocol=5))

    def close(self):
        for stream in self.handles.values():
            stream.close()
