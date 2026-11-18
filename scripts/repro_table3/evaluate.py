"""Evaluate one SAC checkpoint under the three paired Table 3 policies."""

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.distributions import Categorical

from rl_components.agent_net import ActorNet
from scripts.repro_table3.metrics import (
    ACTION_COUNT_KEYS,
    AGG_FIELDS,
    COMPARISON_FIELDS,
    DC_STEP_FIELDS,
    KEYS,
    MODES,
    PAIRED_FIELDS,
    PER_SEED_FIELDS,
    STEP_FIELDS,
    build_step_metrics,
    paper_comparison,
    summarize_run,
    summarize_seeds,
)


ACTION_FIELDS = KEYS + (
    "task_id", "job_name", "origin_dc_id", "raw_action", "policy_action",
    "effective_action", "forced_origin", "cross_dc_assignment",
)
ACTION_STAT_FIELDS = ("mode", "seed", "kind", "action", "count")
OUTPUT_FILES = (
    "step_metrics.csv", "dc_step_metrics.csv", "action_decisions.csv.gz",
    "metrics_per_seed.csv", "metrics_mean_std.csv", "action_statistics.csv",
    "paired_differences.csv", "paper_table3_comparison.csv", "eval_manifest.json",
)


def sample_policy(logits, origins, mode):
    """Return the unmodified actor draw and the action selected by the ablation."""
    if mode not in MODES or logits.ndim != 2 or logits.shape != (len(origins), 6):
        raise ValueError("Expected one row of six action logits per task and a valid mode")
    if not len(origins):
        empty = torch.empty(0, dtype=torch.long, device=logits.device)
        return empty, empty
    if not all(origin in range(1, 6) for origin in origins):
        raise ValueError("Task origin must be a one-based DC ID from 1 to 5")

    raw = Categorical(logits=logits).sample()
    if mode == "geo_only":
        selected = raw.clone()
        rejected = raw == 0
        if rejected.any():
            # Categorical normalizes the five remaining logits before sampling.
            selected[rejected] = Categorical(logits=logits[rejected, 1:]).sample() + 1
    elif mode == "time_only":
        origin_actions = torch.as_tensor(origins, dtype=torch.long, device=logits.device)
        selected = torch.where(raw == 0, raw, origin_actions)
    else:
        selected = raw
    return raw, selected


class ActionTracker:
    """Audit the effective simulator action and give each arrival a stable run-local ID."""

    def __init__(self, mode, seed):
        self.mode = mode
        self.seed = seed
        self.total_arrivals = 0
        self.arrival_digest = hashlib.sha256()
        self.carried = {}
        self.totals = Counter()
        self.actions = {kind: Counter() for kind in ("raw_action", "policy_action", "effective_action")}

    def record(self, step, timestamp, tasks, raw_actions, policy_actions, *, carried_after):
        rows = []
        counts = dict.fromkeys(ACTION_COUNT_KEYS, 0)
        next_carried = {}
        pending = set(carried_after)
        counts["decisions"] = len(tasks)
        timestamp_utc = timestamp.isoformat()
        for task, raw, policy in zip(tasks, raw_actions, policy_actions, strict=True):
            first_decision = task not in self.carried
            if first_decision:
                self.total_arrivals += 1
                counts["new_arrivals"] += 1
                task_id = self.total_arrivals
                arrival = (timestamp_utc, task_id, task.job_name, task.origin_dc_id)
                self.arrival_digest.update(repr(arrival).encode("utf-8"))
            else:
                task_id = self.carried[task]

            forced = timestamp > task.sla_deadline
            expected = task.origin_dc_id if forced else policy
            effective = 0 if task in pending else task.dest_dc_id
            if effective != expected:
                raise RuntimeError(
                    f"Unexpected simulator action in {self.mode} seed {self.seed} "
                    f"step {step}, task {task_id}: expected {expected}, got {effective}"
                )
            cross_dc = effective != 0 and effective != task.origin_dc_id
            if effective == 0:
                next_carried[task] = task_id
                counts["defer_actions"] += 1
                counts["unique_tasks_deferred"] += int(first_decision)
            counts["deadline_overrides"] += int(forced and policy != effective)
            counts["cross_dc_assignments"] += int(cross_dc)

            rows.append({
                "mode": self.mode, "seed": self.seed, "step": step,
                "timestamp_utc": timestamp_utc, "task_id": task_id,
                "job_name": task.job_name, "origin_dc_id": task.origin_dc_id,
                "raw_action": raw, "policy_action": policy, "effective_action": effective,
                "forced_origin": forced, "cross_dc_assignment": cross_dc,
            })
            for kind, histogram in self.actions.items():
                histogram[rows[-1][kind]] += 1

        if self.mode == "geo_only" and counts["defer_actions"]:
            raise RuntimeError("Geo Only produced an effective Defer action")
        if self.mode == "time_only" and counts["cross_dc_assignments"]:
            raise RuntimeError("Time Only produced an effective cross-DC assignment")
        self.carried = next_carried
        self.totals.update(counts)
        return rows, counts

    def summarize(self, cross_dc_started):
        metadata = {"mode": self.mode, "seed": self.seed}
        for kind, histogram in self.actions.items():
            for action in range(6):
                yield {**metadata, "kind": kind, "action": action, "count": histogram[action]}
        counts = {**self.totals, "cross_dc_started": cross_dc_started}
        for kind, count in counts.items():
            yield {**metadata, "kind": kind, "action": None, "count": count}


@contextmanager
def csv_writer(path, fields):
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        yield writer


def validate_checkpoint(path, config, hashes, *, smoke):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    dims = tuple(checkpoint.get(field) for field in ("obs_dim", "act_dim", "hidden_dim"))
    if checkpoint.get("schema_version") != 1 or dims != (34, 6, 64):
        raise ValueError(f"Checkpoint schema or dimensions are incompatible: {path}")
    if checkpoint.get("input_hashes") != hashes:
        raise ValueError("Input file hashes differ from the training checkpoint")
    trained = checkpoint.get("config", {})
    configs = {"datacenters": config.datacenters, "algorithm": config.algorithm,
               "reward": config.reward, "manifest": config.manifest}
    for name, expected in configs.items():
        if trained.get(name) != expected:
            raise ValueError(f"Evaluation {name} differs from the training checkpoint")
    if checkpoint.get("smoke") != smoke:
        raise ValueError("Formal evaluation requires a formal checkpoint, smoke requires a smoke checkpoint")
    total_steps, completed = checkpoint.get("total_steps", 0), checkpoint.get("global_step", 0)
    if not 0 < completed <= total_steps or not checkpoint.get("training_run_id"):
        raise ValueError("Checkpoint is missing valid completed steps or a training run ID")

    if not smoke:
        from scripts.repro_table3.common import sha256

        final_path = path.with_name("final.pt")
        final = checkpoint if path == final_path else torch.load(
            final_path, map_location="cpu", weights_only=True,
        )
        metadata_path = path.parent.parent / "training_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            final.get("kind") != "final"
            or final.get("training_run_id") != checkpoint["training_run_id"]
            or total_steps != config.algorithm["total_steps"]
            or final.get("global_step") != total_steps
            or metadata.get("status") != "complete"
            or metadata.get("training_run_id") != checkpoint["training_run_id"]
            or metadata.get("global_step") != total_steps
            or metadata.get("final_checkpoint_sha256") != sha256(final_path)
        ):
            raise ValueError("Formal evaluation requires final.pt and metadata from the same completed training run")
    return checkpoint


@torch.inference_mode()
def evaluate_run(config, actor, device, mode, seed, steps, step_writer, dc_writer, action_writer):
    from scripts.repro_table3.common import make_env, seed_everything, start_time

    seed_everything(seed)
    env = make_env(config)
    try:
        obs, reset_info = env.reset(seed=seed)
        if env.current_time != start_time(config):
            raise RuntimeError(f"Evaluation reset at unexpected time {env.current_time!s}")
        # The fixed HVAC controller uses no torch RNG. Initialize policy sampling
        # afresh even if environment construction consumed torch randomness.
        torch.manual_seed(seed)
        dcs = env.cluster_manager.datacenters
        if (
            env.single_action_mode or env.disable_defer_action
            or env.obs_dim_per_task != 34 or env.agent_output_act_dim != 6
            or [dc.dc_id for dc in dcs.values()] != [1, 2, 3, 4, 5]
        ):
            raise RuntimeError("Expected ordered DC IDs 1–5 and the full multi-action environment")

        action_tracker = ActionTracker(mode, seed)
        step_rows = []
        for step in range(1, steps + 1):
            timestamp = env.current_time
            metadata = {
                "mode": mode, "seed": seed, "step": step, "timestamp_utc": timestamp.isoformat(),
            }
            tasks = list(env.current_tasks)
            if len(obs) != len(tasks):
                raise RuntimeError("Observation and current task counts differ")
            if tasks:
                features = torch.as_tensor(np.stack(obs), dtype=torch.float32, device=device)
                logits = actor(features)
                raw, policy = sample_policy(logits, [task.origin_dc_id for task in tasks], mode)
                raw_actions, policy_actions = raw.tolist(), policy.tolist()
            else:
                raw_actions, policy_actions = [], []

            obs, reward, done, truncated, info = env.step(policy_actions)
            decision_rows, counts = action_tracker.record(
                step, timestamp, tasks, raw_actions, policy_actions, carried_after=env.current_tasks,
            )
            row, dc_rows = build_step_metrics(
                info, metadata, counts, in_transit=len(env.in_transit_tasks), reward=reward, datacenters=dcs,
            )
            if mode == "time_only" and (
                row["cross_dc_started"]
                or row["transmission_cost_usd"]
                or row["transmission_energy_kwh"]
                or row["transmission_carbon_kg"]
            ):
                raise RuntimeError("Time Only produced actual cross-DC transfers")
            action_writer.writerows(decision_rows)
            step_writer.writerow(row)
            dc_writer.writerows(dc_rows)
            step_rows.append(row)

            should_end = env.current_time == env.end_time
            if (done or truncated) != should_end or (should_end and step != steps):
                raise RuntimeError(f"Unexpected evaluation termination at step {step}")

        summary = summarize_run(step_rows)
        outstanding = (
            len(action_tracker.carried) + step_rows[-1]["in_transit"]
            + step_rows[-1]["pending_tasks"] + step_rows[-1]["running_tasks"]
        )
        if summary["unfinished_tasks"] != outstanding:
            raise RuntimeError(f"Task accounting failed for {mode} seed {seed}")
        run_details = {
            "mode": mode, "seed": seed, "steps": steps,
            "last_step_utc": step_rows[-1]["timestamp_utc"],
            "arrival_stream_sha256": action_tracker.arrival_digest.hexdigest(),
            "reset": reset_info,
        }
        return summary, list(action_tracker.summarize(summary["cross_dc_started"])), run_details
    finally:
        env.close()


def main(argv=None):
    from scripts.repro_table3.common import (
        DEFAULT_OUTPUT, add_common_args, git_metadata, input_hashes, load_config, resolve_device,
        seed_everything, sha256, write_json,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--checkpoint", type=Path, help="SAC model checkpoint (default: checkpoints/best.pt in output directory)")
    parser.add_argument("--seeds", nargs="+", type=int, help="Paired evaluation seeds (default: configured seeds)")
    parser.add_argument("--smoke-steps", type=int, help="Limit each run to a short window with a smoke checkpoint")
    args = parser.parse_args(argv)

    config = load_config(args.config_dir, "eval")
    smoke = args.smoke_steps is not None
    seeds = args.seeds if args.seeds is not None else config.manifest["evaluation_seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(not 0 <= seed < 2**32 for seed in seeds):
        parser.error("--seeds must be distinct integers between 0 and 2^32 - 1")
    if not smoke and len(seeds) != 10:
        parser.error("formal evaluation requires 10 paired seeds")
    minutes_per_day = 24 * 60
    full_steps = config.sim["duration_days"] * minutes_per_day // config.sim["timestep_minutes"]
    if full_steps != config.manifest["evaluation_steps"]:
        raise ValueError("Evaluation horizon differs from the manifest")
    if smoke and not 1 <= args.smoke_steps < full_steps:
        parser.error(f"--smoke-steps must be between 1 and {full_steps - 1}")
    steps = args.smoke_steps if smoke else full_steps

    output = args.output_dir.resolve()
    if smoke and output == DEFAULT_OUTPUT.resolve():
        parser.error("smoke evaluation requires a separate --output-dir")
    checkpoint_path = (args.checkpoint or output / "checkpoints" / "best.pt").resolve(strict=True)
    checkpoint_hash = sha256(checkpoint_path)
    hashes = input_hashes(config)
    checkpoint = validate_checkpoint(checkpoint_path, config, hashes, smoke=smoke)
    if sha256(checkpoint_path) != checkpoint_hash:
        raise RuntimeError("Checkpoint changed while being loaded")
    seed_everything(seeds[0])
    device = resolve_device(config.algorithm["device"])
    actor = ActorNet(34, 6, hidden_dim=64, use_layer_norm=False).to(device)
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    actor.eval()

    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".evaluation.running"
    lock.mkdir()
    try:
        conflicts = [name for name in OUTPUT_FILES if (output / name).exists()]
        if conflicts:
            raise FileExistsError(f"Evaluation outputs already exist in {output}: {', '.join(conflicts)}")
        with TemporaryDirectory(prefix=".evaluation-", dir=output) as tmp:
            stage = Path(tmp)
            summaries, actions, runs = [], [], []
            started = datetime.now(timezone.utc).isoformat()
            with ExitStack() as writers:
                step_writer = writers.enter_context(csv_writer(stage / "step_metrics.csv", STEP_FIELDS))
                dc_writer = writers.enter_context(csv_writer(stage / "dc_step_metrics.csv", DC_STEP_FIELDS))
                action_writer = writers.enter_context(csv_writer(stage / "action_decisions.csv.gz", ACTION_FIELDS))
                for mode in MODES:
                    for seed in seeds:
                        print(f"Evaluating {mode}, seed {seed}, {steps} steps", flush=True)
                        summary, stats, run = evaluate_run(
                            config, actor, device, mode, seed, steps, step_writer, dc_writer, action_writer,
                        )
                        summaries.append(summary)
                        actions.extend(stats)
                        runs.append(run)

            for seed in seeds:
                arrivals = {run["arrival_stream_sha256"] for run in runs if run["seed"] == seed}
                if len(arrivals) != 1:
                    raise RuntimeError(f"Paired variants disagree on the input arrival stream for seed {seed}")
            aggregate, differences = summarize_seeds(summaries)
            for name, fields, rows in (
                ("metrics_per_seed.csv", PER_SEED_FIELDS, summaries),
                ("metrics_mean_std.csv", AGG_FIELDS, aggregate),
                ("action_statistics.csv", ACTION_STAT_FIELDS, actions),
                ("paired_differences.csv", PAIRED_FIELDS, differences),
                ("paper_table3_comparison.csv", COMPARISON_FIELDS, paper_comparison(aggregate, smoke=smoke)),
            ):
                with csv_writer(stage / name, fields) as writer:
                    writer.writerows(rows)
            write_json(stage / "eval_manifest.json", {
                "status": "complete", "smoke": smoke, "started_at_utc": started,
                "ended_at_utc": datetime.now(timezone.utc).isoformat(), "device": str(device),
                "seeds": seeds, "modes": MODES, "steps_per_run": steps,
                "timestep_minutes": config.sim["timestep_minutes"], "git": git_metadata(),
                "checkpoint": {
                    "path": str(checkpoint_path), "sha256": checkpoint_hash,
                    "training_run_id": checkpoint["training_run_id"], "global_step": checkpoint["global_step"],
                    "total_steps": checkpoint["total_steps"], "kind": checkpoint["kind"],
                    "training_git": checkpoint["git"],
                },
                "inputs_sha256": hashes, "runs": runs,
                "config": {
                    "sim": config.sim, "algorithm": config.algorithm,
                    "datacenters": config.datacenters, "reward": config.reward,
                    "experiment_manifest": config.manifest,
                },
                "output_sha256": {name: sha256(stage / name) for name in OUTPUT_FILES[:-1]},
            })
            moved = []
            try:
                for name in OUTPUT_FILES:
                    os.replace(stage / name, output / name)
                    moved.append(name)
            except BaseException:
                for name in moved:
                    (output / name).unlink()
                raise
    finally:
        lock.rmdir()
    print(f"Evaluation outputs saved to {output}", flush=True)


if __name__ == "__main__":
    main()
