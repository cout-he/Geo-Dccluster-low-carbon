"""Run the repository's five native Table 3 heuristics with SAC evaluation inputs."""
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from statistics import mean, stdev
import platform
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from scripts.repro_table3.common import (
    git_metadata, input_hashes, load_config, make_env, seed_everything, sha256, write_json,
)
from scripts.repro_table3.evaluate import ActionTracker, csv_writer
from scripts.repro_table3.metrics import (
    AGG_FIELDS, DC_STEP_FIELDS, METRICS, PER_SEED_FIELDS, STEP_FIELDS,
    build_step_metrics, summarize_run,
)

OUTPUT = Path(__file__).resolve().parent
SAC = ROOT / "outputs/table3_sac_repro"
SNAPSHOT = SAC / "training_config_snapshot"
STRATEGIES = ("local_only", "lowest_carbon", "lowest_price", "most_available", "round_robin")
SEEDS = tuple(range(10))
STEPS = 2880
PAPER = {
    "local_only": ((101349, 5162), (328, 7.2), (1087, 2.0), (7394, 64), (0.02, 0.03), (3.4, 0), (11.4, 0.2), (0, 0), (0, 0)),
    "lowest_carbon": ((99845, 4765), (311, 7.6), (1122, 8.6), (7336, 70), (1.32, 0.32), (3.2, 0.3), (12.6, 0.8), (3273, 200), (0, 0)),
    "lowest_price": ((94119, 3777), (325, 7.4), (1097, 7.3), (7355, 58), (1.15, 0.54), (3.8, 0.2), (11.9, 0.7), (3572, 213), (0, 0)),
    "most_available": ((107067, 3770), (325, 6.6), (1165, 0.6), (7373, 62), (1.70, 0), (2.4, 0), (15.6, 0), (2927, 38), (0, 0)),
    "round_robin": ((100896, 4450), (329, 7.2), (1087, 1.6), (7427, 62), (0, 0), (3.5, 0), (11.9, 0), (3580, 48), (0, 0)),
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fields, rows):
    with csv_writer(path, fields) as writer:
        writer.writerows(rows)


def run_one(strategy, seed, reference, config_fingerprint, steps=STEPS):
    torch.set_num_threads(1)
    config = load_config(SNAPSHOT, "eval")
    config = replace(config, sim={**config.sim, "strategy": strategy})
    seed_everything(seed)
    env = make_env(config)
    final_dir = OUTPUT / "runs" / strategy / f"seed_{seed:02d}"
    if steps == STEPS and final_dir.exists():
        stored = read_json(final_dir / "run.json")
        if (
            stored["config_fingerprint"] != config_fingerprint
            or stored["arrival_stream_sha256"] != reference["arrival_stream_sha256"]
            or stored["summary"]["steps"] != steps
        ):
            raise RuntimeError(f"Existing result is from a different experiment: {final_dir}")
        for name, checksum in stored["output_sha256"].items():
            if sha256(final_dir / name) != checksum:
                raise RuntimeError(f"Existing result checksum changed: {final_dir / name}")
        env.close()
        return stored
    try:
        _, reset = env.reset(seed=seed)
        if reset != reference["reset"]:
            raise RuntimeError(f"SAC reset input mismatch: {strategy} / {seed}")
        cluster = env.cluster_manager
        get_tasks = cluster.get_tasks_for_timestep
        observed = []

        def record_tasks(timestamp):
            if timestamp != env.current_time or observed:
                raise RuntimeError("Native strategy requested another workload batch within one step")
            tasks = get_tasks(timestamp)
            observed.append(tasks)
            return tasks

        cluster.get_tasks_for_timestep = record_tasks
        tracker = ActionTracker(strategy, seed)
        step_rows = []
        directory = final_dir.parent
        directory.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=f".seed_{seed:02d}-", dir=directory, ignore_cleanup_errors=True) as stage:
            stage = Path(stage)
            with csv_writer(stage / "step_metrics.csv.gz", STEP_FIELDS) as sw, csv_writer(
                stage / "dc_step_metrics.csv.gz", DC_STEP_FIELDS
            ) as dw:
                for step in range(1, steps + 1):
                    observed.clear()
                    timestamp = env.current_time
                    _, reward, done, truncated, info = env.step([])
                    tasks, = observed
                    actions = [task.dest_dc_id for task in tasks]
                    if any(action not in (1, 2, 3, 4, 5) for action in actions):
                        raise RuntimeError(f"Unassigned task: {strategy} / {seed} at {timestamp}")
                    _, counts = tracker.record(step, timestamp, tasks, actions, actions, carried_after=[])
                    row, dc_rows = build_step_metrics(
                        info, {"mode": strategy, "seed": seed, "step": step, "timestamp_utc": timestamp.isoformat()},
                        counts, in_transit=len(env.in_transit_tasks), reward=reward, datacenters=cluster.datacenters,
                    )
                    if (done or truncated) != (step == STEPS) or row["defer_actions"] or row["in_transit"]:
                        raise RuntimeError(f"Invalid rule lifecycle: {strategy} / {seed} step {step}")
                    if strategy == "local_only" and (row["cross_dc_assignments"] or row["transmission_cost_usd"]):
                        raise RuntimeError(f"Local Only transferred tasks: seed {seed} step {step}")
                    sw.writerow(row)
                    dw.writerows(dc_rows)
                    step_rows.append(row)
            summary = summarize_run(step_rows)
            last = step_rows[-1]
            if summary["unfinished_tasks"] != last["pending_tasks"] + last["running_tasks"]:
                raise RuntimeError(f"Task accounting mismatch: {strategy} / {seed}")
            digest = tracker.arrival_digest.hexdigest()
            if steps == STEPS and (digest != reference["arrival_stream_sha256"] or last["timestamp_utc"] != reference["last_step_utc"]):
                raise RuntimeError(f"SAC arrival stream or closing time mismatch: {strategy} / {seed}")
            result = {
                "config_fingerprint": config_fingerprint, "reset": reset, "summary": summary,
                "arrival_stream_sha256": digest, "last_step_utc": last["timestamp_utc"],
                "actions": list(tracker.summarize(summary["cross_dc_started"])),
            }
            if steps == STEPS:
                result["output_sha256"] = {
                    name: sha256(stage / name) for name in ("step_metrics.csv.gz", "dc_step_metrics.csv.gz")
                }
                write_json(stage / "run.json", result)
                stage.rename(final_dir)
            return result
    finally:
        env.close()


def aggregate(rows):
    output = []
    for mode in dict.fromkeys(row["mode"] for row in rows):
        subset = [row for row in rows if row["mode"] == mode]
        if sorted(int(row["seed"]) for row in subset) != list(SEEDS):
            raise RuntimeError(f"Incomplete results for {mode}")
        for metric, (_, unit) in METRICS.items():
            values = [float(row[metric]) for row in subset]
            output.append(dict(mode=mode, metric=metric, unit=unit, n=len(values), mean=mean(values), std=stdev(values)))
    return output


def summarize(runs, sac_rows):
    result_rows = [run["summary"] for run in runs]
    aggregate_rows = aggregate(result_rows)
    write_csv(OUTPUT / "metrics_per_seed.csv", PER_SEED_FIELDS, result_rows)
    write_csv(OUTPUT / "metrics_mean_std.csv", AGG_FIELDS, aggregate_rows)
    write_csv(OUTPUT / "combined_rbc_sac_mean_std.csv", AGG_FIELDS, aggregate(result_rows + sac_rows))
    write_csv(
        OUTPUT / "action_statistics.csv", ("mode", "seed", "kind", "action", "count"),
        (action for run in runs for action in run["actions"]),
    )
    comparisons = []
    for row in aggregate_rows:
        paper_mean, paper_std = dict(zip(METRICS, PAPER[row["mode"]], strict=True))[row["metric"]]
        differences = row["mean"] - paper_mean
        comparisons.append({
            **row, "paper_mean": paper_mean, "paper_std": paper_std, "rbc_minus_paper": differences,
            "relative_difference": differences / abs(paper_mean) if paper_mean else None,
        })
    write_csv(OUTPUT / "paper_table3_comparison.csv",
              (*AGG_FIELDS, "paper_mean", "paper_std", "rbc_minus_paper", "relative_difference"), comparisons)
    paired = []
    sac_lookup = {(row["mode"], int(row["seed"])): row for row in sac_rows}
    sac_modes = ("geo_only", "time_only", "geo_time")
    for row in result_rows:
        for mode in sac_modes:
            reference = sac_lookup[mode, row["seed"]]
            for metric, (_, unit) in METRICS.items():
                rbc, sac = float(row[metric]), float(reference[metric])
                paired.append(dict(rule=row["mode"], sac=mode, seed=row["seed"], metric=metric,
                                   unit=unit, rbc_value=rbc, sac_value=sac, rbc_minus_sac=rbc - sac))
    write_csv(OUTPUT / "paired_differences_vs_sac.csv",
              ("rule", "sac", "seed", "metric", "unit", "rbc_value", "sac_value", "rbc_minus_sac"), paired)
    paired_summary = []
    for rule, mode, metric in product(STRATEGIES, sac_modes, METRICS):
        deltas = [row["rbc_minus_sac"] for row in paired if (row["rule"], row["sac"], row["metric"]) == (rule, mode, metric)]
        paired_summary.append(dict(rule=rule, sac=mode, metric=metric, unit=METRICS[metric][1],
                                   n=len(deltas), mean=mean(deltas), std=stdev(deltas)))
    write_csv(OUTPUT / "paired_mean_std_vs_sac.csv", ("rule", "sac", "metric", "unit", "n", "mean", "std"), paired_summary)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--smoke", action="store_true", help="Test only 8 steps for seed 0 of each rule")
    args = parser.parse_args()
    sac_manifest = read_json(SAC / "eval_manifest.json")
    if sac_manifest["smoke"] or sac_manifest["status"] != "complete" or sac_manifest["steps_per_run"] != STEPS:
        raise RuntimeError("Expected completed official-scale SAC reference")
    config = load_config(SNAPSHOT, "eval")
    for key, value in (("sim", config.sim), ("datacenters", config.datacenters), ("reward", config.reward)):
        if value != sac_manifest["config"][key]:
            raise RuntimeError(f"SAC snapshot differs for {key}")
    hashes = input_hashes(config)
    expected_hashes = sac_manifest["inputs_sha256"]
    if hashes.keys() != expected_hashes.keys():
        raise RuntimeError("SAC input inventory mismatch")
    normalized_csv = []
    for name, digest in hashes.items():
        if digest == expected_hashes[name]:
            continue
        raw = (ROOT / name).read_bytes() if name.endswith(".csv") else b""
        if not raw or hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() != expected_hashes[name]:
            raise RuntimeError(f"SAC input data mismatch (including LF comparison): {name}")
        normalized_csv.append(name)
    sac_rows = csv_rows(SAC / "metrics_per_seed.csv")
    if sha256(SAC / "metrics_per_seed.csv") != sac_manifest["output_sha256"]["metrics_per_seed.csv"]:
        raise RuntimeError("SAC per-seed results SHA256 mismatch")
    source = git_metadata()
    code_sha = sha256(Path(__file__))
    config_fingerprints = {name: sha256(SNAPSHOT / name) for name in config.files}
    fingerprint = hashlib.sha256(json.dumps(
        [hashes, config_fingerprints, source["commit"], code_sha], sort_keys=True
    ).encode()).hexdigest()
    refs = {run["seed"]: run for run in sac_manifest["runs"] if run["mode"] == "time_only"}
    jobs = list(product(STRATEGIES, (0,) if args.smoke else SEEDS))
    results = {}
    started = utc_now()
    print(f"SAC inputs verified ({len(hashes)} files, {len(normalized_csv)} CSV line ending differences). Launching {len(jobs)} runs.", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {
            pool.submit(run_one, rule, seed, refs[seed], fingerprint, 8 if args.smoke else STEPS): (rule, seed)
            for rule, seed in jobs
        }
        for future in as_completed(pending):
            rule, seed = pending[future]
            result = future.result()
            results[rule, seed] = result
            print(f"[{len(results):02d}/{len(jobs)}] {rule} seed {seed} | steps={result['summary']['steps']}", flush=True)
    if args.smoke:
        return
    ordered = [results[job] for job in jobs]
    summarize(ordered, sac_rows)
    write_json(OUTPUT / "run_manifest.json", {
        "status": "complete", "started_at_utc": started, "ended_at_utc": utc_now(),
        "strategies": STRATEGIES, "seeds": SEEDS, "steps_per_run": STEPS,
        "sim": config.sim, "datacenters": config.datacenters, "reward": config.reward,
        "strategy_dispatch": "Native DatacenterClusterManager.strategy_map and immediate destination enqueue",
        "most_available_definition": "Repository implementation: maximum free CPU count among feasible sites",
        "sac_reference_git": sac_manifest["git"], "sac_reference_sha256": sha256(SAC / "eval_manifest.json"),
        "git": source, "runner_sha256": code_sha, "config_fingerprint": fingerprint, "input_sha256": hashes, "sac_reference_input_sha256": expected_hashes,
        "csv_crlf_to_lf_matches": normalized_csv,
        "runtime": dict(platform=platform.platform(), python=platform.python_version(),
                        torch=str(torch.__version__), numpy=np.__version__, pandas=pd.__version__, device="cpu"),
        "runs": [{"rule": rule, "seed": seed, "arrival_stream_sha256": results[rule, seed]["arrival_stream_sha256"],
                  "path": f"runs/{rule}/seed_{seed:02d}/run.json"} for rule, seed in jobs],
    })
    print(f"Verified and summarized {len(jobs)} complete runs in {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()

