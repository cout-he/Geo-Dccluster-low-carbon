"""Verify reproducibility settings, dependencies and selected inputs before a run."""

import argparse
import importlib
import json
import math
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.repro_table3.common import (
    DEFAULT_OUTPUT, add_common_args, git_metadata, input_hashes, input_paths, load_config,
    make_env, resolve_device, seed_everything, sha256, start_time, write_json,
)

GIB = 1024**3


def check_dependencies(stage):
    modules = ["numpy", "pandas", "matplotlib"]
    if stage != "plot":
        modules.extend(("torch", "gymnasium", "psychrolib", "cloudpickle", "tensorboard"))
    versions = {}
    for name in modules:
        try:
            module = importlib.import_module(name)
        except ImportError as error:
            raise RuntimeError(f"Dependency {name} unavailable. Install requirements-repro-table3.txt.") from error
        versions[name] = getattr(module, "__version__", "installed")
    return versions


def validate_data(config):
    import numpy as np
    import pandas as pd

    from utils.transmission_region_mapper import map_location_to_region

    hashes = input_hashes(config)
    paths = input_paths(config)
    year = config.sim["year"]
    year_start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
    hourly = pd.date_range(year_start, year_start.replace(year=year + 1), freq="h", inclusive="left")
    workload = pd.read_pickle(paths[config.sim["workload_path"]])
    if not {"interval_15m", "tasks_matrix"}.issubset(workload):
        raise ValueError("Workload is missing interval_15m or tasks_matrix")
    stamps = pd.DatetimeIndex(workload["interval_15m"])
    if stamps.tz is None or stamps.hasnans or not stamps.is_unique or not stamps.is_monotonic_increasing:
        raise ValueError("Workload must have sorted, unique, timezone-aware timestamps")
    stamps = stamps.tz_convert("UTC")
    if not stamps.equals(stamps.floor("15min")):
        raise ValueError("Workload timestamps fall outside the 15-minute grid")

    start = start_time(config)
    stop = start + pd.Timedelta(days=config.sim["duration_days"])
    if config.sim["randomize_episode_start"]:
        start -= pd.Timedelta(days=7)
        stop += pd.Timedelta(days=7, hours=23)
    start, stop = (time.replace(year=config.manifest["workload"]["trace_year"]) for time in (start, stop))
    if stamps[0] > start or stamps[-1] < stop - pd.Timedelta(minutes=15):
        raise ValueError(f"Workload does not cover {start} through {stop}")
    rows = workload.loc[(stamps >= start) & (stamps < stop), "tasks_matrix"]
    if rows.empty:
        raise ValueError("No workload rows in the requested period")
    counts = []
    for matrix in rows:
        array = np.asarray(matrix)
        if array.ndim != 2 or array.shape[1] < 9:
            raise ValueError("Workload task matrix lacks duration, resource, or bandwidth columns")
        values = array[:, 4:9].astype(float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Workload includes non-finite or negative duration, resource, or bandwidth")
        counts.append(len(matrix))

    carbon_column = "Carbon Intensity gCO" + chr(0x2082) + "eq/kWh (direct)"
    for name, path in paths.items():
        if name.startswith(("data/carbon_intensity/", "data/electricity_prices/")):
            frame = pd.read_csv(path)
            column = carbon_column if name.startswith("data/carbon_intensity/") else "Price (USD/MWh)"
            time = pd.DatetimeIndex(pd.to_datetime(frame["Datetime (UTC)"], utc=True))
            if not time.equals(hourly) or not np.isfinite(frame[column]).all():
                raise ValueError(f"Incomplete hourly UTC series or invalid measurements: {name}")
        elif name.startswith("data/weather/"):
            data = json.loads(path.read_text(encoding="utf-8"))["hourly"]
            time = pd.DatetimeIndex(pd.to_datetime(data["time"]))
            if not time.equals(hourly.tz_localize(None)):
                raise ValueError(f"Incomplete hourly weather series: {name}")
            for field in ("temperature_2m", "relative_humidity_2m", "cloudcover", "windspeed_10m"):
                if len(data[field]) != len(time) or not np.isfinite(np.asarray(data[field], dtype=float)).all():
                    raise ValueError(f"Invalid weather series {field}: {name}")
        elif name == "configs/dcs/dc_config.json":
            if not isinstance(json.loads(path.read_text(encoding="utf-8")), dict):
                raise ValueError(f"Invalid physical datacenter configuration: {name}")

    provider = config.sim["cloud_provider"]
    matrix = pd.read_csv(paths[f"data/network_cost/{provider}_transmission_cost_matrix.csv"], index_col=0)
    regions = [map_location_to_region(dc["location"], provider) for dc in config.datacenters]
    rates = matrix.loc[regions, regions].to_numpy()
    if not np.isfinite(rates).all() or (rates < 0).any():
        raise ValueError("Invalid transmission price in configured AWS region pairs")

    return {
        "window_start_utc": start.isoformat(),
        "window_stop_exclusive_utc": stop.isoformat(),
        "quarters_with_tasks": len(counts),
        "fresh_tasks": int(sum(counts)),
        "max_fresh_tasks_per_quarter": int(max(counts)),
        "input_sha256": hashes,
    }


def check_disk(output, stage, smoke):
    parent = output
    while not parent.exists():
        parent = parent.parent
    if output.exists() and not output.is_dir():
        raise ValueError(f"Output must be a directory: {output}")
    free = shutil.disk_usage(parent).free
    # A full replay and its temporary atomic replacement can coexist.
    required = (1 if smoke or stage != "train" else 25) * GIB
    if free < required:
        raise OSError(f"{output}: {free / GIB:.1f} GiB free, at least {required / GIB:.0f} GiB required")
    return round(free / GIB, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--stage", choices=("train", "eval", "plot"), default="train")
    parser.add_argument("--checkpoint", type=Path, help="Model checkpoint for evaluation")
    parser.add_argument("--resume", type=Path, help="Trainer state file for resumption")
    parser.add_argument("--smoke-steps", type=int, help="Use the disk threshold for a bounded smoke run")
    parser.add_argument("--env-smoke", action="store_true", help="Reset and step the real simulator once")
    args = parser.parse_args()
    if args.smoke_steps is not None and args.smoke_steps <= 0:
        parser.error("--smoke-steps must be positive")
    if args.env_smoke and args.stage == "plot":
        parser.error("--env-smoke requires the train or eval stage")
    output = args.output_dir.resolve()
    if args.smoke_steps is not None and output == DEFAULT_OUTPUT.resolve():
        parser.error("Smoke runs require an isolated --output-dir")
    stage = args.stage
    config = load_config(args.config_dir, "eval" if stage == "eval" else "train")
    report = {
        "stage": stage,
        "python": sys.version,
        "dependencies": check_dependencies(stage),
        "free_gib": check_disk(output, stage, args.smoke_steps is not None),
        "git": git_metadata(),
    }
    if stage == "eval":
        checkpoint = args.checkpoint or output / "checkpoints/best.pt"
        if not checkpoint.is_file() or not checkpoint.stat().st_size:
            raise FileNotFoundError(f"Evaluation checkpoint is missing or empty: {checkpoint}")
        report["checkpoint"] = {"path": str(checkpoint.resolve()), "sha256": sha256(checkpoint)}
    if args.resume:
        if stage != "train" or not args.resume.is_file() or not args.resume.stat().st_size:
            raise ValueError(f"Provide a nonempty trainer state with --stage train: {args.resume}")
        report["resume"] = str(args.resume.resolve())
    if stage == "plot":
        if not output.is_dir():
            raise FileNotFoundError(f"Missing results directory: {output}")
    else:
        import torch

        report["device"] = str(resolve_device(config.algorithm["device"]))
        if report["device"] == "cuda":
            report["gpu"] = torch.cuda.get_device_name(0)
            report["gpu_memory_gib"] = round(torch.cuda.get_device_properties(0).total_memory / GIB, 2)
        report["data"] = validate_data(config)
        if args.env_smoke:
            seed_everything(config.algorithm["training_seed"])
            env = make_env(config)
            try:
                observations, reset = env.reset(seed=config.algorithm["training_seed"])
                _, reward, ended, truncated, _ = env.step([1] * len(observations))
                if not len(observations) or not math.isfinite(reward) or ended or truncated:
                    raise RuntimeError("Unexpected single-step simulator result")
                report["env_smoke"] = {"tasks": len(observations), "reward": reward, "reset": reset}
            finally:
                env.close()

    output.mkdir(parents=True, exist_ok=True)
    write_json(output / f"preflight_{stage}.json", report)
    print(f"{stage} preflight passed | output={output} | free GiB={report['free_gib']}", flush=True)
    if "device" in report:
        print(f"device={report['device']} | checked {len(report['data']['input_sha256'])} inputs", flush=True)


if __name__ == "__main__":
    main()
