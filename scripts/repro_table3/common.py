"""Configuration, provenance and environment construction for the Table 3 experiment."""

import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/repro_table3"
DEFAULT_OUTPUT = ROOT / "outputs/table3_sac_repro"
RUN_BASELINE = "c79e8f78a9229b904f123276168a9ef7746177fd"
WORKLOAD = "data/workload/alibaba_2020_dataset/result_df_full_year_2020.pkl"
WORKLOAD_SHA256 = "3ba8a8e0067288f8d2542752a292f7da04cb79305689ba145cc6060e96cb7f2e"
PHYSICS = "configs/dcs/dc_config.json"
CONFIG_FILES = (
    "sim_train.yaml", "sim_eval.yaml", "algorithm.yaml", "reward.yaml",
    "datacenters.yaml", "experiment_manifest.yaml",
)
SIMULATION = {
    "year": 2023, "month": 7, "init_day": 1, "init_hour": 0, "duration_days": 30,
    "timestep_minutes": 15, "workload_path": WORKLOAD, "cloud_provider": "aws",
    "strategy": "manual_rl", "single_action_mode": False, "disable_defer_action": False,
    "align_simulation_time": True,
}
ALGORITHM = {
    "name": "SAC", "gamma": 0.99, "alpha": 0.01, "actor_learning_rate": 1e-4,
    "critic_learning_rate": 3e-4, "optimizer": "Adam", "batch_size": 512, "tau": 0.005,
    "replay_buffer_size": 100_000, "warmup_steps": 1_000, "total_steps": 10_000_000,
    "update_frequency": 1, "policy_update_frequency": 2, "hidden_dim": 64,
    "max_tasks": 400, "device": "auto", "use_attention": False, "use_layer_norm": False,
    "training_seed": 42, "save_interval": 10_000, "log_interval": 100,
}
REWARDS = {
    "energy_price": {"weight": 0.5, "args": {"normalize_factor": 100_000, "use_metered_totals": True}},
    "carbon_emissions": {"weight": 0.3, "args": {"normalize_factor": 10, "use_metered_totals": True}},
    "transmission_cost": {"weight": 0.2, "args": {"normalize_factor": 1}},
    "sla_penalty": {"weight": 0.1, "args": {"penalty_per_violation": 5.0}},
    "transmission_emissions": {"weight": 0.1, "args": {"normalize_factor": 1}},
}
DATACENTERS = (
    ("US-CAL-CISO", -7, 0.18, 50_000, 1_000, 80_000),
    ("DE-LU", 1, 0.22, 85_000, 600, 80_000),
    ("CL-SIC", -5, 0.20, 110_000, 300, 60_000),
    ("SG", 8, 0.25, 75_000, 700, 50_000),
    ("AU-NSW", 11, 0.15, 65_000, 300, 60_000),
)
DATACENTER_FIELDS = (
    "location", "timezone_shift", "population_weight", "total_cores", "total_gpus", "total_mem",
)


@dataclass(frozen=True)
class Config:
    directory: Path
    sim: dict
    algorithm: dict
    reward: dict
    datacenters: list
    manifest: dict
    files: dict


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG, help="Directory with the six reproduction YAML files")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Directory for experiment artifacts")


def require_values(actual, expected, section):
    if not isinstance(actual, dict):
        raise ValueError(f"{section} must be a mapping")
    for key, value in expected.items():
        found = actual.get(key)
        if found != value or (isinstance(value, bool) and not isinstance(found, bool)):
            raise ValueError(f"{section}.{key} must be {value!r}, got {found!r}")


def load_config(config_dir, mode):
    if mode not in ("train", "eval"):
        raise ValueError(f"Unknown experiment mode {mode!r}")
    directory = Path(config_dir).resolve()
    files = {name: (directory / name).read_text(encoding="utf-8") for name in CONFIG_FILES}
    sections = {name: yaml.safe_load(raw) for name, raw in files.items()}
    for name in ("train", "eval"):
        require_values(sections[f"sim_{name}.yaml"]["simulation"], {
            **SIMULATION, "shuffle_datacenters": name == "train",
            "use_tensorboard": name == "train", "randomize_episode_start": name == "train",
        }, f"sim_{name}")
    sim = sections[f"sim_{mode}.yaml"]["simulation"]
    algorithm = sections["algorithm.yaml"]["algorithm"]
    reward = sections["reward.yaml"]["reward"]
    datacenters = sections["datacenters.yaml"]["datacenters"]
    manifest = sections["experiment_manifest.yaml"]["experiment"]
    require_values(algorithm, ALGORITHM, "algorithm")
    require_values(reward, {"normalize": False, "components": REWARDS}, "reward")
    if not isinstance(datacenters, list) or len(datacenters) != len(DATACENTERS):
        raise ValueError("Table 8 requires exactly five data centers")
    for index, (dc, values) in enumerate(zip(datacenters, DATACENTERS), start=1):
        require_values(dc, {
            **dict(zip(DATACENTER_FIELDS, values)), "dc_id": index,
            "dc_config_file": PHYSICS, "use_rl_hvac": False, "hru_enabled": False,
        }, f"datacenters[{index}]")
    require_values(manifest, {
        "baseline_commit": RUN_BASELINE, "observation_dim": 34, "action_dim": 6,
        "evaluation_steps": 2880, "evaluation_seeds": list(range(10)),
        "evaluation_modes": ["geo_only", "time_only", "geo_time"],
        "workload": {
            "trace_year": 2020, "task_scale": 5, "scale": 1, "group_size": 1,
            "sla_multiplier": 1.5, "active_hours_local": [8, 20],
            "activity_active": 1.0, "activity_inactive": 0.3,
        },
    }, "experiment")
    return Config(directory, sim, algorithm, reward, datacenters, manifest, files)


def snapshot_config(config, target):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    for filename, raw in config.files.items():
        (target / filename).write_text(raw, encoding="utf-8", newline="\n")
    shutil.copyfile(ROOT / PHYSICS, target / "dc_config.json")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_workload(config):
    path = ROOT / config.sim["workload_path"]
    archive = path.with_suffix(".zip")
    if path.is_file():
        if not path.stat().st_size:
            raise ValueError(f"Workload is empty: {path}")
        return path
    if not archive.is_file():
        raise FileNotFoundError(f"Workload and tracked archive are missing: {path}, {archive}")
    # Extract only the expected member, with CRC validation, and only publish a complete file.
    with zipfile.ZipFile(archive) as source, source.open(path.name) as payload:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".workload-", delete=False) as temporary:
            staged = Path(temporary.name)
            try:
                shutil.copyfileobj(payload, temporary)
                temporary.close()
                staged.replace(path)
            finally:
                temporary.close()
                staged.unlink(missing_ok=True)
    return path


def input_paths(config):
    paths = {config.sim["workload_path"]: resolve_workload(config), PHYSICS: ROOT / PHYSICS}
    year = config.sim["year"]
    provider = config.sim["cloud_provider"]
    data_files = [f"data/network_cost/{provider}_transmission_cost_matrix.csv"]
    for dc in config.datacenters:
        loc = dc["location"]
        data_files.extend((
            f"data/carbon_intensity/{loc}/{year}/{loc}_{year}_hourly.csv",
            f"data/electricity_prices/standardized/{loc}/{year}/{loc}_electricity_prices_{year}.csv",
            f"data/weather/{loc}/{year}.json",
        ))
    paths.update({name: ROOT / name for name in data_files})
    for name, path in paths.items():
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"Missing or empty experiment input: {name} ({path})")
    return paths


def input_hashes(config):
    hashes = {name: sha256(path) for name, path in input_paths(config).items()}
    if hashes[WORKLOAD] != WORKLOAD_SHA256:
        raise ValueError(f"Workload SHA256 differs from the baseline archive: {hashes[WORKLOAD]}")
    return hashes


def git_metadata():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()

    return {
        "baseline_commit": RUN_BASELINE,
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=normal")),
    }


def write_json(path, data):
    path = Path(path)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}-", encoding="utf-8",
        newline="\n", delete=False,
    ) as handle:
        staged = Path(handle.name)
        try:
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.close()
            staged.replace(path)
        finally:
            handle.close()
            staged.unlink(missing_ok=True)


def seed_everything(seed):
    import numpy as np
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name="auto"):
    import torch

    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def start_time(config):
    import pandas as pd

    sim = config.sim
    return pd.Timestamp(
        year=sim["year"], month=sim["month"], day=sim["init_day"], hour=sim["init_hour"], tz="UTC",
    )


def make_env(config, writer=None):
    import pandas as pd

    from envs.task_scheduling_env import TaskSchedulingEnv
    from rewards.predefined.composite_reward import CompositeReward
    from simulation.cluster_manager import DatacenterClusterManager

    sim = config.sim
    start = start_time(config)
    datacenters = copy.deepcopy(config.datacenters)
    for dc in datacenters:
        dc["dc_config_file"] = str(ROOT / dc["dc_config_file"])
    cluster = DatacenterClusterManager(
        config_list=datacenters,
        simulation_year=start.year,
        init_day=start.dayofyear - 1,
        init_hour=start.hour,
        strategy=sim["strategy"],
        tasks_file_path=str(resolve_workload(config)),
        shuffle_datacenter_order=sim["shuffle_datacenters"],
        cloud_provider=sim["cloud_provider"],
    )
    reward_fn = CompositeReward(**config.reward)
    env = TaskSchedulingEnv(
        cluster_manager=cluster,
        start_time=start,
        end_time=start + pd.Timedelta(days=sim["duration_days"]),
        reward_fn=reward_fn,
        writer=writer if sim["use_tensorboard"] else None,
        sim_config=sim,
        initial_seed_for_resets=config.algorithm["training_seed"],
    )
    dimensions = (env.obs_dim_per_task, env.agent_output_act_dim)
    expected = (config.manifest["observation_dim"], config.manifest["action_dim"])
    if dimensions != expected or env.single_action_mode or env.disable_defer_action:
        raise RuntimeError(f"Expected multi-action with Defer and dimensions {expected}, got {dimensions}")
    return env
