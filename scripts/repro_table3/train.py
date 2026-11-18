"""Train one Geo+Time SAC policy from scratch or an exact local run state."""

import argparse
from collections import deque
from datetime import datetime, timezone
from importlib import metadata as package_metadata
import json
from math import fsum
import os
from pathlib import Path
import pickle
import platform
import random
import signal
import sys
import uuid

import cloudpickle
import numpy as np
import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.repro_table3.common import (
    DEFAULT_OUTPUT, add_common_args, git_metadata, input_hashes, load_config,
    make_env, resolve_device, seed_everything, snapshot_config, write_json,
)
from scripts.repro_table3.learning import ReplayBuffer, SAC
from scripts.repro_table3.training_io import COMPONENTS, TrainingStorage, writer_lock


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def runtime_metadata(device):
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "architecture": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "dependencies": {
            name: package_metadata.version(name)
            for name in ("cloudpickle", "gymnasium", "pandas", "psychrolib", "tensorboard", "PyYAML")
        },
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if device.type == "cuda" else None,
        "device": str(device),
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def serialize_environment(env):
    # Input data are separately hashed and reloaded on resume. TensorBoard
    # remains attached to the trainer and is opened at the resumed step.
    cluster = env.cluster_manager
    tasks = cluster.tasks
    cluster.tasks = None
    try:
        return cloudpickle.dumps(env, protocol=5)
    finally:
        cluster.tasks = tasks


class ReplayRewardStats:
    """The scalar recurrence from train_rl_agent.RunningStats."""
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-5

    def update_and_normalize(self, reward):
        self.count += 1
        last_mean = self.mean
        self.mean += (reward - self.mean) / self.count
        self.var += (reward - last_mean) * (reward - self.mean)
        return (reward - self.mean) / max(np.sqrt(self.var / self.count), 1e-6)


class TrainingWindow:
    def __init__(self):
        self.steps = 0
        self.total_reward = 0.0
        self.components = dict.fromkeys(COMPONENTS, 0.0)
        self.loss_sums = {}
        self.loss_counts = {}
        self.total_cost_usd = 0.0
        self.total_carbon_kg = 0.0
        self.tasks_finished = 0
        self.sla_violated = 0

    def record(self, reward, components, losses, info):
        self.steps += 1
        self.total_reward += reward
        for name in COMPONENTS:
            self.components[name] += components[name]
        dc_stats = [dc["__common__"] for dc in info["datacenter_infos"].values()]
        self.total_cost_usd += (
            fsum(dc["energy_cost_USD"] for dc in dc_stats) + info["transmission_cost_total_usd"]
        )
        self.total_carbon_kg += (
            fsum(dc["carbon_emissions_kg"] for dc in dc_stats) + info["transmission_emissions_total_kg"]
        )
        self.tasks_finished += sum(dc["finished_tasks_count"] for dc in dc_stats)
        self.sla_violated += sum(dc["__sla__"]["violated"] for dc in dc_stats)
        for name, value in losses.items():
            self.loss_sums[name] = self.loss_sums.get(name, 0) + value
            self.loss_counts[name] = self.loss_counts.get(name, 0) + 1

    def averages(self):
        result = {name: float(value) / self.loss_counts[name] for name, value in self.loss_sums.items()}
        if not all(np.isfinite(value) for value in result.values()):
            raise FloatingPointError(f"Nonfinite SAC training losses: {result}")
        return result


class Trainer:
    def __init__(self, config, environment, device, provenance, settings, resume=None):
        self.settings = settings
        self.provenance = provenance
        self.env = environment
        self.device = device
        self.seed = int(config.algorithm["training_seed"])
        self.best_model = None
        self.best_score = float("-inf")
        self.global_step = 0
        self.episode_number = 0
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.episode_rewards = deque(maxlen=10)
        self.critic_updates = 0
        self.actor_updates = 0
        self.reward_stats = ReplayRewardStats()
        self.window = TrainingWindow()

        if resume is None:
            self.obs, self.reset_info = self.env.reset(seed=self.seed)
        obs_dim = self.env.obs_dim_per_task
        act_dim = self.env.agent_output_act_dim
        if self.env.single_action_mode or self.env.disable_defer_action or (obs_dim, act_dim) != (34, 6):
            raise ValueError("Training requires 34 features, six actions, multi-action mode and Defer")
        self.learner = SAC(settings, device, obs_dim, act_dim)
        self.replay = ReplayBuffer(settings["replay_buffer_size"], settings["max_tasks"], obs_dim, act_dim)
        if resume is not None:
            self.restore(resume)

    def restore(self, state):
        saved_env = cloudpickle.loads(state["environment"])
        saved_env.cluster_manager.tasks = self.env.cluster_manager.tasks
        self.env = saved_env
        self.learner.load_state_dict(state["learner"])
        self.replay = state["replay"]
        self.reward_stats.__dict__.update(state["reward_stats"])
        self.window.__dict__.update(state["window"])
        self.episode_rewards.extend(state["episode_rewards"])
        for name, value in state["progress"].items():
            setattr(self, name, value)
        restore_rng(state["rng"])

    def model(self, kind):
        return {
            "schema_version": 1,
            "kind": kind,
            "global_step": self.global_step,
            "obs_dim": 34,
            "act_dim": 6,
            "hidden_dim": self.settings["hidden_dim"],
            "use_attention": False,
            "use_layer_norm": False,
            "actor_state_dict": self.learner.actor.state_dict(),
            "critic_state_dict": self.learner.critic.state_dict(),
            **self.provenance,
        }

    def save_resume(self, storage):
        self.window.averages()
        progress = {
            name: getattr(self, name) for name in (
                "obs", "reset_info", "global_step", "episode_number", "episode_steps", "episode_reward",
                "best_score", "best_model", "actor_updates", "critic_updates",
            )
        }
        storage.save_resume({
            "schema_version": 1,
            "provenance": self.provenance,
            "settings": self.settings,
            "progress": progress,
            "learner": self.learner.state_dict(),
            "replay": self.replay,
            "reward_stats": vars(self.reward_stats),
            "window": vars(self.window),
            "episode_rewards": list(self.episode_rewards),
            "environment": serialize_environment(self.env),
            "rng": rng_state(),
        })

    def step(self, storage, writer=None):
        step_index = self.global_step
        if not len(self.obs):
            actions = []
        elif step_index < self.settings["warmup_steps"]:
            actions = np.random.randint(6, size=len(self.obs)).tolist()
        else:
            with torch.no_grad():
                obs = torch.as_tensor(np.asarray(self.obs), dtype=torch.float32, device=self.device)
                logits = self.learner.actor(obs)
                actions = torch.distributions.Categorical(probs=logits.softmax(-1)).sample().cpu().tolist()

        next_obs, reward, done, truncated, info = self.env.step(actions)
        terminal = done or truncated
        reward = float(reward)
        if not np.isfinite(reward):
            raise FloatingPointError(f"Nonfinite environment reward at step {self.global_step + 1}")
        normalized_reward = self.reward_stats.update_and_normalize(reward)
        components = dict(self.env.reward_fn.get_last_components())
        if actions:
            self.replay.add(self.obs, actions, normalized_reward, next_obs, terminal)
        self.obs = next_obs
        self.global_step += 1
        self.episode_steps += 1
        self.episode_reward += reward
        if terminal:
            # The original reset reseeds NumPy before the terminal replay draw.
            self.finish_episode(storage, writer)

        ready = (
            step_index >= self.settings["warmup_steps"]
            and step_index % self.settings["update_frequency"] == 0
            and len(self.replay) >= self.settings["batch_size"]
        )
        losses = {}
        if ready:
            batch = self.replay.sample(self.settings["batch_size"], self.device)
            losses = self.learner.update(batch, step_index)
            self.critic_updates += 1
            self.actor_updates += int("actor_loss" in losses)

        self.window.record(reward, components, losses, info)

    def finish_episode(self, storage, writer):
        self.episode_number += 1
        average = self.episode_reward / self.episode_steps
        self.episode_rewards.append(average)
        last10 = float(np.mean(self.episode_rewards)) if len(self.episode_rewards) == 10 else None
        storage.record(
            "train_episodes.csv",
            episode=self.episode_number,
            start_step=self.global_step - self.episode_steps + 1,
            end_step=self.global_step,
            total_reward=self.episode_reward,
            mean_reward=average,
            mean_reward_last10=last10,
            episode_length=self.episode_steps,
            completed_at_utc=utc_now(),
            reset_seed=self.reset_info["reset_seed"],
            start_time_utc=self.reset_info["episode_start_utc"],
            workload_start_time_utc=self.reset_info["workload_start_utc"],
            dc_source_indices=json.dumps(self.reset_info["dc_source_indices"], sort_keys=True),
        )
        if writer:
            writer.add_scalar("reward/episode_mean", average, self.global_step)
            if last10 is not None:
                writer.add_scalar("reward/last10_mean", last10, self.global_step)
        # Baseline uses seed + zero-based step index after a terminal step.
        self.obs, self.reset_info = self.env.reset(seed=self.seed + self.global_step - 1)
        self.episode_steps = 0
        self.episode_reward = 0.0
        if last10 is not None and last10 > self.best_score:
            self.best_score = last10
            path, _ = storage.publish_model(self.model("best"), f"best_step_{self.global_step:08d}.pt")
            self.best_model = path.name
            storage.repair_best_alias(self.best_model)

    def log_window(self, storage, writer):
        window = self.window
        if not window.steps:
            return
        averages = window.averages()
        common = {"global_step": self.global_step, "interval_steps": window.steps}
        mean_reward = window.total_reward / window.steps
        operational = {
            "mean_total_cost_usd": window.total_cost_usd / window.steps,
            "mean_total_carbon_kg": window.total_carbon_kg / window.steps,
            "sla_violated": window.sla_violated,
            "tasks_finished": window.tasks_finished,
            "sla_violation_pct_finished": 100 * window.sla_violated / window.tasks_finished
            if window.tasks_finished else None,
        }
        storage.record(
            "train_steps.csv",
            **common,
            mean_reward=mean_reward,
            **averages,
            **operational,
            critic_updates=self.critic_updates,
            actor_updates=self.actor_updates,
            interval_critic_updates=window.loss_counts.get("critic_loss", 0),
            interval_actor_updates=window.loss_counts.get("actor_loss", 0),
            num_overflow_steps=self.replay.overflow_steps,
            max_tasks_seen=self.replay.max_tasks_seen,
        )
        components = {name: value / window.steps for name, value in window.components.items()}
        storage.record("train_reward_components.csv", **common, **components)
        if writer:
            writer.add_scalar("reward/window_mean", mean_reward, self.global_step)
            for name, val in averages.items():
                writer.add_scalar(f"SAC/{name}", val, self.global_step)
            for name, val in components.items():
                writer.add_scalar(f"reward_components/{name}", val, self.global_step)
            for name, val in operational.items():
                if val is not None:
                    writer.add_scalar(f"operations/{name}", val, self.global_step)
        print(f"step {self.global_step}/{self.settings['total_steps']}  mean reward {mean_reward:.6g}"
              f"  critic updates {self.critic_updates}  replay overflow steps {self.replay.overflow_steps}",
              flush=True)
        self.window = TrainingWindow()


def metadata(config, algorithm, device, smoke):
    return {
        "training_run_id": uuid.uuid4().hex,
        "training_seed": int(config.algorithm["training_seed"]),
        "smoke": smoke,
        "total_steps": algorithm["total_steps"],
        "config": {
            "sim": config.sim,
            "algorithm": config.algorithm,
            "reward": config.reward,
            "datacenters": config.datacenters,
            "manifest": config.manifest,
        },
        "effective_algorithm": algorithm,
        "input_hashes": input_hashes(config),
        "git": git_metadata(),
        "runtime": runtime_metadata(device),
        "random_configuration": {
            "training_seed": int(config.algorithm["training_seed"]),
            "reset_seed_rule": "initial seed; subsequent resets use seed plus zero-based terminal global step",
            "python_numpy_torch_cuda_seeded": True,
        },
        "replay_reward_preprocessing": "baseline running scalar mean and variance, updated before normalization",
        "replay_target_semantics": "baseline same-position next task bootstrap with fixed alpha",
        "started_at_utc": utc_now(),
    }


def read_resume(path, output, provenance, settings):
    if path.resolve() != (output / "checkpoints" / "resume_state.pkl").resolve():
        raise ValueError("Resume requires checkpoints/resume_state.pkl in the same output directory as its CSV logs")
    # This pickle is for this trainer's own local state, never an external model.
    with path.open("rb") as stream:
        state = pickle.load(stream)
    if state["schema_version"] != 1:
        raise ValueError("Unsupported training resume schema")
    saved = state["provenance"]
    for key in ("training_seed", "smoke", "total_steps", "config", "input_hashes", "git", "runtime"):
        if saved[key] != provenance[key]:
            raise ValueError(f"Exact resume requires the same {key}")
    if state["settings"] != settings:
        raise ValueError("Exact resume requires identical training settings")
    return state


def train(args):
    config = load_config(args.config_dir, "train")
    algorithm = dict(config.algorithm)
    output = Path(args.output_dir).resolve()
    if args.smoke_steps:
        if output == Path(DEFAULT_OUTPUT).resolve():
            raise ValueError("Short smoke training needs a separate --output-dir")
        algorithm.update(
            total_steps=args.smoke_steps,
            warmup_steps=0,
            batch_size=min(algorithm["batch_size"], 4),
            replay_buffer_size=min(algorithm["replay_buffer_size"], 64),
            save_interval=min(algorithm["save_interval"], 4),
            log_interval=min(algorithm["log_interval"], 2),
        )
    target = algorithm["total_steps"]
    stop_step = args.stop_step if args.stop_step is not None else target
    if not 0 < stop_step <= target:
        raise ValueError(f"--stop-step must be between 1 and {target}")
    output.mkdir(parents=True, exist_ok=True)

    device = resolve_device(algorithm["device"])
    if args.smoke_steps:
        torch.set_num_threads(1)
    seed_everything(int(algorithm["training_seed"]))
    with writer_lock(output):
        provenance = metadata(config, algorithm, device, bool(args.smoke_steps))
        state = read_resume(args.resume, output, provenance, algorithm) if args.resume else None
        already_complete = False
        if state is not None:
            provenance = state["provenance"]
            saved_step = state["progress"]["global_step"]
            already_complete = saved_step == target == stop_step
            if saved_step >= stop_step and not already_complete:
                raise ValueError("The requested stop step has already been reached")
        offsets = state["csv_offsets"] if state else None
        environment = make_env(config)
        trainer = Trainer(config, environment, device, provenance, algorithm, state)
        if state is None:
            provenance["initial_reset"] = trainer.reset_info
        del state, environment
        storage = TrainingStorage(output, offsets)
        writer = None
        stop_requested = False

        def stop_gracefully(signum, frame):
            nonlocal stop_requested
            stop_requested = True

        signals = (signal.SIGINT, signal.SIGTERM)
        handlers = {}
        try:
            for signum in signals:
                handlers[signum] = signal.signal(signum, stop_gracefully)
            if args.resume:
                storage.repair_best_alias(trainer.best_model)
            else:
                snapshot_config(config, output / "training_config_snapshot")
            if config.sim["use_tensorboard"]:
                from torch.utils.tensorboard import SummaryWriter

                writer = SummaryWriter(
                    log_dir=str(output / "tensorboard"),
                    purge_step=trainer.global_step + 1 if args.resume else None,
                )
            write_json(output / "training_metadata.json", {
                **provenance, "status": "running", "global_step": trainer.global_step,
            })
            if not args.resume:
                trainer.save_resume(storage)

            while trainer.global_step < stop_step and not stop_requested:
                trainer.step(storage, writer)
                completed = trainer.global_step
                if completed % algorithm["log_interval"] == 0:
                    trainer.log_window(storage, writer)
                if completed % algorithm["save_interval"] == 0 and completed < target:
                    storage.publish_model(trainer.model("periodic"), f"step_{completed:08d}.pt")
                    trainer.save_resume(storage)

            complete = trainer.global_step == target
            result = {
                **provenance,
                "status": "complete" if complete else "paused",
                "global_step": trainer.global_step,
                "episodes_completed": trainer.episode_number,
                "current_episode_reset": trainer.reset_info,
                "critic_updates": trainer.critic_updates,
                "actor_updates": trainer.actor_updates,
                "overflow_steps": trainer.replay.overflow_steps,
                "max_tasks_seen": trainer.replay.max_tasks_seen,
                "best_checkpoint": f"checkpoints/{trainer.best_model}" if trainer.best_model else None,
                "ended_at_utc": utc_now(),
            }
            if complete:
                trainer.log_window(storage, writer)
                if already_complete:
                    path, digest = storage.final_model(trainer.global_step)
                else:
                    path, digest = storage.publish_model(trainer.model("final"), "final.pt")
                result["final_checkpoint"] = path.relative_to(output).as_posix()
                result["final_checkpoint_sha256"] = digest
            trainer.save_resume(storage)
            write_json(output / "training_metadata.json", result)
            print(f"Training {result['status']} at {trainer.global_step}; artifacts in {output}", flush=True)
        finally:
            storage.close()
            if writer:
                writer.close()
            for signum, handler in handlers.items():
                signal.signal(signum, handler)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--resume", type=Path, help="Resume a trusted checkpoints/resume_state.pkl in the same output")
    parser.add_argument("--stop-step", type=int, help="Pause on this absolute completed step and save resumable state")
    parser.add_argument("--smoke-steps", type=int, help="Override the run for a short smoke test in a separate output")
    args = parser.parse_args(argv)
    if args.smoke_steps is not None and not 1 <= args.smoke_steps <= 1000:
        parser.error("--smoke-steps must be between 1 and 1000")
    return args


if __name__ == "__main__":
    train(parse_args())
