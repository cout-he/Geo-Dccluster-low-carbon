import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import pytest
import torch

from scripts.repro_table3.evaluate import ActionTracker, sample_policy, validate_checkpoint
from scripts.repro_table3.metrics import (
    MODES,
    build_step_metrics,
    paper_comparison,
    summarize_run,
    summarize_seeds,
)


def test_geo_only_masks_and_renormalizes_without_argmax():
    torch.manual_seed(19)
    probabilities = torch.tensor([[0.80, 0.04, 0.07, 0.02, 0.05, 0.02]])
    logits = probabilities.log().expand(12000, -1)
    raw, selected = sample_policy(logits, [1] * len(logits), "geo_only")
    assert (raw == 0).any()
    assert (selected > 0).all()
    assert torch.equal(raw[raw > 0], selected[raw > 0])
    empirical = torch.bincount(selected, minlength=6)[1:] / len(selected)
    expected = probabilities[0, 1:] / probabilities[0, 1:].sum()
    torch.testing.assert_close(empirical, expected, rtol=0, atol=0.018)


def test_time_only_maps_to_one_based_origin_and_preserves_defer():
    logits = torch.full((6, 6), -torch.inf)
    logits.fill_diagonal_(0)
    origins = [3, 5, 1, 2, 5, 1]
    raw, selected = sample_policy(logits, origins, "time_only")
    assert raw.tolist() == [0, 1, 2, 3, 4, 5]
    assert selected.tolist() == [0, *origins[1:]]
    raw, selected = sample_policy(logits, origins, "geo_time")
    assert torch.equal(raw, selected)
    with pytest.raises(ValueError, match="origin"):
        sample_policy(logits, [0] * len(logits), "time_only")


@dataclass(eq=False)
class Task:
    job_name: str
    origin_dc_id: int
    sla_deadline: datetime
    dest_dc_id: int | None = None


def test_action_counts_distinguish_repeated_defers_and_deadline_override():
    now = datetime(2023, 7, 1)
    tasks = [Task("same-name", 2, now + timedelta(minutes=10))]
    tracker = ActionTracker("geo_time", seed=0)
    first, counts = tracker.record(1, now, tasks, [0], [0], carried_after=tasks)
    assert counts["defer_actions"] == counts["unique_tasks_deferred"] == 1
    task = tasks[0]
    again, counts = tracker.record(2, now, tasks, [0], [0], carried_after=tasks)
    assert again[0]["task_id"] == first[0]["task_id"]
    assert counts["defer_actions"] == 1
    assert counts["unique_tasks_deferred"] == 0
    task.dest_dc_id = 2
    expired, counts = tracker.record(
        3, now + timedelta(minutes=15), tasks, [0], [0], carried_after=[]
    )
    assert expired[0]["effective_action"] == 2
    assert expired[0]["forced_origin"] is True
    assert counts["defer_actions"] == counts["cross_dc_assignments"] == 0


def dc_info(energy, cost, carbon, water, cpu, gpu, met, violated):
    return {
        "agent_dc": {"dc_water_usage": water},
        "__common__": {
            "energy_consumption_kwh": energy,
            "energy_cost_USD": cost,
            "carbon_emissions_kg": carbon,
            "cpu_util_percent": cpu,
            "gpu_util_percent": gpu,
            "mem_util_percent": 3.0,
            "pending_tasks": 6,
            "running_tasks": 12,
            "tasks_assigned": 2,
            "routed_tasks_this_step": [
                Task("remote", 2, datetime(2023, 7, 1), dest_dc_id=1)
            ],
            "finished_tasks_count": met + violated,
            "__sla__": {"met": met, "violated": violated},
        },
    }


def test_step_and_run_metrics_aggregate_source_units_once():
    info = {
        "datacenter_infos": {
            "a": dc_info(1200, 32, 180, 2000, 3, 4, 3, 1),
            "b": dc_info(1300, 10, 210, 5000, 9, 8, 0, 0),
        },
        "transmission_energy_total_kwh": 5,
        "transmission_emissions_total_kg": 10,
        "transmission_cost_total_usd": 4,
    }
    metadata = {"mode": "geo_time", "seed": 0, "step": 1, "timestamp_utc": "2023-07-01T00:00:00"}
    actions = {
        "new_arrivals": 6,
        "decisions": 6,
        "defer_actions": 3,
        "unique_tasks_deferred": 2,
        "cross_dc_assignments": 2,
        "deadline_overrides": 0,
    }
    step, dc_steps = build_step_metrics(info, metadata, actions, in_transit=1, reward=-1.5)
    assert sum(row["electricity_cost_usd"] for row in dc_steps) == 42
    assert step["total_cost_usd"] == 46
    assert step["total_carbon_kg"] == 400
    assert step["total_energy_kwh"] == 2505
    assert step["water_l"] == 7000
    assert step["cross_dc_started"] == 2
    assert step["cpu_util_pct"] == 6
    assert step["gpu_util_pct"] == 6
    summary = summarize_run([step, {**step, "step": 2}])
    assert summary["total_cost_usd"] == 92
    assert summary["total_co2_t"] == 0.8
    assert summary["total_energy_mwh"] == 5.01
    assert summary["total_water_m3"] == 14
    assert summary["sla_violation_pct_finished"] == 25
    assert summary["tasks_deferred"] == 6
    assert summary["unique_tasks_deferred"] == 4
    assert summary["tasks_finished"] == 8
    assert summary["unfinished_tasks"] == 4

    seed_runs = [
        {**summary, "mode": mode, "seed": seed, "total_cost_usd": 92 + 2 * seed + off}
        for off, mode in enumerate(MODES)
        for seed in range(2)
    ]
    aggregate, differences = summarize_seeds(seed_runs)
    cost = next(row for row in aggregate if row["mode"] == "geo_only" and row["metric"] == "total_cost_usd")
    assert cost["mean"] == 93
    assert cost["std"] == pytest.approx(math.sqrt(2))
    assert next(row for row in differences if row["metric"] == "total_cost_usd")["difference_a_minus_b"] == -1
    comparison = paper_comparison(aggregate, smoke=False)
    assert next(row for row in comparison if row["mode"] == "geo_only" and row["metric"] == "sla_violation_pct_finished")["paper_mean"] == 25.42
    assert next(row for row in comparison if row["mode"] == "time_only" and row["metric"] == "transmission_cost_usd")["relative_difference"] is None
    assert all(row["absolute_difference"] is None for row in paper_comparison(aggregate, smoke=True))
    with pytest.raises(ValueError, match="paired"):
        summarize_seeds(seed_runs[:-1])



def test_plot_cli_exports_training_and_evaluation_from_csv(tmp_path):
    import json

    import pandas as pd

    from scripts.repro_table3.metrics import METRICS
    from scripts.repro_table3.plot_results import main

    tables = {
        "train_episodes": [
            {"end_step": step, "mean_reward": -10 / step, "mean_reward_last10": -10 / step}
            for step in (4, 8)
        ],
        "train_steps": [
            {"global_step": step, "actor_loss": -10 / step, "critic_loss": 2 / step, "entropy": 1.3,
             "mean_total_cost_usd": 8.3, "mean_total_carbon_kg": 13, "sla_violation_pct_finished": 25}
            for step in (1, 8)
        ],
        "train_reward_components": [
            {"global_step": step, **dict.fromkeys(
                ("energy_price", "carbon_emissions", "transmission_cost", "sla_penalty",
                 "transmission_emissions"), -0.2
            )}
            for step in (1, 8)
        ],
        "metrics_mean_std": [
            {"mode": mode, "metric": metric, "unit": unit, "mean": 2.0, "std": 0.5}
            for mode in MODES
            for metric, (_, unit) in METRICS.items()
        ],
        "step_metrics": [
            {"mode": mode, "seed": seed, "step": step, "total_cost_usd": 100,
             "total_carbon_kg": 150}
            for mode in MODES
            for seed in (0, 1)
            for step in (1, 2)
        ],
        "dc_step_metrics": [
            {"mode": mode, "seed": seed, "step": step, "dc_id": dc, "dc_location": f"DC {dc}",
             "cpu_util_pct": 10, "gpu_util_pct": 3}
            for mode in MODES
            for seed in (0, 1)
            for step in (1, 2)
            for dc in range(1, 6)
        ],
        "action_statistics": [
            {"mode": mode, "seed": seed, "kind": kind, "action": action, "count": 20}
            for mode in MODES
            for seed in (0, 1)
            for kind in ("raw_action", "effective_action")
            for action in range(6)
        ],
        "paired_differences": [
            {"mode_a": a, "mode_b": b, "seed": seed, "metric": metric, "difference_a_minus_b": 0.5}
            for a, b in (("geo_only", "time_only"), ("geo_only", "geo_time"), ("time_only", "geo_time"))
            for seed in (0, 1)
            for metric in METRICS
        ],
    }
    tables["paper_table3_comparison"] = paper_comparison(
        [{**row, "n": 2} for row in tables["metrics_mean_std"]], smoke=False,
    )
    for filename, rows in tables.items():
        pd.DataFrame(rows).to_csv(tmp_path / f"{filename}.csv", index=False)
    (tmp_path / "eval_manifest.json").write_text(
        json.dumps({"timestep_minutes": 15, "smoke": False}), encoding="utf-8",
    )

    main(["--output-dir", str(tmp_path), "--dpi", "75"])
    plots = tmp_path / "figures"
    expected = {
        "train_episode_reward", "train_losses", "train_entropy", "train_reward_components",
        "train_operational_metrics",
        "eval_nine_metrics", "eval_cost", "eval_carbon", "eval_sla", "eval_cumulative",
        "eval_dc_utilization", "eval_action_distribution", "eval_paired_differences", "eval_paper_table3",
    }
    assert {path.stem for path in plots.glob("*.png")} == expected
    assert {path.stem for path in plots.glob("*.pdf")} == expected
    assert all(path.stat().st_size > 1000 for path in plots.iterdir())


def test_formal_evaluation_requires_metadata_from_completed_training(tmp_path):
    import json
    from types import SimpleNamespace

    from scripts.repro_table3.common import sha256

    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    best, final = (checkpoints / f"{kind}.pt" for kind in ("best", "final"))
    hashes = {"workload": "hash"}
    config = SimpleNamespace(algorithm={"total_steps": 10_000_000}, datacenters=[], reward={}, manifest={})
    model = {
        "schema_version": 1, "obs_dim": 34, "act_dim": 6, "hidden_dim": 64,
        "config": vars(config), "input_hashes": hashes, "smoke": False,
        "total_steps": 10_000_000, "training_run_id": "new-run",
    }
    torch.save({**model, "global_step": 5_000_000, "kind": "best"}, best)
    torch.save({**model, "global_step": 10_000_000, "kind": "final"}, final)
    metadata = tmp_path / "training_metadata.json"
    training = {
        "status": "running", "training_run_id": "new-run",
        "global_step": 10_000_000, "final_checkpoint_sha256": sha256(final),
    }
    metadata.write_text(json.dumps(training), encoding="utf-8")
    with pytest.raises(ValueError, match="same completed training run"):
        validate_checkpoint(best, config, hashes, smoke=False)
    metadata.write_text(json.dumps({**training, "status": "complete"}), encoding="utf-8")
    assert validate_checkpoint(best, config, hashes, smoke=False)["kind"] == "best"
    with pytest.raises(ValueError, match="Input file hashes"):
        validate_checkpoint(best, config, {"workload": "different"}, smoke=False)
