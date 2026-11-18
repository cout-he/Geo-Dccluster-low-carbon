"""Table 3 aggregation from simulator measurements in their native units."""

from itertools import combinations
from math import fsum, isfinite
from statistics import mean, stdev


MODES = ("geo_only", "time_only", "geo_time")
MODE_LABELS = dict(zip(MODES, ("Geo Only", "Time Only", "Geo+Time")))
METRICS = {
    "total_cost_usd": ("Total cost", "USD"),
    "total_co2_t": ("Total CO₂", "t"),
    "total_energy_mwh": ("Total energy", "MWh"),
    "total_water_m3": ("Total water", "m³"),
    "sla_violation_pct_finished": ("SLA violations", "%"),
    "avg_cpu_util_pct": ("Mean CPU utilization", "%"),
    "avg_gpu_util_pct": ("Mean GPU utilization", "%"),
    "transmission_cost_usd": ("Transmission cost", "USD"),
    "tasks_deferred": ("Tasks deferred", "actions"),
}

# The printed Table 3 values are authoritative for this comparison. The main
# prose's claim of zero Geo Only SLA violations conflicts with the table.
PAPER_TABLE3 = {
    "geo_only": (
        (92618, 4139), (308, 7.4), (1037, 2.2), (7247, 67), (25.42, 0.28),
        (4.7, 0.0), (6.6, 0.3), (4131, 38), (0, 0),
    ),
    "time_only": (
        (101346, 5185), (327, 7.2), (1086, 2.0), (7394, 64), (23.12, 0.05),
        (3.4, 0.0), (11.4, 0.2), (0, 0), (432, 171),
    ),
    "geo_time": (
        (92401, 4134), (308, 7.6), (1037, 2.1), (7253, 66), (25.47, 0.30),
        (5.0, 0.0), (6.5, 0.3), (4203, 37), (474, 165),
    ),
}

KEYS = ("mode", "seed", "step", "timestamp_utc")
ACTION_COUNT_KEYS = (
    "new_arrivals", "decisions", "defer_actions", "unique_tasks_deferred",
    "cross_dc_assignments", "deadline_overrides",
)
DC_SUM_KEYS = (
    "dc_energy_kwh", "electricity_cost_usd", "dc_carbon_kg", "water_l",
    "tasks_started", "cross_dc_started", "running_tasks", "pending_tasks",
    "tasks_finished", "sla_met", "sla_violated",
)
UTIL_KEYS = ("cpu_util_pct", "gpu_util_pct", "mem_util_pct")
TRANSMISSION_KEYS = (
    "transmission_cost_usd", "transmission_energy_kwh", "transmission_carbon_kg",
)
TOTAL_KEYS = ("total_cost_usd", "total_carbon_kg", "total_energy_kwh")
DC_STEP_FIELDS = KEYS + ("dc", "dc_id", "dc_location", "manager_time") + DC_SUM_KEYS + UTIL_KEYS
STEP_FIELDS = (
    KEYS + DC_SUM_KEYS + UTIL_KEYS + TRANSMISSION_KEYS + TOTAL_KEYS
    + ACTION_COUNT_KEYS + ("in_transit", "reward")
)
PER_SEED_FIELDS = (
    ("mode", "seed", "steps") + tuple(METRICS)
    + ("electricity_cost_usd", "dc_carbon_kg", "transmission_carbon_kg",
       "dc_energy_kwh", "transmission_energy_kwh", "new_arrivals", "tasks_finished",
       "unfinished_tasks", "sla_met", "sla_violated", "unique_tasks_deferred",
       "cross_dc_assignments", "cross_dc_started", "deadline_overrides")
)
AGG_FIELDS = ("mode", "metric", "unit", "n", "mean", "std")
PAIRED_FIELDS = ("mode_a", "mode_b", "seed", "metric", "unit", "value_a", "value_b", "difference_a_minus_b")
COMPARISON_FIELDS = (
    "mode", "metric", "unit", "paper_mean", "paper_std", "repro_mean", "repro_std",
    "absolute_difference", "relative_difference",
)


def percentage(numerator, denominator):
    """Leave rates undefined for windows with no qualifying tasks."""
    return 100.0 * numerator / denominator if denominator else None


def build_step_metrics(info, metadata, actions, *, in_transit, reward, datacenters=None):
    dc_rows = []
    for name, values in info["datacenter_infos"].items():
        common = values["__common__"]
        dc = datacenters[name] if datacenters is not None else None
        row = {
            **metadata,
            "dc": name,
            "dc_id": dc.dc_id if dc is not None else None,
            "dc_location": dc.location if dc is not None else None,
            "manager_time": common.get("time"),
            "dc_energy_kwh": float(common["energy_consumption_kwh"]),
            "electricity_cost_usd": float(common["energy_cost_USD"]),
            "dc_carbon_kg": float(common["carbon_emissions_kg"]),
            "water_l": float(values["agent_dc"]["dc_water_usage"]),
            "tasks_started": int(common["tasks_assigned"]),
            "cross_dc_started": sum(
                task.dest_dc_id != task.origin_dc_id for task in common["routed_tasks_this_step"]
            ),
            "running_tasks": int(common["running_tasks"]),
            "pending_tasks": int(common["pending_tasks"]),
            "tasks_finished": int(common["finished_tasks_count"]),
            "sla_met": int(common["__sla__"]["met"]),
            "sla_violated": int(common["__sla__"]["violated"]),
            "cpu_util_pct": float(common["cpu_util_percent"]),
            "gpu_util_pct": float(common["gpu_util_percent"]),
            "mem_util_pct": float(common["mem_util_percent"]),
        }
        if row["tasks_finished"] != row["sla_met"] + row["sla_violated"]:
            raise ValueError(f"Finished tasks and SLA counts disagree at {metadata} in {name}")
        if row["cross_dc_started"] > row["tasks_started"]:
            raise ValueError(f"Cross-DC starts exceed all starts at {metadata} in {name}")
        dc_rows.append(row)

    if not dc_rows:
        raise ValueError(f"No data centers reported metrics at {metadata}")
    step = {
        **metadata,
        **{key: fsum(row[key] for row in dc_rows) for key in DC_SUM_KEYS[:4]},
        **{key: sum(row[key] for row in dc_rows) for key in DC_SUM_KEYS[4:]},
        **{key: mean(row[key] for row in dc_rows) for key in UTIL_KEYS},
        "transmission_cost_usd": float(info["transmission_cost_total_usd"]),
        "transmission_energy_kwh": float(info["transmission_energy_total_kwh"]),
        "transmission_carbon_kg": float(info["transmission_emissions_total_kg"]),
        **actions,
        "in_transit": in_transit,
        "reward": float(reward),
    }
    step["total_cost_usd"] = step["electricity_cost_usd"] + step["transmission_cost_usd"]
    step["total_carbon_kg"] = step["dc_carbon_kg"] + step["transmission_carbon_kg"]
    step["total_energy_kwh"] = step["dc_energy_kwh"] + step["transmission_energy_kwh"]
    if not all(isfinite(step[key]) for key in TOTAL_KEYS + UTIL_KEYS + ("water_l", "reward")):
        raise ValueError(f"Non-finite simulator metrics at {metadata}")
    return step, dc_rows


def summarize_run(steps):
    if not steps:
        raise ValueError("Cannot summarize a run without simulated steps")

    total = {key: fsum(row[key] for row in steps) for key in
             DC_SUM_KEYS[:4] + TRANSMISSION_KEYS + TOTAL_KEYS}
    counts = {key: sum(row[key] for row in steps) for key in
              ACTION_COUNT_KEYS + ("sla_met", "sla_violated", "tasks_finished", "cross_dc_started")}
    unfinished = counts["new_arrivals"] - counts["tasks_finished"]
    if unfinished < 0:
        raise ValueError("More tasks finished than arrived")
    return {
        "mode": steps[0]["mode"],
        "seed": steps[0]["seed"],
        "steps": len(steps),
        "total_cost_usd": total["total_cost_usd"],
        "total_co2_t": total["total_carbon_kg"] / 1000,
        "total_energy_mwh": total["total_energy_kwh"] / 1000,
        "total_water_m3": total["water_l"] / 1000,
        "sla_violation_pct_finished": percentage(counts["sla_violated"], counts["tasks_finished"]),
        "avg_cpu_util_pct": mean(row["cpu_util_pct"] for row in steps),
        "avg_gpu_util_pct": mean(row["gpu_util_pct"] for row in steps),
        "transmission_cost_usd": total["transmission_cost_usd"],
        "tasks_deferred": counts["defer_actions"],
        "unfinished_tasks": unfinished,
        **{key: total[key] for key in ("electricity_cost_usd", "dc_carbon_kg",
                                      "transmission_carbon_kg", "dc_energy_kwh",
                                      "transmission_energy_kwh")},
        **{key: counts[key] for key in ("new_arrivals", "tasks_finished", "sla_met",
                                       "sla_violated", "unique_tasks_deferred",
                                       "cross_dc_assignments", "cross_dc_started",
                                       "deadline_overrides")},
    }


def summarize_seeds(rows):
    by_run = {(row["mode"], row["seed"]): row for row in rows}
    seeds = sorted({row["seed"] for row in rows})
    if (not seeds or len(by_run) != len(rows)
            or set(by_run) != {(mode, seed) for mode in MODES for seed in seeds}):
        raise ValueError("Expected one paired run for every mode and seed")

    aggregate = []
    differences = []
    for metric, (_, unit) in METRICS.items():
        for mode in MODES:
            values = [by_run[mode, seed][metric] for seed in seeds]
            valid = [value for value in values if value is not None]
            aggregate.append({
                "mode": mode, "metric": metric, "unit": unit, "n": len(valid),
                "mean": mean(valid) if valid else None,
                "std": stdev(valid) if len(valid) > 1 else None,
            })

        for mode_a, mode_b in combinations(MODES, 2):
            for seed in seeds:
                a, b = by_run[mode_a, seed][metric], by_run[mode_b, seed][metric]
                differences.append({
                    "mode_a": mode_a, "mode_b": mode_b, "seed": seed, "metric": metric,
                    "unit": unit, "value_a": a, "value_b": b,
                    "difference_a_minus_b": a - b if a is not None and b is not None else None,
                })
    return aggregate, differences


def paper_comparison(aggregate, *, smoke):
    paper = {
        (mode, metric): result
        for mode, pairs in PAPER_TABLE3.items()
        for metric, result in zip(METRICS, pairs, strict=True)
    }
    rows = []
    for row in aggregate:
        mode, metric = row["mode"], row["metric"]
        paper_mean, paper_std = paper[mode, metric]
        repro_mean = None if smoke else row["mean"]
        rows.append({
            "mode": mode, "metric": metric, "unit": row["unit"],
            "paper_mean": paper_mean, "paper_std": paper_std,
            "repro_mean": repro_mean, "repro_std": None if smoke else row["std"],
            "absolute_difference": abs(repro_mean - paper_mean) if repro_mean is not None else None,
            "relative_difference": (repro_mean - paper_mean) / abs(paper_mean)
            if repro_mean is not None and paper_mean else None,
        })
    return rows
