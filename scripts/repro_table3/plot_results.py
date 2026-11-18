"""Render Table 3 training and evaluation figures from saved experiment files."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scripts.repro_table3.common import DEFAULT_OUTPUT
from scripts.repro_table3.metrics import METRICS, MODES, MODE_LABELS


COLORS = dict(zip(MODES, ("#0072B2", "#D55E00", "#009E73")))
LINE_STYLES = dict(zip(MODES, ("-", "--", "-.")))
REWARD_COMPONENTS = {
    "energy_price": "Electricity cost",
    "carbon_emissions": "DC emissions",
    "transmission_cost": "Transmission cost",
    "sla_penalty": "SLA penalty",
    "transmission_emissions": "Transmission emissions",
}
KEY_METRICS = {
    "total_cost_usd": "cost",
    "total_co2_t": "carbon",
    "sla_violation_pct_finished": "sla",
}
OPTIONAL_TRAIN_METRICS = {
    "mean_total_cost_usd": "Mean total cost (USD)",
    "mean_total_carbon_kg": "Mean CO₂ (kg)",
    "sla_violation_pct_finished": "SLA violations (%)",
}


def save_figure(fig, target, name, dpi):
    try:
        fig.savefig(target / f"{name}.png", dpi=dpi)
        fig.savefig(target / f"{name}.pdf")
    finally:
        plt.close(fig)


def read_table(source, name):
    return pd.read_csv(source / f"{name}.csv")


def time_lines(frame, columns, target, name, dpi):
    fig, axes = plt.subplots(len(columns), 1, figsize=(7, 1.68 * len(columns) + 0.55),
                             sharex=True, squeeze=False, layout="constrained")
    for ax, (column, ylabel) in zip(axes.flat, columns.items(), strict=True):
        if frame[column].notna().any():
            ax.plot(frame["global_step"], frame[column], color="#0072B2", linewidth=1.2,
                    marker="." if frame[column].count() < 15 else None)
        else:
            ax.text(0.5, 0.5, "No observations recorded", transform=ax.transAxes,
                    va="center", ha="center", color="#444444")
        ax.set_ylabel(ylabel)
    axes[-1, 0].set_xlabel("Global training steps")
    save_figure(fig, target, name, dpi)


def plot_training(source, target, dpi):
    episodes = read_table(source, "train_episodes").sort_values("end_step")
    steps = read_table(source, "train_steps").sort_values("global_step")
    rewards = read_table(source, "train_reward_components").sort_values("global_step")

    fig, ax = plt.subplots(figsize=(7, 3.1), layout="constrained")
    if episodes.empty:
        ax.text(0.5, 0.5, "No episode completed in this run", transform=ax.transAxes,
                va="center", ha="center", color="#444444")
        if not steps.empty:
            ax.set_xlim(0, max(steps["global_step"]))
    else:
        ax.plot(episodes["end_step"], episodes["mean_reward"], label="Episode mean",
                color="#8AB6CF", marker=".", markersize=3, linewidth=1)
        ax.plot(episodes["end_step"], episodes["mean_reward_last10"], label="Last 10 episodes",
                color="#14537B", marker=".", markersize=3, linewidth=1.5)
        ax.legend(loc="best")
    ax.set(xlabel="Global training steps", ylabel="Mean episode reward")
    save_figure(fig, target, "train_episode_reward", dpi)
    time_lines(steps, {"actor_loss": "Actor loss", "critic_loss": "Critic loss"},
               target, "train_losses", dpi)
    time_lines(steps, {"entropy": "Policy entropy (nats)"}, target, "train_entropy", dpi)
    time_lines(rewards, REWARD_COMPONENTS, target, "train_reward_components", dpi)
    optional = {key: label for key, label in OPTIONAL_TRAIN_METRICS.items() if key in steps.columns}
    if optional:
        time_lines(steps, optional, target, "train_operational_metrics", dpi)


def mode_handles(*, bars=False):
    if bars:
        return [Patch(color=COLORS[mode], label=MODE_LABELS[mode]) for mode in MODES]
    return [
        Line2D([0, 1], [0, 0], color=COLORS[mode], linestyle=LINE_STYLES[mode],
               label=MODE_LABELS[mode])
        for mode in MODES
    ]


def metric_bars(ax, summary, metric, *, show_modes):
    entries = summary.loc[summary["metric"] == metric].set_index("mode").reindex(MODES)
    for i, (mode, entry) in enumerate(entries.iterrows()):
        if pd.notna(entry["mean"]):
            ax.bar(i, entry["mean"], width=0.62, color=COLORS[mode],
                   yerr=entry["std"] if pd.notna(entry["std"]) else None,
                   error_kw={"linewidth": 0.85, "capsize": 3})
        else:
            ax.text(i, 0.03, "N/A", transform=ax.get_xaxis_transform(), ha="center", fontsize=8)
    label, unit = METRICS[metric]
    ax.set(title=label, ylabel=unit, xlim=(-0.55, 2.55))
    ax.set_xticks(range(len(MODES)), [MODE_LABELS[mode] if show_modes else "" for mode in MODES])
    ax.tick_params(axis="x", length=0)


def plot_metric_summary(summary, target, dpi):
    fig, axes = plt.subplots(3, 3, figsize=(9, 7), layout="constrained")
    for metric, ax in zip(METRICS, axes.flat, strict=True):
        metric_bars(ax, summary, metric, show_modes=False)
    fig.legend(handles=mode_handles(bars=True), ncols=3, loc="outside lower center",
               title="Bars show means. Error bars show sample SD.")
    save_figure(fig, target, "eval_nine_metrics", dpi)

    for metric, name in KEY_METRICS.items():
        fig, ax = plt.subplots(figsize=(4.6, 3.1), layout="constrained")
        metric_bars(ax, summary, metric, show_modes=True)
        save_figure(fig, target, f"eval_{name}", dpi)


def plot_cumulative(steps, minutes, target, dpi):
    metrics = ["total_cost_usd", "total_carbon_kg"]
    steps = steps.sort_values(["mode", "seed", "step"]).copy()
    steps[metrics] = steps.groupby(["mode", "seed"], sort=False)[metrics].cumsum()
    means = steps.groupby(["mode", "step"], sort=False)[metrics].mean()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.15), layout="constrained")
    for mode in MODES:
        per_step = means.loc[mode]
        elapsed = per_step.index.to_numpy() * minutes / 1440
        axes[0].plot(elapsed, per_step[metrics[0]], color=COLORS[mode], linestyle=LINE_STYLES[mode])
        axes[1].plot(elapsed, per_step[metrics[1]] / 1000, color=COLORS[mode], linestyle=LINE_STYLES[mode])
    axes[0].set_ylabel("Cumulative total cost (USD)")
    axes[1].set_ylabel("Cumulative CO₂ (t)")
    for ax in axes:
        ax.set_xlabel("Elapsed simulated days")
    fig.legend(handles=mode_handles(), ncols=3, loc="outside lower center",
               title="Mean across seeds")
    save_figure(fig, target, "eval_cumulative", dpi)


def plot_dc_utilization(dc_steps, minutes, target, dpi):
    columns = ["cpu_util_pct", "gpu_util_pct"]
    names = dc_steps[["dc_id", "dc_location"]].drop_duplicates().sort_values("dc_id")
    means = dc_steps.groupby(["dc_id", "mode", "step"])[columns].mean()
    fig, axes = plt.subplots(len(names), 2, figsize=(8.4, 9), squeeze=False, sharex=True,
                             layout="constrained")
    for row, (dc_id, location) in enumerate(names.itertuples(index=False, name=None)):
        for mode in MODES:
            dc_mode = means.loc[(dc_id, mode)]
            elapsed = dc_mode.index.to_numpy() * minutes / 1440
            for col, field in enumerate(columns):
                axes[row, col].plot(elapsed, dc_mode[field], color=COLORS[mode],
                                    linestyle=LINE_STYLES[mode], linewidth=0.95)
        axes[row, 0].set_ylabel(f"{location}\nUtilization (%)")
    axes[0, 0].set_title("CPU")
    axes[0, 1].set_title("GPU")
    for ax in axes[-1]:
        ax.set_xlabel("Elapsed simulated days")
    fig.legend(handles=mode_handles(), ncols=3, loc="outside lower center",
               title="Mean across seeds")
    save_figure(fig, target, "eval_dc_utilization", dpi)


def plot_actions(actions, target, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.35), sharey=True, layout="constrained")
    labels = ["Defer", "DC 1", "DC 2", "DC 3", "DC 4", "DC 5"]
    categories = ("raw_action", "effective_action")
    totals = actions[actions["kind"].isin(categories)].groupby(["kind", "mode", "action"])["count"].sum()
    x = np.arange(len(labels))
    for ax, kind in zip(axes, categories, strict=True):
        for i, mode in enumerate(MODES):
            counts = totals.loc[(kind, mode)].reindex(x, fill_value=0)
            pct = counts / counts.sum() * 100 if counts.sum() else counts.astype(float) * np.nan
            ax.bar(x + (i - 1) * 0.24, pct, width=0.22, color=COLORS[mode])
        ax.set_xticks(x, labels)
    axes[0].set(title="Actor samples", ylabel="Share of decisions (%)")
    axes[1].set_title("Effective simulator actions")
    fig.legend(handles=mode_handles(bars=True), ncols=3, loc="outside lower center")
    save_figure(fig, target, "eval_action_distribution", dpi)


def plot_paired(differences, target, dpi):
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 7), layout="constrained")
    short_pairs = ("Geo − Time", "Geo − Full", "Time − Full")
    pairs = tuple(combinations(MODES, 2))
    for ax, (metric, (label, unit)) in zip(axes.flat, METRICS.items(), strict=True):
        ax.axvline(0, color="#555555", linewidth=0.8)
        for i, (a, b) in enumerate(pairs):
            selection = differences.loc[
                (differences["metric"] == metric) & (differences["mode_a"] == a)
                & (differences["mode_b"] == b), "difference_a_minus_b"
            ].dropna()
            offsets = np.linspace(-0.18, 0.18, len(selection)) if len(selection) > 1 else np.zeros(len(selection))
            ax.scatter(selection, i + offsets, s=14, color=COLORS[a], alpha=0.75, linewidths=0)
        ax.set(title=label, xlabel=unit)
        ax.set_yticks(range(len(pairs)), short_pairs)
        ax.set_ylim(len(pairs) - 0.5, -0.55)
    fig.suptitle("Paired seed differences")
    save_figure(fig, target, "eval_paired_differences", dpi)


def plot_paper(comparison, smoke, target, dpi):
    fig, axes = plt.subplots(3, 3, figsize=(10, 7), layout="constrained")
    x = np.arange(len(MODES))
    paper_color, repro_color = "#757575", "#0072B2"
    for ax, (metric, (label, unit)) in zip(axes.flat, METRICS.items(), strict=True):
        frame = comparison.loc[comparison["metric"] == metric].set_index("mode").reindex(MODES)
        ax.bar(x if smoke else x - 0.18, frame["paper_mean"], width=0.34, color=paper_color,
               yerr=frame["paper_std"], error_kw={"linewidth": 0.8, "capsize": 2.5})
        if not smoke:
            for i, (_, row) in enumerate(frame.iterrows()):
                if pd.notna(row["repro_mean"]):
                    ax.bar(i + 0.18, row["repro_mean"], width=0.34, color=repro_color,
                           yerr=row["repro_std"] if pd.notna(row["repro_std"]) else None,
                           error_kw={"linewidth": 0.8, "capsize": 2.5})
        ax.set(title=label, ylabel=unit)
        ax.set_xticks(x, ["Geo", "Time", "Geo+Time"])
    if smoke:
        fig.suptitle("Published Table 3 (short-run evaluation excluded)")
    else:
        fig.legend(handles=[Patch(color=paper_color, label="Paper Table 3"),
                            Patch(color=repro_color, label="Reproduction")],
                   ncols=2, loc="outside lower center", title="Mean ± SD")
    save_figure(fig, target, "eval_paper_table3", dpi)


def plot_evaluation(source, target, dpi):
    manifest = json.loads((source / "eval_manifest.json").read_text(encoding="utf-8"))
    minutes, smoke = manifest["timestep_minutes"], manifest["smoke"]
    plot_metric_summary(read_table(source, "metrics_mean_std"), target, dpi)
    plot_cumulative(read_table(source, "step_metrics"), minutes, target, dpi)
    plot_dc_utilization(read_table(source, "dc_step_metrics"), minutes, target, dpi)
    plot_actions(read_table(source, "action_statistics"), target, dpi)
    plot_paired(read_table(source, "paired_differences"), target, dpi)
    plot_paper(read_table(source, "paper_table3_comparison"), smoke, target, dpi)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Directory with experiment CSVs")
    parser.add_argument("--section", choices=("train", "eval", "all"), default="all", help="Select figure inputs")
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution (default: 300 DPI)")
    args = parser.parse_args(argv)
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    target = args.output_dir / "figures"
    target.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#DEDEDE",
        "grid.linewidth": 0.5, "pdf.fonttype": 42, "legend.frameon": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    }):
        if args.section in ("train", "all"):
            plot_training(args.output_dir, target, args.dpi)
        if args.section in ("eval", "all"):
            plot_evaluation(args.output_dir, target, args.dpi)
    print(f"Figures saved to {target}", flush=True)


if __name__ == "__main__":
    main()
