"""Per-step physical accounting and two-level train/test seed summaries."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


class Metrics:
    def __init__(self):
        self.energy = self.co2 = self.cost = self.water = self.tx_cost = 0.
        self.cpu = self.gpu = self.dc_samples = self.steps = 0
        self.completed = self.violated = self.arrivals = 0
        self.deferred = self.forced = self.remote = self.local = 0

    def add(self, info):
        self.steps += 1
        for dc in info['datacenter_infos'].values():
            common = dc['__common__']
            self.energy += common['energy_consumption_kwh']
            self.co2 += common['carbon_emissions_kg']
            self.cost += common['energy_cost_USD']
            # Original physical model returns liters per 15-minute step.
            self.water += dc['agent_dc']['dc_water_usage'] / 1000
            self.cpu += common['cpu_util_percent']
            self.gpu += common['gpu_util_percent']
            self.dc_samples += 1
            self.completed += common['finished_tasks_count']
            self.violated += common['__sla__']['violated']
        paper = info['paper']
        self.energy += paper['transmission_energy_total_kwh']
        self.co2 += paper['transmission_emissions_total_kg']
        self.tx_cost += paper['transmission_cost_total_usd']
        self.arrivals += paper['new_tasks']
        self.deferred += paper['defer_decisions']
        self.forced += paper['forced_origin']
        self.remote += paper['remote_migrations']
        self.local += paper.get('policy_local_executions', 0)

    def result(self, uncompleted, uncompleted_overdue, terminal_completed=0, terminal_violated=0):
        completed = self.completed + terminal_completed
        violated = self.violated + terminal_violated
        return dict(total_cost_usd=self.cost + self.tx_cost, energy_cost_usd=self.cost,
                    total_co2_t=self.co2 / 1000,
                    total_energy_mwh=self.energy / 1000, water_m3=self.water,
                    sla_violation_rate_pct=100 * violated / completed if completed else None,
                    avg_cpu_utilization_pct=self.cpu / self.dc_samples if self.dc_samples else None,
                    avg_gpu_utilization_pct=self.gpu / self.dc_samples if self.dc_samples else None,
                    transmission_cost_usd=self.tx_cost, tasks_deferred=self.deferred,
                    new_arrival_tasks=self.arrivals, completed_tasks=completed,
                    unfinished_tasks=uncompleted, unfinished_overdue_tasks=uncompleted_overdue,
                    defer_decisions=self.deferred, forced_origin_count=self.forced,
                    remote_migration_count=self.remote, policy_local_execution_count=self.local,
                    sla_violations=violated, actual_steps=self.steps)


def write_result(stem, result):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    jp, cp = stem.with_suffix('.json'), stem.with_suffix('.csv')
    if jp.exists() or cp.exists():
        raise FileExistsError('Refusing to overwrite existing evaluation result')
    jp.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    pd.DataFrame([result]).to_csv(cp, index=False)


def _clean(value):
    return json.loads(value.to_json(orient='records'))


def summarize_records(rows):
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError('No completed evaluation records')
    frame = frame[frame.status == 'complete']
    if frame.empty:
        raise ValueError('No completed evaluation records')
    if frame.duplicated(['train_seed', 'eval_seed', 'strategy']).any():
        raise ValueError('Duplicate train/eval/strategy cell; choose one experiment run')
    excluded = {'train_seed', 'eval_seed', 'actual_steps'}
    metrics = [c for c in frame.select_dtypes(include='number').columns if c not in excluded]
    groups = frame.groupby(['strategy', 'train_seed'], sort=True)
    per_model = groups[metrics].agg(['mean', 'std'])  # pandas std uses ddof=1
    per_model.columns = [f'{name}_{stat}' for name, stat in per_model.columns]
    per_model['n_eval_seeds'] = groups.eval_seed.nunique()
    per_model = per_model.reset_index()
    means = groups[metrics].mean().reset_index()
    overall = means.groupby('strategy')[metrics].agg(['mean', 'std'])
    overall.columns = [f'{name}_{stat}' for name, stat in overall.columns]
    overall['n_train_seeds'] = means.groupby('strategy').train_seed.nunique()
    return dict(per_run=_clean(frame), per_model=_clean(per_model),
                across_models=_clean(overall.reset_index()), ddof=1,
                interpretation='Average test seeds within each trained model, then sample SD across trained models')
