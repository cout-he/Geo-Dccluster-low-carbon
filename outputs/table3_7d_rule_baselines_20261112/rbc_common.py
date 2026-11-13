"""Read-only SAC references and identical nine-metric arithmetic for RBC runs."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SAC = ROOT / 'outputs/table3_7d_formal_20261112'
PREFLIGHT = ROOT / 'outputs/table3_7d_preflight'
STRATEGIES = ['local_only', 'lowest_carbon', 'lowest_price', 'most_available', 'round_robin']
SEEDS = [0, 10, 20, 30, 40]
METRICS = ['total_cost_usd', 'total_co2_t', 'total_energy_mwh', 'total_water_m3',
           'sla_violation_pct_finished', 'avg_cpu_util_pct', 'avg_gpu_util_pct',
           'transmission_cost_usd', 'tasks_deferred']

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def update_hash(digest, value):
    digest.update(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())

def metrics(steps):
    total = steps.sum(numeric_only=True)
    finished = int(total.sla_met + total.sla_violated)
    return {'total_cost_usd': float(total.dc_energy_cost_usd + total.transmission_cost_usd),
            'total_co2_t': float(total.dc_carbon_kg + total.tx_carbon_kg) / 1000,
            'total_energy_mwh': float(total.dc_energy_kwh + total.tx_energy_kwh) / 1000,
            'total_water_m3': float(total.water_l) / 1000,
            'sla_violation_pct_finished': 100 * float(total.sla_violated) / finished if finished else None,
            'avg_cpu_util_pct': float(steps.avg_cpu_util_pct.mean()),
            'avg_gpu_util_pct': float(steps.avg_gpu_util_pct.mean()),
            'transmission_cost_usd': float(total.transmission_cost_usd), 'tasks_deferred': int(total.tasks_deferred)}

def sac_records():
    index = read_json(SAC / 'validated_index_15_runs.json')
    assert index['status'] == 'complete' and len(index['runs']) == 15
    records = index['runs']
    assert {(r['strategy'], r['seed']) for r in records} == {
        (s, seed) for s in ['geo_only', 'time_only', 'geo_time'] for seed in SEEDS}
    return records

def regression():
    results = []
    for record in sac_records():
        folder = ROOT / record['output_directory']
        for name, digest in record['result_files_sha256'].items():
            assert file_hash(folder / name) == digest, f'SAC artifact changed: {folder/name}'
        calculated = metrics(pd.read_csv(folder / 'steps.csv'))
        for key, value in calculated.items():
            assert np.isclose(value, record['metrics'][key], atol=1e-7, rtol=1e-12), key
        results.append({'strategy': record['strategy'], 'seed': record['seed'], 'nine_metrics_pass': True,
                        'max_absolute_error': max(abs(calculated[k] - record['metrics'][k]) for k in METRICS)})
    report = {'status': 'passed', 'runs': 15, 'metrics_per_run': 9,
              'absolute_tolerance': 1e-7, 'relative_tolerance': 1e-12, 'results': results}
    write_json(OUT / 'sac_offline_regression.json', report)
    return report

if __name__ == '__main__':
    print(json.dumps(regression(), ensure_ascii=False))
