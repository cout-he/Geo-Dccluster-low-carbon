"""Observe unmodified built-in cluster RBC routing; stop on cohort mismatch."""
import argparse
import copy
import csv
import hashlib
import random
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rbc_common import *
sys.path.insert(0, str(ROOT))
import torch
import yaml
from train_SAC_agent import make_env

def run(strategy, seed):
    folder = OUT / strategy / f'seed_{seed}'
    folder.mkdir(parents=True, exist_ok=False)
    with (folder / 'run.log').open('x', encoding='utf-8') as log:
        with redirect_stdout(log), redirect_stderr(log):
            return evaluate(strategy, seed, folder)

def evaluate(strategy, seed, folder):
    started = time.perf_counter()
    config = yaml.safe_load((ROOT / 'configs/env/sim_config_table3_7d_aligned.yaml').read_text(encoding='utf-8'))
    baseline = copy.deepcopy(config)
    config['simulation']['strategy'] = strategy
    test = copy.deepcopy(config); test['simulation']['strategy'] = 'manual_rl'
    assert test == baseline
    sim_path = folder / 'sim_eval.yaml'
    sim_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = make_env(str(sim_path), 'configs/env/datacenters.yaml', 'configs/env/reward_config.yaml', writer=None, logger=None)
    cluster = env.cluster_manager
    assert cluster.strategy == strategy and strategy in cluster.strategy_map
    original_loader = cluster.get_tasks_for_timestep
    captured, loader_times = [], []
    workload_hash, input_hash = hashlib.sha256(), hashlib.sha256()
    def capture(timestamp):
        tasks = original_loader(timestamp)
        captured.extend(tasks); loader_times.append(timestamp.isoformat())
        update_hash(workload_hash, {'time': timestamp.isoformat(), 'tasks': [
            [t.job_name, t.origin_dc_id, t.cores_req, t.gpu_req, t.mem_req, t.duration,
             t.bandwidth_gb, t.sla_deadline.isoformat()] for t in tasks]})
        return tasks
    cluster.get_tasks_for_timestep = capture
    try:
        env.reset(seed=seed)
        assert not captured and not loader_times, 'RBC reset unexpectedly loaded tasks.'
        assert env.start_time == pd.Timestamp('2023-07-01T00:00Z')
        assert env.end_time == pd.Timestamp('2023-07-08T00:00Z')
        dcs = list(cluster.datacenters.values())
        assert [dc.dc_id for dc in dcs] == [1, 2, 3, 4, 5]
        assert env.obs_dim_per_task == 34 and env.agent_output_act_dim == 6
        assert cluster.random_init_day == 181 and cluster.random_init_hour == 0
        norms = []
        for dc in dcs:
            assert not dc.use_rl_hvac
            dc.aligned_inputs.validate_window(env.start_time, env.end_time)
            frame = dc.aligned_inputs.frame.loc[env.start_time:env.start_time + pd.Timedelta(days=30)]
            norms.append({'dc_id': dc.dc_id, 'start_utc': frame.index[0].isoformat(),
                'end_utc_inclusive': frame.index[-1].isoformat(), 'rows': len(frame),
                'minimum': frame.min().to_dict(), 'maximum': frame.max().to_dict(),
                'span': (frame.max()-frame.min()).clip(lower=1e-8).to_dict()})
        assert norms == read_json(PREFLIGHT / 'normalization_frozen.json'), 'Normalization changed.'
        write_json(folder / 'runtime_config.json', {'seed': seed, 'strategy': strategy,
            'simulation': config['simulation'], 'dc_order': [dc.dc_id for dc in dcs],
            'obs_dim': env.obs_dim_per_task, 'act_dim': env.agent_output_act_dim,
            'normalization': norms, 'checkpoint_loaded': False, 'actor_inference': False,
            'routing': 'DatacenterClusterManager.step built-in rule-based branch',
            'instrumentation': 'Observe same returned loader tasks; no extra calls or RNG interception.'})
        fields = ['step', 'time_utc', 'task', 'arrival_time_utc', 'sla_deadline_utc',
                  'origin_dc_id', 'destination_dc_id', 'effective_decision', 'cores_req', 'gpu_req',
                  'mem_req', 'duration', 'bandwidth_gb']
        input_fields = ['step', 'time_utc', 'datacenter', 'location', 'carbon_intensity_g_kwh',
                        'electricity_price_usd_mwh', 'ambient_temperature_c']
        records, arrived, remote = [], 0, 0
        received = {dc.dc_id: 0 for dc in dcs}
        with (folder / 'decisions.csv').open('x', newline='', encoding='utf-8-sig') as df, \
             (folder / 'inputs.csv').open('x', newline='', encoding='utf-8-sig') as inf:
            decisions = csv.DictWriter(df, fieldnames=fields); decisions.writeheader()
            inputs = csv.DictWriter(inf, fieldnames=input_fields); inputs.writeheader()
            for step in range(672):
                timestamp = env.current_time
                assert timestamp == env.start_time + step * pd.Timedelta(minutes=15)
                assert env.current_episode_count == 1
                captured.clear()
                for dc in dcs:
                    assert dc.aligned_inputs.current_time == timestamp
                    row = dict(step=step, time_utc=timestamp.isoformat(), datacenter=dc.dc_id,
                        location=dc.location, **dc.aligned_inputs.frame.iloc[dc.aligned_inputs.position].to_dict())
                    inputs.writerow(row); update_hash(input_hash, row)
                _, reward, terminated, truncated, info = env.step([])
                assert len(loader_times) == step + 1 and loader_times[-1] == timestamp.isoformat()
                assert bool(terminated or truncated) == (step == 671)
                assert not env.deferred_tasks and not env.current_tasks and not env.in_transit_tasks
                arrived += len(captured)
                for task in captured:
                    assert task.dest_dc_id in received, 'Rule returned no valid DC.'
                    received[task.dest_dc_id] += 1
                    remote += int(task.dest_dc_id != task.origin_dc_id)
                    if strategy == 'local_only': assert task.dest_dc_id == task.origin_dc_id
                    decisions.writerow(dict(zip(fields, [step, timestamp.isoformat(), task.job_name,
                        task.arrival_time.isoformat(), task.sla_deadline.isoformat(), task.origin_dc_id,
                        task.dest_dc_id, 'route_rule', task.cores_req, task.gpu_req,
                        task.mem_req, task.duration, task.bandwidth_gb])))
                infos = list(info['datacenter_infos'].values()); common = [x['__common__'] for x in infos]
                assert all(c['input_time_utc'] == timestamp.isoformat() for c in common)
                records.append({'step': step, 'time_utc': timestamp.isoformat(), 'new_arrivals': len(captured),
                    'tasks_finished': sum(c['finished_tasks_count'] for c in common),
                    'sla_met': sum(c['__sla__']['met'] for c in common),
                    'sla_violated': sum(c['__sla__']['violated'] for c in common),
                    'tasks_deferred': 0, 'forced_origin_decisions': 0,
                    'dc_energy_kwh': sum(c['energy_consumption_kwh'] for c in common),
                    'dc_energy_cost_usd': sum(c['energy_cost_USD'] for c in common),
                    'dc_carbon_kg': sum(c['carbon_emissions_kg'] for c in common),
                    'tx_energy_kwh': info['total_energy_kwh'], 'tx_carbon_kg': info['total_emissions_kg'],
                    'transmission_cost_usd': info['transmission_cost_total_usd'],
                    'water_l': sum(x['agent_dc']['dc_water_usage'] for x in infos),
                    'avg_cpu_util_pct': np.mean([c['cpu_util_percent'] for c in common]),
                    'avg_gpu_util_pct': np.mean([c['gpu_util_percent'] for c in common]), 'reward': float(reward)})
                if (step + 1) % 96 == 0: print(f'{strategy} seed={seed}: {step+1}/672', flush=True)
        steps = pd.DataFrame(records)
        steps.to_csv(folder / 'steps.csv', index=False, encoding='utf-8-sig')
        assert len(steps) == 672 and env.current_time == env.end_time
        assert np.isfinite(steps.select_dtypes('number').to_numpy()).all()
        unfinished = [t for dc in dcs for t in list(dc.pending_tasks) + list(dc.running_tasks)]
        total = steps.sum(numeric_only=True); finished = int(total.tasks_finished)
        assert arrived == finished + len(unfinished)
        assert finished == total.sla_met + total.sla_violated
        result = {'strategy': strategy, 'seed': seed, 'steps': len(steps),
            'start_utc': steps.time_utc.iloc[0], 'last_step_utc': steps.time_utc.iloc[-1],
            'end_utc_exclusive': env.current_time.isoformat(), 'new_arrivals': arrived,
            'tasks_finished': finished, 'unfinished_tasks': len(unfinished),
            'late_unfinished_tasks': sum(env.current_time > t.sla_deadline for t in unfinished),
            'sla_violated': int(total.sla_violated), 'remote_route_decisions': remote,
            'tasks_received_by_dc': received, 'dc_order': [dc.dc_id for dc in dcs],
            'dc_energy_kwh': float(total.dc_energy_kwh), 'tx_energy_kwh': float(total.tx_energy_kwh),
            'dc_carbon_kg': float(total.dc_carbon_kg), 'tx_carbon_kg': float(total.tx_carbon_kg),
            **metrics(steps), 'realized_workload_sha256': workload_hash.hexdigest(),
            'external_inputs_sha256': input_hash.hexdigest(),
            'elapsed_seconds': time.perf_counter() - started, 'status': 'pending_hash_validation'}
        write_json(folder / 'summary.json', result)
        reference = next(r for r in sac_records() if r['seed'] == seed)
        assert result['realized_workload_sha256'] == reference['realized_workload_sha256'], 'ACTUAL WORKLOAD HASH MISMATCH; STOP BATCH'
        assert result['external_inputs_sha256'] == reference['external_inputs_sha256'], 'EXTERNAL INPUT HASH MISMATCH; STOP BATCH'
        if strategy == 'local_only':
            assert remote == 0 and all(result[k] == 0 for k in ['transmission_cost_usd','tx_energy_kwh','tx_carbon_kg'])
        result['status'] = 'complete_verified'; write_json(folder / 'summary.json', result)
        record = {'seed': seed, 'strategy': strategy, 'status': 'complete_verified',
            'actual_steps': 672, 'actual_start_utc': result['start_utc'],
            'actual_last_step_utc': result['last_step_utc'], 'actual_end_utc_exclusive': result['end_utc_exclusive'],
            'checkpoint_loaded': False, 'comparison_checkpoint_sha256': reference['checkpoint_sha256'],
            'config_sha256': reference['config_sha256'], 'runtime_sim_config_sha256': file_hash(sim_path),
            'workload_file_sha256': reference['workload_file_sha256'],
            'realized_workload_sha256': result['realized_workload_sha256'],
            'external_source_set_sha256': reference['external_source_set_sha256'],
            'external_inputs_sha256': result['external_inputs_sha256'], 'metrics': result,
            'output_directory': folder.relative_to(ROOT).as_posix(),
            'result_files_sha256': {n: file_hash(folder/n) for n in ['summary.json','steps.csv','inputs.csv','decisions.csv','sim_eval.yaml','runtime_config.json']}}
        write_json(folder / 'run_record.json', record)
        return record
    except Exception:
        traceback.print_exc(); raise
    finally:
        env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['seed0', 'remaining'], required=True)
    args = parser.parse_args(); torch.set_num_threads(2)
    manifest_path = OUT / 'run_manifest.json'
    if args.phase == 'seed0':
        assert not manifest_path.exists(), 'Refusing to overwrite existing batch.'
        report = regression(); frozen = read_json(PREFLIGHT / 'frozen_manifest.json')
        protected = {r['path']: r['sha256'] for r in frozen['frozen_files']}
        for record in sac_records():
            folder = Path(record['output_directory'])
            protected.update({(folder/n).as_posix(): h for n,h in record['result_files_sha256'].items()})
            protected[(folder/'run_record.json').as_posix()] = file_hash(ROOT/folder/'run_record.json')
        for n in ['validated_index_15_runs.json', 'verified_aggregate.json']:
            protected[(SAC/n).relative_to(ROOT).as_posix()] = file_hash(SAC/n)
        assert all(file_hash(ROOT/p) == h for p,h in protected.items())
        manifest = {'status': 'running_seed0_gate', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'strategies': STRATEGIES, 'seeds': SEEDS, 'required_runs': 25, 'completed_runs': [],
            'python': sys.executable, 'python_version': sys.version, 'torch': torch.__version__,
            'torch_threads': torch.get_num_threads(), 'device': 'cpu', 'numpy': np.__version__, 'pandas': pd.__version__,
            'protected_files_sha256': protected, 'sac_regression_passed': report['status'] == 'passed',
            'only_simulation_config_change': 'simulation.strategy', 'checkpoint_loaded': False, 'training': False,
            'known_differences': ['Most Available uses absolute available CPU cores, not percentage.',
                'Built-in RBC enqueues directly; SAC uses TaskSchedulingEnv in-transit delivery.',
                'Capacity candidate checks occur before DC scheduling; no reservation between same-step assignments.']}
        write_json(manifest_path, manifest); seeds = [0]
    else:
        manifest = read_json(manifest_path)
        assert manifest['status'] == 'seed0_gate_passed' and len(manifest['completed_runs']) == 5
        seeds = [10, 20, 30, 40]
    try:
        for seed in seeds:
            for strategy in STRATEGIES:
                print(f'START {strategy} seed={seed}', flush=True)
                record = run(strategy, seed); folder = ROOT / record['output_directory']
                record['result_files_sha256']['run.log'] = file_hash(folder/'run.log')
                write_json(folder/'run_record.json', record)
                manifest['completed_runs'].append({'strategy': strategy, 'seed': seed, 'record': str(folder/'run_record.json')})
                write_json(manifest_path, manifest)
                print(f'PASS {strategy} seed={seed}: cost={record["metrics"]["total_cost_usd"]:.3f}; workload/input hashes match SAC', flush=True)
        assert all(file_hash(ROOT/p) == h for p,h in manifest['protected_files_sha256'].items()), 'Protected file changed.'
        manifest['status'] = 'seed0_gate_passed' if args.phase == 'seed0' else 'runs_complete_pending_collection'
    except Exception as exc:
        manifest.update(status='failed_stopped', error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        manifest['new_code_sha256'] = {p.name: file_hash(p) for p in OUT.glob('*.py')}
        write_json(manifest_path, manifest)
    print(f'BATCH STATUS: {manifest["status"]}; completed={len(manifest["completed_runs"])}', flush=True)

if __name__ == '__main__':
    main()
