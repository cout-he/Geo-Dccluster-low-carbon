"""Dataset/capacity checks. No replay allocation and no production training."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'scripts.paper_sac'

import argparse
from pathlib import Path
import shutil
import numpy as np
from .config import DEFAULT_CONFIG, load_bundle, resolve
from .data import workload_source, workload_counts, required_window, external_data, validate_coverage, time_index
from .provenance import new_id, fingerprints, utc_now, relative, software, git_version
from .replay import replay_bytes, atomic_json


def _allocated_bytes(path):
    stat = path.stat()
    if hasattr(stat, 'st_blocks'):
        return stat.st_blocks * 512
    if __import__('os').name == 'nt':
        import ctypes
        from ctypes import wintypes
        high = wintypes.DWORD()
        function = ctypes.WinDLL('kernel32', use_last_error=True).GetCompressedFileSizeW
        function.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
        function.restype = wintypes.DWORD
        low = function(str(path), ctypes.byref(high))
        if low == 0xffffffff and ctypes.get_last_error():
            raise ctypes.WinError(ctypes.get_last_error())
        return (high.value << 32) | low
    return 0  # Unknown allocation: conservatively grant no disk credit.


def capacity_report(bundle, output, seeds, existing_replay=None):
    a = bundle['algorithm']
    base = resolve(output)
    base.mkdir(parents=True, exist_ok=True)
    size = replay_bytes(a['replay_buffer_size'], a['max_tasks'], 34)
    row = replay_bytes(1, a['max_tasks'], 34)
    episode_steps = int(bundle['train_sim']['duration_days'] * 96)
    protected_slots = min(a['replay_buffer_size'], a['save_interval'] + episode_steps)
    journals = 2 * protected_slots * (row + 8) + 16
    # Per-seed arrays persist for independent resume. Include two bounded undo
    # journals and a network/checkpoint reserve instead of assuming RAM storage.
    allocated = 0
    if existing_replay is not None:
        directory = Path(existing_replay)
        files = [p for p in directory.iterdir() if p.suffix in ('.dat', '.bin')]
        allocated = min(size + journals, sum(_allocated_bytes(p) for p in files))
    required_disk = max(0, seeds * (size + journals) - allocated) + 2 * 1024**3
    available = shutil.disk_usage(base).free
    memory_needed = a['batch_size'] * row * 16 + 1024**3
    try:
        import psutil
        ram = psutil.virtual_memory().available
    except ImportError:
        ram = None
    return dict(replay_bytes_per_seed=size, replay_gib_per_seed=size/1024**3,
        journal_max_bytes_per_seed=journals, seeds=seeds, disk_required_bytes=required_disk,
        disk_free_bytes=available, estimated_batch_ram_bytes=memory_needed, ram_available_bytes=ram,
        existing_allocated_bytes=allocated, resume_reuses_existing_replay=existing_replay is not None,
        replay_residency='OS-managed memmap pages; no 11.52GB Python array allocation')


def check(bundle, output='outputs/paper_sac_v1', seeds=3, check_environment=True, existing_replay=None):
    report = dict(status='checking', started_at=utc_now(), checks={}, warnings=[], errors=[])
    a = bundle['algorithm']
    report['checks']['capacity'] = capacity_report(bundle, output, seeds, existing_replay)
    capacity = report['checks']['capacity']
    if capacity['ram_available_bytes'] is None:
        report['warnings'].append('psutil missing: available RAM could not be checked')
    if capacity['disk_free_bytes'] < capacity['disk_required_bytes']:
        report['errors'].append('Insufficient disk for all seeds, undo journals and checkpoint reserve')
    if capacity['ram_available_bytes'] is not None and capacity['ram_available_bytes'] < capacity['estimated_batch_ram_bytes']:
        report['errors'].append('Insufficient available RAM for estimated SAC working batch')
    for phase in ('train_sim', 'eval_sim'):
        sim = bundle[phase]
        try:
            source, member = workload_source(sim['workload_path'])
            report['checks'][phase] = dict(configured_workload=sim['workload_path'],
                configured_workload_exists=resolve(sim['workload_path']).is_file(),
                actual_workload=relative(source), zip_member=member)
            counts = workload_counts(sim, bundle['datacenters'])
            report['checks'][phase]['tasks'] = counts
            if counts['maximum'] > a['max_tasks']:
                report['errors'].append(f'{phase}: arrivals exceed max_tasks={a["max_tasks"]}')
            if counts['all_defer_queue_upper_bound'] > a['max_tasks']:
                report['warnings'].append(f'{phase}: all-Defer queue bound {counts["all_defer_queue_upper_bound"]} exceeds 400; runtime checks stop on actual overflow, never truncate')
            start, end = required_window(sim)
            externals = []
            for dc in bundle['datacenters']:
                data = external_data(dc['location'], sim['year'])
                validate_coverage(data, start, end)
                externals.append(dict(dc_id=dc['dc_id'], location=dc['location'],
                    paths={k: relative(v) for k, v in data['paths'].items()},
                    timestamps='UTC, hourly source, exact 15-minute time interpolation',
                    coverage={k: [str(v.index[0]), str(v.index[-1])] for k, v in data['series'].items()}))
            report['checks'][phase]['external_data'] = externals
            if check_environment:
                from .environment import make_env
                env = make_env(bundle, phase=phase, episode_steps=2)
                try:
                    obs, _ = env.reset(seed=42 if phase == 'train_sim' else 0)
                    dims = dict(obs_dim=env.observation_space.shape[0], act_dim=env.agent_output_act_dim,
                        dc_order=[dc.dc_id for dc in env.cluster_manager.datacenters.values()],
                        initial_task_count=len(obs), current_time=str(env.current_time),
                        reward_components=[c['name'] for c in env.reward_fn.components])
                    for _ in range(2):
                        now = env.current_time
                        _, reward, _, _, info = env.step([t.origin_dc_id for t in env.current_tasks])
                        if not np.isfinite(reward):
                            raise ValueError('Non-finite real environment reward')
                        for dc in env.cluster_manager.datacenters.values():
                            common = info['datacenter_infos'][f'DC{dc.dc_id}']['__common__']
                            if common['price_USD_kwh'] != dc.price_manager.prices[time_index(now)]:
                                raise ValueError('Physical price timestamp is misaligned')
                            if common['ci'] != dc.ci_manager.carbon_smooth[time_index(now)]:
                                raise ValueError('Physical carbon timestamp is misaligned')
                            physical = info['datacenter_infos'][f'DC{dc.dc_id}']
                            if physical['agent_bat']['bat_avg_CI'] != dc.ci_manager.carbon_smooth[time_index(now)]:
                                raise ValueError('Battery consumed a different UTC carbon intensity')
                            if physical['agent_dc']['dc_exterior_ambient_temp'] != dc.weather_manager.temperature_data[time_index(now)]:
                                raise ValueError('Cooling model consumed a different UTC temperature')
                    report['checks'][phase]['environment'] = dims
                finally:
                    env.close()
        except Exception as exc:
            report['errors'].append(f'{phase}: {type(exc).__name__}: {exc}')
    try:
        report['fingerprints'] = fingerprints(bundle)
    except Exception as exc:
        report['errors'].append(f'fingerprints: {exc}')
    report['status'] = 'passed' if not report['errors'] else 'failed'
    report['ended_at'] = utc_now()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--output', default='outputs/paper_sac_v1')
    args = parser.parse_args()
    bundle = load_bundle(args.config)
    report = check(bundle, output=args.output)
    report.update(software=software(), source_version=git_version(), config=bundle)
    destination = resolve(args.output) / 'manifests' / new_id('preflight') / 'preflight.json'
    atomic_json(destination, report)
    print(f'Preflight {report["status"]}: {relative(destination)}')
    for phase in ('train_sim', 'eval_sim'):
        print(phase, report['checks'].get(phase, {}).get('tasks'))
    for item in report['warnings']:
        print('WARNING:', item)
    for item in report['errors']:
        print('ERROR:', item)
    raise SystemExit(0 if report['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
