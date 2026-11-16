"""Relative-path configuration and strict experiment invariants."""
from pathlib import Path
import copy
import hashlib
import json
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = 'configs/experiments/paper_sac_v1/algorithm.yaml'
TRAIN_SEEDS = (42, 43, 44)
EVAL_SEEDS = (0, 1, 2)
STRATEGIES = ('geo_only', 'time_only', 'geo_time')


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def load_bundle(config=DEFAULT_CONFIG):
    parent = resolve(config).parent
    result = {}
    for name, key in [('train_sim', 'simulation'), ('eval_sim', 'simulation'),
                      ('reward', 'reward'), ('datacenters', 'datacenters')]:
        result[name] = yaml.safe_load((parent / f'{name}.yaml').read_text(encoding='utf-8'))[key]
    result['algorithm'] = yaml.safe_load(resolve(config).read_text(encoding='utf-8'))['algorithm']
    for name in ('actor_learning_rate', 'critic_learning_rate'):
        result['algorithm'][name] = float(result['algorithm'][name])
    validate_bundle(result)
    return result


def validate_bundle(cfg):
    a = cfg['algorithm']
    for k in ('total_steps', 'batch_size', 'replay_buffer_size', 'max_tasks', 'hidden_dim',
              'update_frequency', 'policy_update_frequency', 'save_interval', 'log_interval'):
        if not isinstance(a[k], int) or a[k] <= 0:
            raise ValueError(f'{k} must be a positive integer')
    if a['batch_size'] > a['replay_buffer_size']:
        raise ValueError('Replay capacity must cover batch_size')
    if a.get('use_attention') or a.get('use_layer_norm') or a['name'] != 'SAC':
        raise ValueError('Paper network requires SAC MLP without attention/LayerNorm')
    if a.get('eval_frequency', 0) != 0 or a.get('eval_episodes', 0) != 0:
        raise ValueError('Use the independent evaluator; training evaluation must be disabled')
    if a['max_tasks'] != 400 or a['hidden_dim'] != 64:
        raise ValueError('Paper dimensions require max_tasks=400 and hidden_dim=64')
    for phase in ('train_sim', 'eval_sim'):
        s = cfg[phase]
        if s['shuffle_datacenters']:
            raise ValueError('shuffle_datacenters must be false')
        if s['single_action_mode'] or s['disable_defer_action'] or s['strategy'] != 'manual_rl':
            raise ValueError('Multi-action + Defer + manual_rl required')
        if s['timestep_minutes'] != 15:
            raise ValueError('Only 15-minute simulation is supported')
        if s.get('reset_mode', 'fixed') not in ('fixed', 'randomized_aligned'):
            raise ValueError('Unknown reset_mode')
    dcs = cfg['datacenters']
    expected = [('US-CAL-CISO', 50000, 1000, 80000), ('DE-LU', 85000, 600, 80000),
                ('CL-SIC', 110000, 300, 60000), ('SG', 75000, 700, 50000),
                ('AU-NSW', 65000, 300, 60000)]
    if [d['dc_id'] for d in dcs] != [1, 2, 3, 4, 5]:
        raise ValueError('DC IDs/order must be 1..5')
    for d, values in zip(dcs, expected):
        if tuple(d[k] for k in ('location', 'total_cores', 'total_gpus', 'total_mem')) != values:
            raise ValueError('Original DC resources must remain unchanged')
        if d.get('use_rl_hvac') is not False or d.get('hvac_controller_type') != 'none':
            raise ValueError('Fixed HVAC is required')
        if d['dc_config_file'] != 'configs/dcs/dc_config.json':
            raise ValueError('Use original physical configuration')


def smoke_bundle(cfg):
    cfg = copy.deepcopy(cfg)
    cfg['algorithm'].update(total_steps=64, replay_buffer_size=128, batch_size=8,
                            warmup_steps=8, device='cpu', save_interval=32, log_interval=8)
    cfg['smoke'] = True
    return cfg
