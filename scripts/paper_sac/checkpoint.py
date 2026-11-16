"""Transactional full episode-boundary checkpoint and inference snapshots."""
from pathlib import Path
import os
import random
import uuid
import numpy as np
import torch
from .config import digest
from .learner import RunningStats
from .replay import MemmapReplayBuffer


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch_cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def set_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    if state['cuda']:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Full CUDA resume requires the same CUDA device count')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def save_training(path, agent, replay, stats, state, config, fingerprints):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite historical checkpoint: {path}')
    replay.flush()
    checkpoint_ref = Path(os.path.relpath(path, replay.directory)).as_posix()
    data = dict(format=1, actor=agent.actor.state_dict(), critic=agent.critic.state_dict(),
                target=agent.target.state_dict(), actor_optimizer=agent.actor_opt.state_dict(),
                critic_optimizer=agent.critic_opt.state_dict(), state=state,
                running_stats=vars(stats).copy(), rng=rng_state(), metadata=agent.metadata,
                config=config, config_hash=digest(config), fingerprints=fingerprints,
                replay_path=Path(os.path.relpath(replay.directory, path.parent)).as_posix(),
                replay={**replay.metadata(checkpoint_ref), 'generation': replay.generation + 1},
                episode_boundary=True)
    tmp = path.with_name(path.name + f'.{uuid.uuid4().hex}.tmp')
    with tmp.open('xb') as stream:
        torch.save(data, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    if os.name != 'nt':
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    # commit.json is the sole publication point. Crash before this restores the
    # prior checkpoint using its undo journal; crash after this opens the new one.
    replay.commit(checkpoint_ref)
    return data


def load_data(path):
    # Our checkpoints include Python/NumPy RNG objects. Load only trusted local
    # experiment files. map_location CPU keeps RNG ByteTensors portable.
    return torch.load(path, map_location='cpu', weights_only=False)


def validate_resume_data(data):
    required = {'actor', 'critic', 'target', 'actor_optimizer', 'critic_optimizer',
                'state', 'running_stats', 'rng', 'metadata', 'config_hash',
                'fingerprints', 'replay_path', 'replay'}
    if (data.get('format') != 1 or not data.get('episode_boundary') or
            data.get('inference_only') or not required.issubset(data)):
        raise ValueError('A full resume requires a complete episode-boundary training checkpoint, not an inference snapshot')


def restore_training(path, agent, config, fingerprints, restore_rng=True):
    path = Path(path).resolve()
    data = load_data(path)
    validate_resume_data(data)
    if data['config_hash'] != digest(config):
        raise ValueError('Training configuration hash mismatch')
    if data['fingerprints'] != fingerprints:
        raise ValueError('Dataset/source fingerprint mismatch')
    if data['metadata'] != agent.metadata:
        raise ValueError('Network/action metadata mismatch')
    directory = (path.parent / data['replay_path']).resolve()
    import json
    committed = json.loads((directory / 'commit.json').read_text())
    if committed != data['replay']:
        raise ValueError('Stale checkpoint: full resume requires latest committed replay checkpoint')
    replay = MemmapReplayBuffer.open(directory)
    agent.actor.load_state_dict(data['actor'])
    agent.critic.load_state_dict(data['critic'])
    agent.target.load_state_dict(data['target'])
    agent.actor_opt.load_state_dict(data['actor_optimizer'])
    agent.critic_opt.load_state_dict(data['critic_optimizer'])
    stats = RunningStats()
    stats.__dict__.update(data['running_stats'])
    if restore_rng:
        set_rng(data['rng'])
    return data['state'], replay, stats
