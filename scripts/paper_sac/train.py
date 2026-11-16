"""Sequential independent-seed training. Production requires --production.

Development uses --smoke (64 total steps, 128 replay slots, CPU). No training
starts on import. Periodic full checkpoints are delayed to episode boundaries.
"""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'scripts.paper_sac'

import argparse
from collections import deque
from pathlib import Path
import json
import os
import random
import signal
import uuid
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from .config import DEFAULT_CONFIG, TRAIN_SEEDS, load_bundle, smoke_bundle, resolve
from .learner import SAC, RunningStats
from .environment import make_env
from .preflight import check
from .provenance import new_id, run_paths, create_manifest, utc_now, relative
from .checkpoint import save_training, restore_training, load_data, rng_state, set_rng, validate_resume_data
from .replay import MemmapReplayBuffer, atomic_json


def save_inference(path, agent, state, config, hashes, boundary):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_suffix('.tmp')
    with tmp.open('xb') as stream:
        torch.save(dict(format=1, actor=agent.actor.state_dict(), critic=agent.critic.state_dict(),
            target=agent.target.state_dict(), metadata=agent.metadata, state=state,
            config=config, fingerprints=hashes, inference_only=True,
            episode_boundary=boundary), stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def reset_episode(env, seed):
    # Environment reset uses its explicit seed. Keep the separate learner RNG
    # streams at their checkpointed positions on BOTH fresh and resumed paths.
    state = rng_state()
    try:
        return env.reset(seed=seed)[0]
    finally:
        set_rng(state)


def train_seed(bundle, seed, output, run_id, hashes, resume=None, stop_after=None,
               progress=None):
    paths = run_paths(output, run_id, seed)
    for name, path in paths.items():
        if name != 'replay_buffer':
            path.mkdir(parents=True, exist_ok=bool(resume))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # CPU thread oversubscription is costly for these small MLPs. This is saved
    # as runtime metadata and held constant for same-host deterministic resume.
    torch.set_num_threads(1)
    agent = SAC(bundle['algorithm'])
    a = bundle['algorithm']
    state = dict(seed=seed, run_id=run_id, global_step=0, episode=0,
        best_avg10=None, recent_rewards=[], next_save_step=a['save_interval'],
        last_checkpoint=None, torch_threads=1)
    if resume:
        saved = load_data(resume)
        state, replay, stats = restore_training(resume, agent, bundle, hashes, restore_rng=False)
        if state['seed'] != seed or state['run_id'] != run_id:
            replay.close()
            raise ValueError('Resume seed/run identity mismatch')
        set_rng(saved['rng'])
    else:
        replay = MemmapReplayBuffer(paths['replay_buffer'], a['replay_buffer_size'], a['max_tasks'], 34)
        stats = RunningStats()
    env = make_env(bundle, episode_steps=16 if bundle.get('smoke') else None)
    writer = SummaryWriter(str(paths['tensorboard']), purge_step=state['global_step'] if resume else None)
    recent = deque(state['recent_rewards'], maxlen=10)
    log = (paths['logs'] / 'training.jsonl').open('a' if resume else 'x', encoding='utf-8')
    episode_reward = 0.
    episode_steps = 0
    at_boundary = True
    last_losses = {}
    last_checkpoint = Path(resume) if resume else None

    def checkpoint():
        nonlocal last_checkpoint
        path = paths['checkpoints'] / f'step_{state["global_step"]:09d}.pt'
        if path.exists():
            # A crash after rename but before publication can leave an orphan.
            # Preserve it and publish the replayed boundary under a new name.
            path = path.with_name(f'{path.stem}_retry_{uuid.uuid4().hex[:8]}.pt')
        state['recent_rewards'] = list(recent)
        state['last_checkpoint'] = relative(path)
        save_training(path, agent, replay, stats, dict(state), bundle, hashes)
        atomic_json(paths['checkpoints'] / 'latest.json', dict(checkpoint=path.name,
            global_step=state['global_step'], episode=state['episode'], resume_kind='full'))
        last_checkpoint = path
        if progress:
            progress(dict(state))

    try:
        if not resume:
            checkpoint()  # An interrupted first episode can restart from step 0.
        obs = reset_episode(env, seed + state['episode'])
        target = a['total_steps']
        while state['global_step'] < target:
            step = state['global_step']
            at_boundary = False
            if len(obs) == 0:
                actions = []
            elif step < a['warmup_steps']:
                actions = np.random.randint(0, 6, len(obs)).tolist()
            else:
                with torch.no_grad():
                    logits = agent.actor(torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=agent.device))
                    actions = torch.multinomial(logits.softmax(-1), 1).squeeze(-1).cpu().tolist()
            nxt, reward, done, truncated, info = env.step(actions)
            stats.update(reward)
            if len(obs):
                replay.add(obs, actions, stats.normalize(reward), nxt, done or truncated)
            state['global_step'] += 1
            episode_steps += 1
            episode_reward += reward
            if state['global_step'] >= a['warmup_steps'] and len(replay) >= a['batch_size'] and state['global_step'] % a['update_frequency'] == 0:
                last_losses.update(agent.update(replay.sample(a['batch_size']), state['global_step']))
            obs = nxt
            if state['global_step'] % a['log_interval'] == 0:
                physical = env.final_metrics()
                record = dict(step=state['global_step'], episode=state['episode'], **last_losses,
                              **physical, reward_components=env.reward_fn.get_last_components())
                log.write(json.dumps(record, allow_nan=False) + '\n')
                log.flush()
                for name, value in last_losses.items():
                    writer.add_scalar(f'Train/{name}', value, state['global_step'])
                for name, value in env.reward_fn.get_last_components().items():
                    writer.add_scalar(f'RewardComponents/{name}', value, state['global_step'])
                for name in ('sla_violations', 'tasks_deferred', 'energy_cost_usd', 'total_cost_usd', 'total_co2_t',
                             'avg_cpu_utilization_pct', 'avg_gpu_utilization_pct'):
                    if physical[name] is not None:
                        writer.add_scalar(f'Physical/{name}', physical[name], state['global_step'])
            if done or truncated:
                at_boundary = True
                state['episode'] += 1
                recent.append(episode_reward / episode_steps)
                writer.add_scalar('Reward/Episode', episode_reward, state['global_step'])
                writer.add_scalar('Reward/EpisodePerStep', episode_reward / episode_steps, state['global_step'])
                is_best = False
                if state['episode'] % 10 == 0 and len(recent) == 10:
                    avg10 = float(np.mean(recent))
                    writer.add_scalar('Reward/Avg10', avg10, state['global_step'])
                    if state['best_avg10'] is None or avg10 > state['best_avg10']:
                        state['best_avg10'] = avg10
                        is_best = True
                record = dict(step=state['global_step'], episode=state['episode'],
                    episode_reward=episode_reward, episode_steps=episode_steps,
                    avg10=state['best_avg10'], metrics=env.final_metrics())
                log.write(json.dumps(record, allow_nan=False) + '\n')
                log.flush()
                due = state['global_step'] >= state['next_save_step']
                stopping = stop_after is not None and state['global_step'] >= stop_after
                if due or is_best or state['global_step'] == target or stopping:
                    while state['next_save_step'] <= state['global_step']:
                        state['next_save_step'] += a['save_interval']
                    checkpoint()
                if is_best:
                    best = paths['checkpoints'] / f'best_avg10_step_{state["global_step"]:09d}.pt'
                    save_inference(best, agent, dict(state), bundle, hashes, True)
                    atomic_json(paths['checkpoints'] / 'best.json', dict(checkpoint=best.name, avg10=state['best_avg10']))
                if stopping and state['global_step'] < target:
                    return dict(state, status='stopped', checkpoint=relative(last_checkpoint))
                if state['global_step'] < target:
                    obs = reset_episode(env, seed + state['episode'])
                    episode_reward, episode_steps = 0., 0
        # Exactly 10M need not be divisible by 672. Save final inference weights
        # without pretending the partial last episode is a complete resume point.
        final = paths['checkpoints'] / f'final_step_{state["global_step"]:09d}.pt'
        save_inference(final, agent, dict(state), bundle, hashes, at_boundary)
        atomic_json(paths['checkpoints'] / 'final.json', dict(checkpoint=final.name,
            global_step=state['global_step'], inference_only=True,
            latest_full_checkpoint=relative(last_checkpoint)))
        return dict(state, status='complete', checkpoint=relative(final),
                    latest_full_checkpoint=relative(last_checkpoint), final_episode_boundary=at_boundary)
    except BaseException:
        if progress:
            progress(dict(state, status='interrupted', checkpoint=relative(last_checkpoint) if last_checkpoint else None,
                          resume_step=load_data(last_checkpoint)['state']['global_step'] if last_checkpoint else None))
        raise
    finally:
        writer.flush()
        writer.close()
        log.close()
        env.close()
        replay.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--output', default='outputs/paper_sac_v1')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--production', action='store_true')
    parser.add_argument('--resume', help='Latest complete episode-boundary checkpoint (one seed)')
    parser.add_argument('--stop-after', type=int, help='Smoke-only controlled stop at the next episode boundary')
    args = parser.parse_args()
    if args.stop_after is not None and (not args.smoke or not 1 <= args.stop_after < 64):
        parser.error('--stop-after requires --smoke and 1..63 steps')
    bundle = load_bundle(args.config)
    if args.smoke:
        bundle = smoke_bundle(bundle)
    seeds = args.seeds or ([42] if args.smoke else list(TRAIN_SEEDS))
    if len(set(seeds)) != len(seeds):
        parser.error('Duplicate seeds')
    if not args.smoke and (any(s not in TRAIN_SEEDS for s in seeds) or seeds != sorted(seeds)):
        parser.error('Production seeds must be an ordered subset of 42 43 44; defaults train all three')
    if args.resume and len(seeds) != 1:
        parser.error('Full resume handles exactly one explicitly selected seed')
    # Do not accept SIGTERM as a silent successful termination. The last
    # transaction remains recoverable even for SIGKILL/power loss.
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    existing_replay = None
    if args.resume:
        saved_for_preflight = load_data(resolve(args.resume))
        validate_resume_data(saved_for_preflight)
        existing_replay = (resolve(args.resume).parent / saved_for_preflight['replay_path']).resolve()
    report = check(bundle, args.output, seeds=len(seeds), existing_replay=existing_replay)
    if report['status'] != 'passed':
        path = resolve(args.output) / 'manifests' / new_id('failed_preflight') / 'preflight.json'
        atomic_json(path, report)
        raise SystemExit(f'Preflight failed: {report["errors"]}; report: {relative(path)}')
    hashes = report['fingerprints']
    if args.resume:
        saved = load_data(resolve(args.resume))
        run_id = saved['state']['run_id']
        manifest_path = resolve(args.output) / 'manifests' / run_id / 'experiment_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest.setdefault('resume_events', []).append(dict(time=utc_now(), checkpoint=relative(resolve(args.resume))))
        manifest.update(status='running', ended_at=None)
    else:
        run_id = new_id('smoke' if args.smoke else 'train')
        manifest_path, manifest = create_manifest(args.output, run_id, bundle, hashes, 'smoke' if args.smoke else 'training')
    manifest['preflight'] = report
    atomic_json(manifest_path, manifest)
    print(f'Run {run_id}; seeds={seeds}; total_steps_per_seed={bundle["algorithm"]["total_steps"]}', flush=True)
    try:
        for seed in seeds:  # sequential: no multiprocessing and no history model loading
            item = dict(train_seed=seed, status='running', started_at=utc_now(), ended_at=None,
                        config_file=args.config, actual_steps=0)
            manifest['runs'].append(item)
            def progress(state):
                item.update(state, actual_steps=state['global_step'])
                atomic_json(manifest_path, manifest)
            result = train_seed(bundle, seed, args.output, run_id, hashes,
                                resume=resolve(args.resume) if args.resume else None,
                                stop_after=args.stop_after, progress=progress)
            item.update(result, actual_steps=result['global_step'], ended_at=utc_now())
            manifest['status'] = result['status']
            atomic_json(manifest_path, manifest)
            print(f'Seed {seed}: {result["status"]}, steps={result["global_step"]}, checkpoint={result["checkpoint"]}', flush=True)
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=repr(exc))
        raise
    finally:
        manifest['ended_at'] = utc_now()
        atomic_json(manifest_path, manifest)


if __name__ == '__main__':
    main()
