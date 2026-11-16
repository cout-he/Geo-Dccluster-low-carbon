"""Separate stochastic evaluation. Full matrix requires --production explicitly."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'scripts.paper_sac'

import argparse
import random
import numpy as np
import torch
from .config import DEFAULT_CONFIG, EVAL_SEEDS, STRATEGIES, load_bundle, resolve, digest
from .learner import SAC
from .checkpoint import load_data
from .environment import make_env
from .policy import sample_strategy
from .preflight import check
from .metrics import write_result
from .provenance import new_id, create_manifest, utc_now, relative, run_paths, sha256_file
from .replay import atomic_json


def evaluate_one(bundle, checkpoint, eval_seed, strategy, steps=None):
    data = load_data(checkpoint)
    seed = data['state']['seed']
    random.seed(eval_seed)
    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)
    agent = SAC(bundle['algorithm'])
    if data['metadata'] != agent.metadata:
        raise ValueError('Model architecture/action/reward temperature metadata mismatch')
    agent.actor.load_state_dict(data['actor'])
    agent.actor.eval()
    generator = torch.Generator(device='cpu').manual_seed(eval_seed)
    env = make_env(bundle, phase='eval_sim', episode_steps=steps)
    started = utc_now()
    try:
        obs, _ = env.reset(seed=eval_seed)
        realized_start = str(env.start_time)
        while True:
            with torch.no_grad():
                logits = agent.actor(torch.as_tensor(np.asarray(obs, dtype=np.float32).reshape(-1, 34), device=agent.device))
                actions = sample_strategy(logits, [t.origin_dc_id for t in env.current_tasks], strategy, generator)
            obs, _, done, truncated, _ = env.step(actions)
            if done or truncated:
                break
        result = env.final_metrics()
        tasks_signature = digest([(t.job_name, str(t.arrival_time), int(t.origin_dc_id), t.cores_req,
                                   t.gpu_req, t.mem_req, t.duration) for t in env.ledger.values()])
        result.update(train_seed=seed, eval_seed=eval_seed, strategy=strategy,
            checkpoint=relative(checkpoint), checkpoint_sha256=sha256_file(checkpoint),
            started_at=started, ended_at=utc_now(), status='complete',
            run_kind='smoke' if steps else 'production', realized_start=realized_start,
            end_exclusive=str(env.end_time), workload_and_origin_sha256=tasks_signature,
            sampling='stochastic, independent CPU torch.Generator seeded by eval_seed',
            reset_mode=bundle['eval_sim'].get('reset_mode', 'fixed'))
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--eval-seeds', nargs='+', type=int)
    parser.add_argument('--strategies', nargs='+', choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument('--output', default='outputs/paper_sac_v1')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--production', action='store_true')
    parser.add_argument('--steps', type=int, help='Smoke-only episode length, 1..64; default 8')
    args = parser.parse_args()
    if args.steps is not None and (not args.smoke or not 1 <= args.steps <= 64):
        parser.error('--steps requires --smoke and a value in 1..64')
    seeds = args.eval_seeds if args.eval_seeds is not None else ([0] if args.smoke else list(EVAL_SEEDS))
    if len(set(seeds)) != len(seeds) or len(set(args.strategies)) != len(args.strategies):
        parser.error('Duplicate eval seeds/strategies')
    checkpoints = [resolve(p) for p in args.checkpoints]
    model_seeds = [load_data(p)['state']['seed'] for p in checkpoints]
    if len(set(model_seeds)) != len(model_seeds):
        parser.error('Select exactly one checkpoint for each training seed')
    if args.smoke and (len(checkpoints) != 1 or len(seeds) != 1):
        parser.error('Smoke evaluation is limited to one model and one eval seed')
    if args.production and (sorted(model_seeds) != [42, 43, 44] or sorted(seeds) != [0, 1, 2] or set(args.strategies) != set(STRATEGIES)):
        parser.error('Production matrix requires three models x seeds 0 1 2 x all three strategies')
    bundle = load_bundle(args.config)
    torch.set_num_threads(1)
    report = check(bundle, args.output, seeds=0)
    if report['status'] != 'passed':
        path = resolve(args.output) / 'manifests' / new_id('eval_preflight_failed') / 'preflight.json'
        atomic_json(path, report)
        raise SystemExit(f'Preflight failed: {relative(path)}: {report["errors"]}')
    hashes = report['fingerprints']
    for path in checkpoints:
        saved = load_data(path)
        if saved['fingerprints'] != hashes:
            raise ValueError(f'Dataset/source fingerprints differ from checkpoint: {relative(path)}')
        if saved['config']['datacenters'] != bundle['datacenters'] or saved['config']['reward'] != bundle['reward']:
            raise ValueError('Model physics/reward config differs from evaluation configuration')
        if args.production and (saved['config'].get('smoke') or saved['state']['global_step'] != 10000000):
            raise ValueError('Production evaluation requires finished 10M-step paper models')
    run_id = new_id('eval_smoke' if args.smoke else 'eval')
    manifest_path, manifest = create_manifest(args.output, run_id, bundle, hashes, 'evaluation_smoke' if args.smoke else 'evaluation')
    manifest['preflight'] = report
    paired = {}
    try:
        for checkpoint in checkpoints:
            for eval_seed in seeds:
                for strategy in args.strategies:
                    item = dict(checkpoint=relative(checkpoint), eval_seed=eval_seed, strategy=strategy,
                                status='running', started_at=utc_now(), ended_at=None)
                    manifest['runs'].append(item)
                    atomic_json(manifest_path, manifest)
                    result = evaluate_one(bundle, checkpoint, eval_seed, strategy,
                                          steps=(args.steps or 8) if args.smoke else None)
                    result.update(evaluation_run_id=run_id, config_file=args.config,
                                  source_sha256=hashes['source_sha256'], data_sha256=digest(hashes['data']))
                    token = (result['realized_start'], result['end_exclusive'], result['workload_and_origin_sha256'])
                    if eval_seed in paired and paired[eval_seed] != token:
                        raise ValueError('Paired strategies/models did not see identical workload/origins/calendar')
                    paired[eval_seed] = token
                    destination = run_paths(args.output, run_id, result['train_seed'])['evaluations'] / f'{strategy}_eval_{eval_seed}'
                    write_result(destination, result)
                    item.update(result, result_file=relative(destination.with_suffix('.json')))
                    atomic_json(manifest_path, manifest)
                    print(f'{run_id}: train={result["train_seed"]} eval={eval_seed} strategy={strategy} steps={result["actual_steps"]}', flush=True)
        manifest['status'] = 'complete'
    except BaseException as exc:
        manifest.update(status='failed', error=repr(exc))
        if manifest['runs'] and manifest['runs'][-1]['status'] == 'running':
            manifest['runs'][-1].update(status='failed', error=repr(exc), ended_at=utc_now())
        raise
    finally:
        manifest['ended_at'] = utc_now()
        atomic_json(manifest_path, manifest)
    print(f'Manifest: {relative(manifest_path)}')


if __name__ == '__main__':
    main()
