"""Portable run identities, source/data hashes and software records."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import platform
import re
import subprocess
import uuid
from .config import ROOT, digest, resolve
from .data import workload_source, external_paths
from .replay import atomic_json


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix):
    return f'{prefix}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}'


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def relative(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        # Explicit user paths may be external; manifests record their name/hash.
        return path.name


def fingerprints(bundle):
    paths = {workload_source(bundle[phase]['workload_path'])[0] for phase in ('train_sim', 'eval_sim')}
    for dc in bundle['datacenters']:
        paths.add(resolve(dc['dc_config_file']))
        for phase in ('train_sim', 'eval_sim'):
            paths.update(external_paths(dc['location'], bundle[phase]['year']).values())
    paths.update(p for p in resolve('data/network_cost').rglob('*') if p.is_file() and p.suffix in ('.csv', '.json'))
    data = {relative(p): sha256_file(p) for p in sorted(paths)}
    sources = set()
    for folder in ('scripts/paper_sac', 'envs', 'simulation', 'rl_components', 'rewards', 'utils', 'data/network_cost'):
        sources.update(resolve(folder).rglob('*.py'))
    sources.add(resolve('train_SAC_agent.py'))
    source = {relative(p): sha256_file(p) for p in sorted(sources)}
    return dict(data=data, source=source, source_sha256=digest(source))


def software():
    versions = {}
    for name in ('numpy', 'pandas', 'torch', 'gymnasium', 'tensorboard', 'PyYAML', 'psychrolib'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=platform.python_version(), platform=platform.platform(), packages=versions)


def git_version():
    def run(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True, encoding='utf-8').strip()
    return dict(commit=run('rev-parse', 'HEAD'), branch=run('branch', '--show-current'),
                dirty=bool(run('status', '--porcelain')))


def run_paths(output, run_id, seed):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('Run ID must contain only letters, digits, underscore or hyphen')
    base = resolve(output)
    return {name: base / name / f'seed_{seed}' / run_id for name in
            ('checkpoints', 'logs', 'tensorboard', 'replay_buffer', 'evaluations')}


def create_manifest(output, run_id, bundle, hashes, kind):
    path = resolve(output) / 'manifests' / run_id / 'experiment_manifest.json'
    if path.exists():
        raise FileExistsError(f'Run already exists: {run_id}')
    data = dict(run_id=run_id, kind=kind, status='running', started_at=utc_now(), ended_at=None,
                config=bundle, config_hash=digest(bundle), fingerprints=hashes,
                software=software(), source_version=git_version(), runs=[])
    atomic_json(path, data)
    return path, data
