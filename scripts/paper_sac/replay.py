"""Persistent FastReplayBuffer layout with bounded write-ahead undo journals.

Only the most recent published commit supports full resume. A checkpoint is
published AFTER array flush, and undo bytes reach disk BEFORE ring overwrites.
Two journal files alternate; reclaiming the older transaction never deletes
experiment checkpoints, results, or data.
"""
from pathlib import Path
import json
import os
import shutil
import struct
import uuid
import numpy as np
import torch


def layout(capacity, max_tasks, obs_dim):
    return {
        'obs': ((capacity, max_tasks, obs_dim), 'float32'),
        'actions': ((capacity, max_tasks), 'int64'),
        'reward': ((capacity,), 'float32'),
        'next_obs': ((capacity, max_tasks, obs_dim), 'float32'),
        'done': ((capacity,), 'float32'),
        'mask': ((capacity, max_tasks), 'float32'),
        'mask_next': ((capacity, max_tasks), 'float32'),
    }


def replay_bytes(capacity, max_tasks, obs_dim):
    return sum(int(np.prod(shape)) * np.dtype(dtype).itemsize
               for shape, dtype in layout(capacity, max_tasks, obs_dim).values())


def sync_directory(path):
    if os.name != 'nt':
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{uuid.uuid4().hex}.tmp')
    with tmp.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    # Persist the rename on POSIX. Windows fsyncs the file above.
    sync_directory(path.parent)


class MemmapReplayBuffer:
    def __init__(self, directory, capacity, max_tasks=400, obs_dim=34):
        if min(capacity, max_tasks, obs_dim) <= 0:
            raise ValueError('Replay dimensions must be positive')
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self._acquire_lock()
        self.capacity, self.max_tasks, self.obs_dim = capacity, max_tasks, obs_dim
        self.spec = layout(capacity, max_tasks, obs_dim)
        need = replay_bytes(capacity, max_tasks, obs_dim)
        if shutil.disk_usage(self.directory).free < need + 64 * 1024**2:
            raise OSError(f'Insufficient replay disk space: need {need} plus reserve')
        self.arrays = {k: np.memmap(self.directory / f'{k}.dat', mode='w+', dtype=dtype, shape=shape)
                       for k, (shape, dtype) in self.spec.items()}
        self.pos = self.size = self.generation = self.committed_size = 0
        self._journal = None
        self._backed = set()
        atomic_json(self.directory / 'commit.json', self.metadata(None))

    def metadata(self, checkpoint):
        return dict(format=1, capacity=self.capacity, max_tasks=self.max_tasks,
                    obs_dim=self.obs_dim, pos=self.pos, size=self.size,
                    generation=self.generation, checkpoint=checkpoint)

    @classmethod
    def open(cls, directory):
        self = cls.__new__(cls)
        self.directory = Path(directory).resolve()
        self._acquire_lock()
        meta = json.loads((self.directory / 'commit.json').read_text())
        if meta['format'] != 1:
            raise ValueError('Unsupported replay format')
        for key in ('capacity', 'max_tasks', 'obs_dim', 'pos', 'size', 'generation'):
            setattr(self, key, meta[key])
        self.committed_size = self.size
        self.spec = layout(self.capacity, self.max_tasks, self.obs_dim)
        self.arrays = {}
        for name, (shape, dtype) in self.spec.items():
            path = self.directory / f'{name}.dat'
            if path.stat().st_size != int(np.prod(shape)) * np.dtype(dtype).itemsize:
                raise ValueError(f'Truncated replay file: {path}')
            self.arrays[name] = np.memmap(path, mode='r+', dtype=dtype, shape=shape)
        self._journal = None
        self._backed = set()
        journal = self.directory / f'undo-{self.generation % 2}.bin'
        if journal.exists():
            with journal.open('rb') as stream:
                header = stream.read(8)
                if len(header) == 8 and struct.unpack('<Q', header)[0] == self.generation:
                    while True:
                        slot = stream.read(8)
                        if len(slot) < 8:
                            break
                        index = struct.unpack('<Q', slot)[0]
                        row = stream.read(self.row_bytes)
                        if len(row) != self.row_bytes:
                            # Interrupted append has not yet touched mmap data.
                            break
                        if index >= self.capacity:
                            raise ValueError('Corrupt replay journal index')
                        offset = 0
                        for name, (shape, dtype) in self.spec.items():
                            count = int(np.prod(shape[1:]))
                            nbytes = count * np.dtype(dtype).itemsize
                            value = np.frombuffer(row[offset:offset+nbytes], dtype=dtype).reshape(shape[1:])
                            self.arrays[name][index] = value
                            offset += nbytes
                        self._backed.add(index)
            self.flush()
            # Recovered state is now durable. Start a new clean journal for the
            # same commit; otherwise a torn tail could hide subsequent records.
            self._backed.clear()
            with journal.open('wb') as stream:
                stream.write(struct.pack('<Q', self.generation))
                stream.flush()
                os.fsync(stream.fileno())
        return self

    @property
    def row_bytes(self):
        return replay_bytes(1, self.max_tasks, self.obs_dim)

    def _acquire_lock(self):
        # OS locks are released on crash; a stale file never blocks recovery.
        # Keep the lock handle alive for the entire mmap lifetime.
        self._lock = (self.directory / '.writer.lock').open('a+b')
        self._lock.seek(0, os.SEEK_END)
        if self._lock.tell() == 0:
            self._lock.write(b'0')
            self._lock.flush()
        self._lock.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self._lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._lock.close()
            raise RuntimeError(f'Replay buffer is already in use: {self.directory}') from exc

    def _backup(self, index):
        if index >= self.committed_size or index in self._backed:
            return
        if self._journal is None:
            path = self.directory / f'undo-{self.generation % 2}.bin'
            self._journal = path.open('wb')
            self._journal.write(struct.pack('<Q', self.generation))
        self._journal.write(struct.pack('<Q', index))
        for value in self.arrays.values():
            self._journal.write(np.asarray(value[index]).tobytes())
        self._journal.flush()
        os.fsync(self._journal.fileno())
        # The first transaction using either alternate journal must also make
        # the journal's directory entry durable BEFORE any mmap overwrite.
        sync_directory(self.directory)
        self._backed.add(index)

    def add(self, obs, actions, reward, next_obs, done):
        n, m = len(obs), len(next_obs)
        if max(n, m) > self.max_tasks:
            raise ValueError(f'Task count {max(n, m)} exceeds max_tasks={self.max_tasks}; no truncation')
        if len(actions) != n:
            raise ValueError('Action count does not match current tasks')
        obs = np.asarray(obs, dtype=np.float32).reshape(n, self.obs_dim)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(m, self.obs_dim)
        actions = np.asarray(actions, dtype=np.int64)
        if ((actions < 0) | (actions >= 6)).any() or not np.isfinite(reward):
            raise ValueError('Invalid replay action/reward')
        if not np.isfinite(obs).all() or not np.isfinite(next_obs).all():
            raise ValueError('Non-finite observation')
        self._backup(self.pos)
        for key in self.arrays:
            self.arrays[key][self.pos] = -1 if key == 'actions' else 0
        self.arrays['obs'][self.pos, :n] = obs
        self.arrays['next_obs'][self.pos, :m] = next_obs
        self.arrays['actions'][self.pos, :n] = actions
        self.arrays['reward'][self.pos] = reward
        self.arrays['done'][self.pos] = done
        self.arrays['mask'][self.pos, :n] = 1
        self.arrays['mask_next'][self.pos, :m] = 1
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if not self.size:
            raise ValueError('Cannot sample empty replay')
        indices = np.random.randint(0, self.size, size=batch_size)
        return tuple(torch.from_numpy(self.arrays[name][indices].copy()) for name in self.spec)

    def flush(self):
        for name, value in self.arrays.items():
            value.flush()
            # Windows FlushFileBuffers requires a writable handle.
            with (self.directory / f'{name}.dat').open('r+b') as stream:
                os.fsync(stream.fileno())

    def commit(self, checkpoint):
        self.flush()
        next_meta = self.metadata(checkpoint)
        next_meta['generation'] += 1
        atomic_json(self.directory / 'commit.json', next_meta)
        if self._journal is not None:
            self._journal.close()
            self._journal = None
        self.generation += 1
        self.committed_size = self.size
        self._backed.clear()
        return next_meta

    def close(self):
        self.flush()
        if self._journal is not None:
            self._journal.close()
            self._journal = None
        for value in self.arrays.values():
            value._mmap.close()
        self.arrays.clear()
        self._lock.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(self._lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
        self._lock.close()

    def __len__(self):
        return self.size
