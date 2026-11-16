"""Behavioral contracts for the isolated paper experiment (no production runs)."""
import json
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.paper_sac.config import load_bundle, validate_bundle
from scripts.paper_sac.policy import sample_strategy
from scripts.paper_sac.replay import MemmapReplayBuffer, replay_bytes
from scripts.paper_sac.learner import SAC, RunningStats
from scripts.paper_sac.checkpoint import save_training, restore_training
from scripts.paper_sac.metrics import Metrics, write_result, summarize_records


def tiny_config():
    cfg = load_bundle()
    cfg['algorithm'].update(replay_buffer_size=4, batch_size=2, max_tasks=400,
                            total_steps=64, warmup_steps=8, device='cpu')
    return cfg


def transition(value=1, count=2, following=3):
    return (np.full((count, 34), value, np.float32), [1] * count, float(value),
            np.full((following, 34), value + 1, np.float32), False)


def test_config_rejects_shuffling_and_accepts_exact_dimensions():
    cfg = load_bundle()
    validate_bundle(cfg)
    cfg['train_sim']['shuffle_datacenters'] = True
    with pytest.raises(ValueError, match='shuffle'):
        validate_bundle(cfg)


def test_replay_memory_includes_int64_actions_and_two_float_masks():
    assert replay_bytes(100000, 400, 34) == 11520800000


@pytest.mark.parametrize('strategy', ['geo_only', 'time_only', 'geo_time'])
def test_sampling_is_reproducible_and_obeys_constraints(strategy):
    logits = torch.zeros(6000, 6)
    origins = [i % 5 + 1 for i in range(6000)]
    a = sample_strategy(logits, origins, strategy, torch.Generator().manual_seed(17))
    torch.manual_seed(999)
    b = sample_strategy(logits, origins, strategy, torch.Generator().manual_seed(17))
    assert a == b
    if strategy == 'geo_only':
        assert set(a) == {1, 2, 3, 4, 5}
        counts = np.bincount(a, minlength=6)[1:]
        assert max(abs(counts - 1200)) < 150
    elif strategy == 'time_only':
        assert all(x == 0 or x == origin for x, origin in zip(a, origins))
        assert 850 < a.count(0) < 1150
    else:
        assert set(a) == set(range(6))


def test_geo_only_removes_defer_before_softmax_even_for_extreme_logits():
    logits = torch.tensor([[100000., 0., -100000., -100000., -100000., -100000.]])
    assert sample_strategy(logits, [5], 'geo_only', torch.Generator()) == [1]


def test_buffer_reopens_committed_data_and_rolls_back_ring_overwrites(tmp_path):
    buf = MemmapReplayBuffer(tmp_path / 'buffer', 4, 400, 34)
    for n in range(4):
        buf.add(*transition(n))
    buf.commit('checkpoint-4.pt')
    expected = buf.arrays['obs'].copy()
    buf.add(*transition(99))
    buf.add(*transition(98))
    buf.flush()
    buf.close()
    recovered = MemmapReplayBuffer.open(tmp_path / 'buffer')
    assert recovered.pos == 0 and len(recovered) == 4
    np.testing.assert_array_equal(recovered.arrays['obs'], expected)
    batch = recovered.sample(2)
    assert batch[0].shape == (2, 400, 34)
    assert batch[-1].sum() == 6
    recovered.close()


def test_buffer_overflow_does_not_change_pointer_or_truncate(tmp_path):
    buf = MemmapReplayBuffer(tmp_path / 'buffer', 4, 400, 34)
    with pytest.raises(ValueError, match='400'):
        buf.add(*transition(count=401))
    assert len(buf) == 0 and buf.pos == 0
    with pytest.raises(ValueError, match='400'):
        buf.add(*transition(following=401))
    buf.close()


def test_buffer_rejects_a_second_live_writer(tmp_path):
    buf = MemmapReplayBuffer(tmp_path / 'buffer', 4, 400, 34)
    try:
        with pytest.raises(RuntimeError, match='in use'):
            other = MemmapReplayBuffer.open(tmp_path / 'buffer')
            other.close()
    finally:
        buf.close()


def test_recovery_ignores_torn_journal_tail_and_can_resume_writing(tmp_path):
    directory = tmp_path / 'buffer'
    buf = MemmapReplayBuffer(directory, 2, 400, 34)
    buf.add(*transition(1))
    buf.add(*transition(2))
    buf.commit('old.pt')
    buf.add(*transition(88))
    generation = buf.generation
    buf.close()
    with (directory / f'undo-{generation % 2}.bin').open('ab') as stream:
        stream.write(b'bad')
    recovered = MemmapReplayBuffer.open(directory)
    assert recovered.arrays['obs'][0, 0, 0] == 1
    recovered.add(*transition(99))
    recovered.close()
    again = MemmapReplayBuffer.open(directory)
    assert again.arrays['obs'][0, 0, 0] == 1
    again.close()


def test_journal_directory_sync_failure_prevents_committed_row_overwrite(tmp_path, monkeypatch):
    from scripts.paper_sac import replay as module
    buf = MemmapReplayBuffer(tmp_path / 'buffer', 1, 400, 34)
    buf.add(*transition(1))
    buf.commit('saved.pt')
    def fail(path):
        raise OSError('directory not durable')
    with monkeypatch.context() as patch:
        patch.setattr(module, 'sync_directory', fail)
        with pytest.raises(OSError, match='not durable'):
            buf.add(*transition(77))
    assert buf.arrays['obs'][0, 0, 0] == 1
    buf.close()


def test_resume_capacity_credits_existing_disk_allocation(tmp_path):
    from scripts.paper_sac.preflight import capacity_report
    cfg = tiny_config()
    buf = MemmapReplayBuffer(tmp_path / 'buffer', 4, 400, 34)
    buf.add(*transition())
    buf.flush()
    buf.close()
    fresh = capacity_report(cfg, tmp_path, 1)
    resumed = capacity_report(cfg, tmp_path, 1, existing_replay=tmp_path / 'buffer')
    assert resumed['existing_allocated_bytes'] > 0
    assert fresh['disk_required_bytes'] - resumed['disk_required_bytes'] == resumed['existing_allocated_bytes']
    assert resumed['replay_bytes_per_seed'] == 460832


def test_checkpoint_restores_target_optimizers_statistics_rng_and_replay(tmp_path):
    cfg = tiny_config()
    torch.manual_seed(42)
    agent = SAC(cfg['algorithm'])
    buf = MemmapReplayBuffer(tmp_path / 'replay', 4, 400, 34)
    for i in range(4):
        buf.add(*transition(i + 1))
    loss = agent.update(buf.sample(2), 10)
    assert all(np.isfinite(v) for v in loss.values())
    stats = RunningStats()
    stats.update(-123)
    state = dict(global_step=64, episode=1, best_avg10=-2., recent_rewards=[-3.], seed=42)
    path = tmp_path / 'step-64.pt'
    save_training(path, agent, buf, stats, state, cfg, {'data': 'abc'})
    expected_rng = (random.random(), np.random.rand(), torch.rand(3))
    expected_batch = buf.sample(2)
    expected_losses = agent.update(expected_batch, 12)
    expected_weights = {k: v.clone() for k, v in agent.actor.state_dict().items()}
    buf.add(*transition(100))
    buf.close()
    other = SAC(cfg['algorithm'])
    restored, reopened, restored_stats = restore_training(path, other, cfg, {'data': 'abc'})
    assert restored['global_step'] == 64 and restored['episode'] == 1
    assert restored_stats.__dict__ == stats.__dict__
    assert expected_rng[0] == random.random()
    assert expected_rng[1] == np.random.rand()
    torch.testing.assert_close(expected_rng[2], torch.rand(3))
    actual_losses = other.update(reopened.sample(2), 12)
    assert actual_losses == expected_losses
    for key, value in other.actor.state_dict().items():
        torch.testing.assert_close(value, expected_weights[key], rtol=0, atol=0)
    reopened.close()
    with pytest.raises(ValueError, match='fingerprint'):
        restore_training(path, other, cfg, {'data': 'changed'})


def test_update_handles_empty_next_tasks_without_nan(tmp_path):
    cfg = tiny_config()
    agent = SAC(cfg['algorithm'])
    buf = MemmapReplayBuffer(tmp_path / 'replay', 4, 400, 34)
    buf.add(*transition(following=0))
    assert all(np.isfinite(v) for v in agent.update(buf.sample(2), 2).values())
    buf.close()


def test_metrics_do_not_double_count_dc_and_transmission_and_use_completed_sla(tmp_path):
    metrics = Metrics()
    info = {'datacenter_infos': {'DC1': {'__common__': {
        'energy_consumption_kwh': 1000., 'carbon_emissions_kg': 100.,
        'energy_cost_USD': 50., 'cpu_util_percent': 10., 'gpu_util_percent': 20.,
        'finished_tasks_count': 4, '__sla__': {'met': 3, 'violated': 1}},
        'agent_dc': {'dc_water_usage': 2000.}}},
        'paper': {'transmission_energy_total_kwh': 10.,
                  'transmission_emissions_total_kg': 2., 'transmission_cost_total_usd': 3.,
                  'new_tasks': 5, 'defer_decisions': 2, 'forced_origin': 1, 'remote_migrations': 1,
                  'policy_local_executions': 2}}
    metrics.add(info)
    result = metrics.result(uncompleted=1, uncompleted_overdue=1)
    assert result['total_cost_usd'] == 53
    assert result['total_energy_mwh'] == 1.01
    assert result['total_co2_t'] == .102
    assert result['water_m3'] == 2
    assert result['sla_violation_rate_pct'] == 25
    assert result['unfinished_tasks'] == 1
    assert result['policy_local_execution_count'] == 2
    assert result['energy_cost_usd'] == 50
    write_result(tmp_path / 'result', result)
    assert json.loads((tmp_path / 'result.json').read_text())['total_cost_usd'] == 53
    assert pd.read_csv(tmp_path / 'result.csv').iloc[0].total_cost_usd == 53


def test_inference_snapshot_is_rejected_as_full_resume(tmp_path):
    cfg = tiny_config()
    agent = SAC(cfg['algorithm'])
    path = tmp_path / 'weights.pt'
    torch.save(dict(format=1, episode_boundary=True, inference_only=True), path)
    with pytest.raises(ValueError, match='full resume'):
        restore_training(path, agent, cfg, {})


def test_remaining_training_seed_can_be_started_without_retraining_completed_models(monkeypatch):
    from scripts.paper_sac import train
    def checked_then_stop(*args, **kwargs):
        raise RuntimeError('preflight reached; no training started')
    monkeypatch.setattr(train, 'check', checked_then_stop)
    monkeypatch.setattr('sys.argv', ['train.py', '--production', '--seeds', '44'])
    with pytest.raises(RuntimeError, match='preflight reached'):
        train.main()


def test_summary_keeps_train_and_eval_seeds_and_uses_sample_sd():
    rows = [dict(train_seed=t, eval_seed=e, strategy='geo_time', total_cost_usd=v,
                 status='complete') for t, e, v in [(42, 0, 1), (42, 1, 3), (43, 0, 5), (43, 1, 7)]]
    out = summarize_records(rows)
    assert len(out['per_model']) == 2
    total = out['across_models'][0]
    assert total['total_cost_usd_mean'] == 4
    assert total['total_cost_usd_std'] == pytest.approx(2.8284271247461903)
    assert total['n_train_seeds'] == 2
