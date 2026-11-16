"""Real simulator checks: a few steps only, never production training."""
import numpy as np
import pandas as pd
import pytest

from scripts.paper_sac.config import load_bundle
from scripts.paper_sac.environment import make_env
from scripts.paper_sac.data import workload_source, workload_counts


def test_workload_read_only_fallback_and_task_scale_count():
    cfg = load_bundle()
    path, member = workload_source(cfg['train_sim']['workload_path'])
    assert path.exists()
    stats = workload_counts(cfg['train_sim'], cfg['datacenters'])
    assert stats['steps'] == 672
    assert stats['maximum'] <= 400
    assert stats['p95'] <= stats['maximum']


@pytest.mark.parametrize('mode', ['fixed', 'randomized_aligned'])
def test_real_environment_clocks_dimensions_order_and_forced_return(mode):
    cfg = load_bundle()
    cfg['train_sim']['reset_mode'] = mode
    env = make_env(cfg, phase='train_sim', episode_steps=4)
    obs, _ = env.reset(seed=42)
    assert env.observation_space.shape == (34,)
    assert env.agent_output_act_dim == 6
    assert [dc.dc_id for dc in env.cluster_manager.datacenters.values()] == [1, 2, 3, 4, 5]
    start = env.current_time
    if mode == 'fixed':
        assert start == pd.Timestamp('2023-08-01T05:00Z')
    assert all(x.shape == (34,) for x in obs)
    # Deliberately expire one task: environment must override an actual Defer.
    first = env.current_tasks[0]
    first.sla_deadline = env.current_time - pd.Timedelta(minutes=1)
    _, _, _, _, info = env.step([0] * len(obs))
    assert info['paper']['forced_origin'] == 1
    assert info['paper']['defer_decisions'] == len(obs) - 1
    assert first.dest_dc_id == first.origin_dc_id
    interim = env.final_metrics()
    assert interim['new_arrival_tasks'] == interim['completed_tasks'] + interim['unfinished_tasks']
    for dc in env.cluster_manager.datacenters.values():
        assert dc.current_time_task == start
        # The next observation uses current UTC data, not the old DC clock.
        assert dc.price_manager.index == int((env.current_time - pd.Timestamp('2023-01-01T00:00Z')).total_seconds() / 900)
    for _ in range(3):
        # Execute locally to avoid manufacturing task overflow in this test.
        obs, _, done, _, info = env.step([t.origin_dc_id for t in env.current_tasks])
        for dc in env.cluster_manager.datacenters.values():
            idx = int((dc.current_time_task - pd.Timestamp('2023-01-01T00:00Z')).total_seconds() / 900)
            physical = info['datacenter_infos'][f'DC{dc.dc_id}']
            assert physical['__common__']['price_USD_kwh'] == dc.price_manager.prices[idx]
            assert physical['agent_bat']['bat_avg_CI'] == dc.ci_manager.carbon_smooth[idx]
            assert physical['agent_dc']['dc_exterior_ambient_temp'] == dc.weather_manager.temperature_data[idx]
    assert done
    totals = env.final_metrics()
    assert totals['actual_steps'] == 4
    assert totals['new_arrival_tasks'] == totals['completed_tasks'] + totals['unfinished_tasks']
    assert all(np.isfinite(totals[k]) for k in ('total_cost_usd', 'total_energy_mwh', 'water_m3'))
    env.close()


def test_unpublished_checkpoint_does_not_block_replayed_boundary(tmp_path, monkeypatch):
    from scripts.paper_sac.config import smoke_bundle
    from scripts.paper_sac.train import train_seed
    from scripts.paper_sac.replay import MemmapReplayBuffer
    cfg = smoke_bundle(load_bundle())
    original = MemmapReplayBuffer.commit
    def fail_publication(self, checkpoint):
        if '000000032' in checkpoint:
            raise OSError('injected interruption before replay publication')
        return original(self, checkpoint)
    monkeypatch.setattr(MemmapReplayBuffer, 'commit', fail_publication)
    with pytest.raises(OSError, match='injected'):
        train_seed(cfg, 42, tmp_path, 'publication_test', {'fixture': 'same'}, stop_after=32)
    checkpoints = tmp_path / 'checkpoints' / 'seed_42' / 'publication_test'
    orphan = checkpoints / 'step_000000032.pt'
    assert orphan.exists()
    monkeypatch.setattr(MemmapReplayBuffer, 'commit', original)
    result = train_seed(cfg, 42, tmp_path, 'publication_test', {'fixture': 'same'},
                        resume=checkpoints / 'step_000000000.pt', stop_after=32)
    assert result['global_step'] == 32 and result['status'] == 'stopped'
    assert orphan.exists()  # Unpublished artifacts are preserved, never overwritten.
    assert len(list(checkpoints.glob('step_000000032*.pt'))) == 2
