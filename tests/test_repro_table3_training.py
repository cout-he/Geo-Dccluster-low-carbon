import numpy as np
import torch

from scripts.repro_table3.learning import ReplayBuffer, SAC


def test_replay_keeps_overflow_in_one_positional_transition():
    replay = ReplayBuffer(capacity=2, max_tasks=400, obs_dim=34, act_dim=6)
    obs = np.arange(403 * 34, dtype=np.float32).reshape(403, 34)
    nxt = np.arange(402 * 34, dtype=np.float32).reshape(402, 34) + 20000
    actions = np.arange(403) % 6

    replay.add(obs, actions, -0.75, nxt, False)
    batch = replay.sample(1, torch.device("cpu"))

    assert len(replay) == 1
    np.testing.assert_array_equal(batch.obs.numpy(), obs)
    np.testing.assert_array_equal(batch.actions.numpy(), actions)
    np.testing.assert_array_equal(batch.next_obs.numpy(), nxt)
    np.testing.assert_array_equal(batch.next_indices.numpy(), np.arange(402))
    np.testing.assert_array_equal(batch.rewards.numpy(), np.full(403, -0.75))
    assert replay.max_tasks_seen == 403
    assert replay.overflow_steps == 1


def test_replay_terminal_and_wrapped_slots():
    replay = ReplayBuffer(capacity=1, max_tasks=400, obs_dim=34, act_dim=6)
    replay.add(np.ones((3, 34)), [0, 1, 2], -1, np.zeros((4, 34)), False)
    replay.add(np.zeros((1, 34)), [5], 2, np.ones((2, 34)), True)
    batch = replay.sample(2, torch.device("cpu"))
    assert len(replay) == 1
    assert batch.actions.tolist() == [5, 5]
    assert batch.rewards.tolist() == [2, 2]
    assert not len(batch.next_indices)


def test_original_sac_mlp_update_and_fixed_alpha():
    cfg = dict(hidden_dim=64, actor_learning_rate=1e-4, critic_learning_rate=3e-4,
               alpha=0.01, gamma=0.99, tau=0.005, policy_update_frequency=2)
    learner = SAC(cfg, torch.device("cpu"), obs_dim=34, act_dim=6)
    assert [layer.out_features for layer in learner.actor.net if isinstance(layer, torch.nn.Linear)] == [64, 64, 6]
    for model in (learner.actor, learner.critic, learner.target_critic):
        for parameter in model.parameters():
            parameter.data.zero_()
    replay = ReplayBuffer(capacity=2, max_tasks=400, obs_dim=34, act_dim=6)
    replay.add(np.ones((2, 34)), [1, 1], 1, np.zeros((1, 34)), False)

    losses = learner.update(replay.sample(1, torch.device("cpu")), step_index=0)

    soft_value = 0.01 * np.log(6)
    expected_loss = (1 + (1 + 0.99 * soft_value) ** 2) / 2
    assert np.isclose(losses["critic_loss"], expected_loss)
    assert np.isclose(losses["entropy"], np.log(6))
    assert "actor_loss" in losses
    assert learner.alpha == 0.01



def test_training_resume_reproduces_an_uninterrupted_run(tmp_path, monkeypatch):
    from argparse import Namespace
    from pathlib import Path
    import random
    from types import SimpleNamespace

    from scripts.repro_table3 import train
    from scripts.repro_table3.training_io import COMPONENTS

    class FakeReward:
        def get_last_components(self):
            return dict.fromkeys(COMPONENTS, -0.02)

    class FakeEnvironment:
        obs_dim_per_task = 34
        agent_output_act_dim = 6
        single_action_mode = False
        disable_defer_action = False

        def __init__(self):
            self.cluster_manager = SimpleNamespace(tasks=np.ones((4,)))
            self.reward_fn = FakeReward()
            self.steps = 0

        def reset(self, seed):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            self.steps = 0
            self.private_random = np.random.default_rng(seed)
            info = dict(reset_seed=seed, episode_start_utc="2023-07-01T00:00:00+00:00",
                        workload_start_utc="2020-07-01T00:00:00+00:00", dc_source_indices={})
            return np.zeros((2, 34), dtype=np.float32), info

        def step(self, actions):
            assert self.cluster_manager.tasks is not None
            self.steps += 1
            obs = np.full((2, 34), self.private_random.random(), dtype=np.float32)
            reward = (random.random() + np.random.random() + torch.rand(()).item()) * sum(actions)
            dc = {"energy_cost_USD": 1.0, "carbon_emissions_kg": 5.0, "finished_tasks_count": 1,
                  "__sla__": {"violated": self.steps == 3}}
            info = {"datacenter_infos": {"DC1": {"__common__": dc}},
                    "transmission_cost_total_usd": 0.1, "transmission_emissions_total_kg": 0.01}
            return obs, reward, self.steps == 3, False, info

    config = SimpleNamespace(
        algorithm=dict(training_seed=42, hidden_dim=64, actor_learning_rate=1e-4,
                       critic_learning_rate=3e-4, alpha=.01, gamma=.99, tau=.005,
                       policy_update_frequency=2, update_frequency=1, device="cpu",
                       max_tasks=400, batch_size=512, replay_buffer_size=100000,
                       save_interval=10000, log_interval=100),
        sim={"use_tensorboard": False}, reward={}, datacenters=[], manifest={},
    )
    monkeypatch.setattr(train, "load_config", lambda *args: config)
    monkeypatch.setattr(train, "make_env", lambda *args: FakeEnvironment())
    monkeypatch.setattr(train, "input_hashes", lambda *args: {})
    monkeypatch.setattr(train, "git_metadata", lambda: {"commit": "test"})
    monkeypatch.setattr(train, "snapshot_config", lambda *args: None)

    def run(output, stop=None, resume=None):
        return train.train(Namespace(config_dir=Path("."), output_dir=output, smoke_steps=9,
                                     stop_step=stop, resume=resume))

    sample = ReplayBuffer.sample
    sampling_states = []

    def observed_sample(self, *args):
        sampling_states.append(np.random.get_state())
        return sample(self, *args)

    monkeypatch.setattr(ReplayBuffer, "sample", observed_sample)
    direct, resumed = (tmp_path / name for name in ("direct", "resumed"))
    baseline = run(direct)
    # Updates begin at completed step 4. At step 6, the episode reset
    # must seed the NumPy replay draw before the update, as in master.
    seed_state = np.random.RandomState(42 + 5).get_state()
    assert sampling_states[2][0] == seed_state[0]
    np.testing.assert_array_equal(sampling_states[2][1], seed_state[1])
    assert sampling_states[2][2:] == seed_state[2:]
    paused = run(resumed, stop=5)
    assert paused["status"] == "paused"
    assert paused["critic_updates"] > 0
    assert not (resumed / "checkpoints" / "final.pt").exists()
    with (resumed / "train_steps.csv").open("a", encoding="utf-8") as stream:
        stream.write("simulated CSV write after the last committed checkpoint\n")
    result = run(resumed, resume=resumed / "checkpoints" / "resume_state.pkl")
    assert result["status"] == "complete"
    assert result["training_run_id"] == paused["training_run_id"]
    assert result["critic_updates"] == baseline["critic_updates"]
    for name in ("train_steps.csv", "train_reward_components.csv"):
        assert (direct / name).read_bytes() == (resumed / name).read_bytes()
    left, right = (
        torch.load(output / "checkpoints" / "final.pt", weights_only=True)
        for output in (direct, resumed)
    )
    for name in ("actor_state_dict", "critic_state_dict"):
        assert all(torch.equal(left[name][key], right[name][key]) for key in left[name])

    manifest = resumed / "checkpoints_manifest.csv"
    manifest_data = manifest.read_bytes()
    # Resume after an interruption between publishing the final state and metadata.
    (resumed / "training_metadata.json").unlink()
    finalized = run(resumed, resume=resumed / "checkpoints" / "resume_state.pkl")
    assert finalized["final_checkpoint_sha256"] == result["final_checkpoint_sha256"]
    assert manifest.read_bytes() == manifest_data
