"""Multi-task discrete SAC with compact positional replay."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from rl_components.agent_net import ActorNet, CriticNet


@dataclass
class ReplayBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    next_indices: torch.Tensor


class ReplayBuffer:
    """Baseline ring sampling with replacement, one entry per nonempty step.

    max_tasks is the block width. An overflowing step is kept as additional
    blocks in the same entry, without changing the step's sampling probability.
    """
    def __init__(self, capacity, max_tasks, obs_dim, act_dim):
        self.capacity = capacity
        self.max_tasks = max_tasks
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.entries = []
        self.position = 0
        self.overflow_steps = 0
        self.max_tasks_seen = 0

    def add(self, obs, actions, reward, next_obs, terminal):
        obs = np.asarray(obs, dtype=np.float32).reshape(-1, self.obs_dim)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1, self.obs_dim)
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (len(obs),) or not len(obs):
            raise ValueError("Replay requires one action per current task and a nonempty current observation")
        if ((actions < 0) | (actions >= self.act_dim)).any():
            raise ValueError("Replay received an action outside the policy's action space")
        if not np.isfinite(reward) or not np.isfinite(obs).all() or not np.isfinite(next_obs).all():
            raise ValueError("Replay received nonfinite inputs")

        task_count = max(len(obs), len(next_obs))
        self.max_tasks_seen = max(self.max_tasks_seen, task_count)
        self.overflow_steps += int(task_count > self.max_tasks)
        transition = (
            tuple(obs[i:i + self.max_tasks] for i in range(0, len(obs), self.max_tasks)),
            tuple(actions[i:i + self.max_tasks] for i in range(0, len(actions), self.max_tasks)),
            np.float32(reward),
            tuple(next_obs[i:i + self.max_tasks] for i in range(0, len(next_obs), self.max_tasks)),
            bool(terminal),
        )
        if len(self.entries) < self.capacity:
            self.entries.append(transition)
        else:
            self.entries[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, device):
        indices = np.random.randint(0, len(self), size=batch_size)
        observations, actions, rewards, next_observations, next_indices = [], [], [], [], []
        offset = 0
        for index in indices:
            obs_blocks, action_blocks, reward, next_blocks, terminal = self.entries[index]
            for i, obs in enumerate(obs_blocks):
                observations.append(obs)
                actions.append(action_blocks[i])
                rewards.append(np.full(len(obs), reward, dtype=np.float32))
                if not terminal and i < len(next_blocks):
                    paired = min(len(obs), len(next_blocks[i]))
                    next_observations.append(next_blocks[i][:paired])
                    next_indices.append(np.arange(offset, offset + paired, dtype=np.int64))
                offset += len(obs)

        empty_next = np.empty((0, self.obs_dim), dtype=np.float32)
        empty_indices = np.empty(0, dtype=np.int64)
        return ReplayBatch(*[
            torch.from_numpy(array).to(device)
            for array in (
                np.concatenate(observations), np.concatenate(actions), np.concatenate(rewards),
                np.concatenate(next_observations) if next_observations else empty_next,
                np.concatenate(next_indices) if next_indices else empty_indices,
            )
        ])

    def __len__(self):
        return len(self.entries)


class SAC:
    def __init__(self, config, device, obs_dim, act_dim):
        self.cfg = config
        self.device = device
        self.alpha = float(config["alpha"])
        hidden = config["hidden_dim"]
        self.actor = ActorNet(obs_dim, act_dim, hidden_dim=hidden, use_layer_norm=False).to(device)
        self.critic = CriticNet(obs_dim, act_dim, hidden_dim=hidden, use_layer_norm=False).to(device)
        self.target_critic = CriticNet(obs_dim, act_dim, hidden_dim=hidden, use_layer_norm=False).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(config["actor_learning_rate"]))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=float(config["critic_learning_rate"]))

    def update(self, batch, step_index):
        # The baseline bootstraps from the next observation in the same list
        # position. Positions do not necessarily refer to the same task.
        with torch.no_grad():
            targets = batch.rewards.clone()
            if len(batch.next_indices):
                logits = self.actor(batch.next_obs)
                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)
                next_q1, next_q2 = self.target_critic.forward_all(batch.next_obs)
                soft_value = (probs * (torch.minimum(next_q1, next_q2) - self.alpha * log_probs)).sum(-1)
                targets[batch.next_indices] += self.cfg["gamma"] * soft_value

        q1, q2 = self.critic(batch.obs, batch.actions)
        critic_loss = 0.5 * (F.mse_loss(q1, targets) + F.mse_loss(q2, targets))
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        losses = {"critic_loss": critic_loss.detach()}
        if step_index % self.cfg["policy_update_frequency"] == 0:
            logits = self.actor(batch.obs)
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            with torch.no_grad():
                q1, q2 = self.critic.forward_all(batch.obs)
                q = torch.minimum(q1, q2)
            actor_loss = (probs * (self.alpha * log_probs - q)).sum(-1).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
            with torch.no_grad():
                for parameter, target in zip(self.critic.parameters(), self.target_critic.parameters()):
                    target.copy_(self.cfg["tau"] * parameter + (1 - self.cfg["tau"]) * target)
            losses.update(
                actor_loss=actor_loss.detach(),
                entropy=(-(probs * log_probs).sum(-1).mean()).detach(),
            )
        return losses

    def state_dict(self):
        return {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "target_critic_state_dict": self.target_critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_opt.state_dict(),
            "critic_optimizer_state_dict": self.critic_opt.state_dict(),
        }

    def load_state_dict(self, state):
        self.actor.load_state_dict(state["actor_state_dict"])
        self.critic.load_state_dict(state["critic_state_dict"])
        self.target_critic.load_state_dict(state["target_critic_state_dict"])
        self.actor_opt.load_state_dict(state["actor_optimizer_state_dict"])
        self.critic_opt.load_state_dict(state["critic_optimizer_state_dict"])
