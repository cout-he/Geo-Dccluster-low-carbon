"""Discrete SAC, retaining the existing multi-task target/reward conventions."""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from rl_components.agent_net import ActorNet, CriticNet


class RunningStats:
    """Same scalar recurrence as train_SAC_agent.RunningStats; serializable."""
    def __init__(self, eps=1e-5):
        self.mean, self.var, self.count = 0.0, 1.0, eps

    def update(self, x):
        x = float(x)
        self.count += 1
        previous = self.mean
        self.mean += (x - self.mean) / self.count
        self.var += (x - previous) * (x - self.mean)

    def normalize(self, x):
        return (x - self.mean) / max(np.sqrt(self.var / self.count), 1e-6)


class SAC:
    def __init__(self, algorithm):
        self.cfg = copy.deepcopy(algorithm)
        device = algorithm['device']
        self.device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                                   if device == 'auto' else device)
        self.actor = ActorNet(34, 6, 64, use_layer_norm=False).to(self.device)
        self.critic = CriticNet(34, 6, 64, use_layer_norm=False).to(self.device)
        self.target = copy.deepcopy(self.critic)
        self.target.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(algorithm['actor_learning_rate']))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=float(algorithm['critic_learning_rate']))

    @property
    def metadata(self):
        return dict(obs_dim=34, act_dim=6, hidden_dim=64, use_layer_norm=False,
                    use_attention=False, dc_order=[1, 2, 3, 4, 5], defer_action=0,
                    target_semantics='legacy_positional_next_task', alpha=float(self.cfg['alpha']))

    def update(self, batch, step):
        obs, actions, reward, next_obs, done, mask, mask_next = [x.to(self.device) for x in batch]
        b, t, dim = obs.shape
        valid = mask.reshape(-1).bool()
        if not valid.any():
            return {}
        current = obs.reshape(-1, dim)[valid]
        chosen = actions.reshape(-1)[valid]
        if ((chosen < 0) | (chosen >= 6)).any():
            raise ValueError('Invalid replay action')
        # Preserve the original positional next-task Bellman convention. Tasks in
        # successive slots are NOT guaranteed to have the same identity.
        with torch.no_grad():
            nxt = next_obs.reshape(-1, dim)
            logits = self.actor(nxt)
            p, lp = logits.softmax(-1), logits.log_softmax(-1)
            q1, q2 = self.target.forward_all(nxt)
            value = (p * (torch.minimum(q1, q2) - self.cfg['alpha'] * lp)).sum(-1)
            value *= mask_next.reshape(-1)
            targets = reward[:, None] + self.cfg['gamma'] * (1 - done[:, None]) * value.reshape(b, t)
            targets = targets.reshape(-1)[valid]
        q1, q2 = self.critic(current, chosen)
        loss = .5 * (F.mse_loss(q1, targets) + F.mse_loss(q2, targets))
        self.critic_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_opt.step()
        out = {'critic_loss': float(loss.detach())}
        if step % self.cfg['policy_update_frequency'] == 0:
            logits = self.actor(current)
            p, lp = logits.softmax(-1), logits.log_softmax(-1)
            with torch.no_grad():
                q1, q2 = self.critic.forward_all(current)
                q = torch.minimum(q1, q2)
            policy_loss = (p * (self.cfg['alpha'] * lp - q)).sum(-1).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            policy_loss.backward()
            self.actor_opt.step()
            with torch.no_grad():
                for parameter, target in zip(self.critic.parameters(), self.target.parameters()):
                    target.lerp_(parameter, self.cfg['tau'])
            out.update(actor_loss=float(policy_loss.detach()), policy_entropy=float(-(p * lp).sum(-1).mean().detach()))
        if not all(np.isfinite(v) for v in out.values()):
            raise FloatingPointError(f'Non-finite SAC losses: {out}')
        return out
