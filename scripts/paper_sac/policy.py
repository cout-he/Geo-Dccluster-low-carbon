"""Stochastic ablation protocols; the generator never uses global torch RNG."""
import torch


def sample_strategy(logits, origins, strategy, generator):
    if strategy not in ('geo_only', 'time_only', 'geo_time'):
        raise ValueError(f'Unknown strategy: {strategy}')
    if logits.ndim != 2 or logits.shape[1] != 6 or len(origins) != len(logits):
        raise ValueError('Expected one six-action row per task')
    if not all(1 <= int(x) <= 5 for x in origins):
        raise ValueError('Task origin DC must be 1..5')
    if not len(origins):
        return []
    # CPU sampling makes the explicit CPU generator portable between CPU/CUDA.
    values = logits.detach().cpu().clone()
    if strategy == 'geo_only':
        values[:, 0] = -torch.inf
    probs = torch.softmax(values, dim=-1)
    if not torch.isfinite(probs).all():
        raise ValueError('Non-finite policy probabilities')
    sampled = torch.multinomial(probs, 1, generator=generator).squeeze(1).tolist()
    if strategy == 'time_only':
        sampled = [0 if a == 0 else int(origin) for a, origin in zip(sampled, origins)]
    return sampled
