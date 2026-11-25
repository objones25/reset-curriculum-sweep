"""Generalized advantage estimation that keeps truncation separate from termination."""

import torch
from torch import Tensor


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    bootstrap_values: Tensor,
    *,
    gamma: float,
    lam: float,
) -> tuple[Tensor, Tensor]:
    """GAE(gamma, lambda) advantages and value targets for a `(T, N)` rollout.

    `values[t]` is V(s_t). `bootstrap_values[t]` is V(s_{t+1}) of the true next state: the
    critic on `StepOut.final_obs`, the observation before any auto-reset. So it is right at a
    truncated step and at the last rollout step alike, and the observation after a reset (which
    belongs to the next episode) is never used.

        delta_t = r_t + gamma * (1 - terminated_t) * bootstrap_values[t] - values[t]
        A_t     = delta_t + gamma * lam * (1 - done_t) * A_{t+1},   done = terminated | truncated

    Returns `(advantages, returns = advantages + values)`, both `(T, N)`.
    Same recursion as robot-learning-expert `scripts/rl_math.py::gae`.
    """
    shape = rewards.shape
    if rewards.dim() != 2 or any(
        x.shape != shape for x in (values, terminated, truncated, bootstrap_values)
    ):
        raise ValueError(
            "all inputs must share shape (T, N), got "
            f"{[tuple(x.shape) for x in (rewards, values, terminated, truncated, bootstrap_values)]}"
        )
    if not (0.0 <= gamma <= 1.0 and 0.0 <= lam <= 1.0):
        raise ValueError(f"gamma and lam must be in [0, 1], got {gamma}, {lam}")
    not_term = (~terminated.bool()).to(rewards.dtype)
    not_done = (~(terminated.bool() | truncated.bool())).to(rewards.dtype)
    delta = rewards + gamma * not_term * bootstrap_values - values
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(rewards[0])
    for t in reversed(range(shape[0])):
        last = delta[t] + gamma * lam * not_done[t] * last
        adv[t] = last
    return adv, adv + values
