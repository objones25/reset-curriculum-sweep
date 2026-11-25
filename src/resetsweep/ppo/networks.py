"""Separate actor and critic MLPs; Gaussian policy with a state-independent log-std."""

import torch
from torch import Tensor, nn
from torch.distributions import Normal


def _mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for width in hidden:
        layers += [nn.Linear(in_dim, width), nn.ELU()]
        in_dim = width
    layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Unbounded Gaussian policy and a value head on the same (normalised) observation.

    The env clips actions to the actuator range, so log-probs are always of the unclipped
    sample `act` returns; never clip before storing it. ELU and an initial std of 1.0 follow
    mjlab's G1 velocity config (robot-learning-expert `references/concepts/policy-gradient.md`).
    Tensors may carry any leading batch dims: `(..., obs_dim)` in, `(...)` log-probs and values out.
    """

    def __init__(
        self, obs_dim: int, act_dim: int, *, hidden: tuple[int, ...] = (256, 256)
    ) -> None:
        super().__init__()
        if obs_dim <= 0 or act_dim <= 0:
            raise ValueError(
                f"obs_dim and act_dim must be positive, got {obs_dim}, {act_dim}"
            )
        self.actor = _mlp(obs_dim, hidden, act_dim)
        self.critic = _mlp(obs_dim, hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def _dist(self, obs: Tensor) -> Normal:
        mean = self.actor(obs)
        return Normal(mean, self.log_std.exp().expand_as(mean))

    @torch.no_grad()
    def act(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Rollout step, without gradients: `(action, logp, value)`."""
        dist = self._dist(obs)
        action = dist.sample()
        return action, dist.log_prob(action).sum(dim=-1), self.value(obs)

    def evaluate(self, obs: Tensor, action: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """`(logp, entropy, value)` of stored actions, with gradients."""
        dist = self._dist(obs)
        return (
            dist.log_prob(action).sum(dim=-1),
            dist.entropy().sum(dim=-1),
            self.value(obs),
        )

    def value(self, obs: Tensor) -> Tensor:
        return self.critic(obs).squeeze(-1)
