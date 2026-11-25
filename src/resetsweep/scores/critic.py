"""S4 and S5: scores read off the PPO critic.

`critic_loss` is S4 (per-IC value loss, MaxMC or |GAE|); `critic_value` is S5 (V at the IC's
start state). Both take their data from `observe_rollout`; `record` is a no-op.
"""

from typing import Literal

import torch
from torch import Tensor

from resetsweep.scores.base import SCORES, BaseScore, median_fill
from resetsweep.types import Outcomes, RolloutStats

Form = Literal["pos_value_loss", "max_mc", "abs_gae"]
_FORMS: tuple[Form, ...] = ("pos_value_loss", "max_mc", "abs_gae")


class _PerICScore(BaseScore):
    """A latest-value-per-IC store with a trusted flag; subclasses set it in `observe_rollout`."""

    def __init__(self, *, capacity: int, device: str, beta: float) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._set_beta(beta)
        self._capacity = capacity
        self._device = device
        # The extra last row absorbs ignored (`ic_id == -1`) entries, so no shape depends on data.
        self._lp = torch.zeros(capacity + 1, dtype=torch.float32, device=device)
        self._seen = torch.zeros(capacity + 1, dtype=torch.bool, device=device)

    def _check_ids(self, ic_ids: Tensor) -> None:
        if ic_ids.numel() and bool(((ic_ids < 0) | (ic_ids >= self._capacity)).any()):
            raise ValueError(f"ic_ids must be in [0, {self._capacity})")

    def _slots(self, ic_ids: Tensor) -> Tensor:
        """Flatten rollout ids; `-1` goes to the spare row, anything else out of range raises."""
        flat = ic_ids.reshape(-1)
        if flat.numel() and bool(((flat < -1) | (flat >= self._capacity)).any()):
            raise ValueError(f"ic_ids must be -1 or in [0, {self._capacity})")
        return torch.where(flat >= 0, flat, torch.full_like(flat, self._capacity))

    def record(self, outcomes: Outcomes) -> None:
        del outcomes

    def values(self, ic_ids: Tensor) -> tuple[Tensor, Tensor]:
        self._check_ids(ic_ids)
        trusted = self._seen[ic_ids]
        return median_fill(self._lp[ic_ids], trusted), trusted

    def forget(self, ic_ids: Tensor) -> None:
        self._check_ids(ic_ids)
        self._lp[ic_ids] = 0.0
        self._seen[ic_ids] = False

    def _store(self, slots: Tensor, new_lp: Tensor, touched: Tensor) -> None:
        """Overwrite the score of every IC in `touched` (spare row never counts)."""
        touched = touched.clone()
        touched[self._capacity] = False
        self._lp = torch.where(touched, new_lp, self._lp)
        self._seen = self._seen | touched


@SCORES.register("critic_loss")
class CriticLoss(_PerICScore):
    """Per-IC regret proxy from the rollout; each rollout replaces the score of the ICs in it.

    With rows `t` of one IC (rollout rows with `ic_id == -1` are excluded):
    - `pos_value_loss`: mean of `relu(adv_t)`, the GAE advantage being the positive-clipped
      value loss of PVL (papers/text/2408.15099 L125-138).
    - `max_mc`: `max(returns) - mean(values)`. The paper defines MaxMC as
      `mean_t(Rmax - V(o_t))` (L140-146), which equals `Rmax - mean(V)` for a fixed `Rmax`;
      the paper's Rmax is the highest return ever obtained on the level, here it is the
      highest of this rollout's rows for the IC, since no history is kept.
    - `abs_gae`: mean of `|adv_t|`.
    """

    def __init__(
        self, *, capacity: int, device: str, form: Form, beta: float = 0.1
    ) -> None:
        if form not in _FORMS:
            raise ValueError(f"form must be one of {_FORMS}, got {form!r}")
        super().__init__(capacity=capacity, device=device, beta=beta)
        self._form: Form = form

    def observe_rollout(self, stats: RolloutStats) -> None:
        slots = self._slots(stats.ic_ids)
        rows = self._capacity + 1
        count = torch.zeros(rows, device=self._device).scatter_add_(
            0, slots, torch.ones_like(slots, dtype=torch.float32)
        )
        if self._form == "max_mc":
            top = torch.full((rows,), -torch.inf, device=self._device).scatter_reduce(
                0, slots, stats.returns.reshape(-1).float(), "amax"
            )
            total = torch.zeros(rows, device=self._device).scatter_add_(
                0, slots, stats.values.reshape(-1).float()
            )
            new_lp = top - total / count.clamp(min=1.0)
        else:
            adv = stats.adv.reshape(-1).float()
            term = adv.clamp(min=0.0) if self._form == "pos_value_loss" else adv.abs()
            total = torch.zeros(rows, device=self._device).scatter_add_(0, slots, term)
            new_lp = total / count.clamp(min=1.0)
        self._store(slots, new_lp, count > 0)


@SCORES.register("critic_value")
class CriticValue(_PerICScore):
    """lp = V(s0) * (vmax - V(s0)), the uniform-target ablation of ProCuRL
    (papers/text/2405.02481 L263-266 for the target form, L378-381 for the uniform form).

    The paper samples `P(c) ~ exp(beta * V/Vmax * (Vmax - V))` (beta = 90 to 130, L852-853).
    `values` returns the un-normalised `V(vmax - V)`; `probs` is
    `softmax(lp / vmax / temperature)`, i.e. the paper's rule with `beta = 1 / temperature`.
    `temperature = 1` is therefore beta = 1; the paper's settings are temperature ~ 0.01.

    `stats.start_obs` must have one row per IC id (shape `(capacity, obs_dim)`, row `i` the
    observation at IC `i`'s start state). Only ICs that appear in the rollout are updated
    and trusted, so rows for other ICs may hold anything.
    """

    def __init__(
        self,
        *,
        capacity: int,
        device: str,
        vmax: float,
        temperature: float = 1.0,
        beta: float = 0.1,
    ) -> None:
        if vmax <= 0.0:
            raise ValueError(f"vmax must be positive, got {vmax}")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        super().__init__(capacity=capacity, device=device, beta=beta)
        self._vmax = vmax
        self._temperature = temperature

    def observe_rollout(self, stats: RolloutStats) -> None:
        if stats.start_obs.shape[0] != self._capacity:
            raise ValueError(
                f"start_obs needs one row per IC ({self._capacity}), "
                f"got {stats.start_obs.shape[0]}"
            )
        slots = self._slots(stats.ic_ids)
        present = torch.zeros(self._capacity + 1, dtype=torch.bool, device=self._device)
        present[slots] = True
        with torch.no_grad():
            v = stats.critic(stats.start_obs).reshape(-1).float()
        if v.shape != (self._capacity,):
            raise ValueError(
                f"critic must return one value per start_obs row, got shape {tuple(v.shape)}"
            )
        new_lp = torch.cat([v * (self._vmax - v), v.new_zeros(1)])
        self._store(slots, new_lp, present)

    def probs(self, ic_ids: Tensor) -> Tensor:
        lp, _ = self.values(ic_ids)
        return torch.softmax(lp / (self._vmax * self._temperature), dim=0)
