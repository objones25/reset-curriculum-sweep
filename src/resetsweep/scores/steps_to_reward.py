"""S6: steps-to-reward along a stored successful trajectory (backward order)."""

import torch
from torch import Tensor

from resetsweep.scores.base import SCORES, BaseScore, median_fill
from resetsweep.types import Outcomes


@SCORES.register("steps_to_reward")
class StepsToReward(BaseScore):
    """lp = -steps, so the IC closest to the reward ranks first (Backplay's backward order;
    reset_curriculum_shortlist.md section 4, item 6).

    `own_success` calls `set_steps(ic_ids, steps)` for ICs cut from a successful trajectory.
    Only those ICs are trusted; the rest get the median of the trusted ones.
    """

    def __init__(self, *, capacity: int, device: str, beta: float = 0.1) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._set_beta(beta)
        self._capacity = capacity
        self._steps = torch.zeros(capacity, dtype=torch.float32, device=device)
        self._has_steps = torch.zeros(capacity, dtype=torch.bool, device=device)

    def _check_ids(self, ic_ids: Tensor) -> None:
        if ic_ids.numel() and bool(((ic_ids < 0) | (ic_ids >= self._capacity)).any()):
            raise ValueError(f"ic_ids must be in [0, {self._capacity})")

    def record(self, outcomes: Outcomes) -> None:
        del outcomes

    def set_steps(self, ic_ids: Tensor, steps: Tensor) -> None:
        self._check_ids(ic_ids)
        if ic_ids.shape != steps.shape:
            raise ValueError(
                f"ic_ids and steps must have the same shape, got {tuple(ic_ids.shape)} "
                f"and {tuple(steps.shape)}"
            )
        if steps.numel() and bool((steps < 0).any()):
            raise ValueError("steps must be non-negative")
        self._steps[ic_ids] = steps.to(torch.float32)
        self._has_steps[ic_ids] = True

    def values(self, ic_ids: Tensor) -> tuple[Tensor, Tensor]:
        self._check_ids(ic_ids)
        trusted = self._has_steps[ic_ids]
        return median_fill(-self._steps[ic_ids], trusted), trusted

    def forget(self, ic_ids: Tensor) -> None:
        self._check_ids(ic_ids)
        self._steps[ic_ids] = 0.0
        self._has_steps[ic_ids] = False
