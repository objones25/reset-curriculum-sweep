"""S2: visit count. Prefers ICs the method has restarted from least."""

from typing import Literal

import torch
from torch import Tensor

from resetsweep.scores.base import SCORES, BaseScore
from resetsweep.scores.success_band import SuccessBand
from resetsweep.types import Outcomes

PostReward = Literal["count", "count_x_learnability"]
_POST_REWARD: tuple[PostReward, ...] = ("count", "count_x_learnability")


@SCORES.register("visit_count")
class VisitCount(BaseScore):
    """lp = 1 / sqrt(count + 1), the Go-Explore cell weight (papers/text/2004.12919 L1632-1633;
    reset_curriculum_shortlist.md section 4, item 2).

    Counts are owned by the method, which calls `set_counts(ic_ids, counts)`; an IC that was
    never set has count 0. Every IC is trusted, since a count is always defined.

    After `notify_first_reward()` the `count_x_learnability` form multiplies by
    `success_band(pq) + eps`, where `eps` keeps the product positive while no IC has a
    success yet (a plain product would be zero everywhere). The `count` form never changes.
    The success-band part is an inner `SuccessBand(form="pq")` fed by `record`.
    """

    def __init__(
        self,
        *,
        capacity: int,
        device: str,
        post_reward: PostReward = "count",
        eps: float = 0.05,
        history: int = 100,
        min_count: int = 1,
        beta: float = 0.1,
    ) -> None:
        if post_reward not in _POST_REWARD:
            raise ValueError(
                f"post_reward must be one of {_POST_REWARD}, got {post_reward!r}"
            )
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        self._set_beta(beta)
        self._capacity = capacity
        self._band = SuccessBand(
            capacity=capacity,
            device=device,
            history=history,
            form="pq",
            min_count=min_count,
        )
        self._counts = torch.zeros(capacity, dtype=torch.int64, device=device)
        self._post_reward: PostReward = post_reward
        self._eps = eps
        self._rewarded = False

    def record(self, outcomes: Outcomes) -> None:
        self._band.record(outcomes)

    def notify_first_reward(self) -> None:
        self._rewarded = True

    def set_counts(self, ic_ids: Tensor, counts: Tensor) -> None:
        self._band.table.check_ids(ic_ids)
        if ic_ids.shape != counts.shape:
            raise ValueError(
                f"ic_ids and counts must have the same shape, got {tuple(ic_ids.shape)} "
                f"and {tuple(counts.shape)}"
            )
        if counts.numel() and bool((counts < 0).any()):
            raise ValueError("counts must be non-negative")
        self._counts[ic_ids] = counts.to(torch.int64)

    def values(self, ic_ids: Tensor) -> tuple[Tensor, Tensor]:
        self._band.table.check_ids(ic_ids)
        lp = 1.0 / torch.sqrt(self._counts[ic_ids].to(torch.float32) + 1.0)
        if self._rewarded and self._post_reward == "count_x_learnability":
            learnability, _ = self._band.values(ic_ids)
            lp = lp * (learnability + self._eps)
        return lp, torch.ones_like(ic_ids, dtype=torch.bool)

    def forget(self, ic_ids: Tensor) -> None:
        self._band.forget(ic_ids)
        self._counts[ic_ids] = 0
