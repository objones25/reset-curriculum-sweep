"""M5 `buffer_scored`: score-weighted draw over the visited buffer, with PLR staleness,
discounted per-IC outcomes, a minimum count, a rehearsal share and lanes.

PLR (papers/text/2010.03934_prioritized-level-replay.txt):
- Eq. 1, L174-175: P_replay = (1 - rho) * P_S + rho * P_C, rho in [0, 1] the staleness
  coefficient (`staleness_coef`; rho = 0.3 was best on Procgen, L428).
- Eq. 4, L229-240: P_C(l_i) = (c - C_i) / sum_j (c - C_j), where c is the count of total
  episodes sampled so far and C_i is the episode count at which l_i was last sampled.
- Algorithm 4, L1787/L1802/L1812/L1817: c <- c + 1 per sampled episode, C_i <- c when l_i is
  sampled, and a newly added level starts with C_i = 0.
Here a level is a buffer slot: a newly written slot gets C_i = 0, every reset row (task start
or buffer) adds 1 to c, and a drawn slot gets C_i = c.
"""

import torch
from torch import Tensor

from resetsweep.methods._slots import SlotBuffer
from resetsweep.methods.base import METHODS, MethodContext
from resetsweep.methods.visited import CAPACITY_DEFAULT
from resetsweep.scores.base import median_fill
from resetsweep.types import Outcomes, StartStates

REHEARSAL_SUCCESS_MAX = 0.5  # "recent success below 0.5" (plan, Task 16)


@METHODS.register("buffer_scored")
class BufferScored(SlotBuffer):
    """Draw distribution over filled slots, computed in this order:

    1. P_S = `score.probs(slots)`, the score's own prioritisation (rank for most scores,
       softmax for `sgs` and `critic_value`).
    2. Trust: a slot is trusted when the score trusts it and the method has seen at least
       `min_count` full outcomes for it since it was written. Untrusted slots get the median
       P_S of the trusted ones; if none is trusted, P_S is uniform. Then renormalise.
    3. P = (1 - staleness_coef) * P_S + staleness_coef * P_C (PLR, module docstring).
    4. Rehearsal: slots that ever succeeded and whose discounted recent success is below 0.5
       get a `rehearsal` share, split uniformly: P = (1 - rehearsal) * P + rehearsal * U.
       With no such slot P is unchanged.

    Recent success per slot is discounted per outcome: each new outcome multiplies the old
    success sum and count by `discount` (2607.26417 L799-800: old evidence must decay).
    `lanes` consecutive reset rows share each drawn slot, so `lanes` envs that reset together
    measure one IC in one iteration; envs that reset alone get their own draw.

    Outcome bookkeeping (stale outcomes dropped, `score.forget` on overwrite) is in `SlotBuffer`.
    """

    name = "buffer_scored"
    accepts_scores = frozenset(
        {
            "success_band",
            "slope",
            "critic_loss",
            "critic_value",
            "success_classifier",
            "rnd",
            "random",
            "inverted",
        }
    )

    def __init__(
        self,
        ctx: MethodContext,
        *,
        capacity: int = CAPACITY_DEFAULT,
        min_fill: int | None = None,
        staleness_coef: float = 0.3,
        discount: float = 0.99,
        min_count: int = 3,
        rehearsal: float = 0.1,
        lanes: int = 1,
    ) -> None:
        if not 0.0 <= staleness_coef <= 1.0:
            raise ValueError(f"staleness_coef must be in [0, 1], got {staleness_coef}")
        if not 0.0 < discount <= 1.0:
            raise ValueError(f"discount must be in (0, 1], got {discount}")
        if min_count < 1:
            raise ValueError(f"min_count must be at least 1, got {min_count}")
        if not 0.0 <= rehearsal <= 1.0:
            raise ValueError(f"rehearsal must be in [0, 1], got {rehearsal}")
        if lanes < 1:
            raise ValueError(f"lanes must be at least 1, got {lanes}")
        super().__init__(ctx, capacity=capacity, min_fill=min_fill)
        self._rho = staleness_coef
        self._discount = discount
        self._min_count = min_count
        self._rehearsal = rehearsal
        self._lanes = lanes
        d = ctx.device
        rows = capacity + 1
        self._count = torch.zeros(rows, dtype=torch.long, device=d)
        self._disc_count = torch.zeros(rows, device=d)
        self._disc_success = torch.zeros(rows, device=d)
        self._stamp = torch.zeros(rows, dtype=torch.long, device=d)  # C_i
        self._episodes = 0  # c

    def reset(
        self, env_ids: Tensor, outcomes: Outcomes | None
    ) -> tuple[StartStates, Tensor]:
        self._episodes += env_ids.numel()
        return super().reset(env_ids, outcomes)

    def trusted(self) -> Tensor:
        """`(buffer_size,)` bool: which filled slots count as trusted (step 2 above)."""
        return self._trusted(torch.arange(int(self._size), device=self.ctx.device))

    def diagnostics(self) -> dict[str, float]:
        """Adds `rehearsal_candidates`: filled slots currently eligible for rehearsal."""
        diag = super().diagnostics()
        ids = torch.arange(int(self._size), device=self.ctx.device)
        diag["rehearsal_candidates"] = float(self._forgotten(ids).sum())
        return diag

    def _trusted(self, ids: Tensor) -> Tensor:
        _, score_trusted = self._score.values(ids)
        return score_trusted & (self._count[ids] >= self._min_count)

    def _forgotten(self, ids: Tensor) -> Tensor:
        recent = self._disc_success[ids] / self._disc_count[ids].clamp(min=1e-12)
        return (
            self._ever[ids] & (self._count[ids] > 0) & (recent < REHEARSAL_SUCCESS_MAX)
        )

    def _probs(self, size: int) -> Tensor:
        ids = torch.arange(size, device=self.ctx.device)
        uniform = torch.full((size,), 1.0 / size, device=self.ctx.device)
        p_s = median_fill(self._score.probs(ids), self._trusted(ids))
        p_s = _normalise_or(p_s, uniform)
        p_c = _normalise_or((self._episodes - self._stamp[ids]).float(), uniform)
        p = (1.0 - self._rho) * p_s + self._rho * p_c
        forgotten = self._forgotten(ids).float()
        p_r = _normalise_or(forgotten, p)
        return (1.0 - self._rehearsal) * p + self._rehearsal * p_r

    def _on_draw(self, idx: Tensor) -> None:
        self._stamp[idx] = self._episodes

    def _on_outcomes(self, outcomes: Outcomes) -> None:
        slot = self._recorded_slots(outcomes)
        n_new = torch.zeros_like(self._disc_count).index_add_(
            0, slot, torch.ones_like(slot, dtype=torch.float32)
        )
        s_new = torch.zeros_like(self._disc_count).index_add_(
            0, slot, outcomes.success.float()
        )
        # Outcomes of one batch count as equally old: decay by discount^(new outcomes).
        decay = self._discount**n_new
        self._disc_count = decay * self._disc_count + n_new
        self._disc_success = decay * self._disc_success + s_new
        self._count += n_new.long()

    def _forget_slots(self, ids: Tensor) -> None:
        self._count[ids] = 0
        self._disc_count[ids] = 0.0
        self._disc_success[ids] = 0.0
        self._stamp[ids] = 0


def _normalise_or(weights: Tensor, fallback: Tensor) -> Tensor:
    """`weights / weights.sum()`, or `fallback` when the weights sum to zero (no host sync)."""
    total = weights.sum()
    return torch.where(total > 0, weights / total.clamp(min=1e-30), fallback)
