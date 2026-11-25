"""Score protocol, the shared per-IC outcome table and the rank-based default sampler."""

from typing import Protocol

import torch
from torch import Tensor

from resetsweep.registry import make_registry
from resetsweep.types import Outcomes, RolloutStats

SCORES = make_registry("score")


class Score(Protocol):
    """Per-IC learning potential. `values` returns `(lp, trusted)`; untrusted entries of `lp`
    already hold the median of the trusted ones. `probs` turns lp into sampling weights."""

    def record(self, outcomes: Outcomes) -> None: ...

    def observe_rollout(self, stats: RolloutStats) -> None: ...

    def values(self, ic_ids: Tensor) -> tuple[Tensor, Tensor]: ...

    def probs(self, ic_ids: Tensor) -> Tensor: ...

    def forget(self, ic_ids: Tensor) -> None: ...


class ICTable:
    """Per-IC circular history of the last `history` outcomes, all in device tensors.

    Only rows with `full=True` and `ic_id >= 0` are recorded: `-1` marks task-start resets
    and `full=False` marks staggered first episodes. Ignored rows are routed to a spare
    row at index `capacity`, so `record` has no data-dependent shapes and no host sync
    beyond one bounds check.
    """

    def __init__(self, capacity: int, device: str, *, history: int = 100) -> None:
        if capacity <= 0 or history <= 0:
            raise ValueError(
                f"capacity and history must be positive, got {capacity}, {history}"
            )
        self.capacity = capacity
        self.history = history
        rows = capacity + 1  # last row absorbs ignored outcomes
        self._success = torch.zeros(rows, history, dtype=torch.bool, device=device)
        self._ret = torch.zeros(rows, history, dtype=torch.float32, device=device)
        self._seen = torch.zeros(rows, dtype=torch.int64, device=device)
        self._succeeded = torch.zeros(rows, dtype=torch.int64, device=device)

    def record(self, outcomes: Outcomes) -> None:
        ic_ids = outcomes.ic_ids
        if ic_ids.numel() == 0:
            return
        if bool((ic_ids >= self.capacity).any()):
            raise ValueError(f"ic_id out of range for capacity {self.capacity}")
        keep = outcomes.full & (ic_ids >= 0)
        slot_ic = torch.where(keep, ic_ids, torch.full_like(ic_ids, self.capacity))
        # Sort so rows of the same IC are adjacent (stable keeps their time order), then give
        # each row its rank within its IC: the k-th new outcome lands k slots after `seen`.
        order = torch.argsort(slot_ic, stable=True)
        ic = slot_ic[order]
        idx = torch.arange(ic.numel(), device=ic.device)
        is_start = torch.cat(
            [torch.ones_like(ic[:1], dtype=torch.bool), ic[1:] != ic[:-1]]
        )
        rank = (
            idx
            - torch.cummax(
                torch.where(is_start, idx, torch.zeros_like(idx)), dim=0
            ).values
        )
        pos = (self._seen[ic] + rank) % self.history
        success = outcomes.success[order].bool()
        self._success.index_put_((ic, pos), success)
        self._ret.index_put_((ic, pos), outcomes.ret[order].to(torch.float32))
        self._seen.scatter_add_(0, ic, torch.ones_like(ic))
        self._succeeded.scatter_add_(0, ic, success.to(torch.int64))

    def check_ids(self, ic_ids: Tensor) -> None:
        """Reject ids outside [0, capacity); the spare row at `capacity` is not addressable."""
        if ic_ids.numel() and bool(((ic_ids < 0) | (ic_ids >= self.capacity)).any()):
            raise ValueError(f"ic_ids must be in [0, {self.capacity})")

    def forget(self, ic_ids: Tensor) -> None:
        self.check_ids(ic_ids)
        self._success[ic_ids] = False
        self._ret[ic_ids] = 0.0
        self._seen[ic_ids] = 0
        self._succeeded[ic_ids] = 0

    def count(self, ic_ids: Tensor) -> Tensor:
        """Outcomes currently in each IC's window (at most `history`)."""
        self.check_ids(ic_ids)
        return self._seen[ic_ids].clamp(max=self.history)

    def success_rate(self, ic_ids: Tensor) -> Tensor:
        """Mean success over the window; 0 where nothing is recorded."""
        total = self._success[ic_ids].sum(dim=1).to(torch.float32)
        return total / self.count(ic_ids).clamp(min=1).to(torch.float32)

    def ever_succeeded(self, ic_ids: Tensor) -> Tensor:
        return self._succeeded[ic_ids] > 0

    def recent(self, ic_ids: Tensor, n: int) -> tuple[Tensor, Tensor, Tensor]:
        """Last `n` outcomes per IC in time order (oldest first), as `(ret, success, valid)`
        of shape `(k, n)`. `valid` is False where the IC has fewer than `n` outcomes."""
        self.check_ids(ic_ids)
        if not 0 < n <= self.history:
            raise ValueError(f"n must be in [1, history={self.history}], got {n}")
        seen = self._seen[ic_ids].unsqueeze(1)
        logical = seen - n + torch.arange(n, device=seen.device).unsqueeze(0)
        valid = logical >= 0
        slot = logical.clamp(min=0) % self.history
        rows = ic_ids.unsqueeze(1)
        return self._ret[rows, slot], self._success[rows, slot], valid


def median_fill(lp: Tensor, trusted: Tensor) -> Tensor:
    """Replace untrusted entries by the median of the trusted ones (0 if none are trusted)."""
    n_trusted = trusted.sum()
    # Sorting with untrusted pushed to +inf puts the trusted values first, in order.
    ordered = torch.where(trusted, lp, torch.full_like(lp, torch.inf)).sort().values
    mid = ((n_trusted - 1) // 2).clamp(min=0)
    median = torch.where(
        n_trusted > 0, ordered[mid], torch.zeros_like(lp[:1]).squeeze(0)
    )
    return torch.where(trusted, lp, median)


class BaseScore:
    """Defaults: no rollout data used, and PLR rank sampling from `values`.

    `beta` is PLR's temperature (default 0.1, the Procgen setting): see `probs`.
    """

    beta: float = 0.1

    def _set_beta(self, beta: float) -> None:
        if beta <= 0.0:
            raise ValueError(f"beta must be positive, got {beta}")
        self.beta = beta

    def record(self, outcomes: Outcomes) -> None:
        raise NotImplementedError

    def observe_rollout(self, stats: RolloutStats) -> None:
        del stats

    def values(self, ic_ids: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def forget(self, ic_ids: Tensor) -> None:
        raise NotImplementedError

    def probs(self, ic_ids: Tensor) -> Tensor:
        """PLR rank prioritisation: p_i = h_i^(1/beta) / sum_j h_j^(1/beta), h = 1/rank.

        Rank 1 is the largest lp; ties rank in input order (stable sort). Source:
        papers/text/2010.03934_prioritized-level-replay.txt L204-214 (definition) and
        L299, L848 (beta = 0.1 for Procgen).
        """
        lp, _ = self.values(ic_ids)
        order = torch.argsort(lp, descending=True, stable=True)
        rank = torch.empty_like(order)
        rank[order] = torch.arange(1, order.numel() + 1, device=order.device)
        weight = rank.to(torch.float32) ** (-1.0 / self.beta)
        return weight / weight.sum()


class TableScore(BaseScore):
    """A score computed from an `ICTable`; subclasses implement `_raw_values`."""

    def __init__(
        self, *, capacity: int, device: str, history: int, min_count: int, beta: float
    ) -> None:
        if not 1 <= min_count <= history:
            raise ValueError(
                f"min_count must be in [1, history={history}], got {min_count}"
            )
        self._set_beta(beta)
        self.table = ICTable(capacity, device, history=history)
        self._min_count = min_count

    def record(self, outcomes: Outcomes) -> None:
        self.table.record(outcomes)

    def forget(self, ic_ids: Tensor) -> None:
        self.table.forget(ic_ids)

    def values(self, ic_ids: Tensor) -> tuple[Tensor, Tensor]:
        trusted = self.table.count(ic_ids) >= self._min_count
        return median_fill(self._raw_values(ic_ids), trusted), trusted

    def _raw_values(self, ic_ids: Tensor) -> Tensor:
        raise NotImplementedError
