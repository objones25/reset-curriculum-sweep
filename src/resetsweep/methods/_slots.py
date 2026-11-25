"""Shared base for visited-buffer methods whose IC ids are buffer slots (`visited_count`,
`buffer_scored`).

Not a method itself (the leading underscore keeps `METHODS.load` from importing it as one).
"""

import dataclasses
import math

import torch
from torch import Tensor

from resetsweep.methods.base import MethodContext
from resetsweep.methods.visited import CAPACITY_DEFAULT, Visited
from resetsweep.scores.base import Score
from resetsweep.types import Outcomes, RolloutStats, StartStates


class SlotBuffer(Visited):
    """`Visited` ring buffer where the slot index is a stable IC id in `[0, capacity)`.

    Contract (Tasks 17-19 rely on it):

    - `reset` returns the slot id for every buffer draw and `-1` for task-start rows.
    - The method drives its own score: `reset` records the finished episodes' outcomes into
      `ctx.score` and `update` passes the rollout to `score.observe_rollout`. train.py must
      not call `score.record` or `score.observe_rollout` itself, or outcomes count twice.
      The score must be built with the method's `capacity`.
    - Overwriting a slot never leaks outcomes into the new state. `offer_visited` only marks
      overwritten slots (device ops, no host sync); the next `reset` or `update` calls
      `score.forget` on them before anything reads or records scores. Each slot also carries
      a version, bumped on every overwrite; an env remembers the version of the slot it
      started from, and an outcome whose slot version has moved on gets `ic_id = -1` before
      recording, so the score ignores it.
    - `ic_states()` returns all `capacity` rows (see `Method.ic_states`).

    Subclasses implement `_probs(size)`, the draw distribution over filled slots, and may
    hook `_on_write`, `_on_outcomes`, `_on_reset`, `_on_draw` and `_forget_slots`.
    """

    _lanes: int = 1

    def __init__(
        self,
        ctx: MethodContext,
        *,
        capacity: int = CAPACITY_DEFAULT,
        min_fill: int | None = None,
    ) -> None:
        super().__init__(ctx, capacity=capacity, min_fill=min_fill)
        if ctx.score is None:
            raise ValueError(f"{self.name} needs a score; ctx.score is None")
        self._score: Score = ctx.score
        d = ctx.device
        rows = capacity + 1  # the spare row absorbs rejected offers, as in `Visited`
        self._version = torch.zeros(rows, dtype=torch.long, device=d)
        self._dirty = torch.zeros(rows, dtype=torch.bool, device=d)
        self._ever = torch.zeros(rows, dtype=torch.bool, device=d)
        self._env_version = torch.full((ctx.num_envs,), -1, dtype=torch.long, device=d)

    def offer_visited(
        self,
        env_ids: Tensor,
        qpos: Tensor,
        qvel: Tensor,
        extra: Tensor,
        ep_step: Tensor,
    ) -> None:
        del ep_step
        slot = self._write(qpos, qvel, extra)
        self._version.index_add_(0, slot, torch.ones_like(slot))
        self._dirty[slot] = True
        self._on_write(env_ids, slot, qpos, qvel, extra)

    def reset(
        self, env_ids: Tensor, outcomes: Outcomes | None
    ) -> tuple[StartStates, Tensor]:
        self._flush_forget()
        if outcomes is not None:
            fresh = self._drop_stale(outcomes)
            self._score.record(fresh)
            self._track(fresh)
            self._on_outcomes(fresh)
        states, ids = super().reset(env_ids, None)
        started = self._version[ids.clamp(min=0)]
        self._env_version[env_ids] = torch.where(
            ids >= 0, started, torch.full_like(started, -1)
        )
        self._on_reset(env_ids, ids)
        return states, ids

    def update(self, it: int, stats: RolloutStats) -> dict[str, float]:
        del it
        self._flush_forget()
        self._score.observe_rollout(stats)
        return {}

    def ic_states(self) -> StartStates:
        cap = self._capacity
        return StartStates(
            self._qpos[:cap].clone(),
            self._qvel[:cap].clone(),
            self._extra[:cap].clone(),
        )

    def slot_probs(self) -> Tensor:
        """Draw distribution over the filled slots `[0, buffer_size)` (before `p_task` mixing)."""
        return self._probs(int(self._size))

    def diagnostics(self) -> dict[str, float]:
        """`sampling_entropy`: nats over the task-start atom (p_task) plus each filled slot
        ((1 - p_task) * slot_probs); 0.0 below `min_fill`. `never_succeeded_share`: share of
        filled slots with no recorded success since they were last written; NaN when empty."""
        size = int(self._size)
        entropy = 0.0
        if size >= self._min_fill:
            p_task = self.ctx.p_task
            p = self._probs(size) * (1.0 - p_task)
            entropy = -float(torch.special.xlogy(p, p).sum())
            entropy -= p_task * math.log(p_task) if p_task > 0 else 0.0
        never = float((~self._ever[:size]).float().mean()) if size > 0 else float("nan")
        return {
            "sampling_entropy": entropy,
            "buffer_size": float(size),
            "never_succeeded_share": never,
        }

    def _draw(self, k: int, size: int) -> tuple[StartStates, Tensor]:
        """`k` slot ids drawn from `_probs`; consecutive groups of `_lanes` rows share one."""
        groups = -(-k // self._lanes)
        pick = torch.multinomial(
            self._probs(size), groups, replacement=True, generator=self.ctx.gen
        )
        idx = pick.repeat_interleave(self._lanes)[:k]
        self._on_draw(idx)
        return StartStates(self._qpos[idx], self._qvel[idx], self._extra[idx]), idx

    def _flush_forget(self) -> None:
        ids = self._dirty[: self._capacity].nonzero(as_tuple=True)[0]
        self._dirty.zero_()
        if ids.numel() > 0:
            self._score.forget(ids)
            self._ever[ids] = False
            self._forget_slots(ids)

    def _drop_stale(self, outcomes: Outcomes) -> Outcomes:
        ic = outcomes.ic_ids
        same = self._version[ic.clamp(min=0)] == self._env_version[outcomes.env_ids]
        current = (ic >= 0) & same
        return dataclasses.replace(
            outcomes, ic_ids=torch.where(current, ic, torch.full_like(ic, -1))
        )

    def _track(self, outcomes: Outcomes) -> None:
        successes = torch.zeros_like(self._version)
        successes.index_add_(0, self._recorded_slots(outcomes), outcomes.success.long())
        self._ever |= successes > 0
        self._ever[self._capacity] = False

    def _recorded_slots(self, outcomes: Outcomes) -> Tensor:
        """Each outcome's slot, or the spare row for rows the score ignores."""
        keep = outcomes.full & (outcomes.ic_ids >= 0)
        return torch.where(
            keep, outcomes.ic_ids, torch.full_like(outcomes.ic_ids, self._capacity)
        )

    def _probs(self, size: int) -> Tensor:
        raise NotImplementedError

    def _on_write(
        self, env_ids: Tensor, slot: Tensor, qpos: Tensor, qvel: Tensor, extra: Tensor
    ) -> None:
        del env_ids, slot, qpos, qvel, extra

    def _on_outcomes(self, outcomes: Outcomes) -> None:
        del outcomes

    def _on_reset(self, env_ids: Tensor, ids: Tensor) -> None:
        del env_ids, ids

    def _on_draw(self, idx: Tensor) -> None:
        del idx

    def _forget_slots(self, ids: Tensor) -> None:
        del ids
