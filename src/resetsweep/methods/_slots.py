"""Shared base for visited-buffer methods with a stable IC table (`visited_count`,
`buffer_scored`), and the IC versioning those and `own_success` share (`ICVersions`).

Not a method itself (the leading underscore keeps `METHODS.load` from importing it as one).
"""

import dataclasses
import math

import torch
from torch import Tensor

from resetsweep.methods.base import (
    TASK_START_IC_ID,
    MethodContext,
    mix_task_start,
    to_device,
)
from resetsweep.methods.visited import CAPACITY_DEFAULT, Visited
from resetsweep.scores.base import Score
from resetsweep.types import Outcomes, RolloutStats, StartStates

IC_CAPACITY_DEFAULT = 1024
PROMOTE_FRAC_DEFAULT = 0.1


class ICVersions:
    """Version of each IC slot and of each env's episode-start IC, on device.

    A slot's version bumps whenever its IC is replaced (`bump`). `start` stamps each
    resetting env with the version of the IC it starts from (-1 for a task start). Outcomes
    and rollout cells whose stamp no longer matches the slot's version came from a replaced
    IC; `drop_stale` and `current_cells` turn their ic id into -1 so scores ignore them.
    """

    def __init__(
        self, capacity: int, num_envs: int, device: str, *, owner: str
    ) -> None:
        self.capacity = capacity
        self._owner = owner
        self.slot = torch.zeros(capacity, dtype=torch.long, device=device)
        self._env = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self._ic_outcomes = torch.zeros((), dtype=torch.long, device=device)
        self._stale_outcomes = torch.zeros((), dtype=torch.long, device=device)

    def start(self, env_ids: Tensor, ids: Tensor) -> None:
        started = self.slot[ids.clamp(min=0)]
        self._env[env_ids] = torch.where(
            ids >= 0, started, torch.full_like(started, -1)
        )

    def bump(self, targets: Tensor) -> None:
        self.slot[targets] += 1

    def episode_versions(self) -> Tensor:
        """`(num_envs,)` version of each env's current episode-start IC (-1: task start)."""
        return self._env.clone()

    def started_from_table(self, env_ids: Tensor) -> Tensor:
        """`(k,)` bool: the env's current episode started from an IC slot."""
        return self._env[env_ids] >= 0

    def drop_stale(self, outcomes: Outcomes) -> Outcomes:
        ic = outcomes.ic_ids
        same = self.slot[ic.clamp(min=0)] == self._env[outcomes.env_ids]
        current = (ic >= 0) & same
        self._ic_outcomes += (ic >= 0).sum()
        self._stale_outcomes += ((ic >= 0) & ~same).sum()
        return dataclasses.replace(
            outcomes, ic_ids=torch.where(current, ic, torch.full_like(ic, -1))
        )

    def current_cells(self, stats: RolloutStats) -> RolloutStats:
        if stats.ic_versions is None:
            raise ValueError(
                f"{self._owner} needs RolloutStats.ic_versions (stack episode_versions() per "
                "step next to ic_ids); without it replaced ICs would get stale rollout cells"
            )
        if stats.ic_versions.shape != stats.ic_ids.shape:
            raise ValueError(
                f"ic_versions shape {tuple(stats.ic_versions.shape)} must match ic_ids "
                f"{tuple(stats.ic_ids.shape)}"
            )
        ic = stats.ic_ids
        lookup = ic.clamp(0, self.capacity - 1)
        current = (ic >= 0) & (stats.ic_versions == self.slot[lookup])
        # Out-of-range ids pass through unchanged, so the score still rejects them loudly.
        keep = current | (ic >= self.capacity)
        return dataclasses.replace(
            stats, ic_ids=torch.where(keep, ic, torch.full_like(ic, -1))
        )

    def stale_share(self) -> float:
        """Cumulative share of IC-started outcomes dropped as stale (NaN before any)."""
        total = int(self._ic_outcomes)
        return int(self._stale_outcomes) / total if total > 0 else float("nan")


class SlotBuffer(Visited):
    """Visit ring (`Visited`, the candidate pool) plus a stable IC table of `ic_capacity` slots.

    Why two tables: at sweep scale a ring slot lives `capacity * visit_every / num_envs` steps
    (65536 * 8 / 4096 = 128), shorter than many episodes, so ring slots as ICs would lose most
    outcomes to staleness and never reach a minimum count. The IC table changes only at
    promotion, once every `update_every` updates.

    Contract (Tasks 17-19 rely on it):

    - IC ids are IC-table slots in `[0, ic_capacity)`; task-start rows are `-1`. The score must
      have `capacity == ic_capacity` (checked at construction). `ic_states()` is the table.
    - Until the first promotion fills a slot, every reset goes to the task start.
    - The method drives its score: `reset` records outcomes and `update` calls
      `score.observe_rollout`. train.py must not call either itself.
    - `update(it, stats)` needs `stats.ic_versions` (raises ValueError if None): train.py
      stacks `episode_versions()` once per step next to the ic ids. Cells whose IC was replaced
      after their episode started become `-1` before the score sees them; outcomes get the
      same treatment in `reset`. Then, every `update_every` iterations, promotion runs.
    - Promotion: empty slots are filled first, all at once, from distinct ring states drawn
      uniformly (ours: an empty slot holds nothing to lose). Once full, up to
      `ceil(promote_frac * ic_capacity)` slots with the lowest current draw probability are
      replaced by uniform ring draws. PLR replaces `l_min = argmin P_replay(l)` when its
      level buffer is full (papers/text/2010.03934_prioritized-level-replay.txt L1015-1027);
      unlike PLR, which replaces only if the new level scores higher, a candidate here has no
      score yet, so it replaces unconditionally (ours). Replaced slots get `score.forget`,
      a version bump and fresh per-slot state.

    Subclasses implement `_probs(size)` and may hook `_on_write`, `_on_outcomes`,
    `_on_reset`, `_on_draw`, `_on_promote` and `_after_promotion`.
    """

    _lanes: int = 1

    def __init__(
        self,
        ctx: MethodContext,
        *,
        capacity: int = CAPACITY_DEFAULT,
        min_fill: int | None = None,
        ic_capacity: int = IC_CAPACITY_DEFAULT,
        promote_frac: float = PROMOTE_FRAC_DEFAULT,
        update_every: int = 1,
    ) -> None:
        if ctx.score is None:
            raise ValueError(f"{self.name} needs a score; ctx.score is None")
        super().__init__(ctx, capacity=capacity, min_fill=min_fill)
        if not 1 <= ic_capacity <= capacity:
            raise ValueError(
                f"ic_capacity must be in [1, capacity={capacity}], got {ic_capacity}"
            )
        if not 0.0 < promote_frac <= 1.0:
            raise ValueError(f"promote_frac must be in (0, 1], got {promote_frac}")
        if update_every < 1:
            raise ValueError(f"update_every must be at least 1, got {update_every}")
        if ctx.score.capacity != ic_capacity:
            raise ValueError(
                f"score capacity {ctx.score.capacity} must equal ic_capacity {ic_capacity}"
            )
        self._score: Score = ctx.score
        self._ic_capacity = ic_capacity
        self._replace_max = math.ceil(promote_frac * ic_capacity)
        self._update_every = update_every
        self._ic_size = 0  # filled slots are the prefix [0, _ic_size)
        d = ctx.device
        self._ic_qpos = torch.zeros(ic_capacity, self._qpos.shape[1], device=d)
        self._ic_qvel = torch.zeros(ic_capacity, self._qvel.shape[1], device=d)
        self._ic_extra = torch.zeros(ic_capacity, self._extra.shape[1], device=d)
        self._versions = ICVersions(ic_capacity, ctx.num_envs, d, owner=self.name)
        self._ever = torch.zeros(ic_capacity + 1, dtype=torch.bool, device=d)

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
        self._on_write(env_ids, slot, qpos, qvel, extra)

    def reset(
        self, env_ids: Tensor, outcomes: Outcomes | None
    ) -> tuple[StartStates, Tensor]:
        if outcomes is not None:
            fresh = self._versions.drop_stale(outcomes)
            self._score.record(fresh)
            self._track(fresh)
            self._on_outcomes(fresh)
        if self._ic_size == 0:
            n = env_ids.numel()
            states = to_device(
                self.ctx.task.sample_start(n, self.ctx.gen), self.ctx.device
            )
            ids = torch.full(
                (n,), TASK_START_IC_ID, dtype=torch.long, device=self.ctx.device
            )
        else:
            states, ids = mix_task_start(self.ctx, env_ids, self._draw_ics)
        self._versions.start(env_ids, ids)
        self._on_reset(env_ids, ids)
        return states, ids

    def update(self, it: int, stats: RolloutStats) -> dict[str, float]:
        self._score.observe_rollout(self._versions.current_cells(stats))
        if it % self._update_every == 0:
            self._promote()
        return {}

    def episode_versions(self) -> Tensor:
        """`(num_envs,)` version of each env's current episode-start IC (-1: task start)."""
        return self._versions.episode_versions()

    def ic_versions(self) -> Tensor:
        """`(ic_capacity,)` current version of each IC slot; it bumps on every replacement."""
        return self._versions.slot.clone()

    def ic_states(self) -> StartStates:
        return StartStates(
            self._ic_qpos.clone(), self._ic_qvel.clone(), self._ic_extra.clone()
        )

    def slot_probs(self) -> Tensor:
        """Draw distribution over the filled IC slots `[0, buffer_size)`, before `p_task`."""
        return self._probs(self._ic_size)

    def diagnostics(self) -> dict[str, float]:
        """`buffer_size`: filled IC slots; `ring_size`: visited states in the ring.
        `sampling_entropy`: nats over the task-start atom (p_task) plus each filled IC
        ((1 - p_task) * slot_probs); 0.0 before the first promotion. `never_succeeded_share`:
        filled ICs with no success since promoted (NaN when empty). `stale_outcome_share`:
        cumulative share of IC-started outcomes dropped because the IC was replaced."""
        size = self._ic_size
        entropy = 0.0
        if size > 0:
            p_task = self.ctx.p_task
            p = self._probs(size) * (1.0 - p_task)
            entropy = -float(torch.special.xlogy(p, p).sum())
            entropy -= p_task * math.log(p_task) if p_task > 0 else 0.0
        never = float((~self._ever[:size]).float().mean()) if size > 0 else float("nan")
        stale = self._versions.stale_share()
        return {
            "sampling_entropy": entropy,
            "buffer_size": float(size),
            "ring_size": float(int(self._size)),
            "never_succeeded_share": never,
            "stale_outcome_share": stale,
        }

    def _draw_ics(self, k: int) -> tuple[StartStates, Tensor]:
        """`k` IC slots drawn from `_probs`; consecutive groups of `_lanes` rows share one."""
        groups = -(-k // self._lanes)
        pick = torch.multinomial(
            self._probs(self._ic_size), groups, replacement=True, generator=self.ctx.gen
        )
        idx = pick.repeat_interleave(self._lanes)[:k]
        self._on_draw(idx)
        return self._ic_rows(idx), idx

    def _ic_rows(self, idx: Tensor) -> StartStates:
        return StartStates(self._ic_qpos[idx], self._ic_qvel[idx], self._ic_extra[idx])

    def _promote(self) -> None:
        ring = int(self._size)
        if ring < self._min_fill:
            return
        if self._ic_size < self._ic_capacity:
            m = min(self._ic_capacity - self._ic_size, ring)
            targets = torch.arange(
                self._ic_size, self._ic_size + m, device=self.ctx.device
            )
            self._ic_size += m
        else:
            m = min(self._replace_max, ring)
            order = torch.argsort(self._probs(self._ic_size), stable=True)
            targets = order[:m]
        gen, device = self.ctx.gen, self.ctx.device
        candidates = torch.randperm(ring, generator=gen, device=device)[:m]
        self._ic_qpos[targets] = self._qpos[candidates]
        self._ic_qvel[targets] = self._qvel[candidates]
        self._ic_extra[targets] = self._extra[candidates]
        self._versions.bump(targets)
        self._ever[targets] = False
        self._score.forget(targets)
        self._on_promote(targets, candidates)
        self._after_promotion()

    def _track(self, outcomes: Outcomes) -> None:
        successes = torch.zeros_like(self._ever, dtype=torch.long)
        successes.index_add_(0, self._recorded_slots(outcomes), outcomes.success.long())
        self._ever |= successes > 0
        self._ever[self._ic_capacity] = False

    def _recorded_slots(self, outcomes: Outcomes) -> Tensor:
        """Each outcome's IC slot, or the spare row `ic_capacity` for rows scores ignore."""
        keep = outcomes.full & (outcomes.ic_ids >= 0)
        return torch.where(
            keep, outcomes.ic_ids, torch.full_like(outcomes.ic_ids, self._ic_capacity)
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

    def _on_promote(self, targets: Tensor, candidates: Tensor) -> None:
        """`targets` IC slots now hold the ring states at `candidates`."""
        del targets, candidates

    def _after_promotion(self) -> None:
        pass
