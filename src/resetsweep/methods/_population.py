"""Shared base for the population methods `walk`, `ball` and `genetic` (spec section 5.3,
M6 and M7).

Not a method itself (the leading underscore keeps `METHODS.load` from importing it as one).

Population. `K = num_envs // R` ICs; env `i` runs IC `min(i // R, K - 1)`, so `R` envs share
each IC and measure it together (envs past `K * R`, when `R` does not divide `num_envs`,
join the last IC). The table starts as `K` task-start draws. A `p_task` share of resets goes
to the task start (`mix_task_start`); every other reset of env `i` restarts from its IC.

Contract (train.py relies on it):

- IC ids are population slots in `[0, K)`; task-start rows are `-1`. The score must have
  `capacity == K` (checked at construction; `population_size` computes K). `ic_states()` is
  the `K`-row table.
- The method drives its score: `reset` records outcomes (stale ones dropped) and `update`
  calls `score.observe_rollout` (stale cells dropped). train.py must not call either.
- `update(it, stats)` needs `stats.ic_versions` (ValueError if None), stacked from
  `episode_versions()` once per step. Every `update_every` iterations the subclass's plan
  runs: it names the slots to replace and proposes one child per slot. Plans rank, cull
  and pick parents among trusted ICs only (`_ranked`); untrusted ICs stay until measured.
- Children go through `_children`: a candidate is kept only when it is finite, inside
  `task.start_box` and accepted by `task.is_valid`, and (when `screen_seconds > 0`) passes
  a zero-action hold screen with `speed_max` = the largest |qvel| bound of `start_box`.
  Rejected children are redrawn up to `tries` times, then fall back to `task.sample_start`.
- Replacing a slot bumps its version (`ICVersions`), clears its ever-succeeded bit and calls
  `score.forget`; outcomes and cells from episodes that started on the old IC become -1.

Host syncs: `reset` adds none beyond `mix_task_start`'s. `update` syncs freely, but only on
update iterations, which is also the only place rollouts and screens run.
"""

import math
from collections.abc import Callable
from typing import ClassVar

import torch
from torch import Tensor

from resetsweep.methods._slots import ICVersions
from resetsweep.methods.base import MethodBase, MethodContext, mix_task_start, to_device
from resetsweep.scores.base import Score
from resetsweep.tasks.base import canonicalize
from resetsweep.types import InvalidStartState, Outcomes, RolloutStats, StartStates

R_DEFAULT = 4
UPDATE_EVERY_DEFAULT = 20
TRIES_DEFAULT = 8
SPREAD_EPS = 1e-8

# Spec section 5.4: these methods take S1, S3, S4, S5, S7 (classifier, RND) and the controls.
POPULATION_SCORES = frozenset(
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

# Maps child row indices (into the plan's target list) to one candidate start per row.
ProposeFn = Callable[[Tensor], StartStates]


def population_size(num_envs: int, *, R: int = R_DEFAULT) -> int:
    """`K = num_envs // R`, the number of ICs; the score's capacity must equal it."""
    if R < 1:
        raise ValueError(f"R must be at least 1, got {R}")
    k = num_envs // R
    if k < 1:
        raise ValueError(f"num_envs {num_envs} must be at least R = {R}")
    return k


class PopulationMixin(MethodBase):
    """K ICs shared by R envs each, IC versioning and the child filter; see the module
    docstring. Subclasses implement `_plan`."""

    accepts_scores: ClassVar[frozenset[str]] = POPULATION_SCORES

    def __init__(
        self,
        ctx: MethodContext,
        *,
        R: int,
        update_every: int,
        tries: int,
        screen_seconds: float,
    ) -> None:
        if ctx.score is None:
            raise ValueError(f"{self.name} needs a score; ctx.score is None")
        k = population_size(ctx.num_envs, R=R)
        if ctx.score.capacity != k:
            raise ValueError(
                f"score capacity {ctx.score.capacity} must equal K = {k} "
                f"(num_envs // R = {ctx.num_envs} // {R})"
            )
        if update_every < 1 or tries < 1:
            raise ValueError(
                f"update_every and tries must be at least 1, got {update_every}, {tries}"
            )
        if screen_seconds < 0:
            raise ValueError(f"screen_seconds must be >= 0, got {screen_seconds}")
        super().__init__(ctx)
        self._score: Score = ctx.score
        self._r = R
        self._k = k
        self._update_every = update_every
        self._tries = tries
        self._screen_seconds = screen_seconds
        d = ctx.device
        start = to_device(ctx.task.sample_start(k, ctx.gen), d)
        self._qpos = start.qpos.clone()
        self._qvel = start.qvel.clone()
        self._extra = start.extra.clone()
        self._nq = self._qpos.shape[1]
        self._versions = ICVersions(k, ctx.num_envs, d, owner=self.name)
        self._ever = torch.zeros(k + 1, dtype=torch.bool, device=d)
        box = ctx.task.start_box
        self._low = box.low.to(d, torch.float32)
        self._high = box.high.to(d, torch.float32)
        span = self._high - self._low
        if bool((span < 0).any()):
            raise ValueError(f"{ctx.task.name} start_box has high < low")
        self._span = torch.where(span > 0, span, torch.ones_like(span))
        self._flat = span == 0  # dims with one admissible value
        speed = torch.maximum(self._low[self._nq :].abs(), self._high[self._nq :].abs())
        self._speed_max = float(speed.max()) if speed.numel() else 0.0
        if screen_seconds > 0 and self._speed_max <= 0:
            raise ValueError(
                f"screening needs a positive |qvel| bound in start_box, got {self._speed_max}"
            )
        self._replaced_total = 0
        self._fallback_total = 0

    # Method protocol -----------------------------------------------------------------

    def reset(
        self, env_ids: Tensor, outcomes: Outcomes | None
    ) -> tuple[StartStates, Tensor]:
        if outcomes is not None:
            fresh = self._versions.drop_stale(outcomes)
            self._score.record(fresh)
            self._track(fresh)
        states, ids = mix_task_start(self.ctx, env_ids, self._placeholder)
        from_ic = ids >= 0
        ic = (env_ids // self._r).clamp(max=self._k - 1)
        ids = torch.where(from_ic, ic, ids)
        rows = from_ic.unsqueeze(1)
        states = StartStates(
            torch.where(rows, self._qpos[ic], states.qpos),
            torch.where(rows, self._qvel[ic], states.qvel),
            torch.where(rows, self._extra[ic], states.extra),
        )
        self._versions.start(env_ids, ids)
        return states, ids

    def update(self, it: int, stats: RolloutStats) -> dict[str, float]:
        self._score.observe_rollout(self._versions.current_cells(stats))
        if it % self._update_every == 0:
            plan = self._plan()
            if plan is not None:
                targets, propose = plan
                self._write(targets, self._children(targets.numel(), propose))
        return {}

    def episode_versions(self) -> Tensor:
        """`(num_envs,)` version of each env's current episode-start IC (-1: task start)."""
        return self._versions.episode_versions()

    def ic_states(self) -> StartStates:
        return StartStates(self._qpos.clone(), self._qvel.clone(), self._extra.clone())

    def diagnostics(self) -> dict[str, float]:
        """`buffer_size`: K. `sampling_entropy`: nats over the task-start atom (p_task) and
        K equal IC atoms ((1 - p_task) / K each). `never_succeeded_share`: ICs with no
        success since they entered the population. `stale_outcome_share`: cumulative share
        of IC-started outcomes dropped because the IC was replaced. `replaced_total`:
        slots replaced so far (children and `replace` calls); `fallback_total`: children
        that fell back to a task start after `tries` rejected proposals."""
        p_task = self.ctx.p_task
        entropy = -p_task * math.log(p_task) if p_task > 0 else 0.0
        if p_task < 1:
            entropy -= (1 - p_task) * math.log((1 - p_task) / self._k)
        return {
            "sampling_entropy": entropy,
            "buffer_size": float(self._k),
            "never_succeeded_share": float((~self._ever[: self._k]).float().mean()),
            "stale_outcome_share": self._versions.stale_share(),
            "replaced_total": float(self._replaced_total),
            "fallback_total": float(self._fallback_total),
        }

    def replace(self, slots: Tensor, states: StartStates) -> None:
        """Put `states` into population `slots` (bump versions, `score.forget`). Raises
        `InvalidStartState` for a row outside `start_box` or rejected by `task.is_valid`."""
        if slots.shape != (states.n,):
            raise ValueError(
                f"slots {tuple(slots.shape)} must be one per state ({states.n})"
            )
        states = to_device(states, self.ctx.device)
        ok = self._valid(states)
        if not bool(ok.all()):
            row = int((~ok).nonzero()[0, 0])
            raise InvalidStartState(
                self.name, row, "outside start_box or rejected by task.is_valid"
            )
        self._write(slots.to(self.ctx.device), states)

    # Subclass hook ---------------------------------------------------------------------

    def _plan(self) -> tuple[Tensor, ProposeFn] | None:
        """Slots to replace (ascending) and a proposer of one child per slot, or None."""
        raise NotImplementedError

    # Helpers for subclasses --------------------------------------------------------------

    def _ranked(self) -> tuple[Tensor, Tensor, Tensor]:
        """`(order, lp, trusted)`: `order` lists only the trusted slots, best first, with
        ties broken at random so equal scores never favour low slot ids. Untrusted slots
        hold a median placeholder in `lp`, not a measurement, so selection never ranks them:
        they are never culled and never parents until their score trusts them."""
        lp, trusted = self._score.values(torch.arange(self._k, device=self.ctx.device))
        measured = trusted.nonzero(as_tuple=True)[0]
        perm = measured[
            torch.randperm(
                measured.numel(), generator=self.ctx.gen, device=self.ctx.device
            )
        ]
        order = perm[torch.argsort(lp[perm], descending=True, stable=True)]
        return order, lp, trusted

    @staticmethod
    def _spread_ok(lp: Tensor, trusted: Tensor, threshold: float) -> bool:
        """Cull only when at least 2 ICs are trusted and their relative spread
        `(max - min) / max(|median|, 1e-8)` reaches `threshold` (2503.19037 L298-312
        analogue: no selection while all scores are near equal)."""
        values = lp[trusted]
        if values.numel() < 2:
            return False
        scale = max(abs(float(values.median())), SPREAD_EPS)
        return float(values.max() - values.min()) / scale >= threshold

    def _keep_plan(
        self, order: Tensor, *, keep_top: float, parent_power: float
    ) -> tuple[Tensor, Tensor, Tensor] | None:
        """Of the `n` trusted slots in `order` (from `_ranked`), keep the top
        `ceil(keep_top * n)` and replace the other trusted ones; untrusted slots are left
        alone. Returns `(targets, parents, quality)`: ascending targets, a kept parent slot
        for each drawn with weight `rank^-parent_power` (rank 1 = best kept), and the
        parent's quality in the kept pool, 1 for the best and 0 for the worst. None when
        fewer than 2 slots are trusted or nothing is replaced."""
        n = order.numel()
        if n < 2:
            return None
        n_keep = min(max(math.ceil(keep_top * n), 1), n)
        if n_keep == n:
            return None
        kept = order[:n_keep]
        targets = order[n_keep:].sort().values
        rank = torch.arange(1, n_keep + 1, device=self.ctx.device, dtype=torch.float32)
        pick = torch.multinomial(
            rank ** (-parent_power),
            targets.numel(),
            replacement=True,
            generator=self.ctx.gen,
        )
        return targets, kept[pick], self._pool_quality(pick, n_keep)

    @staticmethod
    def _pool_quality(pool_index: Tensor, pool_size: int) -> Tensor:
        """1 for pool index 0 (the best parent) down to 0 for the last one."""
        if pool_size == 1:
            return torch.ones_like(pool_index, dtype=torch.float32)
        return 1.0 - pool_index.to(torch.float32) / (pool_size - 1)

    def _unit(self, slots: Tensor) -> Tensor:
        """`(n, nq + nv)` IC states in normalised `start_box` coordinates ([0, 1] per dim)."""
        x = torch.cat([self._qpos[slots], self._qvel[slots]], dim=1)
        return torch.where(self._flat, 0.0, (x - self._low) / self._span)

    def _from_unit(self, u: Tensor, extra: Tensor) -> StartStates:
        # A zero-width dim admits one value: noise must never move it off `low`.
        u = torch.where(self._flat, 0.0, u)
        x = self._low + u * self._span
        return StartStates(x[:, : self._nq], x[:, self._nq :], extra)

    def _rows(self, slots: Tensor) -> StartStates:
        return StartStates(self._qpos[slots], self._qvel[slots], self._extra[slots])

    # Internals -------------------------------------------------------------------------

    def _placeholder(self, k: int) -> tuple[StartStates, Tensor]:
        """Rows `mix_task_start` routes away from the task start; `reset` swaps in each env's
        own IC afterwards, because `draw_fn` is not told which envs it serves."""
        d = self.ctx.device
        states = StartStates(
            torch.zeros(k, self._qpos.shape[1], device=d),
            torch.zeros(k, self._qvel.shape[1], device=d),
            torch.zeros(k, self._extra.shape[1], device=d),
        )
        return states, torch.zeros(k, dtype=torch.long, device=d)

    def _admit(self, states: StartStates) -> tuple[StartStates, Tensor]:
        """Children canonicalized (`tasks.base.canonicalize`), then checked by `_valid`:
        `(states, ok)`. Only the children path canonicalizes; `_valid` checks states that
        are stored as given (`replace`)."""
        states = canonicalize(self.ctx.task, states)
        return states, self._valid(states)

    def _valid(self, states: StartStates) -> Tensor:
        x = torch.cat([states.qpos, states.qvel], dim=1)
        in_box = ((x >= self._low) & (x <= self._high)).all(dim=1)
        finite = torch.isfinite(x).all(dim=1)
        return finite & in_box & self.ctx.task.is_valid(states)

    def _children(self, m: int, propose: ProposeFn) -> StartStates:
        """One admitted child per target row: proposals are filtered (and screened), the
        rejected rows redrawn up to `tries` times, then filled from `task.sample_start`."""
        d = self.ctx.device
        qpos = torch.zeros(m, self._qpos.shape[1], device=d)
        qvel = torch.zeros(m, self._qvel.shape[1], device=d)
        extra = torch.zeros(m, self._extra.shape[1], device=d)
        pending = torch.arange(m, device=d)
        for _ in range(self._tries):
            cand, ok = self._admit(to_device(propose(pending), d))
            if self._screen_seconds > 0 and bool(ok.any()):
                idx = ok.nonzero(as_tuple=True)[0]
                ok[idx] = self.ctx.rollouts.hold_screen(
                    cand.index(idx), self._screen_seconds, speed_max=self._speed_max
                ).to(d)
            rows = pending[ok]
            qpos[rows] = cand.qpos[ok]
            qvel[rows] = cand.qvel[ok]
            extra[rows] = cand.extra[ok]
            pending = pending[~ok]
            if pending.numel() == 0:
                break
        n_left = pending.numel()
        if n_left > 0:
            fallback = to_device(self.ctx.task.sample_start(n_left, self.ctx.gen), d)
            qpos[pending] = fallback.qpos
            qvel[pending] = fallback.qvel
            extra[pending] = fallback.extra
            self._fallback_total += n_left
        return StartStates(qpos, qvel, extra)

    def _write(self, slots: Tensor, states: StartStates) -> None:
        self._qpos[slots] = states.qpos
        self._qvel[slots] = states.qvel
        self._extra[slots] = states.extra
        self._versions.bump(slots)
        self._ever[slots] = False
        self._score.forget(slots)
        self._replaced_total += slots.numel()

    def _track(self, outcomes: Outcomes) -> None:
        keep = outcomes.full & (outcomes.ic_ids >= 0)
        slot = torch.where(
            keep, outcomes.ic_ids, torch.full_like(outcomes.ic_ids, self._k)
        )
        successes = torch.zeros_like(self._ever, dtype=torch.long)
        successes.index_add_(0, slot, outcomes.success.long())
        self._ever |= successes > 0
        self._ever[self._k] = False
