"""M4 `own_success`: restart from states on the agent's own first successful trajectories
(shortlist section 3, method 4; finding C).

Capture. `offer_visited` arrives every `visit_every` steps, so each env keeps the states
offered during its current episode with their `ep_step`. When an episode ends with
`reached_reward`, its states up to and including `Outcomes.first_reward_step` (the episode
clock on the step the reward was first > 0) are stored as one trajectory, each tagged with
steps-to-reward `first_reward_step - ep_step`; states after the first reward are never
stored. Stored states go through `prepare_candidates` (unlimited hinges wrapped into
[-pi, pi), then `task.is_valid`): invalid rows are dropped, the rest keep their order and
tags, and a trajectory with no valid state is not stored. A trajectory has room for every offer of a whole episode: `M =
states_per_trajectory(task) = ceil(task.max_steps / visit_every)` (66 for the 528-step
maze_trap), so nothing before the reward is cut. Episodes that started from a stored state
are not captured (ours): their trajectory is a tail of one already stored, and storing
them would fill the slots with tails of the first success within a few resets.

IC table. Fixed: `max_trajectories x M` slots (`table_capacity(task)`); trajectory `t`
owns slots `[t * M, t * M + n_t)`, ordered in time (the slot nearest the reward last).
Once `max_trajectories` are stored nothing is overwritten (finding C: keep the earliest
successes), so slot versions never change (all 0); `episode_versions()` and the
`ic_versions` filter still follow the `SlotBuffer` contract so train.py treats every IC
method alike.

Draw. `uniform`: over every stored state of every stored trajectory. `frontier`: one
frontier index per trajectory, starting at the state nearest the reward; candidates are the
frontier states, one per trajectory. Over the candidates the draw is uniform, or, with a
score, buffer_scored's step 2 (`score.probs` over the trusted candidates, the median of
those for untrusted ones, uniform when none is trusted). A `p_task` share goes to the task
start, and until the first trajectory is stored every reset does (inert). With the
`steps_to_reward` score this is the score's documented PLR rank weighting (beta = 0.1):
almost every draw lands on the candidates nearest the reward, so `uniform` + S6 is close to
"restart next to the reward", not uniform along the trajectory.

Frontier moves (2607.26417 L90: advance and retreat thresholds per context). The window is
the last `window` outcomes of episodes that started at the trajectory's current frontier
index. When it is full and its success rate is >= `advance` the index moves one state back
(further from the reward); when it is <= `retreat` it moves one forward. The index stays
inside the trajectory, and the window empties whenever the index moves.
"""

import dataclasses
import math
from typing import Literal

import torch
from torch import Tensor

from resetsweep.methods._slots import ICVersions
from resetsweep.methods.base import (
    METHODS,
    TASK_START_IC_ID,
    MethodBase,
    MethodContext,
    mix_task_start,
    to_device,
)
from resetsweep.scores.base import ICTable, Score, median_fill
from resetsweep.scores.steps_to_reward import StepsToReward
from resetsweep.sim.env import VISIT_EVERY_DEFAULT
from resetsweep.tasks.base import Task, prepare_candidates
from resetsweep.types import Outcomes, RolloutStats, StartStates

Order = Literal["uniform", "frontier"]
_ORDERS: tuple[Order, ...] = ("uniform", "frontier")

MAX_TRAJECTORIES_DEFAULT = 16


def states_per_trajectory(task: Task) -> int:
    """Most offers one episode of `task` gets: `ceil(max_steps / visit_every)`. Offers come
    every `visit_every` vector steps, and an episode spans at most `max_steps` consecutive
    steps, which hold at most this many multiples of `visit_every`."""
    return -(-task.max_steps // VISIT_EVERY_DEFAULT)


def table_capacity(
    task: Task, *, max_trajectories: int = MAX_TRAJECTORIES_DEFAULT
) -> int:
    """IC-table size for `task`; the score's capacity must equal it."""
    return max_trajectories * states_per_trajectory(task)


@METHODS.register("own_success")
class OwnSuccess(MethodBase):
    """Restarts along stored successful trajectories; see the module docstring.

    Like `SlotBuffer` methods it drives its score: `reset` records outcomes (stale ones
    dropped) and `update` calls `score.observe_rollout` on current cells; with the
    `steps_to_reward` score it also calls `set_steps` for every stored state. The score is
    optional (`accepts_scores`); `phased` hands it a `success_band(pq)` score directly.

    Host syncs: none in `offer_visited` (it stops writing once the table is full). `reset`
    syncs to find newly successful envs while the table has room, and, in frontier order,
    to clear the windows of moved frontiers.
    """

    name = "own_success"
    accepts_scores = frozenset({"none", "steps_to_reward"})

    def __init__(
        self,
        ctx: MethodContext,
        *,
        order: Order = "uniform",
        advance: float = 0.8,
        retreat: float = 0.15,
        window: int = 10,
        max_trajectories: int = MAX_TRAJECTORIES_DEFAULT,
    ) -> None:
        if order not in _ORDERS:
            raise ValueError(f"order must be one of {_ORDERS}, got {order!r}")
        if not 0.0 <= retreat < advance <= 1.0:
            raise ValueError(
                f"retreat must be below advance, both in [0, 1]; got {retreat}, {advance}"
            )
        if window < 1 or max_trajectories < 1:
            raise ValueError(
                "window and max_trajectories must be at least 1, "
                f"got {window}, {max_trajectories}"
            )
        capacity = table_capacity(ctx.task, max_trajectories=max_trajectories)
        if ctx.score is not None and ctx.score.capacity != capacity:
            raise ValueError(
                f"score capacity {ctx.score.capacity} must equal the IC table size "
                f"{capacity} (table_capacity(task, max_trajectories=...))"
            )
        super().__init__(ctx)
        self._score: Score | None = ctx.score
        self._order: Order = order
        self._advance = advance
        self._retreat = retreat
        self._window = window
        self._k = max_trajectories
        self._m = states_per_trajectory(ctx.task)
        self._capacity = capacity
        d, n_env = ctx.device, ctx.num_envs
        canon = ctx.task.canonical_start()
        nq, nv, ne = canon.qpos.shape[1], canon.qvel.shape[1], ctx.task.extra_dim
        # IC table
        self._qpos = torch.zeros(capacity, nq, device=d)
        self._qvel = torch.zeros(capacity, nv, device=d)
        self._extra = torch.zeros(capacity, ne, device=d)
        self._steps = torch.zeros(capacity, dtype=torch.long, device=d)
        self._ever = torch.zeros(capacity + 1, dtype=torch.bool, device=d)
        self._versions = ICVersions(capacity, n_env, d, owner=self.name)
        self._n_traj = 0
        self._stored = torch.zeros(0, dtype=torch.long, device=d)  # filled slot ids
        self._traj_len = torch.zeros(max_trajectories, dtype=torch.long, device=d)
        self._frontier = torch.zeros(max_trajectories, dtype=torch.long, device=d)
        self._frontier_window = ICTable(max_trajectories, d, history=window)
        # Current episode per env: one row per offer, M rows hold a whole episode.
        rows = self._m
        self._rows = rows
        self._ep_qpos = torch.zeros(n_env, rows, nq, device=d)
        self._ep_qvel = torch.zeros(n_env, rows, nv, device=d)
        self._ep_extra = torch.zeros(n_env, rows, ne, device=d)
        self._ep_step = torch.full((n_env, rows), -1, dtype=torch.long, device=d)
        self._ep_count = torch.zeros(n_env, dtype=torch.long, device=d)

    def offer_visited(
        self,
        env_ids: Tensor,
        qpos: Tensor,
        qvel: Tensor,
        extra: Tensor,
        ep_step: Tensor,
    ) -> None:
        if self._n_traj >= self._k:
            return
        # A ring per env; it never wraps at the default visit_every (M rows per episode).
        pos = self._ep_count[env_ids] % self._rows
        self._ep_qpos[env_ids, pos] = qpos
        self._ep_qvel[env_ids, pos] = qvel
        self._ep_extra[env_ids, pos] = extra
        self._ep_step[env_ids, pos] = ep_step
        self._ep_count[env_ids] += 1

    def reset(
        self, env_ids: Tensor, outcomes: Outcomes | None
    ) -> tuple[StartStates, Tensor]:
        if outcomes is not None:
            fresh = self._versions.drop_stale(outcomes)
            if self._score is not None:
                self._score.record(fresh)
            self._track(fresh)
            if self._order == "frontier" and self._n_traj > 0:
                self._move_frontiers(fresh)
        self.capture(env_ids, outcomes)
        if self._n_traj == 0:
            n = env_ids.numel()
            states = to_device(
                self.ctx.task.sample_start(n, self.ctx.gen), self.ctx.device
            )
            ids = torch.full(
                (n,), TASK_START_IC_ID, dtype=torch.long, device=self.ctx.device
            )
        else:
            states, ids = mix_task_start(self.ctx, env_ids, self._draw)
        self._versions.start(env_ids, ids)
        return states, ids

    def capture(self, env_ids: Tensor, outcomes: Outcomes | None) -> None:
        """Store the trajectories of `outcomes` that reached reward (table room permitting)
        and start a fresh trajectory for each of `env_ids`. `reset` calls it; `phased`
        calls it directly while another method draws the starts."""
        if outcomes is not None and self._n_traj < self._k:
            ok = (
                outcomes.reached_reward
                & (outcomes.first_reward_step >= 0)
                & ~self._versions.started_from_table(outcomes.env_ids)
            )
            envs = outcomes.env_ids[ok].tolist()
            firsts = outcomes.first_reward_step[ok].tolist()
            for env, first in zip(envs, firsts, strict=True):  # at most k stores
                if self._n_traj >= self._k:
                    break
                self._store(env, first)
        self._ep_step[env_ids] = -1
        self._ep_count[env_ids] = 0
        self._versions.start(env_ids, torch.full_like(env_ids, TASK_START_IC_ID))

    def update(self, it: int, stats: RolloutStats) -> dict[str, float]:
        del it
        cells = self._versions.current_cells(stats)
        if self._score is not None:
            self._score.observe_rollout(cells)
        return {}

    def episode_versions(self) -> Tensor:
        """`(num_envs,)` 0 for an episode started from a stored state, -1 for a task start."""
        return self._versions.episode_versions()

    def ic_states(self) -> StartStates:
        return StartStates(self._qpos.clone(), self._qvel.clone(), self._extra.clone())

    def diagnostics(self) -> dict[str, float]:
        """`buffer_size`: stored states; `trajectories`: stored trajectories.
        `sampling_entropy`: nats over the task-start atom (p_task) plus each candidate
        ((1 - p_task) * draw weight); 0.0 while inert. `never_succeeded_share`: stored
        states with no success from them yet (NaN while inert). `frontier_steps_to_reward`
        (frontier order): mean steps-to-reward of the frontier states (NaN while inert)."""
        n_stored = self._stored.numel()
        entropy = 0.0
        never = float("nan")
        frontier_steps = float("nan")
        if n_stored > 0:
            p_task = self.ctx.p_task
            p = self._weights(self._candidates()) * (1.0 - p_task)
            entropy = -float(torch.special.xlogy(p, p).sum())
            entropy -= p_task * math.log(p_task) if p_task > 0 else 0.0
            never = float((~self._ever[self._stored]).float().mean())
            front = self._frontier_slots()
            frontier_steps = float(self._steps[front].float().mean())
        diag = {
            "sampling_entropy": entropy,
            "buffer_size": float(n_stored),
            "never_succeeded_share": never,
            "trajectories": float(self._n_traj),
        }
        if self._order == "frontier":
            diag["frontier_steps_to_reward"] = frontier_steps
        return diag

    def _store(self, env: int, first: int) -> None:
        steps = self._ep_step[env]
        keep = ((steps >= 0) & (steps <= first)).nonzero(as_tuple=True)[0]
        rows = keep[torch.argsort(steps[keep], stable=True)][-self._m :]
        raw = StartStates(
            self._ep_qpos[env, rows],
            self._ep_qvel[env, rows],
            self._ep_extra[env, rows],
        )
        states, valid = prepare_candidates(self.ctx.task, raw)
        # Invalid rows are dropped; the rest keep their time order and steps-to-reward.
        rows, states = rows[valid], states.index(valid)
        n = rows.numel()
        if n == 0:  # no valid state before the reward: nothing to restart from
            return
        t = self._n_traj
        ids = torch.arange(t * self._m, t * self._m + n, device=self.ctx.device)
        self._qpos[ids] = states.qpos
        self._qvel[ids] = states.qvel
        self._extra[ids] = states.extra
        self._steps[ids] = first - steps[rows]
        self._traj_len[t] = n
        self._frontier[t] = n - 1
        self._stored = torch.cat([self._stored, ids])
        self._n_traj += 1
        if isinstance(self._score, StepsToReward):
            self._score.set_steps(ids, self._steps[ids])

    def _frontier_slots(self) -> Tensor:
        t = torch.arange(self._n_traj, device=self.ctx.device)
        return t * self._m + self._frontier[: self._n_traj]

    def _candidates(self) -> Tensor:
        return self._stored if self._order == "uniform" else self._frontier_slots()

    def _weights(self, cands: Tensor) -> Tensor:
        uniform = torch.full(
            (cands.numel(),), 1.0 / cands.numel(), device=self.ctx.device
        )
        if self._score is None:
            return uniform
        _, trusted = self._score.values(cands)
        p_s = torch.zeros_like(uniform)
        trusted_ids = cands[trusted]
        if trusted_ids.numel() > 0:
            p_s[trusted] = self._score.probs(trusted_ids)
        p_s = median_fill(p_s, trusted)
        total = p_s.sum()
        return torch.where(total > 0, p_s / total.clamp(min=1e-30), uniform)

    def _draw(self, k: int) -> tuple[StartStates, Tensor]:
        cands = self._candidates()
        pick = torch.multinomial(
            self._weights(cands), k, replacement=True, generator=self.ctx.gen
        )
        idx = cands[pick]
        return StartStates(self._qpos[idx], self._qvel[idx], self._extra[idx]), idx

    def _track(self, outcomes: Outcomes) -> None:
        keep = outcomes.full & (outcomes.ic_ids >= 0)
        slot = torch.where(
            keep, outcomes.ic_ids, torch.full_like(outcomes.ic_ids, self._capacity)
        )
        successes = torch.zeros_like(self._ever, dtype=torch.long)
        successes.index_add_(0, slot, outcomes.success.long())
        self._ever |= successes > 0
        self._ever[self._capacity] = False

    def _move_frontiers(self, outcomes: Outcomes) -> None:
        ic = outcomes.ic_ids
        traj = ic.clamp(min=0) // self._m
        at_front = (ic >= 0) & (ic == traj * self._m + self._frontier[traj])
        tagged = torch.where(at_front, traj, torch.full_like(traj, -1))
        self._frontier_window.record(dataclasses.replace(outcomes, ic_ids=tagged))
        t = torch.arange(self._n_traj, device=self.ctx.device)
        full = self._frontier_window.count(t) >= self._window
        rate = self._frontier_window.success_rate(t)
        back = (full & (rate >= self._advance)).long()
        forward = (full & (rate <= self._retreat)).long()
        old = self._frontier[: self._n_traj]
        last = self._traj_len[: self._n_traj] - 1
        new = torch.minimum((old - back + forward).clamp(min=0), last)
        moved = t[new != old]
        self._frontier[: self._n_traj] = new
        if moved.numel() > 0:
            self._frontier_window.forget(moved)
