"""M9 `phased`: the shortlist's section 1 item 5 "suggested build" (ours): restart from
visited states weighted by visit count until reward appears, then along the agent's own
successful trajectories with a success-band score.

Composition, not inheritance: it builds an inner `visited_count` (with its own
`visit_count` score) and an inner `own_success` (with its own `success_band(form="pq")`
score) through `METHODS` and `SCORES`, so the arm's score stays None.

Contract for train.py:

- Before the first outcome with `reached_reward`, every call goes to `visited_count`;
  `own_success` still gets every `offer_visited` and is told about every reset
  (`capture`), so the first successful trajectory is stored even though `visited_count`
  drew its start.
- The switch happens inside the `reset` call that receives that outcome; that call and all
  later ones go to `own_success`. `phase` (0 or 1) tells train.py which is active.
- `ic_states()`, `update()` and `score` forward to the active inner method. After a
  switch, `ic_states()` returns a different table (`visited_count`'s `IC_CAPACITY_DEFAULT`
  rows before, `own_success.table_capacity(task)` rows after) and `score` a different
  object, so train.py reads both once per update, after the rollout, as for any IC method.
- Phase epoch: `episode_versions()` returns the active inner version `v` tagged with the
  phase, `2 * v + phase` (task starts stay -1). `update` decodes the tags and turns every
  rollout cell recorded under the other phase, or with a negative tag, into ic id -1
  before forwarding,
  so no cell from before the switch reaches `own_success`, whatever the inner versions are.
  train.py stacks `episode_versions()` per step as for any IC method and never decodes it.
- Outcomes: phased remembers the phase each env's episode started in and sets ic id -1 on
  outcomes of episodes from the other phase before forwarding, so a `visited_count` slot
  id (up to 1023) never indexes `own_success`'s smaller table. The rewarded outcome that
  triggers the switch is one of these; `own_success` still captures its trajectory, since
  capture keys on the env (stamped -1 by `capture`), not on the ic id.
- As with every IC method, the method drives the active score (`record`,
  `observe_rollout`); train.py must not call those itself.
"""

import dataclasses
import math

import torch
from torch import Tensor

from resetsweep.methods._slots import IC_CAPACITY_DEFAULT
from resetsweep.methods.base import METHODS, MethodBase, MethodContext
from resetsweep.methods.own_success import Order, OwnSuccess, table_capacity
from resetsweep.methods.visited_count import CountRule, VisitedCount
from resetsweep.scores.base import SCORES, Score
from resetsweep.scores.success_band import SuccessBand
from resetsweep.scores.visit_count import VisitCount
from resetsweep.types import Outcomes, RolloutStats, StartStates

_PHASES = 2  # the phase epoch is packed into episode versions as 2 * v + phase


@METHODS.register("phased")
class Phased(MethodBase):
    """`visited_count` until the first reward, then `own_success`; see the module docstring.

    Diagnostics are the active inner method's, plus `phase` (0.0 or 1.0), `switch_step`
    (the total step count at the switch; NaN before) and `first_reward_step` (the same
    value, since the switch is the first reward).
    """

    name = "phased"
    accepts_scores = frozenset({"none"})

    def __init__(
        self,
        ctx: MethodContext,
        *,
        count: CountRule = "neighbour",
        radius: float = 0.5,
        order: Order = "frontier",
    ) -> None:
        if ctx.score is not None:
            raise ValueError(
                "phased builds its own scores (visit_count, then success_band pq); "
                f"ctx.score must be None, got {type(ctx.score).__name__}"
            )
        super().__init__(ctx)
        explore_score = SCORES.construct(
            "visit_count", capacity=IC_CAPACITY_DEFAULT, device=ctx.device
        )
        exploit_score = SCORES.construct(
            "success_band",
            capacity=table_capacity(ctx.task),
            device=ctx.device,
            form="pq",
        )
        if not isinstance(explore_score, VisitCount) or not isinstance(
            exploit_score, SuccessBand
        ):
            raise TypeError(
                f"registry built {type(explore_score).__name__} and "
                f"{type(exploit_score).__name__}"
            )
        explore = METHODS.construct(
            "visited_count",
            dataclasses.replace(ctx, score=explore_score),
            count=count,
            radius=radius,
        )
        exploit = METHODS.construct(
            "own_success",
            dataclasses.replace(ctx, score=exploit_score),
            order=order,
        )
        if not isinstance(explore, VisitedCount) or not isinstance(exploit, OwnSuccess):
            raise TypeError(
                f"registry built {type(explore).__name__} and {type(exploit).__name__}"
            )
        self._explore = explore
        self._exploit = exploit
        self._scores: tuple[Score, Score] = (explore_score, exploit_score)
        self._phase = 0
        self._switch_step = float("nan")
        # Phase each env's current episode started in; its ic id names that phase's table.
        self._env_phase = torch.zeros(ctx.num_envs, dtype=torch.long, device=ctx.device)

    @property
    def phase(self) -> int:
        return self._phase

    @property
    def score(self) -> Score:
        """The active inner method's score."""
        return self._scores[self._phase]

    def offer_visited(
        self,
        env_ids: Tensor,
        qpos: Tensor,
        qvel: Tensor,
        extra: Tensor,
        ep_step: Tensor,
    ) -> None:
        if self._phase == 0:
            self._explore.offer_visited(env_ids, qpos, qvel, extra, ep_step)
        self._exploit.offer_visited(env_ids, qpos, qvel, extra, ep_step)

    def reset(
        self, env_ids: Tensor, outcomes: Outcomes | None
    ) -> tuple[StartStates, Tensor]:
        if (
            self._phase == 0
            and outcomes is not None
            and bool(outcomes.reached_reward.any())
        ):
            self._phase = 1
            self._switch_step = float(self.ctx.counter.total)
        if outcomes is not None:
            outcomes = self._same_phase(outcomes)
        self._env_phase[env_ids] = self._phase
        if self._phase == 1:
            return self._exploit.reset(env_ids, outcomes)
        starts = self._explore.reset(env_ids, outcomes)
        self._exploit.capture(env_ids, outcomes)
        return starts

    def _same_phase(self, outcomes: Outcomes) -> Outcomes:
        """`outcomes` with ic id -1 for episodes that started in the other phase: their ic
        ids name the other inner method's table, which may be larger than the active one."""
        other = self._env_phase[outcomes.env_ids] != self._phase
        ic = outcomes.ic_ids
        return dataclasses.replace(
            outcomes, ic_ids=torch.where(other, torch.full_like(ic, -1), ic)
        )

    def update(self, it: int, stats: RolloutStats) -> dict[str, float]:
        tags = stats.ic_versions
        if tags is None or tags.shape != stats.ic_ids.shape:
            return self._active().update(it, stats)  # the inner method raises on these
        # Keep only cells of IC-started episodes recorded in the active phase. A negative tag
        # is a task start or an episode the active method did not start (own_success stamps
        # -1 on episodes visited_count began), whose ic id may be out of its range.
        keep = (tags >= 0) & (tags % _PHASES == self._phase)
        versions = torch.where(tags >= 0, tags // _PHASES, tags)
        # The decoded inner versions are kept: an ic id of -1 alone makes scores skip a cell.
        current = dataclasses.replace(
            stats,
            ic_ids=torch.where(keep, stats.ic_ids, torch.full_like(tags, -1)),
            ic_versions=versions,
        )
        return self._active().update(it, current)

    def episode_versions(self) -> Tensor:
        """`(num_envs,)` active inner version `v` tagged as `2 * v + phase`; -1 stays -1."""
        v = self._active().episode_versions()
        return torch.where(v >= 0, v * _PHASES + self._phase, v)

    def ic_states(self) -> StartStates:
        return self._active().ic_states()

    def diagnostics(self) -> dict[str, float]:
        diag = self._active().diagnostics()
        diag["phase"] = float(self._phase)
        diag["switch_step"] = self._switch_step
        if not math.isnan(self._switch_step):
            diag["first_reward_step"] = self._switch_step
        return diag

    def _active(self) -> VisitedCount | OwnSuccess:
        return self._exploit if self._phase == 1 else self._explore
