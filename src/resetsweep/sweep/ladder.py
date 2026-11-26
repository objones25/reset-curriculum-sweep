"""Expected arm counts per rung of the scheduler's ladder (final-review ruling C2b).

`main.size_rungs` turns these counts into a rung-0 step budget, so the rung sizes come from the
real arm space rather than a share heuristic. The model mirrors `Scheduler._promote` and the
two expansion stages of `arm_space`:

- In every group ``(task, task_kw)``, the quota is ``ceil(n / eta)`` of the ``n`` expected
  non-protected arms; protected arms (`uniform`, `point`) always go up.
- Which arms win is unknown, so each non-protected arm wins with probability
  ``presence * min(1, quota / n)``, independently (an assumption: in reality winners are
  correlated through shared methods, which this ignores).
- Promotion out of rung 0 adds `arm_space("rung1_expand", survivors=[a])` for each winner a;
  promotion into the final rung adds `arm_space("final_expand", survivors=[a])`. An arm that
  several possible winners would add (a dense copy shared by the same config on two tasks, a
  variant group's anchors) is present with ``1 - prod(1 - p_winner)``: that is the dedupe.
"""

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from resetsweep.sweep.arms import Arm, arm_space
from resetsweep.sweep.scheduler import PROTECTED_METHODS, GroupKey, group_of


@dataclass(frozen=True)
class LadderProjection:
    """`per_rung[k]` maps task name -> expected number of arms at rung k."""

    per_rung: tuple[Mapping[str, float], ...]

    def totals(self) -> tuple[float, ...]:
        return tuple(sum(counts.values()) for counts in self.per_rung)


def _merge(pool: dict[Arm, float], arm: Arm, p: float) -> None:
    """Combine an independent chance `p` of `arm` being present into `pool`."""
    pool[arm] = 1.0 - (1.0 - pool.get(arm, 0.0)) * (1.0 - p)


def _promote(
    pool: Mapping[Arm, float], *, eta: int, rung: int, last: int
) -> dict[Arm, float]:
    groups: dict[GroupKey, list[Arm]] = defaultdict(list)
    for arm in pool:
        groups[group_of(arm)].append(arm)
    nxt: dict[Arm, float] = {}
    win: dict[Arm, float] = {}
    for members in groups.values():
        contenders = [a for a in members if a.method not in PROTECTED_METHODS]
        n = sum(pool[a] for a in contenders)
        share = min(1.0, math.ceil(n / eta) / n) if n > 0 else 0.0
        for a in members:
            p = pool[a] if a.method in PROTECTED_METHODS else pool[a] * share
            nxt[a] = p
            if a.method not in PROTECTED_METHODS:
                win[a] = p
    stages = (["rung1_expand"] if rung == 0 else []) + (
        ["final_expand"] if rung + 1 == last else []
    )
    for stage in stages:
        for a, p in win.items():
            for e in arm_space(stage, survivors=[a]):  # pyright: ignore[reportArgumentType] # literal stage names
                if e not in pool:
                    _merge(nxt, e, p)
    return nxt


def project_ladder(
    arms: Sequence[Arm], *, eta: int, n_rungs: int = 3
) -> LadderProjection:
    """Expected arm counts per task at each of `n_rungs` rungs, starting from the rung-0 `arms`."""
    if eta < 2:
        raise ValueError(f"eta must be >= 2, got {eta}")
    if n_rungs < 1:
        raise ValueError(f"n_rungs must be >= 1, got {n_rungs}")
    if not arms:
        raise ValueError("arms is empty")
    pool: dict[Arm, float] = dict.fromkeys(arms, 1.0)
    per_rung: list[Mapping[str, float]] = []
    for rung in range(n_rungs):
        counts: dict[str, float] = defaultdict(float)
        for arm, p in pool.items():
            counts[arm.task] += p
        per_rung.append(dict(counts))
        if rung + 1 < n_rungs:
            pool = _promote(pool, eta=eta, rung=rung, last=n_rungs - 1)
    return LadderProjection(per_rung=tuple(per_rung))
