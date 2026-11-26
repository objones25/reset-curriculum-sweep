"""The arm space (spec sections 5.4 and 7): which (task, method, score) combinations run.

An `Arm` is one cell of the sweep. Compatibility is checked at construction against the
method classes' own `accepts_scores`, so a nonsensical pair cannot exist. Which of the
accepted scores each method actually gets is the table `_SCORE_PLANS` below (the controller's
rulings on spec 5.4); `_check_plans_cover_methods` makes a newly registered method fail loudly
until it has a plan.

`score_kw` never carries `seed`: `train.py` injects the score seed itself.

Expansion stages (spec 5.4, binding over the plan's Task 20 text; final-review ruling C2a):
- Each expansion changes one setting at a time; they are not crossed.
- `rung1_expand` (rung-0 survivors entering rung 1) adds only the p_task variants
  {0.2, 0.8} and the task variants: `maze` (lanes=5) and `maze_trap` (trap_scale 0.5, 2.0),
  each variant with its own `uniform`/`point` anchors (each variant is its own ranking group).
  `visited` gets no p_task variants (rung 0 ran the whole sweep). Other settings stay at their
  first-listed value (rung 0's), so own_success never gets its other `order`.
- `final_expand` (arms promoted into the final rung) adds `inverted(x)` for each scored
  survivor whose method accepts it, and a copy of each survivor on every dense task (task_kw
  cleared), plus `uniform`/`point` on the dense tasks once any dense copy exists. Spec 5.4:
  the dense tasks only run finalists, and the inverted control only costs money where it
  matters.
- Both return only arms not already in the input or rung 0, deduplicated; the scheduler skips
  any arm it already holds.
- `critic_value` arms (and `inverted(critic_value)`) are never put on the dense tasks: the
  `arm_dense` and `claw` rewards are mostly negative, so `V * (vmax - V)` (scores/critic.py) is
  meaningless there.
- The smoke test (tests/test_smoke.py) covers 32 envs on CPU only, not the sweep's sizes.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, get_args

from resetsweep import methods as _methods  # noqa: F401 # registers every method
from resetsweep import scores as _scores  # noqa: F401 # registers every score
from resetsweep.methods import METHODS, P_TASK_DEFAULT, P_TASK_SWEEP
from resetsweep.methods.own_success import Order
from resetsweep.methods.visited_count import CountRule
from resetsweep.scores.critic import Form as CriticForm
from resetsweep.scores.slope import Form as SlopeForm
from resetsweep.scores.success_band import Form as BandForm
from resetsweep.scores.visit_count import PostReward
from resetsweep.tasks.base import TASKS

TASKS.load("resetsweep.tasks")

Kw = tuple[tuple[str, Any], ...]
Stage = Literal["rung0", "rung1_expand", "final_expand"]

DENSE_TASKS: tuple[str, ...] = ("pendulum_dense", "arm_dense", "claw")
SPARSE_TASKS: tuple[str, ...] = tuple(
    n for n in TASKS.names() if not n.endswith("_dense") and n != "claw"
)

P_TASK_RUNG1: tuple[float, ...] = (0.2, 0.8)
# Methods whose draw ignores p_task (they never mix in the task start themselves).
_NO_P_TASK = frozenset({"uniform", "point"})
# Task variants for arms that reach rung 1 (spec 5.4): task name -> task_kw settings.
_TASK_VARIANTS: dict[str, tuple[Kw, ...]] = {
    "maze": ((("lanes", 5),),),
    "maze_trap": ((("trap_scale", 0.5),), (("trap_scale", 2.0),)),
}
# beta = 1 / temperature; the paper's beta is 90-130 (scores/critic.py), so 0.01.
_CRITIC_VALUE_KW: Kw = (("temperature", 0.01),)


@dataclass(frozen=True)
class Arm:
    task: str
    task_kw: Kw
    method: str
    method_kw: Kw
    score: str | None
    score_kw: Kw

    def __post_init__(self) -> None:
        accepted: frozenset[str] = METHODS.get(self.method).accepts_scores  # pyright: ignore[reportAttributeAccessIssue] # every registered method declares it
        if (self.score or "none") not in accepted:
            raise ValueError(
                f"method {self.method!r} does not accept score {self.score!r}; "
                f"accepts: {sorted(accepted)}"
            )

    def label(self) -> str:
        parts = [_fmt(self.task, self.task_kw), _fmt(self.method, self.method_kw)]
        parts.append(_fmt(self.score, self.score_kw) if self.score else "-")
        return "/".join(parts)

    def method_kwargs(self) -> dict[str, Any]:
        return dict(self.method_kw)

    def task_kwargs(self) -> dict[str, Any]:
        return dict(self.task_kw)

    def score_kwargs(self) -> dict[str, Any]:
        """`score_kw` as the dict `RunSpec` takes; `inner_kw` (of `inverted`) is nested."""
        out = dict(self.score_kw)
        if "inner_kw" in out:
            out["inner_kw"] = dict(out["inner_kw"])
        return out


def _fmt(name: str | None, kw: Kw) -> str:
    if not kw:
        return str(name)
    inner = ",".join(f"{k}={_fmt_value(v)}" for k, v in kw)
    return f"{name}[{inner}]"


def _fmt_value(value: Any) -> str:
    if isinstance(value, tuple):
        return "{" + ",".join(f"{k}={_fmt_value(v)}" for k, v in value) + "}"  # pyright: ignore[reportUnknownVariableType] # nested kw pairs
    return str(value)


# --- which scores each method gets ------------------------------------------------------------

# (score, score_kw, method_kw) per method; method_kw excludes p_task, which is set below.
type _Plan = tuple[str | None, Kw, Kw]


def _forms(name: str, forms: tuple[str, ...]) -> list[_Plan]:
    return [(name, (("form", f),), ()) for f in forms]


def _population_scores() -> list[_Plan]:
    """S1 + S3 + S4 forms, S5, success_classifier, rnd, random: 2+3+3+1+1+1+1 = 12."""
    return [
        *_forms("success_band", get_args(BandForm)),
        *_forms("slope", get_args(SlopeForm)),
        *_forms("critic_loss", get_args(CriticForm)),
        ("critic_value", _CRITIC_VALUE_KW, ()),
        ("success_classifier", (), ()),
        ("rnd", (), ()),
        ("random", (), ()),
    ]


def _visited_count_plans() -> list[_Plan]:
    return [
        ("visit_count", (("post_reward", post),), (("count", rule),))
        for post in get_args(PostReward)
        for rule in get_args(CountRule)
    ]


def _score_plans() -> dict[str, list[_Plan]]:
    none: list[_Plan] = [(None, (), ())]
    order = get_args(Order)[0]  # first-listed value at rung 0
    return {
        "uniform": none,
        "point": none,
        "visited": none,
        "phased": none,
        "own_success": [
            (None, (), (("order", order),)),
            ("steps_to_reward", (), (("order", order),)),
        ],
        "visited_count": _visited_count_plans(),
        "buffer_scored": _population_scores(),
        "walk": _population_scores(),
        "ball": _population_scores(),
        "genetic": _population_scores(),
    }


def _check_plans_cover_methods(plans: dict[str, list[_Plan]]) -> None:
    missing = set(METHODS.names()) - set(plans)
    unknown = set(plans) - set(METHODS.names())
    if missing or unknown:
        raise ValueError(
            f"score plans out of step with METHODS: missing {sorted(missing)}, "
            f"unknown {sorted(unknown)}"
        )


def _p_tasks(method: str) -> tuple[float | None, ...]:
    if method in _NO_P_TASK:
        return (None,)
    return P_TASK_SWEEP if method == "visited" else (P_TASK_DEFAULT,)


def _rung0_arms_for(task: str) -> list[Arm]:
    plans = _score_plans()
    _check_plans_cover_methods(plans)
    arms: list[Arm] = []
    for method in METHODS.names():
        for score, score_kw, method_kw in plans[method]:
            for p_task in _p_tasks(method):
                p_kw: Kw = () if p_task is None else (("p_task", p_task),)
                arms.append(Arm(task, (), method, p_kw + method_kw, score, score_kw))
    return arms


# --- rung 1 ------------------------------------------------------------------------------------


def _with_method_kw(arm: Arm, **changes: Any) -> Arm:
    kw = {**arm.method_kwargs(), **changes}
    return replace(arm, method_kw=tuple(kw.items()))


def _expand_rung1(arm: Arm) -> list[Arm]:
    """What rung 1 adds for one rung-0 survivor: p_task variants and task variants."""
    out: list[Arm] = []
    if arm.method not in _NO_P_TASK and arm.method != "visited":
        out += [_with_method_kw(arm, p_task=p) for p in P_TASK_RUNG1]
    out += [replace(arm, task_kw=kw) for kw in _TASK_VARIANTS.get(arm.task, ())]
    return out


def _expand_final(arm: Arm) -> list[Arm]:
    """What the final rung adds for one finalist: its inverted control and its dense copies."""
    out: list[Arm] = []
    accepted: frozenset[str] = METHODS.get(arm.method).accepts_scores  # pyright: ignore[reportAttributeAccessIssue] # every registered method declares it
    if arm.score is not None and arm.score != "inverted" and "inverted" in accepted:
        inverted_kw: Kw = (("inner", arm.score), ("inner_kw", arm.score_kw))
        out.append(replace(arm, score="inverted", score_kw=inverted_kw))
    if arm.method not in _NO_P_TASK and not _uses_critic_value(arm):
        out += [replace(arm, task=t, task_kw=()) for t in DENSE_TASKS]
    return out


def _uses_critic_value(arm: Arm) -> bool:
    return "critic_value" in (arm.score, dict(arm.score_kw).get("inner"))


def _unique(arms: Sequence[Arm]) -> list[Arm]:
    return list(dict.fromkeys(arms))


def _controls_for(task: str, task_kw: Kw) -> list[Arm]:
    return [
        replace(c, task_kw=task_kw)
        for c in _rung0_arms_for(task)
        if c.method in _NO_P_TASK
    ]


def arm_space(stage: Stage, *, survivors: Sequence[Arm] = ()) -> list[Arm]:
    """Arms for `stage` (see the module docstring). `rung0`: every compatible method x score
    on the sparse tasks. `rung1_expand`: what rung-0 survivors add at rung 1. `final_expand`:
    what arms promoted into the final rung add there. Survivors are never repeated."""
    if stage == "rung0":
        return _unique([a for t in SPARSE_TASKS for a in _rung0_arms_for(t)])
    if stage == "rung1_expand":
        for s in survivors:
            if s.task not in SPARSE_TASKS or s.task_kw:
                raise ValueError(f"survivor must be a rung 0 arm, got {s.label()}")
        added = [a for s in survivors for a in _expand_rung1(s)]
        variant_groups = dict.fromkeys((a.task, a.task_kw) for a in added if a.task_kw)
        controls = [c for t, kw in variant_groups for c in _controls_for(t, kw)]
    elif stage == "final_expand":
        for s in survivors:
            if s.task not in SPARSE_TASKS:
                raise ValueError(f"survivor must be on a sparse task, got {s.label()}")
        added = [a for s in survivors for a in _expand_final(s)]
        dense = dict.fromkeys(a.task for a in added if a.task in DENSE_TASKS)
        controls = [c for t in dense for c in _controls_for(t, ())]
    else:
        raise ValueError(
            f"unknown stage {stage!r}; expected 'rung0', 'rung1_expand' or 'final_expand'"
        )
    already_run = set(arm_space("rung0")) | set(survivors)
    return [a for a in _unique(controls + added) if a not in already_run]
