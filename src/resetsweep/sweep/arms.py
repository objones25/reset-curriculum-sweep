"""The arm space (spec sections 5.4 and 7): which (task, method, score) combinations run.

An `Arm` is one cell of the sweep. Compatibility is checked at construction against the
method classes' own `accepts_scores`, so a nonsensical pair cannot exist. Which of the
accepted scores each method actually gets is the table `_SCORE_PLANS` below (the controller's
rulings on spec 5.4); `_check_plans_cover_methods` makes a newly registered method fail loudly
until it has a plan.

`score_kw` never carries `seed`: `train.py` injects the score seed itself.

`rung1_expand` notes:
- Each expansion changes one setting at a time (p_task, own_success order, inverted score,
  task variant, dense task); they are not crossed.
- Task variants exist for `maze` (lanes=5) and `maze_trap` (trap_scale 0.5, 2.0) only.
- It returns only arms that rung 0 did not run, deduplicated in `arm_space`: `visited` gets
  no p_task variants (rung 0 ran the whole sweep), and `uniform`/`point` are added on the
  dense tasks and on every task variant it creates (each variant is its own ranking group
  and needs its own normalisation anchor), never on the plain sparse tasks.
- `critic_value` arms are never put on the dense tasks: the `arm_dense` and `claw` rewards are
  mostly negative, so `V * (vmax - V)` (scores/critic.py) is meaningless there.
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
Stage = Literal["rung0", "rung1_expand"]

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


def _expand(arm: Arm) -> list[Arm]:
    """Everything rung 1 adds for one rung-0 survivor, one change at a time."""
    out: list[Arm] = []
    if arm.method not in _NO_P_TASK and arm.method != "visited":
        out += [_with_method_kw(arm, p_task=p) for p in P_TASK_RUNG1]
    if arm.method == "own_success":
        other = [o for o in get_args(Order) if o != arm.method_kwargs()["order"]]
        out += [_with_method_kw(arm, order=o) for o in other]
    accepted: frozenset[str] = METHODS.get(arm.method).accepts_scores  # pyright: ignore[reportAttributeAccessIssue] # every registered method declares it
    if arm.score is not None and "inverted" in accepted:
        inverted_kw: Kw = (("inner", arm.score), ("inner_kw", arm.score_kw))
        out.append(replace(arm, score="inverted", score_kw=inverted_kw))
    out += [replace(arm, task_kw=kw) for kw in _TASK_VARIANTS.get(arm.task, ())]
    if not _uses_critic_value(arm):
        out += [replace(arm, task=t) for t in DENSE_TASKS]
    return out


def _uses_critic_value(arm: Arm) -> bool:
    return "critic_value" in (arm.score, dict(arm.score_kw).get("inner"))


def _unique(arms: Sequence[Arm]) -> list[Arm]:
    return list(dict.fromkeys(arms))


def arm_space(stage: Stage, *, survivors: Sequence[Arm] = ()) -> list[Arm]:
    """Arms for `stage`. `rung0`: every compatible method x score on the sparse tasks.
    `rung1_expand`: `uniform` and `point` on every dense task and every new task variant,
    plus what each rung-0 survivor adds (p_task, order, inverted score, task variants, dense
    tasks); survivors themselves are not repeated."""
    if stage == "rung0":
        return _unique([a for t in SPARSE_TASKS for a in _rung0_arms_for(t)])
    if stage != "rung1_expand":
        raise ValueError(f"unknown stage {stage!r}; expected 'rung0' or 'rung1_expand'")
    for s in survivors:
        if s.task not in SPARSE_TASKS or s.task_kw:
            raise ValueError(f"survivor must be a rung 0 arm, got {s.label()}")
    dense_controls = [
        a
        for a in (arm for t in DENSE_TASKS for arm in _rung0_arms_for(t))
        if a.method in _NO_P_TASK
    ]
    added = [a for s in survivors for a in _expand(s)]
    variant_controls = [
        replace(c, task_kw=a.task_kw)
        for a in _unique(added)
        if a.task_kw
        for c in _rung0_arms_for(a.task)
        if c.method in _NO_P_TASK
    ]
    already_run = set(arm_space("rung0")) | set(survivors)
    candidates = _unique(dense_controls + variant_controls + added)
    return [a for a in candidates if a not in already_run]
