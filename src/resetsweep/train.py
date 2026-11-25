"""One complete training run: the composition root that wires task, score, method, envs,
PPO and evaluation together, and reports metrics to a sink.

Loop, per iteration (spec sections 5.5, 5.6 and 6):

1. Roll out `rollout_len` vector steps with the observation statistics frozen
   (`collect_rollout`); store normalised obs, values and the critic on `final_obs`.
2. GAE on raw rewards (`compute_gae`), bootstrapping truncated steps from `final_obs`.
3. Build `RolloutStats` (raw GAE, per-IC start obs and features from `method.ic_states()`)
   and call `method.update`. Methods with scores drive them themselves; this loop never
   calls `score.record` or `score.observe_rollout`.
4. `norm.update(raw obs)` once, then `ppo_update`.
5. Log, and evaluate when `eval_every_steps` more steps have passed.

The run stops once `counter.total >= step_budget`. The counter includes the extra steps
methods spend on walks, screens and scouting (anti-goal 2), so an arm that walks trains on
fewer env steps. The last iteration may overshoot the budget by at most one iteration;
the counter's hard cap (`_hard_budget`) only catches a runaway method.

Host syncs: none per step beyond `VecEnv`'s accepted `done.nonzero()`; per-step metrics are
accumulated in device tensors and read with `.item()`/`float()` once per iteration.
"""

import dataclasses
import hashlib
import inspect
import itertools
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import torch
from torch import Tensor

from resetsweep.methods import METHODS
from resetsweep.methods._population import R_DEFAULT, population_size
from resetsweep.methods._slots import IC_CAPACITY_DEFAULT
from resetsweep.methods.base import P_TASK_DEFAULT, Method, MethodContext
from resetsweep.methods.own_success import MAX_TRAJECTORIES_DEFAULT, table_capacity
from resetsweep.metrics import MetricsSink
from resetsweep.ppo.gae import compute_gae
from resetsweep.ppo.networks import ActorCritic
from resetsweep.ppo.normalizer import RunningNorm
from resetsweep.ppo.update import Batch, PPOConfig, ppo_update
from resetsweep.scores import SCORES
from resetsweep.scores.base import Score
from resetsweep.sim.batch import BatchSim
from resetsweep.sim.env import VecEnv
from resetsweep.sim.rollouts import Rollouts
from resetsweep.tasks.base import TASKS, Task
from resetsweep.types import Outcomes, RolloutStats, StartStates, StepCounter

TASKS.load("resetsweep.tasks")

EXTRA_KINDS = ("walk", "screen", "scout")
TASK_START_IC_ID = -1
# Eval steps are never charged to the arm; their own counter only guards against a bug.
_EVAL_BUDGET = 2**62
# Worst-group success: mean per-IC success of the worst quarter of ICs seen (ours).
_WORST_GROUP_SHARE = 0.25


# --- spec and result -------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class RunSpec:
    """Everything that defines one run. `method_kw` may hold `p_task`, which goes to
    `MethodContext` (default `P_TASK_DEFAULT`); every other key goes to the method."""

    task: str
    task_kw: Mapping[str, Any] = field(default_factory=dict)
    method: str
    method_kw: Mapping[str, Any] = field(default_factory=dict)
    score: str | None
    score_kw: Mapping[str, Any] = field(default_factory=dict)
    seed: int
    step_budget: int
    num_envs: int
    rollout_len: int
    ppo: PPOConfig
    eval_every_steps: int
    eval_n: int = 512
    eval_seed: int = 12345
    device: str

    def __post_init__(self) -> None:
        for name in (
            "step_budget",
            "num_envs",
            "rollout_len",
            "eval_every_steps",
            "eval_n",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        batch = self.num_envs * self.rollout_len
        if self.ppo.minibatch_size > batch:
            raise ValueError(
                f"ppo.minibatch_size {self.ppo.minibatch_size} exceeds the rollout batch "
                f"num_envs * rollout_len = {batch}"
            )

    def arm_id(self) -> str:
        """Stable hash of every field except `seed` and `device`: runs of one arm share it."""
        fields = {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name not in ("seed", "device")
        }
        fields["ppo"] = dataclasses.asdict(self.ppo)
        blob = json.dumps(_plain(fields, "spec"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def config(self) -> dict[str, Any]:
        """Plain-dict view for metric sinks (W&B config), with `arm_id` added."""
        out = dataclasses.asdict(self)
        out["arm_id"] = self.arm_id()
        return out


def _plain(value: object, path: str) -> object:
    """`value` as JSON-ready plain data: any Mapping becomes a dict, tuples become lists.
    Raises TypeError naming the field path for anything else, so arm ids never depend on
    a repr."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v, f"{path}.{k}") for k, v in value.items()}  # pyright: ignore[reportUnknownVariableType] # kw values are arbitrary
    if isinstance(value, list | tuple):
        return [_plain(v, f"{path}[{i}]") for i, v in enumerate(value)]  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType] # kw values are arbitrary
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(
        f"{path.removeprefix('spec.')} has a {type(value).__module__}.{type(value).__qualname__} "
        "value; RunSpec fields must be plain data (str, int, float, bool, None, lists, mappings)"
    )


@dataclass(frozen=True)
class RunResult:
    """Outcome of one run. `ever_succeeded_share` is `1 - never_succeeded_share` from the
    method's last diagnostics, and NaN for methods without per-IC outcomes (uniform, point,
    visited): ranking code must treat NaN as "not applicable", not as 0."""

    arm_id: str
    seed: int
    status: Literal["ok", "failed"]
    error: str | None
    auc: float
    final_success: float
    reached_reward: bool
    steps_to_first_reward: int | None
    ever_succeeded_share: float
    env_steps_train: int
    env_steps_extra: dict[str, int]
    wall_s: float


# --- score capacity --------------------------------------------------------------------------


def _slot_capacity(task: Task, num_envs: int, kw: Mapping[str, Any]) -> int:
    del task, num_envs
    return int(kw.get("ic_capacity", IC_CAPACITY_DEFAULT))


def _own_success_capacity(task: Task, num_envs: int, kw: Mapping[str, Any]) -> int:
    del num_envs
    return table_capacity(
        task, max_trajectories=int(kw.get("max_trajectories", MAX_TRAJECTORIES_DEFAULT))
    )


def _population_capacity(task: Task, num_envs: int, kw: Mapping[str, Any]) -> int:
    del task
    return population_size(num_envs, R=int(kw.get("R", R_DEFAULT)))


# The score's capacity must equal the method's IC capacity (checked again by each method).
_CAPACITY_RULES: dict[str, Callable[[Task, int, Mapping[str, Any]], int]] = {
    "buffer_scored": _slot_capacity,
    "visited_count": _slot_capacity,
    "own_success": _own_success_capacity,
    "walk": _population_capacity,
    "ball": _population_capacity,
    "genetic": _population_capacity,
}


def score_capacity(
    method: str, *, task: Task, num_envs: int, method_kw: Mapping[str, Any]
) -> int:
    """Capacity the score of `method` must have, from the helper that method exports."""
    rule = _CAPACITY_RULES.get(method)
    if rule is None:
        raise ValueError(f"method {method!r} takes no score")
    return rule(task, num_envs, method_kw)


# --- rollout ---------------------------------------------------------------------------------


class _Versioned(Protocol):
    def episode_versions(self) -> Tensor: ...


@dataclass(frozen=True)
class Rollout:
    """One rollout, all `(T, N)` unless noted. `obs` is normalised with the frozen stats,
    `raw_obs` is `(T * N, obs_dim)` for `norm.update`. Cell `t` belongs to the episode that
    was running when `obs[t]` was observed: `ic_ids[t]` and `ic_versions[t]` are both read
    after step `t - 1`'s resets (after `reset_all` for `t = 0`), at the same point."""

    obs: Tensor
    raw_obs: Tensor
    actions: Tensor
    logp: Tensor
    values: Tensor
    rewards: Tensor
    terminated: Tensor
    truncated: Tensor
    bootstrap: Tensor
    ic_ids: Tensor
    ic_versions: Tensor | None
    outcomes: list[Outcomes]
    next_obs: Tensor
    first_reward_step: Tensor  # () long: counter total at the first reward > 0, or -1


def collect_rollout(
    env: VecEnv,
    model: ActorCritic,
    norm: RunningNorm,
    obs: Tensor,
    *,
    steps: int,
    versioned: _Versioned | None,
    counter: StepCounter | None = None,
    first_reward_step: Tensor | None = None,
) -> Rollout:
    """`steps` vector steps from `obs` with `norm` frozen. With `versioned`, its
    `episode_versions()` is stacked next to each cell's ic ids (`ic_versions`)."""
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    n, d = env.num_envs, env.device
    obs_buf = torch.zeros(steps, n, env.task.obs_dim, device=d)
    raw_buf = torch.zeros(steps, n, env.task.obs_dim, device=d)
    act_buf = torch.zeros(steps, n, env.task.act_dim, device=d)
    logp_buf, val_buf, rew_buf, boot_buf = (
        torch.zeros(steps, n, device=d) for _ in range(4)
    )
    term_buf = torch.zeros(steps, n, dtype=torch.bool, device=d)
    trunc_buf = torch.zeros(steps, n, dtype=torch.bool, device=d)
    ids_buf = torch.zeros(steps, n, dtype=torch.long, device=d)
    ver_buf = (
        torch.zeros(steps, n, dtype=torch.long, device=d)
        if versioned is not None
        else None
    )
    first = (
        first_reward_step
        if first_reward_step is not None
        else torch.full((), -1, dtype=torch.long, device=d)
    )
    outcomes: list[Outcomes] = []
    ic_ids = env.ic_ids
    for t in range(steps):
        ids_buf[t] = ic_ids
        if ver_buf is not None and versioned is not None:
            ver_buf[t] = versioned.episode_versions()
        obs_n = norm.normalize(obs)
        action, logp, value = model.act(obs_n)
        out = env.step(action)
        with torch.no_grad():
            boot_buf[t] = model.value(norm.normalize(out.final_obs))
        raw_buf[t], obs_buf[t], act_buf[t] = obs, obs_n, action
        logp_buf[t], val_buf[t], rew_buf[t] = logp, value, out.reward
        term_buf[t], trunc_buf[t] = out.terminated, out.truncated
        if counter is not None:
            hit = (out.reward > 0).any() & (first < 0)
            first = torch.where(hit, counter.total, first)
        if out.outcomes is not None:
            outcomes.append(out.outcomes)
        obs = out.obs
        ic_ids = out.ic_ids
    return Rollout(
        obs=obs_buf,
        raw_obs=raw_buf.reshape(steps * n, -1),
        actions=act_buf,
        logp=logp_buf,
        values=val_buf,
        rewards=rew_buf,
        terminated=term_buf,
        truncated=trunc_buf,
        bootstrap=boot_buf,
        ic_ids=ids_buf,
        ic_versions=ver_buf,
        outcomes=outcomes,
        next_obs=obs,
        first_reward_step=first,
    )


# --- start-state observation -----------------------------------------------------------------


class _StartObserver:
    """Observes IC start states on a private `BatchSim` (never the training envs). The sim
    is rebuilt when the table size changes (phased switches tables once)."""

    def __init__(self, task: Task, device: str) -> None:
        self._task = task
        self._device = device
        self._sim: BatchSim | None = None

    def observe(self, states: StartStates) -> Tensor:
        n = states.n
        if self._sim is None or self._sim.qpos.shape[0] != n:
            self._sim = BatchSim(
                self._task.mjcf,
                nworld=n,
                device=self._device,
                frame_skip=self._task.frame_skip,
            )
        rows = torch.arange(n, device=self._device)
        self._sim.set_state(
            rows, states.qpos.to(self._device), states.qvel.to(self._device)
        )
        return self._task.observe(self._sim, states.extra.to(self._device)).clone()


# --- evaluation ------------------------------------------------------------------------------


class FixedStarts:
    """Eval reset rule: env `i` always starts from fixed state `i` (ic id -1)."""

    name = "eval_fixed_starts"

    def __init__(self, states: StartStates) -> None:
        self._states = states

    def reset(
        self, env_ids: Tensor, outcomes: Outcomes | None
    ) -> tuple[StartStates, Tensor]:
        del outcomes
        ids = torch.full_like(env_ids, TASK_START_IC_ID)
        return self._states.index(env_ids), ids

    def offer_visited(
        self,
        env_ids: Tensor,
        qpos: Tensor,
        qvel: Tensor,
        extra: Tensor,
        ep_step: Tensor,
    ) -> None:
        del env_ids, qpos, qvel, extra, ep_step


class Evaluator:
    """Plays exactly one full episode from each fixed state, all clocks at 0 (no stagger),
    on its own `VecEnv` and its own uncharged counter. The env generator is reseeded with
    `seed` before every eval, so tasks that draw `extra` at reset see the same draws in
    every eval of every arm and seed."""

    def __init__(
        self, task: Task, states: StartStates, *, device: str, seed: int
    ) -> None:
        self._task = task
        self._seed = seed
        self._gen = torch.Generator(device=device).manual_seed(seed)
        self._env = VecEnv(
            task,
            num_envs=states.n,
            device=device,
            counter=StepCounter(budget=_EVAL_BUDGET),
            gen=self._gen,
        )
        self._env.bind(
            FixedStarts(
                StartStates(
                    states.qpos.to(device),
                    states.qvel.to(device),
                    states.extra.to(device),
                )
            )
        )

    def run(self, policy: Callable[[Tensor], Tensor]) -> tuple[float, float]:
        """`(success rate, mean return)` over the fixed states under `policy`."""
        env = self._env
        self._gen.manual_seed(self._seed)
        n, d = env.num_envs, env.device
        done = torch.zeros(n, dtype=torch.bool, device=d)
        success = torch.zeros(n, dtype=torch.bool, device=d)
        ret = torch.zeros(n, device=d)
        obs = env.reset_all(stagger=False)
        # Every episode lasts at most max_steps, so max_steps steps finish each env once.
        for _ in range(self._task.max_steps):
            out = env.step(policy(obs))
            if out.outcomes is not None:
                o = out.outcomes
                first = ~done[o.env_ids]
                success[o.env_ids] = torch.where(first, o.success, success[o.env_ids])
                ret[o.env_ids] = torch.where(first, o.ret, ret[o.env_ids])
                done[o.env_ids] = True
            obs = out.obs
        if not bool(done.all()):
            raise RuntimeError("eval ended with envs that never finished an episode")
        return float(success.float().mean()), float(ret.mean())


# --- metrics helpers -------------------------------------------------------------------------


def auc(steps: Sequence[int], success: Sequence[float], *, step_budget: int) -> float:
    """Trapezoid area of eval success over env steps, divided by `step_budget`. Points past
    the budget (the last iteration may overshoot) are moved to `step_budget`."""
    if len(steps) != len(success) or len(steps) < 2:
        raise ValueError(
            f"need matching steps and success with at least 2 points, got {len(steps)}, {len(success)}"
        )
    if step_budget <= 0:
        raise ValueError(f"step_budget must be positive, got {step_budget}")
    x = [min(s, step_budget) for s in steps]
    if any(b < a for a, b in itertools.pairwise(x)):
        raise ValueError(f"steps must be non-decreasing, got {list(steps)}")
    area = sum(
        (x1 - x0) * (y0 + y1) / 2
        for x0, x1, y0, y1 in zip(x, x[1:], success, success[1:], strict=False)
    )
    return area / step_budget


def _ic_success(outcomes: list[Outcomes], capacity: int) -> tuple[float, float]:
    """Per-IC success over this window's full IC-started episodes: `(mean over ICs, mean of
    the worst quarter of ICs)`, NaN when no IC finished an episode. Ids outside the current
    table (phased, right after its switch) are dropped."""
    nan = float("nan")
    if not outcomes or capacity <= 0:
        return nan, nan
    o = Outcomes.cat(outcomes)
    keep = o.full & (o.ic_ids >= 0) & (o.ic_ids < capacity)
    slot = torch.where(keep, o.ic_ids, torch.full_like(o.ic_ids, capacity))
    dev = slot.device
    count = torch.zeros(capacity + 1, device=dev).scatter_add_(
        0, slot, torch.ones_like(slot, dtype=torch.float32)
    )[:capacity]
    wins = torch.zeros(capacity + 1, device=dev).scatter_add_(
        0, slot, o.success.float()
    )[:capacity]
    seen = count > 0
    m = int(seen.sum())
    if m == 0:
        return nan, nan
    rate = (wins[seen] / count[seen]).sort().values
    worst = rate[: max(1, math.ceil(_WORST_GROUP_SHARE * m))]
    return float(rate.mean()), float(worst.mean())


def _mean_return(outcomes: list[Outcomes]) -> float:
    if not outcomes:
        return float("nan")
    return float(Outcomes.cat(outcomes).ret.float().mean())


def _hard_budget(spec: RunSpec) -> int:
    # Runaway guard only: the loop stops at step_budget. One iteration may overshoot by its
    # rollout plus the method's extra steps; a method spending more than step_budget extra
    # steps in one update is a bug and fails the run with BudgetExceeded.
    return 2 * spec.step_budget + spec.num_envs * spec.rollout_len


# --- composition -----------------------------------------------------------------------------


@dataclass
class _Parts:
    task: Task
    counter: StepCounter
    env: VecEnv
    method: Method
    score: Score | None
    model: ActorCritic
    norm: RunningNorm
    opt: torch.optim.Optimizer
    evaluator: Evaluator


# Score seeds are derived apart from spec.seed, which already seeds the env/method generator
# and torch's global RNG: equal seeds give equal streams, which would tie the `random`
# control's per-IC values to the method's own first draws. Affine map, so distinct run
# seeds keep distinct score seeds; it never equals spec.seed for any seed >= 0.
_SCORE_SEED_MULTIPLIER = 1_000_003
_SCORE_SEED_OFFSET = 17


def _score_seed(run_seed: int) -> int:
    return run_seed * _SCORE_SEED_MULTIPLIER + _SCORE_SEED_OFFSET


def _seeded_score_kw(name: str, kw: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """`kw` plus `seed=seed` (already derived by `_score_seed`) when score `name` takes a
    `seed` keyword and `kw` sets none, applied recursively with the same seed to an
    `inner` score's `inner_kw` (`inverted` wrapping `random`).
    The seed is injected here, not stored in `score_kw`, so the arm id stays seed-free."""
    params = inspect.signature(SCORES.get(name)).parameters
    out = dict(kw)
    if "seed" in params and "seed" not in out:
        out["seed"] = seed
    if "inner" in params and "inner" in out:
        out["inner_kw"] = _seeded_score_kw(
            str(out["inner"]), out.get("inner_kw") or {}, seed
        )
    return out


def build_score(spec: RunSpec, task: Task) -> Score | None:
    """The arm's score with the method's IC capacity, seeded with `_score_seed(spec.seed)`
    (inner scores too) when it draws random values; None for scoreless arms."""
    if spec.score is None:
        return None
    method_kw = {k: v for k, v in spec.method_kw.items() if k != "p_task"}
    capacity = score_capacity(
        spec.method, task=task, num_envs=spec.num_envs, method_kw=method_kw
    )
    kw = _seeded_score_kw(spec.score, spec.score_kw, _score_seed(spec.seed))
    return SCORES.construct(spec.score, capacity=capacity, device=spec.device, **kw)  # pyright: ignore[reportReturnType] # registry returns object


def _build(spec: RunSpec) -> _Parts:
    d = spec.device
    task: Task = TASKS.construct(spec.task, **spec.task_kw)  # pyright: ignore[reportAssignmentType] # registry returns object
    method_cls = METHODS.get(spec.method)
    accepted: frozenset[str] = method_cls.accepts_scores  # pyright: ignore[reportAttributeAccessIssue] # every registered method declares it
    if (spec.score or "none") not in accepted:
        raise ValueError(
            f"method {spec.method!r} does not accept score {spec.score!r}; "
            f"accepted: {sorted(accepted)}"
        )
    method_kw = dict(spec.method_kw)
    p_task = float(method_kw.pop("p_task", P_TASK_DEFAULT))
    score = build_score(spec, task)
    gen = torch.Generator(device=d).manual_seed(spec.seed)
    counter = StepCounter(budget=_hard_budget(spec))
    ctx = MethodContext(
        task=task,
        num_envs=spec.num_envs,
        device=d,
        gen=gen,
        score=score,
        rollouts=Rollouts(task, device=d, counter=counter, gen=gen),
        counter=counter,
        p_task=p_task,
    )
    method: Method = METHODS.construct(spec.method, ctx, **method_kw)  # pyright: ignore[reportAssignmentType] # registry returns object
    env = VecEnv(task, num_envs=spec.num_envs, device=d, counter=counter, gen=gen)
    env.bind(method)
    model = ActorCritic(task.obs_dim, task.act_dim).to(d)
    norm = RunningNorm((task.obs_dim,)).to(d)
    opt = torch.optim.Adam(model.parameters(), lr=spec.ppo.lr, eps=spec.ppo.adam_eps)
    evaluator = Evaluator(
        task,
        task.eval_states(spec.eval_n, spec.eval_seed),
        device=d,
        seed=spec.eval_seed,
    )
    return _Parts(task, counter, env, method, score, model, norm, opt, evaluator)


def _active_score(parts: _Parts) -> object | None:
    # phased builds its own scores and exposes the active one as `score`.
    return (
        parts.score if parts.score is not None else getattr(parts.method, "score", None)
    )


def _rollout_stats(
    parts: _Parts, roll: Rollout, adv: Tensor, returns: Tensor, observer: _StartObserver
) -> RolloutStats:
    table = parts.method.ic_states()
    obs_dim = parts.task.obs_dim
    d = roll.obs.device
    if table is None:
        start_obs = torch.zeros(0, obs_dim, device=d)
        start_features = torch.zeros(0, 0, device=d)
    else:
        # Same normalisation as the policy input of this rollout (stats still frozen).
        start_obs = parts.norm.normalize(observer.observe(table))
        start_features = parts.task.state_features(
            table.qpos.to(d), table.qvel.to(d), table.extra.to(d)
        )
    model = parts.model

    @torch.no_grad()
    def critic(x: Tensor) -> Tensor:
        return model.value(x)

    return RolloutStats(
        ic_ids=roll.ic_ids,
        adv=adv,
        returns=returns,
        values=roll.values,
        start_obs=start_obs,
        critic=critic,
        start_features=start_features,
        ic_versions=roll.ic_versions,
    )


def _versioned(method: Method) -> _Versioned | None:
    if method.ic_states() is None:
        return None
    if not hasattr(method, "episode_versions"):
        raise TypeError(
            f"method {method.name!r} returns ic_states() but has no episode_versions()"
        )
    return method  # pyright: ignore[reportReturnType] # checked with hasattr above


def _iteration_metrics(
    parts: _Parts,
    roll: Rollout,
    ppo_stats: dict[str, float],
    update_stats: dict[str, float],
    capacity: int,
) -> dict[str, float]:
    counter = parts.counter
    extra = counter.extra
    data: dict[str, float] = {
        "env_steps/total": float(counter.total),
        "env_steps/train": float(counter.train),
        **{f"env_steps/{k}": float(extra[k]) for k in EXTRA_KINDS},
        "train/curriculum_return": _mean_return(roll.outcomes),
    }
    ic_mean, ic_worst = _ic_success(roll.outcomes, capacity)
    data["ic/success"] = ic_mean
    data["ic/worst_group_success"] = ic_worst
    first = int(roll.first_reward_step)
    data["first_reward_step"] = float(first) if first >= 0 else float("nan")
    data.update({f"ppo/{k}": v for k, v in ppo_stats.items()})
    diag = parts.method.diagnostics()
    data["ic/never_succeeded_share"] = diag.pop("never_succeeded_share")
    data["ic/sampling_entropy"] = diag.pop("sampling_entropy")
    # Namespaced apart so a key both report can never overwrite the other.
    for k, v in diag.items():
        data[k if k.startswith("side/") else f"method/{k}"] = float(v)
    for k, v in update_stats.items():
        data[k if k.startswith("side/") else f"method_update/{k}"] = float(v)
    score = _active_score(parts)
    extra_metrics = getattr(score, "extra_metrics", None)
    if callable(extra_metrics):
        side: dict[str, float] = extra_metrics()  # pyright: ignore[reportAssignmentType] # side-model scores return dict[str, float]
        data.update({f"side/{k}": float(v) for k, v in side.items()})
    return data


# --- the run ---------------------------------------------------------------------------------


def run(spec: RunSpec, sink: MetricsSink) -> RunResult:
    """Train one arm for one seed. Exceptions propagate (the worker pool records them as
    failed results); the caller owns and closes `sink`."""
    start_s = time.perf_counter()
    torch.manual_seed(spec.seed)
    parts = _build(spec)
    counter, env, method, model, norm = (
        parts.counter,
        parts.env,
        parts.method,
        parts.model,
        parts.norm,
    )
    observer = _StartObserver(parts.task, spec.device)
    versioned = _versioned(method)

    @torch.no_grad()
    def policy(obs: Tensor) -> Tensor:
        return model.actor(norm.normalize(obs))  # deterministic: the Gaussian mean

    curve_steps: list[int] = []
    curve_success: list[float] = []

    def evaluate() -> dict[str, float]:
        success, ret = parts.evaluator.run(policy)
        curve_steps.append(counter.total)
        curve_success.append(success)
        return {"eval/success": success, "eval/return": ret}

    sink.log(counter.total, {"env_steps/total": float(counter.total), **evaluate()})
    next_eval = spec.eval_every_steps
    obs = env.reset_all()
    first_reward = torch.full((), -1, dtype=torch.long, device=spec.device)
    batch_steps = spec.num_envs * spec.rollout_len
    # Every iteration adds batch_steps train steps, so the budget ends the loop in time.
    max_iters = math.ceil(spec.step_budget / batch_steps) + 1
    it = 0
    diag: dict[str, float] = {}
    while counter.total < spec.step_budget:
        if it >= max_iters:
            raise RuntimeError(
                f"iteration bound {max_iters} reached with {counter.total} of "
                f"{spec.step_budget} steps"
            )
        iter_start_s = time.perf_counter()
        iter_start_steps = counter.total
        table = method.ic_states()
        capacity = table.n if table is not None else 0
        roll = collect_rollout(
            env,
            model,
            norm,
            obs,
            steps=spec.rollout_len,
            versioned=versioned,
            counter=counter,
            first_reward_step=first_reward,
        )
        obs, first_reward = roll.next_obs, roll.first_reward_step
        adv, returns = compute_gae(
            roll.rewards,
            roll.values,
            roll.terminated,
            roll.truncated,
            roll.bootstrap,
            gamma=spec.ppo.gamma,
            lam=spec.ppo.lam,
        )
        stats = _rollout_stats(parts, roll, adv, returns, observer)
        update_stats = method.update(it, stats)
        norm.update(roll.raw_obs)
        batch = Batch(
            obs=roll.obs,
            actions=roll.actions,
            logp=roll.logp,
            values=roll.values,
            advantages=adv,
            returns=returns,
        )
        ppo_stats = ppo_update(model, parts.opt, batch, spec.ppo)
        data = _iteration_metrics(parts, roll, ppo_stats, update_stats, capacity)
        diag = {"never_succeeded_share": data["ic/never_succeeded_share"]}
        iter_s = time.perf_counter() - iter_start_s
        data["perf/steps_per_s"] = (counter.total - iter_start_steps) / max(
            iter_s, 1e-9
        )
        data["perf/wall_s"] = time.perf_counter() - start_s
        if counter.total >= next_eval or counter.total >= spec.step_budget:
            data.update(evaluate())
            while next_eval <= counter.total:
                next_eval += spec.eval_every_steps
        sink.log(counter.total, data)
        it += 1

    first = int(first_reward)
    never = diag.get("never_succeeded_share", float("nan"))
    result = RunResult(
        arm_id=spec.arm_id(),
        seed=spec.seed,
        status="ok",
        error=None,
        auc=auc(curve_steps, curve_success, step_budget=spec.step_budget),
        final_success=curve_success[-1],
        reached_reward=first >= 0,
        steps_to_first_reward=first if first >= 0 else None,
        ever_succeeded_share=1.0 - never,
        env_steps_train=counter.train,
        env_steps_extra=counter.extra,
        wall_s=time.perf_counter() - start_s,
    )
    sink.summary(
        {
            "arm_id": result.arm_id,
            "seed": float(result.seed),
            "status": result.status,
            "auc": result.auc,
            "final_success": result.final_success,
            "reached_reward": float(result.reached_reward),
            "steps_to_first_reward": float(first) if first >= 0 else float("nan"),
            "ever_succeeded_share": result.ever_succeeded_share,
            "env_steps_train": float(result.env_steps_train),
            **{f"env_steps_{k}": float(v) for k, v in result.env_steps_extra.items()},
            "wall_s": result.wall_s,
        }
    )
    return result
