import dataclasses
import math
from types import MappingProxyType
from typing import Any

import pytest
import torch
from torch import Tensor

from resetsweep import train
from resetsweep.methods._population import population_size
from resetsweep.methods.own_success import table_capacity
from resetsweep.methods.uniform import Uniform
from resetsweep.ppo.networks import ActorCritic
from resetsweep.ppo.normalizer import RunningNorm
from resetsweep.ppo.update import PPOConfig
from resetsweep.sim.env import VecEnv
from resetsweep.tasks.base import TASKS, Task
from resetsweep.train import (
    Evaluator,
    RunSpec,
    auc,
    build_score,
    collect_rollout,
    run,
    score_capacity,
)
from resetsweep.types import Outcomes, StartStates, StepCounter

TASKS.load("resetsweep.tasks")

NUM_ENVS = 8
ROLLOUT_LEN = 8  # 64 train steps per iteration
PPO = PPOConfig(
    gamma=0.99,
    lam=0.95,
    epochs=1,
    minibatch_size=32,
    lr=1e-3,
    ent_coef=0.0,
    vf_coef=1.0,
    target_kl=None,
)

REQUIRED_KEYS = {
    "env_steps/total",
    "env_steps/train",
    "env_steps/walk",
    "env_steps/screen",
    "env_steps/scout",
    "eval/success",
    "eval/return",
    "train/curriculum_return",
    "ic/worst_group_success",
    "ic/never_succeeded_share",
    "ic/sampling_entropy",
    "first_reward_step",
    "ppo/approx_kl",
    "ppo/clip_frac",
    "ppo/explained_var",
    "ppo/entropy",
    "ppo/value_loss",
    "ppo/policy_loss",
    "ppo/kl_stopped",
    "ppo/n_minibatches",
    "perf/steps_per_s",
    "perf/wall_s",
}


def _spec(**overrides: Any) -> RunSpec:
    base: dict[str, Any] = {
        "task": "pendulum_sparse",
        "task_kw": {},
        "method": "uniform",
        "method_kw": {},
        "score": None,
        "score_kw": {},
        "seed": 0,
        "step_budget": 200,
        "num_envs": NUM_ENVS,
        "rollout_len": ROLLOUT_LEN,
        "ppo": PPO,
        "eval_every_steps": 100,
        "eval_n": 4,
        "device": "cpu",
    }
    base.update(overrides)
    return RunSpec(**base)


class _Recorder:
    def __init__(self) -> None:
        self.logs: list[tuple[int, dict[str, float]]] = []
        self.summaries: list[dict[str, float | str]] = []
        self.closed = False

    def log(self, step: int, data: dict[str, float]) -> None:
        self.logs.append((step, dict(data)))

    def summary(self, data: dict[str, float | str]) -> None:
        self.summaries.append(dict(data))

    def close(self) -> None:
        self.closed = True

    def logged_keys(self) -> set[str]:
        return {k for _, d in self.logs for k in d}


# --- RunSpec --------------------------------------------------------------------------------


def test_arm_id_ignores_seed_and_device() -> None:
    a = _spec(seed=0, device="cpu")
    b = _spec(seed=7, device="cuda:0")

    assert a.arm_id() == b.arm_id()


@pytest.mark.parametrize(
    "change",
    [
        {"method_kw": {"p_task": 0.2}},
        {"task": "pendulum_dense"},
        {"step_budget": 400},
        {"ppo": dataclasses.replace(PPO, lr=3e-4)},
        {"eval_seed": 1},
    ],
)
def test_arm_id_changes_with_any_other_field(change: dict[str, Any]) -> None:
    assert _spec(**change).arm_id() != _spec().arm_id()


def test_arm_id_does_not_depend_on_kw_order() -> None:
    a = _spec(method="walk", method_kw={"R": 2, "walk_steps": 5}, score="random")
    b = _spec(method="walk", method_kw={"walk_steps": 5, "R": 2}, score="random")

    assert a.arm_id() == b.arm_id()


def test_arm_id_accepts_read_only_mappings() -> None:
    kw = {"R": 2, "nested": {"a": [1, 2]}}
    proxied = MappingProxyType({"R": 2, "nested": MappingProxyType({"a": (1, 2)})})

    a = _spec(method="walk", method_kw=kw, score="random")
    b = _spec(method="walk", method_kw=proxied, score="random")

    assert a.arm_id() == b.arm_id()


def test_arm_id_rejects_values_that_are_not_plain_data() -> None:
    spec = _spec(method_kw={"device": torch.device("cpu")})

    with pytest.raises(TypeError, match=r"method_kw\.device.*torch\.device"):
        spec.arm_id()


def test_run_spec_rejects_minibatch_larger_than_rollout() -> None:
    with pytest.raises(ValueError, match="minibatch_size 128 exceeds"):
        _spec(ppo=dataclasses.replace(PPO, minibatch_size=128))


# --- score capacity -------------------------------------------------------------------------


def test_score_capacity_uses_population_size_for_population_methods() -> None:
    task: Task = TASKS.construct("pendulum_sparse")  # pyright: ignore[reportAssignmentType] # registry is untyped

    got = score_capacity("walk", task=task, num_envs=16, method_kw={"R": 2})

    assert got == population_size(16, R=2) == 8


def test_score_capacity_uses_table_capacity_for_own_success() -> None:
    task: Task = TASKS.construct("pendulum_sparse")  # pyright: ignore[reportAssignmentType] # registry is untyped

    got = score_capacity(
        "own_success", task=task, num_envs=16, method_kw={"max_trajectories": 3}
    )

    assert got == table_capacity(task, max_trajectories=3)


def test_score_capacity_uses_ic_capacity_for_slot_methods() -> None:
    task: Task = TASKS.construct("pendulum_sparse")  # pyright: ignore[reportAssignmentType] # registry is untyped

    got = score_capacity(
        "buffer_scored", task=task, num_envs=16, method_kw={"ic_capacity": 64}
    )

    assert got == 64


def test_score_capacity_rejects_a_method_without_scores() -> None:
    task: Task = TASKS.construct("pendulum_sparse")  # pyright: ignore[reportAssignmentType] # registry is untyped

    with pytest.raises(ValueError, match="'uniform' takes no score"):
        score_capacity("uniform", task=task, num_envs=16, method_kw={})


def _random_draws(**overrides: Any) -> Tensor:
    task: Task = TASKS.construct("pendulum_sparse")  # pyright: ignore[reportAssignmentType] # registry is untyped
    spec = _spec(method="walk", method_kw={"R": 2}, score="random", **overrides)
    score = build_score(spec, task)
    assert score is not None
    lp, _ = score.values(torch.arange(score.capacity))
    return lp


def test_random_score_draws_differ_across_run_seeds() -> None:
    assert not torch.equal(_random_draws(seed=0), _random_draws(seed=1))


def test_random_score_draws_repeat_for_the_same_run_seed() -> None:
    torch.testing.assert_close(
        _random_draws(seed=3), _random_draws(seed=3), rtol=0, atol=0
    )


def test_random_score_keeps_an_explicit_seed() -> None:
    a = _random_draws(seed=0, score_kw={"seed": 9})
    b = _random_draws(seed=1, score_kw={"seed": 9})

    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_inverted_random_score_is_seeded_from_the_run_seed() -> None:
    task: Task = TASKS.construct("pendulum_sparse")  # pyright: ignore[reportAssignmentType] # registry is untyped

    def draws(seed: int) -> Tensor:
        spec = _spec(
            method="walk",
            method_kw={"R": 2},
            score="inverted",
            score_kw={"inner": "random"},
            seed=seed,
        )
        score = build_score(spec, task)
        assert score is not None
        return score.values(torch.arange(score.capacity))[0]

    assert not torch.equal(draws(0), draws(1))
    torch.testing.assert_close(draws(2), draws(2), rtol=0, atol=0)


def test_score_seed_differs_from_env_generator_seed() -> None:
    # The env/method generator and torch.manual_seed use spec.seed itself; the random
    # score's draws must not repeat that stream.
    lp = _random_draws(seed=0)
    env_stream = torch.rand(lp.numel(), generator=torch.Generator().manual_seed(0))

    assert not torch.equal(lp, env_stream)


def test_seeding_does_not_change_arm_id() -> None:
    assert (
        _spec(score="random", seed=0).arm_id() == _spec(score="random", seed=1).arm_id()
    )


def test_build_score_is_none_for_scoreless_specs() -> None:
    task: Task = TASKS.construct("pendulum_sparse")  # pyright: ignore[reportAssignmentType] # registry is untyped

    assert build_score(_spec(), task) is None


def test_run_rejects_score_the_method_does_not_accept() -> None:
    with pytest.raises(ValueError, match="does not accept score 'slope'"):
        run(_spec(method="own_success", score="slope"), _Recorder())


# --- AUC ------------------------------------------------------------------------------------


def test_auc_is_trapezoid_over_steps_normalised_by_budget() -> None:
    # 0 -> 0.5 over [0, 100], 0.5 -> 1.0 over [100, 200]: 25 + 75 = 100 over a budget of 200.
    assert auc([0, 100, 200], [0.0, 0.5, 1.0], step_budget=200) == pytest.approx(0.5)


def test_auc_clips_the_last_point_at_the_budget() -> None:
    # An iteration may overshoot the budget; its eval point moves to x=200 with its value:
    # 0.5 * 200 * (0.0 + 1.0) = 100 over a budget of 200.
    assert auc([0, 300], [0.0, 1.0], step_budget=200) == pytest.approx(0.5)


# --- rollout alignment ----------------------------------------------------------------------


class _CountingMethod:
    """Every reset gives a fresh ic id; its version is 10x the id, so a misaligned stack of
    `episode_versions()` next to `ic_ids` is visible cell by cell."""

    name = "counting"

    def __init__(self, task: Task, num_envs: int) -> None:
        self._task = task
        self._next = 0
        self._ids = torch.full((num_envs,), -1, dtype=torch.long)

    def reset(
        self, env_ids: Tensor, outcomes: Outcomes | None
    ) -> tuple[StartStates, Tensor]:
        del outcomes
        k = env_ids.numel()
        ids = torch.arange(self._next, self._next + k)
        self._next += k
        self._ids[env_ids] = ids
        gen = torch.Generator().manual_seed(self._next)
        return self._task.sample_start(k, gen), ids

    def offer_visited(self, *args: Tensor) -> None:
        del args

    def episode_versions(self) -> Tensor:
        return self._ids * 10


def test_rollout_ic_versions_are_stacked_at_the_same_point_as_ic_ids() -> None:
    torch.manual_seed(0)
    task: Task = TASKS.construct("pendulum_sparse")  # pyright: ignore[reportAssignmentType] # registry is untyped
    env = VecEnv(
        task,
        num_envs=4,
        device="cpu",
        counter=StepCounter(budget=10**6),
        gen=torch.Generator().manual_seed(0),
    )
    method = _CountingMethod(task, 4)
    env.bind(method)
    obs = env.reset_all()
    model = ActorCritic(task.obs_dim, task.act_dim, hidden=(8,))
    # 150 steps > max_steps (100), so every env resets at least once inside the rollout.
    roll = collect_rollout(
        env, model, RunningNorm((task.obs_dim,)), obs, steps=150, versioned=method
    )

    assert roll.ic_versions is not None
    assert torch.equal(roll.ic_versions, roll.ic_ids * 10)
    assert int(roll.ic_ids.max()) >= 4  # resets happened inside the rollout
    # Cell 0 belongs to the episodes reset_all started (ids 0..3).
    assert roll.ic_ids[0].tolist() == [0, 1, 2, 3]


# --- eval -----------------------------------------------------------------------------------


def test_evaluator_plays_exactly_one_episode_of_max_steps() -> None:
    task: Task = TASKS.construct("pendulum_sparse")  # pyright: ignore[reportAssignmentType] # registry is untyped
    states = task.eval_states(3, 12345)
    evaluator = Evaluator(task, states, device="cpu", seed=12345)
    calls: list[int] = []

    def policy(obs: Tensor) -> Tensor:
        calls.append(obs.shape[0])
        return torch.zeros(obs.shape[0], task.act_dim)

    success, ret = evaluator.run(policy)

    assert calls == [3] * task.max_steps
    assert success == pytest.approx(
        0.0, abs=0
    )  # hanging with zero torque never succeeds
    assert ret == pytest.approx(0.0, abs=1e-6)  # no reward, no action cost


def test_eval_uses_fixed_states_same_across_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Tensor] = []
    original = train.FixedStarts.reset

    def spy(
        self: train.FixedStarts, env_ids: Tensor, outcomes: Outcomes | None
    ) -> tuple[StartStates, Tensor]:
        states, ids = original(self, env_ids, outcomes)
        if outcomes is None:  # reset_all of one eval
            seen.append(states.qpos.clone())
        return states, ids

    monkeypatch.setattr(train.FixedStarts, "reset", spy)
    task: Task = TASKS.construct("pendulum_sparse")  # pyright: ignore[reportAssignmentType] # registry is untyped
    expected = task.eval_states(4, 12345).qpos

    torch.manual_seed(0)
    run(_spec(seed=0), _Recorder())
    n_first = len(seen)
    torch.manual_seed(0)
    run(_spec(seed=1), _Recorder())

    assert n_first >= 2  # an eval at step 0 and at least one more
    assert len(seen) == 2 * n_first
    for qpos in seen:
        torch.testing.assert_close(qpos, expected, rtol=0, atol=0)


# --- whole runs -----------------------------------------------------------------------------


def test_run_stops_at_step_budget_including_extra_steps() -> None:
    torch.manual_seed(0)
    budget = 200
    spec = _spec(
        method="walk",
        method_kw={"R": 2, "update_every": 1, "walk_steps": 5, "keep_top": 0.5},
        score="random",
        step_budget=budget,
    )

    sink = _Recorder()
    result = run(spec, sink)

    extra = sum(result.env_steps_extra.values())
    total = result.env_steps_train + extra
    assert result.env_steps_extra["walk"] > 0
    assert total >= budget
    # Train steps alone stay below the budget: the walks were charged against it.
    assert result.env_steps_train < budget
    assert result.env_steps_train % (NUM_ENVS * ROLLOUT_LEN) == 0
    # Upper bound: the iteration before the last one ended below the budget, so the run
    # overshoots by at most one iteration (N*T train steps plus that iteration's extras).
    iteration_totals = [d["env_steps/total"] for s, d in sink.logs if s > 0]
    assert iteration_totals[-1] == total
    assert iteration_totals[-2] < budget
    last_extra = sink.logs[-1][1]["env_steps/walk"] - sink.logs[-2][1]["env_steps/walk"]
    assert total - budget < NUM_ENVS * ROLLOUT_LEN + last_extra


def test_run_result_auc_is_zero_when_never_successful() -> None:
    # 3 iterations on pendulum_sparse from hanging starts: the swing-up takes several
    # pumps at 1 N*m, far beyond what an untrained deterministic policy does.
    torch.manual_seed(0)

    result = run(_spec(), _Recorder())

    assert result.status == "ok"
    assert result.error is None
    assert result.auc == pytest.approx(0.0, abs=0)
    assert result.final_success == pytest.approx(0.0, abs=0)


def test_run_logs_every_required_metric() -> None:
    torch.manual_seed(0)
    sink = _Recorder()

    run(_spec(), sink)

    assert REQUIRED_KEYS <= sink.logged_keys()
    steps = [s for s, _ in sink.logs]
    assert steps == sorted(steps)
    assert steps[0] == 0  # the eval before training


def test_run_writes_summary_with_the_result() -> None:
    torch.manual_seed(0)
    sink = _Recorder()

    result = run(_spec(), sink)

    assert len(sink.summaries) == 1
    summary = sink.summaries[0]
    assert summary["auc"] == pytest.approx(result.auc)
    assert summary["status"] == "ok"
    assert summary["arm_id"] == result.arm_id
    assert not sink.closed  # the caller owns the sink


def test_run_logs_side_model_cost() -> None:
    torch.manual_seed(0)
    sink = _Recorder()
    spec = _spec(method="walk", method_kw={"R": 2}, score="success_classifier")

    run(spec, sink)

    assert "side/wall_ms_per_update" in sink.logged_keys()


def test_run_reports_ever_succeeded_share_for_ic_methods() -> None:
    torch.manual_seed(0)

    result = run(_spec(method="walk", method_kw={"R": 2}, score="random"), _Recorder())

    assert result.ever_succeeded_share == pytest.approx(0.0, abs=0)


def test_run_reports_nan_ever_succeeded_share_without_ics() -> None:
    torch.manual_seed(0)

    result = run(_spec(), _Recorder())

    assert math.isnan(result.ever_succeeded_share)
    assert result.reached_reward is False
    assert result.steps_to_first_reward is None


def test_run_records_first_reward_step_on_dense_task() -> None:
    torch.manual_seed(0)

    result = run(_spec(task="pendulum_dense"), _Recorder())

    assert result.reached_reward is True
    first = result.steps_to_first_reward
    assert first is not None
    # Counted in vector steps of NUM_ENVS; dense height reward appears within the run.
    assert first % NUM_ENVS == 0
    assert 0 < first <= result.env_steps_train


def test_run_logs_update_stats_and_diagnostics_that_share_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # uniform's diagnostics report buffer_size 0.0; make its update report 7.0 under the
    # same key. Both must reach the sink, neither overwriting the other.
    monkeypatch.setattr(Uniform, "update", lambda self, it, stats: {"buffer_size": 7.0})
    torch.manual_seed(0)
    sink = _Recorder()

    run(_spec(), sink)

    last = sink.logs[-1][1]
    assert last["method/buffer_size"] == pytest.approx(0.0, abs=0)
    assert last["method_update/buffer_size"] == pytest.approx(7.0, abs=0)


def test_run_propagates_exceptions() -> None:
    with pytest.raises(KeyError, match="unknown task 'nope'"):
        run(_spec(task="nope"), _Recorder())


# --- critic_value vmax ------------------------------------------------------------------------


def _critic_score(**overrides: Any) -> Any:
    task: Task = TASKS.construct("pendulum_sparse")  # pyright: ignore[reportAssignmentType] # registry is untyped
    spec = _spec(
        method="walk",
        method_kw={"R": 2},
        score="critic_value",
        score_kw={"temperature": 0.01},
        **overrides,
    )
    score = build_score(spec, task)
    assert score is not None
    return task, score


def test_critic_value_vmax_is_the_discounted_return_of_the_best_episode() -> None:
    task, score = _critic_score()

    gamma = PPO.gamma
    expected = 1.0 * (1 - gamma**task.max_steps) / (1 - gamma)
    assert score._vmax == pytest.approx(expected)  # pyright: ignore[reportAttributeAccessIssue] # private on purpose


def test_explicit_vmax_in_score_kw_is_kept() -> None:
    task: Task = TASKS.construct("pendulum_sparse")  # pyright: ignore[reportAssignmentType] # registry is untyped
    spec = _spec(
        method="walk",
        method_kw={"R": 2},
        score="critic_value",
        score_kw={"temperature": 0.01, "vmax": 7.5},
    )
    score = build_score(spec, task)
    assert score is not None
    assert score._vmax == 7.5  # pyright: ignore[reportAttributeAccessIssue] # private on purpose


def test_inverted_critic_value_gets_vmax_for_its_inner_score() -> None:
    task: Task = TASKS.construct("pendulum_sparse")  # pyright: ignore[reportAssignmentType] # registry is untyped
    spec = _spec(
        method="walk",
        method_kw={"R": 2},
        score="inverted",
        score_kw={"inner": "critic_value", "inner_kw": {"temperature": 0.01}},
    )
    assert build_score(spec, task) is not None


def test_vmax_is_rejected_for_a_task_with_no_positive_reward() -> None:
    task: Task = TASKS.construct("maze_dense")  # pyright: ignore[reportAssignmentType] # registry is untyped
    spec = _spec(
        task="maze_dense",
        method="walk",
        method_kw={"R": 2},
        score="critic_value",
        score_kw={"temperature": 0.01},
    )
    with pytest.raises(ValueError, match="reward_max"):
        build_score(spec, task)
