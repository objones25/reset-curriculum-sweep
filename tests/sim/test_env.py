from collections.abc import Callable
from dataclasses import dataclass, field
from math import pi

import pytest
import torch
from torch import Tensor

from resetsweep.sim.env import VecEnv
from resetsweep.tasks.base import Task
from resetsweep.types import (
    BudgetExceeded,
    InvalidStartState,
    Outcomes,
    StartStates,
    StepCounter,
)

NUM_ENVS = 8


@dataclass
class StubMethod:
    """Reset rule that returns `start(env_ids)` and records everything it is told."""

    start: Callable[[Tensor], StartStates]
    name: str = "stub_method"
    reset_calls: list[tuple[Tensor, Outcomes | None]] = field(default_factory=list)
    visits: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]] = field(
        default_factory=list
    )

    def reset(
        self, env_ids: Tensor, outcomes: Outcomes | None
    ) -> tuple[StartStates, Tensor]:
        self.reset_calls.append((env_ids.clone(), outcomes))
        return self.start(env_ids), 100 + env_ids

    def offer_visited(
        self,
        env_ids: Tensor,
        qpos: Tensor,
        qvel: Tensor,
        extra: Tensor,
        ep_step: Tensor,
    ) -> None:
        self.visits.append((env_ids, qpos, qvel, extra, ep_step))


def _states(qpos: list[float]) -> Callable[[Tensor], StartStates]:
    """Start state per env id: `qpos[env_id % len(qpos)]`, zero velocity."""

    def start(env_ids: Tensor) -> StartStates:
        q = torch.tensor(qpos)[env_ids % len(qpos)].unsqueeze(1)
        return StartStates(q, torch.zeros_like(q), torch.zeros(len(env_ids), 0))

    return start


def _env(
    task: Task,
    method: StubMethod | None = None,
    *,
    num_envs: int = NUM_ENVS,
    budget: int = 10_000,
    visit_every: int = 8,
) -> tuple[VecEnv, StubMethod, StepCounter]:
    counter = StepCounter(budget=budget)
    env = VecEnv(
        task,
        num_envs=num_envs,
        device="cpu",
        counter=counter,
        gen=torch.Generator().manual_seed(0),
        visit_every=visit_every,
    )
    method = method or StubMethod(_states([pi]))
    env.bind(method)
    return env, method, counter


def _zeros(n: int = NUM_ENVS) -> Tensor:
    return torch.zeros(n, 1)


def _collect(env: VecEnv, steps: int, t0: int = 0) -> list[tuple[int, Outcomes]]:
    """Step `steps` times; return (steps_completed, outcomes) for steps that finished envs."""
    got = []
    for t in range(t0 + 1, t0 + steps + 1):
        out = env.step(_zeros(env.num_envs))
        if out.outcomes is not None:
            got.append((t, out.outcomes))
    return got


def test_episode_truncates_at_max_steps_not_terminates(
    make_task: Callable[..., Task],
) -> None:
    env, _, _ = _env(make_task(max_steps=4))
    env.reset_all()
    env.ep_step.fill_(0)  # remove the staggered offsets for this test

    outs = [env.step(_zeros()) for _ in range(4)]
    flags = [(o.terminated.any().item(), o.truncated.all().item()) for o in outs]

    assert flags == [(False, False), (False, False), (False, False), (False, True)]


@pytest.mark.parametrize("max_steps", [3, 10])
def test_termination_at_step_3_is_not_truncation(
    make_task: Callable[..., Task], max_steps: int
) -> None:
    env, _, _ = _env(
        make_task(max_steps=max_steps, terminate_script=[False, False, True])
    )
    env.reset_all()
    env.ep_step.fill_(0)

    outs = [env.step(_zeros()) for _ in range(3)]

    assert outs[2].terminated.all().item() is True
    assert outs[2].truncated.any().item() is False
    assert outs[2].outcomes is not None
    assert outs[2].outcomes.ep_len.tolist() == [3] * NUM_ENVS


def test_final_obs_is_pre_reset_observation(make_task: Callable[..., Task]) -> None:
    reset_angle = 0.5
    env, method, _ = _env(make_task(max_steps=2))
    env.reset_all()  # starts at pi
    env.ep_step.fill_(0)
    method.start = _states([reset_angle])  # the reset after truncation starts elsewhere

    env.step(_zeros())
    out = env.step(_zeros())

    assert out.truncated.all().item() is True
    assert torch.allclose(out.obs, torch.tensor([[reset_angle, 0.0]] * NUM_ENVS))
    assert torch.allclose(out.final_obs[:, 0], torch.full((NUM_ENVS,), pi), atol=0.05)


@pytest.mark.parametrize(
    ("script", "expect_success"),
    [
        ([True, True, True, False, False, False], True),
        ([True, True, False, True, True, False], False),
        ([False, True, True, True, False, False], True),
        ([True, True, False, True, True, True], True),
    ],
)
def test_success_requires_hold_steps_consecutive(
    make_task: Callable[..., Task], script: list[bool], expect_success: bool
) -> None:
    env, _, _ = _env(make_task(max_steps=6, hold_steps=3, success_script=script))
    env.reset_all()
    env.ep_step.fill_(0)

    outs = [env.step(_zeros()).outcomes for _ in range(6)]

    final = outs[5]
    assert final is not None
    assert final.success.tolist() == [expect_success] * NUM_ENVS


def test_first_episodes_have_random_clocks_and_full_false(
    make_task: Callable[..., Task],
) -> None:
    max_steps = 6
    env, _, _ = _env(make_task(max_steps=max_steps))

    env.reset_all()

    clocks = env.ep_step.clone()
    assert int(clocks.min()) >= 0
    assert int(clocks.max()) < max_steps
    assert len(set(clocks.tolist())) > 1
    first = _collect(env, max_steps)
    second = _collect(env, max_steps, t0=max_steps)
    first_ids = torch.cat([o.env_ids for _, o in first]).sort().values
    assert first_ids.tolist() == list(range(NUM_ENVS))  # each env finishes exactly once
    assert not torch.cat([o.full for _, o in first]).any()
    assert torch.cat([o.full for _, o in second]).all()
    assert torch.cat([o.ep_len for _, o in first]).tolist() == [max_steps] * NUM_ENVS


def test_outcomes_report_env_ids_ic_ids_returns(make_task: Callable[..., Task]) -> None:
    # env 0 starts upright (reward 1 per step), env 1 hangs down (never rewarded)
    method = StubMethod(_states([pi, 0.0]))
    env, _, _ = _env(make_task(max_steps=3, hold_steps=2), method, num_envs=2)
    env.reset_all()
    zeros2 = torch.zeros(2, 1)
    for _ in range(3):  # first (staggered) wave
        env.step(zeros2)

    waves = []
    for t in range(
        4, 7
    ):  # second wave: every env finishes exactly once, with full=True
        out = env.step(zeros2)
        if out.outcomes is not None:
            waves.append((t, out.outcomes))

    oc = Outcomes.cat([o for _, o in waves])
    order = oc.env_ids.argsort()
    assert oc.env_ids[order].tolist() == [0, 1]
    assert oc.ic_ids[order].tolist() == [100, 101]
    assert oc.ret[order].tolist() == [3.0, 0.0]
    assert oc.success[order].tolist() == [True, False]
    assert oc.reached_reward[order].tolist() == [True, False]
    assert oc.ep_len[order].tolist() == [3, 3]
    assert oc.full[order].tolist() == [True, True]
    finish_t = torch.cat([torch.full((o.env_ids.numel(),), t) for t, o in waves])
    assert oc.start_step.tolist() == (finish_t - oc.ep_len).tolist()
    # the method was told about exactly these finished episodes on its reset calls
    told = torch.cat([o.env_ids for _, o in method.reset_calls[1:] if o is not None])
    assert sorted(told.tolist()) == sorted([0, 1] * 2)


def test_outcomes_report_first_reward_step_or_minus_one(
    make_task: Callable[..., Task],
) -> None:
    # env 0 starts upright (rewarded from its first step), env 1 hangs down (never)
    method = StubMethod(_states([pi, 0.0]))
    env, _, _ = _env(make_task(max_steps=3, hold_steps=2), method, num_envs=2)
    env.reset_all()
    _collect(env, 3)  # first (staggered) wave

    oc = Outcomes.cat([o for _, o in _collect(env, 3, t0=3)])

    order = oc.env_ids.argsort()
    assert oc.first_reward_step[order].tolist() == [1, -1]


def test_reset_all_gives_method_no_outcomes(make_task: Callable[..., Task]) -> None:
    env, method, _ = _env(make_task())

    env.reset_all()

    ids, outcomes = method.reset_calls[0]
    assert ids.tolist() == list(range(NUM_ENVS))
    assert outcomes is None


def test_invalid_start_state_raises_naming_method(
    make_task: Callable[..., Task],
) -> None:
    def nan_start(env_ids: Tensor) -> StartStates:
        n = len(env_ids)
        return StartStates(
            torch.full((n, 1), float("nan")), torch.zeros(n, 1), torch.zeros(n, 0)
        )

    env, _, _ = _env(make_task(), StubMethod(nan_start, name="bad_method"))

    with pytest.raises(InvalidStartState, match="bad_method") as excinfo:
        env.reset_all()

    assert excinfo.value.method_name == "bad_method"
    assert excinfo.value.row == 0


def test_start_state_outside_start_box_raises_with_row(
    make_task: Callable[..., Task],
) -> None:
    env, _, _ = _env(
        make_task(), StubMethod(_states([pi, 9.0]), name="wild"), num_envs=2
    )

    with pytest.raises(InvalidStartState, match="wild") as excinfo:
        env.reset_all()

    assert excinfo.value.row == 1


def test_invalid_start_on_later_reset_raises(make_task: Callable[..., Task]) -> None:
    method = StubMethod(_states([pi]), name="late_bad")
    env, _, _ = _env(make_task(max_steps=2), method)
    env.reset_all()
    env.ep_step.fill_(0)
    method.start = _states([float("nan")])

    env.step(_zeros())
    with pytest.raises(InvalidStartState, match="late_bad"):
        env.step(_zeros())


def test_counter_charges_num_envs_per_step(make_task: Callable[..., Task]) -> None:
    env, _, counter = _env(make_task(max_steps=100))
    env.reset_all()

    for _ in range(3):
        env.step(_zeros())

    assert counter.train == 3 * NUM_ENVS
    assert counter.extra == {"walk": 0, "screen": 0, "scout": 0}


def test_step_over_budget_raises(make_task: Callable[..., Task]) -> None:
    env, _, _ = _env(make_task(max_steps=100), budget=2 * NUM_ENVS)
    env.reset_all()
    env.step(_zeros())
    env.step(_zeros())

    with pytest.raises(BudgetExceeded):
        env.step(_zeros())


def test_offer_visited_every_visit_every_steps(make_task: Callable[..., Task]) -> None:
    env, method, _ = _env(make_task(max_steps=100), visit_every=2)
    env.reset_all()
    env.ep_step.fill_(0)

    for _ in range(5):
        env.step(_zeros())

    assert len(method.visits) == 2
    env_ids, qpos, qvel, extra, ep_step = method.visits[1]
    assert env_ids.tolist() == list(range(NUM_ENVS))
    assert qpos.shape == (NUM_ENVS, 1)
    assert qvel.shape == (NUM_ENVS, 1)
    assert extra.shape == (NUM_ENVS, 0)
    assert ep_step.tolist() == [4] * NUM_ENVS


def test_step_before_bind_raises(make_task: Callable[..., Task]) -> None:
    env = VecEnv(
        make_task(),
        num_envs=2,
        device="cpu",
        counter=StepCounter(budget=100),
        gen=torch.Generator().manual_seed(0),
    )

    with pytest.raises(RuntimeError, match="bind"):
        env.reset_all()


def test_step_rejects_wrong_action_shape(make_task: Callable[..., Task]) -> None:
    env, _, _ = _env(make_task())
    env.reset_all()

    with pytest.raises(ValueError, match="action"):
        env.step(torch.zeros(NUM_ENVS, 2))


def test_final_obs_survives_reset_when_observe_returns_view(
    make_task: Callable[..., Task],
) -> None:
    task = make_task(max_steps=2)
    task.observe = lambda sim, extra: sim.qpos[:, :1]  # a view of sim state, not a copy
    env, method, _ = _env(task)
    env.reset_all()
    env.ep_step.fill_(0)
    method.start = _states([0.5])

    env.step(_zeros())
    out = env.step(_zeros())

    assert out.truncated.all().item() is True
    assert torch.allclose(out.obs, torch.full((NUM_ENVS, 1), 0.5))
    assert torch.allclose(out.final_obs, torch.full((NUM_ENVS, 1), pi), atol=0.05)


def test_hold_and_success_reset_between_episodes(
    make_task: Callable[..., Task],
) -> None:
    # episode 1 ends with 2 success steps, episode 2 opens with 1: a leaked hold counter
    # would reach hold_steps=3 on episode 2's first step.
    script = [False, False, True, True, True, False, False, False]
    env, _, _ = _env(make_task(max_steps=4, hold_steps=3, success_script=script))
    env.reset_all()
    env.ep_step.fill_(0)

    outs = [env.step(_zeros()).outcomes for _ in range(8)]

    first, second = outs[3], outs[7]
    assert first is not None
    assert second is not None
    assert first.success.tolist() == [False] * NUM_ENVS
    assert second.success.tolist() == [False] * NUM_ENVS


def test_rejects_generator_on_wrong_device(make_task: Callable[..., Task]) -> None:
    # A CPU generator with a "cuda" env must fail in __init__, before any CUDA work, so this
    # runs on CPU-only machines (a meta-device Generator cannot be created).
    with pytest.raises(ValueError, match="generator"):
        VecEnv(
            make_task(),
            num_envs=2,
            device="cuda",
            counter=StepCounter(budget=100),
            gen=torch.Generator(device="cpu"),
        )


def test_task_applied_force_hook_acts_on_the_sim(
    make_task: Callable[..., Task],
) -> None:
    task = make_task()
    task.applied_force = lambda sim, extra: torch.full_like(sim.qvel, 3.0)  # pyright: ignore[reportAttributeAccessIssue] # optional hook, set on the instance
    pushed, _, _ = _env(task)
    plain, _, _ = _env(make_task())
    pushed.reset_all()
    plain.reset_all()

    pushed.step(_zeros())
    plain.step(_zeros())

    assert not torch.allclose(pushed.sim.qvel, plain.sim.qvel)


def test_task_init_extra_hook_overrides_extra_the_method_supplied(
    make_task: Callable[..., Task],
) -> None:
    task = make_task(extra_dim=1)
    seen: list[tuple[int, torch.Generator]] = []

    def init_extra(states: StartStates, gen: torch.Generator) -> Tensor:
        seen.append((states.n, gen))
        return torch.full((states.n, 1), 7.0)

    task.init_extra = init_extra  # pyright: ignore[reportAttributeAccessIssue] # optional hook, set on the instance

    def start(env_ids: Tensor) -> StartStates:
        q = torch.full((len(env_ids), 1), pi)
        return StartStates(q, torch.zeros_like(q), torch.zeros(len(env_ids), 1))

    env, _, _ = _env(task, StubMethod(start))

    env.reset_all()

    assert env.extra.tolist() == [[7.0]] * NUM_ENVS
    assert seen[0][0] == NUM_ENVS
    assert seen[0][1] is env._gen  # pyright: ignore[reportPrivateUsage] # the env's own generator, so draws are reproducible


def test_init_extra_hook_with_wrong_shape_is_rejected(
    make_task: Callable[..., Task],
) -> None:
    task = make_task(extra_dim=1)
    task.init_extra = lambda states, gen: torch.zeros(states.n, 3)  # pyright: ignore[reportAttributeAccessIssue] # optional hook

    def start(env_ids: Tensor) -> StartStates:
        q = torch.full((len(env_ids), 1), pi)
        return StartStates(q, torch.zeros_like(q), torch.zeros(len(env_ids), 1))

    env, _, _ = _env(task, StubMethod(start))

    with pytest.raises(ValueError, match="init_extra"):
        env.reset_all()
