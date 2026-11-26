"""Contract every registered task must meet. Parametrised over the registry, so a new
task file in `resetsweep.tasks` is picked up with no edit here."""

import pytest
import torch

from resetsweep.sim.batch import BatchSim
from resetsweep.tasks.base import TASKS, Task

TASKS.load("resetsweep.tasks")

N = 6


def _make(name: str) -> Task:
    return TASKS.construct(name)  # pyright: ignore[reportReturnType] # registry is untyped


def _sim_for(task: Task, n: int = N) -> BatchSim:
    return BatchSim(task.mjcf, nworld=n, device="cpu", frame_skip=task.frame_skip)


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_registry_has_tasks() -> None:
    assert {"pendulum_sparse", "pendulum_dense"} <= set(TASKS.names())


@pytest.mark.parametrize("name", TASKS.names())
def test_observation_has_shape_n_by_obs_dim(name: str) -> None:
    task = _make(name)
    sim = _sim_for(task)
    starts = task.sample_start(N, _gen())
    sim.set_state(torch.arange(N), starts.qpos, starts.qvel)

    obs = task.observe(sim, starts.extra)

    assert obs.shape == (N, task.obs_dim)


@pytest.mark.parametrize("name", TASKS.names())
def test_sample_start_states_are_valid(name: str) -> None:
    task = _make(name)

    starts = task.sample_start(64, _gen())

    assert starts.n == 64
    assert starts.extra.shape == (64, task.extra_dim)
    assert bool(task.is_valid(starts).all())


@pytest.mark.parametrize("name", TASKS.names())
def test_is_valid_rejects_nan_and_out_of_box_states(name: str) -> None:
    task = _make(name)
    good = task.sample_start(2, _gen())
    nan_row = torch.full_like(good.qpos[:1], float("nan"))
    far_row = good.qpos[:1] + 1e3
    bad = type(good)(
        torch.cat([nan_row, far_row]),
        torch.cat([good.qvel[:1], good.qvel[:1]]),
        torch.cat([good.extra[:1], good.extra[:1]]),
    )

    assert task.is_valid(bad).tolist() == [False, False]


@pytest.mark.parametrize("name", TASKS.names())
def test_eval_states_identical_across_calls_and_global_rng(name: str) -> None:
    task = _make(name)
    torch.manual_seed(1)
    first = task.eval_states(8, seed=123)
    torch.manual_seed(2)
    second = task.eval_states(8, seed=123)

    assert torch.equal(first.qpos, second.qpos)
    assert torch.equal(first.qvel, second.qvel)
    assert torch.equal(first.extra, second.extra)


@pytest.mark.parametrize("name", TASKS.names())
def test_eval_states_leaves_global_rng_untouched(name: str) -> None:
    task = _make(name)
    torch.manual_seed(7)
    expected = torch.rand(3)
    torch.manual_seed(7)

    task.eval_states(8, seed=5)

    assert torch.equal(torch.rand(3), expected)


@pytest.mark.parametrize("name", TASKS.names())
def test_eval_states_differ_across_seeds(name: str) -> None:
    task = _make(name)

    a = task.eval_states(8, seed=1)
    b = task.eval_states(8, seed=2)

    assert not torch.equal(a.qpos, b.qpos)


@pytest.mark.parametrize("name", TASKS.names())
def test_canonical_start_is_one_valid_state(name: str) -> None:
    task = _make(name)

    start = task.canonical_start()

    assert start.n == 1
    assert bool(task.is_valid(start).all())


@pytest.mark.parametrize("name", TASKS.names())
def test_reward_and_success_have_shape_n(name: str) -> None:
    task = _make(name)
    sim = _sim_for(task)
    starts = task.sample_start(N, _gen())
    sim.set_state(torch.arange(N), starts.qpos, starts.qvel)
    action = torch.zeros(N, task.act_dim)

    reward = task.reward(sim, starts.extra, action)
    success = task.success_now(sim, starts.extra)
    terminated = task.terminated(sim, starts.extra)

    assert reward.shape == (N,)
    assert success.shape == (N,)
    assert success.dtype == torch.bool
    assert terminated.shape == (N,)
    assert terminated.dtype == torch.bool


@pytest.mark.parametrize("name", TASKS.names())
def test_state_features_have_one_row_per_state(name: str) -> None:
    task = _make(name)
    starts = task.sample_start(N, _gen())

    feats = task.state_features(starts.qpos, starts.qvel, starts.extra)

    assert feats.ndim == 2
    assert feats.shape[0] == N


@pytest.mark.parametrize("name", TASKS.names())
def test_start_box_has_nq_plus_nv_dimensions(name: str) -> None:
    task = _make(name)
    sim = _sim_for(task, 1)
    dims = sim.qpos.shape[1] + sim.qvel.shape[1]

    assert task.start_box.low.shape == (dims,)
    assert task.start_box.high.shape == (dims,)
    assert bool((task.start_box.low < task.start_box.high).all())


@pytest.mark.parametrize("name", TASKS.names())
def test_dense_sibling_is_a_registered_task(name: str) -> None:
    task = _make(name)

    assert task.dense_sibling in TASKS.names()
    assert task.name == name


@pytest.mark.parametrize("name", TASKS.names())
def test_start_states_extra_width_matches_extra_dim(name: str) -> None:
    task = _make(name)

    sampled = task.sample_start(N, _gen())
    canonical = task.canonical_start()
    evaluated = task.eval_states(N, seed=0)

    assert sampled.extra.shape == (N, task.extra_dim)
    assert canonical.extra.shape == (1, task.extra_dim)
    assert evaluated.extra.shape == (N, task.extra_dim)
