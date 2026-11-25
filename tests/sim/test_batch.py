from collections.abc import Callable

import pytest
import torch
import warp as wp

from resetsweep.sim.batch import BatchSim
from resetsweep.types import BudgetExceeded, StepCounter

NWORLD = 4
FRAME_SKIP = 5


def _make(mjcf: str, device: str = "cpu") -> BatchSim:
    return BatchSim(mjcf, nworld=NWORLD, device=device, frame_skip=FRAME_SKIP)


def test_set_then_get_state_round_trips(pendulum_mjcf: str) -> None:
    sim = _make(pendulum_mjcf)
    others = torch.tensor([0, 2])
    sim.set_state(others, torch.tensor([[0.1], [0.2]]), torch.tensor([[0.3], [0.4]]))
    before_q, before_v = sim.get_state(torch.arange(NWORLD))
    idx = torch.tensor([1, 3])
    q = torch.tensor([[0.5], [-0.25]])
    v = torch.tensor([[1.0], [2.0]])

    sim.set_state(idx, q, v)

    got_q, got_v = sim.get_state(idx)
    assert torch.equal(got_q, q)
    assert torch.equal(got_v, v)
    other = torch.tensor([0, 2])
    kept_q, kept_v = sim.get_state(other)
    assert torch.equal(kept_q, before_q[other])
    assert torch.equal(kept_v, before_v[other])


def test_get_state_returns_a_copy(pendulum_mjcf: str) -> None:
    sim = _make(pendulum_mjcf)
    q, _ = sim.get_state(torch.tensor([0]))
    q += 5.0
    assert sim.qpos[0, 0].item() == 0.0


def test_restored_state_gives_same_trajectory_in_two_worlds(pendulum_mjcf: str) -> None:
    sim = _make(pendulum_mjcf)
    idx = torch.tensor([1, 2])
    sim.set_state(idx, torch.full((2, 1), 0.7), torch.full((2, 1), -0.3))
    ctrl = torch.full((NWORLD, 1), 0.2)

    for _ in range(50):
        sim.step(ctrl)

    assert torch.allclose(sim.qpos[1], sim.qpos[2], atol=1e-5)


@pytest.mark.gpu
def test_restored_state_agrees_across_worlds_on_gpu(
    pendulum_mjcf: str, record_property: Callable[[str, object], None]
) -> None:
    sim = _make(pendulum_mjcf, device="cuda:0")
    idx = torch.tensor([1, 2], device=sim.qpos.device)
    device = sim.qpos.device
    sim.set_state(
        idx,
        torch.full((2, 1), 0.7, device=device),
        torch.full((2, 1), -0.3, device=device),
    )
    ctrl = torch.full((NWORLD, 1), 0.2, device=sim.qpos.device)

    for _ in range(50):
        sim.step(ctrl)

    deviation_max = (sim.qpos[1] - sim.qpos[2]).abs().max().item()
    record_property("max_deviation", deviation_max)
    assert torch.allclose(sim.qpos[1], sim.qpos[2], atol=1e-3)


def test_step_advances_time_by_frame_skip_times_timestep(
    pendulum_mjcf: str, pendulum_timestep_s: float
) -> None:
    sim = _make(pendulum_mjcf)
    sim.step(torch.zeros((NWORLD, 1)))
    sim.step(torch.zeros((NWORLD, 1)))
    time_s = wp.to_torch(sim.data.time)
    expected = 2 * FRAME_SKIP * pendulum_timestep_s
    assert torch.allclose(time_s, torch.full((NWORLD,), expected), atol=1e-6)


def test_views_have_expected_shapes(pendulum_mjcf: str) -> None:
    sim = _make(pendulum_mjcf)
    assert sim.qpos.shape == (NWORLD, 1)
    assert sim.qvel.shape == (NWORLD, 1)
    assert sim.ctrl.shape == (NWORLD, 1)
    assert sim.xpos.shape == (NWORLD, 2, 3)
    assert sim.site_xpos.shape == (NWORLD, 1, 3)


def test_set_state_refreshes_derived_site_position(pendulum_mjcf: str) -> None:
    sim = _make(pendulum_mjcf)
    sim.set_state(
        torch.tensor([1]), torch.tensor([[torch.pi / 2]]), torch.zeros((1, 1))
    )
    # Hanging down the tip is at z=0.5; rotated 90 deg about y it is at x=-0.5 or +0.5, z=1.
    tip = sim.site_xpos[1, 0]
    assert tip[2].item() == pytest.approx(1.0, abs=1e-5)
    assert abs(tip[0].item()) == pytest.approx(0.5, abs=1e-5)


def test_step_counter_raises_budget_exceeded_past_budget() -> None:
    counter = StepCounter(budget=10)
    counter.add_train(10)
    with pytest.raises(BudgetExceeded, match="11 > 10"):
        counter.add_extra(1, "walk")


def test_step_counter_splits_train_and_extra() -> None:
    counter = StepCounter(budget=100)
    counter.add_train(30)
    counter.add_extra(5, "walk")
    counter.add_extra(7, "screen")
    counter.add_extra(2, "walk")
    assert counter.train == 30
    assert counter.extra == {"walk": 7, "screen": 7, "scout": 0}
    assert counter.total == 44


@pytest.mark.parametrize("shape", [(1,), (1, 1), (NWORLD,), (NWORLD, 2)])
def test_step_rejects_broadcastable_ctrl_shape(
    pendulum_mjcf: str, shape: tuple[int, ...]
) -> None:
    sim = _make(pendulum_mjcf)
    with pytest.raises(ValueError, match=r"ctrl shape .* expected"):
        sim.step(torch.zeros(shape))


def test_set_state_rejects_row_count_mismatch(pendulum_mjcf: str) -> None:
    sim = _make(pendulum_mjcf)
    with pytest.raises(ValueError, match="row count"):
        sim.set_state(torch.tensor([1, 3]), torch.zeros((2, 1)), torch.zeros((3, 1)))
    with pytest.raises(ValueError, match="row count"):
        sim.set_state(torch.tensor([1, 3]), torch.zeros((1, 1)), torch.zeros((1, 1)))
