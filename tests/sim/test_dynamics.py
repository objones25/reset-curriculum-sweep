import torch

from resetsweep.sim.batch import BatchSim
from resetsweep.sim.dynamics import step_task

NWORLD = 3


class _PlainTask:
    pass


class _PushingTask:
    """Applies a constant generalized force of 0.3 to the one joint."""

    def applied_force(self, sim: BatchSim, extra: torch.Tensor) -> torch.Tensor:
        return torch.full((sim.qpos.shape[0], 1), 0.3)


def _sim(mjcf: str) -> BatchSim:
    return BatchSim(mjcf, nworld=NWORLD, device="cpu", frame_skip=5)


def test_step_task_without_a_force_hook_is_a_plain_step(pendulum_mjcf: str) -> None:
    via_task, plain = _sim(pendulum_mjcf), _sim(pendulum_mjcf)
    action = torch.full((NWORLD, 1), 0.2)

    step_task(_PlainTask(), via_task, torch.zeros(NWORLD, 0), action)  # pyright: ignore[reportArgumentType] # minimal stand-in task
    plain.step(action)

    assert torch.equal(via_task.qvel, plain.qvel)


def test_step_task_applies_the_task_force(pendulum_mjcf: str) -> None:
    pushed, driven = _sim(pendulum_mjcf), _sim(pendulum_mjcf)

    step_task(_PushingTask(), pushed, torch.zeros(NWORLD, 0), torch.zeros(NWORLD, 1))  # pyright: ignore[reportArgumentType] # minimal stand-in task
    driven.step(torch.full((NWORLD, 1), 0.3))  # the motor has gear 1

    assert torch.allclose(pushed.qvel, driven.qvel, atol=1e-5)
