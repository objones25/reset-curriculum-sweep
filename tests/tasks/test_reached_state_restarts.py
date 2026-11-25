"""End to end: methods that restart from reached states, driven through `VecEnv` by a
scripted swing-up, must hand the env only valid (canonical) start states."""

import pytest
import torch
from controllers import PendulumSwingup

from resetsweep.methods.base import METHODS, MethodContext
from resetsweep.methods.own_success import OwnSuccess
from resetsweep.methods.phased import Phased
from resetsweep.sim.env import VecEnv
from resetsweep.sim.rollouts import Rollouts
from resetsweep.tasks.base import TASKS, Task
from resetsweep.tasks.pendulum import TORQUE_LIMIT_NM
from resetsweep.types import StepCounter

TASKS.load("resetsweep.tasks")
METHODS.load("resetsweep.methods")

NUM_ENVS = 16


def _restarted_from_stored(method: OwnSuccess | Phased) -> bool:
    """Some env's current episode started from a stored own_success state."""
    tags = method.episode_versions()
    if isinstance(method, Phased):
        return method.phase == 1 and bool(((tags >= 0) & (tags % 2 == 1)).any())
    return bool((tags >= 0).any())


@pytest.mark.parametrize("name", ["own_success", "phased"])
def test_swingup_capture_and_restart_from_stored_states(name: str) -> None:
    task: Task = TASKS.construct("pendulum_sparse")  # pyright: ignore[reportAssignmentType] # registry is untyped
    gen = torch.Generator().manual_seed(0)
    counter = StepCounter(budget=10 * NUM_ENVS * task.max_steps)
    rollouts = Rollouts(task, device="cpu", counter=counter, gen=gen, nworld_max=16)
    ctx = MethodContext(
        task=task,
        num_envs=NUM_ENVS,
        device="cpu",
        gen=gen,
        score=None,
        rollouts=rollouts,
        counter=counter,
        p_task=0.5,
    )
    method = METHODS.construct(name, ctx)
    assert isinstance(method, OwnSuccess | Phased)
    env = VecEnv(task, num_envs=NUM_ENVS, device="cpu", counter=counter, gen=gen)
    env.bind(method)
    controller = PendulumSwingup(task.mjcf, TORQUE_LIMIT_NM)
    obs = env.reset_all()
    restarted = False

    for _ in range(4 * task.max_steps):  # raises InvalidStartState on a bad restart
        obs = env.step(controller(obs)).obs
        restarted = restarted or _restarted_from_stored(method)

    assert method.diagnostics()["trajectories"] >= 1.0
    assert restarted
