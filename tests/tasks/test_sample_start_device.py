"""`sample_start` draws on the generator's device and returns tensors there.

train.py builds `torch.Generator(device="cuda:0")`; torch refuses a non-CPU generator in a
draw that does not name the same device ("Expected a 'cpu' device type for generator").
On a Mac, an MPS generator raises the same error, so the CPU suite catches the bug too.
"""

import pytest
import torch

from resetsweep.methods.base import P_TASK_DEFAULT, MethodContext
from resetsweep.methods.point import Point
from resetsweep.sim.rollouts import Rollouts
from resetsweep.tasks.base import TASKS, Task
from resetsweep.types import StartStates, StepCounter

TASKS.load("resetsweep.tasks")

N = 5

_ACCELERATORS = [
    pytest.param(
        "mps",
        marks=pytest.mark.skipif(
            not torch.backends.mps.is_available(), reason="needs MPS"
        ),
    ),
]


def _make(name: str) -> Task:
    return TASKS.construct(name)  # pyright: ignore[reportReturnType] # registry is untyped


def _assert_on(states: StartStates, device: torch.device) -> None:
    assert states.qpos.device.type == device.type
    assert states.qvel.device.type == device.type
    assert states.extra.device.type == device.type


@pytest.mark.parametrize("device", ["cpu", *_ACCELERATORS])
@pytest.mark.parametrize("name", TASKS.names())
def test_sample_start_returns_tensors_on_generator_device(
    name: str, device: str
) -> None:
    gen = torch.Generator(device=device).manual_seed(0)

    states = _make(name).sample_start(N, gen)

    assert states.n == N
    _assert_on(states, gen.device)


@pytest.mark.parametrize("name", TASKS.names())
def test_sample_start_on_cpu_generator_stays_valid(name: str) -> None:
    task = _make(name)
    states = task.sample_start(64, torch.Generator().manual_seed(0))

    assert bool(task.is_valid(states).all())


def _point_ctx(task: Task, device: str) -> MethodContext:
    gen = torch.Generator(device=device).manual_seed(0)
    counter = StepCounter(budget=10**9)
    rollouts = Rollouts(task, device=device, counter=counter, gen=gen, nworld_max=8)
    return MethodContext(
        task=task,
        num_envs=4,
        device=device,
        gen=gen,
        score=None,
        rollouts=rollouts,
        counter=counter,
        p_task=P_TASK_DEFAULT,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("name", TASKS.names())
def test_g1_sample_start_and_random_point_on_cuda(name: str) -> None:
    """G1: a CUDA generator gives CUDA starts, and `point(source="random")` builds."""
    task = _make(name)
    gen = torch.Generator("cuda").manual_seed(0)

    states = task.sample_start(N, gen)
    _assert_on(states, torch.device("cuda"))

    method = Point(_point_ctx(task, "cuda:0"), source="random")
    starts, ic_ids = method.reset(torch.arange(4, device="cuda:0"), None)
    _assert_on(starts, torch.device("cuda"))
    assert ic_ids.device.type == "cuda"
