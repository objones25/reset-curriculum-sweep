"""The `Task` protocol: everything a reset method and the env need to know about a task."""

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from resetsweep.registry import make_registry
from resetsweep.sim.batch import BatchSim
from resetsweep.types import Box, StartStates

TASKS = make_registry("task")


class Task(Protocol):
    name: str
    mjcf: str
    obs_dim: int
    act_dim: int
    max_steps: int
    frame_skip: int
    hold_steps: int
    start_box: Box
    dense_sibling: str
    extra_dim: int

    def sample_start(self, n: int, gen: torch.Generator) -> StartStates: ...

    def canonical_start(self) -> StartStates: ...

    def observe(self, sim: BatchSim, extra: Tensor) -> Tensor: ...

    def reward(self, sim: BatchSim, extra: Tensor, action: Tensor) -> Tensor: ...

    def terminated(self, sim: BatchSim, extra: Tensor) -> Tensor: ...

    def success_now(self, sim: BatchSim, extra: Tensor) -> Tensor: ...

    def state_features(self, qpos: Tensor, qvel: Tensor, extra: Tensor) -> Tensor: ...

    def is_valid(self, states: StartStates) -> Tensor: ...

    def eval_states(self, n: int, seed: int) -> StartStates: ...


@runtime_checkable
class ControlsAction(Protocol):
    """Optional hook: tasks whose dynamics depend on where the agent is (a pit that pulls it
    back) turn the policy's action into the sim control. `VecEnv.step` calls it when present;
    without it the action goes to the sim unchanged."""

    def control(self, sim: BatchSim, extra: Tensor, action: Tensor) -> Tensor: ...


@runtime_checkable
class InitsExtra(Protocol):
    """Optional hook: tasks whose `extra` is drawn at reset (a coin flip) set it here.
    `VecEnv` calls it for every episode start, with its own generator, and the result replaces
    the `extra` the reset method supplied. It returns shape `(states.n, extra_dim)`."""

    def init_extra(self, states: StartStates, gen: torch.Generator) -> Tensor: ...
