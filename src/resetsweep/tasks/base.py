"""The `Task` protocol: everything a reset method and the env need to know about a task."""

from typing import Protocol

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
