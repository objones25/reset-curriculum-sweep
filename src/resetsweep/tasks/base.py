"""The `Task` protocol: everything a reset method and the env need to know about a task."""

import math
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
class AppliesForce(Protocol):
    """Optional hook: tasks whose dynamics depend on where the agent is (a pit that pulls it
    back and slows it) return the generalized force `(nworld, nv)` to apply, or None. Called
    before every physics substep by `resetsweep.sim.dynamics.step_task`."""

    def applied_force(self, sim: BatchSim, extra: Tensor) -> Tensor | None: ...


@runtime_checkable
class InitsExtra(Protocol):
    """Optional hook: tasks whose `extra` is drawn at reset (a coin flip) set it here.
    `VecEnv` calls it for every episode start, with its own generator, and the result replaces
    the `extra` the reset method supplied. It returns shape `(states.n, extra_dim)`."""

    def init_extra(self, states: StartStates, gen: torch.Generator) -> Tensor: ...


@runtime_checkable
class Canonicalizes(Protocol):
    """Optional hook: tasks with unlimited hinge joints wrap those qpos entries into
    [-pi, pi). A swinging hinge accumulates angle past +-pi; the state is physically the
    same as the wrapped one, which is the one `start_box` and `is_valid` describe. Limited
    joints and slides are never touched. Callers use `canonicalize`, not the hook."""

    def canonicalize(self, states: StartStates) -> StartStates: ...


def canonicalize(task: Task, states: StartStates) -> StartStates:
    """`task.canonicalize(states)` for tasks with the hook, else `states` unchanged."""
    if isinstance(task, Canonicalizes):
        return task.canonicalize(states)
    return states


REWARD_MAX_DEFAULT = 1.0


@runtime_checkable
class BoundsReward(Protocol):
    """Optional attribute: the largest per-step reward the task can give. Sparse tasks pay a
    success indicator minus a non-negative action cost, so their bound is the default 1.0."""

    reward_max: float


def reward_max(task: Task) -> float:
    """`task.reward_max` for tasks that set it, else `REWARD_MAX_DEFAULT`."""
    if isinstance(task, BoundsReward):
        return task.reward_max
    return REWARD_MAX_DEFAULT


def prepare_candidates(task: Task, states: StartStates) -> tuple[StartStates, Tensor]:
    """The one path for reached states kept for later restarts: canonicalize, then mark the
    rows that are finite and accepted by `task.is_valid`. Returns `(states, valid)`."""
    states = canonicalize(task, states)
    finite = (
        torch.isfinite(states.qpos).all(dim=1)
        & torch.isfinite(states.qvel).all(dim=1)
        & torch.isfinite(states.extra).all(dim=1)
    )
    return states, finite & task.is_valid(states)


def wrap_angle(theta: Tensor) -> Tensor:
    """`theta` wrapped into [-pi, pi]. Angles already in [-pi, pi) pass through bit for bit
    (the remainder arithmetic would perturb them by float rounding); a wrapped angle can
    land exactly on pi through float32 rounding, which `start_box` still accepts."""
    wrapped = torch.remainder(theta + math.pi, 2 * math.pi) - math.pi
    inside = (theta >= -math.pi) & (theta < math.pi)
    return torch.where(inside, theta, wrapped)


def wrap_columns(states: StartStates, columns: tuple[int, ...]) -> StartStates:
    """`states` with the qpos `columns` (unlimited hinges) wrapped by `wrap_angle`."""
    qpos = states.qpos.clone()
    qpos[:, list(columns)] = wrap_angle(qpos[:, list(columns)])
    return StartStates(qpos, states.qvel, states.extra)
