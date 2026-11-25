import pytest

PENDULUM_TIMESTEP_S = 0.002

PENDULUM_MJCF = f"""
<mujoco>
  <option timestep="{PENDULUM_TIMESTEP_S}"/>
  <worldbody>
    <body name="arm" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.01"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02" mass="1"/>
      <site name="tip" pos="0 0 -0.5"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="hinge" gear="1"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def pendulum_mjcf() -> str:
    return PENDULUM_MJCF


@pytest.fixture
def pendulum_timestep_s() -> float:
    return PENDULUM_TIMESTEP_S


# --- stub task -------------------------------------------------------------

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor

from resetsweep.sim.batch import BatchSim
from resetsweep.types import Box, StartStates

_SUCCESS_TOL_RAD = 0.1


@dataclass
class StubTask:
    """Pendulum swing-up: success when the arm is within 0.1 rad of upright (angle pi).

    `success_script` and `terminate_script` override the physics-based flags: call i
    (one call per env step) returns `script[min(i, len - 1)]` for every env.
    """

    mjcf: str
    max_steps: int = 10
    hold_steps: int = 3
    success_script: list[bool] | None = None
    terminate_script: list[bool] | None = None
    name: str = "stub"
    obs_dim: int = 2
    act_dim: int = 1
    frame_skip: int = 5
    dense_sibling: str = "stub"
    extra_dim: int = 0
    start_box: Box = field(
        default_factory=lambda: Box(
            torch.tensor([-4.0, -20.0]), torch.tensor([4.0, 20.0])
        )
    )
    _success_calls: int = 0
    _terminate_calls: int = 0

    def sample_start(self, n: int, gen: torch.Generator) -> StartStates:
        u = torch.rand(n, 2, generator=gen)
        x = self.start_box.low + u * (self.start_box.high - self.start_box.low)
        return StartStates(x[:, :1], x[:, 1:], torch.zeros(n, 0))

    def canonical_start(self) -> StartStates:
        return StartStates(
            torch.full((1, 1), math.pi), torch.zeros(1, 1), torch.zeros(1, 0)
        )

    def observe(self, sim: BatchSim, extra: Tensor) -> Tensor:
        return torch.cat([sim.qpos, sim.qvel], dim=1)

    def reward(self, sim: BatchSim, extra: Tensor, action: Tensor) -> Tensor:
        return self.success_now(sim, extra, _count=False).float()

    def terminated(self, sim: BatchSim, extra: Tensor) -> Tensor:
        if self.terminate_script is None:
            return torch.zeros(sim.qpos.shape[0], dtype=torch.bool)
        flag = self.terminate_script[
            min(self._terminate_calls, len(self.terminate_script) - 1)
        ]
        self._terminate_calls += 1
        return torch.full((sim.qpos.shape[0],), flag, dtype=torch.bool)

    def success_now(
        self, sim: BatchSim, extra: Tensor, *, _count: bool = True
    ) -> Tensor:
        if self.success_script is None:
            wrapped = (sim.qpos[:, 0] - math.pi + math.pi) % (2 * math.pi) - math.pi
            return wrapped.abs() < _SUCCESS_TOL_RAD
        flag = self.success_script[
            min(self._success_calls, len(self.success_script) - 1)
        ]
        self._success_calls += int(_count)
        return torch.full((sim.qpos.shape[0],), flag, dtype=torch.bool)

    def state_features(self, qpos: Tensor, qvel: Tensor, extra: Tensor) -> Tensor:
        return torch.cat([qpos, qvel], dim=1)

    def is_valid(self, states: StartStates) -> Tensor:
        x = torch.cat([states.qpos, states.qvel], dim=1)
        inside = ((x >= self.start_box.low) & (x <= self.start_box.high)).all(dim=1)
        return inside & torch.isfinite(x).all(dim=1)

    def eval_states(self, n: int, seed: int) -> StartStates:
        gen = torch.Generator().manual_seed(seed)
        return self.sample_start(n, gen)


@pytest.fixture
def make_task(pendulum_mjcf: str) -> Callable[..., StubTask]:
    def factory(**overrides: object) -> StubTask:
        return StubTask(pendulum_mjcf, **overrides)  # pyright: ignore[reportArgumentType]

    return factory
