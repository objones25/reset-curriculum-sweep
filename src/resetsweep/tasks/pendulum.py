"""Hinge pendulum swing-up: a sparse task (T2) and its dense sibling (T7), sharing one MJCF.

Angle convention: `qpos[0] = 0` hangs straight down, `qpos[0] = pi` is upright.
"""

import math

import torch
from torch import Tensor

from resetsweep.sim.batch import BatchSim
from resetsweep.tasks.base import TASKS, wrap_columns
from resetsweep.types import Box, StartStates

# Capsule of mass 1 kg, 0.5 m long, so its centre of mass is l = 0.25 m below the hinge and
# gravity's peak torque is m*g*l = 1 * 9.81 * 0.25 = 2.45 N*m. The motor's ctrlrange is
# [-1, 1] with gear 1, i.e. 1 N*m = 41% of that. Held static, 1 N*m tips the pendulum to
# asin(1 / 2.45) = 24 deg; swinging from rest it peaks near twice that (48 deg), well short
# of pi/2. So the arm cannot be pushed up directly and has to pump energy over several swings.
# timestep 0.01 s with frame_skip 5 gives the 0.05 s control step; 100 steps make 5 s.
TORQUE_LIMIT_NM = 1.0
_MJCF = f"""
<mujoco model="pendulum">
  <option timestep="0.01"/>
  <worldbody>
    <body name="arm" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.01"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="hinge" gear="1" ctrllimited="true" ctrlrange="-{TORQUE_LIMIT_NM} {TORQUE_LIMIT_NM}"/>
  </actuator>
</mujoco>
"""

_FRAME_SKIP = 5
_MAX_STEPS = 100
_HOLD_STEPS = 10  # 0.5 s upright
# Holding the band is feasible: gravity torque at 0.2 rad is 2.45 * sin(0.2) = 0.49 N*m < 1.0.
_UPRIGHT_TOL_RAD = 0.2
_ACTION_COST = 0.01
_START_NOISE = 0.1
_MAX_SPEED_RAD_S = 8.0


def _angle_from_upright(theta: Tensor) -> Tensor:
    """Absolute angular distance to upright, in [0, pi], for any unwrapped angle."""
    return (torch.remainder(theta, 2 * math.pi) - math.pi).abs()


_HINGE_COLUMNS = (0,)  # unlimited hinges: the hinge (no `limited` in the MJCF)


class _Pendulum:
    mjcf = _MJCF
    obs_dim = 3  # cos(theta), sin(theta), omega
    act_dim = 1
    max_steps = _MAX_STEPS
    frame_skip = _FRAME_SKIP
    hold_steps = _HOLD_STEPS
    dense_sibling = "pendulum_dense"
    extra_dim = 0
    name: str

    def __init__(self) -> None:
        self.start_box = Box(
            torch.tensor([-math.pi, -_MAX_SPEED_RAD_S]),
            torch.tensor([math.pi, _MAX_SPEED_RAD_S]),
        )

    def sample_start(self, n: int, gen: torch.Generator) -> StartStates:
        noise = (
            torch.rand(n, 2, generator=gen, device=gen.device) * 2 - 1
        ) * _START_NOISE
        return StartStates(
            noise[:, :1], noise[:, 1:], torch.zeros(n, 0, device=gen.device)
        )

    def canonical_start(self) -> StartStates:
        return StartStates(torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 0))

    def observe(self, sim: BatchSim, extra: Tensor) -> Tensor:
        theta = sim.qpos[:, 0]
        return torch.stack([theta.cos(), theta.sin(), sim.qvel[:, 0]], dim=1)

    def terminated(self, sim: BatchSim, extra: Tensor) -> Tensor:
        return torch.zeros(sim.qpos.shape[0], dtype=torch.bool, device=sim.qpos.device)

    def success_now(self, sim: BatchSim, extra: Tensor) -> Tensor:
        return _angle_from_upright(sim.qpos[:, 0]) < _UPRIGHT_TOL_RAD

    def state_features(self, qpos: Tensor, qvel: Tensor, extra: Tensor) -> Tensor:
        return torch.cat([qpos.cos(), qpos.sin(), qvel], dim=1)

    def canonicalize(self, states: StartStates) -> StartStates:
        return wrap_columns(states, _HINGE_COLUMNS)

    def is_valid(self, states: StartStates) -> Tensor:
        x = torch.cat([states.qpos, states.qvel], dim=1)
        low = self.start_box.low.to(x.device)
        high = self.start_box.high.to(x.device)
        return ((x >= low) & (x <= high)).all(dim=1) & torch.isfinite(x).all(dim=1)

    def eval_states(self, n: int, seed: int) -> StartStates:
        # A local generator: never touches the global torch RNG.
        return self.sample_start(n, torch.Generator().manual_seed(seed))

    @staticmethod
    def _action_cost(action: Tensor) -> Tensor:
        return _ACTION_COST * action.square().sum(dim=1)


@TASKS.register("pendulum_sparse")
class PendulumSparse(_Pendulum):
    name = "pendulum_sparse"

    def reward(self, sim: BatchSim, extra: Tensor, action: Tensor) -> Tensor:
        return self.success_now(sim, extra).float() - self._action_cost(action)


@TASKS.register("pendulum_dense")
class PendulumDense(_Pendulum):
    name = "pendulum_dense"

    def reward(self, sim: BatchSim, extra: Tensor, action: Tensor) -> Tensor:
        height01 = (1 - sim.qpos[:, 0].cos()) / 2  # 0 hanging, 1 upright
        return height01 - self._action_cost(action)
