"""Cart-pole swing-up: a sparse task (T2) and its dense sibling, sharing one MJCF.

Joint order is `qpos = (x, theta)`. `theta = 0` hangs straight down, `theta = pi` is upright.
Actions are in [-1, 1] and scale to the cart force.
"""

import math

import torch
from torch import Tensor

from resetsweep.sim.batch import BatchSim
from resetsweep.tasks.base import TASKS, wrap_columns
from resetsweep.types import Box, StartStates

# Cart 1 kg, pole 0.2 kg with its centre of mass 0.25 m from the hinge. Full force 4 N on the
# ~1.2 kg system accelerates the cart at about 3.3 m/s^2, a third of g. Held steady that tips the
# hanging pole to atan(3.3 / 9.81) = 19 deg; swinging from rest it peaks near twice that, well
# short of pi/2. So the pole cannot be pushed up directly and the cart has to pump energy over
# several swings, within the +-2 m rail. timestep 0.01 s with frame_skip 5 gives a 0.05 s control
# step; 150 steps make 7.5 s.
FORCE_LIMIT_N = 4.0
RAIL_HALF_LENGTH_M = 2.0
_MJCF = f"""
<mujoco model="cartpole">
  <option timestep="0.01"/>
  <worldbody>
    <body name="cart">
      <joint name="slider" type="slide" axis="1 0 0" limited="true"
             range="-{RAIL_HALF_LENGTH_M} {RAIL_HALF_LENGTH_M}" damping="0.05"/>
      <geom type="box" size="0.1 0.1 0.05" mass="1"/>
      <body name="pole">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0.002"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02" mass="0.2"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slider" gear="{FORCE_LIMIT_N}" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""

_FRAME_SKIP = 5
_MAX_STEPS = 150
_HOLD_STEPS = 10  # 0.5 s upright
_UPRIGHT_TOL_RAD = 0.2
_CENTRE_TOL_M = 0.5
_ACTION_COST = 0.01
_CENTRE_COST = 0.05  # dense only: per m^2 of cart offset
_START_NOISE = 0.1
_MAX_CART_SPEED_M_S = 3.0
_MAX_POLE_SPEED_RAD_S = 10.0


def _angle_from_upright(theta: Tensor) -> Tensor:
    """Absolute angular distance to upright, in [0, pi], for any unwrapped angle."""
    return (torch.remainder(theta, 2 * math.pi) - math.pi).abs()


_HINGE_COLUMNS = (1,)  # unlimited hinges: the pole hinge; the cart slider is limited and never wrapped


class _Cartpole:
    mjcf = _MJCF
    obs_dim = 5  # x, cos(theta), sin(theta), x_dot, theta_dot
    act_dim = 1
    max_steps = _MAX_STEPS
    frame_skip = _FRAME_SKIP
    hold_steps = _HOLD_STEPS
    dense_sibling = "cartpole_swingup_dense"
    extra_dim = 0
    name: str

    def __init__(self) -> None:
        # Box over concat(qpos, qvel) = (x, theta, x_dot, theta_dot).
        self.start_box = Box(
            torch.tensor(
                [
                    -RAIL_HALF_LENGTH_M,
                    -math.pi,
                    -_MAX_CART_SPEED_M_S,
                    -_MAX_POLE_SPEED_RAD_S,
                ]
            ),
            torch.tensor(
                [
                    RAIL_HALF_LENGTH_M,
                    math.pi,
                    _MAX_CART_SPEED_M_S,
                    _MAX_POLE_SPEED_RAD_S,
                ]
            ),
        )

    def sample_start(self, n: int, gen: torch.Generator) -> StartStates:
        noise = (torch.rand(n, 4, generator=gen) * 2 - 1) * _START_NOISE
        return StartStates(noise[:, :2], noise[:, 2:], torch.zeros(n, 0))

    def canonical_start(self) -> StartStates:
        return StartStates(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 0))

    def observe(self, sim: BatchSim, extra: Tensor) -> Tensor:
        x, theta = sim.qpos[:, 0], sim.qpos[:, 1]
        return torch.stack(
            [x, theta.cos(), theta.sin(), sim.qvel[:, 0], sim.qvel[:, 1]], dim=1
        )

    def terminated(self, sim: BatchSim, extra: Tensor) -> Tensor:
        return torch.zeros(sim.qpos.shape[0], dtype=torch.bool, device=sim.qpos.device)

    def success_now(self, sim: BatchSim, extra: Tensor) -> Tensor:
        upright = _angle_from_upright(sim.qpos[:, 1]) < _UPRIGHT_TOL_RAD
        return upright & (sim.qpos[:, 0].abs() < _CENTRE_TOL_M)

    def state_features(self, qpos: Tensor, qvel: Tensor, extra: Tensor) -> Tensor:
        theta = qpos[:, 1:2]
        return torch.cat([qpos[:, :1], theta.cos(), theta.sin(), qvel], dim=1)

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


@TASKS.register("cartpole_swingup_sparse")
class CartpoleSwingupSparse(_Cartpole):
    name = "cartpole_swingup_sparse"

    def reward(self, sim: BatchSim, extra: Tensor, action: Tensor) -> Tensor:
        return self.success_now(sim, extra).float() - self._action_cost(action)


@TASKS.register("cartpole_swingup_dense")
class CartpoleSwingupDense(_Cartpole):
    name = "cartpole_swingup_dense"

    def reward(self, sim: BatchSim, extra: Tensor, action: Tensor) -> Tensor:
        height01 = (1 - sim.qpos[:, 1].cos()) / 2  # 0 hanging, 1 upright
        off_centre = _CENTRE_COST * sim.qpos[:, 0].square()
        return height01 - off_centre - self._action_cost(action)
