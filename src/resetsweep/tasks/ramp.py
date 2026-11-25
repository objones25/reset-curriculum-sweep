"""Deceptive ramp: an underpowered cart in a valley whose goal is atop the far slope (T6).

After Kinetix Car-Ramp. The dense sibling's reward (height) points at the near slope's top,
a local maximum; the sparse goal sits on the higher far slope. The motor beats the near slope
but not the far one, so the cart must first drive away from the goal to build momentum.

Design: the valley is a 1-D curve, and `qpos[0] = s` is the arc length along it (0 at the
vertex, negative up the near slope, positive up the far slope). A slide joint moves along a
straight line, so the curve's gravity is supplied through the `applied_force` hook:
`F = -m g dh/ds`. For motion constrained to a curve parametrised by arc length that is the
exact equation of motion, so the physics is a valley, with no contact model to tune.

Profile: near slope (gentle, the motor can climb it), vertex at s = 0, far slope (steep, the
motor cannot climb it from rest), then a flat plateau to the end stop. The goal is the plateau
beyond `GOAL_S_M`; the motor can hold position there, which a slope that outpulls the motor
would not allow. Both end stops are soft joint limits.
"""

import math

import torch
from torch import Tensor

from resetsweep.sim.batch import BatchSim
from resetsweep.tasks.base import TASKS
from resetsweep.types import Box, StartStates

_G_M_S2 = 9.81
MASS_KG = 1.0
# Motor 3 N. Near slope pulls with m g sin(12 deg) = 2.04 N < 3 N (climbable, net 0.96 N);
# far slope with m g sin(25 deg) = 4.15 N > 3 N (the cart cannot climb it from rest).
FORCE_LIMIT_N = 3.0
NEAR_ANGLE_DEG = 12.0
FAR_ANGLE_DEG = 25.0
NEAR_STOP_M = 4.0  # arc length of the near slope; its top is 0.83 m high
FAR_SLOPE_LEN_M = 4.0  # arc length of the far slope; its top is 1.69 m high
FAR_STOP_M = 5.5  # end of the plateau
GOAL_S_M = 4.3  # goal: plateau beyond this arc length
_DAMPING_N_S_M = 0.05
_SIN_NEAR = math.sin(math.radians(NEAR_ANGLE_DEG))
_SIN_FAR = math.sin(math.radians(FAR_ANGLE_DEG))
# The end stops are soft joint limits. Measured give: 0.03 m hit at 6 m/s, 0.06 m at the 7.6 m/s
# top speed of the pumped rollouts, under 0.15 m for a hit at 8 m/s (a test). 0.15 m is that margin.
_STOP_TOL_M = 0.15
_SPEED_MAX_M_S = 10.0
_START_HALF_WIDTH_M = 0.3
_START_SPEED_M_S = 0.05

_MJCF = f"""
<mujoco model="ramp">
  <option timestep="0.01" gravity="0 0 0"/>
  <worldbody>
    <body name="cart" pos="0 0 0">
      <joint name="s" type="slide" axis="1 0 0" damping="{_DAMPING_N_S_M}"
             limited="true" range="{-NEAR_STOP_M} {FAR_STOP_M}"/>
      <geom type="box" size="0.1 0.1 0.05" mass="{MASS_KG}" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="s" gear="{FORCE_LIMIT_N}" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""

_FRAME_SKIP = 5
_MAX_STEPS = 200  # 10 s at 0.05 s per control step
_HOLD_STEPS = 10  # 0.5 s on the plateau


def height_slope(s: Tensor) -> Tensor:
    """d height / d s at arc length `s`: -sin(near) on the near slope, +sin(far) on the far."""
    return torch.where(
        s < 0, -_SIN_NEAR, torch.where(s < FAR_SLOPE_LEN_M, _SIN_FAR, 0.0)
    )


def height_m(s: Tensor) -> Tensor:
    """Cart height above the vertex at arc length `s` (flat beyond the far slope)."""
    return torch.where(s < 0, -s * _SIN_NEAR, s.clamp(max=FAR_SLOPE_LEN_M) * _SIN_FAR)


class _Ramp:
    mjcf = _MJCF
    obs_dim = 2  # s, ds/dt
    act_dim = 1
    max_steps = _MAX_STEPS
    frame_skip = _FRAME_SKIP
    hold_steps = _HOLD_STEPS
    dense_sibling = "ramp_dense"
    extra_dim = 0
    name: str

    def __init__(self) -> None:
        self.start_box = Box(
            torch.tensor([-NEAR_STOP_M, -_SPEED_MAX_M_S]),
            torch.tensor([FAR_STOP_M, _SPEED_MAX_M_S]),
        )

    def sample_start(self, n: int, gen: torch.Generator) -> StartStates:
        u = torch.rand(n, 2, generator=gen) * 2 - 1
        return StartStates(
            u[:, :1] * _START_HALF_WIDTH_M,
            u[:, 1:] * _START_SPEED_M_S,
            torch.zeros(n, 0),
        )

    def canonical_start(self) -> StartStates:
        return StartStates(torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 0))

    def applied_force(self, sim: BatchSim, extra: Tensor) -> Tensor:
        """Gravity along the valley, `-m g dh/ds` (N), evaluated at every substep."""
        return -MASS_KG * _G_M_S2 * height_slope(sim.qpos[:, :1])

    def observe(self, sim: BatchSim, extra: Tensor) -> Tensor:
        return torch.cat([sim.qpos, sim.qvel], dim=1)

    def terminated(self, sim: BatchSim, extra: Tensor) -> Tensor:
        return torch.zeros(sim.qpos.shape[0], dtype=torch.bool, device=sim.qpos.device)

    def success_now(self, sim: BatchSim, extra: Tensor) -> Tensor:
        return sim.qpos[:, 0] >= GOAL_S_M  # False for NaN

    def state_features(self, qpos: Tensor, qvel: Tensor, extra: Tensor) -> Tensor:
        return torch.cat([qpos, qvel], dim=1)

    def is_valid(self, states: StartStates) -> Tensor:
        """Finite and inside the start box. The box is padded by `_STOP_TOL_M` along s: the
        end stops are soft, so a cart at or against one sits slightly past the joint range.
        That is the only interpenetration there is (the cart has no other contacts)."""
        x = torch.cat([states.qpos, states.qvel], dim=1)
        pad = torch.tensor([_STOP_TOL_M, 0.0], device=x.device)
        low = self.start_box.low.to(x.device) - pad
        high = self.start_box.high.to(x.device) + pad
        return ((x >= low) & (x <= high)).all(dim=1) & torch.isfinite(x).all(dim=1)

    def eval_states(self, n: int, seed: int) -> StartStates:
        return self.sample_start(n, torch.Generator().manual_seed(seed))


@TASKS.register("ramp_deceptive")
class RampDeceptive(_Ramp):
    name = "ramp_deceptive"

    def reward(self, sim: BatchSim, extra: Tensor, action: Tensor) -> Tensor:
        return self.success_now(sim, extra).float()


@TASKS.register("ramp_dense")
class RampDense(_Ramp):
    name = "ramp_dense"
    # reward = cart height: at most the far slope's top (1.69 m); the near top is 0.83 m and
    # the plateau beyond the far slope is flat. There is no cost term.
    reward_max = FAR_SLOPE_LEN_M * _SIN_FAR

    def reward(self, sim: BatchSim, extra: Tensor, action: Tensor) -> Tensor:
        return height_m(sim.qpos[:, 0])
