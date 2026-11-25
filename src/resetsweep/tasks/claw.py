"""Planar two-finger claw turning a hinged disc to a goal angle (T7, dense).

Everything lies in the z = 0 plane (no gravity). The gripper is a base on two slide joints
carrying two fingers that slide toward each other, so `qpos = (base_x, base_y, left_finger,
right_finger, disc_angle)`, in m and rad. The base body sits at the origin so its slides read
the base position directly. The left finger slides along +x and the right along -x from fully
open at 0, so closing is a positive left value and a negative right value.

The disc turns only through finger contact. The contacts have sliding friction (condim 3,
friction coefficient 1): a finger pressed on the rim and dragged along it carries the rim
round. Squeezing both fingers at the same height is a dead end, since their friction torques
cancel and the base is then pinned to the hinge; turning needs one finger pressing alone, or an
asymmetric grip. The disc has light damping and Coulomb friction on its hinge so it stays where
it is left, which is what makes the goal angle holdable.
"""

import torch
from torch import Tensor

from resetsweep.sim.batch import BatchSim
from resetsweep.tasks.base import TASKS
from resetsweep.tasks.contacts import InterpenetrationCheck
from resetsweep.types import Box, StartStates

DISC_XY = (0.40, 0.0)
DISC_RADIUS_M = 0.08
FINGER_RADIUS_M = 0.02
GOAL_ANGLE_RAD = 0.8
GOAL_TOL_RAD = 0.15

_FINGER_OFFSET_M = 0.12  # each finger sits this far either side of the base
_FINGER_STROKE_M = 0.10
_BASE_X_RANGE_M = (-0.3, 0.9)
_BASE_Y_RANGE_M = 0.5
_DISC_RANGE_RAD = 3.3
_MJCF = f"""
<mujoco model="claw">
  <compiler angle="radian"/>
  <option timestep="0.01" gravity="0 0 0"/>
  <default>
    <geom condim="3" friction="1.0 0.005 0.0001"/>
  </default>
  <worldbody>
    <body name="base">
      <joint name="bx" type="slide" axis="1 0 0" limited="true"
             range="{_BASE_X_RANGE_M[0]} {_BASE_X_RANGE_M[1]}" damping="5"/>
      <joint name="by" type="slide" axis="0 1 0" limited="true"
             range="-{_BASE_Y_RANGE_M} {_BASE_Y_RANGE_M}" damping="5"/>
      <geom type="box" size="0.02 0.02 0.02" mass="1" contype="0" conaffinity="0"/>
      <body name="left_finger" pos="-{_FINGER_OFFSET_M} 0 0">
        <joint name="fl" type="slide" axis="1 0 0" limited="true" range="0 {_FINGER_STROKE_M}"
               damping="5"/>
        <geom type="sphere" pos="0 -0.03 0" size="{FINGER_RADIUS_M}" mass="0.05"
              contype="1" conaffinity="0"/>
        <geom type="sphere" pos="0 0.03 0" size="{FINGER_RADIUS_M}" mass="0.05"
              contype="1" conaffinity="0"/>
      </body>
      <body name="right_finger" pos="{_FINGER_OFFSET_M} 0 0">
        <joint name="fr" type="slide" axis="1 0 0" limited="true" range="-{_FINGER_STROKE_M} 0"
               damping="5"/>
        <geom type="sphere" pos="0 -0.03 0" size="{FINGER_RADIUS_M}" mass="0.05"
              contype="1" conaffinity="0"/>
        <geom type="sphere" pos="0 0.03 0" size="{FINGER_RADIUS_M}" mass="0.05"
              contype="1" conaffinity="0"/>
      </body>
    </body>
    <body name="disc" pos="{DISC_XY[0]} {DISC_XY[1]} 0">
      <joint name="turn" type="hinge" axis="0 0 1" limited="true"
             range="-{_DISC_RANGE_RAD} {_DISC_RANGE_RAD}" damping="0.05" frictionloss="0.01"/>
      <geom type="cylinder" size="{DISC_RADIUS_M} 0.05" mass="0.2" contype="0" conaffinity="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="bx" gear="2.5" ctrllimited="true" ctrlrange="-1 1"/>
    <motor joint="by" gear="2.5" ctrllimited="true" ctrlrange="-1 1"/>
    <motor joint="fl" gear="2.5" ctrllimited="true" ctrlrange="-1 1"/>
    <motor joint="fr" gear="2.5" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""

# Contacts are soft, so a finger pressed on the rim sags into it. Random bang-bang rollouts from
# valid starts (46 worlds, 400 control steps) sagged at most 7.9 mm; a finger sphere is 20 mm
# in radius, so beyond 15 mm its centre is nearly at the rim and the pose is real overlap.
PENETRATION_TOL_M = 0.015
_SLIDE_MARGIN_M = (
    0.1  # soft joint limits sag past their range, and is_valid must accept that
)
_DISC_MARGIN_RAD = 0.4
_SLIDE_SPEED_M_S = 3.0
_DISC_SPEED_RAD_S = 30.0
_START_X_M = (0.0, 0.15)
_START_Y_M = (-0.15, 0.15)
_START_DISC_NOISE_RAD = 0.05
_REACH_RADIUS_M = 0.15  # the gripper's pull toward the disc stops inside this distance
_ANGLE_WEIGHT = 2.0
_ACTION_COST = 0.01


@TASKS.register("claw")
class Claw:
    name = "claw"
    mjcf = _MJCF
    obs_dim = 11  # base xy, two fingers, cos/sin disc angle, five velocities
    act_dim = 4
    max_steps = 100  # 0.05 s control step: 5 s
    frame_skip = 5
    hold_steps = 5  # 0.25 s inside the goal band
    extra_dim = 0
    dense_sibling = "claw"

    def __init__(self) -> None:
        low = torch.tensor(
            [_BASE_X_RANGE_M[0] - _SLIDE_MARGIN_M, -_BASE_Y_RANGE_M - _SLIDE_MARGIN_M]
            + [-_FINGER_STROKE_M - _SLIDE_MARGIN_M] * 2
            + [-_DISC_RANGE_RAD - _DISC_MARGIN_RAD]
            + [-_SLIDE_SPEED_M_S] * 4
            + [-_DISC_SPEED_RAD_S]
        )
        high = torch.tensor(
            [_BASE_X_RANGE_M[1] + _SLIDE_MARGIN_M, _BASE_Y_RANGE_M + _SLIDE_MARGIN_M]
            + [_FINGER_STROKE_M + _SLIDE_MARGIN_M] * 2
            + [_DISC_RANGE_RAD + _DISC_MARGIN_RAD]
            + [_SLIDE_SPEED_M_S] * 4
            + [_DISC_SPEED_RAD_S]
        )
        self.start_box = Box(low, high)
        self._contacts = InterpenetrationCheck(
            self.mjcf,
            frame_skip=self.frame_skip,
            canonical=self.canonical_start(),
            tolerance_m=PENETRATION_TOL_M,
        )

    def sample_start(self, n: int, gen: torch.Generator) -> StartStates:
        x = _START_X_M[0] + torch.rand(n, 1, generator=gen) * (
            _START_X_M[1] - _START_X_M[0]
        )
        y = _START_Y_M[0] + torch.rand(n, 1, generator=gen) * (
            _START_Y_M[1] - _START_Y_M[0]
        )
        angle = (torch.rand(n, 1, generator=gen) * 2 - 1) * _START_DISC_NOISE_RAD
        qpos = torch.cat([x, y, torch.zeros(n, 2), angle], dim=1)
        return StartStates(qpos, torch.zeros(n, 5), torch.zeros(n, 0))

    def canonical_start(self) -> StartStates:
        x, y = sum(_START_X_M) / 2, sum(_START_Y_M) / 2
        qpos = torch.tensor([[x, y, 0.0, 0.0, 0.0]])
        return StartStates(qpos, torch.zeros(1, 5), torch.zeros(1, 0))

    def observe(self, sim: BatchSim, extra: Tensor) -> Tensor:
        q, v = sim.qpos, sim.qvel
        return torch.cat([q[:, :4], q[:, 4:5].cos(), q[:, 4:5].sin(), v], dim=1)

    def terminated(self, sim: BatchSim, extra: Tensor) -> Tensor:
        return torch.zeros(sim.qpos.shape[0], dtype=torch.bool, device=sim.qpos.device)

    def success_now(self, sim: BatchSim, extra: Tensor) -> Tensor:
        return (sim.qpos[:, 4] - GOAL_ANGLE_RAD).abs() < GOAL_TOL_RAD

    def reward(self, sim: BatchSim, extra: Tensor, action: Tensor) -> Tensor:
        disc = torch.tensor(DISC_XY, dtype=sim.qpos.dtype, device=sim.qpos.device)
        reach_m = (sim.qpos[:, :2] - disc).norm(dim=1).clamp(min=_REACH_RADIUS_M)
        angle_error = (sim.qpos[:, 4] - GOAL_ANGLE_RAD).abs()
        shaped = -(reach_m + _ANGLE_WEIGHT * angle_error)
        cost = _ACTION_COST * action.square().sum(dim=1)
        return shaped + self.success_now(sim, extra).float() - cost

    def state_features(self, qpos: Tensor, qvel: Tensor, extra: Tensor) -> Tensor:
        return torch.cat(
            [qpos[:, :4], qpos[:, 4:5].cos(), qpos[:, 4:5].sin(), qvel], dim=1
        )

    def is_valid(self, states: StartStates) -> Tensor:
        x = torch.cat([states.qpos, states.qvel], dim=1)
        low = self.start_box.low.to(x.device)
        high = self.start_box.high.to(x.device)
        in_box = ((x >= low) & (x <= high)).all(dim=1) & torch.isfinite(x).all(dim=1)
        no_overlap = self._contacts.ok(states.qpos.cpu(), in_box.cpu())
        return in_box & no_overlap.to(x.device)

    def eval_states(self, n: int, seed: int) -> StartStates:
        # A local generator: never touches the global torch RNG.
        return self.sample_start(n, torch.Generator().manual_seed(seed))
