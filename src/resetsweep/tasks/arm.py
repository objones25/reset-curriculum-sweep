"""Planar 3-link arm pushing a puck to a goal disc (T2, §12: push, not pick-and-place).

Everything lies in the z = 0 plane (no gravity): the table is implicit, and the puck's sliding
friction is joint damping. `qpos = (q1, q2, q3, puck_x, puck_y, puck_angle)`, in rad and m, with
`q` the arm's hinge angles (each relative to the previous link). The puck body sits at the origin
so its two slide joints read the puck's world position directly (a slide joint's qpos is the
offset from the body's reference position, not an absolute coordinate).

Collisions: links collide only with the puck (not with each other), by contype/conaffinity. The
contacts are frictionless (condim 1), so the puck is only ever shoved along the contact normal.
"""

import torch
from torch import Tensor

from resetsweep.sim.batch import BatchSim
from resetsweep.tasks.base import TASKS
from resetsweep.tasks.contacts import InterpenetrationCheck
from resetsweep.types import Box, StartStates

LINK_LENGTHS_M = (0.30, 0.28, 0.22)
GOAL_XY = (0.60, 0.0)
GOAL_RADIUS_M = 0.06
PUCK_RADIUS_M = 0.04

# Three links, 0.8 m reach. Hinge torque is 1.5 N*m at full action against joint damping of
# 1 N*m*s/rad, so a joint settles near 1.5 rad/s; the puck (0.2 kg, damping 1 N*s/m, 0.04 m
# radius) is shoved at about the hand's speed and coasts to rest in ~0.2 s. timestep 0.01 s with
# frame_skip 5 gives a 0.05 s control step; 100 steps make 5 s.
_REST_QPOS = (1.0, -1.0, 0.0)  # arm folded up and away from the puck's start region
_ARM_RANGE_RAD = 2.6
_JOINT_MARGIN = (
    0.4  # soft joint limits sag past their range, and is_valid must accept that
)
_PUCK_RANGE_M = 1.0
_PUCK_SPIN_RANGE_RAD = 3.14159
_ARM_SPEED_RAD_S = 10.0
_PUCK_SPEED_M_S = 5.0
_PUCK_SPIN_SPEED_RAD_S = 10.0
_PUCK_START_X_M = (0.30, 0.42)
_PUCK_START_Y_M = (-0.12, 0.12)
_ARM_START_NOISE_RAD = 0.15
# Contacts are soft, so a link pressing the puck sags a little into it. Real interpenetration is
# a start state that overlaps by more than this: half the puck's radius. Hard random pushes
# were measured to sag up to 1.3 cm, and the soft-contact time constant cannot go below twice
# the timestep, so the contacts cannot be made stiffer.
PENETRATION_TOL_M = 0.02
_ACTION_COST = 0.01
_MJCF = f"""
<mujoco model="arm_push">
  <compiler angle="radian"/>
  <option timestep="0.01" gravity="0 0 0"/>
  <default>
    <geom condim="1"/>
    <joint damping="1.0"/>
  </default>
  <worldbody>
    <body name="link1">
      <joint name="j1" type="hinge" axis="0 0 1" limited="true" range="-{_ARM_RANGE_RAD} {_ARM_RANGE_RAD}"/>
      <geom type="capsule" fromto="0 0 0 {LINK_LENGTHS_M[0]} 0 0" size="0.02" mass="0.5"
            contype="1" conaffinity="0"/>
      <body name="link2" pos="{LINK_LENGTHS_M[0]} 0 0">
        <joint name="j2" type="hinge" axis="0 0 1" limited="true" range="-{_ARM_RANGE_RAD} {_ARM_RANGE_RAD}"/>
        <geom type="capsule" fromto="0 0 0 {LINK_LENGTHS_M[1]} 0 0" size="0.02" mass="0.4"
              contype="1" conaffinity="0"/>
        <body name="link3" pos="{LINK_LENGTHS_M[1]} 0 0">
          <joint name="j3" type="hinge" axis="0 0 1" limited="true" range="-{_ARM_RANGE_RAD} {_ARM_RANGE_RAD}"/>
          <geom type="capsule" fromto="0 0 0 {LINK_LENGTHS_M[2]} 0 0" size="0.02" mass="0.3"
                contype="1" conaffinity="0"/>
          <site name="hand" pos="{LINK_LENGTHS_M[2]} 0 0" size="0.01"/>
        </body>
      </body>
    </body>
    <body name="puck">  <!-- at the origin: slide qpos is then the puck position -->
      <joint name="px" type="slide" axis="1 0 0" limited="true" range="-{_PUCK_RANGE_M} {_PUCK_RANGE_M}"/>
      <joint name="py" type="slide" axis="0 1 0" limited="true" range="-{_PUCK_RANGE_M} {_PUCK_RANGE_M}"/>
      <joint name="pa" type="hinge" axis="0 0 1" limited="true" range="-{_PUCK_SPIN_RANGE_RAD} {_PUCK_SPIN_RANGE_RAD}"
             damping="0.01"/>
      <geom type="sphere" size="{PUCK_RADIUS_M}" mass="0.2" contype="0" conaffinity="1"/>
    </body>
    <geom name="goal" type="cylinder" pos="{GOAL_XY[0]} {GOAL_XY[1]} -0.05"
          size="{GOAL_RADIUS_M} 0.001" contype="0" conaffinity="0" rgba="0 1 0 0.3"/>
  </worldbody>
  <actuator>
    <motor joint="j1" gear="1.5" ctrllimited="true" ctrlrange="-{_PUCK_RANGE_M} {_PUCK_RANGE_M}"/>
    <motor joint="j2" gear="1.5" ctrllimited="true" ctrlrange="-{_PUCK_RANGE_M} {_PUCK_RANGE_M}"/>
    <motor joint="j3" gear="1.5" ctrllimited="true" ctrlrange="-{_PUCK_RANGE_M} {_PUCK_RANGE_M}"/>
  </actuator>
</mujoco>
"""


class _Arm:
    mjcf = _MJCF
    obs_dim = 10  # q (3), q_dot (3), puck xy (2), puck velocity (2)
    act_dim = 3
    max_steps = 100
    frame_skip = 5
    hold_steps = 5  # 0.25 s with the puck in the goal
    extra_dim = 0
    name: str
    dense_sibling: str

    def __init__(self) -> None:
        arm_q = _ARM_RANGE_RAD + _JOINT_MARGIN
        puck_q = _PUCK_RANGE_M + _JOINT_MARGIN
        spin_q = _PUCK_SPIN_RANGE_RAD + _JOINT_MARGIN
        # Box over concat(qpos, qvel) = (q1..q3, puck x, y, angle, then their velocities).
        bound = torch.tensor(
            [arm_q] * 3
            + [puck_q] * 2
            + [spin_q]
            + [_ARM_SPEED_RAD_S] * 3
            + [_PUCK_SPEED_M_S] * 2
            + [_PUCK_SPIN_SPEED_RAD_S]
        )
        self.start_box = Box(-bound, bound)
        self._contacts = InterpenetrationCheck(
            self.mjcf,
            frame_skip=self.frame_skip,
            canonical=self.canonical_start(),
            tolerance_m=PENETRATION_TOL_M,
        )

    def sample_start(self, n: int, gen: torch.Generator) -> StartStates:
        arm = (
            torch.tensor(_REST_QPOS)
            + (torch.rand(n, 3, generator=gen) * 2 - 1) * _ARM_START_NOISE_RAD
        )
        puck_x = _PUCK_START_X_M[0] + torch.rand(n, 1, generator=gen) * (
            _PUCK_START_X_M[1] - _PUCK_START_X_M[0]
        )
        puck_y = _PUCK_START_Y_M[0] + torch.rand(n, 1, generator=gen) * (
            _PUCK_START_Y_M[1] - _PUCK_START_Y_M[0]
        )
        qpos = torch.cat([arm, puck_x, puck_y, torch.zeros(n, 1)], dim=1)
        return StartStates(qpos, torch.zeros(n, 6), torch.zeros(n, 0))

    def canonical_start(self) -> StartStates:
        x = sum(_PUCK_START_X_M) / 2
        y = sum(_PUCK_START_Y_M) / 2
        qpos = torch.tensor([[*_REST_QPOS, x, y, 0.0]])
        return StartStates(qpos, torch.zeros(1, 6), torch.zeros(1, 0))

    def observe(self, sim: BatchSim, extra: Tensor) -> Tensor:
        q, v = sim.qpos, sim.qvel
        return torch.cat([q[:, :3], v[:, :3], q[:, 3:5], v[:, 3:5]], dim=1)

    def terminated(self, sim: BatchSim, extra: Tensor) -> Tensor:
        return torch.zeros(sim.qpos.shape[0], dtype=torch.bool, device=sim.qpos.device)

    def success_now(self, sim: BatchSim, extra: Tensor) -> Tensor:
        return _puck_to_goal_m(sim.qpos) < GOAL_RADIUS_M

    def state_features(self, qpos: Tensor, qvel: Tensor, extra: Tensor) -> Tensor:
        return torch.cat(
            [qpos[:, :3].cos(), qpos[:, :3].sin(), qpos[:, 3:5], qvel], dim=1
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


def _hand_xy(qpos: Tensor) -> Tensor:
    """Hand (tip of link 3) position `(n, 2)` by planar forward kinematics, on qpos' device."""
    angles = torch.cumsum(qpos[:, :3], dim=1)
    lengths = torch.tensor(LINK_LENGTHS_M, dtype=qpos.dtype, device=qpos.device)
    return torch.stack(
        [(lengths * angles.cos()).sum(dim=1), (lengths * angles.sin()).sum(dim=1)],
        dim=1,
    )


def _puck_to_goal_m(qpos: Tensor) -> Tensor:
    goal = torch.tensor(GOAL_XY, dtype=qpos.dtype, device=qpos.device)
    return (qpos[:, 3:5] - goal).norm(dim=1)


def _action_cost(action: Tensor) -> Tensor:
    return _ACTION_COST * action.square().sum(dim=1)


@TASKS.register("arm_sparse")
class ArmSparse(_Arm):
    name = "arm_sparse"
    dense_sibling = "arm_dense"

    def reward(self, sim: BatchSim, extra: Tensor, action: Tensor) -> Tensor:
        return self.success_now(sim, extra).float() - _action_cost(action)


@TASKS.register("arm_dense")
class ArmDense(_Arm):
    name = "arm_dense"
    dense_sibling = "arm_dense"

    def reward(self, sim: BatchSim, extra: Tensor, action: Tensor) -> Tensor:
        hand_to_puck = (sim.qpos[:, 3:5] - _hand_xy(sim.qpos)).norm(dim=1)
        shaped = -(hand_to_puck + _puck_to_goal_m(sim.qpos))
        return shaped + self.success_now(sim, extra).float() - _action_cost(action)
