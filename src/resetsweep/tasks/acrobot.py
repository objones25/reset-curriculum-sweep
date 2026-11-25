"""Acrobot swing-up: a sparse task (T2) and its dense sibling, sharing one MJCF.

Two links hang from a fixed pivot. Only the elbow is driven; the shoulder is passive.
`qpos = (shoulder, elbow)`: `(0, 0)` is hanging straight down, `(pi, 0)` is straight up.
Actions are in [-1, 1] and scale to the elbow torque.
"""

import torch
from torch import Tensor

from resetsweep.sim.batch import BatchSim
from resetsweep.tasks.base import TASKS, wrap_columns
from resetsweep.types import Box, StartStates

# Two 1 kg capsules of 0.5 m, so the tip reaches 1.0 m from the pivot. Gravity on the second link
# alone gives up to m*g*l/2 = 2.45 N*m at the elbow; the motor's 1.5 N*m is 61% of that, and the
# shoulder has no motor at all, so the arm cannot be pushed up and has to swing, pumping energy
# through the elbow. timestep 0.01 s with frame_skip 5 gives the 0.05 s control step; 200 steps
# make 10 s.
LINK_LENGTH_M = 0.5
TORQUE_LIMIT_NM = 1.5
_MJCF = f"""
<mujoco model="acrobot">
  <option timestep="0.01"/>
  <worldbody>
    <body name="link1" pos="0 0 1">
      <joint name="shoulder" type="hinge" axis="0 1 0" damping="0.01"/>
      <geom type="capsule" fromto="0 0 0 0 0 -{LINK_LENGTH_M}" size="0.02" mass="1"/>
      <body name="link2" pos="0 0 -{LINK_LENGTH_M}">
        <joint name="elbow" type="hinge" axis="0 1 0" damping="0.01"/>
        <geom type="capsule" fromto="0 0 0 0 0 -{LINK_LENGTH_M}" size="0.02" mass="1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="elbow" gear="{TORQUE_LIMIT_NM}" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""

_FRAME_SKIP = 5
_MAX_STEPS = 200
_HOLD_STEPS = 10  # 0.5 s above the target
# Tip height above the pivot needed for success; the most it can reach is 2 * LINK_LENGTH_M = 1.0.
_TARGET_HEIGHT_M = 0.75
_ACTION_COST = 0.01
_START_NOISE = 0.1
_MAX_SPEED_RAD_S = 10.0
_PI = 3.141592653589793


def _tip_height_m(qpos: Tensor) -> Tensor:
    """Tip height above the pivot: -2 * l hanging, +2 * l straight up."""
    q1, q2 = qpos[:, 0], qpos[:, 1]
    return -LINK_LENGTH_M * (q1.cos() + (q1 + q2).cos())


_HINGE_COLUMNS = (0, 1)  # unlimited hinges: shoulder and elbow (neither `limited` in the MJCF)


class _Acrobot:
    mjcf = _MJCF
    obs_dim = 6  # cos/sin of both joints, then both velocities
    act_dim = 1
    max_steps = _MAX_STEPS
    frame_skip = _FRAME_SKIP
    hold_steps = _HOLD_STEPS
    dense_sibling = "acrobot_swingup_dense"
    extra_dim = 0
    name: str

    def __init__(self) -> None:
        self.start_box = Box(
            torch.tensor([-_PI, -_PI, -_MAX_SPEED_RAD_S, -_MAX_SPEED_RAD_S]),
            torch.tensor([_PI, _PI, _MAX_SPEED_RAD_S, _MAX_SPEED_RAD_S]),
        )

    def sample_start(self, n: int, gen: torch.Generator) -> StartStates:
        noise = (torch.rand(n, 4, generator=gen) * 2 - 1) * _START_NOISE
        return StartStates(noise[:, :2], noise[:, 2:], torch.zeros(n, 0))

    def canonical_start(self) -> StartStates:
        return StartStates(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 0))

    def observe(self, sim: BatchSim, extra: Tensor) -> Tensor:
        q = sim.qpos
        return torch.cat([q.cos(), q.sin(), sim.qvel], dim=1)

    def terminated(self, sim: BatchSim, extra: Tensor) -> Tensor:
        return torch.zeros(sim.qpos.shape[0], dtype=torch.bool, device=sim.qpos.device)

    def success_now(self, sim: BatchSim, extra: Tensor) -> Tensor:
        return _tip_height_m(sim.qpos) > _TARGET_HEIGHT_M

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


@TASKS.register("acrobot_swingup_sparse")
class AcrobotSwingupSparse(_Acrobot):
    name = "acrobot_swingup_sparse"

    def reward(self, sim: BatchSim, extra: Tensor, action: Tensor) -> Tensor:
        return self.success_now(sim, extra).float() - self._action_cost(action)


@TASKS.register("acrobot_swingup_dense")
class AcrobotSwingupDense(_Acrobot):
    name = "acrobot_swingup_dense"

    def reward(self, sim: BatchSim, extra: Tensor, action: Tensor) -> Tensor:
        height01 = (
            _tip_height_m(sim.qpos) / (2 * LINK_LENGTH_M) + 1
        ) / 2  # 0 hanging, 1 up
        return height01 - self._action_cost(action)
