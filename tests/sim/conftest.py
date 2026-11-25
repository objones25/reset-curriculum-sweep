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
