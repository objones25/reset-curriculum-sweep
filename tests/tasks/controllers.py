"""Scripted solvers for task tests: prove each task is solvable, not only hard."""

import math

import mujoco
import torch
from torch import Tensor

_GRAVITY_M_S2 = 9.81


class PendulumSwingup:
    """Energy pump far from upright, PD catch near it. Observations are (cos, sin, omega)."""

    def __init__(self, mjcf: str, torque_limit: float) -> None:
        model = mujoco.MjModel.from_xml_string(mjcf)  # pyright: ignore[reportAttributeAccessIssue] # no stub for the C-extension class
        self._inertia = float(model.dof_M0[0])  # about the hinge
        self._mgl = (
            float(model.body_mass[1])
            * _GRAVITY_M_S2
            * float(abs(model.body_ipos[1, 2]))
        )
        self._limit = torque_limit

    def __call__(self, obs: Tensor) -> Tensor:
        cos_t, sin_t, omega = obs[:, 0], obs[:, 1], obs[:, 2]
        theta = torch.atan2(sin_t, cos_t)
        dev = (
            torch.remainder(theta, 2 * math.pi) - math.pi
        )  # signed distance from upright
        energy = (
            0.5 * self._inertia * omega.square() - self._mgl * cos_t
        )  # target: +mgl
        pump = torch.sign((self._mgl - energy) * omega)
        pump = torch.where(
            omega.abs() < 1e-3, torch.ones_like(pump), pump
        )  # kick off from rest
        catch = -(25.0 * dev + 4.0 * omega)
        near = (dev.abs() < 0.6) & (omega.abs() < 6.0)
        u = torch.where(near, catch, pump * self._limit)
        return u.clamp(-self._limit, self._limit).unsqueeze(1)
