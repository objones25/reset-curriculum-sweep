"""Scripted solvers for task tests: prove each task is solvable, not only hard."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from torch import Tensor

from resetsweep.sim.batch import BatchSim
from resetsweep.sim.env import VecEnv
from resetsweep.tasks.base import Task
from resetsweep.types import Outcomes, StartStates, StepCounter

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


@dataclass
class _FixedStartMethod:
    """Reset rule that hands out `states` (one row, tiled) or draws from the task's own `sample_start`."""

    task: Task
    gen: torch.Generator
    states: StartStates | None = None
    name: str = "task_start"

    def reset(
        self, env_ids: Tensor, outcomes: Outcomes | None
    ) -> tuple[StartStates, Tensor]:
        n = env_ids.numel()
        if self.states is None:
            return self.task.sample_start(n, self.gen), env_ids.clone()
        return self.states.index(torch.zeros(n, dtype=torch.long)), env_ids.clone()

    def offer_visited(
        self,
        env_ids: Tensor,
        qpos: Tensor,
        qvel: Tensor,
        extra: Tensor,
        ep_step: Tensor,
    ) -> None:
        return None


# A policy sees the env (for state-based planners) and the observation.
Policy = Callable[[VecEnv, Tensor], Tensor]


def run_first_episode(
    task: Task,
    policy: Policy,
    start: StartStates | None = None,
    *,
    num_envs: int = 1,
    seed: int = 0,
) -> Outcomes:
    """Run `policy` through VecEnv for exactly one full episode from every env's first start.

    Starts come from `start` (one row, tiled) or `task.sample_start`. `reset_all` staggers the
    episode clocks, so they are zeroed here: every env's first episode is then a whole one.
    Returns the outcomes recorded when that episode ends, the env's own held-success flags.
    """
    gen = torch.Generator().manual_seed(seed)
    env = VecEnv(
        task,
        num_envs=num_envs,
        device="cpu",
        counter=StepCounter(budget=2 * num_envs * task.max_steps),
        gen=gen,
    )
    env.bind(_FixedStartMethod(task, gen, start))
    obs = env.reset_all()
    env.ep_step.zero_()
    for _ in range(task.max_steps - 1):
        obs = env.step(policy(env, obs)).obs
    outcomes = env.step(policy(env, obs)).outcomes
    if outcomes is None or not bool((outcomes.ep_len == task.max_steps).all()):
        raise RuntimeError("the first episode did not end after max_steps steps")
    return outcomes


def first_episode_success(
    task: Task, policy: Policy, num_envs: int, *, seed: int
) -> Tensor:
    """Held-success flag of each env's first full episode from `task.sample_start`."""
    return run_first_episode(task, policy, num_envs=num_envs, seed=seed).success


def _solve_dare(
    a: np.ndarray,
    b: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    *,
    iters_max: int = 20000,
) -> np.ndarray:
    """Discrete LQR gain by iterating the Riccati recursion to a fixed point (no scipy)."""
    p = q.copy()
    for _ in range(iters_max):
        gain = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
        p_next = q + a.T @ p @ (a - b @ gain)
        if np.abs(p_next - p).max() < 1e-10:
            return gain
        p = p_next
    raise RuntimeError(f"Riccati recursion did not converge in {iters_max} iterations")


class CartpoleSwingup:
    """Energy pump for the pole plus cart centring far from upright, LQR near it.

    Observations are (x, cos theta, sin theta, x_dot, theta_dot); actions are in [-1, 1].
    The LQR gain comes from finite differences of the model about upright, over one control step.
    """

    def __init__(self, mjcf: str, *, frame_skip: int = 5) -> None:
        model = mujoco.MjModel.from_xml_string(mjcf)  # pyright: ignore[reportAttributeAccessIssue] # no stub for the C-extension class
        data = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]
        data.qpos[:] = [0.0, math.pi]
        mujoco.mj_forward(model, data)  # pyright: ignore[reportAttributeAccessIssue]
        a1 = np.zeros((4, 4))
        b1 = np.zeros((4, 1))
        mujoco.mjd_transitionFD(model, data, 1e-6, True, a1, b1, None, None)  # pyright: ignore[reportAttributeAccessIssue]
        a = np.linalg.matrix_power(a1, frame_skip)
        b = sum(np.linalg.matrix_power(a1, k) @ b1 for k in range(frame_skip))
        gain = _solve_dare(a, b, np.diag([1.0, 20.0, 0.1, 0.5]), np.array([[0.1]]))
        self._gain = torch.tensor(gain[0], dtype=torch.float32)
        self._inertia = float(model.dof_M0[1])  # pole about its hinge
        self._mgl = (
            float(model.body_mass[2])
            * _GRAVITY_M_S2
            * float(abs(model.body_ipos[2, 2]))
        )

    def __call__(self, obs: Tensor) -> Tensor:
        x, cos_t, sin_t, xd, td = obs.unbind(dim=1)
        theta = torch.atan2(sin_t, cos_t)
        phi = (
            torch.remainder(theta, 2 * math.pi) - math.pi
        )  # signed distance from upright
        energy = 0.5 * self._inertia * td.square() - self._mgl * cos_t  # target +mgl
        pump = torch.sign((self._mgl - energy) * td * cos_t)
        pump = torch.where(td.abs() < 1e-3, torch.ones_like(pump), pump)
        swing = pump - 0.3 * x - 0.3 * xd
        state = torch.stack([x, phi, xd, td], dim=1)
        catch = -(state @ self._gain)
        near = (phi.abs() < 0.4) & (td.abs() < 6.0) & (x.abs() < 1.5)
        return torch.where(near, catch, swing).clamp(-1.0, 1.0).unsqueeze(1)


class SamplingPlanner:
    """Receding-horizon CEM over action sequences, scored by a dense task's reward.

    Generic: it needs only a task's `mjcf`, `frame_skip`, `act_dim`, `extra_dim` and `reward`.
    Candidates are rolled forward in its own `BatchSim`, whose state is set to each env's current
    state, so the real env is never touched. Call it with the envs' `(qpos, qvel, extra)`; it
    returns `(num_envs, act_dim)` actions in `[-action_limit, action_limit]`. The plan is warm-started
    from the previous call, so use one planner per run and do not share it across env batches.
    """

    def __init__(
        self,
        task: Task,
        *,
        num_envs: int,
        seed: int,
        horizon: int = 24,
        knot: int = 4,
        samples: int = 128,
        elites: int = 12,
        iterations: int = 3,
        init_std: float = 0.6,
        fine_std: float = 0.2,
        action_limit: float = 1.0,
    ) -> None:
        if not 0 < elites <= samples:
            raise ValueError(f"need 0 < elites <= samples, got {elites}, {samples}")
        if num_envs <= 0 or iterations <= 0 or knot <= 0 or horizon % knot != 0:
            raise ValueError(
                "num_envs, iterations and knot must be positive and horizon a multiple of knot, "
                f"got {num_envs}, {iterations}, {knot}, {horizon}"
            )
        self._task = task
        self._n, self._k, self._h, self._knot = num_envs, samples, horizon, knot
        self._elites, self._iterations = elites, iterations
        self._init_std, self._fine_std, self._limit = init_std, fine_std, action_limit
        self._gen = torch.Generator().manual_seed(seed)
        self._sim = BatchSim(
            task.mjcf,
            nworld=num_envs * samples,
            device="cpu",
            frame_skip=task.frame_skip,
        )
        self._mean = torch.zeros(num_envs, horizon, task.act_dim)

    def __call__(self, qpos: Tensor, qvel: Tensor, extra: Tensor) -> Tensor:
        if qpos.shape[0] != self._n:
            raise ValueError(f"planner built for {self._n} envs, got {qpos.shape[0]}")
        rep = torch.arange(self._n).repeat_interleave(self._k)
        q, v, e = qpos[rep], qvel[rep], extra[rep]
        mean = self._mean
        for it in range(self._iterations):
            shrink = 0.5**it  # CEM: later iterations search closer to the elite mean
            # Mostly noise held for `knot` control steps: white noise per step averages out and never
            # finds a coherent push, while a handful of knots per plan is easy to search. A little
            # per-step noise on top gives the fast corrections that balancing needs.
            knots = torch.randn(
                self._n,
                self._k,
                self._h // self._knot,
                mean.shape[2],
                generator=self._gen,
            )
            fine = torch.randn(
                knots.shape[0],
                knots.shape[1],
                self._h,
                mean.shape[2],
                generator=self._gen,
            )
            noise = (
                self._init_std * knots.repeat_interleave(self._knot, dim=2)
                + self._fine_std * fine
            )
            noise[:, 0] = (
                0.0  # candidate 0 is the current mean, so the plan never gets worse
            )
            cand = (mean.unsqueeze(1) + shrink * noise).clamp(-self._limit, self._limit)
            score = self._rollout(cand, q, v, e)
            best = score.topk(self._elites, dim=1).indices
            mean = cand[torch.arange(self._n).unsqueeze(1), best].mean(dim=1)
        first = mean[:, 0].clone()
        self._mean = torch.cat(
            [mean[:, 1:], mean[:, -1:]], dim=1
        )  # shift the plan one step
        return first

    def _rollout(self, cand: Tensor, q: Tensor, v: Tensor, e: Tensor) -> Tensor:
        flat = cand.reshape(self._n * self._k, self._h, -1)
        self._sim.set_state(torch.arange(self._n * self._k), q, v)
        total = torch.zeros(self._n * self._k)
        for t in range(self._h):
            self._sim.step(flat[:, t])
            total += self._task.reward(self._sim, e, flat[:, t])
        return total.reshape(self._n, self._k)
