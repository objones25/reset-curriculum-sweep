"""Vectorised episode logic on top of `BatchSim`: clocks, truncation, hold-time success."""

from typing import Protocol

import torch
from torch import Tensor

from resetsweep.sim.batch import BatchSim
from resetsweep.sim.dynamics import step_task
from resetsweep.tasks.base import InitsExtra, Task
from resetsweep.types import (
    InvalidStartState,
    Outcomes,
    StartStates,
    StepCounter,
    StepOut,
)

VISIT_EVERY_DEFAULT = 8


class _ResetMethod(Protocol):
    """The slice of the Method protocol (defined later) that the env calls."""

    name: str

    def reset(
        self, env_ids: Tensor, outcomes: Outcomes | None
    ) -> tuple[StartStates, Tensor]: ...

    def offer_visited(
        self,
        env_ids: Tensor,
        qpos: Tensor,
        qvel: Tensor,
        extra: Tensor,
        ep_step: Tensor,
    ) -> None: ...


def _check_generator_device(gen: torch.Generator, device: str) -> None:
    want = torch.device(device)
    if gen.device.type != want.type or (
        gen.device.index is not None
        and want.index is not None
        and gen.device.index != want.index
    ):
        raise ValueError(f"generator is on {gen.device}, but the env device is {want}")


class VecEnv:
    """`num_envs` copies of `task`, auto-reset through the bound method.

    No `.item()`/`.cpu()` on the per-step path, but `done.nonzero()` syncs the host
    once per step (it returns a variable-size result). Validating the method's start
    states adds a second sync, only on steps where some env finished.
    """

    def __init__(
        self,
        task: Task,
        *,
        num_envs: int,
        device: str,
        counter: StepCounter,
        gen: torch.Generator,
        visit_every: int = VISIT_EVERY_DEFAULT,
    ) -> None:
        if num_envs <= 0 or visit_every <= 0:
            raise ValueError(
                f"num_envs and visit_every must be positive, got {num_envs}, {visit_every}"
            )
        if task.max_steps <= 0 or task.hold_steps <= 0:
            raise ValueError(
                f"max_steps and hold_steps must be positive, got {task.max_steps}, {task.hold_steps}"
            )
        _check_generator_device(gen, device)
        self.task = task
        self.num_envs = num_envs
        self.device = device
        self._counter = counter
        self._gen = gen
        self._visit_every = visit_every
        self.sim = BatchSim(
            task.mjcf, nworld=num_envs, device=device, frame_skip=task.frame_skip
        )
        self.extra = torch.zeros(num_envs, task.extra_dim, device=device)
        self.ep_step = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._ret = torch.zeros(num_envs, device=device)
        self._hold = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._success = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._reached = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._first_reward = torch.full(
            (num_envs,), -1, dtype=torch.long, device=device
        )
        self._full = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._start_step = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._ic_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._t = 0  # vector steps completed
        self._method: _ResetMethod | None = None

    def bind(self, method: _ResetMethod) -> None:
        self._method = method

    @property
    def ic_ids(self) -> Tensor:
        """`(num_envs,)` ic id of each env's current episode (a copy)."""
        return self._ic_ids.clone()

    def reset_all(self, *, stagger: bool = True) -> Tensor:
        """Start every env from the method. With `stagger` (training), clocks get random
        offsets so episodes end at different times and these first episodes are not `full`.
        Without it (evaluation), every clock starts at 0, so each env plays one full episode
        from the state the method returned."""
        method = self._require_method()
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._start_episodes(method, env_ids, None, full=not stagger)
        if not stagger:
            return self.task.observe(self.sim, self.extra)
        offsets = torch.randint(
            0,
            self.task.max_steps,
            (self.num_envs,),
            generator=self._gen,
            device=self.device,
        )
        self.ep_step.copy_(offsets)
        self._start_step.copy_(self._t - offsets)
        return self.task.observe(self.sim, self.extra)

    def step(self, action: Tensor) -> StepOut:
        method = self._require_method()
        if action.shape != self.sim.ctrl.shape:
            raise ValueError(
                f"action shape {tuple(action.shape)}, expected {tuple(self.sim.ctrl.shape)}"
            )
        self._counter.add_train(self.num_envs)
        step_task(self.task, self.sim, self.extra, action)
        self._t += 1
        self.ep_step += 1

        reward = self.task.reward(self.sim, self.extra, action)
        self._ret += reward
        self._first_reward = torch.where(
            (reward > 0) & ~self._reached, self.ep_step, self._first_reward
        )
        self._reached |= reward > 0
        terminated = self.task.terminated(self.sim, self.extra)
        success_now = self.task.success_now(self.sim, self.extra)
        self._hold = torch.where(
            success_now, self._hold + 1, torch.zeros_like(self._hold)
        )
        self._success |= self._hold >= self.task.hold_steps
        truncated = (self.ep_step >= self.task.max_steps) & ~terminated
        # clone: `observe` may return a view of sim state, which the reset below overwrites
        final_obs = self.task.observe(self.sim, self.extra).clone()

        if self._t % self._visit_every == 0:
            qpos, qvel = self.sim.get_state(
                torch.arange(self.num_envs, device=self.device)
            )
            method.offer_visited(
                torch.arange(self.num_envs, device=self.device),
                qpos,
                qvel,
                self.extra.clone(),
                self.ep_step.clone(),
            )

        done = terminated | truncated
        env_ids = done.nonzero(as_tuple=True)[0]
        outcomes = None
        obs = final_obs
        if env_ids.numel() > 0:
            outcomes = self._outcomes(env_ids)
            self._start_episodes(method, env_ids, outcomes, full=True)
            obs = self.task.observe(self.sim, self.extra)
        return StepOut(
            obs,
            reward,
            terminated,
            truncated,
            final_obs,
            self._ic_ids.clone(),
            outcomes,
        )

    def _outcomes(self, env_ids: Tensor) -> Outcomes:
        return Outcomes(
            env_ids=env_ids,
            ic_ids=self._ic_ids[env_ids],
            ret=self._ret[env_ids],
            success=self._success[env_ids],
            ep_len=self.ep_step[env_ids],
            reached_reward=self._reached[env_ids],
            first_reward_step=self._first_reward[env_ids],
            full=self._full[env_ids],
            start_step=self._start_step[env_ids],
        )

    def _start_episodes(
        self,
        method: _ResetMethod,
        env_ids: Tensor,
        outcomes: Outcomes | None,
        *,
        full: bool,
    ) -> None:
        states, ic_ids = method.reset(env_ids, outcomes)
        if states.n != env_ids.numel() or ic_ids.shape != env_ids.shape:
            raise ValueError(
                f"method {method.name!r} returned {states.n} states and {tuple(ic_ids.shape)} ic_ids "
                f"for {env_ids.numel()} envs"
            )
        self._check_valid(method.name, states)
        extra = states.extra
        if isinstance(self.task, InitsExtra):
            extra = self.task.init_extra(states, self._gen)
            if extra.shape != (states.n, self.task.extra_dim):
                raise ValueError(
                    f"task {self.task.name!r} init_extra returned shape {tuple(extra.shape)}, "
                    f"expected {(states.n, self.task.extra_dim)}"
                )
        self.sim.set_state(env_ids, states.qpos, states.qvel)
        self.extra[env_ids] = extra
        self._ic_ids[env_ids] = ic_ids
        self.ep_step[env_ids] = 0
        self._ret[env_ids] = 0.0
        self._hold[env_ids] = 0
        self._success[env_ids] = False
        self._reached[env_ids] = False
        self._first_reward[env_ids] = -1
        self._full[env_ids] = full
        self._start_step[env_ids] = self._t

    def _check_valid(self, method_name: str, states: StartStates) -> None:
        finite = (
            torch.isfinite(states.qpos).all(dim=1)
            & torch.isfinite(states.qvel).all(dim=1)
            & torch.isfinite(states.extra).all(dim=1)
        )
        valid = finite & self.task.is_valid(states)
        if bool(valid.all()):
            return
        row = int((~valid).nonzero()[0, 0])
        reason = (
            "non-finite values"
            if not bool(finite[row])
            else "rejected by task.is_valid"
        )
        raise InvalidStartState(method_name, row, reason)

    def _require_method(self) -> _ResetMethod:
        if self._method is None:
            raise RuntimeError(
                "no method bound; call bind(method) before reset_all/step"
            )
        return self._method
