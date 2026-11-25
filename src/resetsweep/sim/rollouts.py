"""Off-policy simulator work for reset methods: random walks and zero-action hold screens.

Both run on their own `BatchSim`, so the training envs are never touched, and both step
through `step_task`, so a task's applied forces (the trap's pull) act exactly as in training.
Every step is charged to the `StepCounter` under its extra kind.
"""

import mujoco
import torch
from torch import Tensor

from resetsweep.sim.batch import BatchSim
from resetsweep.sim.dynamics import step_task
from resetsweep.tasks.base import Task
from resetsweep.types import StartStates, StepCounter

SPEED_MAX_DEFAULT = 5.0


class Rollouts:
    """Random walks and hold screens on a private `BatchSim` of `nworld_max` worlds.

    Starts are processed in chunks of at most `nworld_max`. The sim is built on first use.
    `speed_max` bounds `max |qvel|` over a hold screen, in the task's own velocity units.
    """

    def __init__(
        self,
        task: Task,
        *,
        device: str,
        counter: StepCounter,
        gen: torch.Generator,
        nworld_max: int = 4096,
        speed_max: float = SPEED_MAX_DEFAULT,
    ) -> None:
        if nworld_max <= 0 or speed_max <= 0:
            raise ValueError(
                f"nworld_max and speed_max must be positive, got {nworld_max}, {speed_max}"
            )
        self._task = task
        self._device = device
        self._counter = counter
        self._gen = gen
        self._nworld_max = nworld_max
        self._speed_max = speed_max
        self._sim: BatchSim | None = None
        self._extra = torch.zeros(nworld_max, task.extra_dim, device=device)
        model = mujoco.MjModel.from_xml_string(task.mjcf)  # pyright: ignore[reportAttributeAccessIssue] # mujoco ships no stub for the C-extension class
        self._control_dt_s = float(model.opt.timestep) * task.frame_skip

    def random_walk(
        self, starts: StartStates, steps: int, action_hold: int
    ) -> StartStates:
        """Final states after `steps` control steps of uniform random actions in [-1, 1]^act_dim,
        each redrawn every `action_hold` steps. Charges `starts.n * steps` as kind "walk".

        The result is not filtered: it may be outside `task.start_box` or non-finite, and the
        caller must check `task.is_valid` before using it as a start.
        """
        if steps <= 0 or action_hold <= 0:
            raise ValueError(
                f"steps and action_hold must be positive, got {steps}, {action_hold}"
            )
        self._counter.add_extra(starts.n * steps, "walk")
        out: list[StartStates] = []
        for chunk in self._chunks(starts):
            sim = self._load(chunk)
            action = torch.zeros(sim.ctrl.shape, device=self._device)
            for t in range(steps):
                if t % action_hold == 0:
                    action = self._random_action(sim.ctrl.shape)
                step_task(self._task, sim, self._extra, action)
            out.append(self._read(sim, chunk.n))
        return StartStates.cat(out)

    def hold_screen(
        self, starts: StartStates, seconds: float, *, speed_max: float | None = None
    ) -> Tensor:
        """True for starts that stay put under zero action for `seconds`: the task never
        terminates and `max |qvel|` never exceeds `speed_max` (the constructor's bound when
        None). Charges every start for the full duration (kind "screen"), also those that
        fail early.
        """
        if seconds <= 0:
            raise ValueError(f"seconds must be positive, got {seconds}")
        if speed_max is None:
            speed_max = self._speed_max
        if speed_max <= 0:
            raise ValueError(f"speed_max must be positive, got {speed_max}")
        steps = max(1, round(seconds / self._control_dt_s))
        self._counter.add_extra(starts.n * steps, "screen")
        keep: list[Tensor] = []
        for chunk in self._chunks(starts):
            sim = self._load(chunk)
            zero = torch.zeros(sim.ctrl.shape, device=self._device)
            ok = torch.ones(sim.qpos.shape[0], dtype=torch.bool, device=self._device)
            for _ in range(steps):
                step_task(self._task, sim, self._extra, zero)
                ok &= ~self._task.terminated(sim, self._extra)
                ok &= sim.qvel.abs().amax(dim=1) <= speed_max
            keep.append(ok[: chunk.n])
        return torch.cat(keep)

    def _chunks(self, starts: StartStates) -> list[StartStates]:
        n = starts.n
        return [
            StartStates(
                starts.qpos[i : i + self._nworld_max],
                starts.qvel[i : i + self._nworld_max],
                starts.extra[i : i + self._nworld_max],
            )
            for i in range(0, n, self._nworld_max)
        ]

    def _load(self, chunk: StartStates) -> BatchSim:
        if self._sim is None:
            self._sim = BatchSim(
                self._task.mjcf,
                nworld=self._nworld_max,
                device=self._device,
                frame_skip=self._task.frame_skip,
            )
        sim = self._sim
        rows = torch.arange(chunk.n, device=self._device)
        sim.set_state(rows, chunk.qpos.to(self._device), chunk.qvel.to(self._device))
        self._extra[: chunk.n] = chunk.extra.to(self._device)
        return sim

    def _read(self, sim: BatchSim, n: int) -> StartStates:
        rows = torch.arange(n, device=self._device)
        qpos, qvel = sim.get_state(rows)
        return StartStates(qpos, qvel, self._extra[:n].clone())

    def _random_action(self, shape: torch.Size) -> Tensor:
        u = torch.rand(shape, generator=self._gen, device=self._device)
        return u * 2.0 - 1.0
