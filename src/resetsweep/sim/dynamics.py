"""The one place a task's dynamics are stepped: every stepper calls `step_task`.

`VecEnv`, the test planners and rollout helpers all go through it, so a task whose physics
need more than the MJCF (a pit's pull and drag) behaves the same however it is driven. The
actuators stay clamped to their MJCF `ctrlrange` by the simulator; a task adds forces only
through the optional `applied_force` hook.
"""

from torch import Tensor

from resetsweep.sim.batch import BatchSim
from resetsweep.tasks.base import AppliesForce, Task


def step_task(task: Task, sim: BatchSim, extra: Tensor, action: Tensor) -> None:
    """Advance `sim` one control step under `action`, plus the task's applied force if it has one.

    The force is recomputed before every physics substep, not once per control step: drag
    depends on velocity, and over a 0.05 s control step a 10 N*s/m drag changes the velocity
    by a large fraction, so a force frozen at the step's start would be a different system.
    """
    if isinstance(task, AppliesForce):
        sim.step(action, applied_force=lambda s: task.applied_force(s, extra))
    else:
        sim.step(action)
