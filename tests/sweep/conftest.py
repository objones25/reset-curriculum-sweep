"""Fakes for the calibration and sweep-loop tests: a clock that only moves when the fake pool
says time passed, and a pool that "runs" a job by advancing that clock and writing the JSONL
lines the real runner (`resetsweep.sweep.pool.train_job`) would write. No process is spawned,
nothing sleeps, nothing trains.

Fake world (every number here is ours, picked so hand-computed values stay simple):
- per-job steps/s = min(num_envs, saturate_envs) * 100 / sqrt(n_workers), so the aggregate over
  n concurrent jobs grows like sqrt(n) and stops growing past `saturate_envs`;
- each job also pays `overhead_s` (spawn, compile, evals);
- final eval return 100 at lr 3e-4 and 50 at lr 1e-3; kl_stopped 0.6 at lr 1e-3, 0.1 at 3e-4;
- AUC of a sweep job is a hash of its arm id in [0, 1).
"""

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from resetsweep.sweep.pool import RunnerConfig, failed_result, run_log_path
from resetsweep.sweep.scheduler import Job
from resetsweep.train import RunResult, RunSpec


class FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@dataclass
class Submission:
    job: Job
    spec: RunSpec
    n_workers: int
    session: int


@dataclass
class FakeWorld:
    clock: FakeClock
    saturate_envs: int = 10**9
    overhead_s: float = 2.0
    fails: Callable[[RunSpec], bool] = lambda spec: False
    slowdown: Callable[[RunSpec], float] = lambda spec: 1.0  # actual / modelled duration
    enforce_timeouts: bool = True  # False: a pool that never kills, to test callers' own stops
    submissions: list[Submission] = field(default_factory=list)
    session: int = 0

    def steps_per_s(self, spec: RunSpec, n_workers: int) -> float:
        return min(spec.num_envs, self.saturate_envs) * 100.0 / math.sqrt(n_workers)

    def duration_s(self, spec: RunSpec, n_workers: int) -> float:
        modelled = spec.step_budget / self.steps_per_s(spec, n_workers) + self.overhead_s
        return modelled * self.slowdown(spec)

    def finish(self, spec: RunSpec, n_workers: int, cfg: RunnerConfig) -> RunResult:
        wall_s = self.duration_s(spec, n_workers)
        if self.fails(spec):
            return failed_result(spec, "RuntimeError('fake CUDA OOM')", wall_s=wall_s)
        self._write_log(spec, n_workers, cfg)
        h = int(hashlib.sha256(spec.arm_id().encode()).hexdigest()[:8], 16)
        auc = (h % 1000) / 1000.0
        return RunResult(
            arm_id=spec.arm_id(),
            seed=spec.seed,
            status="ok",
            error=None,
            auc=auc,
            final_success=auc,
            reached_reward=auc > 0,
            steps_to_first_reward=1 if auc > 0 else None,
            ever_succeeded_share=0.5,
            env_steps_train=spec.step_budget,
            env_steps_extra={},
            wall_s=wall_s,
        )

    def _write_log(self, spec: RunSpec, n_workers: int, cfg: RunnerConfig) -> None:
        batch = spec.num_envs * spec.rollout_len
        iters = max(1, math.ceil(spec.step_budget / batch))
        slow_lr = math.isclose(spec.ppo.lr, 1e-3)
        lines = [{"type": "log", "step": 0, "eval/return": 0.0, "eval/success": 0.0}]
        for i in range(iters):
            lines.append(
                {
                    "type": "log",
                    "step": (i + 1) * batch,
                    "perf/steps_per_s": self.steps_per_s(spec, n_workers),
                    "ppo/kl_stopped": 0.6 if slow_lr else 0.1,
                    "method/ic_churn": 0.8,
                }
            )
        lines[-1]["eval/return"] = 50.0 if slow_lr else 100.0
        path = run_log_path(cfg.runs_dir, spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")


class FakePool:
    def __init__(self, world: FakeWorld, n_workers: int, device: str, cfg: RunnerConfig) -> None:
        self._world = world
        self._n = n_workers
        self._device = device
        self._cfg = cfg
        # (arm_id, seed) -> (job, spec, end time, timed out)
        self._running: dict[tuple[str, int], tuple[Job, RunSpec, float, bool]] = {}
        self.closed = False

    def submit(self, job: Job, spec: RunSpec, *, timeout_s: float) -> None:
        assert not self.closed
        assert len(self._running) < self._n, "no free worker"
        key = (spec.arm_id(), spec.seed)
        assert key not in self._running
        assert timeout_s > 0
        assert spec.device == self._device
        self._world.submissions.append(
            Submission(job=job, spec=spec, n_workers=self._n, session=self._world.session)
        )
        now = self._world.clock.t
        finish = now + self._world.duration_s(spec, self._n)
        timed_out = self._world.enforce_timeouts and finish > now + timeout_s
        end = now + timeout_s if timed_out else finish
        self._running[key] = (job, spec, end, timed_out)

    def poll(self, timeout_s: float) -> list[RunResult]:
        clock = self._world.clock
        if not self._running:
            clock.t += timeout_s
            return []
        t_next = min(end for _, _, end, _ in self._running.values())
        if t_next > clock.t + timeout_s:
            clock.t += timeout_s
            return []
        clock.t = max(clock.t, t_next)
        done = [key for key, (_, _, end, _) in self._running.items() if end <= clock.t]
        out: list[RunResult] = []
        for key in done:
            _, spec, end, timed_out = self._running.pop(key)
            if timed_out:
                out.append(failed_result(spec, "timed out; worker killed", wall_s=end))
            else:
                out.append(self._world.finish(spec, self._n, self._cfg))
        return out

    def kill(self, job: Job) -> None:
        (key,) = [k for k, (j, _, _, _) in self._running.items() if j == job]
        self._running.pop(key)

    def running(self) -> list[Job]:
        return [job for job, _, _, _ in self._running.values()]

    def free_slots(self) -> int:
        return self._n - len(self._running)

    def close(self) -> None:
        self._running.clear()
        self.closed = True


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(t=1000.0)


@pytest.fixture
def world(clock: FakeClock) -> FakeWorld:
    return FakeWorld(clock=clock)


@pytest.fixture
def pool_factory(world: FakeWorld) -> Callable[[int, str, RunnerConfig], FakePool]:
    def make(n_workers: int, device: str, cfg: RunnerConfig) -> FakePool:
        return FakePool(world, n_workers, device, cfg)

    return make
