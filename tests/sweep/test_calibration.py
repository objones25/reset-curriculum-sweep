"""Calibration (spec 5.5 and 7.1, Task 11 ruling): PPO lr comparison on the dense sibling,
steps/s over the num_envs x n_workers grid, performance probes, the $10 cap, reuse on restart.

Runs through the fake pool of tests/sweep/conftest.py; prices give 0.01 USD per pod-second.
"""

import json
import math
from pathlib import Path
from typing import Any

import pytest
from resetsweep.sweep.budget import BudgetGuard
from resetsweep.sweep.calibration import (
    CalibrationError,
    CalibrationSettings,
    calibrate,
    choose_plan,
)
from resetsweep.sweep.pool import RunnerConfig

GPU_USD_PER_H = 36.0  # 0.01 USD/s


def _settings(**kw: Any) -> CalibrationSettings:
    base: dict[str, Any] = {
        "num_envs_grid": (64, 128, 256),
        "workers_grid": (1, 2, 4),
        "rollout_len": 8,
        "config_steps": 4000,
        "throughput_iters": 2,
        "eval_n": 16,
        "device": "cpu",
        "cap_usd": 10.0,
        "job_timeout_s": 600.0,
        "poll_s": 5.0,
    }
    return CalibrationSettings(**{**base, **kw})


def _guard(tmp_path: Path, clock: Any) -> BudgetGuard:
    return BudgetGuard(
        cap_usd=100.0,
        margin_usd=5.0,
        gpu_usd_per_h=GPU_USD_PER_H,
        volume_usd_per_h=0.0,
        ledger_path=tmp_path / "ledger.jsonl",
        clock=clock,
    )


def _runner_cfg(tmp_path: Path) -> RunnerConfig:
    return RunnerConfig(project="p", group="g", wandb_mode="disabled", runs_dir=tmp_path / "runs")


def _calibrate(tmp_path: Path, clock: Any, pool_factory: Any, families: tuple[str, ...], **kw: Any):
    return calibrate(
        families=families,
        settings=_settings(**kw),
        guard=_guard(tmp_path, clock),
        pool_factory=pool_factory,
        runner_cfg=_runner_cfg(tmp_path),
        path=tmp_path / "calibration.json",
        clock=clock,
    )


def test_calibration_compares_lr_and_keeps_higher_final_return(
    tmp_path: Path, clock: Any, pool_factory: Any
) -> None:
    record = _calibrate(tmp_path, clock, pool_factory, ("pendulum_dense",))

    plan = choose_plan(record, arms_per_family={"pendulum_dense": 10}, settings=_settings())

    configs = {m.lr: (m.final_return, m.kl_stopped_rate) for m in record.config("pendulum_dense")}
    assert configs == {1e-3: (50.0, pytest.approx(0.6)), 3e-4: (100.0, pytest.approx(0.1))}
    assert plan.families["pendulum_dense"].ppo.lr == 3e-4


def test_calibration_runs_uniform_on_the_dense_sibling(
    tmp_path: Path, clock: Any, pool_factory: Any, world: Any
) -> None:
    _calibrate(tmp_path, clock, pool_factory, ("pendulum_dense",))

    sweep_like = {
        (s.spec.task, s.spec.method)
        for s in world.submissions
        if not s.job.arm.label().startswith("pendulum_sparse")
    }
    assert sweep_like == {("pendulum_dense", "uniform")}
    assert {s.job.rung for s in world.submissions} == {-1}


def test_calibration_measures_every_num_envs_and_worker_count(
    tmp_path: Path, clock: Any, pool_factory: Any
) -> None:
    record = _calibrate(tmp_path, clock, pool_factory, ("pendulum_dense",))

    grid = record.throughput("pendulum_dense")

    # aggregate over n concurrent jobs = n * N * 100 / sqrt(n) = N * 100 * sqrt(n)
    assert grid == {
        (n_envs, n): pytest.approx(n_envs * 100.0 * math.sqrt(n))
        for n_envs in (64, 128, 256)
        for n in (1, 2, 4)
    }


def test_plan_picks_smallest_num_envs_within_10pct_and_fastest_worker_count(
    tmp_path: Path, clock: Any, pool_factory: Any, world: Any
) -> None:
    world.saturate_envs = 128  # 128 and 256 envs give the same steps/s
    record = _calibrate(tmp_path, clock, pool_factory, ("pendulum_dense",))

    plan = choose_plan(record, arms_per_family={"pendulum_dense": 10}, settings=_settings())

    fam = plan.families["pendulum_dense"]
    assert (plan.n_workers, fam.num_envs) == (4, 128)
    assert fam.steps_per_s == pytest.approx(128 * 100.0 * 2.0)  # aggregate at n=4
    assert fam.calibrated is True


def test_plan_overhead_is_measured_from_config_jobs(
    tmp_path: Path, clock: Any, pool_factory: Any, world: Any
) -> None:
    record = _calibrate(tmp_path, clock, pool_factory, ("pendulum_dense",))

    plan = choose_plan(record, arms_per_family={"pendulum_dense": 10}, settings=_settings())

    # fake jobs pay overhead_s = 2 on top of steps / steps_per_s; the fake pool reports at once
    assert plan.families["pendulum_dense"].overhead_s == pytest.approx(world.overhead_s)


def test_failed_throughput_batch_is_unusable(
    tmp_path: Path, clock: Any, pool_factory: Any, world: Any
) -> None:
    world.fails = lambda spec: spec.num_envs == 256 and spec.seed == 3  # one of 4 workers OOMs
    record = _calibrate(tmp_path, clock, pool_factory, ("pendulum_dense",))

    plan = choose_plan(record, arms_per_family={"pendulum_dense": 10}, settings=_settings())

    # without (256, 4) the best is 2 workers at 256 envs: 256 * 100 * sqrt(2) = 36,204 steps/s,
    # above 4 workers at 128 envs: 128 * 100 * 2 = 25,600
    assert record.throughput("pendulum_dense")[(256, 4)] is None
    assert (plan.n_workers, plan.families["pendulum_dense"].num_envs) == (2, 256)


def test_calibration_saved_and_reused_on_restart(
    tmp_path: Path, clock: Any, pool_factory: Any, world: Any
) -> None:
    first = _calibrate(tmp_path, clock, pool_factory, ("pendulum_dense",))
    n_first = len(world.submissions)

    second = _calibrate(tmp_path, clock, pool_factory, ("pendulum_dense",))

    assert len(world.submissions) == n_first
    assert second == first
    assert json.loads((tmp_path / "calibration.json").read_text())["version"] == 1


def test_calibration_stops_at_calibration_cap(
    tmp_path: Path, clock: Any, pool_factory: Any, world: Any
) -> None:
    # One config batch costs 4000 / (128 * 100 / sqrt 2) + 2 = 2.44 s = 0.0244 USD, so a
    # 0.03 USD cap is reached after the first batch... the second batch starts below the cap
    # (0.0244 < 0.03) and the third does not.
    record = _calibrate(tmp_path, clock, pool_factory, ("pendulum_dense", "arm_dense"), cap_usd=0.03)

    assert record.capped is True
    assert sorted(record.batch_keys()) == ["config/arm_dense", "config/pendulum_dense"]
    assert record.spent_usd >= 0.03


def test_calibration_resumes_where_the_cap_stopped_it(
    tmp_path: Path, clock: Any, pool_factory: Any, world: Any
) -> None:
    _calibrate(tmp_path, clock, pool_factory, ("pendulum_dense",), cap_usd=0.01)
    n_first = len(world.submissions)  # the config batch only

    record = _calibrate(tmp_path, clock, pool_factory, ("pendulum_dense",), cap_usd=10.0)

    keys = [(s.spec.arm_id(), s.spec.seed, s.n_workers) for s in world.submissions]
    assert n_first == 2
    assert len(keys) == len(set(keys))  # nothing ran twice
    assert record.capped is False
    assert len(record.throughput("pendulum_dense")) == 9


def test_probes_record_steps_per_s_and_method_diagnostics(
    tmp_path: Path, clock: Any, pool_factory: Any
) -> None:
    record = _calibrate(tmp_path, clock, pool_factory, ("arm_dense",))

    probes = {m.label: m for m in record.probes()}

    assert sorted(probes) == [
        "arm_sparse/buffer_scored[p_task=0.5]/random",
        "arm_sparse/visited[p_task=0.5]/-",
    ]
    m = probes["arm_sparse/visited[p_task=0.5]/-"]
    assert (m.steps_per_s, m.diagnostics) == (128 * 100.0, {"method/ic_churn": 0.8})


def test_uncalibrated_family_gets_conservative_fallback(
    tmp_path: Path, clock: Any, pool_factory: Any
) -> None:
    record = _calibrate(tmp_path, clock, pool_factory, ("pendulum_dense",))

    plan = choose_plan(
        record, arms_per_family={"pendulum_dense": 10, "claw": 3}, settings=_settings()
    )

    claw = plan.for_task("claw")
    assert claw.calibrated is False
    assert claw.num_envs == 128  # middle of the grid
    assert claw.steps_per_s == plan.families["pendulum_dense"].steps_per_s


def test_plan_for_sparse_task_uses_its_dense_sibling(
    tmp_path: Path, clock: Any, pool_factory: Any
) -> None:
    record = _calibrate(tmp_path, clock, pool_factory, ("pendulum_dense",))

    plan = choose_plan(record, arms_per_family={"pendulum_dense": 10}, settings=_settings())

    assert plan.for_task("pendulum_sparse") == plan.families["pendulum_dense"]


def test_plan_without_any_throughput_raises(tmp_path: Path, clock: Any, pool_factory: Any) -> None:
    record = _calibrate(tmp_path, clock, pool_factory, ("pendulum_dense",), cap_usd=0.01)

    with pytest.raises(CalibrationError, match="no usable steps/s measurement"):
        choose_plan(record, arms_per_family={"pendulum_dense": 10}, settings=_settings())
