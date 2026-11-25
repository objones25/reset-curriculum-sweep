from collections.abc import Callable, Sequence
from typing import Any

import pytest
import torch
from torch import Tensor

from resetsweep.methods.base import MethodContext
from resetsweep.methods.walk import Walk
from resetsweep.types import Outcomes, RolloutStats, StartStates

MakeCtx = Callable[..., MethodContext]
MakeScore = Callable[[int], Any]


def _outcomes(env_ids: Sequence[int], ic_ids: Sequence[int]) -> Outcomes:
    n = len(env_ids)
    return Outcomes(
        env_ids=torch.tensor(env_ids),
        ic_ids=torch.tensor(ic_ids),
        ret=torch.zeros(n),
        success=torch.zeros(n, dtype=torch.bool),
        ep_len=torch.full((n,), 10),
        reached_reward=torch.zeros(n, dtype=torch.bool),
        first_reward_step=torch.full((n,), -1),
        full=torch.ones(n, dtype=torch.bool),
        start_step=torch.zeros(n, dtype=torch.long),
    )


def _stats(k: int, ic_ids: Tensor, ic_versions: Tensor | None) -> RolloutStats:
    return RolloutStats(
        ic_ids=ic_ids,
        adv=torch.zeros(ic_ids.shape),
        returns=torch.zeros(ic_ids.shape),
        values=torch.zeros(ic_ids.shape),
        start_obs=torch.zeros(k, 3),
        critic=lambda obs: obs[:, 0],
        start_features=torch.zeros(k, 3),
        ic_versions=ic_versions,
    )


def _no_cells(k: int) -> RolloutStats:
    ids = torch.full((1, 1), -1)
    return _stats(k, ids, torch.full((1, 1), -1))


def _build(
    make_ctx: MakeCtx,
    fixed_score: MakeScore,
    *,
    num_envs: int = 16,
    p_task: float = 0.0,
    **settings: object,
) -> tuple[Walk, Any, MethodContext]:
    k = num_envs // 4
    score = fixed_score(k)
    ctx = make_ctx(num_envs=num_envs, p_task=p_task, score=score)
    return Walk(ctx, **settings), score, ctx  # pyright: ignore[reportArgumentType] # typed settings passed through


def test_walk_env_i_restarts_from_ic_i_div_r(
    make_ctx: MakeCtx, fixed_score: MakeScore
) -> None:
    torch.manual_seed(0)
    method, _, _ = _build(make_ctx, fixed_score)

    states, ids = method.reset(torch.tensor([0, 3, 4, 9, 15]), None)
    table = method.ic_states()

    assert ids.tolist() == [0, 0, 1, 2, 3]
    assert table.n == 4
    assert torch.equal(states.qpos, table.qpos[ids])
    assert torch.equal(states.qvel, table.qvel[ids])


def test_walk_p_task_one_sends_every_reset_to_task_start(
    make_ctx: MakeCtx, fixed_score: MakeScore
) -> None:
    torch.manual_seed(0)
    method, _, _ = _build(make_ctx, fixed_score, p_task=1.0)

    _, ids = method.reset(torch.arange(16), None)

    assert ids.tolist() == [-1] * 16
    assert method.episode_versions().tolist() == [-1] * 16


def test_walk_rejects_score_capacity_other_than_population(
    make_ctx: MakeCtx, fixed_score: MakeScore
) -> None:
    ctx = make_ctx(num_envs=16, p_task=0.0, score=fixed_score(8))

    with pytest.raises(ValueError, match=r"score capacity 8 must equal K = 4"):
        Walk(ctx)


def test_walk_requires_a_score(make_ctx: MakeCtx) -> None:
    ctx = make_ctx(num_envs=16, p_task=0.0, score=None)

    with pytest.raises(ValueError, match="needs a score"):
        Walk(ctx)


def test_walk_children_are_reached_states_and_charged(
    make_ctx: MakeCtx, fixed_score: MakeScore, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(0)
    method, score, ctx = _build(
        make_ctx, fixed_score, keep_top=0.25, walk_steps=6, action_hold=3
    )
    score.lp = torch.tensor([0.0, 1.0, 3.0, 2.0])  # IC 2 is the only one kept
    parent = method.ic_states().index(torch.tensor([2]))
    walked: list[StartStates] = []
    real_walk = ctx.rollouts.random_walk

    def spy(starts: StartStates, steps: int, action_hold: int) -> StartStates:
        out = real_walk(starts, steps, action_hold)
        walked.append(out)
        return out

    monkeypatch.setattr(ctx.rollouts, "random_walk", spy)

    method.update(0, _no_cells(4))
    table = method.ic_states()

    replaced = torch.tensor([0, 1, 3])
    assert len(walked) == 1
    assert torch.equal(table.qpos[replaced], walked[0].qpos)
    assert torch.equal(table.qvel[replaced], walked[0].qvel)
    assert torch.equal(table.qpos[2], parent.qpos[0])
    assert ctx.counter.extra["walk"] == 3 * 6
    assert score.forgotten == [[0, 1, 3]]


def test_walk_parents_are_kept_ics_only(
    make_ctx: MakeCtx, fixed_score: MakeScore, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(0)
    method, score, ctx = _build(make_ctx, fixed_score, keep_top=0.25)
    score.lp = torch.tensor([0.0, 1.0, 3.0, 2.0])
    parent = method.ic_states().index(torch.tensor([2]))
    starts_seen: list[StartStates] = []
    real_walk = ctx.rollouts.random_walk

    def spy(starts: StartStates, steps: int, action_hold: int) -> StartStates:
        starts_seen.append(starts)
        return real_walk(starts, steps, action_hold)

    monkeypatch.setattr(ctx.rollouts, "random_walk", spy)

    method.update(0, _no_cells(4))

    assert torch.equal(starts_seen[0].qpos, parent.qpos.expand(3, -1))


def test_walk_invalid_children_redrawn_then_fall_back_to_task_start(
    make_ctx: MakeCtx,
    fixed_score: MakeScore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    method, score, ctx = _build(
        make_ctx, fixed_score, keep_top=0.25, walk_steps=5, tries=3
    )
    score.lp = torch.tensor([0.0, 1.0, 3.0, 2.0])
    monkeypatch.setattr(
        ctx.task, "is_valid", lambda s: torch.zeros(s.n, dtype=torch.bool)
    )

    method.update(0, _no_cells(4))
    table = method.ic_states()

    replaced = torch.tensor([0, 1, 3])
    # pendulum_sparse task starts are uniform in [-0.1, 0.1] for angle and speed
    assert bool((table.qpos[replaced].abs() <= 0.1).all())
    assert bool((table.qvel[replaced].abs() <= 0.1).all())
    assert ctx.counter.extra["walk"] == 3 * 5 * 3


def test_walk_updates_only_every_update_every_iterations(
    make_ctx: MakeCtx, fixed_score: MakeScore
) -> None:
    torch.manual_seed(0)
    method, score, ctx = _build(make_ctx, fixed_score, update_every=20)
    score.lp = torch.tensor([0.0, 1.0, 3.0, 2.0])

    for it in range(1, 20):
        method.update(it, _no_cells(4))

    assert ctx.counter.extra["walk"] == 0
    assert score.forgotten == []
    assert len(score.observed) == 19


def test_walk_update_needs_ic_versions(
    make_ctx: MakeCtx, fixed_score: MakeScore
) -> None:
    method, _, _ = _build(make_ctx, fixed_score)

    with pytest.raises(ValueError, match="ic_versions"):
        method.update(1, _stats(4, torch.zeros(1, 16, dtype=torch.long), None))


def test_replaced_ic_outcomes_dropped_by_version(
    make_ctx: MakeCtx, fixed_score: MakeScore
) -> None:
    torch.manual_seed(0)
    method, score, _ = _build(make_ctx, fixed_score, keep_top=0.5)
    score.lp = torch.tensor([3.0, 2.0, 1.0, 0.0])  # ICs 2 and 3 get replaced
    _, ids = method.reset(torch.arange(16), None)
    versions = method.episode_versions()

    method.update(0, _no_cells(4))
    method.update(1, _stats(4, ids.unsqueeze(0), versions.unsqueeze(0)))
    method.reset(torch.tensor([0, 8, 12]), _outcomes([0, 8, 12], [0, 2, 3]))

    assert score.recorded[-1].ic_ids.tolist() == [0, -1, -1]
    cells = score.observed[-1].ic_ids[0]
    assert cells[:8].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert cells[8:].tolist() == [-1] * 8


def test_walk_diagnostics_report_population(
    make_ctx: MakeCtx, fixed_score: MakeScore
) -> None:
    method, _, _ = _build(make_ctx, fixed_score, p_task=0.5)

    diag = method.diagnostics()

    # entropy of {task: 0.5, each of 4 ICs: 0.125}
    expected = -(
        0.5 * torch.log(torch.tensor(0.5)) + 4 * 0.125 * torch.log(torch.tensor(0.125))
    )
    assert diag["buffer_size"] == 4.0
    assert diag["sampling_entropy"] == pytest.approx(float(expected))
    assert diag["never_succeeded_share"] == pytest.approx(1.0)


def test_walk_high_parent_power_draws_every_parent_from_the_best_kept_ic(
    make_ctx: MakeCtx, fixed_score: MakeScore, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(0)
    method, score, ctx = _build(
        make_ctx, fixed_score, num_envs=32, keep_top=0.25, parent_power=50.0
    )
    score.lp = torch.arange(8, dtype=torch.float32)  # kept: 7 (best) and 6
    best = method.ic_states().index(torch.tensor([7]))
    starts_seen: list[StartStates] = []
    real_walk = ctx.rollouts.random_walk

    def spy(starts: StartStates, steps: int, action_hold: int) -> StartStates:
        starts_seen.append(starts)
        return real_walk(starts, steps, action_hold)

    monkeypatch.setattr(ctx.rollouts, "random_walk", spy)

    method.update(0, _no_cells(8))

    # rank 2 has weight 2^-50 against 1 for rank 1
    assert torch.equal(starts_seen[0].qpos, best.qpos.expand(6, -1))
