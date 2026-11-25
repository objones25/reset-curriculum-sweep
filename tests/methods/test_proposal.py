import math
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import Tensor

from resetsweep.methods.base import MethodContext
from resetsweep.methods.proposal import Ball, Genetic, knn_novelty
from resetsweep.types import RolloutStats, StartStates

MakeCtx = Callable[..., MethodContext]
MakeScore = Callable[[int], Any]

# pendulum start_box: angle in [-pi, pi], speed in [-8, 8]
_SPAN = torch.tensor([2 * math.pi, 16.0])
_LOW = torch.tensor([-math.pi, -8.0])


def _norm(states: StartStates) -> Tensor:
    return (torch.cat([states.qpos, states.qvel], dim=1) - _LOW) / _SPAN


def _no_cells(k: int) -> RolloutStats:
    ids = torch.full((1, 1), -1)
    return RolloutStats(
        ic_ids=ids,
        adv=torch.zeros(ids.shape),
        returns=torch.zeros(ids.shape),
        values=torch.zeros(ids.shape),
        start_obs=torch.zeros(k, 3),
        critic=lambda obs: obs[:, 0],
        start_features=torch.zeros(k, 3),
        ic_versions=torch.full((1, 1), -1),
    )


def _states(qpos: list[float], qvel: list[float]) -> StartStates:
    n = len(qpos)
    return StartStates(
        torch.tensor(qpos).unsqueeze(1),
        torch.tensor(qvel).unsqueeze(1),
        torch.zeros(n, 0),
    )


def _build(
    cls: type[Ball] | type[Genetic],
    make_ctx: MakeCtx,
    fixed_score: MakeScore,
    *,
    k: int = 4,
    **settings: object,
) -> tuple[Any, Any, MethodContext]:
    score = fixed_score(k)
    ctx = make_ctx(num_envs=4 * k, p_task=0.0, score=score)
    return cls(ctx, **settings), score, ctx  # pyright: ignore[reportArgumentType] # typed settings passed through


def test_ball_children_within_radius_and_valid(
    make_ctx: MakeCtx, fixed_score: MakeScore
) -> None:
    torch.manual_seed(0)
    method, score, ctx = _build(
        Ball, make_ctx, fixed_score, keep_top=0.25, r_min=0.05, r_max=0.3
    )
    method.replace(torch.arange(4), _states([0.0, 0.5, 3.0, 1.0], [0.0, 0.0, 0.0, 7.5]))
    score.lp = torch.tensor([0.0, 1.0, 3.0, 2.0])  # IC 2 (angle 3.0) is the only parent

    method.update(0, _no_cells(4))
    table = method.ic_states()

    children = table.index(torch.tensor([0, 1, 3]))
    parent = table.index(torch.tensor([2]))
    distance = (_norm(children) - _norm(parent)).norm(dim=1)
    assert bool((distance <= 0.05 + 1e-6).all())
    assert bool(ctx.task.is_valid(children).all())
    assert table.qpos[2].item() == pytest.approx(3.0)


def test_ball_invalid_children_fall_back_to_task_start(
    make_ctx: MakeCtx, fixed_score: MakeScore, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(0)
    method, score, ctx = _build(Ball, make_ctx, fixed_score, keep_top=0.25, tries=4)
    method.replace(torch.arange(4), _states([0.0, 0.5, 3.0, 1.0], [0.0, 0.0, 0.0, 7.5]))
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
    assert score.forgotten[-1] == [0, 1, 3]


def test_ball_children_stay_inside_start_box_even_when_is_valid_accepts_all(
    make_ctx: MakeCtx, fixed_score: MakeScore, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(0)
    method, score, ctx = _build(
        Ball, make_ctx, fixed_score, keep_top=0.25, r_min=0.3, r_max=0.3
    )
    method.replace(
        torch.arange(4), _states([0.0, 0.0, math.pi - 0.01, 0.0], [0.0, 0.0, 7.9, 0.0])
    )
    score.lp = torch.tensor([0.0, 1.0, 3.0, 2.0])
    monkeypatch.setattr(
        ctx.task, "is_valid", lambda s: torch.ones(s.n, dtype=torch.bool)
    )

    method.update(0, _no_cells(4))

    unit = _norm(method.ic_states())
    assert bool(((unit >= 0.0) & (unit <= 1.0)).all())


def test_genetic_replaces_bottom_p_with_children_of_top_p(
    make_ctx: MakeCtx, fixed_score: MakeScore
) -> None:
    torch.manual_seed(0)
    method, score, _ = _build(
        Genetic, make_ctx, fixed_score, k=10, p=0.2, sigma_min=1e-3, sigma_max=1e-3
    )
    angles = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    method.replace(torch.arange(10), _states(angles, [0.0] * 10))
    score.lp = torch.arange(10, dtype=torch.float32)  # ICs 0, 1 worst; 8, 9 best

    method.update(0, _no_cells(10))
    table = method.ic_states()

    parents = _norm(table.index(torch.tensor([8, 9])))
    children = _norm(table.index(torch.tensor([0, 1])))
    nearest = torch.cdist(children, parents).min(dim=1).values
    assert bool((nearest < 0.01).all())
    assert table.qpos[2:, 0].tolist() == pytest.approx(angles[2:])
    assert score.forgotten[-1] == [0, 1]


def _mean_child_novelty(
    make_ctx: MakeCtx, fixed_score: MakeScore, *, n_candidates: int
) -> float:
    """Genetic on 40 ICs all at one state, so a child's novelty is its feature distance to
    that state. p = 0.5: 20 children, about half from parents in the low half of the pool."""
    torch.manual_seed(0)
    method, score, ctx = _build(
        Genetic,
        make_ctx,
        fixed_score,
        k=40,
        p=0.5,
        sigma_min=0.05,
        sigma_max=0.05,
        n_candidates=n_candidates,
    )
    method.replace(torch.arange(40), _states([0.0] * 40, [0.0] * 40))
    score.lp = torch.arange(40, dtype=torch.float32)
    centre = ctx.task.state_features(
        torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 0)
    )

    method.update(0, _no_cells(40))

    children = method.ic_states().index(torch.arange(20))
    features = ctx.task.state_features(children.qpos, children.qvel, children.extra)
    return float((features - centre).norm(dim=1).mean())


def test_genetic_low_score_parent_keeps_most_novel_candidate(
    make_ctx: MakeCtx, fixed_score: MakeScore
) -> None:
    single = _mean_child_novelty(make_ctx, fixed_score, n_candidates=1)
    best_of_64 = _mean_child_novelty(make_ctx, fixed_score, n_candidates=64)

    # Expected ratio about 1.6: the low-pool half of the children move from a typical
    # Gaussian draw (1.25 sigma in 2-D) to the farthest of 64 (about 2.9 sigma).
    assert best_of_64 > 1.3 * single


def test_knn_novelty_is_mean_distance_to_k_nearest() -> None:
    population = torch.tensor([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]])
    candidates = torch.tensor([[0.0, 1.0], [10.0, 0.0]])

    novelty = knn_novelty(candidates, population, k=2)

    assert novelty.tolist() == pytest.approx(
        [(1.0 + math.sqrt(2.0)) / 2, (5.0 + 9.0) / 2]
    )


@pytest.mark.parametrize("cls", [Ball, Genetic])
def test_no_cull_when_scores_flat(
    cls: type[Ball] | type[Genetic], make_ctx: MakeCtx, fixed_score: MakeScore
) -> None:
    torch.manual_seed(0)
    method, score, _ = _build(cls, make_ctx, fixed_score, spread_threshold=0.1)
    before = method.ic_states()
    score.lp = torch.tensor([1.0, 1.02, 0.99, 1.01])  # spread 0.03 / median 1.0 < 0.1

    method.update(0, _no_cells(4))

    assert torch.equal(method.ic_states().qpos, before.qpos)
    assert score.forgotten == []


@pytest.mark.parametrize("cls", [Ball, Genetic])
def test_no_cull_when_fewer_than_two_ics_trusted(
    cls: type[Ball] | type[Genetic], make_ctx: MakeCtx, fixed_score: MakeScore
) -> None:
    torch.manual_seed(0)
    # threshold 0 so only the trusted-count rule can stop the cull
    method, score, _ = _build(cls, make_ctx, fixed_score, spread_threshold=0.0)
    before = method.ic_states()
    score.lp = torch.tensor([0.0, 5.0, 0.0, 0.0])
    score.trusted = torch.tensor([False, True, False, False])

    method.update(0, _no_cells(4))

    assert torch.equal(method.ic_states().qpos, before.qpos)
    assert score.forgotten == []


@pytest.mark.parametrize("cls", [Ball, Genetic])
def test_screening_charges_screen_steps(
    cls: type[Ball] | type[Genetic], make_ctx: MakeCtx, fixed_score: MakeScore
) -> None:
    torch.manual_seed(0)
    settings: dict[str, object] = {"keep_top": 0.5} if cls is Ball else {"p": 0.5}
    method, score, ctx = _build(
        cls, make_ctx, fixed_score, screen_seconds=0.5, **settings
    )
    score.lp = torch.tensor([0.0, 1.0, 2.0, 3.0])  # 2 children

    method.update(0, _no_cells(4))

    # Task starts hang still, so their children pass on the first try; 0.5 s is 10 control
    # steps of 0.05 s each.
    assert ctx.counter.extra["screen"] == 2 * 10


def test_screening_uses_task_speed_bound_not_default(
    make_ctx: MakeCtx, fixed_score: MakeScore
) -> None:
    torch.manual_seed(0)
    method, score, _ = _build(
        Genetic,
        make_ctx,
        fixed_score,
        p=0.5,
        sigma_min=1e-4,
        sigma_max=1e-4,
        screen_seconds=0.5,
    )
    # Swinging through the bottom at 6 rad/s: above the Rollouts default of 5, below the
    # pendulum box bound of 8 (and too slow to go over the top, so it stays below 8).
    method.replace(torch.arange(4), _states([0.0] * 4, [6.0] * 4))
    score.lp = torch.tensor([0.0, 1.0, 2.0, 3.0])

    method.update(0, _no_cells(4))

    assert method.ic_states().qvel[:2, 0].tolist() == pytest.approx(
        [6.0, 6.0], abs=0.01
    )


def test_genetic_rejects_score_capacity_other_than_population(
    make_ctx: MakeCtx, fixed_score: MakeScore
) -> None:
    ctx = make_ctx(num_envs=16, p_task=0.0, score=fixed_score(3))

    with pytest.raises(ValueError, match=r"score capacity 3 must equal K = 4"):
        Genetic(ctx)


def test_replace_rejects_invalid_state(
    make_ctx: MakeCtx, fixed_score: MakeScore
) -> None:
    method, _, _ = _build(Ball, make_ctx, fixed_score)

    with pytest.raises(ValueError, match="invalid start state at row 1"):
        method.replace(torch.tensor([0, 1]), _states([0.0, 4.0], [0.0, 0.0]))
