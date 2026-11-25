import math
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import Tensor

from resetsweep.methods.base import MethodContext
from resetsweep.methods.proposal import Ball, Genetic, knn_novelty
from resetsweep.methods.walk import Walk
from resetsweep.types import Box, RolloutStats, StartStates

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
    cls: type[Ball] | type[Genetic] | type[Walk],
    make_ctx: MakeCtx,
    fixed_score: MakeScore,
    *,
    k: int = 4,
    task: Any = None,
    **settings: object,
) -> tuple[Any, Any, MethodContext]:
    score = fixed_score(k)
    ctx = make_ctx(num_envs=4 * k, p_task=0.0, score=score, task=task)
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


# Selection uses measured (trusted) ICs only ------------------------------------------

_CULL_SETTINGS: dict[type, dict[str, object]] = {
    Walk: {"keep_top": 0.5},
    Ball: {"keep_top": 0.5},
    Genetic: {"p": 0.5},
}


@pytest.mark.parametrize("cls", [Walk, Ball, Genetic])
def test_untrusted_ics_are_never_culled(
    cls: type[Walk] | type[Ball] | type[Genetic],
    make_ctx: MakeCtx,
    fixed_score: MakeScore,
) -> None:
    torch.manual_seed(0)
    method, score, _ = _build(cls, make_ctx, fixed_score, k=8, **_CULL_SETTINGS[cls])
    before = method.ic_states()
    score.lp = torch.tensor([0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0])
    score.trusted = torch.tensor([True] * 4 + [False] * 4)

    method.update(0, _no_cells(8))

    assert torch.equal(method.ic_states().qpos[4:], before.qpos[4:])
    # 4 trusted: walk/ball keep ceil(0.5 * 4) = 2, genetic culls min(ceil(0.5 * 4), 2) = 2
    assert score.forgotten == [[0, 1]]


def test_parents_drawn_only_from_trusted_ics(
    make_ctx: MakeCtx, fixed_score: MakeScore, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(0)
    method, score, ctx = _build(Walk, make_ctx, fixed_score, k=8, keep_top=0.5)
    method.replace(
        torch.arange(8),
        _states([0.0, 0.2, 0.4, 0.6, 1.0, 1.2, 1.4, 1.6], [0.0] * 8),
    )
    # Untrusted ICs read as the trusted median (1.0), above trusted ICs 0 and 1.
    score.lp = torch.tensor([0.0, 1.0, 2.0, 10.0, 0.0, 0.0, 0.0, 0.0])
    score.trusted = torch.tensor([True] * 4 + [False] * 4)
    starts_seen: list[StartStates] = []
    real_walk = ctx.rollouts.random_walk

    def spy(starts: StartStates, steps: int, action_hold: int) -> StartStates:
        starts_seen.append(starts)
        return real_walk(starts, steps, action_hold)

    monkeypatch.setattr(ctx.rollouts, "random_walk", spy)

    method.update(0, _no_cells(8))

    # kept: trusted ICs 3 and 2 (angles 0.6 and 0.4)
    parent_angles = starts_seen[0].qpos[:, 0]
    kept = torch.tensor([0.4, 0.6])
    assert parent_angles.numel() == 2  # trusted ICs 0 and 1 are replaced
    assert bool(
        ((parent_angles.unsqueeze(1) - kept).abs().min(dim=1).values < 1e-6).all()
    )


@pytest.mark.parametrize("cls", [Walk, Ball])
def test_children_survive_until_measured(
    cls: type[Walk] | type[Ball], make_ctx: MakeCtx, fixed_score: MakeScore
) -> None:
    torch.manual_seed(0)
    method, score, _ = _build(cls, make_ctx, fixed_score, k=8, keep_top=0.5)
    score.untrust_on_forget = True
    score.lp = torch.arange(8, dtype=torch.float32)  # 0-3 replaced at the first update

    method.update(0, _no_cells(8))
    children = method.ic_states().index(torch.arange(4))
    method.update(20, _no_cells(8))

    assert torch.equal(method.ic_states().qpos[:4], children.qpos)
    assert score.forgotten == [[0, 1, 2, 3], [4, 5]]


# Zero-width box dims and quality-dependent step size ---------------------------------


def test_flat_box_dim_is_never_moved_by_ball_noise(
    make_ctx: MakeCtx, fixed_score: MakeScore, pendulum: Any
) -> None:
    class _FixedSpeedPendulum(type(pendulum)):  # pyright: ignore[reportGeneralTypeIssues] # subclass of a registry-built task
        def __init__(self) -> None:
            super().__init__()
            # qvel admits exactly one value
            self.start_box = Box(
                torch.tensor([-math.pi, 0.0]), torch.tensor([math.pi, 0.0])
            )

        def sample_start(self, n: int, gen: torch.Generator) -> StartStates:
            angle = (torch.rand(n, 1, generator=gen) * 2 - 1) * 0.1
            return StartStates(angle, torch.zeros(n, 1), torch.zeros(n, 0))

    torch.manual_seed(0)
    method, score, _ = _build(
        Ball,
        make_ctx,
        fixed_score,
        task=_FixedSpeedPendulum(),
        keep_top=0.25,
        r_min=0.05,
        r_max=0.05,
    )
    method.replace(torch.arange(4), _states([0.0, 0.0, 2.0, 0.0], [0.0] * 4))
    score.lp = torch.tensor([0.0, 1.0, 3.0, 2.0])

    method.update(0, _no_cells(4))
    table = method.ic_states()

    # Children are near the parent at angle 2.0 (not task-start fallbacks near 0) and
    # their speed is exactly the one admissible value.
    assert table.qvel[:, 0].tolist() == [0.0] * 4
    angles = table.qpos[torch.tensor([0, 1, 3]), 0]
    assert bool(((angles - 2.0).abs() <= 0.05 * 2 * math.pi + 1e-6).all())


def _nearest(children: Tensor, parents: Tensor) -> tuple[Tensor, Tensor]:
    """Index of each child's nearest parent and the distance to it (normalised units)."""
    distance = torch.cdist(children, parents)
    return distance.argmin(dim=1), distance.min(dim=1).values


def test_worse_parent_gets_larger_sigma(
    make_ctx: MakeCtx, fixed_score: MakeScore
) -> None:
    torch.manual_seed(0)
    method, score, _ = _build(
        Genetic,
        make_ctx,
        fixed_score,
        k=16,
        p=0.5,
        sigma_min=0.001,
        sigma_max=0.1,
        n_candidates=1,
    )
    # Parents are ICs 8-15. The better four (12-15, pool quality 1 to 4/7) sit at corner A,
    # the worse four (8-11, quality 3/7 to 0) at corner B, far apart in normalised units.
    method.replace(
        torch.arange(16),
        _states([0.0] * 8 + [2.0] * 4 + [-2.0] * 4, [0.0] * 8 + [5.0] * 4 + [-5.0] * 4),
    )
    score.lp = torch.arange(16, dtype=torch.float32)

    method.update(0, _no_cells(16))
    table = method.ic_states()

    corners = _norm(table.index(torch.tensor([15, 8])))  # A (better), B (worse)
    which, distance = _nearest(_norm(table.index(torch.arange(8))), corners)
    assert which.tolist().count(0) >= 1
    assert which.tolist().count(1) >= 1
    # Mean sigma is about 0.03 at A and 0.08 at B, so B's children land about 3x farther.
    assert distance[which == 1].mean().item() > 2 * distance[which == 0].mean().item()


def test_worse_parent_gets_larger_radius(
    make_ctx: MakeCtx, fixed_score: MakeScore
) -> None:
    torch.manual_seed(0)
    method, score, _ = _build(
        Ball,
        make_ctx,
        fixed_score,
        k=8,
        keep_top=0.25,
        parent_power=0.0,
        r_min=0.01,
        r_max=0.2,
    )
    # Kept ICs 7 (best, radius r_min) and 6 (worst kept, radius r_max), far apart.
    method.replace(
        torch.arange(8), _states([0.0] * 6 + [2.0, -2.0], [0.0] * 6 + [5.0, -5.0])
    )
    score.lp = torch.arange(8, dtype=torch.float32)

    method.update(0, _no_cells(8))
    table = method.ic_states()

    parents = _norm(table.index(torch.tensor([7, 6])))
    which, distance = _nearest(_norm(table.index(torch.arange(6))), parents)
    assert which.tolist().count(0) >= 1
    assert which.tolist().count(1) >= 1
    assert distance[which == 0].max().item() <= 0.01 + 1e-6
    # uniform in a 2-D ball of radius 0.2: P(distance <= 0.01) = (0.01 / 0.2)^2 = 0.0025
    assert distance[which == 1].min().item() > 0.01
