import dataclasses
import math
from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from resetsweep.methods import visited_count as visited_count_module
from resetsweep.methods.base import MethodContext
from resetsweep.methods.visited_count import VisitedCount
from resetsweep.scores.side_models import StepDistance
from resetsweep.scores.visit_count import PostReward, VisitCount
from resetsweep.tasks.base import Task
from resetsweep.tasks.maze import Maze
from resetsweep.types import Outcomes, RolloutStats

MakeCtx = Callable[..., MethodContext]


def _stats(ic_capacity: int) -> RolloutStats:
    """Rollout data with one task-start cell: `update` needs it, the score never reads it."""
    return RolloutStats(
        ic_ids=torch.full((1, 1), -1),
        adv=torch.zeros(1, 1),
        returns=torch.zeros(1, 1),
        values=torch.zeros(1, 1),
        start_obs=torch.zeros(ic_capacity, 1),
        critic=lambda obs: obs[:, 0],
        start_features=torch.zeros(ic_capacity, 1),
        ic_versions=torch.zeros(1, 1, dtype=torch.long),
    )


def _build(
    make_ctx: MakeCtx,
    *,
    capacity: int,
    num_envs: int,
    ic_capacity: int | None = None,
    p_task: float = 0.0,
    task: Task | None = None,
    post_reward: PostReward = "count",
    **settings: object,
) -> tuple[VisitedCount, VisitCount]:
    ic_capacity = capacity if ic_capacity is None else ic_capacity
    score = VisitCount(capacity=ic_capacity, device="cpu", post_reward=post_reward)
    ctx = make_ctx(num_envs=num_envs, p_task=p_task, score=score, task=task)
    method = VisitedCount(ctx, capacity=capacity, ic_capacity=ic_capacity, **settings)  # pyright: ignore[reportArgumentType] # typed settings passed through
    return method, score


def _offer(method: VisitedCount, qpos: Tensor, qvel: Tensor) -> None:
    n = qpos.shape[0]
    method.offer_visited(
        torch.arange(n), qpos, qvel, torch.zeros(n, 0), torch.ones(n, dtype=torch.long)
    )


def _pendulum_features(qpos: Tensor, qvel: Tensor) -> Tensor:
    return torch.cat([qpos.cos(), qpos.sin(), qvel], dim=1)


def test_neighbour_count_matches_bruteforce_small(
    make_ctx: MakeCtx, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(0)
    ring, table_size = 24, 10
    # 3 table rows per cdist chunk against the 24-state ring: the count runs over 4 chunks.
    monkeypatch.setattr(visited_count_module, "_CDIST_ELEMENTS_MAX", 3 * ring)
    method, score = _build(
        make_ctx, capacity=ring, num_envs=ring, ic_capacity=table_size, radius=0.8
    )
    qpos = (torch.rand(ring, 1) * 2 - 1) * 3.0
    qvel = (torch.rand(ring, 1) * 2 - 1) * 2.0
    _offer(method, qpos, qvel)

    method.update(0, _stats(table_size))

    f = _pendulum_features(qpos, qvel)
    mean, std = f.mean(dim=0), f.std(dim=0, unbiased=False)
    table = method.ic_states()
    z_table = (_pendulum_features(table.qpos, table.qvel) - mean) / std
    z_ring = (f - mean) / std
    dist = torch.cdist(z_table, z_ring, compute_mode="donot_use_mm_for_euclid_dist")
    expected = (dist <= 0.8).sum(
        dim=1
    )  # ring states near each IC, its own copy included
    lp, _ = score.values(torch.arange(table_size))
    assert torch.equal(method.counts(), expected)
    assert torch.allclose(lp, 1.0 / torch.sqrt(expected.float() + 1.0))


def test_counts_and_table_refresh_only_every_update_every_iterations(
    make_ctx: MakeCtx,
) -> None:
    torch.manual_seed(0)
    method, _ = _build(
        make_ctx,
        capacity=8,
        num_envs=4,
        ic_capacity=4,
        update_every=2,
        promote_frac=1.0,
    )
    _offer(method, torch.zeros(4, 1), torch.zeros(4, 1))
    method.update(0, _stats(4))
    first = method.counts().clone()
    versions = method.ic_versions()

    _offer(method, torch.zeros(4, 1), torch.zeros(4, 1))
    method.update(1, _stats(4))

    assert first.tolist() == [4, 4, 4, 4]  # 4 identical ring states
    assert method.counts().tolist() == [4, 4, 4, 4]
    assert torch.equal(method.ic_versions(), versions)


def _chain(make_ctx: MakeCtx, *, ic_capacity: int, promote_frac: float) -> VisitedCount:
    """One env and a ring as small as the IC table; offering the latest state `ic_capacity`
    times fills the whole ring with it, so every promotion copies that state."""
    method, _ = _build(
        make_ctx,
        capacity=ic_capacity,
        num_envs=1,
        ic_capacity=ic_capacity,
        promote_frac=promote_frac,
        count="lineage_depth",
    )
    _offer(method, torch.tensor([[0.1]]), torch.zeros(1, 1))  # episode from task start
    method.update(0, _stats(ic_capacity))
    return method


def test_lineage_depth_increments_along_chain(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method = _chain(make_ctx, ic_capacity=1, promote_frac=1.0)
    depths = [method.depths().tolist()]

    method.reset(torch.tensor([0]), None)  # restart from the depth-0 IC
    _offer(method, torch.tensor([[0.2]]), torch.zeros(1, 1))
    method.update(1, _stats(1))
    depths.append(method.depths().tolist())
    method.reset(torch.tensor([0]), None)
    _offer(method, torch.tensor([[0.3]]), torch.zeros(1, 1))
    method.update(2, _stats(1))
    depths.append(method.depths().tolist())

    assert depths == [[0], [1], [2]]


def test_lineage_depth_is_zero_after_task_start_reset(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method = _chain(make_ctx, ic_capacity=1, promote_frac=1.0)
    method.reset(torch.tensor([0]), None)
    _offer(method, torch.tensor([[0.2]]), torch.zeros(1, 1))
    method.update(1, _stats(1))
    method.ctx = dataclasses.replace(method.ctx, p_task=1.0)  # later resets: task start

    _, ids = method.reset(torch.tensor([0]), None)
    _offer(method, torch.tensor([[0.3]]), torch.zeros(1, 1))
    method.update(2, _stats(1))

    assert ids.tolist() == [-1]
    assert method.depths().tolist() == [0]


def test_lineage_depth_prefers_deeper_states(make_ctx: MakeCtx) -> None:
    # The first version fed depth itself as the visit count, so deeper states were drawn
    # less: a misreading. Depth stands in for "further from the start / rarer" (shortlist
    # section 3, method 3), so the draw weight must grow with depth.
    torch.manual_seed(0)
    method = _chain(make_ctx, ic_capacity=2, promote_frac=0.5)  # slot 0: depth 0
    method.reset(torch.tensor([0]), None)
    # Offered twice: the two-slot ring now holds only the depth-1 state.
    _offer(method, torch.tensor([[0.2]]), torch.zeros(1, 1))
    _offer(method, torch.tensor([[0.2]]), torch.zeros(1, 1))

    method.update(1, _stats(2))  # fills slot 1 with the depth-1 state

    probs = method.slot_probs()
    assert method.depths().tolist() == [0, 1]
    assert method.counts().tolist() == [1, 0]  # pseudo-count max_depth - depth
    assert probs[1].item() / probs[0].item() == pytest.approx(2**0.5)


def test_count_weighted_draw_prefers_rare_states(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    n = 512
    method, _ = _build(make_ctx, capacity=n, num_envs=n, radius=0.5)
    qpos = torch.full((n, 1), 0.5) + torch.randn(n, 1) * 1e-3
    qvel = torch.randn(n, 1) * 1e-3
    qpos[-1, 0], qvel[-1, 0] = -2.5, 5.0  # one rare state, far from the cluster
    _offer(method, qpos, qvel)
    method.update(0, _stats(n))
    rare = int((method.ic_states().qpos[:, 0] == -2.5).nonzero()[0, 0])

    _, ids = method.reset(torch.arange(n), None)

    probs = method.slot_probs()
    common = (rare + 1) % n
    # Cluster ICs have 511 ring neighbours (lp = 1/sqrt(512)), the rare one only its own
    # copy (lp = 1/sqrt(2)): a weight ratio of 16.
    assert probs[rare].item() / probs[common].item() == pytest.approx(16.0, rel=1e-4)
    assert int((ids == rare).sum()) >= 5  # expected about 16; uniform gives 1


def _maze_share(make_ctx: MakeCtx, qpos: Tensor, *, radius: float) -> float:
    torch.manual_seed(0)
    n = qpos.shape[0]
    method, _ = _build(make_ctx, capacity=n, num_envs=n, radius=radius, task=Maze())
    _offer(method, qpos, torch.zeros(n, 2))
    method.update(0, _stats(n))
    return method.diagnostics()["across_wall_neighbour_share"]


def _free_centres() -> Tensor:
    grid = Maze().grid
    free = (~grid.occupied).nonzero()
    return (free.flip(1).float() + 0.5) * grid.cell_m


# Cells (1, 1) and (3, 1) are free with the wall row 2 between them: these two points are
# 0.5 m apart in a straight line and 16 cells (6.4 m) apart through the maze.
_ACROSS = torch.tensor([[0.6, 0.75], [0.6, 1.25]])


def test_across_wall_share_positive_in_maze_fixture(make_ctx: MakeCtx) -> None:
    share = _maze_share(make_ctx, torch.cat([_free_centres(), _ACROSS]), radius=1.0)

    assert 0.0 < share < 1.0


def test_across_wall_share_is_one_when_every_neighbour_is_across_a_wall(
    make_ctx: MakeCtx,
) -> None:
    # Only the two across-wall points: standardised they sit 2 apart in y, inside radius 2.5;
    # radius_m = 2.5 * mean(std_x = 0, std_y = 0.25) = 0.3125 m, and 6.4 m > 2 * 0.3125 m.
    share = _maze_share(make_ctx, _ACROSS, radius=2.5)

    assert share == 1.0


def test_across_wall_share_is_zero_within_one_corridor(make_ctx: MakeCtx) -> None:
    # Centres of row 1 only: every path is the straight line, at most 2 * radius_m apart.
    row = torch.tensor([[0.4 * c + 0.2, 0.6] for c in range(1, 9)])

    share = _maze_share(make_ctx, row, radius=1.0)

    assert share == 0.0


def test_across_wall_share_absent_without_path_distance(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method, _ = _build(make_ctx, capacity=8, num_envs=4)
    _offer(method, torch.zeros(4, 1), torch.zeros(4, 1))

    method.update(0, _stats(8))

    assert "across_wall_neighbour_share" not in method.diagnostics()


def test_ic_ids_are_table_slots(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method, _ = _build(make_ctx, capacity=16, num_envs=4, ic_capacity=4)
    _offer(method, torch.tensor([[0.1], [0.2], [0.3], [0.4]]), torch.zeros(4, 1))
    method.update(0, _stats(4))

    states, ids = method.reset(torch.arange(4), None)

    table = method.ic_states()
    assert table.n == 4
    assert bool(((ids >= 0) & (ids < 4)).all())
    assert torch.equal(table.qpos[ids], states.qpos)


def test_draws_only_change_at_update(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method, _ = _build(make_ctx, capacity=8, num_envs=8, ic_capacity=4)
    _offer(method, torch.zeros(8, 1), torch.zeros(8, 1))
    method.update(0, _stats(4))
    table = method.ic_states().qpos.clone()

    _offer(method, torch.ones(8, 1), torch.zeros(8, 1))  # the whole ring is overwritten
    states, _ = method.reset(torch.arange(8), None)

    # Resets draw from the IC table, which only promotion in `update` changes.
    assert torch.equal(method.ic_states().qpos, table)
    assert bool((states.qpos == 0.0).all())


def _outcome(env_id: int, *, reached: bool) -> Outcomes:
    return Outcomes(
        env_ids=torch.tensor([env_id]),
        ic_ids=torch.tensor([-1]),
        ret=torch.tensor([float(reached)]),
        success=torch.tensor([reached]),
        ep_len=torch.tensor([10]),
        reached_reward=torch.tensor([reached]),
        full=torch.tensor([True]),
        start_step=torch.tensor([0]),
    )


def test_first_reward_notifies_score_and_records_step(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method, score = _build(
        make_ctx, capacity=8, num_envs=4, post_reward="count_x_learnability"
    )
    method.ctx.counter.add_train(100)
    method.reset(torch.tensor([0]), _outcome(0, reached=False))
    before = method.diagnostics()["first_reward_step"]
    method.ctx.counter.add_train(20)

    method.reset(torch.tensor([1]), _outcome(1, reached=True))
    method.ctx.counter.add_train(30)
    method.reset(torch.tensor([2]), _outcome(2, reached=True))

    # Notified: lp now carries the (learnability + eps) factor; no IC has outcomes, so
    # learnability is 0 and lp = 1/sqrt(0 + 1) * 0.05.
    lp, _ = score.values(torch.arange(8))
    assert math.isnan(before)  # NaN until the first reward
    assert lp.tolist() == pytest.approx([0.05] * 8)
    assert method.diagnostics()["first_reward_step"] == 120.0


def test_embedding_pairs_are_consecutive_offers_within_one_episode(
    make_ctx: MakeCtx, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(0)
    pairs: list[tuple[Tensor, Tensor]] = []

    def record(self: StepDistance, features_t: Tensor, features_t1: Tensor) -> None:
        del self
        pairs.append((features_t.clone(), features_t1.clone()))

    monkeypatch.setattr(StepDistance, "observe_transitions", record)
    method, _ = _build(make_ctx, capacity=16, num_envs=2, count="embedding")
    _offer(method, torch.tensor([[0.1], [0.2]]), torch.zeros(2, 1))
    method.reset(torch.tensor([1]), None)  # env 1 starts a new episode between offers
    _offer(method, torch.tensor([[0.3], [0.4]]), torch.zeros(2, 1))

    method.update(0, _stats(16))

    assert len(pairs) == 1
    first, second = pairs[0]
    assert first.shape == (1, 3)
    # Standardised features keep order along the sin column: 0.1 < 0.3 in qpos.
    assert first[0, 1].item() < second[0, 1].item()


def test_embedding_mode_reports_side_model_wall_time(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method, _ = _build(make_ctx, capacity=16, num_envs=4, count="embedding")
    _offer(method, torch.rand(4, 1), torch.zeros(4, 1))
    _offer(method, torch.rand(4, 1), torch.zeros(4, 1))

    method.update(0, _stats(16))

    diag = method.diagnostics()
    assert diag["side/wall_ms_per_update"] > 0.0
    assert diag["buffer_size"] == 8.0
    assert diag["ring_size"] == 8.0


def test_rejects_a_score_other_than_visit_count(make_ctx: MakeCtx) -> None:
    with pytest.raises(TypeError, match="visit_count"):
        VisitedCount(make_ctx(num_envs=4), capacity=8)


def test_rejects_score_capacity_other_than_ic_capacity(make_ctx: MakeCtx) -> None:
    score = VisitCount(capacity=16, device="cpu")

    with pytest.raises(ValueError, match="ic_capacity"):
        VisitedCount(make_ctx(num_envs=4, score=score), capacity=32, ic_capacity=8)


def test_rejects_unknown_count_rule(make_ctx: MakeCtx) -> None:
    with pytest.raises(ValueError, match="count"):
        _build(make_ctx, capacity=8, num_envs=4, count="hash")
