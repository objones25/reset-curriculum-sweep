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


def _stats() -> RolloutStats:
    """Rollout data `visited_count` never reads; `update` only needs the argument."""
    return RolloutStats(
        ic_ids=torch.zeros(1, 1, dtype=torch.long),
        adv=torch.zeros(1, 1),
        returns=torch.zeros(1, 1),
        values=torch.zeros(1, 1),
        start_obs=torch.zeros(1, 1),
        critic=lambda obs: obs[:, 0],
        start_features=torch.zeros(1, 1),
    )


def _build(
    make_ctx: MakeCtx,
    *,
    capacity: int,
    num_envs: int,
    p_task: float = 0.0,
    task: Task | None = None,
    post_reward: PostReward = "count",
    **settings: object,
) -> tuple[VisitedCount, VisitCount]:
    score = VisitCount(capacity=capacity, device="cpu", post_reward=post_reward)
    ctx = make_ctx(num_envs=num_envs, p_task=p_task, score=score, task=task)
    method = VisitedCount(ctx, capacity=capacity, **settings)  # pyright: ignore[reportArgumentType] # typed settings passed through
    return method, score


def _offer(
    method: VisitedCount,
    qpos: Tensor,
    qvel: Tensor,
    *,
    env_ids: Tensor | None = None,
    ep_step: Tensor | None = None,
) -> None:
    n = qpos.shape[0]
    method.offer_visited(
        torch.arange(n) if env_ids is None else env_ids,
        qpos,
        qvel,
        torch.zeros(n, 0),
        torch.ones(n, dtype=torch.long) if ep_step is None else ep_step,
    )


def _pendulum_features(qpos: Tensor, qvel: Tensor) -> Tensor:
    return torch.cat([qpos.cos(), qpos.sin(), qvel], dim=1)


def test_neighbour_count_matches_bruteforce_small(
    make_ctx: MakeCtx, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(0)
    capacity = 24
    # 5 rows of 24 per cdist chunk: the count runs over 5 chunks.
    monkeypatch.setattr(visited_count_module, "_CDIST_ELEMENTS_MAX", 5 * capacity)
    method, score = _build(make_ctx, capacity=capacity, num_envs=capacity, radius=0.8)
    qpos = (torch.rand(capacity, 1) * 2 - 1) * 3.0
    qvel = (torch.rand(capacity, 1) * 2 - 1) * 2.0
    _offer(method, qpos, qvel)

    method.update(0, _stats())

    f = _pendulum_features(qpos, qvel)
    z = (f - f.mean(dim=0)) / f.std(dim=0, unbiased=False)
    dist = torch.cdist(z, z, compute_mode="donot_use_mm_for_euclid_dist")
    expected = (dist <= 0.8).sum(dim=1) - 1  # a state is not its own neighbour
    lp, _ = score.values(torch.arange(capacity))
    assert torch.equal(method.counts(), expected)
    assert torch.allclose(lp, 1.0 / torch.sqrt(expected.float() + 1.0))


def test_counts_refresh_only_every_update_every_iterations(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method, _ = _build(make_ctx, capacity=8, num_envs=4, update_every=2)
    _offer(method, torch.zeros(4, 1), torch.zeros(4, 1))
    method.update(0, _stats())
    first = method.counts().clone()

    _offer(method, torch.zeros(4, 1), torch.zeros(4, 1))
    method.update(1, _stats())

    assert first.tolist() == [3, 3, 3, 3]
    assert method.counts().tolist() == [3, 3, 3, 3, 0, 0, 0, 0]


def test_lineage_depth_increments_along_chain(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method, _ = _build(
        make_ctx, capacity=8, num_envs=1, min_fill=1, count="lineage_depth"
    )
    _offer(method, torch.tensor([[0.1]]), torch.zeros(1, 1))  # episode from task start
    method.reset(torch.tensor([0]), None)  # only slot 0 exists: restart from depth 0
    _offer(method, torch.tensor([[0.2]]), torch.zeros(1, 1))
    after_one = method.depths().tolist()

    _, ids = method.reset(torch.tensor([0]), None)
    _offer(method, torch.tensor([[0.3]]), torch.zeros(1, 1))

    depth = method.depths()
    assert after_one == [0, 1]
    assert int(depth[2]) == int(depth[int(ids[0])]) + 1


def test_lineage_depth_is_zero_after_task_start_reset(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method, _ = _build(
        make_ctx, capacity=8, num_envs=1, min_fill=1, count="lineage_depth"
    )
    _offer(method, torch.tensor([[0.1]]), torch.zeros(1, 1))
    method.reset(torch.tensor([0]), None)
    _offer(method, torch.tensor([[0.2]]), torch.zeros(1, 1))
    method.ctx = dataclasses.replace(method.ctx, p_task=1.0)  # later resets: task start

    _, ids = method.reset(torch.tensor([0]), None)
    _offer(method, torch.tensor([[0.3]]), torch.zeros(1, 1))

    assert ids.tolist() == [-1]
    assert method.depths().tolist() == [0, 1, 0]


def test_lineage_depth_prefers_deeper_states(make_ctx: MakeCtx) -> None:
    # The first version fed depth itself as the visit count, so deeper states were drawn
    # less: a misreading. Depth stands in for "further from the start / rarer" (shortlist
    # section 3, method 3), so the draw weight must grow with depth.
    torch.manual_seed(0)
    method, score = _build(
        make_ctx, capacity=8, num_envs=1, min_fill=1, count="lineage_depth"
    )
    _offer(method, torch.tensor([[0.1]]), torch.zeros(1, 1))  # depth 0
    method.reset(torch.tensor([0]), None)
    _offer(method, torch.tensor([[0.2]]), torch.zeros(1, 1))  # depth 1
    method.update(0, _stats())
    method.reset(torch.tensor([0]), None)
    _offer(method, torch.tensor([[0.3]]), torch.zeros(1, 1))  # depth 1 or 2

    method.update(1, _stats())

    depth = method.depths()
    probs = method.slot_probs()
    lp, _ = score.values(torch.arange(3))
    deeper = depth.unsqueeze(1) > depth.unsqueeze(
        0
    )  # deeper[i, j]: slot i deeper than j
    assert int(depth[0]) == 0
    assert int(depth.max()) >= 1
    assert bool((probs.unsqueeze(1) > probs.unsqueeze(0))[deeper].all())
    # Pseudo-count max_depth - depth: the deepest slot has count 0, so lp = 1.
    assert method.counts().tolist() == (depth.max() - depth).tolist()
    assert lp[depth.argmax()].item() == pytest.approx(1.0)


def test_count_weighted_draw_prefers_rare_states(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    n = 512
    method, _ = _build(make_ctx, capacity=n, num_envs=n, radius=0.5)
    qpos = torch.full((n, 1), 0.5) + torch.randn(n, 1) * 1e-3
    qvel = torch.randn(n, 1) * 1e-3
    qpos[-1, 0], qvel[-1, 0] = -2.5, 5.0  # one rare state, far from the cluster
    _offer(method, qpos, qvel)
    method.update(0, _stats())

    _, ids = method.reset(torch.arange(n), None)

    probs = method.slot_probs()
    # Cluster states have 510 neighbours (lp = 1/sqrt(511)), the rare one none (lp = 1).
    assert probs[-1].item() / probs[0].item() == pytest.approx(511**0.5, rel=1e-4)
    assert int((ids == n - 1).sum()) >= 10  # expected about 22; uniform gives 1


def test_across_wall_share_positive_in_maze_fixture(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    maze = Maze()
    grid = maze.grid
    free = (~grid.occupied).nonzero()
    centres = (free.flip(1).float() + 0.5) * grid.cell_m
    # Cells (1, 1) and (3, 1) are free and the wall row 2 lies between them: the two points
    # are 0.5 m apart in a straight line and 16 cells apart through the maze.
    across = torch.tensor([[0.6, 0.75], [0.6, 1.25]])
    qpos = torch.cat([centres, across])
    n = qpos.shape[0]
    method, _ = _build(make_ctx, capacity=n, num_envs=n, radius=1.0, task=maze)
    _offer(method, qpos, torch.zeros(n, 2))

    method.update(0, _stats())

    share = method.diagnostics()["across_wall_neighbour_share"]
    assert 0.0 < share < 1.0


def test_across_wall_share_absent_without_path_distance(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method, _ = _build(make_ctx, capacity=8, num_envs=4)
    _offer(method, torch.zeros(4, 1), torch.zeros(4, 1))

    method.update(0, _stats())

    assert "across_wall_neighbour_share" not in method.diagnostics()


def test_ic_ids_are_buffer_slots(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method, _ = _build(make_ctx, capacity=16, num_envs=4)
    _offer(method, torch.tensor([[0.1], [0.2], [0.3], [0.4]]), torch.zeros(4, 1))

    states, ids = method.reset(torch.arange(4), None)

    table = method.ic_states()
    assert table.n == 16
    assert bool(((ids >= 0) & (ids < 4)).all())
    assert torch.equal(table.qpos[ids], states.qpos)


def test_overwritten_slot_count_is_forgotten(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method, score = _build(make_ctx, capacity=4, num_envs=4)
    _offer(method, torch.zeros(4, 1), torch.zeros(4, 1))
    method.update(0, _stats())
    lp_before, _ = score.values(torch.arange(4))

    _offer(method, torch.ones(4, 1), torch.zeros(4, 1))  # overwrites every slot
    method.reset(torch.tensor([0]), None)

    lp_after, _ = score.values(torch.arange(4))
    assert lp_before.tolist() == pytest.approx([0.5] * 4)  # 3 neighbours each
    assert lp_after.tolist() == pytest.approx([1.0] * 4)  # forgotten: count 0


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

    assert math.isnan(before)  # NaN until the first reward
    # Notified: lp now carries the (learnability + eps) factor; no IC has outcomes, so
    # learnability is 0 and lp = 1/sqrt(0 + 1) * 0.05.
    lp, _ = score.values(torch.arange(8))
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

    method.update(0, _stats())

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

    method.update(0, _stats())

    diag = method.diagnostics()
    assert diag["side/wall_ms_per_update"] > 0.0
    assert diag["buffer_size"] == 8.0


def test_rejects_a_score_other_than_visit_count(make_ctx: MakeCtx) -> None:
    with pytest.raises(TypeError, match="visit_count"):
        VisitedCount(make_ctx(num_envs=4), capacity=8)


def test_rejects_unknown_count_rule(make_ctx: MakeCtx) -> None:
    with pytest.raises(ValueError, match="count"):
        _build(make_ctx, capacity=8, num_envs=4, count="hash")


def test_draw_distribution_is_fixed_between_updates(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method, _ = _build(make_ctx, capacity=16, num_envs=8)
    _offer(method, torch.zeros(8, 1), torch.zeros(8, 1))
    method.update(0, _stats())

    _offer(method, torch.ones(8, 1), torch.zeros(8, 1))  # slots 8-15, after the update
    _, before = method.reset(torch.arange(8), None)
    method.update(1, _stats())
    _, after = method.reset(torch.arange(8), None)

    # Reset only draws from the table built at the last update (O(k log n) per reset);
    # new slots join at the next update, where their 7 neighbours make them as likely.
    assert bool((before < 8).all())
    assert bool((after >= 8).any())
