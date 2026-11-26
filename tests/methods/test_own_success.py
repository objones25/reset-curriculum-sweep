import math
from collections.abc import Callable, Sequence

import pytest
import torch
from torch import Tensor

from resetsweep.methods.base import MethodContext
from resetsweep.methods.own_success import OwnSuccess, table_capacity
from resetsweep.scores.base import Score
from resetsweep.scores.steps_to_reward import StepsToReward
from resetsweep.types import Outcomes, RolloutStats

MakeCtx = Callable[..., MethodContext]
STATES_PER_TRAJ = 13  # pendulum_sparse: ceil(max_steps 100 / visit_every 8)


def _build(
    make_ctx: MakeCtx,
    *,
    num_envs: int = 2,
    p_task: float = 0.0,
    score: Score | None = None,
    max_trajectories: int = 2,
    **settings: object,
) -> OwnSuccess:
    ctx = make_ctx(num_envs=num_envs, p_task=p_task, score=score)
    return OwnSuccess(
        ctx,
        max_trajectories=max_trajectories,
        **settings,  # pyright: ignore[reportArgumentType] # typed settings passed through
    )


def _offer(
    method: OwnSuccess, env_id: int, qpos: float, ep_step: int, *, qvel: float = 0.0
) -> None:
    """Offer one state for `env_id` only (the other envs are not offered)."""
    method.offer_visited(
        torch.tensor([env_id]),
        torch.tensor([[qpos]]),
        torch.tensor([[qvel]]),
        torch.zeros(1, 0),
        torch.tensor([ep_step]),
    )


def _outcomes(
    env_ids: Sequence[int],
    *,
    ic_ids: Sequence[int] | None = None,
    success: Sequence[bool] | None = None,
    first_reward_step: Sequence[int] | None = None,
) -> Outcomes:
    n = len(env_ids)
    first = [-1] * n if first_reward_step is None else list(first_reward_step)
    succ = [False] * n if success is None else list(success)
    return Outcomes(
        env_ids=torch.tensor(env_ids),
        ic_ids=torch.tensor([-1] * n if ic_ids is None else list(ic_ids)),
        ret=torch.tensor(succ, dtype=torch.float32),
        success=torch.tensor(succ),
        ep_len=torch.full((n,), 40),
        reached_reward=torch.tensor([f >= 0 for f in first]),
        first_reward_step=torch.tensor(first),
        full=torch.ones(n, dtype=torch.bool),
        start_step=torch.zeros(n, dtype=torch.long),
    )


def _store(method: OwnSuccess, env_id: int, qpos: Sequence[float]) -> None:
    """Offer `qpos` at ep_steps 8, 16, ... for `env_id`, then end its episode with reward
    first reached at the last offered step."""
    for i, q in enumerate(qpos):
        _offer(method, env_id, q, 8 * (i + 1))
    method.reset(
        torch.tensor([env_id]),
        _outcomes([env_id], first_reward_step=[8 * len(qpos)]),
    )


def _stats(capacity: int, *, ic_versions: Tensor | None) -> RolloutStats:
    return RolloutStats(
        ic_ids=torch.full((1, 1), -1),
        adv=torch.zeros(1, 1),
        returns=torch.zeros(1, 1),
        values=torch.zeros(1, 1),
        start_obs=torch.zeros(capacity, 1),
        critic=lambda obs: obs[:, 0],
        start_features=torch.zeros(capacity, 1),
        ic_versions=ic_versions,
    )


def test_own_success_inert_before_first_reward(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method = _build(make_ctx)
    _offer(method, 0, 0.3, 8)
    _offer(method, 0, 0.4, 16)

    _, ids = method.reset(torch.tensor([0, 1]), _outcomes([0, 1]))

    assert ids.tolist() == [-1, -1]
    assert method.diagnostics()["buffer_size"] == 0.0
    assert bool((method.ic_states().qpos == 0.0).all())


def test_stores_trajectory_on_first_reward_with_steps_to_reward(
    make_ctx: MakeCtx,
) -> None:
    torch.manual_seed(0)
    score = StepsToReward(capacity=2 * STATES_PER_TRAJ, device="cpu")
    method = _build(make_ctx, score=score)
    for q, step in [(0.1, 8), (0.2, 16), (0.3, 24), (0.4, 32)]:
        _offer(method, 0, q, step)

    # Reward first reached at ep_step 26: the state offered at 32 came after it.
    method.reset(torch.tensor([0]), _outcomes([0], first_reward_step=[26]))

    table = method.ic_states().qpos[:, 0]
    lp, trusted = score.values(torch.arange(3))
    assert table[:4].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.0])
    assert lp.tolist() == [-18.0, -10.0, -2.0]  # lp = -(26 - ep_step)
    assert trusted.tolist() == [True, True, True]
    assert method.diagnostics()["buffer_size"] == 3.0


def test_frontier_moves_back_on_high_success_and_forward_on_collapse(
    make_ctx: MakeCtx,
) -> None:
    torch.manual_seed(0)
    method = _build(make_ctx, order="frontier", window=2)
    _store(method, 0, [0.1, 0.2, 0.3, 0.4])
    _, start = method.reset(torch.tensor([0, 1]), None)

    _, back = method.reset(
        torch.tensor([0, 1]), _outcomes([0, 1], ic_ids=[3, 3], success=[True, True])
    )
    _, forward = method.reset(
        torch.tensor([0, 1]), _outcomes([0, 1], ic_ids=[2, 2], success=[False, False])
    )

    assert start.tolist() == [3, 3]  # the frontier starts next to the reward
    assert back.tolist() == [2, 2]  # 2/2 successes >= advance 0.8: one step back
    assert forward.tolist() == [3, 3]  # 0/2 successes <= retreat 0.15: one step forward


def test_frontier_ignores_outcomes_from_behind_the_frontier(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method = _build(make_ctx, order="frontier", window=2)
    _store(method, 0, [0.1, 0.2, 0.3, 0.4])
    method.reset(torch.tensor([0, 1]), None)
    method.reset(
        torch.tensor([0, 1]), _outcomes([0, 1], ic_ids=[3, 3], success=[True, True])
    )

    # Episodes that started at slot 3 finish after the frontier moved to 2: not counted.
    _, ids = method.reset(
        torch.tensor([0, 1]), _outcomes([0, 1], ic_ids=[2, 2], success=[True, True])
    )
    _, still = method.reset(torch.tensor([0, 1]), None)

    assert ids.tolist() == [1, 1]
    assert still.tolist() == [1, 1]


def test_frontier_stays_inside_the_trajectory(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method = _build(make_ctx, order="frontier", window=2)
    _store(method, 0, [0.1, 0.2])
    method.reset(torch.tensor([0, 1]), None)

    _, ids = method.reset(
        torch.tensor([0, 1]), _outcomes([0, 1], ic_ids=[1, 1], success=[False, False])
    )

    assert ids.tolist() == [1, 1]  # already at the last state: retreat is clamped


def test_uniform_order_draws_along_whole_trajectory(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    n = 2000
    method = _build(make_ctx, num_envs=n)
    _store(method, 0, [0.1, 0.2, 0.3])
    _store(method, 1, [0.5, 0.6])

    _, ids = method.reset(torch.arange(n), None)

    counts = torch.bincount(ids, minlength=2 * STATES_PER_TRAJ)
    stored = [0, 1, 2, STATES_PER_TRAJ, STATES_PER_TRAJ + 1]
    # Five stored states in two trajectories, 400 draws expected each;
    # 4.5 binomial standard deviations is 4.5 * sqrt(2000 * 0.2 * 0.8) = 80.5.
    assert counts.nonzero()[:, 0].tolist() == stored
    assert all(abs(counts[i].item() - 400) < 81 for i in stored)


def test_task_start_share_follows_p_task(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    n = 2000
    method = _build(make_ctx, num_envs=n, p_task=0.5)
    _store(method, 0, [0.1, 0.2])

    _, ids = method.reset(torch.arange(n), None)

    task_share = (ids == -1).float().mean().item()
    assert task_share == pytest.approx(0.5, abs=0.05)


def test_table_keeps_earliest_trajectories_once_full(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method = _build(make_ctx, max_trajectories=1)
    _store(method, 0, [0.1, 0.2])

    _store(method, 1, [0.7, 0.8, 0.9])

    assert method.ic_states().qpos[:4, 0].tolist() == pytest.approx(
        [0.1, 0.2, 0.0, 0.0]
    )


def test_success_from_a_stored_start_is_not_stored_again(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method = _build(make_ctx)
    _store(method, 0, [0.1, 0.2])
    _, ids = method.reset(torch.tensor([1]), None)
    _offer(method, 1, 0.25, 8)

    method.reset(
        torch.tensor([1]), _outcomes([1], ic_ids=ids.tolist(), first_reward_step=[8])
    )

    assert method.diagnostics()["buffer_size"] == 2.0


def test_trajectory_buffer_restarts_at_each_reset(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method = _build(make_ctx)
    _offer(method, 0, 0.9, 8)  # from an episode that ended without reward
    method.reset(torch.tensor([0]), _outcomes([0]))

    _store(method, 0, [0.1, 0.2])

    assert method.ic_states().qpos[:3, 0].tolist() == pytest.approx([0.1, 0.2, 0.0])


def test_stores_a_whole_episode_of_max_steps(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method = _build(make_ctx)
    qpos = [0.05 * (i + 1) for i in range(STATES_PER_TRAJ)]
    # Offers at ep_steps 4, 12, ..., 100: the most a 100-step episode can get.
    for i, q in enumerate(qpos):
        _offer(method, 0, q, 4 + 8 * i)

    method.reset(torch.tensor([0]), _outcomes([0], first_reward_step=[100]))

    table = method.ic_states().qpos[:STATES_PER_TRAJ, 0]
    assert table.tolist() == pytest.approx(qpos)
    assert method.diagnostics()["buffer_size"] == float(STATES_PER_TRAJ)


def test_table_capacity_follows_task_horizon(make_ctx: MakeCtx) -> None:
    method = _build(make_ctx, max_trajectories=3)

    assert table_capacity(method.ctx.task, max_trajectories=3) == 3 * STATES_PER_TRAJ
    assert method.ic_states().n == 3 * STATES_PER_TRAJ


def test_rejects_score_with_wrong_capacity(make_ctx: MakeCtx) -> None:
    score = StepsToReward(capacity=3, device="cpu")

    with pytest.raises(ValueError, match="score capacity 3 must equal"):
        _build(make_ctx, score=score)


def test_rejects_retreat_not_below_advance(make_ctx: MakeCtx) -> None:
    with pytest.raises(ValueError, match="retreat must be below advance"):
        _build(make_ctx, advance=0.5, retreat=0.5)


def test_update_requires_ic_versions(make_ctx: MakeCtx) -> None:
    method = _build(make_ctx)

    with pytest.raises(ValueError, match="needs RolloutStats.ic_versions"):
        method.update(0, _stats(2 * STATES_PER_TRAJ, ic_versions=None))


def test_episode_versions_mark_stored_starts_zero_and_task_starts_minus_one(
    make_ctx: MakeCtx,
) -> None:
    torch.manual_seed(0)
    method = _build(make_ctx, p_task=0.0)
    _store(method, 0, [0.1])
    method.reset(torch.tensor([0]), None)

    assert method.episode_versions().tolist() == [0, -1]


def test_own_success_skips_invalid_states_and_keeps_tags(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    score = StepsToReward(capacity=2 * STATES_PER_TRAJ, device="cpu")
    method = _build(make_ctx, score=score, order="frontier")
    _offer(method, 0, 0.1, 8)
    _offer(method, 0, 0.2 + 2 * math.pi, 16)  # past +pi: wrapped to 0.2, kept
    _offer(method, 0, 0.3, 24, qvel=1e3)  # faster than the start box allows: dropped
    _offer(method, 0, 0.4, 32)

    method.reset(torch.tensor([0]), _outcomes([0], first_reward_step=[32]))

    lp, trusted = score.values(torch.arange(4))
    _, ids = method.reset(torch.tensor([0, 1]), None)
    assert method.ic_states().qpos[:4, 0].tolist() == pytest.approx(
        [0.1, 0.2, 0.4, 0.0], abs=1e-5
    )
    assert lp[:3].tolist() == [-24.0, -16.0, 0.0]  # tags of the kept states, in order
    assert trusted.tolist() == [True, True, True, False]
    assert ids.tolist() == [2, 2]  # the frontier indexes the stored states


def test_own_success_does_not_store_trajectory_without_valid_states(
    make_ctx: MakeCtx,
) -> None:
    torch.manual_seed(0)
    method = _build(make_ctx)
    _offer(method, 0, 0.1, 8, qvel=1e3)

    method.reset(torch.tensor([0]), _outcomes([0], first_reward_step=[8]))

    assert method.diagnostics()["trajectories"] == 0.0
