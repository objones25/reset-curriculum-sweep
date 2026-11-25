import dataclasses
import math
from collections.abc import Callable

import pytest
import torch

from resetsweep.methods.base import MethodContext
from resetsweep.methods.own_success import table_capacity
from resetsweep.methods.phased import Phased
from resetsweep.scores.steps_to_reward import StepsToReward
from resetsweep.scores.success_band import SuccessBand
from resetsweep.scores.visit_count import VisitCount
from resetsweep.types import Outcomes, RolloutStats

MakeCtx = Callable[..., MethodContext]
CAPACITY = 1024  # visited_count's default IC table


def _outcome(env_id: int, *, first_reward_step: int) -> Outcomes:
    return Outcomes(
        env_ids=torch.tensor([env_id]),
        ic_ids=torch.tensor([-1]),
        ret=torch.tensor([float(first_reward_step >= 0)]),
        success=torch.tensor([first_reward_step >= 0]),
        ep_len=torch.tensor([40]),
        reached_reward=torch.tensor([first_reward_step >= 0]),
        first_reward_step=torch.tensor([first_reward_step]),
        full=torch.tensor([True]),
        start_step=torch.tensor([0]),
    )


def _offer(method: Phased, qpos: list[float], ep_step: int) -> None:
    n = len(qpos)
    method.offer_visited(
        torch.arange(n),
        torch.tensor(qpos).unsqueeze(1),
        torch.zeros(n, 1),
        torch.zeros(n, 0),
        torch.full((n,), ep_step),
    )


def _stats(versions: torch.Tensor) -> RolloutStats:
    return RolloutStats(
        ic_ids=torch.full((1, versions.numel()), -1),
        adv=torch.zeros(1, versions.numel()),
        returns=torch.zeros(1, versions.numel()),
        values=torch.zeros(1, versions.numel()),
        start_obs=torch.zeros(CAPACITY, 1),
        critic=lambda obs: obs[:, 0],
        start_features=torch.zeros(CAPACITY, 1),
        ic_versions=versions.unsqueeze(0),
    )


def test_phased_switches_on_first_reward_and_logs_step(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method = Phased(make_ctx(num_envs=2, p_task=0.0))
    method.reset(torch.arange(2), None)
    before = method.diagnostics()
    _offer(method, [0.1, 0.5], 8)
    _offer(method, [0.2, 0.6], 16)
    method.ctx.counter.add_train(100)

    # env 0 first reaches reward at ep_step 20: its trajectory (0.1, 0.2) is stored.
    _, ids = method.reset(torch.tensor([0]), _outcome(0, first_reward_step=20))

    after = method.diagnostics()
    assert (before["phase"], after["phase"]) == (0.0, 1.0)
    assert math.isnan(before["switch_step"])
    assert after["switch_step"] == 100.0
    assert ids.tolist() == [
        1
    ]  # own_success's frontier: the stored state nearest reward
    assert method.ic_states().qpos[:3, 0].tolist() == pytest.approx([0.1, 0.2, 0.0])


def test_phased_score_follows_the_active_phase(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method = Phased(make_ctx(num_envs=2, p_task=0.0))
    method.reset(torch.arange(2), None)
    before = method.score
    _offer(method, [0.1, 0.5], 8)

    method.reset(torch.tensor([0]), _outcome(0, first_reward_step=10))

    assert isinstance(before, VisitCount)
    assert isinstance(method.score, SuccessBand)
    assert method.score.capacity == table_capacity(method.ctx.task)


def test_phased_delegates_to_visited_count_before_reward(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method = Phased(make_ctx(num_envs=2, p_task=0.0))
    method.reset(torch.arange(2), None)
    _offer(method, [0.1, 0.5], 8)
    method.update(
        0, _stats(method.episode_versions())
    )  # promotes the ring into the table

    _, ids = method.reset(torch.arange(2), _outcome(0, first_reward_step=-1))

    assert method.diagnostics()["phase"] == 0.0
    assert bool((ids >= 0).all())  # drawn from visited_count's IC table
    assert sorted(method.ic_states().qpos[:2, 0].tolist()) == pytest.approx([0.1, 0.5])


def test_phased_takes_no_score(make_ctx: MakeCtx) -> None:
    ctx = make_ctx(score=StepsToReward(capacity=CAPACITY, device="cpu"))

    with pytest.raises(ValueError, match="phased builds its own scores"):
        Phased(ctx)


def test_phased_stores_only_the_rewarded_episode(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method = Phased(make_ctx(num_envs=2, p_task=0.0))
    method.reset(torch.arange(2), None)
    _offer(method, [0.9, 0.9], 8)  # an episode of env 0 that ends without reward
    method.reset(torch.tensor([0]), _outcome(0, first_reward_step=-1))
    _offer(method, [0.1, 0.5], 8)

    method.reset(torch.tensor([0]), _outcome(0, first_reward_step=10))

    assert method.ic_states().qpos[:2, 0].tolist() == pytest.approx([0.1, 0.0])


def test_phased_drops_pre_switch_rollout_cells(
    make_ctx: MakeCtx, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(0)
    method = Phased(make_ctx(num_envs=2, p_task=0.0))
    method.reset(torch.arange(2), None)
    _offer(method, [0.1, 0.5], 8)
    _offer(method, [0.2, 0.6], 16)
    method.reset(torch.tensor([0]), _outcome(0, first_reward_step=20))
    seen: list[torch.Tensor] = []
    monkeypatch.setattr(
        method.score, "observe_rollout", lambda stats: seen.append(stats.ic_ids)
    )
    # Both cells name slot 1 with inner version 0. The first was recorded before the
    # switch (tag 2 * 0 + phase 0), the second after it (tag 2 * 0 + phase 1), so only the
    # first must be dropped, even though its inner version matches own_success's.
    stats = dataclasses.replace(
        _stats(torch.tensor([0, 1])), ic_ids=torch.tensor([[1, 1]])
    )

    method.update(1, stats)

    assert method.episode_versions().tolist() == [1, -1]  # env 0: own slot, phase 1
    assert seen[0].tolist() == [[-1, 1]]


def _switched(make_ctx: MakeCtx) -> Phased:
    """A phased method that switched on env 0's reward; env 1 is still in its phase-0
    episode."""
    method = Phased(make_ctx(num_envs=2, p_task=0.0))
    method.reset(torch.arange(2), None)
    _offer(method, [0.1, 0.5], 8)
    method.reset(torch.tensor([0]), _outcome(0, first_reward_step=10))
    return method


def test_phased_switch_with_rewarded_outcome_from_high_visited_slot(
    make_ctx: MakeCtx,
) -> None:
    torch.manual_seed(0)
    method = Phased(make_ctx(num_envs=2, p_task=0.0))
    method.reset(torch.arange(2), None)
    _offer(method, [0.1, 0.5], 8)
    high = (
        table_capacity(method.ctx.task) + 300
    )  # a visited_count slot own_success lacks
    rewarded = dataclasses.replace(
        _outcome(0, first_reward_step=10), ic_ids=torch.tensor([high])
    )

    method.reset(torch.tensor([0]), rewarded)

    assert method.phase == 1
    assert method.ic_states().qpos[:2, 0].tolist() == pytest.approx([0.1, 0.0])


def test_phased_post_switch_stale_outcome_from_visited_slot(
    make_ctx: MakeCtx, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(0)
    method = _switched(make_ctx)
    recorded: list[torch.Tensor] = []
    monkeypatch.setattr(
        method.score, "record", lambda outcomes: recorded.append(outcomes.ic_ids)
    )
    high = table_capacity(method.ctx.task) + 300
    stale = dataclasses.replace(
        _outcome(1, first_reward_step=-1), ic_ids=torch.tensor([high])
    )

    method.reset(torch.tensor([1]), stale)

    assert recorded[0].tolist() == [-1]


def test_phased_update_masks_negative_tag_cells(
    make_ctx: MakeCtx, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(0)
    method = _switched(make_ctx)
    seen: list[torch.Tensor] = []
    monkeypatch.setattr(
        method.score, "observe_rollout", lambda stats: seen.append(stats.ic_ids)
    )
    # env 0: own_success slot 1 (tag 2 * 0 + 1); env 1: still in its visited_count
    # episode from slot 500, which own_success stamps -1.
    stats = dataclasses.replace(
        _stats(torch.tensor([1, -1])), ic_ids=torch.tensor([[1, 500]])
    )

    method.update(1, stats)

    assert seen[0].tolist() == [[1, -1]]
