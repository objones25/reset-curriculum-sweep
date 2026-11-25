from collections.abc import Callable, Sequence

import pytest
import torch
from torch import Tensor

from resetsweep.methods.base import MethodContext
from resetsweep.methods.buffer_scored import BufferScored
from resetsweep.scores.success_band import SuccessBand
from resetsweep.types import Outcomes, RolloutStats

MakeCtx = Callable[..., MethodContext]


def _offer(method: BufferScored, qpos_values: Sequence[float]) -> None:
    n = len(qpos_values)
    method.offer_visited(
        torch.arange(n),
        torch.tensor(qpos_values).unsqueeze(1),
        torch.zeros(n, 1),
        torch.zeros(n, 0),
        torch.ones(n, dtype=torch.long),
    )


def _outcomes(
    env_ids: Sequence[int], ic_ids: Tensor, success: Sequence[bool]
) -> Outcomes:
    n = len(env_ids)
    return Outcomes(
        env_ids=torch.tensor(env_ids),
        ic_ids=ic_ids,
        ret=torch.tensor(success, dtype=torch.float32),
        success=torch.tensor(success, dtype=torch.bool),
        ep_len=torch.full((n,), 10),
        reached_reward=torch.tensor(success, dtype=torch.bool),
        full=torch.ones(n, dtype=torch.bool),
        start_step=torch.zeros(n, dtype=torch.long),
    )


def _band(capacity: int) -> SuccessBand:
    return SuccessBand(capacity=capacity, device="cpu", form="pq")


def _build(
    make_ctx: MakeCtx, *, capacity: int = 8, num_envs: int = 8, **settings: object
) -> tuple[BufferScored, SuccessBand]:
    score = _band(capacity)
    ctx = make_ctx(num_envs=num_envs, p_task=0.0, score=score)
    method = BufferScored(ctx, capacity=capacity, **settings)  # pyright: ignore[reportArgumentType] # test passes typed settings through
    return method, score


def _report(
    method: BufferScored,
    env_ids: Sequence[int],
    ic_ids: Tensor,
    success: Sequence[bool],
) -> Tensor:
    """Finish the episodes of `env_ids` (started on `ic_ids`) and return their new ic ids."""
    _, new_ids = method.reset(
        torch.tensor(env_ids), _outcomes(env_ids, ic_ids, success)
    )
    return new_ids


def test_staleness_raises_probability_of_unvisited_ic(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method, _ = _build(make_ctx, capacity=4, num_envs=4, staleness_coef=0.3)
    _offer(method, [0.1, 0.2, 0.3, 0.4])

    _, drawn = method.reset(torch.tensor([0, 1]), None)

    probs = method.slot_probs()
    visited = torch.zeros(4, dtype=torch.bool)
    visited[drawn] = True
    # No slot has min_count outcomes, so the score part is uniform (0.25 each); a slot drawn
    # at the latest episode count has staleness 0 and keeps only (1 - rho) * 0.25.
    assert probs[visited].tolist() == pytest.approx([0.7 * 0.25] * int(visited.sum()))
    assert bool((probs[~visited].min() > probs[visited].max()).item())
    assert probs.sum().item() == pytest.approx(1.0)


def test_staleness_coef_zero_ignores_staleness(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method, _ = _build(make_ctx, capacity=4, num_envs=4, staleness_coef=0.0)
    _offer(method, [0.1, 0.2, 0.3, 0.4])

    method.reset(torch.tensor([0, 1]), None)

    assert method.slot_probs().tolist() == pytest.approx([0.25] * 4)


def test_min_count_marks_untrusted(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method, _ = _build(
        make_ctx, min_count=3, lanes=4, staleness_coef=0.0, rehearsal=0.0
    )
    _offer(method, [0.1 * i for i in range(1, 9)])
    _, ids = method.reset(torch.arange(8), None)
    slot_a, slot_b = int(ids[0]), int(ids[4])

    _report(method, [0, 1], ids[[0, 1]], [True, False])  # 2 outcomes on slot_a
    _report(method, [4, 5, 6], ids[[4, 5, 6]], [True, False, False])  # 3 on slot_b

    trusted = method.trusted()
    assert slot_a != slot_b
    assert not bool(trusted[slot_a])
    assert bool(trusted[slot_b])
    assert int(trusted.sum()) == 1


def test_untrusted_slots_get_the_median_trusted_probability(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method, _ = _build(
        make_ctx, min_count=3, lanes=4, staleness_coef=0.0, rehearsal=0.0
    )
    _offer(method, [0.1 * i for i in range(1, 9)])
    _, ids = method.reset(torch.arange(8), None)
    slot_a, slot_b = int(ids[0]), int(ids[4])

    _report(method, [0, 1], ids[[0, 1]], [True, False])
    _report(method, [4, 5, 6], ids[[4, 5, 6]], [True, False, False])

    probs = method.slot_probs()
    # Only slot_b is trusted, so every slot carries its probability: uniform 1/8.
    assert slot_a != slot_b
    assert probs.tolist() == pytest.approx([1 / 8] * 8)


def _forgotten_setup(make_ctx: MakeCtx, *, rehearsal: float) -> tuple[Tensor, int]:
    """Slot probabilities after one slot succeeded once and then failed five times."""
    torch.manual_seed(0)
    method, _ = _build(make_ctx, lanes=8, staleness_coef=0.0, rehearsal=rehearsal)
    _offer(method, [0.1 * i for i in range(1, 9)])
    _, ids = method.reset(torch.arange(8), None)
    _report(method, [0, 1, 2, 3, 4, 5], ids[:6], [True] + [False] * 5)
    return method.slot_probs(), int(ids[0])


def test_rehearsal_share_goes_to_forgotten_ics(make_ctx: MakeCtx) -> None:
    base, slot = _forgotten_setup(make_ctx, rehearsal=0.0)

    mixed, mixed_slot = _forgotten_setup(make_ctx, rehearsal=0.3)

    # Recent success 1/6 < 0.5 after a success: the slot is the only rehearsal candidate.
    others = torch.arange(8) != slot
    assert mixed_slot == slot
    assert mixed[slot].item() == pytest.approx(0.3 + 0.7 * base[slot].item())
    assert mixed[others].tolist() == pytest.approx((0.7 * base[others]).tolist())


def test_rehearsal_ignores_slots_still_succeeding(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method, _ = _build(make_ctx, lanes=8, staleness_coef=0.0, rehearsal=0.3)
    _offer(method, [0.1 * i for i in range(1, 9)])
    _, ids = method.reset(torch.arange(8), None)

    _report(method, [0, 1, 2], ids[:3], [True, True, False])  # recent success 2/3

    assert method.diagnostics()["rehearsal_candidates"] == 0.0


def test_lanes_assigns_same_ic_to_lane_group(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method, _ = _build(make_ctx, capacity=16, num_envs=16, lanes=4)
    _offer(method, [0.05 * i for i in range(1, 17)])

    states, ids = method.reset(torch.arange(16), None)

    groups = ids.view(4, 4)
    assert bool((groups == groups[:, :1]).all())
    assert bool((ids >= 0).all())
    assert torch.equal(states.qpos[:, 0], method.ic_states().qpos[ids, 0])


def test_ic_ids_are_buffer_slots_and_ic_states_has_capacity_rows(
    make_ctx: MakeCtx,
) -> None:
    torch.manual_seed(0)
    method, _ = _build(make_ctx, capacity=8, num_envs=4)
    _offer(method, [0.1, 0.2, 0.3, 0.4])

    states, ids = method.reset(torch.arange(4), None)
    table = method.ic_states()

    assert table.n == 8
    assert bool(((ids >= 0) & (ids < 4)).all())
    assert torch.equal(table.qpos[ids], states.qpos)


def test_overwritten_slot_is_forgotten_in_score(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method, score = _build(make_ctx, capacity=4, num_envs=4, lanes=4)
    _offer(method, [0.1, 0.2, 0.3, 0.4])
    _, ids = method.reset(torch.arange(4), None)
    slot = int(ids[0])
    _report(method, [0, 1], ids[:2], [True, False])
    recorded = int(score.table.count(torch.tensor([slot]))[0])

    _offer(method, [0.5, 0.6, 0.7, 0.8])  # overwrites every slot
    method.reset(torch.tensor([3]), None)

    assert recorded == 2
    assert int(score.table.count(torch.tensor([slot]))[0]) == 0


def test_outcome_from_overwritten_slot_is_dropped(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method, score = _build(make_ctx, capacity=4, num_envs=4, lanes=4)
    _offer(method, [0.1, 0.2, 0.3, 0.4])
    _, ids = method.reset(torch.arange(4), None)
    slot = int(ids[0])

    _offer(method, [0.5, 0.6, 0.7, 0.8])  # the state envs 0-3 started from is gone
    _report(method, [0, 1, 2], ids[:3], [True, False, True])

    assert int(score.table.count(torch.tensor([slot]))[0]) == 0


def test_outcome_from_kept_slot_is_recorded(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method, score = _build(make_ctx, capacity=4, num_envs=4, lanes=4)
    _offer(method, [0.1, 0.2, 0.3, 0.4])
    _, ids = method.reset(torch.arange(4), None)
    slot = int(ids[0])

    _report(method, [0, 1, 2], ids[:3], [True, False, True])

    assert int(score.table.count(torch.tensor([slot]))[0]) == 3


def test_update_passes_rollout_to_score(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method, score = _build(make_ctx)
    seen: list[RolloutStats] = []
    score.observe_rollout = seen.append  # type: ignore[method-assign] # record the call
    stats = RolloutStats(
        ic_ids=torch.zeros(1, 1, dtype=torch.long),
        adv=torch.zeros(1, 1),
        returns=torch.zeros(1, 1),
        values=torch.zeros(1, 1),
        start_obs=torch.zeros(8, 1),
        critic=lambda obs: obs[:, 0],
        start_features=torch.zeros(8, 1),
    )

    method.update(0, stats)

    assert seen == [stats]


def test_requires_a_score(make_ctx: MakeCtx) -> None:
    with pytest.raises(ValueError, match="needs a score"):
        BufferScored(make_ctx(num_envs=4), capacity=8)


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("staleness_coef", 1.5),
        ("discount", 0.0),
        ("min_count", 0),
        ("rehearsal", -0.1),
        ("lanes", 0),
    ],
)
def test_rejects_out_of_range_settings(
    make_ctx: MakeCtx, setting: str, value: float
) -> None:
    with pytest.raises(ValueError, match=setting):
        _build(make_ctx, **{setting: value})  # pyright: ignore[reportArgumentType] # one setting per case
