from collections.abc import Callable, Sequence

import pytest
import torch
from torch import Tensor

from resetsweep.methods.base import MethodContext
from resetsweep.methods.buffer_scored import BufferScored
from resetsweep.scores.base import Score
from resetsweep.scores.critic import CriticLoss
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


def _stats(
    ic_capacity: int,
    *,
    ic_ids: Tensor | None = None,
    ic_versions: Tensor | None = None,
    adv: Tensor | None = None,
) -> RolloutStats:
    """Rollout data; by default one task-start cell, which no score reads."""
    ids = torch.full((1, 1), -1) if ic_ids is None else ic_ids
    return RolloutStats(
        ic_ids=ids,
        adv=torch.zeros(ids.shape) if adv is None else adv,
        returns=torch.zeros(ids.shape),
        values=torch.zeros(ids.shape),
        start_obs=torch.zeros(ic_capacity, 1),
        critic=lambda obs: obs[:, 0],
        start_features=torch.zeros(ic_capacity, 1),
        ic_versions=torch.zeros(ids.shape, dtype=torch.long)
        if ic_versions is None
        else ic_versions,
    )


def _build(
    make_ctx: MakeCtx,
    *,
    capacity: int = 8,
    num_envs: int = 8,
    ic_capacity: int = 8,
    score: Score | None = None,
    **settings: object,
) -> tuple[BufferScored, Score]:
    score = score or SuccessBand(capacity=ic_capacity, device="cpu", form="pq")
    ctx = make_ctx(num_envs=num_envs, p_task=0.0, score=score)
    method = BufferScored(ctx, capacity=capacity, ic_capacity=ic_capacity, **settings)  # pyright: ignore[reportArgumentType] # typed settings passed through
    return method, score


def _filled(make_ctx: MakeCtx, **settings: object) -> tuple[BufferScored, Score]:
    """A method whose 8-slot IC table was filled from an 8-state ring at update 0."""
    method, score = _build(make_ctx, **settings)  # pyright: ignore[reportArgumentType] # typed settings passed through
    _offer(method, [0.1 * i for i in range(1, 9)])
    method.update(0, _stats(8))
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


def _count(score: Score, slot: int) -> int:
    assert isinstance(score, SuccessBand)
    return int(score.table.count(torch.tensor([slot]))[0])


def test_resets_use_task_start_until_first_promotion(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method, _ = _build(make_ctx)
    _offer(method, [0.1 * i for i in range(1, 9)])

    _, before = method.reset(torch.arange(8), None)
    method.update(0, _stats(8))
    _, after = method.reset(torch.arange(8), None)

    assert before.tolist() == [-1] * 8
    assert bool(((after >= 0) & (after < 8)).all())


def test_ic_ids_are_table_slots_and_ic_states_is_the_table(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method, _ = _filled(make_ctx, capacity=16, ic_capacity=4)

    states, ids = method.reset(torch.arange(8), None)
    table = method.ic_states()

    assert table.n == 4
    assert bool(((ids >= 0) & (ids < 4)).all())
    assert torch.equal(table.qpos[ids], states.qpos)


def test_staleness_raises_probability_of_unvisited_ic(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method, _ = _filled(make_ctx, ic_capacity=4, staleness_coef=0.3)

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
    method, _ = _filled(make_ctx, ic_capacity=4, staleness_coef=0.0)

    method.reset(torch.tensor([0, 1]), None)

    assert method.slot_probs().tolist() == pytest.approx([0.25] * 4)


def _two_slots_with_outcomes(make_ctx: MakeCtx) -> tuple[BufferScored, int, int]:
    """slot_a gets 2 outcomes (1 success), slot_b 3 (1 success); min_count is 3."""
    method, _ = _filled(
        make_ctx, min_count=3, lanes=4, staleness_coef=0.0, rehearsal=0.0
    )
    _, ids = method.reset(torch.arange(8), None)
    _report(method, [0, 1], ids[[0, 1]], [True, False])
    _report(method, [4, 5, 6], ids[[4, 5, 6]], [True, False, False])
    return method, int(ids[0]), int(ids[4])


def test_min_count_marks_untrusted(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)

    method, slot_a, slot_b = _two_slots_with_outcomes(make_ctx)

    trusted = method.trusted()
    assert slot_a != slot_b
    assert not bool(trusted[slot_a])
    assert bool(trusted[slot_b])
    assert int(trusted.sum()) == 1


def test_untrusted_slots_get_the_median_trusted_probability(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)

    method, slot_a, slot_b = _two_slots_with_outcomes(make_ctx)

    # Only slot_b is trusted, so every slot carries its probability: uniform 1/8.
    assert slot_a != slot_b
    assert method.slot_probs().tolist() == pytest.approx([1 / 8] * 8)


def test_score_part_ranks_only_trusted_slots(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method, _ = _filled(
        make_ctx, min_count=3, lanes=4, staleness_coef=0.0, rehearsal=0.0
    )
    _, ids = method.reset(torch.arange(8), None)
    slot_a, slot_b = int(ids[0]), int(ids[4])
    # slot_a: success 1/3, pq = 2/9; slot_b: success 0/3, pq = 0. Both trusted.
    _report(method, [0, 1, 2], ids[:3], [True, False, False])
    _report(method, [4, 5, 6], ids[4:7], [False, False, False])

    probs = method.slot_probs()

    # Rank weights over the 2 trusted slots only, beta = 0.1: 1 and 2^-10. The 6 untrusted
    # slots take the lower median of the two, 2^-10. Normalised over 8 slots.
    total = 1.0 + 7 * 2.0**-10
    assert slot_a != slot_b
    assert probs[slot_a].item() == pytest.approx(1.0 / total)
    assert probs[slot_b].item() == pytest.approx(2.0**-10 / total)


def _forgotten_probs(make_ctx: MakeCtx, *, rehearsal: float) -> tuple[Tensor, int]:
    """Slot probabilities after one slot succeeded once and then failed five times."""
    torch.manual_seed(0)
    method, _ = _filled(make_ctx, lanes=8, staleness_coef=0.0, rehearsal=rehearsal)
    _, ids = method.reset(torch.arange(8), None)
    _report(method, [0, 1, 2, 3, 4, 5], ids[:6], [True] + [False] * 5)
    return method.slot_probs(), int(ids[0])


def test_rehearsal_share_goes_to_forgotten_ics(make_ctx: MakeCtx) -> None:
    base, slot = _forgotten_probs(make_ctx, rehearsal=0.0)

    mixed, mixed_slot = _forgotten_probs(make_ctx, rehearsal=0.3)

    # Recent success 1/6 < 0.5 after a success: the slot is the only rehearsal candidate.
    others = torch.arange(8) != slot
    assert mixed_slot == slot
    assert mixed[slot].item() == pytest.approx(0.3 + 0.7 * base[slot].item())
    assert mixed[others].tolist() == pytest.approx((0.7 * base[others]).tolist())


def test_rehearsal_ignores_slots_still_succeeding(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method, _ = _filled(make_ctx, lanes=8, staleness_coef=0.0, rehearsal=0.3)
    _, ids = method.reset(torch.arange(8), None)

    _report(method, [0, 1, 2], ids[:3], [True, True, False])  # recent success 2/3

    assert method.diagnostics()["rehearsal_candidates"] == 0.0


def test_lanes_assigns_same_ic_to_lane_group(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method, _ = _build(make_ctx, capacity=16, num_envs=16, lanes=4)
    _offer(method, [0.05 * i for i in range(1, 17)])
    method.update(0, _stats(8))

    states, ids = method.reset(torch.arange(16), None)

    groups = ids.view(4, 4)
    assert bool((groups == groups[:, :1]).all())
    assert torch.equal(states.qpos[:, 0], method.ic_states().qpos[ids, 0])


def test_promotion_replaces_lowest_probability_slots(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method, _ = _filled(
        make_ctx, ic_capacity=4, promote_frac=0.5, staleness_coef=1.0, rehearsal=0.0
    )
    # Pure staleness: slots drawn at the latest episode count have the lowest probability.
    _, drawn = method.reset(torch.tensor([0, 1, 2, 3, 4, 5, 6]), None)
    probs = method.slot_probs()
    lowest = set(torch.argsort(probs, stable=True)[:2].tolist())
    before = method.ic_states().qpos.clone()
    versions = method.ic_versions()

    method.update(1, _stats(4))

    changed = set((method.ic_versions() != versions).nonzero()[:, 0].tolist())
    assert changed == lowest
    assert len(set(drawn.tolist())) >= 2
    kept = [i for i in range(4) if i not in lowest]
    assert torch.equal(method.ic_states().qpos[kept], before[kept])


def test_promotion_only_every_update_every_iterations(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method, _ = _filled(make_ctx, ic_capacity=4, promote_frac=1.0, update_every=2)
    versions = method.ic_versions()

    method.update(1, _stats(4))
    after_odd = method.ic_versions()
    method.update(2, _stats(4))

    assert torch.equal(after_odd, versions)
    assert bool((method.ic_versions() > versions).all())


def test_replaced_slot_is_forgotten_in_score(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method, score = _filled(make_ctx, ic_capacity=4, lanes=4, promote_frac=1.0)
    _, ids = method.reset(torch.arange(4), None)
    slot = int(ids[0])
    _report(method, [0, 1], ids[:2], [True, False])
    recorded = _count(score, slot)

    method.update(1, _stats(4))  # promote_frac 1.0 replaces every slot

    assert recorded == 2
    assert _count(score, slot) == 0


def test_outcome_from_replaced_slot_is_dropped(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method, score = _filled(make_ctx, ic_capacity=4, lanes=4, promote_frac=1.0)
    _, ids = method.reset(torch.arange(4), None)
    slot = int(ids[0])

    method.update(1, _stats(4))  # the state envs 0-3 started from is replaced
    _report(method, [0, 1, 2], ids[:3], [True, False, True])

    assert _count(score, slot) == 0
    assert method.diagnostics()["stale_outcome_share"] == 1.0


def test_outcome_from_kept_slot_is_recorded(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method, score = _filled(make_ctx, ic_capacity=4, lanes=4)
    _, ids = method.reset(torch.arange(4), None)

    _report(method, [0, 1, 2], ids[:3], [True, False, True])

    assert _count(score, int(ids[0])) == 3
    assert method.diagnostics()["stale_outcome_share"] == 0.0


def test_rollout_cells_from_replaced_slot_are_dropped(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    score = CriticLoss(capacity=1, device="cpu", form="abs_gae")
    method, _ = _build(
        make_ctx,
        num_envs=2,
        ic_capacity=1,
        score=score,
        update_every=2,
        promote_frac=1.0,
    )
    _offer(method, [0.1, 0.2])
    method.update(0, _stats(1))  # fills the single IC slot
    method.reset(torch.tensor([0]), None)  # env 0 starts on slot 0
    method.update(2, _stats(1))  # slot 0 replaced while env 0's episode runs on
    method.reset(torch.tensor([1]), None)  # env 1 starts on the new occupant
    versions = method.episode_versions()  # [old, new]
    stats = _stats(
        1,
        ic_ids=torch.zeros(2, 2, dtype=torch.long),
        ic_versions=versions.expand(2, 2).clone(),
        adv=torch.tensor([[5.0, 1.0], [5.0, 1.0]]),  # env 0 (old occupant) adv 5
    )

    method.update(3, stats)  # odd iteration: observe only, no promotion

    lp, trusted = score.values(torch.tensor([0]))
    assert int(versions[0]) != int(versions[1])
    assert bool(trusted[0])
    assert lp.item() == pytest.approx(1.0)  # only env 1's cells; 3.0 if env 0 leaked


def test_update_requires_ic_versions(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    method, _ = _build(make_ctx)
    stats = RolloutStats(
        ic_ids=torch.zeros(1, 1, dtype=torch.long),
        adv=torch.zeros(1, 1),
        returns=torch.zeros(1, 1),
        values=torch.zeros(1, 1),
        start_obs=torch.zeros(8, 1),
        critic=lambda obs: obs[:, 0],
        start_features=torch.zeros(8, 1),
    )

    with pytest.raises(ValueError, match="ic_versions"):
        method.update(0, stats)


def _simulate_turnover(method: BufferScored, *, num_envs: int, updates: int) -> None:
    """Pendulum-like run: every tick (8 env steps) all envs offer a state; episodes last 13
    ticks (104 steps), staggered; an update every 16 ticks. Successes are coin flips."""
    gen = torch.Generator().manual_seed(1)
    left = torch.randint(1, 14, (num_envs,), generator=gen)
    ids = torch.full((num_envs,), -1)
    for tick in range(16 * updates):
        qpos = (torch.rand(num_envs, 1, generator=gen) * 2 - 1) * 3.0
        method.offer_visited(
            torch.arange(num_envs),
            qpos,
            torch.zeros(num_envs, 1),
            torch.zeros(num_envs, 0),
            torch.full((num_envs,), 13) - left,
        )
        left -= 1
        done = (left == 0).nonzero()[:, 0]
        success = torch.rand(done.numel(), generator=gen) < 0.5
        outcomes = _outcomes(done.tolist(), ids[done], success.tolist())
        _, ids[done] = method.reset(done, outcomes)
        left[done] = 13
        if (tick + 1) % 16 == 0:
            method.update(tick // 16, _stats(method.ic_states().n))


def test_ic_slots_become_trusted_under_realistic_turnover(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    num_envs = 512
    # Ring lifetime: 2048 slots / 512 offers per tick = 4 ticks = 32 steps, well under the
    # 104-step episodes, so ring slots never live long enough to collect an outcome.
    method, _ = _build(
        make_ctx, capacity=2048, num_envs=num_envs, ic_capacity=64, min_count=3
    )

    _simulate_turnover(method, num_envs=num_envs, updates=8)

    diag = method.diagnostics()
    assert int(method.trusted().sum()) >= 32
    assert diag["stale_outcome_share"] < 0.2


def test_rejects_score_capacity_other_than_ic_capacity(make_ctx: MakeCtx) -> None:
    score = SuccessBand(capacity=16, device="cpu", form="pq")

    with pytest.raises(ValueError, match="ic_capacity"):
        _build(make_ctx, ic_capacity=8, score=score)


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
        ("promote_frac", 0.0),
        ("update_every", 0),
    ],
)
def test_rejects_out_of_range_settings(
    make_ctx: MakeCtx, setting: str, value: float
) -> None:
    with pytest.raises(ValueError, match=setting):
        _build(make_ctx, **{setting: value})  # pyright: ignore[reportArgumentType] # one setting per case


def test_rejects_ic_capacity_above_ring_capacity(make_ctx: MakeCtx) -> None:
    with pytest.raises(ValueError, match="ic_capacity"):
        _build(make_ctx, capacity=8, ic_capacity=16)
