import pytest
import torch

from resetsweep.scores import SCORES
from resetsweep.scores.side_models import RND, StepDistance, SuccessClassifier, band_lp

CAP = 40


def _ids(n: int = CAP) -> torch.Tensor:
    return torch.arange(n)


def _clusters(seed: int = 0) -> torch.Tensor:
    """ICs 0-19 near (-2, -2), ICs 20-39 near (+2, +2)."""
    gen = torch.Generator().manual_seed(seed)
    centre = torch.cat([-2 * torch.ones(20, 2), 2 * torch.ones(20, 2)])
    return centre + 0.3 * torch.randn(CAP, 2, generator=gen)


def _auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    return (pos[:, None] > neg[None, :]).float().mean().item()


def _classifier(**kw: object) -> SuccessClassifier:
    return SuccessClassifier(capacity=CAP, device="cpu", **kw)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


def test_band_lp_is_one_inside_band_and_falls_off_linearly() -> None:
    p = torch.tensor([0.3, 0.5, 0.7, 0.15, 0.85, 0.0, 1.0])
    lp = band_lp(p, (0.3, 0.7))
    assert lp.tolist() == pytest.approx([1.0, 1.0, 1.0, 0.5, 0.5, 0.0, 0.0])


def test_classifier_flat_before_first_success(make_stats, make_outcomes) -> None:
    score = _classifier()
    score.record(make_outcomes([(i, False, 0.0) for i in range(CAP)]))
    score.observe_rollout(
        make_stats(ic_ids=[list(range(CAP))], start_features=_clusters())
    )
    lp, trusted = score.values(_ids())
    assert not trusted.any()
    assert torch.unique(lp).numel() == 1


def test_classifier_trusted_after_first_success(make_stats, make_outcomes) -> None:
    score = _classifier()
    score.record(make_outcomes([(0, True, 1.0)]))
    score.observe_rollout(
        make_stats(ic_ids=[list(range(CAP))], start_features=_clusters())
    )
    _, trusted = score.values(_ids())
    assert trusted.all()


def test_classifier_learns_separable_success(make_stats, make_outcomes) -> None:
    score = _classifier(steps_per_update=200)
    score.record(make_outcomes([(i, i >= 20, 0.0) for i in range(CAP)]))
    score.observe_rollout(
        make_stats(ic_ids=[list(range(CAP))], start_features=_clusters())
    )
    p = score.success_prob(_ids())
    assert _auc(p[20:], p[:20]) > 0.9


def test_classifier_ignores_task_start_and_partial_rows(
    make_stats, make_outcomes
) -> None:
    score = _classifier()
    score.record(make_outcomes([(0, True, 1.0), (-1, True, 1.0)], full=[False, True]))
    score.observe_rollout(
        make_stats(ic_ids=[list(range(CAP))], start_features=_clusters())
    )
    _, trusted = score.values(_ids())
    assert not trusted.any()


def test_classifier_rejects_wrong_feature_rows(make_stats) -> None:
    score = _classifier()
    with pytest.raises(ValueError, match="start_features"):
        score.observe_rollout(make_stats(ic_ids=[[0]], start_features=_clusters()[:5]))


def test_classifier_rejects_out_of_range_rollout_ids(make_stats) -> None:
    score = _classifier()
    with pytest.raises(ValueError, match="ic_ids"):
        score.observe_rollout(make_stats(ic_ids=[[CAP]], start_features=_clusters()))


def test_rnd_error_higher_for_unseen_region(make_stats) -> None:
    score = RND(capacity=CAP, device="cpu", steps_per_update=300)
    # Only ICs 0-19 (region A) are ever started from; region B stays unseen.
    score.observe_rollout(
        make_stats(ic_ids=[list(range(20))], start_features=_clusters())
    )
    lp, trusted = score.values(_ids())
    assert trusted.all()
    assert lp[20:].mean() > 3 * lp[:20].mean()


def test_rnd_untrusted_before_first_rollout() -> None:
    score = RND(capacity=CAP, device="cpu")
    lp, trusted = score.values(_ids())
    assert not trusted.any()
    assert lp.tolist() == [0.0] * CAP


def test_step_distance_embeds_consecutive_closer_than_far(make_stats) -> None:
    score = StepDistance(capacity=CAP, device="cpu", steps_per_update=300)
    t = torch.arange(200, dtype=torch.float32)
    path = torch.stack([0.05 * t, torch.sin(0.05 * t)], dim=1)
    score.observe_transitions(path[:-1], path[1:])
    z = score.embed(path)
    assert z.shape == (200, 16)
    near = (z[1:] - z[:-1]).norm(dim=1).mean()
    far = (z[100:] - z[:-100]).norm(dim=1).mean()
    assert far > 2 * near


def test_step_distance_rejects_mismatched_pairs() -> None:
    score = StepDistance(capacity=CAP, device="cpu")
    with pytest.raises(ValueError, match="same shape"):
        score.observe_transitions(torch.zeros(4, 2), torch.zeros(3, 2))


@pytest.mark.parametrize("name", ["success_classifier", "rnd", "step_distance"])
def test_side_models_are_registered(name: str) -> None:
    assert name in SCORES.names()


def test_side_models_report_wall_ms(make_stats, make_outcomes) -> None:
    clf = _classifier(steps_per_update=5)
    rnd = RND(capacity=CAP, device="cpu", steps_per_update=5)
    sd = StepDistance(capacity=CAP, device="cpu", steps_per_update=5)
    assert clf.extra_metrics() == {"wall_ms_per_update": 0.0}
    clf.record(make_outcomes([(0, True, 1.0)]))
    stats = make_stats(ic_ids=[[0]], start_features=_clusters())
    clf.observe_rollout(stats)
    rnd.observe_rollout(stats)
    sd.observe_transitions(_clusters()[:-1], _clusters()[1:])
    for model in (clf, rnd, sd):
        assert model.extra_metrics()["wall_ms_per_update"] > 0.0
