import time

import pytest
import torch
from torch import Tensor

from resetsweep.ppo.gae import compute_gae
from resetsweep.ppo.networks import ActorCritic
from resetsweep.ppo.normalizer import RunningNorm
from resetsweep.ppo.update import START_CONFIG, Batch, PPOConfig, ppo_update
from resetsweep.sim.env import VecEnv
from resetsweep.tasks.base import TASKS, Task
from resetsweep.types import Outcomes, StartStates, StepCounter

TASKS.load("resetsweep.tasks")

OBS_DIM = 3
ACT_DIM = 2


def _config(**overrides: object) -> PPOConfig:
    base: dict[str, object] = {
        "gamma": 0.99,
        "lam": 0.95,
        "epochs": 4,
        "minibatch_size": 32,
        "lr": 1e-3,
        "ent_coef": 0.0,
        "vf_coef": 1.0,
        "target_kl": None,
    }
    base.update(overrides)
    return PPOConfig(**base)  # pyright: ignore[reportArgumentType] # test-only kwargs bag


def _fixed_batch(model: ActorCritic, *, t: int, n: int, seed: int) -> Batch:
    gen = torch.Generator().manual_seed(seed)
    obs = torch.randn(t, n, OBS_DIM, generator=gen)
    action, logp, value = model.act(obs)
    # Advantage tied to the action sign so there is a consistent direction to improve.
    advantages = action[..., 0].sign() + 0.1 * torch.randn(t, n, generator=gen)
    returns = obs.sum(dim=-1)
    return Batch(
        obs=obs,
        actions=action,
        logp=logp,
        values=value,
        advantages=advantages,
        returns=returns,
    )


def _losses(model: ActorCritic, batch: Batch) -> tuple[float, float]:
    """(value MSE, clipped-free surrogate -mean(ratio * adv)) on the whole batch."""
    with torch.no_grad():
        logp, _, value = model.evaluate(batch.obs, batch.actions)
        ratio = (logp - batch.logp).exp()
        adv = (batch.advantages - batch.advantages.mean()) / batch.advantages.std()
        return (
            float((value - batch.returns).square().mean()),
            float(-(ratio * adv).mean()),
        )


def test_ppo_update_reduces_loss_on_fixed_batch() -> None:
    model = ActorCritic(OBS_DIM, ACT_DIM, hidden=(32, 32))
    batch = _fixed_batch(model, t=8, n=16, seed=1)
    cfg = _config()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=cfg.adam_eps)
    value_before, surrogate_before = _losses(model, batch)

    stats = ppo_update(model, opt, batch, cfg)
    value_after, surrogate_after = _losses(model, batch)

    assert value_after < 0.8 * value_before
    assert surrogate_after < surrogate_before - 0.01
    assert set(stats) == {
        "approx_kl",
        "clip_frac",
        "explained_var",
        "entropy",
        "value_loss",
        "policy_loss",
        "kl_stopped",
        "n_minibatches",
    }


@pytest.mark.parametrize(
    ("t", "n", "minibatch_size", "epochs", "expected_steps"),
    [
        (4, 8, 8, 1, 4),  # 32 // 8
        (4, 8, 5, 1, 6),  # 32 // 5, remainder dropped
        (4, 8, 32, 3, 3),  # one minibatch per epoch
        (8, 16, 32, 2, 8),  # 128 // 32 = 4, times 2 epochs
    ],
)
def test_minibatch_count_derived_from_batch(
    t: int, n: int, minibatch_size: int, epochs: int, expected_steps: int
) -> None:
    model = ActorCritic(OBS_DIM, ACT_DIM, hidden=(16,))
    batch = _fixed_batch(model, t=t, n=n, seed=2)
    cfg = _config(minibatch_size=minibatch_size, epochs=epochs)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=cfg.adam_eps)
    steps: list[int] = []
    opt.register_step_post_hook(lambda *_: steps.append(1))

    ppo_update(model, opt, batch, cfg)

    assert len(steps) == expected_steps


def test_ppo_update_rejects_minibatch_larger_than_batch() -> None:
    model = ActorCritic(OBS_DIM, ACT_DIM, hidden=(16,))
    batch = _fixed_batch(model, t=2, n=4, seed=3)
    cfg = _config(minibatch_size=9)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=cfg.adam_eps)

    with pytest.raises(ValueError, match="minibatch_size 9 exceeds the batch of 8"):
        ppo_update(model, opt, batch, cfg)


def test_target_kl_stops_update_early() -> None:
    model = ActorCritic(OBS_DIM, ACT_DIM, hidden=(32, 32))
    batch = _fixed_batch(model, t=8, n=16, seed=4)
    # A huge lr makes the first minibatch move the policy far past the KL target.
    cfg = _config(lr=0.5, epochs=4, minibatch_size=32, target_kl=1e-6)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=cfg.adam_eps)
    steps: list[int] = []
    opt.register_step_post_hook(lambda *_: steps.append(1))

    ppo_update(model, opt, batch, cfg)

    assert 1 <= len(steps) < cfg.epochs * (8 * 16 // 32)


def test_early_stop_reports_kl_stopped_and_minibatch_count() -> None:
    model = ActorCritic(OBS_DIM, ACT_DIM, hidden=(32, 32))
    batch = _fixed_batch(model, t=8, n=16, seed=4)
    # Minibatch 1 sees the rollout policy (KL ~0) and steps with a huge lr; minibatch 2's KL is
    # then far above 1.5 * 1e-6, so the update stops there without stepping on it.
    cfg = _config(lr=0.5, epochs=4, minibatch_size=32, target_kl=1e-6)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=cfg.adam_eps)

    stats = ppo_update(model, opt, batch, cfg)

    assert stats["kl_stopped"] == pytest.approx(1.0, abs=0)
    assert stats["n_minibatches"] == pytest.approx(1.0, abs=0)
    # Only the triggering minibatch has a large KL; without it the mean would be ~0.
    assert stats["approx_kl"] > 1e-3


def test_no_early_stop_reports_full_minibatch_count() -> None:
    model = ActorCritic(OBS_DIM, ACT_DIM, hidden=(32, 32))
    batch = _fixed_batch(model, t=8, n=16, seed=4)
    cfg = _config(epochs=3, minibatch_size=32, target_kl=None)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=cfg.adam_eps)

    stats = ppo_update(model, opt, batch, cfg)

    assert stats["kl_stopped"] == pytest.approx(0.0, abs=0)
    assert stats["n_minibatches"] == pytest.approx(
        12.0, abs=0
    )  # 3 epochs x (128 // 32)


def _minibatches_run(kl_stop_factor: float) -> float:
    torch.manual_seed(0)
    model = ActorCritic(OBS_DIM, ACT_DIM, hidden=(32, 32))
    batch = _fixed_batch(model, t=8, n=16, seed=4)
    cfg = _config(
        lr=0.5,
        epochs=2,
        minibatch_size=32,
        target_kl=1e-6,
        kl_stop_factor=kl_stop_factor,
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=cfg.adam_eps)
    return ppo_update(model, opt, batch, cfg)["n_minibatches"]


def test_kl_stop_threshold_scales_with_kl_stop_factor() -> None:
    # Same seed, model and batch; only the factor differs. A factor large enough to lift the
    # threshold above any reachable KL must let all 2 x (128 // 32) minibatches run.
    default_run = _minibatches_run(1.5)
    lifted_run = _minibatches_run(1e12)

    assert default_run == pytest.approx(1.0, abs=0)
    assert lifted_run == pytest.approx(8.0, abs=0)


def test_value_loss_is_unscaled_mse() -> None:
    # One epoch, one minibatch: the reported value loss is the MSE before the only step.
    model = ActorCritic(OBS_DIM, ACT_DIM, hidden=(16,))
    batch = _fixed_batch(model, t=4, n=8, seed=5)
    cfg = _config(epochs=1, minibatch_size=32)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=cfg.adam_eps)
    with torch.no_grad():
        mse = float((model.value(batch.obs) - batch.returns).square().mean())

    stats = ppo_update(model, opt, batch, cfg)

    assert stats["value_loss"] == pytest.approx(mse, rel=1e-5)


def test_act_log_prob_is_of_the_unclipped_sample() -> None:
    model = ActorCritic(OBS_DIM, ACT_DIM, hidden=(16,))
    obs = torch.randn(64, OBS_DIM)

    action, logp, _ = model.act(obs)
    logp_eval, _, _ = model.evaluate(obs, action)

    assert bool(
        (action.abs() > 1.0).any()
    )  # std starts at 1, so some samples exceed [-1, 1]
    torch.testing.assert_close(logp, logp_eval)


# --- learning test --------------------------------------------------------------------------


class _TaskStarts:
    """Minimal reset method: every episode starts from the task's own start distribution."""

    name = "task_start"

    def __init__(self, task: Task, gen: torch.Generator) -> None:
        self._task, self._gen = task, gen

    def reset(
        self, env_ids: Tensor, outcomes: Outcomes | None
    ) -> tuple[StartStates, Tensor]:
        return self._task.sample_start(env_ids.numel(), self._gen), env_ids.clone()

    def offer_visited(
        self,
        env_ids: Tensor,
        qpos: Tensor,
        qvel: Tensor,
        extra: Tensor,
        ep_step: Tensor,
    ) -> None:
        return None


NUM_ENVS = 256
ROLLOUT_STEPS = 32
TRAIN_ITERATIONS = 100
WALL_CLOCK_S_MAX = 60.0
BASELINE_ROLLOUTS = (
    6  # untrained policy; the first 2 are dropped (all envs start hanging)
)
# Threshold, measured 2026-11-24 on an Apple-silicon Mac CPU with this exact setup (256 envs x
# 32 steps, START_CONFIG with a 2048 minibatch, kl_stop_factor 1.5, unscaled value MSE). The
# metric is the mean per-step dense reward of a rollout on pendulum_dense (height01 - 0.01 u^2:
# 0 hanging, 1 upright); "gain" is the mean of the last 3 rollouts minus the untrained baseline.
# - Untrained steady state (rollouts 3-6): about 0.02 (spread about 0.01 rollout to rollout).
# - This test exactly (seed 0, after the warm-up step): baseline 0.0174, final 0.5079, gain
#   0.490 at 100 iterations, identical over two runs (CPU is deterministic here); about 40 s.
# - Without the warm-up step (which advances the env generator), seeds 0, 1, 2 at 100
#   iterations gained 0.582, 0.429, 0.277; at 70 iterations 0.443, 0.272, 0.393. So the margin
#   belongs to this seed, not to every seed.
# - No-learning control (lr 1e-9): gain 0.003.
# Requiring a gain of more than 0.3 leaves 0.19 of margin (1.6x) on this run and is about 100x
# the control's gain. Iterations, not wall clock, bound the training, so the
# verdict does not depend on machine speed; the 60 s budget is a separate assertion.
IMPROVEMENT_MIN = 0.3


@pytest.mark.slow
def test_ppo_learns_pendulum_dense_on_cpu() -> None:
    torch.manual_seed(0)
    task: Task = TASKS.construct("pendulum_dense")  # pyright: ignore[reportAssignmentType] # registry is untyped
    gen = torch.Generator().manual_seed(0)
    env = VecEnv(
        task,
        num_envs=NUM_ENVS,
        device="cpu",
        counter=StepCounter(budget=10**9),
        gen=gen,
    )
    env.bind(_TaskStarts(task, gen))
    # Warm the Warp kernels (compiled on first use) outside the timed section.
    env.reset_all()
    env.step(torch.zeros(NUM_ENVS, task.act_dim))
    model = ActorCritic(task.obs_dim, task.act_dim)
    norm = RunningNorm((task.obs_dim,))
    # START_CONFIG except the minibatch: 24,576 is larger than this CPU batch of 256 x 32 = 8192,
    # so the test holds 4 minibatches per epoch instead (2048 samples each).
    cfg = PPOConfig(**{**START_CONFIG.__dict__, "minibatch_size": 2048})
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=cfg.adam_eps)

    start = time.monotonic()
    obs = env.reset_all()
    untrained: list[float] = []
    for _ in range(BASELINE_ROLLOUTS):
        _, obs, mean_reward = _rollout(env, model, norm, obs, cfg)
        untrained.append(mean_reward)
    baseline = sum(untrained[2:]) / len(untrained[2:])
    curve: list[float] = []
    for _ in range(TRAIN_ITERATIONS):
        batch, obs, mean_reward = _rollout(env, model, norm, obs, cfg)
        curve.append(mean_reward)
        ppo_update(model, opt, batch, cfg)
    elapsed_s = time.monotonic() - start

    final = sum(curve[-3:]) / 3
    assert final - baseline > IMPROVEMENT_MIN, (baseline, final, curve)
    assert elapsed_s < WALL_CLOCK_S_MAX, (
        f"learning passed, but {TRAIN_ITERATIONS} iterations took {elapsed_s:.1f} s, "
        f"over the {WALL_CLOCK_S_MAX:.0f} s CPU budget"
    )


def _rollout(
    env: VecEnv, model: ActorCritic, norm: RunningNorm, obs: Tensor, cfg: PPOConfig
) -> tuple[Batch, Tensor, float]:
    n, t = env.num_envs, ROLLOUT_STEPS
    obs_buf = torch.zeros(t, n, env.task.obs_dim)
    act_buf = torch.zeros(t, n, env.task.act_dim)
    logp_buf, val_buf, rew_buf, boot_buf = (torch.zeros(t, n) for _ in range(4))
    term_buf = torch.zeros(t, n, dtype=torch.bool)
    trunc_buf = torch.zeros(t, n, dtype=torch.bool)
    raw_obs: list[Tensor] = []
    for step in range(t):
        raw_obs.append(obs)
        obs_n = norm.normalize(obs)
        action, logp, value = model.act(obs_n)
        out = env.step(action)
        with torch.no_grad():
            boot_buf[step] = model.value(norm.normalize(out.final_obs))
        obs_buf[step], act_buf[step] = obs_n, action
        logp_buf[step], val_buf[step], rew_buf[step] = logp, value, out.reward
        term_buf[step], trunc_buf[step] = out.terminated, out.truncated
        obs = out.obs
    norm.update(torch.cat(raw_obs))
    adv, returns = compute_gae(
        rew_buf, val_buf, term_buf, trunc_buf, boot_buf, gamma=cfg.gamma, lam=cfg.lam
    )
    batch = Batch(
        obs=obs_buf,
        actions=act_buf,
        logp=logp_buf,
        values=val_buf,
        advantages=adv,
        returns=returns,
    )
    return batch, obs, float(rew_buf.mean())
