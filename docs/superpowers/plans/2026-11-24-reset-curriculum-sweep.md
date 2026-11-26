# Reset-Curriculum Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement every shortlist task, reset method and score on GPU MuJoCo with a PyTorch PPO learner, plus a UCB scheduler that spends at most $100 on RunPod to rank (task, method, score) combinations, all launched by one local command and logged to W&B.

**Architecture:** `src/resetsweep/` has one folder per pluggable kind (`tasks/`, `methods/`, `scores/`), each with a `Protocol` in `base.py` and a decorator registry. `sim/` wraps MuJoCo Warp's batched `Data` behind torch views. `train.py` composes one run. `sweep/` composes many runs under a budget guard. `launch/` talks to RunPod REST v2.

**Tech Stack:** Python 3.12, uv 0.13.x, mujoco 4.0.1, mujoco-warp 4.0.1, warp-lang 1.19.0, torch 2.15.0, numpy, wandb 0.31.2, httpx 0.28.1, python-dotenv 1.2.4; dev: pytest 9.2.0, hypothesis, ruff, pyright. Versions read from PyPI on 2026-11-24.

**Spec:** `docs/superpowers/specs/2026-11-24-reset-curriculum-sweep-design.md`. Read it before any task: section numbers below (§x) refer to it. Shortlist citations (e.g. "shortlist §4.2") refer to `reset_curriculum_shortlist.md`.

## Global Constraints

- Python pinned to `3.12` (`.python-version` and `requires-python = ">=3.12,<3.13"`). All packages pinned exactly in `pyproject.toml`; `uv.lock` committed.
- Skills to load when the task touches them:
  - `pytest-expert`: every task; tests first.
  - `pytorch`: `ppo/`, `scores/` side models, `sim/`.
  - `massively-parallel-rl` and `robot-learning-expert`: `ppo/`, `train.py`, `sim/env.py`, `sweep/stats.py`.
  - `design-patterns-expert`: registries, protocols, composition roots.
  - `observability-expert`: W&B and logging.
  - `negative-space-programming`: budget guard and scheduler.
  - `bash-scripting-specialist`: `scripts/pod_bootstrap.sh`.
- Never write a library API from memory. Before using a `mujoco_warp`, `warp`, `torch`, `wandb`, `httpx` or RunPod call for the first time, check it against current docs (Context7 MCP, or the installed source under `.venv/lib/python3.12/site-packages/`), and note the source in the commit message body.
- Verified MuJoCo Warp facts (spike on macOS arm64 CPU, 2026-11-24):
  - Load models with `mjw.put_model(mjm)` and `mjw.make_data(mjm, nworld=N)`; step with `mjw.step(m, d)`; `mjw.reset_data(m, d, reset=None)`.
  - `wp.to_torch(d.qpos)` gives a zero-copy `(nworld, nq)` float32 tensor; writes through it are seen by `mjw.step`.
  - Device choice: `with wp.ScopedDevice(device):`. CPU works on macOS. Kernels compile on first use, about 0.05–0.3 s each, then come from the cache.
- Pod: Secure Cloud, `gpu.minCudaVersion = "13.2"`. The torch 2.15.0 Linux wheel depends on `cuda-toolkit==13.2.2` (PyPI metadata).
- Base image `runpod/pytorch:2.2.0-cu1290-torch291-ubuntu2404` (Docker Hub, updated 2026-11-24). Its torch is ignored: the repo's uv venv brings its own.
- `.env`, `.env.*`, `.claude/`, `.DS_Store`, `runs/`, `wandb/` stay gitignored (already in `.gitignore`). The repo will be **public**.
- Secrets: never logged, never in W&B config, never in printed request bodies. The redaction placeholder is the literal `"***"`.
- Budget constants (§8):
  - `CAP_USD_DEFAULT = 100.0`
  - `MARGIN_USD = 5.0`
  - `CALIBRATION_CAP_USD = 10.0`
  - `NETWORK_VOLUME_USD_PER_GB_MONTH = 0.07` (docs.runpod.io/storage/network-volumes, read 2026-11-24; the API exposes no storage price)
  - `NETWORK_VOLUME_GB = 20`
- Scheduler constants (§7, §12): `ETA = 3`, `BETA = 2.0`, rung 0 sized to `0.35` of the sweep budget, seeds per rung `(1, 3, 5)`.
- Method defaults:
  - `P_TASK_DEFAULT = 0.5`; the `visited` sweep uses `P_TASK_SWEEP = (0.0, 0.2, 0.5, 0.8, 1.0)`.
  - `own_success` frontier: advance at `0.8`, retreat at `0.15`; alternative bar `0.2`.
  - Success-classifier band `(0.3, 0.7)`.
- Commits: Conventional Commits, one per task step that says commit. Each message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Work happens on branch `feat/sweep`, cut from `docs/sweep-design`.
- Every test run uses `uv run pytest`. Warnings are errors (`filterwarnings = ["error"]`, with exact-message ignores only for third-party warnings, each with a comment saying why). Unknown markers are errors.
- Tests that need CUDA carry `@pytest.mark.gpu` and are skipped when `torch.cuda.is_available()` is false. The pod runs `uv run pytest -m gpu` before the sweep.

## Review Focus

1. **A worker process crashes mid-run** (CUDA OOM, NaN loss, MuJoCo Warp solver overflow). Expected: the result is recorded as `status="failed"` with the error text. The arm is retried once with a new seed, then marked failed and excluded from ranking. The sweep keeps going, and the spend still counts. Test goes in Task 23.
2. **The pod restarts or is preempted mid-sweep.** Expected: on start, `sweep-run` reloads scheduler state and results from `/workspace/sweep/`. Jobs that were running are resubmitted, finished ones aren't. The spend ledger includes the time before the restart. Test goes in Task 22 (state resume) and Task 23 (ledger resume).
3. **Live price lookup or create-pod fails** (network error, 400 no capacity, 422). Expected: the launcher retries 429/5xx a bounded number of times. On 400 it tries the next GPU in the fallback list. On a failed pod create after the volume was created, it prints the orphaned volume id and the delete command. It never creates a second volume silently. Test goes in Task 24.
4. **`.env` missing a key or holding a blank value.** Expected: the launcher exits non-zero naming the missing variable, before any HTTP call, and never prints any value. Test goes in Task 24.
5. **A method returns an invalid start state** (NaN, outside `start_box`, inside a wall). Expected: the env refuses it with `InvalidStartState` naming the method and the bad row, so a bug fails loudly instead of training on garbage. Methods that propose points (`ball`, `genetic`) must filter with `task.is_valid` before returning. Test goes in Task 3 (env check) and Task 18 (methods filter).

---

## File structure

```
pyproject.toml, uv.lock, .python-version, README.md
scripts/pod_bootstrap.sh                 in-pod: uv sync, gpu tests, sweep-run, self-terminate
src/resetsweep/
  __init__.py
  registry.py                            make_registry(kind) -> (register decorator, get, names)
  device.py                              pick_device() -> "cuda:0" | "cpu"
  types.py                               StartStates, Outcomes, RolloutStats, Box, StepCounter, BudgetExceeded, InvalidStartState
  sim/batch.py                           BatchSim: MuJoCo Warp model+data, torch views, state get/set, step
  sim/env.py                             VecEnv: clocks, truncation/termination, hold-success, resets via Method
  sim/rollouts.py                        Rollouts: random walks and hold screens, charged to StepCounter
  tasks/base.py                          Task protocol + registry
  tasks/pendulum.py tasks/cartpole.py tasks/acrobot.py tasks/maze.py tasks/maze_variants.py
  tasks/arm.py tasks/ramp.py tasks/claw.py
  ppo/networks.py ppo/gae.py ppo/normalizer.py ppo/update.py
  scores/base.py                         Score protocol + registry + ICTable (per-IC tensors)
  scores/success_band.py scores/slope.py scores/controls.py scores/critic.py
  scores/steps_to_reward.py scores/visit_count.py scores/side_models.py
  methods/base.py                        Method protocol + registry + MethodContext + p_task mixing
  methods/uniform.py methods/point.py methods/visited.py methods/visited_count.py
  methods/buffer_scored.py methods/own_success.py methods/phased.py methods/walk.py methods/proposal.py
  train.py                               RunSpec, RunResult, run()
  metrics.py                             MetricsSink protocol, WandbSink, JsonlSink, MultiSink, NullSink
  sweep/arms.py sweep/stats.py sweep/scheduler.py sweep/budget.py sweep/pool.py sweep/main.py sweep/report.py
  launch/env_file.py launch/runpod_client.py launch/launcher.py
tests/   mirrors src/ (tests/sim/, tests/tasks/, ...), plus tests/test_no_secrets.py
```

---

### Task 1: Project scaffold, registry, device, secret scan

**Files:**
- Create: `pyproject.toml`, `.python-version`, `src/resetsweep/__init__.py`, `src/resetsweep/registry.py`, `src/resetsweep/device.py`, `tests/test_registry.py`, `tests/test_no_secrets.py`, `tests/conftest.py`

**Interfaces:**
- Produces:
  - `make_registry(kind: str) -> Registry`. `Registry` has `.register(name: str) -> Callable[[type], type]`, `.get(name: str) -> type` (raises `KeyError` listing the valid names), `.names() -> list[str]` (sorted), and `.construct(name: str, /, *args, **settings)`. `construct` raises `ValueError` listing the valid keyword names when a setting doesn't bind to `__init__`, checked with `inspect.signature(...).bind`.
  - Auto-discovery: `Registry.load(package: str)` imports every non-underscore module in the package once.
  - `pick_device() -> str`: `"cuda:0"` if `warp.is_cuda_available()`, else `"cpu"`.

- [ ] **Step 1:** `git checkout -b feat/sweep`. Then `uv init --package --name resetsweep --python 3.12` (keep the existing `.gitignore`), and `uv add` the exact pins from Tech Stack (`uv add --dev` for dev tools). In `[tool.pytest.ini_options]` set:
  - `testpaths = ["tests"]`
  - `addopts = ["-ra", "--strict-markers"]`
  - `xfail_strict = true`
  - `filterwarnings = ["error"]`
  - `markers = ["gpu: needs CUDA", "slow: > 10 s"]`

  Add `[project.scripts]` with `sweep-launch = "resetsweep.launch.launcher:main"` and `sweep-run = "resetsweep.sweep.main:main"`.
- [ ] **Step 2: Write failing tests** in `tests/test_registry.py`:
  - `test_register_and_get_returns_class`
  - `test_duplicate_name_different_class_raises_value_error`
  - `test_get_unknown_lists_valid_names` (`pytest.raises(KeyError, match="valid: a, b")`)
  - `test_construct_rejects_unknown_setting_listing_valid` (class with `__init__(self, x: int, *, y=1)`; `construct("c", 1, z=2)` raises `ValueError` matching `"y"`)
  - `test_construct_passes_settings`

  In `tests/test_no_secrets.py`, `test_no_tracked_file_contains_secret_like_strings`:
  - runs `git ls-files` (as an argument list, no shell);
  - fails if any tracked file matches `r"(RUNPOD_API_KEY|WANDB_API_KEY)\s*=\s*\S{8,}"` or `r"rpa_[A-Za-z0-9]{20,}"`;
  - fails if `.env` is tracked.
- [ ] **Step 3:** Run `uv run pytest tests/test_registry.py -v`. Expected: FAIL with `ModuleNotFoundError: resetsweep.registry`.
- [ ] **Step 4:** Implement `registry.py` and `device.py`.
- [ ] **Step 5:** Run `uv run pytest -v`. Expected: all pass. Run `uv run ruff check . && uv run pyright src`. Expected: clean.
- [ ] **Step 6: Commit** `chore: scaffold uv project, registry and secret scan`.

### Task 2: BatchSim over MuJoCo Warp

**Files:**
- Create: `src/resetsweep/types.py`, `src/resetsweep/sim/__init__.py`, `src/resetsweep/sim/batch.py`, `tests/sim/test_batch.py`

**Interfaces:**
- Produces in `types.py`:
  - `@dataclass(frozen=True) StartStates(qpos: Tensor, qvel: Tensor, extra: Tensor)`, each with shape `(n, ·)`. `extra` holds task-specific state (e.g. a coin-flip draw) and may be `(n, 0)`. Also `.n` and `.index(idx) -> StartStates`, plus `StartStates.cat(list) -> StartStates`.
  - `@dataclass(frozen=True) Box(low: Tensor, high: Tensor)` over `concat(qpos, qvel)`.
  - `class StepCounter`:
    - `__init__(self, *, budget: int)`;
    - `add_train(n: int)`;
    - `add_extra(n: int, kind: Literal["walk", "screen", "scout"])`;
    - properties `total`, `train`, `extra: dict[str, int]`;
    - raises `BudgetExceeded` when `total > budget`.
  - Exceptions `BudgetExceeded(RuntimeError)` and `InvalidStartState(ValueError)`.
- Produces in `batch.py`:
  - `class BatchSim`:
    - `__init__(self, mjcf: str, *, nworld: int, device: str, frame_skip: int)`;
    - views `qpos`, `qvel`, `ctrl`, `xpos`, `site_xpos` (torch, zero-copy);
    - `get_state(idx: Tensor) -> tuple[Tensor, Tensor]` (cloned);
    - `set_state(idx: Tensor, qpos: Tensor, qvel: Tensor) -> None`, which also zeroes `qacc_warmstart` (field name checked against `mujoco_warp` source) and runs `mjw.forward` so derived quantities are fresh;
    - `step(ctrl: Tensor) -> None`, which copies into `ctrl` then runs `mjw.step` `frame_skip` times.

  Steps are captured in a CUDA graph (`wp.ScopedCapture`) only when the device is CUDA, per the MuJoCo Warp tutorial.

- [ ] **Step 1: Write failing tests** (a pendulum MJCF fixture in `tests/sim/conftest.py`, `nworld=4`, `device="cpu"`):
  - `test_set_then_get_state_round_trips`: set rows `[1, 3]`, then `get_state` equals the inputs exactly, and rows `[0, 2]` are unchanged.
  - `test_restored_state_gives_same_trajectory_in_two_worlds`: same state into worlds 1 and 2, same ctrl for 50 steps, `torch.allclose(qpos[1], qpos[2], atol=1e-5)`. (Shortlist F check. On GPU the same test runs with `atol=1e-3`, marked `gpu`, and logs the max deviation.)
  - `test_step_advances_time_by_frame_skip_times_timestep`.
  - `test_step_counter_raises_budget_exceeded_past_budget`, plus `test_step_counter_splits_train_and_extra`.
- [ ] **Step 2:** Run `uv run pytest tests/sim -v`. Expected: FAIL (import error for `resetsweep.sim.batch`).
- [ ] **Step 3:** Implement `types.py` and `batch.py`.
- [ ] **Step 4:** Run `uv run pytest tests/sim -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: batched MuJoCo Warp sim with torch state views`.

### Task 3: Task protocol and VecEnv episode logic

**Files:**
- Create: `src/resetsweep/tasks/__init__.py`, `src/resetsweep/tasks/base.py`, `src/resetsweep/sim/env.py`, `tests/sim/test_env.py`

**Interfaces:**
- Consumes: `BatchSim`, `StartStates`, `Box`, `StepCounter`, `InvalidStartState`.
- Produces in `tasks/base.py`: `TASKS = make_registry("task")` and a `Protocol Task` with:
  - attributes `name`, `mjcf`, `obs_dim`, `act_dim`, `max_steps`, `frame_skip`, `hold_steps`, `start_box: Box`, `dense_sibling: str`, `extra_dim: int`;
  - `sample_start(n: int, gen: torch.Generator) -> StartStates`;
  - `canonical_start() -> StartStates`;
  - `observe(sim: BatchSim, extra: Tensor) -> Tensor`;
  - `reward(sim, extra, action) -> Tensor`;
  - `terminated(sim, extra) -> Tensor[bool]`;
  - `success_now(sim, extra) -> Tensor[bool]`;
  - `state_features(qpos, qvel, extra) -> Tensor`;
  - `is_valid(states: StartStates) -> Tensor[bool]`;
  - `eval_states(n: int, seed: int) -> StartStates`.
- Produces in `sim/env.py`: `class VecEnv` with:
  - `__init__(self, task, *, num_envs, device, counter: StepCounter, gen)`;
  - `bind(method)`;
  - `reset_all() -> Tensor obs`, which assigns starts via `method.reset`, then sets each env's clock to `randint(0, max_steps)`. Those first episodes are flagged `full=False` (shortlist F, staggered clocks).
  - `step(action) -> StepOut`.
- `StepOut` fields: `obs`, `reward`, `terminated`, `truncated`, `final_obs` (the pre-reset obs for truncation bootstrap), `ic_ids`, and `outcomes: Outcomes | None`.
- `Outcomes` (in `types.py`): `env_ids, ic_ids, ret, success, ep_len, reached_reward, full, start_step` (tensors).
- On every `step`, VecEnv calls `method.offer_visited(env_ids, qpos, qvel, extra, ep_step)` every `visit_every` steps (default 8). It checks each start state with `task.is_valid`, and NaNs, and raises `InvalidStartState(method_name, row)`.

- [ ] **Step 1: Write failing tests**, using a stub task in `tests/sim/conftest.py` (pendulum MJCF, success when angle near π) and a stub method that returns `canonical_start`:
  - `test_episode_truncates_at_max_steps_not_terminates`
  - `test_final_obs_is_pre_reset_observation`
  - `test_success_requires_hold_steps_consecutive`
  - `test_first_episodes_have_random_clocks_and_full_false`
  - `test_outcomes_report_env_ids_ic_ids_returns`
  - `test_invalid_start_state_raises_naming_method` (stub method returns NaN qpos)
  - `test_counter_charges_num_envs_per_step`
- [ ] **Step 2:** Run `uv run pytest tests/sim/test_env.py -v`. Expected: FAIL (import error).
- [ ] **Step 3:** Implement `tasks/base.py`, the `Outcomes`/`StepOut` additions to `types.py`, and `sim/env.py`.
- [ ] **Step 4:** Run `uv run pytest tests/sim -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: task protocol and vectorized episode env`.

### Task 4: Pendulum tasks (sparse and dense): the pattern every task follows

**Files:**
- Create: `src/resetsweep/tasks/pendulum.py`, `tests/tasks/test_pendulum.py`, `tests/tasks/test_task_contract.py`

**Interfaces:**
- Produces: registered tasks `pendulum_sparse` and `pendulum_dense`, which share an MJCF.
  - Hinge pendulum, torque limit below m·g·l, so a swing-up is needed. `frame_skip` gives a 0.05 s control step; episodes are 5 s.
  - `sample_start`: hanging ± small noise. `start_box`: angle in [−π, π], ω in [−8, 8].
  - Sparse reward: `1.0` when upright within tolerance, minus `0.01·|u|²`.
  - Dense reward: `cos θ` mapped to [0, 1], minus the same action cost.
  - `dense_sibling = "pendulum_dense"` for both.
- `tests/tasks/test_task_contract.py` is parametrised over `TASKS.names()` and checks every task:
  - observation shape is `(n, obs_dim)`;
  - `sample_start` states pass `is_valid`;
  - `eval_states(n, seed)` is identical across two calls and across two `Generator` states;
  - `canonical_start().n == 1`;
  - reward and `success_now` have shape `(n,)`;
  - `start_box` has `nq + nv` dimensions.

  Every later task gets this for free.

- [ ] **Step 1: Write failing tests**:
  - the contract test;
  - `test_sparse_reward_zero_when_hanging_and_one_when_upright`;
  - `test_torque_limit_cannot_lift_directly` (full torque from hanging for 1 s keeps the angle below π/2);
  - `test_dense_reward_increases_with_height`.
- [ ] **Step 2:** Run `uv run pytest tests/tasks -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `pendulum.py`.
- [ ] **Step 4:** Run `uv run pytest tests/tasks -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: sparse and dense pendulum tasks with task contract tests`.

### Task 5: Cartpole and acrobot swing-up (sparse)

**Files:**
- Create: `src/resetsweep/tasks/cartpole.py`, `src/resetsweep/tasks/acrobot.py`, `tests/tasks/test_cartpole.py`, `tests/tasks/test_acrobot.py`

**Interfaces:**
- Produces: `cartpole_swingup_sparse` and `cartpole_swingup_dense` (dense is the sibling), and `acrobot_swingup_sparse` and `acrobot_swingup_dense` (dense is the sibling).
  - Sparse reward: success indicator minus small action cost (§5.3 T2).
  - Success: pole upright and cart near centre (cartpole); tip above a height (acrobot), held `hold_steps`.
  - Write the geometry fresh in MJCF; the builder picks dimensions.
- [ ] **Step 1: Write failing tests**:
  - `test_cartpole_sparse_zero_from_hanging_start`
  - `test_cartpole_success_when_set_upright_and_held`
  - `test_acrobot_sparse_zero_from_hanging_start`
  - `test_acrobot_success_when_tip_above_target_and_held`
  - contract tests pick these tasks up automatically
- [ ] **Step 2:** Run `uv run pytest tests/tasks -v`. Expected: the new tests FAIL.
- [ ] **Step 3:** Implement both files.
- [ ] **Step 4:** Run `uv run pytest tests/tasks -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: sparse cartpole and acrobot swing-up tasks`.

### Task 6: Maze family

**Files:**
- Create: `src/resetsweep/tasks/maze.py`, `tests/tasks/test_maze.py`

**Interfaces:**
- Produces:
  - `maze` (settings `lanes: int = 3`, `passage_width: float = 0.4`): a point mass on two slide joints, actuated by force, with damping. Walls are box geoms forming a serpentine corridor, the start is at one end, and the goal region is at the other.
  - `maze_narrow`: `maze` with `passage_width = 0.15` and an N-shaped layout, after 2610.03395.
  - `maze_dense`: `maze` plus a reward of minus normalised shortest-path distance to the goal, from a precomputed grid BFS. This is the sibling for every maze-family task.
  - Reward for `maze` and `maze_narrow`: `1.0` inside the goal, else `0`, minus action cost.
  - `is_valid`: rejects points inside walls (using the BFS occupancy grid).
  - `state_features`: (x, y, vx, vy).
  - Also exports `path_distance(task, xy_a, xy_b) -> Tensor` (BFS-grid geodesic) for the across-wall diagnostic in Task 16.
- [ ] **Step 1: Write failing tests**:
  - `test_point_cannot_cross_wall_with_full_force` (start next to a wall, push into it for 2 s, x stays on its side);
  - `test_goal_reward_only_inside_goal`;
  - `test_lanes_setting_changes_path_length` (the BFS path length for `lanes=5` is greater than for `lanes=3`);
  - `test_is_valid_rejects_point_inside_wall`;
  - `test_path_distance_exceeds_euclidean_across_wall`.
- [ ] **Step 2:** Run `uv run pytest tests/tasks/test_maze.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `maze.py`.
- [ ] **Step 4:** Run `uv run pytest tests/tasks -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: maze, narrow maze and dense maze sibling`.

### Task 7: Maze variants: trap, twin, infeasible mass, coin flip

**Files:**
- Create: `src/resetsweep/tasks/maze_variants.py`, `tests/tasks/test_maze_variants.py`

**Interfaces:**
- Consumes: the maze builder from Task 6 (refactor it into `build_maze(...) -> (mjcf, grid)` if needed).
- Produces:
  - `maze_trap` (setting `trap_scale: float = 1.0`): a pit region (sunken floor with a lip) the point can't climb out of, because actuator force is below what the lip needs.
  - `maze_trap_twin`: identical geometry, with enough actuator force inside the pit to escape. The pit's force is set per world through a site-local control gain in `extra`, or by using two actuators; the builder decides.
  - `maze_infeasible`: `sample_start` draws easy starts (near the goal, about 90%), hard-but-feasible starts (about 0.01%) and infeasible starts (about 9.99%, sealed pockets with no path) (§5.3 T4, ToyLevels). `eval_states` still uses only the feasible start end.
  - `maze_coinflip`: a block of cells. When an episode starts inside it, `extra[:,0]` holds a Bernoulli(0.5) draw made at reset, and `success_now` is that draw, whatever the policy does.
  - Each exports `diagnostic_region(qpos) -> Tensor[bool]` (inside the trap / infeasible pocket / coin-flip block) for the "wasted resets" metric.
- [ ] **Step 1: Write failing tests**:
  - `test_trap_cannot_be_escaped_at_full_force`;
  - `test_twin_trap_can_be_escaped`;
  - `test_infeasible_mass_fractions_within_tolerance` (100k draws: easy 0.90 ± 0.01, infeasible 0.0999 ± 0.005);
  - `test_infeasible_pocket_has_no_bfs_path_to_goal`;
  - `test_coinflip_success_independent_of_actions` (two action sequences, same `extra`, same success);
  - `test_coinflip_rate_near_half`.
- [ ] **Step 2:** Run `uv run pytest tests/tasks/test_maze_variants.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `uv run pytest tests/tasks -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: trap, twin, infeasible and coin-flip maze variants`.

### Task 8: Arm push (sparse, dense)

**Files:**
- Create: `src/resetsweep/tasks/arm.py`, `tests/tasks/test_arm.py`

**Interfaces:**
- Produces: `arm_sparse` and `arm_dense` (the sibling).
  - A 3-link planar arm (hinge joints, motors) pushes a puck (two slide joints plus a hinge) to a goal disc on a table plane (§12: push, not pick-and-place).
  - Sparse reward: `1.0` when the puck centre is inside the goal, minus action cost.
  - Dense reward: minus (hand-to-puck distance + puck-to-goal distance), plus the sparse term.
  - `is_valid`: rejects starts where any geom penetrates another; checked with `mjw.forward` on a 1-world sim, using `ncon` with `dist < 0`.
- [ ] **Step 1: Write failing tests**:
  - `test_puck_moves_when_pushed_by_link`;
  - `test_sparse_reward_only_when_puck_in_goal`;
  - `test_is_valid_rejects_interpenetrating_start`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `uv run pytest tests/tasks -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: sparse and dense planar arm push task`.

### Task 9: Deceptive ramp

**Files:**
- Create: `src/resetsweep/tasks/ramp.py`, `tests/tasks/test_ramp.py`

**Interfaces:**
- Produces: `ramp_deceptive` and `ramp_dense` (the sibling; reward is the cart's height).
  - An underpowered cart on a slide joint along a valley built from two inclined box geoms. The goal is atop the far slope; the near slope is behind the start.
  - Sparse goal reward (§5.3 T6, after Kinetix Car-Ramp).
- [ ] **Step 1: Write failing tests**:
  - `test_full_throttle_forward_from_start_never_reaches_goal` (10 s);
  - `test_back_then_forward_policy_reaches_goal` (a scripted bang-bang policy succeeds);
  - `test_sparse_reward_zero_until_goal`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `uv run pytest tests/tasks -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: deceptive ramp task`.

### Task 10: Claw (lowest priority)

**Files:**
- Create: `src/resetsweep/tasks/claw.py`, `tests/tasks/test_claw.py`

**Interfaces:**
- Produces: `claw` (dense, T7). A two-finger planar gripper on a 2-DoF base turns a disc on a hinge to a goal angle, with contact friction. Success: the angle is within tolerance and held.
- [ ] **Step 1: Write failing tests**:
  - `test_disc_turns_only_with_finger_contact`;
  - `test_success_at_goal_angle_held`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `uv run pytest tests/tasks -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: claw disc-turning task`.

### Task 11: PPO

**Files:**
- Create: `src/resetsweep/ppo/__init__.py`, `ppo/networks.py`, `ppo/gae.py`, `ppo/normalizer.py`, `ppo/update.py`, `tests/ppo/test_gae.py`, `tests/ppo/test_normalizer.py`, `tests/ppo/test_update.py`

**Interfaces:**
- Produces:
  - `ActorCritic(obs_dim: int, act_dim: int, *, hidden: tuple[int, ...] = (256, 256))`: Gaussian policy with state-independent log-std; methods `act(obs) -> (action, logp, value)`, `evaluate(obs, action) -> (logp, entropy, value)`, `value(obs) -> Tensor`.
  - `compute_gae(rewards, values, terminated, truncated, bootstrap_values, *, gamma, lam) -> (adv, returns)`, all `(T, N)`. At a truncated step it bootstraps from `bootstrap_values`, which is the critic on `final_obs`. At a terminated step it doesn't bootstrap.
  - `RunningNorm(shape)`: `update(x)` and `normalize(x)`, Welford/Chan merge.
  - `PPOConfig` (frozen dataclass): `gamma, lam, clip=0.2, epochs, minibatch_size, lr, ent_coef, vf_coef, max_grad_norm=0.5, adam_eps=1e-5, target_kl`.
  - `ppo_update(model, opt, batch, cfg) -> dict[str, float]`, returning keys `approx_kl, clip_frac, explained_var, entropy, value_loss, policy_loss`. The minibatch count is derived as `N*T // minibatch_size` (massively-parallel-rl non-negotiable 2).
- [ ] **Step 1: Write failing tests**:
  - `test_gae_matches_hand_computed_three_step_case` (write the 3-step numbers in the test, worked by hand in a comment);
  - `test_gae_bootstraps_on_truncation_not_termination`;
  - `test_running_norm_matches_numpy_mean_var_over_batches` (a Hypothesis property over random batch splits);
  - `test_ppo_update_reduces_loss_on_fixed_batch`;
  - `test_minibatch_count_derived_from_batch`;
  - `test_ppo_learns_pendulum_dense_on_cpu` (marked `slow`, 256 envs, return improves by more than a stated threshold within 60 s).

  Cross-check the GAE numbers with the robot-learning-expert oracle (`scripts/rl_math.py`) before freezing them in the test.
- [ ] **Step 2:** Run `uv run pytest tests/ppo -v -m "not slow"`. Expected: FAIL.
- [ ] **Step 3:** Implement. Starting values: adapt them from robot-learning-expert `references/training/on-policy-at-scale.md` and massively-parallel-rl `references/large-batch-on-policy.md`, and cite the source line in the `PPOConfig` docstring.
- [ ] **Step 4:** Run `uv run pytest tests/ppo -v`. Expected: PASS, including `slow`.
- [ ] **Step 5: Commit** `feat: PPO with truncation-aware GAE and obs normalization`.

### Task 12: Score protocol, IC table, S1, S3 and control scores

**Files:**
- Create: `src/resetsweep/scores/__init__.py`, `scores/base.py`, `scores/success_band.py`, `scores/slope.py`, `scores/controls.py`, `tests/scores/test_success_band.py`, `tests/scores/test_slope.py`, `tests/scores/test_controls.py`

**Interfaces:**
- Consumes: `Outcomes`; `RolloutStats` (added to `types.py` here) with fields:
  - `ic_ids (T, N)`, `adv`, `returns`, `values`;
  - `start_obs: Tensor` (obs at each IC's start);
  - `critic: Callable[[Tensor], Tensor]`.
- Produces in `scores/base.py`:
  - `SCORES = make_registry("score")`.
  - `ICTable(capacity: int, device)`: per-IC circular outcome history and counters, with `record(outcomes)`, `forget(ic_ids)`, `count`, `success_rate`, `ever_succeeded`.
  - `Protocol Score`:
    - `__init__(self, *, capacity: int, device: str, **settings)`;
    - `record(outcomes) -> None`;
    - `observe_rollout(stats: RolloutStats) -> None` (default no-op);
    - `values(ic_ids: Tensor) -> tuple[Tensor, Tensor]` (`lp`, `trusted`);
    - `forget(ic_ids) -> None`.
  - Untrusted ICs get the median of the trusted ICs.
- Registered scores:
  - `success_band` (settings `history=100, target=0.5, form: Literal["sgs", "pq"] = "sgs", min_count=1`). `pq` is p(1−p). `sgs` is the Beta-shaped weight peaking at `target`, per SGS 2610.12465; check the formula against that paper's text, and if the PDF isn't in the repo, mark the formula as taken from the shortlist's description.
  - `slope` (settings `form: Literal["signed", "abs", "fast_minus_slow"] = "signed"`, `window=20`, `fast=0.1`, `slow=0.01`).
  - `random` (settings `seed`): an iid uniform value per IC, fixed until `forget`.
  - `inverted` (setting `inner: str`, plus `inner_kw`): `max(lp) − lp` of the inner score, built through the registry.
- [ ] **Step 1: Write failing tests**:
  - `test_pq_score_peaks_at_half_success` (ICs with rates 0, 0.5, 1 give 0, 0.25, 0);
  - `test_sgs_weight_peaks_at_target`;
  - `test_untrusted_ics_get_median`;
  - `test_forget_resets_history`;
  - `test_slope_positive_for_rising_returns_and_abs_counts_falling`;
  - `test_fast_minus_slow_positive_after_jump`;
  - `test_random_score_fixed_until_forget`;
  - `test_inverted_reverses_ranking_of_inner`.
- [ ] **Step 2:** Run `uv run pytest tests/scores -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `uv run pytest tests/scores -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: score protocol with success-band, slope and control scores`.

### Task 13: S2 visit count, S4 critic-loss scores, S5 V(s0), S6 steps-to-reward

**Files:**
- Create: `scores/visit_count.py`, `scores/critic.py`, `scores/steps_to_reward.py`, `tests/scores/test_visit_count.py`, `tests/scores/test_critic.py`, `tests/scores/test_steps_to_reward.py`

**Interfaces:**
- Consumes: `ICTable`, `RolloutStats`.
- Produces:
  - `visit_count` (settings `post_reward: Literal["count", "count_x_learnability"] = "count"`, `eps=0.05`): lp = `1/sqrt(count+1)`. Counts come from the method through `set_counts(ic_ids, counts)`. After `notify_first_reward()`, the `count_x_learnability` form multiplies by `success_band(pq) + eps` (shortlist §4.2).
  - `critic_loss` (setting `form: Literal["pos_value_loss", "max_mc", "abs_gae"]`): per IC, mean of `relu(adv)`, `max(returns) − mean(values)` (MaxMC per 2408.15099), or `|adv|`, computed with `scatter_reduce` over `ic_ids`.
  - `critic_value` (settings `vmax: float`, `temperature: float = 1.0`): lp = `V(s0)·(vmax − V(s0))` from `stats.critic(stats.start_obs)` (2405.02481 uniform-target ablation).
  - `steps_to_reward`: for ICs that carry `steps_to_reward` metadata (set by `own_success` through `set_steps(ic_ids, steps)`), lp ranks smaller steps first (backward order). Other ICs are untrusted.
- [ ] **Step 1: Write failing tests**:
  - `test_visit_count_score_decreases_with_count`;
  - `test_count_x_learnability_zero_before_reward_is_replaced_by_count` (before first reward the score equals `1/sqrt(count+1)`);
  - `test_pos_value_loss_matches_manual_scatter_mean`;
  - `test_max_mc_matches_manual`;
  - `test_abs_gae_matches_manual`;
  - `test_critic_value_peaks_at_half_vmax`;
  - `test_steps_to_reward_prefers_fewer_steps`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `uv run pytest tests/scores -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: visit-count, critic-loss, critic-value and steps-to-reward scores`.

### Task 14: S7 side-model scores

**Files:**
- Create: `scores/side_models.py`, `tests/scores/test_side_models.py`

**Interfaces:**
- Consumes: `state_features` (passed in `RolloutStats.start_features: Tensor (n_ics, f)`; add that field) and `Outcomes`.
- Produces:
  - `success_classifier` (settings `band=(0.3, 0.7)`, `hidden=64`, `lr=1e-3`, `steps_per_update=50`): an MLP on start features predicting success. lp = 1 inside the band, else a distance-based falloff. Flat (all untrusted) until the first success.
  - `rnd` (settings `hidden=64`, `out=32`): random-network distillation; lp = prediction error on start features.
  - `step_distance` (settings `emb=16`, `margin=1.0`): an encoder trained so consecutive visited states sit close together, per 2407.08464. Exposes `embed(features) -> Tensor` for `visited_count` neighbour counting. Used only for near / not-near decisions (shortlist §4.7).
  - Every side model records `wall_ms_per_update` in `extra_metrics() -> dict[str, float]`.
- [ ] **Step 1: Write failing tests**:
  - `test_classifier_flat_before_first_success`;
  - `test_classifier_learns_separable_success` (two clusters, AUC > 0.9 after 200 steps);
  - `test_rnd_error_higher_for_unseen_region`;
  - `test_step_distance_embeds_consecutive_closer_than_far`;
  - `test_side_models_report_wall_ms`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `uv run pytest tests/scores -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: classifier, RND and step-distance side-model scores`.

### Task 15: Method protocol, uniform, point, visited

**Files:**
- Create: `src/resetsweep/methods/__init__.py`, `methods/base.py`, `methods/uniform.py`, `methods/point.py`, `methods/visited.py`, `src/resetsweep/sim/rollouts.py`, `tests/methods/test_base.py`, `tests/methods/test_visited.py`

**Interfaces:**
- Produces in `methods/base.py`:
  - `METHODS = make_registry("method")`.
  - `MethodContext` (frozen dataclass): `task, num_envs, device, gen, score: Score | None, rollouts: Rollouts, counter: StepCounter, p_task: float`.
  - `Protocol Method`:
    - `__init__(self, ctx: MethodContext, **settings)`;
    - `reset(env_ids: Tensor, outcomes: Outcomes | None) -> tuple[StartStates, Tensor ic_ids]`;
    - `offer_visited(env_ids, qpos, qvel, extra, ep_step) -> None`;
    - `update(it: int, stats: RolloutStats) -> dict[str, float]`;
    - `diagnostics() -> dict[str, float]`, which must include `sampling_entropy`, `buffer_size`, `never_succeeded_share`.
  - Helper `mix_task_start(ctx, env_ids, draw_fn) -> (StartStates, ic_ids)`: sends a `p_task` share of `env_ids` to `task.sample_start`, with reserved `ic_id = -1`.
  - Class attribute `accepts_scores: frozenset[str]`, used by Task 20.
- Produces in `sim/rollouts.py`: `Rollouts(task, *, device, counter, gen, nworld_max=4096)` with:
  - `random_walk(starts: StartStates, steps: int, action_hold: int) -> StartStates`, which charges `kind="walk"`;
  - `hold_screen(starts, seconds: float) -> Tensor[bool]`, which keeps a start only if the zero-action rollout doesn't terminate and speed stays bounded, and charges `kind="screen"`.

  Both use a separate `BatchSim`, so training envs are untouched.
- Methods:
  - `uniform` (`accepts_scores = {"none"}`): `sample_start` on every reset.
  - `point` (`source: Literal["canonical", "random"] = "canonical"`): one fixed IC.
  - `visited` (`capacity=65536`, `p_task=P_TASK_DEFAULT`): a ring buffer of visited states taken from all time offsets, drawn uniformly. Until the buffer holds `min_fill=num_envs` states, it falls back to the task start.
- [ ] **Step 1: Write failing tests** (stub task on CPU):
  - `test_mix_task_start_share_close_to_p_task` (10k resets, within ±0.02);
  - `test_p_task_zero_and_one_extremes`;
  - `test_point_always_same_state`;
  - `test_visited_ring_buffer_overwrites_oldest`;
  - `test_visited_draw_uniform_over_buffer` (chi-square over 10 cells, p > 0.01, seeded);
  - `test_visited_falls_back_until_min_fill`;
  - `test_random_walk_charges_counter_extra_walk`;
  - `test_hold_screen_rejects_falling_state`.
- [ ] **Step 2:** Run `uv run pytest tests/methods -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `uv run pytest tests/methods -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: method protocol, uniform, point and visited-buffer methods`.

### Task 16: visited_count (M3) and buffer_scored (M5)

**Files:**
- Create: `methods/visited_count.py`, `methods/buffer_scored.py`, `tests/methods/test_visited_count.py`, `tests/methods/test_buffer_scored.py`

**Interfaces:**
- Consumes: the `visited` buffer (subclass it or compose it, the builder's choice), `visit_count`, `step_distance`, and maze `path_distance` for diagnostics.
- Produces:
  - `visited_count`: settings `count: Literal["neighbour", "lineage_depth", "embedding"] = "neighbour"`, `radius=0.5` (on standardised features), `p_task`. `accepts_scores = {"visit_count"}`.
    - Neighbour counts are batched on device: chunked `torch.cdist` against the buffer.
    - Lineage depth = the number of restarts along the chain back to the task start, stored per buffer entry.
    - Draw is ∝ the score's lp.
    - If the task exposes `path_distance`, it logs `across_wall_neighbour_share`: the share of neighbours whose path distance exceeds 2×radius. That is the measurement shortlist §7 step 2 asks for.
  - `buffer_scored`: settings `staleness_coef=0.3`, `discount=0.99`, `min_count=3`, `rehearsal=0.1`, `lanes=1`, `p_task`. `accepts_scores = {"success_band", "slope", "critic_loss", "critic_value", "success_classifier", "rnd", "random", "inverted"}`.
    - Draws ∝ `(1 − staleness_coef)·rank_prob(lp) + staleness_coef·staleness_prob`, per PLR 2010.03934.
    - A `rehearsal` share goes to ICs with `ever_succeeded` and recent success below 0.5.
    - `lanes` envs share each drawn IC.
- [ ] **Step 1: Write failing tests**:
  - `test_neighbour_count_matches_bruteforce_small`;
  - `test_lineage_depth_increments_along_chain`;
  - `test_count_weighted_draw_prefers_rare_states`;
  - `test_across_wall_share_positive_in_maze_fixture`;
  - `test_staleness_raises_probability_of_unvisited_ic`;
  - `test_min_count_marks_untrusted`;
  - `test_rehearsal_share_goes_to_forgotten_ics`;
  - `test_lanes_assigns_same_ic_to_lane_group`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `uv run pytest tests/methods -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: count-weighted and score-weighted visited-buffer methods`.

### Task 17: own_success (M4) and phased (M9)

**Files:**
- Create: `methods/own_success.py`, `methods/phased.py`, `tests/methods/test_own_success.py`, `tests/methods/test_phased.py`

**Interfaces:**
- Produces:
  - `own_success`: settings `order: Literal["uniform", "frontier"] = "uniform"`, `advance=0.8`, `retreat=0.15`, `window=10`, `max_trajectories=16`, `p_task`. `accepts_scores = {"none", "steps_to_reward"}`.
    - Keeps a per-env trajectory of visited states for the current episode, using `offer_visited`. When an episode first reaches reward, it stores that trajectory, each state tagged with steps-to-reward.
    - Inert (all task start) until the first stored trajectory.
    - `frontier` moves the start index back when the success rate over the window behind it is ≥ `advance`, and forward when it's ≤ `retreat`.
    - Calls `score.set_steps` when the score is `steps_to_reward`.
  - `phased`: settings `count="neighbour"`, `radius=0.5`, `order="frontier"`, `p_task`. `accepts_scores = {"none"}`: it builds its own inner scores through `SCORES.construct`, `visit_count` before reward and `success_band(form="pq")` after, so the arm's score field stays `None`.
    - Delegates to an inner `visited_count` until any outcome has `reached_reward`, then to an inner `own_success`.
    - Logs `phase` and `switch_step`.
- [ ] **Step 1: Write failing tests**:
  - `test_own_success_inert_before_first_reward`;
  - `test_stores_trajectory_on_first_reward_with_steps_to_reward`;
  - `test_frontier_moves_back_on_high_success_and_forward_on_collapse`;
  - `test_uniform_order_draws_along_whole_trajectory`;
  - `test_phased_switches_on_first_reward_and_logs_step`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `uv run pytest tests/methods -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: own-success trajectory restarts and phased method`.

### Task 18: walk (M6), ball and genetic (M7)

**Files:**
- Create: `methods/walk.py`, `methods/proposal.py`, `tests/methods/test_walk.py`, `tests/methods/test_proposal.py`

**Interfaces:**
- Consumes: `Rollouts`, `task.start_box`, `task.is_valid`, `task.state_features`.
- Produces:
  - `PopulationMixin`, shared: `K = num_envs // R` ICs with R envs each; IC versioning, so outcomes from a replaced IC are dropped; `replace(slots, states)` calls `score.forget`.
  - `walk`: settings `R=4`, `walk_steps=20`, `action_hold=5`, `parent_power=1.0`, `keep_top=0.2`, `update_every=20`, `p_task`. `accepts_scores = {"success_band", "slope", "critic_loss", "critic_value", "success_classifier", "rnd", "random", "inverted"}` (spec §5.4).
  - `ball`: settings `R=4`, `parent_power=1.0`, `radius_power=1.0`, `r_min=0.02`, `r_max=0.3`, `keep_top=0.2`, `tries=8`, `screen_seconds=0.0` (0 disables screening), `spread_threshold=0.1`, `update_every`, `p_task`. Same `accepts_scores`.
  - `genetic`: settings `R=4`, `p=0.2`, `sigma_min=0.02`, `sigma_max=0.3`, `n_candidates=4`, `novelty_k=5`, `screen_seconds=0.0`, `spread_threshold=0.1`, `update_every`, `p_task`. Same `accepts_scores`.
  - Culling is skipped when `(lp.max() − lp.min()) / max(|median|, 1e-8) < spread_threshold` (2503.19037 analogue).
  - Children outside `start_box` or failing `is_valid` are redrawn up to `tries`, then fall back to `sample_start`.
- [ ] **Step 1: Write failing tests**:
  - `test_walk_children_are_reached_states_and_charged`;
  - `test_ball_children_within_radius_and_valid`;
  - `test_ball_invalid_children_fall_back_to_task_start` (stub `is_valid` rejects all);
  - `test_genetic_replaces_bottom_p_with_children_of_top_p`;
  - `test_no_cull_when_scores_flat`;
  - `test_screening_charges_screen_steps`;
  - `test_replaced_ic_outcomes_dropped_by_version`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `uv run pytest tests/methods -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: walk, ball and genetic population methods`.

### Task 19: train.py and metrics sinks

**Files:**
- Create: `src/resetsweep/train.py`, `src/resetsweep/metrics.py`, `tests/test_train.py`, `tests/test_metrics.py`

**Interfaces:**
- Consumes: everything above.
- Produces in `metrics.py`: `Protocol MetricsSink` with `log(step: int, data: dict[str, float]) -> None`, `summary(data: dict[str, float | str]) -> None`, `close() -> None`.
  - `NullSink`, `JsonlSink(path: Path)` (append, one JSON object per line, flushed), `MultiSink(*sinks)`.
  - `WandbSink(*, project, group, name, config, mode: Literal["online", "offline", "disabled"])`, which wraps `wandb.init` / `run.log` / `run.summary` / `run.finish`. Check these against the current W&B docs via Context7 before writing (load the `observability-expert` skill).
  - `config` must not contain keys matching `r"(?i)key|token|secret"`. Raise `ValueError` if it does.
- Produces in `train.py`:
  - `RunSpec` (frozen dataclass): `task: str, task_kw, method: str, method_kw, score: str | None, score_kw, seed: int, step_budget: int, num_envs: int, rollout_len: int, ppo: PPOConfig, eval_every_steps: int, eval_n: int = 512, eval_seed: int = 12345, device: str`. Method `arm_id() -> str`, a stable hash of every field except `seed` and `device`.
  - `RunResult` (frozen dataclass): `arm_id, seed, status: Literal["ok", "failed"], error: str | None, auc: float, final_success: float, reached_reward: bool, steps_to_first_reward: int | None, ever_succeeded_share: float, env_steps_train: int, env_steps_extra: dict[str, int], wall_s: float`.
  - `run(spec: RunSpec, sink: MetricsSink) -> RunResult`.
- `run` composes `TASKS`, `SCORES`, `METHODS` through `construct`. It seeds torch and the generators from `spec.seed`. It stops once `counter.total ≥ step_budget`, where the counter includes extra steps (anti-goal 2).
- Every `eval_every_steps` it runs a deterministic-policy eval from `task.eval_states(eval_n, eval_seed)` in a separate `VecEnv`. Eval steps aren't charged to the arm.
- It logs every key listed in spec §5.6 as named metrics: `env_steps/{total,train,walk,screen,scout}`, `eval/success`, `eval/return`, `train/curriculum_return`, `ic/worst_group_success`, `ic/never_succeeded_share`, `ic/sampling_entropy`, `first_reward_step`, `ppo/*`, `perf/steps_per_s`, `perf/wall_s`, and `side/*`.
- Per-step host syncs are banned: aggregate on device and call `.item()` only at log points.
- AUC = trapezoid of eval success over `env_steps/total`, normalised by `step_budget`.
- [ ] **Step 1: Write failing tests**:
  - `test_run_stops_at_step_budget_including_extra_steps` (walk method, tiny budget);
  - `test_run_result_auc_is_zero_when_never_successful`;
  - `test_eval_uses_fixed_states_same_across_seeds`;
  - `test_jsonl_sink_writes_one_line_per_log`;
  - `test_wandb_sink_rejects_secret_like_config_keys` (`mode="disabled"`);
  - `test_arm_id_ignores_seed_and_device`.

  (The all-combinations smoke test belongs to Task 20, because it needs the arm space.)
- [ ] **Step 2:** Run `uv run pytest tests/test_train.py tests/test_metrics.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `uv run pytest -v -m "not slow"`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: single-run training loop with W&B and JSONL metrics`.

### Task 20: Arm space and compatibility rules

**Files:**
- Create: `src/resetsweep/sweep/__init__.py`, `sweep/arms.py`, `tests/sweep/test_arms.py`, `tests/test_smoke.py`

**Interfaces:**
- Consumes: `METHODS[...].accepts_scores`, `TASKS.names()`.
- Produces:
  - `Arm` (frozen dataclass): `task, task_kw: tuple[tuple[str, Any], ...], method, method_kw: tuple, score: str | None, score_kw: tuple`; `.label() -> str`.
  - `SPARSE_TASKS: tuple[str, ...]`: every registered task whose name doesn't end in `_dense`, excluding `claw`.
  - `DENSE_TASKS = ("pendulum_dense", "arm_dense", "claw")`.
  - `arm_space(stage: Literal["rung0", "rung1_expand"], *, survivors: Sequence[Arm] = ()) -> list[Arm]`, with these §5.4 rules:
    - methods without scores get `score=None`;
    - `visited_count` gets `visit_count` (both `post_reward` forms) once per counting rule;
    - `own_success` gets `None` and `steps_to_reward`;
    - `phased` gets `None` (its inner scores are fixed);
    - `buffer_scored`, `walk`, `ball`, `genetic` get each S1 form, S3 form and S4 form, plus S5, `success_classifier`, `rnd` and `random` (spec §5.4).
  - At `rung0`: `p_task = 0.5`, except `visited`, which gets every `P_TASK_SWEEP` value. Sparse tasks only, default task settings.
  - `rung1_expand(survivors)` adds, for each survivor:
    - `p_task ∈ {0.2, 0.8}`;
    - `inverted(<its score>)` if it has one;
    - task variants (`maze lanes ∈ {5}`, `maze_trap trap_scale ∈ {0.5, 2.0}`);
    - the same arm on `DENSE_TASKS`.
  - `uniform` and `point` are always included on every task.
- [ ] **Step 1: Write failing tests**:
  - `test_no_scoreless_method_paired_with_score`;
  - `test_steps_to_reward_only_with_own_success`;
  - `test_visit_count_only_with_visited_count`;
  - `test_side_model_scores_only_with_population_and_buffer_methods`;
  - `test_visited_gets_all_p_task_values_at_rung0`;
  - `test_rung0_has_no_dense_tasks_or_variants`;
  - `test_rung1_expand_adds_inverted_and_dense_for_survivors_only`;
  - `test_uniform_and_point_on_every_task`;
  - `test_arm_labels_unique`;
  - `test_rung0_arm_count_printed` (asserts an exact count; compute it once and pin it, so any later change to the space is deliberate).

  In `tests/test_smoke.py`: `test_every_rung0_method_score_pair_runs_two_iterations`, marked `slow`. It covers each distinct (method, score) pair on `pendulum_sparse` and `maze` with `num_envs=32` on CPU.
- [ ] **Step 2:** Run `uv run pytest tests/sweep/test_arms.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `uv run pytest tests/sweep tests/test_smoke.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: arm space with compatibility rules`.

### Task 21: Statistics

**Files:**
- Create: `sweep/stats.py`, `tests/sweep/test_stats.py`

**Interfaces:**
- Produces:
  - `iqm(x: np.ndarray) -> float`: mean of the middle 50%, using the trimming convention of 2108.13264.
  - `bootstrap_ci(x, *, stat=iqm, n_boot=2000, alpha=0.05, seed=0) -> tuple[float, float]`.
  - `stratified_bootstrap_ci(scores_by_task: dict[str, np.ndarray], ...)`: resamples seeds within each task.
  - `normalise(score, *, low, high) -> float`: maps `uniform` → 0 and best → 1, and is safe when `high == low`.
  - `rung0_key(results: Sequence[RunResult]) -> tuple[float, float, float, float]` (§6 zero ties): `(iqm auc, share reached_reward, −median steps_to_first_reward (inf if none), share ever_succeeded)`, compared lexicographically.
- [ ] **Step 1: Write failing tests**. Cross-check each expected value with robot-learning-expert `scripts/run_stats.py` before freezing it:
  - `test_iqm_of_known_array` (`[1..8]` → 4.5);
  - `test_iqm_robust_to_outlier`;
  - `test_bootstrap_ci_contains_point_estimate_and_narrows_with_n`;
  - `test_stratified_resamples_within_task`;
  - `test_normalise_maps_uniform_to_zero_best_to_one`;
  - `test_rung0_key_breaks_zero_auc_ties_by_reached_reward_then_steps`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `uv run pytest tests/sweep -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: IQM, bootstrap intervals and rung-0 tie-break key`.

### Task 22: Scheduler

**Files:**
- Create: `sweep/scheduler.py`, `tests/sweep/test_scheduler.py`

**Interfaces:**
- Consumes: `Arm`, `RunResult`, `stats`.
- Produces:
  - `Job` (frozen dataclass): `arm: Arm, seed: int, rung: int, step_budget: int`.
  - `ArmState`: `arm, rung, results: list[RunResult], status: Literal["active", "promoted", "pruned", "final"]`.
  - `Scheduler(*, arms: list[Arm], rung_budgets: tuple[int, int, int], seeds_per_rung=(1, 3, 5), eta=3, beta=2.0, state_dir: Path)` with:
    - `next_jobs(k: int) -> list[Job]`;
    - `report(result: RunResult) -> None`;
    - `promotions() -> None`, run after a rung's jobs finish for a task;
    - `done() -> bool`;
    - `save() / Scheduler.load(state_dir)`.
- Rules (§7):
  - Rung 0 runs every arm once before any UCB choice.
  - UCB = mean + β·sd/√n, with `sd` = that task's spread at the rung when n < 2.
  - At rung 0 the "mean" is replaced by the `rung0_key` ordering (rank-transformed to [0, 1]).
  - Promote the top `ceil(n_task_arms/eta)` per task.
  - Prune when UCB < the leader's LCB.
  - Never prune `uniform` or `point`.
  - Done when every task has an arm whose LCB > every other arm's UCB, or no jobs remain.
  - Failed results: retry once with `seed + 1000`, then exclude.
- State is appended to `state_dir/results.jsonl` and `state_dir/scheduler.json`, each written to a temp file then `os.replace`.
- [ ] **Step 1: Write failing tests**:
  - `test_rung0_schedules_every_arm_once_before_repeats`;
  - `test_ucb_prefers_uncertain_arm_with_close_mean` (hand-computed);
  - `test_promote_top_third_per_task`;
  - `test_prune_when_ucb_below_leader_lcb`;
  - `test_uniform_and_point_never_pruned`;
  - `test_failed_job_retried_once_then_excluded`;
  - `test_done_when_leader_separated`;
  - `test_save_load_round_trip_resubmits_running_jobs` (Review Focus 2);
  - a Hypothesis property: `next_jobs` never returns a pruned arm's job.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `uv run pytest tests/sweep -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: UCB successive-rung scheduler with resume`.

### Task 23: Budget guard, worker pool, calibration, sweep entry point, report

**Files:**
- Create: `sweep/budget.py`, `sweep/pool.py`, `sweep/main.py`, `sweep/report.py`, `tests/sweep/test_budget.py`, `tests/sweep/test_pool.py`, `tests/sweep/test_main.py`

**Interfaces:**
- Produces in `budget.py`:
  - `BudgetGuard(*, cap_usd: float, margin_usd: float, gpu_usd_per_h: float, volume_usd_per_h: float, ledger_path: Path, clock: Callable[[], float] = time.monotonic)`:
    - `spent() -> float`, including previous sessions from the ledger;
    - `can_start(est_job_s: float, in_flight_s: float) -> bool`;
    - `heartbeat() -> None`, which appends `{t, spent}` to the ledger at least every 60 s.
  - `volume_usd_per_h(gb) = gb * 0.07 / 730`, using 730 h/month (state this in the docstring).
- Produces in `pool.py`:
  - `WorkerPool(*, n_workers: int, device: str)`, using `multiprocessing` with the `spawn` context: `submit(job: Job, spec: RunSpec) -> None` and `poll() -> list[RunResult]`.
  - A crashing worker yields `RunResult(status="failed", error=<repr>)` (Review Focus 1).
- Produces in `main.py`: `main(argv=None) -> int` (the `sweep-run` script):
  1. Load settings from env: `SWEEP_CAP_USD`, `SWEEP_GPU_USD_PER_H`, `SWEEP_STATE_DIR` (default `/workspace/sweep`), `WANDB_PROJECT` (default `reset-curriculum-sweep`).
  2. Calibrate: for each sparse task family, run `uniform` on `dense_sibling` with 2 PPO configs (from Task 11 references) for a short budget. Keep the better config and measure steps/s at `num_envs ∈ {2048, 4096, 8192}` and `n_workers ∈ {1, 2, 4}`. Stop calibration at `CALIBRATION_CAP_USD`. Save to `state_dir/calibration.json`, and reuse it on restart.
  3. Size the rungs: `r0` such that `len(arm_space("rung0")) × r0 / steps_per_s × usd_per_s ≈ 0.35 × remaining`; `r1 = 3·r0`, `r2 = 9·r0`.
  4. Loop: `next_jobs` → `can_start` → submit → poll → `report` → `save`, until `done()` or `can_start` is false.
  5. Write the report.
  6. Return 0.

  With `SWEEP_SMOKE=1`: calibrate one task family (`pendulum`), then run only `uniform` and `visited` (`p_task=0.5`) on `pendulum_sparse` for one rung of `r0` and write the report. This is the mode Task 25 pays for.
- Produces in `report.py`:
  - `write_report(scheduler, path_md, path_json, sink)`: the per-task table (IQM AUC, CI, seeds, zero-seed count, rung) and the overall normalised IQM ranking. Pruned arms are listed separately and unranked.
  - Also logged as a `wandb.Table` in a sweep-level run.
- [ ] **Step 1: Write failing tests**:
  - `test_spent_includes_prior_ledger_entries` (Review Focus 2);
  - `test_can_start_false_when_projected_exceeds_cap_minus_margin`;
  - `test_volume_cost_per_hour`;
  - `test_worker_crash_returns_failed_result` (a job that raises);
  - `test_main_stops_when_budget_exhausted` (fake clock, fake pool, tiny cap);
  - `test_main_resumes_from_state_dir`;
  - `test_smoke_mode_runs_only_uniform_and_visited_on_pendulum`;
  - `test_report_ranks_only_final_rung_arms_and_lists_pruned`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `uv run pytest -v -m "not slow"`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: budget guard, worker pool, calibration and sweep entry point`.

### Task 24: RunPod client, launcher, pod bootstrap

**Files:**
- Create: `launch/__init__.py`, `launch/env_file.py`, `launch/runpod_client.py`, `launch/launcher.py`, `scripts/pod_bootstrap.sh`, `tests/launch/test_env_file.py`, `tests/launch/test_runpod_client.py`, `tests/launch/test_launcher.py`, `tests/launch/test_bootstrap.bats` (or a pytest that runs `bash -n` and `shellcheck` when available)

**Interfaces:**
- Produces in `env_file.py`:
  - `load_secrets(path: Path) -> Secrets`, using `dotenv_values`. Check the `python-dotenv` API via Context7 first.
  - `Secrets` (frozen dataclass) with `runpod_api_key: str` and `wandb_api_key: str`, and `__repr__` returning `Secrets(***)`.
  - Raises `MissingSecret(name)` for an absent or blank key (Review Focus 4).
- Produces in `runpod_client.py`: `RunPodClient(api_key: str, *, transport: httpx.BaseTransport | None = None, base_url="https://api.runpod.io/v2", timeout_s=30.0, max_retries=4)` with:
  - `gpu_types() -> list[GpuType]`, from `GET /catalog/gpus?include=AVAILABILITY&product=POD&cloud=SECURE`;
  - `datacenters() -> list[DataCenter]`, from `GET /catalog/datacenters?include=GPU_AVAILABILITY`;
  - `create_volume(name, size_gb, datacenter) -> str`, via `POST /network-volumes`;
  - `create_pod(body: dict) -> dict`, via `POST /pods`;
  - `delete_pod(pod_id) -> None`, via `DELETE /pods/{id}`, which treats 204 as success;
  - `balance_usd() -> float`. Find the endpoint in the REST v2 reference (docs.runpod.io/api-reference-v2) before writing it; the MCP tool `get-balance` shows it exists. If REST v2 has none, use the GraphQL `myself { clientBalance }` query from docs.runpod.io/sdks/graphql and say so in the commit.

  Retries on 429/5xx use exponential backoff with full jitter, honouring `Retry-After`. Errors raise `RunPodError(status, detail)`, with the API key never in the message.
- Produces in `launcher.py`: `main(argv=None) -> int` (`sweep-launch`), with flags `--gpu` (default `"NVIDIA GeForce RTX 4090"`, fallback list `("NVIDIA A40", "NVIDIA L40S", "NVIDIA RTX A6000")`), `--cap` (default 100), `--ref` (default the current git branch), `--repo-url` (default from `git remote get-url origin`), `--dry-run`, `--smoke` (adds `SWEEP_SMOKE=1` to the pod env; test `test_smoke_flag_sets_env`).
  1. Load secrets.
  2. Read the local spend ledger (`~/.resetsweep/ledger.json`).
  3. Check that `remaining = cap − ledger_spent` is positive and that the balance is ≥ remaining.
  4. Pick the GPU and a data center where it is available with `"STANDARD" in networkVolumeTypes`.
  5. Compute `max_hours = (remaining − margin) / (gpu_usd_per_h + volume_usd_per_h)` and print it.
  6. Create a 20 GB volume, then the pod with body:
     ```
     {name, image, cloud: "SECURE",
      gpu: {id, count: 1, minCudaVersion: "13.2"},
      disk: 40,
      mounts: {network: [{volumeId, path: "/workspace"}]},
      dataCenterIds: [dc],
      env: {RUNPOD_API_KEY, WANDB_API_KEY, REPO_URL, REPO_REF,
            SWEEP_CAP_USD: remaining, SWEEP_GPU_USD_PER_H, SWEEP_MAX_HOURS},
      cmd: ["bash", "-c", <clone-and-bootstrap one-liner>]}
     ```
  7. On pod-create 400, try the next GPU type in the same data center. If all fail, print the orphaned volume id and the exact delete command, and exit 2 (Review Focus 3).
  8. Record the launch in the ledger, and print the pod id, the hourly cost and the W&B project URL.

  `--dry-run` prints the bodies with secrets replaced by `"***"`, and makes no POST.
- The clone one-liner: `set -euo pipefail; cd /workspace; if [ -d repo ]; then git -C repo fetch --depth 1 origin "$REPO_REF" && git -C repo checkout FETCH_HEAD; else git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" repo; fi; exec bash repo/scripts/pod_bootstrap.sh`.
- `pod_bootstrap.sh` (load `bash-scripting-specialist`):
  1. `python3 -m pip install uv==0.13.10` (no curl-pipe-shell);
  2. `uv sync --frozen`;
  3. `uv run pytest -m gpu -q`, stopping on failure;
  4. `uv run sweep-run`;
  5. a `trap` that always runs the self-terminate (`curl -fsS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID"`) on exit, success or failure, so a crash can't leave a billing pod;
  6. logs to `/workspace/sweep/bootstrap.log`.

  `RUNPOD_POD_ID` is set inside pods per the docs.runpod.io example (`runpodctl pod stop $RUNPOD_POD_ID`); the script checks it is non-empty first. The API key is never echoed: the script runs `set +x` around any line that uses it.
- [ ] **Step 1: Write failing tests**, using `httpx.MockTransport`, which needs no extra dependency (confirm in the httpx docs):
  - `test_missing_or_blank_key_raises_naming_key_before_any_http`;
  - `test_secrets_repr_hides_values`;
  - `test_retries_on_429_honouring_retry_after_then_succeeds`;
  - `test_gives_up_after_max_retries_with_runpod_error_without_key_in_message`;
  - `test_delete_pod_204_is_success`;
  - `test_create_pod_body_has_network_mount_secure_cloud_min_cuda_and_env`;
  - `test_dry_run_redacts_secrets_and_makes_no_post`;
  - `test_refuses_launch_when_balance_below_remaining_cap`;
  - `test_refuses_launch_when_ledger_exhausted_cap`;
  - `test_falls_back_to_next_gpu_on_400_and_reports_orphan_volume_when_all_fail`;
  - `test_bootstrap_script_passes_bash_n_and_shellcheck` (shellcheck skipped if not installed, and the skip reason printed).
- [ ] **Step 2:** Run `uv run pytest tests/launch -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `uv run pytest -v -m "not slow"`. Expected: PASS. Then run `uv run sweep-launch --dry-run --repo-url https://github.com/objones25/reset-curriculum-sweep --ref feat/sweep`. Expected: redacted bodies and the cost check printed, exit 0, no resources created. (This calls the catalog GETs with the real key. That is read-only and free.)
- [ ] **Step 5: Commit** `feat: RunPod launcher, client and pod bootstrap`.

### Task 25: README, public repo, paid GPU smoke test

**Files:**
- Create: `README.md`
- Modify: nothing else unless the smoke test exposes a bug, which then gets a test first.

**Interfaces:**
- Consumes: everything.

- [ ] **Step 1:** Write `README.md` (load `documentation-expert`, and run its checks): what the sweep does, `uv sync`, `uv run pytest`, `uv run sweep-launch --dry-run`, `uv run sweep-launch`, where results land, how to delete the network volume, and the cost model. Every command in it must have been run.
- [ ] **Step 2:** Run the whole suite: `uv run pytest -v` (including `slow`), `uv run ruff check .`, `uv run pyright src`. Expected: all green. Paste the summary lines into the task report.
- [ ] **Step 3:** Create the public repo and push (the user approved a public repo): `gh repo create objones25/reset-curriculum-sweep --public --source . --remote origin --push`. Before pushing, run `git ls-files | grep -E '(^|/)\.env|\.claude/'`. Expected: no output.
- [ ] **Step 4: Paid smoke test** (user-approved, a few dollars). Run `uv run sweep-launch --cap 5 --ref feat/sweep --smoke`. `--smoke` adds `SWEEP_SMOKE=1` to the pod env; add the flag and its body test to Task 24's launcher.
  - Watch it with the RunPod MCP `stream-pod-logs`.
  - Expected: GPU tests pass, the W&B runs appear with the §5.6 metric names, the report is written to `/workspace/sweep/report.md`, and the pod self-terminates.
  - Confirm termination with `get-pod` (404), and record the actual spend from `list-pod-billing`.
- [ ] **Step 5:** Commit any fixes, then `docs: README with launch and cost instructions`. Report to the user: the smoke test's spend, measured steps/s per task, the projected rung sizes for a $100 sweep, the W&B link, the network volume id with its monthly cost, and its delete command.
