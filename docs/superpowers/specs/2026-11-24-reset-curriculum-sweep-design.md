# Reset-curriculum sweep: design

Date: 2026-11-24. Status: draft for approval.

## 1. Intent

Find out which **(task, method, score)** combinations from `reset_curriculum_shortlist.md` work best for massively parallel PPO. "Best" means learning the task from its own start states. Every shortlist method, score and task gets implemented fresh on GPU MuJoCo. One local command launches the whole sweep on RunPod, and every metric goes to Weights & Biases. A scheduler spends a hard budget of $100 on the combinations that are most uncertain and have the most room to beat the current leader.

The CS291K `bucketing` repo is inspiration only. Nothing is ported from it, and the sweep needn't stay compatible with it.

## 2. Goals, non-goals, anti-goals

**Goals**
- Implement every method (M1–M8), score (S1–S7) and task (T1–T7) in shortlist sections 3–5, plus the section 6 controls.
- One command, `uv run sweep-launch`, does all of this:
  1. reads `RUNPOD_API_KEY` and `WANDB_API_KEY` from `.env`;
  2. creates a network volume and a pod;
  3. passes both keys to the pod as environment variables;
  4. has the pod clone this repo and run the sweep.
- Pick runs adaptively (section 7). Prune combinations that can't win. Give the leaders more seeds and longer training.
- A final ranking per task and overall, with IQM and bootstrap intervals. The ranking lives in W&B and on the network volume.
- Test-first development with pytest, using uv for packages.

**Non-goals**
- Reproducing the bucketing repo's numbers, physics or API.
- Sim-to-real, rendering, vision observations.
- Tuning PPO hyperparameters per combination. One PPO config is used per task (section 6).
- Distributed training across several GPUs or pods.

**Anti-goals** (any of these makes the project a failure, even if the code runs)
- Spending more than $100 of RunPod credit, network-volume storage included.
- A ranking that reflects unequal compute: every environment step charged to an arm counts, including walks, screening rollouts and scouting.
- A ranking built from single seeds or reported without uncertainty.
- Secrets in git, logs, W&B config or the pod's printed start command.

## 3. Constraints

- **Simulator:** MuJoCo Warp (`mujoco-warp`), batched over `nworld` environments on an NVIDIA GPU. The docs list a CPU mode for development and debugging (robot-learning-expert `simulators-and-tooling.md`, checked 2026-11-02). Python ≥ 3.11 per the same source. The exact Python and package pins get re-checked against PyPI at plan time.
- **Learner:** PyTorch PPO, written in this repo. Warp arrays are shared with torch without copying via `wp.to_torch` / `wp.from_torch` (NVIDIA Warp docs, PyTorch interop page, read via Context7 2026-11-24).
- **Restoring states:** a start state is (qpos, qvel, plus any task-specific extra state), written into the selected worlds' `data.qpos` / `data.qvel` through those torch views.
- **Determinism:** batched GPU MuJoCo isn't reproducible run to run. GPUSimBench (2607.13059) classes MuJoCo Warp as differing both between envs in a run and between runs. So conclusions come from seeds, never from one run.
- **Cloud:** RunPod REST v2, `POST /v2/pods` (fields `image`, `gpu.id`, `env`, `args`, `mounts.network[].volumeId/path`, `cloud`, `dataCenterIds`) and `/v2/network-volumes` (`size` 10–4096 GB, `dataCenter` fixed at creation). Network volumes attach only to Secure Cloud pods in the volume's data center, and storage is billed hourly per GB from creation until deletion. Termination is `DELETE /v2/pods/{id}`. All from docs.runpod.io, read 2026-11-24.
- **Prices** (RunPod catalog, read 2026-11-24, Secure Cloud): RTX 4090 $0.89/h, A40 $0.59/h, L40S $1.09/h, A100 SXM $2.19/h. The launcher reads live prices at launch and never uses these constants.
- **Packages:** uv with a committed `uv.lock`, Python version pinned.
- **Git:** `.env` and `.claude/` are gitignored from the first commit. Work goes on branches (`type/short-description`).
- **Library facts:** every API used is checked against current docs (Context7, PyPI, official docs) at plan or implementation time, never written from memory.

## 4. Approaches considered

**Simulator and learner**
- **A. MuJoCo Warp + own PyTorch PPO (chosen).** Each task is a small MJCF model we write. Restarting from any stored state is a direct write through torch views. Fits the PyTorch skill set the user asked for.
- B. MJX + JAX PPO. Cheaper per run, because a whole run compiles to one program and seeds can be vmapped. Rejected: the learner wouldn't be PyTorch.
- C. MuJoCo Playground. Rejected: 1.x's registry lists `CartpoleBalance`, two pick-cube tasks and G1 tracking (Playground `AGENTS.md`, via Context7), not the sparse classics the shortlist needs. Tasks would have to be written anyway.

**Spending the budget**
- **UCB with successive rungs (chosen).** Every valid arm gets a short run. Arms are then ranked by an upper confidence bound and promoted to longer runs and more seeds, and pruned when their upper bound falls below the leader's lower bound. This is the user's "most uncertainty + potential to improve" rule.
- Full grid at fixed seeds. Rejected: about 1,500 runs doesn't fit $100.
- Gaussian-process Bayesian optimisation. Rejected: the arms are discrete and unordered, so a GP kernel over them adds little that a per-arm posterior doesn't give.

## 5. What gets built

### 5.1 Layout (my call: one package, one folder per pluggable kind)

```
src/resetsweep/
  sim/        MuJoCo Warp batch env: step, state read/write, staggered clocks, truncation vs termination
  tasks/      one file per task (MJCF + obs + reward + success + start distribution + eval states)
  methods/    one file per reset method
  scores/     one file per score
  ppo/        PPO, GAE with truncation bootstrap, running obs normalisation, health metrics
  train.py    one run: (task, method, score, seed, step budget) -> metrics, logged to W&B
  sweep/      arm space, scheduler, worker pool, results store, budget guard, final report
  launch/     local CLI: .env -> network volume + pod; and the in-pod entry point
tests/
```

Registries are plain dicts filled by a decorator, one per kind (task, method, score). Each kind has a `Protocol` describing its interface. Adding a new idea means one new file. The composition root is `train.py` (one run) and `sweep/` (many runs). That is the only code that knows concrete classes.

### 5.2 Interfaces (behaviour, not final signatures)

- **Task:** builds its MJCF model and has these parts:
  - `sample_start(n)`: the task's own start distribution, as start states;
  - `observe`;
  - `reward`, which is sparse wherever the shortlist says sparse;
  - `terminated`, for real terminal states only;
  - `success`, with a hold time;
  - `eval_states`: fixed, drawn once from `eval_seed`, the same for every arm and seed;
  - `state_features`, a vector used by count, distance and novelty scores;
  - optional variant parameters, such as maze size or trap scale;
  - `start_box`: per-dimension bounds of a start state (positions and velocities) plus a validity check, used by M7 and by `point`'s random source;
  - `dense_sibling`: the dense task used for PPO tuning (section 5.5).
- **Method:** the reset rule. On every reset it is told which envs finished, plus their outcomes: return, success, IC id, start time. It returns start states for those envs. It is offered visited states every k steps. It may ask the sim for extra rollouts (walks, screening); those steps are charged to the run's step budget through one counter the method can't bypass.
- **Score:** per-IC learning potential, computed from outcomes **and** optional PPO data: per-IC value loss, MaxMC, mean |GAE|, and V(s0) from one batched critic call. Side-model scores (success classifier, RND, step-distance embedding) own their networks. The training loop gives scores what they need, which fixes the gap where scores could only see returns.

### 5.3 Shortlist coverage

**Tasks** (each a MuJoCo model; the geometry is left to the builder, the behaviour is fixed here):

| ID | Task | Required behaviour |
|---|---|---|
| T1 | `maze` | Point mass in a walled corridor maze. Sparse reward in a goal region. Variants: `narrow` (narrow passages, after the N-maze of 2610.03395) and a size sweep (number of lanes). |
| T2 | `pendulum_sparse`, `cartpole_swingup_sparse`, `acrobot_swingup_sparse`, `arm_sparse` | Swing-up / reach tasks. The reward is a success indicator plus small action penalties (as in the SGS manipulation reward, shortlist §5.2). `arm_sparse`: 3-link planar arm moving an object to a goal (my call: a push task, to avoid fragile grasp contacts). |
| T3 | `maze_trap` + `maze_trap_twin` | A region that can't be escaped (an underpowered agent in a pit), and a twin with identical geometry that can be escaped. A parameter scales the region. |
| T4 | `maze_infeasible` | Task-parameter mass split as in ToyLevels: about 90% easy, 0.01% hard-but-feasible, about 9.99% infeasible (e.g. start pockets with no path to the goal). |
| T5 | `maze_coinflip` | A block of states where success is decided by a coin flip the policy can't influence. |
| T6 | `ramp_deceptive` | An underpowered cart in a valley that must first move away from the goal to build momentum, after Kinetix Car-Ramp. Sparse goal. |
| T7 | `pendulum_dense`, `arm_dense`, `claw` | Sanity tasks: a curriculum shouldn't hurt here. `claw`: two-finger planar gripper turning a disc; lowest priority. |

**Methods:**

| ID | Method | Settings exposed |
|---|---|---|
| M1 | `uniform` (control) | — |
| M2 | `visited`: ring buffer of visited states from all time offsets, uniform draw | `p_task` ∈ {0, 0.2, 0.5, 0.8, 1.0} |
| M3 | `visited_count`: draw ∝ 1/sqrt(count+1) | count = raw-state neighbour count (fixed radius on standardised features), lineage depth, or step-distance-embedding neighbours; `p_task` |
| M4 | `own_success`: restart from states on the agent's own first successful trajectories | `uniform` along the trajectory or a `frontier` (advance at 0.8, retreat at 0.15, per 2607.26417; alternative bar 0.2 per 1812.03381); `p_task` |
| M5 | `buffer_scored`: score-weighted draw over the visited buffer | staleness term, discounting, minimum count, rehearsal share, ever-succeeded bit, lanes (several envs per IC); `p_task` |
| M6 | `walk`: a population of K ICs, each shared by R envs. Every update, the lowest-scored ICs are replaced by states reached from a parent IC by a random-action walk. Parents are chosen by score rank. | walk length, action hold, parent power, keep-top share; steps charged; `p_task` |
| M7 | `ball`: children drawn inside a ball around a parent in `start_box` coordinates, with the radius set by the parent's score. `genetic`: bottom 20% by score replaced by mutated children of the top 20%; high-score parents mutate less; low-score parents keep their most novel candidate (k-NN on `state_features`). Invalid children are redrawn, then fall back to the task start. | optional hold-rollout screening (charged); cull only when the score spread exceeds a threshold; `p_task` |
| M8 | `point` (control): one IC for every env, never updated | `source` = the task's canonical start or one random draw |
| M9 | `phased` (the shortlist's §1 item 5 "suggested build"): `visited_count` until any env first reaches reward, then `own_success` with a success-band score | the settings of both, `p_task` |

Every method mixes a `p_task` share of resets at the task start. The default is 0.5, as used in 2602.20809 (shortlist §3).

**Scores:** S1 success band (SGS weight and p(1−p)); S2 visit count (and count × (learnability + ε) after the first reward); S3 return slope (signed and absolute, plus fast-minus-slow); S4 positive value loss, MaxMC, mean \|GAE\|; S5 critic V(s0)·(Vmax − V(s0)); S6 steps-to-reward along a stored successful trajectory; S7 success classifier (0.3–0.7 band), RND novelty, step-distance embedding. **Control scores:** `random` and `inverted(<score>)`.

### 5.4 Removing nonsensical pairs

The arm space is built from explicit compatibility rules, which are tested:
- `uniform`, `point`, `visited` take no score.
- `visited_count` takes S2, or S2 × (S1 + ε) after the first reward (shortlist §4.2). The counting rule is a separate variant.
- `own_success` takes only S6, or none (uniform along the trajectory).
- `phased` takes S2 before reward and S1 after; no other pairing.
- **Rung 0 uses one setting per method:** `p_task` = 0.5 and the first-listed variant of each setting. The `p_task` sweep {0, 0.2, 0.5, 0.8, 1.0} runs from rung 0 for `visited` only, because that is the shortlist's §7 step 1. For any other method it runs only once the method reaches rung 1. Task variants (maze size sweep, trap scale) likewise run only for arms that reach rung 1.
- `buffer_scored`, `walk`, `genetic`, `ball` take S1, S3, S4 (each sub-variant), S5, S7 (classifier, RND), and the `random` control.
- `inverted(x)` is added only for an x that reaches the final rung (section 7), so the control costs money only where it matters.
- Every arm runs on every task, except that the dense T7 tasks get only the arms that reach the final rung on at least one sparse task. That's my call: their job is to show "doesn't hurt", not to rank.

### 5.5 PPO (one config per task, my call)

- Truncation is separate from termination. Bootstrap at truncation from the pre-reset observation.
- Running observation normalisation; Adam with eps 1e-5 and global-norm clip 0.5.
- Fixed minibatch size, with the minibatch count derived from N×T.
- Starting values come from massively-parallel-rl and robot-learning-expert references with their sources. They are tuned once per task family on that family's `dense_sibling` with `uniform`, because `uniform` gives no signal on the sparse tasks. The maze family's sibling is a maze with a distance-to-goal shaping term. The values then carry over to the sparse members unchanged.
- Calibration and tuning are charged to a "calibration" line **inside** the $100 cap, and capped at $10 of it.
- Every arm gets random initial episode-time offsets (shortlist F, 2511.21011).
- Health metrics logged: approx KL, clip fraction, explained variance, entropy, value loss.

### 5.6 Metrics (everything to W&B, also JSONL on the network volume)

Per run, every K iterations, read from device tensors (no per-step host syncs):
- env steps, total and split into training vs extra (walk, screen, scout);
- task-start eval success and return on the fixed eval states; return on curriculum envs next to it;
- per-IC success (not per finished episode), worst-group success, share of ICs that have never succeeded, buffer concentration (entropy of the sampling distribution);
- steps to first reward;
- PPO health; wall-clock and steps/s; side-model cost per iteration.

Per sweep: arm table with posterior mean, CI, UCB, rung, status (running / promoted / pruned); dollars spent and projected; final ranking.

## 6. Ranking metric

- **Per run:** area under the task-start success curve over the run's step budget, plus final success. The area rewards speed, which is the point of a curriculum. A run that never reaches reward scores 0, and "share of seeds that reach any reward" is reported next to it.
- **Per (task, arm):** IQM over seeds, with a stratified bootstrap CI. Seeds that end at zero are counted and reported.
- **Overall:** per task, normalise arms against `uniform` (0) and the best arm (1), then take the IQM across tasks.
- **Ties at zero** (common at rung 0 on sparse tasks) are broken, in order, by: share of seeds that reached any reward, then median steps to first reward, then share of ICs that have ever succeeded. The UCB and promotion rules use this composite at rung 0. From rung 1 on, AUC alone is used.
- **Equal compute:** the final ranking includes only arms that reached the last rung, with equal step budgets and equal seed counts. Arms pruned earlier are listed with the rung they reached, unranked. `uniform` and `point` are never pruned, so the normalisation anchor always exists at the final rung.

## 7. Scheduler

1. **Calibrate** (on the pod, first thing): measure steps/s per task at the chosen N and the number of concurrent worker processes per GPU. Those numbers turn dollars into env steps.
2. **Rungs:** step budgets r0 < r1 < r2, with ratio η = 3 (my call, in the style of successive halving). Each rung length is set from calibration so that rung 0 covers every arm once on the sparse tasks within about 35% of the budget.
3. **Pick the next job** with the highest UCB = mean + β·sd/√n among arms on their current rung that can still win. An arm with fewer than 2 seeds uses a wide prior sd from that task's spread at the rung. β is set in config; my default is 2.
4. **Promote** the top 1/η of arms per task by UCB to the next rung. Seeds rise to at least 3 at r1 and 5 at r2. Five is the minimum for a decision claim, per massively-parallel-rl rule 4 and robot-learning-expert `evaluation.md`.
5. **Prune** an arm when its UCB is below the task leader's LCB.
6. **Stop** at the budget (section 8) or when every task has one arm whose LCB beats every other arm's UCB.
7. **State and resume:** the full scheduler state and all results live on the network volume (append-only JSONL), so a restarted pod resumes where it stopped.

## 8. Budget guard ($100 hard cap)

- Before creating anything, the launcher reads live GPU and storage prices. It refuses to launch if `pod $/h × max_hours + volume $/GB·h × GB × max_hours > cap`.
- `max_hours` is derived from the cap. The cap defaults to $100 and can be lowered.
- In the pod, the guard counts spend from wall-clock × live price. It stops scheduling new jobs once projected spend, including all in-flight jobs, would pass `cap − margin` (margin $5, my call). After the final report is written, the pod terminates itself with `DELETE /v2/pods/$RUNPOD_POD_ID`.
- Before submitting, the launcher also checks the account balance, and refuses if the balance is below the cap.
- The cap is account-wide for this project. The launcher keeps a spend ledger on the network volume and locally, covering every pod it has launched (failed launches, GPU smoke tests, the sweep). Each launch is allowed only the cap minus what the ledger shows already spent.
- The network volume is **not** deleted automatically, because it holds the results. The launcher prints its id, its hourly cost and the delete command. Deleting it is the user's call.

## 9. Launcher

- `uv run sweep-launch [--gpu ...] [--cap 100] [--dry-run]`:
  1. reads `.env` without printing values;
  2. picks a data center that has the GPU available and supports network volumes;
  3. creates the volume, then the pod: Secure Cloud, the volume mounted at `/workspace`, `env` = {`RUNPOD_API_KEY`, `WANDB_API_KEY`, repo URL/ref, cap};
  4. prints the pod id and the W&B URL.
- The pod's start command installs uv, clones the repo at the given ref into `/workspace`, runs `uv sync --frozen`, and runs the sweep entry point.
- **Repo (user's decision):** a new **public** GitHub repo under the user's account, created with `gh` once the first code is ready to push. Pod secrets stay at two, `RUNPOD_API_KEY` and `WANDB_API_KEY`. Because the repo is public, a test fails if any tracked file matches a secret pattern, on top of `.env` being gitignored.
- **Who spends what (user's decision):** the builder runs one short paid GPU smoke test: one pod, calibration plus one arm end to end, a few dollars, recorded in the ledger. The user launches the full sweep.
- The pod image is an official RunPod PyTorch image whose CUDA matches what MuJoCo Warp and torch need (exact tag pinned at plan time).
- `--dry-run` prints the request bodies with secrets redacted and the cost check, and creates nothing.
- HTTP goes through one small RunPod client that has timeouts, bounded retries with jitter on 429/5xx, and an injectable transport for tests.

## 10. Testing (test-first, pytest)

- Unit tests on CPU, for:
  - the compatibility rules and the arm space;
  - UCB, promotion and pruning against hand-computed cases;
  - budget arithmetic;
  - IQM and bootstrap against known values;
  - GAE with truncation against a hand-computed oracle;
  - each method's reset rule;
  - each score against hand-built outcome sequences;
  - `.env` parsing and secret redaction;
  - RunPod client request bodies, via the injected transport;
  - W&B in `mode="disabled"`.
- Sim tests (MuJoCo Warp CPU mode, small `nworld`): writing and reading states round-trips; restoring one stored state into two worlds and stepping the same actions agrees within a stated tolerance (shortlist F, the simulator benchmark check); each task's success and reward fire where they should; trap and infeasible regions behave as specified.
- Smoke test: every arm × one tiny task × a few iterations.
- An end-to-end `--dry-run` launcher test.
- A GPU-only test marker for checks that need CUDA, run on the pod before the sweep starts. The sweep doesn't start if they fail.

## 11. Left to the builder

- MJCF geometry and constants for each task, within the behaviour in 5.3.
- Exact signatures, file names inside each folder, the JSONL schema, the W&B panel layout.
- N, T and PPO values per task, chosen in calibration from the cited references.
- Worker-process count per GPU, chosen from calibration.

## 12. My calls (for the user to check)

- Package name `resetsweep`, `src/` layout.
- `arm_sparse` is a push task, not pick-and-place.
- Dense T7 tasks only run finalists.
- `inverted(x)` controls only for finalists.
- Rung ratio η = 3, β = 2, $5 margin, rung 0 at about 35% of budget.
- Network volume kept at the end, not deleted.
- Ranking by area under the task-start success curve, normalised per task between `uniform` and the best arm.
- Definitions of `point`, `walk`, `ball`, `genetic` (5.3), written fresh from their one-line descriptions in the bucketing README.
- `phased` (M9) added, because the shortlist names it as its headline suggestion.
- `p_task` = 0.5 by default for every method. The `p_task` sweep and task variants run only from rung 1, except for `visited`.
- Zero-tie composite at rung 0; final ranking only among last-rung arms; `uniform` and `point` never pruned.
- PPO tuned on a dense sibling per task family; calibration capped at $10 inside the $100.
- The $100 cap is account-wide for this project, tracked in a spend ledger.
