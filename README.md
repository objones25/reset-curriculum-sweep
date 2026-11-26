# reset-curriculum-sweep

A sweep of reset-curriculum methods for GPU MuJoCo PPO, run on one RunPod pod under a $100 cap.

## What the sweep does

- **Grid.** Tasks x methods x scores, taken from `reset_curriculum_shortlist.md`. The tasks are in
  `src/resetsweep/tasks/`, the methods in `src/resetsweep/methods/`, the scores in
  `src/resetsweep/scores/`, and the allowed combinations in `src/resetsweep/sweep/arms.py`.
- **Training.** MuJoCo Warp (`mujoco-warp`) for batched GPU simulation, and a PyTorch PPO learner.
- **Scheduling.** A UCB successive-rung scheduler (`src/resetsweep/sweep/scheduler.py`). Rung 0
  runs every arm once on a short step budget. Promising arms are promoted to rungs with 3x and 9x
  the steps, and arms that cannot win are pruned. Rung sizes are set from calibration
  measurements and the remaining budget, not hard-coded: the whole expected ladder (rung 0, the
  promoted arms and their expansions, with 1, 3 and 5 seeds) is sized to cost 90% of what
  calibration leaves, each job weighted by its measured calibration-probe slowdown. Task variants and other `p_task` values join at rung 1; `inverted(x)`
  controls and the dense tasks run only for finalists. If the rung-0 budget would fall below
  `SWEEP_MIN_R0_STEPS` (default 2,000,000 env steps) the sweep refuses to start.
- **Budget.** Account-wide cap of $100 (`--cap` accepts at most 100), with a $5 margin ($0.50
  for a `--smoke` run, so `--cap 5 --smoke` leaves $4.50 to spend).
- **Output.** A final per-task and overall ranking (IQM with bootstrap intervals).

## Setup

Requires Python 3.12 (`requires-python = ">=3.12,<3.13"`) and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create a `.env` file in the repo root that sets these two keys, written as `NAME=value` lines:

- `RUNPOD_API_KEY`, from the RunPod console
- `WANDB_API_KEY`, from Weights & Biases

`.env` is git-ignored. The launcher reads it and never prints a value.

## Tests

```bash
uv run pytest                 # everything that runs on this machine, including slow tests
uv run pytest -m "not slow"   # skip tests marked slow (more than 10 s)
uv run pytest -m gpu          # tests that need CUDA
```

The full suite takes about 20 minutes or more. The `gpu` tests are skipped without CUDA; the
pod runs them (`uv run pytest -m gpu -q`) before the sweep starts.

Lint and type check:

```bash
uv run ruff check .
uv run pyright src
```

## Usage

`sweep-launch` (`resetsweep.launch.launcher:main`) creates a network volume and a pod on RunPod.
The pod fetches this repo at one pinned commit, runs `scripts/pod_bootstrap.sh`, and that runs `sweep-run`
(`resetsweep.sweep.main:main`).

The repo must be reachable by the pod: `--repo-url` (an `https://` URL) defaults to
`git remote get-url origin`. The pod checks out a commit SHA, never a moving branch: `--ref`
(a branch, tag or SHA) is resolved on the remote with `git ls-remote`, and without it the launcher
uses `git rev-parse HEAD`, or with `--volume-id` the SHA recorded for that volume in the ledger.
If the remote does not serve that commit, the launcher exits 2 with "push your branch first". The
SHA is recorded in the ledger.

1. **Dry run.** Reads from RunPod only (GETs), prints the request bodies, creates nothing. Secrets
   are printed as `***`.

   ```bash
   uv run sweep-launch --dry-run
   ```

   Output shape (run 2026-11-25; the dry run made no POST):

   ```text
   ledger: 0.00 USD spent by 0 earlier launch(es); cap 100.00
   data center EU-RO-1; GPUs to try: NVIDIA GeForce RTX 4090
   remaining=100.00 USD pod_cap=100.00 USD balance=<your balance> USD
   NVIDIA GeForce RTX 4090: 0.892 USD/h (gpu 0.89 + 20 GB volume 0.0019)
   max_hours=106.51 on NVIDIA GeForce RTX 4090
   dry run: POST /v2/network-volumes        { name, size: 20, dataCenter, type: STANDARD }
   dry run: POST /v2/pods (first GPU)       { name, image, gpu, disk, mounts, env: {...}, cmd }
   dry run: nothing created
   ```

   In the pod body, `SWEEP_RUNPOD_API_KEY` and `WANDB_API_KEY` appear as `"***"`; the other `env`
   entries are `REPO_URL`, `REPO_REF`, `SWEEP_CAP_USD`, `SWEEP_PROJECT_CAP_USD` (always
   100, the default account cap), `SWEEP_GPU_USD_PER_H`, `SWEEP_MAX_HOURS`, `SWEEP_STATE_DIR` and `WANDB_PROJECT`.

2. **Smoke run (costs money, not run for this README).** One short GPU run: calibrate the pendulum
   family and run the arm-family probes (`visited` and `buffer_scored` on `arm_sparse`), then
   run `uniform` and `visited` on `pendulum_sparse` and `uniform` on `maze_trap` for one rung.
   That rung is sized to cost half of what calibration left, counting every job without its
   own calibration probe (here `visited`, and `maze_trap`, whose family smoke does not
   calibrate) at 3x uniform's time, with r0 clamped to [100,000, 1,000,000] env steps
   (`SWEEP_SMOKE_MIN_R0_STEPS`, `SWEEP_SMOKE_R0_MAX_STEPS`). Smoke never exits 5.
   After calibration the smoke prints a `full-ladder projection` line (also in report.md,
   report.json and the W&B summary as `full_projection_*`): the full sweep's expected arms
   and USD per rung, r0, and whether a full run would exit 5, at `SWEEP_PROJECT_CAP_USD`
   (the launcher always sends 100, so a $5 smoke still answers the $100 question) less the $5 margin and the $10 calibration cap. It uses the
   throughput smoke measured (pendulum) and a fallback for every other family, so treat it
   as a rough check.

   ```bash
   uv run sweep-launch --cap 5 --smoke
   ```

3. **Full sweep (costs money, not run for this README).**

   ```bash
   uv run sweep-launch
   ```

The launcher prints the pod id, the hourly cost, `max_hours`, the W&B project, the resume command
and the volume delete command.

### Flags

| Flag | Meaning |
|---|---|
| `--gpu ID` | RunPod GPU type id. Default `NVIDIA GeForce RTX 4090`. Tried first; if it has no stock the launcher tries fallbacks (`NVIDIA RTX A5000`, `NVIDIA A40`, `NVIDIA L4`, `NVIDIA RTX A6000`, `NVIDIA L40S`, `NVIDIA RTX 6000 Ada Generation`, `NVIDIA A100-SXM4-80GB`) by ascending live price. A GPU named with `--gpu` ignores `--max-usd-per-h`. |
| `--max-usd-per-h X` | Price ceiling for automatic picks (the default GPU and the fallbacks). Default 1.20. If nothing at or under it has stock, the launcher exits 2 and lists the in-stock GPUs above it with the command that picks each. |
| `--cap X` | Account-wide cap in USD, at most 100 (default 100). |
| `--volume-id ID` | Reuse this network volume (and its data center) to resume a sweep. |
| `--ref REF` | Branch, tag or SHA the pod checks out, resolved to a commit the remote serves. Default: the SHA recorded for `--volume-id`, else `git rev-parse HEAD`. A resume with a different SHA prints a warning: `sweep-run` records the commit (`REPO_REF`) in `sweep_meta.json` on first start and refuses to resume that state dir on another commit (exit 4). |
| `--repo-url URL` | Repo the pod clones. Default: `git remote get-url origin`. |
| `--dry-run` | Print the request bodies, create nothing. |
| `--smoke` | Short GPU smoke run (`SWEEP_SMOKE=1`). Uses state dir `/workspace/sweep-smoke`. |

## Where results land

- **Weights & Biases.** Project `reset-curriculum-sweep` (your entity), group = sweep id. Per-run
  metrics, plus a sweep-level run with live metrics (spend, projected spend, jobs running and
  done, arms by status, current rung per group) and, at the end, the arm table and ranking.
- **Network volume**, mounted at `/workspace`:
  - `/workspace/sweep/` for a full sweep, `/workspace/sweep-smoke/` for a smoke run
  - `report.md` and `report.json`: arm table (mean, CI, LCB/UCB, rung, status) and final ranking.
    If the final rung was not reached, the ranking is marked "provisional: ranked at rung k".
    To rebuild both from the saved state (no W&B, while no `sweep-run` uses the dir):
    `uv run sweep-report --state-dir /workspace/sweep`
  - `sweep_metrics.jsonl`: a local copy of every live sweep metric (also holds `sink_errors`
    if W&B logging failed)
  - `bootstrap.log`, scheduler state, calibration record, and the pod's spend ledger
- **Local ledger** on the machine running `sweep-launch`: `~/.resetsweep/ledger.json`.

## Cost model and safety bounds

Cost is pod time x the pod's hourly price, plus storage for a 20 GB STANDARD network volume at
$0.07 per GB-month (from docs.runpod.io/storage/network-volumes, read 2026-11-24), about
$0.0019/h or $1.40/month. The 2026-11-25 dry run quoted the default RTX 4090 at $0.89/h.

Each of these limits is enforced in code:

- **Launcher budget check.** Before creating anything, the launcher settles its ledger from the
  RunPod billing API (every earlier launch's pod and volume spend), never below a floor of
  price x min(hours since the launch, its `max_hours`) plus volume storage while billing lags or
  reads 0 for a deleted pod. It exits 3 unless
  `cap - spent` is more than the margin ($5, or $0.50 with `--smoke`) and the account balance is at least `cap - spent`.
- **One pod at a time.** It exits 2 (dry run included) while any pod in its ledger still exists
  (any state except 404 or TERMINATED, so a stopped pod counts: it still bills its disk), and
  refuses while an earlier pod create is unresolved.
- **Price ceiling.** Automatic GPU picks cost at most `--max-usd-per-h` (default 1.20).
- **`SWEEP_MAX_HOURS` wall clock.** `max_hours = (remaining - margin) / (gpu USD/h + volume USD/h)`.
  The bootstrap counts it from the pod's first start (kept on the volume, so a restart does not
  reset it) and runs `uv sync`, the GPU tests and `sweep-run` under one deadline of
  `SWEEP_MAX_HOURS` minus a 10-minute reserve. Not covered by this bound: time before the
  bootstrap starts (image pull, clone) and volume storage while no pod runs.
- **In-pod ledger.** `sweep-run` tracks spend as wall clock x price in a ledger that survives pod
  restarts. It stops starting jobs that would pass `cap - margin`, and at `cap - margin` it kills
  running jobs and writes the report. Killed jobs are kept as in flight, so a resumed sweep reruns
  them. Calibration has its own $10 cap. It also stops 300 s before the bootstrap's deadline
  (`SWEEP_DEADLINE_EPOCH`), killing running jobs and writing the report.
- **Hang guard.** Each worker touches a heartbeat file on every metrics log. A job is killed
  only if that file has not moved for `SWEEP_HANG_S` (default 900 s) or it runs past 4x its
  estimate (a calibration probe's measured slowdown, or 3x uniform's time for an unprobed
  method), so slow arms are not dropped for being slow.
- **Self-delete trap.** The pod's command installs an EXIT trap that deletes the pod before it
  clones the repo, and the bootstrap installs its own. On any exit (success, failure, SIGTERM) the
  pod deletes itself through the RunPod API with retries, using `SWEEP_RUNPOD_API_KEY`, then
  RunPod's injected key, then `runpodctl`. A `DONE-<pod id>` marker on the volume stops a
  restarted container from redoing paid work.
- **Resume allowance.** When resuming on a volume, the pod cap is reduced by $2 per earlier launch
  on that volume (`SETUP_ALLOWANCE_USD`), an estimate for setup time that has not been measured
  on a GPU yet.

## Resume

The pod deletes itself when it exits, so a stopped sweep is resumed on a new pod that mounts the
same volume. The volume id is printed at launch.

```bash
uv run sweep-launch --volume-id <volume id>             # add --smoke if the first launch had it
```

`sweep-run` finds the saved scheduler state and continues; calibration is not repeated and the
arm plan does not change.

## Delete the network volume

The volume is kept after the pod ends and bills about $1.40/month until deleted. The launcher
prints the exact command at launch. With `RUNPOD_API_KEY` set in your shell:

```bash
curl -fsS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" \
  https://api.runpod.io/v2/network-volumes/<volume id>
```

RunPod refuses (409) a delete within five minutes of the volume's creation. A volume left behind by a failed launch is reported by the
launcher as orphaned, with the same command.

## Exit codes

`sweep-launch`:

| Code | Meaning |
|---|---|
| 0 | Launched, or dry run done |
| 1 | Bad config (missing `.env` or key, corrupt ledger, git default unreadable), volume problem, or RunPod API error |
| 2 | No pod could be created, no data center fits, an earlier pod still exists, a usage error, a non-https repo URL, or a commit the remote does not serve ("push your branch first") |
| 3 | Refused by the budget check |

`sweep-run` (on the pod): 0 finished (also when it stopped at the budget or 300 s before the
bootstrap's deadline, `SWEEP_DEADLINE_EPOCH`, having written the report); 128 + signal number
after SIGTERM/SIGINT (workers killed, state saved, report written); 2 bad settings; 3 budget already spent, or calibration
produced no usable measurement; 4 state dir made by another mode or arm set or on another commit
(`REPO_REF`, recorded in `sweep_meta.json`; skipped when unset, as in local runs), or scheduler state
without its calibration; 5 the rung-0 budget would be below `SWEEP_MIN_R0_STEPS` (the log names
the numbers and the knobs: fewer tasks, fewer arms, fewer seeds); 6 the device is the CPU on a
RunPod pod (`RUNPOD_POD_ID` set). Before the GPU tests, `scripts/pod_bootstrap.sh` runs a CUDA
gate (`torch.cuda.is_available()` and `warp.is_cuda_available()` must both be true); if it
fails, the bootstrap exits and the pod deletes itself. `scripts/pod_bootstrap.sh` exits 1 if it refuses
to start, 124 if a step hit the deadline, 137 if a step ignored TERM and was killed, otherwise
the failing step's exit code.

## Known limits

- With 3 seeds, the IQM (25% trimmed mean) drops no runs, so it equals the mean. At 5 seeds it
  keeps 3 (`src/resetsweep/sweep/stats.py`).
- The calibration measurements (steps/s per task family, the `num_envs` and worker grid, the
  per-launch setup allowance, pre-bootstrap pod time) have not been taken on a GPU yet. Rung
  sizes are therefore unknown until a GPU run reports them.
- GPU-only checks still open: the CUDA one-substep graph with per-substep callback, torch/warp
  stream ordering, CUDA generators, CPU contact checks inside arm `visited` methods, and
  host syncs in `buffer_scored`. They are covered only by `pytest -m gpu` (tests/test_gpu.py:
  G2 a GPU smoke of every method x score pair on pendulum and arm and every method on the two
  maze tasks, G3 CPU vs CUDA dynamics, G4 the worker pool on CUDA; plus G1 CUDA start states
  and G5 the CUDA gate) and the smoke run. None has run on a GPU yet.
- GPU batched MuJoCo is not reproducible run to run, so conclusions come from seeds, not single
  runs.
- The paid launches (`--smoke` and the full sweep) were not run while writing this README.
