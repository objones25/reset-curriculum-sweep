#!/usr/bin/env bash
# pod_bootstrap.sh: set up the repo inside the RunPod pod, run the GPU tests, run the sweep,
# and always delete the pod on exit (success, failure or crash) so it cannot keep billing.
#
# Started by the launcher's clone one-liner (`exec bash repo/scripts/pod_bootstrap.sh`).
# bash-target: 5.2 (Ubuntu 24.04 in runpod/pytorch:2.2.0-cu1290-torch291-ubuntu2404); the tests
# also run it on the local bash (3.2 and 5.3 on macOS).
# Needs: python3 with pip, curl, date, awk; GNU coreutils `timeout` if present (else a
# watchdog does its job); git ran in the clone before this script.
# Env: RUNPOD_POD_ID (set by RunPod in every pod, docs.runpod.io/pods/templates/
#      environment-variables), SWEEP_RUNPOD_API_KEY and WANDB_API_KEY (from the launcher),
#      SWEEP_MAX_HOURS (from the launcher), SWEEP_STATE_DIR (default /workspace/sweep),
#      SWEEP_DEADLINE_RESERVE_S (default 600). RunPod injects its own pod-scoped
#      RUNPOD_API_KEY; this script uses that name only as a self-delete fallback.
#
# Wall-clock bound: SWEEP_MAX_HOURS is the hard limit on this pod's life, counted from its first
# start (epoch seconds kept in $SWEEP_STATE_DIR/START-$RUNPOD_POD_ID, so a container restart
# does not reset the clock). uv sync, the GPU tests and sweep-run all run under one deadline,
# SWEEP_MAX_HOURS minus the reserve; a step still running then gets TERM, and KILL 120 s later.
# The self-delete retries stop at SWEEP_MAX_HOURS. The launcher derives SWEEP_MAX_HOURS from the
# cap minus all billed spend, so this pod's time cannot cost more than the money left.
# On exit (any exit) the EXIT trap writes $SWEEP_STATE_DIR/DONE-$RUNPOD_POD_ID, then deletes the
# pod: our key, then RunPod's injected key, then runpodctl, retried with backoff.
# Steps: install uv, uv sync, the CUDA gate (torch and warp must both see a GPU), the GPU
# tests, sweep-run.
# Exit codes: 0 the sweep finished; 1 refused to start (no RUNPOD_POD_ID or key, bad
#             SWEEP_MAX_HOURS, corrupt START file, deadline already passed); 124 a step hit the
#             deadline and ended on TERM; 137 it ignored TERM and was KILLed 120 s later
#             (GNU timeout's 128+9; the no-timeout watchdog reports 124 for both); otherwise
#             the exit code of the step that failed.
# Overrun: the deadline is checked between self-delete rounds, and one round (three curl
# attempts with --max-time 30 --retry 5, plus runpodctl) can take a few minutes, so the last
# round can end a few minutes past SWEEP_MAX_HOURS. The 120 s KILL grace adds to that too;
# the 600 s reserve is meant to absorb both.
set -Eeuo pipefail
shopt -s inherit_errexit 2>/dev/null || true

readonly UV_VERSION="0.13.7" # the uv that made uv.lock
readonly PODS_URL="https://api.runpod.io/v2/pods"
readonly STATE_DIR="${SWEEP_STATE_DIR:-/workspace/sweep}"
readonly LOG_FILE="${STATE_DIR}/bootstrap.log"
readonly RESERVE_S="${SWEEP_DEADLINE_RESERVE_S:-600}"
readonly KILL_AFTER_S=120
readonly DELETE_FALLBACK_S=3600 # self-delete retry window when no deadline could be computed
# CUDA gate (final-review I1): a pod without a working GPU would run the sweep on the CPU at GPU
# prices. Exits non-zero (so the EXIT trap deletes the pod) unless both torch and warp see CUDA.
# torch.cuda.is_available (torch/cuda/__init__.py:222, torch 2.15.0); warp.is_cuda_available
# (warp/_src/context.py:9277, get_cuda_device_count() > 0), warp-lang 1.19.0.
readonly CUDA_GATE='import sys, torch, warp; t = torch.cuda.is_available(); w = warp.is_cuda_available(); print(f"CUDA gate: torch.cuda.is_available()={t} warp.is_cuda_available()={w}"); sys.exit(0 if t and w else 3)'

tee_pid=""
start_epoch=""   # this pod's first start, epoch seconds
work_deadline="" # epoch seconds by which every step must have ended
pod_deadline=""  # epoch seconds: start + SWEEP_MAX_HOURS

log() {
  local now
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ) || now="?"
  printf '[bootstrap %s] %s\n' "$now" "$*" >&2
}

now_s() {
  date +%s
}

# SWEEP_MAX_HOURS in whole seconds; fails if it is missing or not a non-negative number.
max_hours_s() {
  local hours=${SWEEP_MAX_HOURS:-}
  [[ $hours =~ ^[0-9]+([.][0-9]+)?$ ]] || return 1
  awk -v h="$hours" 'BEGIN { printf "%d", h * 3600 }'
}

# Sets start_epoch from START-<pod id>, writing it on the first start.
load_start() {
  local file="${STATE_DIR}/START-${RUNPOD_POD_ID}" saved
  if [[ -f $file ]]; then
    saved=$(<"$file")
    if [[ ! $saved =~ ^[0-9]+$ ]]; then
      log "${file} does not hold epoch seconds; refusing to guess this pod's start"
      return 1
    fi
    start_epoch=$saved
    log "restart: this pod first started at epoch ${start_epoch}"
    return 0
  fi
  start_epoch=$(now_s)
  printf '%s\n' "$start_epoch" >"$file"
}

# Runs "$@" (without any RunPod key in its environment) with TERM at `deadline` and KILL
# KILL_AFTER_S later, in its own process group so children go too. Returns 124 on timeout, as
# GNU timeout does. Used when the image has no `timeout`.
run_with_watchdog() {
  local deadline=$1 child watchdog status=0 t
  shift
  set -m # job control: each background job gets its own process group
  env -u SWEEP_RUNPOD_API_KEY -u RUNPOD_API_KEY "$@" &
  child=$!
  (
    t=$(now_s)
    while ((t < deadline)); do
      sleep 1
      t=$(now_s)
    done
    kill -TERM -- "-${child}" 2>/dev/null || exit 0
    while ((t < deadline + KILL_AFTER_S)); do
      kill -0 -- "-${child}" 2>/dev/null || exit 0
      sleep 1
      t=$(now_s)
    done
    kill -KILL -- "-${child}" 2>/dev/null || true
  ) &
  watchdog=$!
  set +m
  wait "$child" || status=$?
  kill -TERM -- "-${watchdog}" 2>/dev/null || true
  wait "$watchdog" 2>/dev/null || true
  t=$(now_s)
  if ((status != 0 && t >= deadline)); then
    return 124
  fi
  return "$status"
}

# Runs one step before the work deadline, without any RunPod key in its environment: only
# this script's EXIT trap needs one.
run_step() {
  local remaining t
  t=$(now_s)
  remaining=$((work_deadline - t))
  if ((remaining <= 0)); then
    log "deadline reached before step: $*"
    return 124
  fi
  log "step (${remaining} s before the deadline): $*"
  if command -v timeout >/dev/null 2>&1; then
    # GNU coreutils timeout (`timeout --help`, coreutils 9.10): TERM after DURATION, KILL
    # --kill-after later; without --foreground the command's children are timed out too.
    timeout --signal=TERM --kill-after="${KILL_AFTER_S}s" "${remaining}s" \
      env -u SWEEP_RUNPOD_API_KEY -u RUNPOD_API_KEY "$@"
  else
    run_with_watchdog "$((work_deadline))" "$@"
  fi
}

flush_log() {
  # Close our ends of the tee pipe and wait for tee, so the log on the volume is complete.
  if [[ -n $tee_pid ]]; then
    exec 1>&- 2>&-
    wait "$tee_pid" 2>/dev/null || true
  fi
}

# Sends one DELETE for this pod with the key in $1. The header goes in on stdin (`-H @-`,
# curl >= 7.55) from the printf builtin, so no key ever appears in argv / ps.
delete_with_key() {
  set +x # never trace a line that carries a key
  printf 'Authorization: Bearer %s\n' "$1" \
    | curl -fsS -X DELETE --max-time 30 --retry 5 --retry-all-errors \
      -H @- "${PODS_URL}/${RUNPOD_POD_ID}"
}

# One round of self-delete attempts: our key, then RunPod's injected pod-scoped key (if set and
# different), then `runpodctl pod delete` (docs.runpod.io/runpodctl/reference/runpodctl-pod)
# if installed. Returns 0 as soon as one works.
try_delete_once() {
  # shellcheck disable=SC2310 # predicate call: its failure is handled right here
  if delete_with_key "$SWEEP_RUNPOD_API_KEY"; then return 0; fi
  log "self-delete with SWEEP_RUNPOD_API_KEY failed"
  if [[ -n ${RUNPOD_API_KEY:-} && $RUNPOD_API_KEY != "$SWEEP_RUNPOD_API_KEY" ]]; then
    # shellcheck disable=SC2310 # predicate call: its failure is handled right here
    if delete_with_key "$RUNPOD_API_KEY"; then return 0; fi
    log "self-delete with the pod's injected RUNPOD_API_KEY failed"
  fi
  if command -v runpodctl >/dev/null 2>&1; then
    if runpodctl pod delete "$RUNPOD_POD_ID"; then return 0; fi
    log "runpodctl pod delete failed"
  fi
  return 1
}

# Epoch seconds at which a failing self-delete gives up: the pod deadline when known, else
# SWEEP_MAX_HOURS (or DELETE_FALLBACK_S) from now.
delete_deadline() {
  local total t
  if [[ -n $pod_deadline ]]; then
    printf '%s\n' "$pod_deadline"
    return 0
  fi
  # shellcheck disable=SC2310 # predicate call: its failure is handled right here
  total=$(max_hours_s) || total=$DELETE_FALLBACK_S
  t=$(now_s)
  printf '%s\n' "$((t + total))"
}

terminate_pod() {
  local status=$?
  trap - EXIT
  if [[ -z ${RUNPOD_POD_ID:-} ]]; then
    log "RUNPOD_POD_ID is empty: cannot self-terminate; delete this pod in the console"
    flush_log
    exit "$status"
  fi
  # Any exit is terminal for this pod: if RunPod restarts the container, the pod command sees
  # this marker and deletes the pod instead of redoing paid work.
  if ! { : >"${STATE_DIR}/DONE-${RUNPOD_POD_ID}"; } 2>/dev/null; then
    log "could not write the DONE marker in ${STATE_DIR}"
  fi
  if [[ -z ${SWEEP_RUNPOD_API_KEY:-} ]]; then
    log "SWEEP_RUNPOD_API_KEY is empty: cannot self-terminate pod ${RUNPOD_POD_ID}"
    flush_log
    exit "$status"
  fi
  log "exit status ${status}: terminating pod ${RUNPOD_POD_ID}"
  local deadline delay=30 wait_s left t
  deadline=$(delete_deadline)
  # Keep trying rather than exit: an exited container may be restarted. Bounded by the deadline.
  # shellcheck disable=SC2310 # predicate call: its failure is handled right here
  until try_delete_once; do
    t=$(now_s)
    left=$((deadline - t))
    if ((left <= 0)); then
      log "self-delete failed and the SWEEP_MAX_HOURS deadline passed: giving up; delete pod ${RUNPOD_POD_ID} by hand"
      break
    fi
    wait_s=$((left < delay ? left : delay))
    log "self-delete failed; retrying in ${wait_s} s"
    sleep "$wait_s"
    delay=$((delay * 2 > 600 ? 600 : delay * 2))
  done
  flush_log
  exit "$status"
}

main() {
  # The trap goes first: even a failure to create the state dir must still delete the pod.
  trap terminate_pod EXIT
  mkdir -p -- "$STATE_DIR"
  # shellcheck disable=SC2312 # if tee dies the steps still run and the EXIT trap still fires
  exec > >(tee -a -- "$LOG_FILE") 2>&1
  tee_pid=$!

  if [[ -z ${RUNPOD_POD_ID:-} ]]; then
    log "RUNPOD_POD_ID is empty: refusing to start work this pod could not stop"
    exit 1
  fi
  if [[ -z ${SWEEP_RUNPOD_API_KEY:-} ]]; then
    log "SWEEP_RUNPOD_API_KEY is empty: refusing to start work this pod could not stop"
    exit 1
  fi
  if [[ ! $RESERVE_S =~ ^[0-9]+$ ]]; then
    log "SWEEP_DEADLINE_RESERVE_S must be whole seconds"
    exit 1
  fi
  local total_s
  # shellcheck disable=SC2310 # predicate call: its failure is handled right here
  if ! total_s=$(max_hours_s); then
    log "SWEEP_MAX_HOURS is missing or not a non-negative number: refusing to start an unbounded pod"
    exit 1
  fi
  # shellcheck disable=SC2310 # predicate call: its failure is handled right here
  load_start || exit 1
  pod_deadline=$((start_epoch + total_s))
  work_deadline=$((pod_deadline - RESERVE_S))
  local t
  t=$(now_s)
  if ((t >= work_deadline)); then
    log "deadline already passed (SWEEP_MAX_HOURS=${SWEEP_MAX_HOURS} from epoch ${start_epoch}, ${RESERVE_S} s reserved): refusing to start"
    exit 1
  fi

  local repo_dir
  repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
  cd -- "$repo_dir"
  log "pod ${RUNPOD_POD_ID}, repo ${repo_dir}, state ${STATE_DIR}, steps must end by epoch ${work_deadline}"

  # Pinned uv from PyPI, never curl | sh. PIP_BREAK_SYSTEM_PACKAGES guards against images whose
  # system Python is marked externally managed (PEP 668); the pod is disposable. Unverified on
  # runpod/pytorch:2.2.0-cu1290-torch291-ubuntu2404 until the Task 25 smoke run.
  export PIP_BREAK_SYSTEM_PACKAGES=1
  export PATH="${HOME}/.local/bin:${PATH}"
  run_step python3 -m pip install "uv==${UV_VERSION}"
  run_step uv sync --frozen
  run_step uv run python -c "$CUDA_GATE"
  run_step uv run pytest -m gpu -q
  run_step uv run sweep-run
  log "sweep finished"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  main "$@"
fi
