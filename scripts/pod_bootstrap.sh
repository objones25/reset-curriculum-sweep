#!/usr/bin/env bash
# pod_bootstrap.sh: set up the repo inside the RunPod pod, run the GPU tests, run the sweep,
# and always delete the pod on exit (success, failure or crash) so it cannot keep billing.
#
# Started by the launcher's clone one-liner (`exec bash repo/scripts/pod_bootstrap.sh`).
# bash-target: 5.2 (Ubuntu 24.04 in runpod/pytorch:2.2.0-cu1290-torch291-ubuntu2404); the tests
# also run it on the local bash (3.2 and 5.3 on macOS).
# Needs: python3 with pip, curl, git (for the clone before it).
# Env: RUNPOD_POD_ID (set by RunPod in every pod, docs.runpod.io/pods/templates/
#      environment-variables), SWEEP_RUNPOD_API_KEY and WANDB_API_KEY (from the launcher),
#      SWEEP_STATE_DIR (default /workspace/sweep). RunPod injects its own pod-scoped
#      RUNPOD_API_KEY; this script never uses that name, only SWEEP_RUNPOD_API_KEY.
# On exit (any exit) the EXIT trap writes $SWEEP_STATE_DIR/DONE-$RUNPOD_POD_ID, then deletes the
# pod: our key, then RunPod's injected key, then runpodctl, retried with backoff until
# SWEEP_MAX_HOURS after start.
# Exit codes: 0 the sweep finished; 1 RUNPOD_POD_ID or SWEEP_RUNPOD_API_KEY is empty (nothing run);
#             otherwise the exit code of the step that failed.
set -Eeuo pipefail
shopt -s inherit_errexit 2>/dev/null || true

readonly UV_VERSION="0.13.7" # the uv that made uv.lock
readonly PODS_URL="https://api.runpod.io/v2/pods"
readonly STATE_DIR="${SWEEP_STATE_DIR:-/workspace/sweep}"
readonly LOG_FILE="${STATE_DIR}/bootstrap.log"

tee_pid=""

log() {
  local now
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ) || now="?"
  printf '[bootstrap %s] %s\n' "$now" "$*" >&2
}

# Runs a step without any RunPod key in its environment: only this script's EXIT trap needs one.
run_step() {
  log "step: $*"
  env -u SWEEP_RUNPOD_API_KEY -u RUNPOD_API_KEY "$@"
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

# Seconds after script start (bash's SECONDS) at which a failing self-delete gives up:
# SWEEP_MAX_HOURS, the budget's own bound on this pod's life (default 1 h).
deadline_s() {
  local hours=${SWEEP_MAX_HOURS:-1}
  if [[ ! $hours =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    log "SWEEP_MAX_HOURS is not a number; giving self-delete 1 h"
    hours=1
  fi
  awk -v h="$hours" 'BEGIN { printf "%d", h * 3600 }'
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
  if ! : >"${STATE_DIR}/DONE-${RUNPOD_POD_ID}"; then
    log "could not write the DONE marker in ${STATE_DIR}"
  fi
  if [[ -z ${SWEEP_RUNPOD_API_KEY:-} ]]; then
    log "SWEEP_RUNPOD_API_KEY is empty: cannot self-terminate pod ${RUNPOD_POD_ID}"
    flush_log
    exit "$status"
  fi
  log "exit status ${status}: terminating pod ${RUNPOD_POD_ID}"
  local deadline delay=30 wait_s
  # shellcheck disable=SC2310 # predicate call: its failure is handled right here
  deadline=$(deadline_s) || deadline=3600
  # Keep trying rather than exit: an exited container may be restarted. Bounded by the deadline.
  # shellcheck disable=SC2310 # predicate call: its failure is handled right here
  until try_delete_once; do
    if ((SECONDS >= deadline)); then
      log "self-delete failed and the SWEEP_MAX_HOURS deadline passed: giving up; delete pod ${RUNPOD_POD_ID} by hand"
      break
    fi
    wait_s=$((deadline - SECONDS < delay ? deadline - SECONDS : delay))
    log "self-delete failed; retrying in ${wait_s} s"
    sleep "$wait_s"
    delay=$((delay * 2 > 600 ? 600 : delay * 2))
  done
  flush_log
  exit "$status"
}

main() {
  mkdir -p -- "$STATE_DIR"
  # shellcheck disable=SC2312 # if tee dies the steps still run and the EXIT trap still fires
  exec > >(tee -a -- "$LOG_FILE") 2>&1
  tee_pid=$!
  trap terminate_pod EXIT

  if [[ -z ${RUNPOD_POD_ID:-} ]]; then
    log "RUNPOD_POD_ID is empty: refusing to start work this pod could not stop"
    exit 1
  fi
  if [[ -z ${SWEEP_RUNPOD_API_KEY:-} ]]; then
    log "SWEEP_RUNPOD_API_KEY is empty: refusing to start work this pod could not stop"
    exit 1
  fi

  local repo_dir
  repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
  cd -- "$repo_dir"
  log "pod ${RUNPOD_POD_ID}, repo ${repo_dir}, state ${STATE_DIR}"

  # Pinned uv from PyPI, never curl | sh. PIP_BREAK_SYSTEM_PACKAGES guards against images whose
  # system Python is marked externally managed (PEP 668); the pod is disposable. Unverified on
  # runpod/pytorch:2.2.0-cu1290-torch291-ubuntu2404 until the Task 25 smoke run.
  export PIP_BREAK_SYSTEM_PACKAGES=1
  export PATH="${HOME}/.local/bin:${PATH}"
  run_step python3 -m pip install "uv==${UV_VERSION}"
  run_step uv sync --frozen
  run_step uv run pytest -m gpu -q
  run_step uv run sweep-run
  log "sweep finished"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  main "$@"
fi
