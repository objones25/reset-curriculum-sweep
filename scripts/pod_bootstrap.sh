#!/usr/bin/env bash
# pod_bootstrap.sh: set up the repo inside the RunPod pod, run the GPU tests, run the sweep,
# and always delete the pod on exit (success, failure or crash) so it cannot keep billing.
#
# Started by the launcher's clone one-liner (`exec bash repo/scripts/pod_bootstrap.sh`).
# bash-target: 5.2 (Ubuntu 24.04 in runpod/pytorch:2.2.0-cu1290-torch291-ubuntu2404); the tests
# also run it on the local bash (3.2 and 5.3 on macOS).
# Needs: python3 with pip, curl, git (for the clone before it).
# Env: RUNPOD_POD_ID (set by RunPod in every pod, docs.runpod.io/pods/templates/
#      environment-variables), RUNPOD_API_KEY and WANDB_API_KEY (from the launcher),
#      SWEEP_STATE_DIR (default /workspace/sweep).
# Exit codes: 0 the sweep finished; 1 RUNPOD_POD_ID or RUNPOD_API_KEY is empty (nothing run);
#             otherwise the exit code of the step that failed.
set -Eeuo pipefail
shopt -s inherit_errexit 2>/dev/null || true

readonly UV_VERSION="0.13.10"
readonly PODS_URL="https://api.runpod.io/v2/pods"
readonly STATE_DIR="${SWEEP_STATE_DIR:-/workspace/sweep}"
readonly LOG_FILE="${STATE_DIR}/bootstrap.log"

tee_pid=""

log() {
  local now
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ) || now="?"
  printf '[bootstrap %s] %s\n' "$now" "$*" >&2
}

# Runs a step without the RunPod key in its environment: only this script's EXIT trap needs it.
run_step() {
  log "step: $*"
  env -u RUNPOD_API_KEY "$@"
}

flush_log() {
  # Close our ends of the tee pipe and wait for tee, so the log on the volume is complete.
  if [[ -n $tee_pid ]]; then
    exec 1>&- 2>&-
    wait "$tee_pid" 2>/dev/null || true
  fi
}

terminate_pod() {
  local status=$?
  trap - EXIT
  if [[ -z ${RUNPOD_POD_ID:-} ]]; then
    log "RUNPOD_POD_ID is empty: cannot self-terminate; delete this pod in the console"
  elif [[ -z ${RUNPOD_API_KEY:-} ]]; then
    log "RUNPOD_API_KEY is empty: cannot self-terminate pod ${RUNPOD_POD_ID}"
  else
    log "exit status ${status}: terminating pod ${RUNPOD_POD_ID}"
    set +x # never trace the line that carries the key
    # The header goes in on stdin (`-H @-`, curl >= 7.55), so the key is never in argv / ps.
    if ! printf 'Authorization: Bearer %s\n' "$RUNPOD_API_KEY" \
      | curl -fsS -X DELETE --max-time 30 --retry 5 --retry-all-errors \
        -H @- "${PODS_URL}/${RUNPOD_POD_ID}"; then
      log "self-terminate failed: delete pod ${RUNPOD_POD_ID} by hand"
    fi
  fi
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
  if [[ -z ${RUNPOD_API_KEY:-} ]]; then
    log "RUNPOD_API_KEY is empty: refusing to start work this pod could not stop"
    exit 1
  fi

  local repo_dir
  repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
  cd -- "$repo_dir"
  log "pod ${RUNPOD_POD_ID}, repo ${repo_dir}, state ${STATE_DIR}"

  # Pinned uv from PyPI, never curl | sh. The base image's system Python may be marked
  # externally managed (PEP 668); this pod is disposable, so allow the system install.
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
