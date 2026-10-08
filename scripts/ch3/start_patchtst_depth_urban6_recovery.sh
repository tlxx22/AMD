#!/usr/bin/env bash
set -euo pipefail
M6_DEPTH_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
M6_DEPTH_PY='/public/home/yueweiting/大论文/amd-execution-envs/m5-source-smoke-8sr2d3d_/bin/python'
M6_DEPTH_ENTRY="$M6_DEPTH_ROOT/m6_patchtst_depth_urban6_recovery_entry.py"
M6_DEPTH_ACTION=${1:-dry-run}
if [[ "$M6_DEPTH_ACTION" == start || "$M6_DEPTH_ACTION" == arm ]]; then
  shift
  env PYTHONDONTWRITEBYTECODE=1 "$M6_DEPTH_PY" -B "$M6_DEPTH_ENTRY" preflight "$@"
  M6_DEPTH_SESSION='ch3-m6-patchtst-depth-urban6-r2'
  if tmux has-session -t "$M6_DEPTH_SESSION" 2>/dev/null; then echo 'retained depth/Urban6 session' >&2; exit 2; fi
  M6_DEPTH_LOG='/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/baseline-unified-v3-ms-seal-m128-recovery1/moderntcn-etth1-numeric-diagnosis-v1/baseline-numeric-admission-v1/patchtst-depth-urban6-repair-v1/patchtst-enc2-adoption-v1/patch-enc1-observation-repair-v1/followup-launcher.log'
  (set -o noclobber; : > "$M6_DEPTH_LOG")
  M6_DEPTH_TOKEN=$(env PYTHONDONTWRITEBYTECODE=1 "$M6_DEPTH_PY" -B "$M6_DEPTH_ENTRY" prepare-launch --wrapper-pid "$$" "$@")
  printf -v M6_DEPTH_CMD '%q ' env PYTHONDONTWRITEBYTECODE=1 CH3_TYPE1_LAUNCH_TOKEN="$M6_DEPTH_TOKEN" "$M6_DEPTH_PY" -B "$M6_DEPTH_ENTRY" start "$@"
  printf -v M6_DEPTH_QLOG '%q' "$M6_DEPTH_LOG"
  exec tmux new-session -d -s "$M6_DEPTH_SESSION" -c "$M6_DEPTH_ROOT" "$M6_DEPTH_CMD >$M6_DEPTH_QLOG 2>&1"
fi
exec env PYTHONDONTWRITEBYTECODE=1 "$M6_DEPTH_PY" -B "$M6_DEPTH_ENTRY" "${@:-dry-run}"
