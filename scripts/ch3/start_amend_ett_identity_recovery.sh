#!/usr/bin/env bash
set -euo pipefail
M6_ETT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
M6_ETT_PY='/public/home/yueweiting/大论文/amd-execution-envs/m5-source-smoke-8sr2d3d_/bin/python'
M6_ETT_ENTRY="$M6_ETT_ROOT/m6_amend_ett_identity_recovery_entry.py"
M6_ETT_ACTION=${1:-dry-run}
if [[ "$M6_ETT_ACTION" == start || "$M6_ETT_ACTION" == arm ]]; then
  shift
  env PYTHONDONTWRITEBYTECODE=1 "$M6_ETT_PY" -B "$M6_ETT_ENTRY" preflight "$@"
  M6_ETT_SESSION='ch3-m6-amend-ett-identity-r1'
  if tmux has-session -t "$M6_ETT_SESSION" 2>/dev/null; then echo 'retained ETT-identity recovery session' >&2; exit 2; fi
  M6_ETT_LOG='/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/baseline-unified-v3-ms-seal-m128-recovery1/moderntcn-etth1-numeric-diagnosis-v1/baseline-numeric-admission-v1/amend-ett-identity-repair-v1/followup-launcher.log'
  (set -o noclobber; : > "$M6_ETT_LOG")
  M6_ETT_TOKEN=$(env PYTHONDONTWRITEBYTECODE=1 "$M6_ETT_PY" -B "$M6_ETT_ENTRY" prepare-launch --wrapper-pid "$$" "$@")
  printf -v M6_ETT_CMD '%q ' env PYTHONDONTWRITEBYTECODE=1 CH3_TYPE1_LAUNCH_TOKEN="$M6_ETT_TOKEN" "$M6_ETT_PY" -B "$M6_ETT_ENTRY" start "$@"
  printf -v M6_ETT_QLOG '%q' "$M6_ETT_LOG"
  exec tmux new-session -d -s "$M6_ETT_SESSION" -c "$M6_ETT_ROOT" "$M6_ETT_CMD >$M6_ETT_QLOG 2>&1"
fi
exec env PYTHONDONTWRITEBYTECODE=1 "$M6_ETT_PY" -B "$M6_ETT_ENTRY" "${@:-dry-run}"
