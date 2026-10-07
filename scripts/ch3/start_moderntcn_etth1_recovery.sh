#!/usr/bin/env bash
set -euo pipefail
M6_NUMERIC_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
M6_NUMERIC_PY='/public/home/yueweiting/大论文/amd-execution-envs/m5-source-smoke-8sr2d3d_/bin/python'
M6_NUMERIC_ENTRY="$M6_NUMERIC_ROOT/m6_moderntcn_etth1_recovery_entry.py"
M6_NUMERIC_ACTION=${1:-dry-run}
if [[ "$M6_NUMERIC_ACTION" == start || "$M6_NUMERIC_ACTION" == arm ]]; then
  shift
  env PYTHONDONTWRITEBYTECODE=1 "$M6_NUMERIC_PY" -B "$M6_NUMERIC_ENTRY" preflight "$@"
  M6_NUMERIC_SESSION='ch3-m6-m128-moderntcn-etth1-numeric-r1'
  if tmux has-session -t "$M6_NUMERIC_SESSION" 2>/dev/null; then echo 'retained numeric-recovery session' >&2; exit 2; fi
  M6_NUMERIC_LOG='/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/baseline-unified-v3-ms-seal-m128-recovery1/moderntcn-etth1-numeric-diagnosis-v1/followup-launcher.log'
  (set -o noclobber; : > "$M6_NUMERIC_LOG")
  M6_NUMERIC_TOKEN=$(env PYTHONDONTWRITEBYTECODE=1 "$M6_NUMERIC_PY" -B "$M6_NUMERIC_ENTRY" prepare-launch --wrapper-pid "$$" "$@")
  printf -v M6_NUMERIC_CMD '%q ' env PYTHONDONTWRITEBYTECODE=1 CH3_TYPE1_LAUNCH_TOKEN="$M6_NUMERIC_TOKEN" "$M6_NUMERIC_PY" -B "$M6_NUMERIC_ENTRY" start "$@"
  printf -v M6_NUMERIC_QLOG '%q' "$M6_NUMERIC_LOG"
  exec tmux new-session -d -s "$M6_NUMERIC_SESSION" -c "$M6_NUMERIC_ROOT" "$M6_NUMERIC_CMD >$M6_NUMERIC_QLOG 2>&1"
fi
exec env PYTHONDONTWRITEBYTECODE=1 "$M6_NUMERIC_PY" -B "$M6_NUMERIC_ENTRY" "${@:-dry-run}"
