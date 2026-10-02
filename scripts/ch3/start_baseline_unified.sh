#!/usr/bin/env bash
set -euo pipefail
UNIFIED_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
UNIFIED_PY='/public/home/yueweiting/大论文/amd-execution-envs/m5-source-smoke-8sr2d3d_/bin/python'
UNIFIED_ACTION=${1:-dry-run}
if [[ "$UNIFIED_ACTION" == start ]]; then
  shift
  env PYTHONDONTWRITEBYTECODE=1 "$UNIFIED_PY" -B "$UNIFIED_ROOT/m6_baseline_unified_entry.py" preflight "$@"
  UNIFIED_SESSION='ch3-baseline-unified96-oc01-v1'
  if tmux has-session -t "$UNIFIED_SESSION" 2>/dev/null; then echo 'retained unified session' >&2; exit 2; fi
  UNIFIED_LOG='/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/baseline-unified96-onecycle001-v1/unified-launcher.log'
  (set -o noclobber; : > "$UNIFIED_LOG")
  UNIFIED_TOKEN=$(env PYTHONDONTWRITEBYTECODE=1 "$UNIFIED_PY" -B "$UNIFIED_ROOT/m6_baseline_unified_entry.py" prepare-launch --wrapper-pid "$$" "$@")
  printf -v UNIFIED_CMD '%q ' env PYTHONDONTWRITEBYTECODE=1 CH3_UNIFIED_LAUNCH_TOKEN="$UNIFIED_TOKEN" "$UNIFIED_PY" -B "$UNIFIED_ROOT/m6_baseline_unified_entry.py" start "$@"
  printf -v UNIFIED_QLOG '%q' "$UNIFIED_LOG"
  exec tmux new-session -d -s "$UNIFIED_SESSION" -c "$UNIFIED_ROOT" "$UNIFIED_CMD >$UNIFIED_QLOG 2>&1"
fi
exec env PYTHONDONTWRITEBYTECODE=1 "$UNIFIED_PY" -B "$UNIFIED_ROOT/m6_baseline_unified_entry.py" "${@:-dry-run}"
