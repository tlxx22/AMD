#!/usr/bin/env bash
set -euo pipefail
M6_RECOVERY_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
M6_RECOVERY_PY='/public/home/yueweiting/大论文/amd-execution-envs/m5-source-smoke-8sr2d3d_/bin/python'
M6_RECOVERY_ENTRY="$M6_RECOVERY_ROOT/m6_probe_schema_recovery_entry.py"
M6_RECOVERY_ACTION=${1:-dry-run}
if [[ "$M6_RECOVERY_ACTION" == start || "$M6_RECOVERY_ACTION" == arm ]]; then
  shift
  env PYTHONDONTWRITEBYTECODE=1 "$M6_RECOVERY_PY" -B "$M6_RECOVERY_ENTRY" preflight "$@"
  M6_RECOVERY_SESSION='ch3-baseline-type1-followup-v3-recovery1-probe-schema-r1'
  if tmux has-session -t "$M6_RECOVERY_SESSION" 2>/dev/null; then echo 'retained schema-recovery session' >&2; exit 2; fi
  M6_RECOVERY_LOG='/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/baseline-unified-v3-ms-seal-m128-recovery1/probe-schema-repair-v1/followup-launcher.log'
  (set -o noclobber; : > "$M6_RECOVERY_LOG")
  M6_RECOVERY_TOKEN=$(env PYTHONDONTWRITEBYTECODE=1 "$M6_RECOVERY_PY" -B "$M6_RECOVERY_ENTRY" prepare-launch --wrapper-pid "$$" "$@")
  printf -v M6_RECOVERY_CMD '%q ' env PYTHONDONTWRITEBYTECODE=1 CH3_TYPE1_LAUNCH_TOKEN="$M6_RECOVERY_TOKEN" "$M6_RECOVERY_PY" -B "$M6_RECOVERY_ENTRY" start "$@"
  printf -v M6_RECOVERY_QLOG '%q' "$M6_RECOVERY_LOG"
  exec tmux new-session -d -s "$M6_RECOVERY_SESSION" -c "$M6_RECOVERY_ROOT" "$M6_RECOVERY_CMD >$M6_RECOVERY_QLOG 2>&1"
fi
exec env PYTHONDONTWRITEBYTECODE=1 "$M6_RECOVERY_PY" -B "$M6_RECOVERY_ENTRY" "${@:-dry-run}"
