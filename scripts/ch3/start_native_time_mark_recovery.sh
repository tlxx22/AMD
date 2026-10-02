#!/usr/bin/env bash
set -euo pipefail
RECOVERY_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
RECOVERY_PY='/public/home/yueweiting/大论文/amd-execution-envs/m5-source-smoke-8sr2d3d_/bin/python'
RECOVERY_ACTION=${1:-dry-run}
if [[ "$RECOVERY_ACTION" == start ]]; then
  shift
  env PYTHONDONTWRITEBYTECODE=1 "$RECOVERY_PY" -B "$RECOVERY_ROOT/m6_native_recovery_entry.py" recovery-preflight "$@"
  RECOVERY_SESSION='ch3-native-tmark-chain-v4-recovery1'
  if tmux has-session -t "$RECOVERY_SESSION" 2>/dev/null; then echo 'retained recovery session' >&2; exit 2; fi
  RECOVERY_LOG='/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/native-time-mark-chain-v4-recovery1/recovery-launcher.log'
  (set -o noclobber; : > "$RECOVERY_LOG")
  RECOVERY_TOKEN=$(env PYTHONDONTWRITEBYTECODE=1 "$RECOVERY_PY" -B "$RECOVERY_ROOT/m6_native_recovery_entry.py" prepare-launch --wrapper-pid "$$" "$@")
  printf -v RECOVERY_CMD '%q ' env PYTHONDONTWRITEBYTECODE=1 CH3_NATIVE_RECOVERY_TOKEN="$RECOVERY_TOKEN" "$RECOVERY_PY" -B "$RECOVERY_ROOT/m6_native_recovery_entry.py" start "$@"
  printf -v RECOVERY_QLOG '%q' "$RECOVERY_LOG"
  exec tmux new-session -d -s "$RECOVERY_SESSION" -c "$RECOVERY_ROOT" "$RECOVERY_CMD >$RECOVERY_QLOG 2>&1"
fi
exec env PYTHONDONTWRITEBYTECODE=1 "$RECOVERY_PY" -B "$RECOVERY_ROOT/m6_native_recovery_entry.py" "$@"
