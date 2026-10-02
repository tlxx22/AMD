#!/usr/bin/env bash
set -euo pipefail
CHAIN_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
CHAIN_PY='/public/home/yueweiting/大论文/amd-execution-envs/m5-source-smoke-8sr2d3d_/bin/python'
CHAIN_ACTION=${1:-dry-run}
if [[ "$CHAIN_ACTION" == start ]]; then
  shift
  env PYTHONDONTWRITEBYTECODE=1 "$CHAIN_PY" -B "$CHAIN_ROOT/m6_native_chain_entry.py" preflight "$@"
  CHAIN_SESSION='ch3-native-time-mark-chain-v3'
  if tmux has-session -t "$CHAIN_SESSION" 2>/dev/null; then echo 'retained native successor session' >&2; exit 2; fi
  CHAIN_LOG='/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/native-time-mark-chain-v3/native-chain-launcher.log'
  (set -o noclobber; : > "$CHAIN_LOG")
  CHAIN_TOKEN=$(env PYTHONDONTWRITEBYTECODE=1 "$CHAIN_PY" -B "$CHAIN_ROOT/m6_native_chain_entry.py" prepare-launch --wrapper-pid "$$" "$@")
  printf -v CHAIN_CMD '%q ' env PYTHONDONTWRITEBYTECODE=1 CH3_NATIVE_CHAIN_TOKEN="$CHAIN_TOKEN" "$CHAIN_PY" -B "$CHAIN_ROOT/m6_native_chain_entry.py" start "$@"
  printf -v CHAIN_QLOG '%q' "$CHAIN_LOG"
  exec tmux new-session -d -s "$CHAIN_SESSION" -c "$CHAIN_ROOT" "$CHAIN_CMD >$CHAIN_QLOG 2>&1"
fi
exec env PYTHONDONTWRITEBYTECODE=1 "$CHAIN_PY" -B "$CHAIN_ROOT/m6_native_chain_entry.py" "$@"
