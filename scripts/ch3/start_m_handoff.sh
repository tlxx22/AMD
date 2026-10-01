#!/usr/bin/env bash
set -euo pipefail
M_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
M_PY='/public/home/yueweiting/大论文/amd-execution-envs/m5-source-smoke-8sr2d3d_/bin/python'
M_ACTION=${1:-dry-run}
if [[ "$M_ACTION" == start ]]; then
  shift
  env PYTHONDONTWRITEBYTECODE=1 "$M_PY" -B "$M_ROOT/m6_m_handoff_entry.py" preflight "$@"
  M_SESSION='ch3-m-baselines-handoff-v1'
  if tmux has-session -t "$M_SESSION" 2>/dev/null; then echo 'retained M handoff session' >&2; exit 2; fi
  M_LOG='/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/m-handoff-launcher.log'
  (set -o noclobber; : > "$M_LOG")
  M_TOKEN=$(env PYTHONDONTWRITEBYTECODE=1 "$M_PY" -B "$M_ROOT/m6_m_handoff_entry.py" prepare-launch --wrapper-pid "$$" "$@")
  printf -v M_CMD '%q ' env PYTHONDONTWRITEBYTECODE=1 CH3_M_LAUNCH_TOKEN="$M_TOKEN" "$M_PY" -B "$M_ROOT/m6_m_handoff_entry.py" start "$@"
  printf -v M_QLOG '%q' "$M_LOG"
  exec tmux new-session -d -s "$M_SESSION" -c "$M_ROOT" "$M_CMD >$M_QLOG 2>&1"
fi
exec env PYTHONDONTWRITEBYTECODE=1 "$M_PY" -B "$M_ROOT/m6_m_handoff_entry.py" "$@"
