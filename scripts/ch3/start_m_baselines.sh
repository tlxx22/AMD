#!/usr/bin/env bash
set -euo pipefail
M_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
M_PY='/public/home/yueweiting/大论文/amd-execution-envs/m5-source-smoke-8sr2d3d_/bin/python'
M_ACTION=${1:-dry-run}
M_MODE=()

if [[ "$M_ACTION" == start ]]; then
  shift
  env PYTHONDONTWRITEBYTECODE=1 "$M_PY" -B "$M_ROOT/m6_m_entry.py" preflight "${M_MODE[@]}" "$@"
  M_SESSION='ch3-m-baselines-v1'
  if tmux has-session -t "$M_SESSION" 2>/dev/null; then echo 'retained M session; explicit audit required' >&2; exit 2; fi
  M_PARENT='/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-formal-launch-dhozikhu/m-tasks/m-baselines-v1/queues'
  mkdir -p "$M_PARENT"
  M_LOG="$M_PARENT/m-baselines-v1-launcher.log"
  (set -o noclobber; : > "$M_LOG")
  M_TOKEN=$(env PYTHONDONTWRITEBYTECODE=1 "$M_PY" -B "$M_ROOT/m6_m_entry.py" prepare-launch "${M_MODE[@]}" --wrapper-pid "$$" "$@")
  printf -v M_CMD '%q ' env PYTHONDONTWRITEBYTECODE=1 CH3_M_LAUNCH_TOKEN="$M_TOKEN" "$M_PY" -B "$M_ROOT/m6_m_entry.py" start "${M_MODE[@]}" "$@"
  printf -v M_QLOG '%q' "$M_LOG"
  exec tmux new-session -d -s "$M_SESSION" -c "$M_ROOT" "$M_CMD >$M_QLOG 2>&1"
fi
exec env PYTHONDONTWRITEBYTECODE=1 "$M_PY" -B "$M_ROOT/m6_m_entry.py" "$@" "${M_MODE[@]}"
