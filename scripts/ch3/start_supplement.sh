#!/usr/bin/env bash
set -euo pipefail
CH3_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
CH3_PY=/public/home/yueweiting/大论文/amd-execution-envs/m5-source-smoke-8sr2d3d_/bin/python
CH3_ACTION=${1:-dry-run}
if [[ "$CH3_ACTION" == start ]]; then
  shift
  PYTHONDONTWRITEBYTECODE=1 "$CH3_PY" -B "$CH3_ROOT/m6_extension_entry.py" preflight "$@"
  CH3_SESSION=ch3-epf4-timemixer-v1
  if tmux has-session -t "$CH3_SESSION" 2>/dev/null; then echo 'retained session; audit required' >&2; exit 2; fi
  CH3_OUT=/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-formal-launch-dhozikhu/supplements/epf4-timemixer-v1
  mkdir -p "$CH3_OUT"
  CH3_LOG="$CH3_OUT/launcher-$(date +%s%N).log"
  printf -v CH3_CMD '%q ' env PYTHONDONTWRITEBYTECODE=1 "$CH3_PY" -B "$CH3_ROOT/m6_extension_entry.py" start "$@"
  printf -v CH3_QLOG '%q' "$CH3_LOG"
  exec tmux new-session -d -s "$CH3_SESSION" -c "$CH3_ROOT" "$CH3_CMD >$CH3_QLOG 2>&1"
fi
exec env PYTHONDONTWRITEBYTECODE=1 "$CH3_PY" -B "$CH3_ROOT/m6_extension_entry.py" "$@"
