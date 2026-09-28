#!/usr/bin/env bash
set -euo pipefail
CH3_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
CH3_PY='/public/home/yueweiting/大论文/amd-execution-envs/m5-source-smoke-8sr2d3d_/bin/python'
export PYTHONDONTWRITEBYTECODE=1
CH3_ACTION=${1:-dry-run}
if [[ "$CH3_ACTION" == start ]]; then
 shift
 "$CH3_PY" -B "$CH3_ROOT/m6_revision_entry.py" preflight "$@"
 command -v tmux >/dev/null
 if tmux has-session -t ch3-timemixer-fixedlr-v2 2>/dev/null; then echo 'revision session exists' >&2; exit 2; fi
 CH3_LOG='/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-formal-launch-dhozikhu/revisions/timemixer-fixedlr-v2/controller.log'
 [[ ! -e "$CH3_LOG" ]] || { echo 'retained revision log; no duplicate start' >&2; exit 2; }
 mkdir -p -- "$(dirname -- "$CH3_LOG")"
 printf -v CH3_CMD '%q ' env PYTHONDONTWRITEBYTECODE=1 "$CH3_PY" -B "$CH3_ROOT/m6_revision_entry.py" start "$@"
 printf -v CH3_LOG_Q '%q' "$CH3_LOG"
 exec tmux new-session -d -s ch3-timemixer-fixedlr-v2 -c "$CH3_ROOT" "set -C; $CH3_CMD >$CH3_LOG_Q 2>&1"
fi
exec "$CH3_PY" -B "$CH3_ROOT/m6_revision_entry.py" "$@"
