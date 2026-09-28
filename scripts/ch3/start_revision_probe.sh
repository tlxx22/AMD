#!/usr/bin/env bash
set -euo pipefail
CH3_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
CH3_PY=/public/home/yueweiting/大论文/amd-execution-envs/m5-source-smoke-8sr2d3d_/bin/python
CH3_ACTION=${1:-dry-run}
if [[ "$CH3_ACTION" == start ]]; then
 shift
 env PYTHONDONTWRITEBYTECODE=1 "$CH3_PY" -B "$CH3_ROOT/m6_revision_probe_entry.py" preflight "$@"
 CH3_SESSION=ch3-timemixer-revision-numeric
 if tmux has-session -t "$CH3_SESSION" 2>/dev/null; then echo 'retained revision numeric probe session; audit required' >&2; exit 2; fi
 CH3_BASE=/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc
 [[ ! -e "$CH3_BASE/revision-numeric-probe-v1" && ! -e "$CH3_BASE/revision-numeric-probe-controller.log" ]] || { echo 'retained probe evidence; no fresh retry' >&2; exit 2; }
 printf -v CH3_CMD '%q ' env PYTHONDONTWRITEBYTECODE=1 "$CH3_PY" -B "$CH3_ROOT/m6_revision_probe_entry.py" start "$@"
 printf -v CH3_LOG '%q' "$CH3_BASE/revision-numeric-probe-controller.log"
 exec tmux new-session -d -s "$CH3_SESSION" -c "$CH3_ROOT" "$CH3_CMD >$CH3_LOG 2>&1"
fi
exec env PYTHONDONTWRITEBYTECODE=1 "$CH3_PY" -B "$CH3_ROOT/m6_revision_probe_entry.py" "$@"
