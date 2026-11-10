#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: $0 absolute-config absolute-AUTH" >&2
    exit 2
fi
W=/public/home/yueweiting/大论文/ModernTCN-U+V
PY=/public/home/yueweiting/大论文/amd-execution-envs/m5-source-smoke-8sr2d3d_/bin/python
cd "$W"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
"$PY" -m m6_ch3_supplement_v10.entry preflight --config "$1" --auth "$2"
command -v tmux >/dev/null
printf -v command '%q ' "$PY" -m m6_ch3_supplement_v10.entry supervise --config "$1" --auth "$2"
tmux new-session -d -s m6-ch3-supplement-v10 "$command"
echo 'tmux created; verify runtime/worker PID-start and logs before disconnecting SSH.'
