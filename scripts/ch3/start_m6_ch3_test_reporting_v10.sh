#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo 'usage: absolute-config absolute-AUTH' >&2
    exit 2
fi
W=/public/home/yueweiting/大论文/ModernTCN-U+V
PY=/public/home/yueweiting/大论文/amd-execution-envs/m5-source-smoke-8sr2d3d_/bin/python
cd "$W"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
"$PY" -B -m m6_ch3_test_reporting_v10.entry preflight --config "$1" --auth "$2"
command -v tmux >/dev/null
printf -v command '%q ' "$PY" -B -m m6_ch3_test_reporting_v10.entry supervise --config "$1" --auth "$2"
tmux new-session -d -s m6-ch3-test-reporting-v10 "$command"
echo 'Verify runtime, owned worker identity, test access ledger and first receipt before disconnecting SSH.'
