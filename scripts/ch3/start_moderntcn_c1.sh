#!/usr/bin/env bash
set -euo pipefail
ROOT='/public/home/yueweiting/大论文/ModernTCN-C1-baseline'
PY='/public/home/yueweiting/miniconda/envs/amd/bin/python'
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=''
unset AMD_RR_CONFIG AMD_RR_CONFIG_SHA256 PYTHONPATH
case "${1:-}" in
  arm|preflight|status|stop|check|plan) exec "$PY" -B "$ROOT/moderntcn_c1_entry.py" "$1" ;;
  logs) exec tail -n 100 -f '/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-C1-round2-rerun-v1/launcher.log' ;;
  tmux) exec tmux attach-session -t m6-moderntcn-C1-round2-v1 ;;
  *) echo 'usage: start_moderntcn_c1.sh {arm|preflight|status|stop|check|plan|logs|tmux}' >&2; exit 2 ;;
esac
