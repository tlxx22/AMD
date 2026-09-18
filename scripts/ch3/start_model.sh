#!/usr/bin/env bash
set -euo pipefail
CH3_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
CH3_PY=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["execution"]["python"])' "$CH3_ROOT/configs/ch3_formal_profiles.json")
CH3_ACTION=preflight
CH3_MODEL=
CH3_PREVIOUS=
for CH3_VALUE in "$@"; do
  if [[ "$CH3_PREVIOUS" == --model ]]; then CH3_MODEL=$CH3_VALUE; fi
  case "$CH3_VALUE" in preflight|dry-run|start|status|logs|complete|safe-stop) CH3_ACTION=$CH3_VALUE;; esac
  CH3_PREVIOUS=$CH3_VALUE
done
case "$CH3_MODEL" in AMD|J|DLinear|PatchTST|iTransformer|TimeMixer|ModernTCN|TimeXer|N|S) ;; *) echo 'Exact --model required' >&2; exit 2;; esac
cd -- "$CH3_ROOT"
if [[ "$CH3_ACTION" == start ]]; then
  # Preflight is repeated by the daemon before any load or artifact creation.
  CH3_ARGS=()
  for CH3_VALUE in "$@"; do
    if [[ "$CH3_VALUE" == start ]]; then CH3_ARGS+=(preflight); else CH3_ARGS+=("$CH3_VALUE"); fi
  done
  PYTHONDONTWRITEBYTECODE=1 "$CH3_PY" "$CH3_ROOT/ch3_runner.py" "${CH3_ARGS[@]}"
  command -v tmux >/dev/null || { echo 'Verified tmux required; no untested fallback' >&2; exit 2; }
  printf -v CH3_COMMAND '%q ' env PYTHONDONTWRITEBYTECODE=1 "$CH3_PY" "$CH3_ROOT/ch3_runner.py" "$@"
  CH3_LOG=$("$CH3_PY" -c 'import json,sys,time;print(json.load(open(sys.argv[1]))["execution"]["evidence"]+"/model-"+sys.argv[2]+"-controller-"+str(time.time_ns())+".log")' "$CH3_ROOT/configs/ch3_formal_profiles.json" "$CH3_MODEL")
  printf -v CH3_LOG_QUOTED '%q' "$CH3_LOG"
  CH3_COMMAND+=" >$CH3_LOG_QUOTED 2>&1"
  exec tmux new-session -d -s "ch3-model-$CH3_MODEL" -c "$CH3_ROOT" "$CH3_COMMAND"
fi
exec env PYTHONDONTWRITEBYTECODE=1 "$CH3_PY" "$CH3_ROOT/ch3_runner.py" "$@"
