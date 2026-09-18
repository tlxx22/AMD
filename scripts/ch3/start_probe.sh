#!/usr/bin/env bash
set -euo pipefail
CH3_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
CH3_PY=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["execution"]["python"])' "$CH3_ROOT/configs/ch3_formal_profiles.json")
CH3_ACTION=${1:-preflight}
cd -- "$CH3_ROOT"
if [[ "$CH3_ACTION" == start ]]; then
  shift
  PYTHONDONTWRITEBYTECODE=1 "$CH3_PY" tools/restricted_regression/m5_formal_entry.py preflight "$@"
  command -v tmux >/dev/null || { echo 'Verified tmux required; no untested fallback' >&2; exit 2; }
  printf -v CH3_COMMAND '%q ' env PYTHONDONTWRITEBYTECODE=1 "$CH3_PY" "$CH3_ROOT/tools/restricted_regression/m5_formal_entry.py" start "$@"
  CH3_LOG=$("$CH3_PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["execution"]["evidence"]+"/probe-controller.log")' "$CH3_ROOT/configs/ch3_formal_profiles.json")
  [[ ! -e "$CH3_LOG" ]] || { echo 'Controller log exists; audit retained attempt first' >&2; exit 2; }
  printf -v CH3_LOG_QUOTED '%q' "$CH3_LOG"
  CH3_COMMAND+=" >$CH3_LOG_QUOTED 2>&1"
  exec tmux new-session -d -s ch3-resource-probe -c "$CH3_ROOT" "$CH3_COMMAND"
fi
if [[ $# == 0 ]]; then set -- preflight; fi
exec env PYTHONDONTWRITEBYTECODE=1 "$CH3_PY" tools/restricted_regression/m5_formal_entry.py "$@"
