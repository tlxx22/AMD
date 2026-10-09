#!/usr/bin/env bash
set -euo pipefail
M6_THIRD_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
M6_THIRD_PY='/public/home/yueweiting/大论文/amd-execution-envs/m5-source-smoke-8sr2d3d_/bin/python'
exec env PYTHONDONTWRITEBYTECODE=1 "$M6_THIRD_PY" -B "$M6_THIRD_ROOT/m6_third_round_after_selection_entry.py" "${@:-dry-run}"
