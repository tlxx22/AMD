#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 5 ] || [ "$#" -gt 6 ]; then
  echo 'usage: bash start_timemixer_uv_v6_r4.sh CONFIG TASK_IDS REVIEWED_AUTHORIZATION RESULT_ROOT ATTEMPT [--resume]' >&2
  exit 2
fi
task_config=$1
task_ids=$2
task_authorization=$3
task_results=$4
task_attempt=$5
task_resume=${6:-}
if [[ ! "$task_attempt" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,90}$ ]]; then exit 2; fi
if [ -n "$task_resume" ] && [ "$task_resume" != '--resume' ]; then exit 2; fi
task_worktree='/public/home/yueweiting/大论文/TimeMixer-U+V-v1'
task_python='/public/home/yueweiting/miniconda/envs/amd/bin/python'
cd "$task_worktree"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
"$task_python" -B timemixer_train_v6_r4_entry.py preflight --config "$task_config" --task-list "$task_ids" --authorization "$task_authorization" --root "$task_results" --attempt "$task_attempt"
task_results=$("$task_python" -B -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$task_results")
mkdir -p "$task_results/launch-logs"
task_log="$task_results/launch-logs/$task_attempt-$(date -u +%Y%m%dT%H%M%S)-$$.log"
task_command=(env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 "$task_python" -B "$task_worktree/timemixer_train_v6_r4_entry.py" supervise --config "$task_config" --task-list "$task_ids" --authorization "$task_authorization" --root "$task_results" --attempt "$task_attempt")
if [ -n "$task_resume" ]; then task_command+=(--resume); fi
printf -v task_quoted '%q ' "${task_command[@]}"
printf -v task_log_quoted '%q' "$task_log"
task_shell="exec $task_quoted > $task_log_quoted 2>&1"
if command -v tmux >/dev/null 2>&1; then
  tmux new-session -d -s "uv6-r4-$task_attempt" -c "$task_worktree" "$task_shell"
elif command -v systemd-run >/dev/null 2>&1; then
  systemd-run --user --unit="uv6-r4-$task_attempt" --working-directory="$task_worktree" /bin/bash -c "$task_shell"
else
  nohup setsid /bin/bash -c "$task_shell" < /dev/null &
  echo "$!"
fi
echo "launch log: $task_log"
