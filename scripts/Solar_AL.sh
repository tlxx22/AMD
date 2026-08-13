#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

data_path_name="solar_AL.txt"
model_id_name="solar_AL"
data_name="solar_AL"
seq_len=512
artifact_root="${PROJECT_ROOT}/artifacts"
python_bin="${PYTHON_BIN:-python}"
read -r -a seeds <<< "${SEEDS:-2024}"

for seed in "${seeds[@]}"; do
  for pred_len in 96 192 336 720; do
    "${python_bin}" -u "${PROJECT_ROOT}/main.py" \
    --implementation_variant "AMD-mdm-u-to-ddi-v1" \
    --seed "${seed}" \
    --dataset_id "${data_name}" \
    --data "${PROJECT_ROOT}/data/${data_path_name}" \
    --feature_type M \
    --target OT \
    --artifact_root "${artifact_root}" \
    --name "${model_id_name}" \
    --device "cuda:0" \
    --seq_len "${seq_len}" \
    --pred_len "${pred_len}" \
    --n_block 1 \
    --alpha 1.0 \
    --mix_layer_num 3 \
    --mix_layer_scale 2 \
    --patch 8 \
    --norm True \
    --layernorm True \
    --dropout 0.1 \
    --train_epochs 10 \
    --batch_size 128 \
    --learning_rate 0.00002
  done
done
