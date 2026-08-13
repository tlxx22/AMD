<p align="center">
  <img src="assets/logo.png" width="150" alt="AMD logo">
</p>

# Adaptive Multi-Scale Decomposition for Time Series Forecasting

This branch contains an AMD-derived, paper-close interpretation with a
corrected and reproducible experiment runner:

```text
AMD-paper-norm-wd-ddi-v1
```

It keeps the requested `U -> DDI` connection and changes three explicit
paper/code mismatches: actual LayerNorm at the flag-controlled entries, the
paper's DDI hidden-width rule, and the paper's Adam weight decay. It is not
claimed to be an exact reconstruction of every underspecified paper detail.

## Paper-close interpretation contract

- When `layernorm=True`, the MDM and DDI entry normalizers are
  `torch.nn.LayerNorm(seq_len)` applied to `[batch, channel, sequence]`. Thus
  each channel is normalized independently over its final look-back axis.
- The paper does not specify the LayerNorm axes and Algorithm 1 only explicitly
  shows LayerNorm at the DDI entry. Applying the same released switch at both
  MDM and DDI entries is this variant's documented interpretation.
- DDI's patch-internal `norm1`, and `norm2` when `alpha > 0`, remain the
  released flattened `BatchNorm1d` operators; they are not controlled by the
  `layernorm` flag.
- When channel mixing is enabled (`alpha > 0`), its hidden width is
  `max(32, 2**ceil(log2(feature_count)))`.
- Adam weight decay is fixed at the paper value `1e-7`.
- The selector produces one expert-weight vector per input window/channel and
  shares it across every forecast horizon by expanding it from one dimension
  to `T`; the paper's stated `S` shape is intentionally not reimplemented yet.
- The released dense Top-K emphasis and auxiliary-loss reduction are unchanged.
- Public dataset split boundaries and standardized-space metrics are retained.

## MDM-U-to-DDI module connection

The model path is now explicitly:

```text
X -> MDM -> U -> DDI block(s) -> AMS experts
             \-------------> AMS selector
```

Previously, the released wiring sent raw `X` into DDI while using `U` only for
the selector. In this corrected tree, the first DDI block receives `U`; later
DDI blocks remain sequential, and the selector continues to receive the same
`U`.

Artifacts and checkpoints labeled `AMD-public-code-fixed-v1` or
`AMD-mdm-u-to-ddi-v1` are different model variants. They remain stored
separately and cannot be resumed as `AMD-paper-norm-wd-ddi-v1` runs.

## What the reproducible runner fixes

- `False` is now parsed as false for `--norm` and `--layernorm`. In particular,
  ETTh2 runs with `norm=False, layernorm=False` as written in the public script.
- Solar-Energy is read as a headerless `52560 x 137` matrix, so its first row is
  no longer discarded.
- Validation is deterministic and keeps its final partial batch.
- Validation/test MSE and MAE are accumulated globally by element count rather
  than averaging batch means.
- The selected best model changes only on strict validation-MSE improvement;
  the committed `last.pt` also embeds that best snapshot, and final testing
  reloads the derived `best.pt` from disk.
- Every run has a collision-free artifact directory with resolved configuration,
  data/source fingerprints, preprocessing state, history, metrics, and resumable
  optimizer/RNG/DataLoader state.
- Per-run locking rejects concurrent writers. `last.pt` is the single epoch
  commit point, so `best.pt`, history, and manifest can be reconciled after an
  interruption without mixing epochs.
- Invalid model/data shapes fail early with actionable errors.

## Environment

Use the single-manager Conda environment. To replace an existing `amd`
environment, deactivate it first and recreate it from the lock-style file. Do
not run `pip install -r requirements.txt` and then install PyTorch/NumPy again
with Conda; overlapping binary ownership can recreate the NumPy/Pandas ABI
failure.

```bash
conda deactivate
conda env remove --name amd
conda env create --file environment.yml
conda activate amd
```

The supported stack is Python 3.11, NumPy 1.24.3, Pandas 2.0.3, SciPy 1.11.4,
scikit-learn 1.3.2, PyTorch 2.0.1, and PyTorch CUDA 11.8.

## Data

Place the benchmark files under `data/`:

```text
data/
├── electricity.csv
├── exchange_rate.csv
├── ETTh1.csv
├── ETTh2.csv
├── ETTm1.csv
├── ETTm2.csv
├── solar_AL.txt
├── traffic.csv
└── weather.csv
```

`--dataset_id` is not only a display label: it selects the reader and public
split policy, independently of the filename. Matching is case-insensitive:

- `ETTm*`: fixed endpoints `34560 / 46080 / 57600`.
- `ETTh*`: fixed endpoints `8640 / 11520 / 14400`.
- `PEMS*`: 60% train, 20% validation, 20% test. The input must be an `.npz`
  archive with a `data` array shaped `[time, feature, channel]`; this variant preserves
  the public reader's channel-0 selection.
- `Solar*`: 70% train, 10% validation, 20% test and headerless CSV/TXT input.
- All other IDs: the same 70%/10%/20% generic split.

Validation and test windows receive `seq_len` points of preceding context, but
their targets stay strictly inside their own split. `StandardScaler` is fitted
on the training split only. The formal scripts use multivariate `M` mode. For
direct `main.py` runs, `S` selects only `--target`, while `MS` uses all inputs
and predicts only that target. Headerless Solar/PEMS integer columns can be
addressed with a string such as `--target 0`. There is no bundled PEMS script.

## Running experiments

Each script is path-independent and defaults to seed 2024. From Git Bash:

```bash
bash scripts/ETTh1.sh
```

Supply the fixed five-seed matrix without editing scripts:

```bash
SEEDS="2024 2025 2026 2027 2028" bash scripts/ETTh1.sh
```

The scripts run horizons `96, 192, 336, 720` and fail immediately if any run
fails. Formal scripts explicitly require `cuda:0`. For a CPU diagnostic, call
`python main.py ... --device cpu` directly; the shell scripts intentionally do
not override their formal CUDA setting.

A successfully completed run is isolated as:

```text
artifacts/AMD-paper-norm-wd-ddi-v1/
└── <dataset>/sl<seq>_pl<pred>/seed<seed>/<timestamp>-<uuid>/
    ├── manifest.json
    ├── config.resolved.json
    ├── best.pt
    ├── last.pt
    ├── history.jsonl
    └── metrics.json
```

The persistent `.run.lock` file may also be present; only its operating-system
lock state matters. A failed run can contain a subset of the files above. It is
resumable only after at least one epoch has atomically committed `last.pt`;
otherwise start a new run. Resume granularity is one complete epoch, not a
partially processed batch.

Existing legacy files under `checkpoints/` are not read by this runner and must
not be mixed into this variant's summaries.

To resume an interrupted run, repeat its original scientific arguments, set the
target total epoch count, and add its directory:

```bash
python main.py <original arguments> \
  --train_epochs 20 \
  --resume artifacts/AMD-paper-norm-wd-ddi-v1/.../<run_id>
```

Variant, source, runtime, data, preprocessing, and scientific configuration
hashes must match. Completed runs are immutable and cannot be resumed.

## Summaries

Generate run-level and seed-aggregate CSVs using only completed, internally
consistent artifacts from this variant:

```bash
python summarize_results.py
```

This writes:

```text
summaries/AMD-paper-norm-wd-ddi-v1.csv
summaries/AMD-paper-norm-wd-ddi-v1-aggregate.csv
```

Aggregate standard deviation is the sample standard deviation (`ddof=1`). If
multiple completed runs exist for the same scientific configuration and seed,
the summarizer refuses to choose one automatically.

## Verification

Run the regression suite with no extra test dependency:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

The tests lock the documented LayerNorm axes, DDI hidden rule, retained
patch-internal BatchNorm and horizon-shared selector, the `U -> DDI`
connection, dataset/window policies, strict boolean parsing, global metrics,
checkpoint selection, deterministic resume equivalence, script paths, and
result aggregation.

## Citation

```bibtex
@inproceedings{hu2025adaptive,
  title={Adaptive Multi-Scale Decomposition Framework for Time Series Forecasting},
  author={Hu, Yifan and Liu, Peiyuan and Zhu, Peng and Cheng, Dawei and Dai, Tao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```
