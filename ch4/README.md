# M7 implementation and comparison preparation

This package implements the approved full-v10 spatial extension and 18-model,
12-task comparison. It is **not admitted for execution yet**. The authoritative
contract is `docs/AMD_EV_Thesis_Final_Implementation_Plan_v2.1.md`; actual progress
and open decisions are in the sole M7 milestone.

`comparison-plan.json` contains every task, source identity, separated structure
and training settings, legal-window counts, and proposed update ceilings. It is
a preparation configuration, not AUTH. Missing artifacts and resource admission
are explicit blockers. The seven temporal structures remain bound to adopted
CH3 UrbanEV MS profiles; author-default fallback applies only to the nine new
spatial baselines. PatchSTG is excluded; its downloaded
source and early vendor copy are retained only as historical material.

`dependencies.py` binds local imports, package initializers and deferred imports
conservatively, with explicit roots for dynamically loaded author models.
Runtime assets and provenance-only source snapshots are separate. The small
`runtime_io.py` helpers avoid importing a historical launcher just for PID/start.
`pattern_registry.json` explicitly names the fixed-graph and adopted pattern
identities; a missing successful pattern identity always blocks PDFormer.

The source-native input adapters are in `baselines.py`. Calendar fields are
derived from timestamps and inserted in each native field position. TESTAM
keeps 12 internal slots and selects the same approved slot in predictions,
experts, routing, losses and masks. Models receive all approved history fields;
native channel independence is not changed to manufacture auxiliary influence.

`prepare_native_graphs.py` creates independently identified fixed relations.
CHARGED uses the approved full-matrix Gaussian scale and 0.1 threshold. GWN
row-normalized supports and D2 transposed supports are separate files; STDN
non-PEMS adjacency excludes self loops. PDFormer receives weights once, plus
its own train-only DTW and pattern cache. Main-model nearest/cosine relations
are separate. Graph diagnostics never authorize threshold tuning or node removal.

All domains use Adam and OneCycle. UrbanEV/CHARGED use cap50/patience20; PEMS
cap200/patience40. `protocol.py` supplies the common settings, `optimization.py`
keeps method-specific auxiliaries separate. `state.py` preserves full optimizer,
schedule, RNG, best, patience, epoch and update state. Early stopping never
resizes the OneCycle schedule. `prepare_data.py` exposes only train/validation
preparation on its CLI. The private final-data path is called by the queue only
after all training receipts and the complete best manifest are frozen.

CPU acceptance uses the isolated interpreter:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  /public/home/yueweiting/大论文/amd-execution-envs/m7-v10-spatial-r1/bin/python -B \
  -m unittest tests.test_ch4_core tests.test_ch4_amendments tests.test_ch4_execution
```

The command documents the checks; do not mechanically rerun passed checks.
Per-baseline synthetic checks are selected with `-m tests.ch4_baseline_case MODEL`.
They never use real task windows, third-chapter checkpoints or GPUs. Historical
failures and subsequent targeted results are retained in the external evidence
scope's `checks/` directory.

Concurrency admission uses measured GPU memory/headroom, OOM risk, step and
end-to-end throughput, GPU busy time and resource contention only. The finite
5 preparation + 20 measured updates and 4 GPU-hour cap remain; q8 requires
measured headroom and a total-throughput benefit. No serial/concurrent parameter,
gradient, Adam, step-loss or prediction equivalence comparison is performed,
including renamed smoke/warmup/stability equivalents. Record its status as
`not_executed_by_user_policy`, never a newly measured equivalence Passed.
`protocol.resource_admission_policy()` is serialized into the comparison plan;
frozen resource decisions retain this status and disable persistent step-state
dumps. Do not produce numeric-full-step-*.bin or equivalent full step trajectories.
Normal finite/OOM checks, owner identities, best/last recovery checkpoints,
data/test boundaries and artifact verification remain unchanged. Freeze actual
batch, concurrency, device, threads and task order; concurrency-only OOM reduces
q first, while singleton OOM follows the approved domain-wide batch sequence.

After engineering review, finite resource probes, final batch/concurrency
freeze, precise Git closure and actual AUTH binding, the single future launch
interface is:

```bash
cd /public/home/yueweiting/大论文/CH4-v10-spatial
tmux new-session -d -s M7-comparison \
  'PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=<reviewed-visible-device> <bound-python> -B -m ch4.queue start --config <frozen-config> --auth <actual-reviewed-AUTH> --folder <new-M7-attempt> > <new-supervisor-log> 2>&1'
```

This is an interface description with deliberately unresolved bindings, **not
the executable handoff command**. No future commit, AUTH, GPU UUID, concurrent
waves or attempt is fabricated. The executable one-line launch is delivered
only after those real identities exist. Initial preflight checks actual remote
Git once; background workers use bound local bytes and owned parent identity.

For an actual owned attempt, `python -B -m ch4.queue status --folder ATTEMPT`
reports receipts/completion/failure/STOP. Logs are `ATTEMPT/logs`, PID/start and
session are `owner.json`, `runtime.json` and wave child records. Completion is
`complete.json` plus matching `manifest.json`/checksums and all 216 train + 216
test receipts. The safe stop interface is
`python -B -m ch4.queue stop --folder ATTEMPT`; it validates PID/start/runtime
ownership and writes only that attempt's STOP. It does not kill other processes.

Any technical failure halts later scheduling and preserves its partial work.
There is no automatic retry. An existing test claim always blocks reevaluation,
including ambiguous interrupted evaluation. Resume helpers do not themselves
authorize resumption; reviewed recovery must retain checkpoint progress and
separately account for failed partial work.

Sealing closes all summary files, verifies task/receipt/claim/best/metric sets,
checks every payload in Python, runs system `sha256sum --strict -c`, and only
then atomically publishes `complete.json`. The manifest lists payload only;
checksums adds the manifest; completion binds both controls. No control file
hashes itself. A failed check preserves files and publishes no success marker.

Native CPU pattern preparation is separate from the 216-task training queue.
`prepare_native_patterns` accepts a new output directory, a fixed-graph identity,
and an optional audited DTW reuse reference. `pattern_batch` is a finite CPU-only
worklist with PID/start, logs, STOP and per-domain receipts; it neither changes
the registered adoption identities during a run nor starts model training.
Any long preparation launch is delivered with actual paths and bindings after
the user-last-start boundary; an unfinished/failed fit is never adopted.

Final outputs retain all models/cities/tasks, train-standardized MSE/MAE,
separately labeled physical-unit MSE/MAE, Urban four-H equal-weight results,
PEMS positions, predictions/truth/masks/timestamps/node IDs and source/cost
ledgers. There is no winner selection or additional accuracy threshold.
