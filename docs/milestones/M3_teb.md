# M3：TEB 目标—外生桥接

状态：Closed
日期：2026-08-24（UTC）
canonical 内部版本：v2.1-R1
implementation gate：Passed
implementation review：Passed
Git closure：Authorized by user

## 1. 阶段范围

本里程碑实现 Global Target-Conditioned Residual TEB，并将它接入 AMDEnhanced、正式增强 runner、可追溯配置、checkpoint/resume 和 artifact 合同。数据流固定为：

~~~text
RevIN -> MDM -> DDI -> PMCR -> TEB -> AMS -> Forecast

AMS experts  <- TEB 后的 v_final
AMS selector <- 原始 u_mdm
~~~

本轮没有修改冻结 AMD，没有运行正式性能实验，没有实现 StateAdapter、H_time、graph mode 或任何空间模块，也没有进入 M4。

## 2. 继承 Git 状态

~~~text
repository:
/public/home/yueweiting/大论文/AMD

branch:
AMD-paper-repro-custom-modules-v1

implementation start HEAD:
ac4f2c2a98f8a5c4c3cafe48b36b9c2e68451b75

parent:
f906a969f6fbd250f1d4520ec4db8be0d4f3d0df

local tracking ref at implementation:
ac4f2c2a98f8a5c4c3cafe48b36b9c2e68451b75

ahead/behind:
0/0

immutable baseline tag:
amd_reproduced_baseline_v1
-> fa9665627e6fcfb1d0c2bc22d943ca9666304fd6
~~~

本轮未 checkout、reset、stash、clean、stage、commit、push 或移动 tag。本文档不写入尚未存在的自身 commit SHA。

## 3. Canonical 更新

~~~text
before SHA-256:
79b9bebec6591f9334d174900e2a2e8b799b7e83aad9760029ce8a8cab9ff45d

after SHA-256:
d5d2c3b2654e335338aca45d63832b8fe53848d407707a53e701d5c13180dbe5
~~~

内部版本保持 v2.1-R1。更新仅消歧 M3/TEB：

- 固定 Global Target-Conditioned Residual TEB v1；
- Query 固定为 Linear(T,d) 后接 LayerNorm(d)，不做 mean pooling；
- 外生历史来自 RevIN 后的 x_norm，所有变量共享 projector；
- 固定 single-target 与 parallel 的张量流、对角 mask、空 aux 和 C=1 策略；
- 固定共享可学习 scalar gamma_teb，初始化 1e-3；
- 固定 d 等于 teb_context_dim，不增加 teb_hidden_dim；
- 固定 checkpoint source-kind、正式 runner、target selection 和 artifact 合同；
- 消歧 U0-U4、F0 与 F1-F4 的公平对照范围；
- Patch-conditioned TEB 仅登记为 M5 验收失败且用户重新确认后的预留方向。

第三章、第四章总体路线、M1 数据合同、PMCR 方法、StateAdapter 和空间模块定义未改。

## 4. TimeXer 来源边界

服务器现有独立参考仓库：

~~~text
path:
/public/home/yueweiting/大论文/TimeXer

remote:
https://github.com/thuml/TimeXer.git

branch:
main

commit:
76011909357972bd55a27adba2e1be994d81b327

status:
clean
~~~

论文：

~~~text
/public/home/yueweiting/大论文/paper/NeurIPS-2024-timexer-empowering-transformers-for-time-series-forecasting-with-exogenous-variables-Paper-Conference.pdf
SHA-256:
c2ec27241da87e0559c4f797f5a23d20f725c215c1decb8f007afeb5bfd85964
~~~

实现只借鉴 TimeXer 的目标—外生交互思想，没有复制源码，也没有引入 patchify、多层 Transformer、自注意力主干或 TimeXer prediction head。

## 5. Global TEB v1 合同

模块：models/modules/target_exogenous_bridge.py

固定成员：

~~~text
query_projection:
Linear(T,d,bias=True)

query_norm:
LayerNorm(d,eps=1e-5)

exogenous_projection:
shared Linear(T,d,bias=True)

exogenous_norm:
LayerNorm(d,eps=1e-5)

cross_attention:
MultiheadAttention(
    embed_dim=d,
    num_heads=teb_heads,
    dropout=teb_dropout,
    bias=True,
    batch_first=True,
)

output_projection:
Linear(d,T,bias=True)

gamma_teb:
shared unconstrained scalar nn.Parameter, init=1e-3
~~~

第一版不包含 variable identity embedding、q residual、attention 后 FFN、额外 output dropout、自注意力、多层 encoder、未来真实外生输入或 Patch-conditioned TEB。

## 6. Single-target 张量流

~~~text
hidden [B,C,T]
normalized_input [B,T,C]

H_y = hidden[:,target_idx,:]
q = LN(Linear(T,d)(H_y)).unsqueeze(1)
# [B,1,d]

X_aux = normalized_input[:,:,aux_idx].transpose(1,2)
# [B,m,T]

E_aux = LN(shared Linear(T,d)(X_aux))
# [B,m,d]

c_exo = MHA(q,E_aux,E_aux)
# [B,1,d]

delta_y = Linear(d,T)(c_exo.squeeze(1))
# [B,T]

hidden_out[:,target_idx,:] =
    H_y + gamma_teb * delta_y

exo_context = c_exo.squeeze(1)
# [B,d]
~~~

只有目标通道被更新；其他通道逐元素不变。aux_idx 是显式有序 tuple，保留调用方顺序，并拒绝 bool、重复、越界和 target_idx。

## 7. Parallel 张量流

~~~text
Q = LN(Linear(T,d)(hidden))
# [B,C,d]

E = LN(shared Linear(T,d)(normalized_input.transpose(1,2)))
# [B,C,d]

mask = eye(C,dtype=bool)
# [C,C], True 表示禁止 self-attention

context_all = one vectorized MHA(Q,E,E,attn_mask=mask)
# [B,C,d]

delta_all = Linear(d,T)(context_all)
hidden_out = hidden + gamma_teb * delta_all
# [B,C,T]

exo_context = context_all[:,target_idx,:]
# [B,d]
~~~

所有变量均参与更新和预测。target_idx 仅作为 state_source 和可选分析锚点，不限制 parallel 输出。

## 8. 空 aux、C=1 与 F0

~~~text
TEB off + aux_idx empty:
合法严格旁路

target_exogenous + TEB on + aux_idx empty:
ValueError("TEB requires at least one auxiliary variable.")

parallel_multivariate + TEB on + C=1:
ValueError("Parallel TEB requires at least two variables.")

parallel_multivariate + TEB off + C=1:
合法严格旁路
~~~

F0 只有 volume，target_idx=0、aux_idx=()；只用于 AMD-TargetOnly 和可选 AMD-TargetOnly + PMCR。F0 不得标记为 AMD-Concat、TEB 或完整 EL-AMD，F0 上 EL-AMD 为 N/A。U1-U4 固定使用同一套非空 F1-F4 schema。

## 9. AMDEnhanced 接入与 state_source

AMDEnhanced 保持 legacy M0 路径，并为正式模式增加 task_mode、aux_idx 和 TEB 配置。use_teb=False 时：

~~~text
self.teb = None
state_dict 无 teb.* keys
forward 不执行 attention/dropout
exo_context = v_local.new_zeros([B,teb_context_dim])
~~~

真实 forward：

~~~text
x -> RevIN norm -> x_norm
x_norm.transpose -> MDM -> u_mdm
u_mdm -> all DDI blocks -> v
PMCR? -> v_local
TEB? -> v_final, exo_context
AMS(v_final,u_mdm) -> pred_all_norm
full-channel RevIN denorm -> task selection
~~~

正式 target_exogenous 输出 [B,H,1]；parallel_multivariate 输出 [B,H,C]。target_idx 是唯一目标索引，正式 runner 固定 target_slice=None。

状态源：

~~~text
state_source = concat(
    v_final[:,target_idx,:],
    u_mdm[:,target_idx,:],
    exo_context,
)

shape = [B,2*T+teb_context_dim]
~~~

return_state_source 只增加返回值，不改变 prediction 或 MoE loss。

## 10. Checkpoint source-kind 与 resume

结构迁移在写入参数前验证完整 key set 和 tensor shape：

| 来源 | 目标 | 唯一允许缺失 |
|---|---|---|
| baseline | PMCR off / TEB off | 空，strict=True |
| baseline | PMCR on / TEB off | 全部 pmcr.* |
| baseline | PMCR off / TEB on | 全部 teb.* |
| baseline | PMCR on / TEB on | 全部 pmcr.* 与 teb.* |
| PMCR-only | PMCR on / TEB on | 全部 teb.* |
| TEB-only | PMCR on / TEB on | 全部 pmcr.* |
| 完整同结构 | 完整同结构 | 空，strict=True |

source_kind 必须显式为 baseline、pmcr_only 或 teb_only，不根据缺失 key 猜测。不允许部分 enhancement key、unexpected key、非 enhancement missing key或全局 silent strict=False。

Resume 不使用结构迁移 importer，只允许完全同结构普通 load_state_dict(strict=True)。variant、task mode、target/aux/schema、PMCR/TEB 全部结构参数、seq_len 和 target selection policy 均进入 scientific config hash。

## 11. Runner、config 与 artifact

模型选择：

~~~text
AMD-paper-norm-wd-ddi-v1 -> frozen AMD
el-amd-pmcr-teb-v1      -> AMDEnhanced
~~~

同一增强 variant 通过 ablation_id 表达 U0-U4、target_only_pmcr 和 parallel M0-M3。矛盾的 PMCR/TEB/aux/task 配置在模型或 artifact 创建前拒绝。

正式增强路径：

~~~text
artifacts/<variant>/<dataset>/<task_mode>/<target>/
  horizon_<h>/fold_<fold>/seed_<seed>/<run_id>/
~~~

原生生成：

~~~text
manifest.json
config.resolved.json
history.jsonl
metrics.json
best.pt
last.pt
sys.argv.json
command.txt
stdout.log
stderr.log
train.log
checksums.sha256
source_fingerprint.json
data_fingerprint.json
~~~

M3 不生成 graph_fingerprint.json。checksums.sha256 使用 sha256sum 兼容格式，精确覆盖其余 13 个受控文件；Python verifier 和系统 sha256sum -c 均通过后才提交 completed 结果。source fingerprint 确定性递归包含 models/**/*.py，当前 fingerprint 为：

~~~text
e54420a9302dd02b5089486b8d9c64f4cff339ade163aec5eef43fa319b8e023
source files: 17
~~~

## 12. 公平消融合同

~~~text
U0: aux empty,  PMCR off, TEB off, AMD-TargetOnly
U1: aux nonempty, PMCR off, TEB off, AMD-Concat
U2: aux nonempty, PMCR off, TEB on,  AMD-Concat + TEB
U3: aux nonempty, PMCR on,  TEB off, AMD-Concat + PMCR
U4: aux nonempty, PMCR on,  TEB on,  EL-AMD
~~~

U1 与 U2 的 feature/target/aux 名称与顺序、schema、fold、split、scaler、horizon、seed、训练预算和评价流程相同；唯一结构差异为 TEB。M3 的 tiny synthetic smoke 只验证工程流，不用于宣称性能改善。

## 13. 新增与修改文件

新增代码：

- models/modules/target_exogenous_bridge.py

修改代码：

- models/modules/__init__.py
- models/tsAMD_enhanced.py
- main.py

新增测试：

- tests/test_teb.py
- tests/test_teb_parallel.py
- tests/test_teb_disabled_zero_context.py

修改测试：

- tests/test_tsAMD_enhanced.py
- tests/test_runner.py

修改文档：

- docs/AMD_EV_Thesis_Final_Implementation_Plan_v2.1.md

新增文档：

- docs/milestones/M3_teb.md

## 14. 测试结果

| 组别 | 结果 | failed | skipped | unittest 耗时 |
|---|---:|---:|---:|---:|
| TEB 模块 | 16/16 passed | 0 | 0 | 0.215 s |
| AMDEnhanced / public architecture / runner | 47/47 passed | 0 | 0 | 1.571 s |
| PMCR 保护 | 15/15 passed | 0 | 0 | 0.495 s |
| M1 保护 | 21/21 passed | 0 | 0 | 3.104 s |
| 完整回归 | 123/123 passed | 0 | 0 | 5.067 s |

关键数值：

~~~text
M0 AMD/AMDEnhanced off/off:
CPU prediction/MoE/state parity max_abs = 0
CUDA prediction/MoE/state parity max_abs = 0

M1 dual-interface:
raw x/y, y_time, state_source, MoE loss max_abs = 0

PMCR reparameterization:
CPU float64 temporal 2.22044604925e-16
CPU float64 delta    6.66133814775e-16
CPU float64 forward  1.73472347598e-18
CPU float32 temporal 2.38418579102e-07
CPU float32 delta    4.17232513428e-07
CPU float32 forward  2.98023223877e-08
CUDA float32 temporal 2.38418579102e-07
CUDA float32 delta    3.57627868652e-07
CUDA float32 forward  9.31322574615e-10
~~~

测试中的 CUDA 路径在本机可用并实际执行。仅出现 PyTorch 已知 CuDNN Conv1d workaround warning，不影响结果。

## 15. 实际测试命令

~~~bash
time env GIT_OPTIONAL_LOCKS=0 PYTHONDONTWRITEBYTECODE=1 /public/home/yueweiting/miniconda/envs/amd/bin/python -B -m unittest -v tests/test_teb.py tests/test_teb_parallel.py tests/test_teb_disabled_zero_context.py

time env GIT_OPTIONAL_LOCKS=0 PYTHONDONTWRITEBYTECODE=1 /public/home/yueweiting/miniconda/envs/amd/bin/python -B -m unittest -v tests/test_tsAMD_enhanced.py tests/test_public_architecture.py tests/test_runner.py

time env GIT_OPTIONAL_LOCKS=0 PYTHONDONTWRITEBYTECODE=1 /public/home/yueweiting/miniconda/envs/amd/bin/python -B -m unittest -v tests/test_pmcr.py tests/test_pmcr_no_cross_variable.py tests/test_pmcr_reparameterization.py

time env GIT_OPTIONAL_LOCKS=0 PYTHONDONTWRITEBYTECODE=1 /public/home/yueweiting/miniconda/envs/amd/bin/python -B -m unittest -v tests/test_target_offset.py tests/test_fold_scaler_no_leakage.py tests/test_temporal_graph_loader_consistency.py tests/test_state_restore_node_order.py tests/test_urbanev_data_contract.py

time env GIT_OPTIONAL_LOCKS=0 PYTHONDONTWRITEBYTECODE=1 /public/home/yueweiting/miniconda/envs/amd/bin/python -B -m unittest discover -s tests -p 'test_*.py' -v
~~~

## 16. 冻结资产保护

~~~text
models/tsAMD.py:
fa72cdbe34348364344c0d9c0755668a82d22f6a37ee061c7ece93ecfaf90ba1

M0 milestone:
e2a20131664391752340e92a9d9a5302b0078cac48d7dcdbb4a4841a16f62cdd

M1 milestone:
bcd9b3e3d821a1cf609423a3e8b58ecb4129bc202238e9fe1f8d7cf3a361c70b

M2 milestone:
8a4343e054f9dacd9cd623b930b07376a55e507f2d9085e56c634323b6dbda54

baseline tag:
fa9665627e6fcfb1d0c2bc22d943ca9666304fd6
~~~

UrbanEV、ModernTCN 和 TimeXer 参考仓库均保持 clean。既有 2026-08-14 至 2026-08-15 缓存未被当作本轮文件处理；本轮可明确归因的 3 个 pyc 和唯一新建空缓存目录已精确移除。

## 17. Patch-conditioned TEB 预留

Patch-conditioned TEB 仅是 M5 模块验收失败、限定调参仍无效且用户重新确认后的首选替代方向。本轮没有实现其 flag、参数、checkpoint key、runner 配置或测试，也没有把它写成当前完成方法。

## 18. 当前未完成事项

- 尚未获得 M3 Git closure 授权；
- 未 stage、commit 或 push；
- 未运行 UrbanEV、EPF、ETTh1、Weather、ECL、Exchange 或其他正式训练；
- 未进行 M5/M6 性能判断；
- 未实现 StateAdapter、H_time 或图模型；
- 未进入 M4。

下一步必须先由用户审核 canonical、源码、测试和本 milestone；只有用户另行授权后才执行 M3 Git closure。

## 19. Implementation review 与 repair 边界

只读 implementation review 确认 Global Target-Conditioned Residual TEB v1、formal RevIN 临时矩阵等价性和 source-kind checkpoint importer 可保留；同时确认当时的增强 runner 仍缺少 M1 UrbanEV production loader、UrbanEV horizon/fold 强制、原子 artifact 发布、真实 command/log provenance、checksum-aware summarizer 与对应永久门禁。本节之后记录的是 production/artifact repair 后的最终现场；前述第 1--18 节保留为实现轮历史。

本 repair 未实现 Patch-conditioned TEB，没有修改冻结 PMCR 方法，没有新增 EPF 数据管线，没有启动正式训练，也没有进入 M4。

## 20. Production runner 与 UrbanEV 数据合同修复

UrbanEV `target_exogenous` 正式路径现为：

~~~text
UrbanEVRawData.load(data_root)
-> UrbanEVFoldPreprocessor(raw).fit_transform(fold,preset)
-> one shared UrbanEVFoldBundle
-> TemporalRegionDataset(train/validation/test)
-> torch DataLoader
~~~

train、validation、test 共享同一 FoldBundle；train 可 shuffle，validation/test 保持确定顺序；窗口和 scaler 继续完全复用 M1，validation/test 不借前一 split 历史。FeatureSchema/FoldBundle 是 `feature_names`、`target_idx`、有序 `aux_idx` 与 schema fingerprint 的事实来源，CLI 同名字段仅作为 expected value 交叉核验。

UrbanEV 第一版在创建模型、loader 或 artifact 前强制：`seq_len=history_len=12`、`label_horizon in {3,6,9,12}`、`model_pred_len=pred_len=1`、`fold in {1,...,6}`、`target=volume`、`preset in {F0,...,F4}`。F0 固定空 aux 且 TEB off。artifact identity 使用 `artifact_horizon=label_horizon`；标准 parallel 数据仍以 `pred_len` 同时作为 model/artifact horizon，fold 固定为 `official`。

UrbanEV data fingerprint 文档直接记录 M1 raw data fingerprint、preprocessing-state fingerprint、feature-schema fingerprint、timestamp/node-order identity、fold/preset/horizon、目标与有序特征/aux、以及三段 split identity。scientific config、checkpoint、manifest、resume mismatch 与 artifact path 均区分 `label_horizon`、`model_pred_len` 和 `artifact_horizon`。

真实 UrbanEV F1/fold1/h3 工程 smoke 使用服务器实际数据并得到：

~~~text
x:             [4,12,6]
y:             [4,1]
prediction:    [4,1,1]
state_source:  [4,28]  # teb_context_dim=4
target_idx:    0
aux_idx:       (1,2,3,4,5)
finite:        true
published path contains: horizon_3/fold_1
~~~

该 smoke 只做单 batch 工程核验；测试内的 train/evaluate 被限制为确定性单 batch 路径，没有 optimizer.step，不构成正式训练或性能结论。

## 21. RevIN、loss shape 与 gamma 修复

`AMDEnhanced` 已删除私有 `_denormalize_all()`。formal path 将 AMS 的 `[B,C,H]` 输出转置为 `[B,H,C]`，在 `norm=True` 时直接调用冻结 RevIN 的 `denorm` 与 `slice(None)` 做全通道反归一化，再按任务返回 `[B,H,1]` 或 `[B,H,C]`。`target_exogenous` loss 显式对 prediction 执行 `squeeze(-1)` 并要求与 `[B,H]` label 完全同形；parallel 要求完整多变量 shape 完全一致，禁止广播。

永久 formal parity 覆盖 target_idx=0/非零、target_exogenous/parallel、norm true/false、affine true/false、CPU float32/float64 与可用时 CUDA float32。最终 prediction 与 MoE loss 最大绝对误差均为 0；既有 legacy M0 parity 同样保持 0。

`gamma_teb` 的模块直接 API、AMDEnhanced 和 runner 全部固定初始化为 `1e-3`，并永久拒绝 0、1e-2、-1e-3、NaN 与 Inf；未修改 `gamma_pmcr`。

## 22. Artifact schema v2、provenance 与 summarizer

增强 artifact schema version 固定为 2。completed 发布顺序为：

~~~text
hidden .<run_id>.staging
-> write all mutable artifacts
-> close stdout/stderr/train.log writers
-> write final completed manifest
-> write checksums.sha256 for the exact 13 controlled files
-> Python exact-set/digest verifier
-> system sha256sum -c checksums.sha256
-> release and remove staging .run.lock
-> os.replace(staging, final_run_dir) on the same filesystem
-> immutable completed artifact
~~~

final run directory 在训练和验证期间不存在；失败只保留 staging，且 final 不出现。故障注入覆盖 completed-manifest 后、checksum 后、Python verifier、系统 verifier、atomic rename 前等 false-completed 窗口。成功发布后无 `.run.lock`，checksum 覆盖文件不再变化。legacy AMD artifact 行为保持不变，历史产物不迁移。

`sys.argv.json` 保存运行时真实 `sys.argv`；`command.txt` 使用 `shlex.join([sys.executable,*sys.argv])`，正式 subprocess 测试确认包含真实 Python executable、main.py 和完整 argv。stdout/stderr/train.log 来自真实捕获并在 seal 前关闭，不使用空占位文件。

`summarize_results.py` 对 legacy AMD 保持原读取路径；对 schema-v2 enhanced artifact 仅接受 final completed 目录，忽略 hidden staging，要求并验证精确 checksum 集与 manifest/config/path identity，拒绝 missing/mismatch/running/failed/tampered artifact、unsupported variant 及同一 scientific identity/seed 的多个成功 run。summarizer 不写回 artifact。

## 23. Checkpoint 与永久门禁补强

source_kind 继续必须显式。baseline、pmcr_only 与 teb_only importer 在任何参数写入前验证完整 key 集和 tensor shape；partial `pmcr.*`、partial `teb.*`、unexpected key、非 enhancement missing、shape mismatch、错误或缺失 source_kind 均拒绝，并逐参数证明失败后模型完全不变。Resume 仍只允许同结构 `strict=True`；TEB/PMCR/schema/horizon/fold 任一 scientific config 变化均拒绝。

Repair 后测试结果：

| 组别 | 结果 | failed | skipped | unittest 耗时 |
|---|---:|---:|---:|---:|
| TEB single/parallel/zero-context | 17/17 passed | 0 | 0 | 0.279 s |
| AMDEnhanced formal RevIN/checkpoint | 15/15 passed | 0 | 0 | 0.797 s |
| runner + atomic artifact + summarizer | 38/38 passed | 0 | 0 | 4.327 s |
| PMCR + public architecture protection | 27/27 passed | 0 | 0 | 0.752 s |
| M1 protection | 21/21 passed | 0 | 0 | 3.084 s |
| 完整回归 | 136/136 passed | 0 | 0 | 7.946 s |

关键数值保持：formal RevIN prediction/MoE max_abs=0；M1 dual-interface 全项 max_abs=0；PMCR CPU float64 forward max_abs=1.73472347598e-18、CPU float32 forward max_abs=2.98023223877e-08、CUDA float32 forward max_abs=9.31322574615e-10。

## 24. Canonical repair 与文件指纹

Canonical 内部版本继续为 v2.1-R1；repair 前 SHA-256 为 `d5d2c3b2654e335338aca45d63832b8fe53848d407707a53e701d5c13180dbe5`，repair 后为 `d3a2d480454cd5d0c38d6d29ef2dcb043d90f3b77fd9e568b55186dd476cd5da`。修正范围仅包括 RevIN/shape、Patch-conditioned 第一替代方向、UrbanEV production runner、command provenance 和 staging/atomic publication；未改变 PMCR、M1 或第四章方法合同。

Repair 后源码 fingerprint 为 `961efb5a54742db972b55b6556a2dc2e05939e580b2c9e61b595f3ff9983f330`（17 个文件）。主要交付文件 SHA-256：

~~~text
main.py
bca8e104d4373e5ba0746a8ff0c0103b99976cec871cf8459a93f6858d7bcb22
models/tsAMD_enhanced.py
466e47b80e3c5ce3b9fc1d997bce72363dab3398a167214965158c5099a06670
models/modules/target_exogenous_bridge.py
c389157fd20ed66911163b6db0df3e7cd96f66b6f0bb112c432b77cf37588b2e
summarize_results.py
8f2f6667aa2c0f86d944e7e2119df8c62719ee18357048cccba9d533e6e04cc5
tests/test_runner.py
3420dd5cc586c2afba2539f156f6762b363027061d3d1619223353e8fcd603f3
tests/test_summarize_results.py
f8f9e3b75d7ab84d383ff827c65c47c5255fe95bbf58cf0c60ab7f50d08c2cde
tests/test_tsAMD_enhanced.py
29484e3bbd39d4ef8ae106e634fc2bf23db9984a8754e69715ec10d4d61a27f4
tests/test_teb.py
990476215857373c93ccccbca387f8387d11c81a6ae9cbd7c2850210c113e0ce
tests/test_teb_disabled_zero_context.py
8238dc0255ca9930bcf515b1441f5604482569bd01a3964cc91673f15ed2627e
tests/test_teb_parallel.py
0924d04d788a5bc1dc3f0f3937509d7096e7d57908a8a1ebb99e29eb41d45920
~~~

本 milestone 自身的最终 SHA-256 由 Codex 最终回复报告，避免文档自引用递归。

## 25. Repair 后、Git closure 前状态（历史快照）

本节记录 production/artifact repair 完成后、最终 review 与 Git closure 开始前的现场；当前最终状态以第 26 节 Git closure 为准。

状态保持 `Implementation complete; awaiting final review and Git closure`，implementation gate 为 Passed，implementation review 为 Pending final user review，Git closure 未授权。尚未 stage、commit 或 push；未运行正式训练；未实现 Patch-conditioned TEB、StateAdapter、H_time 或 graph mode；未进入 M4。

只有用户完成最终 review 并另行明确授权后，才能执行 M3 Git closure。

## 26. Git closure

- 用户已明确授权 M3 Git closure。
- M3 implementation review：Passed；M3 production/artifact repair review：Passed；implementation gate：Passed。
- closure 前使用正式 `amd` 环境重新运行完整回归，结果为 136/136 passed、failed=0、skipped=0（unittest 耗时 8.604 s）。
- closure 范围严格限于以下 13 个文件：

~~~text
docs/AMD_EV_Thesis_Final_Implementation_Plan_v2.1.md
docs/milestones/M3_teb.md
main.py
models/modules/__init__.py
models/modules/target_exogenous_bridge.py
models/tsAMD_enhanced.py
summarize_results.py
tests/test_runner.py
tests/test_summarize_results.py
tests/test_tsAMD_enhanced.py
tests/test_teb.py
tests/test_teb_disabled_zero_context.py
tests/test_teb_parallel.py
~~~

- canonical 最终 SHA-256 为 `d3a2d480454cd5d0c38d6d29ef2dcb043d90f3b77fd9e568b55186dd476cd5da`，内部版本保持 v2.1-R1。
- 冻结 `models/tsAMD.py`、PMCR、M0/M1/M2 milestone 和不可变 baseline tag 均未变化。
- 本次 closure 未运行正式训练，未实现 Patch-conditioned TEB，未实现 StateAdapter、H_time 或 graph mode，未进入 M4。
- 本次 closure 使用一个 M3 commit，并推送至 `origin/AMD-paper-repro-custom-modules-v1`。
- 最终 closure commit 的完整 SHA 由 Codex closure 回执报告；本文档不写入其自身所属 commit SHA，也不为记录该 SHA 创建递归补充提交。
- M3 closure 后冻结，后续阶段不得再向本 milestone 追加 M4 或更晚阶段结果。
