---
title: "基于 AMD 与异构双图的城市 EV 充电需求预测：最终模块魔改与实验实施方案"
author: "面向服务器 Codex 开发的唯一权威实施文档"
date: "2026-08-15"
version: "v2.1-R1（替代版：修复数据双接口、目标输出、状态接口、图归一化、空间残差与实验协议）"
---

# 0. 文档定位

本文件替代此前的 `AMD_EV_Thesis_Final_Implementation_Plan_v2.1.md`，作为后续实现、实验登记和论文写作的唯一权威方案。旧版及更早方案移入 `docs/archive/plans/`，它们只保留历史记录，不得参与当前实现决策。

当前仓库与开发位置：

```text
仓库：https://github.com/tlxx22/AMD
可执行冻结基准：amd_reproduced_baseline_v1 -> fa9665627e6fcfb1d0c2bc22d943ca9666304fd6
论文语义审计锚点：AMD-paper-norm-wd-ddi-v1 @ 5a718d5
开发分支：AMD-paper-repro-custom-modules-v1
时间模型 variant：el-amd-pmcr-teb-v1
时空模型 variant：st-el-amd-hst-sadr-sc-simgca-v1
```

`5a718d5` 仅用于解释 paper-close 语义演变；所有数值等价测试、基准重跑、checkpoint 对照和后续实验，统一以不可变标签 `amd_reproduced_baseline_v1` 指向的 `fa96656...` 为唯一可执行基准。不得在实现阶段同时存在两个“基准真值”。

决策优先级：

```text
本 v2.1-R1 及用户后续明确确认
    > amd_reproduced_baseline_v1 上 models/tsAMD.py 的实际 forward
    > 六篇来源论文的模块边界
    > 归档旧方案
```

本方案满足以下硬性要求：

1. 第三章以完整 AMD 为时间基准；
2. 在 AMD 上加入至少两个来自近三年正式论文、并经过明确修改的模块；
3. 第四章以 HSTGCN 类地理—需求异构空间结构为基准；
4. 在空间基准上加入至少两个来自近三年正式论文、并经过明确修改的模块；
5. 每个模块都有独立开关、独立消融、同输入基线和可追溯来源；
6. 第三、第四章均进行多数据集、多模型比较；
7. 结构和超参数只依据训练集与验证集，测试集只在方案冻结后使用；
8. 新增模块全部关闭时，增强入口必须与冻结 AMD 在相同权重、`eval()`、相同输入下数值等价；
9. “模块临时关闭”只能用于排障，不能在最终论文中替代“两模块均经过验证”的硬性要求；若某个来源模块经过限定调参仍失败，必须换用同来源的合理变体或更换另一篇近三年论文模块。

# 1. 最终锁定路线

## 1.1 第三章：纯时间模型

冻结 AMD 主干：

```text
RevIN -> MDM -> DDI -> AMS -> Forecast
```

新增模块执行顺序：

```text
RevIN
  -> x_ch
  -> MDM -> u_mdm
  -> DDI -> v_ddi
  -> PMCR? -> v_local
  -> TEB?  -> v_final, exo_context
  -> AMS(experts=v_final, selector=u_mdm)
  -> Forecast
```

最终时间模型：

```text
EL-AMD：Exogenous-and-Local Enhanced AMD
variant：el-amd-pmcr-teb-v1
```

两个来源模块：

- **PMCR：Peak-preserving Modern Convolution Refinement**
  - 来源：ModernTCN，ICLR 2024 Spotlight；
  - 借鉴可重参数化大核/小核 depthwise temporal convolution、ConvFFN1 和残差结构；
  - 删除与 AMD-DDI 重复的 ConvFFN2 跨变量分支；
  - 改成只补偿局部时间细节的单块轻量旁路。

- **TEB：Target–Exogenous Bridge**
  - 来源：TimeXer，NeurIPS 2024；
  - 借鉴内生全局表示、外生 variate token 和 exogenous-to-endogenous cross-attention；
  - 改成插在 AMD 隐表示上的轻量目标导向桥接层；
  - 不复制 TimeXer 的 Transformer 主干与预测头。

## 1.2 第四章：时空模型

空间基准：

```text
固定地理图 A_geo
+
按 fold 仅由训练切片构造的静态 DTW 需求图 A_DTW
```

第三章已经承担时间建模，因此只借鉴 HSTGCN 的异构关系与双分支思想，不复制其 GRU 和 region-specific prediction head。

两个来源模块：

- **SADR：State-Adaptive Demand Residual Graph**
  - 来源：ASTGRN，Applied Energy 383 (2025) 125320；
  - 借鉴动态节点 embedding、embedding projection 和相似度构图；
  - 改成由 EL-AMD 当前区域状态驱动、且只对 HSTGCN 长期 DTW 需求先验做残差修正。

- **SC-SimGCA：State-Conditioned Sim-GCA**
  - 来源：G-STAN，Sustainable Energy, Grids and Networks 44 (2025) 101975；
  - 保留多层 GCN、层间融合、stack 和 SimAM 特征细化；
  - 把原模型全局固定融合系数改成样本—节点—关系条件化门控；
  - 模块只输出空间残差，禁止再次把 `H_time` 作为预测旁路叠加到最终 head。

最终流程：

```text
历史多变量序列
      |
      v
EL-AMD -> state_source, y_time
      |
      v
trained StateAdapter（M4） -> H_time
      |
      +-----------------------------+
      |                             |
      v                             v
   A_geo                    A_DTW + SADR
      |                             |
      v                             v
SC-SimGCA residual       SC-SimGCA residual
      |                             |
      +------ heterogeneous fusion--+
                    |
                    v
        y_hat = y_time + spatial residual
```

# 2. 六篇核心论文与模块边界

| 角色 | 正式论文 | 出处 | 本方案使用范围 |
|---|---|---|---|
| 时间基准 | Adaptive Multi-Scale Decomposition Framework for Time Series Forecasting | AAAI 2025 | 完整 MDM + DDI + AMS |
| 时间模块 T1 | TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables | NeurIPS 2024 | 外生 variate token、内生全局表示、cross-attention 桥接 |
| 时间模块 T2 | ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis | ICLR 2024 Spotlight | Reparam large/small DWConv、ConvFFN1、residual |
| 空间基准 | Predicting Electric Vehicle Charging Demand Using a Heterogeneous Spatio-Temporal Graph Convolutional Network | TR-C 2023 | Geographic Graph + DTW Demand Graph + heterogeneous fusion |
| 空间模块 S1 | An Adaptive Spatio-Temporal Graph Recurrent Network for Short-Term Electric Vehicle Charging Demand Prediction | Applied Energy 383 (2025) 125320 | 动态 embedding、embedding projection、相似度图 |
| 空间模块 S2 | An Electric Vehicle Charging Demand Prediction Approach Based on a Graph-based Spatio-Temporal Attention Network | SEGAN 44 (2025) 101975 | Sim-GCA：多层 GCN、层融合、stack、SimAM |

论文贡献表述必须使用：

> 受 HSTGCN 地理—需求异构关系建模和 ASTGRN 自适应图学习启发，本文保留长期稳定的需求相似先验，并利用前一章时间状态对需求关系进行样本级残差修正。

不得使用：

> 本文首次提出动态需求图。

# 3. 冻结基准、Git 与 artifact 合同

## 3.1 不可变基准验证

禁止无条件重复创建或移动 tag。M0-A 使用以下逻辑：

```bash
EXPECTED=fa9665627e6fcfb1d0c2bc22d943ca9666304fd6

# 本地不存在时才创建
if ! git rev-parse -q --verify refs/tags/amd_reproduced_baseline_v1 >/dev/null; then
  git tag -a amd_reproduced_baseline_v1 "$EXPECTED" \
    -m "Immutable paper-close AMD baseline before thesis modules"
fi

# 必须核验指向，不一致则立即停止，禁止移动标签
ACTUAL=$(git rev-list -n 1 amd_reproduced_baseline_v1)
test "$ACTUAL" = "$EXPECTED"

git push origin refs/tags/amd_reproduced_baseline_v1

git ls-remote --tags origin refs/tags/amd_reproduced_baseline_v1
```

后续开发分支固定：

```text
AMD-paper-repro-custom-modules-v1
```

## 3.2 M0-A：只读审计

M0 唯一阶段报告：

```text
docs/milestones/M0_baseline_freeze_and_equivalence.md
```

机器生成的 patch、清单和原始证据统一放在：

```text
docs/evidence/M0/
```

至少记录：

- 相对 `upstream/main` 的累计行为差异，而不是只看最后一个 commit；
- norm、layernorm、weight decay、DDI 输入路径、best checkpoint、验证加载、布尔参数和 Solar header 修复；
- 已成功复现的数据集、命令、seed、指标和日志位置；
- `seq_len=12` 下 DDI 的 `patch={3,4,6,12}` 实际行为；
- checkpoint、数据、环境和源码 SHA-256；
- tag 的本地与远端指向；
- 当前 worktree 中 tracked/untracked 文件及其处理方式。

禁止使用：

```text
git clean -fd
git reset --hard
删除唯一论文或数据文件
用 stash 伪装干净状态
```

## 3.3 M0-B：pass-through 增强入口

只新建最小 `models/tsAMD_enhanced.py` 和对应测试；M0-B 不实现 PMCR、TEB、UrbanEV DataLoader、`H_time` 或随机 `StateProjection`，也不进入 M1。

增强入口必须逐句保持 baseline tag 上 `models/tsAMD.py` 的真实主路径：

```python
u_mdm = MDM(x_ch)
v = u_mdm
for block in DDI_blocks:
    v = block(v)
pred_all_norm, moe_loss = AMS(v, u_mdm)
```

M0-B 只增加 `return_state_source`。默认关闭时返回值仍为 `(pred, moe_loss)`；开启时额外返回确定性的 `state_source`，但预测与 MoE loss 必须完全不变：

```python
exo_context = v.new_zeros((v.shape[0], teb_context_dim))
state_source = torch.cat(
    (v[:, target_idx, :], u_mdm[:, target_idx, :], exo_context),
    dim=-1,
)
```

`target_idx` 必须是经过范围校验的单一目标通道索引，`teb_context_dim` 必须是固定正整数。形状合同为：

```text
state_source: [B, 2 * seq_len + teb_context_dim]
```

等价测试必须让 base/enhanced 使用相同权重与输入并同时处于 eval 模式；分别执行前恢复相同 CPU RNG 状态和所有可用 CUDA RNG 状态。`return_state_source=False` 与 `True` 两条路径的预测、MoE loss 相对 baseline 最大绝对误差都必须小于 `1e-6`。

## 3.4 artifact 路径

```text
artifacts/<variant>/<dataset>/<task_mode>/<target>/horizon_<h>/fold_<fold>/seed_<seed>/<run_id>/
```

受控字段固定为：

```text
标准 parallel multivariate：
task_mode=parallel_multivariate
target=all
fold=official

UrbanEV/CHARGED 纯时间：
task_mode=target_exogenous

第四章时空任务：
task_mode=graph_spatiotemporal
```

UrbanEV/CHARGED 的 `target` 与 `fold` 继续按真实目标和实验 fold 显式记录，不使用标准数据集占位符。

所有未来正式 run 必须原生保存：

```text
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
graph_fingerprint.json（第四章）
```

`sys.argv.json` 必须逐项保存运行时 `sys.argv`；`command.txt` 必须保存可重放的完整 shell-escaped command；`stdout.log`、`stderr.log` 和 `train.log` 必须完整保留正式运行输出。不得用事后推断的命令冒充原始命令证据。

`checksums.sha256` 必须作为独立文件生成，至少覆盖 `best.pt`、`last.pt`、`config.resolved.json`、`history.jsonl`、`metrics.json`、`manifest.json` 和 `train.log`。run 只有在所有文件原子落盘且 `sha256sum -c checksums.sha256` 通过后才能标记 completed。

resume 时必须核对 variant、dataset、task_mode、target、horizon、fold、seed、run_id、源码哈希、数据/图哈希和科学配置哈希；不一致时拒绝恢复。

# 4. 两类数据接口：第三章与第四章不得混用

原 v2.1 只描述了区域展开样本，但第四章必须恢复完整节点集合。最终固定两类 DataLoader。

## 4.1 TemporalRegionDataset：第三章纯时间训练

先按时间切分 train/val/test，再创建滑动窗口，最后把区域展开为独立样本：

```text
原始：X_graph [B_time,T,N,C]
展开：x_region [B_region,T,C]
标签：y_region [B_region,H_out] 或 [B_region,1]
```

训练集可以在窗口生成后 shuffle 区域样本；验证和测试保持确定顺序。禁止先从完整时间轴随机生成窗口后再切分。

该接口使 AMD-DDI 只在同一区域内部处理变量，不发生区域间消息传播。

## 4.2 GraphWindowDataset：第四章时空训练

每个 batch 必须包含同一时间窗口下的全部 N 个节点，并保持固定 node order：

```text
x_graph [B,T,N,C]
y_graph [B,N,H_out]
node_ids [N]
```

graph wrapper 按阶段执行：

```text
[B,T,N,C]
 -> permute/reshape
[B*N,T,C]
 -> node-wise EL-AMD(return_state_source=True)
state_source [B*N,2*T+d_teb]
y_time [B*N,H_out]
 -> M4 中经过已训练 StateAdapter
 -> restore
H_time [B,N,d]
y_time [B,N,H_out]
```

M1 只需验证 flatten/restore 后的 `state_source` 与 `y_time`；在 M4 创建并训练 StateAdapter 之前，不得伪造 `H_time`。

禁止把第三章任意 shuffle 后的 `B_region` 样本直接拼回图。必须通过时间窗口 ID 和固定 node order 恢复，且有单元测试验证。

## 4.3 两接口一致性测试

同一组完整图窗口在 `eval()` 下：

1. 用 `GraphWindowDataset` 内部 flatten；
2. 用 `TemporalRegionDataset` 按相同 node order 手工展开；
3. M1 验证两者得到的 node-wise `y_time` 和 `state_source` 必须一致；
4. M4 在同一已训练 StateAdapter 下再验证两者恢复的 `H_time` 一致。

# 5. 任务模式、输入变量与目标输出合同

## 5.1 第三章两种任务模式

### 模式 A：单目标 + 外生变量

```text
task_mode=target_exogenous
```

用于：

```text
UrbanEV
CHARGED
EPF-PJM（以及可选的其他 EPF 市场）
```

目标：

```text
UrbanEV/CHARGED：volume
EPF：price
```

外生变量只提供辅助信息，不计算未来预测损失。

### 模式 B：标准 M-to-M 多变量预测

```text
task_mode=parallel_multivariate
```

用于：

```text
ETTh1
Weather
ECL
Exchange
PEMS04/PEMS08（图任务中目标通常为节点主变量）
```

标准长序列数据输出全部目标变量；TEB 使用 TimeXer 的并行多变量模式。

## 5.2 UrbanEV 第一版变量

历史输入候选：

```text
volume
e_price
s_price
weather_central.csv: T -> Ta
weather_central.csv: P -> P
weather_central.csv: U -> h
hour_sin, hour_cos
weekday_sin, weekday_cos
is_weekend
```

规则：

- 非日历变量只使用历史窗口内观测值；
- UrbanEV 第一版天气固定使用 `weather_central.csv` 的 `T/P/U`，分别映射为 `Ta/P/h`；
- UrbanEV 第一版暂不使用 `P0`、`nRAIN`、`Td` 和 `weather_airport.csv`，不包含降水；
- 不输入未来真实天气；
- 所有公平比较模型获得同一组可用变量；
- 第一版不把 occupancy、duration 作为主输入；
- POI、区域面积、道路长度和桩数量留到第四章附加实验；
- 辅助变量选择只使用训练/验证集。

## 5.3 target-only 输出与 RevIN

UrbanEV/CHARGED 的 AMD-Concat 与 EL-AMD 第一版保留 AMD 的全部通道内部计算和 AMS 输出，以最大程度复用冻结基准；完成 RevIN 全通道反归一化后，只返回 `target_idx` 对应的目标预测，并只在该目标上计算 loss。

```text
pred_all_norm [B_region,C,H_out]
 -> RevIN denorm with per-channel statistics
pred_target = pred_all[:, target_idx, :]
loss = criterion(pred_target, y_target)
```

禁止把单通道预测直接送入要求 C 通道统计量的通用 RevIN `denorm`。若后续增加 target-specific denorm，必须先与“全通道 denorm 后选 target”做数值等价测试。

## 5.4 UrbanEV 标签

```text
history_len = 12
label_horizon in {3,6,9,12}
model_pred_len = 1
x = data[t:t+12]
y = volume[t+12+label_horizon-1]
```

四个 horizon 独立训练。禁止把 AMD `pred_len` 直接设为 3/6/9/12 并误做连续多步输出。

# 6. PMCR：ModernTCN-inspired 局部细化模块

## 6.1 插入位置

```text
RevIN -> MDM -> DDI -> PMCR -> TEB -> AMS -> Forecast
```

M2 尚未实现 TEB，因此本阶段真实路径为：

```text
RevIN -> MDM -> DDI -> PMCR -> AMS -> Forecast
```

其中 `AMS` experts 使用 PMCR 后的 `v`，selector 始终使用原始 `u_mdm`；禁止让 DDI 重新接回 `x_ch`，也禁止把 PMCR 后的 `v` 送入 selector。

## 6.2 精确张量合同

```text
H [B,C,T]
H_bc = reshape(H, [B*C,1,T])
U = Conv1d(1,d,kernel_size=1)(H_bc)       # [B*C,d,T]
V = ReparamLargeKernelDWConv(U)           # groups=d，[B*C,d,T]
V = FeatureWiseLayerNorm(V)               # [B*C,d,T]
V = ConvFFN1(V)                           # [B*C,d,T]
delta_bc = Conv1d(d,1,kernel_size=1)(V)   # [B*C,1,T]
delta = reshape(delta_bc, [B,C,T])
H_out = H + gamma_pmcr * delta
```

公开分析接口固定为：

```python
delta = pmcr.compute_delta(H)  # 未乘 gamma_pmcr
H_out = pmcr(H)                # H + gamma_pmcr * delta
```

`B*C` 仅用于将每个变量视作独立样本，所有变量共享同一套 PMCR 参数；PMCR 内不得发生跨变量数据混合。保留 variable-independent temporal processing、large/small temporal depthwise convolution、结构重参数化、ConvFFN1 和外层 residual；删除 patchify stem、多 stage backbone、ConvFFN2、完整 ModernTCN 预测头及其他重复的跨变量建模。

## 6.3 归一化与 ConvFFN1

归一化固定为 feature-wise LayerNorm：

```text
[B*C,d,T]
 -> transpose [B*C,T,d]
 -> LayerNorm(d, eps=1e-5)
 -> transpose [B*C,d,T]
```

禁止 BatchNorm、跨 batch 统计以及分支内独立 GroupNorm/LayerNorm。large/small 两个纯线性卷积分支先求和，再经过一个公共 feature-wise LayerNorm：

```text
large DWConv ----\
                  + -> feature-wise LayerNorm -> ConvFFN1
small DWConv ----/
```

公共 LayerNorm 不参与卷积核融合，部署形态中继续保留。

ConvFFN1 固定为：

```text
Conv1d(d,2d,kernel_size=1,bias=True)
 -> GELU
 -> Dropout(0.1)
 -> Conv1d(2d,d,kernel_size=1,bias=True)
 -> Dropout(0.1)
```

固定 expansion ratio 为 2；ConvFFN1 内部不增加 residual，PMCR 只保留外层 `H_out = H + gamma_pmcr * delta`。

## 6.4 gamma 与初始化

`gamma_pmcr` 固定为所有变量共享的可学习全局标量 `nn.Parameter`，初始化为 `1e-3`，不施加 sigmoid、softplus 或非负约束。禁止固定常数、每变量 gamma、每特征 gamma 和零初始化。

初始化固定为：

```text
input/output 1x1 projection：Xavier uniform，bias=0
ConvFFN1 两个 1x1 projection：Xavier uniform，bias=0
large/small DWConv：Kaiming fan-in、linear；各分支权重乘 1/sqrt(2)，bias=0
LayerNorm：weight=1，bias=0
gamma_pmcr：1e-3
```

禁止 output projection 零初始化，以保证首次反向传播时 PMCR 内部参数可获得非零梯度。

## 6.5 kernel、padding 与重参数化

固定：

```text
stride = 1
dilation = 1
padding_mode = zeros
padding = kernel_size // 2
k_small > 0，k_large > 0
k_small、k_large 均为奇数
k_large > k_small
```

在 `AMDEnhanced` 集成时要求 `k_large <= seq_len`。独立 `ReparamLargeKernelDWConv` 不绑定固定 T，但必须保持 same padding 和时间长度不变。短/长序列参数只作为显式配置建议，不得依据 dataset 名称或 `seq_len` 自动选择：

| 数据类型 | k_small | k_large | d |
|---|---:|---:|---:|
| UrbanEV/CHARGED，T=12 | 3 | 7 | 4 或 8 |
| 标准长序列，L>=96 | 5 | 31 | 8 或 16 |

固定提供：

```python
get_equivalent_kernel_bias()
switch_to_deploy()
to_deploy()
```

语义固定为：

- `get_equivalent_kernel_bias()` 只计算，不修改模块；
- `switch_to_deploy()` 显式、原地、幂等，不在普通 `forward()` 中自动执行；
- `to_deploy()` 深拷贝当前模块，对副本执行 `eval()` 和 `switch_to_deploy()`，不修改原训练模块；
- small kernel 居中补零后融合：`K_eq = K_large + center_pad(K_small)`，`b_eq = b_large + b_small`；
- 部署后只合并 large/small temporal DWConv；公共 LayerNorm、ConvFFN1、输入/输出投影和外层 residual 均保留。

## 6.6 开关、配置与 checkpoint

```text
use_pmcr=False：
  self.pmcr = None
  不产生 pmcr.* state_dict key
  直接旁路
  冻结 AMD checkpoint 继续 strict=True 加载

use_pmcr=True：
  显式提供 hidden_dim、kernel_small、kernel_large
  实例化完整 PMCR
  完整 PMCR checkpoint 使用 strict=True 恢复
  从冻结 AMD 初始化时使用专用 allowlist importer
```

专用 importer 只允许：

```text
missing keys == 当前模型全部 pmcr.* keys
unexpected keys == 空集
```

校验必须在修改参数前完成；禁止把全局静默 `strict=False` 作为兼容方案。`use_pmcr`、hidden dim、两个 kernel、dropout、gamma init 及 deploy 形态均应进入可追溯配置/checkpoint 元数据；正式 runner 接入在 M3 完成完整 EL-AMD 后统一处理。

## 6.7 必做测试

- shape 与 dtype/device；
- `use_pmcr=False` 严格 pass-through；
- 参数和输入梯度非零；
- 修改输入某一变量时，其他变量的 PMCR `delta` 不应变化，验证无跨变量混合；
- 训练态双分支与导出后重参数化卷积数值等价；
- `T=12` 下 kernel/padding 不改变长度。

# 7. TEB：TimeXer-inspired 目标—外生桥接

## 7.1 单目标模式

DDI+PMCR 输出：

```text
H [B,C,T]
H_y = H[:,target_idx,:]
```

目标 Query：

```text
q = W_q Pool(LayerNorm(H_y))      # [B,1,d]
```

每个外生变量独立编码为 variate token：

```text
e_j = W_j LayerNorm(X_aux,j)      # [B,d]
E_aux = stack(e_1,...,e_m)         # [B,m,d]
```

Cross-attention：

```text
c_exo = MHA(q,E_aux,E_aux)         # [B,1,d]
delta = W_o(c_exo).squeeze(1)      # [B,T]
H_y' = H_y + gamma_teb * delta
exo_context = c_exo.squeeze(1)     # [B,d]
```

仅替换目标通道，其他通道保持原值。

推荐：

```text
d=32
heads=4
dropout=0.1
gamma_teb=1e-3
```

第一版不改变 AMS selector 使用的 `u_mdm`，只改变 AMS experts 的输入。

## 7.2 TEB 关闭合同

```text
use_teb=False：
    v 保持逐元素不变
    exo_context = zeros([B,d_teb], device=v.device, dtype=v.dtype)
```

不得返回 `None`，不得改变 StateAdapter 输入维度。

## 7.3 M-to-M 并行模式

用于 ETTh1/Weather/ECL/Exchange：

- 每个变量同时作为一个 Query；
- 其他变量作为 Key/Value；
- 参数共享；
- 使用 diagonal mask，禁止变量只复制自身；
- 一次向量化完成，禁止 Python 循环逐变量运行完整 AMD；
- 输出仍为原 AMD 的全部变量。

## 7.4 公平对照

UrbanEV/EPF 必须比较：

```text
AMD-TargetOnly
AMD-Concat（与 Ours 完全相同的辅助变量）
AMD-Concat + TEB
```

只有 `AMD-Concat + TEB` 优于或至少不劣于 `AMD-Concat`，才能把增益归因于桥接结构。

# 8. 第三章最终 forward 与时间状态接口

## 8.1 前向流程

```python
x_norm = RevIN_norm(x)
x_ch = x_norm.transpose(1,2)               # [B,C,T]

u_mdm = MDM(x_ch)
v = u_mdm
for block in DDI_blocks:
    v = block(v)

if use_pmcr:
    v = PMCR(v)

exo_context = v.new_zeros((v.shape[0], teb_context_dim))
if use_teb:
    v, exo_context = TEB(
        hidden=v,
        raw_aux=x_norm,
        target_idx=target_idx,
        aux_idx=aux_idx,
    )

pred_all_norm, moe_loss = AMS(v, u_mdm)
pred_all = RevIN_denorm_all(pred_all_norm)
pred = select_target_or_all(pred_all, task_mode, target_idx)
```

不得再出现：

```text
h = x_ch 后送入 DDI
```

## 8.2 return_state_source 与后续 StateAdapter

第三章增强模型只公开 M0-B 已冻结的原始状态源接口，不直接返回未训练的 `StateProjection` 或 `H_time`：

```text
v_target     = v[:,target_idx,:]       # [B_region,T]
u_target     = u_mdm[:,target_idx,:]   # [B_region,T]
exo_context  = [B_region,d_teb]
state_source = concat(v_target,u_target,exo_context)
             # [B_region,2*T+d_teb]
```

调用合同固定为：

```python
pred, moe_loss, state_source = model(
    x,
    return_state_source=True,
)
```

`target_idx` 必须显式提供并经过范围校验；TEB 关闭或尚未实现时，`exo_context` 必须是 dtype/device 正确的确定性固定零张量。`return_state_source=True` 只增加返回值，不得改变预测或 MoE loss；默认调用仍返回 `(pred, moe_loss)`。

M0-B 不创建 `StateProjection`、`StateAdapter` 或 `H_time`。到 M4 的第四章 graph mode 才允许新增并训练独立 StateAdapter，以 `state_source` 为输入：

```text
s_v = Linear_v(LayerNorm(v_target))
s_u = Linear_u(LayerNorm(u_target))
s_e = Linear_e(exo_context)
H_region = MLP(LayerNorm(concat(s_v,s_u,s_e)))
H_time = reshape(H_region,[B,N,state_dim])
y_time = reshape(pred,[B,N,H_out])
```

该 StateAdapter 必须属于第四章模型及其 checkpoint，并参与训练；不得把随机、未训练投影冒充第三章 EL-AMD 输出。推荐 `d_s=16`、`state_dim=32`，最终值只依据训练/验证集确定。

# 9. 第三章实验设计

原 v2.1 把 UrbanEV 的 `AMD-V/AMD-Concat` 消融直接套到 Weather/ECL，任务语义不一致。本替代版将实验拆成两种协议。

## 9.1 必做数据集

### A. 单目标 + 外生变量协议

| 数据集 | 协议 | 作用 |
|---|---|---|
| UrbanEV | 12 h -> t+3/t+6/t+9/t+12 单点；6 folds | 核心 EV 场景 |
| EPF-PJM | 168 -> 24；TimeXer 官方切分/预处理 | 独立验证 TEB，不让外生模块只在 UrbanEV 一套数据上成立 |

完成最低版本后，优先扩展 EPF-NP/BE/FR/DE，形成五市场表。

### B. 标准 M-to-M 协议

| 数据集 | 协议 | 作用 |
|---|---|---|
| ETTh1 | pred 96/192/336/720 | 标准 benchmark |
| Weather | pred 96/192/336/720 | 多变量局部波动 |
| ECL | pred 96/192/336/720 | 高维电力负荷 |
| Exchange | pred 96/192/336/720 | 低维、非能源分布 |

第三章最低正式数据范围因此为：

```text
UrbanEV + EPF-PJM + ETTh1 + Weather + ECL + Exchange
```

可选：Solar、其余四个 EPF 市场。

## 9.2 Baseline

### 单目标 + 外生协议

```text
Last Observation（仅 UrbanEV）
DLinear
PatchTST
iTransformer
TiDE
TimeXer
ModernTCN
AMD-Concat
EL-AMD
```

### M-to-M 协议

```text
DLinear
PatchTST
iTransformer
TimeMixer
ModernTCN
TimeXer-parallel
AMD
EL-AMD
```

同一 dataset/target/horizon/fold 内统一输入变量、划分、scaler、seed、mask、反归一化和指标。模型专属学习率、hidden、layer、patch、kernel 可以不同，但必须使用相近验证搜索预算并留档。

## 9.3 消融矩阵

### target_exogenous：UrbanEV / EPF-PJM（U0-U4）

| 编号 | 结构 | 问题 |
|---|---|---|
| U0 | AMD-TargetOnly | 只有目标历史 |
| U1 | AMD-Concat | 相同辅助变量直接作为普通通道 |
| U2 | U1 + TEB | 定向外生桥接是否有效 |
| U3 | U1 + PMCR | 局部卷积细化是否有效 |
| U4 | U1 + TEB + PMCR | 最终时间模型 |

### parallel_multivariate：标准多变量（M0-M3）

本小节的 M0-M3 是标准多变量消融编号，与工程里程碑 M0 不同。

| 编号 | 结构 |
|---|---|
| M0 | AMD |
| M1 | AMD + PMCR |
| M2 | AMD + parallel-TEB |
| M3 | AMD + PMCR + parallel-TEB |

ETTh1/Exchange 只跑主 baseline、AMD、两个单模块和 EL-AMD；不重复全部输入变量消融。

## 9.4 UrbanEV 辅助变量消融

| 设置 | 输入 |
|---|---|
| F0 | volume only |
| F1 | volume + calendar |
| F2 | F1 + e_price + s_price |
| F3 | F1 + weather |
| F4 | all selected historical auxiliary features |

同时报告 AMD-Concat 与 EL-AMD。

## 9.5 模块验收线

每个时间模块最终保留必须满足：

1. 相对同输入 AMD 的验证指标平均退化不超过 0.5%；
2. 至少在一个核心场景和一个独立场景出现稳定改善：
   - TEB：UrbanEV/EPF 中至少两项；
   - PMCR：UrbanEV + Weather/ECL/ETTh1 中至少两项；
3. 3 个锁定 seed 下方向基本一致；
4. 参数量和耗时增幅与收益相匹配。

若不满足，不能简单关闭模块后仍宣称满足“两模块”要求，必须执行第 21 节的替换/变体流程。

# 10. 第四章：每个数据域必须独立获得时间基线

禁止把 UrbanEV 的 EL-AMD checkpoint 直接用于 CHARGED 或 PEMS 测试。

训练流程：

### UrbanEV

```text
加载第三章同 fold/horizon 的 UrbanEV EL-AMD checkpoint
-> 训练空间模块
-> 联合微调
```

### CHARGED 每座城市

```text
先在该城市独立训练 time-only EL-AMD（S0）
-> 保存该城市 checkpoint
-> 加入空间模块
-> 联合微调
```

### PEMS04/PEMS08

```text
先按标准 PEMS 协议独立训练 time-only EL-AMD（S0）
-> 加入空间模块
-> 联合微调
```

默认两阶段训练：

1. 加载同数据域 S0 checkpoint；
2. 前 5 epoch 冻结时间编码器，只训练空间模块；
3. 解冻后时间侧学习率设为空间侧的 0.1；
4. frozen epoch 数作为配置，可依据验证集在 `{3,5,10}` 中选择；
5. 不进行跨数据集权重迁移，除非单独设立迁移学习附加实验。

# 11. HSTGCN-core 与统一图归一化

## 11.1 图来源

### UrbanEV

```text
A_geo_raw：官方 adj.csv
A_DTW_raw：当前 fold 训练切片的平均周模式（168 维）计算 DTW
```

### CHARGED

```text
一座城市一张图
A_geo_raw：distance.csv -> KNN Gaussian
A_DTW_raw：该城市当前 fold 训练切片
```

### PEMS04/08

```text
A_geo_raw：标准道路距离/邻接图
A_DTW_raw：当前训练切片的平均日模式或锁定低维摘要
```

## 11.2 行随机消息传递合同

SADR 使用节点级 `lambda` 后，融合图可能非对称。因此全章统一使用带方向的 row-stochastic message passing，而不是融合后再套对称 Kipf 归一化。

静态图：

```text
A_geo = row_normalize(A_geo_raw + I)
A_DTW = row_normalize(A_DTW_raw + I)
```

GCN 基本操作：

```text
G = activation(A @ X @ W + b)
```

要求：

```text
A >= 0
每行和约为 1
包含 self-loop
```

所有图构建与缓存必须记录 node order、阈值/KNN、归一化方式和哈希。

## 11.3 HSTGCN-core 基线

```text
R_geo = GCN_stack(H_time,A_geo)
R_dem = GCN_stack(H_time,A_DTW)
R_sp = alpha * R_geo + (1-alpha) * R_dem
```

`alpha=sigmoid(alpha_logit)`，初始为 0.5。

该适配版实验名称固定：

```text
HSTGCN-core / HSTGCN-style static dual graph
```

只有完整重实现 GCN-GRU 与原预测模块时，才可在表中写 `HSTGCN`。

# 12. SADR：ASTGRN-inspired 状态需求残差图

基础 embedding：

```text
E0 [N,d_a]
```

EL-AMD 状态偏移：

```text
DeltaE_b = W_s H_time,b
E_b = E0 + gamma_e * LayerNorm(DeltaE_b)
```

状态相似度：

```text
S_b = ReLU(E_b E_b^T / sqrt(d_a))
```

Top-k：

```text
M_b = symmetric_union_topk(S_b,k)
A_state,b = row_softmax(mask(S_b,M_b))
```

必须显式保留 self-loop。

节点门控：

```text
lambda_b,n = sigmoid(MLP_lambda(H_time,b,n) + b_lambda)
b_lambda = -4
```

需求图融合：

```text
A_dem,b[n,:] =
    (1-lambda_b,n) * A_DTW[n,:]
  + lambda_b,n     * A_state,b[n,:]
```

两项均为 row-stochastic，因此融合后每行仍和为 1。

推荐：

```text
d_a=16
k=8
gamma_e=1e-3
```

## 12.1 大节点图的内存合同

完整 `B*N*N` 只允许在显存预算内使用。默认：

```text
N <= 512：可直接计算完整相似度
N > 512：使用 blockwise top-k，不保留完整 B*N*N
```

CHARGED 城市必须先记录 N、理论相似度张量大小和峰值显存，再选择实现。

可使用候选边并集：

```text
DTW top-k
union Geo KNN
union State blockwise top-k
```

但候选策略必须在所有比较模型和 folds 中固定，不得根据测试集改变。

# 13. SC-SimGCA：只输出空间残差

原 v2.1 定义 `H_r = H_time + ...`，随后又在最终 head 中加到 `y_time`，会让“空间残差”包含额外纯时间旁路，削弱 S3/S5 的归因。本替代版改为纯空间残差输出。

对关系分支 `r in {geo,demand}`：

```text
C_r^0 = H_time
G_r^1 = GCN_1(C_r^0,A_r)
rho_r^1 = sigmoid(MLP_r^1(H_time))
C_r^1 = (1-rho_r^1) * G_r^1 + rho_r^1 * C_r^0

G_r^2 = GCN_2(C_r^1,A_r)
rho_r^2 = sigmoid(MLP_r^2(H_time))
C_r^2 = (1-rho_r^2) * G_r^2 + rho_r^2 * C_r^1
```

层聚合：

```text
C_stack = concat(C_r^1,C_r^2,dim=feature)  # [B,N,2d]
```

Graph-SimAM：

```text
[B,N,2d]
 -> [B,2d,N,1]
 -> parameter-free SimAM energy attention
 -> [B,N,2d]
 -> Linear(2d,d)
```

输出：

```text
R_r = GraphSimAM(C_stack)       # 纯空间分支残差，不加 H_time
```

地理与需求分支的 `rho` 网络不共享。

启用模块时不再设置第二个零初始化内门控，避免与最终 `gamma_sp` 形成双零门控导致空间模块早期无梯度。模块关闭时显式旁路为对应普通 GCN 分支。

# 14. 第四章最终 forward

```text
H_time, y_time = EL_AMD_graph_mode(X_graph)

A_dem = SADR(H_time,A_DTW)

R_geo = SC_SimGCA(H_time,A_geo,relation='geo')
R_dem = SC_SimGCA(H_time,A_dem,relation='demand')

alpha = sigmoid(alpha_logit)
R_sp = alpha * R_geo + (1-alpha) * R_dem

y_hat = y_time + gamma_sp * SpatialHead(R_sp)
```

推荐：

```text
alpha_logit=0
gamma_sp=1e-3
```

模块全部关闭时由显式开关返回 `y_time`，而不是依赖 `gamma_sp` 恰好为 0。

输出头：

```text
UrbanEV/CHARGED：H_out=1
PEMS04/08：按标准协议 H_out=12
```

# 15. 第四章实验数据集

| 数据集 | 正式任务 | 作用 |
|---|---|---|
| UrbanEV | 12 h -> t+3/t+6/t+9/t+12 单点；6 folds | 第一核心 EV 区域级数据 |
| CHARGED-AMS/JHB/LOA/MEL/SPO/SZH | 每城市独立训练；官方 12 h -> 下一小时协议；fold/切分以官方代码审计锁定 | 第二核心 EV、多城市/站点级验证 |
| PEMS04 | 标准过去 12 点 -> 未来 12 点 | 跨领域适用性 |
| PEMS08 | 标准过去 12 点 -> 未来 12 点 | 跨领域适用性 |

CHARGED 六城市不能拼为一个跨洲图；每座城市独立 scaler、图、checkpoint 和结果。

UrbanEV 主目标：

```text
volume
```

辅助：

| 目标/口径 | 实验范围 |
|---|---|
| volume-11kW | 最强时空 baseline、EL-AMD、Ours |
| occupancy | 3-4 个代表模型 |
| duration | 时间充分时放附录 |

三个目标是同一数据集的三个预测变量，不能称为三个数据集。

# 16. 第四章 Baseline 与消融

## 16.1 UrbanEV

```text
EL-AMD（S0）
GCN-LSTM
ASTGCN
HSTGCN-core（必做内部静态双图基准）
AGCRN
ASTGRN（重实现，直接来源强基线）
STAEformer
Ours
```

若完整忠实重实现原 HSTGCN，可另列 `HSTGCN`；不得把 HSTGCN-core 与完整 HSTGCN 混成同一名称。

G-STAN 完整模型无官方代码，不是最低版本强制 baseline；若重实现成功，放扩展表。

## 16.2 CHARGED 六城市

```text
EL-AMD
GCN-LSTM
HSTGCN-core
ASTGRN 或 AGCRN（优先 ASTGRN，工程失败时使用 AGCRN 并说明）
STAEformer
Ours
```

结果表：

```text
Model | AMS | JHB | LOA | MEL | SPO | SZH | Avg Rank
```

## 16.3 PEMS04/08

```text
GCN-LSTM
AGCRN
STAEformer
HSTGCN-core adapted
EL-AMD
Ours
```

## 16.4 空间消融

| 编号 | 结构 | 用途 |
|---|---|---|
| S0 | EL-AMD | 纯时间 |
| S1 | EL-AMD + Geo only | 地理关系作用 |
| S2 | EL-AMD + DTW Demand only | 长期需求关系作用 |
| S3 | EL-AMD + Static Dual Graph | HSTGCN-core 基准 |
| S4 | S3 + SADR | ASTGRN 来源模块增益 |
| S5 | S3 + SC-SimGCA | G-STAN 来源模块增益 |
| S6 | S3 + SADR + SC-SimGCA | 最终模型 |

关键比较：

```text
S4 vs S3
S5 vs S3
S6 vs S4/S5
```

# 17. 数据泄漏、公平性与可复现性

- scaler 只使用当前 fold 训练切片拟合；
- DTW、Pearson、cosine、需求聚类等统计关系只使用训练切片；
- 官方地理 adjacency/distance/coordinates 可固定跨 fold 使用，但要记录来源与哈希；
- 若所谓“官方图”实际由全期需求统计生成，仍按 train-only 图处理；
- node order 与图矩阵严格一致；
- 不输入未来真实天气；
- 主表同一 dataset/target/horizon/fold 使用相同输入变量、划分、scaler、seed、mask、反归一化和聚合指标；
- 模型专属超参数可以不同，但搜索空间、预算和验证选择依据必须留档；
- 主表报告相同 seed 列表的均值和标准差；
- 单 seed 探索结果不得混入正式主表；
- TEB attention 只能称“注意力分配/关联权重”，不能直接解释为因果重要性；
- 测试集只在模型结构、变量和超参数冻结后运行。

图 cache key 至少包含：

```text
dataset
task_mode
target
horizon
fold
train_start/train_end
data_hash
node_order_hash
graph_method
graph_params
normalization
```

# 18. 代码目录与新增测试

```text
models/
├── tsAMD.py
├── tsAMD_enhanced.py
├── modules/
│   ├── target_exogenous_bridge.py
│   ├── modern_conv_refinement.py
│   └── state_adapter.py
└── spatial/
    ├── graph_conv.py
    ├── static_heterogeneous_graph.py
    ├── state_adaptive_demand_residual.py
    ├── state_sim_gca.py
    └── ev_spatiotemporal_model.py

utils/
├── dataloader_urbanev.py
├── dataloader_charged.py
├── dataloader_graph.py
├── temporal_region_dataset.py
├── graph_window_dataset.py
├── feature_schema.py
├── graph_builder.py
└── result_logger.py

tests/
├── test_amd_equivalence.py
├── test_ddi_effective.py
├── test_pmcr_no_cross_variable.py
├── test_pmcr_reparameterization.py
├── test_teb_disabled_zero_context.py
├── test_target_only_revin_denorm.py
├── test_target_offset.py
├── test_temporal_graph_loader_consistency.py
├── test_state_restore_node_order.py
├── test_fold_scaler_no_leakage.py
├── test_graph_node_alignment.py
├── test_graph_row_stochastic.py
├── test_sadr_sparse_topk.py
├── test_demand_stat_graph_train_only.py
├── test_spatial_zero_bypass.py
└── test_checkpoint_manifest.py
```

# 19. Codex 执行里程碑

| 阶段 | 任务 | 完成标志 |
|---|---|---|
| M0-A | tag、全量 diff、audit、artifact 和环境审计 | `docs/milestones/M0_baseline_freeze_and_equivalence.md`，基准可追溯 |
| M0-B | pass-through AMDEnhanced + return_state_source 空壳 | pred/MoE loss <1e-6；zero context；target denorm 测试 |
| G0 | 总门禁 | M0-A/M0-B 通过，worktree 干净 |
| M1 | TemporalRegionDataset + GraphWindowDataset | 标签、切分、node order、`state_source`/`y_time` 双接口一致性测试通过 |
| M2 | PMCR | shape、gradient、无跨变量、reparam 测试通过 |
| M3 | TEB | AMD-Concat 公平对照、parallel mode、zero context 测试通过 |
| M4 | 训练 StateAdapter 与 graph mode | `H_time [B,N,d]`、target-only output、适配后一致性测试通过 |
| M5 | 第三章筛选 | UrbanEV + EPF-PJM + Weather/ECL 验证结果 |
| M6 | 第三章正式实验 | 6 数据集主表、消融、效率 |
| M7 | HSTGCN-core、图归一化、train-only DTW | S0-S3 跑通，图测试通过 |
| M8 | SADR | S4、blockwise top-k、关系可视化 |
| M9 | SC-SimGCA | S5，纯空间 residual 测试通过 |
| M10 | 第四章完整实验 | UrbanEV + CHARGED + PEMS04/08，S6 与主表 |

# 20. 结果表与可视化

## 20.1 第三章

表 A：单目标外生协议

```text
Model | UrbanEV h3/h6/h9/h12/Avg | EPF-PJM MSE/MAE | Avg Rank
```

表 B：标准 M-to-M

```text
Model | ETTh1 | Weather | ECL | Exchange | Avg Rank
```

不同数据集指标不直接求数值平均，只报告平均排名。

可视化：

- TEB 对价格、服务费、天气的注意力分配；
- PMCR 在峰值和突变窗口的残差响应；
- AMD 与 EL-AMD 峰值预测案例；
- 不把 attention 权重表述为因果贡献。

## 20.2 第四章

- UrbanEV horizon 详细表；
- CHARGED 六城市表；
- PEMS04/08 标准表；
- S0-S6 消融；
- 参数量、显存、epoch 时间、推理时间；
- `A_DTW`、`A_state`、`A_dem` 热力图；
- 早高峰、晚高峰、深夜关系变化；
- `lambda` 和 `rho` 分布；
- 典型住宅、商业、交通枢纽案例。

# 21. 模块失败与替换合同

临时降级可以用于定位问题，但最终毕业版本仍必须有两个经过修改且通过消融的时间模块，以及两个经过修改且通过消融的空间模块。

| 模块 | 第一轮允许调整 | 保留来源的备选实现 | 最终仍失败时 |
|---|---|---|---|
| TEB | 缩小 aux；d/heads；dropout | TimeXer-inspired gated additive target bridge | 更换另一篇近三年外生变量模块，不得简单删除 |
| PMCR | k=7->5；d 减小；仅 target | 保留 Reparam DWConv+ConvFFN1 的 target-only PMCR | 更换另一篇近三年局部时间模块 |
| SADR | b_lambda 更负；k/d_a；正则 | ASTGRN global adaptive graph 与 DTW 的残差融合 | 更换另一篇近三年空间图模块；退回静态双图只算排障结果 |
| SC-SimGCA | rho 初始化；层数；SimAM lambda | 保留 G-STAN 层融合，移除 Graph-SimAM，改名 SC-GCF | 若仍失败，更换另一篇近三年空间传播模块 |

单模块正式通过线：

1. 相对同输入/同骨干基线平均退化不超过 0.5%；
2. 至少在一个 EV 数据域和一个外部数据域或第二城市上产生稳定改善；
3. 3 seed 方向基本一致；
4. 最终组合不因模块交互产生稳定退化。

# 22. 代码与复现难度

| 论文 | 官方代码 | 本方案使用难度 |
|---|---|---|
| AMD | https://github.com/TROUBADOUR000/AMD | 已复现 |
| TimeXer | https://github.com/thuml/TimeXer | TEB：低—中，2-4 个有效开发日 |
| ModernTCN | https://github.com/luodhhh/ModernTCN | PMCR：中，3-5 日 |
| HSTGCN | 未检索到可核验作者仓库 | HSTGCN-core：中，3-6 日 |
| ASTGRN | 未检索到可核验作者仓库 | graph learner：低—中，2-3 日；完整 baseline 5-8 日 |
| G-STAN | 未检索到可核验作者仓库 | SC-SimGCA：低—中，2-4 日 |

数据/官方仓库：

```text
UrbanEV：https://github.com/IntelligentSystemsLab/UrbanEV
CHARGED：https://github.com/IntelligentSystemsLab/CHARGED
```

# 23. 正式参考文献

[T0] Hu, Y., Liu, P., Zhu, P., Cheng, D., and Dai, T. Adaptive Multi-Scale Decomposition Framework for Time Series Forecasting. AAAI, 2025.

[T1] Wang, Y., Wu, H., Dong, J., et al. TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables. NeurIPS, 2024.

[T2] Luo, D., and Wang, X. ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis. ICLR Spotlight, 2024.

[S0] Wang, S., Chen, A., Wang, P., and Zhuge, C. Predicting Electric Vehicle Charging Demand Using a Heterogeneous Spatio-Temporal Graph Convolutional Network. Transportation Research Part C, 153, 104205, 2023.

[S1] Wang, S., Li, Y., Shao, C., Wang, P., Wang, A., and Zhuge, C. An Adaptive Spatio-Temporal Graph Recurrent Network for Short-Term Electric Vehicle Charging Demand Prediction. Applied Energy, 383, 125320, 2025.

[S2] Jiang, D., Gong, X., Wei, Y., Peng, B., and Xu, Z. An Electric Vehicle Charging Demand Prediction Approach Based on a Graph-based Spatio-Temporal Attention Network. Sustainable Energy, Grids and Networks, 44, 101975, 2025.

[D1] Li, H., Qu, H., Tan, X., et al. UrbanEV: An Open Benchmark Dataset for Urban Electric Vehicle Charging Demand Prediction. Scientific Data, 12, 523, 2025.

[D2] Guo, Z., You, L., Zhu, R., et al. A City-scale and Harmonized Dataset for Global Electric Vehicle Charging Demand Analysis. Scientific Data, 12, 1254, 2025.

# 24. 最终一句话路线

```text
第三章：AMD + ModernTCN-inspired PMCR + TimeXer-inspired TEB
数据：UrbanEV + EPF-PJM + ETTh1 + Weather + ECL + Exchange

第四章：EL-AMD + HSTGCN-core + ASTGRN-inspired SADR + G-STAN-inspired SC-SimGCA
数据：UrbanEV + CHARGED 六城市 + PEMS04 + PEMS08

所有数据域独立训练；时空模型使用完整图窗口；空间模块只输出 residual；
产物按 variant/dataset/task_mode/target/horizon/fold/seed/run_id 隔离。
```
