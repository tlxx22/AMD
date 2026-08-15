---
title: "[已归档 v2.0] 基于 AMD 与异构双图的城市 EV 充电需求预测：最终模块魔改与实验实施方案"
author: "面向服务器 Codex 开发的最终实施文档"
date: "2026-08-14"
version: "v2.0（依据六篇正式论文及 ASTGRN 正式期刊版复核）"
---

> **归档声明（2026-08-15）：** 本 v2.0 快照仅供追溯，不得参与实现决策。唯一权威方案为 `docs/AMD_EV_Thesis_Final_Implementation_Plan_v2.1.md`。

# 0. 文档定位

本文件用于指导服务器端 Codex 在已成功复现的 AMD 分支上进行后续开发、实验登记和论文写作。当前基准仓库与分支为：

```text
仓库：https://github.com/tlxx22/AMD
分支：AMD-paper-norm-wd-ddi-v1
```

本方案满足以下硬性要求：

1. 第三章以一个完整时间基准模型 AMD 为骨干；
2. 在 AMD 上加入至少两个来自近三年正式论文的模块；
3. 两个模块均经过明确改造，不能原样复制；
4. 第四章以 HSTGCN 类地理图—需求图异构空间结构为基准；
5. 在空间基准上加入至少两个来自近三年正式论文的模块；
6. 每个新增模块都有独立开关、独立消融和基线保持设计；
7. 第三章与第四章均进行多数据集、多模型基线比较；
8. 任何结构取舍只依据训练集与验证集，不根据测试集反复改模。

# 1. 最终锁定结论

## 1.1 第三章：纯时间模型

时间骨干保持完整 AMD：

```text
RevIN -> MDM -> DDI -> AMS -> Forecast
```

在 DDI 与 AMS 之间加入两个改造模块：

```text
RevIN -> MDM -> DDI -> PMCR -> TEB -> AMS -> Forecast
```

最终时间模型暂称：

```text
EL-AMD：Exogenous-and-Local Enhanced AMD
```

其中：

- **TEB：Target–Exogenous Bridge，目标—外生变量桥接模块**
  - 来源：TimeXer，NeurIPS 2024；
  - 借鉴外生变量 token、内生全局表示和外生到内生 cross-attention；
  - 改造成适配 AMD 隐表示的轻量、目标导向桥接层。

- **PMCR：Peak-preserving Modern Convolution Refinement，峰值保真现代卷积细化模块**
  - 来源：ModernTCN，ICLR 2024 Spotlight；
  - 借鉴大核/小核可重参数化 depthwise temporal convolution、ConvFFN1 与残差结构；
  - 改造成单块、轻量、仅补偿局部时间细节的 AMD 旁路；删除与 DDI 重复的跨变量 ConvFFN2。

## 1.2 第四章：时空模型

空间基准采用 HSTGCN 的核心异构关系思想：

```text
固定地理图 A_geo + 静态需求图 A_DTW
```

第三章已经负责时间建模，因此第四章不照搬 HSTGCN 的 GRU 和 region-specific prediction head，只保留并重构空间骨干：

```text
H_time -> Geo branch
       -> Demand branch
       -> Heterogeneous fusion
```

在此基础上加入两个改造模块：

- **SADR：State-Adaptive Demand Residual Graph，状态自适应需求残差图**
  - 来源：ASTGRN，Applied Energy 2025；
  - ASTGRN 正式版明确以随时间更新的节点 embedding 生成 A_t；
  - 本文不宣称首次提出动态图，而是将 ASTGRN 类动态 embedding 学习改造成“对 HSTGCN 长期 DTW 需求先验的状态残差修正”。

- **SC-SimGCA：State-Conditioned Sim-GCA，状态条件化简单图卷积注意模块**
  - 来源：G-STAN，Sustainable Energy, Grids and Networks 2025；
  - 保留多层 GCN、层间融合连接、stack 与 SimAM 特征细化；
  - 将原模型全局固定融合系数 alpha 改成由样本、节点和关系类型共同决定的状态门控。

最终时空流程为：

```text
历史多变量序列
      |
      v
EL-AMD
      |
      v
H_time [B,N,d]
   |                  |
   v                  v
A_geo             A_DTW + SADR state residual
   |                  |
SC-SimGCA          SC-SimGCA
   |                  |
   +-------- HSTGCN-style fusion --------+
                         |
                         v
          y_time + spatial residual head
```

# 2. 六篇核心论文及模块边界

| 角色 | 正式论文 | 出处 | 本方案使用范围 |
|---|---|---|---|
| 时间基准 | Adaptive Multi-Scale Decomposition Framework for Time Series Forecasting | AAAI 2025 | 完整 MDM + DDI + AMS |
| 时间模块 T1 | TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables | NeurIPS 2024 | 外生 token、内生全局表示、cross-attention 桥接思想 |
| 时间模块 T2 | ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis | ICLR 2024 Spotlight | Reparam large/small DWConv、ConvFFN1、残差块 |
| 空间基准 | Predicting Electric Vehicle Charging Demand Using a Heterogeneous Spatio-Temporal Graph Convolutional Network | TR-C 2023 | Geographic Graph + DTW Demand Graph + 双关系融合 |
| 空间模块 S1 | An Adaptive Spatio-Temporal Graph Recurrent Network for Short-Term Electric Vehicle Charging Demand Prediction | Applied Energy 383 (2025) 125320 | 动态节点 embedding、embedding projection、相似度构图 |
| 空间模块 S2 | An Electric Vehicle Charging Demand Prediction Approach Based on a Graph-based Spatio-Temporal Attention Network | SEGAN 44 (2025) 101975 | Sim-GCA：多层 GCN、层融合、stack、SimAM |

## 2.1 ASTGRN 正式版带来的关键修正

ASTGRN 正式期刊版明确写出：

```text
E_init,t in R^(Nxd)
A_t = softmax(ReLU(E_init,t E_init,t^T))
E_hid,t = E_init,t W_t + b_t
A'_t = softmax(ReLU(E_hid,t E_hid,t^T))
```

因此 ASTGRN 应被视为真正的时变/自适应构图方法。本文第四章不能把“根据当前状态生成动态图”本身写成独立原创点。本文与 ASTGRN 的区别必须固定为：

```text
ASTGRN：完全依赖动态 latent graph，不保留显式地理/需求先验
本文：固定地理图 + 长期 DTW 需求先验 + EL-AMD 状态残差修正
```

论文贡献表述应使用：

> 受 HSTGCN 地理—需求异构关系建模和 ASTGRN 自适应图学习启发，本文保留长期稳定的需求相似先验，并利用前一章时间状态对需求关系进行样本级残差修正。

不要使用：

> 本文首次提出动态需求图。

# 3. 开发前：冻结成功复现的 AMD

## 3.1 Git 操作

```bash
git checkout AMD-paper-norm-wd-ddi-v1
git status
git log -1 --oneline

git tag amd_reproduced_baseline_v1
git checkout -b thesis/final-modules-v2
```

若尚未配置上游：

```bash
git remote add upstream https://github.com/TROUBADOUR000/AMD.git
git fetch upstream
```

要求 Codex 输出：

```text
docs/baseline_audit.md
```

至少记录：

- 当前分支相对上游 AMD 的所有行为变化；
- norm、layernorm、weight decay、DDI、最佳模型保存和验证加载修复；
- 已成功复现的数据集、命令、随机种子、指标；
- `seq_len=12` 下 DDI patch 的实际行为；
- 当前 checkpoint 的 SHA-256、Git commit 和环境版本。

## 3.2 基准文件保护

建议：

```text
models/tsAMD.py          # 基准行为冻结
models/common.py         # 非必要不改
models/tsmoe.py          # 原 AMS 不改
models/tsAMD_enhanced.py # 新模型入口
```

关闭所有新增模块时，新模型必须与基准数值等价：

```python
base.eval()
enhanced.eval()
with torch.no_grad():
    y0, l0 = base(x)
    y1, l1 = enhanced(x, use_teb=False, use_pmcr=False)
assert (y0 - y1).abs().max() < 1e-6
assert (l0 - l1).abs().max() < 1e-6
```

# 4. 第三章数据接口与“纯时间”边界

## 4.1 UrbanEV 数据形状

原始区域数据：

```text
X_graph: [B,T,N,C]
```

第三章先按时间划分 train/val/test，再将区域展开为独立样本：

```text
一个样本 = 某一时间窗口内某一区域的多变量序列
x: [T,C]
y: [1]
DataLoader: [B_region,T,C]
```

禁止先生成所有区域窗口后随机切分。这样 AMD 的 DDI 只建模同一区域内部变量关系，不发生区域之间消息传播。

## 4.2 UrbanEV 第一版变量

预测目标：

```text
volume
```

历史辅助变量候选：

```text
e_price
s_price
weather_central 中缺失率低、语义清楚的温度/湿度/气压/降水变量
hour_sin, hour_cos
weekday_sin, weekday_cos
is_weekend
```

主实验规则：

- 非日历变量只使用历史窗口内观测值；
- 不输入未来真实天气；
- 所有模型获得完全相同的可用变量；
- 第一版不把 occupancy 和 duration 作为主输入；
- 静态 POI、区域面积、道路长度和桩数量留到第四章或附录。

## 4.3 UrbanEV 标签定义

每个 horizon 独立训练：

```text
h in {3,6,9,12}
x = data[t:t+12]
y = volume[t+12+h-1]
```

AMD 内部输出长度：

```text
model_pred_len = 1
```

数据标签距离使用独立参数：

```text
label_horizon = h
```

禁止把 AMD 的 `pred_len` 直接设为 3、6、9、12 后输出连续序列。

## 4.4 DDI patch 审计

当前分支必须实测 `patch in {3,4,6,12}`：

- 输出是否退化为输入；
- DDI 参数是否有非零梯度；
- forward/backward 是否稳定。

若 `seq_len=12, patch=12` 在当前分支仍无有效更新，UrbanEV 主设置使用：

```text
patch = 3
```

标准 AMD 数据集继续使用已成功复现的原配置。

# 5. 时间模块 T1：TEB

## 5.1 论文出处与借鉴点

来源：

> Wang et al., TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables, NeurIPS 2024.

TimeXer 将内生目标切成 patch token，并使用一个内生全局 token 作为桥梁；外生变量被编码成 variate token，全局内生 token 作为 Query，通过 cross-attention 从外生变量中选择信息。

本方案不复制 TimeXer Transformer，只借用“目标表示定向查询辅助变量”的机制。

## 5.2 最终结构

AMD 的 DDI 输出：

```text
H: [B,C,T]
```

目标通道：

```text
H_y = H[:, target_idx, :]
```

目标 Query：

```text
q = W_q Pool(LayerNorm(H_y))       # [B,1,d]
```

每个外生变量单独编码：

```text
e_j = W_j LayerNorm(X_aux,j)       # [B,d]
E_aux = stack(e_1,...,e_m)          # [B,m,d]
```

Cross-attention：

```text
c_exo = MHA(q, E_aux, E_aux)        # [B,1,d]
```

注入目标通道：

```text
delta = W_o(c_exo)                  # [B,T]
H_y' = H_y + gamma_teb * delta
```

推荐初值：

```text
d = 32
heads = 4
dropout = 0.1
gamma_teb = 1e-3
```

第一版不改变 AMS selector 使用的 MDM 表示，只改变 AMS predictor 的输入。

## 5.3 与 TimeXer 的明确差异

| TimeXer | TEB |
|---|---|
| Transformer 主干 | AMD 主干 |
| 内生 patch self-attention | 不复制，由 MDM/DDI 负责时间建模 |
| 多层 global token bridge | 单个轻量桥接层 |
| 完整预测头 | 保留 AMD 的 AMS |
| 通用外生预测框架 | 针对区域 volume 目标及辅助因素 |

论文中写“受 TimeXer 启发设计 TEB”，不能写“将 TimeXer 原模块直接接入 AMD”。

## 5.4 标准多变量数据集的并行模式

在 Weather/ECL/ETTh1/Exchange 的 M-to-M 设置中，沿用 TimeXer 的并行多变量思想：

- 每个变量同时作为一个 Query；
- 其他变量作为 Key/Value；
- 参数共享；
- 对角 mask 防止变量只复制自身；
- 输出仍为原 AMD 的全部目标变量。

该模式必须向量化，禁止 Python 循环逐变量运行模型。

# 6. 时间模块 T2：PMCR

## 6.1 论文出处与借鉴点

来源：

> Luo and Wang, ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis, ICLR 2024 Spotlight.

ModernTCN 的核心包括：

- variable-independent embedding；
- 大感受野 depthwise temporal convolution；
- large/small kernel 的结构重参数化；
- ConvFFN1 负责变量内部特征变换；
- ConvFFN2 负责跨变量依赖；
- residual block。

## 6.2 为什么不能只放两个普通 Conv1d

仅写：

```text
DWConv(k=3) + DWConv(k=7)
```

不足以体现 ModernTCN 模块来源。最终 PMCR 必须至少保留：

```text
Reparam large/small DWConv + ConvFFN1 + residual
```

同时删除 ConvFFN2，因为 AMD 的 DDI 已经承担跨变量交互，重复加入会造成解释和优化冲突。

## 6.3 最终结构

输入：

```text
H: [B,C,T]
```

变量独立特征投影：

```text
U = W_in(H) -> [B,C,d,T]
```

时间卷积：

```text
V = ReparamLargeKernelDWConv(U, k_large, k_small)
```

变量内 ConvFFN：

```text
V1 = PWConv_expand(V)
V2 = GELU(Norm(V1))
V3 = PWConv_reduce(V2)
```

回投影与残差：

```text
delta = W_out(V3) -> [B,C,T]
H' = H + gamma_pmcr * delta
```

主参数：

| 数据类型 | k_small | k_large | d |
|---|---:|---:|---:|
| UrbanEV / CHARGED，T=12 | 3 | 7 | 4 或 8 |
| 标准长序列，L>=96 | 5 | 31 | 8 或 16 |

其他：

```text
gamma_pmcr = 1e-3
dropout = 0.1
优先 GroupNorm/LayerNorm，避免区域展开后 BatchNorm 统计耦合
```

训练时保留 large/small 两分支；推理导出时可使用 ModernTCN 的结构重参数化方式合并卷积核。

# 7. 第三章最终前向流程

```python
x_norm = RevIN(x)
x_ch = x_norm.transpose(1, 2)       # [B,C,T]

h_mdm = MDM(x_ch)
h = x_ch
for block in DDI_blocks:
    h = block(h)

if use_pmcr:
    h = PMCR(h)

if use_teb:
    h, exo_context = TEB(
        hidden=h,
        raw_aux=x_norm,
        target_idx=target_idx,
        aux_idx=aux_idx,
    )

pred, moe_loss = AMS(h, h_mdm)
pred = RevIN_denorm(pred)

if return_state:
    H_time = StateProjection(
        concat(
            Pool(h_target),
            Pool(h_mdm_target),
            exo_context,
        )
    )
```

推荐：

```text
state_dim = 32（第一版）
```

# 8. 第三章实验设计

## 8.1 必做数据集

| 数据集 | 角色 | 协议 | 指标 |
|---|---|---|---|
| UrbanEV | 核心 EV 应用 | 12 h 输入，分别预测 t+3/t+6/t+9/t+12 单点；6 folds | MAE、RMSE、RAE、WAPE |
| ETTh1 | 标准多变量 benchmark | 沿用已复现 AMD 配置；96/192/336/720 | MSE、MAE |
| Weather | 多变量、局部波动显著 | 沿用 AMD；96/192/336/720 | MSE、MAE |
| ECL | 高维电力负荷 | 沿用 AMD；96/192/336/720 | MSE、MAE |
| Exchange | 低维、非能源分布 | 沿用 AMD；96/192/336/720 | MSE、MAE |

可选扩展：

- Solar：完成上述五个后再补；
- EPF-NP/PJM/BE/FR/DE：若需要专门强化 TEB 的外生变量论证，再使用 TimeXer 官方协议 168->24。

## 8.2 Baseline

所有五个必做数据集：

1. DLinear；
2. PatchTST；
3. iTransformer；
4. TimeMixer；
5. TimeXer；
6. ModernTCN；
7. AMD；
8. EL-AMD（Ours）。

UrbanEV 额外加入：

- Last Observation；
- LSTM。

所有模型在同一个数据集上必须获得相同变量、相同时间划分和相同 scaler。

## 8.3 模块消融

| 编号 | 结构 | 验证问题 |
|---|---|---|
| T0 | AMD-V：仅 volume | 目标自身历史基线 |
| T1 | AMD-Concat：volume + 同一组辅助变量 | 更多输入本身的收益 |
| T2 | AMD-Concat + TEB | 目标—外生桥接是否有效 |
| T3 | AMD-Concat + PMCR | 局部现代卷积细化是否有效 |
| T4 | AMD-Concat + TEB + PMCR | 最终时间模型 |

完整 T0-T4：

```text
UrbanEV + Weather + ECL
```

ETTh1/Exchange：

```text
AMD、AMD+TEB、AMD+PMCR、EL-AMD 及主 baseline
```

## 8.4 UrbanEV 辅助变量消融

| 输入设置 | 内容 |
|---|---|
| F0 | volume only |
| F1 | volume + calendar |
| F2 | F1 + e_price + s_price |
| F3 | F1 + weather |
| F4 | all selected historical auxiliary features |

同时比较 AMD-Concat 与 EL-AMD，避免将“变量更多”和“模块有效”混为一谈。

## 8.5 效率实验

至少报告：

- 参数量；
- 可训练参数增量；
- 峰值显存；
- 单 epoch 时间；
- 单 batch 推理时间；
- UrbanEV 6 folds 全部训练总 GPU 时长。

## 8.6 模块验收线

单模块通过条件：

1. 在 UrbanEV 筛选设置中相对同输入 AMD 的验证 MAE 不劣于 0.5%；
2. 至少在 UrbanEV 多个 horizon 或 Weather/ECL 中产生稳定改善；
3. 三个随机种子下方向一致；
4. 参数和耗时增幅与收益相匹配。

不能预先保证测试集一定不下降；零初始化残差仅保证新模型的函数空间包含或接近基线。

# 9. 第四章时间状态接口与训练方式

第三章完成后，输出：

```text
H_time: [B,N,d]
y_time: [B,N,1]
```

推荐两阶段训练：

1. 加载第三章 EL-AMD checkpoint；
2. 前 5 个 epoch 冻结时间编码器，只训练空间模块；
3. 再解冻时间编码器，时间部分学习率设为空间部分的 0.1；
4. 最终预测采用残差形式：

```text
y_hat = y_time + gamma_sp * SpatialHead(H_sp)
gamma_sp 初始为 0
```

这样初始模型严格接近第三章时间模型。

# 10. HSTGCN-core 静态异构空间基准

## 10.1 原论文核心

HSTGCN：

- 地理图：基于区域间距离阈值；
- 需求图：基于区域需求序列 DTW 距离阈值；
- 两张图分别经过 GCN+GRU；
- 两个隐藏表示通过可学习 alpha 线性组合。

本方案只借用空间关系与双分支融合：

```text
Z_geo = GCN_stack(H_time, A_geo)
Z_dem = GCN_stack(H_time, A_DTW)
H_sp = alpha Z_geo + (1-alpha) Z_dem
```

不得把该适配版直接称为完整 HSTGCN。实验表中应写：

```text
HSTGCN-core / HSTGCN-style static dual graph
```

若单独重实现原始 GCN-GRU 版本，可另列为 HSTGCN baseline。

## 10.2 图构建

### UrbanEV

```text
A_geo：官方 adj.csv
A_DTW：仅用当前 fold 训练部分构造
```

建议先对每个节点计算训练集平均周模式：

```text
168 维（小时级）
```

标准化后计算 DTW，保留每个节点 Top-k 相似节点，主设置：

```text
k = 8
```

### CHARGED

- 一座城市一张独立图；
- 不把六座城市拼成一个图；
- A_geo：由官方 distance.csv 构造 KNN Gaussian 图；
- A_DTW：仅使用该城市训练部分的平均周需求模式；
- k 使用 `min(8, N_city-1)`。

### PEMS04/PEMS08

- A_geo：采用标准距离/邻接图；
- A_DTW：仅训练集，使用平均日模式或完整训练序列的低维摘要；
- 不使用验证/测试信息。

# 11. 空间模块 S1：SADR

## 11.1 论文出处与边界

来源：

> Wang et al., An Adaptive Spatio-Temporal Graph Recurrent Network for Short-Term Electric Vehicle Charging Demand Prediction, Applied Energy 383 (2025) 125320.

ASTGRN 正式版通过随时间变化的节点 embedding 和 embedding projection 生成自适应邻接 A_t。本文不复制其 GRU 或多 horizon 预测层，也不把动态构图本身作为原创点。

## 11.2 最终结构

基础可学习 embedding：

```text
E0: [N,d_a]
```

由 EL-AMD 状态产生样本级偏移：

```text
DeltaE_b = W_s H_time,b
E_b = E0 + gamma_e * LayerNorm(DeltaE_b)
```

相似度与稀疏化：

```text
S_b = ReLU(E_b E_b^T / sqrt(d_a))
M_b = symmetric_union_topk(S_b, k)
A_state,b = row_softmax(mask(S_b, M_b))
```

状态门控：

```text
lambda_b,n = sigmoid(MLP_lambda(H_time,b,n) + b_lambda)
b_lambda = -4
```

需求图残差融合：

```text
A_dem,b[n,:] =
    (1-lambda_b,n) * A_DTW[n,:]
  + lambda_b,n     * A_state,b[n,:]
```

推荐：

```text
d_a = 16
k = 8
gamma_e = 1e-3
```

## 11.3 与 ASTGRN 的差异

| ASTGRN | SADR |
|---|---|
| 动态 embedding 直接生成唯一自适应图 | 动态状态只修正需求关系分支 |
| 不依赖预定义图 | 保留 HSTGCN 的 DTW 长期先验 |
| Graph-GRU 统一时空建模 | 时间状态由 EL-AMD 提供 |
| 无显式 geo/demand 语义分解 | 固定地理图与需求图显式分开 |

论文贡献写法：

> 设计长期需求先验与短期状态关系的残差融合，而非重复提出一般意义上的动态图学习。

# 12. 空间模块 S2：SC-SimGCA

## 12.1 论文出处与借鉴点

来源：

> Jiang et al., An Electric Vehicle Charging Demand Prediction Approach Based on a Graph-based Spatio-Temporal Attention Network, Sustainable Energy, Grids and Networks 44 (2025) 101975.

G-STAN 的 Sim-GCA 包含：

- 多层 GCN；
- `conv(l)=(1-alpha)G(l)+alpha conv(l-1)`；
- 不同层输出 stack；
- SimAM 参数无关注意力。

仅把 alpha 换成 MLP gate 太薄，因此最终模块必须同时保留层融合与 SimAM。

## 12.2 最终结构

对关系分支 `r in {geo, demand}`：

```text
C_r^0 = H_time
G_r^1 = GCN_1(C_r^0, A_r)
rho_r^1 = sigmoid(MLP_r^1(H_time))
C_r^1 = (1-rho_r^1) G_r^1 + rho_r^1 C_r^0

G_r^2 = GCN_2(C_r^1, A_r)
rho_r^2 = sigmoid(MLP_r^2(H_time))
C_r^2 = (1-rho_r^2) G_r^2 + rho_r^2 C_r^1
```

层表示聚合：

```text
C_stack = concat(C_r^1, C_r^2, dim=feature)
```

Graph-SimAM：

```text
[B,N,2d] -> [B,2d,N,1]
parameter-free energy attention
-> [B,N,2d]
-> Linear(2d,d)
```

残差输出：

```text
H_r = H_time + gamma_r * GraphSimAM(C_stack)
gamma_r 初始为 0
```

地理分支与需求分支使用独立门控网络，不共享 `rho`。

## 12.3 与 G-STAN 的差异

| G-STAN | SC-SimGCA |
|---|---|
| 一张属性加权邻接图 | geo/demand 两个语义分支 |
| 全局固定 alpha | 样本—节点—关系条件化 rho |
| Res-TCN 输出 | EL-AMD 时间状态 |
| 原 CNN 特征图上的 SimAM | 适配图节点特征 `[B,N,d]` 的 Graph-SimAM |
| 完整 G-STAN 预测框架 | 仅抽取并改造空间模块 |

# 13. 第四章最终前向流程

```text
H_time, y_time = EL_AMD(X)

A_dem = SADR(H_time, A_DTW)

H_geo = SC_SimGCA(H_time, A_geo, relation='geo')
H_dem = SC_SimGCA(H_time, A_dem, relation='demand')

H_sp = alpha * H_geo + (1-alpha) * H_dem

y_hat = y_time + gamma_sp * Head(H_sp)
```

第一版保持 HSTGCN 的单个可学习 alpha；不要再叠加第三个复杂双图 gate。若 scalar fusion 明显受限，再把 `Concat+MLP` 作为次级消融，而非新的主要贡献。

# 14. 第四章实验数据集

## 14.1 必做数据集

| 数据集 | 正式任务 | 作用 |
|---|---|---|
| UrbanEV | 12 h 输入，t+3/t+6/t+9/t+12 单点；6 folds | 第一核心 EV 区域级数据 |
| CHARGED-AMS/JHB/LOA/MEL/SPO/SZH | 每座城市独立训练；12 h -> 下一小时；6 folds | 第二核心 EV 数据，多城市与站点级验证 |
| PEMS04 | 标准交通图时序协议 | 跨领域适用性 |
| PEMS08 | 标准交通图时序协议 | 跨领域适用性 |

CHARGED 不进行 UrbanEV checkpoint 零样本迁移；同一方法在每个城市独立重新训练。

## 14.2 UrbanEV 预测目标

正文完整主任务：

```text
volume
```

辅助实验：

| 目标/口径 | 实验范围 |
|---|---|
| volume-11kW | 最强时空 baseline、EL-AMD、Ours；用于估算口径稳健性 |
| occupancy | 3-4 个代表模型；用于目标语义稳健性 |
| duration | 时间充分时放附录 |

不能把三个目标称为三个数据集。

# 15. 第四章 Baseline 与消融

## 15.1 UrbanEV 完整 baseline

1. EL-AMD（无空间）；
2. GCN-LSTM；
3. ASTGCN；
4. HSTGCN / HSTGCN-core；
5. AGCRN；
6. ASTGRN；
7. STAEformer；
8. Ours。

说明：

- ASTGRN 是 S1 的直接来源，应作为强 baseline；
- G-STAN 若无官方代码，不作为毕业最低版本的强制完整 baseline；
- 若完整重实现成功，可加入扩展表。

## 15.2 CHARGED 六城市核心 baseline

1. EL-AMD；
2. GCN-LSTM；
3. HSTGCN-core；
4. ASTGRN 或 AGCRN；
5. STAEformer；
6. Ours。

六座城市形成一张表：

```text
Model | AMS | JHB | LOA | MEL | SPO | SZH | Avg Rank
```

## 15.3 PEMS04/08 baseline

1. GCN-LSTM；
2. AGCRN；
3. STAEformer；
4. HSTGCN-core adapted；
5. Ours。

## 15.4 空间消融

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
S4 vs S3：状态残差图是否有效
S5 vs S3：状态条件化 Sim-GCA 是否有效
S6 vs S4/S5：两个空间模块是否互补
```

# 16. 结果表与可视化

## 16.1 第三章主表

```text
Model | UrbanEV Avg MAE/Rank | ETTh1 | Weather | ECL | Exchange | Avg Rank
```

不同数据集指标不能直接平均，使用平均排名汇总。

UrbanEV 另设：

```text
Model | h=3 | h=6 | h=9 | h=12 | Avg
```

## 16.2 第四章主表

- UrbanEV：horizon 详细表；
- CHARGED：六城市横向表；
- PEMS04/08：标准交通结果表；
- 消融表；
- 参数/显存/时间表。

## 16.3 必做可视化

第三章：

- TEB 对电价、服务费、天气的注意力分布；
- PMCR 在峰值/突变时段的局部残差响应；
- AMD 与 EL-AMD 的峰值预测案例。

第四章：

- A_DTW、A_state、融合 A_demand 的热力图；
- 早高峰、晚高峰、深夜三类窗口的关系变化；
- lambda 的节点/时间分布；
- geo 与 demand 分支的层融合 rho；
- 典型住宅、商业、交通枢纽节点案例。

# 17. 数据泄漏与公平性清单

- scaler 只使用训练部分；
- 图只使用训练部分构建；
- DTW 图按 fold 单独缓存；
- node order 必须与图矩阵完全一致；
- 不输入未来真实天气；
- 所有模型使用同一变量、同一划分、同一指标；
- 超参数选择只看验证集；
- 测试集只在结构和参数冻结后运行；
- 报告 3 个随机种子或至少对 Ours/强 baseline 报告均值与标准差。

# 18. 代码目录

```text
models/
├── tsAMD.py
├── tsAMD_enhanced.py
├── modules/
│   ├── target_exogenous_bridge.py
│   └── modern_conv_refinement.py
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
├── feature_schema.py
├── graph_builder.py
└── result_logger.py

tests/
├── test_amd_equivalence.py
├── test_ddi_effective.py
├── test_module_shapes.py
├── test_module_gradients.py
├── test_target_offset.py
├── test_fold_scaler_no_leakage.py
├── test_graph_node_alignment.py
└── test_graph_train_only.py
```

# 19. Codex 执行里程碑

| 阶段 | 任务 | 完成标志 |
|---|---|---|
| M0 | 冻结 AMD、仓库审计、环境锁定 | baseline_audit.md + tag |
| M1 | AMDEnhanced 空壳与等价测试 | 模块关闭误差 <1e-6 |
| M2 | UrbanEV 多变量时间 DataLoader | 无空间泄漏与标签偏移测试通过 |
| M3 | 实现 PMCR | shape/gradient/零门控测试通过 |
| M4 | 实现 TEB | AMD-Concat 公平对照跑通 |
| M5 | 第三章筛选 | UrbanEV/Weather/ECL 验证结果 |
| M6 | 第三章最终实验 | 五数据集主表、消融、效率 |
| M7 | HSTGCN-core 与训练集 DTW 图 | S0-S3 跑通 |
| M8 | 实现 SADR | S4 与动态图可视化 |
| M9 | 实现 SC-SimGCA | S5 跑通 |
| M10 | 完整第四章 | UrbanEV + CHARGED + PEMS04/08 |

# 20. 模块失败时的降级路径

| 风险 | 允许的降级 |
|---|---|
| TEB 无收益 | 缩小辅助变量集合；将 MHA 改为 gated additive context；仍保留目标导向桥接 |
| PMCR 无收益 | k_large 从 7 改 5；减少 d；仅作用于目标通道；不得退化成无出处普通 Conv1d 后仍声称 ModernTCN 模块 |
| SADR 不稳定 | 增大 b_lambda 负值；加强 A_state 熵/稀疏正则；最终可退回 Static Dual Graph |
| SC-SimGCA 不稳定 | 保留状态层融合，暂时移除 Graph-SimAM；需在附录报告负结果 |
| 双图不互补 | 以验证结果选择单图，但如实报告；不强行保留双图 |
| 辅助变量全量输入掉点 | 按训练/验证集进行特征选择，不使用测试集选择 |

毕业最低版本仍需满足：

```text
AMD + 两个经过修改且经消融验证的时间模块
HSTGCN-core + 两个经过修改且经消融验证的空间模块
UrbanEV + 多个时间序列数据集 + CHARGED + PEMS04/08
```

# 21. 代码与复现难度

截至 2026-08-14：

| 论文 | 可核验官方代码 | 本方案使用难度 |
|---|---|---|
| AMD | https://github.com/TROUBADOUR000/AMD | 已复现 |
| TimeXer | https://github.com/thuml/TimeXer | TEB：低—中，2-4 个有效开发日 |
| ModernTCN | https://github.com/luodhhh/ModernTCN | PMCR：中，3-5 个有效开发日 |
| HSTGCN | 本次未检索到作者官方仓库 | HSTGCN-core：中，3-6 日；完整 GCN-GRU：约 1-2 周 |
| ASTGRN | 本次未检索到作者官方仓库 | graph learner：低—中，2-3 日；完整 baseline：5-8 日 |
| G-STAN | 本次未检索到作者官方仓库 | SC-SimGCA：低—中，2-4 日；完整模型：7-12 日 |

UrbanEV 与 CHARGED 均有官方数据/代码仓库：

```text
https://github.com/IntelligentSystemsLab/UrbanEV
https://github.com/IntelligentSystemsLab/CHARGED
```

# 22. 正式参考文献

[T0] Hu, Y., Liu, P., Zhu, P., Cheng, D., and Dai, T. Adaptive Multi-Scale Decomposition Framework for Time Series Forecasting. AAAI, 2025. Code: https://github.com/TROUBADOUR000/AMD

[T1] Wang, Y., Wu, H., Dong, J., et al. TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables. NeurIPS, 2024. Code: https://github.com/thuml/TimeXer

[T2] Luo, D., and Wang, X. ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis. ICLR Spotlight, 2024. Code: https://github.com/luodhhh/ModernTCN

[S0] Wang, S., Chen, A., Wang, P., and Zhuge, C. Predicting Electric Vehicle Charging Demand Using a Heterogeneous Spatio-Temporal Graph Convolutional Network. Transportation Research Part C, 153, 104205, 2023. DOI: 10.1016/j.trc.2023.104205

[S1] Wang, S., Li, Y., Shao, C., Wang, P., Wang, A., and Zhuge, C. An Adaptive Spatio-Temporal Graph Recurrent Network for Short-Term Electric Vehicle Charging Demand Prediction. Applied Energy, 383, 125320, 2025. DOI: 10.1016/j.apenergy.2025.125320

[S2] Jiang, D., Gong, X., Wei, Y., Peng, B., and Xu, Z. An Electric Vehicle Charging Demand Prediction Approach Based on a Graph-based Spatio-Temporal Attention Network. Sustainable Energy, Grids and Networks, 44, 101975, 2025. DOI: 10.1016/j.segan.2025.101975

[D1] Li, H., Qu, H., Tan, X., et al. UrbanEV: An Open Benchmark Dataset for Urban Electric Vehicle Charging Demand Prediction. Scientific Data, 12, 523, 2025. DOI: 10.1038/s41597-025-04874-4

[D2] Guo, Z., You, L., Zhu, R., et al. A City-scale and Harmonized Dataset for Global Electric Vehicle Charging Demand Analysis. Scientific Data, 12, 1254, 2025. DOI: 10.1038/s41597-025-05584-7

# 23. 最终一句话路线

```text
第三章：AMD + TimeXer-inspired TEB + ModernTCN-inspired PMCR
第四章：EL-AMD + HSTGCN-core + ASTGRN-inspired SADR + G-STAN-inspired SC-SimGCA
实验：第三章 UrbanEV/ETTh1/Weather/ECL/Exchange；第四章 UrbanEV/CHARGED六城市/PEMS04/PEMS08
```
