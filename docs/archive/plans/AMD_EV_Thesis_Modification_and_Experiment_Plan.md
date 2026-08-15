---
title: "[已归档] AMD 基准模型魔改与城市 EV 充电需求预测实验落地方案"
author: "面向服务器 Codex 开发的实施文档"
date: "2026-08-14"
---

> **归档声明（2026-08-15）：** 本文档已失效，仅用于保留历史决策记录，不得再参与实现、实验或论文写作决策。唯一权威方案为 `docs/AMD_EV_Thesis_Final_Implementation_Plan_v2.1.md`。

# 1. 文档定位与最终锁定结论

## 1.1 当前基线

- 基准仓库：`https://github.com/tlxx22/AMD`
- 已成功复现分支：`AMD-paper-norm-wd-ddi-v1`
- 上游论文：Hu et al., **Adaptive Multi-Scale Decomposition Framework for Time Series Forecasting**, AAAI 2025。
- 上游 AMD 的核心结构为：

```text
RevIN -> MDM -> DDI -> AMS -> Forecast
```

本方案不再更换时间骨干，也不再采用“只抽取 MDM”的 RA-MDM 路线。第三章以你已复现成功的完整 AMD 为唯一时间基准，在其上增加两个来自近三年论文、且经过任务化修改的模块；第四章以 HSTGCN 类地理图—需求图异构空间结构为基准，再加入两个经过修改的近期空间模块。

## 1.2 最终模块组合

| 章节 | 基准 | 新增模块 1 | 新增模块 2 | 最终作用 |
|---|---|---|---|---|
| 第三章纯时间模型 | AMD（AAAI 2025） | TimeXer 启发的目标—辅助变量桥接模块 TAB | ModernTCN 启发的峰值感知局部残差模块 PALR | 提升外生信息利用与局部突变建模 |
| 第四章时空模型 | HSTGCN 类静态异构双图 | ASTGRN 启发的状态残差需求图 SDRG | G-STAN 启发的状态条件化图层融合 SCGF | 在稳定先验上动态修正需求关系，并自适应控制图传播深度 |

最终结构固定为：

```text
第三章：AMD + TAB' + PALR'
第四章：第三章时间模型 + HSTGCN-core + SDRG' + SCGF'
```

其中 `'` 表示没有原样照搬论文模块，而是针对当前 AMD 接口、EV 区域级任务和论文递进关系进行了修改。

## 1.3 明确排除的主线方案

- 不使用 `AMD + Pathformer Router`：完整 AMD 已有 AMS 对时间模式和预测器进行自适应选择，继续加入多尺度 Router 功能重叠，消融解释困难。
- 不把 ASTGRN 直接作为第四章母模型：ASTGRN 已经逐时刻学习动态图，难以清晰体现“基准 + 两个新增模块”的增量；它更适合做强 baseline 和模块来源。
- 不直接把 UrbanEV 的 275 个区域当作 AMD 的 275 个变量通道：这会使 DDI 在第三章提前混合区域信息，破坏“无空间传播”的实验边界。
- 不用测试集决定是否保留模块；所有结构选择仅根据训练/验证集完成。

# 2. 开发前必须完成的 Codex 仓库审计

由于你的分支包含 norm、weight decay、DDI 等复现修复，开发前必须冻结当前成功结果，禁止直接覆盖基线文件。

## 2.1 Git 操作

```bash
git checkout AMD-paper-norm-wd-ddi-v1
git status
git log -1 --oneline

git tag amd_reproduced_baseline_v1
git checkout -b thesis/temporal-modules-v1
```

若未配置上游仓库：

```bash
git remote add upstream https://github.com/TROUBADOUR000/AMD.git
git fetch upstream
```

让 Codex 执行：

```bash
git diff upstream/main...HEAD -- \
  models/tsAMD.py models/common.py models/tsmoe.py \
  main.py utils/dataloader.py
```

并生成：

```text
docs/baseline_audit.md
```

审计文档至少记录：

1. 当前分支相对上游 AMD 的所有行为变化；
2. 当前成功复现使用的数据集、配置、随机种子和结果；
3. `norm`、`layernorm`、weight decay、最佳模型保存、验证集加载、Solar 读取和 DDI 的实际修复状态；
4. `seq_len=12` 时 DDI 的 `patch` 是否会退化。

## 2.2 DDI 的特殊风险检查

上游 DDI 的循环从 `n_history * patch` 开始。如果 `seq_len=12` 且 `patch=12`，循环不会进入，DDI 基本只复制输入。你的分支名包含 `ddi-v1`，因此不能假设仍是上游实现，必须用单元测试确认。

Codex 应增加：

```python
# tests/test_ddi_effective.py
# 对 seq_len=12、patch=3/4/6/12 分别检查：
# 1. 输出是否与输入完全相同；
# 2. DDI 参数是否获得非零梯度；
# 3. 前向和反向是否稳定。
```

若当前分支仍与上游逻辑一致，UrbanEV 的主设置采用：

```text
seq_len = 12
patch = 3
```

敏感性只测 `patch in {3, 4, 6}`，不再使用 12。

# 3. 代码目录与开发隔离

建议新增文件，不修改已复现基准的核心行为：

```text
models/
├── tsAMD.py                         # 已复现 AMD，禁止改行为
├── common.py                        # 仅在确有必要时追加公共层
├── tsmoe.py                         # 原 AMS，主线不改
├── tsAMD_enhanced.py                # 第三章最终时间模型
├── modules/
│   ├── __init__.py
│   ├── target_aux_bridge.py         # TAB，TimeXer-inspired
│   └── peak_local_residual.py       # PALR，ModernTCN-inspired
└── spatial/
    ├── __init__.py
    ├── graph_conv.py                # 纯 PyTorch 稀疏图卷积
    ├── static_dual_graph.py         # HSTGCN-core
    ├── state_residual_graph.py      # SDRG，ASTGRN-inspired
    ├── state_conditioned_fusion.py  # SCGF，G-STAN-inspired
    └── ev_spatiotemporal_model.py

utils/
├── dataloader.py                    # 原 AMD loader，尽量不改
├── dataloader_urbanev.py
├── dataloader_charged.py
├── dataloader_graph.py
├── feature_schema.py
├── graph_builder.py
└── result_logger.py

configs/
├── temporal/
├── spatial/
└── datasets/

scripts/
├── run_temporal_screen.sh
├── run_temporal_final.sh
├── run_spatial_screen.sh
└── run_spatial_final.sh

tests/
├── test_amd_equivalence.py
├── test_module_shapes.py
├── test_no_temporal_spatial_leakage.py
├── test_fold_scaler_no_leakage.py
├── test_graph_node_alignment.py
└── test_sparse_dense_consistency.py

docs/
├── baseline_audit.md
├── data_protocol.md
└── experiment_registry.md
```

建议新增独立入口：

```text
main_temporal.py
main_spatial.py
```

不要把所有新逻辑继续塞进原 `main.py`，否则很容易破坏已经复现的 AMD 命令和结果。

# 4. 第三章纯时间模型：数据接口

## 4.1 “纯时间”的严格定义

第三章允许同一个区域内部的多变量交互，但禁止不同区域之间的消息传递。

UrbanEV 原始张量建议组织为：

```text
X: [B, T, N, C]
```

第三章训练时有两种等价实现。推荐第一种。

### 推荐：数据集层面展开区域

在按时间划分 train/val/test 之后，将区域展开为样本：

```text
一个样本 = 某一时间窗口 + 某一区域
x: [T, C]
y: [1] 或 [pred_len]
```

DataLoader 输出：

```text
[B_region, T, C]
```

优点：

- 完全兼容现有 AMD `[B,T,C]` 接口；
- 不需要在模型内部处理 `B*N`；
- 第三章不会发生跨区域传播；
- 可直接复用现有训练循环。

必须先按时间切分，再展开区域，禁止先生成所有窗口后随机切分。

### 第四章图数据接口

第四章才保留完整节点维度：

```text
X_graph: [B, T, N, C]
```

时间编码器内部临时 reshape：

```text
[B,T,N,C] -> [B*N,T,C]
```

得到区域状态后再恢复：

```text
[B*N,d] -> [B,N,d]
```

## 4.2 UrbanEV 第一版动态输入

主预测目标：`volume`。

第一版建议使用：

```text
目标通道：
- volume

辅助动态通道：
- e_price
- s_price
- weather_central 中经核验的数值气象变量
- hour_sin, hour_cos
- weekday_sin, weekday_cos
- is_weekend
```

天气字段必须依据 `weather_header.txt` 实际解析。优先选温度、湿度、气压/降水等缺失率低、语义清楚的字段，第一版不超过 5 个天气通道。

暂不作为主输入：

- occupancy；
- duration；
- POI、区域面积、桩数量等静态属性；
- 未来真实天气；
- 区域 ID embedding。

原因：occupancy 和 duration 本身也是可预测充电状态，第一版加入会使问题从“辅助因素增强 volume 预测”变成多任务/多状态联合预测。静态属性更适合第四章或附加消融。

## 4.3 数据泄漏规则

- 所有 scaler 仅用当前 fold 的训练部分拟合；
- `e_price`、`s_price`、天气在主实验只使用历史窗口内的真实值；
- 目标时刻的 hour/day/calendar 可以使用，因为它们部署时天然已知，但必须让所有 baseline 获得相同信息；
- 不允许用验证集或测试集构造需求图、均值周曲线、PCA、归一化参数或缺失填充值。

# 5. 第三章模块 T1：Target-Auxiliary Bridge（TAB）

## 5.1 论文出处

来源思想：

> Wang et al., **TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables**, NeurIPS 2024.

TimeXer 使用内生目标的全局 token 作为桥梁，通过变量级 cross-attention 有方向地吸收外生变量信息。本方案不引入完整 Transformer，只借用“目标查询辅助变量”的桥接思想。

## 5.2 为什么 AMD 仍需要 TAB

AMD 的 DDI 可以混合变量，但其通道交互主要是统一的通道映射；它没有明确区分：

```text
预测目标 volume
与
辅助因素 price / service / weather / calendar
```

TAB 将交互改成目标导向：由目标状态主动查询辅助变量，而不是把所有变量完全等价地混在一起。

## 5.3 改造后的模块

输入：

```text
h: [B, C, T]        # DDI 输出
h_mdm: [B, C, T]    # MDM 输出，可选
```

步骤：

1. 将每个变量的时间序列映射为变量 token：

```text
z_c = Linear_T_to_D(LayerNorm(h_c))
Z = [z_1, ..., z_C] -> [B,C,D]
```

2. UrbanEV 中：

```text
Q = z_volume
K,V = z_price, z_service, z_weather, z_calendar
```

3. 标准多变量数据集中，为保持原 AMD 的多变量到多变量任务：

```text
Q = 所有目标变量 token
K,V = 所有变量 token
```

使用对角 mask，避免某变量只复制自身信息。

4. Cross-attention：

```text
context = MHA(Q, K, V)
```

5. 将 context 投影回时间维度，只对目标通道做残差调制：

```text
delta = Linear_D_to_T(context)
h_target_new = h_target + gamma_tab * delta
```

建议：

```text
D = 32
heads = 4
gamma_tab 初始值 = 1e-3
dropout = 0.1
```

## 5.4 与原 TimeXer 的差异

| TimeXer | 本方案 TAB |
|---|---|
| Transformer 主骨干 | AMD 主骨干 |
| patch-level self-attention | 不引入，继续使用 MDM/DDI |
| 全局内生 token + 变量级 cross-attention | 保留思想，压缩为单个轻量桥接层 |
| 独立预测头 | 不使用，仍由 AMD 的 AMS 预测 |
| 通用外生预测 | UrbanEV 中显式 target/aux；标准数据中采用共享目标—辅助桥接 |

因此不能在论文中写“加入 TimeXer”，应写：

> 受 TimeXer 中全局内生表示桥接外生变量机制启发，设计了适配 AMD 的轻量目标—辅助变量桥接模块。

## 5.5 必做公平对照

```text
AMD-V       : 仅目标变量
AMD-Concat  : 目标 + 辅助变量直接作为普通通道输入
AMD+TAB     : 与 AMD-Concat 使用完全相同的变量，但增加目标导向桥接
```

只有 `AMD+TAB` 优于 `AMD-Concat`，才能证明收益来自模块而不只是来自更多输入。

# 6. 第三章模块 T2：Peak-Aware Local Residual（PALR）

## 6.1 论文出处

来源思想：

> Luo and Wang, **ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis**, ICLR 2024 Spotlight.

ModernTCN 通过现代化的大核卷积结构扩大有效感受野，并保持卷积模型的效率。本方案不复制完整 ModernTCN，只保留大小核 depthwise temporal convolution 的局部模式补偿思想。

## 6.2 解决的问题

AMD 的 MDM 使用平均池化形成粗粒度尺度，这有利于趋势和周期，但可能弱化：

- 突发充电尖峰；
- 峰谷快速切换；
- 局部高频波动；
- 非周期异常需求。

PALR 作为并行残差支路，补偿这些局部信息。

## 6.3 结构

输入使用 RevIN 后的序列：

```text
x_norm: [B,C,T]
```

训练结构：

```text
r_small = DWConv1d(k_small)(x_norm)
r_large = DWConv1d(k_large)(x_norm)
r = PointwiseConv(GELU(r_small + r_large))
h_new = h + gamma_palr * r
```

建议主参数：

| 场景 | k_small | k_large |
|---|---:|---:|
| UrbanEV / CHARGED，T=12 | 3 | 7 |
| ETTh1 / Weather / ECL / Exchange | 3 | 25 或 31，且不得超过序列长度 |

其他参数：

```text
groups = C
dropout = 0.1
gamma_palr 初始值 = 1e-3
```

## 6.4 与原 ModernTCN 的差异

- 不采用完整多 stage backbone；
- 不采用 ModernTCN 自己的变量 FFN 和预测头；
- 不替换 AMD；
- 只将大小核局部卷积改成 AMD 的轻量残差补偿器；
- UrbanEV 的短窗口采用小得多的有效大核，避免核长超过输入长度。

论文表述：

> 受 ModernTCN 大感受野卷积建模思想启发，设计峰值感知双核局部残差分支，以补偿多尺度平均混合可能造成的局部突变信息损失。

# 7. 第三章最终模型的代码插入点

上游 AMD 的核心流程为：

```python
x = rev_norm(x, 'norm')
x = x.transpose(1, 2)             # [B,C,T]
time_embedding = MDM(x)
for ddi in DDI_blocks:
    x = ddi(x)
pred, moe_loss = AMS(x, time_embedding)
```

建议新建 `AMDEnhanced`，内部复用已复现 AMD 的模块，不复制参数定义。

伪代码：

```python
class AMDEnhanced(nn.Module):
    def __init__(self, base_amd, cfg):
        super().__init__()
        self.base = base_amd
        self.tab = TargetAuxiliaryBridge(...)
        self.palr = PeakAwareLocalResidual(...)
        self.state_proj = nn.Sequential(
            nn.Linear(cfg.seq_len * 2 + cfg.tab_dim, cfg.state_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.state_dim),
        )

    def forward(self, x, *, target_idx=None, aux_idx=None,
                use_tab=True, use_palr=True, return_state=False):
        x_norm = self.base.rev_norm(x, 'norm') if self.base.norm else x
        x_ch = x_norm.transpose(1, 2)                 # [B,C,T]

        h_mdm = self.base.pastmixing(x_ch)
        h = x_ch
        for block in self.base.fc_blocks:
            h = block(h)

        if use_palr:
            h = h + self.palr(x_ch, target_idx)

        tab_context = None
        if use_tab:
            h, tab_context = self.tab(h, target_idx, aux_idx)

        pred, moe_loss = self.base.moe(h, h_mdm)
        pred = pred.transpose(1, 2)

        if self.base.norm:
            pred = self.base.rev_norm(pred, 'denorm', self.base.target_slice)
        if self.base.target_slice:
            pred = pred[:, :, self.base.target_slice]

        if not return_state:
            return pred, moe_loss

        state = build_state(h, h_mdm, tab_context)
        return pred, moe_loss, state
```

第一版不要修改 AMS 的 gating 输入，仍使用原 `h_mdm`。这样可以隔离模块作用，降低训练风险。

# 8. 基线保持与单元测试

## 8.1 等价性测试

当两个模块关闭时，新模型必须与已复现 AMD 完全一致：

```python
base.eval()
enhanced.eval()
with torch.no_grad():
    y0, l0 = base(x)
    y1, l1 = enhanced(x, use_tab=False, use_palr=False)
assert torch.max(torch.abs(y0 - y1)) < 1e-6
assert torch.max(torch.abs(l0 - l1)) < 1e-6
```

## 8.2 必做测试

1. `test_amd_equivalence.py`：关闭模块时完全等价；
2. `test_module_shapes.py`：M、MS、S 三种 target_slice 形状正确；
3. `test_module_gradients.py`：TAB/PALR 参数和 residual gate 均获得梯度；
4. `test_no_temporal_spatial_leakage.py`：交换 DataLoader 中区域样本顺序不影响对应单样本输出；
5. `test_fold_scaler_no_leakage.py`：验证/测试时间戳不参与 scaler；
6. `test_target_offset.py`：UrbanEV 的模型 `pred_len=1`，标签偏移由 `h` 决定；
7. `test_resume_checkpoint.py`：新模型 checkpoint 能完整恢复。

# 9. 第三章实验数据集与协议

## 9.1 必做数据集

| 数据集 | 角色 | 协议 | 主指标 |
|---|---|---|---|
| UrbanEV | 核心 EV 应用 | 过去 12 h，分别预测 t+3/t+6/t+9/t+12 单点；6 folds | MAE、RMSE、RAE、WAPE |
| ETTh1 | 标准多变量时序 | 完全沿用已复现 AMD 配置；pred_len 96/192/336/720 | MSE、MAE |
| Weather | 多变量气象与波动 | 沿用 AMD 配置；96/192/336/720 | MSE、MAE |
| Electricity/ECL | 电力负荷与高维变量 | 沿用 AMD 配置；96/192/336/720 | MSE、MAE |
| Exchange | 低维、非能源分布 | 沿用 AMD 配置；96/192/336/720 | MSE、MAE |

可选第六数据集：Solar。只有前五个全部完成后再补。

## 9.2 UrbanEV 特殊标签规则

对每个 horizon `h in {3,6,9,12}` 独立训练：

```text
x = data[t : t+12]
y = volume[t+12+h-1]
```

AMD 模型内部：

```text
pred_len = 1
```

禁止将 AMD 的 `pred_len` 设置为 3/6/9/12，否则会错误地变成多步序列输出。

## 9.3 第三章 baseline

主表至少包括：

1. DLinear；
2. PatchTST；
3. iTransformer；
4. TimeMixer；
5. TimeXer；
6. ModernTCN；
7. AMD；
8. AMD + TAB + PALR（Ours）。

UrbanEV 另加：

- Last Observation；
- LSTM。

所有模型在同一数据集上使用相同可用变量。UrbanEV 主比较中，除 `volume-only` 对照外，所有深度模型均获得同一组历史辅助变量；模型差异只能来自结构。

## 9.4 第三章消融

| 编号 | 结构 | 目的 |
|---|---|---|
| T0 | AMD-V | 仅 volume 基线 |
| T1 | AMD-Concat | 加入同样辅助变量但不使用 TAB |
| T2 | AMD-Concat + TAB | 验证目标—辅助桥接 |
| T3 | AMD-Concat + PALR | 验证局部残差 |
| T4 | AMD-Concat + TAB + PALR | 最终时间模型 |

完整消融只在：

```text
UrbanEV + Weather + ECL
```

完成。ETTh1 和 Exchange 只跑最终主表与 AMD/TAB/PALR 关键版本。

## 9.5 运行规模控制

### 筛选阶段

- UrbanEV：fold 6，h=3 和 h=12，seed=2025；
- Weather：pred_len=96；
- ECL：pred_len=96；
- 每个模块只搜索预先规定的 4 组以内配置。

### 结构锁定阶段

- UrbanEV：fold 5 和 6，h=3/6/12，2 seeds；
- Weather/ECL：96 和 336，2 seeds。

### 最终阶段

- UrbanEV：6 folds × 4 horizons；
- 标准数据集：全部 4 个 pred_len；
- AMD、TimeXer、ModernTCN、Ours 至少 3 seeds；
- 其他 baseline 可先 1 seed，若结果接近再补 3 seeds。

# 10. 第三章结果表模板

## 10.1 多数据集平均排名表

| Model | UrbanEV Avg MAE | ETTh1 Avg MSE | Weather Avg MSE | ECL Avg MSE | Exchange Avg MSE | Avg Rank |
|---|---:|---:|---:|---:|---:|---:|

不同数据集指标不能直接求数值平均，应使用平均排名或相对提升。

## 10.2 UrbanEV 详细表

| Model | h=3 MAE | h=6 MAE | h=9 MAE | h=12 MAE | Avg MAE | Params |
|---|---:|---:|---:|---:|---:|---:|

## 10.3 模块消融表

| Model | TAB | PALR | MAE | RMSE | WAPE | Params | Peak GPU |
|---|---:|---:|---:|---:|---:|---:|---:|

## 10.4 辅助变量消融

| Input | AMD-Concat | Ours |
|---|---:|---:|
| Volume only | | |
| + Calendar | | |
| + Price + Service | | |
| + Weather | | |
| All historical features | | |

# 11. 第三章向第四章输出的时间状态

第四章不能使用最终预测值代替时间状态。建议在 AMS 之前构造：

```text
h_target      : DDI + TAB + PALR 后的目标通道序列 [B*N,T]
h_mdm_target  : MDM 目标通道序列 [B*N,T]
tab_context   : TAB 上下文 [B*N,D]
```

拼接并投影：

```text
state = MLP([h_target, h_mdm_target, tab_context])
state: [B*N, d_state]
```

主设置：

```text
d_state = 64
```

第四章恢复：

```text
H_time: [B,N,64]
```

第三章仍用原 AMS 输出预测；第四章使用同一预测作为基础预测，再由图模型产生空间残差：

```text
y_final = y_temporal + gamma_spatial * delta_y_spatial
```

这样关闭空间模块时，第四章严格退化回第三章最终时间模型。

# 12. 第四章空间基准：HSTGCN-core

## 12.1 论文出处

> Wang et al., **Predicting Electric Vehicle Charging Demand Using a Heterogeneous Spatio-Temporal Graph Convolutional Network**, Transportation Research Part C, 2023.

HSTGCN 使用：

```text
A_geo + A_demand_static
```

其中地理图由距离阈值构造，需求图由需求序列 DTW 相似度构造，两分支分别提取表示后做加权融合。

本方案只借用其异构地理—需求关系骨干，不照搬原 GRU 和 region-specific prediction 模块，因为时间建模已由第三章完成。

## 12.2 静态基准结构

```text
Z_geo = GCN(H_time, A_geo)
Z_dem = GCN(H_time, A_dtw)
alpha = softmax([a_geo, a_dem])
Z_sp = alpha_geo * Z_geo + alpha_dem * Z_dem
delta_y = Linear(Z_sp)
y_final = y_temporal + gamma_spatial * delta_y
```

建议：

```text
gamma_spatial 初始值 = 0
```

# 13. 静态图构造

## 13.1 UrbanEV

- `A_geo`：优先使用官方 `adj.csv`；
- `distance.csv` 用于核验节点顺序和替代 KNN 实验；
- `A_dtw`：仅用当前 fold 的训练部分计算；
- 主设置 `k_geo=8`、`k_demand=8`；
- 275 节点允许计算完整 pairwise DTW，但仍应缓存每 fold 图文件。

## 13.2 CHARGED

每座城市是一张独立图，禁止把六座城市拼成一个 4280 节点图。

- `A_geo`：由每城 `distance.csv` 生成 KNN 高斯图；
- `A_dtw`：先用训练集生成每个 site 的平均周模式（168 维）；
- 先用 Pearson/cosine 筛选每节点 32 个候选；
- 只对候选计算 DTW；
- 最终保留 top-8。

这一近似是工程可行性处理，论文必须明确说明，而不能声称与 HSTGCN 的全量 DTW 完全相同。

## 13.3 PEMS04 / PEMS08

- 使用选定 baseline 仓库的标准邻接矩阵；
- 节点顺序以数据文件为唯一准则；
- 需求图仍只基于训练数据构造；
- 图文件保存 `node_ids`、`edge_index`、`edge_weight`、数据 hash 和 fold。

# 14. 空间模块 S1：State-conditioned Demand Residual Graph（SDRG）

## 14.1 论文出处

来源思想：

> Wang et al., **An Adaptive Spatio-Temporal Graph Recurrent Network for Short-Term Electric Vehicle Charging Demand Prediction**, Applied Energy, 2025.

ASTGRN 通过随充电需求演变的节点 embedding，在每个时间步生成自适应邻接矩阵。

## 14.2 本方案的修改

不使用 ASTGRN 的独立动态图 embedding，也不完全丢弃 HSTGCN 的长期需求先验。

使用第三章状态：

```text
Q = H_time W_Q
K = H_time W_K
score_ij = Q_i K_j / sqrt(d_a)
```

在候选边集合上生成：

```text
A_state = row_softmax(topk(score))
```

再与静态需求先验融合：

```text
lambda_i = sigmoid(MLP_lambda(H_time_i) + bias_lambda)
A_demand_t(i,:) = (1-lambda_i) * A_dtw(i,:) + lambda_i * A_state(i,:)
```

主参数：

```text
d_a = 16
candidate_k = 32
final_top_k = 8
bias_lambda = -4
```

`bias_lambda=-4` 使初始 lambda 约为 0.018，训练开始时模型几乎等于静态 HSTGCN-core。

## 14.3 与 ASTGRN 的差异

| ASTGRN | 本方案 SDRG |
|---|---|
| 独立动态 node embedding | 直接使用第三章 AMD 时间状态 |
| 动态图替代预定义图 | 保留长期 DTW 需求先验 |
| 单个 latent adaptive graph | 显式地理分支 + 需求分支 |
| Graph-GRU 时空耦合 | 时间与空间递进解耦 |

# 15. 空间模块 S2：State-conditioned Graph Fusion（SCGF）

## 15.1 论文出处

来源思想：

> Jiang et al., **An Electric Vehicle Charging Demand Prediction Approach Based on a Graph-based Spatio-temporal Attention Network**, Sustainable Energy, Grids and Networks, 2025.

G-STAN 的 Sim-GCA 使用多层 GCN 和融合连接，以固定比例控制前层特征与深层图卷积特征的混合。

## 15.2 本方案的修改

对地理分支和需求分支分别计算：

```text
Z1_r = GCN1(H_time, A_r)
Z2_r = GCN2(Z1_r, A_r)
rho_r(i) = sigmoid(MLP_r(H_time_i))
Z_r = Z2_r + gamma_r * rho_r(i) * (Z1_r - Z2_r)
```

其中：

```text
r in {geo, demand}
gamma_r 初始值 = 0
```

这将 G-STAN 的固定层融合比例修改为：

- 样本相关；
- 节点相关；
- 图关系类型相关；
- 由第三章时间状态驱动。

不在第一版加入完整 SimAM，以避免同时改变图注意力、层融合和时间模块，导致消融无法定位。

# 16. 第四章最终模型

```text
输入 X [B,T,N,C]
        |
        v
第三章 AMD+TAB+PALR
        |
        +--> y_temporal [B,H,N]
        |
        +--> H_time [B,N,d]
                    |
         +----------+----------+
         |                     |
      A_geo             A_dtw + SDRG
         |                     |
      GCN1/2                GCN1/2
         |                     |
      SCGF_geo             SCGF_dem
         +----------+----------+
                    |
        HSTGCN-style branch fusion
                    |
             delta_y_spatial
                    |
y_final = y_temporal + gamma_spatial * delta_y_spatial
```

# 17. 图计算的可扩展实现

CHARGED 某些城市节点较多，禁止对所有 batch 构造完整 `[B,N,N]` 动态矩阵。

主实现采用稀疏候选边：

```text
candidate edges = union(
  geographic top-k,
  static demand top-k,
  training-correlation top-k
)
```

只在候选边上计算 QK 分数，复杂度从：

```text
O(B*N^2)
```

降为：

```text
O(B*N*K_candidate)
```

`graph_conv.py` 优先用纯 PyTorch `index_add_`/`scatter_reduce_`，避免额外依赖已停止维护的 PyG Temporal。必须增加小图下稀疏与稠密结果一致性测试。

# 18. 第四章数据集与协议

## 18.1 必做数据集

| 数据集 | 协议 | 角色 |
|---|---|---|
| UrbanEV | 12 h 输入，h=3/6/9/12 单点，6 folds | 第一主 EV 数据集 |
| CHARGED 六城市 | 每城独立训练；12 h 输入预测下一小时；8:1:1、6 folds | 第二主 EV 数据集与多城市验证 |
| PEMS04 | 标准 12->12 交通流预测 | 跨领域适用性 |
| PEMS08 | 标准 12->12 交通流预测 | 跨领域适用性 |

CHARGED 正文只采用官方 `h=1`，不额外强制 3/6/9/12。

## 18.2 UrbanEV 预测目标

正文完整实验只做：

```text
volume
```

辅助稳健性：

1. `volume-11kW`：必做，检验充电量估算口径；
2. `occupancy`：只跑时间模型、最强时空 baseline、Ours；
3. `duration`：时间和算力充足再放附录。

三个目标属于同一数据集，不能当作三个数据集报告。

# 19. 第四章 baseline

## 19.1 UrbanEV 主表

1. LSTM；
2. EL-AMD（第三章最终时间模型）；
3. GCN-LSTM；
4. ASTGCN；
5. Graph WaveNet；
6. AGCRN；
7. HSTGCN-core；
8. ASTGRN-Reimpl；
9. STAEformer；
10. Ours。

## 19.2 CHARGED 六城市

控制工作量，保留：

1. EL-AMD；
2. GCN-LSTM；
3. HSTGCN-core；
4. AGCRN；
5. ASTGRN-Reimpl；
6. Ours。

## 19.3 PEMS04 / PEMS08

1. GCN-LSTM；
2. Graph WaveNet；
3. AGCRN；
4. STAEformer；
5. EL-AMD；
6. Ours。

G-STAN 完整模型没有必要作为最低必做 baseline；本章通过 S2 消融验证其来源思想的改造。若后续找到可靠官方代码，再加入扩展表。

# 20. 第四章消融

| 编号 | 时间模型 | 静态双图 | SDRG | SCGF | 目的 |
|---|---|---:|---:|---:|---|
| S0 | EL-AMD | 否 | 否 | 否 | 纯时间基准 |
| S1 | EL-AMD | 是 | 否 | 否 | HSTGCN-core 空间基准 |
| S2 | EL-AMD | 是 | 是 | 否 | ASTGRN 来源模块增量 |
| S3 | EL-AMD | 是 | 否 | 是 | G-STAN 来源模块增量 |
| S4 | EL-AMD | 是 | 是 | 是 | 最终时空模型 |

补充图消融：

```text
Geo only
Static demand only
Geo + static demand
Geo + state demand only
Geo + static demand + state residual
```

完整 S0-S4 消融在：

```text
UrbanEV + CHARGED-SZH + CHARGED-AMS
```

完成。CHARGED 其他四城和 PEMS 只跑 S0、S1、S4 与强 baseline。

# 21. 模块“至少不变差”的验收规则

无法在训练前数学保证测试集一定提升，但可以预先设置非劣门槛和基线保持结构。

## 21.1 时间模块

TAB 或 PALR 单独通过需满足：

1. UrbanEV 筛选设置的平均验证 MAE 相对 `AMD-Concat` 不劣于 0.5%；
2. 在 UrbanEV、Weather、ECL 三个验证任务中至少两个改善；
3. 参数量增幅不超过 AMD 的 20%；
4. 不出现训练不稳定、NaN 或显存增长超过 30%。

## 21.2 空间模块

SDRG 或 SCGF 单独通过需满足：

1. 相对 S1 的验证 MAE 不劣于 0.5%；
2. UrbanEV 至少两个 horizon 改善；
3. CHARGED-SZH 或 AMS 至少一个改善；
4. 动态图无大面积孤立节点，平均有效度数稳定。

## 21.3 允许的有限调参范围

```text
TAB: D in {16,32}, heads in {2,4}
PALR: kernels in {(3,5),(3,7)} for T=12
SDRG: final_top_k in {8,16}, bias_lambda in {-4,-2}
SCGF: hidden in {16,32}, gamma init in {0,1e-3}
```

达到预设上限后仍不通过，不允许看 test 后无限修改结构。应记录为失败候选，再启用备用模块，而不是隐藏负结果。

# 22. Codex 里程碑任务单

## M0：冻结基线与审计

交付：

```text
docs/baseline_audit.md
tests/test_baseline_reproduction.py
```

验收：能用原命令复现当前结果，且记录 commit、环境、数据 hash。

## M1：AMDEnhanced 空壳与等价性

交付：

```text
models/tsAMD_enhanced.py
tests/test_amd_equivalence.py
```

验收：关闭模块时输出误差 `<1e-6`。

## M2：UrbanEV 纯时间数据管线

交付：

```text
utils/dataloader_urbanev.py
configs/datasets/urbanev_temporal.yaml
tests/test_target_offset.py
```

验收：时间切分、区域展开、标签 h 和 scaler 均无泄漏。

## M3：PALR

先做 PALR，因为不依赖辅助变量数据管线。

交付：

```text
models/modules/peak_local_residual.py
```

验收：形状、梯度、基线保持和 smoke train 通过。

## M4：TAB

交付：

```text
models/modules/target_aux_bridge.py
utils/feature_schema.py
```

验收：AMD-Concat 与 AMD+TAB 使用完全相同输入，TAB 能单独开关。

## M5：第三章筛选和最终实验

交付：

```text
results/chapter3/*.jsonl
results/chapter3/summary.csv
```

验收：完成 T0-T4、五数据集主表和效率统计。

## M6：第四章静态双图

交付：

```text
utils/graph_builder.py
models/spatial/static_dual_graph.py
```

验收：S1 相比 S0 在至少一个主验证设置上产生合理空间增益；图节点顺序严格一致。

## M7：SDRG

交付：

```text
models/spatial/state_residual_graph.py
```

验收：lambda 初始接近 0；关闭后严格退化为静态需求图；稀疏计算通过。

## M8：SCGF 与完整模型

交付：

```text
models/spatial/state_conditioned_fusion.py
models/spatial/ev_spatiotemporal_model.py
```

验收：S0-S4 可独立配置，关闭空间残差后严格等于第三章预测。

## M9：第四章最终实验与可视化

交付：

```text
results/chapter4/
figures/graph_snapshots/
figures/error_by_region/
```

至少生成：

- 早高峰/深夜状态需求图对比；
- 静态 DTW 图与动态修正图差异；
- 各区域 MAE 地图；
- lambda 和 rho 分布；
- 参数量、显存和推理时间表。

# 23. 建议新增的命令行参数

```text
--model AMD | AMDEnhanced | EVST
--dataset UrbanEV | ETTh1 | Weather | ECL | Exchange | CHARGED | PEMS04 | PEMS08
--task temporal | spatial
--target volume
--seq_len
--pred_len
--label_horizon
--fold
--seed

--use_tab
--tab_dim
--tab_heads
--tab_gamma_init

--use_palr
--palr_kernel_small
--palr_kernel_large
--palr_gamma_init

--return_state
--state_dim

--use_static_dual_graph
--use_sdrg
--use_scgf
--k_geo
--k_demand
--candidate_k
--state_graph_dim
--lambda_bias
--spatial_gamma_init
```

所有实验配置保存为 YAML，并在结果文件中记录完整展开后的配置。

# 24. 结果记录与复现要求

每一次运行至少保存：

```text
run_id
git_commit
branch
dataset_version
data_sha256
fold
horizon
seed
model
module_flags
hyperparameters
best_epoch
val_metrics
test_metrics
parameter_count
peak_gpu_memory
train_time
inference_time
checkpoint_path
```

主结果中：

- 报告均值和标准差；
- 最终模型与最强 baseline 做配对检验或至少按 fold/seed 报告胜负次数；
- 不只报告最佳 seed；
- 不使用跨数据集绝对误差平均，使用平均排名或相对提升。

# 25. 推荐开发顺序

```text
1. 冻结 AMD 基线
2. 建立 AMDEnhanced 等价接口
3. 完成 UrbanEV 时间数据管线
4. 插入 PALR
5. 插入 TAB
6. 完成第三章筛选与五数据集实验
7. 固定第三章最终时间模型
8. 输出 H_time 并建立静态双图
9. 插入 SDRG
10. 插入 SCGF
11. UrbanEV 完整时空实验
12. CHARGED 六城市
13. PEMS04/08
14. volume-11kW、occupancy、缺失鲁棒性与可视化
```

禁止在第三章时间模型未稳定前同时开发空间模块。

# 26. 参考文献与模块出处

1. Hu, Y., Liu, P., Zhu, P., Cheng, D., & Dai, T. Adaptive Multi-Scale Decomposition Framework for Time Series Forecasting. *Proceedings of the AAAI Conference on Artificial Intelligence*, 2025. DOI: 10.1609/aaai.v39i16.33908.
2. Wang, Y., Wu, H., Dong, J., et al. TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables. *NeurIPS*, 2024.
3. Luo, D., & Wang, X. ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis. *ICLR*, 2024.
4. Wang, S., Chen, A., Wang, P., & Zhuge, C. Predicting Electric Vehicle Charging Demand Using a Heterogeneous Spatio-Temporal Graph Convolutional Network. *Transportation Research Part C*, 153, 104205, 2023. DOI: 10.1016/j.trc.2023.104205.
5. Wang, S., Li, Y., Shao, C., et al. An Adaptive Spatio-Temporal Graph Recurrent Network for Short-Term Electric Vehicle Charging Demand Prediction. *Applied Energy*, 383, 125320, 2025. DOI: 10.1016/j.apenergy.2025.125320.
6. Jiang, D., Gong, X., Wei, Y., Peng, B., & Xu, Z. An Electric Vehicle Charging Demand Prediction Approach Based on a Graph-based Spatio-temporal Attention Network. *Sustainable Energy, Grids and Networks*, 44, 101975, 2025. DOI: 10.1016/j.segan.2025.101975.
7. Li, H., Qu, H., Tan, X., et al. UrbanEV: An Open Benchmark Dataset for Urban Electric Vehicle Charging Demand Prediction. *Scientific Data*, 12, 523, 2025. DOI: 10.1038/s41597-025-04874-4.
8. Guo, Z., You, L., Zhu, R., et al. A City-scale and Harmonized Dataset for Global Electric Vehicle Charging Demand Analysis. *Scientific Data*, 12, 1254, 2025. DOI: 10.1038/s41597-025-05584-7.

# 27. 最终验收清单

- [ ] 原 AMD 分支和结果保持可复现；
- [ ] 新模型关闭模块时与 AMD 数值等价；
- [ ] TAB 和 PALR 分别有独立开关、独立消融和论文来源；
- [ ] UrbanEV 时间模型没有跨区域传播；
- [ ] UrbanEV `pred_len=1` 与 `label_horizon` 分离；
- [ ] 第三章完成 5 个数据集和至少 7 个 baseline；
- [ ] HSTGCN 静态地理图与需求图均只使用训练信息；
- [ ] SDRG 和 SCGF 分别有独立消融；
- [ ] CHARGED 六城市分别训练，不拼成跨洲图；
- [ ] CHARGED 动态图使用稀疏候选边；
- [ ] 第四章完成 UrbanEV、CHARGED、PEMS04、PEMS08；
- [ ] 主任务完整跑 volume，补 volume-11kW；
- [ ] 所有结构选择只看验证集；
- [ ] 结果记录 commit、数据 hash、fold、seed 和完整配置；
- [ ] 论文不宣称发明双图或首次提出动态图，而准确表述为对已有模块的状态条件化改造。
