# ETTh1: AMD-mdm-u-to-ddi-v1 与 AMD 论文结果对比

生成日期：2026-08-13

## 1. 实验身份与口径

- 实现变体：`AMD-mdm-u-to-ddi-v1`
- Git 提交：`e6eed2e36b3d42788ce72c47eaa9c2da429b8627`
- 源码指纹：`14a5f542ebcc50a96f4233beefd433d120b0ca62a37c2a0f78a41fe7dcd073e5`
- 数据 SHA256：`f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`
- 随机种子：2025（单次运行）
- 设备：NVIDIA GeForce RTX 4070 SUPER；PyTorch 2.0.1，CUDA 11.8
- 任务：ETTh1，多变量预测多变量（M），输入长度 512，预测长度 96/192/336/720
- 公共脚本超参数：10 epochs、batch size 128、Adam、学习率 5e-5、patch 16、alpha 0、1 个 DDI block、RevIN 开启、`layernorm=True`
- 本实现指标：训练集 StandardScaler 空间中的全样本、全预测步、全通道逐元素 MSE/MAE；最终测试严格加载验证 MSE 最优 checkpoint。

论文来源为 arXiv:2406.03751v2。Table 7 报告五个随机种子的均值和一个未明确定义的 `±` 离散量，因此下文不把它擅自称为标准差或标准误。

## 2. 结果

| 预测长度 | 本次 MSE | 论文 MSE | MSE 差值 | 相对差值 | 本次 MAE | 论文 MAE | MAE 差值 | 相对差值 | 最优 epoch |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 96  | 0.370437 | 0.3691 ± 0.0008 | +0.001337 | +0.36% | 0.398979 | 0.3969 ± 0.0001 | +0.002079 | +0.52% | 9 |
| 192 | 0.400580 | 0.4008 ± 0.0007 | -0.000220 | -0.05% | 0.418060 | 0.4160 ± 0.0002 | +0.002060 | +0.50% | 7 |
| 336 | 0.418361 | 0.4177 ± 0.0005 | +0.000661 | +0.16% | 0.433457 | 0.4272 ± 0.0002 | +0.006257 | +1.46% | 4 |
| 720 | 0.462329 | 0.4389 ± 0.0009 | +0.023429 | +5.34% | 0.476149 | 0.4541 ± 0.0002 | +0.022049 | +4.86% | 2 |
| 四长度均值 | 0.412927 | 0.406625 | +0.006302 | +1.55% | 0.431661 | 0.423550 | +0.008111 | +1.92% | - |

结论：96/192/336 的 MSE 与论文均值非常接近（绝对差不超过 0.00134），主要差距集中在 720。720 的验证最优出现在 epoch 2，之后验证 MSE 从 1.4630 逐步恶化到 1.5177，说明当前实现/超参数在最长预测任务上较早进入过拟合或优化退化；它贡献了绝大多数四长度平均差距。

## 3. 为什么不能期待与论文逐值相同

### 3.1 当前变体仍不是论文描述的完整实现

本次只把模块连接改成论文算法中的 `X -> MDM -> U -> DDI`，AMS selector 也继续读取 U。其他算子仍冻结公开源码语义：

- 论文规定 ETTh1 的 DDI hidden dimension 为 `max(32, 2^ceil(log2(C)))`。ETTh1 有 7 个通道，所以论文值应为 32；当前公开代码规则为 `2^ceil(log2(C))`，实际只有 8。
- 论文 Table 6 写 `Layer Norm=True`；当前公开实现的同名开关实际使用展平后的 `BatchNorm1d`，不是 LayerNorm。
- 论文给出 weight decay `1e-7`；当前公开脚本和本变体固定为 `1e-9`。
- 论文算法令 selector 权重 `S` 具有 `m x T` 形状；当前公开 selector 先产生每个 expert 一个权重，再沿所有预测步复制，属于 horizon-shared gate。

这些差异会改变模型容量、归一化统计、正则强度和长预测步的专家组合，其中 DDI hidden 8 vs 32 和 horizon-shared selector 尤其可能放大 720 步预测差距。

### 3.2 论文是五随机种子，本次是一个随机种子

论文 Table 7 明确使用五个随机种子，本次正式对比仅使用 seed 2025。单次结果不能估计均值或方差，也不能据此做显著性判断。96/192/336 的接近程度说明主数据流程与量纲大体吻合，但 720 的约 5% 差距明显大于论文表中的 `±` 数字，不能仅用普通 seed 波动解释。

### 3.3 训练与评价细节并未在论文中完全定义

- 论文没有说明 Table 1/7 的 `±` 类型。
- 论文给出 MSE/MAE 公式，但没有说明跨 batch、样本、通道及尾批的完整累计方式，也没有明确结果是在标准化空间还是原始量纲。
- 当前 runner 使用全元素加权指标并纳入验证/测试尾批；公开原始 runner 按 batch 均值再平均，尾批权重不同。
- 当前 runner 严格选择验证 MSE 最优 epoch；公开原始 runner 的最后 epoch 条件会覆盖此前 best checkpoint，实际可能总是测试最后一轮。论文没有说明最终表格究竟采用哪一种代码路径。
- 论文说输入长度从 `{96,192,336,512,672,720}` 搜索，但没有披露 ETTh1 表格最终采用的长度；当前公开脚本固定为 512。

### 3.4 论文结果生成代码的拓扑来源无法证明

论文和算法明确写 DDI 输入 U，但作者公开代码基线原本让首个 DDI 读取 X。当前变体按用户要求改为 U，却无法从论文确认 Table 7 数值究竟来自论文所述拓扑、公开仓库拓扑，还是另一份未发布训练代码。因此当前结果应标为独立架构变体复现，而不是作者数值的逐位复刻。

### 3.5 硬件与确定性边界

论文使用 NVIDIA V100 32GB，本次使用 RTX 4070 SUPER。当前配置固定随机种子、cuDNN deterministic 并关闭 benchmark，但 PyTorch 未强制所有 deterministic algorithms，GPU 架构和底层 kernel 仍可能带来小幅数值差异。这可以解释微小偏差，通常不足以单独解释 720 上约 5% 的差距。

## 4. 运行经过与产物完整性

`SEEDS=2025 bash scripts/ETTh1.sh` 先完成 96 和 192。启动第三项时，本地 `data/ETTh1.csv` 路径出现 Windows `PermissionError`，脚本按 `set -e` 安全停止；两个已完成 run 未受影响。同时检测到另一个用户启动的 iTransformer 任务占用 `cuda:0`，因此等待其结束，没有并发争抢 GPU。

随后使用与脚本完全相同的模型、优化器和数据参数补跑 336/720，只把 `--data` 指向 G 盘中经 Git blob SHA 和文件 SHA256 校验、内容完全相同的 ETTh1 副本。四个 run 的 manifest、metrics、config、best.pt、last.pt 和 history.jsonl 均完整且一致。

四个 run 的训练活动时间合计 201.77 秒，即约 3 分 22 秒；实际墙钟时间因上述权限中断和等待另一训练任务而更长。产物约 1.06 GiB。

## 5. 完整实验耗时与容量估计

- ETTh1 单 seed、4 个预测长度：本机实测约 3.4 分钟，建议预留 4-6 分钟。
- 9 个长期预测数据集、单 seed、36 个 run：中心估计约 33 GPU 小时，合理区间约 24-50 小时。
- 9 个数据集、5 seeds、180 个 run：中心估计约 165 GPU 小时，即连续约 6.9 天；考虑大数据集吞吐和系统占用，建议按 5-10 天规划。
- 预计产物：约 9.7 GB/seed、约 48.5 GB/5 seeds。D 盘当前空间不足以安全保留完整五 seed，正式全量训练应使用 G 盘。

## 6. 后续建议

1. 先完成同一提交下 5 seeds 的 ETTh1，报告均值、样本标准差和逐 seed 结果。
2. 单独建立更严格的 paper-operator variant：DDI hidden 最小 32、真正 LayerNorm、weight decay 1e-7、horizon-specific selector；不要覆盖当前 `AMD-mdm-u-to-ddi-v1`。
3. 对 720 做受控消融：只改变一个差异项，并保持数据 SHA、seed 和评价器不变。优先顺序为 DDI hidden、selector horizon 维、归一化、weight decay。
4. 所有全量训练写入 G 盘独立 campaign root，避免不同源码指纹和旧 artifacts 混在同一汇总目录。

## 7. 可复核文件

- 论文 PDF：`tmp/pdfs/amd-paper.pdf`
- 新 run 汇总：`summaries/etth1_comparison_2025/AMD-mdm-u-to-ddi-v1.csv`
- 聚合汇总：`summaries/etth1_comparison_2025/AMD-mdm-u-to-ddi-v1-aggregate.csv`
- 原始产物根目录：`artifacts/AMD-mdm-u-to-ddi-v1/ETTh1/`

