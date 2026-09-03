# M4：时间模块诊断与候选迭代

状态：In Progress

开始日期：2026-08-28（UTC）

当前轮次：第二十轮，CrossLinear-inspired Late CCE production capability 与 paired development 实验

canonical 内部版本：v2.1-R1

## 1. 阶段范围与禁止事项

M4 只处理第三章时间模块的训练/验证诊断、证据边界澄清和经用户另行授权的来源保持型候选迭代。M4 当前不执行模型筛选或结构冻结，M5-M13 尚未开始。

本阶段不得提前实现 StateAdapter、StateProjection、`H_time`、Graph Mode、HSTGCN-core、地理图或需求图、SADR、SC-SimGCA 及任何其他空间模块。ETTm1 自第五轮起是 development-only benchmark，可使用 train/validation/test 开发候选；UrbanEV、EPF-PJM、ETTh1、Weather、ECL、Exchange 的 test 仍不得用于 M4/M5 结构或超参数选择。任何单数据集、单 seed、10 epoch 诊断均不得写成正式性能验收。

M0、M1、M2、M3 均保持 Closed，不重新打开，也不向其 milestone 追加 M4 结果。

## 2. 继承状态

| 项目 | 继承值 |
|---|---|
| branch | `AMD-paper-repro-custom-modules-v1` |
| M4 起始 HEAD | `5fcac2010afe0154c900a3992de5b9d3803a2737` |
| HEAD parent | `ac4f2c2a98f8a5c4c3cafe48b36b9c2e68451b75` |
| baseline tag | `amd_reproduced_baseline_v1 -> fa9665627e6fcfb1d0c2bc22d943ca9666304fd6` |
| canonical 更新前 SHA-256 | `d3a2d480454cd5d0c38d6d29ef2dcb043d90f3b77fd9e568b55186dd476cd5da` |
| canonical 第三轮候选身份 repair 后 SHA-256 | `be3e051ffb1b680f0c3566427b41394760a5e9219d533804e1e49e6c69f533e3` |
| canonical 第五轮 ETTm1 development 协议提交后 SHA-256 | `33a45667270dfab37ac2137ecf96873a42e96215bd6a91c132cf153d41b4edac` |
| M0 milestone SHA-256 | `e2a20131664391752340e92a9d9a5302b0078cac48d7dcdbb4a4841a16f62cdd` |
| M1 milestone SHA-256 | `bcd9b3e3d821a1cf609423a3e8b58ecb4129bc202238e9fe1f8d7cf3a361c70b` |
| M2 milestone SHA-256 | `8a4343e054f9dacd9cd623b930b07376a55e507f2d9085e56c634323b6dbda54` |
| M3 milestone SHA-256 | `be3ff6606a52c9668ca4f1c2d47aaf9771be1e25e69b8fe991031f7fb9c192ed` |
| source fingerprint | `961efb5a54742db972b55b6556a2dc2e05939e580b2c9e61b595f3ff9983f330`；算法 `sha256_length_prefixed_relative_path_and_content_v1`；17 个 Python 源文件 |
| M3 closure 回归 | `136/136 passed`，failed=0，skipped=0（继承证据；本轮未重跑） |

M4 第一轮只读 validation 诊断结束时，AMD 顶层、UrbanEV、ModernTCN 和 TimeXer 工作区均为 clean；HEAD、tracking 与 baseline tag 无变化。

## 3. 用户最新阶段重排

旧 canonical 将 M4 定义为 StateAdapter / Graph Mode，与用户最新阶段决定冲突。自本轮起唯一有效顺序为：

```text
M4  时间模块诊断与候选迭代
M5  模型筛选与结构冻结
M6  第三章正式实验与定稿
M7  时间状态接口与 Graph Mode
M8  HSTGCN-core 与双图构建
M9  SADR 状态需求残差图
M10 SC-SimGCA 状态条件图传播
M11 第四章正式实验与定稿
M12 论文正文、图表与结果分析
M13 终稿审校、复现材料与答辩
```

canonical 已同步该顺序。M0-M3 milestone 保留其历史现场，不因后续重排而修改。

## 4. pre-M4 test 与 ETTm1 development 登记

现有 pre-M4 ETTm1 smoke 包含 4 类模型 × 4 个 horizon、单训练 seed、10 epoch。其 test 结果在 M4 建立前已经被查看，并触发了 M4 时间模块诊断。

用户在第五轮将 ETTm1 重新指定为 M4 development-only benchmark。其 train、validation 和 test 均可继续用于 M4 候选结构、容量与超参数开发；这些 test 结果不进入 M6 正式主表，也不再解释为未见测试集泛化结果。pre-M4 test 不再被登记为“永久禁止参与后续决策”。

UrbanEV、EPF-PJM、ETTh1、Weather、ECL、Exchange 的 test 继续隔离：M4/M5 只允许 train/validation 参与选择，M6 在 M5 结构冻结后才使用 test。本决定只适用于 ETTm1。

## 5. 第一轮 artifact、源码与协议审计

- 16 个科学身份均唯一：AMD、AMD + PMCR、AMD + TEB、AMD + PMCR + TEB，各含 horizon 96/192/336/720。
- 12 个增强模型 schema-v2 artifact 的受控 checksum 集全部通过。
- 本轮使用的 4 个 AMD schema-v1 run 在当前 clean HEAD 上生成，通过 source gate B；其 legacy 合同不要求 schema-v2 文件集。
- 旧 M0 ETTm1 artifact 被单独识别并排除，没有静默择一或用历史 run 补表。
- 16 个 `best.pt` 均由与各自 config 完全相同的模型结构以 `strict=True` 恢复；missing、unexpected 和 shape mismatch 均为空。
- 数据、split、scaler、seed、epoch、batch、优化器、主干参数、损失和 validation 聚合无未解释 protocol confounder；允许差异仅为 PMCR/TEB 开关、对应模块参数和 ablation identity。

## 6. 共享初始化公平性补核

### 6.1 逐 run 初始化来源

| Horizon | 模型 | run_id | 初始化 | source_kind | source checkpoint SHA-256 | resume |
|---:|---|---|---|---|---|---|
| 96 | AMD | `20260824T142010.470888Z-1483456b` | from scratch | N/A | N/A | false |
| 192 | AMD | `20260824T142326.363972Z-63dcec3d` | from scratch | N/A | N/A | false |
| 336 | AMD | `20260824T142743.219070Z-b4b0d8c4` | from scratch | N/A | N/A | false |
| 720 | AMD | `20260824T143345.017095Z-8a0eb04d` | from scratch | N/A | N/A | false |
| 96 | AMD + PMCR | `20260824T142510.894757Z-41da0bd8` | from scratch | N/A | N/A | false |
| 192 | AMD + PMCR | `20260824T143116.292410Z-c15d9daa` | from scratch | N/A | N/A | false |
| 336 | AMD + PMCR | `20260824T143727.819123Z-c21ba8ec` | from scratch | N/A | N/A | false |
| 720 | AMD + PMCR | `20260824T144259.209542Z-7b7c6cc5` | from scratch | N/A | N/A | false |
| 96 | AMD + TEB | `20260824T142537.384319Z-4335edd3` | from scratch | N/A | N/A | false |
| 192 | AMD + TEB | `20260824T143148.209808Z-f28f563e` | from scratch | N/A | N/A | false |
| 336 | AMD + TEB | `20260824T143802.584750Z-69fb044b` | from scratch | N/A | N/A | false |
| 720 | AMD + TEB | `20260824T144322.792887Z-e7ec961a` | from scratch | N/A | N/A | false |
| 96 | AMD + PMCR + TEB | `20260824T140559.764397Z-6e0e796c` | from scratch | N/A | N/A | false |
| 192 | AMD + PMCR + TEB | `20260824T140936.093161Z-590e6a2e` | from scratch | N/A | N/A | false |
| 336 | AMD + PMCR + TEB | `20260824T141248.387312Z-774377bd` | from scratch | N/A | N/A | false |
| 720 | AMD + PMCR + TEB | `20260824T141609.954041Z-a68e8be6` | from scratch | N/A | N/A | false |

4 个 AMD schema-v1 run 没有 `command.txt`，本轮采用其 `config.resolved.json` 还原等价 argv；12 个增强 run 使用原生 `command.txt`。所有 config 的 `run.resume` 均为 `null`，没有 `resume_invocations`；command 与 best/last checkpoint metadata 均无 source checkpoint、`source_kind` 或 resume 来源。故 source/resume checkpoint 路径和 SHA-256 均为 N/A，而不是缺失后被猜测。

### 6.2 runner 顺序与公共 AMD state parity

runner 在构造 DataLoader 前调用统一 `set_seed(seed=2024)`，依次设置 Python、NumPy、CPU torch 与可用 CUDA RNG；随后创建独立 train generator 并构造 DataLoader，最后调用模型 factory。`AMDEnhanced` 内部先通过 `AMD.__init__` 构造完整 AMD 主干，再按开关实例化 PMCR，最后按开关实例化 TEB。

本轮用每个 run 的真实配置、相同 runner seed 流程与模型 factory 在 `/tmp` 重建训练起始模型；不执行 forward、训练或评价。每个 horizon 的公共 AMD state 包含 60 个 key，其中 57 个 parameter key、3 个 persistent-buffer key。核验覆盖 key 名、类别、shape、dtype 和 tensor value：

| 对比 | 跨 4 horizon 最大绝对误差 | 不一致 key |
|---|---:|---|
| AMD vs AMD + PMCR | 0 | 无 |
| AMD vs AMD + TEB | 0 | 无 |
| AMD vs AMD + PMCR + TEB | 0 | 无 |

结论：`shared AMD initialization parity: Passed`。

证据边界：该结论证明当前同源码、同真实配置和同 runner 初始化顺序可逐元素重建公共 AMD 初值；它不是仅依据“相同 seed”的推断，也不声称重建了训练过程中的全部随机轨迹。

## 7. RNG 与 validation 确定性修正

AMD 论文公式包含 noisy selector，但当前实际 `TopKGating` 源码只在 `training=True` 时采样噪声。`eval()` validation 不消费 selector RNG；PMCR/TEB dropout 在 eval 下关闭。

16 个 run 的同 batch 正式 forward 重复误差均为：prediction `0`、MoE loss `0`。统一 validation 复算与各自 history 中 best validation 的绝对误差均为 `0`。因此不得写“历史 best selection 受到随机 AMS validation 影响”。

M0 的 CPU/CUDA RNG 保存与恢复仍作为防御性数值等价合同保留，以防未来源码或其他随机组件改变；这不改变当前 eval 路径确定性的事实。

## 8. Validation 结果

下表均为 ETTm1、train-standardized、全部 7 变量、单训练 seed、10 epoch 的 validation-only 诊断；不是 M5 筛选结果，也不是论文正式性能表。

### 8.1 MSE

| Horizon | AMD | AMD + PMCR | AMD + TEB | AMD + PMCR + TEB |
|---:|---:|---:|---:|---:|
| 96 | 0.37007348 | 0.36860838 | 0.37602077 | 0.37303565 |
| 192 | 0.49616223 | 0.49529328 | 0.49283903 | 0.49386202 |
| 336 | 0.64766912 | 0.64662651 | 0.65069948 | 0.64236317 |
| 720 | 0.95571715 | 0.95557406 | 0.95184801 | 0.95277967 |
| Macro mean | 0.61740550 | 0.61652556 | 0.61785182 | 0.61551013 |

### 8.2 MAE

| Horizon | AMD | AMD + PMCR | AMD + TEB | AMD + PMCR + TEB |
|---:|---:|---:|---:|---:|
| 96 | 0.40250664 | 0.40204834 | 0.40477602 | 0.40378113 |
| 192 | 0.46346370 | 0.46302766 | 0.46199174 | 0.46303697 |
| 336 | 0.52858624 | 0.52797802 | 0.53072787 | 0.52775674 |
| 720 | 0.63946526 | 0.64010678 | 0.63921554 | 0.63958094 |
| Macro mean | 0.50850546 | 0.50829020 | 0.50917779 | 0.50853895 |

### 8.3 两种相对统计

`mean_horizon_relative_change` 是先逐 horizon 计算相对 AMD 变化，再对 4 个 horizon 等权平均：

| 模型 | MSE | MAE |
|---|---:|---:|
| PMCR | -0.1867% | -0.0557% |
| TEB | +0.2501% | +0.1531% |
| Full | -0.1974% | +0.0214% |

`relative_change_of_macro_means` 是先计算四 horizon 指标的算术宏平均，再相对 AMD 宏平均计算变化：

| 模型 | MSE | MAE |
|---|---:|---:|
| PMCR | -0.1425% | -0.0423% |
| TEB | +0.0723% | +0.1322% |
| Full | -0.3070% | +0.0066% |

两种统计口径不得统称为同一个 “Macro relative”。负值表示误差降低，正值表示误差升高。

## 9. PMCR v1 诊断结论

- `gamma_pmcr` 已显著离开 `1e-3` 初值。
- validation gated residual 幅度约为主表示的 2.6%-23.3%，尾部分布无明显爆炸。
- 两类 PMCR-enabled checkpoint、四个 horizon 共 8 个同-checkpoint 对照中，PMCR on 均优于 PMCR bypass（8/8）。这只证明各联合训练 checkpoint 已依赖 PMCR，不等于独立训练模型的正式增益。
- 独立训练的 AMD + PMCR 相对 AMD validation 收益仍非常小。
- “PMCR 普遍因残差过弱而无效”不受当前证据支持。
- PMCR v1 仍只是工程候选，尚未通过最终性能验收。

## 10. Global TEB v1 诊断结论

- `gamma_teb` 已显著离开 `1e-3` 初值；gated residual 约为主表示的 7%-11%。
- 可用 attention 诊断显示分布并非完全均匀，effective keys 约为 3.5-4/6；attention 非均匀不等于信息有效，也不构成因果或特征重要性结论。
- TEB-only validation 在四个 horizon 中两好两坏。
- Full checkpoint 内，保留 TEB 相对 no-TEB bypass 的 validation 方向为有利；这只属于同一联合训练 checkpoint 的反事实，不能替代独立训练比较。
- T→d 全局压缩瓶颈与 parallel 跨变量噪声均未被当前证据证明或排除。
- h336 Full 的 `need_weights=True` attention 重建未通过 dtype-aware 内部等价门禁，因此该 run 的 attention entropy、变量权重和 top-2 统计不可用；不依赖 attention weights 的正式 residual/gamma 诊断仍有效。
- Global TEB v1 仍只是工程候选，尚未通过最终性能验收。

## 11. 组合交互结论

联合训练 Full checkpoint 的四状态反事实显示：h96 为描述性不利交互，h192/h336 为描述性有利交互，h720 近似加性。当前证据不支持“PMCR 与 TEB 存在稳定负交互”。

交互量只描述同一联合训练 checkpoint 内的旁路结果，不等同于独立训练模型间的正式交互效应，也不构成因果结论。

## 12. H1-H7 修正版

| 假设 | 当前判定 | 证据边界 |
|---|---|---|
| H1：10 epoch 训练不足 | Mixed / horizon-dependent | h96/h192 弱支持训练预算可能不足；h336 混合；h720 不支持统一延长训练。不得据此决定新 epoch。 |
| H2：PMCR v1 实际残差过弱 | Not supported | gamma、gated residual 与 8/8 on/bypass 不支持“普遍过弱”；独立训练收益仍小。 |
| H3：PMCR v1 残差过强或方向不稳定 | Not supported | 未观察到明显爆炸尾部，旁路方向一致；不等于已通过性能验收。 |
| H4：Global TEB 存在 T→d 全局压缩瓶颈 | Not testable with current evidence | 结构压缩事实存在，但没有受控替代结构证据。 |
| H5：Parallel TEB 引入跨变量噪声 | Not testable / Inconclusive | attention 非均匀不能证明有效或噪声；TEB-only 方向两好两坏，且一项 attention 重建不可用。不得写 `Not supported`。 |
| H6：PMCR 与 TEB 存在负交互 | Not supported as a stable adverse interaction | 同-checkpoint 交互符号随 horizon 改变。 |
| H7：现有证据不足以区分训练问题与结构问题 | Supported | 单数据集、单 seed、10 epoch 且缺少受控候选对照，无法完成归因。 |

## 13. 当前身份与未作决定

- PMCR v1：M2 已闭环的工程候选，不是最终模块。
- Global TEB v1：M3 已闭环的工程候选，不是最终模块。
- EL-AMD：第三章增强时间模型的项目名称（模型族名称），不等同于任何具体候选 variant。
- `el-amd-pmcr-teb-v1`：M3 时点 PMCR v1 + Global TEB v1 的工程组合候选 variant，不是最终正式 variant。
- 最终内部结构和正式 implementation variant 只能在 M5 冻结。
- 本轮没有决定任何新 kernel、hidden/context dimension、heads、dropout、learning rate、epoch、候选结构或替代论文来源。
- Patch-conditioned TEB 仅为 M4 可评估的来源保持型方向；实现必须由用户另行授权。
- practical-effect threshold 尚未锁定；0.5% 仅为安全底线，不能单独决定保留。
- 最终第三章仍须满足至少两个近三年来源模块的硬性要求。

## 14. 当前门禁与下一步

M4 状态保持 `In Progress`。当前已完成第一轮 validation 诊断、第二轮共享初始化公平性补核与阶段文档建立，以及第三轮 canonical 候选身份残余冲突 repair。

第二轮共享初始化补核与阶段文档建立没有重新评价模型；第一轮只读 validation 复算及内部诊断已经记录在本 milestone 第 5、7-12 节。第三轮只做文档一致性 repair，同样不重新评价模型。

第三轮未训练、未读取或复述 test 数值、未实现或选择候选、未创建新 variant、未进入 M5、未实现 M7 或任何空间模块、未 commit/push。后续任何候选实现、训练或筛选均需用户另行授权。

## 15. 第四轮只读 runner 审计

第四轮逐函数核验了当前 production runner 的真实路径：

```text
train
-> validation
-> validation MSE 选择 best checkpoint
-> 重新加载 best.pt
-> 无条件 test
-> schema-v2 completed artifact
```

当前不存在不访问 test Dataset/DataLoader/evaluation 的 validation-only 安全路径；schema-v2 completed artifact 与正式 summarizer 也要求 test 字段。用户决定不实现 validation-only runner、独立 validation-only schema、summarizer 或 artifact purpose，因为 ETTm1 已转为 development-only 数据集。该决定不得扩展到正式评价数据集。

## 16. 第五轮 Stage A：P1/T1 容量 sanity 预登记

第五轮 Stage A 固定使用 ETTm1、`parallel_multivariate`、全部 7 个变量、`seq_len=512`、seed 2024、10 epochs，以及 horizon 96/192/336/720。实验使用现有 production runner 和完整 schema-v2 artifact；ETTm1 test 在本阶段是 development 指标，不是正式论文结果。

P1 固定为 `P1_PMCR_CAP16`：

```text
PMCR v1 architecture
hidden_dim: 8 -> 16
kernel_small/kernel_large: 5/31
dropout: 0.1
TEB: off
```

T1 固定为 `T1_GLOBAL_TEB_D64_H4`：

```text
Global TEB v1 architecture
context_dim: 32 -> 64
heads: 4（head_dim: 8 -> 16）
dropout: 0.1
PMCR: off
```

P1/T1 仅改变 v1 architecture 的一个容量参数，不创建新 class 或新 implementation variant，也不实现 P2/T2/T3。其余 AMD、数据、split/scaler、优化、seed、训练预算、输出与 best-checkpoint 合同必须逐字段复用对应 P0/T0 completed artifact。M4 状态继续为 `In Progress`。

## 17. 第五轮协议文档 Git closure

第五轮实验启动前，ETTm1 development-only 协议与 P1/T1 预登记已通过一个 docs-only commit 闭环：

- commit：`adad7fee12688769c53bf5736c7d1fcc3bb33c6e`
- parent：`57aa772c873f73bce53bbccdc35361bf5d0a935c`
- title：`docs(m4): designate ETTm1 development benchmark`
- push：已推送至 `origin/AMD-paper-repro-custom-modules-v1`；实验启动前 local、tracking 与远端 branch HEAD 一致，ahead/behind 为 `0/0`，工作区 clean。
- 提交范围：仅 canonical 与本 milestone。

闭环后的固定合同是：ETTm1 的 train/validation/test 可用于 M4 development；其 test 已参与候选开发反馈，不能作为 M6 未见测试集结果、正式主表结果或无偏泛化主张。六个正式评价数据集的 test 仍受 M5/M6 边界保护。本轮未实现 validation-only runner、独立 schema、summarizer 或 artifact purpose。

## 18. P0/T0 对照与 P1/T1 artifact

所有 run 均为 ETTm1、`parallel_multivariate`、全部 7 变量、`seq_len=512`、seed 2024、10 epochs、train-standardized metric space。数据 SHA-256 均为 `6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e`；source fingerprint 均为 `961efb5a54742db972b55b6556a2dc2e05939e580b2c9e61b595f3ff9983f330`。每个科学身份均只有一个可接受的 completed run；精确 13-file checksum 集和系统 `sha256sum -c` 均通过。P0/T0 直接复用，不重新训练；hidden staging 未被接受或用于择优。

### 18.1 P0：PMCR v1，`d=8`

| H | run_id | config hash | 绝对路径 | checksum |
|---:|---|---|---|---|
| 96 | `20260824T142510.894757Z-41da0bd8` | `9fee6de64f01868f8c77aeb6788267ae320aa93ec8a160181c2f61dcd980690b` | `/public/home/yueweiting/大论文/AMD/artifacts/manual-smoke/pmcr-only/el-amd-pmcr-teb-v1/ETTm1/parallel_multivariate/all/horizon_96/fold_official/seed_2024/20260824T142510.894757Z-41da0bd8` | 13/13 passed |
| 192 | `20260824T143116.292410Z-c15d9daa` | `7b179e8f3b66fbd4528b069964c9e33b68400a66f766cf09a1b70579140950a2` | `/public/home/yueweiting/大论文/AMD/artifacts/manual-smoke/pmcr-only/el-amd-pmcr-teb-v1/ETTm1/parallel_multivariate/all/horizon_192/fold_official/seed_2024/20260824T143116.292410Z-c15d9daa` | 13/13 passed |
| 336 | `20260824T143727.819123Z-c21ba8ec` | `33ffddb0b32d2efa446b5f5972d8c381c46be58c6600ec9e93246a5c433ab68e` | `/public/home/yueweiting/大论文/AMD/artifacts/manual-smoke/pmcr-only/el-amd-pmcr-teb-v1/ETTm1/parallel_multivariate/all/horizon_336/fold_official/seed_2024/20260824T143727.819123Z-c21ba8ec` | 13/13 passed |
| 720 | `20260824T144259.209542Z-7b7c6cc5` | `0495a53447e8741e968abcf67981e9037ee35cd0ea49badd9aae5abfbdd61545` | `/public/home/yueweiting/大论文/AMD/artifacts/manual-smoke/pmcr-only/el-amd-pmcr-teb-v1/ETTm1/parallel_multivariate/all/horizon_720/fold_official/seed_2024/20260824T144259.209542Z-7b7c6cc5` | 13/13 passed |

### 18.2 T0：Global TEB v1，`d=32, heads=4`

| H | run_id | config hash | 绝对路径 | checksum |
|---:|---|---|---|---|
| 96 | `20260824T142537.384319Z-4335edd3` | `cb5ee9b612f4652ad4dc63c63b9b721af0dd89abfd679dcf0aeef8738efc1da7` | `/public/home/yueweiting/大论文/AMD/artifacts/manual-smoke/teb-only/el-amd-pmcr-teb-v1/ETTm1/parallel_multivariate/all/horizon_96/fold_official/seed_2024/20260824T142537.384319Z-4335edd3` | 13/13 passed |
| 192 | `20260824T143148.209808Z-f28f563e` | `bf97ebdeeaa9e797c166c512e2de6120bb738ad909d66e77571ec935e9d64f56` | `/public/home/yueweiting/大论文/AMD/artifacts/manual-smoke/teb-only/el-amd-pmcr-teb-v1/ETTm1/parallel_multivariate/all/horizon_192/fold_official/seed_2024/20260824T143148.209808Z-f28f563e` | 13/13 passed |
| 336 | `20260824T143802.584750Z-69fb044b` | `40382f721963d108f049898b97f1fc121ceb2976c0ba154845ddf3ca5d328fb2` | `/public/home/yueweiting/大论文/AMD/artifacts/manual-smoke/teb-only/el-amd-pmcr-teb-v1/ETTm1/parallel_multivariate/all/horizon_336/fold_official/seed_2024/20260824T143802.584750Z-69fb044b` | 13/13 passed |
| 720 | `20260824T144322.792887Z-e7ec961a` | `9e5cca4a964d9067bddf66c008885adf0c9ae7eecf1234bebf5ff269508888b4` | `/public/home/yueweiting/大论文/AMD/artifacts/manual-smoke/teb-only/el-amd-pmcr-teb-v1/ETTm1/parallel_multivariate/all/horizon_720/fold_official/seed_2024/20260824T144322.792887Z-e7ec961a` | 13/13 passed |

### 18.3 P1：`P1_PMCR_CAP16`

| H | run_id | config hash | best epoch | 绝对路径 | checksum |
|---:|---|---|---:|---|---|
| 96 | `20260828T130529.230907Z-ee88487b` | `baef766dd2cb742b047ec23ad09f772336da375c02b325348bc8a8a9da7f09e8` | 10 | `/public/home/yueweiting/大论文/AMD/artifacts/m4-development/ettm1-stage-a-capacity-v1/el-amd-pmcr-teb-v1/ETTm1/parallel_multivariate/all/horizon_96/fold_official/seed_2024/20260828T130529.230907Z-ee88487b` | 13/13 passed |
| 192 | `20260828T130916.916853Z-493ea9bf` | `64a959872f5463147d96cb0841c94ae22d1eedab2168aaaec33cc78b15f143ce` | 10 | `/public/home/yueweiting/大论文/AMD/artifacts/m4-development/ettm1-stage-a-capacity-v1/el-amd-pmcr-teb-v1/ETTm1/parallel_multivariate/all/horizon_192/fold_official/seed_2024/20260828T130916.916853Z-493ea9bf` | 13/13 passed |
| 336 | `20260828T131303.474933Z-f83b3934` | `19c464460c0fd4d079f006b06ad6f88eceb14379ec955cf200e3448f9fad3f0e` | 9 | `/public/home/yueweiting/大论文/AMD/artifacts/m4-development/ettm1-stage-a-capacity-v1/el-amd-pmcr-teb-v1/ETTm1/parallel_multivariate/all/horizon_336/fold_official/seed_2024/20260828T131303.474933Z-f83b3934` | 13/13 passed |
| 720 | `20260828T131703.587008Z-9d320a71` | `f61e081a0cfdf8596af5156c9f5138e0bf984478c1e10235b063ba0fe58f2d21` | 4 | `/public/home/yueweiting/大论文/AMD/artifacts/m4-development/ettm1-stage-a-capacity-v1/el-amd-pmcr-teb-v1/ETTm1/parallel_multivariate/all/horizon_720/fold_official/seed_2024/20260828T131703.587008Z-9d320a71` | 13/13 passed |

### 18.4 T1：`T1_GLOBAL_TEB_D64_H4`

| H | run_id | config hash | best epoch | 绝对路径 | checksum |
|---:|---|---|---:|---|---|
| 96 | `20260828T132118.638904Z-d4f094fe` | `9c03473356b359ddd050f22fd99691a0c1b6a2f361f04a45f90c3d7bcbd04797` | 8 | `/public/home/yueweiting/大论文/AMD/artifacts/m4-development/ettm1-stage-a-capacity-v1/el-amd-pmcr-teb-v1/ETTm1/parallel_multivariate/all/horizon_96/fold_official/seed_2024/20260828T132118.638904Z-d4f094fe` | 13/13 passed |
| 192 | `20260828T132448.746197Z-50f92b99` | `c601990b6612a36034f417eae311c40c39a7c63bc3a676de7d9d924d2f32c72b` | 7 | `/public/home/yueweiting/大论文/AMD/artifacts/m4-development/ettm1-stage-a-capacity-v1/el-amd-pmcr-teb-v1/ETTm1/parallel_multivariate/all/horizon_192/fold_official/seed_2024/20260828T132448.746197Z-50f92b99` | 13/13 passed |
| 336 | `20260828T132929.912951Z-539f43ce` | `d794e27356e8754d4403c251395670dd3c74e05f591305f34586f79a7ca9e372` | 5 | `/public/home/yueweiting/大论文/AMD/artifacts/m4-development/ettm1-stage-a-capacity-v1/el-amd-pmcr-teb-v1/ETTm1/parallel_multivariate/all/horizon_336/fold_official/seed_2024/20260828T132929.912951Z-539f43ce` | 13/13 passed |
| 720 | `20260828T133418.321029Z-fad4a267` | `965cc61a132426ed0a2b360f52c26a5baab39fa39ecba1446b99081c36bee238` | 4 | `/public/home/yueweiting/大论文/AMD/artifacts/m4-development/ettm1-stage-a-capacity-v1/el-amd-pmcr-teb-v1/ETTm1/parallel_multivariate/all/horizon_720/fold_official/seed_2024/20260828T133418.321029Z-fad4a267` | 13/13 passed |

## 19. P1 容量 sanity：PMCR `d=8 -> 16`

负相对变化表示误差降低，正相对变化表示误差升高。以下均为 ETTm1 development-only 指标，test 已用于候选开发，不是正式论文结果。

### 19.1 Validation MSE

| H | P0 | P1 | Relative change |
|---:|---:|---:|---:|
| 96 | 0.36860838 | 0.36898725 | +0.102785% |
| 192 | 0.49529328 | 0.49541245 | +0.024060% |
| 336 | 0.64662651 | 0.64630554 | -0.049638% |
| 720 | 0.95557406 | 0.95557643 | +0.000247% |
| Macro mean | 0.61652556 | 0.61657042 | +0.007276% |

`mean_horizon_relative_change = +0.019364%`；`relative_change_of_macro_means = +0.007276%`。

### 19.2 Validation MAE

| H | P0 | P1 | Relative change |
|---:|---:|---:|---:|
| 96 | 0.40204834 | 0.40223365 | +0.046090% |
| 192 | 0.46302766 | 0.46343251 | +0.087434% |
| 336 | 0.52797802 | 0.52782581 | -0.028828% |
| 720 | 0.64010678 | 0.63990043 | -0.032238% |
| Macro mean | 0.50829020 | 0.50834810 | +0.011390% |

`mean_horizon_relative_change = +0.018115%`；`relative_change_of_macro_means = +0.011390%`。

### 19.3 Test MSE

| H | P0 | P1 | Relative change |
|---:|---:|---:|---:|
| 96 | 0.28989010 | 0.29018928 | +0.103203% |
| 192 | 0.32904546 | 0.32912871 | +0.025301% |
| 336 | 0.36365691 | 0.36396959 | +0.085983% |
| 720 | 0.42511255 | 0.42534631 | +0.054987% |
| Macro mean | 0.35192625 | 0.35215847 | +0.065984% |

`mean_horizon_relative_change = +0.067368%`；`relative_change_of_macro_means = +0.065984%`。

### 19.4 Test MAE

| H | P0 | P1 | Relative change |
|---:|---:|---:|---:|
| 96 | 0.34447625 | 0.34478304 | +0.089060% |
| 192 | 0.36711148 | 0.36736905 | +0.070162% |
| 336 | 0.38619270 | 0.38666383 | +0.121995% |
| 720 | 0.41742995 | 0.41747935 | +0.011835% |
| Macro mean | 0.37880259 | 0.37907382 | +0.071601% |

`mean_horizon_relative_change = +0.073263%`；`relative_change_of_macro_means = +0.071601%`。

### 19.5 Gamma、参数与资源

| H | P0 gamma best / last | P1 gamma best / last | P0 / P1 total params | P1 wall-clock | P1 artifact |
|---:|---:|---:|---:|---:|---:|
| 96 | -0.04686904 / -0.04686904 | +0.03898283 / +0.03898283 | 10,245,368 / 10,246,504 | 186.648 s | 196.310 MiB |
| 192 | +0.04588981 / +0.04588981 | -0.03958452 / -0.03958452 | 11,819,000 / 11,820,136 | 188.133 s | 226.322 MiB |
| 336 | +0.04984254 / +0.05440913 | -0.03693775 / -0.04023543 | 14,179,448 / 14,180,584 | 197.381 s | 271.341 MiB |
| 720 | -0.01566763 / -0.04720909 | +0.02121408 / +0.04246049 | 20,473,976 / 20,475,112 | 195.393 s | 391.393 MiB |

PMCR 模块参数从 626 增至 1,762，增加 1,136；完整模型在每个 horizon 同样只增加 1,136 个参数。P1 四个 run 合计 runner wall-clock 为 767.555 s，artifact 合计约 1,085.366 MiB。

**P1 development signal：negative-or-negligible。** Validation 变化为混合且极小；四个 horizon 的 test MSE/MAE 均轻微升高。当前 ETTm1 development 证据不支持“PMCR v1 的 `d=8` 是明显容量限制”，但不能据此冻结 PMCR、否定 P2 或外推到 EV/正式数据集。

## 20. T1 容量 sanity：Global TEB `d=32 -> 64, heads=4`

### 20.1 Validation MSE

| H | T0 | T1 | Relative change |
|---:|---:|---:|---:|
| 96 | 0.37602077 | 0.37943502 | +0.907996% |
| 192 | 0.49283903 | 0.49532662 | +0.504747% |
| 336 | 0.65069948 | 0.64885843 | -0.282934% |
| 720 | 0.95184801 | 0.95221528 | +0.038585% |
| Macro mean | 0.61785182 | 0.61895884 | +0.179172% |

`mean_horizon_relative_change = +0.292099%`；`relative_change_of_macro_means = +0.179172%`。

### 20.2 Validation MAE

| H | T0 | T1 | Relative change |
|---:|---:|---:|---:|
| 96 | 0.40477602 | 0.40718437 | +0.594983% |
| 192 | 0.46199174 | 0.46343968 | +0.313412% |
| 336 | 0.53072787 | 0.52963580 | -0.205767% |
| 720 | 0.63921554 | 0.64016221 | +0.148100% |
| Macro mean | 0.50917779 | 0.51010551 | +0.182201% |

`mean_horizon_relative_change = +0.212682%`；`relative_change_of_macro_means = +0.182201%`。

### 20.3 Test MSE

| H | T0 | T1 | Relative change |
|---:|---:|---:|---:|
| 96 | 0.30180387 | 0.30271411 | +0.301598% |
| 192 | 0.33628009 | 0.34250621 | +1.851468% |
| 336 | 0.37065268 | 0.37312595 | +0.667275% |
| 720 | 0.42554159 | 0.42704505 | +0.353304% |
| Macro mean | 0.35856956 | 0.36134783 | +0.774820% |

`mean_horizon_relative_change = +0.793411%`；`relative_change_of_macro_means = +0.774820%`。

### 20.4 Test MAE

| H | T0 | T1 | Relative change |
|---:|---:|---:|---:|
| 96 | 0.35277633 | 0.35327623 | +0.141703% |
| 192 | 0.37304086 | 0.37616152 | +0.836545% |
| 336 | 0.39189362 | 0.39316304 | +0.323920% |
| 720 | 0.41877910 | 0.42027104 | +0.356261% |
| Macro mean | 0.38412248 | 0.38571796 | +0.415357% |

`mean_horizon_relative_change = +0.414607%`；`relative_change_of_macro_means = +0.415357%`。

### 20.5 Gamma、参数与资源

| H | T0 gamma best / last | T1 gamma best / last | T0 / T1 total params | T1 wall-clock | T1 artifact |
|---:|---:|---:|---:|---:|---:|
| 96 | +0.05907373 / +0.05907373 | +0.04093082 / +0.04383150 | 10,298,823 / 10,360,583 | 169.865 s | 198.431 MiB |
| 192 | +0.06089771 / +0.06089771 | +0.04576709 / +0.05180440 | 11,872,455 / 11,934,215 | 181.411 s | 228.445 MiB |
| 336 | +0.06229828 / +0.07270923 | +0.04085442 / +0.05065113 | 14,232,903 / 14,294,663 | 180.016 s | 273.461 MiB |
| 720 | +0.05281645 / +0.07394664 | -0.04142861 / -0.05390761 | 20,527,431 / 20,589,191 | 190.254 s | 393.558 MiB |

TEB 模块参数从 54,081 增至 115,841，增加 61,760；完整模型在每个 horizon 同样只增加 61,760 个参数。T1 四个 run 合计 runner wall-clock 为 721.546 s，artifact 合计约 1,093.895 MiB。八个新 run 合计 wall-clock 为 1,489.101 s，artifact 合计约 2,179.261 MiB。

**T1 development signal：negative-or-negligible。** h336 validation 有小幅改善，其余 validation 与全部 test MSE/MAE 均退化；test MSE 宏平均升高约 0.775%。当前 ETTm1 development 证据不支持“Global TEB v1 的 `d=32` 是明显容量限制”，但不能据此冻结 TEB、取消 T2、启动 T3 或外推到正式泛化能力。

## 21. 第五轮 Stage A 结论与边界

事实：P1/T1 共 8 个新 run 全部 completed；每个 artifact 均为 schema-v2、10 epochs、指标有限值、13/13 checksum 通过，source/data fingerprint 一致。命令相对各自 P0/T0 的允许科学变化仅为 artifact root、docs-only Git HEAD，以及唯一容量参数（P1 `hidden_dim=16`；T1 `context_dim=64`）；P1/T1 沿用现有 `el-amd-pmcr-teb-v1` 与对应 `M1`/`M2` ablation，不创建新 class 或 variant。

ETTm1 development 推断：增大 PMCR hidden capacity 没有形成一致正向信号；增大 Global TEB context capacity 形成总体负向 development signal。二者的小容量都未被本轮证明为明显限制。

不能推出：P2、T2 或 T3 是否有效；最终 PMCR/TEB 结构；多 seed 可重复性；EV 场景有效性；正式数据集 test 表现；M6 正式泛化能力。第五轮没有实现 P2/T2/T3，没有继续设计或保留 T4/T5，没有进入 M5，也没有实现任何空间模块。

本节及第 18-20 节记录第五轮 Stage A 结束、Git closure 前的历史现场；最终 closure 状态以第 22 节为准。M4 状态继续为 `In Progress`。

## 22. 第六轮 Stage A closure 与 T2 工程合同

### 22.1 Stage A Git closure

- repair 将第 2 节 canonical SHA 消歧为第三轮候选身份 repair SHA 与第五轮 ETTm1 development 协议 SHA，未改变任何 P1/T1 artifact、指标或结论。
- closure commit：`36104e7a9fd0681d283b91e740c7d4339a7df1a9`。
- parent：`adad7fee12688769c53bf5736c7d1fcc3bb33c6e`。
- title：`docs(m4): record capacity sanity results`。
- commit 仅包含 `docs/milestones/M4_temporal_diagnostics.md`，已普通 push 至 `origin/AMD-paper-repro-custom-modules-v1`；local、tracking 与 live remote 闭环一致。
- P1 与 T1 均为 `negative-or-negligible` development signal；不继续 PMCR `d=32/d=128`、Global TEB heads/context 网格，也不据此冻结最终结构。

### 22.2 T2 授权、身份与来源边界

用户明确授权第六轮实现 T2 Patch-Conditioned TEB，但未授权训练。固定身份：

```text
class = PatchConditionedTargetExogenousBridge
implementation_variant = el-amd-m4-t2-patch-teb-v1
ablation_id = M4_T2
teb_architecture = patch_conditioned_v1
```

T2 是 TimeXer-inspired hierarchical representation adaptation：保留 target patch-level queries、whole-series exogenous variate tokens 与 global target context；patch queries 直接查询外生 variate tokens。不复制 TimeXer 完整 self-attention、FFN、Transformer stack 或 prediction head。Global TEB v1 的 class、文件、keys、forward 和 `el-amd-pmcr-teb-v1` 身份保持冻结。

### 22.3 精确张量与参数合同

- non-overlap target patch，`N=ceil(T/P)`，只做 right-zero padding，输出 unpatch 后 crop 回 T。
- `Q_patch=LayerNorm(Linear(P,d)(patches)+fixed_sinusoidal)`；位置 buffer 不可学习、`persistent=False`。
- `q_global=LayerNorm(Linear(T,d)(H_y))`；外生 token 继续来自 RevIN 后 `normalized_input`，共享 `Linear(T,d)+LayerNorm`。
- single 模式拼接 `N` 个 patch query 与一个 global query，只替换 target 通道；parallel 模式以 `[B,C*(N+1),d]` 一次向量化 MHA，并用 `[C*(N+1),C]` owner mask 排除自身 key。
- patch context 经共享 `Linear(d,P)`、unpatch/crop 后形成 residual；global context 进入 `state_source`，其顺序与 `[B,2*T+d]` 宽度不变。
- `d=32`、heads=4、dropout=0.1、gamma init=1e-3、padding=`right_zero_crop`、position=`fixed_sinusoidal`；ETTm1 `P=32`、UrbanEV `P=3` 均须显式配置。
- 参数量公式固定为 `2*P*d + 2*T*d + 4*d*d + 13*d + P + 1`；ETTm1 为 39,361，UrbanEV 为 5,476。

### 22.4 Config、checkpoint 与阶段边界

T2 的 variant、ablation、architecture、patch size/padding/position、TEB 参数、seq_len、task/target/aux/schema、source/data fingerprint 必须进入 scientific config、checkpoint、manifest、resume mismatch 与 summarizer identity。只允许 from scratch 和完全同结构 `load_state_dict(strict=True)`；禁止 Global v1/T2 跨结构加载、source-kind importer、部分 key、`strict=False` 或自动补 key。

本轮不训练 T2，不运行真实 ETTm1/UrbanEV，不实现 P2/T3/T4/T5，不进入 M5，不实现空间模块。T2 实现和测试完成后仍保持未提交，必须先通过用户与 ChatGPT review。

### 22.5 工程实现与文件范围

第六轮在 Stage A closure 后的 clean HEAD 上完成独立 T2 class 和工程接入。实际实现文件为：

- 新增 `models/modules/patch_conditioned_target_exogenous_bridge.py`：single/parallel patch-conditioned bridge、固定 sinusoidal buffer、owner mask 与显式输入合同。
- 修改 `models/modules/__init__.py`：公开独立 T2 class 与固定合同常量。
- 修改 `models/tsAMD_enhanced.py`：按 `teb_architecture` 条件实例化 Global v1 或 T2；保持 DDI/PMCR/AMS/state-source 路由；严格同结构恢复先验证完整 key/shape，失败不污染参数；T2 禁止 source-kind importer。
- 修改 `main.py`：新增 `el-amd-m4-t2-patch-teb-v1`、`M4_T2`、三个显式 patch CLI 字段、候选 scientific/manifest/checkpoint/resume/artifact 身份；旧 Global v1 scientific config 不增加 patch 字段。
- 修改 `summarize_results.py`：独立加载 T2 variant，核验 patch candidate contract、路径/manifest/config、13-file checksum 和重复科学身份，不与 Global v1 混分组。
- 新增 `tests/test_patch_conditioned_teb.py`、`tests/test_patch_conditioned_teb_parallel.py`、`tests/test_patch_conditioned_teb_checkpoint.py`。
- 修改 `tests/test_tsAMD_enhanced.py`、`tests/test_public_architecture.py`、`tests/test_runner.py`、`tests/test_summarize_results.py`，增加 T2 集成、旧 v1 key/hash、runner/artifact/resume 与 summarizer 门禁。
- 修改 canonical 与本 milestone，仅记录已授权的 T2 精确合同、实现证据和阶段边界。

T2 使用一个向量化 parallel MHA；single 模式非目标通道逐元素不变；parallel 的 `target_idx` 只锚定导出的 global context。固定位置 buffer 不进入 `state_dict`。真实 P0/T0 h96 回归 fixture 证明旧 Global v1 scientific/config comparison hash 语义未因 T2 字段而改变。

### 22.6 定向测试与完整回归

所有命令均使用 `/public/home/yueweiting/miniconda/envs/amd/bin/python -B`、`PYTHONDONTWRITEBYTECODE=1` 和 `GIT_OPTIONAL_LOCKS=0`；仅执行 unit-test synthetic fixture、受控 forward/backward 和 `TemporaryDirectory` artifact，不执行真实候选训练。

| 门禁组 | 命令范围 | 结果 | failed | skipped | wall-clock |
|---|---|---:|---:|---:|---:|
| T2 模块/parallel/checkpoint | `test_patch_conditioned_teb*.py` 三文件 | 19/19 passed | 0 | 0 | 2.009 s |
| 集成/runner/summarizer | `test_tsAMD_enhanced.py`、`test_public_architecture.py`、`test_runner.py`、`test_summarize_results.py` | 75/75 passed | 0 | 0 | 7.359 s |
| Global TEB v1 保护 | `test_teb.py`、`test_teb_parallel.py`、`test_teb_disabled_zero_context.py` | 17/17 passed | 0 | 0 | 1.805 s |
| PMCR v1 保护 | `test_pmcr.py`、`test_pmcr_no_cross_variable.py`、`test_pmcr_reparameterization.py` | 15/15 passed | 0 | 0 | 1.895 s |
| M1 保护 | M1 五个既有数据合同文件 | 21/21 passed | 0 | 0 | 4.727 s |
| 完整回归 | `unittest discover -s tests -p 'test_*.py' -v` | 165/165 passed | 0 | 0 | 11.640 s |

CUDA 可用且 T2 CUDA float32 分支实际执行。参数量门禁实测为 ETTm1 `T=512,P=32,d=32` 39,361 与 UrbanEV `T=12,P=3,d=32` 5,476。既有 synthetic 梯度测试把 `hidden_out` 与 `context` 同时纳入测试 loss，证明两个模块输出联合可导且全部逻辑参数组可获得有限非零梯度；它不证明 production forecast loss 单独训练全部逻辑参数，production 梯度语义以第 25 节专项审计为准。owner mask 的被禁止 attention 权重为零；T2 same-structure `strict=True` 成功，partial/mismatch/Global↔T2/`strict=False` 均在写参前拒绝。完整 `state_source` 继续为 `[B,2*T+32]`。

### 22.7 Fingerprint、冻结保护与 review 门禁

- canonical T2 实现版本 SHA-256：`16e8425968f0586a4464e4546b6a8fe7a6969f2fa9daf27e92a4832c5465160b`；内部版本仍为 `v2.1-R1`。
- source fingerprint：`883bbbef80d5a7a13d5353d3dc08e549159dcfbf3beed40a307759db4e20a117`；算法仍为 `sha256_length_prefixed_relative_path_and_content_v1`，文件数由 17 增至 18，新纳入 `models/modules/patch_conditioned_target_exogenous_bridge.py`。
- `models/tsAMD.py`：`fa72cdbe34348364344c0d9c0755668a82d22f6a37ee061c7ece93ecfaf90ba1`。
- Global TEB v1：`c389157fd20ed66911163b6db0df3e7cd96f66b6f0bb112c432b77cf37588b2e`。
- PMCR v1：`4f0507d3512df22152826e55a6113bb06ed2b5e9d39e6fe0a417a528c23ecd56`。
- M0/M1/M2/M3 SHA-256 分别保持 `e2a20131664391752340e92a9d9a5302b0078cac48d7dcdbb4a4841a16f62cdd`、`bcd9b3e3d821a1cf609423a3e8b58ecb4129bc202238e9fe1f8d7cf3a361c70b`、`8a4343e054f9dacd9cd623b930b07376a55e507f2d9085e56c634323b6dbda54`、`be3ff6606a52c9668ca4f1c2d47aaf9771be1e25e69b8fe991031f7fb9c192ed`。
- 不可变 baseline tag 仍解析到 `fa9665627e6fcfb1d0c2bc22d943ca9666304fd6`；UrbanEV、ModernTCN 与 TimeXer 参考仓库未修改。

本轮没有训练 T2，没有运行真实 ETTm1/UrbanEV T2，没有产生正式 artifact，没有实现 P2/T3/T4/T5，没有进入 M5，也没有实现空间模块。T2 代码、测试、canonical 与本 milestone 均保持未 stage、未 commit、未 push；M4 状态继续为 `In Progress`。下一步必须先完成 T2 implementation review，未经新授权不得开始训练或 Git closure。

## 23. 第七轮 T2 implementation review

- 变更范围复核：工作区精确包含已授权的 14 个 T2 代码、测试与文档文件，无第 15 个变化文件；独立 `PatchConditionedTargetExogenousBridge`、single/parallel 张量合同、AMS 路由、`state_source`、初始化、variant/config/checkpoint/artifact 合同均通过源码审查。
- 冻结保护：`models/tsAMD.py`、Global TEB v1、PMCR v1 与 M0-M3 均保持既有 SHA-256；旧 Global v1 class、state keys、forward、CLI/config identity 未被 T2 覆盖或重新定义。
- ETTm1 real validation batch smoke：`x=[128,512,7]`、`y=[128,96,7]`、prediction=`[128,96,7]`、`state_source=[128,1056]`；prediction、state 与标量 MoE loss 均有限，T2 参数量为 39,361。
- UrbanEV M1 production pipeline F4/fold 1 real validation batch smoke：`x=[4,12,11]`、`y=[4,1]`、prediction=`[4,1,1]`、`state_source=[4,56]`；`target_idx=0`，`aux_idx=[1,2,3,4,5,6,7,8,9,10]` 有序非空，全部输出有限，T2 参数量为 5,476。两项 smoke 均为 `eval()+no_grad()`，未构造 optimizer、未训练、未创建 artifact，也未遍历 UrbanEV test。
- 公共 AMD 初始化 parity：按真实 runner 的 seed→loader→model 顺序，在 h96/h192/h336/h720 分别重建 T0 Global TEB v1 与 T2；每个 horizon 的 60 个公共 AMD parameter/persistent buffer keys 完全一致，`max_abs_error=0`，不一致 key 为空。
- 第七轮复跑：T2 模块/parallel/checkpoint 19/19、集成/runner/summarizer 75/75、Global TEB v1 保护 17/17、PMCR v1 保护 15/15、M1 保护 21/21、完整回归 165/165，均 `failed=0, skipped=0`；CUDA 可用且 T2 CUDA float32 测试实际执行。
- implementation review：**Passed**。M4 状态继续为 `In Progress`；本结论只完成工程实现审查，尚未完成 T2 性能验收，尚未进入 M5。

## 24. 第七轮 T2 Git closure 与 ETTm1 development 实验

### 24.1 T2 工程 closure

第 23 节 review 通过后，14 个已授权文件以一个 commit 完成工程 closure：

```text
commit = 4979e5fd9738da28e2999edf8a6b7dc1ff0266d9
parent = 36104e7a9fd0681d283b91e740c7d4339a7df1a9
title = feat(m4): implement patch-conditioned TEB candidate
push = origin/AMD-paper-repro-custom-modules-v1 succeeded
local/tracking/live remote = 4979e5fd9738da28e2999edf8a6b7dc1ff0266d9
ahead/behind = 0/0
```

closure 后 source fingerprint 为 `883bbbef80d5a7a13d5353d3dc08e549159dcfbf3beed40a307759db4e20a117`（`sha256_length_prefixed_relative_path_and_content_v1`，18 files）。用于 closure 的本 milestone SHA-256 为 `c7601572b2154c11f113e05a5ea3e531f8a44c46dc81f933021e36cb1e861d55`。提交并推送后 worktree/index clean，才开始下述实验。

### 24.2 同源码 T0-refresh 与 T2 artifact

以下 8 个 run 均为 ETTm1 development-only、`parallel_multivariate`、7 变量、`seq_len=512`、seed 2024、10 epochs、train-standardized metric space；均 `status=completed`、schema-v2、history=10、指标有限、精确 13/13 checksum 与系统 `sha256sum -c` 通过。共同 source fingerprint 为 `883bbbef80d5a7a13d5353d3dc08e549159dcfbf3beed40a307759db4e20a117`，共同 data SHA-256 为 `6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e`，Git commit 均为 `4979e5fd9738da28e2999edf8a6b7dc1ff0266d9` 且 `dirty=false`。

T0-refresh（Global TEB v1）：

| H | run_id | config hash | best epoch | best.pt SHA-256 | last.pt SHA-256 | artifact path |
|---:|---|---|---:|---|---|---|
| 96 | `20260828T163747.443682Z-7cebe0aa` | `02336c26bba997583eaf9863c7e8042e6e95c11207276133fcbab0e1e66b68e8` | 10 | `335f824129721304ccf5967dfd59b4b84cc785c517820ffaa2f8b50f757fd550` | `64847938fa73a09f5faa699b9658fa18c8cc398b15da3b38210365214000dec3` | `artifacts/m4-development/ettm1-stage-b-t2-v1/el-amd-pmcr-teb-v1/ETTm1/parallel_multivariate/all/horizon_96/fold_official/seed_2024/20260828T163747.443682Z-7cebe0aa` |
| 192 | `20260828T164101.355353Z-a9ab23df` | `9d476dc6aefdd4038675274433aafedc8ffa7532ebead469a162243da9f86762` | 10 | `2c6ace2851a3a83de4f6a0a87cc737737e79776d6dcfb2e5e0e469a6d258a4d9` | `094c7fc6f0fa33916749a87a7a3860038d4c9d039cb219e9c8a47422fb8ac1b4` | `artifacts/m4-development/ettm1-stage-b-t2-v1/el-amd-pmcr-teb-v1/ETTm1/parallel_multivariate/all/horizon_192/fold_official/seed_2024/20260828T164101.355353Z-a9ab23df` |
| 336 | `20260828T164428.040240Z-04b4302a` | `32f628f4568f7e6ee21b36a0e5b8fb80af0db8bbf4a1b8304d54fdcfcf5efec0` | 6 | `1242443ec635a4abc8939f8cea375c4228b4818ba462e697ba42509231829faf` | `1f902fd98074296e6a6685448955cf4136f2204335a0b8b8f23b378d4a79529b` | `artifacts/m4-development/ettm1-stage-b-t2-v1/el-amd-pmcr-teb-v1/ETTm1/parallel_multivariate/all/horizon_336/fold_official/seed_2024/20260828T164428.040240Z-04b4302a` |
| 720 | `20260828T164733.847735Z-bed6e79b` | `dca615ba104843a931ba50cb4c54ec6e089948ed5d83e2ab4585284a29ae20d3` | 4 | `585b39f28f609834e6512310b697d440e18802762c3fe58c7c299af92f0794c9` | `589af37e0f9366e5840bff714995d85b006c165eb2f77efe2524365b94c8b013` | `artifacts/m4-development/ettm1-stage-b-t2-v1/el-amd-pmcr-teb-v1/ETTm1/parallel_multivariate/all/horizon_720/fold_official/seed_2024/20260828T164733.847735Z-bed6e79b` |

T2（Patch-Conditioned TEB）：

| H | run_id | config hash | best epoch | best.pt SHA-256 | last.pt SHA-256 | artifact path |
|---:|---|---|---:|---|---|---|
| 96 | `20260828T165152.007177Z-239ec4ce` | `80dd9752e391899cfd46d5ab3c25c34d74d5d936f8b21deb23cae57322d7acca` | 10 | `fb0f1841056f5f02c7d9a17d919ca9feb559b19b4ba47984ae0ef138871d4668` | `659e105fbd503dc66f12c676a5e7e448e09372ed46a60b0f8c16ba4945fd46e7` | `artifacts/m4-development/ettm1-stage-b-t2-v1/el-amd-m4-t2-patch-teb-v1/ETTm1/parallel_multivariate/all/horizon_96/fold_official/seed_2024/20260828T165152.007177Z-239ec4ce` |
| 192 | `20260828T165448.863614Z-d14a962b` | `8cf9c9a814af3145aecd107706ca662c8d67598bc701480d0cdd980f6ac5a94c` | 7 | `5b47fa8e06805cd13071fef092e8de7a110fe227c702389f38266a51d419ceb5` | `d0400c4de87a56aa357a3275eb5d2480a4cda39696e9aab1f43b0ef33324d417` | `artifacts/m4-development/ettm1-stage-b-t2-v1/el-amd-m4-t2-patch-teb-v1/ETTm1/parallel_multivariate/all/horizon_192/fold_official/seed_2024/20260828T165448.863614Z-d14a962b` |
| 336 | `20260828T165800.999127Z-6bf6519d` | `66282315c200cc3ca10d01291cebb691d611c027bf421f5c75edfb12cc554b29` | 6 | `acc2cdb8f324357e8bff9d1906b918ac954ad87f8752040a8ba8facd907a8f0d` | `97b7b1b9052cf4357f4013fa2991e365086c3894657e9fab856688c984b1d042` | `artifacts/m4-development/ettm1-stage-b-t2-v1/el-amd-m4-t2-patch-teb-v1/ETTm1/parallel_multivariate/all/horizon_336/fold_official/seed_2024/20260828T165800.999127Z-6bf6519d` |
| 720 | `20260828T170128.403877Z-4816d7a2` | `2fcb1b561d80d230db8084282ad757fa3e1d222b3aa66b6ef70c7ceb5be96eda` | 6 | `ada629807cd638d7eb2a3a8fe5ee412227f709c65bbe9454bf3583af02ceebf5` | `8fa86adb83fa97073fbafcc534f82426e08e497a75d47f3ae2bff6b93bafe16f` | `artifacts/m4-development/ettm1-stage-b-t2-v1/el-amd-m4-t2-patch-teb-v1/ETTm1/parallel_multivariate/all/horizon_720/fold_official/seed_2024/20260828T170128.403877Z-4816d7a2` |

相对旧 T0，T0-refresh 只改变 artifact root 与自动记录的 docs/T2 closure commit、source fingerprint。四个 horizon 的 validation/test MSE/MAE、best epoch、history 曲线与 best/last gamma 均逐值完全一致；两种 relative 统计均为 `0.0000%`。这为 T2 提供了同源码、同 runner 且可复现的直接对照。

### 24.3 T2 相对同源码 T0-refresh

负 relative 表示误差下降。

Validation MSE：

| Horizon | T0-refresh | T2 | Relative change |
|---:|---:|---:|---:|
| 96 | 0.37602077 | 0.37115965 | -1.2928% |
| 192 | 0.49283903 | 0.50013897 | +1.4812% |
| 336 | 0.65069948 | 0.64641350 | -0.6587% |
| 720 | 0.95184801 | 0.95247168 | +0.0655% |
| Macro mean | 0.61785182 | 0.61754595 | -0.0495% |

Validation MAE：

| Horizon | T0-refresh | T2 | Relative change |
|---:|---:|---:|---:|
| 96 | 0.40477602 | 0.40300325 | -0.4380% |
| 192 | 0.46199174 | 0.46881942 | +1.4779% |
| 336 | 0.53072787 | 0.52858440 | -0.4039% |
| 720 | 0.63921554 | 0.63922116 | +0.0009% |
| Macro mean | 0.50917779 | 0.50990705 | +0.1432% |

Test MSE（ETTm1 development-only）：

| Horizon | T0-refresh | T2 | Relative change |
|---:|---:|---:|---:|
| 96 | 0.30180387 | 0.29777238 | -1.3358% |
| 192 | 0.33628009 | 0.33464689 | -0.4857% |
| 336 | 0.37065268 | 0.36694654 | -0.9999% |
| 720 | 0.42554159 | 0.42190152 | -0.8554% |
| Macro mean | 0.35856956 | 0.35531683 | -0.9071% |

Test MAE（ETTm1 development-only）：

| Horizon | T0-refresh | T2 | Relative change |
|---:|---:|---:|---:|
| 96 | 0.35277633 | 0.35107041 | -0.4836% |
| 192 | 0.37304086 | 0.37234873 | -0.1855% |
| 336 | 0.39189362 | 0.38916411 | -0.6965% |
| 720 | 0.41877910 | 0.41770466 | -0.2566% |
| Macro mean | 0.38412248 | 0.38257198 | -0.4036% |

两种 relative 汇总不得混称：

| Metric | mean_horizon_relative_change | relative_change_of_macro_means |
|---|---:|---:|
| Validation MSE | -0.1012% | -0.0495% |
| Validation MAE | +0.1592% | +0.1432% |
| Test MSE | -0.9192% | -0.9071% |
| Test MAE | -0.4055% | -0.4036% |

### 24.4 训练端点、成本与曲线末段

| H | T0 best/last gamma | T2 best/last gamma | T0/T2 total params | T0/T2 wall-clock | T0/T2 artifact MiB |
|---:|---:|---:|---:|---:|---:|
| 96 | 0.05907373 / 0.05907373 | 0.08463841 / 0.08463841 | 10,298,823 / 10,284,103 | 188.415 / 170.919 s | 197.28 / 196.99 |
| 192 | 0.06089771 / 0.06089771 | 0.07496016 / 0.08428805 | 11,872,455 / 11,857,735 | 200.795 / 186.258 s | 227.31 / 227.05 |
| 336 | 0.06229828 / 0.07270923 | 0.06646292 / 0.08232092 | 14,232,903 / 14,218,183 | 173.833 / 201.223 s | 272.29 / 272.06 |
| 720 | 0.05281645 / 0.07394664 | 0.06837399 / 0.08211943 | 20,527,431 / 20,512,711 | 179.929 / 172.951 s | 392.37 / 392.10 |

Global TEB v1 模块为 54,081 parameters，T2 为 39,361，故 T2 每个 horizon 少 14,720 parameters。四个 T0-refresh 合计 wall-clock 742.972 s、artifact 1,089.253 MiB；四个 T2 合计 731.350 s、1,088.201 MiB。该单次 wall-clock 只描述本轮执行，不构成稳定吞吐基准。

validation MSE 最后三轮：T0-refresh h96 单调改善，h192 混合后改善，h336/h720 回升；T2 h96 单调改善，h192/h336 混合，h720 回升。best epoch 分别为 T0-refresh `[10,10,6,4]`、T2 `[10,7,6,6]`；`last.pt` 均为完成第 10 epoch 的训练末端 checkpoint。

### 24.5 T2 相对既有 AMD baseline（补充、跨 source 版本）

既有 AMD baseline 来自 M4 第一轮纳入的四个 clean schema-v1 run，source fingerprint 为 `961efb5a54742db972b55b6556a2dc2e05939e580b2c9e61b595f3ff9983f330`；T2 使用当前 `883bbbef...a117`。因此本比较是用户要求的补充 development 参照，不替代上节同源码 T0-refresh 对照。

| Metric | H96 AMD/T2 | H192 AMD/T2 | H336 AMD/T2 | H720 AMD/T2 | AMD/T2 macro | macro relative |
|---|---:|---:|---:|---:|---:|---:|
| Validation MSE | 0.37007348 / 0.37115965 | 0.49616223 / 0.50013897 | 0.64766912 / 0.64641350 | 0.95571715 / 0.95247168 | 0.61740550 / 0.61754595 | +0.0227% |
| Validation MAE | 0.40250664 / 0.40300325 | 0.46346370 / 0.46881942 | 0.52858624 / 0.52858440 | 0.63946526 / 0.63922116 | 0.50850546 / 0.50990705 | +0.2756% |
| Test MSE | 0.29053000 / 0.29777238 | 0.32905525 / 0.33464689 | 0.36377269 / 0.36694654 | 0.42211783 / 0.42190152 | 0.35136894 / 0.35531683 | +1.1236% |
| Test MAE | 0.34479168 / 0.35107041 | 0.36706965 / 0.37234873 | 0.38649264 / 0.38916411 | 0.41598824 / 0.41770466 | 0.37858555 / 0.38257198 | +1.0530% |

对应 mean_horizon_relative_change 依次为 Validation MSE `+0.1404%`、Validation MAE `+0.3101%`、Test MSE `+1.2533%`、Test MAE `+1.0908%`。这说明 T2 相对 Global TEB v1 的改进不能被表述为已经优于冻结 AMD baseline。

### 24.6 Development signal 与证据边界

**T2 development signal：positive development signal（相对同源码 T0-refresh）。** 固定 primary（四 horizon test MSE macro）下降 0.9071%，secondary（test MAE macro）下降 0.4036%，且四个 horizon 的 test MSE/MAE 均同向改善，不由单个 horizon 驱动；validation 则为两好两坏，MSE 宏平均近似持平、MAE 宏平均轻微退化。T0-refresh 与旧 T0 完全复现，排除了本轮 source/runner refresh 对 Global v1 数值的扰动；T2 参数量更少且本轮总 wall-clock 相近。

该结论仅表示：在 ETTm1 development-only、单 seed、10 epochs 下，patch-conditioned 表示相对 Global TEB v1 出现值得继续验证的开发信号。它不表示 T2 已冻结、已经优于 AMD、具有正式泛化提升或已通过 M5；也不能推出多 seed、UrbanEV/正式数据集有效性。ETTm1 test 已用于候选开发，不能进入 M6 未见测试主张或正式性能主表。本轮未实现 P2/T3/T4/T5，未运行任何正式评价数据集 test，未进入 M5，也未实现空间模块。M4 状态继续为 `In Progress`；本节实验结果保持未 stage、未 commit、未 push，等待用户与 ChatGPT review。

## 25. 第八轮：TEB-first 顺序修订与 T2 global-query 梯度路径审计

### 25.1 用户锁定的 M4 开发顺序

M4 后续采用 `TEB 候选路线先收敛 > PMCR/P2 候选路线`。T2 结果之后必须先核验 global-query 的任务梯度与功能路径；在 TEB 分支形成 M4 候选终点，或用户明确停止继续迭代 TEB 之前，不启动 P2。这里的“候选终点”只表示 M4 内不再继续改变 TEB 结构，不等于 M5 最终冻结。

T3 confidence-gate proposal 暂缓，未获实现或训练授权；T4 Hidden-KV 与 T5 sparse/top-k 继续排除。若本轮发现 global-query 路径问题，任何 patch-global interaction 或其他修复候选都须由用户再次明确决定。

### 25.2 前置状态与静态依赖图

审计基于 `4979e5fd9738da28e2999edf8a6b7dc1ff0266d9`（`feat(m4): implement patch-conditioned TEB candidate`），local/tracking/live remote 一致、ahead/behind 为 `0/0`；开始时 index 为空，工作区仅本 milestone 保留第七轮实验结果修改。source fingerprint 为 `883bbbef80d5a7a13d5353d3dc08e549159dcfbf3beed40a307759db4e20a117`（18 files）。M0-M3、冻结 AMD、Global TEB v1、PMCR v1、baseline tag 及三个参考仓库均通过保护性核验。

真实 T2 路径为：

```text
hidden -> patchify -> Q_patch ---------------------> MHA -> C_patch
normalized_input -> shared exogenous tokens E ----/          |
hidden -> Linear(T,d) -> LayerNorm -> q_global ----/          v
                                                     output projection
                                                          -> temporal delta
                                                          -> v_final
                                                          -> AMS experts
                                                          -> prediction

q_global -> 同一次 MHA 的独立 query rows -> C_global
                                              -> exo_context
                                              -> state_source（仅 return_state_source=True）

u_mdm -> AMS selector -> moe_loss
```

标准 `MultiheadAttention` 对各 query row 分别查询同一组 K/V；将 `Q_patch` 与 `q_global` 拼接后进行一次 MHA，不会让 query rows 彼此交互。源码中没有 `C_global -> C_patch`、`q_global -> temporal delta` 或 `state_source -> production loss` 路径。

依赖关系如下：

| 输出/损失 | Q_patch | q_global | C_patch | C_global | exo_context | state_source |
|---|---:|---:|---:|---:|---:|---:|
| prediction | 是 | 否 | 是 | 否 | 否 | 否 |
| prediction MSE | 是 | 否 | 是 | 否 | 否 | 否 |
| AMS selector / moe loss | 否 | 否 | 否 | 否 | 否 | 否 |
| state_source | 否 | 是 | 否 | 是 | 是 | 是 |

### 25.3 Production loss 与既有梯度测试的证明边界

production runner 的单 batch 总目标是：

```text
prediction_loss = MSE(prediction_for_loss, batch_y)
total_loss = prediction_loss + auxiliary_loss
auxiliary_loss = AMS/selector 返回的 moe_loss
```

其中 AMS experts 使用 `v_final`，selector 与 auxiliary loss 仍只使用 `u_mdm`；production forward 不请求 `state_source`，也不把 `exo_context` 加入目标。optimizer 为 Adam，虽配置 weight decay，但本轮判定只使用 `optimizer.step()` 前的 raw backward gradient。

既有 `tests/test_patch_conditioned_teb.py` 梯度测试使用：

```text
loss = sum(hidden_out * hidden_weight)
     + sum(exo_context * context_weight)
```

它明确把 `exo_context` 人工加入 synthetic loss，因此有效证明 T2 两个输出及各逻辑参数组的联合可导性，却不证明 production 时间预测损失会训练 global-query 专属参数。此前“所有逻辑参数组非零”的表述在该边界内保留，不能再解释为 production forecast-loss 全参数覆盖。

### 25.4 动态审计合同

ETTm1 使用第七轮四个 T2 completed artifact 的 `best.pt`，逐 run 按自身 config 同结构构造并 `load_state_dict(strict=True)`；每个 horizon 取确定性真实 train batch `[128,512,7]`。UrbanEV 使用 M1 production pipeline 的 F4/fold1/h3 真实 train batch `[4,12,11]`，`target_idx=0`、`aux_idx=[1,...,10]`，按固定 seed 从 scratch 构造 P=3、d=32、heads=4 的 T2。所有 production 实验均为 `model.train()` 下单次真实 `MSE + moe_loss` backward，未创建 optimizer、未执行 `optimizer.step()`、未训练 epoch、未写 artifact。

另以 ETTm1 h96 T0-refresh `best.pt` 的 Global TEB v1 作为正控制；state-control 和参数扰动均使用独立 forward，并与 production 表严格分开。

### 25.5 Production-loss 中间张量梯度

真实 loss 数值如下，均为 finite：

| 数据/路径 | prediction MSE | selector auxiliary | total |
|---|---:|---:|---:|
| ETTm1 parallel h96 | 0.229860440 | 0.109931052 | 0.339791477 |
| ETTm1 parallel h192 | 0.271037757 | 0.175532803 | 0.446570575 |
| ETTm1 parallel h336 | 0.312774807 | 0.173528373 | 0.486303180 |
| ETTm1 parallel h720 | 0.373621345 | 0.149816945 | 0.523438275 |
| UrbanEV single F4/fold1/h3 | 2.445301056 | 7.524266243 | 9.969567299 |

下表的统计顺序为 `L1 / L2 / max_abs / exact_nonzero/total`。`grad None` 与 finite 另行明确列出。

| 数据/路径 | Q_patch（MHA patch rows 相同） | C_patch | temporal delta |
|---|---|---|---|
| ETTm1 h96 | `[128,7,16,32]`; `0.198692599 / 7.57693e-4 / 5.14628e-5 / 458752/458752` | `[128,7,16,32]`; `0.099041144 / 2.92288e-4 / 1.55662e-5 / 458752/458752` | `[128,7,512]`; `0.071615954 / 2.30132e-4 / 2.41263e-5 / 458752/458752` |
| ETTm1 h192 | `[128,7,16,32]`; `0.138796845 / 5.56992e-4 / 3.09720e-5 / 458752/458752` | `[128,7,16,32]`; `0.067524857 / 1.98116e-4 / 1.03311e-5 / 458752/458752` | `[128,7,512]`; `0.042743522 / 1.33661e-4 / 1.40622e-5 / 458752/458752` |
| ETTm1 h336 | `[128,7,16,32]`; `0.051819530 / 2.27648e-4 / 3.26088e-5 / 458752/458752` | `[128,7,16,32]`; `0.037816823 / 1.15784e-4 / 7.10352e-6 / 458752/458752` | `[128,7,512]`; `0.032484235 / 1.08935e-4 / 1.09790e-5 / 458752/458752` |
| ETTm1 h720 | `[128,7,16,32]`; `0.067424009 / 3.82257e-4 / 5.85117e-5 / 458752/458752` | `[128,7,16,32]`; `0.036107270 / 1.24773e-4 / 1.02455e-5 / 458752/458752` | `[128,7,512]`; `0.026879080 / 9.85976e-5 / 1.82017e-5 / 458752/458752` |
| UrbanEV single | `[4,4,32]`; `1.23466e-4 / 9.44358e-6 / 2.89293e-6 / 512/512` | `[4,4,32]`; `0.001684865 / 1.15847e-4 / 2.16127e-5 / 512/512` | `[4,12]`; `8.62053e-4 / 1.85656e-4 / 9.24530e-5 / 48/48` |

上述 `Q_patch`、MHA patch rows、`C_patch` 与 temporal delta 的 `grad_is_none=False`、finite=True。与之对照：

| 数据/路径 | q_global / MHA global rows / C_global | exo_context |
|---|---|---|
| ETTm1 h96/h192/h336/h720 | 各为 `[128,7,32]`；`grad_is_none=False`、finite=True、`L1=L2=max_abs=0`、`0/28672` | `[128,32]`；`grad_is_none=True`、finite=N/A、`0/4096` |
| UrbanEV single | 各为 `[4,32]`（MHA row 语义等价为单 global row）；`grad_is_none=False`、finite=True、`L1=L2=max_abs=0`、`0/128` | `[4,32]`；`grad_is_none=True`、finite=N/A、`0/128` |

因此 single 与 parallel 两条 production 路径给出同一结构事实：patch residual 路径获得非零任务梯度；global-only 输出虽然仍在 autograd graph 中，但 production loss 对其导数为严格零或未被使用。

### 25.6 Production-loss 参数梯度

实际参数名按逻辑组为：

- patch query projection：`teb.patch_query_projection.weight/bias`；
- patch query normalization：`teb.patch_query_norm.weight/bias`；
- global query projection：`teb.global_query_projection.weight/bias`；
- global query normalization：`teb.global_query_norm.weight/bias`；
- exogenous projection：`teb.exogenous_projection.weight/bias`；
- exogenous normalization：`teb.exogenous_norm.weight/bias`；
- MHA Q/K/V：实际参数为 packed `teb.cross_attention.in_proj_weight/bias`，审计同时逐片核验 Q/K/V；out 为 `teb.cross_attention.out_proj.weight/bias`；
- patch output projection：`teb.patch_output_projection.weight/bias`；
- gate：`teb.gamma_teb`。

下表对组内实际 parameter tensor 合并统计，格式仍为 `L1 / L2 / max_abs / exact_nonzero/total`。所有列出的 TEB parameter 均 `grad_is_none=False` 且 finite=True；这不把 packed Q/K/V slice 冒充独立 parameter。

| 参数组 | ETT h96 | ETT h192 | ETT h336 | ETT h720 | UrbanEV single |
|---|---|---|---|---|---|
| patch query projection | `0.164331 / 0.006987 / 0.001411 / 1056/1056` | `0.154074 / 0.006219 / 6.98255e-4 / 1056/1056` | `0.068201 / 0.002776 / 4.06341e-4 / 1056/1056` | `0.083650 / 0.003230 / 3.49427e-4 / 1056/1056` | `2.24603e-4 / 2.51344e-5 / 5.45078e-6 / 128/128` |
| patch query norm | `0.006834 / 0.001194 / 5.18966e-4 / 64/64` | `0.005190 / 8.71754e-4 / 3.28721e-4 / 64/64` | `0.002463 / 3.94354e-4 / 1.34938e-4 / 64/64` | `0.004567 / 7.79692e-4 / 2.99776e-4 / 64/64` | `9.00669e-5 / 1.47694e-5 / 5.49576e-6 / 64/64` |
| **global query projection** | **`0 / 0 / 0 / 0/16416`** | **`0 / 0 / 0 / 0/16416`** | **`0 / 0 / 0 / 0/16416`** | **`0 / 0 / 0 / 0/16416`** | **`0 / 0 / 0 / 0/416`** |
| **global query norm** | **`0 / 0 / 0 / 0/64`** | **`0 / 0 / 0 / 0/64`** | **`0 / 0 / 0 / 0/64`** | **`0 / 0 / 0 / 0/64`** | **`0 / 0 / 0 / 0/64`** |
| exogenous projection | `5.478709 / 0.053727 / 0.002081 / 16416/16416` | `2.914316 / 0.029008 / 0.001045 / 16416/16416` | `2.237202 / 0.022422 / 8.15031e-4 / 16416/16416` | `1.931601 / 0.019244 / 6.92673e-4 / 16416/16416` | `0.001659 / 1.43579e-4 / 5.76902e-5 / 416/416` |
| exogenous norm | `0.015498 / 0.002541 / 9.45958e-4 / 64/64` | `0.005394 / 0.001012 / 4.45908e-4 / 64/64` | `0.004737 / 8.56972e-4 / 3.21328e-4 / 64/64` | `0.006530 / 0.001035 / 4.52929e-4 / 64/64` | `3.08205e-4 / 5.50577e-5 / 2.38134e-5 / 64/64` |
| MHA packed Q/K/V | `0.496394 / 0.012557 / 0.001246 / 3168/3168` | `0.253793 / 0.006171 / 6.36160e-4 / 3168/3168` | `0.168143 / 0.004927 / 5.73714e-4 / 3168/3168` | `0.171892 / 0.004608 / 6.80176e-4 / 3168/3168` | `0.007221 / 2.17466e-4 / 2.99139e-5 / 3168/3168` |
| MHA out | `0.122089 / 0.004920 / 5.70741e-4 / 1056/1056` | `0.032906 / 0.001267 / 1.34262e-4 / 1056/1056` | `0.075574 / 0.003030 / 3.09546e-4 / 1056/1056` | `0.045207 / 0.001827 / 2.31127e-4 / 1056/1056` | `0.004519 / 2.22074e-4 / 3.90837e-5 / 1056/1056` |
| patch output projection | `0.091244 / 0.003641 / 4.83791e-4 / 1056/1056` | `0.020695 / 8.56916e-4 / 1.29060e-4 / 1056/1056` | `0.051155 / 0.002202 / 2.67213e-4 / 1056/1056` | `0.028704 / 0.001174 / 1.62414e-4 / 1056/1056` | `0.001984 / 3.20747e-4 / 2.00322e-4 / 99/99` |
| gamma_teb | `0.022884 / 0.022884 / 0.022884 / 1/1` | `0.002346 / 0.002346 / 0.002346 / 1/1` | `0.039020 / 0.039020 / 0.039020 / 1/1` | `0.017975 / 0.017975 / 0.017975 / 1/1` | `0.014905 / 0.014905 / 0.014905 / 1/1` |

Packed MHA 的 Q/K/V weight slices 在五个实验中分别均为 finite、非零；例如 h96 的 Q/K/V L2 为 `0.006243 / 0.008236 / 0.006987`，UrbanEV 为 `8.48863e-5 / 7.07804e-5 / 1.77170e-4`。这说明共享的 MHA 与外生 K/V 参数可经 patch residual 受训，不能把 global-query 专属参数为零误写成整个 MHA 或整个 T2 无梯度。

AMD 主干汇总 L2 正控制依次为 h96 `1.363702`、h192 `2.207142`、h336 `2.061705`、h720 `1.645152`、UrbanEV `21.627258`，均 finite 且存在大量非零元素。UrbanEV 中四个 `fc_blocks.0` 参数因该配置的未使用分支为 `grad=None`，与 T2 global-query 结论无关。

### 25.7 State-output 控制与 Global TEB v1 正控制

state-control 使用独立 forward，损失仅为带确定性非均匀权重的 `exo_context` mean；它不是 production loss。格式仍为 `L1 / L2 / max_abs / exact_nonzero/total`：

| 对象 | ETTm1 parallel h96 state-control | UrbanEV single state-control |
|---|---|---|
| exo_context | `1 / 0.0163033 / 3.66211e-4 / 4096/4096` | `1 / 0.0922255 / 0.0117188 / 128/128` |
| C_global | `1 / 0.0163033 / 3.66211e-4 / 4096/28672` | `1 / 0.0922255 / 0.0117188 / 128/128` |
| q_global / MHA global rows | `0.255930 / 0.00556628 / 5.03292e-4 / 4096/28672` | `0.050197 / 0.00558782 / 0.00159081 / 128/128` |
| global projection weight | `55.610524 / 0.627400 / 0.0426780 / 16384/16384` | `0.421680 / 0.0291063 / 0.00696556 / 384/384` |
| global projection bias | `0.404197 / 0.0926386 / 0.0459018 / 32/32` | `0.058588 / 0.0123558 / 0.00660495 / 32/32` |
| global norm weight | `0.039629 / 0.00887182 / 0.00360118 / 32/32` | `0.021789 / 0.00485073 / 0.00179170 / 32/32` |
| global norm bias | `0.150080 / 0.0344257 / 0.0182887 / 32/32` | `0.040886 / 0.00868336 / 0.00469807 / 32/32` |

这些梯度全部 `grad_is_none=False`、finite=True。patch-only 路径在 state-control 下为零，进一步隔离了控制目标。结果证明 global 分支本身没有 detach，问题是 production loss 没有消费该分支。

Global TEB v1 的 ETTm1 h96 正控制采用同类真实 production backward。其 `q_global` 梯度为 `L1=0.0607208, L2=7.90316e-4, max_abs=8.29262e-5, 28672/28672 nonzero`；`context_all` 为 `0.0390326 / 4.04572e-4 / 3.33665e-5 / 28672/28672`；temporal delta 为 `0.0490897 / 1.63550e-4 / 1.87088e-5 / 458752/458752`。query projection weight/bias 与 norm weight/bias 的 L2 分别为 `0.0444232 / 0.00111641 / 0.00108151 / 7.37684e-4`，均 finite、全量非零。该正控制验证了本轮 hook/backward 诊断能够识别真正进入 temporal residual 的 global-query 路径。

### 25.8 Global-query 专属参数功能扰动

在 `eval()`、相同输入和权重副本下，只对 `teb.global_query_projection.weight/bias` 与 `teb.global_query_norm.weight/bias` 施加确定性、finite、最大绝对值约 `1e-4` 的小扰动：

| 路径 | prediction | moe_loss | temporal delta | C_patch | C_global | exo_context / state_source last d |
|---|---|---|---|---|---|---|
| ETTm1 parallel h96 | `0; equal=True` | `0; True` | `0; True` | `0; True` | `0.0352865; False` | `0.0232835; False` |
| UrbanEV single F4/fold1/h3 | `0; equal=True` | `0; True` | `0; True` | `0; True` | `4.14848e-5; False` | `4.14848e-5; False` |

表内为 `max_abs_error; torch.equal`，所有比较前后张量均 finite。扰动不改变 prediction、moe loss、temporal delta 或 C_patch，却明确改变 C_global 与 state_source 最后 d 维，符合 `forecast-loss-disconnected、state-output-live` 分类；未发现文档之外的 global-to-patch 依赖。

### 25.9 最终裁决与旧表述消歧

**当前 T2 的 global-query 专属分支与第三章时间预测损失断开；它是 state-output-live，但 forecast-loss-disconnected。共享的 exogenous/MHA 参数仍可通过 patch residual 获得梯度，因此 C_global 并非所有组成部分都完全静止，但 global query 没有被直接优化成对预测或状态有用的表示。**

这一裁决同时由静态依赖、ETTm1 四 horizon production backward、UrbanEV single-target production backward、state-control、Global TEB v1 正控制与专属参数扰动支持。它不表示整个 T2 没有梯度、不把整个 exo_context 称为完全随机、不使既有 T2 development 结果失效，也不证明任何下一结构必然更好。

第六/第七轮“所有逻辑参数组及输入梯度均存在、finite、非零”应严格理解为：在同时消费 `hidden_out` 与 `exo_context` 的 synthetic module-output loss 下，联合可导性测试通过。它不能解释为 production forecast loss 会训练 global-query 专属 Linear/LayerNorm；本节完成了这一证明边界的纠正。

### 25.10 仅比较、未获实现授权的最小方向

本轮确认断开后只进行无代码比较，未把任何方向登记为已批准 T3。

#### 方向 A：forecast-supervised patch-context pooling

```text
single:   exo_context = mean_N(C_patch)
parallel: exo_context = mean_N(C_patch[:, target_idx, :, :])
```

该 context 继承 patch residual 的 forecast supervision；由于只替换 state 输出，T2 prediction 可保持逐元素不变，global query 可删除。相应删除 `Linear(T,d)` 与 global LayerNorm：ETTm1 T=512 时减少 16,480 parameters，T2 从 39,361 降为 22,881；UrbanEV T=12 时减少 480，T2 从 5,476 降为 4,996。它离 TimeXer global-token 语义更远，只修复 state context 的监督来源，不增加新的时间预测能力；必须使用新 candidate identity，并与现有 T2 checkpoint/config 严格隔离。

#### 方向 B：global-mediated patch interaction

最小合同 B1：

```text
C_fused = C_patch + beta * broadcast_N(C_global)
delta   = unpatch(output_projection(C_fused))
```

仅增加一个共享标量 `beta`。`beta=0` 可在初始化时严格退化为 T2，但 global-query 专属参数首个 backward 仍为零；`beta=1e-3` 会产生立即可用的 forecast gradient和小幅而非严格零的初始扰动。

最小合同 B2：

```text
C_fused = C_patch
        + beta * W_f(concat(C_patch, broadcast_N(C_global)))
W_f: 2d -> d
```

d=32 时增加 `2*d*d + d + 1 = 2,081` parameters（含 bias 与 beta）。同样存在 `beta=0` 的严格等价/首步 global 梯度为零与 `beta=1e-3` 的立即监督/小扰动取舍。两者都不需要 self-attention；B1 是最小广播交互，B2 提供可学习的 patch/global 融合，均为 TimeXer-inspired 而非原实现复刻。

对 parallel ETTm1，N=16、C=7：B1 每样本约增加 `7*16*32=3,584` 个逐元素融合操作，B2 的 fusion Linear 约 `7*16*2*32*32=229,376` MACs。UrbanEV T=12、P=3 时 N=4：single 分别约 128 个操作与 8,192 MACs；F4 parallel（C=11）分别约 1,408 与 90,112。两个合同都须建立新 candidate identity、从 scratch 或使用显式且完整的来源合同；不得以 `strict=False` 让现有 T2 checkpoint 静默补参数。

| 方向 | forecast gradient 到 global query | prediction 初始合同 | state context | 参数/来源边界 |
|---|---|---|---|---|
| A：Pool(C_patch) | 不再保留 global query；context 直接受 forecast supervision | 与 T2 完全不变 | forecast-supervised patch 汇总 | 删除 global 参数；偏离 global-token 语义；只修 state |
| B1：broadcast residual | beta 非零时有 | beta=0 严格等价；1e-3 小扰动 | 保留 C_global | +1；最小 global-to-patch 路径 |
| B2：Linear fusion | beta 非零时有 | beta=0 严格等价；1e-3 小扰动 | 保留 C_global | +2,081；表达力与成本更高 |

后续必须等待用户在“保留 T2、采用 patch-context pooling、实现一个 global-mediated patch 候选、或其他处理”之间作出明确决定。本轮不选择、不实现、不训练上述方向。

### 25.11 阶段边界与停止点

本轮只修改 canonical 与本 milestone；未修改任何 Python、测试、shell、配置模板、数据或 artifact，未创建 optimizer，未执行 optimizer step，未训练新候选，未访问 UrbanEV test。未启动 P2/T3，T4/T5 继续排除；未进入 M5/M7，未实现 StateAdapter、Graph Mode 或空间模块。

审计使用 `/public/home/yueweiting/miniconda/envs/amd/bin/python -B`，临时目录为 `/tmp/m4_t2_global_gradient_audit_9oO8pV`，仅包含一次性审计脚本/终端结果，完成最终核验后删除。M4 状态保持 `In Progress`；两份文档均不 stage、不 commit、不 push，等待用户与 ChatGPT 审核。

## 26. 第九轮：T2G Global-Mediated Patch-Conditioned TEB 工程实现

### 26.1 第八轮文档 closure

第八轮 canonical 与本 milestone 在保持字节不变的前提下完成一个文档 commit：

```text
commit: daa8b2685a419d0e59cb4f4d184ea77cb3c2ad16
parent: 4979e5fd9738da28e2999edf8a6b7dc1ff0266d9
title: docs(m4): record T2 global gradient audit
push: origin/AMD-paper-repro-custom-modules-v1 succeeded
```

提交范围精确为两份文档；提交后 local/tracking/live remote 相同、ahead/behind=`0/0`、worktree/index clean。closure 版本 canonical SHA-256 为 `56cca7ccece13b7556a59c284d153750e2ffecda59bb1f6607d94779b9346764`，本 milestone SHA-256 为 `0f639b995e4602401ac7e0b0834c20aaa0f90afff86cd8a0b2c0f975a6437a6a`。当时 source fingerprint 仍为 `883bbbef80d5a7a13d5353d3dc08e549159dcfbf3beed40a307759db4e20a117`（18 files）；M0-M3、baseline 及冻结源码未变化。

### 26.2 用户决定、来源边界与候选身份

用户决定实现 T2G，并固定：global cross-attention 使用 `q_global + A_global` residual 后接 LayerNorm；patch 直接查询外生变量后不增加 `Q_patch` residual；不把 patch residual 与 global-mediated interaction 捆绑在同一候选。T3 confidence gate 继续暂缓，T4/T5 继续排除，TEB-first 顺序不变，P2 不启动。

TimeXer 的 target patch/global endogenous tokens 先做带 residual+LayerNorm 的 endogenous self-attention，global token 查询 exogenous variate tokens 后也使用 global residual+LayerNorm；原结构不让各 patch 直接查询外生 tokens。T2/T2G 的 patch-direct-query 路径属于 TimeXer-inspired 改造。T2G 固定身份：

```text
class = GlobalMediatedPatchTargetExogenousBridge
variant = el-amd-m4-t2g-global-mediated-patch-teb-v1
ablation_id = M4_T2G
teb_architecture = global_mediated_patch_v1
```

它是 T2 的单因素 M4 工程候选，不是 T3、最终 TEB 或最终 EL-AMD，不覆盖 Global v1/T2/M3 variant。

### 26.3 精确结构、初始化和参数

实现严格遵循：

```text
A_patch,A_global = MHA(concat(Q_patch,q_global),E,E)
G_global = LayerNorm_global_bridge(q_global + A_global)
a_patch = 2*sigmoid(Linear_gate([A_patch;broadcast(G_global)]))
F_patch = A_patch + beta_global*a_patch*broadcast(G_global)
delta = crop(unpatch(Linear(d,P)(F_patch)))
H_out = H + gamma_teb*delta
exo_context = G_global（parallel 由 target_idx 选择）
```

`A_patch` 是 raw MHA response；不存在 `Q_patch+A_patch`、patch post-cross norm、patch FFN/self-attention/to-patch attention 或 target-only gate shortcut。gate Linear weight/bias 零初始化，初始 gate 逐元素严格等于 1；`beta_global=1e-3`，`gamma_teb=1e-3`。新增 keys 仅 `global_bridge_norm.weight/bias`、`global_injection_gate.weight/bias`、`beta_global`，合计 130 parameters。实测模块参数为 ETTm1 `39,491`、UrbanEV `5,606`；fixed sinusoidal buffer 不进入 state dict。

`beta_global=0`、相同 T2 base weights 时，T2G hidden output 与 patch-output projection input 对 T2 的误差均通过 `atol/rtol=1e-6`。production `beta=1e-3` 的真实 ETTm1 validation batch 初始差异为：prediction `4.47034836e-7`、MoE `0`、raw `A_patch=0`、raw `A_global=0`、patch projection input `0.00384790450`、delta patch `0.00238600373`，全部 finite；state source 因 `G_global` 语义变化而最大差 `3.17841768`。

### 26.4 工程接入与文件范围

- 新增 `models/modules/global_mediated_patch_target_exogenous_bridge.py`：独立 T2G class；T2 文件不改。
- 修改 `models/modules/__init__.py`：公开 T2G class 与固定合同常量。
- 修改 `models/tsAMD_enhanced.py`：条件实例化 T2G，保持 DDI/PMCR/AMS/state-source 路由，限制 strict same-structure restore。
- 修改 `main.py`：增加独立 variant/ablation/architecture、六个显式 T2G-only 字段、scientific/comparison/checkpoint/manifest/resume/artifact identity；旧 Global/T2 字段与 hash 语义不变。
- 修改 `summarize_results.py`：严格核验 T2G candidate contract、checksum、path/config/manifest 和 duplicate identity，与 Global/T2 分组隔离。
- 新增 `tests/test_global_mediated_patch_teb.py`、`tests/test_global_mediated_patch_teb_parallel.py`、`tests/test_global_mediated_patch_teb_checkpoint.py`。
- 修改 `tests/test_tsAMD_enhanced.py`、`tests/test_public_architecture.py`、`tests/test_runner.py`、`tests/test_summarize_results.py`，覆盖路由、旧 identity/hash、schema-v2 synthetic artifact、resume/duplicate/tamper。
- 修改 canonical 与本 milestone，且未创建第二份 M4 文档。

T2G-only fields 为 global residual、patch residual=`none`、scalar-per-patch gate、gate input、identity gate init 与 beta init；它们条件进入 T2G scientific/comparison config、manifest candidate contract、checkpoint/resume mismatch 和 summarizer。Global/T2 的 historical scientific/comparison fixture 原值通过。T2G 只允许 from scratch 或同结构 `strict=True`；Global↔T2G、T2↔T2G、partial/unexpected/shape/patch mismatch、`strict=False` 与全部 source-kind importer 均在写参前拒绝，失败不污染 parameter/buffer。

### 26.5 真实 batch smoke 与 production-loss 梯度

两项均为真实 production loader、固定 seed、`model.train()` 下单次 `MSE(prediction,y)+selector auxiliary` backward；未创建 optimizer，未执行 `optimizer.step()`，未训练 epoch，未写 artifact。

| 项目 | ETTm1 parallel | UrbanEV F4 single |
|---|---:|---:|
| input / target | `[128,512,7]` / `[128,96,7]` | `[4,12,11]` / `[4,1]` |
| prediction | `[128,96,7]` | `[4,1,1]` |
| state_source | `[128,1056]` | `[4,56]` |
| target_idx / aux_idx | `6 / []` | `0 / [1,...,10]` |
| prediction MSE / auxiliary / total | `0.521593213 / 0.111157767 / 0.632750988` | `2.445301294 / 7.524266243 / 9.969567299` |

全部输出 finite。下表为 raw backward 的 `L2 / max_abs / exact_nonzero/total`，所有列均 `grad_is_none=False`、finite=True：

| 梯度对象 | ETTm1 parallel | UrbanEV single |
|---|---:|---:|
| q_global | `4.33190e-10 / 3.97319e-11 / 28672/28672` | `8.07252e-8 / 2.36768e-8 / 128/128` |
| A_global | `4.32921e-10 / 3.96727e-11 / 28672/28672` | `8.10313e-8 / 2.30965e-8 / 128/128` |
| G_global（forecast fusion tensor） | `4.51643e-10 / 4.31116e-11 / 28672/28672` | `8.30049e-8 / 2.45561e-8 / 128/128` |
| gate | `4.59203e-10 / 6.61888e-11 / 14336/14336` | `4.17723e-8 / 2.93940e-8 / 16/16` |
| A_patch | `4.44308e-7 / 1.13514e-8 / 458752/458752` | `1.15847e-4 / 2.16127e-5 / 512/512` |

返回给 `state_source` 的 squeezed `exo_context` 视图未被 production loss 直接消费，因此其 retained grad 为 None；这不影响 forecast fusion 使用的 pre-squeeze `G_global` 获得非零梯度。关键参数组均 finite/nonzero：

| 参数组 | ETTm1 L2 / nonzero | UrbanEV L2 / nonzero |
|---|---:|---:|
| global query projection+norm | `3.50539e-8 / 16480` | `3.32218e-7 / 480` |
| global bridge norm | `1.47190e-9 / 64` | `1.17490e-7 / 64` |
| gate Linear | `2.24847e-9 / 65` | `1.39323e-7 / 65` |
| beta_global | `9.52255e-8 / 1` | `3.67939e-6 / 1` |
| patch query | `4.10442e-7 / 1120` | `2.91526e-5 / 192` |
| patch output | `2.51531e-6 / 1056` | `3.20917e-4 / 99` |
| gamma_teb | `1.82245e-4 / 1` | `1.49014e-2 / 1` |
| AMD backbone | `2.21932 / 10244742` | `21.62726 / 229573` |

UrbanEV 的四个 `fc_blocks.0` 参数属于该配置未使用分支，grad=None，与 T2G 路径无关。ETTm1/UrbanEV 的 global、gate、beta 和 patch 路径均在真实 production forecast loss 下立即获得任务梯度，修复了 T2 第八轮确认的 global-query 专属断开。

公共 AMD 初始化 parity：同 seed/config 下 T2 与 T2G 的 60 个 AMD parameter/persistent-buffer keys 逐元素一致，`max_abs_error=0`、mismatch 为空。现有 T2 h96 `best.pt` 对第八轮 closure class 与当前 class 均同结构 `strict=True`、missing/unexpected 为空；同一真实 validation batch的 prediction、MoE loss、state_source 均 `max_abs_error=0` 且 `torch.equal=True`。

### 26.6 测试结果

| 测试组 | 结果 | failed | skipped | unittest time / wall |
|---|---:|---:|---:|---:|
| T2G module/parallel/checkpoint | 16/16 | 0 | 0 | `0.298 / 1.91 s` |
| T2 保护 | 19/19 | 0 | 0 | `0.376 / 2.41 s` |
| Global TEB v1 保护 | 17/17 | 0 | 0 | `0.375 / 2.09 s` |
| AMDEnhanced/architecture/runner/summarizer | 82/82 | 0 | 0 | `5.258 / 7.81 s` |
| PMCR/M1 保护 | 36/36 | 0 | 0 | `3.251 / 4.92 s` |
| 完整 discover 回归 | **188/188** | **0** | **0** | `9.228 / 17.98 s` |

CUDA 可用，设备为 NVIDIA A800 80GB PCIe；T2G、T2、Global、PMCR 和 AMD CUDA 分支实际执行。既有 165 个测试全部保留，未删除、放宽或改写失败测试；新完整回归增加到 188。既有 parity `1e-6` 门槛未放宽。

### 26.7 Source fingerprint、保护资产与阶段边界

实现后 source fingerprint 为 `221c049a20ee17aa5ca806bf3e4e59e66361128df55c1c0572a65322a48844bb`，算法 `sha256_length_prefixed_relative_path_and_content_v1`，19 files；新 T2G module 已纳入，模块 SHA-256 为 `6a6a6d1ab6f8b4b7d8f79c188049e8b89baafcfbb45ccb1b89195af4bac5e32d`。

冻结 `models/tsAMD.py`、Global TEB v1、PMCR v1、T2 module、M0-M3 与 baseline tag 均未修改；旧 T2 state keys、参数量、strict restore、prediction/state parity 和 Global/T2 scientific identity 保护通过。UrbanEV/ModernTCN/TimeXer 参考仓库保持 clean。

第九轮没有训练 T2G，没有生成真实 T2G artifact，没有运行任何正式评价数据集 test，没有实现 P2/T3/T4/T5，没有进入 M5，也没有实现空间模块。未来 T2G development 必须执行已预登记的 normal exogenous、batch permutation 与 `A_global` bypass 诊断，不得将 target shortcut 带来的改善直接归因于外生变量。M4 状态继续为 `In Progress`；T2G 代码、测试、canonical 与本 milestone 均保持未 stage、未 commit、未 push，等待 review。


## 27. 第十轮：T2G Git closure、同源码 T2 对照与 ETTm1 development 实验

### 27.1 T2G closure、测试资产与最终回归

第九轮已审核的 14 文件 T2G 实现以一个 commit 完成 closure 并推送：

```text
commit: 636cb09e7a446350050db36f1be72ff609df58b8
parent: daa8b2685a419d0e59cb4f4d184ea77cb3c2ad16
title: feat(m4): implement global-mediated patch TEB candidate
push: origin/AMD-paper-repro-custom-modules-v1 succeeded
```

提交范围精确为第九轮批准的 14 个代码、测试和文档文件；提交后 local/tracking/live remote 相同，ahead/behind=`0/0`，worktree/index clean。closure 版本 canonical SHA-256 为 `28a17672e1b9dbf4ba66db666382e36271140f957e3257661ffae5e2204bb42c`，本 milestone SHA-256 为 `8baf43438aa35aa2c40aae6c826347b6a6ecb13d02693c9b3142d5f75fa95323`。source fingerprint 为 `221c049a20ee17aa5ca806bf3e4e59e66361128df55c1c0572a65322a48844bb`（算法 `sha256_length_prefixed_relative_path_and_content_v1`，19 files）。

以下三个新增测试固定为 `permanent_regression_test` 并随实现提交，未删除或改写：

- `tests/test_global_mediated_patch_teb.py`：保护 T2G 数学、初始化、无 patch residual、beta=0 T2 parity 和 forecast gradient。
- `tests/test_global_mediated_patch_teb_parallel.py`：保护 parallel owner mask、一次向量化 MHA、变量置换等变和 target context 选择。
- `tests/test_global_mediated_patch_teb_checkpoint.py`：保护同结构 strict restore、候选隔离、失败无参数污染和 artifact identity。

一次性初始化、命令构造、artifact 汇总、full-pass wrapper 与 counterfactual 脚本仅位于 `/tmp/m4_t2g_development_GBjEYgL5/`，不进入 `tests/` 或 Git；最终保护核验后已删除该目录。最终回归是在 closure 前、未修改实现字节的条件下重新执行：

| 测试组 | 结果 | failed | skipped | wall |
|---|---:|---:|---:|---:|
| T2G module/parallel/checkpoint | 16/16 | 0 | 0 | `2.17 s` |
| T2 保护 | 19/19 | 0 | 0 | `1.72 s` |
| Global TEB v1 保护 | 17/17 | 0 | 0 | `1.71 s` |
| AMDEnhanced/architecture/runner/summarizer | 82/82 | 0 | 0 | `7.81 s` |
| PMCR/M1 保护 | 36/36 | 0 | 0 | `5.05 s` |
| 完整 discover 回归 | **188/188** | **0** | **0** | `11.08 s` |

CUDA 可用，T2G/T2/PMCR CUDA 测试实际执行；既有 165 tests 与 `1e-6` parity 门槛均保留。

### 27.2 同源码初始化与命令公平性

四个 horizon 均按 production runner 的 seed/config/factory 从 scratch 构造 T2-refresh 与 T2G。公共 AMD 的 60 个 parameter/persistent-buffer keys 以及两者共享的 19 个 T2-base keys（patch/global query projection/norm、exogenous projection/norm、MHA、patch output、gamma）均为相同 key/shape/dtype/value：

| Horizon | AMD keys / max error / mismatch | T2-base keys / max error / mismatch | fixed PE |
|---:|---|---|---|
| 96 | `60 / 0 / none` | `19 / 0 / none` | `[1,16,32]`, exact |
| 192 | `60 / 0 / none` | `19 / 0 / none` | `[1,16,32]`, exact |
| 336 | `60 / 0 / none` | `19 / 0 / none` | `[1,16,32]`, exact |
| 720 | `60 / 0 / none` | `19 / 0 / none` | `[1,16,32]`, exact |

T2G 额外 state keys 精确为 `teb.beta_global`、`teb.global_bridge_norm.{weight,bias}`、`teb.global_injection_gate.{weight,bias}`。因此本轮 T2/T2G 差异不来自公共初始化漂移。

命令逐 horizon 从第七轮 T2 `command.txt` 解析。T2-refresh 仅改变 artifact root；T2G 相对 refresh 仅改变 variant、ablation、派生 architecture/display name 和六个 T2G-only contract fields。dataset/data/split/scaler/feature order、task/target、T/horizon、seed/epochs/batch/optimizer、AMD 主干、Global/Patch TEB 共用参数、best rule 与 metric aggregation 均无未授权差异。

### 27.3 八个 completed artifacts

共同合同：ETTm1、parallel multivariate、all 7 variables、`T=512`、seed 2024、10 epochs、schema-v2；source fingerprint `221c049a...844bb`，data fingerprint `6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e`。每个 run 均 `status=completed`、history=10、best/last present、metrics finite、manifest/config/path identity 一致、13/13 checksums 且系统 `sha256sum -c` passed；不存在残留 staging 或重复 identity。

Artifact root：`/public/home/yueweiting/大论文/AMD/artifacts/m4-development/ettm1-stage-c-t2g-v1/`。

| Model | H | run_id | best epoch | config hash | best / last SHA-256 | wall / size |
|---|---:|---|---:|---|---|---:|
| T2-refresh | 96 | `20260831T125614.088680Z-28c264dc` | 10 | `7db5ee38...c89404` | `91d68322...25eb2` / `638c93f2...06ce52` | `184.159 s / 197.075 MiB` |
| T2-refresh | 192 | `20260831T125946.781047Z-d356c00a` | 7 | `8b15bf66...598e52` | `067ea9ab...2afc8c` / `1cd5bf16...6d3645` | `202.728 s / 227.095 MiB` |
| T2-refresh | 336 | `20260831T130339.244629Z-cd81fda5` | 6 | `24951e61...1af01` | `ed96ecf3...6ee02` / `1e8f4146...f4327e` | `198.628 s / 272.144 MiB` |
| T2-refresh | 720 | `20260831T130809.284851Z-fe91d1e1` | 6 | `08d774ff...f09dc` | `e504b6de...d6dc7` / `c29e927f...40104` | `193.971 s / 392.187 MiB` |
| T2G | 96 | `20260831T131316.843353Z-28429be0` | 10 | `6a3c2040...a1b4` | `e70d13e4...67c0b` / `5e3eb192...84cd2` | `203.548 s / 197.129 MiB` |
| T2G | 192 | `20260831T131718.867739Z-a96e5e00` | 7 | `8d89d06b...d28b0` | `4f53f936...e7cea` / `55eb36ee...aa295` | `195.997 s / 227.115 MiB` |
| T2G | 336 | `20260831T132101.766850Z-23f16d8c` | 6 | `fd1f9bba...c7e0` | `3706d784...c37282` / `7eb9b9ce...5523c` | `203.194 s / 272.134 MiB` |
| T2G | 720 | `20260831T132455.900770Z-65ee2812` | 6 | `28190d49...891a` | `32c94a8a...e0342` / `7fc7f2b8...eff1` | `191.253 s / 392.206 MiB` |

完整路径由上述 root 依次接：`<variant>/ETTm1/parallel_multivariate/all/horizon_<H>/fold_official/seed_2024/<run_id>`。T2-refresh 总 wall/space 为 `779.486 s / 1088.502 MiB`，T2G 为 `793.992 s / 1088.584 MiB`；T2G 每个 horizon 仅多 130 parameters，模型总参数依次为 `10,284,233 / 11,857,865 / 14,218,313 / 20,512,841`。

### 27.4 Normal performance：T2G vs 同源码 T2-refresh

以下均为 **ETTm1 development-only；test used for candidate development；not a formal paper result**。负相对值表示 T2G 误差下降。

| H | Val MSE T2 / T2G / rel% | Val MAE T2 / T2G / rel% | Test MSE T2 / T2G / rel% | Test MAE T2 / T2G / rel% |
|---:|---:|---:|---:|---:|
| 96 | `0.37115965 / 0.37139266 / +0.06278` | `0.40300325 / 0.40304378 / +0.01006` | `0.29777238 / 0.29870051 / +0.31169` | `0.35107041 / 0.35167332 / +0.17174` |
| 192 | `0.50013897 / 0.49992212 / -0.04336` | `0.46881942 / 0.46879847 / -0.00447` | `0.33464689 / 0.33472641 / +0.02376` | `0.37234873 / 0.37242933 / +0.02165` |
| 336 | `0.64641350 / 0.64621341 / -0.03095` | `0.52858440 / 0.52855252 / -0.00603` | `0.36694654 / 0.36708114 / +0.03668` | `0.38916411 / 0.38927930 / +0.02960` |
| 720 | `0.95247168 / 0.95236755 / -0.01093` | `0.63922116 / 0.63920692 / -0.00223` | `0.42190152 / 0.42203582 / +0.03183` | `0.41770466 / 0.41786549 / +0.03850` |
| Macro | `0.61754595 / 0.61747393 / -0.01166` | `0.50990705 / 0.50990042 / -0.00130` | `0.35531683 / 0.35563597 / +0.08982` | `0.38257198 / 0.38281186 / +0.06270` |
| mean horizon rel% | `-0.00562` | `-0.00067` | `+0.10099` | `+0.06537` |

固定 development primary（test MSE macro）与 secondary（test MAE macro）均小幅变差，且四个 test horizon 的 MSE/MAE 全部同向变差；validation 只有接近数值平局的微小改善。最后三轮 validation MSE：

| H | T2-refresh epochs 8-10 | T2G epochs 8-10 |
|---:|---|---|
| 96 | `0.37340881,0.37270323,0.37115965` | `0.37317685,0.37268224,0.37139266` |
| 192 | `0.50175004,0.50462597,0.50097802` | `0.50143894,0.50398372,0.50025376` |
| 336 | `0.64734166,0.64712390,0.64862325` | `0.64707475,0.64687374,0.64817753` |
| 720 | `0.95743055,0.95832105,0.96076194` | `0.95764866,0.95848619,0.96110287` |

T2-refresh 与第七轮旧 T2 的 4 horizons × 4 metrics、macro、best epoch 均逐值完全相同，所有 relative changes=`0`；这既验证同源码 refresh，也排除 refresh 自身漂移。

### 27.5 补充对照：Global TEB v1 与 AMD

下表为 T2G 相对既有对照的每-horizon百分比变化；AMD 是 legacy cross-source 补充证据，不具有本轮同源码 T2-refresh 的控制强度。

| 对照 | H | Val MSE% | Val MAE% | Test MSE% | Test MAE% |
|---|---:|---:|---:|---:|---:|
| Global v1 | 96 | `-1.23081` | `-0.42795` | `-1.02827` | `-0.31267` |
| Global v1 | 192 | `+1.43720` | `+1.47334` | `-0.46202` | `-0.16393` |
| Global v1 | 336 | `-0.68942` | `-0.40988` | `-0.96358` | `-0.66710` |
| Global v1 | 720 | `+0.05458` | `-0.00135` | `-0.82384` | `-0.21816` |
| Global v1 macro | — | `-0.06116` | `+0.14192` | `-0.81814` | `-0.34120` |
| Global v1 mean-horizon | — | `-0.10711` | `+0.15854` | `-0.81943` | `-0.34046` |
| AMD | 96 | `+0.35646` | `+0.13345` | `+2.81228` | `+1.99588` |
| AMD | 192 | `+0.75779` | `+1.15107` | `+1.72347` | `+1.46013` |
| AMD | 336 | `-0.22476` | `-0.00638` | `+0.90948` | `+0.72101` |
| AMD | 720 | `-0.35048` | `-0.04040` | `-0.01943` | `+0.45128` |
| AMD macro | — | `+0.01109` | `+0.27433` | `+1.21440` | `+1.11634` |
| AMD mean-horizon | — | `+0.13475` | `+0.30943` | `+1.35645` | `+1.15707` |

T2G 的 Global-v1 test macro 较低，但其同源码、单因素 T2-refresh 对照优先级更高；对 AMD 的 development primary/secondary 均较差。本节不把跨源码补充比较升级为正式性能结论。

### 27.6 T2G 参数训练动力学

初始值统一为 `gamma=0.001`、`beta=0.001`、gate weight/bias=0、bridge-norm weight=1/bias=0。表中 q-proj/q-norm 为 checkpoint 相对同 seed/factory 初始参数的 L2；bridge 列为 weight-vs-one/bias L2。

| H | epoch best/last | gamma best/last | beta best/last | gate W L2 best/last | gate bias best/last | bridge W/B best/last | q proj / q norm best→last |
|---:|---:|---:|---:|---:|---:|---|---|
| 96 | `10/10` | `.085178/.085178` | `.128975/.128975` | `.467343/.467343` | `.125871/.125871` | `.360106/.255264 → same` | `1.205312/.165088 → same` |
| 192 | `7/10` | `.075040/.084526` | `.097404/.125887` | `.318294/.407148` | `.093994/.121567` | `.192930/.219875 → .341179/.328401` | `1.042862/.151909 → 1.150183/.156248` |
| 336 | `6/10` | `.066730/.082290` | `.087039/.125917` | `.340425/.467806` | `.081717/.113904` | `.136797/.116822 → .306465/.186592` | `1.013722/.151015 → 1.183779/.124957` |
| 720 | `6/10` | `.068672/.082270` | `.087872/.124193` | `.390356/.500924` | `.072907/.091427` | `.154193/.131322 → .324529/.143432` | `.990252/.161216 → 1.114232/.161030` |

Raw gradient 非零不等于参数产生有意义移动；本轮端点证明 beta、gate、bridge norm、global query 与 gamma 均已明显离开初始化，初始约 `1e-6` 的 global 写回路径确实参与了训练，而不是只在理论上可导。

### 27.7 Gate、表示范数与 global contribution

每行统计完整 split 的样本×变量×patch（norm 统计按相应 token）。`r_global=||beta*gate*G_global||/(||A_patch||+eps)`。

| H/split | gate mean/median/p10/p90/p99/min/max | `<.1 / >1.9` | r_global mean/median/p10/p90/p99/max |
|---|---|---:|---|
| 96/val | `1.4428/1.5775/.7514/1.8976/1.9583/.0464/1.9903` | `.0541%/9.5867%` | `.2149/.2017/.1022/.3411/.5325/1.7747` |
| 96/test | `1.4101/1.5541/.6573/1.8990/1.9581/.0423/1.9918` | `.0634%/9.8274%` | `.2113/.1992/.0855/.3424/.5401/1.5259` |
| 192/val | `1.1875/1.2454/.6932/1.5756/1.7160/.2441/1.8333` | `0/0` | `.1253/.1056/.0400/.2372/.3866/1.9209` |
| 192/test | `1.1942/1.2574/.7097/1.5641/1.7036/.2551/1.8294` | `0/0` | `.1313/.1109/.0400/.2499/.4026/1.0581` |
| 336/val | `1.3266/1.3493/.9818/1.6401/1.7735/.2922/1.9113` | `0/.0003%` | `.1223/.1043/.0647/.2058/.3174/.8167` |
| 336/test | `1.3190/1.3421/.9717/1.6321/1.7818/.2894/1.9223` | `0/.0004%` | `.1290/.1128/.0698/.2115/.3198/.8972` |
| 720/val | `1.5531/1.7571/.7649/1.9502/1.9800/.0414/1.9938` | `.1652%/25.6461%` | `.1204/.1136/.0717/.1814/.2828/.8475` |
| 720/test | `1.5461/1.7349/.7971/1.9457/1.9787/.0368/1.9938` | `.1195%/23.3997%` | `.1226/.1161/.0730/.1835/.2763/.6937` |

| H/split | mean ||A_patch|| / ||q|| / ||A_global|| / ||G|| / ||injection|| | mean A_global/q | mean cosine(q,A_global) |
|---|---|---:|---:|
| 96/val | `6.0483/5.5734/5.9127/6.0878/1.1343` | `1.0607` | `.0110` |
| 96/test | `6.0501/5.5703/5.7004/6.0822/1.1080` | `1.0232` | `-.0263` |
| 192/val | `7.3077/5.5605/6.7743/5.9298/.6889` | `1.2181` | `.0401` |
| 192/test | `7.1302/5.5585/6.4962/5.9265/.6923` | `1.1686` | `.0383` |
| 336/val | `6.5186/5.5288/6.1361/5.7958/.6698` | `1.1098` | `-.1212` |
| 336/test | `6.1079/5.5288/5.8665/5.7951/.6659` | `1.0611` | `-.0723` |
| 720/val | `7.4314/5.5465/6.9218/5.8321/.7966` | `1.2474` | `.0587` |
| 720/test | `7.2413/5.5445/6.8063/5.8305/.7927` | `1.2272` | `.1040` |

变量顺序为 `[HUFL,HULL,MUFL,MULL,LUFL,LULL,OT]`；以下为 gate mean（变量列表；随后为 patch positions 0-15）：

| H/split | variable means | patch-position means |
|---|---|---|
| 96/val | `[1.4319,1.4739,1.4295,1.4683,1.4477,1.4395,1.4086]` | `[1.3930,1.4252,1.4117,1.4477,1.4825,1.4603,1.4767,1.4873,1.4698,1.4763,1.5205,1.4546,1.4012,1.4121,1.4173,1.3484]` |
| 96/test | `[1.4159,1.4433,1.4143,1.4388,1.4186,1.3860,1.3537]` | `[1.3567,1.3850,1.3491,1.4080,1.4407,1.3980,1.4552,1.4561,1.4132,1.4655,1.4932,1.4193,1.4041,1.3929,1.3725,1.3518]` |
| 192/val | `[1.2556,1.1827,1.2674,1.1972,1.1527,1.1303,1.1267]` | `[1.1840,1.1871,1.1566,1.1416,1.1373,1.1498,1.1654,1.1745,1.1785,1.1713,1.1792,1.2159,1.2578,1.2613,1.2363,1.2033]` |
| 192/test | `[1.2334,1.1910,1.2349,1.1928,1.2058,1.1567,1.1448]` | `[1.1957,1.1967,1.1688,1.1500,1.1449,1.1607,1.1748,1.1779,1.1827,1.1760,1.1822,1.2198,1.2576,1.2644,1.2446,1.2106]` |
| 336/val | `[1.3175,1.3476,1.3097,1.3433,1.3500,1.3093,1.3088]` | `[1.2889,1.3167,1.3397,1.3730,1.3725,1.3758,1.3677,1.3704,1.3861,1.3729,1.3570,1.3156,1.2586,1.2328,1.2301,1.2679]` |
| 336/test | `[1.3322,1.3348,1.3251,1.3361,1.3235,1.3036,1.2779]` | `[1.2759,1.3171,1.3331,1.3529,1.3716,1.3676,1.3458,1.3647,1.3752,1.3570,1.3507,1.3103,1.2465,1.2320,1.2333,1.2710]` |
| 720/val | `[1.5530,1.5783,1.5397,1.5709,1.5650,1.5448,1.5199]` | `[1.6006,1.5786,1.5888,1.6548,1.6885,1.6801,1.6689,1.6560,1.6420,1.6574,1.6376,1.5482,1.4479,1.3147,1.2084,1.2772]` |
| 720/test | `[1.5697,1.5740,1.5625,1.5689,1.5484,1.5448,1.4541]` | `[1.5896,1.5739,1.5758,1.6319,1.6682,1.6703,1.6534,1.6425,1.6340,1.6463,1.6274,1.5576,1.4594,1.3170,1.2106,1.2794]` |

Gate 没有整体停留在 1；h720 有明显高端集中。global injection 相对 raw patch response 的平均范数比例约 12.0%-21.5%，说明 learned global path 数值上并非消失，但不能据此推断它改善了误差。

### 27.8 同 checkpoint 归因诊断

每个 best checkpoint 的 formal forward 与 diagnostic wrapper 在 prediction、MoE、A_patch、A_global、G_global、gate、delta 上最大误差均为 `0`。Normal 指标与 artifact 精确相同；所有四个 checkpoint 均以同结构 `strict=True` 恢复，missing/unexpected 为空。每个状态从同一 loader/RNG 起点完整遍历；loader `num_workers=0`、无独立 generator、无随机 transform/collate。诊断后完整 state dict 的 84 keys 逐元素 `torch.equal`，没有 parameter/buffer 变化。表内为 `changed MSE/MAE (相对 Normal MSE%/MAE%)`：

| Split/H | beta=0 | A_global=0 | q residual=0 | 3 cyclic exogenous permutations mean |
|---|---|---|---|---|
| val/96 | `.36957485/.40209290 (-.48946/-.23592)` | `.36960690/.40214270 (-.48083/-.22357)` | `.37195030/.40341179 (+.15015/+.09131)` | `.37200341/.40353747 (+.16445/+.12249)` |
| val/192 | `.49985033/.46859615 (-.01436/-.04316)` | `.49967321/.46856373 (-.04979/-.05007)` | `.50005275/.46886355 (+.02613/+.01388)` | `.50032300/.46895893 (+.08019/+.03423)` |
| val/336 | `.64682085/.52884255 (+.09400/+.05487)` | `.64622236/.52859670 (+.00138/+.00836)` | `.64640498/.52862573 (+.02964/+.01385)` | `.64642026/.52865378 (+.03201/+.01916)` |
| val/720 | `.95268318/.63929828 (+.03314/+.01429)` | `.95233615/.63918756 (-.00330/-.00303)` | `.95253636/.63927669 (+.01773/+.01092)` | `.95251586/.63931865 (+.01557/+.01748)` |
| test/96 | `.29975946/.35273438 (+.35452/+.30172)` | `.29947864/.35261144 (+.26050/+.26676)` | `.29890542/.35169634 (+.06860/+.00654)` | `.29921720/.35194066 (+.17298/+.07602)` |
| test/192 | `.33474987/.37220388 (+.00701/-.06053)` | `.33456404/.37218740 (-.04851/-.06496)` | `.33482153/.37248480 (+.02842/+.01489)` | `.33492142/.37251134 (+.05826/+.02202)` |
| test/336 | `.36706822/.38925604 (-.00352/-.00598)` | `.36707138/.38929777 (-.00266/+.00474)` | `.36706977/.38927334 (-.00310/-.00153)` | `.36715356/.38928560 (+.01973/+.00162)` |
| test/720 | `.42197853/.41774580 (-.01358/-.02864)` | `.42197845/.41778599 (-.01359/-.01903)` | `.42209515/.41789440 (+.01406/+.00692)` | `.42202666/.41789482 (-.00217/+.00702)` |

三个 permutation 的 MSE/MAE ranges：val h96 `[.37167412,.37232798]/[.40327706,.40379924]`，h192 `[.50011811,.50052663]/[.46887610,.46904150]`，h336 `[.64630769,.64653544]/[.52859653,.52871379]`，h720 `[.95243086,.95260721]/[.63925904,.63938091]`；test h96 `[.29893705,.29949676]/[.35178953,.35209118]`，h192 `[.33480798,.33504393]/[.37245988,.37257084]`，h336 `[.36710637,.36720722]/[.38927553,.38929984]`，h720 `[.42201950,.42203890]/[.41787348,.41792088]`。所有 split 的不可置换 B=1 batch 数为 0。

各状态对内部张量/预测的全 split 最大绝对变化，tuple 顺序为 `prediction/A_patch/A_global/G_global/gate`：

| Split/H | beta=0 | A_global=0 | q residual=0 | permutation max over shifts |
|---|---|---|---|---|
| val/96 | `.17467/0/0/0/0` | `.14778/0/6.51910/4.06700/.67458` | `.13318/0/0/5.20290/.95425` | `.96340/7.52427/6.31766/3.96637/1.78759` |
| val/192 | `.05240/0/0/0/0` | `.06842/0/5.87030/3.56853/.72656` | `.05064/0/0/5.28228/.83853` | `.84711/7.02210/6.29703/3.43142/1.01883` |
| val/336 | `.03533/0/0/0/0` | `.04557/0/6.37727/4.07178/.29909` | `.04334/0/0/5.59749/.58023` | `.43661/7.10389/5.87279/4.02733/.85362` |
| val/720 | `.04511/0/0/0/0` | `.04032/0/5.87905/4.04695/.44353` | `.03911/0/0/5.30544/.51181` | `.76478/8.10504/6.30310/3.89410/1.89221` |
| test/96 | `.12780/0/0/0/0` | `.16836/0/6.01151/3.81889/.66707` | `.10675/0/0/5.48224/.99900` | `.78328/8.05981/6.27515/3.91950/1.87382` |
| test/192 | `.05433/0/0/0/0` | `.04993/0/5.79388/3.67948/.74198` | `.03210/0/0/4.89092/.79981` | `.79639/7.89504/6.90600/3.39083/.88304` |
| test/336 | `.03106/0/0/0/0` | `.04056/0/5.52199/3.78859/.31836` | `.03897/0/0/5.20538/.51361` | `.41032/6.82280/6.89836/3.92455/.88814` |
| test/720 | `.04338/0/0/0/0` | `.05347/0/6.14384/3.72501/.43321` | `.03221/0/0/4.98694/.54124` | `.53046/7.49647/7.02561/3.42424/1.82184` |

解释边界：`beta=0` 是整个 global-mediated writeback 的同-checkpoint bypass；`A_global=0` 保留 `LayerNorm(q_global)`；q-residual bypass 保留 `LayerNorm(A_global)`；permutation 同时破坏 patch/global K/V 对齐。结果显示 global path 被使用但贡献方向依 horizon/split 而变：beta bypass 在 test h96 明显恶化，却在 val h96 改善；A_global bypass 同样不稳定；q residual bypass在多数行造成微小退化；外生 permutation 在全部 validation 以及 test h96/h192/h336 增加 MSE，test h720 MSE 轻微下降而 MAE上升。它证明可测的外生依赖，但不是独立训练消融、可加性分解或因果证明。

### 27.9 Development signal、证据边界与停止点

事实：T2-refresh 精确复现旧 T2；T2G 公共初始化完全公平；T2G global/gate/beta 参数显著移动，global injection 不是数值零路径，外生 permutation 可改变表示与预测。但在最高优先级的同源码单因素比较中，T2G 四个 test horizons 的 MSE/MAE 全部比 T2-refresh 略差，test MSE/MAE macro 分别退化 `+0.08982%/+0.06270%`；validation 仅有 `-0.01166%/-0.00130%` 的近零改善，且同-checkpoint bypass方向不稳定。

因此本轮分类为：**negative-or-negligible development signal**。这是 ETTm1、single seed、10 epochs 的 development 判断，不冻结或淘汰 T2G，不把 T2G 声明为最终 TEB，也不构成正式论文性能结果。它不能回答 EV 场景、多 seed 可重复性或 M5 最终结构；等待用户与 ChatGPT 决定 T2G 是否为 TEB 的 M4 候选终点、是否退回 T2、是否仅做一个有限 TEB repair，以及何时转入 PMCR/P2。

本轮没有实现或启动 P2/T3/T4/T5，没有进入 M5，没有运行 UrbanEV、EPF-PJM、ETTh1、Weather、ECL 或 Exchange test，没有实现 StateAdapter/Graph Mode/空间模块。M0-M3、frozen AMD、Global TEB v1、PMCR v1、T2 与 baseline tag 均未变化。M4 状态保持 `In Progress`；canonical 保持 closure SHA `28a17672e1b9dbf4ba66db666382e36271140f957e3257661ffae5e2204bb42c` 且实验阶段未改。本节作为唯一 M4 milestone 的实验追加保持未 stage、未 commit、未 push，等待 review。

## 28. 第十一轮 Stage A：T2G 结果 closure 与 T3 合同锁定

### 28.1 第十轮结果审查结论

第十轮 T2-refresh 精确复现旧 T2；T2G 相对同源码 T2-refresh 的四个 test horizon MSE/MAE 均轻微退化，test MSE/MAE macro 分别退化 `+0.08982%/+0.06270%`，validation 仅接近数值平局，因此固定分类为 **negative-or-negligible development signal**。T2G 的 global/gate/beta 参数明显离开初始化，global writeback 和外生 permutation 均能改变预测，故结论不是“结构没有训练起来”；它只说明当前简单 global-mediated patch injection 未带来额外 ETTm1 development 收益。

T2G 保留代码、永久测试、strict-restorable artifacts 和独立候选身份，作为可追溯负向工程候选；不删除、不在 M5 前宣称正式淘汰，也不把该结果扩展为其他数据集、所有 global/patch interaction 或 TimeXer global token 设计无效。用户决定不再围绕 T2G 搜索 beta、gate、MLP 或更复杂 global injection，T2G 退出当前 leading-candidate 竞争。

### 28.2 用户授权与 T3 身份

用户正式解除 T3 暂缓并明确授权最后一个 TEB 结构候选：

```text
candidate = T3 Selective Patch TEB
class = SelectivePatchTargetExogenousBridge
variant = el-amd-m4-t3-selective-patch-teb-v1
ablation_id = M4_T3
teb_architecture = selective_patch_v1
```

T3 直接从 T2 派生，不继承 T2G，不含 `q_global` residual、global bridge norm、global-to-patch injection、`beta_global` 或 global injection gate。它不是最终 TEB、最终 EL-AMD 或 M5 frozen variant，不覆盖 Global v1/T2/T2G。

### 28.3 精确 gate 合同

T3 沿用 T2 全部 patch/global token、fixed PE、whole-series exogenous K/V、一次向量化 MHA、owner mask、projection、padding/crop、gamma residual、exo context 与 state source。唯一变化为：

```text
D_patch = patch_output_projection(A_patch)
gate_input = concat(Q_patch,A_patch,-1)
gate_logits = F.linear(gate_input, gate_weight, gate_bias)
g_patch = 2*sigmoid(gate_logits)
D_effective = g_patch*D_patch
```

Gate 固定在含 bias 的 output projection **之后**；禁止 `output_projection(g*A_patch)`。Gate 是 `[B,N,1]` 或 `[B,C,N,1]` 的 shared scalar-per-patch gate，只控制外生 residual 写回量，不把 `Q_patch` 加进 residual。Gate 参数必须显式零 Parameter 初始化，不允许先随机构造 Linear 再清零；初始 gate 严格等于 1，且新增参数不消费 CPU/CUDA RNG。新增 keys 只允许 `teb.patch_confidence_gate_weight/bias`，`d=32` 时增加 65，模块总参数必须为 ETTm1 39,426、UrbanEV 5,541。

T3 的 global query 完全保持 T2 的 state-only、forecast-loss-disconnected 语义；其专属参数在 production forecast loss 下预计零梯度，但 state-control 应证明图未 detach。T3 不实现 StateAdapter、pooling 或其他 global repair。

### 28.4 Identity、恢复与 endpoint 规则

T3-only candidate contract fields 固定为：

```text
teb_patch_confidence_gate = scalar_per_patch_post_projection
teb_patch_gate_input = query_and_attention_response
teb_patch_gate_activation = two_sigmoid
teb_patch_gate_init = explicit_zero_identity
teb_global_prediction_role = state_only_forecast_disconnected
```

这些字段条件进入 T3 config/hash/checkpoint/manifest/resume/summarizer identity，不改变旧 Global/T2/T2G identity。T3 训练 from scratch；只允许同结构 T3 `strict=True` restore，拒绝所有跨候选、partial、mismatch、`strict=False` 和 source-kind importer，失败不污染模型。

Endpoint 规则已预登记：T3 为 positive 时 T3 成为 TEB M4 leading candidate；T3 为 mixed 或 negative-or-negligible 时 T2 保持 leading candidate。无论哪一种，随后登记 `TEB branch reaches M4 candidate endpoint`，但不等于 M5 freeze。本轮完成 T3 证据与裁决后必须停止，不自动启动 P2；T4/T5 继续排除。

本节为 Stage A 文档 closure 前状态；第十轮所有 artifact、指标与诊断原值完整保留。M4 状态继续 `In Progress`。

## 29. 第十一轮 Stage B：T3 implementation review

### 29.1 Stage A 文档 closure

T2G development 结论与 T3 exact contract 已通过单独文档 closure：

```text
commit = 953dfaf514be4d5a5c669b42978de72d3f5bdbcd
parent = 636cb09e7a446350050db36f1be72ff609df58b8
title = docs(m4): conclude T2G development and lock T3 contract
push = origin/AMD-paper-repro-custom-modules-v1
local/tracking/live remote = equal
ahead/behind = 0/0
canonical SHA-256 = bdcb154429463ee75bdbf46dbb83b0e277a0cb267decc87ad5a705c8b2d1af01
M4 closure SHA-256 = ae8390cee192f7e99263327bfee3ec204b670decfb9a9d0458e9ea546d46da6c
```

该 closure 只包含 canonical 与唯一 M4 milestone；M0-M3、baseline tag 与 19-file source fingerprint 未变化。

### 29.2 实现范围与合同核验

T3 以独立 `SelectivePatchTargetExogenousBridge` 实现，直接继承 T2，未继承或组合 T2G。新增模块之外，只修改模块公开出口、`AMDEnhanced`、runner、summarizer 以及四个长期集成测试；新增三个 `permanent_regression_test`。T2、T2G、Global TEB v1、PMCR v1 与 frozen AMD 文件字节未改。

实现的唯一结构变化为 post-projection scalar gate：

```text
D_patch = patch_output_projection(A_patch)
g_patch = 2*sigmoid(F.linear([Q_patch;A_patch], gate_weight, gate_bias))
D_effective = g_patch*D_patch
```

Gate weight/bias 以显式零 Parameter 创建，不消费 RNG；初始 gate 严格为 1。新增 state keys 精确为 `teb.patch_confidence_gate_weight` 和 `teb.patch_confidence_gate_bias`，`d=32` 时增加 65 参数；模块参数量 ETTm1 `39,426`、UrbanEV `5,541`。不存在 T2G keys、patch residual/norm/FFN、learnable position、top-k、Hidden-KV 或第二 selector。T2 common state、fixed PE、CPU/CUDA RNG 与初始 eval/train output parity 均由永久测试保护。

Runner 接受且只接受：

```text
variant = el-amd-m4-t3-selective-patch-teb-v1
ablation = M4_T3
architecture = selective_patch_v1
```

五个 T3-only fields 条件进入 scientific/comparison identity、checkpoint metadata、manifest candidate contract、resume mismatch 和 summarizer。Global/T2/T2G 携带 T3 fields、T3 携带 T2G fields、缺失/错误 gate contract、partial/unexpected/shape mismatch、跨候选恢复、`strict=False` 与 source-kind importer 均被拒绝；失败恢复保持目标 parameter/buffer 逐元素不变。Synthetic schema-v2 artifact 与 duplicate/tamper 拒绝测试通过，没有创建真实 T3 artifact。

### 29.3 测试结果

统一使用 `/public/home/yueweiting/miniconda/envs/amd/bin/python -B`，设置 `GIT_OPTIONAL_LOCKS=0` 与 `PYTHONDONTWRITEBYTECODE=1`：

| 测试组 | 结果 | failed | skipped |
|---|---:|---:|---:|
| T3 permanent regression | 19/19 passed | 0 | 0 |
| T2 protection | 19/19 passed | 0 | 0 |
| T2G protection | 16/16 passed | 0 | 0 |
| Global TEB v1 | 17/17 passed | 0 | 0 |
| AMDEnhanced/architecture/runner/summarizer | 87/87 passed | 0 | 0 |
| PMCR/M1 protection | 36/36 passed | 0 | 0 |
| full discovery | 212/212 passed | 0 | 0 |

T2、T2G、T3 的 CUDA float32 tests 均实际执行；既有 188 tests 全部保留，既有 `1e-6` parity 门槛未放宽。

### 29.4 真实数据单 batch production-loss smoke

仅在 `/tmp/m4_t3_development_3Lcaeu2C/` 执行一次性诊断；没有构造 optimizer、没有 `optimizer.step()`、没有 checkpoint/artifact 写入。总目标严格为 production `MSE(prediction,label) + selector auxiliary loss`。

ETTm1 parallel h96：

```text
x = [128,512,7]
y = [128,96,7]
prediction = [128,96,7]
state_source = [128,1056]
gate = [128,7,16,1]
objective = 0.6327509880
```

UrbanEV M1 production pipeline，F4/fold1/label-horizon3：

```text
x = [128,12,11]
y = [128,1]
prediction = [128,1,1]
state_source = [128,56]
gate = [128,4,1]
target_idx = 0
aux_idx = [1,2,3,4,5,6,7,8,9,10]
objective = 3.6380765438
```

两条路径的 prediction/state/auxiliary 均 finite。ETTm1 与 UrbanEV 的 gate weight/bias、`Q_patch`、`A_patch`、`D_patch`、`D_effective`、`gamma_teb` 与 AMD backbone raw backward gradients 均 finite、nonzero。ETTm1 gate weight/bias L2 分别为 `7.9087433e-7/9.1074583e-8`；UrbanEV 为 `1.6191830e-5/3.7283808e-6`。Global-query projection/norm 的 production task gradient 在两条路径均为严格零，符合预登记的 state-only、forecast-loss-disconnected 合同，不构成实现失败。

### 29.5 Implementation gate 与停止点

Implementation review：**Passed**。当前 executable source fingerprint 为：

```text
algorithm = sha256_length_prefixed_relative_path_and_content_v1
file_count = 20
sha256 = a3766cbc4f79eef2cc6db19020131abfc7f8eb8e4c56677c6eab5e995d952910
new included file = models/modules/selective_patch_target_exogenous_bridge.py
```

T3 尚未完成 performance/development 验收。本阶段未训练 T3、未生成真实 T3 artifact、未实现或启动 P2/T4/T5、未进入 M5/M7，M4 状态继续为 `In Progress`。

## 30. 第十一轮 Stage D-F：T3 development 与 TEB M4 候选终点

### 30.1 T3 implementation closure、测试资产与公平门禁

T3 implementation review 通过后，以一个 commit 完成工程 closure 并推送：

```text
commit = 82da6d0198652dc725b867656b0cf9bc2dc955c2
parent = 953dfaf514be4d5a5c669b42978de72d3f5bdbcd
title = feat(m4): implement selective patch TEB candidate
push = origin/AMD-paper-repro-custom-modules-v1 succeeded
local/tracking/live remote = equal
ahead/behind = 0/0
source fingerprint = a3766cbc4f79eef2cc6db19020131abfc7f8eb8e4c56677c6eab5e995d952910 (20 files)
```

三个 T3 新测试均固定为 `permanent_regression_test` 并保留：

- `tests/test_selective_patch_teb.py`：保护精确 post-projection gate、显式零初始化、参数量、forecast/state gradient 与 single 路径。
- `tests/test_selective_patch_teb_parallel.py`：保护一次向量化 MHA、owner mask、变量置换等变与 shared gate。
- `tests/test_selective_patch_teb_checkpoint.py`：保护同结构 strict restore、跨候选隔离与失败无污染。

一次性命令构造、初始化、artifact 校验、checkpoint dynamics 和 full-split wrapper 仅位于 `/tmp/m4_t3_development_3Lcaeu2C/`，不进入 `tests/` 或 Git；最终保护核验后删除。closure 前定向组和完整回归为 212/212 passed、failed=0、skipped=0，CUDA float32 实际执行。

四个 horizon 的 T2/T3 公共 AMD 60 keys、公共 T2-base 19 keys、fixed PE、模型构造后 CPU/CUDA RNG、冻结真实 batch 均逐元素一致；eval/train matched-RNG 的 prediction/MoE/state/Q/A/raw/effective 最大误差均为 0，T3 gate 严格为 1。T3 额外 keys 精确为 gate weight/bias。

### 30.2 八个 completed artifacts

共同合同为 ETTm1 development-only、parallel multivariate、all 7 variables、T=512、seed 2024、10 epochs、schema-v2；source fingerprint `a3766cbc...52910`，data fingerprint `6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e`。全部 8 runs 均 completed、history=10、best/last present、metrics finite、13/13 checksums 且系统 `sha256sum -c` passed，无残留 staging 或 duplicate identity。

Artifact root：`/public/home/yueweiting/大论文/AMD/artifacts/m4-development/ettm1-stage-d-t3-v1/`。

| Model | H | run_id | best epoch | config hash | best / last SHA-256 | wall / size / params |
|---|---:|---|---:|---|---|---:|
| T2-refresh | 96 | `20260831T160647.766907Z-7fa4da04` | 10 | `edddb313...f2228c` | `d7c0a395...24d59c` / `bc9a68d8...95b8ab` | `204.231 s / 197.007 MiB / 10,284,103` |
| T2-refresh | 192 | `20260831T161201.439116Z-8d4956cf` | 7 | `e575b0dd...998e71` | `0601b418...2efb63` / `2439182d...e7c716` | `194.238 s / 227.015 MiB / 11,857,735` |
| T2-refresh | 336 | `20260831T161551.521593Z-4a8d44b6` | 6 | `173733f3...34b400` | `c0e86311...1d4906` / `444fddb3...6a8e55` | `208.341 s / 272.062 MiB / 14,218,183` |
| T2-refresh | 720 | `20260831T161955.811343Z-3e44c62b` | 6 | `dad5e7e6...46e19a` | `2ea6ed40...bd7fc1` / `111e730c...d5deda` | `201.487 s / 392.118 MiB / 20,512,711` |
| T3 | 96 | `20260831T162524.130527Z-fd0774fb` | 10 | `45550eab...5e9bbb` | `072047e6...808c9a` / `2934bfb7...68350f` | `172.012 s / 196.977 MiB / 10,284,168` |
| T3 | 192 | `20260831T162900.454117Z-cb403993` | 7 | `aebc7673...4a4581` | `d4e1fbc2...fea46e` / `91ee6aad...7d590f` | `174.476 s / 227.003 MiB / 11,857,800` |
| T3 | 336 | `20260831T163235.556343Z-bd670d7d` | 5 | `b078d635...3abbb0` | `8e5ecbf6...39fdf3` / `9117fb9f...6b66f5` | `184.738 s / 272.030 MiB / 14,218,248` |
| T3 | 720 | `20260831T163608.458436Z-cb95d420` | 4 | `71405431...4e31b6` | `df239c66...ef51bb` / `21e37a6d...7f0626` | `193.289 s / 392.128 MiB / 20,512,776` |

T2_refresh 合计：`808.297 s / 1088.203 MiB`。

T3 合计：`724.514 s / 1088.138 MiB`。

### 30.3 Normal performance：T3 vs 同源码 T2-refresh

以下均为 **ETTm1 development-only；test used for candidate development；not a formal paper result**。负值表示 T3 误差下降。

| H | Val MSE control / T3 / rel | Val MAE control / T3 / rel | Test MSE control / T3 / rel | Test MAE control / T3 / rel |
|---:|---:|---:|---:|---:|
| 96 | `0.37115965 / 0.37002611 / -0.30541%` | `0.40300325 / 0.40246724 / -0.13300%` | `0.29777238 / 0.29869404 / +0.30952%` | `0.35107041 / 0.35189844 / +0.23586%` |
| 192 | `0.50013897 / 0.49987798 / -0.05218%` | `0.46881942 / 0.46867267 / -0.03130%` | `0.33464689 / 0.33517127 / +0.15670%` | `0.37234873 / 0.37291844 / +0.15300%` |
| 336 | `0.64641350 / 0.64731730 / +0.13982%` | `0.52858440 / 0.52896106 / +0.07126%` | `0.36694654 / 0.36821109 / +0.34461%` | `0.38916411 / 0.38995937 / +0.20435%` |
| 720 | `0.95247168 / 0.95256170 / +0.00945%` | `0.63922116 / 0.63950634 / +0.04461%` | `0.42190152 / 0.42413013 / +0.52823%` | `0.41770466 / 0.41763543 / -0.01657%` |
| Macro | `0.61754595 / 0.61744577 / -0.01622%` | `0.50990705 / 0.50990183 / -0.00103%` | `0.35531683 / 0.35655163 / +0.34752%` | `0.38257198 / 0.38310292 / +0.13878%` |
| mean horizon rel | `-0.05208%` | `-0.01211%` | `+0.33477%` | `+0.14416%` |

T2-refresh 与第十轮旧 T2 在 4 horizons × 4 metrics、macro 与 best epoch 上逐值完全相同，所有 relative changes=0，排除 refresh 自身漂移。固定 development primary（test MSE macro）退化 `+0.34752%`，secondary（test MAE macro）退化 `+0.13878%`；test MSE 四个 horizon 全部退化，validation macro 仅为接近数值平局的 `-0.01622%/-0.00103%`。

补充比较（均为 T3 相对对照的 macro-mean relative；AMD 为 legacy cross-source 补充证据）：

| Control | Val MSE | Val MAE | Test MSE | Test MAE |
|---|---:|---:|---:|---:|
| Global TEB v1 | `-0.06572%` | `+0.14220%` | `-0.56277%` | `-0.26543%` |
| T2G | `-0.00456%` | `+0.00028%` | `+0.25747%` | `+0.07603%` |
| AMD | `+0.00652%` | `+0.27460%` | `+1.47500%` | `+1.19322%` |

### 30.4 Gate 与 gamma 训练动力学

初始统一为 gamma=0.001、gate weight L2=0、gate bias=0。best/last checkpoint 均以完全同结构 `strict=True` 恢复：

| H | epoch best/last | gamma init/best/last | gate W L2 init/best/last | gate bias init/best/last | T3-vs-T2 best common parameter L2 / max |
|---:|---:|---:|---:|---:|---:|
| 96 | `10/10` | `0.001000/0.076264/0.076264` | `0/0.410775/0.410775` | `0/0.071992/0.071992` | `1.327351/0.045904` |
| 192 | `7/10` | `0.001000/0.068132/0.075767` | `0/0.350725/0.376888` | `0/0.071506/0.077075` | `1.060403/0.045295` |
| 336 | `5/10` | `0.001000/0.056968/0.074652` | `0/0.326615/0.377537` | `0/0.055644/0.052491` | `2.064275/0.038630` |
| 720 | `4/10` | `0.001000/0.049966/0.071145` | `0/0.308644/0.405742` | `0/0.050997/0.077225` | `4.594041/0.037886` |

Gate 与 gamma 均明显离开初始化，故 T3 不是未训练起来。T2/T3 是独立训练模型，公共参数的 checkpoint 差异只描述联合优化轨迹，不能被解释为 gate 的单独因果效果。h336/h720 的 BN `num_batches_tracked` 因双方 selected-best epoch 不同而不同，未混入上表 parameter-only 统计。

### 30.5 完整 split gate 与 residual 分布

Formal forward 与 diagnostic wrapper 在每个 horizon/split 的 prediction、MoE、Q_patch、A_patch、D_patch、gate、D_effective、delta、exo_context/state 上最大误差均为 0；Normal 与 artifact 指标仅有 float64 聚合末位差。诊断后 81 个 state keys 和全部 tensor 逐元素不变。

| H/split | gate mean/median/p10/p90/p99/min/max | `<.1 / >1.9` | effective/raw mean/median/p10/p90/p99/max | gamma-delta/hidden mean/median/p10/p90/p99/max |
|---|---|---:|---|---|
| 96/val | `1.6433/1.6893/1.3661/1.8528/1.9103/0.2929/1.9566` | `0.0000%/1.9030%` | `1.7135/1.7366/1.5832/1.8112/1.8525/1.8971` | `0.1876/0.1841/0.1241/0.2546/0.3219/0.3959` |
| 96/test | `1.6294/1.6813/1.3305/1.8494/1.9088/0.3203/1.9576` | `0.0000%/1.7097%` | `1.6921/1.7226/1.5266/1.8125/1.8529/1.8827` | `0.1783/0.1736/0.1105/0.2530/0.3226/0.4054` |
| 192/val | `1.6104/1.6453/1.3358/1.8351/1.9021/0.4377/1.9503` | `0.0000%/1.1398%` | `1.6229/1.6479/1.3941/1.8008/1.8473/1.8838` | `0.2451/0.2327/0.1184/0.3889/0.5131/0.5957` |
| 192/test | `1.6208/1.6603/1.3367/1.8484/1.9120/0.3946/1.9578` | `0.0000%/2.0077%` | `1.6372/1.6770/1.3831/1.8127/1.8576/1.8941` | `0.2482/0.2287/0.1106/0.4191/0.5535/0.6597` |
| 336/val | `1.4069/1.4390/1.0465/1.7177/1.8266/0.4824/1.8988` | `0.0000%/0.0000%` | `1.4212/1.4517/1.1004/1.6836/1.7885/1.8415` | `0.1135/0.1143/0.0700/0.1539/0.1953/0.2639` |
| 336/test | `1.4181/1.4392/1.1285/1.6783/1.7924/0.4233/1.8927` | `0.0000%/0.0000%` | `1.4337/1.4455/1.2085/1.6432/1.7395/1.8108` | `0.1043/0.1035/0.0608/0.1487/0.1827/0.2132` |
| 720/val | `1.6750/1.7313/1.3842/1.8787/1.9248/0.4850/1.9543` | `0.0000%/4.8782%` | `1.7381/1.7846/1.5664/1.8578/1.8876/1.9053` | `0.1291/0.1303/0.0570/0.1969/0.2364/0.2686` |
| 720/test | `1.6661/1.7158/1.3875/1.8709/1.9257/0.4404/1.9586` | `0.0000%/4.1922%` | `1.7290/1.7669/1.5675/1.8542/1.8908/1.9108` | `0.1224/0.1215/0.0542/0.1889/0.2319/0.2623` |

Raw/effective/gamma-delta/hidden 的按样本 L2 mean（其余 quantiles 保留于本轮临时诊断输出，结束时删除）：

| H/split | D_patch | D_effective | gamma*delta | hidden |
|---|---:|---:|---:|---:|
| 96/val | `91.2997` | `157.3240` | `11.9981` | `63.9698` |
| 96/test | `88.4602` | `151.0470` | `11.5194` | `64.6136` |
| 192/val | `135.3606` | `224.1622` | `15.2726` | `62.3167` |
| 192/test | `136.1017` | `229.1621` | `15.6133` | `62.9375` |
| 336/val | `89.8088` | `123.7502` | `7.0498` | `62.1551` |
| 336/test | `81.9582` | `114.9739` | `6.5499` | `62.8184` |
| 720/val | `90.0999` | `160.2201` | `8.0055` | `61.9955` |
| 720/test | `86.8189` | `153.5601` | `7.6727` | `62.7125` |

变量顺序 `[HUFL,HULL,MUFL,MULL,LUFL,LULL,OT]`。Gate 已明显偏离恒等 1；h96/h720 的 gate 更强、h720 高端集中更明显，但所有 split 的 `<0.1` 比例为 0，`>1.9` 最高约 4.88%，不存在大规模双端饱和。各变量/patch mean：

| H/split | per-variable mean | patch-position mean (0-15) |
|---|---|---|
| 96/val | `[1.6315,1.6483,1.6299,1.6432,1.6418,1.6553,1.6526]` | `[1.5178,1.5729,1.6007,1.6361,1.6524,1.6357,1.6660,1.6659,1.6685,1.6911,1.7061,1.6626,1.6390,1.6512,1.6774,1.6484]` |
| 96/test | `[1.6384,1.6269,1.6385,1.6244,1.6197,1.6215,1.6365]` | `[1.5060,1.5553,1.5717,1.6179,1.6348,1.6106,1.6552,1.6509,1.6462,1.6844,1.6936,1.6478,1.6390,1.6447,1.6618,1.6505]` |
| 192/val | `[1.6192,1.6150,1.6158,1.6126,1.6171,1.6000,1.5933]` | `[1.5066,1.5327,1.6081,1.6478,1.6600,1.6407,1.6251,1.6294,1.6661,1.6861,1.6539,1.6160,1.5574,1.5297,1.5810,1.6260]` |
| 192/test | `[1.6399,1.6310,1.6387,1.6297,1.6203,1.6165,1.5698]` | `[1.5193,1.5468,1.6141,1.6494,1.6597,1.6419,1.6314,1.6413,1.6767,1.6945,1.6635,1.6259,1.5722,1.5518,1.5994,1.6456]` |
| 336/val | `[1.3975,1.4095,1.3968,1.4047,1.4077,1.4180,1.4137]` | `[1.3338,1.3605,1.3969,1.4560,1.4856,1.5019,1.4796,1.4786,1.4734,1.4671,1.4568,1.4068,1.3318,1.2886,1.2713,1.3209]` |
| 336/test | `[1.4245,1.4124,1.4249,1.4103,1.4123,1.4073,1.4351]` | `[1.3476,1.3737,1.4055,1.4653,1.4954,1.5087,1.4888,1.4872,1.4792,1.4770,1.4653,1.4168,1.3470,1.3044,1.2870,1.3410]` |
| 720/val | `[1.6753,1.6839,1.6713,1.6807,1.6774,1.6727,1.6636]` | `[1.6425,1.6387,1.6646,1.7251,1.7579,1.7542,1.7468,1.7434,1.7480,1.7602,1.7421,1.6769,1.6079,1.5431,1.4995,1.5489]` |
| 720/test | `[1.6775,1.6772,1.6758,1.6746,1.6643,1.6644,1.6287]` | `[1.6297,1.6331,1.6513,1.7088,1.7432,1.7383,1.7301,1.7303,1.7337,1.7497,1.7316,1.6729,1.6070,1.5431,1.5027,1.5513]` |

### 30.6 同-checkpoint identity/bypass/permutation 诊断

表内为 `MSE/MAE (相对 Normal MSE%/MAE%)`。Permutation 为三个 deterministic cyclic shifts 的均值；所有 batch 均 B>1，不可置换 batch=0。

| H/split | Normal | identity gate | TEB residual bypass | exogenous permutation mean [MSE range; MAE range] |
|---|---:|---:|---:|---:|
| 96/val | `0.37002611/0.40246724` | `0.36865937/0.40132666 (-0.36936%/-0.28340%)` | `0.37856803/0.40829895 (+2.30847%/+1.44899%)` | `0.37417622/0.40515866 (+1.12157%/+0.66873%) [0.37034361,0.37905389; 0.40271107,0.40787389]` |
| 96/test | `0.29869404/0.35189844` | `0.29561581/0.35054864 (-1.03056%/-0.38358%)` | `0.30047419/0.35553930 (+0.59598%/+1.03464%)` | `0.29883349/0.35216151 (+0.04669%/+0.07476%) [0.29854793,0.29901461; 0.35175487,0.35273795]` |
| 192/val | `0.49987798/0.46867267` | `0.50049780/0.46721342 (+0.12399%/-0.31136%)` | `0.51073205/0.47153012 (+2.17134%/+0.60969%)` | `0.50193360/0.46980618 (+0.41123%/+0.24186%) [0.50007221,0.50467623; 0.46874611,0.47118828]` |
| 192/test | `0.33517127/0.37291844` | `0.33619850/0.37404438 (+0.30648%/+0.30193%)` | `0.34986974/0.38527388 (+4.38536%/+3.31317%)` | `0.33706690/0.37402538 (+0.56557%/+0.29683%) [0.33530801,0.33854695; 0.37296557,0.37474061]` |
| 336/val | `0.64731730/0.52896106` | `0.64777183/0.52917553 (+0.07022%/+0.04055%)` | `0.65052880/0.53087697 (+0.49612%/+0.36220%)` | `0.64817906/0.52941874 (+0.13313%/+0.08652%) [0.64741581,0.64884984; 0.52900980,0.52975431]` |
| 336/test | `0.36821109/0.38995937` | `0.36777101/0.38983405 (-0.11952%/-0.03214%)` | `0.36918858/0.39128040 (+0.26547%/+0.33876%)` | `0.36881768/0.39020023 (+0.16474%/+0.06177%) [0.36823063,0.36929701; 0.38995115,0.39053658]` |
| 720/val | `0.95256170/0.63950634` | `0.95307753/0.63943757 (+0.05415%/-0.01075%)` | `0.95529608/0.64031048 (+0.28706%/+0.12574%)` | `0.95321983/0.63980143 (+0.06909%/+0.04614%) [0.95259303,0.95362112; 0.63953611,0.64002424]` |
| 720/test | `0.42413013/0.41763543` | `0.42360197/0.41750444 (-0.12453%/-0.03136%)` | `0.42424048/0.41845119 (+0.02602%/+0.19533%)` | `0.42454723/0.41789978 (+0.09834%/+0.06330%) [0.42409070,0.42502370; 0.41762122,0.41814203]` |

全 split 最大绝对变化 tuple=`prediction/A_patch/gate/D_patch/D_effective`；permutation 取三个 shifts 的最大值：

| H/split | identity | residual bypass | permutation |
|---|---|---|---|
| 96/val | `0.68497/0.00000/0.95662/0.00000/8.68132` | `1.52188/0.00000/0.00000/0.00000/17.77885` | `1.33291/8.67469/1.15232/11.57911/19.66361` |
| 96/test | `0.57394/0.00000/0.95757/0.00000/8.18635` | `1.18984/0.00000/0.00000/0.00000/16.80619` | `1.40545/9.07153/1.16340/12.69410/19.63291` |
| 192/val | `0.45230/0.00000/0.95028/0.00000/9.99701` | `0.94619/0.00000/0.00000/0.00000/20.68417` | `1.22004/7.01565/1.00130/13.86351/21.96820` |
| 192/test | `0.57598/0.00000/0.95782/0.00000/10.73243` | `1.19720/0.00000/0.00000/0.00000/22.04968` | `1.43519/7.24636/1.20941/13.57427/23.53697` |
| 336/val | `0.22863/0.00000/0.89878/0.00000/4.92275` | `0.52785/0.00000/0.00000/0.00000/10.44240` | `0.91693/6.83113/0.79714/9.07475/15.94980` |
| 336/test | `0.14913/0.00000/0.89270/0.00000/4.79981` | `0.39989/0.00000/0.00000/0.00000/10.36489` | `0.53212/6.47166/0.85515/9.30732/15.22577` |
| 720/val | `0.24018/0.00000/0.95429/0.00000/6.32349` | `0.51516/0.00000/0.00000/0.00000/12.95222` | `0.56317/6.12194/1.20840/7.93113/13.87164` |
| 720/test | `0.22230/0.00000/0.95856/0.00000/6.65238` | `0.45832/0.00000/0.00000/0.00000/13.61181` | `0.48340/6.46848/1.20635/8.67705/14.47310` |

Identity gate 在 test h96/h336/h720 以及 val h96 降低 MSE，在 test h192 与其余 validation 行不稳定；特别是 test h96，强制 gate=1 将 MSE 降低约 1.03%。因此 learned gate 的确移动并产生功能影响，但没有形成稳定选择性收益。TEB residual bypass 在所有 horizon/split 提高误差，说明各 T3 checkpoint 仍依赖 patch TEB residual。三个外生 permutation 的平均 MSE 在所有 horizon/split 均高于 Normal，表示可测的外生依赖；这仍是同-checkpoint 扰动，不是独立训练消融或因果证明。

### 30.7 T3 development signal 与 TEB M4 候选终点

事实：T2-refresh 完全复现旧 T2；T3 与 T2 公共初始化/RNG 完全公平；T3 gate/gamma 已实际移动，gate 分布有变量/patch 差异，外生 permutation 可改变 A_patch、gate、residual 与 prediction。另一方面，最高优先级同源码单因素比较中，T3 四个 test horizon MSE 全部退化，test MSE/MAE macro 分别 `+0.34752%/+0.13878%`；validation macro 近零，identity-gate 反事实还显示 learned gate 在若干关键行不如恒等 gate。

因此 T3 固定分类为：**negative-or-negligible development signal**。按预登记 endpoint 规则：

```text
TEB M4 leading candidate = T2 Patch-Conditioned TEB
TEB branch reaches M4 candidate endpoint
not M5 freeze
```

T2G 与 T3 均保留为可追溯工程候选、永久测试和 strict-restorable artifacts，但不作为当前 leading candidate。本结论只适用于 ETTm1、single seed、10 epochs 的 development 证据；不等同于 M5 frozen TEB，不构成正式论文性能主张，也不能外推到 EV、多 seed 或其他正式数据集。

本轮没有实现或启动 P2，没有新增 TEB candidate，没有调整 gate 维度/激活/位置/输入，没有实现 T4/T5、完整 TimeXer self-attention、StateAdapter/Graph Mode/空间模块，没有进入 M5。下一工程方向必须等待用户审核后才可转入 PMCR/P2。M4 状态继续 `In Progress`；本节结果追加保持未 stage、未 commit、未 push。

### 30.8 补充对照的逐 horizon 相对变化

下表为 T3 相对各既有对照的逐 horizon 百分比变化，列顺序为 Validation MSE / Validation MAE / Test MSE / Test MAE；负值表示 T3 误差更低。Global TEB v1 与 AMD 属既有开发对照，其中 AMD 仍是 legacy cross-source 补充证据；同源码 T2-refresh 始终是最高优先级控制。

| Control | H96 | H192 | H336 | H720 | Macro-mean relative | Mean-horizon relative |
|---|---|---|---|---|---|---|
| Global TEB v1 | `-1.59424/-0.57038/-1.03041/-0.24885%` | `+1.42825/+1.44611/-0.32973/-0.03282%` | `-0.51978/-0.33290/-0.65873/-0.49357%` | `+0.07498/+0.04549/-0.33169/-0.27310%` | `-0.06572/+0.14220/-0.56277/-0.26543%` | `-0.15270/+0.14708/-0.58764/-0.26208%` |
| T2G | `-0.36795/-0.14305/-0.00217/+0.06401%` | `-0.00883/-0.02683/+0.13290/+0.13133%` | `+0.17082/+0.07729/+0.30782/+0.17470%` | `+0.02039/+0.04684/+0.49624/-0.05506%` | `-0.00456/+0.00028/+0.25747/+0.07603%` | `-0.04639/-0.01144/+0.23370/+0.07875%` |
| AMD | `-0.01280/-0.00979/+2.81005/+2.06118%` | `+0.74890/+1.12392/+1.85866/+1.59337%` | `-0.05432/+0.07091/+1.22010/+0.89697%` | `-0.33017/+0.00642/+0.47672/+0.39597%` | `+0.00652/+0.27460/+1.47500/+1.19322%` | `+0.08790/+0.29787/+1.59138/+1.23687%` |

该补充表不改变第 30.7 节裁决：T3 对同源码 T2-refresh 的 development primary/secondary 均退化，故 T3 为 `negative-or-negligible development signal`，T2 仍为 TEB M4 leading candidate。
## 31. 第十二轮 Stage A：撤销旧 TEB endpoint 与目标任务预登记

### 31.1 用户最新决定对第十一轮历史判断的取代

第十一轮的原始指标、artifact、T3 `negative-or-negligible development signal`、T2G `negative-or-negligible development signal` 以及“T2 是已测试 TEB 中表现最好者”等事实继续有效。第十一轮第 30.7 节按当时预登记规则形成的：

```text
TEB M4 leading candidate = T2 Patch-Conditioned TEB
TEB branch reaches M4 candidate endpoint
not M5 freeze
```

是可追溯的历史判断，但已被第十二轮用户最新治理决定正式取代。当前状态修正为：

```text
T2 = best among tested TEB variants
TEB development adequacy gate = not yet passed
TEB branch endpoint = withdrawn
```

T2 相对 Global TEB v1 虽有改善，但相对同输入 AMD baseline 仍是负向 development signal，因而不得把“若干负收益候选中最好的一个”解释为 TEB 已通过。转入 PMCR/P2 前，必须有一个 TimeXer-inspired TEB 候选在同输入、同输出、同源码 AMD control 下取得明确正向 development signal，或由用户明确停止当前路线并授权更换另一篇近三年外生变量模块来源。在此之前 P2 继续阻塞，也不得用 PMCR 潜在收益掩盖 TEB 负收益。

### 31.2 ETTm1 target-exogenous U1/U2 公平合同

本轮不新增 TEB architecture，只对现有 T2 做功能定位一致的目标—外生任务检查。ETTm1 仍为 development-only benchmark；train/validation/test 可用于本轮开发，但不进入 M6 正式主表，不构成未见测试泛化证据，也不表示正式 ETTh1/Weather/ECL/Exchange 改用 `target_exogenous`。

```text
dataset = ETTm1
feature order = [HUFL,HULL,MUFL,MULL,LUFL,LULL,OT]
task_mode = target_exogenous
target = OT
target_idx = 6
aux_idx = [0,1,2,3,4,5]
aux_feature_names = [HUFL,HULL,MUFL,MULL,LUFL,LULL]
seq_len = 512
horizons = [96,192,336,720]
seed = 2024
epochs = 10
```

公平对照固定为：

```text
U1 = AMD-Concat; implementation_variant=el-amd-pmcr-teb-v1; ablation_id=U1; PMCR off; TEB off
U2 = AMD-Concat + T2; implementation_variant=el-amd-m4-t2-patch-teb-v1; ablation_id=M4_T2; PMCR off; TEB on
```

U1/U2 必须共享七变量历史输入、feature type、feature order、OT target、aux 顺序、数据 SHA、57,600 used rows、split、train-only scaler、窗口、seed、batch、optimizer、AMD 主干、target-only 输出、metric space 和 validation-best checkpoint 规则。只允许 implementation/ablation/TEB architecture 及其结构参数、参数量、identity/hash/path 不同；不得用 parallel 七变量输出后切 OT 的 workaround。

### 31.3 Zero-code runner capability gate 与 development 判定

在任何训练前必须只读审计 generic ETTm1 runner 的 CLI、数据/标签、模型输出、criterion shape、artifact/checkpoint/manifest 和 summarizer。只有 verdict 为 `Supported without code changes`，并通过 212/212 回归、真实数据 U1/U2 shape/data parity、U1/frozen AMD OT-slice parity 和公共 AMD 初始化门禁，才可启动八个 run。若不支持，本轮只记录具体阻塞项，不修改代码、不创建 workaround、不训练。

U2 仅在 full-precision test MSE macro 低于 U1、test MAE macro 不高于 U1、至少 3/4 test MSE horizon 改善、validation MSE macro 不高于 U1，且收益非舍入假象或单一 horizon 驱动时分类为 `positive development signal`。否则按证据分类为 mixed 或 negative-or-negligible，并登记 `TEB M4 development adequacy gate remains failed`。本轮不得自动执行 warm-start rescue、更换来源模块、新建 T4/T5/T6、调 patch/d/heads/gate、实现 P2、进入 M5/M7 或实现空间模块。

## 32. 第十二轮 Stage B：generic ETTm1 target-exogenous capability audit

### 32.1 审计范围与结论

从阶段 A 文档 closure 后的 clean HEAD `6638d36336b9f46ed4d636c2b9d88bac80640b34` 只读审计了 `parse_args`、`prepare_args`、`_prepare_enhanced_contract`、generic runtime 构造、`CustomDataLoader`、`CustomDataset.__getitem__`、`AMDEnhanced.forward`、T2 single-target forward、`_prediction_for_loss`/`_assert_prediction_batch`、`train_one_epoch`、`evaluate`、inverse-transform/metric space、scientific config、checkpoint/resume、artifact path/manifest 和 `summarize_results.py`。本轮 capability verdict 固定为：

```text
Not supported without code changes
```

该结论由 criterion 前的真实 label/prediction shape 冲突触发；因此按第 31.3 节门禁，本轮没有运行完整回归、模型 forward/backward smoke、U1/U2 训练或归因诊断，也没有创建本轮 artifact。

### 32.2 已具备的身份、数据与模型能力

- CLI 和 `_prepare_enhanced_contract` 已允许 generic ETTm1 的 `U1`：`el-amd-pmcr-teb-v1`、`target_exogenous`、`feature_type=MS`、`use_pmcr=False`、`use_teb=False`、非空有序 aux schema；U1 构造语义为 `pmcr=None`、`teb=None`，零 `exo_context` 只保留 state 接口。
- T2 variant 已允许相同 generic target task 的 `M4_T2`，并强制 PMCR off、TEB on、d=32、heads=4、dropout=0.1、gamma=0.001 与显式 patch 合同。T2 single-target 路径只更新 `target_idx=6` 的 OT hidden；`aux_idx=(0,1,2,3,4,5)` 按顺序从 RevIN 后的七变量输入抽取六个 K/V，aux 不产生未来标签或独立预测损失，global context 只进入 `state_source`。
- schema-v2 path 已能表达 `ETTm1/target_exogenous/OT/horizon_<H>/fold_official/seed_2024`；scientific config 保存 feature order、target/aux schema、task、fold/horizon、模型开关和 T2 patch fields。checkpoint/resume 绑定完整 scientific hash并使用同结构 `strict=True`；manifest 保留 task/target/horizon/fold/seed，T2 另有 candidate contract。
- summarizer 的 enhanced path identity 显式包含 dataset/task/target/horizon/fold/seed；comparison hash 覆盖 scientific config（仅移除 seed，并加入完成 epoch），所以 U1/U2 及 target/parallel 不会被静默混组，重复 scientific identity + seed 会被拒绝。

### 32.3 真实 loader 事实与确定 blocker

真实 `data/ETTm1.csv` 探针结果：

| feature_type | input sample | label sample | target indices | 是否满足 OT-only 语义 |
|---|---:|---:|---|---|
| `M` | `[512,7]` | `[96,7]` | `[0,1,2,3,4,5,6]` | 否，标签包含全部七变量 |
| `MS` | `[512,7]` | `[96,1]` | `[6]` | 是，唯一语义正确的 feature type |

`MS` 的 batch 标签因此为 `[B,H,1]`。与此同时，`AMDEnhanced(target_exogenous)` 的 prediction 为 `[B,H,1]`，但 `_prediction_for_loss` 当前只接受 prediction `[B,H,1]` 配对 target `[B,H]`，先把 prediction squeeze 为 `[B,H]`；当真实 generic MS target 同为 `[B,H,1]` 时，它在 criterion 前抛出：

```text
RuntimeError: target_exogenous prediction must be [B,H,1] while target is [B,H],
got (1, 96, 1) and (1, 96, 1)
```

因此：

- `feature_type=M` 不能作为 workaround，因为它改变为七变量标签和七变量 metric；
- `feature_type=MS` 是唯一正确数据语义，却无法通过现有 production loss adapter；
- 不允许在模型外事后 squeeze/cut、把 parallel 指标切 OT、包装 runner 或利用 PyTorch broadcasting；
- U1/U2 无法到达 production criterion，后续 validation-best/test/artifact 生命周期因而不能安全启动。

探针同时确认 ETTm1 used rows=`57,600`，split endpoints=`34,560/46,080/57,600`，window counts（h96）=`33,953/11,425/11,425`，列顺序为 `[HUFL,HULL,MUFL,MULL,LUFL,LULL,OT]`。训练 scaler 仍仅 fit train split；M/MS 共用同一 generic backend，但这一数据公平性事实不能绕过 shape blocker。

### 32.4 最小代码缺口与停止点

最小安全 repair 是让正式 task adapter 显式兼容两种合法、非广播目标形态：

```text
prediction [B,H,1] + target [B,H]   -> 显式 squeeze prediction
prediction [B,H,1] + target [B,H,1] -> 保持二者三维并严格同形
其他 shape                           -> 拒绝
```

相较于修改 `CustomDataLoader` 的既有 MS label 维度，优先修正 `_prediction_for_loss`/`_assert_prediction_batch` 可以保留 loader、scaler 和 inverse-transform 合同；但任何 repair 都必须另经用户授权，并补充 generic MS target-exogenous runner/integration 测试、无 broadcasting 断言、U1 frozen-AMD OT parity 及 U1/U2 artifact/summarizer 测试。本轮没有执行这些改动。

Stage B 到此停止：TEB adequacy gate 仍为 `not yet passed`；T2 仍只是 `best among tested TEB variants`。P2、新 TEB、warm-start rescue、来源替换、M5/M7 和空间模块均未启动。M4 状态继续 `In Progress`；本节保持未 stage、未 commit、未 push，等待用户与 ChatGPT 审核。

### 32.5 独立 artifact metadata 缺口

除 criterion shape blocker 外，standard U1 的 manifest 还有一个独立治理缺口：U1 的 `config.resolved.json` 及 checkpoint 内嵌 resolved scientific config 会保存 `target_idx=6`、有序 `aux_idx`/feature names/schema；T2 的 manifest 也会通过 candidate contract 保存 target/aux schema。但 U1 不生成 candidate contract，当前 enhanced manifest 顶层只保存 task、target、horizon、fold、seed 等字段，不保存 `target_idx` 或 `aux_idx`。因此它不能满足本轮“config/manifest/checkpoint 均显式登记 target_idx 与 aux 顺序”的 artifact 合同，即使 shape adapter 另行修复也仍需处理。

最小治理 repair 是在所有 enhanced manifest 的共同身份区显式写入并由 summarizer 校验 `target_idx`、`aux_idx`、feature/aux names 与 schema fingerprint，同时保持旧 schema-v2 artifact 的兼容边界清晰；这需要 runner/summarizer 与永久测试的另行授权修改，本轮未实施。该 metadata 缺口与首要 shape blocker 均属于 `Not supported without code changes` 的依据。

## 33. 第十三轮 Stage A：Generic target-exogenous 最小 repair 授权与预登记

### 33.1 用户授权、继承事实与边界

用户授权修复 generic ETTm1 `feature_type=MS` 的合法 `[B,H,1]` 单目标标签无法通过 production loss adapter 的问题，并为 repair 后新生成的 enhanced `target_exogenous` artifact 增加公共、显式、可校验的 target/aux schema 身份。授权同时覆盖既有三份永久回归测试、真实 ETTm1 MS/UrbanEV 无训练 smoke、synthetic artifact smoke，以及全部门禁通过后的 implementation Git closure；本轮不训练 U1/U2，不生成真实 development artifact。

第十二轮事实原样保留：`feature_type=M` 返回七变量标签而不满足 OT-only；`feature_type=MS` 返回七变量历史输入和 `[B,H,1]` OT 标签，是唯一正确的数据语义；repair 前 adapter 拒绝 prediction `[B,H,1]` + target `[B,H,1]`；U1 manifest 缺少公共 target/aux schema block；第十二轮没有训练、forward 或 artifact；TEB development adequacy gate 仍为 `not yet passed`。

### 33.2 Production loss 的唯一 shape adapter 合同

`task_mode=target_exogenous` 的 prediction 必须为 `[B,H,1]`。合法组合只允许：

```text
prediction [B,H,1] + target [B,H]
  -> prediction.squeeze(-1) / target，criterion 前均为 [B,H]

prediction [B,H,1] + target [B,H,1]
  -> prediction / target 原样保留，criterion 前均为 [B,H,1]
```

统一 adapter 必须由 `train_one_epoch`、validation `evaluate` 和 final test `evaluate` 共同调用；criterion 前 shape 必须逐元组完全相同。只允许 `squeeze(-1)`，不得裸 `squeeze()`，不得改变 target rank、内容、顺序或数值。prediction channel >1、三维 target channel >1、batch/horizon mismatch、target rank 1/4、prediction rank 非 3 以及任何需要 broadcasting 的组合均须在 criterion/metric 前报错并打印实际 shape。Generic MS 指标只聚合 OT；UrbanEV 原二维 `[B,H]` target 行为、inverse-transform 与 scaler 语义保持不变；`parallel_multivariate` 行为不变；`CustomDataLoader` 不修改。

### 33.3 `target_exogenous_schema_v1` provenance 合同

所有 repair 后新生成且满足 `enhanced variant AND task_mode=target_exogenous` 的 resolved scientific config 与 checkpoint metadata，条件增加：

```text
target_exogenous_schema_contract_version = target_exogenous_schema_v1
```

completed manifest 在 seal/checksum 前必须写入并验证：

```json
"target_exogenous_schema": {
  "contract_version": "target_exogenous_schema_v1",
  "feature_type": "<runtime resolved value>",
  "feature_names": ["...runtime order..."],
  "target_feature_name": "<runtime target>",
  "target_idx": 0,
  "target_indices": [0],
  "aux_idx": [1, 2],
  "aux_feature_names": ["...runtime order..."],
  "schema_fingerprint": "<runtime fingerprint>"
}
```

实际值必须来自已验证的 runtime schema，而非照抄 CLI。Generic ETT 使用 `CustomDataLoader`/runtime preprocessing metadata；UrbanEV 使用 M1 `FoldBundle`/`FeatureSchema`。CLI/config 只做 expected-value 交叉核验，不一致须在训练或 staging 前拒绝。当前单目标下 `target_indices=[target_idx]`；`aux_idx` 保留原顺序、无重复、不含 target，names 与索引逐项对应；索引序列写 JSON list。

ETTm1 U1/U2 公共块必须逐元素相同：`feature_type=MS`、feature order `[HUFL,HULL,MUFL,MULL,LUFL,LULL,OT]`、target `OT`/`6`/`[6]`、aux `[0,1,2,3,4,5]` 与对应六个名称。U1 必须拥有公共 block；U2 同时拥有公共 block 与 T2 candidate contract，candidate contract 不得替代公共数据 schema。

### 33.4 Legacy、summarizer、checkpoint 与 artifact 边界

artifact 继续使用 schema-v2、既有路径、13-file checksum、staging/atomic publish 和 candidate contract。summarizer 对 config 声明 v1 的 artifact 严格核验 manifest block、version、path/task/target、feature type/name/order、target indices、aux index/name order、fingerprint、candidate contract（如适用）、checksum 与 completed status；任何 mismatch 拒绝整个 run。历史 config 没有 version 时按现有 legacy schema-v2 规则读取，不回写、不补齐、不重算 hash，并显式标记 schema contract 为 `legacy`。

以下冲突必须拒绝：config 有 version 但 manifest 无 block；manifest 有 v1 block但 config 无 version；version 不一致；target/aux/name/order/fingerprint 篡改；`parallel_multivariate` 错带 v1 contract。U1/U2 公共 block 必须一致，target/parallel 不得混组，duplicate scientific identity + seed 继续拒绝。resume 不得跨 legacy/v1 contract，mismatch 必须在写参数前拒绝；parallel historical scientific/comparison identity 不变。

### 33.5 实现与测试预登记、停止点

实现范围只允许 `main.py`、`summarize_results.py`、`tests/test_runner.py`、`tests/test_summarize_results.py`、`tests/test_tsAMD_enhanced.py` 与本 milestone；不得新增测试文件。三份测试均为 `permanent_regression_test`，将长期保护正式 runner、双合法 target shape、无 broadcasting、artifact provenance、U1/frozen AMD OT parity、U2/T2 单目标合同及 legacy/parallel 兼容。一次性 probe 只位于 `/tmp/m4_target_exogenous_repair_<timestamp>/` 并在结束前删除。

本 repair 不改变 U1、T2、T2G、T3、PMCR、Global TEB 或 frozen AMD 数学结构，不改变 ETTm1 数据/split/scaler/window/label，不改变 UrbanEV M1 数据合同。本轮不训练 U1/U2，不执行 validation epoch 或 final test，不创建真实 artifact，不实现 warm-start、新 TEB 或 P2，不进入 M5/M7，不实现空间模块。通过测试和 smoke 只构成 runner/provenance 工程证据，不形成 TEB performance 结论；M4 状态继续 `In Progress`，TEB adequacy gate 仍未通过。
## 34. 第十三轮 Stage B：runner/provenance repair 实现与验证

### 34.1 Stage A 文档 closure

第十二轮 capability audit 与第十三轮 repair 合同已由唯一文档 commit `2172b83b6631f74b14913aadc2ba6b517336d315` 提交并推送；parent=`6638d36336b9f46ed4d636c2b9d88bac80640b34`，title=`docs(m4): authorize target-exogenous runner repair`。推送后 local/tracking/live remote 一致、ahead/behind=`0/0`、worktree/index clean。阶段 A canonical SHA-256=`55defcfba6c52cc94201bc44d7a26714f32790a74bfda56cdb191d50bf82a2a6`；本 milestone 的 docs-closure SHA-256=`bcd47a91cef6f09dfec922b385f4557d54f9be84c291b5b545341bc4582e7212`。

### 34.2 Production loss shape repair

`main.py::_prediction_for_loss` 是唯一 adapter，`_assert_prediction_batch` 直接委托它；`train_one_epoch`、validation `evaluate` 与 final-test `evaluate` 均走同一调用链。实现后的严格行为为：

- prediction 必须为 `[B,H,1]`；
- target 为 `[B,H]` 时，只对 prediction 执行 `squeeze(-1)`，criterion 前双方严格为 `[B,H]`；
- target 为 `[B,H,1]` 时，prediction/target 均保持三维，criterion 前严格为 `[B,H,1]`；
- prediction/target rank、channel、batch 或 horizon 不符均在 criterion/metric 前拒绝并报告真实 shape，不允许 broadcasting；
- `_accumulate_errors` 只接收 adapter 后严格同形张量，按全部目标元素累计 SSE/SAE；
- generic MS 的 target-only inverse transform 仍使用 loader 的目标 scaler；UrbanEV 二维 target 行为不变；`parallel_multivariate` 分支未改变。

实际 call site：`train_one_epoch`（adapter + MSE + selector auxiliary）、`evaluate`（adapter + 全元素 metric），主循环分别用于 train、validation 与 validation-best checkpoint 载入后的 final test。本轮未修改 `CustomDataLoader`。

### 34.3 Schema-v1、resume 与 summarizer

新增版本常量 `target_exogenous_schema_v1`。仅当 `enhanced variant AND task_mode=target_exogenous` 时，runtime preprocessing 生成公共块：

`contract_version / feature_type / feature_names / target_feature_name / target_idx / target_indices / aux_idx / aux_feature_names / schema_fingerprint`。

Generic ETT 的事实来源是 `CustomDataLoader.metadata()` 的实际 columns、resolved target 与 target indices；UrbanEV 的事实来源是 M1 FoldBundle/FeatureSchema 及其 fingerprint。CLI 只作为 expected value；不一致在 artifact path/staging 创建前拒绝。该版本与全部 schema 字段进入 resolved scientific config，checkpoint 通过内嵌 resolved config 携带；completed manifest 在 checksum/seal 前再次执行字段集和内部一致性校验。U1 与 T2 synthetic artifacts 的公共块逐元素相同，U1 无 candidate contract，T2 同时保留自身 candidate contract；schema-v2 的 13-file checksum、目录与 atomic publish 未改变。

Resume 先比较 manifest 公共块，再比较 resolved scientific schema，之后才允许 checkpoint deserialization/参数写入；legacy↔v1、字段或顺序 mismatch 均拒绝。Summarizer 对 v1 严格校验 config/manifest/path/task/target/schema/candidate/checksum/completed status；无版本且无 block 的历史 schema-v2 继续读取并显式标记为 `legacy`；有版本无 block、无版本有 block、version/target/indices/aux order/names/feature order/fingerprint 篡改及 parallel 错带 v1 均拒绝。Parallel 既有 scientific/comparison hash fixture 保持不变，真实 parallel synthetic artifact 也明确不含 version/block。

### 34.4 修改范围与永久回归

实现修改精确为 `main.py`、`summarize_results.py`、`tests/test_runner.py`、`tests/test_summarize_results.py`、`tests/test_tsAMD_enhanced.py` 和本 milestone；没有新增测试文件。三份既有测试继续分类为 `permanent_regression_test`，分别保护双合法 target shape/无广播/train-evaluate 共用 adapter/runtime schema 与原子 artifact、v1/legacy/tamper/parallel/duplicate summarizer 合同、U1/frozen AMD OT-slice parity 与 enhancement-empty state。

最终验证结果：

| 组 | passed/total | failed | skipped | wall-clock |
|---|---:|---:|---:|---:|
| runner + summarizer + AMDEnhanced | 80/80 | 0 | 0 | 5.676 s |
| Global/T2/T2G/T3 TEB 保护 | 71/71 | 0 | 0 | 0.642 s |
| PMCR + M1 保护 | 36/36 | 0 | 0 | 3.205 s |
| 完整 discovery | 220/220 | 0 | 0 | 9.586 s |

repair 前既有 212 项全部保留，新增 8 项；既有 `1e-6` parity 门槛未放宽。CUDA 可用且 T2/T2G/T3、PMCR 与 M0 equivalence 的 CUDA 路径均实际执行。

### 34.5 真实 ETTm1 MS 与 UrbanEV smoke

真实 ETTm1（MS、OT、target_idx=6、aux_idx=`[0,1,2,3,4,5]`、seq=512、pred=96）：

| 项目 | 实际值 |
|---|---|
| train sample x / y | `[512,7]` / `[96,1]` |
| train/validation batch x / y | `[4,512,7]` / `[4,96,1]` |
| U1 prediction / adapter / target | `[4,96,1]` / `[4,96,1]` / `[4,96,1]` |
| T2 prediction / adapter / target | `[4,96,1]` / `[4,96,1]` / `[4,96,1]` |
| state_source | U1/T2 均为 `[4,1056]` |
| U1 vs frozen AMD OT slice | prediction max abs=`0`、MoE max abs=`0`、两者均 `torch.equal=True` |
| U1 enhancement keys | `0` |
| target inverse-transform shape | `[2,96,1]` |

T2 使用真实 train batch 和 production `MSE + selector auxiliary` 仅 backward 一次；loss=`4.939964771270752` 且有限，未创建 optimizer、未调用 `optimizer.step()`。梯度 L2：patch-query=`2.8418864896048054e-07`（1120/1120 非零）、exogenous projector/norm=`6.4224020117987575e-06`（16480/16480 非零）、gamma=`3.598617040552199e-04`（1/1 非零）、AMD backbone=`78.06347028950478`（10243941/10244742 非零），全部有限。T2 global-query 专属参数有 raw grad tensor 但严格全零（L2=`0`、0/16480 非零），继续符合第八轮的 forecast-loss-disconnected 合同。

真实 UrbanEV M1 production pipeline（F4、fold=1、history=12、label horizon=3、model pred_len=1）只取一个 train batch：x=`[4,12,11]`、target=`[4,1]`、prediction=`[4,1,1]`、adapter=`[4,1]`、state_source=`[4,28]`，target_idx=`0`，aux_idx=`[1,2,3,4,5,6,7,8,9,10]`；criterion 前严格同形、prediction/loss 有限、无 broadcasting。FoldBundle preprocessing fingerprint=`8a90c280e2aa0a78e25ef014478ca914768562d24d3364966ecf22dbcb085a19`，FeatureSchema fingerprint=`8e43cc3835b913f43357d98573c57c902e3c42d38024df32b6ea93735c00a0f8`；未遍历 UrbanEV test。

### 34.6 Synthetic artifact、source 与停止点

现有 tiny fixture 在 `TemporaryDirectory` 内生成并回收 U1/T2 schema-v1 artifacts：两者 completed manifest 公共 block 完全相同，T2 candidate contract 保留，13/13 checksum 与 system `sha256sum -c` 通过，summarizer 接受合法 v1；legacy fixture 可读且标记 `legacy`，各类 tamper 被拒绝。仓库内和正式 artifact root 均未产生真实 ETTm1/UrbanEV artifact。

实现后 source fingerprint：

`sha256_length_prefixed_relative_path_and_content_v1`，20 个文件，`bffb7f1975f4f4f9448e44576bc626a0e82c75e54902fda4800847c89611065e`。

一次性目录 `/tmp/m4_target_exogenous_repair_20260831T1938silDum/` 已确认为空并删除。Frozen AMD、PMCR、Global TEB、T2、T2G、T3、M0-M3 与 baseline 均未修改；`utils/dataloader.py`、任何模型模块和数据均未修改。

本轮没有 U1/U2 production training、development validation epoch 或 final-test evaluation，没有真实 development artifact，没有新增 TEB、warm-start 或 P2，没有进入 M5/M7，也没有实现空间模块。工程 repair 通过不构成 TEB performance evidence；TEB adequacy gate 仍为 `not yet passed`，M4 继续 `In Progress`。本节将随唯一 implementation commit `fix(m4): support generic target-exogenous runner` 提交，最终 commit SHA 由 Codex closure 回执报告。

## 35. 第十四轮：ETTm1 target_exogenous U1/U2 公平开发对照与 TEB adequacy gate

### 35.1 起始状态、回归与测试生命周期

本轮从 clean、已推送的 commit be2185c3382ec42c7287e4bcc9b2cad5c07fdbad（parent=2172b83b6631f74b14913aadc2ba6b517336d315，title=fix(m4): support generic target-exogenous runner）启动；branch=AMD-paper-repro-custom-modules-v1，local/tracking/live remote 一致，ahead/behind=0/0。起始 source fingerprint 为 sha256_length_prefixed_relative_path_and_content_v1、20 个文件、bffb7f1975f4f4f9448e44576bc626a0e82c75e54902fda4800847c89611065e。第十三轮 generic target_exogenous production runner、loss-shape 与 artifact provenance repair 已完成 Git closure。

使用 /public/home/yueweiting/miniconda/envs/amd/bin/python -B -m unittest discover -s tests -p 'test_*.py' -v 重新运行完整回归：220/220 passed、failed=0、skipped=0，Python 测试计时 10.829 s，shell wall-clock 14 s。CUDA 可用，T2/T2G/T3、PMCR 与 M0 equivalence 的 CUDA 路径实际执行。首次尝试的外部 /usr/bin/time 包装器因该可执行文件不存在而在 Python 启动前返回 127；随后改用 Bash SECONDS，这不是测试失败。

本轮未新增、修改或删除任何测试文件；仓库内全部 tests/test_*.py 均为永久回归资产。一次性公平性、梯度和归因 probe 只在 /tmp/m4_ettm1_target_exogenous_u1_u2_20260901T0955Z_iHAlAa/ 范围内执行，未写入仓库。

### 35.2 Artifact root、数据与 schema 公平性

artifact root 为 artifacts/m4-development/ettm1-stage-e-target-exogenous-t2-v1。启动前该 root 不存在，因此没有相同 scientific identity 的 completed run、hidden staging、running/failed staging 或 checksum-invalid final；未发生静默择一、覆盖或清理。

U1=AMD-Concat（el-amd-pmcr-teb-v1、ablation_id=U1、PMCR off、TEB off）；U2=AMD-Concat+T2（el-amd-m4-t2-patch-teb-v1、ablation_id=M4_T2、PMCR off、T2 on）。两者共同 runtime schema 为：

| 字段 | 事实值 |
|---|---|
| dataset / feature type / task | ETTm1 / MS / target_exogenous |
| input / target | 7 variables / OT only |
| feature order | [HUFL,HULL,MUFL,MULL,LUFL,LULL,OT] |
| target / aux index | 6 / [0,1,2,3,4,5] |
| schema contract / fingerprint | target_exogenous_schema_v1 / f6dd94841b5d9d0b7515b19e0ff1876bf6476068054eacdc02ac6fcab3f084dc |
| data SHA-256 / rows | 6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e / raw 69680, used 57600 |
| split endpoints | train 34560, validation 46080, test 57600 |
| context starts | train 0, validation 34048, test 45568 |
| target scaler mean / scale | 17.12495024489946 / 9.173035273908697 |
| x / y batch shape | [128,512,7] / [128,H,1] |

Window counts（train/validation/test）依 horizon 分别为：h96=33953/11425/11425，h192=33857/11329/11329，h336=33713/11185/11185，h720=33329/10801/10801。四个 horizon 上固定样本、首个 shuffled train batch、首个 validation/test batch以及 DataLoader generator state 均逐元素相同；torch.equal=True、最大绝对误差=0。U1/U2 config 与 manifest 中公共 schema block 逐字段相同；U1 无 candidate contract，U2 另带 T2 candidate contract。

### 35.3 Frozen AMD parity、初始化与 RNG 边界

四个 horizon 上，U1 与 frozen AMD 的全通道输出取 OT slice 后 prediction 与 MoE loss 均 torch.equal=True、最大绝对误差=0；U1 state_source=[128,1056]，最后 32 维 context 为严格零，且无 pmcr.* / teb.* state keys 或 enhancement modules。

U1/U2 共同 AMD parameter 与 persistent buffer 共 60 keys；相同 runner seed/factory 下 key/shape/dtype/value 全部一致，最大绝对误差=0、mismatch=none。两者 train DataLoader generator state 和首 batch完全一致。模型构造后 CPU RNG state 不一致，因为 U2 额外初始化 T2 参数；CUDA RNG state 一致。本公平性门禁证明共同 AMD 初值与训练样本顺序一致，但不声称两个独立模型逐 step 共享同一全局 RNG 轨迹。

### 35.4 实际命令与 scientific config 差异

八个 completed artifact 的 command.txt 均保存实际 Python executable 与完整 shell-escaped argv。绝对路径如下（每个文件第二行即完整可重放命令）：

| Model | H | command.txt |
|---|---:|---|
| U1 | 96 | /public/home/yueweiting/大论文/AMD/artifacts/m4-development/ettm1-stage-e-target-exogenous-t2-v1/el-amd-pmcr-teb-v1/ETTm1/target_exogenous/OT/horizon_96/fold_official/seed_2024/20260901T095811.286299Z-6e33fa77/command.txt |
| U1 | 192 | /public/home/yueweiting/大论文/AMD/artifacts/m4-development/ettm1-stage-e-target-exogenous-t2-v1/el-amd-pmcr-teb-v1/ETTm1/target_exogenous/OT/horizon_192/fold_official/seed_2024/20260901T100147.203364Z-f03509fd/command.txt |
| U1 | 336 | /public/home/yueweiting/大论文/AMD/artifacts/m4-development/ettm1-stage-e-target-exogenous-t2-v1/el-amd-pmcr-teb-v1/ETTm1/target_exogenous/OT/horizon_336/fold_official/seed_2024/20260901T100520.627934Z-0c9f399c/command.txt |
| U1 | 720 | /public/home/yueweiting/大论文/AMD/artifacts/m4-development/ettm1-stage-e-target-exogenous-t2-v1/el-amd-pmcr-teb-v1/ETTm1/target_exogenous/OT/horizon_720/fold_official/seed_2024/20260901T100834.831317Z-5f71979f/command.txt |
| U2 | 96 | /public/home/yueweiting/大论文/AMD/artifacts/m4-development/ettm1-stage-e-target-exogenous-t2-v1/el-amd-m4-t2-patch-teb-v1/ETTm1/target_exogenous/OT/horizon_96/fold_official/seed_2024/20260901T101200.472821Z-265147d9/command.txt |
| U2 | 192 | /public/home/yueweiting/大论文/AMD/artifacts/m4-development/ettm1-stage-e-target-exogenous-t2-v1/el-amd-m4-t2-patch-teb-v1/ETTm1/target_exogenous/OT/horizon_192/fold_official/seed_2024/20260901T101543.845343Z-954e54fd/command.txt |
| U2 | 336 | /public/home/yueweiting/大论文/AMD/artifacts/m4-development/ettm1-stage-e-target-exogenous-t2-v1/el-amd-m4-t2-patch-teb-v1/ETTm1/target_exogenous/OT/horizon_336/fold_official/seed_2024/20260901T101921.289921Z-7bbb1947/command.txt |
| U2 | 720 | /public/home/yueweiting/大论文/AMD/artifacts/m4-development/ettm1-stage-e-target-exogenous-t2-v1/el-amd-m4-t2-patch-teb-v1/ETTm1/target_exogenous/OT/horizon_720/fold_official/seed_2024/20260901T102301.738668Z-5c2d85f0/command.txt |

所有命令固定 seed=2024、epochs=10、batch=128、lr=3e-5、weight decay=1e-7、seq=512、pred_len=H、AMD n_block=1/alpha=0/mix=3,2/patch=16/norm+layernorm/dropout=true,true,0.1。U1/U2 的 scientific config 差异仅为 implementation_variant、experiment.ablation_id/display_name、model.use_teb 及 T2 的 architecture/patch size=32/padding=right_zero_crop/position=fixed_sinusoidal/target-selection contract；数据、训练、AMD 主干和所有共同字段完全一致。

### 35.5 Completed artifact 清单

全部 8 个 run 均为 schema-v2 completed、history=10 epochs、13/13 checksum Python exact-set+digest 及 system sha256sum -c 通过，source commit=be2185c3382ec42c7287e4bcc9b2cad5c07fdbad、dirty=false、source fingerprint=bffb7f1975f4f4f9448e44576bc626a0e82c75e54902fda4800847c89611065e、data SHA=6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e，best/last checkpoints 存在且所有指标 finite。最终 root 恰有 8 个 manifest，无 hidden staging。

| Model | H | run_id | config hash | best epoch | best.pt SHA-256 | last.pt SHA-256 | wall s | bytes |
|---|---:|---|---|---:|---|---|---:|---:|
| U1 | 96 | 20260901T095811.286299Z-6e33fa77 | fa2c4da41f34eca232907e4d6462305cb8ef3ef15fc8996f7c67baa0411ddb2d | 7 | 66458be335ac7948889156bf6a7a91af7221f3b75838a1522fd701e8e78b42d0 | 667ae4bba8becf458adc13cd218395780b63a9f3d2ac15e44fe3bea3a4d679a1 | 176.181 | 205064834 |
| U1 | 192 | 20260901T100147.203364Z-f03509fd | 1585152dbf1ff74d935f7404f8b2699881b85c7224252371e939db585f2611d0 | 3 | f8b0308578b10f09ade232cf2c6ac2e7826b1e28ceb994c2a321d44c4563be6a | 48df0c13636db15006806d24a5e7f6181c844166a5db2c883c2baedfcb9e710d | 173.946 | 236537485 |
| U1 | 336 | 20260901T100520.627934Z-0c9f399c | 45ea9453083d6fb38381d03ba3a8455191e28e6221a2c9f86d0dee72ba8e8ff5 | 3 | 89da2c854dce4e124c09575835d52e313bf3a6aeab2b556a060c7174709cb530 | dba0f43a990ab2242ee212fd75383e6e32c37c3e2bf8092159c442158459dd98 | 162.284 | 283746640 |
| U1 | 720 | 20260901T100834.831317Z-5f71979f | 93ddc59a0879b435a4cf4742e76c75460c254283ca00b2138164c4fe735d51cd | 7 | a13126522bb2cf8f5871c46272222242b8ebb6077145658558199c9473ba109b | 403dabcf76d471efbc520c7da3cc915e1f3d0980b3ceb8bc9f7f43523068ea87 | 169.717 | 409637178 |
| U2 | 96 | 20260901T101200.472821Z-265147d9 | 669772e9f8ac399465e510abe21ad269fdd8e030726f626cf58334dfd144853d | 3 | 058bfa02446e299a07849d4a8195eec8cb848d7f1e964d342e814f1e6fce5fd7 | ed0e837060cf4340f7ab74fca3279c5db207ae5b68bb8cbbf87939dc7ac5aa83 | 185.842 | 205887746 |
| U2 | 192 | 20260901T101543.845343Z-954e54fd | f590a5a39a561378d2c7dfa22f77ab24a8530a027b234e6ea60cde95ad3a3b55 | 3 | ea81861b91f6a1ca160dd8a4c6af65b5088692db64dbd1c68c6d1dd66820d93f | 3120419bcb7104f5c7896fe345f1b1bd747aa453147e31ee5f6a6993ae9183bc | 185.017 | 237360421 |
| U2 | 336 | 20260901T101921.289921Z-7bbb1947 | 07a4f9d53dbb9d82bfc776be3c13fcc3583a5d5c293bd1779b1a889f515984cf | 2 | 8fb3b1214b3edd8e5dcb461a1b1d907a95aa68d8873ee892bc8394303e8d62ef | 16fbdfb8a46bf5523dc3a329d5305a028fb0019dbff17b7a240feb7514e4a9d7 | 173.273 | 284569509 |
| U2 | 720 | 20260901T102301.738668Z-5c2d85f0 | 715aa1ebe5557946585bec62c98774d4df4dc99a1891ab5785e6d456c7c753ab | 4 | d54fa638d796e425e2d692855b2c536398a6e95c51c0bc809273d6c54ed64407 | c100e3aefd4e882a67c59a0839c07235c3c409bd459d42327ea57232622bcbe0 | 188.906 | 410460044 |

### 35.6 U2 相对 U1 的 development 指标

负相对变化表示 U2 误差下降。ETTm1 为 development-only，test 已用于候选开发，以下不是论文正式结果。

Validation MSE：

| H | U1 | U2 | relative change |
|---:|---:|---:|---:|
| 96 | 0.05116432 | 0.05135259 | +0.36797% |
| 192 | 0.07350609 | 0.07359417 | +0.11983% |
| 336 | 0.09000857 | 0.09012128 | +0.12523% |
| 720 | 0.10404531 | 0.10480060 | +0.72592% |
| Macro mean | 0.07968107 | 0.07996716 | +0.35904% |

Validation MAE：

| H | U1 | U2 | relative change |
|---:|---:|---:|---:|
| 96 | 0.16807296 | 0.16888409 | +0.48261% |
| 192 | 0.20784405 | 0.20780507 | -0.01875% |
| 336 | 0.23508584 | 0.23554143 | +0.19380% |
| 720 | 0.25514636 | 0.25599622 | +0.33309% |
| Macro mean | 0.21653730 | 0.21705670 | +0.23987% |

Test MSE（development primary）：

| H | U1 | U2 | relative change |
|---:|---:|---:|---:|
| 96 | 0.02759721 | 0.02819966 | +2.18301% |
| 192 | 0.04116124 | 0.04116471 | +0.00844% |
| 336 | 0.05261028 | 0.05341865 | +1.53652% |
| 720 | 0.07035230 | 0.06954740 | -1.14409% |
| Macro mean | 0.04793026 | 0.04808261 | +0.31786% |

Test MAE（development secondary）：

| H | U1 | U2 | relative change |
|---:|---:|---:|---:|
| 96 | 0.12624871 | 0.12831446 | +1.63625% |
| 192 | 0.15472727 | 0.15478672 | +0.03842% |
| 336 | 0.17431307 | 0.17614072 | +1.04848% |
| 720 | 0.20091614 | 0.19986873 | -0.52132% |
| Macro mean | 0.16405130 | 0.16477766 | +0.44276% |

两种相对汇总严格区分：

| Metric | mean_horizon_relative_change | relative_change_of_macro_means |
|---|---:|---:|
| Validation MSE | +0.33474% | +0.35904% |
| Validation MAE | +0.24768% | +0.23987% |
| Test MSE | +0.64597% | +0.31786% |
| Test MAE | +0.55046% | +0.44276% |

### 35.7 曲线、gamma 与成本

| Model | H | best epoch | epoch-1 val | best val | last val | last vs best | last-3 val |
|---|---:|---:|---:|---:|---:|---:|---|
| U1 | 96 | 7 | 0.05397703 | 0.05116432 | 0.05213227 | +1.89184% | 0.05191388,0.05153775,0.05213227 |
| U1 | 192 | 3 | 0.07464176 | 0.07350609 | 0.07673102 | +4.38729% | 0.07578068,0.07620048,0.07673102 |
| U1 | 336 | 3 | 0.09162994 | 0.09000857 | 0.09238960 | +2.64535% | 0.09267142,0.09137627,0.09238960 |
| U1 | 720 | 7 | 0.10782488 | 0.10404531 | 0.10482275 | +0.74721% | 0.10541365,0.10466343,0.10482275 |
| U2 | 96 | 3 | 0.05401062 | 0.05135259 | 0.05406593 | +5.28374% | 0.05295763,0.05332140,0.05406593 |
| U2 | 192 | 3 | 0.07474114 | 0.07359417 | 0.08372224 | +13.76206% | 0.08052580,0.08269039,0.08372224 |
| U2 | 336 | 2 | 0.09172329 | 0.09012128 | 0.10404136 | +15.44595% | 0.10043911,0.09956302,0.10404136 |
| U2 | 720 | 4 | 0.10785528 | 0.10480060 | 0.11526754 | +9.98748% | 0.11215577,0.11395104,0.11526754 |

所有 run 的最后 epoch 均不是 best。Train objective 为 prediction MSE + selector auxiliary，而 validation 为纯 prediction MSE，二者 objective 不同，不计算或解释 train-validation gap。

U2 gamma_teb（init/best/last）依次为：h96=0.001/0.03187034/0.06876934，h192=0.001/0.03561231/0.08853173，h336=0.001/0.01873618/0.09187824，h720=0.001/0.04767828/0.10154618。Patch query、exogenous projector、MHA 与 patch output 的 best/last 参数均相对同 seed 初始值发生有限非零移动；global-query projection/norm 虽因 Adam coupled weight decay 发生参数移动，但 production raw backward gradient 在四个 horizon 均为严格零，不能称为受任务监督。

参数量 U1/U2（增量恒为 39,361）：h96=10,244,742/10,284,103，h192=11,818,374/11,857,735，h336=14,178,822/14,218,183，h720=20,473,350/20,512,711。四 horizon 总训练 wall-clock U1=682.128 s、U2=733.036 s，U2/U1=1.07463（约 +7.46%）。artifact 总空间 U1=1,134,986,137 bytes，U2=1,138,277,720 bytes。

### 35.8 T2 residual、同-checkpoint bypass 与 aux-K/V permutation

在每个 U2 best checkpoint 的完整 validation/test 上，diagnostic wrapper 与 formal forward 的 prediction、MoE、A_patch、delta、exo_context/state 均最大误差=0；模型 `legacy_unversioned_audit_digest` 前后相同。所有表示、ratio 和指标 finite，无 NaN/Inf。

每样本 r_teb=||gamma*delta||/(||hidden||+eps)：

| H | Split | mean | median | p10 | p90 | p99 | max |
|---:|---|---:|---:|---:|---:|---:|---:|
| 96 | val | 0.024467 | 0.023358 | 0.014765 | 0.035730 | 0.044748 | 0.053900 |
| 96 | test | 0.025189 | 0.023332 | 0.014050 | 0.039072 | 0.049099 | 0.054448 |
| 192 | val | 0.022405 | 0.022738 | 0.014699 | 0.029486 | 0.034174 | 0.038627 |
| 192 | test | 0.022078 | 0.022085 | 0.014198 | 0.029574 | 0.035473 | 0.044266 |
| 336 | val | 0.005307 | 0.005286 | 0.003993 | 0.006652 | 0.007716 | 0.008384 |
| 336 | test | 0.005228 | 0.005176 | 0.004053 | 0.006480 | 0.007433 | 0.009474 |
| 720 | val | 0.033940 | 0.032631 | 0.020814 | 0.048182 | 0.067132 | 0.082536 |
| 720 | test | 0.035880 | 0.034854 | 0.021507 | 0.051540 | 0.068529 | 0.078116 |

Patch-position mean ratio 亦 finite：h96 val/test 范围约 0.02447–0.03947/0.02789–0.03761；h192 0.02386–0.03112/0.02479–0.03008；h336 0.00628–0.00725/0.00622–0.00688；h720 0.03774–0.05072/0.04193–0.05073。h336 写回尤其小，但没有爆炸尾部。

同-checkpoint 将 TEB temporal residual 旁路后，误差相对 Normal 的变化如下；负值代表旁路更优：

| H | val MSE | val MAE | test MSE | test MAE | prediction max change val/test |
|---:|---:|---:|---:|---:|---:|
| 96 | -0.21048% | -0.01621% | -0.35058% | -0.14872% | 0.03237/0.02086 |
| 192 | -0.32913% | -0.02378% | -0.17130% | -0.08422% | 0.02221/0.01989 |
| 336 | -0.03111% | -0.00304% | -0.01510% | -0.01038% | 0.00290/0.00189 |
| 720 | -0.32027% | -0.12039% | -0.06447% | -0.01870% | 0.02391/0.01644 |

因此 8/8 split×horizon 上旁路均描述性地降低误差；这是同一已联合训练 checkpoint 内的反事实依赖，不是独立训练消融或因果证明。

Aux-K/V permutation 对每个 batch 使用相同样本循环错配、保持 target hidden/query 与样本内变量关系不变；每个 split 使用三个 cyclic shifts 取均值，B=1 尾 batch数为 0。相对 Normal 的聚合变化：

| H | val MSE | val MAE | test MSE | test MAE |
|---:|---:|---:|---:|---:|
| 96 | +0.05646% | +0.05017% | +0.05156% | +0.02038% |
| 192 | -0.01192% | +0.00014% | +0.03620% | +0.00225% |
| 336 | -0.00187% | -0.00074% | +0.00459% | +0.00191% |
| 720 | +0.00002% | -0.00027% | -0.00334% | -0.00844% |

所有 horizon 的 A_patch、delta 和 prediction 均发生非零逐元素变化，证明样本对齐的 auxiliary K/V 会影响表示和预测；但 aggregate metric 效应极小且方向混合，不能形成稳定有利外生依赖证据。这同样是同-checkpoint 扰动，不是因果结论。

### 35.9 Development signal、adequacy gate 与证据边界

本轮固定结论为 **negative-or-negligible development signal**：U2 相对严格公平 U1 的 validation MSE/MAE macro 分别恶化 +0.35904%/+0.23987%，development test MSE/MAE macro 分别恶化 +0.31786%/+0.44276%；四个 test horizon 中只有 h720 改善，且 validation MSE 四个 horizon 均未改善。T2 参数确实学习、residual finite 且能改变 prediction，但这不足以转化为稳定指标收益；旁路结果在所有 split×horizon 上描述性更优，aux permutation 的 aggregate 影响又极小/混合。

因此 **TEB development adequacy gate remains failed**。本结论只回答单数据集 ETTm1、单 seed=2024、10 epochs、OT-only target_exogenous 下 T2 相对 AMD-Concat 的开发信号；ETTm1 test 已用于 development，不能进入 M6 正式主表或解释为未见测试泛化。它不能证明 UrbanEV/其他正式数据集表现、多 seed 稳定性、warm-start 可行性、其他 TEB 结构或 PMCR/P2 的效果，也不自动废弃既有可复现候选。

下一步只允许等待用户与 ChatGPT 裁决是否停止 TEB 结构迭代、是否另行授权有限 repair/warm-start，或何时转入 PMCR/P2；本轮未自动执行任何下一步。M4 保持 In Progress，没有新增 TEB architecture、没有训练 T2G/T3、没有实现 warm-start/P2、没有进入 M5/M7、没有运行任何正式评价数据集 test，也没有实现空间模块。
## 36. 第十五轮：第十四轮 closure 与 warm-start/adapter-style T2 rescue 合同审计

### 36.1 第十四轮结果 closure 与审计边界

第十四轮结果版本已由单一 commit `e5d1b4aecdf34705bd9f908fa13644c74fddebf5` 提交并推送，parent=`be2185c3382ec42c7287e4bcc9b2cad5c07fdbad`，title=`docs(m4): record target-exogenous T2 adequacy failure`。提交范围仅为本 milestone；push 后 local/tracking/live remote 一致，ahead/behind=`0/0`，worktree/index clean。closure 版本 milestone SHA-256 为 `f5240dcfb6bb44bbc37eb75161501972dfc8b8896bff3f49a41a97ae5218b060`。Canonical、M0-M3、20-file source fingerprint、baseline 和冻结源码均未变化。

本轮后续只做只读合同审计：没有重新读取 `metrics.json` 数值，没有执行 validation/test evaluation，没有训练 epoch，没有创建或更新 artifact/checkpoint，也没有执行 `optimizer.step()`。动态探针只使用 ETTm1 target_exogenous h96 的一个真实 train batch，并且只构造 train DataLoader。

### 36.2 四个 U1 source artifact

四个 source 均为唯一 completed schema-v2 U1（`el-amd-pmcr-teb-v1`、`ablation_id=U1`、PMCR/TEB off），13/13 Python exact-set/digest 与系统 `sha256sum -c` 全部通过；没有 duplicate completed identity、hidden staging、running/failed staging。共同身份为 `ETTm1`、`target_exogenous`、`feature_type=MS`、target=`OT`、target_idx=`6`、target_indices=`[6]`、aux_idx=`[0,1,2,3,4,5]`、`target_exogenous_schema_v1`、data SHA-256=`6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e`、source commit=`be2185c3382ec42c7287e4bcc9b2cad5c07fdbad`、source fingerprint=`bffb7f1975f4f4f9448e44576bc626a0e82c75e54902fda4800847c89611065e`、dirty=`false`。每个 best checkpoint 的 metadata 与 config/manifest 一致，state dict 无 `pmcr.*` 或 `teb.*`。

| H | run_id | config hash | best epoch | best.pt SHA-256 | artifact path |
|---:|---|---|---:|---|---|
| 96 | `20260901T095811.286299Z-6e33fa77` | `fa2c4da41f34eca232907e4d6462305cb8ef3ef15fc8996f7c67baa0411ddb2d` | 7 | `66458be335ac7948889156bf6a7a91af7221f3b75838a1522fd701e8e78b42d0` | `artifacts/m4-development/ettm1-stage-e-target-exogenous-t2-v1/el-amd-pmcr-teb-v1/ETTm1/target_exogenous/OT/horizon_96/fold_official/seed_2024/20260901T095811.286299Z-6e33fa77` |
| 192 | `20260901T100147.203364Z-f03509fd` | `1585152dbf1ff74d935f7404f8b2699881b85c7224252371e939db585f2611d0` | 3 | `f8b0308578b10f09ade232cf2c6ac2e7826b1e28ceb994c2a321d44c4563be6a` | `artifacts/m4-development/ettm1-stage-e-target-exogenous-t2-v1/el-amd-pmcr-teb-v1/ETTm1/target_exogenous/OT/horizon_192/fold_official/seed_2024/20260901T100147.203364Z-f03509fd` |
| 336 | `20260901T100520.627934Z-0c9f399c` | `45ea9453083d6fb38381d03ba3a8455191e28e6221a2c9f86d0dee72ba8e8ff5` | 3 | `89da2c854dce4e124c09575835d52e313bf3a6aeab2b556a060c7174709cb530` | `artifacts/m4-development/ettm1-stage-e-target-exogenous-t2-v1/el-amd-pmcr-teb-v1/ETTm1/target_exogenous/OT/horizon_336/fold_official/seed_2024/20260901T100520.627934Z-0c9f399c` |
| 720 | `20260901T100834.831317Z-5f71979f` | `93ddc59a0879b435a4cf4742e76c75460c254283ca00b2138164c4fe735d51cd` | 7 | `a13126522bb2cf8f5871c46272222242b8ebb6077145658558199c9473ba109b` | `artifacts/m4-development/ettm1-stage-e-target-exogenous-t2-v1/el-amd-pmcr-teb-v1/ETTm1/target_exogenous/OT/horizon_720/fold_official/seed_2024/20260901T100834.831317Z-5f71979f` |

### 36.3 当前 checkpoint、resume 与 runner 事实

- CLI 只有 `--resume`，没有 source checkpoint、`source_kind` 或 warm-start 参数。`--resume` 只接受当前 running/failed run 的 `last.pt`，拒绝 completed artifact；它严格核对同 variant/config/data/schema，并恢复 model、optimizer、completed epoch、best、history、train generator 和 Python/NumPy/CPU/CUDA RNG。因此 U1 best -> 新 T2 必须是新的 warm-start initialization，绝不能伪装成 resume。
- `AMDEnhanced.load_state_dict(strict=True)` 在写入前检查 key 和 shape，但没有显式 dtype 检查。T2/T2G/T3 明确拒绝 `strict=False`。`load_enhancement_state_dict(source_kind=baseline|pmcr_only|teb_only)` 只接收 raw state mapping，结构预检后补全新模块再 strict load；它不读取 checkpoint container、implementation variant、artifact schema、target schema或 source lineage，也不证明来源就是 immutable baseline tag。
- T2/T2G/T3 在 public importer 入口直接拒绝：`T2/T2G/T3 permit only from-scratch initialization or same-structure strict restore`。h96 实测拒绝前后 full-state `legacy_unversioned_audit_digest` 完全相同。
- 现有 `_main_impl` 在 seed 和 loader 构造后先解析 artifact path、取得 lock、绑定 staging 并写 provenance/config/manifest，随后才构造 model/optimizer。它没有在 staging 前完成 source artifact、checkpoint metadata、key/shape/dtype 和 compatibility proof 的入口。
- 未来安全路径必须在任何 artifact staging/写入、optimizer 构造和参数写入之前，验证 completed/checksum/config/manifest/checkpoint metadata、同 horizon/seq_len、task/target/aux/schema、完整 source key 集、shape 与 dtype；unexpected 必须为空，唯一 missing 必须精确等于目标 T2 的全部 `teb.*`。预检失败应丢弃新构造目标模型或证明全 state 逐元素不变。

**Warm-start source loading: Not supported without code changes.**

### 36.4 四 horizon source -> fresh T2 映射

按每个 run 自身 config/factory/seed 构造并在内存中验证：U1 均为 60 keys；T2 均为 79 keys；公共 AMD 为 60 keys；PMCR/source TEB keys 均为 0；missing 精确为 19 个 T2 keys；unexpected、shape mismatch、dtype mismatch 均为空。安全映射后公共 AMD 与 source 逐元素相同，fresh T2 subset digest 不变。

共同 missing key 完整列表：`teb.gamma_teb`；`teb.patch_query_projection.{weight,bias}`；`teb.patch_query_norm.{weight,bias}`；`teb.global_query_projection.{weight,bias}`；`teb.global_query_norm.{weight,bias}`；`teb.exogenous_projection.{weight,bias}`；`teb.exogenous_norm.{weight,bias}`；`teb.cross_attention.{in_proj_weight,in_proj_bias,out_proj.weight,out_proj.bias}`；`teb.patch_output_projection.{weight,bias}`。

| H | source U1 digest | mapped T2 AMD-subset digest | fresh T2 subset digest（before=after） |
|---:|---|---|---|
| 96 | `95ebbbb744b9b112207152de728ab35ba21eb98ab87030aa4aeccf072c54dd23` | `95ebbbb744b9b112207152de728ab35ba21eb98ab87030aa4aeccf072c54dd23` | `5d079a22f13b7d602aa0eff037452bd8041dea6be84de3037bbfbedc79927ad5` |
| 192 | `a1a72d1c1768dbc5c2f22faa3b89a232976effbd226cb1222415ac7862f0e821` | `a1a72d1c1768dbc5c2f22faa3b89a232976effbd226cb1222415ac7862f0e821` | `c096352f038c9fb652db6c2a19b31d0b3ab6225c337c7510eb9a1f84e078132d` |
| 336 | `f5f13396ca4090a9540b5de99cbafbeffa7c7d389d135abbb31cdf6d4d5abb40` | `f5f13396ca4090a9540b5de99cbafbeffa7c7d389d135abbb31cdf6d4d5abb40` | `0213c44a065b861ab26b1314f7991808d557dbcda9cba945742755dfe43c3c23` |
| 720 | `24bde4a53df88e3428dcddb4a79bd6858f28437f2699e0a339ab92760df49e9e` | `24bde4a53df88e3428dcddb4a79bd6858f28437f2699e0a339ab92760df49e9e` | `074be8c3a70d18630cf910abddfb4f43c5f0647b0d13e96a9ab3439a9c377edd` |

本表 digest 现正式标记为 `legacy_unversioned_audit_digest`。保留的本地 Codex session 记录恢复了当时一次性脚本的精确实现：按 Python 字符串排序 key；key/dtype/raw length、rank 与每个 shape dimension 均为 8-byte little-endian，dimension 使用 signed encoding；无 header 或 key count。它覆盖 parameter 与 persistent buffer，不使用 `torch.save` pickle SHA。任意不同 horizon 交叉映射均先出现 horizon metadata mismatch，并在 AMS expert 输出层形成 16 个 shape mismatch（首个为 `moe.experts.0.net.3.bias`）；未写入目标，failure 前后 digest 相同。该历史逐 tensor equality、key/shape/dtype 与 mapping 证据继续有效，但 legacy 字符串不得与后续其他 framing 的 digest 直接比较。

### 36.5 Freeze、buffer 与 module mode

AMD 侧 RevIN/LayerNorm 无 running buffer；MDM 无 dropout；每个 DDI 的 `norm1` 是 BatchNorm1d，当前 `alpha=0` 因而没有启用 `norm2` 路径，`dropout_t` 在 train mode 生效；AMS 每个 expert 含 Dropout，TopKGating 在 train mode 调用 `torch.randn_like`；T2 的 MultiheadAttention dropout 在 T2 train mode 生效。当前 `train_one_epoch()` 每个 epoch 无条件调用 `model.train()`，会递归把此前手工设为 eval 的 AMD backbone 重新切回 train。

h96 naive probe 使用真实 train batch `x=[128,512,7]`、`y=[128,96,1]`。仅把非 T2 parameters 设为 `requires_grad=False` 后调用 `model.train()`，forward 不改变任何 frozen parameter，却改变 `fc_blocks.0.norm1.running_mean`、`running_var` 和 `num_batches_tracked`；AMD DDI dropout、AMS expert dropout、selector Gaussian noise与 T2 attention dropout均激活。CUDA RNG 被消费；相同输入连续两次 prediction 不相等，max abs=`0.3012093902`，auxiliary max abs=`0.0151400194`。这证明 parameter freeze 不等于 buffer/mode/stochastic freeze。

在独立模型上执行 `model.eval(); model.teb.train()`，AMD/RevIN/MDM/DDI/AMS/selector 均为 eval，T2/MHA 为 train：AMD BatchNorm buffer不变，selector noise与 AMD dropout关闭，T2 attention dropout保留；CPU RNG不变、CUDA RNG因T2 dropout改变。validation/test仍应使用全模型 eval。该 mixed mode 由 PyTorch公共 API可表达，但现有 runner 会在每个 epoch开头覆盖它，没有稳定重应用机制。

**Frozen parameter policy: Not supported without code changes.**

**Frozen buffer and module-mode policy: Not supported without code changes.**

### 36.6 h96 mixed-mode production backward

在 source U1 AMD 已安全映射、非 T2参数冻结、AMD eval/T2 train 的独立内存模型上，使用真实 production objective `MSE(prediction,y)+selector auxiliary` 做一次 backward；没有 optimizer/step。prediction/target 均为 `[128,96,1]`，prediction loss=`0.0581101477`、auxiliary=`0.0116437199`、total=`0.0697538704`。auxiliary 不依赖 T2且 `requires_grad=false`。完整 `legacy_unversioned_audit_digest` before/after 均为 `71347d364275d182b799af9475a66ea3b0441ec9d3fbb1652089a2c3cfb91b2e`；AMD parameter grad 全部 `None`，AMD buffer无变化。

| T2 group | 参数 tensors / elements | grad L2 | max abs | exact nonzero | 结论 |
|---|---:|---:|---:|---:|---|
| forecast-connected 合计 | 15 / 22,881 | `1.8969084e-4` | `1.8884671e-4` | 22,881 | finite、首 backward 非零 |
| `gamma_teb` | 1 / 1 | `1.8884671e-4` | `1.8884671e-4` | 1 | finite、非零 |
| patch query projection/norm | 4 / 1,120 | `1.1170399e-6` | `1.1228893e-7` | 1,120 | finite、非零 |
| exogenous projection/norm | 4 / 16,480 | `1.5935888e-5` | `6.4560629e-7` | 16,480 | finite、非零 |
| shared cross-attention | 4 / 4,224 | `6.1272740e-6` | `7.1979571e-7` | 4,224 | finite、通过 patch forecast 路径训练 |
| patch output projection | 2 / 1,056 | `5.1762217e-6` | `1.7747381e-6` | 1,056 | finite、非零 |
| global-query-only projection/norm | 4 / 16,480 | `0` | `0` | 0 | graph-connected zero tensor，不受任务监督 |

将 global-query-only 4 个 parameters 明确冻结后，其 grad 从严格零 tensor 变为 `None`；其余 forecast-connected 数值不变。`exo_context` 仍可因 shared exogenous projector/KV/MHA 的训练而改变，但 fixed fresh global query projection/norm 本身不被 forecast loss优化，因此应描述为“state interface 的 shared部分可适配，global-query-only映射固定”，不能称整个 state context 已受任务监督。

推荐的 forecast-connected key 集（未锁定）：`teb.gamma_teb`；`teb.patch_query_projection.*`；`teb.patch_query_norm.*`；`teb.exogenous_projection.*`；`teb.exogenous_norm.*`；`teb.cross_attention.*`；`teb.patch_output_projection.*`。建议排除 `teb.global_query_projection.*` 与 `teb.global_query_norm.*`。

### 36.7 Optimizer、gamma、epoch 0 与 objective

当前 runner 使用 PyTorch 2.0.1 `torch.optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-7)`，76 tensors/10,284,103 elements全部进入单一参数组；无 scheduler；`optimizer.zero_grad()` 在当前环境默认 `set_to_none=True`。它没有 freeze policy、trainable-only filter或独立 optimizer-state policy，resume 会恢复旧 optimizer state。T2共19 tensors/39,361 elements，其中 forecast-connected 15 tensors/22,881 elements，global-query-only 4 tensors/16,480 elements。

当前 Adam 的 weight decay 是 coupled L2：参数只要拥有严格零但非 `None` 的 grad tensor，就会加上 `weight_decay * parameter` 后移动。这足以解释第十四轮 global-query raw task gradient严格为零、参数却相对初始化移动的现象。若 gamma=0，除 gamma 外的 T2 参数首 backward均得到严格零 tensor；保留 `1e-7` 会在其获得任务梯度前先按权重衰减移动。推荐默认（未锁定）是全新 Adam、只接收 forecast-connected `requires_grad=True` 参数、adapter weight decay=`0`，并显式排除 global-query-only参数。

gamma 两种候选合同：

- `1e-3`：当前 class/config唯一支持值；h96 fresh T2 相对 source U1 的初始 prediction max abs=`7.8320503e-5`，不再 source-equivalent；首 backward全部15个 forecast-connected tensors有任务梯度。
- `0`：当前 config/class拒绝；审计中仅在内存将已构造 T2 scalar置零。初始 prediction与U1逐元素相同（prediction/MoE max abs=`0`、`torch.equal=true`）；首 backward只有 gamma 非零（`1.8846584e-4`），其余 forecast-connected和global-only均为严格零，后续 gamma离开0后才建立其他任务梯度。若未来选择此策略，可由显式 warm-start初始化 policy在构造后、optimizer前原子置零并记录，而不必改变T2数学 class，但不得把它伪装成原 `teb_gamma_init=1e-3`。

当前 runner 不评价或保存 epoch 0：`best_mse=inf`、`best_epoch=None`，从 epoch 1 validation开始选择，history只允许1..completed_epoch，summarizer也要求 `1 <= best_epoch <= train_epochs`。不纳入epoch 0实现最简单，但若所有adapter epoch更差，仍会强制发布某个更差训练态；纳入epoch 0可安全回退，但只有gamma=0时才严格等于source U1。推荐默认（未锁定）：gamma=0并把初始化candidate登记为明确的epoch-0 validation/checkpoint候选，不计作已训练epoch；最终test仍只在best确定后执行。

冻结 AMD 后，selector auxiliary只依赖固定 `u_mdm`/selector，不对T2产生梯度；加入total只造成随batch变化但对adapter梯度为常数的数值偏移。它不影响backward方向，也不影响基于纯validation MSE的best选择，但会使train objective日志不等于prediction MSE。推荐默认（未锁定）保留原 `prediction MSE + auxiliary` 以维持production语义，并继续分开记录两项；若选择prediction-only，必须作为新的training policy/scientific identity，不能静默改变。

### 36.8 Controls、预算与停止线

- Primary：rescue从每个horizon同一个U1 best AMD出发，只训练fresh T2，与固定、不再训练的source U1 best比较。这直接回答adapter能否改善固定AMD，但candidate拥有额外数据遍历/优化预算。
- Secondary：只有rescue先出现positive signal，才增加4个matched-budget U1 continuation：同source、reset optimizer、相同数据顺序/epoch/validation频率、继续训练整个AMD但不加T2。它用于判断收益是否仅是额外训练预算；当前runner同样缺少“completed best作为新run warm-start”的lineage路径，不能用resume替代，且延长已训练AMD可能引入特定过拟合偏差。
- No-op：冻结所有参数、重复遍历数据而不更新，prediction恒定，不提供额外科学信息，不建议生成wall-clock matching artifact。

R-min为4 horizon × 1 adapter、seed 2024、最多10 adapter epochs，共4个run；R-control仅在R-min positive后再增加4个U1 continuation，最大8个新run。基于第十四轮实际U2四run `733.036 s / 1,138,277,720 bytes`、U1四run `682.128 s / 1,134,986,137 bytes`，R-min保守估计约12分钟和1.14 GB，R-control总计约24分钟和2.27 GB；冻结backbone可能更快，但估算不是承诺。R-pilot只跑h96/h720会减少首阶段成本，但中途看test再决定补horizon会增加选择偏差；若采用pilot，只允许用validation的预登记门禁扩展。推荐默认（未锁定）直接R-min，最终最大预算8run，避免继续扩大搜索自由度。

建议而未锁定的positive定义同时要求：test MSE macro低于fixed U1、test MAE macro不高于U1、至少3/4 horizon test MSE改善、validation MSE macro不高于U1、至少若干horizon best epoch>0、gamma/residual有限非零、residual bypass不再稳定优于Normal、aux permutation呈稳定不利影响；若运行matched-budget control，改善不能被普通AMD continuation完全解释。部分方向改善但核心条件不齐为mixed；宏平均/多数horizon无改善、全部回退epoch0或bypass继续稳定占优为negative-or-negligible。

若一次预注册 rescue 仍非 positive：

```text
当前 TimeXer-inspired TEB 路线在 M4 有限开发中失败
```

之后不继续T4/T5/T6，不再调d、heads、patch、gate或beta，不启动P2，等待用户授权更换另一篇近三年外生模块来源。

### 36.9 Identity、provenance 与 source compatibility gate

T2数学结构不变，建议继续使用implementation variant `el-amd-m4-t2-patch-teb-v1`；warm-start/freeze是训练协议，不能复用from-scratch `M4_T2` identity。proposal only：`ablation_id=M4_T2_ADAPTER`，`rescue_protocol_id=m4_t2_u1_warmstart_frozen_adapter_v1`。是否采用仍需用户确认。

除机器相关绝对 `source_artifact_path` 外，指令列出的 initialization/source checkpoint语义、source run/SHA/config/executable/data/best epoch/task/target/schema、parameter/buffer/mode、trainable/global-query policy、optimizer state/scope/lr/wd/adapter seed、gamma policy、epoch-zero policy和protocol id都应进入scientific config及checkpoint metadata；comparison hash保留这些预先决定的字段并只移除adapter seed；第十六轮用户已锁定：`completed_epochs`、`best_epoch`与best role均为runtime outcome，不进入scientific/comparison hash。绝对path写入resolved run/manifest但不进scientific hash；source SHA/run id等稳定lineage进入manifest。artifact root可显式使用protocol语义，既有dataset/task/target/horizon/fold/seed/run_id后缀保持不变。

Summarizer必须区分from-scratch T2、warm-start adapter T2和matched-budget U1 continuation，按protocol/source SHA/source run/policy形成comparison identity，并以comparison hash+seed定义duplicate identity；completed epochs不同不得分裂identity，并拒绝lineage缺失、SHA篡改或identity冲突。建议保留schema-v2和现有13-file checksum：把`warm_start_contract_version=v1`、lineage和compatibility proof嵌入已受checksum保护的config/manifest/checkpoint，无需增加受控文件或schema-v3。

最小source gate：

1. source artifact完整性：completed、精确13-file checksum和系统校验、config/manifest/checkpoint metadata/data/schema/source SHA一致；
2. source model compatibility：逐字节核验`models/tsAMD.py`、`models/common.py`、`models/tsmoe.py`、`models/tsAMD_enhanced.py`，并核验ETTm1路径所需`utils/dataloader.py`、`utils/general.py`、`models/modules/target_exogenous_bridge.py`；目标T2另记录`models/modules/patch_conditioned_target_exogenous_bridge.py`。未来`main.py`必然因warm-start治理改变，故对未改的`prepare_args`相关合同、`_build_generic_runtime_data`、`_build_model`、`_prediction_for_loss`与schema/target-selection函数记录可验证的function-source hash，而不能仅比较整个文件SHA；
3. target implementation：允许新增runner/freeze/provenance逻辑，但不得改变AMD key/shape/forward和数据/target-selection语义；
4. source/current全局fingerprint不同必须在manifest显式记录并附逐依赖compatibility proof。strict-load成功不等于语义等价。

### 36.10 Production capability verdicts

| Capability | Verdict | 核心原因 |
|---|---|---|
| U1 best -> T2 backbone warm-start | Not supported without code changes | 无source CLI，T2 importer明确拒绝 |
| Atomic source validation | Not supported without code changes | 无checkpoint/lineage入口，现有preflight缺dtype与完整metadata |
| Backbone parameter freeze | Not supported without code changes | 无freeze policy，optimizer收全部parameters |
| Backbone buffer freeze | Not supported without code changes | `model.train()`使DDI BatchNorm buffers漂移 |
| Mixed AMD-eval / T2-train mode | Not supported without code changes | 公共API可手工表达，但epoch入口会递归覆盖 |
| Trainable-only optimizer | Not supported without code changes | runner固定`model.parameters()` |
| Global-query-only parameter exclusion | Not supported without code changes | 无显式scope/policy |
| Epoch-0 best selection | Not supported without code changes | best/history/summarizer均从epoch 1开始 |
| Warm-start lineage in artifact | Not supported without code changes | config/manifest/checkpoint无source lineage contract |
| Summarizer contract | Not supported without code changes | 当前只认from-scratch T2固定ablation/candidate contract |
| Matched-budget U1 continuation | Not supported without code changes | completed best不能作为新run warm-start，resume语义错误 |
| Overall warm-start adapter rescue | Not supported without code changes | 上述source/freeze/mode/optimizer/epoch0/provenance门禁缺失 |

### 36.11 最小未来实现范围（未执行）

- `main.py`：新增warm-start CLI与protocol字段；在staging/optimizer前执行source完整性、metadata、key/shape/dtype和compatibility gate；安全映射U1 AMD、设freeze/mixed mode并在每个epoch重应用；构造trainable-only新Adam；支持明确epoch0 candidate；写lineage/checkpoint/manifest。不得复用resume。
- `summarize_results.py`：接受并严格验证adapter/continuation protocol、lineage、epoch0语义、comparison/duplicate identity与tamper拒绝，同时保持历史from-scratch identity不变。
- `tests/test_runner.py`：扩展source atomicity、cross-horizon/dtype拒绝、freeze/buffer/mode、optimizer scope、global-only排除、epoch0、resume隔离、artifact lineage测试。
- `tests/test_summarize_results.py`：扩展adapter/continuation/legacy区分、duplicate和lineage tamper测试。
- `tests/test_tsAMD_enhanced.py`：仅在公共mapping helper放入model层时扩展；优先把协议留在runner，不改变T2 class/state keys/forward。
- 无需修改frozen AMD、Global/T2/T2G/T3/PMCR、DataLoader或数据；无需新增TEB class或新production模块文件。建议不新增测试文件，扩展现有永久回归测试。若用户锁定rescue协议，执行前canonical需登记该训练治理，唯一M4 milestone继续记录实现/实验；本轮不做这些修改。

### 36.12 待用户确认与本轮停止点

下列均是待确认项，不是已授权合同：gamma使用`0`还是`1e-3`；epoch0是否纳入best；trainable scope用全部T2还是forecast-connected-only；global-query-only冻结还是保留；adapter weight decay用`0`还是`1e-7`；保留production objective还是prediction-only；adapter最大epoch数；matched-budget continuation是否只在positive后运行；variant/`M4_T2_ADAPTER`/protocol命名；R-min/R-pilot及最大4或8 run预算。

本轮建议默认但未锁定：gamma=0、epoch0纳入best、forecast-connected-only、global-query-only冻结、adapter weight decay=0、保留production objective、R-min四个horizon最多10 epochs；只有R-min positive才追加matched-budget U1 continuation，总预算最多8run。

本轮没有新增、修改或删除测试；所有既有测试继续为`permanent_regression_test`。第十四轮已在同一代码HEAD完成220/220，本轮未修改代码，故未重跑完整回归。一次性审计仅在`/tmp/m4_t2_warmstart_audit_9Lhn7V/`范围内进行，结束前删除。

M4继续`In Progress`。本轮未修改canonical、代码、测试、M0-M3、数据或artifact；未训练、未运行validation/test、未创建rescue artifact；未实现warm-start、新TEB或P2，未进入M5/M7，未实现空间模块。审计结果只保留在本milestone的未stage、未commit、未push修改中，等待用户与ChatGPT审核。

## 37. 第十六轮：warm-start/adapter-style T2 rescue production capability 实现

### 37.1 用户锁定合同与实现边界

用户已将第十五轮建议锁定为唯一一次预注册 rescue：每个 horizon 从对应 U1 `best.pt` 加载 60-key AMD 主干，T2 按 seed=2024 fresh 构造且不加载任何旧 TEB 参数，随后以无 RNG 的 policy 将 effective `gamma_teb` 置为严格 0。Epoch 0 是未训练的 source-equivalent initialization candidate，纳入 strict-improvement best selection；AMD parameter、persistent buffer 和 module mode 分别固定为 frozen/frozen/eval，T2 为 train；只训练 15 个 forecast-connected T2 tensors（22,881 parameters），4 个 global-query-only tensors（16,480 parameters）冻结。Optimizer 为 fresh Adam、lr=`3e-5`、weight decay=`0`；objective 仍为 prediction MSE + frozen selector auxiliary，但三项分开记录且 validation best 只看 prediction MSE。

Adapter identity 固定为 implementation variant=`el-amd-m4-t2-patch-teb-v1`、ablation=`M4_T2_ADAPTER`、protocol=`m4_t2_u1_warmstart_frozen_adapter_v1`；matched-budget capability identity固定为 implementation variant=`el-amd-pmcr-teb-v1`、ablation=`M4_U1_CONTINUATION`、protocol=`m4_u1_matched_budget_continuation_v1`。二者均使用 `warm_start_contract_v1`，是训练协议而非新 architecture。第一阶段未来最多 4 个 adapter run；只有 provisional positive 且再次获授权，才能运行 4 个 continuation，总预算上限 8。一次 rescue 仍非 positive 时停止当前 TimeXer-inspired TEB 路线，不继续 T4/T5/T6、不再调 TEB 参数、不启动 P2，等待用户授权更换新的近三年外生模块来源。

本轮只修改 canonical、本 milestone、`main.py`、`summarize_results.py`、`tests/test_runner.py`、`tests/test_summarize_results.py`；没有修改任何模型模块、DataLoader、数据、M0-M3 或既有 artifact，也没有新增文件。没有运行四个真实 adapter、没有运行 continuation、没有读取新的 ETTm1 test 指标、没有创建真实 rescue artifact。

### 37.2 Runner、source preflight 与原子映射

Runner 新增 standard/adapter/continuation 三类显式 protocol contract。Standard 不携带 warm-start block；warm-start CLI 必须显式给出 completed U1 source、role=`best` 和预期 checkpoint SHA。ETTm1/`target_exogenous`/MS/OT、target=6、aux=[0..5]、seq=512、seed=2024、horizon∈{96,192,336,720} 及 optimizer/epoch/variant/ablation 的矛盾组合均在 artifact path、staging、log 和 optimizer 创建前拒绝；warm-start source 与 `--resume` 不能作为同一个新-run initialization 使用。

四个锁定 U1 source 均通过 formal preflight：completed schema-v2、精确 13-file checksum set、Python digest与系统 `sha256sum -c`、config/manifest/checkpoint/data/source/schema/dirty=false、run/config/comparison/checkpoint identity、无 `pmcr.*`/`teb.*`。h336 指令中的 source checkpoint SHA 原文本多写了末尾字符 `2`（65 hex）；本轮以 artifact `sha256sum` 和既有第十四/十五轮证据的实际 64-hex 值 `89da2c854dce4e124c09575835d52e313bf3a6aeab2b556a060c7174709cb530` 修正授权文件中的证据转录，不修改 source artifact。

全局 executable fingerprint 因 runner/summarizer 实现而不同，故使用 `source_compatibility_proof_v1`。七个 critical files 的 source/current SHA 全部逐字节相同：`models/tsAMD.py`=`fa72cdbe...90ba1`、`models/common.py`=`570f47c3...a3afb`、`models/tsmoe.py`=`d6c78884...ee0f1`、`models/tsAMD_enhanced.py`=`3eacf0cd...04f5`、`models/modules/__init__.py`=`1c381c66...61e04`、T2 module=`5bba40e...8f69`、`utils/dataloader.py`=`27b68d05...c87c`。Proof 与稳定 lineage 写入 checksum 覆盖的 config/checkpoint/manifest；absolute source path只留在 resolved runtime/manifest，不进入 scientific/comparison hash。

Adapter mapping 在任何参数写入前验证完整 key、shape 与 dtype：source=60、target=79、mapped=60、allowed missing精确为19个 `teb.*`、unexpected/shape/dtype mismatch均为空。实现先保存目标完整 parameter/buffer clone，构造 merged state 后用普通 `load_state_dict(strict=True)` 一次写入；任何异常恢复原 clone并核对当时的 `pre_repair_production_unversioned_digest`。Continuation 为60→60同结构 strict mapping。永久测试覆盖 unexpected、shape、dtype、cross-horizon与污染拒绝；四个真实 source 均通过。h96 production smoke 当时记录的是 `pre_repair_production_unversioned_digest`：source state 与 mapped AMD digest同为 `b4111c4cf3898f8f5617be00ae9f57d2ab8383220447a919388c874be7cfe835`；fresh T2 subset mapping前后同为 `6bc1bb67afcfce02650a8b5c4dd16c54499778f65f1e893b47c5e1a20e7d12cc`。该旧 production helper 的 framing 为 key length=4-byte big-endian、dtype length/rank=2-byte big-endian、dimension/raw length=8-byte big-endian，无 header/key count；这些字符串同样不与 production v1 直接比较。

### 37.3 Gamma、freeze/mode、optimizer 与 objective

Gamma-zero helper在 `torch.no_grad()` 下仅改变 `teb.gamma_teb`，effective值严格为0，CPU与全部CUDA RNG state均不变，其他T2 parameter/buffer逐元素不变。每个adapter训练epoch由唯一mode helper重施 `model.eval(); model.teb.train()`：top-level与所有非TEB modules为eval，AMD BatchNorm/dropout和selector noise关闭，T2/MHA dropout保持train；validation/test仍为全模型eval。

Forecast-connected allowlist精确为：`teb.gamma_teb`；`teb.patch_query_projection.{weight,bias}`；`teb.patch_query_norm.{weight,bias}`；`teb.exogenous_projection.{weight,bias}`；`teb.exogenous_norm.{weight,bias}`；`teb.cross_attention.{in_proj_weight,in_proj_bias,out_proj.weight,out_proj.bias}`；`teb.patch_output_projection.{weight,bias}`。Global-only freeze list精确为：`teb.global_query_projection.{weight,bias}`、`teb.global_query_norm.{weight,bias}`。未知/缺失 TEB key 或计数不符会在 optimizer 前拒绝。Fresh Adam 只接收15个 allowlist tensor，无重复、无 frozen parameter、state初始为空、lr=`3e-5`、weight decay=`0`。

训练日志对 warm protocol分别保存 prediction、selector auxiliary和total；production backward仍使用总和，但永久测试证明 auxiliary 对adapter参数无梯度，total与prediction-only产生逐元素相同的adapter gradient。Validation selection继续只使用纯prediction MSE，不能把total解释成prediction metric。

### 37.4 Epoch-0、resume、lineage 与 identity

Warm-start新run在source preflight、target构造/mapping、adapter gamma-zero、scope/mode和fresh optimizer之后，训练前完整评价validation并保存可恢复的epoch-0 `best.pt`/`last.pt`。Best从epoch0 MSE初始化，只有严格更小的训练epoch可替换，相等不替换。History只允许1..`completed_epochs`；`completed_epochs`是实际执行的额外训练epoch数，`best_epoch`允许0..N。最终best=0时，test仅在全部预算完成后加载epoch-0 best执行，不强制改选epoch1。

Warm-start staging可从自身epoch0或训练epoch恢复：核对sealed protocol/source SHA/lineage/proof/scope/optimizer/gamma/epoch-zero/objective，直接恢复该target run自己的`last.pt`、optimizer/best/history/train-generator/RNG；不重新打开source、不重复mapping或zero gamma。Completed U1 source不是resume target；standard resume语义保持不变。Continuation只提供同结构U1 source strict restore、fresh full-AMD Adam、epoch0与额外epoch1..10能力，不自动触发，本轮没有运行。

Schema仍为v2且受控文件仍精确13个。Training protocol、稳定source lineage和compatibility proof进入config/checkpoint/manifest。Scientific identity包含预先决定的protocol/source/policy/max epochs/seed等；comparison identity仅移除adapter seed，duplicate identity使用comparison hash+seed。`completed_epochs`、`best_epoch`、best role、wall-clock、artifact size和最终指标均不进入scientific/comparison/duplicate identity；绝对source path也不进入hash。Summarizer分别验证standard、adapter与continuation，接受合法warm best_epoch=0，拒绝standard best_epoch=0、lineage/proof/policy/role/history tamper、parallel spoof和同identity不同completed epochs的重复run，并且不重新访问absolute source path。

### 37.5 回归、真实 h96 smoke 与 synthetic 生命周期

永久测试只修改并保留`tests/test_runner.py`与`tests/test_summarize_results.py`，没有新增、删除或弱化测试。Runner/summarizer定向组73/73通过（failed=0、skipped=0）；TEB/AMDEnhanced保护组106/106通过；PMCR/M1保护组36/36通过；完整discovery 233/233通过（failed=0、skipped=0），高于原220项。CUDA测试在NVIDIA A800 80GB PCIe上实际执行；完整回归用时25.792 s（unittest自身22.649 s），既有1e-6 parity门槛未放宽。

真实h96 smoke只构造train与validation DataLoader，各冻结一个batch；未调用`get_test()`、未遍历完整validation、未创建artifact或checkpoint、未运行epoch。Train/validation的x均为`[128,512,7]`，y均为`[128,96,1]`。Formal source preflight为13/13与七个critical SHA通过，60→79映射通过，目标staging未创建。Epoch0 source U1与adapter prediction均为`[128,96,1]`，MoE均finite；prediction与MoE均`torch.equal=True`、max abs=0。State source均为`[128,1056]`，但其最后32维不要求等价且实测不相等（max abs=`0.8773267269`），符合U1零context与fresh T2 live context语义。

独立真实train-batch两步probe使用正式mixed mode与optimizer：gamma `0 -> -2.9998407626e-5 -> -5.9238991525e-5`；step1/step2 prediction loss=`0.0581099615`/`0.0581099503`，auxiliary=`0.0116437199`/`0.0116437199`，total=`0.0697536841`/`0.0697536692`，全部finite，auxiliary `requires_grad=false`。两步AMD parameter、AMD persistent buffer和global-only最大变化始终0；step1非gamma forecast最大变化0，step2为`1.7639482394e-5`；全部parameter/buffer finite。Top-level=false、T2=true、非TEB/AMD BatchNorm/dropout均eval、T2 dropout=train；CPU RNG未消费，CUDA RNG因T2 dropout被消费。该两步仅是工程门禁，不构成candidate训练或性能证据。

`TemporaryDirectory` synthetic A-E均通过：A（训练更差）best_epoch=0、role正确、history=[1,2]、completed=2并被summarizer接受；B（有改善）best_epoch=1/`trained_epoch`并接受；C continuation带lineage/epoch0且无PMCR/TEB并接受；D篡改lineage被拒绝；E仅completed epochs和absolute path不同但稳定identity相同的两个run仍按duplicate拒绝。A/B/C均为schema-v2、精确13/13且系统`sha256sum -c`通过，fixture目录自动回收。

### 37.6 Backward compatibility、fingerprint 与停止点

Standard AMD/U1/Global/T2/T2G/T3/PMCR保持既有CLI、optimizer、best从epoch1、history/metrics/manifest/resume语义；standard scientific/resolved/checkpoint/manifest不携带warm block，历史hash fixture逐值不变。T2 class、source/state keys与所有模型模块/DataLoader字节未改。第十四轮8个U1/U2 completed artifacts再次通过精确13/13 Python/system checksum，未读取其metrics数值，也未迁移或重写artifact。

实现后source fingerprint算法仍为`sha256_length_prefixed_relative_path_and_content_v1`，文件数仍为20，新SHA-256为`4cd47a74687cdcdb7a1288522268a3e3c87a7c59e10cc0dfbc8347cd8aa7a2e6`。M0-M3、baseline、frozen AMD、DataLoader、Global TEB、PMCR、T2/T2G/T3及参考仓库保持不变。

M4继续`In Progress`，TEB adequacy gate继续`remains failed`。本轮没有运行四个真实adapter、没有运行continuation、没有读取新的ETTm1 test、没有创建真实rescue artifact、没有修改T2数学结构、没有新增TEB、没有实现P2、没有进入M5/M7或空间模块。本实现和文档均保持未stage、未commit、未push，等待用户与ChatGPT review；不得自动启动R-min、continuation或Git closure。

### 37.7 第十六轮 state-digest provenance repair

本 repair 不改变当前轮次；M4 仍为第十六轮、状态仍为 **In Progress**。触发点是第十五轮 `legacy_unversioned_audit_digest` 与第十六轮 `pre_repair_production_unversioned_digest` 的两组历史字符串不同：h96 source U1 分别为 `95ebbbb744b9b112207152de728ab35ba21eb98ab87030aa4aeccf072c54dd23` 与 `b4111c4cf3898f8f5617be00ae9f57d2ab8383220447a919388c874be7cfe835`；fresh T2 subset 分别为 `5d079a22f13b7d602aa0eff037452bd8041dea6be84de3037bbfbedc79927ad5` 与 `6bc1bb67afcfce02650a8b5c4dd16c54499778f65f1e893b47c5e1a20e7d12cc`。

保留的本地 Codex session 记录使 legacy 一次性算法得以精确重建，而不是猜测：它按字符串 key 排序，key/dtype/raw length、rank、shape dimension均为8-byte little-endian（dimension signed），无 header/key count。Repair 前 production helper 也已从源码与四 horizon probe逐值复现：key length为4-byte big-endian，dtype length与rank为2-byte big-endian，dimension/raw length为8-byte big-endian，无 header/key count。对同一组 h96 source和fresh T2 tensor同时运行两套已恢复 framing，分别精确得到上述两组历史值；四 horizon legacy 值也逐项重现第十五轮表。因此精确根因是 field-width、endianness 与 framing 合同不同，不是 checkpoint/source mapping 或 underlying tensor state 异常。Fresh initialization digest一般仍依赖精确 RNG 起点和模型构造顺序；本次之所以可把这四个 fresh T2 值归因于 framing，是因为同一内存 tensor同时计算两种 digest，而不是仅凭两次独立构造的字符串比较。

Production 合同现唯一固定为：

```text
state_digest_contract_version =
sha256_length_prefixed_state_dict_v1

key_policy =
exact_state_dict_keys_no_prefix_normalization
```

输入只允许唯一字符串 key 对应的 dense strided、非 meta、非 quantized tensor；精确 key 不去除或补写任何 prefix，按 UTF-8 bytes 排序。算法只允许 little-endian host。固定 byte stream 为 `sha256_length_prefixed_state_dict_v1\0` header、big-endian uint64 key count；随后每个 tensor写入 `LP(key_utf8)`、`LP(dtype_utf8)`、big-endian uint64 rank、逐维 big-endian uint64 shape、`LP(raw bytes)`，其中 `LP` 为 big-endian uint64 length 加 payload，dtype 使用 `str(tensor.dtype)`，raw bytes 来自 detached CPU contiguous flattened uint8 view 的 NumPy C-order bytes。Device与requires_grad不进入；`torch.save`/pickle、JSON、repr及无 framing 拼接禁止。非 Mapping、非字符串 key、非 tensor、sparse/其他 layout、meta、quantized均拒绝。

四个锁定 U1 source artifact均再次通过精确13-file checksum和系统 `sha256sum -c`；probe不解析 `metrics.json` 数值。Production v1 source/mapped结果如下，且每个 key另由 `torch.equal` 逐 tensor通过：

| H | algorithm | source keys | target AMD keys | source digest | mapped digest | equal |
|---:|---|---:|---:|---|---|---|
| 96 | `sha256_length_prefixed_state_dict_v1` | 60 | 60 | `f65b7f01a7a898d31cb56c37131d25db4c7dee1307731282195ff2094962863d` | `f65b7f01a7a898d31cb56c37131d25db4c7dee1307731282195ff2094962863d` | true |
| 192 | `sha256_length_prefixed_state_dict_v1` | 60 | 60 | `f955deac3265d029658952e5054eda491b1debe9081f59f3b11fc351cfb5b22b` | `f955deac3265d029658952e5054eda491b1debe9081f59f3b11fc351cfb5b22b` | true |
| 336 | `sha256_length_prefixed_state_dict_v1` | 60 | 60 | `c5b715e1b92393d13f1267eac4d48941ad51d928dd92261cd53db6995414224c` | `c5b715e1b92393d13f1267eac4d48941ad51d928dd92261cd53db6995414224c` | true |
| 720 | `sha256_length_prefixed_state_dict_v1` | 60 | 60 | `668c9b56934af6d69fda96349b968e47ef88cc86507999f4e84f8effe6ff9d79` | `668c9b56934af6d69fda96349b968e47ef88cc86507999f4e84f8effe6ff9d79` | true |

Fresh T2 subset在相同 mapping 的前后保持19 keys且逐 tensor相等：

| H | algorithm | T2 keys | before digest | after digest | equal |
|---:|---|---:|---|---|---|
| 96 | `sha256_length_prefixed_state_dict_v1` | 19 | `32336718131e5dd518a0dbef9c56af792a4bed45fa44ed0d4a14a94811fba308` | `32336718131e5dd518a0dbef9c56af792a4bed45fa44ed0d4a14a94811fba308` | true |
| 192 | `sha256_length_prefixed_state_dict_v1` | 19 | `2ec396db15c68e11dc7dc8a3b8656bb4d9df45573fa21ce04b30d241066ff8d0` | `2ec396db15c68e11dc7dc8a3b8656bb4d9df45573fa21ce04b30d241066ff8d0` | true |
| 336 | `sha256_length_prefixed_state_dict_v1` | 19 | `ad78513c9460612e3c2be8e88d81bf2aa67b1a7b5e72844e411a7e5b4825307b` | `ad78513c9460612e3c2be8e88d81bf2aa67b1a7b5e72844e411a7e5b4825307b` | true |
| 720 | `sha256_length_prefixed_state_dict_v1` | 19 | `a95d2c2e035c58569f8fae6d84c0dec57cd692d1a7dfd73acced0156c0f933cf` | `a95d2c2e035c58569f8fae6d84c0dec57cd692d1a7dfd73acced0156c0f933cf` | true |

Atomic mapping/rollback的全部 production call site统一使用该唯一 helper，内部 mapping report明确携带版本；失败 mapping前后完整 target v1 digest相同。Digest helper覆盖 parameter与persistent buffer，并以 `torch.equal` 作第二重确定性证明。当前 `source_compatibility_proof_v1` 不存任何 state digest，故本 repair没有增加artifact文件、没有升级schema-v2、没有改变13-file checksum set，也没有修改summarizer sealed contract；`summarize_results.py`与`tests/test_summarize_results.py`的repair前后字节不变。Digest version/value只作为provenance/integrity evidence，不进入scientific/comparison/duplicate identity；standard hash fixture、warm-start identity、completed_epochs/best_epoch identity边界均保持。

永久测试未新增文件，只修改并保留 `tests/test_runner.py`。新增4个test methods，覆盖版本常量与empty golden、重复/插入顺序/deep clone、non-contiguous/contiguous、device/requires_grad、value/dtype/shape/key/key-set敏感性、非法mapping/value/sparse/meta/quantized拒绝；既有四source mapping测试增补source/mapped与fresh before/after v1 digest，既有失败mapping测试继续覆盖rollback无污染，身份测试确认diagnostic digest变化不改变scientific identity。Runner定向52/52通过；runner+summarizer保护77/77通过；完整回归237/237通过（failed=0、skipped=0），原233项全部保留。CUDA digest CPU canonicalization测试和既有CUDA回归均在NVIDIA A800 80GB PCIe上实际执行；既有1e-6门槛未放宽。

Repair后 executable source fingerprint为 `sha256_length_prefixed_relative_path_and_content_v1`、20 files、`f3a0a9de6bb3296437202c5a9ba4cebfd88424191232db212f55d151db144a4a`。只读probe目录为 `/tmp/m4_state_digest_repair_BaHgLwK3/`，只包含临时script/result/log，结束前删除。Probe只做四个source strict restore、fresh T2构造、state subset、mapping、`torch.equal`与digest；未构造DataLoader、未执行model forward/backward、未创建optimizer或artifact。

Warm-start source preflight、60→79原子映射、19个fresh T2 keys、gamma=0、epoch-0 parity、freeze/mixed mode、15/4 tensor scope、epoch-0 selection、identity隔离、233项原回归、h96 two-step smoke与synthetic artifact lifecycle的第十六轮既有结论均不因digest framing消歧而撤销。本 repair没有运行真实adapter训练、validation/test、R-min或continuation，没有创建/修改真实artifact，没有修改模型/DataLoader或第十四轮8个artifact。TEB adequacy gate remains failed，P2未启动，M4保持In Progress；Git closure仍未授权，本轮不stage、不commit、不push。

## 38. 第十七轮：Frozen-AMD + Fresh-T2 R-min 四 horizon development 实验

### 38.1 起始现场、回归与 artifact root

本轮从 branch AMD-paper-repro-custom-modules-v1、HEAD 08e704aa163953360c0a2675154e23141af90085 启动；parent=e5d1b4aecdf34705bd9f908fa13644c74fddebf5，title=feat(m4): implement warm-start T2 rescue protocol。local HEAD、tracking 与 live remote 完全一致，ahead/behind=0/0，worktree/index clean。baseline tag amd_reproduced_baseline_v1 的本地与远端 peeled commit 均为 fa9665627e6fcfb1d0c2bc22d943ca9666304fd6。

Canonical SHA-256 为 fc418a32cf07c8cd9854f65efe2effbbad9137464c39512f0dbccbdce706ed6e，起始 M4 SHA-256 为 ef20281e3dd369bcdbd602f58106252e6726b319d440f7f4c283928e2d182ce4。M0-M3 相对 parent 无差异；UrbanEV、ModernTCN、TimeXer 各自 worktree/index clean 且 ahead/behind=0/0。Executable source fingerprint 为 sha256_length_prefixed_relative_path_and_content_v1、20 files、f3a0a9de6bb3296437202c5a9ba4cebfd88424191232db212f55d151db144a4a。

四个第十四轮 U1 source 在不解析 source test 指标的前提下完成 best.pt SHA 交叉核验和系统 sha256sum -c；每个均为 13/13 OK。随后按锁定命令运行完整永久回归：237/237 passed、failed=0、skipped=0，unittest 自身 24.289 s。

固定 artifact root 为 artifacts/m4-development/ettm1-stage-f-t2-adapter-rescue-v1；启动前该路径不存在，因而没有 completed、hidden staging、failed、running 或其他可复用内容。本轮没有覆盖、删除或静默复用 artifact。

### 38.2 四个 source、实际命令与 lineage

| H | source run_id | source config hash | source comparison hash | source best epoch | source best.pt SHA-256 |
|---:|---|---|---|---:|---|
| 96 | 20260901T095811.286299Z-6e33fa77 | fa2c4da41f34eca232907e4d6462305cb8ef3ef15fc8996f7c67baa0411ddb2d | 4fbf51cca6fa7bad95bc8e35ddfc416d6dc45a78c331aead19e007f6d24ef74b | 7 | 66458be335ac7948889156bf6a7a91af7221f3b75838a1522fd701e8e78b42d0 |
| 192 | 20260901T100147.203364Z-f03509fd | 1585152dbf1ff74d935f7404f8b2699881b85c7224252371e939db585f2611d0 | 771cdff549663c52cc213c5cfaf9ed731362ce5b544457776bf7653ff0c950bf | 3 | f8b0308578b10f09ade232cf2c6ac2e7826b1e28ceb994c2a321d44c4563be6a |
| 336 | 20260901T100520.627934Z-0c9f399c | 45ea9453083d6fb38381d03ba3a8455191e28e6221a2c9f86d0dee72ba8e8ff5 | 8ccb82795987bd3df127f1cd087c8da7ff2cae53ee1d8c282fd565e730392d4b | 3 | 89da2c854dce4e124c09575835d52e313bf3a6aeab2b556a060c7174709cb530 |
| 720 | 20260901T100834.831317Z-5f71979f | 93ddc59a0879b435a4cf4742e76c75460c254283ca00b2138164c4fe735d51cd | 3f5f973cd636b205f813ba4589845c5a5ffb91fdf86a379726dd2cc6d039c291 | 7 | a13126522bb2cf8f5871c46272222242b8ebb6077145658558199c9473ba109b |

四个 source artifact 路径继续使用第 36.2 节的精确路径。每个 target run 的 command.txt 保存实际 Python executable 与完整 argv；绝对路径为：

| H | command.txt |
|---:|---|
| 96 | /public/home/yueweiting/大论文/AMD/artifacts/m4-development/ettm1-stage-f-t2-adapter-rescue-v1/el-amd-m4-t2-patch-teb-v1/ETTm1/target_exogenous/OT/horizon_96/fold_official/seed_2024/20260901T153705.487633Z-5916049f/command.txt |
| 192 | /public/home/yueweiting/大论文/AMD/artifacts/m4-development/ettm1-stage-f-t2-adapter-rescue-v1/el-amd-m4-t2-patch-teb-v1/ETTm1/target_exogenous/OT/horizon_192/fold_official/seed_2024/20260901T153933.304452Z-45b7e8bc/command.txt |
| 336 | /public/home/yueweiting/大论文/AMD/artifacts/m4-development/ettm1-stage-f-t2-adapter-rescue-v1/el-amd-m4-t2-patch-teb-v1/ETTm1/target_exogenous/OT/horizon_336/fold_official/seed_2024/20260901T154148.160319Z-07ee1485/command.txt |
| 720 | /public/home/yueweiting/大论文/AMD/artifacts/m4-development/ettm1-stage-f-t2-adapter-rescue-v1/el-amd-m4-t2-patch-teb-v1/ETTm1/target_exogenous/OT/horizon_720/fold_official/seed_2024/20260901T154406.678180Z-f41be476/command.txt |

四条命令除 horizon、pred_len、source artifact 与 source checkpoint SHA 外完全一致：ETTm1/MS/target_exogenous/OT、target_idx=6、aux_idx=[0,1,2,3,4,5]、seq_len=512、seed=2024、batch=128、10 epochs、T2 patch=32/d=32/heads=4/dropout=0.1、AMD n_block=1/alpha=0/mix=3,2/patch=16、lr=3e-5、weight_decay=0、无 early stopping。Identity 固定为 el-amd-m4-t2-patch-teb-v1、M4_T2_ADAPTER、m4_t2_u1_warmstart_frozen_adapter_v1、warm_start_contract_v1。

Formal source preflight 均证明 completed schema-v2、13-file exact checksum、source metadata、data/schema、dirty=false、无 pmcr/teb source state；source/current global fingerprint 不同，但 source_compatibility_proof_v1 的 7 个 critical file SHA 逐项相同。四条 mapping 均为 source=60 keys、target=79 keys、mapped AMD=60 keys、allowed missing 精确 19 个 fresh teb keys，unexpected/shape/dtype mismatch 均为空。

### 38.3 Completed adapter artifacts

全部四个 run 一次成功完成，没有第二身份重跑。正式 summarizer 接受 4/4 run 并形成 4 个 aggregate group，没有 duplicate identity；root 恰有四个 manifest，无 hidden staging。每个 run 均 status=completed、schema-v2、history epochs 精确为 1..10、completed_epochs=10、epoch 0 不在 history 中、best_epoch 属于 [0,10]、指标 finite、Python exact checksum 与系统 sha256sum -c 均为 13/13。

config hash 即 stable scientific_config hash；下表仍同时明确 scientific 身份语义。

| H | run_id | config/scientific hash | comparison hash | best epoch/role | best.pt SHA-256 | last.pt SHA-256 | wall s | bytes |
|---:|---|---|---|---|---|---|---:|---:|
| 96 | 20260901T153705.487633Z-5916049f | ef658ec9b5054465cdcca6ddbc9702a4ac1f9a3d854d75d5b728fee657c3ee9d | e5f791315ccba5b118d656c2be94973342230f165b9aa995a6bf377966c91a2c | 1 / trained_epoch | b57ff0c00fbf59d6c593cd07b482abecdd9acccdf2848f48b28722bcbcd5d8fe | fc192e8a81f26192b6472fcfb9a3749ff9c065fa9051ed3a372177252ed3cc46 | 115.049986 | 123814608 |
| 192 | 20260901T153933.304452Z-45b7e8bc | 5a5432236dea9c3ff8fe73e76e2a4747d0015ab9c841dde84152c6baf4b292d0 | 4ecb72af0110af7df4f2266a3054bc6aa7b8ad58bab8067cf6c36974066e2e04 | 0 / epoch_zero_initialization | 640438a130cc90f3f31bca4a9cc406974a8172d2fe0c7eac930bc0d54d1e3672 | cd60eb7223113b8625098f04795be6059c339145d2dcf148b3cc95618e466a2a | 99.677459 | 142698267 |
| 336 | 20260901T154148.160319Z-07ee1485 | 84fff7e1abcfe0a8d65c0908ddce6a9111fcf5b4e9aca23628f9a4c7e8c29639 | f25f509718ad3cfd46320b602453df286dd8dbc3454b1cbf1076cbaa13f4e4a2 | 0 / epoch_zero_initialization | 4a400554eaeb16f60521702507322ebcd058bf6e8178370fdd1880f6e29ddcb1 | 19c42a973c4396d0e2296617e3d7804ee7950dcef52ca0034c9054cb99399e8b | 112.507478 | 171023722 |
| 720 | 20260901T154406.678180Z-f41be476 | c9030e201371796851fe67805e8370846dc6f2a80ddddfcdea2061ea8576f6f9 | 505873c35c40914b22abcb25400a833e2e05b390277e4e45860c8155ab0600cf | 0 / epoch_zero_initialization | ad2bb2c0e5bc7594bfa8adecb78aaaec732593ce647808e15d8bf806016199aa | ab90dc4ba31a7ec93c34e5a49149a4cd3b72397792d55c57e5ed2d3e0fc8ca74 | 111.657049 | 246558028 |

四条 artifact 的共同 data SHA-256 为 6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e；target_exogenous_schema_v1 fingerprint 为 f6dd94841b5d9d0b7515b19e0ff1876bf6476068054eacdc02ac6fcab3f084dc；current executable fingerprint 为 f3a0a9de6bb3296437202c5a9ba4cebfd88424191232db212f55d151db144a4a。四条总 wall-clock 为 438.892 s，artifact 总空间为 684094625 bytes。

### 38.4 U1 vs Adapter development 指标

ETTm1 继续只作为 development-only benchmark；以下 validation/test 已用于候选开发，不进入 M6 正式主表，也不支持未见测试泛化主张。负相对变化表示 adapter 改善。

Validation MSE：

| H | fixed U1 | Adapter | relative change |
|---:|---:|---:|---:|
| 96 | 0.051164323931 | 0.051164103197 | -0.00043142% |
| 192 | 0.073506087241 | 0.073506087241 | 0% |
| 336 | 0.090008565565 | 0.090008565565 | 0% |
| 720 | 0.104045309980 | 0.104045309980 | 0% |
| Macro mean | 0.079681071679 | 0.079681016496 | -0.00006926% |

Validation MAE：

| H | fixed U1 | Adapter | relative change |
|---:|---:|---:|---:|
| 96 | 0.168072960404 | 0.168067353652 | -0.00333590% |
| 192 | 0.207844046975 | 0.207844046975 | 0% |
| 336 | 0.235085840964 | 0.235085840964 | 0% |
| 720 | 0.255146362864 | 0.255146362864 | 0% |
| Macro mean | 0.216537302802 | 0.216535901114 | -0.00064732% |

Development test MSE：

| H | fixed U1 | Adapter | relative change |
|---:|---:|---:|---:|
| 96 | 0.027597208934 | 0.027597859607 | +0.00235775% |
| 192 | 0.041161235376 | 0.041161235376 | 0% |
| 336 | 0.052610279493 | 0.052610279493 | 0% |
| 720 | 0.070352296784 | 0.070352296784 | 0% |
| Macro mean | 0.047930255147 | 0.047930417815 | +0.00033939% |

Development test MAE：

| H | fixed U1 | Adapter | relative change |
|---:|---:|---:|---:|
| 96 | 0.126248707310 | 0.126249907372 | +0.00095055% |
| 192 | 0.154727269120 | 0.154727269120 | 0% |
| 336 | 0.174313073967 | 0.174313073967 | 0% |
| 720 | 0.200916141056 | 0.200916141056 | 0% |
| Macro mean | 0.164051297863 | 0.164051597879 | +0.00018288% |

| Metric | mean-horizon relative | relative change of macro means |
|---|---:|---:|
| Validation MSE | -0.00010786% | -0.00006926% |
| Validation MAE | -0.00083398% | -0.00064732% |
| Development test MSE | +0.00058944% | +0.00033939% |
| Development test MAE | +0.00023764% | +0.00018288% |

### 38.5 Epoch 0、best/last gamma 与冻结证明

四个 initialization validation 的 MSE、MAE、num_batches 和 num_elements 均与对应 source U1 best validation 在 Python full precision 下严格相等。h192/h336/h720 的重建 epoch-0 79-key full state 又与 best.pt 逐 tensor torch.equal；history 只含 1..10，没有把 epoch 0 伪装成普通训练 epoch。best_epoch>0 的 horizon 数为 1/4。

| H | effective gamma init | best gamma | last gamma | best forecast tensors moved / max abs | last forecast tensors moved / max abs |
|---:|---:|---:|---:|---|---|
| 96 | 0 | -0.0062736641 | -0.0820560753 | 15/15 / 0.0072885901 | 15/15 / 0.1106575429 |
| 192 | 0 | 0 | +0.0894100219 | 0/15 / 0 | 15/15 / 0.1042429283 |
| 336 | 0 | 0 | +0.1028243154 | 0/15 / 0 | 15/15 / 0.1101957988 |
| 720 | 0 | 0 | -0.0871193483 | 0/15 / 0 | 15/15 / 0.1000282764 |

因此四条训练轨迹的 15 个 forecast-connected tensors 均在 last state 实际移动；h192/h336/h720 只是 validation best 严格回退到未移动的 epoch 0。每条 run 的 57 个 AMD parameters 与 3 个 AMD persistent buffers 在 best/last 相对 source-mapped epoch 0 的 max change 均为 0；4 个 global-query-only tensors 在 best/last 的 max change也均为 0。该证据同时区分了训练确实发生与 best checkpoint 回退。

### 38.6 Best-checkpoint residual 与 bypass

每样本 residual ratio 定义为 norm(gamma*delta_target)/(norm(target_hidden)+1e-12)。h192/h336/h720 的 best gamma=0，因此 validation/test 的 mean、median、p10、p90、p99、max 全部严格为 0。h96 为唯一非零 best residual：

| Split | mean | median | p10 | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| h96 validation | 0.001258977 | 0.001223564 | 0.000967429 | 0.001596620 | 0.002007371 | 0.002636101 |
| h96 test | 0.001304640 | 0.001267959 | 0.000994289 | 0.001658338 | 0.002079300 | 0.002557177 |

同一 best checkpoint 将 temporal residual 旁路后，相对 Normal 的变化如下；负值表示旁路更优：

| H | validation MSE | validation MAE | test MSE | test MAE |
|---:|---:|---:|---:|---:|
| 96 | +0.00043142% | +0.00333601% | -0.00235770% | -0.00095054% |
| 192 | 0% | 0% | 0% | 0% |
| 336 | 0% | 0% | 0% | 0% |
| 720 | 0% | 0% | 0% | 0% |

h96 validation 上 Normal 极轻微优于 bypass，test 上 bypass 极轻微优于 Normal；另外三条因 epoch-0 gamma=0 而严格相同。因此 bypass 没有呈现稳定优于 Normal，也不能被解释为独立训练消融或因果证据。

### 38.7 Aux K/V cyclic permutation、wrapper parity 与 state immutability

每个 batch 固定使用 shifts=(1,floor(B/3),floor(2B/3)) 对 auxiliary sample K/V 做三次 torch.roll；target hidden/query 和样本内变量顺序保持不变。B=128 时 shifts=(1,42,85)，所有 validation/test batch 均 B>1，无法置换的 batch 数为 0。下表为三个 shifts 的 aggregate metric 均值相对 Normal；负值表示 permutation 后指标更低：

| H | validation MSE | validation MAE | test MSE | test MAE |
|---:|---:|---:|---:|---:|
| 96 | +0.00014634% | +0.00035565% | +0.00055399% | -0.00008877% |
| 192 | 0% | 0% | 0% | 0% |
| 336 | 0% | 0% | 0% | 0% |
| 720 | 0% | 0% | 0% | 0% |

h96 permutation 的 prediction max change 为 validation/test 0.000954032/0.000875235；A_patch、delta_target、exo_context 最大变化分别为 2.380069/1.502985/2.249131 和 2.473568/1.820715/2.393269。h192/h336/h720 的 A_patch、delta、exo_context 仍发生有限非零变化，但 prediction max change 为 0，因为 best gamma=0。Permutation 因而没有呈现稳定不利指标影响；它只证明 K/V 对内部表示 live，不是独立训练因果消融。

四个 horizon、两个 split 上，formal forward 与诊断 wrapper 的 prediction、MoE loss、state_source 最大绝对误差全部为 0；Normal 重算指标与封存 metrics 在 1e-12 内一致。诊断前后 sha256_length_prefixed_state_dict_v1 digest 分别保持：

| H | before = after |
|---:|---|
| 96 | 717a5a3c645671026fffdbebc37e5a3731fcfbfadf6e3c0bb4bce31715bca825 |
| 192 | 4091958897a25228b9baefbe7adc730239eb4e109050f16eaee26344a74013c7 |
| 336 | 3004997849ddf2f7af5ada174df0b92411803c28089c55a92c8534ab873eb5c3 |
| 720 | 6cb9506901a20ccefa7b8fdeccce03366153cbd583cd2ce8e6f6960681e31df3 |

每个 state tensor 另以 torch.equal 逐项通过，证明诊断没有修改模型 parameter 或 persistent buffer。

### 38.8 Provisional signal 判定与停止线

只使用 canonical 已有硬条件：

| 条件 | 结果 | 依据 |
|---|---|---|
| test MSE macro < fixed U1 | Fail | 0.047930417815 > 0.047930255147，+0.00033939% |
| test MAE macro <= fixed U1 | Fail | 0.164051597879 > 0.164051297863，+0.00018288% |
| 至少 3/4 horizon test MSE 改善 | Fail | 0/4 改善；3 条严格相等，h96 退化 |
| validation MSE macro <= fixed U1 | Pass | 0.079681016496 < 0.079681071679，-0.00006926% |
| 改善不是舍入误差 | Fail / 不适用 | 没有 qualifying test improvement；full-precision test delta 是退化 |
| 改善不只由单一 horizon 驱动 | Fail | 唯一发生数值变化的是 h96；有利 validation macro 也完全来自 h96 |

结论固定为 negative-or-negligible development signal，而不是 provisional positive 或 mixed。额外诊断同样不支持升级：best_epoch>0 仅 1/4；只有 h96 best gamma/residual nonzero，其他三条回退 gamma=0；bypass 不稳定优于 Normal；aux permutation 也不呈稳定不利影响。

按预登记停止线登记：

当前 TimeXer-inspired TEB 路线在 M4 有限开发中失败。

本轮不运行 matched-budget continuation，不继续 T4/T5/T6，不调 d/heads/patch/gate/beta，不实现新 TEB，不启动 P2，不进入 M5/M7，也不实现或运行任何空间模块。下一步只等待用户与 ChatGPT 审核后决定更换另一篇近三年外生变量模块来源。M4 状态继续为 In Progress。

### 38.9 文件、测试与 Git 边界

本轮未新增、修改或删除任何 tests/test_*.py；现有 237 项继续全部作为 permanent regression tests。一次性聚合与诊断只位于 /tmp/m4_t2_adapter_rmin_20260901T1547Z/，结果登记完成后按本轮合同删除，不向仓库或 ChatGPT Project 上传。

本轮只修改本 milestone；canonical、M0-M3、main.py、summarizer、模型、runner、测试、配置模板、数据及既有 artifact 均未修改。不执行 git add、commit 或 push，不做 Git closure。最终预期 tracked worktree 只保留本文件的未 stage 修改。

## 39. 第十八轮：CrossLinear 来源锁定与 CCE v1 production capability

### 39.1 起始现场与来源身份

本轮从 clean Git 现场开始：branch `AMD-paper-repro-custom-modules-v1`，local/tracking/live remote 均为 `4cff9cc0a6bdff7cac9366c5ab77f9979b344777`，ahead/behind=`0/0`，index/worktree clean 且无 untracked。不可变 baseline `amd_reproduced_baseline_v1` 仍指向 `fa9665627e6fcfb1d0c2bc22d943ca9666304fd6`。canonical 与本 milestone 修改前 SHA-256 分别为 `fc418a32cf07c8cd9854f65efe2effbbad9137464c39512f0dbccbdce706ed6e`、`756ee986cf1d4f48a8513b7d80d88ddaf01c93f154f545fa974b84962222e331`；M0-M3 SHA 均与冻结值一致。起始 source fingerprint 为 `f3a0a9de6bb3296437202c5a9ba4cebfd88424191232db212f55d151db144a4a`，20 files。

用户已明确锁定 CrossLinear 的 Cross-Correlation Embedding 为 TimeXer 路线停止后的首选替代来源。本轮只使用服务器已有本地来源，不联网下载、不 clone：

| source field | locked value |
|---|---|
| paper_title | `CrossLinear: Plug-and-Play Cross-Correlation Embedding for Time Series Forecasting with Exogenous Variables` |
| venue | `KDD 2025` |
| DOI | `10.1145/3711896.3736899` |
| PDF SHA-256 | `45557c426ca8bfa88f35ec41f09fd87ab864c9a382eef1c659c2296a4a1b0152` |
| official_repo_url | `https://github.com/mumiao2000/CrossLinear.git` |
| official_repo_commit | `d22366e2f59ced560a02b2b1c7cc673e3c02a13f` |
| official_model_sha256 | `a062ac97231c55384c621f27981b8225bb87822f50704df201b381dd8e037593` |
| retained_component | `cross_correlation_embedding_only` |

论文 Eq. (7)-(8) 与官方代码的来源事实被登记为“归一化变量经 `Conv1D` 形成 cross-correlation embedding，再与 endogenous 输入线性混合”；官方实现另含自身 normalization/de-normalization、unbounded alpha/beta、patch embedding、PE 和 forecasting head。本项目不复制官方源码，只独立实现 Cross-Correlation Embedding，并显式删除第二套 normalization、patch、PE 和 head；因此来源边界清晰，不能称为完整 CrossLinear。

### 39.2 CCE v1 锁定数学与插入合同

候选 implementation variant 固定为 `el-amd-m4-crosslinear-cce-v1`。输入是 AMD RevIN 后、MDM 前的 `x_ch [B,C,T]`；插入顺序固定为 `RevIN -> CCE -> MDM -> DDI -> PMCR? -> AMS`，当前 CCE pair 中 PMCR 与全部旧 TEB 固定关闭，不修改 `models/tsAMD.py`、PMCR 或旧 TEB 数学结构。

gate 固定为 `lambda=sigmoid(logit(0.1)+rho)`，`rho` 是初始化严格为 0 的全局共享 scalar Parameter；effective lambda init 严格为 0.1。卷积固定 `kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True` 和 zero padding。可学习量采用 RNG-neutral identity-residual delta：`delta_weight=0`、`delta_bias=0`，不得先随机初始化再覆盖。

target_exogenous 模式的模块输入顺序是 ordered `aux_idx` 后接 `target_idx`：

```text
delta_target = Conv1d([aux...,target], out_channels=1)
output_target = input_target + lambda * delta_target
```

只替换 target，其他 channel 逐元素不变。parallel_multivariate 模式保持 feature schema 原顺序：

```text
delta_all = Conv1d(x_ch, C -> C)
output = x_ch + lambda * delta_all
```

参数量硬合同为 target `3*C_source+2`，parallel `3*C*C+C+1`。分析接口必须公开 effective lambda、ungated delta 与 selector identity 加 learned delta 的等价 CrossLinear kernel。模块内部不得存在 normalization。

target/aux 索引拒绝 bool、重复、越界及 target-in-aux，并保持 aux 明确顺序；target CCE-on 要求 aux 非空，parallel CCE-on 要求 `C>=2`。CCE-off 固定 `self.cce=None`、无 `cce.*` state keys 且严格旁路；F0/`C=1` 不得伪装为 CCE enabled。CCE 路线的 state source 前两段正常承载间接影响，第三段固定为 dtype/device 正确的 `legacy_width_compatibility_zero`，不是 CrossLinear-derived context。

### 39.3 Runner、artifact、checkpoint 与 development 停止合同

新增可追踪字段 `use_cce`、`cce_kernel_size`、`cce_lambda_init`、`cce_padding_policy`、`cce_input_order_policy`、`cce_parameterization_policy`；v1 强制 kernel 3、lambda init 0.1、`zero_same`、mode-specific input order 与 `identity_residual_delta_v1`。论文、仓库、retained component、插入点、mode/order/kernel/padding/bias、lambda transform/raw/effective init、normalization reuse 与 state placeholder 均进入 scientific identity 和 config/checkpoint/manifest；机器绝对参考仓库路径不得进入 comparison identity。

development protocol 固定 `m4_crosslinear_cce_from_scratch_pair_v1`，只允许 `M4_CCE_CONTROL`（CCE/PMCR/TEB off）与 `M4_CCE`（CCE on、PMCR/TEB off）。两者均为 standard from-scratch：全部 AMD/CCE parameters trainable，Adam，lr=`3e-5`，weight_decay=`1e-7`，best 从 epoch 1 开始，不使用 epoch-0 best/source checkpoint/T2 warm-start；同 run resume 只允许 strict same-structure。

artifact 继续 schema-v2、13-file checksum、hidden staging 与 atomic publication，不创建 schema-v3 或第 14 个文件。summarizer 必须区分 standard AMD/U1、TimeXer TEB、T2 adapter/continuation、CCE control 与 CCE candidate，并拒绝 CCE source/config/order/gate/checkpoint tamper 和 duplicate spoof。

从 AMD/U1 state 初始化 CCE-on 模型只能调用专用 importer：`missing` 必须严格等于完整 `cce.*` key set，`unexpected` 必须为空；写参前核验 key/shape/dtype/task mode/C/feature schema/order/target/aux/schema fingerprint，失败时全部 parameter 与 persistent buffer 原子不变。禁止 `strict=False`、partial CCE、任何 `teb.*` 权重复用、target/parallel 跨模式迁移、不同 schema/order/C 迁移及 T2 lineage 冒充 CCE lineage。

本轮只实现 production capability、永久回归合同与一个 ETTm1 h96 真实 single-batch smoke；不运行四个 control 或四个 CCE development run，不创建真实 completed development artifact。CCE 未通过后续 M4 paired adequacy 前继续阻塞新 PMCR/P2 和 M5；不得称其为最终外生模块或最终 EL-AMD，不进入 M7/空间模块，也不执行 Git closure。

### 39.4 Production implementation 与 identity-preserving 证据

独立模块实现位于 `models/modules/cross_correlation_embedding.py`，未复制官方源码。模块直接以 `torch.zeros` 创建 `delta_weight`、`delta_bias` 与 scalar `rho`，没有构造后清零的随机初始化，也没有 normalization、patch、PE 或 forecasting head。`effective_lambda()` 数值实现保持 `sigmoid(logit(0.1)+rho)` 的有界语义，并对 `rho==0` 的 float64 one-ULP round-trip 做 detached 精确值修正，因此 CPU/CUDA、float32/float64 初始化均严格返回各 dtype 的 `0.1`，同时保留 sigmoid 导数；`rho=±1000` 分别返回 0/1，未使用 clamp。

`AMDEnhanced.forward` 的实际插入点已锁定为 `normalized_input -> transpose -> cce -> pastmixing`，即 RevIN 后、MDM 前；`models/tsAMD.py` 未改。target 模式以 one-hot residual scatter 保留直接恒等 autograd 主路径，既保证非 target channels 逐元素不变，也保证零 delta 首次 backward 时公共 AMD 梯度与 paired control 位级相等。parallel 模式保持 schema 原顺序。分析接口为 `effective_lambda()`、`compute_ungated_delta()` 与 `equivalent_crosslinear_kernel()`。

shape/参数/主要 MAC 合同如下：

| mode | internal shape | parameters | leading MAC complexity |
|---|---|---:|---:|
| target_exogenous | `[B,C_source,T] -> [B,1,T]`，外部仍 `[B,C,T]` | `3*C_source+2` | `O(3*B*T*C_source)` |
| parallel_multivariate | `[B,C,T] -> [B,C,T]` | `3*C*C+C+1` | `O(3*B*T*C^2)` |

永久测试实际覆盖 `T=12`、`T=512` 与 `C=321` forward。ECL `C=321` 时 parallel CCE 单模块参数为 309,445，target 使用全部 321 个 source 时为 965；前者的二次 channel 成本须留待获授权的高维实验单独评估。ETTm1 长序列下 k=3 只提供局部滞后，UrbanEV 短序列下端点 zero padding 占比更高；本轮仅证明能力与合同，不声称性能或可扩展性。

### 39.5 梯度、optimizer 与 checkpoint 原子性

固定 synthetic paired probe 使用相同 seed、输入、目标、Adam `lr=3e-5,weight_decay=1e-7`：初始化 prediction、MoE 和总 loss 均 `torch.equal`。第一次 backward 的六个 aux delta elements 为 `6/6` 非零且有限，L2=`0.29671091531916033`；`rho.grad=0`。公共 AMD parameter gradient 为 `41/41` tensors 位级相等，max abs diff=`0`。第一次 optimizer step 后公共 parameter/persistent-buffer state 为 `44/44` tensors 位级相等，max abs diff=`0`；`delta_weight` max move=`3.000000106112566e-05`，`rho` 仍严格为 0，未因任务梯度或 weight decay 漂移。第二次 synthetic backward 的 `rho.grad=-2.1595949874608777e-05`，有限且非零。

专用 `load_cce_source_state_dict` 只接受 source state 的 missing set 严格等于 `{"cce.delta_weight","cce.delta_bias","cce.rho"}` 且 unexpected 为空；source contract 完整绑定 mode、C、feature schema、target/aux order、schema fingerprint 与 input-order policy。永久测试覆盖合法 AMD/U1-style import、partial CCE、额外 key、缺 key、shape、dtype、mode、schema、target/aux/order mismatch；每个失败案例均核验全部 parameter 与 persistent buffer 未变化。同结构 CCE checkpoint 只允许 `strict=True`，`strict=False` 被拒绝。

新增 CCE 会改变 `models/tsAMD_enhanced.py` 与 `models/modules/__init__.py` 的整文件 SHA；因此第十四轮历史 U1 artifact 在当前 live source 下继续被既有 full-file compatibility proof 严格拒绝，不被伪装为可复用来源。永久测试显式保留此生产拒绝，同时用隔离的 compatibility fixture 继续验证历史 T2 adapter/continuation 的 state mapping、冻结范围与 epoch-0 机制。旧 completed artifacts 的 summarizer 读取合同不变。

### 39.6 Runner、schema-v2 artifact 与 summarizer 回归

production runner 已新增 variant `el-amd-m4-crosslinear-cce-v1`、development protocol `m4_crosslinear_cce_from_scratch_pair_v1` 及 `M4_CCE_CONTROL`/`M4_CCE` 两个且仅两个 ablation。开关、固定 kernel/gate/padding/order/parameterization、CrossLinear 论文与官方仓库身份、retained component、插入点、normalization reuse 和 state zero-placeholder 均进入 scientific config，并由 config、best/last checkpoint 与 manifest/candidate contract 交叉封存；机器绝对 CrossLinear 路径不进入 scientific/comparison identity。control 无 `cce.*` keys，candidate 恰有三个完整 CCE keys；两者 standard best 均从 epoch 1 开始。

永久 runner fixture 在 `TemporaryDirectory` 中生成并自动回收 CCE control/candidate schema-v2 synthetic artifacts：每个 artifact 仍为 13 个 checksummed payload files，Python exact-set/digest 与系统 `sha256sum -c` 均通过，hidden staging/atomic publication 合同未变，没有 schema-v3/第 14 个文件。summarizer 成功把 CCE control、CCE candidate 与旧 standard/TimeXer/warm-start 身份分开，并拒绝 CrossLinear source、input order、gate、switch、checkpoint dtype 及 duplicate scientific identity + seed spoof。所有 synthetic artifact 已随临时目录删除，仓库 `artifacts/` 下没有 CCE variant 目录。

### 39.7 真实 ETTm1 h96 single-batch smoke

只通过 production `prepare_args -> _build_runtime_data -> _validate_loader_contract -> _build_model -> _prediction_for_loss` 取得一个真实 train batch；没有调用完整 runner lifecycle，没有创建 optimizer、执行 optimizer step、遍历 validation/test loader 或写 artifact。实际设备为 `cuda:0`，数据 SHA-256=`6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e`，schema fingerprint=`f6dd94841b5d9d0b7515b19e0ff1876bf6476068054eacdc02ac6fcab3f084dc`。

| probe | observed |
|---|---|
| input / target / prediction | `[4,512,7]` / `[4,96,1]` / `[4,96,1]` |
| state_source | `[4,1056]`；最后 32 维为 zero placeholder |
| paired initialization | prediction/MoE/state_source 均 `torch.equal=True` |
| production objective | `6.7545485496521`，finite |
| six-aux delta gradient | L2=`0.09123487410321596`，18/18 nonzero，finite |
| first `rho.grad` | `0.0` exactly |
| effective lambda init | float32 `0.10000000149011612`，即该 dtype 的精确 0.1 |
| artifact/optimizer | artifact root 不存在；optimizer 未创建；step 未执行 |

该 smoke 没有读取 development validation/test 指标，不是 CCE performance evidence。

### 39.8 永久测试、最终 source identity 与停止点

新增 `tests/test_cross_correlation_embedding.py`，覆盖模块数学、CPU float32/float64、CUDA float32、CPU/全部 CUDA RNG-neutral、identity output、target 首/中/末位置、有序 aux、非法索引、非 target 不变、parallel shape、zero padding、参数量、首/次轮梯度、paired optimizer exactness、无内部 normalization、off-path 与专用 importer 原子性。`tests/test_runner.py` 增加 CCE runner/artifact/summarizer、parallel identity、tamper/duplicate 及历史 source live-gate 回归；未删除或弱化任何旧测试，`tests/test_tsAMD_enhanced.py`、`tests/test_public_architecture.py`、`tests/test_summarize_results.py` 未修改。

最终完整 discovery：`251/251 passed`，failed=`0`，skipped=`0`，用时 `26.324 s`；CUDA CCE/T2/T2G/T3、PMCR 与 M0 parity 分支均实际执行。新 executable source fingerprint 为 `d6e2dd7fe51994dc91f9bad44a692426636518aa1f1f9109db11d5277ac8892a`，21 files；新增的 source file 是 `models/modules/cross_correlation_embedding.py`。

本轮没有运行四个 `M4_CCE_CONTROL` 或四个 `M4_CCE` 真实 development runs，没有创建 completed/failed/staging CCE development artifact，没有训练完整 epoch，没有修改或删除既有 artifact/cache/log。没有启动 PMCR/P2，没有进入 M5/M7 或任何空间模块，M4 继续 `In Progress`。除本轮允许的 canonical 与 M4 外，其他文档、M0-M3、`models/tsAMD.py`、旧 TEB、PMCR、DataLoader、数据与 artifact 均未修改。所有改动保持未 stage、未 commit、未 push，等待用户与 ChatGPT 审核并锁定后续八 run 合同；本轮不执行 Git closure。

## 40. 第十九轮：CrossLinear-inspired CCE v1 四 horizon paired development

### 40.1 起始现场、回归与公平初始化

本轮从已完成第十八轮 closure 的 clean 现场开始：branch `AMD-paper-repro-custom-modules-v1`，local/tracking/live remote 均为 `e3ef3a454f6c2b0b602d5f310c231126fe545757`，ahead/behind=`0/0`，worktree/index clean 且无 untracked。不可变 baseline `amd_reproduced_baseline_v1` 仍指向 `fa9665627e6fcfb1d0c2bc22d943ca9666304fd6`。canonical 与本 milestone 修改前 SHA-256 分别为 `3109bf54b8946520ee7c93c32cdabc320aa7c69acd848300393cf724a8e6da17`、`85af167f2bb9570bbcd90bad930882e571f0698766fbfca929c7990ef74971d9`。executable source fingerprint 为 `d6e2dd7fe51994dc91f9bad44a692426636518aa1f1f9109db11d5277ac8892a`（21 files，`sha256_length_prefixed_relative_path_and_content_v1`）。新 artifact root `artifacts/m4-development/ettm1-stage-g-crosslinear-cce-v1` 启动前不存在。

运行前完整 discovery 结果为 `251/251 passed`，failed=`0`，skipped=`0`，unittest 用时 `24.930 s`。CUDA 分支实际执行，只出现既知 CuDNN `nvrtc.so` workaround warning，不影响通过判定。

一次性 `/tmp` 探针对 h96/192/336/720 分别重建 matched control/candidate 初始化，观测如下：

- 公共 AMD parameter 57 keys 与 persistent buffer 3 keys 的 key/shape/dtype/value 全部严格相等，max abs diff=`0`；
- 同一 train generator 初态严格相等，digest=`88dc2922f3b13c33260a1f2064b3a551422700e416cfc14230f22a0b5f849e56`；取出首个 train batch 后 generator state 仍成对相等；
- 首个 train/validation/test batch 逐元素相等；输入为 `[128,512,7]`，target 为 `[128,H,1]`；
- CCE 构造不消费 CPU 或任何 CUDA RNG，paired construction 后 CPU/CUDA RNG state 相等；
- 初始 prediction、MoE loss 与 `[128,1056]` state_source 均 `torch.equal=True`；
- candidate 只多出完整 `cce.delta_weight`、`cce.delta_bias`、`cce.rho` 三个 state keys，control 无任何 `cce.*` key。

该探针未写入仓库，临时目录已删除。

### 40.2 锁定运行合同与 production 命令证据

八个 run 均使用 `implementation_variant=el-amd-m4-crosslinear-cce-v1`、`development_protocol_id=m4_crosslinear_cce_from_scratch_pair_v1`、ETTm1/MS/OT、`target_exogenous`、`target_idx=6`、ordered `aux_idx=0,1,2,3,4,5`、`seq_len=512`、seed 2024、official fold、10 epochs、batch 128、Adam `lr=3e-5,weight_decay=1e-7`。AMD 固定 `n_block=1,alpha=0,mix_layer_num=3,mix_layer_scale=2,patch=16,norm=true,layernorm=true,dropout=0.1`；PMCR 和全部 TimeXer TEB 关闭，全部参数 from scratch 联合训练，best 从 epoch 1 开始，无 source/warm-start/adapter/continuation/epoch-0 best。

Candidate 固定 kernel 3、lambda init 0.1、`zero_same`、`ordered_aux_then_target`、`identity_residual_delta_v1`；control 只将 `use_cce=false`，candidate 为 `use_cce=true`。数据 SHA-256 为 `6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e`，feature/schema fingerprint 为 `f6dd94841b5d9d0b7515b19e0ff1876bf6476068054eacdc02ac6fcab3f084dc`。

实际顺序为 `control h96 -> CCE h96 -> control h192 -> CCE h192 -> control h336 -> CCE h336 -> control h720 -> CCE h720`。每个 artifact 的 `command.txt` 均由 production runner 在训练前写入该 run 的完整命令，包含全部上述参数；本轮未手工补写、未事后重建命令证据。

### 40.3 八个不可变 artifact

下表路径均相对 `artifacts/m4-development/ettm1-stage-g-crosslinear-cce-v1/el-amd-m4-crosslinear-cce-v1/ETTm1/target_exogenous/OT/`。`config/scientific` 列为封存的 config/scientific identity hash。

| H | arm | run_id / relative artifact path | config/scientific hash | comparison hash | best epoch | best.pt SHA-256 | last.pt SHA-256 | wall-clock (s) | bytes |
|---:|---|---|---|---|---:|---|---|---:|---:|
| 96 | control | `20260903T133823.128197Z-1a2c2339` / `horizon_96/fold_official/seed_2024/20260903T133823.128197Z-1a2c2339` | `03cf18690da9675799fc50d0f5f5f749658037cb8dadd7e340dd405e4c9885af` | `a293995ae65f65179258948f64dc2b5d99a32e046b8942da3c018f30fd84f5bc` | 7 | `466949f4087daeb3f373f6d09a4e7934cbcfd9bece8f63aedf111a38d1f8160e` | `53cbd89a4da7856488a78308cec20e056cdc3d15bc09c3b2f0f204594a7cc962` | 173.041576 | 205075922 |
| 96 | CCE | `20260903T134158.187683Z-251f979a` / `horizon_96/fold_official/seed_2024/20260903T134158.187683Z-251f979a` | `d2a4ceb4cc3252105f9fbff6add2780db09450c74dc495cce9a222ac324cc839` | `d3779c51cc04b35cdb2bcb592a6d458c13eeb8105e5966bdc5b20851d219a63b` | 7 | `513cb81d8b6d0939391aca2398cdd41b27d370d2e638fd5fb3b730dd51326dcd` | `5bbb4d8e958a27eee366b2f71c982b92f7ab56b03a7f7223998d47ce454b2123` | 168.102194 | 205082136 |
| 192 | control | `20260903T134526.964084Z-7aa2f273` / `horizon_192/fold_official/seed_2024/20260903T134526.964084Z-7aa2f273` | `b73d1d2a69ece422436d56386571b9a9e967cdee0e0b9e2a2b54432227b57337` | `0e728438c556964a33746521d71d4317993f9904f16c4302a5dbf1d7d2d78584` | 3 | `74130c8e855fd2e86dae6285bbaa5c1b9d9b8fa4c9122f1c69b00affbd5d6244` | `8cb43b14aa1dcf1684e5f1c73befb186047daade8bc75849ed61738dade8d3b4` | 154.012600 | 236548574 |
| 192 | CCE | `20260903T134835.750261Z-63442539` / `horizon_192/fold_official/seed_2024/20260903T134835.750261Z-63442539` | `80946e4a36c31fca955ce6189d789f9dbc0391539222342804f525fef9bafc6e` | `8541230ed83663f544d81ccca09040f2ccbb82fdcaa975ab65b6a281ba401abc` | 3 | `75f0f95931b563543fab72641829f6cc1e5f8f730b9ce05bc8b79217f40b3880` | `b96ea9e17de14c6af496e73c0918f571296b3c4257f072fd3f44417dbed6cd65` | 163.212772 | 236554779 |
| 336 | control | `20260903T135155.761182Z-3aa72c9d` / `horizon_336/fold_official/seed_2024/20260903T135155.761182Z-3aa72c9d` | `6183b85d768f5eb14bb82e15f43644c2cb5a5c1b521ec36590753edbe4d924da` | `464096b2a683bf1c921d541e101a5639ea39293120954c21ed7548c16452e28c` | 3 | `9821f8b8ea90cd250fb841623d2ca25e088c28676adc873f0fd8f49cb0639544` | `770902dfe83c5f8274636af201ddb5a1d402b1dd9d6d7e34d292ff76245a2cff` | 159.916965 | 283757729 |
| 336 | CCE | `20260903T135522.029340Z-01bc74eb` / `horizon_336/fold_official/seed_2024/20260903T135522.029340Z-01bc74eb` | `d53d0e489747a0b01fd9f671ac8d0c9fd5e601fa090ee9cc6df5df81477e4605` | `86803785c79c1b6307d73f95a4fbfde68ae70287a922fc9829503c0c5013d1c6` | 3 | `58f5a7c01d0288a4de6e91e3f8080c27253cb440350de47231639ca6ea7c5e12` | `f5f951017ddf77f34099a8bcd51129df2ca8f5515b610969f99f6d6333458e19` | 161.982626 | 283763940 |
| 720 | control | `20260903T135837.073075Z-b19c7ca0` / `horizon_720/fold_official/seed_2024/20260903T135837.073075Z-b19c7ca0` | `c31a46fcffae2956e6ec68f8b7108218fe79221d22da43de3f4b63c38e4e132c` | `6d827d903bb086f19013605cbfea97def91a6b80b2ebe8ee59b262790c244f93` | 7 | `bf4f778fce358019ed9d91f4476e70d0468bee3ff980259c4b64a365addb4e9c` | `933cedacd9085c0f8fd65aff158a5f3798061481f2e60beb3d53573f9515c90d` | 173.113178 | 409648272 |
| 720 | CCE | `20260903T140215.465903Z-0f57e1eb` / `horizon_720/fold_official/seed_2024/20260903T140215.465903Z-0f57e1eb` | `3049a941fa630356e65b9545a54afcacff53b8a11b777acae97beea1d9c0b708` | `bd0fbe29e25f53424edbf1150681f8302f7b46d80e87767b2d57e31cf97fac84` | 7 | `0b9ed2507c8a678aeacaf2cf8dee9339736b7733630d0e5e5870f1d38b04bb85` | `69642ec9b1c54c35a1781095426b90aa6f1706a5288377e168bbf1cd38c0de0f` | 163.870678 | 409654475 |

八个 run 合计 artifact size=`2270085827` bytes（约 2.114 GiB），artifact-recorded wall-clock 合计 `1317.252590 s`。逐 run 核验均为status completed、schema-v2/schema-version 1、history 恰为 epochs 1--10、best/last checkpoint 存在、metrics 全 finite、source/data/schema/config/checkpoint/manifest/path identity 一致。每个 artifact 的 13 个 checksummed payload 均通过 Python exact-set/digest 和系统 `sha256sum -c`。正式 summarizer 接受八个 artifact；最终 root 恰有 8 个 completed manifest、8 个唯一 scientific/comparison identity + seed，无 failed/running/hidden staging/重复 completed identity。

### 40.4 四项 matched-control 核心指标

下表的 relative change 均为 `100*(CCE/control-1)`，负值表示 CCE 改善。Macro 行的 relative 是 `relative_change_of_macro_means`；每表后另列四个 horizon relative 的算术平均 `mean_horizon_relative_change`。ETTm1 只是 development-only，本节不是论文正式结果。

Validation MSE：

| horizon | control | CCE | relative change |
|---:|---:|---:|---:|
| 96 | 0.051164323931 | 0.051283546439 | +0.233019% |
| 192 | 0.073506087241 | 0.073552612337 | +0.063294% |
| 336 | 0.090008565565 | 0.090040862991 | +0.035883% |
| 720 | 0.104045309980 | 0.104191630054 | +0.140631% |
| macro | 0.079681071679 | 0.079767162955 | +0.108045% |

`mean_horizon_relative_change=+0.118207%`。

Validation MAE：

| horizon | control | CCE | relative change |
|---:|---:|---:|---:|
| 96 | 0.168072960404 | 0.168290660773 | +0.129527% |
| 192 | 0.207844046975 | 0.207929689125 | +0.041205% |
| 336 | 0.235085840964 | 0.235146953912 | +0.025996% |
| 720 | 0.255146362864 | 0.255406785269 | +0.102068% |
| macro | 0.216537302802 | 0.216693522270 | +0.072144% |

`mean_horizon_relative_change=+0.074699%`。

Development test MSE：

| horizon | control | CCE | relative change |
|---:|---:|---:|---:|
| 96 | 0.027597208934 | 0.027626573597 | +0.106404% |
| 192 | 0.041161235376 | 0.041177171984 | +0.038718% |
| 336 | 0.052610279493 | 0.052622570281 | +0.023362% |
| 720 | 0.070352296784 | 0.070392896863 | +0.057710% |
| macro | 0.047930255147 | 0.047954803181 | +0.051216% |

`mean_horizon_relative_change=+0.056548%`。

Development test MAE：

| horizon | control | CCE | relative change |
|---:|---:|---:|---:|
| 96 | 0.126248707310 | 0.126295513195 | +0.037074% |
| 192 | 0.154727269120 | 0.154756899016 | +0.019150% |
| 336 | 0.174313073967 | 0.174332537814 | +0.011166% |
| 720 | 0.200916141056 | 0.200824188841 | -0.045766% |
| macro | 0.164051297863 | 0.164052284716 | +0.000602% |

`mean_horizon_relative_change=+0.005406%`。

### 40.5 Best epoch、last-vs-best 与最后三轮 validation

| H | arm | best epoch | last validation MSE / MAE（vs best） | last test MSE / MAE（vs best） | validation epoch 8; 9; 10（MSE/MAE） |
|---:|---|---:|---|---|---|
| 96 | control | 7 | 0.052132270 / 0.169383497 (+1.891837% / +0.779742%) | 0.027444374 / 0.125766111 (-0.553807% / -0.382258%) | 0.051913878/0.168732173; 0.051537745/0.168756223; 0.052132270/0.169383497 |
| 96 | CCE | 7 | 0.052323292 / 0.169725407 (+2.027446% / +0.852541%) | 0.027492825 / 0.125836504 (-0.484132% / -0.363441%) | 0.052056760/0.168977472; 0.051706009/0.169043040; 0.052323292/0.169725407 |
| 192 | control | 3 | 0.076731016 / 0.211956744 (+4.387295% / +1.978742%) | 0.041332146 / 0.154918449 (+0.415223% / +0.123559%) | 0.075780676/0.210342253; 0.076200475/0.210927529; 0.076731016/0.211956744 |
| 192 | CCE | 3 | 0.076972072 / 0.212439877 (+4.648999% / +2.169093%) | 0.041418647 / 0.155032386 (+0.586429% / +0.178013%) | 0.075947756/0.210684890; 0.076406216/0.211350058; 0.076972072/0.212439877 |
| 336 | control | 3 | 0.092389603 / 0.238162101 (+2.645345% / +1.308569%) | 0.053267822 / 0.174872425 (+1.249837% / +0.320889%) | 0.092671425/0.238073279; 0.091376269/0.237454255; 0.092389603/0.238162101 |
| 336 | CCE | 3 | 0.092610712 / 0.238654454 (+2.854092% / +1.491620%) | 0.053335988 / 0.174943059 (+1.355725% / +0.350205%) | 0.092805513/0.238408623; 0.091545118/0.237844036; 0.092610712/0.238654454 |
| 720 | control | 7 | 0.104822747 / 0.256006517 (+0.747210% / +0.337122%) | 0.071411360 / 0.202188040 (+1.505371% / +0.633050%) | 0.105413646/0.256573481; 0.104663433/0.255823048; 0.104822747/0.256006517 |
| 720 | CCE | 7 | 0.105077024 / 0.256495044 (+0.849774% / +0.426088%) | 0.071514140 / 0.202117619 (+1.592835% / +0.644061%) | 0.105572991/0.256882364; 0.104792675/0.256153490; 0.105077024/0.256495044 |

四组 pair 的 best epoch 完全相同：h96/720 为 7，h192/336 为 3。这不改变 CCE 四个 test MSE 均退化的主结果。

### 40.6 Gate、delta 参数与 best-checkpoint kernel

初始时 `rho=0`、effective lambda 严格为对应 dtype 的 0.1，delta weight/bias 严格为零。训练后全部 gate 和 delta 均 finite/nonzero；下表的 move 是 best 到 last 的 L2 距离，不是相对初始的移动（相对零初始的 weight/bias 移动就是各 checkpoint 自身 L2 norm）。

| H | checkpoint | rho | effective lambda | delta_weight L2 | delta_bias L2 | best-to-last weight / bias / rho move |
|---:|---|---:|---:|---:|---:|---|
| 96 | best | 0.032534186 | 0.102966405 | 0.067019924 | 4.194981e-05 | 0.023677606 / 6.064736e-06 / 0.010063868 |
| 96 | last | 0.042598054 | 0.103899673 | 0.090420600 | 4.801455e-05 | same row-pair |
| 192 | best | 0.015966600 | 0.101446196 | 0.032121689 | 2.144902e-05 | 0.076175663 / 2.544612e-05 / 0.034451818 |
| 192 | last | 0.050418418 | 0.104630038 | 0.107501626 | 4.689514e-05 | same row-pair |
| 336 | best | 0.020402620 | 0.101851277 | 0.040165508 | 1.428153e-05 | 0.086956662 / 4.523675e-05 / 0.037766462 |
| 336 | last | 0.058169082 | 0.105358377 | 0.125960699 | 3.095523e-05 | same row-pair |
| 720 | best | 0.059833311 | 0.105515353 | 0.135291184 | 7.501144e-05 | 0.047742540 / 8.870560e-05 / 0.017966822 |
| 720 | last | 0.077800132 | 0.107223138 | 0.182682856 | 1.369416e-05 | same row-pair |

下表报告 best checkpoint 下 module input order `HUFL,HULL,MUFL,MULL,LUFL,LULL,OT`的 raw `delta_weight` 及真正等价 CrossLinear kernel `selector_identity + lambda*delta_weight`，每个向量顺序为 lag `[-1,0,+1]`；显示值四舍五入到 8 位小数，checkpoint 保留完整精度。

| H | source | raw delta kernel [-1,0,+1] | equivalent kernel [-1,0,+1] |
|---:|---|---|---|
| 96 | HUFL | `[-0.01885078,-0.01816544,-0.01698866]` | `[-0.00194100,-0.00187043,-0.00174926]` |
| 96 | HULL | `[-0.01134587,-0.01104496,-0.01335085]` | `[-0.00116824,-0.00113726,-0.00137469]` |
| 96 | MUFL | `[-0.01777768,-0.01725085,-0.01570902]` | `[-0.00183050,-0.00177626,-0.00161750]` |
| 96 | MULL | `[-0.01091791,-0.01028410,-0.01225491]` | `[-0.00112418,-0.00105892,-0.00126184]` |
| 96 | LUFL | `[-0.00728045,-0.00659963,-0.00676955]` | `[-0.00074964,-0.00067954,-0.00069704]` |
| 96 | LULL | `[-0.01351823,-0.01318184,-0.01459754]` | `[-0.00139192,-0.00135729,-0.00150306]` |
| 96 | OT target | `[-0.00956228,-0.03176258,+0.00639698]` | `[-0.00098459,+0.99672952,+0.00065867]` |
| 192 | HUFL | `[-0.00728721,-0.00686494,-0.00661619]` | `[-0.00073926,-0.00069642,-0.00067119]` |
| 192 | HULL | `[-0.00860049,-0.00816142,-0.00812643]` | `[-0.00087249,-0.00082794,-0.00082440]` |
| 192 | MUFL | `[-0.00710761,-0.00680197,-0.00653906]` | `[-0.00072104,-0.00069003,-0.00066336]` |
| 192 | MULL | `[-0.00860567,-0.00833848,-0.00873473]` | `[-0.00087301,-0.00084591,-0.00088611]` |
| 192 | LUFL | `[-0.00124249,-0.00171027,-0.00146198]` | `[-0.00012605,-0.00017350,-0.00014831]` |
| 192 | LULL | `[-0.00469612,-0.00503788,-0.00509386]` | `[-0.00047640,-0.00051107,-0.00051675]` |
| 192 | OT target | `[-0.00196211,-0.01531092,+0.00186898]` | `[-0.00019905,+0.99844677,+0.00018960]` |
| 336 | HUFL | `[-0.00827739,-0.00868889,-0.00863621]` | `[-0.00084306,-0.00088497,-0.00087961]` |
| 336 | HULL | `[-0.01086003,-0.01038908,-0.00985292]` | `[-0.00110611,-0.00105814,-0.00100353]` |
| 336 | MUFL | `[-0.00890723,-0.00924897,-0.00914004]` | `[-0.00090721,-0.00094202,-0.00093092]` |
| 336 | MULL | `[-0.01146283,-0.01120895,-0.01069017]` | `[-0.00116750,-0.00114165,-0.00108881]` |
| 336 | LUFL | `[-0.00224254,-0.00231321,-0.00276190]` | `[-0.00022841,-0.00023560,-0.00028130]` |
| 336 | LULL | `[-0.00394334,-0.00374363,-0.00438094]` | `[-0.00040163,-0.00038129,-0.00044620]` |
| 336 | OT target | `[+0.00057239,-0.01953362,+0.00182325]` | `[+0.00005830,+0.99801048,+0.00018570]` |
| 720 | HUFL | `[-0.03202698,-0.03238045,-0.03273229]` | `[-0.00337934,-0.00341664,-0.00345376]` |
| 720 | HULL | `[-0.03523847,-0.03493602,-0.03469572]` | `[-0.00371820,-0.00368629,-0.00366093]` |
| 720 | MUFL | `[-0.03412665,-0.03453387,-0.03526797]` | `[-0.00360089,-0.00364385,-0.00372131]` |
| 720 | MULL | `[-0.03423008,-0.03404135,-0.03412889]` | `[-0.00361180,-0.00359189,-0.00360112]` |
| 720 | LUFL | `[-0.01317689,-0.01255784,-0.01167291]` | `[-0.00139036,-0.00132505,-0.00123167]` |
| 720 | LULL | `[-0.01164375,-0.01077665,-0.01052970]` | `[-0.00122859,-0.00113710,-0.00111104]` |
| 720 | OT target | `[+0.00175090,-0.05950332,+0.00466761]` | `[+0.00018475,+0.99372149,+0.00049250]` |

### 40.7 CCE 写回幅度

在 candidate best checkpoint 下，对每个 sample 计算 `||lambda*delta||_2/(||x_ch||_2+1e-12)`；这是 RevIN 后表示上的 target-channel CCE 写回与全 channel 输入的 L2 比率。

| H | split | mean | median | p10 | p90 | p99 | max |
|---:|---|---:|---:|---:|---:|---:|---:|
| 96 | validation | 0.004616929 | 0.004696637 | 0.003345302 | 0.005702361 | 0.006674247 | 0.007482079 |
| 96 | test | 0.005285786 | 0.005538117 | 0.003407837 | 0.006278925 | 0.007015974 | 0.007322467 |
| 192 | validation | 0.002255430 | 0.002279694 | 0.001635735 | 0.002798170 | 0.003143115 | 0.003518216 |
| 192 | test | 0.002584890 | 0.002757714 | 0.001737921 | 0.003031810 | 0.003364259 | 0.003411890 |
| 336 | validation | 0.002786764 | 0.002794165 | 0.002019790 | 0.003513739 | 0.003901419 | 0.004360912 |
| 336 | test | 0.003289349 | 0.003516489 | 0.002177545 | 0.003822384 | 0.004213944 | 0.004257409 |
| 720 | validation | 0.009796953 | 0.009816233 | 0.007128801 | 0.012288994 | 0.014076245 | 0.015739123 |
| 720 | test | 0.012010297 | 0.012894569 | 0.007821717 | 0.014006642 | 0.015336035 | 0.015449849 |

写回全部 finite/nonzero，但其幅度小不能被自动解释为有效外生利用。

### 40.8 同 checkpoint bypass 与 auxiliary permutation

Bypass 在同一 candidate best checkpoint 上只禁用 CCE 写回，其他 state 不变。relative 为 bypass 相对 normal，负值表示 bypass 更好。

| H | split | normal MSE / MAE | bypass MSE / MAE | bypass relative MSE / MAE | prediction max change |
|---:|---|---|---|---|---:|
| 96 | validation | 0.051283546 / 0.168290661 | 0.051192720 / 0.168101686 | -0.177106% / -0.112291% | 0.012485623 |
| 96 | test | 0.027626574 / 0.126295513 | 0.027616723 / 0.126246052 | -0.035656% / -0.039163% | 0.011724710 |
| 192 | validation | 0.073552612 / 0.207929689 | 0.073524943 / 0.207853673 | -0.037619% / -0.036558% | 0.011008143 |
| 192 | test | 0.041177172 / 0.154756899 | 0.041174993 / 0.154731566 | -0.005293% / -0.016369% | 0.009085178 |
| 336 | validation | 0.090040863 / 0.235146954 | 0.090064931 / 0.235130720 | +0.026730% / -0.006904% | 0.017675996 |
| 336 | test | 0.052622570 / 0.174332538 | 0.052635459 / 0.174320107 | +0.024494% / -0.007130% | 0.009994745 |
| 720 | validation | 0.104191630 / 0.255406785 | 0.104885272 / 0.256038988 | +0.665736% / +0.247528% | 0.031915128 |
| 720 | test | 0.070392897 / 0.200824189 | 0.070606880 / 0.200994510 | +0.303984% / +0.084811% | 0.030093670 |

h96/192 的 bypass 在 validation/test 四项上均更好，h336 混合，h720 normal 更好。因此 bypass 并未呈现稳定一致方向。

Auxiliary permutation 保持每个 batch 的 target 不变，六个 aux 整体分别使用 batch 维 cyclic shift `1,floor(B/3),floor(2B/3)`。下表报告三次指标均值、相对 normal 变化、三次范围和对 normal prediction 的最大变化。

| H | split | perm mean MSE / MAE | relative MSE / MAE | three-shift MSE range | three-shift MAE range | prediction max change |
|---:|---|---|---|---|---|---:|
| 96 | validation | 0.051283113 / 0.168283573 | -0.000845% / -0.004212% | [0.051264922,0.051301522] | [0.168263038,0.168297473] | 0.017612338 |
| 96 | test | 0.027618629 / 0.126260097 | -0.028757% / -0.028042% | [0.027608163,0.027627437] | [0.126229971,0.126299536] | 0.017690420 |
| 192 | validation | 0.073573147 / 0.207944887 | +0.027918% / +0.007309% | [0.073552006,0.073597251] | [0.207926901,0.207980766] | 0.013253212 |
| 192 | test | 0.041178939 / 0.154736067 | +0.004292% / -0.013461% | [0.041176238,0.041183667] | [0.154718754,0.154756356] | 0.010203004 |
| 336 | validation | 0.090097932 / 0.235201227 | +0.063382% / +0.023080% | [0.090040491,0.090137934] | [0.235144726,0.235256968] | 0.018750906 |
| 336 | test | 0.052635421 / 0.174321487 | +0.024420% / -0.006339% | [0.052621803,0.052652138] | [0.174302037,0.174331298] | 0.014073253 |
| 720 | validation | 0.104836113 / 0.256060361 | +0.618555% / +0.255896% | [0.104201212,0.105155623] | [0.255415675,0.256402535] | 0.053235531 |
| 720 | test | 0.070560511 / 0.200944534 | +0.238113% / +0.059926% | [0.070377569,0.070805123] | [0.200802417,0.201223698] | 0.052888751 |

Permutation 对 h96 略有利、h192/336 混合、h720 明显不利，没有跨四 horizon 的稳定不利影响。Bypass 和 permutation 都只是同-checkpoint 描述性诊断，不是独立训练消融，也不是因果证明。

### 40.9 Diagnostic wrapper parity 与 state 不变性

对四个 candidate best checkpoint 的全部 validation/test batches，formal `AMDEnhanced.forward` 与用相同 production modules 展开的诊断 wrapper 对 prediction、MoE loss、state_source 的 max abs difference 全部为 `0`，满足原有 exact/1e-6 门禁。诊断前后 model `state_dict` key/shape/dtype/value 逐 tensor 严格相等，normal forward 在诊断前后的 prediction max difference 为 `0`。诊断未改写 best/last checkpoint、manifest 或 checksums。

### 40.10 预登记 adequacy 裁决

| positive development signal 硬条件 | evidence | result |
|---|---|---|
| test MSE macro < matched control | 0.047954803181 > 0.047930255147，+0.051216% | **Fail** |
| test MAE macro <= matched control | 0.164052284716 > 0.164051297863，+0.000602% | **Fail** |
| 至少 3/4 horizon test MSE 改善 | 0/4 改善，4/4 退化 | **Fail** |
| validation MSE macro <= matched control | 0.079767162955 > 0.079681071679，+0.108045% | **Fail** |
| 改善超过数值舍入噪声 | 不存在 macro/test-MSE 改善 | **Fail** |
| 改善不只由单一 horizon 驱动 | 不存在可用的总体改善；仅 h720 test MAE 改善 | **Fail** |

因六项硬条件全部失败，本轮分类为 **negative-or-negligible development signal**，不登记 CCE 通过 M4 development adequacy gate，也不把 CCE 升级为最终外生模块或最终 EL-AMD。Gate/delta finite/nonzero、h720 normal 在 bypass 诊断中占优，均不能推翻 matched-control 主结果。

按预登记停止边界，本轮不调 kernel、lambda、层数、epoch、学习率或其他超参，不自动转向 Sonnet/XLinear 或启动新候选。没有启动 PMCR/P2，没有进入 M5/M7，没有实现或运行空间模块。M4 状态继续为 **In Progress**，下一步模型来源/结构决策等待用户与 ChatGPT 审核。

### 40.11 文档、测试、artifact 与 Git 边界

canonical 只修正执行授权边界：用户已锁定科学合同且前置 implementation closure 完成后，ChatGPT 可直接下发合同内普通实现、测试和预登记实验；只有改变结构、来源、超参数、预算、停止线、test 边界，启动新候选或进入下一 milestone 时需要用户重新决定。本修正不改变 CCE 科学、模型或实验合同。

本轮没有新增、修改或删除任何测试文件；251 项现有测试继续全部作为 permanent regression tests。没有修改 Python 模型/runner/summarizer、DataLoader、`models/tsAMD.py`、PMCR、旧 TEB、数据或任何既有 artifact；本轮只在启动前不存在的固定 root 创建了上述八个不可变 development artifacts。一次性初始化/聚合/诊断内容只在 `/tmp`，登记完成后删除。

本轮只保留 canonical 与本唯一 M4 milestone 的未 stage 文档修改；M0-M3 未修改，无其他 tracked/untracked 变化。本轮不执行 `git add`、commit、push 或 Git closure。


## 41. 第二十轮：Late CCE 只读审计与 production capability

### 41.1 起始现场与来源锚点

本轮从 clean closure 现场启动：branch `AMD-paper-repro-custom-modules-v1`，local/tracking/live remote 均为 `26f285d8e4dc0b9f250584cadefc906fb5abf006`，ahead/behind `0/0`，index/worktree/untracked 均为空；baseline tag `amd_reproduced_baseline_v1` 仍指向 `fa9665627e6fcfb1d0c2bc22d943ca9666304fd6`。canonical/M4 起始 SHA-256 分别为 `ebb11be3032eef74a3d5118a1e9f719460d19f6f38f5cafb084b022e482b9701` 与 `18d70b0fa20dc0e8da8e058e962c7a6362c12ff6d014cda9adaaa45eb65c2c19`；起始 executable source fingerprint 为 21-file `d6e2dd7fe51994dc91f9bad44a692426636518aa1f1f9109db11d5277ac8892a`。M0--M3 SHA、UrbanEV/ModernTCN/TimeXer/CrossLinear clean 状态均与冻结现场一致。

CrossLinear 来源锚点复核为：KDD 2025 论文 PDF SHA-256 `45557c426ca8bfa88f35ec41f09fd87ab864c9a382eef1c659c2296a4a1b0152`；服务器官方仓库 commit `d22366e2f59ced560a02b2b1c7cc673e3c02a13f`，`models/CrossLinear.py` SHA-256 `a062ac97231c55384c621f27981b8225bb87822f50704df201b381dd8e037593`。`docs/archive` 未作为决策依据。

### 41.2 只读审计结论与锁定边界

对 production 依赖图的逐函数审计确认：Early CCE 在 `AMDEnhanced.forward` 的 RevIN transpose 后、`pastmixing` 前写回 `x_ch`，因此同时影响 MDM、DDI、AMS experts 与 selector；Late CCE 在 MDM/DDI/可选 PMCR 得到 `v_local` 后写回 `v_final`，AMS 固定调用 `moe(v_final,u_mdm)`。因此 Late 只改变 experts input，`x_ch`、`u_mdm`、`v_ddi` 与 selector input 均保持原路径。`state_source` 第一段来自 `v_final` target，第二段来自 `u_mdm` target，第三段仍是 deterministic `legacy_width_compatibility_zero`。

审计 verdict 为 **Supported**：现有 `CrossCorrelationEmbedding` 已具备 `[B,C,T]`、ordered aux+target、target-only writeback、parallel `C->C`、正号 identity residual、RNG-neutral zero delta 与分析接口，可由 `AMDEnhanced` 的显式 route 复用于 hidden `v_local`；不需要且不得新建第二个 Late class。来源表述限定为 CrossLinear-inspired hidden-state/late adaptation；lag `-1/0/+1` 是 AMD hidden-time 局部修正，不是原始物理观测上的一步 lead-lag。

### 41.3 Production capability 实现

实现登记独立 identity：

```text
implementation_variant = el-amd-m4-crosslinear-late-cce-v1
control ablation_id = M4_LATE_CCE_CONTROL
candidate ablation_id = M4_LATE_CCE
development_protocol_id = m4_crosslinear_late_cce_from_scratch_pair_v1
cce_architecture = crosslinear_inspired_hidden_state_late_cce_v1
cce_insertion_point = post_pmcr_pre_ams
cce_input_representation = amd_hidden_v_local
```

`AMDEnhanced` 现在只接受完整 Early 或完整 Late route triplet。Late route 为 `RevIN -> MDM(u_mdm) -> DDI(v_ddi) -> PMCR?(v_local) -> CCE?(v_final) -> AMS(experts=v_final,selector=u_mdm)`；固定公式是 `target_new=target_hidden+lambda*(cross_target-target_hidden)=target_hidden+lambda*delta_target`，没有引入负号。数学仍复用 kernel 3、zero-same、bias、`sigmoid(logit(0.1)+rho)` 与 zero delta；CCE+PMCR/TEB guard 保持，standalone pair 固定 PMCR/全部 TEB off。

runner 将 Late variant/arm/protocol 与三个 route 字段封存在 scientific/resolved/comparison identity、checkpoint 内嵌 resolved metadata和 manifest candidate contract；resume 先按 variant/config hash 拒绝 Early/Late 或 control/candidate 交叉身份，才允许 same-structure `strict=True` 写参。模型 importer 同时按 route/mode/schema/order/key/shape/dtype 作写参前 allowlist 检查并保持 failure atomicity。summarizer 独立识别 Late control/candidate，复核同一合同、checkpoint 和 13-file schema-v2，并拒绝 route tamper 与 duplicate spoof。Early CCE identity、恢复语义和第十九轮 artifact 均未改写。

本阶段实际修改范围精确为九个授权文件：canonical、本 M4 milestone、`main.py`、`models/modules/__init__.py`、现有 CCE module、`models/tsAMD_enhanced.py`、`summarize_results.py`、`tests/test_cross_correlation_embedding.py`、`tests/test_runner.py`。未修改 `models/tsAMD.py`、DataLoader、PMCR/TEB 数学、M0--M3、数据或既有 artifact；未新增 Late module 文件。

### 41.4 永久测试与真实单批探针

新增的永久保护覆盖 Late off/identity/正号、hidden target-only writeback、Early/Late route、selector/state_source、公共初始化与梯度 exact parity、first-backward aux/rho、strict restore/atomic rejection、PMCR guard、runner scientific/comparison/checkpoint/manifest identity、summarizer tamper 与 resume-before-model-write rejection。定向 CCE/AMDEnhanced/public architecture/runner/summarizer 回归为 `131/131 passed`；完整 discovery 为 `256/256 passed, failed=0, skipped=0`，CPU float32/float64 与可用 CUDA float32 路径均执行，未放宽既有 `1e-6` 或 exact 门禁。

一次性真实 ETTm1 h96 train-batch probe 直接复用 production `prepare_args`、runtime loader、model builder 和 prediction+MoE objective，不执行 optimizer step、不创建 artifact。结果：两臂 train generator 初态和首 batch exact；CCE construction CPU/CUDA RNG-neutral；公共 AMD 为 57 parameter tensors + 3 persistent buffers，key/shape/dtype/value exact；control 无 `cce.*`，candidate 恰有 3 keys。初始化 prediction/MoE/state_source max error 均为 `0`，MDM input、`u_mdm`、selector input exact，`v_final==v_local`，非 target hidden exact unchanged。第一次 backward 的 aux delta max-abs gradient `1.9047695968765765e-4`、target delta max-abs gradient `6.4536230638623238e-4`，均 finite，`rho.grad=0`；全部公共 AMD gradients exact。probe 清除 gradient 并恢复训练 forward 产生的 buffers 后，parameter/persistent-buffer digest 与 probe 前 exact；两个 `/tmp/m4_late_cce_impl_probe*` 临时目录均已自动删除。

### 41.5 Closure 前门禁

`git diff --check` 通过。implementation executable source fingerprint 更新为 21-file `adba794cdbc03b6d83a7c89f40d95bb5bf8163d2d32e23d530deff674e566005`。九文件之外没有 tracked/untracked 变化，index 仍为空。上述 capability 在精确 diff/staged scope 再核验后，才允许按本轮用户授权执行 implementation commit/push；八个 development runs 只能从随后 clean、已推送的新 HEAD 启动。此处尚未把 capability 或 probe 称为 development adequacy 通过，M4 保持 **In Progress**。
