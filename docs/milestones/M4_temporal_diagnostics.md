# M4：时间模块诊断与候选迭代

状态：In Progress

开始日期：2026-08-28（UTC）

当前轮次：第十一轮，T2G 结果 closure、T3 Selective Patch TEB 与 TEB 候选终点

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
