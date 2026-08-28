# M4：时间模块诊断与候选迭代

状态：In Progress

开始日期：2026-08-28（UTC）

当前轮次：第三轮，canonical 候选身份残余冲突 repair

canonical 内部版本：v2.1-R1

## 1. 阶段范围与禁止事项

M4 只处理第三章时间模块的训练/验证诊断、证据边界澄清和经用户另行授权的来源保持型候选迭代。M4 当前不执行模型筛选或结构冻结，M5-M13 尚未开始。

本阶段不得提前实现 StateAdapter、StateProjection、`H_time`、Graph Mode、HSTGCN-core、地理图或需求图、SADR、SC-SimGCA 及任何其他空间模块。不得使用 test 指标选择结构或超参数，不得把单数据集、单 seed、10 epoch 的 smoke 诊断写成正式性能验收。

M0、M1、M2、M3 均保持 Closed，不重新打开，也不向其 milestone 追加 M4 结果。

## 2. 继承状态

| 项目 | 继承值 |
|---|---|
| branch | `AMD-paper-repro-custom-modules-v1` |
| M4 起始 HEAD | `5fcac2010afe0154c900a3992de5b9d3803a2737` |
| HEAD parent | `ac4f2c2a98f8a5c4c3cafe48b36b9c2e68451b75` |
| baseline tag | `amd_reproduced_baseline_v1 -> fa9665627e6fcfb1d0c2bc22d943ca9666304fd6` |
| canonical 更新前 SHA-256 | `d3a2d480454cd5d0c38d6d29ef2dcb043d90f3b77fd9e568b55186dd476cd5da` |
| canonical 本轮更新后 SHA-256 | `be3e051ffb1b680f0c3566427b41394760a5e9219d533804e1e49e6c69f533e3` |
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

## 4. pre-M4 test 隔离登记

现有 pre-M4 ETTm1 smoke 包含 4 类模型 × 4 个 horizon、单训练 seed、10 epoch。其 test 结果在 M4 建立前已经被查看，只作为触发性能风险审计的历史事实；具体 test MSE/MAE 不在本 milestone 复述，也永久退出后续结构和超参数选择。

M4 只使用训练/validation 证据。最终 test 只允许在 M5 完成结构冻结后进入 M6 正式实验。

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
