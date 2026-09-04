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
M3 工程候选时间 variant：el-amd-pmcr-teb-v1
M4 外生模块状态：TimeXer-inspired TEB 与 CrossLinear-inspired CCE 均已触发有限开发停止线；Sonnet / Multivariable Coherence Attention（MVCA）S2 target residual 的精确 development-candidate 合同已由用户确认，production capability 已完成且 ChatGPT implementation review 已 Passed，当前等待 Stage B implementation Git closure；ETTm1/UrbanEV paired development 尚未执行
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
7. 除明确登记的 development-only 数据集外，结构和超参数只依据训练集与验证集；正式评价数据集的测试集只在方案冻结后使用；
8. 新增模块全部关闭时，增强入口必须与冻结 AMD 在相同权重、`eval()`、相同输入下数值等价；
9. “模块临时关闭”只能用于排障，不能在最终论文中替代“两模块均经过验证”的硬性要求；若某个来源模块经过限定调参仍失败，必须换用同来源的合理变体或更换另一篇近三年论文模块。

## 0.1 阶段顺序、候选身份与性能治理

自 2026-08-28 起，工程阶段固定为：

```text
M0-M3：已 Closed 的基准、数据管线、PMCR v1 与 Global TEB v1 工程闭环
M4：时间模块诊断与候选迭代
M5：模型筛选与结构冻结
M6：第三章正式实验与定稿
M7：时间状态接口与 Graph Mode
M8：HSTGCN-core 与双图构建
M9：SADR 状态需求残差图
M10：SC-SimGCA 状态条件图传播
M11：第四章正式实验与定稿
M12：论文正文、图表与结果分析
M13：终稿审校、复现材料与答辩
```

M2/M3 的 `Closed` 只表示对应工程实现、测试、文档和 Git 已闭环，不表示 PMCR 或 TEB 已通过最终性能验收。PMCR v1 与 Global TEB v1 均为可追溯工程候选；`el-amd-pmcr-teb-v1` 是 M3 时点的工程组合 variant，不是最终冻结模型。TimeXer-inspired Global/T2/T2G/T3 与 warm-start adapter rescue 的实现、artifact 和诊断继续作为失败路线及历史工程证据保留，不删除、不改写，也不得再称为当前领先外生候选。

M4 只处理第三章时间模块诊断与候选迭代。第十七轮已按预登记停止线确认 TimeXer-inspired TEB 路线在 M4 有限开发中失败；第十九轮 Early CCE 与第二十轮 Late CCE 又均未通过既定 development adequacy gate，因此 CrossLinear-inspired CCE 路线也已在 M4 有限开发中失败。两条来源路线的实现、永久测试、artifact 与诊断仅作为历史工程和负向 development 证据保留。用户已选择 Sonnet / Multivariable Coherence Attention（MVCA）作为当前下一来源，并在第二十二轮进一步确认第 1.1.1 节 S2 精确 development-candidate 合同；这不把 Sonnet/MVCA 称为最终外生模块、最终 EL-AMD 或 M5 冻结结构。最终时间结构与正式 variant 仍只能在 M5 依据训练/验证证据冻结。

M4 继续保持 In Progress。第二十二轮已由用户一次性确认 Sonnet/MVCA 的 S2 保留范围、RevIN 后/MDM 前插入点、target_exogenous-only 任务语义、matched standard from-scratch 协议、全部数值配置、development 数据边界与停止线；精确合同见第 1.1.1 节。Sonnet S2 production capability、永久测试和非训练单批探针已经完成，ChatGPT implementation review 已 Passed；当前等待 Stage B implementation Git closure。ETTm1/UrbanEV paired development 尚未执行，尚无 adequacy 或最终结构结论；Sonnet/MVCA 仍只是 M4 development candidate，不是最终外生模块、最终 EL-AMD 或 M5 冻结结构。XLinear 仍未选择；不得启动 PMCR/P2、进入 M5/M7 或实现任何空间模块。Stage B implementation Git closure 完成后，可以按 canonical 已锁定合同直接执行 paired development，无需用户重复授权。只有改变结构、来源、超参数、预算、停止线、test 边界，启动新候选或进入下一 milestone 时，才需要用户重新决定。

ETTm1 自 M4 第五轮起固定登记为 **development-only diagnostic benchmark**。M4 允许使用 ETTm1 的 train、validation 和 test 进行候选结构、容量与超参数探索；现有 production runner 可以继续按 `train -> validation -> validation 选择 best checkpoint -> test` 运行并生成完整 schema-v2 artifact，不要求为 ETTm1 实现 validation-only runner、独立 schema 或独立 summarizer。

ETTm1 test 已被纳入模型开发反馈，因此不得作为 M6 正式未见测试集结果，不得进入第三章正式性能主表，也不得用于最终无偏泛化主张。pre-M4 ETTm1 test 已查看并触发 M4 的事实继续保留，但不再被解释为“永久禁止参与后续 M4 开发决策”。论文可将 ETTm1 记录为开发集、诊断集或结构搜索数据集。

“测试集只在 M5 冻结后使用”的限制仅适用于当前正式评价数据集：UrbanEV、EPF-PJM、ETTh1、Weather、ECL、Exchange。这些数据集的结构和超参数选择在 M4/M5 仍只依据 train/validation，其 test 只在 M5 结构冻结后于 M6 使用。ETTm1 当前不属于 M6 正式主表；ETTh1 暂时仍保留为正式候选数据集，是否因同属 ETT 数据族而替换，延后至 M5 前由用户决定。若最终保留 ETTh1，论文必须披露 ETTm1 曾用于 M4 development。ETTm1 的 development-only 例外不得扩展到任何正式评价数据集。

“平均退化不超过 0.5%”只是安全底线，不能单独构成模块保留依据；practical-effect threshold 尚未锁定，必须在 M5 正式筛选前由用户确认。最终第三章仍必须包含至少两个来自近三年论文、经过修改并通过消融的来源模块；不得因 v1 暂时表现不佳而取消这一硬性要求。

# 1. 总体来源路线与冻结边界

## 1.1 第三章：纯时间模型

冻结 AMD 主干：

```text
RevIN -> MDM -> DDI -> AMS -> Forecast
```

历史 Early CCE v1 路由（失败工程候选）：

```text
RevIN
  -> x_ch
  -> CCE? -> x_cce
  -> MDM -> u_mdm
  -> DDI -> v_ddi
  -> PMCR? -> v_local
  -> AMS(experts=v_local, selector=u_mdm)
  -> Forecast
```

对应历史插入路径为：

```text
RevIN -> CCE -> MDM -> DDI -> PMCR? -> AMS
```

历史 Late CCE 路由（失败工程候选）：

```text
RevIN -> MDM -> DDI -> PMCR? -> Late CCE -> AMS
```

Early CCE 与 Late CCE 均未通过 M4 development adequacy gate；上述路由只保留为历史实现和负向 development 证据，不构成现行锁定方案。Sonnet/MVCA 的 S2 development-candidate 精确合同现已锁定，XLinear 尚未被选择；production implementation gate 与 ChatGPT implementation review 已 Passed，当前等待 Stage B implementation Git closure，后续 paired development 尚未执行且 adequacy 尚未评估，不能称为最终结构。

M3 时点工程组合：

```text
EL-AMD：Exogenous-and-Local Enhanced AMD
工程 variant：el-amd-pmcr-teb-v1
```

EL-AMD 名称继续作为第三章增强时间模型的项目名称（模型族名称）；`el-amd-pmcr-teb-v1` 只标识 M3 已闭环、可追溯的 PMCR v1 + Global TEB v1 历史工程候选。`el-amd-m4-crosslinear-cce-v1` 与 `el-amd-m4-crosslinear-late-cce-v1` 只标识已经停止的 Early/Late CCE production/development 历史候选。最终内部结构和正式 implementation variant 只能在 M5 冻结。

第三章固定来源目标：

- **PMCR：Peak-preserving Modern Convolution Refinement**
  - 来源：ModernTCN，ICLR 2024 Spotlight；
  - 借鉴可重参数化大核/小核 depthwise temporal convolution、ConvFFN1 和残差结构；
  - 删除与 AMD-DDI 重复的 ConvFFN2 跨变量分支；
  - 改成只补偿局部时间细节的单块轻量旁路。

- **外生模块当前来源候选：Sonnet / MVCA**
  - 来源：Sonnet，AAAI 2026；当前重点组件为 Multivariable Coherence Attention；
  - 用户已明确完成来源选择，但该候选尚不是最终外生模块或最终 EL-AMD；
  - 第二十一轮只读审计建议已经用户审核；第二十二轮锁定 S2、插入点 A、target_exogenous only、matched standard from-scratch 及第 1.1.1 节全部数值、来源裁决、evaluation 与 artifact 合同；
  - production capability 已完成且 ChatGPT implementation review 已 Passed；当前等待 Stage B implementation Git closure，ETTm1/UrbanEV 10-epoch paired development 尚未执行，也尚无 adequacy 或最终结构结论；closure 完成后可按已锁定合同直接执行，无需用户重复授权；
  - XLinear 尚未被选择，本轮不得同时启动；
  - TimeXer 与 CrossLinear 继续作为失败历史路线保留，不删除、不覆盖、不重新定义。

TimeXer-inspired TEB 与 CrossLinear-inspired CCE 的来源边界、实现和失败证据保留在第 7、7A、7B 节与 M4 milestone 中，仅作历史工程及负向 development 证据，不占用当前外生模块候选身份。其中历史 CCE 仅借鉴 CrossLinear（KDD 2025）的单层一维跨变量卷积嵌入，并分别形成 RevIN 后、MDM 前的 Early identity-residual delta 与 post-PMCR/pre-AMS 的 Late hidden-state adaptation；均不复制 CrossLinear 的第二套 normalization、patch embedding、positional embedding 或 forecasting head。

### 1.1.1 Sonnet S2 精确 development-candidate 合同

候选身份固定为：

```text
implementation_variant = el-amd-m4-sonnet-mvca-wavelet-residual-v1
control ablation_id = M4_SONNET_MVCA_CONTROL
candidate ablation_id = M4_SONNET_MVCA
architecture_identity = sonnet_inspired_joint_wavelet_mvca_target_residual_v1
input_identity = amd_revin_normalized_target_exogenous
insertion_identity = after_revin_before_mdm
development_protocol_id = m4_sonnet_mvca_from_scratch_pair_v1
```

唯一结构为 `S2 + 插入点 A + target_exogenous only + matched standard from-scratch`：

```text
AMD RevIN
-> Sonnet-inspired Joint Embedding
-> Learnable Wavelet
-> paper-defined MVCA
-> no-Koopman atom reconstruction
-> Linear(d,1) minimal readout
-> target-only gated residual
-> MDM -> DDI -> AMS
```

令 AMD RevIN 后 `z=[B,T,C]`。必须按 `X_aux=z[:,:,ordered_aux_idx]`、`y=z[:,:,target_idx:target_idx+1]` gather，原始来源顺序为 `[ordered_aux_idx...,target_idx]`，目标位于最后。固定 `d=64`、`K=8`、`alpha=0.5`；`alpha` 是非参数超参数。`E_aux=Linear(C_aux,32,bias=True)(X_aux)`，`E_target=Linear(1,32,bias=True)(y)`，latent 拼接严格采用论文顺序 `E=concat([E_aux,E_target],dim=-1)`，不得采用官方代码相反顺序。joint embedding 后不增加 normalization 或 dropout；不增加第二套 RevIN、InstanceNorm、BatchNorm 或 LayerNorm。

Learnable wavelet 的 `freq_params` 固定为 `[64,8,3]`、standard-normal 初始化且无约束。每次 forward 使用包含端点的 `torch.linspace(0,1,T)`；每个 atom 为：

```text
M_k(t) = exp(-w_alpha,k * t^2)
         * cos(w_beta,k * t + w_gamma,k * t^2)
P = E 与 atoms 逐元素相乘 = [B,8,T,64]
```

不得对 `w_alpha` 使用 `abs`、`softplus`、`clamp` 或正值约束；不得 padding、crop、求和为传统小波系数或压缩时间。

paper-defined MVCA 只接受 `[B,8,T,64]`。`Q/K/V=Linear(64,192,bias=True)` 后沿 latent `d` 维执行 `torch.fft.rfft`。每个 atom/时间位置按频率 bin 求均值得到：

```text
P_qk = Q_f * conj(K_f)
P_qq = Q_f * conj(Q_f)
P_kk = K_f * conj(K_f)
coherence = abs(mean(P_qk))^2 /
            (real(mean(P_qq))*real(mean(P_kk)) + 1e-6)
attention = Dropout(Softmax(coherence/sqrt(64), dim=time), p=0.1)
```

禁止 hard clamp，禁止官方代码实际形成的乘 `sqrt(K)`；权重按时间位置广播乘 `V`，不形成 `T×T` attention、不跨时间求和。随后保留 `weighted_v + Linear(64,64,bias=True)->GELU->Linear(64,64,bias=True)`，再用 `Linear(64,64,bias=True)` output projection，输出仍为 `[B,8,T,64]`。删除论文未定义且官方实现只产生 scalar trace 的 `var_attn`，不得用其他 variable-mixing 参数替代。

重构复用同一组 atoms：`r=sum_k(mvca_output_k*atom_k)=[B,T,64]`，不使用 Koopman。`delta=Linear(64,1,bias=True)(r)=[B,T,1]`，readout weight 为 Xavier uniform、bias 为 0。`gamma_sonnet` 是全样本/全时间共享的无约束 scalar `nn.Parameter`，初始化 `1e-3`；只写回 `y + gamma_sonnet*delta`，全部非目标通道必须 bitwise 不变。模块关闭时不实例化、不计算、不产生 `sonnet_mvca.*` state key，并相对冻结 AMD exact identity；模块开启是 `1e-3`-gated near identity，不要求 exact identity。不得把 gamma 或 readout/MVCA output projection 置零。

明确排除 Sonnet 自有 RevIN、Koopman operator、decoder/forecasting head、multiblock/downsampling wrapper、horizon-dependent head、新 exogenous context pooling/head、paper-only D.8 feature-head-split 插件、官方 `var_attn` 和 parallel_multivariate Sonnet 路径。官方仓库缺少独立 LICENSE/COPYING/NOTICE；本项目只依据论文公式与已审计行为独立实现，不复制其源文件。
来源身份同时固定为：论文 `Sonnet: Spectral Operator Neural Network for Multivariable Time Series Forecasting`，Yuxuan Shu / Vasileios Lampos，AAAI 2026，DOI `10.1609/aaai.v40i30.39736`，PDF SHA-256 `b076e6fed68448d3c3382c96f6f6985a988ea019ef3c470353780385c4011079`；官方仓库 `https://github.com/ClaudiaShu/Sonnet.git` @ `bf3d4801d34c5e7261718490f287c6fb15cadfdb`。核心 SHA-256 为 `Sonnet.py=be4fd33b9d1eb4a4f09be0a325a8aa87d5efd5d754e184606fc8a5808769b684`、`RevIN.py=0139409a58e57aca7c7e5423346db3f9224c6e871fecead418797ec4977e756b`、`lightning_module.py=f25e2e9ee1d12444eabf4ad6616c14f4f77c9bdad9a6886091193c6eca744d62`、`sonnet.yaml=329463667b7bb4aa80cf1a7761c3ac6adc7091c8a2cfda63545e69fd2f756346`、`setup.py=85a9f7773200d374a04ad42006a78efbf580c5680602367150b09ebd9979dcc7`；许可状态登记为 `license_text_missing_classifier_only`。


AMD 路由固定为：

```text
x -> AMD RevIN -> z -> Sonnet adapter? -> z_new
  -> transpose -> MDM(u_mdm) -> DDI(v_ddi) -> PMCR?(本候选固定 off)
  -> AMS(experts=v_local, selector=u_mdm)
  -> full-channel AMD RevIN denorm -> target selection
```

该 development variant 强制 PMCR、全部 TimeXer TEB、Early/Late CCE 关闭；不得实现 Sonnet+PMCR。`state_source=concat(v_final[:,target_idx,:],u_mdm[:,target_idx,:],deterministic_zero_placeholder)`，第三段继续是现有固定宽度、dtype/device 正确的零占位；不得创建 Sonnet `exo_context` 或修改 M7 StateAdapter。

任务只支持 `target_exogenous`，明确拒绝 `parallel_multivariate`。必须校验 `target_idx` 合法；`ordered_aux_idx` 非空、无 bool、无重复、不含目标且全部范围合法；feature schema、名称和顺序完全匹配。F0/空 aux 不支持。

control 与 candidate 使用相同主 seed，先以完全相同顺序构造公共 AMD 主干；candidate 分支在隔离 RNG 上下文中以 `module_init_seed=run seed` 初始化，并恢复全局 CPU/CUDA RNG，不改变 train DataLoader generator 初态。永久门禁必须证明公共 AMD parameter/persistent buffer、全局 CPU/CUDA RNG、train generator state 与首个 train batch逐元素一致。control 与 candidate 均为 standard from-scratch、全部自身参数可训练、fresh Adam；source checkpoint 为 null，不使用 importer、warm-start 或 frozen adapter。非零 gamma/readout/output projection 用来保证非退化 batch 首次 backward 中 joint embedding、wavelet、Q/K/V、residual MLP、output projection、readout 与 gamma 均可获得 finite nonzero 任务梯度；负例必须证明 gamma=0 或 zero readout 会切断上游。

development 协议固定如下；第二十二轮 production capability 阶段未执行这些 10-epoch 实验，paired development 当前仍为 Not started，须在 Stage B implementation Git closure 后按已锁定合同直接执行：

| 数据 | 固定合同 |
|---|---|
| ETTm1 | `target_exogenous`；OT；target 6；aux `[0,1,2,3,4,5]`；T=512；H=`[96,192,336,720]`；seed 2024；10 epochs；batch 32；Adam lr `3e-5`、weight decay `1e-7`；PMCR/旧外生模块 off；允许 train/validation/development test，不能进入 M6 正式主表 |
| UrbanEV | F4；fold 6；history 12；label horizons `[3,6,9,12]`；volume；target 0；aux `[1,...,10]`；seed 2024；10 epochs；batch 128；Adam lr `3e-5`、weight decay `1e-7`；PMCR/旧外生模块 off；仅 train+validation |

UrbanEV 的 `train_validation_only` 必须在数据构造层隔离：不创建 test Dataset/DataLoader，不遍历 test，不预测或输出 test metrics，不伪造 test 字段，也不把 validation 写入 test。完整数据文件只可参与 byte-level fingerprint。必须复用 M1 fold/split/train-only scaler。其主门禁为 validation MSE macro 小于 control、validation MAE macro 不高于 control、至少 3/4 horizon validation MSE 改善、改善不由单一 horizon 驱动且是 full-precision 非舍入差。ETTm1 安全门禁为 validation MSE macro 退化不超过 0.5%、development test MSE/MAE macro 各退化不超过 0.5%、任一 horizon development test MSE 退化不超过 1.0%。只有 UrbanEV 主门禁与 ETTm1 安全门禁同时通过，才是 positive development signal；任一失败即按预登记停止，禁止自动调 d/K/alpha/gamma/dropout/epsilon/预算/batch 或转向其他阶段。

未来 artifact root 固定为 `artifacts/m4-development/ettm1-stage-i-sonnet-mvca-v1` 与 `artifacts/m4-development/urbanev-stage-i-sonnet-mvca-v1`。沿用 schema-v2 的 13-file payload、checksums、hidden staging 与 atomic publication，并显式记录 `evaluation_policy=train_validation_test|train_validation_only`、`artifact_purpose=m4_development_candidate`。`train_validation_only` 的 `metrics.json` 不得含 `test` key，manifest 不得含 `test_mse`、`test_mae` 或其他 test result 字段，但必须固定 `test_access_policy=forbidden`；RuntimeData 不得持有可遍历 test loader；summarizer 使用独立 validation-only 分支并拒绝混合 evaluation policy。不得降低既有 test-inclusive schema-v2 合同。

上述全部结构、source paper/repo/commit/PDF/core SHA、license-text-missing、retained/deleted components、raw/latent order、d/K/alpha/epsilon/dropout/gamma、FFT/denominator/scale/softmax/var_attn/reconstruction/readout/init policy、task/schema/order、evaluation policy 与 module-init-seed policy必须进入 resolved config、scientific/comparison hash、checkpoint、manifest、resume preflight 和 summarizer。resume 只允许同一 Sonnet variant/ablation/task/schema/order/T/d/K/alpha/policy/evaluation/scientific config 的 `strict=True` restore；跨 TimeXer/CrossLinear/XLinear、control/candidate、target/parallel、schema/order或结构不匹配、partial `sonnet_mvca.*`、key/shape/dtype 不匹配均须在写参前原子拒绝。

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
trained StateAdapter（M7） -> H_time
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

# 2. 核心论文与模块边界

| 角色 | 正式论文 | 出处 | 本方案使用范围 |
|---|---|---|---|
| 时间基准 | Adaptive Multi-Scale Decomposition Framework for Time Series Forecasting | AAAI 2025 | 完整 MDM + DDI + AMS |
| 时间模块 T1（失败历史路线） | TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables | NeurIPS 2024 | Global/T2/T2G/T3/rescue 的来源边界与失败工程证据，不是当前候选 |
| 时间模块 T1（失败历史替代路线） | CrossLinear: Plug-and-Play Cross-Correlation Embedding for Time Series Forecasting with Exogenous Variables | KDD 2025 | Early/Late Cross-Correlation Embedding 的来源边界、实现与负向 development 证据；不复制 normalization、patch、PE、head |
| 时间模块 T1 当前 development 候选（精确合同已锁定） | Sonnet: Spectral Operator Neural Network for Multivariable Time Series Forecasting | AAAI 2026 | S2 joint embedding + learnable wavelet + paper-defined MVCA + no-Koopman reconstruction + target residual；插入 RevIN 后/MDM 前；target_exogenous only；不是最终外生模块 |
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

`sys.argv.json` 必须逐项保存运行时真实 `sys.argv`；`command.txt` 必须按 `shlex.join([sys.executable, *sys.argv])` 保存真实 Python executable 与完整 shell-escaped argv，形成可重放的等价命令。`stdout.log`、`stderr.log` 和 `train.log` 必须完整保留正式运行的真实捕获输出，不得用空占位文件或事后推断的命令冒充原始证据。

`checksums.sha256` 必须作为独立文件生成，至少覆盖 `best.pt`、`last.pt`、`config.resolved.json`、`history.jsonl`、`metrics.json`、`manifest.json` 和 `train.log`。增强 artifact 固定采用：隐藏 staging 目录写入全部可变文件 -> 关闭 stdout/stderr/train.log writer -> 写最终 completed manifest -> 生成 checksums -> Python verifier -> 实际执行 `sha256sum -c checksums.sha256` -> 同文件系统目录级 atomic rename 发布到最终 run 目录。最终 run 目录只代表已经验证且不可变的 completed artifact；校验或发布失败时最终目录必须不存在，staging 不得被 summarizer 接受。

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

UrbanEV `target_exogenous` 正式 runner 必须直接消费 M1 的共享数据对象：`UrbanEVRawData.load(data_root) -> UrbanEVFoldPreprocessor.fit_transform(fold,preset) -> one shared UrbanEVFoldBundle -> TemporalRegionDataset(train/validation/test)`。不得另建 scaler、split 或窗口逻辑。固定 `seq_len=history_len=12`、`label_horizon in {3,6,9,12}`、`model_pred_len=1`、`fold in {1,...,6}`；artifact 的 `horizon_<h>` 使用 `label_horizon`，而非 `model_pred_len`。

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
 -> M7 中经过已训练 StateAdapter
 -> restore
H_time [B,N,d]
y_time [B,N,H_out]
```

M1 只需验证 flatten/restore 后的 `state_source` 与 `y_time`；在 M7 创建并训练 StateAdapter 之前，不得伪造 `H_time`。

禁止把第三章任意 shuffle 后的 `B_region` 样本直接拼回图。必须通过时间窗口 ID 和固定 node order 恢复，且有单元测试验证。

## 4.3 两接口一致性测试

同一组完整图窗口在 `eval()` 下：

1. 用 `GraphWindowDataset` 内部 flatten；
2. 用 `TemporalRegionDataset` 按相同 node order 手工展开；
3. M1 验证两者得到的 node-wise `y_time` 和 `state_source` 必须一致；
4. M7 在同一已训练 StateAdapter 下再验证两者恢复的 `H_time` 一致。

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
pred_all_norm_ch [B_region,C,H_out]
 -> transpose
pred_all_norm [B_region,H_out,C]
 -> RevIN full-channel denorm with slice(None)
pred_all [B_region,H_out,C]

pred_target = pred_all[:,:,target_idx:target_idx+1]
# [B_region,H_out,1]

loss = criterion(
    pred_target.squeeze(-1),
    y_target,
)
# y_target [B_region,H_out]
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

M2 工程实现阶段尚未实现 TEB，因此该阶段的历史真实路径为：

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

校验必须在修改参数前完成；禁止把全局静默 `strict=False` 作为兼容方案。`use_pmcr`、hidden dim、两个 kernel、dropout、gamma init 及 deploy 形态均进入可追溯配置/checkpoint 元数据；M3 工程 runner 通过 `el-amd-pmcr-teb-v1` 和显式消融开关统一接入 PMCR/TEB。该 variant 只记录 M3 时点工程候选，最终正式 variant 由 M5 冻结。

## 6.7 必做测试

- shape 与 dtype/device；
- `use_pmcr=False` 严格 pass-through；
- 参数和输入梯度非零；
- 修改输入某一变量时，其他变量的 PMCR `delta` 不应变化，验证无跨变量混合；
- 训练态双分支与导出后重参数化卷积数值等价；
- `T=12` 下 kernel/padding 不改变长度。

# 7. TEB：TimeXer-inspired 目标—外生桥接

当前正式 v1 定义为 **Global Target-Conditioned Residual TEB（全局目标条件残差桥接）**。它是在 AMD 隐表示上的轻量 Target–Exogenous Bridge，不是完整 TimeXer、缩小版 TimeXer Transformer 或第二套时间预测主干。

## 7.1 单目标模式

DDI+PMCR 输出：

```text
H [B,C,T]
H_y = H[:,target_idx,:]
```

目标 Query：

```text
q = LayerNorm_q(Linear_q(T -> d)(H_y)).unsqueeze(1)
# [B,1,d]
```

外生输入固定为 RevIN 后的历史 `normalized_input=x_norm [B,T,C]`，所有辅助变量共享同一个 projector：

```text
X_aux = normalized_input[:,:,aux_idx].transpose(1,2)
# [B,m,T]

E_aux = LayerNorm_exo(Linear_exo(T -> d)(X_aux))
# [B,m,d]
```

Cross-attention：

```text
c_exo = MHA(query=q,key=E_aux,value=E_aux)
# [B,1,d]

delta_y = Linear_out(d -> T)(c_exo.squeeze(1))
# [B,T]

H_y_new = H_y + gamma_teb * delta_y
exo_context = c_exo.squeeze(1)
# [B,d]
```

只替换 `target_idx` 通道，其他通道必须逐元素保持原值。`d = teb_context_dim`，不设置 `teb_hidden_dim`。

推荐：

```text
teb_context_dim=32
teb_heads=4
teb_dropout=0.1
teb_gamma_init=1e-3
```

`Linear_out(d -> T)` 使 TEB checkpoint 与 `seq_len` 绑定；runner/resume 必须把 `seq_len` 作为严格科学配置，不支持跨 `seq_len` 静默加载。第一版不改变 AMS selector 使用的原始 `u_mdm`，只改变 AMS experts 的输入。

`gamma_teb` 是所有变量共享、无约束、可学习的全局 scalar `nn.Parameter`，初始化 `1e-3`。禁止固定常数、零初始化、sigmoid/softplus 约束、每变量或每时间位置 gamma。零初始化会使受门控的 TEB 内部参数第一次反向传播梯度为零；`1e-3` 保持小扰动且梯度非零。

## 7.2 TEB 关闭合同

```text
use_teb=False：
    self.teb = None
    不产生 teb.* state_dict key
    v 保持逐元素不变
    exo_context = v.new_zeros([B,teb_context_dim])
```

不得返回 `None`，不得改变 StateAdapter 输入维度。空输入合同固定为：

```text
use_teb=False + aux_idx=[]：合法严格旁路
target_exogenous + use_teb=True + aux_idx=[]：明确报错
parallel_multivariate + use_teb=True + C=1：明确报错
parallel_multivariate + use_teb=False + C=1：合法旁路
```

单目标空辅助固定报错为 `TEB requires at least one auxiliary variable.`；parallel `C=1` 固定报错为 `Parallel TEB requires at least two variables.`。不得将空 Key/Value 送入 MHA、产生全 `-inf` attention 行、在 `C=1` 时取消 mask，或以 zero context 冒充 TEB enabled。

`target_exogenous` 的 `aux_idx` 必须显式、有序并规范化为 `tuple[int,...]`，保持调用方顺序，拒绝 bool、重复、越界和 `target_idx`。Scientific config 同时保存 feature/target/aux 名称与索引及 schema fingerprint。`parallel_multivariate` 固定 `parallel_aux_policy=all_other_variables`，非空手工 `aux_idx` 必须拒绝。

## 7.3 M-to-M 并行模式

用于 ETTh1/Weather/ECL/Exchange，一次向量化执行：

```text
Q = LayerNorm_q(Linear_q(T -> d)(hidden))
# [B,C,d]

E = LayerNorm_exo(Linear_exo(T -> d)(normalized_input.transpose(1,2)))
# [B,C,d]

diagonal_mask [C,C]
diagonal_mask[i,i] = True
diagonal_mask[i,j] = False, i != j

context_all = MHA(query=Q,key=E,value=E,attn_mask=diagonal_mask)
# [B,C,d]

delta_all = Linear_out(d -> T)(context_all)
hidden_out = hidden + gamma_teb * delta_all
# [B,C,T]

exo_context = context_all[:,target_idx,:]
# [B,d]
```

Mask 中 `True` 表示禁止。禁止 Python 循环逐变量运行完整 AMD；所有变量均参与更新和预测。`context_all [B,C,d]` 只在 parallel 内部更新全部变量；普通 `exo_context [B,d]` 只用于固定 `state_source`。`target_idx` 在 parallel 中只作为状态和可选分析锚点，不限制全部变量预测。

## 7.4 模块边界、checkpoint 与 runner

模块固定包含：

```text
Linear_q(T,d,bias=True) -> LayerNorm(d,eps=1e-5)
共享 Linear_exo(T,d,bias=True) -> LayerNorm(d,eps=1e-5)
MultiheadAttention(d,heads,dropout=teb_dropout,bias=True,batch_first=True)
Linear_out(d,T,bias=True)
scalar gamma_teb
```

除 gamma 外使用 PyTorch 默认初始化。第一版不加入 mean pooling、variable identity embedding、q residual、attention 后 FFN、额外 output dropout、endogenous/exogenous self-attention、多层 Transformer、TimeXer prediction head 或未来真实外生变量。唯一写回 AMD 隐表示的 residual 是 `H_new = H + gamma_teb * delta`。

checkpoint 使用显式 source-kind 精确 allowlist，并在修改参数前校验完整 key 集与 tensor shape：

```text
baseline  -> 允许目标模型全部 pmcr.* / teb.* 缺失
pmcr_only -> 只允许目标模型全部 teb.* 缺失
teb_only  -> 只允许目标模型全部 pmcr.* 缺失
unexpected keys 必须为空；部分 enhancement key 不得接受
完整同结构 `el-amd-pmcr-teb-v1` checkpoint/resume 使用普通 `load_state_dict(strict=True)`
```

禁止全局静默 `strict=False`。M3 已闭环工程 variant 为 `el-amd-pmcr-teb-v1`，并使用显式 `ablation_id` 表达其 v1 消融；它不是 M5 冻结后的正式 variant。M4 若实现任何结构变化，必须使用新的候选 variant，不得覆盖、复用或重新定义 `el-amd-pmcr-teb-v1`。最终正式 implementation variant 由 M5 冻结。runner 必须显式保存任务模式、目标/辅助名称与索引、schema fingerprint、PMCR/TEB 参数及 policy；M3 v1 路径中 `target_idx` 是唯一目标索引来源，`target_slice=None`，并遵循第 3.4 节 artifact/checksum/resume 合同。

## 7.5 公平对照

UrbanEV/EPF 的 U1 与 U2 必须使用完全相同的辅助 schema、变量顺序、划分、scaler、horizon 和 seed：

```text
U1: AMD-Concat，PMCR off，TEB off
U2: AMD-Concat + TEB，PMCR off，TEB on
```

M3 只以 tiny synthetic smoke 验证结构、配置和 artifact 流程，不据此宣称性能改善。M4 对正式评价数据集仅使用训练/验证证据；ETTm1 则按第 0.1 节登记为 development-only benchmark，可使用 train/validation/test 进行经用户授权的候选迭代。M5 完成公平筛选与结构冻结；UrbanEV、EPF-PJM、ETTh1、Weather、ECL、Exchange 的测试集只在结构冻结后的 M6 正式实验中使用。只有满足完整公平协议和 practical-effect threshold 后，才能把稳定增益归因于桥接结构。

## 7.6 M4 工程候选：T2 Patch-Conditioned TEB

用户已在 M4 第六轮明确授权实现 **T2 Patch-Conditioned TEB**。其工程身份固定为：

```text
implementation_variant = el-amd-m4-t2-patch-teb-v1
ablation_id = M4_T2
teb_architecture = patch_conditioned_v1
```

T2 是 M4 工程候选，不是最终 TEB 或最终 EL-AMD，不覆盖 Global TEB v1，不复用或重新定义 `el-amd-pmcr-teb-v1`；是否进入正式模型只能由 M5 冻结。

### 7.6.1 来源边界与固定配置

TimeXer 原始来源语义是 endogenous patch-level tokens 加 global endogenous token、exogenous whole-series variate tokens，以及 global token 对 exogenous tokens 的查询，外生信息再经目标侧 token 交互传播到 patches。本项目 T2 改为 target patch queries 直接查询 exogenous variate tokens，同时保留 global target context。因此 T2 必须表述为 **TimeXer-inspired hierarchical representation adaptation**，不是原样 TimeXer cross-attention、完整 TimeXer 或缩小版 TimeXer Transformer。

固定配置为：

```text
teb_context_dim = 32
teb_heads = 4
teb_dropout = 0.1
teb_gamma_init = 1e-3
teb_patch_padding = right_zero_crop
teb_patch_position = fixed_sinusoidal

ETTm1 / seq_len=512：teb_patch_size=32
UrbanEV / seq_len=12：teb_patch_size=3
```

`teb_patch_size` 必须显式配置，不得依据 dataset 名称自动选择。位置编码固定为不可学习、非 persistent 的 sinusoidal buffer；除共享 scalar `gamma_teb=1e-3` 外，Linear、LayerNorm 和 MHA 沿用 Global TEB v1 的 PyTorch 默认初始化语义。

### 7.6.2 Single-target 精确合同

```text
H_y = hidden[:,target_idx,:]                         # [B,T]
N = ceil(T/P); pad_len = N*P-T
patches = right_zero_pad(H_y).reshape(B,N,P)        # [B,N,P]
Q_patch = LayerNorm(Linear(P,d)(patches) + PE)      # [B,N,d]
q_global = LayerNorm(Linear(T,d)(H_y)).unsqueeze(1) # [B,1,d]

X_aux = normalized_input[:,:,aux_idx].transpose(1,2) # [B,m,T]
E_aux = LayerNorm(shared Linear(T,d)(X_aux))          # [B,m,d]
Q_all = concat(Q_patch,q_global,dim=1)                 # [B,N+1,d]
C_all = MHA(Q_all,E_aux,E_aux)                         # [B,N+1,d]

C_patch = C_all[:,:N,:]
c_global = C_all[:,N,:]                              # [B,d]
delta_patch = shared Linear(d,P)(C_patch)            # [B,N,P]
delta_y = delta_patch.reshape(B,N*P)[:,:T]           # [B,T]
H_y_new = H_y + gamma_teb * delta_y
```

只替换 `target_idx`，其他通道逐元素不变；`exo_context=c_global [B,d]`。`aux_idx` 继续是显式有序 tuple，非空、不含 target、无重复、无 bool 且不越界。

### 7.6.3 Parallel 精确合同

```text
hidden [B,C,T]
 -> patches [B,C,N,P]
 -> Q_patch [B,C,N,d]
q_global [B,C,1,d]
Q_by_variable = concat(Q_patch,q_global,dim=2)       # [B,C,N+1,d]
Q_flat = reshape(Q_by_variable,[B,C*(N+1),d])

E_all = LayerNorm(shared Linear(T,d)(
    normalized_input.transpose(1,2)
))                                                   # [B,C,d]

owner = arange(C).repeat_interleave(N+1)
mask[q,k] = (k == owner[q])                          # [C*(N+1),C]
C_flat = MHA(Q_flat,E_all,E_all,attn_mask=mask)
C_by_variable = reshape(C_flat,[B,C,N+1,d])

C_patch = C_by_variable[:,:,:N,:]
C_global = C_by_variable[:,:,N,:]                   # [B,C,d]
delta_all = Linear(d,P)(C_patch)
             .reshape(B,C,N*P)[:,:,:T]              # [B,C,T]
hidden_out = hidden + gamma_teb * delta_all
exo_context = C_global[:,target_idx,:]               # [B,d]
```

MHA 必须一次向量化执行，mask 中 `True` 表示禁止查询自身变量；不得逐变量运行 AMD 或完整 TEB。全部变量都产生 residual 和预测，`target_idx` 只选择 `state_source` 的 global context。`C=1` 且启用 T2 时固定报错 `Parallel TEB requires at least two variables.`。

T2 后状态源顺序和宽度保持：

```text
state_source = concat(
    v_final[:,target_idx,:],
    u_mdm[:,target_idx,:],
    exo_context,
) # [B,2*T+d]
```

### 7.6.4 模块、checkpoint 与 artifact 合同

T2 使用独立 `PatchConditionedTargetExogenousBridge` class，旧 `TargetExogenousBridge` 文件、class、state keys 和 forward 均不改变。T2 trainable parameter count 固定满足：

```text
2*P*d + 2*T*d + 4*d*d + 13*d + P + 1
T=512,P=32,d=32 -> 39,361
T=12,P=3,d=32   -> 5,476
```

T2 只允许 from-scratch 初始化和完全同结构的普通 `load_state_dict(strict=True)` resume。variant、ablation、architecture、patch size/padding/position、context dim、heads、dropout、gamma init、seq_len、task mode、target/aux/schema 以及 source/data fingerprint 全部进入 scientific config/checkpoint/resume 合同。禁止 Global TEB v1 与 T2 之间 strict load，禁止 source-kind 结构迁移、部分 key、`strict=False` 或自动补齐 missing patch keys。固定 sinusoidal position 不进入 state dict。

T2 使用现有 schema-v2 完整 artifact，路径以 `el-amd-m4-t2-patch-teb-v1` 为 variant 根；summarizer 必须保留该候选身份、核验完整 patch 合同和 13-file checksum、拒绝重复科学身份，不得把 T2 归入 `el-amd-pmcr-teb-v1`。旧 v1 scientific/comparison config 不得无条件增加 patch 字段，历史 hash 语义保持不变。

T2 明确不包含 T3 confidence gate、T4 Hidden-KV、T5 sparse/top-k、外生 patchify、外生或目标 self-attention、完整 Transformer encoder、attention 后 FFN、variable identity embedding、未来真实外生输入、第二套 selector、PMCR/P2/MDM-bypass 或任何空间模块。第六轮完成工程实现与测试，第七轮完成 ETTm1 development 实验；这些结果均不把 T2 冻结为最终结构。第七轮时 T3 尚处于暂缓状态；其后第十一轮已获用户授权、完成实现与 development，但为 `negative-or-negligible`。T4/T5 继续排除，任何后续 TEB architecture 仍须由用户另行确认。

### 7.6.5 第八轮 global-query 梯度语义

标准 `MultiheadAttention(Q,K,V)` 对每个 query row 独立计算 cross-attention；把 `Q_patch` 与 `q_global` 拼接后一次调用 MHA，只共享 K/V 与投影参数，不产生 query-row 间交互。当前 T2 的 production 时间路径严格为：

```text
Q_patch -> C_patch -> patch output projection -> temporal residual
        -> v_final -> AMS experts -> prediction -> prediction MSE

q_global -> C_global -> exo_context -> state_source
```

AMS selector 与 selector auxiliary loss 仍只依赖原始 `u_mdm`；production runner 调用 `model(x)`，不请求 `state_source`，总目标固定为 `MSE(prediction,y) + selector_auxiliary_loss`。因此当前源码没有 `C_global -> C_patch`、`q_global -> temporal residual` 或 `state_source -> 第三章训练目标` 的路径。

第八轮在四个 ETTm1 T2 best checkpoint 和 UrbanEV F4 single-target 真实 train batch 上各执行一次 production-loss backward，未执行 `optimizer.step()`。五个案例中，`C_global`、MHA global query rows 与 `q_global` 的 raw gradient 均为有限的严格全零张量，`exo_context.grad is None`；`global_query_projection.{weight,bias}` 与 `global_query_norm.{weight,bias}` 的 raw gradient 也全部严格为零。与此同时，patch query、`C_patch`、exogenous projector、共享 MHA、patch output projection、`gamma_teb` 与 AMD 主干均获得非零梯度。Global TEB v1 h96 正控制的 query projection/norm 则获得非零 production gradient。

独立 state-control loss 只使用 `exo_context` 时，single 与 parallel 的 `q_global/C_global` 及其专属 Linear/LayerNorm 参数均获得有限非零梯度，证明该图没有 detach。只扰动 global-query 专属 Linear/LayerNorm 参数时，ETTm1 h96 与 UrbanEV single 的 prediction、MoE loss、`C_patch` 和 temporal delta 均逐元素不变，而 `C_global`、`exo_context` 和 `state_source` 最后 `d` 维发生变化。

因此当前 T2 的 global-query 专属分支是 **state-output-live，但 forecast-loss-disconnected**。共享的 exogenous/MHA 参数仍可通过 patch residual 获得任务梯度；不得把这一事实扩大解释为整个 T2 无梯度、整个 `exo_context` 完全静止、T2 development 结果无效、T2 必须废弃，或已证明某一修复结构必然更好。既有“所有逻辑参数组梯度非零”测试使用 `hidden_out` 与 `context` 的联合 synthetic loss，只证明模块两个输出联合可导，不证明 production forecast loss 会训练全部逻辑参数。

本轮只确认事实并保留无代码方案比较，不提前批准下一 TEB architecture。后续可以由用户在保留 T2、用 forecast-supervised `C_patch` pooling 导出状态 context、或建立最小 global-mediated patch interaction 等方向之间另行决定；作出该决定前不得启动 P2。

## 7.7 M4 工程候选：T2G Global-Mediated Patch-Conditioned TEB

用户在 M4 第九轮明确批准实现 T2G；其身份固定为：

```text
candidate = T2G
public class = GlobalMediatedPatchTargetExogenousBridge
implementation_variant = el-amd-m4-t2g-global-mediated-patch-teb-v1
ablation_id = M4_T2G
teb_architecture = global_mediated_patch_v1
```

T2G 是 T2 的单因素 M4 工程扩展，不是原 T3、最终 TEB 或最终 EL-AMD；它不覆盖 Global TEB v1、T2 或 `el-amd-pmcr-teb-v1`。第九轮批准 T2G 时，是否进入 TEB 候选终点仍待 development 证据，T3 confidence gate 也处于暂缓状态；随后 T2G/T3 均已完成且为 `negative-or-negligible`，第十一轮旧 endpoint 又在第十二轮被撤销。TEB-first adequacy 顺序继续有效，T4/T5 继续排除。

### 7.7.1 来源边界与唯一结构变化

TimeXer 的 endogenous patch/global token self-attention 使用 residual + LayerNorm；global token 查询 exogenous variate tokens 后还使用 global residual + LayerNorm，且原结构不让各 patch 直接查询 exogenous tokens。T2 保留其 TimeXer-inspired 层级表示语义，但改为每个 target patch 直接查询 whole-series exogenous variate tokens。T2G 继续保留该 T2 patch 路径，只新增：

```text
global cross-attention response
-> q_global residual + post-cross LayerNorm
-> patch-conditioned global injection
-> temporal residual
```

T2G v1 明确不使用 `LayerNorm(Q_patch + A_patch)`、`Q_patch + A_patch`、额外 patch post-cross LayerNorm、patch FFN、patch self-attention 或 patch-to-patch attention。`Q_patch` 只能作为 patch cross-attention query，并经 raw `A_patch` 影响预测；gate 只能使用 `[A_patch;G_global]`，不得直接引入 `Q_patch`。这样保持 T2 已验证 patch-to-exogenous 路径不变，避免额外 target-only shortcut，也不把 patch residual 与 global-mediated interaction 捆绑为同一候选。

### 7.7.2 精确张量合同

沿用 T2 的 patchify、fixed sinusoidal position、shared exogenous projector、一次向量化 cross-attention、right-zero-pad/crop、owner diagonal mask、patch output projection、`gamma_teb` 和 AMD 外层 residual。MHA 输出命名必须是 raw response：

```text
A_patch, A_global = MHA(concat(Q_patch,q_global), E, E)

G_global = LayerNorm_global_bridge(q_global + A_global)

gate_input = concat(
    A_patch,
    broadcast_N(G_global),
    dim=-1,
)
a_patch = 2 * sigmoid(Linear_gate(2*d,1)(gate_input))

F_patch = A_patch
        + beta_global * a_patch * broadcast_N(G_global)

delta_patch = shared Linear(d,P)(F_patch)
delta = unpatch(delta_patch) -> crop to T
H_out = H + gamma_teb * delta
```

Single-target shapes 为 `A_patch/F_patch [B,N,d]`、`A_global/G_global [B,1,d]`、`a_patch [B,N,1]`。只修改 `target_idx` 通道，其他通道逐元素不变；`exo_context=G_global.squeeze(1) [B,d]`。

Parallel shapes 为 `Q_flat [B,C*(N+1),d]`、一次 MHA、`A_patch/F_patch [B,C,N,d]`、`A_global/G_global [B,C,1,d]`、`a_patch [B,C,N,1]`。mask `[C*(N+1),C]` 中 `True` 禁止 query owner 查询自身 key；所有变量产生 residual，`target_idx` 只选择 `G_global[:,target_idx,0,:]` 作为 `exo_context [B,d]`，`C=1` 继续拒绝。AMS experts 使用 `v_final`，selector 仍只使用 `u_mdm`；状态源固定为：

```text
state_source = concat(
    v_final[:,target_idx,:],
    u_mdm[:,target_idx,:],
    exo_context,
) # [B,2*T+d]
```

### 7.7.3 初始化、参数与解释边界

`global_bridge_norm=LayerNorm(d,eps=1e-5)`，weight=1、bias=0。`global_injection_gate=Linear(2*d,1,bias=True)`，weight=0、bias=0，因此初始 `a_patch=1`。`beta_global` 是所有变量共享、无约束 scalar parameter，固定初始化 `1e-3`；既有 `gamma_teb` 仍初始化 `1e-3`，所以 global-mediated 分支在 AMD hidden 上的初始有效系数量级约为 `1e-6`。禁止将 beta 初始化为 0，也禁止 sigmoid/softplus/非负约束，因为 beta=0 会让 global query、global bridge norm 与 gate 的首个 forecast backward 继续缺少或延迟梯度。

T2G 在 T2 上仅新增：

```text
global_bridge_norm.weight/bias       2*d
global_injection_gate.weight/bias    2*d+1
beta_global                          1
新增合计                              4*d+2
```

`d=32` 时新增 130；`T=512,P=32` 总参数 39,491，`T=12,P=3` 总参数 5,606。fixed positional buffer 继续 `persistent=False`，不进入 state dict。不得增加 learnable position、patch residual norm、FFN、target/exogenous self-attention、Hidden-KV、top-k、variable embedding、第二 selector、PMCR/P2 或空间模块。

因为 `G_global` 含 `q_global` target shortcut，后续 T2G development 必须同时登记正常外生输入、固定 checkpoint 后 batch 内确定性 exogenous permutation、`A_global` 旁路，并分别观察 prediction、`A_patch`、`G_global` 与 gate；任何改善都不得未经该诊断直接归因于外生变量。

### 7.7.4 Config、checkpoint、artifact 与恢复合同

T2G variant 强制 `use_pmcr=False,use_teb=True,d=32,heads=4,dropout=0.1,gamma=1e-3`，patch size 显式配置，padding=`right_zero_crop`，position=`fixed_sinusoidal`。以下 T2G-only 字段必须进入 resolved/scientific/comparison config、checkpoint metadata、manifest candidate contract、resume mismatch 与 summarizer identity，但不得改变旧 Global v1/T2 historical identity：

```text
teb_global_residual = query_plus_attention_post_layernorm
teb_patch_attention_residual = none
teb_global_gate = scalar_per_patch
teb_global_gate_input = patch_attention_and_global_bridge
teb_global_gate_init = identity
teb_beta_global_init = 0.001
```

T2G 只允许 from scratch，或完全同结构的普通 `load_state_dict(strict=True)`。必须拒绝 Global↔T2G、T2↔T2G、partial new keys、patch/beta/gate/global-residual mismatch、`strict=False` 和所有 source-kind importer；完整 key、shape 与 config 必须在任何参数写入前通过，失败后 parameter/buffer 逐元素不变。T2G 使用独立 schema-v2 identity；summarizer 必须检查 candidate contract、13-file checksum 与重复科学身份，不得与 Global v1 或 T2 混分组。

第九轮完成 T2G 工程实现、真实单 batch production-gradient 门禁和测试。第十轮在同源码 T2-refresh 控制下完成 ETTm1 development：T2G 四个 test horizon 的 MSE/MAE 均轻微退化，test MSE/MAE macro 分别相对 T2 退化约 `+0.08982%/+0.06270%`，validation 接近数值平局，因此分类为 **negative-or-negligible development signal**。global/gate/beta 参数均实际移动，故该结果不能解释为“结构没有训练起来”；它只说明简单 global-mediated patch injection 在当前 ETTm1 development 配置下没有额外收益。T2G 保留为可追溯负向工程候选，但退出当前领先候选；M4 不再围绕 T2G 搜索 beta、gate、MLP 或更复杂 global injection。该证据不能扩大为所有数据集、所有 global/patch interaction 或 TimeXer global token 设计均无效，也不构成 M5 淘汰。


## 7.8 M4 最后一个 TEB 结构候选：T3 Selective Patch TEB

用户在 M4 第十一轮明确授权 T3；其固定工程身份为：

```text
candidate = T3
public class = SelectivePatchTargetExogenousBridge
implementation_variant = el-amd-m4-t3-selective-patch-teb-v1
ablation_id = M4_T3
teb_architecture = selective_patch_v1
```

T3 直接从 T2 Patch-Conditioned TEB 派生，不继承 T2G，也不包含 T2G 的 `q_global` residual、`global_bridge_norm`、global-to-patch injection、`beta_global` 或 `global_injection_gate`。它是当时 M4 规划中的最后一个 TEB 结构候选，不是最终 TEB、最终 EL-AMD 或 M5 frozen variant，也不覆盖 Global v1、T2、T2G。

### 7.8.1 唯一结构变化与 post-projection gate

T3 完整沿用 T2 的 non-overlap target patch、fixed sinusoidal position、whole-series exogenous variate tokens、patch/global queries、一次向量化 MHA、parallel owner mask、patch output projection、right-zero-pad/crop、`gamma_teb`、AMD 外层 residual、`exo_context` 与 `state_source`。仅对 T2 raw patch attention response 的投影结果增加共享的 scalar-per-patch confidence gate：

```text
Q_patch = T2 patch query
A_patch = T2 raw patch cross-attention response
D_patch = patch_output_projection(A_patch)

gate_input = concat(Q_patch, A_patch, dim=-1)
gate_logits = F.linear(
    gate_input,
    patch_confidence_gate_weight,
    patch_confidence_gate_bias,
)
g_patch = 2 * sigmoid(gate_logits)

D_effective = g_patch * D_patch
unpatch/crop(D_effective) -> delta
H_out = H + gamma_teb * delta
```

Gate 必须在含 bias 的 patch output projection **之后**相乘；禁止 `patch_output_projection(g_patch*A_patch)`，以确保 gate 接近 0 时同时抑制 projection weight contribution 与 bias。Gate 输入固定为 `[Q_patch;A_patch]`：`Q_patch` 表示目标局部状态，`A_patch` 表示来自其他变量的外生响应；gate 只决定外生 residual 写回量，不直接生成预测内容，不把 `Q_patch` 加入 residual，也不形成 target-only output shortcut。

Single shapes 为 `Q_patch/A_patch [B,N,d]`、`D_patch/D_effective [B,N,P]`、`g_patch [B,N,1]`；只更新 `target_idx`，其他通道逐元素不变。Parallel shapes 为 `Q_patch/A_patch [B,C,N,d]`、`D_patch/D_effective [B,C,N,P]`、`g_patch [B,C,N,1]`；所有变量产生 residual，一次 MHA 的 owner mask 仍为 `[C*(N+1),C]`，`target_idx` 只选择 global context，`C=1` 继续拒绝。所有变量和 patch positions 共享同一组 gate 参数；禁止 per-time/per-feature/per-variable 参数、两层 MLP、attention gate、第二 selector、top-k 或 Hidden-KV。

### 7.8.2 初始化、RNG 与参数合同

为保证 T2/T3 公共初始化和构造后 RNG 完全一致，gate 不得先构造随机初始化的普通 `nn.Linear` 再清零，而须显式创建：

```python
patch_confidence_gate_weight = nn.Parameter(torch.zeros(1, 2*d))
patch_confidence_gate_bias = nn.Parameter(torch.zeros(1))
```

正式计算使用 `F.linear`，因此初始 `gate_logits=0`、`g_patch=1`，相同 T2 base state 下 T3 prediction、MoE、state source、raw/effective patch residual 初始严格退化为 T2。新增参数只允许 `2*d+1`；`d=32` 时为 65，ETTm1 `T=512,P=32` 模块参数固定 39,426，UrbanEV `T=12,P=3` 固定 5,541。不得出现 T2G-only keys、patch norm/FFN、learnable position 或其他未授权组件。

### 7.8.3 Global query 与分析接口

T3 完全沿用 T2 的 `q_global -> A_global -> exo_context -> state_source` 路径，不让 global query 进入预测 residual。其专属 projection/norm 在 production forecast loss 下预计仍为零任务梯度；这不是 T3 实现失败，也不表示 `exo_context` 已受预测损失直接监督。该状态接口留到 M7 处理，本轮不实现 StateAdapter、patch-context pooling 或其他 global repair。

T3 必须能明确分析 `A_patch`、raw `D_patch`、`g_patch`、effective `D_effective` 与 global `exo_context`。`compute_patch_confidence_gate(Q_patch,A_patch)` 返回 gate；`compute_raw_patch_delta(A_patch)` 只返回未门控投影；`compute_effective_patch_delta(Q_patch,A_patch)` 返回 gate 后结果；正式 forward 只消费 `D_effective`，分析接口不得改变正常 forward。

### 7.8.4 Config、checkpoint 与 artifact identity

T3 强制 `use_pmcr=False,use_teb=True,d=32,heads=4,dropout=0.1,gamma=1e-3`，patch size 显式配置，padding=`right_zero_crop`，position=`fixed_sinusoidal`。以下 T3-only fields 条件进入 resolved/scientific/comparison config、checkpoint metadata、manifest candidate contract、resume mismatch 与 summarizer identity，不得改变 Global/T2/T2G historical identity：

```text
teb_patch_confidence_gate = scalar_per_patch_post_projection
teb_patch_gate_input = query_and_attention_response
teb_patch_gate_activation = two_sigmoid
teb_patch_gate_init = explicit_zero_identity
teb_global_prediction_role = state_only_forecast_disconnected
```

T3 训练只允许 from scratch；恢复只允许完全同结构 T3 的普通 `load_state_dict(strict=True)`。必须在写参前拒绝 Global/T2/T2G↔T3、partial gate keys、unexpected/shape/patch/gate-contract mismatch、`strict=False` 与所有 source-kind importer，失败后 parameter/buffer 逐元素不变。“从 T2 派生”只表示结构继承，不授权加载 T2 checkpoint。

### 7.8.5 第十一轮历史 endpoint 与第十二轮治理取代

第十一轮按当时预登记规则形成了历史判断：T3 为 `negative-or-negligible development signal`，T2G 同样为 `negative-or-negligible`，T2 因而是已测试 TEB 中的领先候选，并曾登记 `TEB branch reaches M4 candidate endpoint`。第十二轮用户最新决定正式取代该治理结论：历史指标和 artifact 继续有效，但 endpoint 撤销。T2 只能称为 `best among tested TEB variants`，尚未通过 M4 TEB development adequacy gate，也不是最终 TEB、M5 frozen TEB 或正式 EL-AMD variant。

TEB-first 的退出条件现固定为以下二者至少满足一项：

1. 一个 TimeXer-inspired TEB 候选在与其功能定位一致、同输入、同输出、同源码的 AMD control 下取得明确 `positive development signal`；
2. 用户明确停止当前 TimeXer-inspired 路线，并授权更换另一篇近三年外生变量模块来源。

在此之前不得启动 P2，不得把 PMCR 潜在收益用于掩盖 TEB 负收益，也不得把“若干负收益候选中最好的一个”解释为 TEB 已通过。

### 7.8.6 第十二轮 ETTm1 target-exogenous adequacy 协议

本轮不新增 TEB architecture，只用现有 T2 验证其 target–exogenous 功能定位。ETTm1 仍为 development-only benchmark，允许使用 train/validation/test，但结果不进入 M6 正式主表，也不构成未见测试集泛化证据。本协议不表示正式 ETTh1、Weather、ECL 或 Exchange 的任务模式改为 `target_exogenous`。

固定输入 schema 为：

```text
feature order = [HUFL,HULL,MUFL,MULL,LUFL,LULL,OT]
target = OT
target_idx = 6
aux_idx = [0,1,2,3,4,5]
aux_feature_names = [HUFL,HULL,MUFL,MULL,LUFL,LULL]
```

公平对照固定为：

```text
U1 = AMD-Concat; PMCR off; TEB off
U2 = AMD-Concat + T2 Patch-Conditioned TEB; PMCR off; TEB on
```

U1/U2 必须使用完全相同的七变量历史输入、feature order、OT target、aux 顺序、split、train-only scaler、窗口、seed、batch、optimizer、AMD 主干、输出范围、metric space 和 best-checkpoint 规则；仅对 OT 未来标签计算 loss 和指标。现有 generic runner 必须先通过 zero-code capability audit；若不能安全表达该合同，本轮停止且不实现 workaround。

本轮 U2 只有在以下条件全部满足时才记为 `positive development signal`：full-precision test MSE macro 低于 U1、test MAE macro 不高于 U1、至少 3/4 test MSE horizon 改善、validation MSE macro 不高于 U1，且收益不是舍入假象或单一 horizon 驱动。若仍非 positive，不得自动创建 T4/T5/T6、调整 patch size/d/heads/gate、启动 P2 或继续大规模 ETTm1 test 搜索；下一步只能由用户另行确认一个严格匹配计算预算的 warm-start/adapter-style rescue，或更换来源模块。

### 7.8.7 第十三轮 generic target-exogenous runner / provenance repair 合同

第十二轮 capability audit 的事实继续有效：generic ETTm1 `feature_type=M` 返回七变量标签，不满足 OT-only 任务；`feature_type=MS` 返回七变量历史输入与 `[B,H,1]` OT 标签，是唯一正确的数据语义；repair 前 production adapter 拒绝 prediction `[B,H,1]` 与 target `[B,H,1]` 的合法组合；standard U1 manifest 缺少公共 target/aux schema block；第十二轮没有训练、forward 或 artifact，TEB adequacy gate 仍未通过。第十三轮用户只授权最小 runner/provenance repair，不授权 U1/U2 训练、warm-start、新 TEB、P2、M5/M7 或空间模块。

对于 `task_mode=target_exogenous`，正式 prediction 固定为 `[B,H,1]`。合法 target 只允许以下两种，且 criterion 前 prediction 与 target 的 shape 必须逐元组完全相同：

```text
A. prediction [B,H,1] + target [B,H]
   prediction_for_loss = prediction.squeeze(-1)
   target_for_loss = target
   criterion shapes = [B,H] / [B,H]

B. prediction [B,H,1] + target [B,H,1]
   prediction_for_loss = prediction
   target_for_loss = target
   criterion shapes = [B,H,1] / [B,H,1]
```

除这两种外全部拒绝：prediction rank 不是 3、prediction 最后一维不是 1、target rank 不是 2/3、三维 target 最后一维不是 1、batch 或 horizon 不一致，以及任何依赖 PyTorch broadcasting 的组合。只允许 `squeeze(-1)`，禁止无维度 `squeeze()`；不得改变 target 的 rank、内容、顺序或数值，不得修改 `CustomDataLoader` 的 MS `[B,H,1]` 标签合同。train、validation 与 final test 必须复用同一个严格 adapter，shape 错误必须在 criterion、metric 或 inverse-transform 前拒绝。Generic MS 的 loss/MSE/MAE 仅聚合 OT 的全部元素且不得二次切片；UrbanEV 原有二维 `[B,H]` target 继续仅将 prediction 显式 squeeze 为 `[B,H]`，其 loss、metric 与目标 scaler 语义不变。`parallel_multivariate` 行为保持不变。

所有 repair 后新生成且满足 `enhanced variant AND task_mode=target_exogenous` 的 artifact，必须在 resolved scientific config 与 checkpoint metadata 中条件登记：

```text
target_exogenous_schema_contract_version = target_exogenous_schema_v1
```

completed manifest 必须在 seal/checksum 前写入并验证公共块：

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

示例索引只说明字段形式，实际内容必须来自已验证的运行时 schema。Generic ETT 以 `CustomDataLoader`/runtime preprocessing metadata、真实 feature order、真实 target indices 和 resolved aux schema 为事实来源；UrbanEV 以 M1 `UrbanEVFoldBundle`/`FeatureSchema` 的真实 target/aux/name/fingerprint 为事实来源。CLI/config 只作为 expected value 交叉核验，不得覆盖 runtime facts；任何不一致必须在训练或 artifact staging 前拒绝。索引序列序列化为 list；`target_indices` 在当前单目标合同中必须恰为 `[target_idx]`；`aux_idx` 保留调用方顺序，非空、有序、无重复、不含 target，names 必须逐项对应且所有索引在范围内。

ETTm1 U1/U2 的公共块固定为 `feature_type=MS`、`feature_names=[HUFL,HULL,MUFL,MULL,LUFL,LULL,OT]`、`target_feature_name=OT`、`target_idx=6`、`target_indices=[6]`、`aux_idx=[0,1,2,3,4,5]` 与对应六个 aux names；两者必须逐元素相同。U1 即使没有 architecture candidate contract 也必须具有该公共块；U2 同时具有公共块与自身 T2 candidate contract，二者不得互相替代。

schema 继续使用 schema-v2 的原目录、13-file checksum、staging/atomic publish 和 candidate contract，不升级 schema-v3。summarizer 必须以显式 version 区分新合同与 legacy：config 声明 v1 时，manifest block、version、path/task/target、feature/target/aux 顺序与 schema fingerprint 必须严格一致，任何缺失或篡改拒绝；历史 config 没有 version 时继续按原 legacy schema-v2 读取，不补写、不重算 hash，并显式标记 `legacy`。config 有 version 但 manifest 无 block、manifest 有 v1 block但 config 无 version、version 或字段不一致，以及 `parallel_multivariate` 错带 v1 contract，均必须拒绝。resume 不得跨越 legacy/v1 schema contract，且 mismatch 必须在参数写入前拒绝；旧 parallel scientific/comparison identity 不得因本 repair 改变。

该 repair 不改变 U1、T2 或 frozen AMD 的数学结构，不改变 ETTm1 数据、split、scaler、窗口或标签，不改变 UrbanEV M1 数据合同，也不构成 TEB performance evidence。repair 后仍须另轮、另经审核才能运行 U1/U2；本轮 TEB adequacy gate 仍未通过，P2 继续阻塞。


### 7.8.8 第十六轮 Frozen-AMD + Fresh-T2 warm-start adapter rescue 合同

本节是用户已正式锁定的 M4 训练协议，不是新的 TEB architecture。T2 数学结构和 public class 保持 `patch_conditioned_v1`；不得新增 TEB class、architecture 或 implementation variant。Adapter 和可选 matched-budget continuation 的身份分别固定为：

```text
Adapter:
implementation_variant = el-amd-m4-t2-patch-teb-v1
ablation_id = M4_T2_ADAPTER
training_protocol_id = m4_t2_u1_warmstart_frozen_adapter_v1
warm_start_contract_version = warm_start_contract_v1

Matched-budget continuation capability:
implementation_variant = el-amd-pmcr-teb-v1
ablation_id = M4_U1_CONTINUATION
training_protocol_id = m4_u1_matched_budget_continuation_v1
warm_start_contract_version = warm_start_contract_v1
```

Adapter 的 source 只允许对应 horizon 的第十四轮 completed U1 `best.pt`；source 必须为 PMCR off、TEB off、`ETTm1/MS/target_exogenous/OT`，`target_idx=6`、`target_indices=[6]`、`aux_idx=[0,1,2,3,4,5]`、`seq_len=512`、seed 2024、同 horizon 与同 data/schema fingerprint。目标为按 seed fresh 构造的 T2，不加载任何历史 T2/TEB 参数。四个唯一 source run、config hash、best epoch 与实际 64-hex checkpoint SHA 必须由 artifact 自身和 checksum 共同验证；h336 的实际 `best.pt` SHA 为 `89da2c854dce4e124c09575835d52e313bf3a6aeab2b556a060c7174709cb530`。

Source preflight 必须发生在 target staging、日志/provenance 写入和 optimizer 构造之前，且不得解析 source test 数值。它必须验证 completed schema-v2、精确 13-file checksum、系统 `sha256sum -c`、config/manifest/checkpoint/source/data metadata、dirty=false、source run/config/checkpoint/horizon/task/schema identity，以及 source state 无 `pmcr.*`/`teb.*`。全局 executable fingerprint 不相等时，不得只凭 strict load 接受；固定 `source_compatibility_proof_v1` 必须逐文件证明以下 critical source 与当前工作树相同：

```text
models/tsAMD.py
models/common.py
models/tsmoe.py
models/tsAMD_enhanced.py
models/modules/__init__.py
models/modules/patch_conditioned_target_exogenous_bridge.py
utils/dataloader.py
```

Compatibility proof 同时记录 source/current aggregate fingerprint、是否相等、critical file SHA、source/target/mapped key count、allowed missing、unexpected、shape mismatch 与 dtype mismatch。Adapter 映射固定为 source 60 keys、target 79 keys、映射 AMD 60 keys、allowed missing 精确为 fresh target 的 19 个 `teb.*` keys；continuation 为同结构 60→60。所有 metadata、key、shape 与 dtype 必须先完整验证，再从目标 fresh state 构造 merged state 并执行普通 `load_state_dict(strict=True)`；禁止 importer、`strict=False`、跨 horizon 或逐 key 边检查边写。失败必须恢复并以唯一 production helper 和逐 tensor `torch.equal` 共同证明完整 target parameter/buffer 未污染。

State mapping 与 rollback 的 digest 固定使用：

```text
state_digest_contract_version =
sha256_length_prefixed_state_dict_v1

key_policy =
exact_state_dict_keys_no_prefix_normalization
```

该 v1 输入仅允许 `Mapping[str, torch.Tensor]` 中的 dense strided、非 meta、非 quantized tensor；非字符串 key、非 tensor value、sparse/其他 layout、meta 或 quantized tensor全部拒绝。调用方传入的精确 key 不删除 `module.`/`teb.`、不补前缀、不重命名；key 按 UTF-8 bytes 升序。算法只在 little-endian production host 上定义，其他 host 明确拒绝。每个 tensor先 `detach()`、转 CPU、`contiguous()`；device 与 `requires_grad` 不编码。

固定 SHA-256 byte stream 为 UTF-8 header `sha256_length_prefixed_state_dict_v1\0`，随后是 big-endian uint64 key count。每个排序后的 key 依次写入 `LP(key_utf8)`、`LP(str(tensor.dtype).encode("utf-8"))`、big-endian uint64 rank、每一维的 big-endian uint64 dimension、`LP(raw_tensor_bytes)`；`LP(payload)=big-endian uint64 len(payload)+payload`，raw bytes 等价于 `detach().to("cpu").contiguous().reshape(-1).view(torch.uint8).numpy().tobytes(order="C")`。禁止使用 `torch.save(state_dict)`、pickle、JSON 数值序列、NumPy 默认字符串、tensor repr 或无明确 framing 的字符串拼接。

第十五轮临时审计值保留为 `legacy_unversioned_audit_digest`：保留 session 证据已恢复其实际 framing，即 key/dtype/raw length、rank 与 shape dimension 均使用 8-byte little-endian（dimension 为 signed），无 header/key count。第十六轮 repair 前的 production helper 则使用 key length=4-byte big-endian、dtype length/rank=2-byte big-endian、dimension/raw length=8-byte big-endian，也无 header/key count。相同 tensor 因上述 framing 与 endian 差异得到不同字符串，不表示 state 不同；历史未标版本的字符串不得与 production v1 做值相等判断。Fresh initialization digest 一般还受精确 RNG 起点与构造顺序影响，比较时必须另证这些条件一致。

当前 `source_compatibility_proof_v1` 不存储 source/mapped/target state digest，因此本合同不新增 artifact 文件、不升级 schema-v2、不扩张 summarizer sealed field set。Digest version/value只属于 mapping/rollback 与审计的 provenance/integrity evidence，不进入 architecture、training hyperparameter、scientific/comparison hash 或 duplicate identity。

Adapter 初始化顺序固定为：fresh T2 构造；只映射 source U1 AMD subset；使用 `torch.no_grad()` 将 `model.teb.gamma_teb` 精确置零；证明 CPU/全部 CUDA RNG 不变且除 gamma 外所有 T2 tensor 不变；再配置 freeze/mode/optimizer 和 epoch-0 validation。已有 architecture 字段 `teb_gamma_init=0.001` 继续表示 T2 constructor/default 合同；训练协议额外记录：

```text
gamma_initialization_policy = zero_after_fresh_t2_initialization
effective_teb_gamma_init = 0.0
```

Adapter 参数、buffer 与 module mode 固定为：AMD parameters frozen；AMD persistent buffers frozen；AMD/root module `eval()`；仅 T2 module `train()`；该 mixed mode 在每个训练 epoch 开头重新应用。由此 AMD BatchNorm buffer不更新，AMS selector Gaussian noise与 AMD dropout关闭，T2 MHA dropout保持训练态；validation/test仍使用全模型 eval。冻结只按参数名和 persistent buffer验证，不以“未加入 optimizer”替代。

Trainable allowlist 精确为以下 15 tensors / 22,881 parameters：

```text
teb.gamma_teb
teb.patch_query_projection.weight
teb.patch_query_projection.bias
teb.patch_query_norm.weight
teb.patch_query_norm.bias
teb.exogenous_projection.weight
teb.exogenous_projection.bias
teb.exogenous_norm.weight
teb.exogenous_norm.bias
teb.cross_attention.in_proj_weight
teb.cross_attention.in_proj_bias
teb.cross_attention.out_proj.weight
teb.cross_attention.out_proj.bias
teb.patch_output_projection.weight
teb.patch_output_projection.bias
```

Global-query-only 以下 4 tensors / 16,480 parameters 必须冻结且不进入 optimizer：

```text
teb.global_query_projection.weight
teb.global_query_projection.bias
teb.global_query_norm.weight
teb.global_query_norm.bias
```

Adapter optimizer 固定为 fresh Adam，只接收上述 15 tensors，learning rate=`3e-5`、weight decay=`0`、无 source optimizer/scheduler state。训练 objective 继续为 `prediction MSE + frozen selector auxiliary`，但 history 必须分别记录 prediction、auxiliary 和 total；永久测试必须证明 auxiliary 对 adapter trainable parameters 的 raw gradient 为常数零影响，即 total-loss 与 prediction-only 的 adapter gradient 在相同 RNG/input 下逐元素一致。Validation best 只依据 prediction MSE。

Epoch 0 表示 source-loaded、fresh-T2、effective gamma=0 的初始化 candidate，不表示已完成训练 epoch。它在训练前做完整 validation，进入 strict-improvement best selection，可保存为 `best.pt`/`last.pt`，其 `epoch_zero_checkpoint_role=source_equivalent_initialization`、`best_checkpoint_role=epoch_zero_initialization`。History 只允许 `1..completed_epochs`；`completed_epochs` 是实际执行的 adapter epoch 数，`best_epoch` 为 `0..completed_epochs`。相等不得替换 epoch 0；若所有训练 epoch 更差，最终 best 可保持 epoch 0，最终 test 仍只在全部预算结束并加载 best 后执行。Max adapter epochs=`10`，无 early stopping。

Warm-start run 自身失败后可以 resume 自己的 hidden staging：必须严格恢复该 target run 的 `last.pt`、model/optimizer/RNG/generator/history/best/epoch-0 metadata，不重新打开或映射 source artifact，不重新 zero gamma；last epoch 0 从 epoch 1 开始，last epoch k 从 k+1 开始，best epoch 0 可继续保留。Completed source U1 不得用 `--resume` 冒充 initialization。Standard resume、standard best 从 epoch 1 开始及其 artifact字段保持不变。

Warm artifact 继续使用 schema-v2、原路径后缀和 13-file checksum。受 checksum 保护的 config/checkpoint/manifest 写入完整 training protocol、稳定 source lineage 与 compatibility proof；runtime/manifest 可记录机器相关绝对 `source_artifact_path`，但该 path 不进入 scientific/comparison hash，summarizer也不得要求原机器路径可访问。稳定 lineage 至少包含 source run/variant/ablation/checkpoint role+SHA/config+comparison hash/commit/executable+data fingerprint/best epoch/task/feature/target/indices/schema。Standard from-scratch artifact 不携带 warm-start block，也无需迁移旧 artifact。

Scientific identity 固定包含 max epochs、stopping/epoch-zero/source/protocol/optimizer/freeze/mode/parameter scope/adapter seed/gamma/objective等预先决定的合同；comparison identity只移除 adapter seed，duplicate identity使用 `comparison_config_hash + seed`。以下 runtime outcome **不得**进入 scientific/comparison/duplicate identity：`completed_epochs`、`best_epoch`、best role、wall-clock、artifact size和最终指标。同 protocol/source/seed 即使 completed epochs 不同仍是 duplicate，不得借此绕过拒绝。

Matched-budget U1 continuation 本轮只提供 production capability，不自动执行：对应 horizon U1 best作为同结构source，PMCR/TEB off，普通 strict 60→60，fresh Adam、全部 AMD parameters、lr=`3e-5`、wd=`1e-7`、full model train、epoch 0纳入best、额外 epoch按1..10重新编号、无 early stopping；不恢复 source optimizer/epoch/history，不由 adapter runner 自动触发。

Rescue 的第一阶段固定为四 horizon各一个 adapter run。只有其相对固定 U1 形成 `provisional positive adapter signal`，才允许用户另行授权四个 matched-budget continuation；最终 TEB adequacy pass 必须等待 continuation 后裁决。No-op frozen control不运行。一次预注册 rescue 仍非 positive时，正式停止当前 TimeXer-inspired路线；不得自动运行 R-min、continuation、P2或任何新 TEB。该段前两句保留预登记合同：第十六轮只实现并验证 production capability，第十七轮已完成四个 adapter run 且未形成 provisional positive，因而未运行 continuation 并正式停止 TimeXer-inspired 路线；M4 保持 In Progress，TEB adequacy gate remains failed。

# 7A. CrossLinear-inspired CCE 历史候选与失败证据（Early CCE v1）

TimeXer-inspired Global/T2/T2G/T3/rescue 已按第十七轮结果触发失败停止线；CrossLinear 随后曾由用户锁定为替代来源，本节保留其 Early CCE 独立工程候选的来源、公式、身份、实现与实验合同。Early CCE 与后续 Late CCE 均已失败，本节不再表示当前已选定外生模块候选。来源身份固定为：

| 字段 | 锁定值 |
|---|---|
| paper_title | `CrossLinear: Plug-and-Play Cross-Correlation Embedding for Time Series Forecasting with Exogenous Variables` |
| conference | `KDD 2025` |
| DOI | `10.1145/3711896.3736899` |
| PDF SHA-256 | `45557c426ca8bfa88f35ec41f09fd87ab864c9a382eef1c659c2296a4a1b0152` |
| official_repo_url | `https://github.com/mumiao2000/CrossLinear.git` |
| official_repo_commit | `d22366e2f59ced560a02b2b1c7cc673e3c02a13f` |
| official_model_sha256 | `a062ac97231c55384c621f27981b8225bb87822f50704df201b381dd8e037593` |
| retained_component | `cross_correlation_embedding_only` |

论文原始 many-to-one 公式为：

```text
X_cross = Conv1D(Stack(X_exogenous_normalized, X_endogenous_normalized))
X_embedding = alpha * X_endogenous_normalized
            + (1 - alpha) * X_cross
```

官方代码在 `features=MS` 时把最后一列作为 endogenous target，使用 `Conv1d(dec_in,1,kernel_size=3,padding='same')`；在 `features=M` 时使用 `Conv1d(C,C,kernel_size=3,padding='same')`。官方完整模型另含自身的 instance normalization/de-normalization、可学习 `alpha/beta`、patch/value embedding、learnable positional embedding 与 forecasting head。CCE v1 不复制官方源码，只保留“单层 k=3 跨变量卷积嵌入”这一来源组件，并按 AMD 接口独立重参数化。

## 7A.1 输入、插入点与禁止重复职责

CCE 固定输入是 AMD 已完成 RevIN 后的：

```text
x_ch = normalized_input.transpose(1,2)  # [B,C,T]
```

唯一插入点是：

```text
normalized_input -> transpose -> CCE -> pastmixing/MDM
```

历史 Early CCE 实现路径固定为：

```text
RevIN -> CCE -> MDM -> DDI -> PMCR? -> AMS
```

CCE 内不得再次 normalization，也不得复制 CrossLinear patch embedding、positional embedding 或 forecasting head。AMD 的 RevIN 继续唯一负责输入 normalization/de-normalization，MDM/DDI/AMS 继续负责既有时间分解、channel mixing 与预测；CCE 只负责 RevIN 空间内、进入 MDM 前的局部跨变量线性相关注入。不得修改 `models/tsAMD.py`，不得把 CCE 放到 DDI、PMCR 或 AMS 后。

该历史候选的 PMCR 与全部旧 TEB 固定关闭。旧 TEB 文件、测试、checkpoint、artifact 与数学结构保持不变；CCE 不产生 `teb.*` 或旧 `exo_context`。

## 7A.2 固定 gate、卷积与 identity residual

全局共享 gate 固定为：

```text
lambda = sigmoid(logit(0.1) + rho)
rho = one learnable scalar Parameter, initialized exactly to 0
effective lambda at initialization = exactly 0.1
```

禁止 clamp、无界直接 gate、per-channel gate 或 per-output gate。卷积合同固定为：

```text
kernel_size=3
stride=1
padding=1
dilation=1
groups=1
padding_policy=zero_same
bias=True
parameterization_policy=identity_residual_delta_v1
```

CCE 参数是直接创建的零 `delta_weight` 与零 `delta_bias`，不得先调用随机初始化再清零并静默消费 CPU/CUDA RNG。初始化时 `delta=0`，因此 CCE 输出必须与输入逐元素 `torch.equal`，并使 AMD prediction、MoE loss 和 `state_source` 与 matched CCE-off control 严格相等。

`target_exogenous` 固定按调用方提供的 `ordered aux_idx followed by target_idx` 聚合卷积输入：

```text
source_idx = [*aux_idx, target_idx]
delta_target = Conv1d(x_ch[:,source_idx,:], C_source -> 1, k=3)
output_target = input_target + lambda * delta_target
```

只写回 `target_idx`；其他通道必须逐元素不变。参数量 golden 为：

```text
3 * C_source + 2
# delta_weight + delta_bias + rho
```

`parallel_multivariate` 使用原 feature schema 顺序：

```text
delta_all = Conv1d(x_ch, C -> C, k=3)
output = x_ch + lambda * delta_all
```

参数量 golden 为：

```text
3 * C * C + C + 1
# delta_weight + delta_bias + rho
```

两种模式的外部输入/输出都保持 `[B,C,T] -> [B,C,T]`。target 模式内部卷积为 `[B,C_source,T] -> [B,1,T]`，参数和主要 MAC 复杂度分别为 `O(3*C_source)` 与 `O(3*B*T*C_source)`；parallel 模式内部卷积为 `[B,C,T] -> [B,C,T]`，参数和主要 MAC 复杂度分别为 `O(3*C^2)` 与 `O(3*B*T*C^2)`。因此 ETTm1 `T=512,C=7` 的 target 路径计算随长序列线性增长，但 k=3 仍只表达局部滞后；UrbanEV `T=12` 时两端 zero padding 的边界占比更高，必须保留端点测试；ECL `C=321` 的 parallel 路径单模块即有 `309,445` 个参数并呈二次 channel 成本，而 target 路径若使用全部 321 个 source 只有 `965` 个参数。任何高维 parallel 正式运行都需在后续授权轮单独评估显存、吞吐与收益，本轮不得据 capability smoke 宣称可扩展性或性能。

模块必须提供只读分析接口，至少返回 effective lambda、ungated delta，以及与同一次 forward 等价的 CrossLinear-style kernel：

```text
equivalent_weight = selector_identity_kernel + lambda * delta_weight
equivalent_bias = lambda * delta_bias
```

target 模式的 selector identity 是目标输入在 center tap 上的 1，并按 source order/scatter 明确表达；parallel 模式是每个输出通道对应输入通道 center tap 的单位矩阵。分析接口不得改变参数、buffer 或 RNG。

## 7A.3 模式、索引与状态接口

`target_exogenous + CCE on` 要求 `aux_idx` 非空；`parallel_multivariate + CCE on` 要求 `C>=2`。target/aux 索引必须拒绝 bool、重复、越界及 target 出现在 aux，并严格保留 aux 明确顺序。target 可位于 feature schema 首、中、末任意位置，禁止依赖官方 target-last 假设。

`CCE off` 固定：

```text
self.cce = None
no cce.* state_dict keys
forward strict bypass
```

F0/单变量 `C=1` 不得伪装为 CCE enabled。当前 `state_source` 的 `u_target` 与 `v_target` 正常反映 CCE 对 AMD 主干的间接影响；第三段固定为 dtype/device 正确的确定性零张量，其语义登记为：

```text
legacy_width_compatibility_zero
```

该零段只维持冻结宽度接口，不是 CrossLinear-derived context，不得输出、命名或声称存在独立 CCE context。它对未来 M7 的影响仅登记为 state source 缺少独立外生 context；本轮不设计或实现 M7。

## 7A.4 Runner、artifact 与科学身份

历史候选 implementation variant 固定为：

```text
el-amd-m4-crosslinear-cce-v1
```

runner 必须原生解析并封存：

```text
use_cce
cce_kernel_size
cce_lambda_init
cce_padding_policy
cce_input_order_policy
cce_parameterization_policy
```

v1 强制 `kernel_size=3`、`lambda_init=0.1`、`padding_policy=zero_same`、`parameterization_policy=identity_residual_delta_v1`；target 模式 input order 为 `ordered_aux_then_target`，parallel 模式为 `feature_schema_order`。config、checkpoint、manifest 必须同时封存本节来源身份、retained component、插入点、mode/order/kernel/stride/padding/dilation/groups/bias、lambda transform/raw/effective init、normalization reuse policy 与 state zero-placeholder policy。

上述稳定来源与 CCE 合同进入 scientific identity。机器绝对参考仓库/PDF 路径仅可作为本地审计信息，不得进入 comparison identity；repo URL、commit 与 SHA 等稳定字段必须进入。artifact 继续使用 schema-v2、现有 13-file checksum、hidden staging 与同文件系统 atomic publication；不得新增 schema-v3 或第 14 个文件。

summarizer 必须把旧 standard AMD/U1、旧 TimeXer TEB variants、T2 adapter/continuation、新 CCE control 与新 CCE candidate 分开，并拒绝 CCE source/config/order/gate tamper、manifest/config/checkpoint 不一致及 `comparison_config_hash + seed` duplicate spoof。

## 7A.5 From-scratch development pair

development 身份固定为：

```text
development_protocol_id = m4_crosslinear_cce_from_scratch_pair_v1

M4_CCE_CONTROL:
    CCE off
    PMCR off
    all TEB off

M4_CCE:
    CCE on
    PMCR off
    all TEB off
```

两者均使用 standard from-scratch 训练身份：全部 AMD/CCE parameters trainable；fresh Adam；learning rate=`3e-5`；weight decay=`1e-7`；standard best 从 epoch 1 开始，不使用 epoch-0 best；不使用 source checkpoint、T2 warm-start protocol 或历史 TEB lineage。同一 run 的 resume 必须 strict same-structure，并恢复其自身 model/optimizer/RNG/generator/history/best。

第十九轮已按该锁定合同完成 ETTm1 四 horizon 的 4 个 control + 4 个 Early CCE paired runs。其科学合同曾由用户锁定且前置 implementation closure 已完成，因此属于可直接执行的合同内预登记实验，无需逐个 run 或逐轮重新授权；实际负向裁决见第 7B 节导语。

## 7A.6 Checkpoint 与原子 importer

CCE 关闭时保持现有严格恢复行为。同结构 CCE resume 只允许普通 `load_state_dict(strict=True)`。从 AMD/U1 state 初始化 CCE-on 模型只能调用专用 importer，并要求：

```text
missing == complete current cce.* key set
unexpected == empty
```

importer 在写参前必须校验完整 key set、shape、dtype、task mode、C、feature schema/order、target_idx、ordered aux_idx 与 schema fingerprint。成功时只映射公共 AMD state，fresh CCE state 逐 tensor不变；任一失败时全部 parameter 与 persistent buffer 原子不变。禁止 `strict=False`、partial `cce.*`、任何 `teb.*` 权重复用、target/parallel 间迁移、不同 C/schema/feature order 间迁移，以及把 T2 warm-start lineage 用于 CCE。

## 7A.7 梯度与阶段停止线

零 delta 初始化时任务损失对 `rho` 的第一次 backward 梯度为零，这是乘法结构的预期；由于 effective lambda 初始非零，forecast-connected delta weights（包括 target 模式的 auxiliary taps）在非退化真实 batch 上必须获得 finite nonzero gradient，公共 AMD gradient 必须与 matched CCE-off control 严格相等。第一次 optimizer step 后公共 AMD 参数须与 matched control 严格相等，delta 参数移动，`rho` 不得因 task gradient或 weight decay 漂移；构造 synthetic 非零 delta 后，第二次 backward 必须证明 `rho` 可获得 finite nonzero gradient。

Early CCE 未通过 M4 adequacy gate，后续 Late CCE 也按第 7B.4 节结果失败，因此当前 CrossLinear-inspired CCE 路线已触发有限开发停止线。capability、smoke 与 development 结果均不得称为最终外生模块、最终 EL-AMD 或正式论文性能。用户随后已明确选择 Sonnet/MVCA 作为当前下一来源候选；该选择不恢复 CCE，也不授权实现 Sonnet、启动 XLinear、PMCR/P2、M5/M7 或空间模块，精确合同须经第二十一轮审计与 ChatGPT 审核后另行闭环。

# 7B. CrossLinear-inspired CCE 历史候选与失败证据（Late CCE）

第十九轮 Early CCE v1（RevIN 后、MDM 前）四 horizon paired development 已完成，按预登记六项 adequacy gate 判定为 `negative-or-negligible development signal`：development test MSE/MAE macro 与 validation MSE macro 均退化，test MSE 改善为 0/4 horizon。该结果和八个 Early CCE artifact 保持不变，不得覆盖、改写或用 Late CCE 身份恢复。

用户当时已授权 Late CCE 作为 CrossLinear 路线最后一个有限插入位置候选；本节保留该历史候选的完整工程身份、公式、实现、artifact 与裁决证据。其工程身份固定为：

```text
implementation_variant = el-amd-m4-crosslinear-late-cce-v1
control ablation_id = M4_LATE_CCE_CONTROL
candidate ablation_id = M4_LATE_CCE
development_protocol_id = m4_crosslinear_late_cce_from_scratch_pair_v1
cce_architecture = crosslinear_inspired_hidden_state_late_cce_v1
cce_insertion_point = post_pmcr_pre_ams
cce_input_representation = amd_hidden_v_local
```

该身份在执行时也只属于 M4 工程/开发候选，从未成为最终外生模块、最终 EL-AMD 或 M5 冻结结果；现仅作为失败历史身份保留。

## 7B.1 精确路由、数学与来源边界

Late CCE 复用同一个独立实现 `CrossCorrelationEmbedding`，不新建第二个 Late class。插入位置由 `AMDEnhanced` 显式路由字段决定：

```text
normalized_input = RevIN(x)
x_ch = transpose(normalized_input)
u_mdm = MDM(x_ch)
v_ddi = DDI(u_mdm)
v_local = PMCR(v_ddi) if enabled else v_ddi
v_final = LateCCE(v_local) if enabled else v_local
prediction, moe_loss = AMS(experts_input=v_final, selector_input=u_mdm)
```

standalone Late CCE development 中 PMCR 与全部 TimeXer TEB 固定关闭，因而 `v_local == v_ddi`；`post_pmcr_pre_ams` 仍是体系结构位置。现有 CCE+PMCR coexistence guard 本轮保持，不解除、不重写。Late CCE 不改变 `x_ch`、`u_mdm`、`v_ddi` 或 AMS selector，只修改进入 AMS experts 的 `v_final`。

数学继续完全复用 CCE v1：kernel 3、stride 1、zero-same padding 1、dilation 1、groups 1、bias true；`lambda=sigmoid(logit(0.1)+rho)`，全局共享 scalar `rho` 初始化为 0；`delta_weight/delta_bias` 直接零初始化且 RNG-neutral。固定正号公式为：

```text
source_hidden = gather(v_local, ordered [aux_idx..., target_idx])
delta_target = Conv1d(source_hidden, C_source -> 1, k=3)
cross_target = target_hidden + delta_target
target_new = target_hidden + lambda * (cross_target - target_hidden)
           = target_hidden + lambda * delta_target
```

禁止实现负号。target_exogenous 只写回 `target_idx`，所有非目标 hidden channel 逐元素不变；parallel_multivariate 继续以 feature schema 原顺序执行 `C -> C`。不得增加 normalization、patch、PE、attention、额外 gate、FFN 或 CrossLinear forecasting head。

来源表述只能是 `CrossLinear-inspired hidden-state / late cross-correlation embedding adaptation`。Late kernel 的 lag `-1/0/+1` 表示 AMD 隐状态时间位置之间的局部相关修正，不得表述为原始物理变量数值的一步 lead-lag，也不得声称与原版 CrossLinear 的输入和插入方式完全相同。

## 7B.2 State source、参数与恢复隔离

Late 路线保持冻结宽度接口：

```text
state_source = concat(
    v_final[:, target_idx, :],
    u_mdm[:, target_idx, :],
    legacy_width_compatibility_zero,
)
```

第一段反映 Late CCE 后的 target hidden；第二段保持原始 `u_mdm`；第三段仍是 dtype/device 正确的确定性零占位。总宽度不变，不得称零段或 CCE 为独立 `exo_context`，本轮不设计 M7 StateAdapter。

Late target 参数量仍为 `3*C_source+2`，parallel 为 `3*C*C+C+1`。CCE-off 固定 `self.cce=None` 且无 `cce.*` keys；CCE-on 固定恰有 `cce.delta_weight`、`cce.delta_bias`、`cce.rho`。初始化 output、AMD prediction、MoE、selector input 与 state_source 必须和 matched control 位级相等；第一次 production backward 要求 aux delta taps finite/nonzero、target taps finite、`rho.grad==0`，公共 AMD 与 selector 路径 gradient 和 control 位级相等。

以下字段必须同时封存到 resolved/scientific/comparison config、checkpoint 内嵌 metadata、manifest candidate contract、resume mismatch 与 summarizer：

```text
cce_architecture
cce_insertion_point
cce_input_representation
```

即使 Early/Late 的 `cce.*` key 和 shape 相同，也必须在写入参数前依据 variant、route、mode、schema/order 与 candidate identity 拒绝交叉恢复；普通 `strict=True` 能读取 tensor 不代表科学结构兼容。Late same-structure resume 才允许 strict restore。control→candidate、target↔parallel、不同 C/schema/feature/target/aux order、partial/unexpected/shape/dtype mismatch 都必须原子拒绝。Late 是 standard from-scratch pair，不得继承 AMD/U1/T2 adapter checkpoint 或 lineage。

## 7B.3 八 run development 与停止线

ETTm1 development-only 实验固定为 horizon `96/192/336/720`，每个 horizon 顺序运行 `M4_LATE_CCE_CONTROL` 后 `M4_LATE_CCE`。共同合同逐字段复用第十九轮 Early pair：MS/OT、target_exogenous、feature order `HUFL,HULL,MUFL,MULL,LUFL,LULL,OT`、target 6、ordered aux 0--5、seq_len 512、seed 2024、10 epochs、batch 128、Adam lr `3e-5`、weight decay `1e-7`、`n_block=1,alpha=0,mix_layer_num=3,mix_layer_scale=2,patch=16,norm=true,layernorm=true,dropout=0.1`；PMCR/TEB off，全部参数 from scratch，best 从 epoch 1 开始，无 source/warm-start/adapter/continuation/epoch-0 best。

artifact 使用独立 root：

```text
artifacts/m4-development/ettm1-stage-h-crosslinear-late-cce-v1
```

继续使用 schema-v2、13-file checksum、hidden staging 与 atomic publication，不创建 schema-v3/第 14 个文件，不覆盖第十九轮 artifact。summarizer 必须把 Late control/candidate 与 Early CCE、旧 standard/TEB/adapter/continuation 分开，并拒绝 route/source/config/checkpoint tamper 与 duplicate scientific identity。

adequacy gate 不得事后改变，只有六项全部满足才是 `positive development signal`：test MSE macro 更低、test MAE macro 不高、至少 3/4 horizon test MSE 改善、validation MSE macro 不高、改善超过舍入噪声且不由单一 horizon 驱动。任一失败即登记 `negative-or-negligible development signal`，并正式停止：

> 当前 CrossLinear-inspired CCE 路线在 M4 有限开发中失败。

该段记录实验前预登记的裁决边界：失败后不得自动调 kernel/lambda/gate、再换插入点、转向 Sonnet/XLinear、启动 PMCR/P2 或进入 M5；即使当时取得 positive，Late CCE 也只会成为 CrossLinear 路线的 leading development candidate，不会自动进入 M5 或启动 P2。实际结果与当前停止状态见第 7B.4 节。


## 7B.4 实际 paired development 裁决

Late CCE production capability 已由 commit `43403f6c7f38b06a6cb5b62eb6f554c9ac215c9b`（parent `26f285d8e4dc0b9f250584cadefc906fb5abf006`）提交并推送；local/tracking/live remote 三端闭环后，才从该 clean HEAD 启动固定八 run。四 horizon control/candidate 均完成 schema-v2、history 1--10、13-file checksum 与原子发布；source fingerprint 为 21-file `adba794cdbc03b6d83a7c89f40d95bb5bf8163d2d32e23d530deff674e566005`。

ETTm1 结果只用于 development，不是正式论文结果。matched-control 对比中，Late CCE 的 development-test MSE macro 为 `0.047978709147`，高于 control `0.047930255147`（`+0.101093%`）；test MAE macro 为 `0.164059503280`，高于 `0.164051297863`（`+0.005002%`）；test MSE 改善为 `0/4` horizon；validation MSE macro 为 `0.079832489453`，高于 `0.079681071679`（`+0.190030%`）。六项预登记 adequacy 条件全部失败，h720 test MAE 的单项 `-0.033576%` 不改变总体裁决。

因此 Late CCE 判定为 **negative-or-negligible development signal**，并正式执行停止线：

> 当前 CrossLinear-inspired CCE 路线在 M4 有限开发中失败。

Early 与 Late CCE 的工程实现、永久测试和 development artifact 作为可复核负向证据保留，但不升级为最终外生模块、最终 EL-AMD 或 M5 冻结结构。不得继续调 kernel/lambda/gate 或再换 CCE 插入位置。用户随后已明确选择 Sonnet/MVCA 作为当前下一来源候选，XLinear 尚未被选择；Sonnet/MVCA 的具体范围、插入点、训练协议与 variant 仍待审核和后续 canonical 精确合同闭环，在此之前不得实现，也不得启动 PMCR/P2 或进入 M5。M4 继续保持 In Progress。

# 8. 历史 M3 与历史 CCE 候选的 forward / 时间状态接口

## 8.1 历史 M3 前向流程

本小节只保留 `el-amd-pmcr-teb-v1` 的历史工程合同，不代表当前存在已选定的 M4 外生模块候选。

```python
x_norm = RevIN_norm(x)
# [B,T,C]

x_ch = x_norm.transpose(1,2)
# [B,C,T]

u_mdm = MDM(x_ch)
# [B,C,T]

v = u_mdm
for block in DDI_blocks:
    v = block(v)

if use_pmcr:
    v = PMCR(v)

v_local = v
exo_context = v_local.new_zeros((v_local.shape[0], teb_context_dim))

if use_teb:
    v_final, exo_context = TEB(
        hidden=v_local,
        normalized_input=x_norm,
    )
else:
    v_final = v_local

pred_all_norm, moe_loss = AMS(v_final, u_mdm)
pred_all = RevIN_denorm_all(pred_all_norm)
pred = select_target_or_all(pred_all, task_mode, target_idx)
```

固定双输入语义：

```text
AMS experts  <- v_final
AMS selector <- 原始 u_mdm
```

不得让 DDI 重新接回 `x_ch`，不得把 PMCR/TEB 后的表示送入 selector，也不得先切出单通道再调用通用 RevIN denorm；单目标必须先完成全通道反归一化，再按 `target_idx` 选择 `[B,H,1]`，parallel 输出保持 `[B,H,C]`。

不得再出现：

```text
h = x_ch 后送入 DDI
```

## 8.2 历史 M3 return_state_source 与后续 StateAdapter

第三章增强模型只公开 M0-B 已冻结的原始状态源接口，不直接返回未训练的 `StateProjection` 或 `H_time`：

```text
v_target     = v_final[:,target_idx,:] # [B_region,T]
u_target     = u_mdm[:,target_idx,:]   # [B_region,T]
exo_context  = TEB 目标上下文或确定性零张量
             # [B_region,teb_context_dim]
state_source = concat(v_target,u_target,exo_context)
             # [B_region,2*T+teb_context_dim]
```

调用合同固定为：

```python
pred, moe_loss, state_source = model(
    x,
    return_state_source=True,
)
```

`target_idx` 必须显式提供并经过范围校验；TEB 关闭时，`exo_context` 必须是 dtype/device 正确的确定性固定零张量。`return_state_source=True` 只增加返回值，不得改变预测或 MoE loss；默认调用仍返回 `(pred, moe_loss)`。

M3 仍不创建 `StateProjection`、`StateAdapter` 或 `H_time`。到 M7 的第四章 Graph Mode 才允许新增并训练独立 StateAdapter，以 `state_source` 为输入：

```text
s_v = Linear_v(LayerNorm(v_target))
s_u = Linear_u(LayerNorm(u_target))
s_e = Linear_e(exo_context)
H_region = MLP(LayerNorm(concat(s_v,s_u,s_e)))
H_time = reshape(H_region,[B,N,state_dim])
y_time = reshape(pred,[B,N,H_out])
```

该 StateAdapter 必须属于第四章模型及其 checkpoint，并参与训练；不得把随机、未训练投影冒充第三章 EL-AMD 输出。推荐 `d_s=16`、`state_dim=32`，最终值只依据训练/验证集确定。

## 8.3 历史 Early CCE v1 forward 与 fixed-width state

`el-amd-m4-crosslinear-cce-v1` 的历史实现插入顺序固定为：

```text
normalized_input = RevIN_norm(x)
x_ch             = transpose(normalized_input)
x_cce            = CCE(x_ch)
u_mdm            = MDM(x_cce)
v_final          = DDI(u_mdm)
                  -> PMCR?  # 该历史 CCE pair 固定 off
                  -> TEB?   # 该历史 CCE pair 固定 off
pred             = AMS(v_final, selector=u_mdm)
```

CCE-off control 在同一位置严格旁路，`self.cce=None` 且无 `cce.*` state。CCE-on 的零 delta 初始化必须使 `x_cce` 与 `x_ch` 逐元素相等；因此 matched control 与 candidate 的 prediction、MoE loss 和 state source 在初始化时严格相等。

该历史 Early CCE 路线使用冻结宽度接口：

```text
u_target = u_mdm[:,target_idx,:]
v_target = v_final[:,target_idx,:]
legacy_width_compatibility_zero =
    v_final.new_zeros([B,teb_context_dim])
state_source =
    concat(v_target,u_target,legacy_width_compatibility_zero)
```

`u_target` 与 `v_target` 可以承载 CCE 的间接影响；第三段只维持历史宽度和 dtype/device 合同，不是 CrossLinear-derived context 或独立 CCE context。M7 之前不得据此设计、创建或训练 StateAdapter。

# 9. 第三章实验设计

原 v2.1 把 UrbanEV 的 `AMD-V/AMD-Concat` 消融直接套到 Weather/ECL，任务语义不一致。本替代版将实验拆成两种协议。

本节及后续第四章规划中未带 variant 后缀的 “EL-AMD”，统一表示由 M5 最终冻结的第三章增强时间模型。EL-AMD 是项目名称或模型族名称，不等同于任何具体候选。在 M5 之前，当前 M3 实现必须写作 `el-amd-pmcr-teb-v1`；后续新结构必须使用各自新的候选 variant；不得用无后缀 EL-AMD 暗示某个候选已经入选。

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
| U4 | U1 + Global TEB v1 + PMCR v1 | M3 v1 完整工程候选；最终结构由 M5 决定 |

U0 固定使用 F0：`feature_names=(volume,)`、`target_idx=0`、`aux_idx=()`。U1-U4 必须固定使用同一套非空辅助 schema：默认使用 F4，或只依据训练/验证集预先选定一个 F1-F4 preset；一旦锁定，不得在 U1-U4 之间更换。四组必须统一 feature/target/aux 名称与顺序、schema fingerprint、fold、split、scaler、horizon、seed、训练预算和评价流程。

模块归因固定为：

- `U2 - U1`：TEB；
- `U3 - U1`：PMCR；
- `U4 - U1`：PMCR v1 + Global TEB v1 完整候选组合。
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

F0 的 compact tensor 只有 `volume`，因此 `target_idx=0`、`aux_idx=()`；它只用于 AMD-TargetOnly 和可选的 AMD-TargetOnly + PMCR，不得标记为 AMD-Concat、TEB 或 EL-AMD。F0 上的 EL-AMD 结果记为 N/A。

F1-F4 的 compact tensor 均保持表中 canonical 顺序，目标 `volume` 位于第 0 通道，`aux_idx` 是其余通道的有序索引且非空。每个 preset 内 AMD-Concat 与 EL-AMD 必须使用完全相同的变量、顺序、fold、split、scaler、horizon、seed 和评价流程。

报告拆为两个 panel：

- Panel A：F0 的 target-only 对照；
- Panel B：F1-F4 的同输入 AMD-Concat 与 EL-AMD 对照。

## 9.5 模块验收线

每个时间模块只有在 M5 公平筛选中满足全部条件后才能最终保留：

1. 相对同输入 AMD 的验证指标平均退化不超过 0.5%；该值仅是安全底线，不能单独构成保留依据；
2. 至少在一个核心场景和一个独立场景出现稳定改善：
   - TEB：UrbanEV/EPF 中至少两项；
   - PMCR：UrbanEV + Weather/ECL/ETTh1 中至少两项；
3. 3 个锁定 seed 下方向基本一致；
4. 参数量和耗时增幅与收益相匹配；
5. 达到在 M5 筛选前由用户锁定的 practical-effect threshold；当前阈值尚未确定。

M4 validation 诊断不是正式性能验收。若候选不满足最终条件，不能简单关闭模块后仍宣称满足“两模块”要求，必须执行第 21 节的来源保持型变体或替换流程。

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

本节工程实现归属 M8。`adj.csv` 方向性处理、train-only DTW 需求图、地理—需求双图与 HSTGCN-core 均不得在 M4-M7 提前实现。

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

本节工程实现归属 M9；不得在 M4 时间模块诊断阶段提前实现。

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

本节工程实现归属 M10；不得在 M4 时间模块诊断阶段提前实现。

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

本节完整组合在 M10 形成工程闭环，并在 M11 执行第四章正式实验与定稿。

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
- UrbanEV、EPF-PJM、ETTh1、Weather、ECL、Exchange 的测试集只在模型结构、变量和超参数冻结后运行；ETTm1 按第 0.1 节的 development-only 例外治理。

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
| M3 | TEB | AMD-Concat 公平对照、parallel mode、zero context 测试通过；工程闭环不等于性能通过 |
| M4 | 时间模块诊断与候选迭代 | 保留 TimeXer 与 CrossLinear 失败证据；Sonnet S2/MVCA residual 精确 development 合同已锁定，production capability 与 ChatGPT implementation review 已 Passed，implementation Git closure Pending、paired development Not started、adequacy Not evaluated；XLinear 未选择，不解锁 PMCR/P2/M5 |
| M5 | 模型筛选与结构冻结 | 公平多数据集、多 seed 验证；锁定 practical-effect threshold、最终时间结构与正式 variant |
| M6 | 第三章正式实验与定稿 | 结构冻结后运行正式 test；完成主表、消融、效率与第三章定稿 |
| M7 | 时间状态接口与 Graph Mode | 训练 StateAdapter；`H_time [B,N,d]`、target-only output、适配后一致性测试通过 |
| M8 | HSTGCN-core 与双图构建 | 图归一化、官方地理图、train-only DTW、S0-S3 与图测试通过 |
| M9 | SADR 状态需求残差图 | S4、blockwise top-k、关系可视化 |
| M10 | SC-SimGCA 状态条件图传播 | S5，纯空间 residual 测试通过 |
| M11 | 第四章正式实验与定稿 | UrbanEV + CHARGED + PEMS04/08，S6、主表与第四章定稿 |
| M12 | 论文正文、图表与结果分析 | 全文叙事、图表、结果分析与章节一致性完成 |
| M13 | 终稿审校、复现材料与答辩 | 终稿、复现清单、答辩材料与最终校验完成 |

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

- 历史 TimeXer TEB 的 attention/residual 只作为失败路线诊断；
- 历史 Early/Late CCE 的等价 CrossLinear kernel、effective lambda 与 ungated delta 分布只作为负向 development 诊断；
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

| 模块 | M4 诊断与候选边界 | 保留来源的备选实现 | 最终仍失败时 |
|---|---|---|---|
| TimeXer-inspired TEB（失败历史路线） | Global/T2/T2G/T3/rescue 的 artifact、实现与诊断保持可追溯；第十七轮已触发停止线 | 不再继续 T4/T5/T6、patch/gate/beta 调参或新 TimeXer-derived TEB | 后续 CrossLinear 路线也已失败；当前下一来源候选已转为 Sonnet/MVCA |
| CrossLinear-inspired CCE（失败历史路线） | Early（RevIN 后、MDM 前）与 Late（post-PMCR/pre-AMS）实现、artifact 和诊断保持可追溯；两者均未通过 M4 adequacy gate | 不再调 kernel/lambda/gate 或增加插入位置 | 当前路线已触发有限开发停止线；当前下一来源候选已转为 Sonnet/MVCA |
| Sonnet/MVCA（当前 S2 development 候选，精确合同已闭环） | joint embedding + learnable wavelet + paper-defined MVCA + no-Koopman reconstruction + target residual | RevIN 后/MDM 前；target_exogenous only；matched from-scratch；固定 d/K/alpha/gamma 与双数据 development gate | production implementation gate 与 ChatGPT implementation review 已 Passed；paired development 与 adequacy 尚未执行/评估，在取得 development 证据前不得称为最终外生模块或最终 EL-AMD；XLinear 不同时启动 |
| PMCR | M4 只诊断 kernel、hidden、作用范围与 residual；任何参数或结构候选须经用户确认 | 可评估保留 Reparam DWConv + ConvFFN1 来源边界的候选；不得预先选定 | 更换另一篇近三年局部时间模块 |
| SADR | b_lambda 更负；k/d_a；正则 | ASTGRN global adaptive graph 与 DTW 的残差融合 | 更换另一篇近三年空间图模块；退回静态双图只算排障结果 |
| SC-SimGCA | rho 初始化；层数；SimAM lambda | 保留 G-STAN 层融合，移除 Graph-SimAM，改名 SC-GCF | 若仍失败，更换另一篇近三年空间传播模块 |

单模块正式通过线由 M5 执行：

1. 相对同输入/同骨干基线平均退化不超过 0.5%；该值只是安全底线，不能单独支持保留；
2. 至少在一个 EV 数据域和一个外部数据域或第二城市上产生稳定改善；
3. 3 seed 方向基本一致；
4. 最终组合不因模块交互产生稳定退化；
5. 达到用户在 M5 筛选前锁定的 practical-effect threshold。

# 22. 代码与复现难度

| 论文 | 官方代码 | 本方案使用难度 |
|---|---|---|
| AMD | https://github.com/TROUBADOUR000/AMD | 已复现 |
| CrossLinear | https://github.com/mumiao2000/CrossLinear | CCE 独立重实现与配对合同：低—中；不复制 normalization、patch、PE、head |
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

[T1R] Zhou, P., Liu, Y., Liang, J., Song, Q., and Li, X. CrossLinear: Plug-and-Play Cross-Correlation Embedding for Time Series Forecasting with Exogenous Variables. Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, 2025. DOI: 10.1145/3711896.3736899.

[T2] Luo, D., and Wang, X. ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis. ICLR Spotlight, 2024.

[S0] Wang, S., Chen, A., Wang, P., and Zhuge, C. Predicting Electric Vehicle Charging Demand Using a Heterogeneous Spatio-Temporal Graph Convolutional Network. Transportation Research Part C, 153, 104205, 2023.

[S1] Wang, S., Li, Y., Shao, C., Wang, P., Wang, A., and Zhuge, C. An Adaptive Spatio-Temporal Graph Recurrent Network for Short-Term Electric Vehicle Charging Demand Prediction. Applied Energy, 383, 125320, 2025.

[S2] Jiang, D., Gong, X., Wei, Y., Peng, B., and Xu, Z. An Electric Vehicle Charging Demand Prediction Approach Based on a Graph-based Spatio-Temporal Attention Network. Sustainable Energy, Grids and Networks, 44, 101975, 2025.

[D1] Li, H., Qu, H., Tan, X., et al. UrbanEV: An Open Benchmark Dataset for Urban Electric Vehicle Charging Demand Prediction. Scientific Data, 12, 523, 2025.

[D2] Guo, Z., You, L., Zhu, R., et al. A City-scale and Harmonized Dataset for Global Electric Vehicle Charging Demand Analysis. Scientific Data, 12, 1254, 2025.

# 24. 总体论文路线（最终时间结构由 M5 冻结）

```text
第三章：AMD
  + ModernTCN-inspired PMCR
  + Sonnet-inspired S2 MVCA target-residual development 候选（精确合同已锁定，尚未通过 development gate）
（TimeXer-inspired TEB 与 CrossLinear-inspired CCE 均作为失败历史证据保留；最终内部结构仍只由 M5 冻结）
数据：UrbanEV + EPF-PJM + ETTh1 + Weather + ECL + Exchange

第四章：EL-AMD + HSTGCN-core + ASTGRN-inspired SADR + G-STAN-inspired SC-SimGCA
数据：UrbanEV + CHARGED 六城市 + PEMS04 + PEMS08

所有数据域独立训练；时空模型使用完整图窗口；空间模块只输出 residual；
产物按 variant/dataset/task_mode/target/horizon/fold/seed/run_id 隔离。
```
