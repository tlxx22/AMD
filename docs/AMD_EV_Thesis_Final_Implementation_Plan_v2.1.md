---
title: "基于 AMD 与异构双图的城市 EV 充电需求预测：最终模块魔改与实验实施方案"
author: "面向服务器 Codex 开发的唯一权威实施文档"
date: "2026-08-15"
version: "v2.1-R1（替代版：修复数据双接口、目标输出、状态接口、图归一化、空间残差与实验协议）"
---

# 0. 文档定位

本文件替代此前的 `AMD_EV_Thesis_Final_Implementation_Plan_v2.1.md`，作为后续实现、实验登记和论文写作的唯一权威方案。旧版及更早方案移入 `docs/archive/plans/`，它们只保留历史记录，不得参与当前实现决策。

读取协作渠道（用户本次确认，M4 §69）：ChatGPT 优先通过 Remote Desktop Commander 从服务器直读本 canonical 和当前唯一 milestone，默认无需上传 Project。Codex 继续报告相对路径、变更类型、完整 SHA-256 和摘要；未由 ChatGPT 实际读取的修后版本不得宣称已核验。连接失败时采用可核对至对应 SHA 的片段/diff 或临时上传 fallback，旧上传记录保留历史。直读不改变权威优先级，也不授予任意服务器写入、test/checkpoint 读取或模型/训练启动权限。

当前仓库与开发位置：

```text
仓库：https://github.com/tlxx22/AMD
可执行冻结基准：amd_reproduced_baseline_v1 -> fa9665627e6fcfb1d0c2bc22d943ca9666304fd6
论文语义审计锚点：AMD-paper-norm-wd-ddi-v1 @ 5a718d5
开发分支：AMD-paper-repro-custom-modules-v1
M3 工程候选时间 variant：el-amd-pmcr-teb-v1
M4 外生模块状态：TimeXer-inspired TEB 与 CrossLinear-inspired CCE 均已触发有限开发停止线；Sonnet / Multivariable Coherence Attention（MVCA）S2 target residual 的 production capability、implementation review 与 Git closure 均已完成，ETTm1/UrbanEV 16-run paired development 得到 positive development signal，S2 adequacy gate 已 Passed，现仅是 M4 leading development candidate
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

本方案遵循以下工程约束与用户选择的创新组织策略；其中近期来源模块的数量与组织方式是用户策略，不是已核实的学校或导师硬性要求：

1. 第三章以完整 AMD 为时间基准；
2. 在 AMD 上加入至少两个来自近三年正式论文、并经过明确修改的模块；
3. 第四章以 HSTGCN 类地理—需求异构空间结构为基准；
4. 在空间基准上加入至少两个来自近三年正式论文、并经过明确修改的模块；
5. 每个模块都有独立开关、独立消融、同输入基线和可追溯来源；
6. 第三、第四章均进行多数据集、多模型比较；
7. 除明确登记的 development-only 数据集外，结构和超参数只依据训练集与验证集；正式评价数据集的测试集只在方案冻结后使用；
8. 新增模块全部关闭时，增强入口必须与冻结 AMD 在相同权重、`eval()`、相同输入下数值等价；
9. “模块临时关闭”可用于排障，不能冒充两个来源模块已获验证；两个近期来源模块是用户当前创新组织目标。若有限开发失败，保留失败证据和真实来源，后续变体或来源调整须依用户决定，不因数量目标自动新增候选或实验。

## 0.1 阶段顺序、候选身份与性能治理

2026-09-30最新M6执行决定：正文主实验集为UrbanEV/PJM/Weather/Exchange，完整552任务矩阵及既有结果保留；剩余228任务以DLinear→iTransformer→ModernTCN→TimeXer→N→S六组总队列一次用户启动、组间自动技术交接。此为跨模型启动方式修订，不改变baseline、科学profile、消融或第四章路线。41-run在原绑定版本下本轮审计通过、结果待ChatGPT审核；总队列为未提交待审核工作区，尚未获实际启动许可。正文范围及启动例外详见§9.6与唯一M6 §5.14，旧时点快照不倒写。

**M6当前增量（用户本次批准，工作区待字节审核）**：TimeMixer/UrbanEV/F4的h3/6/9/12采用独立确认政策：初始化、RNG、batch、结构及非浮点/Adam step/参数组精确一致；六步浮点参数、buffer、梯度、Adam moments绝对差≤1e-4；逐步loss与第2/6步MSE、MAE、同元素数归一SSE/SAE差≤1e-6，均rtol=0。仅命中当前冻结T12/pred_len1/C11/B128/LR0.001/seed2024及既有线程/硬件；其他域和旧exact失败不改。四H串行后q4共8 worker，局部48 Adam/80前向/48反向；第6步后复用第2步两个CPU评价batch额外评价，不改变六步训练/RNG。A阶段零模型负载，B须实现字节审核和精确closure后另签实际许可；本轮不提交或执行。完整条款、87＋1来源合并与正式队列边界见唯一M6 §5.13。

2026-09-27 M6增量覆盖授权（外置候选，生产未切换）：用户在已见原正式结果后明确增加NP/BE/FR/DE四市场的八主表模型及TimeMixer的PJM/UrbanEV F4。原495个科学profile保持，新增57个run/最多900 run-epochs；扩展总数552/最多6240，TiDE继续Deferred。新增41个补做任务允许固定AMD→J→PatchTST→TimeMixer跨模型自动衔接，另外16个任务分别追加DLinear/iTransformer/ModernTCN/TimeXer原组尾；原模型组之间仍不自动衔接。此前TimeMixer仅四标准域及495总数是原批次事实，不限制本次获批新增批次；不得倒写新任务为原注册范围。新增EPF保持T168/H24、单fit、70/10/20取整、历史输入、train-only scaler及目标标准化指标，模型数学与原495任务不变。TimeMixer新增域参数按本次明确授权固定，不能迁移回原四域。详见唯一M6 §5。

2026-09-28 当前工程接入：用户授权将同包累计22文件候选精确接入AMD工作区，并补齐作用域与完成审计绑定。运行代码与code_binding来自R，不再由P/candidate执行；P保留数据、来源、计划、收据与历史候选证据。工作区尚未stage/commit/push，不是已发布版本。旧TimeMixer退役回执和目标缺失已轻量核验；不再读取已删旧complete/manifest/checkpoint，不补造完整性Passed，也未再维护Desktop Commander。

后续顺序固定为：共用工程审核/closure和必要准入→用户单独启动timemixer-fixedlr-v2/attempt2原四域16-run→完成并通过必要完成审计→用户另行启动41-run补做入口；两个入口不自动衔接。41入口内部才允许AMD→J→PatchTST→TimeMixer的既定跨模型顺序。DLinear/iTransformer/ModernTCN/TimeXer各4新增市场仍在各自原组尾部，共16追加，和41任务互斥。覆盖552/6240、含重复计划568/6440不变；12个原profile仅LR变化，其余483原profile及57新增配置保持。准入分区51不变原组＋26已接受q1转移依据＋8待新测＋3改LR待复验，完整88组且互斥；原扩展计量18/512、剩494保持，改LR三域144Adam/192前向/144反向仍为单列待审提案。

本次29次精确无模型方法调用全部通过（19＋7＋3），一次CLI发现待重跑索引缺少计划revision/attempt字段，已修复并保留首次失败。267项已接受行及来源完全保持；重跑16行明确attempt2/not-run，无旧指标回填；删除前Blocked仅保留历史。不可执行模板start均在硬件查询/业务前拒绝，源码/环境/硬件、PLAN SHA、退役回执和未来probe报告必须逐项绑定，未来41入口还需当前修订版16项完成审计及实际JSON来源SHA。0模型/前向/反向/Adam/GPU，未运行probe/训练或创建有效许可。详见唯一M6 §5.11与同包workspace-integration-v1；ChatGPT未被声称已审核这些新字节。

以下为此前外置候选时点的记录，当前来源与顺序以本段为准：
本次M6增量仍只修改外置准备包`../amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/candidate`，生产未部署。用户明确授权退役删除旧TimeMixer整组且不备份，并批准原16任务固定LR整组fresh重跑：ETTh1/Weather/ECL=0.001，Exchange=0.0003（T96保持），其余训练/模型/数据/指标合同不变。旧训练确已执行且已见test；这是post-hoc LR修订，不是首次盲测，旧消耗不退款。此前SSH权限和Desktop Commander目标句柄导致dry-run停止的事实保留。此次按新增服务维护授权，从独立Codex终端在desktop/%0发送一次C-c，原服务链及同管道tee退出后，未改删除脚本持项目锁完成精确删除198文件/1,009,496,210字节及103目录，三目标均不存在；随后原pane以缓存0.2.50入口恢复一个remote和一个MCP，本地连接日志已确认。回执为user_authorized_retirement/deletion_completed，不是checkpoint_integrity_passed；无备份、未退款旧成本。PID64747权限局限仍如实保留。新目录为原结果根下`revisions/timemixer-fixedlr-v2/formal-TimeMixer`，attempt2，16次fresh/最多200epoch；覆盖552/6240不变，计划尝试累计568/6440。只改变12个profile的LR，483原profile及57新增profile不变；TimeMixer新增UrbanEV=0.001、EPF/PJM=0.0001不变。统一索引保留其他267已接受身份，新16显示not-run且不从旧结果回填。三个改LR组排除旧数值继承，新数值准入仅提案、未获技术预算，不挪用57任务包剩494前向。26组q1转移依据已接受；计量累计18仍有效。此前23项工程和17项删除专项验收均保留，本次不重跑测试，0模型/前向/反向/Adam/GPU；仅服务维护和已授权退役完成，候选审核/部署closure、新许可及数值准入仍未完成，不自动训练。

2026-09-19当前有效阶段：用户明确授权将J冻结为第三章最终EL-AMD（AMD＋Sonnet/MVCA S2 target residual＋THLS），结构身份`el-amd-s2-thls-v1`；M5阶段结束并随本轮closure封存，当前唯一milestone转为`docs/milestones/M6_formal_experiments.md`。M6已有495个fresh正式run/最多5340 run-epochs获本次授权，按十个模型组各自用户启动，组内按既定dataset/input_variant/H/fold波次自动执行；不自动启动下一模型。源/形状不变的54/54 probe证据沿M5 §24继承，并发只在已验证范围使用。M4原18项总gate Not passed和H192 J/N +1.542951806%风险保持；结构冻结不等于模块正式效果已获证。正式模型数学、T/epoch、AMD家族batch/LR及外部baseline来源batch/LR均未变，std=N/A。此前带日期或§编号的状态是历史快照，不重新开放已经裁决的阶段/参数事项。

2026-09-19当前状态（M5 §24）：RSS复合判据与后端条件数值准入后的10组补测已完成并审核，54/54组技术准入全部Passed（44继承＋10本轮新通过）。本轮实际480 Adam/640前向/480反向，低于594硬上限；80条worker轨迹均finite、0 OOM、0资源归属失败、0受限审计deny。Passed并发为q4 44组、q2 3组、q1 7组。报告与当前protocol/code/environment/hardware/许可、kernel admission均匹配。该结论仅为资源/并发/短轨迹数值技术gate Passed，不是效果gate，不冻结J；M5仍未Closed，M6未开始。以下§23及更早为历史时点。

2026-09-19当前状态（M5 §23）：按用户本次RSS复合判据及条件性数值授权完成接入。RSS需连续增长且净增>32MiB、平均>1MiB/step，并经24步窗口（前6步warm-up）末8步无平台才作CPU持续增长阻塞；显著短窗增长先标NeedsLongWindow，不冒充泄漏。三组24步代表均平台。四个数值作用域的固定Conv1d输入/权重/上游梯度重放均实测默认梯度非逐位、仅诊断切换cuDNN确定性后重复exact；生产确定性设置未改。TimeMixer/Exchange、ModernTCN/Weather采用全浮点1e-4及指标1e-6；ModernTCN/ECL初始界失败保留，条件证据成立后前瞻性采用state1e-3、validation1e-6、loss abs1e-6+rel1e-5并经新轨迹通过；原RSS阻塞还掩盖TimeMixer/Weather数值差异，已在原10组范围内补2串行验证，state1e-4、validation1e-6、loss abs1e-6+rel1e-5通过，正式并发仍待probe。32次CPU方法通过；21诊断worker实际180 Adam/294前向/252反向，机械余108；原48次隔离grad漏计单列补账、钩子修复，旧记录不改。44组155代表继承核验，10组40代表补测只用594既有余额，核心<=480，资源性候选回退按固定顺序限余额，不增加预算。495/5340、全部正式profile与训练数学不变；尚未启动新probe/J冻结/M5关闭/M6。以下§22及更早为历史时点。

2026-09-19当前状态（M5 §22）：20组补测已完成并审核，54组终态44 Passed/10 Blocked（34继承＋10本轮新通过）。实际846 Adam/1128前向/846反向，低于1440上限，新增144额度未实际动用；0 OOM、0资源归属失败。剩余10组中7组由CPU RSS四点严格递增触发，GPU allocated均稳定；3组为数值准入失败：TimeMixer/Exchange超过原具名单张量1e-7或白名单外exact，ModernTCN/Weather与ModernTCN/ECL仍exact不一致。完整report与当前protocol/code/environment/hardware/许可均匹配。结果review Not passed；J未冻结、M5未Closed、M6未开始。以下§21及更早为历史时点。

2026-09-19当前状态（M5 §21）：用户明确确认三组限定数值等价及额外144次补测Adam。仅TimeMixer/ETTh1、TimeMixer/ECL、ModernTCN/ETTh1四H采用全浮点state/gradient/Adam moment atol=1e-4、rtol=0，loss与归一化validation atol=1e-6；初始身份/RNG/batch、optimizer step、非浮点状态和param-group结构继续exact，非finite拒绝。TimeMixer/Exchange原具名单张量1e-7规则保持。实现使用预分配CPU缓冲区写原始sidecar，正式训练数学/profile不变。CPU首验12 passed/1 error/3 unexecuted，唯一机械清理修复后16/16；三组H96串行参考＋两路并发＋独立串行重复共72 Adam/96前向/72反向均通过，最大state差分别1.91e-6、1.91e-6、4.32e-5，最大指标差2.38e-7，均在预注册界内；机械余额288。34组/115代表继承已核对，20组/80代表补测上限1440=原余额1296+新批144，尚未启动。J未冻结、M5未Closed、M6未开始。以下§20及更早为历史时点。

2026-09-19当前状态（M5 §20）：用户授权直接修复并准备补测；已审核第二轮54组终态34 Passed/20 Blocked（10继承＋24新通过），本轮原probe实际1740 Adam/2320前向/1740反向，无已尝试worker OOM或资源归属失败，不外推未执行H。新增固定CPU摘要缓冲区，三个原RSS失败代表复验通过且计算轨迹与原版exact相同；没有删除四点规则或放宽门槛。CPU16+16方法通过；15个合成诊断worker累计90 Adam/120前向/90反向，机械余额360。三组数值诊断初始/RNG/batch相同但串行重复也不exact，最大模型状态差约1.55e-6、7.08e-8、5.25e-5，不能直接套TimeMixer/Exchange单张量1e-7白名单，未新增容差。计划34组继承/20组80代表补测上限1440，原补测剩1296，额外144仅Proposed。工程repair可收口，完整补测仍Blocked；J未冻结，M5未Closed，M6未开始。以下§19及更早为历史时点。

2026-09-19当前状态（M5 §19）：用户明确选择数值等价并继续授权ChatGPT直接执行至工程closure。仅对TimeMixer/Exchange既定四H的enc_embedding.value_embedding.tokenConv.weight及其梯度/Adam两动量，预登记atol=1e-7、rtol=0、NaN/Inf拒绝；训练loss与归一化validation误差同绝对界，初始/随机/批身份、步数及白名单外模型/梯度/optimizer状态继续exact，其他模型/域不放宽。新JSON逐步数值证据与精确残余摘要已接实际probe比较及报告检查。CPU16/16首次通过；H192新六步串行参考、两路各六步及独立串行重复共24 Adam/32前向/24反向，全部在该固定界内，最大梯度差7.450580596923828e-08、动量差1.4901161193847656e-08、参数和loss差0，旧exact Not passed不倒改。机械余额450；3036补测额度未消耗，10组25代表继承已核对，44组170代表仍待用户一次启动。495/5340、全部profile/T/结构/训练/数据/确定性设置不变。技术阻塞已按新合同处理，本轮工程审核closure后可生成实际版本probe许可；不代启补测，不冻结J、不关闭M5、不进入M6。以下§18及更早为历史时点。

2026-09-19当前状态（M5 §18）：用户明确授权ChatGPT接手服务器执行上一轮确认方案及检查工程Git closure。本轮57个来源待决run全部落实，iTransformer/ECL明确采用本模型官方脚本B16；A/J/N/S的274个profile、495任务/5340 run-epochs、54组/195worker及统一T等不变。最终最大optimizer步数算术16,482,750；新增709次补测Adam获批，2327原余额＋709=3036，未消耗。双时点RSS测量保留四点严格增长规则和前序哈希影响说明；CPU16/16首次通过、无复验。限定5worker均正常完成，共30 Adam/40前向/30反向，机械余额474；AMD/ETTh1六步未触发旧增长检查。TimeMixer/Exchange H192初始/RNG/batch及loss一致，但重复串行也有Conv1d权重梯度/Adam微小非逐位差异，exact仍Not passed，不擅调容差或确定性设置。10组25代表继承profile核对通过，44组170代表补测计划保留但全队启动仍Blocked；本轮仅工程版本收口，不生成完整probe许可、不冻结J/关闭M5/进入M6。以下§17及更早为历史时点。

2026-09-19当前限定增量（M5 §17）：用户批准外部六baseline仅按对应论文/明确委托的本模型官方代码迁移batch、eval batch及初始LR；A/J及N/S原训练设置、全模型T/epoch/停止/优化器与信息合同保持。唯一profile覆盖层已接入，未决来源字段保持待审并前置拒绝；这是有限参数迁移，不是完整作者recipe复现、不声称各模型最优或纯结构差异完全隔离。原54组probe已完成：13 Passed、22 ResourceNotVerified、19 Blocked，206worker、1183 Adam调用/1580前向/1184反向，失败不退款。当前CPU24+24方法均通过；CUDA两个代表首次及唯一机械复验累计8 Adam/18前向/8反向，修后iTransformer小形状与TimeMixer ECL B32/T512/H96连通、自然退出监测通过，首次退出监测失败仍保留。九组RSS增长旧阻塞及TimeMixer/Exchange旧数值不一致不倒改。495/5340、54组/195worker不变；10组可提交有条件继承审核，44组170worker补测上限3036 Adam，原首验余额2327，额外709仅Proposed；机械512余额504。来源、内存判据及额度待决，修后review/closure和用户启动仍必需，本轮未启动完整补测/正式训练，J未冻结、M5未关闭、M6未开始。以下§16及更早均为历史时点。

2026-09-18当前准入收口（M5 §16）：用户已明确确认并授权四域数据政策及限定CPU/前缀验收，§14.4相关提案现由本次精确合同替代。继承§15五文件来源摘要，stat与其读后状态一致，本轮未完整扫描或重哈希数据。Weather仅允许非唯一但非递减时间，保留原记录按T/H建窗；ECL/Exchange采用已核验benchmark列槽位和train-standardized指标，原转换/单位及币种解释未知作为必须披露限制；PJM采用锁定TimeXer文件、n52416及36691/41933/52416端点，原EPF §4.1 d−1可得性为documented source assumption，不冒充逐记录vintage审计。CPU首验7 passed/1 error/8 unexecuted，经唯一机械夹具修复后16/16通过，累计24次调用；Weather/PJM各一次限定前缀连通通过，其他域复用旧有效证据。54组195worker的端点与尾批均具备，dry-run/preflight仅拒绝未审核许可及未closure/clean；资源/正式模型尚未实测，完整probe仍由用户在审核和统一closure后启动。495/5340、模型与训练、NVML门禁不变；J未冻结、M5未关闭、M6未开始。§14额外调用偏差与所有旧失败保留。以下§15及更早为历史时点。

2026-09-18当前来源事实（M5 §15）：本轮沿锁定作者README取得具体发布物并直接比对：Weather/ECL/Exchange与iTransformer明确分发包仅CRLF/LF不同（B类文本一致，非原始字节一致）；PJM与TimeXer固定Git blob、ETTh1与固定ETDataset对象为A类字节一致。作者Weather副本同样含零基19043/19044重复时间。PJM实测n=52416，既定端点36691/41933/52416，T168/H24/B128的validation尾批99；本轮只登记事实，生产endpoints及mandatory未变。EPF原论文已有d−1可得的day-ahead外生预测说明，但非逐记录vintage审计；原EPF Zenodo发布文件与TimeXer版本header/列序不同，转换链另列待核。UrbanEV复用Closed M1及Git对象，不重读数据。§14.4降级政策仍Proposed，不修改loader、数据、科学条款或准入；新证据待审核后决定可解除的具体事实项。495/5340、54组/195worker不变，§14额外test调用偏差保留。本轮8个合成审计方法首次通过，0模型/GPU/训练/checkpoint；完整字节比较遍历test字节，test数值解析/统计/评价为0。完整probe未启动、J未冻结、M5未关闭、M6未开始。以下§14及更早为历史时点。

2026-09-18当前工程状态（M5 §14）：§13的495任务/最多5340 run-epochs、54组/195worker与A/J F1–F4合同保持。NVML正常权限下的自有PID映射已实际解决：读取自有/proc sched的内核PID并核对生命周期，1→2→4个16MiB张量进程准入均通过，4实例/27.87秒、0模型/Adam/前向/反向；CPU首次16/16及一次机械测试/状态修正复验16/16；首验另有1次嵌套旧test断言调用（严格调用计数17，超首验上限1次），偏差保留待审核，修后无此调用。仅解除已获现场证据支持的监测归属缺口，不授予正式组并发、模型资源或训练许可。完整probe仍待修后整体审核、实际closure/clean及用户启动，PJM尾批等数据边界须明确；本轮未运行完整probe。四域数据mandatory不改；Weather按原记录建窗、ECL/Exchange匿名固定基准标准化指标及四文件窄字节核验均为§14.4集中待决稿，未批准/执行。J未冻结、M6未启动，以下§13及更早摘要保留历史时点。

2026-09-18最新决定（M5 §13）：用户重新决定恢复UrbanEV **A/J双方F1–F4输入消融**，只新增AMD/F1–F3的72个未执行计划，两者F4复用主表；不恢复任何F0、PJM TargetOnly或其他消融。当前495唯一runs/最多5340 run-epochs；TiDE25/最多260 Deferred，含暂缓为520/最多5600，**此520集合不等于§11旧520集合**。清单生成54组/195代表worker，仅新增三个AMD输入组。`input_variant`及schema v2不变，完整配置指纹更新，旧423/旧520许可、报告及恢复身份不兼容。当前NVML逐活动worker PID匹配缺口已列为probe全队启动前阻塞；必须先有已审核且可实际通过的准入路径，不先跑全队重复发现缺口。本轮不再读取数据前缀/查网站、不改数据mandatory或NVML策略，不启动完整probe/训练、不冻结J或进入M6。以下§12及更早摘要均保留历史时点，不倒写旧授权。

2026-09-18最新决定（M5 §12）：**所有消融仅UrbanEV；模块消融F4 A/N/S/J只新增N/S；输入消融仅J的F1/F2/F3/F4，F4引用主表同run。** 不做额外AMD输入消融或PJM TargetOnly；J/F0及空aux的J/S在声明阶段拒绝，不关S2改名、不造伪辅助。撤下97个未执行计划后，当前为423唯一runs/最多4620 run-epochs，含Deferred TiDE为448/最多4880；447/4860是含非法J/F0的错误建议，从未成为已批准合同。正式schema统一`input_variant`；F2价格和F3天气是并列方案。仍按模型分组，轻量probe从现清单生成51组/183代表worker，完整probe未启动，修后材料待整体审核/closure。Weather重复原始时间、PJM版本端点/单位/as-of、ECL转换单位/列对应、Exchange报价列对应继续mandatory blocked；不把匿名客户真实身份作为补证目标。空间规则、模型数学、统一训练表、M4 gate Not passed/H192/1%及单seed限制不改。J未冻结、M6未开始。下述§11及更早轮次均为历史时点，旧520/5600与545/5860不再是当前合计。

2026-09-18最新实施决定（M5 §11）：**按模型分组＋固定配置轻量资源准入**取代下述三个正式包和未执行的重型1/2/4全覆盖流程。唯一机器任务/结构配置为`configs/ch3_formal_profiles.json`：520唯一runs/最多5600 run-epochs，TiDE25/最多260继续Deferred；同run跨表引用，不新增候选。新正式声明、六域受限前缀路径、独立作者adapter、共享runner/恢复/汇总、模型入口和probe入口已实现，实际短验收与限制见M5 §11。ECL骨干LN=False在新声明下已通过限定CUDA验收，THLS内部feature-LN及旧M4 guard不改。数据连通不等于元数据闭环：Weather已读前缀有重复时间戳；PJM版本端点/单位/as-of、ECL处理链/客户映射和Exchange报价映射仍blocked。PJM **F=1及70/10/20取整政策已确认**，此前“PJM fit Unknown”措辞不再适用；Unknown的是版本数据事实、步骤与耗时。完整probe待修后整体审核、统一closure及用户启动；本轮不运行完整probe、不冻结J、不进入M6、不启动正式训练。以下§10及更早摘要保留历史时点，空间规则不在本轮变更范围。

2026-09-18本轮执行决定（M5 §10）：用户授权八个独立时间作者来源准备、受限合成训练链smoke及ETTh1固定13项loader验收；这不授权正式训练、M6或J冻结。TiDE **Deferred**，本轮不准备/运行、不永久删除或新增替代者；含TiDE完整规划545 runs/最多5860 run-epochs保留，当前推进520/最多5600、暂缓25/最多260。正式训练分为三个独立用户启动包，见§9.6.1。原生结构须各自作者源码绑定，不再从TimeXer附带baseline代取；smoke小配置不覆盖§5.5正式训练表或补作正式结构冻结。ECL骨干layernorm=False与旧J守卫冲突仍blocked，THLS自身feature-LN保持。空间本轮仅确认既定HSTGCN/ASTGRN/G-STAN三个期刊来源例外和原版复现优先顺序（§1.2/§10），没有任何空间准备或执行。以下旧轮次授权及工具Blocked是历史时点，最新实际结果见唯一M5 §10。

2026-09-18最新确认（M5 §9）：用户已确认“正式协议优先”的阶段职责和消融范围，并明确同一数据集所有baseline、AMD/J及实际消融统一epoch、LR、batch和停止规则，精确训练合同见§5.5。M5承担正式协议、必要工程接入/验收准备及提请用户最终结构冻结；M6承担六域主表、UrbanEV完整模块消融、保留的TargetOnly/输入消融与正式效果/成本评价。旧多域单模块效果筛选及新增practical-effect不再作为冻结前置，这是已确认的合同替代，不声称旧要求已满足。原M4总gate Not passed、H192退化1.542951806%及原1%线/科学序列停止不改；J仍未冻结，不按dataset/H切换模型。M5旧§4–8保留历史，旧筛选矩阵不执行；仅旧§6限定ETTh1 loader两文件设计按本次授权启用，正式训练、M6与其他接入包未获授权。以下此前审议/未确认摘要均为历史时点，最新执行状态见唯一M5 §9。

2026-09-18当前审议状态：用户要求M5转为“正式协议优先”的合同收敛，暂停沿M5旧§4筛选包及§6 loader首包推进；旧§4–6保持未批准提案。唯一修订稿见`docs/milestones/M5_model_selection_and_freeze.md` §8，推荐M5以正式协议、必要工程/验收及提请结构冻结为主，M6承担六数据集主表和以UrbanEV为核心的模块消融。精确范围与替代冻结条件均为Proposed，尚未确认，**本段只登记审议状态，不使§9.3/§9.5/§9.6/§19/§21等科学条款变更生效**，也不声称已满足旧多域效果前置。M4原总gate Not passed、H192失败/原1%线及单seed/development-test限制不变；不冻结J、不进入M6，不授权实现、动态验收或训练。

2026-09-16 M5启动登记：用户本次明确批准开始“大论文05｜M5 模型筛选与结构冻结”，本轮仅授权静态可执行性审计、筛选协议提案、首个实施包设计和唯一M5文档建立。A段实际统一closure commit为`3e43a55327eac641445ab5a43d06424d7fc5ed3f`；本轮写入前回核HEAD/tracking/live remote三端一致、ahead/behind=0/0、worktree/index clean，已封存M4 SHA-256=`3b29d43624de290ebdec6a1701066f6dded33f8e788d8b183cc1fa416df4c11e`与用户指定审核字节一致。M4已Closed，不再追加；原18项总gate仍Not passed，ETTm1 H192 J/N退化1.542951806%超过原1%安全线及已知风险保持，J仅为候选审查对象。当前唯一milestone切换为`docs/milestones/M5_model_selection_and_freeze.md`，M5 In Progress（静态审计/协议准备）；新结构、动态测试/训练预算、正式冻结均未获本次授权，M6未启动。下述§68/§69及旧摘要中的M4 In Progress、closure Pending、M5未开始均保留封存前历史时点，不覆盖本段最新状态；阶段职责、科学条款、旧门槛及既有来源不变。

§68结果登记：双数据集N/S/J incremental engineering、ChatGPT implementation review及统一closure已完成，closure/训练HEAD=4abb099d69db04532f066986fde0c43aedf2465a；25-file source=6a11d88f6d22861e49db0122bf685b642b197af2796ab8de4b7a424b4548a97a。已批S/J兼容通过，N证据按已核验前置复用，实际E-S→U-S→E-N→U-N→E-J→U-J六波24run完成：240 run-epochs、1017450 Adam步骤、墙钟39583.71105360985秒。ChatGPT接受完整性及本轮结果review Passed；UrbanEV J/S、J/N各5/5 Passed，ETTm1 J/S 4/4 Passed、J/N 3/4。原18项总gate Not passed，唯一失败是ETTm1 H192 J/N test MSE +1.542951806%超过1%。十二个ETTm1 run文本曲线未发现训练/选择异常，S/N同H也退化；只能确认局部性能取舍，原24-run科学序列停止，不改阈值、不重选best或重训。M4 In Progress，结构未冻结，单模块已接受结果与P2原Not passed保持；详见M4 §68。

§69最新决定（结果已知后的用户准入例外）：服务器直读等工作方式此前已确认；**用户于本次04-03会话明确确认**，在看到§68完整结果、接受ETTm1 H192 J/N development-test MSE退化1.542951806%超过原1%安全线后，允许当前J=AMD+Sonnet S2 target residual+THLS带已知局限保留为后续候选审查对象。此准入例外自本次消息明确确认起成立，不据原草案或§69原有自述认定此前已获批准，不追溯补证。保留N/S及历史比较，不按H或dataset切换模型。原24-run总gate仍Not passed、原科学序列停止，既有结果review Passed不变；不构成新性能验证或EL-AMD冻结，不改模型/阈值、不重选best、不择seed或重复24-run。此前修后版本已由ChatGPT服务器直读并匹配SHA，文档review=Passed。用户现明确批准“结束M4并办理统一收口，M5暂不启动”：M4已批准的诊断、候选开发和有限分析工作结束，无待追加的M4模型、测试或实验任务。本次仅形成最终结束登记待审版本，Git closure尚未执行，尚未完成Closed封存；完整封存以本次修后字节经审核并完成对应累计文件统一Git closure为准。不单独docs closure，M5/M6未开始，本次决定不授予M5启动或正式冻结权限。M5对应任务的正式数据/元数据/MS接口、公平多数据集与独立模块筛选、practical-effect/预算、代价收益及结构/协议冻结要求仍按§5.5、§9.1–9.5、§19、§21执行；M6承担冻结后的正式test、主表、原范围消融、效率与定稿，不升级旧development产物。上述后续任务不全部前置为M4关闭条件，详见M4 §69.3–69.4。

2026-09-13 用户治理澄清：小规模文档更新不单独安排closure，可与下一实质工作统一提交；这不免除审核、训练来源/版本核验或启动器要求的clean Git。§§58–60累计文档已随THLS代码/测试统一closure；§62结果登记与§63接入代码/测试/策略绑定及累计文档已由667a1b82bdf799a10a6ce7d813b424c0f223a5cf统一closure；§64–65组合提案/接入及失败修复保留；§66原提案保留；§67统一接入/工程验收与implementation review已通过，代码/tests/阶段policy及累计文档已由4abb099d69db04532f066986fde0c43aedf2465a统一closure；§68登记已接受24-run结果及有限文本分析，本轮两文档修改不单独docs closure。“两个近期论文来源模块”属于用户选择的创新组织策略，不是学校/导师硬性要求；真实来源与冻结历史不倒改，现有模型及阶段任务表不因此重排。

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

M4 继续保持 In Progress。第二十二轮已由用户一次性确认 Sonnet/MVCA 的 S2 保留范围、RevIN 后/MDM 前插入点、target_exogenous-only 任务语义、matched standard from-scratch 协议、全部数值配置、development 数据边界与停止线；精确合同见第 1.1.1 节。Sonnet S2 production capability、永久测试和非训练单批探针已经完成，ChatGPT implementation review 已 Passed，Stage B implementation 已由 `bd1e0ab45f7329d3eb8c24eed106de19e21d9884` 完成 Git closure。第二十三轮又完整执行 ETTm1/UrbanEV 16-run paired development：UrbanEV 主门禁与 ETTm1 安全门禁均通过，登记 `positive development signal` 与 `Sonnet S2 development adequacy gate = Passed`。Sonnet S2 因而只是当前 M4 leading development candidate，仍不是最终外生模块、最终 EL-AMD 或 M5 冻结结构；既有单 seed、固定 S2 与 development-only ETTm1 证据不能替代 M5 按现行正式合同的筛选；随机初始化稳定性仍未验证，原三 seed 一致性要求依现行政策暂缓。XLinear 仍未选择。P2生产实现、implementation review与Git closure已完成；原24-run development序列已完成，ETTm1两个安全gate Passed，UrbanEV完整性Passed但两个主效果gate Not passed，ChatGPT已接受P2 development adequacy=Not passed。本配置序列停止；有限1路/2路短测及后续1/2/4路、4路×2线程对照均已完成并获接受；合并12组48个有效任务、24,960步，计入中断组后实际26,000步。S4T4仅为当前UrbanEV h3/AMD-Concat负载的优先执行配置候选，不是全局或完整训练冻结。本轮效率探索停止，转为P2既有24-run曲线与16份C权重的只读机制分析；用户随后批准原有限validation激活/旁路提案，本轮8份C权重、504次batch前向已完成，parity/状态与数据隔离检查通过；固定子集显示平均增益与时间变化并存，旁路结果不替代独立训练比较，ChatGPT已接受§58结果；用户已确认“目标历史局部形状残差支路”（THLS）的§59结构/身份、工程验收、UrbanEV 8-run合同及独立160步N并发兼容范围。当前独立模块、A/N入口、前缀分派、严格恢复/汇总和18项永久测试已写入。用户另行明确批准4项真实资产方法在本次THLS验收中暂不执行；当前精确策略区分历史16项真实资产＋1项旧CPU-only依据与这4项当前批准，后者历史证据仍Missing/Not verified，旧recovery_status=incomplete保留。§60.7历史序列中，重建工具v2通过无模型自检后进入业务：新增18项7 passed、1 failure、10 unexecuted；关闭支路A与冻结AMD在CPU T512的输入梯度逐位相等断言失败，按停止线未执行继承314项及真实probe。THLS engineering gate=Not passed（整套验收未完成），不能登记implementation complete。核账发现调用别名导致漏记4次前向，原v2日志保留，实际调用由89次在线记录＋4次静态控制流核对登记为93次前向/API尝试、16次反向、0次Adam.step；用户随后批准有限定位；v2.1在本轮4次CPU前向/4次反向中实际核对计数及包装边界。当前A/冻结AMD的T512输入梯度最大绝对差6.558927697368118e+34；AMS/DDI输出处的反传梯度相等，DDI参数及其上游累计梯度不同，不能判为舍入误差。HEAD版增强/冻结AMD对照又触发非finite检查，按停止线没有生产/测试repair或最终续验；该组详细数值未在断言前落盘，未补造，诊断记录工具修正后仅做无模型检查。§§60.8–60.9的失败、缺失记录与累计102/24/0保持历史事实。§60.10用户明确批准开发分支common.py的限定兼容例外：ddi_cpu_gelu_cotangent_contiguous_v1，仅在DDI所属原生GELU的CPU反向前把cotangent同值连续化；未改前向数学、初始化、state keys、CUDA路径或环境。M0/不可变tag及其源码原样保留；此兼容处理不是新增论文模块，不重定义历史source。§60.10局部布局回归Passed，两条安全路径梯度逐位相等，解析参考最大差3.779108439621844e-11。最终序列在原关闭等价CPU T512的rev_norm.affine_weight处再次停止：预测/MoE、输入梯度及其余50/51参数键逐位相等，但该参数1/7元素最大差7.450580596923828e-09，全部有限，仍不满足原torch.equal。333 ID中1 passed、1 failure、331 unexecuted，skip/error/fixture-blocked为0；必需CUDA、继承314和真实probe未执行。当前反向参考AMD也使用修后的common.py；另以未修复冻结Git源码仅作前向参照并通过两项CPU子case，不冒称整套执行依赖未变。§60.10累计114/30/0及当时预算缺口保持历史事实。§§60.11–60.12的固定参考失败、weight舍入参考核对通过、CUDA bias原逐位失败及当时累计141/44/0均保留历史。§60.13已完成CUDA bias完整参考及仿射组条件性比较，关闭等价四CPU/CUDA子case满足各自要求；当轮333 ID为11 passed、1 error、321 unexecuted，停止于THLS入口缺失常量导入，累计253/70/0。§60.14已补齐main.py的CANONICAL_FEATURE_NAMES单行导入并完成静态检查，当时未获96→128扩额而未运行动态验收，保留该历史事实。最新§60.20用户明确批准新增合成累计上限824/193/49，携入621/153/33，临时fixture修复后的唯一封存验收序列已完整通过：GELU1＋THLS18全部Passed；继承314为293 passed＋21原精确策略skip。共333 ID=312 passed＋21 skipped，failure/error/blocked/unexecuted均0，不能称333项全部通过。CPU/A800 CUDA实际覆盖原必需case；原36 forward/4 backward/0 Adam真实UrbanEV单批probe完成，四horizon A/N的train/validation前缀、CPU/CUDA batch2及h3/h12 CUDA batch128检查通过，test读取越界/解析/构造/迭代/评价哨兵均0。全文件读取仅为既定字节指纹，实际观测解析限原3909时间点前缀；无历史checkpoint读取、完整评价或development产物发布。三个业务阶段非预期受禁访问均0。新增本轮179/40/16、累计800/193/49，剩24/0/0；继承992/298/148及真实36/4/0另计，历史失败成本保留。未新增生产/测试repair；旧CCE可移植性断言在原字节下Passed。仅原指定float32 RevIN仿射参数组保留对称1e-7/1e-6数值规则，其他严格断言不变；报告整理的双方None字段错误从完整原记录修正，未重算模型。THLS implementation complete、engineering/implementation review Passed，统一代码closure=f13dd26dfa72e140c8e4c05b146ea39d6643f889；UrbanEV已提交文档/训练来源为e93f008f9f82546e13bf4c7f2522b1728d81cbdf，24-file source=16343e9298a9c4a1eb477c061f7b58552ab04b0706f751b8b83e42749b703a7f。§61独立N160步兼容和训练前方案已获审；随后按用户明确授权执行A四H→N四H两波，8/8、80 run-epochs、594040有效Adam步骤、墙钟22573.999078273773秒。最新ChatGPT接受完整性Passed、UrbanEV五项gate Passed及严格限本合同的positive development signal：N/A主MSE macro−0.873577%、MAE−0.715574%，四H及全部leave-one-out改善；旧中断A-h3成本与审计脚本修正保留（M4 §62.1–62.4）。ETTm1最小接入incremental engineering及ChatGPT implementation review均Passed，统一closure/训练HEAD=667a1b82bdf799a10a6ce7d813b424c0f223a5cf，24-file source=4d707b1d878d3490a55243e060a3d43258c60a70414edf23d2ff3fc4acedbc9a。已批336/320/320兼容检查完成后采用A四H→N四H两波；8/8、80 run-epochs、84260 Adam步骤、实际墙钟3142.443908929825秒。ChatGPT本轮接受完整性与§62.7四项安全线全部Passed：主validation MSE macro +0.112646595%、development-test MSE −0.032258338%、MAE −0.160544776%，最差H720 test MSE +0.466123118%；两种macro和逐H原值见M4 §64.2，不能称所有H均提升。结合UrbanEV五项gate Passed，THLS单模块本轮开发验证收口，ETTm1仍是development-only test；旧source和审计Pending快照不追写。§66–67双数据集N/S/J比较已完成统一接入、工程/review/closure及兼容，训练来源为4abb099d69db04532f066986fde0c43aedf2465a、25-file source=6a11d88f6d22861e49db0122bf685b642b197af2796ab8de4b7a424b4548a97a。24run按E-S→U-S→E-N→U-N→E-J→U-J六波完成，240epochs/1017450 Adam，墙钟10小时59分43.71秒。§68本轮结果review Passed；UrbanEV J/S与J/N各5/5、ETTm1 J/S 4/4，J/N因H192 test MSE +1.542951806%超过1%而3/4；原18项总gate Not passed、科学序列停止。ETTm1两组validation MSE macro安全项分别−0.049560234%和−0.133132443%，均Passed，不因摘要只列test而省略。有限十二run曲线未发现具体训练/选择异常，不自动重训、改模型或准入。以下§65失败/修复为已被§67续验取代的历史记录，不是当前工程状态：用户当时明确确认§64唯一结构/独立身份、8 fresh runs/80 run-epochs、五项效果gate及三笔独立额度；§65四文件组合接入和八项新增验收已执行至6 passed、1 summary负例failure、1 unexecuted。source负例参数传递错误已最小修复并静态检查；累计123/45/24、剩37/19/8，不足以按原要求在修后版本完整复验，当时engineering Not passed/未完成、implementation review/closure Pending，真实20/8/0未执行。当时25-file source=75458a103145f60b98314bd206330e45442ce65043c3161362208cecb10207e8；原工具/skip/模型数学及数值边界不改，当轮320步兼容和八run未启动。当前已完成新24-run且总gate Not passed；S2 Passed/leading、THLS单模块已接受结果、P2原Not passed、M4 In Progress保持，正式结构/variant未冻结，不启动M5/M6/M7。

2026-09-07 用户已确认“改进 PMCR”方向，并在正式 MS 对齐后确认精确字段目标、历史输入、指标/汇总与正式身份合同；当前工作状态为 **MS精确任务合同 User confirmed，文档审核与Git closure已完成**，**P2工程 Conditional authorization granted**。第二十八轮已实现ETTm1/UrbanEV非Sonnet A/B最小MS接口，并取得永久测试及有限单批工程证据；A/B最小接口 ChatGPT implementation review=Passed，Git closure已由`e62e3ddbe8be96f99fb1d77f010de94dd7546094`完成，完整回归受访问边界限制的skip仍按M4 §51登记。第二十九轮已完成P2候选C生产实现及本轮授权范围的工程验收；合成fixture路径修复后，CPU/CUDA定向、完整受限回归及有限真实单批探针通过，engineering gate=Passed，ChatGPT final implementation review=Passed，P2 Git closure=Completed（2f0bd489b7ebd408fba359811fe580984895fe10，parent=e62e3ddbe8be96f99fb1d77f010de94dd7546094）。原失败记录及续验分别保留于M4 §52.1–52.5、§52.6–52.9，受限项不升级为Passed；工程review/closure与Stage 1授权见M4 §53，Stage 2原执行合同见M4 §54，已接受性能结果见M4 §55，并发收口见M4 §57、已接受机制诊断见§58，新支路已确认合同见§59，完整工程验收见§60.20，已接受UrbanEV结果及ETTm1合同见§62，本次明确批准与接入工程实测见§63。未用于下一轮ETTm1/UrbanEV两项development任务的正式元数据缺口不再阻塞P2工程，但仍阻塞对应正式任务；这是对旧M4 §49.7依赖的有限调整，不改变Not verified事实，详见M4 §50。不以PMCR v1先独立通过target_exogenous/UrbanEV adequacy gate或旧U1/U3独立16-run为前置；v1小幅validation收益与P1扩容无一致收益的历史证据保留，不外推为当前任务有效，也不改写为v1失败。同期三臂方向保持A=AMD-Concat、B=A+PMCR v1、C=A+P2：C vs B检验改进，C vs A检验AMD实际收益，B不是先通过的门禁。三臂关闭Sonnet、CCE及全部TEB；S2 Passed/leading结论与组合guard不变，不进入M5/M7。

P2精确数学、工程身份与初始化合同保持：沿用M4 §§47.2–47.4、§50.6，implementation_variant=el-amd-m4-pmcr-local-change-p2-v1，initialization_policy=matched_amd_pmcr_body_and_isolated_gate_v1，run seed=2024、body_init_seed=2024、gate_init_seed=2025；2025仅为隔离gate子流，不是第二个实验seed。文档closure、A/B最小接口review/closure及C生产实现review/closure均已完成；P2 production implementation complete、engineering gate=Passed、ChatGPT final implementation review=Passed，有限工程证据及skip仍按M4 §52.6–52.9原范围解释。ETTm1 Stage 1已完成12/12 runs、120 run-epochs、串行墙钟约2.194 h，完整性及C vs B/C vs A安全gate均Passed并获ChatGPT接受；未显示P2优于v1，相对AMD的test改善主要由H720带动，移除H720后test MSE macro约退化0.035685%，不外推为跨H一致收益。用户已确认的M4 §§47.6–47.8总24-run、10 epochs及原门槛不变；UrbanEV后半12/12 runs、120 run-epochs现已完成，串行launcher墙钟45795.64994764328 s（约12.721 h）。ChatGPT接受完整性Passed及C vs B、C vs A主效果gate均Not passed：validation MSE macro相对变化分别为-0.088164%和-0.466125%，均未达到≤-0.5%；MAE、4/4 H改善、逐H安全和全部leave-one-out条件通过。精确未舍入值、两种macro及证据见M4 §55；gate仍只用100×(mean(C_h)/mean(R_h)−1)，不以逐H相对变化均值或舍入更换裁决。结合ETTm1两个安全gate Passed，原24-run协议下P2 development adequacy=Not passed，不登记positive development signal；这不撤销工程Passed/Completed，也不外推所有PMCR/P2结构无效。原序列在此停止，不自动追加seed、epoch、调参或组合。已完成独立run并发短测：最终有效轮496步、S₂ median=1.7420710448402421，6组保存短轨迹逐位一致；准备占比较高，不能外推完整训练速度或逐位一致。较长训练段四配置对照现已完成，有限合并证据与排除跨时段R3的描述性敏感性检查均支持本负载优先S4T4，S4T2无稳定额外收益（§9.6.4、M4 §57）。P2曲线/CPU参数分析已完成，不能代替激活贡献证据；四horizon固定validation子集的原504次batch前向及全部上限现已由用户批准并执行，8份权重各读一次，0优化/反向步骤；normal与生产路径逐位一致，参数/buffer/mode/RNG保护及test隔离通过。已训练C在该子集上依赖P2，时间均值化的效果并非四H一致，不能据此重裁原adequacy；ChatGPT已接受本次结果（M4 §58）；新“目标历史局部形状残差支路”以原RevIN目标历史的有符号形状特征直接生成DDI后目标hidden修正，§59结构与工程身份已获用户确认；生产代码及18项永久测试已写入；4项缺历史依据的方法现有用户明确的本次限制批准，历史缺口不改为Recovered。当前策略及重建工具无模型自检通过，CPU f32/f64、A800 CUDA f32的部分模块数学/梯度/部署及单次RevIN路由已实测通过；§60.7历史新增验收在关闭A与冻结AMD的CPU T512输入梯度逐位相等断言处停止，7项通过、1项失败、10项未执行，继承314项和真实probe未启动。§60.8的4/4/0次CPU诊断及HEAD版第二组非finite缺失记录保留。§60.9定位证据已获接受；本轮按用户明确例外实施共同DDI CPU GELU同值cotangent连续化（ddi_cpu_gelu_cotangent_contiguous_v1），局部永久回归Passed。未修复冻结源码的前向参照在CPU T12/T512均逐位相等；修后AMD与关闭A的输入梯度也均逐位相等，反向两侧共同依赖已修复common.py。但CPU T512仍有rev_norm.affine_weight的1/7元素差7.450580596923828e-09，原整模型torch.equal未通过；无NaN/Inf，不因此宣布差异无害。§60.10的1＋18＋314序列1 passed、1 failure、331 unexecuted及累计114/30/0保持历史事实。§§60.11–60.12的参考政策修订、weight通过及bias旧逐位失败保留历史。§60.13的CUDA bias前置、限定float32仿射组对称1e-7/1e-6及关闭等价结果已获ChatGPT接受；当轮生产入口NameError和未完成的后续验收保留历史。§60.14单行导入修复及当时静态-only停止保持历史。最新§60.20用户明确批准新增合成累计上限824/193/49，携入621/153/33，临时fixture修复后的唯一封存验收序列已完整通过：GELU1＋THLS18全部Passed；继承314为293 passed＋21原精确策略skip。共333 ID=312 passed＋21 skipped，failure/error/blocked/unexecuted均0，不能称333项全部通过。CPU/A800 CUDA实际覆盖原必需case；原36 forward/4 backward/0 Adam真实UrbanEV单批probe完成，四horizon A/N的train/validation前缀、CPU/CUDA batch2及h3/h12 CUDA batch128检查通过，test读取越界/解析/构造/迭代/评价哨兵均0。全文件读取仅为既定字节指纹，实际观测解析限原3909时间点前缀；无历史checkpoint读取、完整评价或development产物发布。三个业务阶段非预期受禁访问均0。新增本轮179/40/16、累计800/193/49，剩24/0/0；继承992/298/148及真实36/4/0另计，历史失败成本保留。未新增生产/测试repair；旧CCE可移植性断言在原字节下Passed。仅原指定float32 RevIN仿射参数组保留对称1e-7/1e-6数值规则，其他严格断言不变；报告整理的双方None字段错误从完整原记录修正，未重算模型。THLS implementation complete、engineering/implementation review Passed，统一代码closure=f13dd26dfa72e140c8e4c05b146ea39d6643f889；UrbanEV已提交文档/训练来源为e93f008f9f82546e13bf4c7f2522b1728d81cbdf，24-file source=16343e9298a9c4a1eb477c061f7b58552ab04b0706f751b8b83e42749b703a7f。§61独立N160步兼容和训练前方案已获审；随后按用户明确授权执行A四H→N四H两波，8/8、80 run-epochs、594040有效Adam步骤、墙钟22573.999078273773秒。最新ChatGPT接受完整性Passed、UrbanEV五项gate Passed及严格限本合同的positive development signal：N/A主MSE macro−0.873577%、MAE−0.715574%，四H及全部leave-one-out改善；旧中断A-h3成本与审计脚本修正保留（M4 §62.1–62.4）。ETTm1最小接入incremental engineering及ChatGPT implementation review均Passed，统一closure/训练HEAD=667a1b82bdf799a10a6ce7d813b424c0f223a5cf，24-file source=4d707b1d878d3490a55243e060a3d43258c60a70414edf23d2ff3fc4acedbc9a。已批336/320/320兼容检查完成后采用A四H→N四H两波；8/8、80 run-epochs、84260 Adam步骤、实际墙钟3142.443908929825秒。ChatGPT本轮接受完整性与§62.7四项安全线全部Passed：主validation MSE macro +0.112646595%、development-test MSE −0.032258338%、MAE −0.160544776%，最差H720 test MSE +0.466123118%；两种macro和逐H原值见M4 §64.2，不能称所有H均提升。结合UrbanEV五项gate Passed，THLS单模块本轮开发验证收口，ETTm1仍是development-only test；旧source和审计Pending快照不追写。§66–67双数据集N/S/J比较已完成统一接入、工程/review/closure及兼容，训练来源为4abb099d69db04532f066986fde0c43aedf2465a、25-file source=6a11d88f6d22861e49db0122bf685b642b197af2796ab8de4b7a424b4548a97a。24run按E-S→U-S→E-N→U-N→E-J→U-J六波完成，240epochs/1017450 Adam，墙钟10小时59分43.71秒。§68本轮结果review Passed；UrbanEV J/S与J/N各5/5、ETTm1 J/S 4/4，J/N因H192 test MSE +1.542951806%超过1%而3/4；原18项总gate Not passed、科学序列停止。ETTm1两组validation MSE macro安全项分别−0.049560234%和−0.133132443%，均Passed，不因摘要只列test而省略。有限十二run曲线未发现具体训练/选择异常，不自动重训、改模型或准入。以下§65失败/修复为已被§67续验取代的历史记录，不是当前工程状态：用户当时明确确认§64唯一结构/独立身份、8 fresh runs/80 run-epochs、五项效果gate及三笔独立额度；§65四文件组合接入和八项新增验收已执行至6 passed、1 summary负例failure、1 unexecuted。source负例参数传递错误已最小修复并静态检查；累计123/45/24、剩37/19/8，不足以按原要求在修后版本完整复验，当时engineering Not passed/未完成、implementation review/closure Pending，真实20/8/0未执行。当时25-file source=75458a103145f60b98314bd206330e45442ce65043c3161362208cecb10207e8；原工具/skip/模型数学及数值边界不改，当轮320步兼容和八run未启动。当前已完成新24-run且总gate Not passed；S2 Passed/leading、THLS单模块已接受结果、P2原Not passed、M4 In Progress保持，正式结构/variant未冻结，不启动M5/M6/M7。

ETTm1 自 M4 第五轮起固定登记为 **development-only diagnostic benchmark**。M4 允许使用 ETTm1 的 train、validation 和 test 进行候选结构、容量与超参数探索；现有 production runner 可以继续按 `train -> validation -> validation 选择 best checkpoint -> test` 运行并生成完整 schema-v2 artifact，不要求为 ETTm1 实现 validation-only runner、独立 schema 或独立 summarizer。

ETTm1 test 已被纳入模型开发反馈，因此不得作为 M6 正式未见测试集结果，不得进入第三章正式性能主表，也不得用于最终无偏泛化主张。pre-M4 ETTm1 test 已查看并触发 M4 的事实继续保留，但不再被解释为“永久禁止参与后续 M4 开发决策”。论文可将 ETTm1 记录为开发集、诊断集或结构搜索数据集。

“测试集只在 M5 冻结后使用”的限制仅适用于当前正式评价数据集：UrbanEV、EPF-PJM、ETTh1、Weather、ECL、Exchange。这些数据集的结构和超参数选择在 M4/M5 仍只依据 train/validation，其 test 只在 M5 结构冻结后于 M6 使用。ETTm1 当前不属于 M6 正式主表；ETTh1 按本轮决定继续保留在六个正式数据集中，论文必须披露 ETTm1 曾用于 M4 development；本轮不替换数据源或删减任务。ETTm1 的 development-only 例外不得扩展到任何正式评价数据集。

正式政策 **formal seed list=[2024]** 继续适用于第三/四章正式比较、消融及对应预算；不追加seed、不择优，不改变其他模块已登记的初始化子流。正式结果报固定seed 2024、seed std=N/A，三seed一致性暂缓、稳定性Not evaluated；第三章效果评价范围及冻结前置依本次修订的§9.3/§9.5/§21，不保留旧多域单模块效果或新practical-effect作为冻结前置。MS精确合同见§5、§9及M4 §50，字段选择已User confirmed；Weather/ECL/Exchange/PJM的元数据/接口缺口仍按任务blocked，不因合同批准核销事实。

旧“平均退化不超过0.5%”及筛选前锁定新practical-effect的冻结条件已由本次§9.5/§21替代，历史gate不倒改，不据此把旧失败记Passed。第三章两个近期来源模块、明确修改并接受消融评价仍是用户创新组织策略，并非学校或导师外部硬性要求；实际有效性由M6证据决定，负向结果须收缩论断，不能以结构冻结宣称两个模块均有效或为凑数量新增实验。

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

Early CCE 与 Late CCE 均未通过 M4 development adequacy gate；上述路由只保留为历史实现和负向 development 证据，不构成现行锁定方案。Sonnet/MVCA 的 S2 development-candidate 精确合同现已锁定，XLinear 尚未被选择；production implementation gate、ChatGPT implementation review 与 Git closure 均已完成，第二十三轮 paired development 得到 positive development signal 且 S2 adequacy gate 已 Passed。该结论只使 S2 成为当前 M4 leading development candidate，不能称为最终结构。

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
  - production capability、ChatGPT implementation review 与 Git closure 已完成；ETTm1/UrbanEV 16-run paired development 已按锁定合同完成并通过双数据 gate，当前只登记 S2 为 M4 leading development candidate，尚无最终结构结论；
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

development 协议固定如下；第二十三轮已经从 Stage B clean closure commit 按此合同顺序完成全部 16 个 10-epoch paired runs：

| 数据 | 固定合同 |
|---|---|
| ETTm1 | `target_exogenous`；OT；target 6；aux `[0,1,2,3,4,5]`；T=512；H=`[96,192,336,720]`；seed 2024；10 epochs；batch 32；Adam lr `3e-5`、weight decay `1e-7`；PMCR/旧外生模块 off；允许 train/validation/development test，不能进入 M6 正式主表 |
| UrbanEV | F4；fold 6；history 12；label horizons `[3,6,9,12]`；volume；target 0；aux `[1,...,10]`；seed 2024；10 epochs；batch 128；Adam lr `3e-5`、weight decay `1e-7`；PMCR/旧外生模块 off；仅 train+validation |

UrbanEV 的 `train_validation_only` 必须在数据构造层隔离：不创建 test Dataset/DataLoader，不遍历 test，不预测或输出 test metrics，不伪造 test 字段，也不把 validation 写入 test。完整数据文件只可参与 byte-level fingerprint。必须复用 M1 fold/split/train-only scaler。其主门禁为 validation MSE macro 小于 control、validation MAE macro 不高于 control、至少 3/4 horizon validation MSE 改善、改善不由单一 horizon 驱动且是 full-precision 非舍入差。ETTm1 安全门禁为 validation MSE macro 退化不超过 0.5%、development test MSE/MAE macro 各退化不超过 0.5%、任一 horizon development test MSE 退化不超过 1.0%。只有 UrbanEV 主门禁与 ETTm1 安全门禁同时通过，才是 positive development signal；任一失败即按预登记停止，禁止自动调 d/K/alpha/gamma/dropout/epsilon/预算/batch 或转向其他阶段。

第二十三轮实际结果从 clean implementation commit `bd1e0ab45f7329d3eb8c24eed106de19e21d9884` 启动，8 个 ETTm1 与 8 个 UrbanEV artifact 均完成 10 epochs 并通过 schema-v2、13/13 checksum、Python verifier、系统 `sha256sum -c`、identity、finite 与无 staging 检查。full-precision macro 证据为：

| development gate | control | candidate | relative change of macro means | 结论 |
|---|---:|---:|---:|---|
| UrbanEV validation MSE | 0.487974883666633 | 0.446404234951513 | -8.519014% | 4/4 horizon 改善，主门禁通过 |
| UrbanEV validation MAE | 0.465939237469093 | 0.438367005436726 | -5.917560% | 不退化，主门禁通过 |
| ETTm1 validation MSE | 0.079599271299896 | 0.079622306340827 | +0.028939% | 低于 +0.5% 安全上限 |
| ETTm1 development-test MSE | 0.048442540257879 | 0.047921259864012 | -1.076080% | 改善 |
| ETTm1 development-test MAE | 0.165082955197693 | 0.163868899651695 | -0.735422% | 改善；最大单 horizon test MSE 退化仅 h192 +0.083839% |

因此登记 `positive development signal` 与 `Sonnet S2 development adequacy gate = Passed`。该裁决只覆盖已锁定的 S2、两个 development 数据协议、seed 2024 与 10-epoch 预算；ETTm1 test 是 candidate-development feedback，不能进入 M6 正式主表。它不证明所有 Sonnet 结构有效，也不构成因果归因、M5 冻结、最终外生模块或最终 EL-AMD 结论。Sonnet S2 仅成为当前 M4 leading development candidate；M4 保持 In Progress，下一结构未选择。


第二十三轮实际 artifact root 为 `artifacts/m4-development/ettm1-stage-i-sonnet-mvca-v1` 与 `artifacts/m4-development/urbanev-stage-i-sonnet-mvca-v1`。两者沿用 schema-v2 的 13-file payload、checksums、hidden staging 与 atomic publication，并显式记录 `evaluation_policy=train_validation_test|train_validation_only`、`artifact_purpose=m4_development_candidate`。`train_validation_only` 的 `metrics.json` 不得含 `test` key，manifest 不得含 `test_mse`、`test_mae` 或其他 test result 字段，但必须固定 `test_access_policy=forbidden`；RuntimeData 不得持有可遍历 test loader；summarizer 使用独立 validation-only 分支并拒绝混合 evaluation policy。不得降低既有 test-inclusive schema-v2 合同。

上述全部结构、source paper/repo/commit/PDF/core SHA、license-text-missing、retained/deleted components、raw/latent order、d/K/alpha/epsilon/dropout/gamma、FFT/denominator/scale/softmax/var_attn/reconstruction/readout/init policy、task/schema/order、evaluation policy 与 module-init-seed policy必须进入 resolved config、scientific/comparison hash、checkpoint、manifest、resume preflight 和 summarizer。resume 只允许同一 Sonnet variant/ablation/task/schema/order/T/d/K/alpha/policy/evaluation/scientific config 的 `strict=True` restore；跨 TimeXer/CrossLinear/XLinear、control/candidate、target/parallel、schema/order或结构不匹配、partial `sonnet_mvca.*`、key/shape/dtype 不匹配均须在写参前原子拒绝。

## 1.2 第四章：时空模型

用户已明确批准三个历史来源例外：HSTGCN-core来源HSTGCN、SADR来源ASTGRN、SC-SimGCA来源G-STAN；三篇期刊来源继续保留，不被新替代候选的“2023年及以后顶会”规则自动淘汰，不重问例外、不换成PDFormer或其他新结构。本次新增原版复现准入顺序及有限止损原则，见§10，不改变下述模块数学。

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

## 5.1 第三章统一 MS 精确合同与历史兼容边界（User confirmed）

第三章六个正式数据集统一采用“多变量历史输入、单个指定目标输出”的target_exogenous / MS精确任务合同，已由用户确认，本轮文档待ChatGPT审核/closure；最新确认见§9.1与M4 §50。MS不是S-to-S、target-only input，也不表示pred_len=1。除明确的target-only输入消融外，保留各数据集全部既定历史变量、fold与horizon；字段确认不等于未完成正式接口已经可用。

~~~text
task_mode = target_exogenous
feature_type = MS
input = [B,T,C]                  # 一个指定目标历史 + ordered aux 历史
target_indices = [target_idx]
prediction = [B,H,1]             # 指定目标的完整 H 步；UrbanEV 的 H_out=1 另见 §5.4
~~~

固定目标（去掉时间列后的0-based feature index）为：UrbanEV volume/0；EPF-PJM业务名price ↔ 原CSV字段OT/2；ETTh1 OT/6；Weather T (degC)/1；ECL OT/320；Exchange OT/7。六项字段选择均为User confirmed，绑定M4 §49.3的数据版本/SHA、名称、索引及输入顺序，不改写CSV、不按模型表现重新筛选。数据准入按用户本次确认的M5 §16收口，替代M4 §50.2在这些事实上的旧复合mandatory：版本/字段/端点是硬约束，已接受来源依据和解释范围单列，未知业务解释不能写成Passed。Weather作者分发文件已有重复时间，仅该域允许非唯一但非递减时间，按原记录T/H建窗，逆序拒绝；原行、原序、原值及36887/42157/52696端点不变，禁止删除/去重/排序/插值/补时间/重采样，不声称每步严格10分钟或重复记录是新样本扩增。ECL 321列/OT320与Exchange 8列/OT7的作者benchmark文本身份已证，原split及train-only scaler不变；ECL原转换全链条/客户业务身份/物理单位恢复、Exchange币种/基准币/报价方向未知改为必须披露的解释限制，不要求客户实名，不推导物理用电误差或交易收益。两域只报告既定train-standardized指标。PJM使用TimeXer@76011909357972bd55a27adba2e1be994d81b327的已核验A类CSV，不换Zenodo文件；n=52416，端点36691/41933/52416。价格及系统/COMED负荷forecast角色沿TimeXer/原EPF；原EPF §4.1的d−1可得性是documented source assumption，非record-level vintage audit Passed。逐记录发行审计、TimeXer到原EPF完整转换重建及两forecast未明物理单位保留为限制，不再作为当前运行前置，不声称没有潜在源数据风险。仍仅输入既定历史窗口，不增加未来forecast；全部test隔离、版本/字段顺序/切分/目标/scaler硬约束保持。

原“ETTh1/Weather/ECL/Exchange 采用标准 M-to-M，输出全部变量”的正式矩阵，被本轮 MS 调整方向替代。parallel_multivariate 的既有实现、永久测试、checkpoint 与历史指标保留兼容和证据身份；它不再是当前第三章正式任务规划。第 7 节的 parallel-TEB 仅属失败历史工程合同，不因为保留兼容代码而成为未来正式结构。

第四章仍预测全部节点：单目标指每节点一个目标特征，不是全图只预测一个节点。UrbanEV 的全部 275 区域、CHARGED 六城市、PEMS04/08 的全部节点及未来 12 点多步输出均保留。无辅助输入时 S2 所需非空 aux 尚不成立，须在后续本域输入合同中单列处理；本轮 MS 调整不证明全部图任务已被支持，不添加辅助模块或改图路线。

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

第三章所有 MS 任务的 AMD/AMD-Concat 与 EL-AMD 保留 AMD 的全部通道内部计算和 AMS 输出；完成 RevIN 全通道反归一化后，只返回显式 `target_idx` 对应的预测，并只在该目标上计算主 loss。下例的 B_region 对一般长序列数据即 B；第四章按全部节点恢复形状，不裁减节点。

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

## 5.5 新正式任务的损失、信息集与身份（User confirmed）

正式训练合同（2026-09-19 User confirmed，M5 §17及§18取代此前所有baseline必须使用AMD数据集batch/LR的条款）：下表batch/eval batch/LR严格适用于A/J及既定N/S；epochs和early-stop仍适用于全部模型。DLinear、PatchTST、iTransformer、TimeMixer、ModernTCN、TimeXer仅允许经来源核定的batch/eval batch/LR覆盖，唯一映射为configs/ch3_formal_profiles.json的baseline_training_overrides，按共同项→dataset→外部覆盖解析；不得覆盖T、epochs、patience、scheduler、结构等其他项。论文初始LR作为项目固定LR，eval batch默认等于train batch，不线性缩放LR。§18已由用户明确选择本模型锁定官方脚本补值，57个来源待决run已清零；iTransformer/ECL采用脚本B16而非论文通用B32的差异明确保留。没有对应论文的迁移域维持既定项目值。新增未裁决来源仍Blocked，不能将缺省值冒称来源已核实。当前495任务范围沿M5 §13，无F0/TargetOnly；A/J输入消融与N/S匹配政策不变。

| dataset | train/eval batch | epochs | LR | early-stop |
|---|---|---|---|---|
| UrbanEV | 128/128 | fixed10 | 3e-5 | off |
| PJM | 128/128 | max20 | 5e-5 | patience5 |
| ETTh1 | 128/128 | fixed10 | 5e-5 | off |
| Weather | 128/128 | fixed10 | 5e-5 | off |
| ECL | 128/128 | fixed20 | 3e-4 | off |
| Exchange | 512/512 | fixed10 | 3e-4 | off |

共同使用Adam、betas=(.9,.999)、eps=1e-8、weight_decay=1e-7；固定LR，无scheduler/warmup/OneCycle；float32、梯度累积1、workers0、每进程4线程；train shuffle/drop_last=True，eval不shuffle且保留尾批。每任务每模型单配置、0额外搜索、from-scratch，formal seed=[2024]、std=N/A、稳定性Not evaluated；已登记匹配初始化子流不变。不改变既有TF32/确定性等环境设置，不新增AMP/compile。仅有限目标validation MSE严格下降更新best，相等保留早epoch；PJM严格改善重置耐心，连续5轮无改善的完整epoch结束后停止，最大20轮；其余按fixed预算完成，不早停。不以best更晚为目标，不择run/seed。论文称“统一T、训练轮数/停止及信息评价边界下，外部baseline有限batch/初始LR来源迁移比较”；不称完整作者recipe复现、每模型最优或纯结构差异被完全隔离，不以未经核实的“其他baseline必然做过搜索”解释差异。

PJM项目协议确认F=1：按时间顺序，train=floor(.7n)、test=floor(.2n)、val=n−train−test，无rolling、无train+val额外重训；T168/H24和原历史信息集不变。实际数据版本、市场/as-of/单位等事实和接口缺口继续blocked。合同及545-run规划不授予本轮正式训练；新身份/模型适配/早停/汇总能力须另行接入验收。

主预测 loss、validation best selection 和最终指标只覆盖指定目标。标准长序列任务保留 H=96/192/336/720，对每个窗口的全部未来 H 步评价；不采用 Sonnet 论文 §2 的仅末时刻评价。UrbanEV 则保持 label_horizon 与 model_pred_len=1 的单点语义，EPF-PJM 保持 168→24 完整区间。

原生多输出模型可保留 [B,H,C] 内部输出，但须在训练 loss、validation、final evaluation 三处显式选择同一个命名目标；不能监督所有未来变量后只在末端切目标并称为同一 MS 合同。AMD 的 MoE 等固有辅助正则单列，不能借辅助项引入其他未来标签。

同一数据任务内，各比较模型使用相同lookback：ETTh1/Weather/ECL=512，Exchange=96，UrbanEV=12，PJM=168。输入列、名称与ordered aux沿用M4 §49.3；标准CSV日期列只作索引，不给个别baseline额外calendar/time-mark，UrbanEV保留F4已有历史日历特征。输入列可因已登记的模型接口确定性重排，但须保存原列→模型列→目标的双向映射与ordered aux，不静默增删变量或仅靠“最后一列”定位。DLinear、PatchTST等无跨变量信息路径时如实说明，不擅加融合模块。未来真实天气、负荷、汇率、价格等协变量不得输入；特殊baseline所需未来占位/接口须另行验收，不由本次合同确认自动批准或启用未完成适配。

主指标空间确认metric_space=train-standardized。每run对全部有效目标元素累加SSE/SAE，再除以元素数Q；评价末尾不完整batch全部覆盖，不平均batch均值。标准任务评价所有H点，PJM为24点，UrbanEV为指定未来偏移单点及全部区域样本。仅当目标validation MSE严格下降才更新best，等值保留较早epoch；不以test或事后改用原单位指标选best。

正式汇总先列逐fold/horizon未舍入指标：UrbanEV每个horizon对六fold等权macro，再对四horizon等权macro；标准数据对四horizon等权macro。pooled SSE/SAE÷全部元素数仅作明确加权的补充，另列；不同数据集原始误差不直接平均。原单位指标若后续列报，须使用目标scaler，UrbanEV还须绑定node，单独注明metric space和可信单位。正式test只在M5冻结后使用，冻结前接入须隔离test；ETTm1 development例外不扩散，旧development artifact不升级为正式结果。上述正式macro不把P2 development的fold6扩为六fold，不追溯修改S2等既有development指标、门槛、hash或结论。

正式任务合同名确认 **ch3-ms-specified-target-full-horizon-v1**；本轮只锁定文档语义，不注册生产入口或宣称正式管线可用。未来formal purpose与M4 development purpose分开，按M4 §49.6绑定数据版本/SHA、task/目标字段/原始身份/索引、输入顺序/重排与ordered aux、split/scaler、T/H及标签位置、loss/metric scope、metric space/macro、test policy、formal seed list、模型/模块初始化及训练身份。配置、科学/比较hash、checkpoint、manifest、resume和summarizer须一致；跨task、target、metric scope、purpose、schema等不一致时，在权重反序列化/写参前拒绝恢复，同shape不等于同协议。历史U/M、M-to-M、S2、v1身份及hash不重定义，旧checkpoint/指标不改名复用为新MS结果。

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

2026-09-07 当前状态接口边界：TEB 关闭时继续使用 `state_source = concat(v_final_target, u_mdm_target, zero_context)`，shape 为 `[B,2*T+teb_context_dim]`。第三段是兼容零占位，不是有效外生摘要；S2 经预测路径影响前两段，不等于产生独立 exo_context。M5 冻结后，在 M7 依据最终时间模型落实状态适配器及其后续训练监督；下方 StateAdapter 公式及 `d_s=16/state_dim=32` 仅保留为历史建议，不是当前已锁定的新公式、宽度或训练协议。本轮不恢复失败 TEB、不添加 Sonnet global/context head、不删除零占位、不修改 state keys/宽度，不实现 H_time、Graph Mode 或空间模块。

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

原先按target_exogenous与M-to-M划分正式矩阵的规划，已由用户确认的统一MS精确合同替代；六数据集字段选择及共同lookback已确认，保留两组既有baseline清单和原消融覆盖范围，不扩增模型、目标或消融。字段确认、Not verified元数据、未实现接口及实验授权分别登记，不能将本轮合同确认视为正式矩阵可执行。

本节及后续第四章规划中未带 variant 后缀的 “EL-AMD”，统一表示由 M5 最终冻结的第三章增强时间模型。它是项目/模型族名称，不等同于当前任何候选；S2 仍仅为 M4 leading development candidate，Sonnet+PMCR/P2 组合尚未实现或授权。正式 variant 与模块角色须在 M5 冻结，不能用无后缀名称暗示已入选。

## 9.1 必做数据集

六数据集均采用已确认的多变量历史输入、单个指定目标输出MS合同，原fold/horizon数保持；下表index均为去掉时间列后的0-based feature index：

| 数据集 | 唯一目标与确认状态 | 输入长度 / 预测合同 | 作用 |
|---|---|---|---|
| UrbanEV | volume，index=0；User confirmed，沿用既定合同 | T=12；label_horizon=3/6/9/12；model_pred_len=1；6 folds、全部区域；F4主表及既有输入消融 | 核心EV场景；正式执行仍待冻结 |
| EPF-PJM | price ↔ 原字段OT，index=2；项目双向字段映射User confirmed，不改CSV | T=168→完整24步；原历史输入；F=1、时间顺序70/10/20取整及无rolling/retraining见§5.5 | 锁定TimeXer版本/端点；论文级可得性按M5 §16接受，vintage及转换限制披露；正式训练仍待授权 |
| ETTh1 | OT，index=6；User confirmed，沿用已核验字段与数据版本 | T=512；H=96/192/336/720，完整区间 | 标准benchmark；新正式接口仍待验收 |
| Weather | T (degC)，index=1；字段选择User confirmed | T=512；四个H，完整区间 | 原记录建窗，时间非递减；重复及非严格10分钟限制按M5 §16披露 |
| ECL | OT，index=320；字段选择User confirmed | T=512；四个H，完整区间 | 作者benchmark 321列身份已证；仅标准化评价，转换/物理单位限制披露 |
| Exchange | OT，index=7；字段选择User confirmed | T=96；四个H，完整区间 | 作者benchmark 8列槽位已证；币种/报价方向未知披露，不作交易解释 |

完整dataset/version、target/index/unit、ordered aux、shape与split/scaler证据沿用M4 §49.3，当前来源/准入以M5 §§15–16为准，§50保留历史事实；上述T值现为同任务各比较模型的共同历史长度合同，不套用其他论文的T或仅末时刻指标。目标不由test、相关性排名或模型效果选择，当前用户批准以已核验ECL/Exchange基准列身份开展标准化比较，未知业务解释必须披露；不猜物理单位或币种。M4 development仅用UrbanEV fold6，正式六fold汇总不扩增P2范围。

最低正式范围仍为 UrbanEV + EPF-PJM + ETTh1 + Weather + ECL + Exchange。Solar、其他 EPF 市场仍是另行授权的可选扩展，不计当前默认资源情景；ETTm1 仅为 development，不进入 M6 正式主表。不得为了适配 S2 减少数据集、fold、horizon 或历史变量。

## 9.2 Baseline

统一 MS 不合并或扩大原有两组模型名单：

| 适用数据集组 | 8 个训练模型，保持原数量 |
|---|---|
| UrbanEV / EPF-PJM | DLinear、PatchTST、iTransformer、TiDE、TimeXer、ModernTCN、AMD-Concat、EL-AMD |
| ETTh1 / Weather / ECL / Exchange | DLinear、PatchTST、iTransformer、TimeMixer、ModernTCN、TimeXer、AMD（同输入 MS 输出适配）、EL-AMD |

TiDE本轮Deferred：保留在完整规划名单，UrbanEV/PJM当前分别推进其余7模型；标准四域仍8模型。仅暂缓TiDE对应25 runs，不新增替代模型。作者源码smoke中的AMD-upstream只作参考，实际A和不可变baseline不变；完整Sonnet只核对S2来源，不能作为新增正式比较模型或代替项目S2/J证据。

Last Observation 仍仅在 UrbanEV 单列评价，不计神经网络训练。第二组 TimeXer 的原 parallel 路径拟改为目标外生路径：features=MS、n_vars=1、显式目标末列映射；这是新任务适配，不重命名旧 TimeXer-parallel 结果。保留 TimeMixer，不因统一 MS 而向第二组添加 TiDE，也不添加完整 Sonnet、XLinear 或其他模型。

同一dataset/target/horizon/fold按§5.5统一lookback、信息集、split/scaler、mask、目标损失及指标聚合，并统一epoch/停止规则与共同优化配置；batch/eval batch及初始LR仅外部baseline按§5.5/ M5 §17已确认来源层覆盖，AMD家族保持原表。各模型可保留有来源的原生hidden/layer/patch/kernel等结构项，但每任务每模型单配置、0额外搜索；本次允许上述有限模型专属batch/LR，不新增搜索或暗中增加验证机会。AMD家族同任务公共骨干/共有模块匹配初始化，主表A/J与消融配置兼容时引用同一run。六域主表保留同输入AMD与最终增强模型；用户本次已明确冻结J为EL-AMD，执行身份保留J，正式结构身份为el-amd-s2-thls-v1。正式适配缺口继续按M5登记，来源默认值不是已批准结构表缺项的替代证据。

## 9.3 消融矩阵：当前角色与历史 identity 分离

未来正式消融按 M5 最终冻结的“外生模块”“局部模块”描述，不让历史 Global TEB 或 parallel-TEB 冒充未来结构：

| 当前正式规划角色（尚未注册新 ablation identity） | 已确认覆盖范围 | 比较问题 |
|---|---|---|
| 同输入 AMD / AMD-Concat | 六数据集主表已含 | 基线 |
| 仅最终冻结外生模块 | UrbanEV F4全部6fold×4H | 核心场景外生模块独立贡献 |
| 仅最终冻结局部模块 | UrbanEV F4全部6fold×4H | 核心场景局部模块独立贡献 |
| 最终冻结完整 EL-AMD | 六数据集主表已含 | 完整模型收益 |

全部消融只在UrbanEV：F4完整A/N/S/J模块消融中，A/J直接引用主表同run，只新增N/S；输入消融为A/J双方F1–F4，F1–F3各新增训练、F4各复用主表。§13仅恢复§12撤下的AMD/F1–F3共72个计划，不恢复F0或任何PJM TargetOnly，不增加替代实验。其他五域只有既定主表；六域主表只能检验完整模型表现，不能证明每个模块跨六域独立有效。M4旧artifact不得升级、改purpose/manifest或跨协议恢复；不为换阶段/表格重训。

F0=volume-only字段定义、旧代码能力和合法合成工程测试保留，但不产生当前正式run、probe代表或评价表行。J/F0非法且不以N/F0替代；主比较和模块消融均F4，A/J输入范围见§9.4。同组不能改变输入顺序、fold、scaler、训练政策或评价流程。

历史 identity 仅供追溯，含义不重定义：U0=AMD-TargetOnly、U1=AMD-Concat、U2=U1+TEB、U3=U1+PMCR、U4=M3 Global TEB v1+PMCR v1 完整工程候选；旧 parallel 消融 M0=AMD、M1=AMD+PMCR、M2=AMD+parallel-TEB、M3=AMD+PMCR+parallel-TEB。此处 M0–M3 是旧消融编号，与工程 milestone 不同。已有实现、测试、hash、checkpoint 和结果保留；新 MS 正式角色需独立 identity，不复用这些编号暗示换了任务或模块仍属旧协议。

## 9.4 UrbanEV 辅助变量消融

| input_variant | 输入 | C / aux | 当前角色 |
|---|---|---:|---|
| F0 | volume only | 1 / 0 | 仅定义/历史能力，不产生正式任务 |
| F1 | volume + calendar | 6 / 5 | A/J同输入消融 |
| F2 | F1 + e_price + s_price | 8 / 7 | A/J同输入消融，与F3并列 |
| F3 | F1 + weather | 9 / 8 | A/J同输入消融，与F2并列 |
| F4 | volume + calendar + price + weather | 11 / 10 | A/J输入消融各复用主表；A/N/S/J模块消融 |

F0的compact tensor只有`volume`，`target_idx=0`、`aux_idx=()`。J/S必须有非空有序aux；J/F0在配置/声明阶段明确拒绝，不能自动关闭S2仍叫J，不能用零辅助或复制目标伪造合法输入。当前正式表不产生F0行，旧F0上J的N/A事实不变。

F1–F4保持canonical字段顺序，volume为第0通道，其余为非空有序aux。F2与F3不是逐级加字段。每个F1–F4内A/J的字段/顺序、目标、scaler、fold、标签、训练及评价完全一致；A关闭增强模块，J同时启用S2/THLS。沿既有独立模块RNG子流隔离策略匹配公共AMD初始化，使用独立且同序的数据generator，不仅以seed相同代替匹配条件。汇总按相同输入方案给A/J对照，不新增run或择优统计。F4主表、模块表、输入表引用同一run。schema继续`ch3-target-ms-formal-v2`/`input_variant`，只更新真实任务/配置指纹；旧M4/封存证据不追改。

报告拆为两个 panel：

- 模块表：UrbanEV F4 A/N/S/J，A/J引用主表，只新增N/S。
- 输入表：UrbanEV A/J各F1/F2/F3/F4，双方F4引用主表；不生成F0或PJM TargetOnly表。

## 9.5 模块验收线

用户本次确认以“正式协议优先”替代旧冻结前多域模块效果筛选、0.5%安全/新practical-effect前置；不新增完整M5性能筛选矩阵，不声称旧门槛已满足。M5冻结前须完成正式协议/配置与必要工程验收准备：目标/信息集、身份/恢复、train-only scaler、test隔离、指标/选择及执行配置可核验；涉及结构适用性的事实或配置冲突必须先解决，其他未闭环任务保持blocked。随后依据既有开发证据与明确风险接受，提请用户另作最终结构冻结决定。协议确认或loader通过均不等于冻结J。

M6按§9.3/§9.4完成六域主表、UrbanEV F4独立/组合模块及A/J的F1–F4同输入消融，逐任务报告效果、负交互及参数/耗时等代价，不承诺正收益、不增新量化效果门槛。冻结结构不等于效果Passed；无独立模块外部域实验时不声称跨域单模块有效；负向结果保留并收缩论文论断，不按dataset/H替换模型或自动调参重训。

formal seed=[2024]、std=N/A、随机初始化稳定性Not evaluated保持；fold/H差异不替代seed方差。M4 development及ETTm1 development-test不能进入正式未见test主表，不改旧身份/manifest或旧失败gate；新正式test仅在M5明确冻结后按批准任务执行。两个模块实际有效性仍待正式证据，不以关闭模块冒充“两模块已验证”。

## 9.6 正式实验资源预算与效率预检

本节历史资源情景保留；当前有效授权为已冻结J和现有495个正式run/最多5340 run-epochs，M5已收口、M6进入用户分模型启动准备。可选扩展、额外seed和新增搜索不在本次授权内。第三章矩阵与统一训练合同依§5.5/§9.1–9.5；formal seed=[2024]，多seed稳定性Not evaluated。工程/失败成本、效率预检及正式训练分别授权，旧P2等历史预算不转为新额度。

证据标签统一为：**Measured**＝已有实际运行记录或现场只读观察（注明来源和限制）；**Extrapolated**＝依据已测数据外推；**Scenario**＝条件性算术情景；**Unknown**＝尚无可靠测量或合同未定。不能把后三者写成实测均值、性能结论或保证上界。

### 9.6.1 第三章任务数与完整消融情景

依据§9.1–9.4，含Deferred TiDE时每个主表panel包含8个训练模型；当前UrbanEV/PJM暂推进7个，四标准域8个。六数据集采用用户已确认的统一MS合同，两组保留各自既有模型清单，不新增替代者或目标任务。下列仍是条件性资源核算，不代表被blocked的正式任务已可执行。

| 任务域 | configurations / model / seed | 条件与边界 |
|---|---:|---|
| UrbanEV | 6 fold × 4 horizon = 24 | 正式主表按 F4 情景核算 |
| ETTh1、Weather、ECL、Exchange | 4 数据集 × 4 horizon = 16 | 统一MS及每个数据集一个指定字段已确认；原H不变，元数据/接口缺口仍保留 |
| EPF-PJM | F_PJM=1 | 无rolling/retraining；项目切分政策见§5.5，数据事实缺口不核销 |
| Last Observation | 不计神经网络训练次数 | 单列评价成本，仍按既定适用任务评价 |
| ETTm1 | 不进入 M6 正式主表 | 仅为 development，M4 成本另计 |

主比较训练数为 **N_main = 8 × (40 + 1) = 328**，当前S=1、seed=2024、F_PJM=1已确认。MS不按输出变量数C缩减次数。以下多seed数仅为未授权历史扩展情景：

| seeds S | 1（当前默认） | 3（未授权扩展） | 5（未授权扩展） |
|---|---:|---:|---:|
| 主比较 trains，F_PJM=1 | 328 | 984 | 1640 |

**历史§9时点的545规划快照（已被本节下方423/448规划替代，不再执行）**：当时主表F4，模块消融集中UrbanEV，原输入/TargetOnly保留：

| 组成 | 新增 trains / seed，F_PJM=1 | 去重依据 |
|---|---:|---|
| 主比较 | 328 | 已含 AMD / AMD-Concat 与完整 EL-AMD |
| UrbanEV N/S模块消融 | 2×24 = 48 | A/J已计主表；其他五域不默认N/S |
| UrbanEV/PJM target-only | 24+1 = 25 | 保留原覆盖，不扩增四标准域TargetOnly |
| UrbanEV F1–F3，AMD-Concat / EL-AMD | 3×2×24 = 144 | F4 已在主表，F0 已在 target-only 角色，不重复计数 |
| 合计 | **545唯一runs** | 单配置、seed2024；同run跨表引用 |

按dataset核算：UrbanEV408×10=4080 run-epochs；PJM9×最多20=最多180；ETTh1/Weather/Exchange各32×10=各320；ECL32×20=640。合计545 runs、最多5860 run-epochs。F0上J为N/A；不额外相加旧M5的112+4F筛选包，不沿用旧579或463剩余数。原568+11F/579属于已被替代的六域模块消融规划，历史时延记录不因此改写。

可选F0+PMCR、额外市场/Solar、效率预检及M4 development不在545内；额外配置搜索=0，必要工程/失败成本另列。兼容的正式主表与消融引用同一run，不为换表/阶段重训；旧development/parallel artifact不得改名复用。未知窗口/步骤/实测耗时仍Unknown，1.4只可标并发假设，不据旧耗时情景承诺工期。

**历史三包规划（已被下方按模型分组取代，不再作为启动规则）**：上列545/5860为含TiDE完整规划；TiDE只在UrbanEV/PJM，暂缓24+1=25 runs、24×10+1×20=最多260 run-epochs。当时520 runs/最多5600如下，均为规划而非正式训练授权：

| 独立用户启动包 | 唯一runs | 最多run-epochs | 范围 |
|---|---:|---:|---|
| 包1 | 82 | 920 | A/J六域主表，2×(24+1+16) |
| 包2 | 221 | 2500 | 其余非TiDE baseline主表：5×25+6×16 |
| 包3 | 217 | 2180 | 剩余消融：UrbanEV N/S48＋TargetOnly25＋F1–F3 A/J144 |
| 当前推进合计 | **520** | **5600** | 同run跨表引用，不重复训练 |

上表历史规则原为包内自动、包间不自动启动；本轮由以下模型组规则整体取代。PJM fit不再列Unknown：F=1与时间顺序70/10/20取整已确认，版本端点与窗口已在M5 §§15–16闭环，论文级可得性按用户确认范围接受；逐条vintage及转换限制仍披露，实测步骤耗时未取得。1.4只是历史并发假设；TiDE暂缓不计消耗或完成，M4旧artifact不升级。

| 当前独立用户启动模型组 | 唯一runs | 最多run-epochs |
|---|---:|---:|
| AMD/A（六域主表及UrbanEV F1–F3；PJM仅MS） | 113 | 1180 |
| J（主表及UrbanEV F1–F3；F0 N/A） | 113 | 1180 |
| DLinear、PatchTST、iTransformer、ModernTCN、TimeXer，各自一组 | 各41 | 各460 |
| TimeMixer（四标准域） | 16 | 200 |
| N、S（各自仅UrbanEV F4） | 各24 | 各240 |
| 当前合计 | **495** | **5340** |

当前独立核算：六域主表303＋UrbanEV N/S额外48＋A/J F1–F3额外144＝495；UrbanEV360×10、PJM7×最多20、ETTh1/Weather/Exchange各32×10、ECL32×20＝最多5340 run-epochs。TiDE25/最多260 Deferred，含暂缓完整规划520/最多5600。当前含TiDE的520集合与§11旧520不同，不凭相同计数接受旧计划/许可。§12曾撤下97计划，本次仅恢复AMD/F1–F3的72个；F0的24个及PJM TargetOnly 1个不恢复。447/4860包含非法J/F0，是错误建议，不能登记为用户曾批准。历史产物/失败日志不删。

原规则为用户每次启动一个模型组，组内按dataset→UrbanEV input_variant→固定H/fold波次执行，不跨数据集/输入方案补位、不自动启动下一模型。2026-09-30用户明确授权仅对剩余DLinear→iTransformer→ModernTCN→TimeXer→N→S六组，由用户一次启动总队列，按原波次顺序执行并在每组真实退出、技术核验通过后自动衔接下一组；不等待逐组人工结果审核。任一技术失败或STOP停止后续，不自动重试。项目GPU互斥锁禁止两个模型入口同时占同一GPU，总队列锁与GPU组锁分开持有。原独立启动历史保留；本次总队列不改变科学合同或允许结果驱动调参。A/J主表run直接被消融引用；训练表、信息集、seed2024/0搜索及正式test冻结边界保持。

**第三章正文范围及完整工程矩阵（2026-09-30用户决定）**：正文主要数据集介绍、主结果表与分析围绕UrbanEV、PJM、Weather、Exchange。UrbanEV是核心EV充电需求预测；PJM是电力市场电价预测，目标price↔OT/index2，不称负荷预测；Weather/Exchange用于不同领域时间序列预测适用性。ETTh1/ECL/NP/BE/FR/DE不进入正文主结果表，但已有结果及全部已批准训练任务保留，不调整参数或执行顺序，不缩减为180任务。此为当前正文呈现范围，不倒写为最初全部评测范围；“学校每章不超过四个”尚未核实，不作为硬性校规。不同数据集不求总平均，域内fold/H汇总沿原合同；UrbanEV输入消融F1–F4、F4上的A/N/S/J模块消融保持，无F0或新增消融，J冻结身份和第四章路线保持。Informer/Autoformer未获批准替换或新增。

完整矩阵仍为552目标任务/6240目标run-epochs，含TimeMixer已批准重复尝试上限568/6440。剩余六组228任务/最多2640 run-epochs、7,438,600 optimizer steps、73波：前四模型各45/540，N/S各24/240；四市场追加在前四模型原组末尾。DLinear/iTransformer/TimeXer为15波，ModernTCN为16波（ECL q2两波），N/S各6波q4；PJM及四EPF市场均q1，其余已接受q4。总控制器不计作训练run。每组技术完成标记与结果审核分离，完整权重/科学结果审核在总队列结束后统一办理；实际启动仍须修后字节审核、统一closure及匹配总许可/六子许可。实施与41-run审计范围见唯一M6 §5.14。

### 9.6.2 第四章条件性计数与覆盖缺口

第四章单独核算（Scenario，当前 S=1、formal seed list=[2024]）：若 CHARGED 每城市仅一个正式切分、UrbanEV 主比较仍为 8 模型，且可复用完全匹配的 24 个第三章 S0，则新增主比较约为 **24×7 + 6×6 + 2×6 = 216 trains / seed**。三项分别是 UrbanEV 新增 7 模型、CHARGED 六城市各 6 模型、PEMS04/08 各 6 模型，依据 §10、§15–16。复用须任务、数据切分、输入输出、最终模型与训练身份等完全一致；任一条件不成立时重算，不能默认扣除 S0。

PEMS 未来 12 点是一次多步输出任务，不乘成 12 次独立训练。216 是名义 model/task/seed 主比较计数；空间消融、图构建、时间/空间训练阶段及辅助目标另计，不能把多阶段的实际 fit 次数或耗时隐藏在该数中。第四章总小时为 **Unknown，待本域管线校准**，不凭第三章时间模型外推，也不与时间模型耗时合并冒充实测。

目前覆盖缺口只登记，不因资源表而自行补实现：

- 原M-to-M规划与S2单目标数学之间的不一致已按用户确认的MS精确合同调整，字段选择不再待用户确认；Weather/ECL/Exchange/PJM按M5 §16细分版本事实、接受范围与未核业务解释，旧复合metadata阻塞已被精确新合同替代。S2公开runner/summarizer仍只锁定ETTm1/UrbanEV development身份，正式接入、完整H指标与冻结前test隔离仍需工程验收，不能称覆盖缺口已全部解决。
- EPF-PJM F=1、无rolling/retraining保持；限定train/validation前缀连通已通过。T168/H24/B128：train36500窗、285整批丢20；validation5219窗尾99；test10460窗尾92只作算术。论文级as-of依据与逐记录审计未做必须分开，不能据用户接受把未知forecast单位写成已验证。
- Sonnet+PMCR/P2 组合尚未实现、尚未授权；P2 specification review 不解除组合 guard。
- 正式epoch/早停与搜索合同现已确认于§5.5，相关接口/验收及正式执行预算未自动获批；不是继承M4旧开发身份。

MS精确合同与P2有限依赖调整已获用户确认；不得为补齐资源表重选目标、改成S-to-S、添加parallel Sonnet、恢复失败TEB或减少数据集。P2工程已获条件性授权，MS文档closure已完成；A/B最小接口已有实现、定向测试及单批证据（完整回归有受访问政策限制的skip，见M4 §51），ChatGPT implementation review=Passed且Git closure已完成；候选C production implementation complete、engineering gate=Passed，ChatGPT final implementation review=Passed且Git closure已完成（2f0bd489b7ebd408fba359811fe580984895fe10；有限工程范围见M4 §52.6–52.9）。M4 §§47.6–47.8原24-run现已完成；ETTm1两个安全gate Passed，UrbanEV完整性Passed、两个主效果gate Not passed，P2整体development adequacy=Not passed并停止该序列（M4 §55）。未参与下一轮两项P2 development任务的正式缺口继续blocked，不阻塞该有限工程范围；未实现模型/管线在效率预检中标blocked，不为profiling提前实施。S2结论、第四章路线与state_source/零context/M7边界不变。

### 9.6.3 完整 run 成本、现有证据与时间敏感性

完整 run 成本口径为：

~~~text
T_run = T_prepare
      + sum_epoch(T_train + T_validation)
      + T_final_evaluation
      + T_save_and_verify
~~~

prepare 包括必要的数据/模型准备；save/verify 计入各次 checkpoint 写入、最终保存及校验且不重复计费。H2D、数据预取与计算可能重叠，分项应注明计时范围，不把重叠分项直接相加；以完整 run 墙钟核对。final evaluation 仅指该阶段获准的数据边界：M4 UrbanEV 仍仅 train/validation，正式 test 不参与效率参数或模型选择。

必须分别记录串行等效工作量（各任务在独占条件下完整耗时之和）、实际 GPU 占用小时（按每张物理 GPU 的占用区间并集计时，再跨卡求和）和任务组墙钟时间（makespan）。并发进程重叠时长之和不是物理 GPU 占用小时；排队时间另列，GPU 利用率也不能替代占用时长或完成时间。

| 证据标签 | 已知资源数值 | 来源与适用限制 |
|---|---|---|
| Measured（继承历史记录） | ETTm1 8 runs = 5087.140 s；UrbanEV 8 runs = 31282.383 s；源记录总计 36369.524 s = 10.103 h | M4 §46.3 的 Sonnet paired development 资源记录；非本轮测量，不是正式全矩阵均值或并发 GPU 占用计量 |
| Measured（复用已审审计） | P2三臂ETTm1 Stage 1：12 runs、120 run-epochs；串行launcher墙钟7897.697474479675 s≈2.194 h，各run duration之和7828.4727437496185 s≈2.175 h | M4 §54.2的已封存resource-totals.json；两种时长分列，不冒称物理GPU独占占用小时；不是UrbanEV实测 |
| Extrapolated | P2 三臂约 14.49 h（历史规划值） | M4 §47.8以旧AMD control：ETTm1 40.220 min、UrbanEV 249.620 min，各乘三；不是P2实测或保证上界；ETTm1已完成，UrbanEV已完成，实际阶段墙钟另列，不将旧外推当实测（M4 §55） |
| Measured（复用已接受审计） | UrbanEV Stage 2：12 runs、120 run-epochs；launcher墙钟45795.64994764328 s（12.7210138743 h），run时长和45741.647010564804 s | M4 §55所列审计证据根的final/resource-totals.json；仅串行任务墙钟/工作量，不是物理GPU独占小时、正式矩阵均值或并发实测 |
| Scenario | 串行预留约 22 h | M4 §47.8 的约 1.5 倍余量建议，非保证上界、非预算授权或自动强杀时限 |
| Extrapolated | 六折四 horizon、AMD 10 epoch 约 14.6 h | M1 §12/§21 的训练窗口比例 × M4 §47.8 四个 UrbanEV AMD control 的 249.620 min，推导如下 |
| Unknown | 正式各 dataset/model 的完整 run 均值、epoch/早停、资源峰值与调度效率 | 待经审核的效率预检及正式协议冻结；不能用上述开发记录代替 |

源记录的两个已显示秒数小计之和为 36369.523 s，与总计显示值相差 0.001 s；保留原记录并注明显示精度差异，不重算旧指标、不改写历史资源表。

M1 §12 的六折 train split 长度为 576/1171/1747/2342/2937/3475，T=12，H=3/6/9/12。按 W=max(0, split_length−12−H+1)，每折四 horizon 的时间窗口数和为 2230/4610/6914/9294/11674/13826；六折总和 48548，fold6 为 13826。因此比例为 **48548/13826 = 3.5113554173**，乘共同的 275 区后仍相同（13350700/3802150）；不把六折简单乘六。

于是 249.620 min × 3.5113554173 / 60 = **14.6084 h**。这是历史按训练样本量的外推，当时正式epoch政策未锁定；现在§5.5已确认统一训练项，仍不能把14.6 h当新正式任务实测或保证上界。validation比例、准备、完整epoch、final evaluation和保存/校验等不随训练窗口严格线性变化，目标输出减少也不保证按C倍加速。

以下为 **Scenario**：F_PJM=1、主比较每 seed 328 次；S=1 为当前默认，S=3/5 为未授权扩展。假设平均完整 run 为 20/40/60 min，按每日连续运行 24 h 折算串行等效工作量：

| seeds | 20 min/run | 40 min/run | 60 min/run |
|---|---:|---:|---:|
| 1 | 4.6 日 | 9.1 日 | 13.7 日 |
| 3 | 13.7 日 | 27.3 日 | 41.0 日 |
| 5 | 22.8 日 | 45.6 日 | 68.3 日 |

旧579-run情景在40 min/run下约16.1日/seed仅保留为历史假设，不适用于当前495-run/统一训练合同（含Deferred TiDE为520）。上述均值及任何1.4并发折算都不是新矩阵实测工期或保证上界；缺少实际时延的任务保持Unknown。额外工程、profiling、排队和失败成本另列，不自动追加训练/搜索/seed。

### 9.6.4 A800 batch / 并发效率预检（有限对照完成、适用边界与后续规划）

**历史补测准备（M5 §23；已由§24全通过收口）**：44组/155代表继承、10组/40代表待补测，核心串行＋一个可行并发候选至多480 Adam；总硬上限594，额外额度0。四路资源不适合先按原规则两路；若已耗额度不足以完成可选回退并保留后续组核心额度，则只保留已验证单路，不把未测并发写Passed，不以降并发掩盖数值失败。RSS与条件数值规则见唯一M5 §23，原结果不改。新许可必须绑定实际closure/code/protocol/父证据及kernel证据；full probe由用户启动。

**历史数值准入（M5 §19；当前限定规则以§23–24为准）**：仅TimeMixer/Exchange四H具名tokenConv权重/梯度/Adam两动量采用逐元素atol=1e-7、rtol=0，finite强制；loss/归一化validation同界，初始/RNG/batch/步数与其他状态仍exact，其他模型域不放宽。旧exact失败保留，新六步H192串行/并发对照已通过；44组170代表仍待完整补测。机械余额450，3036补测额度未用，准备review/closure后用户启动，不进入M6。规则与完整证据见唯一M5 §19。

**历史收口（M5 §18）**：10组有条件继承核对、44组170代表补测计划保持；新增709已明确批准，2327＋709=3036，机械余额474单列。源码、范围、父证据和额度在输出创建前检查。双时点RSS检查已限定验收，旧AMD/ETTh1代表6步未触发；TimeMixer/Exchange串行重复亦不exact，完整补测仍Blocked，未放宽数值规则/改变确定性。工程commit/push不等于M5 Closed或M6启动。§17以下是此前时点。

**当前补测准备（M5 §17）**：原54组结果及失败成本见本节最新摘要/唯一M5；不重跑54组、不重置3510首验/512机械额度。新来源batch/LR改变的profile不能沿用旧数值轨迹或资源许可，当前精确补测44组/170worker、10组有条件继承；3036 Adam上限比首验余额2327多709，仅列待批，机械余额504单列，不默认挪用。每worker仍6 Adam/8 forward/6 backward/2 validation，固定四H波次及按模型分组不变。只做来源迁移，不搜索T/batch/LR/线程。新许可须绑定实际修后closure、完整配置/代码/来源/环境/硬件、精确followup摘要及父证据；没有review/额度/来源闭环不能启动。旧54组13 Passed不等于正式训练获批。

**首轮流程历史快照（M5 §13，沿§11–12轻量流程，当前补测见§17；取代下文P0–P3未执行长期流程）**：只测实际使用的model×dataset×input_variant组。标准域直接四H一组；UrbanEV各输入方案以四H/fold1代表GPU形状，不更换代表fold，六fold数据/标签与CPU占用证据另核并沿用。由495清单生成54组、Q=195：四标准域32组128worker、UrbanEV F4九组36、A/J F1–F3六组24、PJM七个单fit组7；47组四任务＋7组单任务，仅新增三个AMD组。正式清单与probe均无F0、无PJM TargetOnly。PJM端点与validation尾批99已由M5 §16限定前缀连通核实；仍仅一个fit，不复制四fit。每worker六次Adam（2 warm-up→一完整validation batch及真实余数尾批→4次更新），使用正式结构/batch与合成数据，保留原生辅助loss；数据前缀连通与模型资源负载分开。

先串行各worker建立唯一资源/初始与RNG/batch/短轨迹参照；q>1且资源准入时直接测该q任务组，最多四路。四路资源不准入/OOM或短包无收益才测相同任务的两路分波，仍不成立则只用已通过单路。每组最多一次各候选，不搜索batch/线程、不做完整1/2/4排列研究；同任务组makespan作分母，资源安全与短包收益分开。身份/初始RNG异常停止整个probe；数值/模型失败阻塞配置，不用降低并发掩盖。只有资源不足允许按计划降并发。短轨迹按§19限定例外及其余exact执行，禁止未获批的进一步放宽；六步不保证完整epoch绝无OOM。

首验保守上限18×195=3510 Adam；单任务不重复并发，实际固定流程至多3426 Adam（1170串行＋1128候选四路＋必要回退最多1128），对应4568 forward/3426 backward/1142 validation batch，另512只为明确机械修复/必要诊断上限、每受影响组一次，不借历史余额。每worker固定8 forward/6 backward/2 validation batch；不新增batch/线程/效果搜索。记录allocated/reserved、NVML整卡/进程与驱动保留显存、GPU UUID/NSpid、实际间隔、CPU RSS/配额；预留max(8GiB,显存10%)并扣除外部占用，不杀无关进程。进程归属/显存未知保持Not verified/null；§14在正常权限下以自有/proc sched内核PID及稳定启动时刻实现映射，已用1/2/4存活张量进程实际通过准入。尚未登记CUDA的自有进程不伪填0显存；未知外部PID、坏采样、UUID不符和余量不足继续拒绝，退出另作生命周期处理。此为监测链修复，不是正式组资源/并发许可；必须审核修后路径后才运行完整队列，不扩建PID平台或绕过保护。旧smoke整进程4GiB Not verified不倒改。M5 §16已落实用户明确确认的细分数据准入，54组/195worker的端点与尾批全部可计算；Exchange H336/H720实际validation仅425/41窗，小于B512，其合成完整batch只是保守资源检查，不能声称真实validation存在整批。完整probe仍待修后整体review、实际closure/clean及用户启动，尚无任何正式组并发许可。入口按完整协议/代码/源/环境/硬件及组覆盖核验，拒绝旧423/旧520许可/报告，不以旧数字本身判身份。正式入口另需冻结/M6授权、数据闭环及资源结果审核。当前数据接受范围、验收及完整命令见唯一M5 §16；§14.4保留当时Proposed历史。

当前第三章训练项以§5.5已确认表为准；本节下文历史效率探索与长期候选流程不授权改变固定batch/LR/epoch/停止或新增搜索。既有并发仅在原已验证范围复用，新的任务/形状不可凭1.4假设直接外推放行；没有实际时延的任务保持Unknown。

2026-09-16当前更新：THLS单模块两dataset既有结果保持；§67双数据集N/S/J工程/review/closure完成，N入口等价/初始化/首batch前置通过并复用既有同形状并发证据。S/J缺口兼容672/640/640已通过，随后按用户指定E-S→U-S→E-N→U-N→E-J→U-J六波、全局最多四路完成24run，无效果前置或逐波人工确认。实际墙钟39583.71105360985秒，不由短测或worker时长相除声称实测加速比。§68登记完整性及结果review Passed、原18项总gate Not passed（ETTm1 H192 J/N唯一安全失败），科学序列停止；本轮仅文本分析，不新增效率负载。下文旧AMD-Concat有限对照保留其历史适用范围，不是当前NSJ审批状态。

P2原24-run序列已结束，development adequacy=Not passed，工程implementation/review/closure的Passed/Completed不变（M4 §55）。效率工作的阶段接口保持：**M4做瓶颈核对与有限并发预检；M5确认实际选定模型的执行配置；M6依据批准清单调度，模型独占效率与任务组调度效率分开报告。** 这不是M5/M6开始、正式并发配置冻结或整体阶段重排授权。

成功短测及完成审计已获接受（M4 §56.1）：最终有效轮13子任务、496个Adam步骤、96.63924797601067 s，三次同任务组1路/2路的S₂中位数1.7420710448402421，范围1.6967539824489621–1.84298501373104；6组保存短轨迹逐位一致、max_abs=0。prepare占worker时间约41.7%–63.3%，合计采样CPU RSS峰值约1.897 GiB，后者不是GPU显存。旧序列另有已完成120步、初次失败≤8步，累计≤624步；496不是全部尝试成本，不追溯补造授权。短测不证明完整训练同样加速或必然逐位一致。

M4 §56四配置对照已获用户批准并执行；一次中断后仅重跑获批的最后完整R3/S2T4组，原11组＋唯一重跑组合并为12组48个有效worker、24,960个Adam步骤/192批validation，含被排除中断组的已完成部分后实际26,000步/200批validation，失败证据保留（M4 §57.1）。按每次重复的同任务组配对比值计算，S2T4、S4T4、S4T2相对S1T4的端到端加速median分别为1.3797898085425668、1.6929321420582246、1.6824019535320853；稳定训练段分别为1.2626251681240717、1.4752309538704576、1.4705046802809107。逐重复值、range及排除整个跨时段R3后的描述性敏感性见M4 §57.2–57.3，不用两个median相除替代配对统计。36个并发任务与同重复串行参照的已保存初始/RNG/batch、warm/final参数及Adam状态、loss记录逐位一致；仅限保存范围，不保证完整训练逐位一致。GPU约34%→99%是采样忙碌时间，约3.72 GiB为CPU RSS，约2.88 GiB为设备显存，不解释为理论算力利用率。

当前结论仅为：**UrbanEV h3/F4/fold6 AMD-Concat、batch128及本次环境下优先S4T4；S4T2没有稳定额外收益。** 这是执行配置候选，未注册全局默认，未外推P2/Sonnet、其他horizon/数据集或正式矩阵。效率探索本轮到此停止，不再测试6/8路，不修改runner、旧launcher、batch/workers/AMP/TF32/compile/MPS。后续真实任务按实际负载另行确认；下列分层流程继续作为长期规划，不代表当前启动许可，不按validation改善选择调度。

**P0：固定已有配置，识别瓶颈。** 优先覆盖 UrbanEV 短序列、低维长序列、高维长序列代表任务；只使用已实现且可用的模型/管线，其余标 blocked。分解 data 等待、H2D、forward/backward、optimizer、validation、checkpoint 和 checksum 耗时；CUDA 计时明确 warm-up、正确同步及多次测量，分别记录冷启动与稳定段。测量须覆盖完整训练状态建立后（含优化器懒创建状态）的 allocated/reserved、进程显存及峰值；单批 probe 不冒充完整 epoch 峰值。同步记录 CPU 线程、workers、RAM、存储和其他 GPU 进程竞争；不以主机总核心数/内存替代实际配额。

**P1：batch 与数据管线候选。** UrbanEV 训练 batch 候选为 128/256/512/1024；2048 只在前序有收益且余量允许时评估。其他任务从已验证配置逐级探测，不统一强制 128。evaluation batch 可独立研究，但保持 eval 语义、全样本覆盖、全元素 SSE/SAE 聚合和预定数值容差。workers、pin_memory、传输/切片优化逐项测，不默认 workers 越多越快；workers/预取等变更必须另验数据顺序、RNG 和 resume 合同。

batch 变化属于训练协议变化，必须同时记录每 epoch/总 optimizer steps、已见样本数、drop_last、LR、epoch/停止政策、BatchNorm 及相关损失行为。不盲目线性放大学习率，不将梯度累积称为完全等价的物理大 batch；吞吐预检不能替代 train/validation 效果检查。同一消融组训练政策一致；跨完整 baseline 可采用经验证的各自配置，但搜索机会和预算须公平。正式 test 不参与效率参数或模型选择。

**P2：多 run 并发预检。** 先比较 1 路与 2 路，仅当同一组任务的总完成时间确有改善后才考虑 4 路。每 run 独立进程、seed/RNG、artifact、日志、锁和 resume 身份，同时控制 CPU 线程、workers、RAM 和存储带宽。比较 makespan，记录 S_k=T_serial/T_concurrent；不以并发数冒充加速比，也不只看 GPU 利用率/显存。模型效率表使用独占、无竞争计时，调度效率另表报告。

MPS 仅列可选项，须核验权限、对共享 GPU 的影响及恢复策略；不得自行启动 MPS、修改 compute mode/MIG 或系统配置。未来若另获多 seed 扩展授权，优先作为独立任务调度，不自动变为预测集成；当前默认仅 seed=2024。

**P3：正式执行配置冻结。** 按 dataset/model/任务形状登记 train/eval batch、precision、workers/threads、并发度、训练预算、初始化/seed、实际吞吐/显存和验证选择依据；经审定后才生成完整运行清单及成本范围。AMP/TF32/compile 和环境升级各为独立候选优化，不与首轮 batch/并发优化同时默认开启。

未来长时任务仍须完成获准实现、review/closure 和独立训练前准备；守护方式实测后按 tmux > systemd-run --user > nohup+setsid 选择，提供完整启动、日志、进程/状态、完成判断和安全停止命令，默认由用户启动。失败/中断先审计 resume，不删除证据、不自动重跑。本节规划不等于执行上述流程的授权。

# 10. 第四章：每个数据域必须独立获得时间基线

空间实施准入顺序（用户本轮确认，仅文档生效，未启动空间阶段）：

1. 当前及未来候选backbone先在原版作者模型上复现指定论文任务，通过后再按既定计划直接使用或改造。选论文中1–2个可取得数据集，使用与原结果可比的任务、处理、指标和训练协议；不用第三章统一训练表去要求原论文数值复现。
2. HSTGCN已知有Boulder公开实验；3h/6h是同一个数据集的两个任务。北京数据不可得时如实记录，不强凑两个数据集，不声称其所有数据均私有。原版复现与后续EL-AMD+HSTGCN-core项目实验分开，不用改造core冒充原版；改造后只做必要工程验收与计划内正式效果/消融，不另设core必须复现原论文指标的关卡。
3. SADR/SC-SimGCA等模块不要求完整复现来源模型整篇论文指标；保留来源核对、既定修改工程验收和本项目消融。HSTGCN/ASTGRN/G-STAN三个已批期刊例外按§1.2保留，不改变模块数学。
4. 复现预算、具体对照数值与容差在相应空间阶段执行前锁定；有限排障后仍无法复现，先记录原因和成本，再提出替换候选，不无限改epoch/seed/结构救援。新选择的替代backbone/模块须2023年及以后顶会、优先作者开源；无代码按更高材料风险审查。替代backbone仍先原版指标复现，替代模块不增加整篇指标复现要求。
5. 结论限定为“指定任务上复现通过/未通过/条件不足”；通过不是整篇论文无造假的证明，未复现也不直接推断作者造假。本轮不clone空间仓库、不准备数据/环境、不跑模型/toy graph/复现，不创建M7/M8等milestone。

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
- 不输入未来真实天气、负荷、汇率或其他未来观测；
- 主预测loss、validation best及最终指标只覆盖指定目标，标准数据覆盖完整H步；多输出内部结构不能引入其他未来标签监督；
- 主表同一dataset/target/horizon/fold按§5.5使用相同lookback、输入变量/ordered aux、划分、scaler、seed、mask、目标反归一化和全元素聚合；标准日期列仅作索引，不给个别baseline额外time-mark；
- 模型原生结构项可不同并记录来源；第三章训练项严格沿§5.5统一表、单配置和0额外搜索，不因模型不同另给LR/batch/epoch或搜索机会；其他阶段仍依各自合同留档，不由本次修改改变第四章路线；
- 第三/四章正式比较及消融固定 formal seed list=[2024]，不自动补 seed、不 seed 择优，模块已登记的内部初始化子流不变；
- 主表报告固定 seed 2024 的实际结果，seed std=N/A，不报伪造的均值±0；fold/horizon/city差异不是seed方差，不宣称随机初始化稳定性已验证；
- 既有 development artifact 不得升级为正式主表；未来按冻结正式协议新运行的 single-seed 结果可进入正式表，原 M5 三 seed 一致性暂缓且不标 Passed；
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
| M4 | 时间模块诊断与候选迭代（Closed） | 已封存；原18项总效果gate Not passed，H192失败及风险接受保留，不追加M4任务 |
| M5 | 模型筛选与结构冻结（Closed） | 正式协议与配置、必要工程接入/验收准备后，依据既有开发证据及风险接受提请用户明确结构冻结；不默认新增完整多域效果筛选或practical-effect前置，不提前冻结J |
| M6 | 第三章正式实验与定稿（In Progress；用户分模型启动） | 明确冻结后按统一训练协议执行六域主表、UrbanEV F4模块消融与A/J的F1–F4同输入消融、正式test和效率报告；seed2024/std=N/A，负向结果照实报告 |
| M7 | 时间状态接口与 Graph Mode | 训练 StateAdapter；`H_time [B,N,d]`、target-only output、适配后一致性测试通过 |
| M8 | HSTGCN-core 与双图构建 | 图归一化、官方地理图、train-only DTW、S0-S3 与图测试通过 |
| M9 | SADR 状态需求残差图 | S4、blockwise top-k、关系可视化 |
| M10 | SC-SimGCA 状态条件图传播 | S5，纯空间 residual 测试通过 |
| M11 | 第四章正式实验与定稿（未开始） | 全部UrbanEV节点、CHARGED六城市、PEMS04/08全部节点/12步；固定seed=2024、std=N/A，S6、主表与定稿 |
| M12 | 论文正文、图表与结果分析 | 全文叙事、图表、结果分析与章节一致性完成 |
| M13 | 终稿审校、复现材料与答辩 | 终稿、复现清单、答辩材料与最终校验完成 |

本次仅按用户确认更新M4封存状态及M5/M6职责，M0–M3和第四章路线不改。M5协议确认不等于正式训练或最终variant冻结，必要工程与具体负载仍按对应授权执行。效率规划保持M5确认执行配置、M6按批准清单调度及单独报告。

# 20. 结果表与可视化

## 20.1 第三章

表 A：UrbanEV / EPF-PJM 的 MS 单目标外生协议（固定 seed 2024）

```text
Model | UrbanEV h3/h6/h9/h12/Avg | EPF-PJM MSE/MAE | Avg Rank
```

表 B：ETTh1 / Weather / ECL / Exchange 的统一MS任务（字段目标已User confirmed；元数据/接口缺口按§9.1保留；固定seed 2024，替代旧M-to-M规划）

```text
Model | ETTh1 | Weather | ECL | Exchange | Avg Rank
```

两表均绑定明确目标与metric scope；标准长序列报告完整H区间，UrbanEV保持偏移单点，不能混入旧M-to-M或development结果。报告固定seed实际值、seed std=N/A；fold/horizon macro不是seed均值，不以其波动表示初始化稳定性。不同数据集指标不直接求数值平均，只报告平均排名。

可视化：

- 历史 TimeXer TEB 的 attention/residual 只作为失败路线诊断；
- 历史 Early/Late CCE 的等价 CrossLinear kernel、effective lambda 与 ungated delta 分布只作为负向 development 诊断；
- PMCR 在峰值和突变窗口的残差响应；
- AMD 与 EL-AMD 峰值预测案例；
- 不把 attention 权重表述为因果贡献。

## 20.2 第四章

正式比较与消融同样固定 seed=2024、seed std=N/A；保留全部节点/城市/预测步，单目标是每节点的目标特征。城市和horizon差异不作seed方差。

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

临时降级可以用于定位问题。两个经过修改且通过消融的时间模块、以及既有空间模块路线，是用户当前的创新组织目标；“两个近期论文来源模块”是用户选择的研究组织策略，不是已核实的学校/导师硬性要求。保留真实来源、历史证据与现有模型，不据此强行凑模块数量、改写来源或自动调整M4/M5/M6任务表；新支路的项目修改与底层卷积来源须分开表述。

| 模块 | M4 诊断与候选边界 | 保留来源的备选实现 | 最终仍失败时 |
|---|---|---|---|
| TimeXer-inspired TEB（失败历史路线） | Global/T2/T2G/T3/rescue 的 artifact、实现与诊断保持可追溯；第十七轮已触发停止线 | 不再继续 T4/T5/T6、patch/gate/beta 调参或新 TimeXer-derived TEB | 后续 CrossLinear 路线也已失败；当前下一来源候选已转为 Sonnet/MVCA |
| CrossLinear-inspired CCE（失败历史路线） | Early（RevIN 后、MDM 前）与 Late（post-PMCR/pre-AMS）实现、artifact 和诊断保持可追溯；两者均未通过 M4 adequacy gate | 不再调 kernel/lambda/gate 或增加插入位置 | 当前路线已触发有限开发停止线；当前下一来源候选已转为 Sonnet/MVCA |
| Sonnet/MVCA（当前 M4 leading development candidate） | joint embedding + learnable wavelet + paper-defined MVCA + no-Koopman reconstruction + target residual | RevIN 后/MDM 前；target_exogenous only；matched from-scratch；固定 d/K/alpha/gamma 与双数据 development gate | implementation/review/closure 完成；16-run paired development 为 positive signal 且 S2 adequacy Passed；证据仅覆盖单 seed 固定 S2，仍不得称为最终外生模块或最终 EL-AMD，XLinear 不同时启动 |
| PMCR | M4 只诊断 kernel、hidden、作用范围与 residual；任何参数或结构候选须经用户确认 | 可评估保留 Reparam DWConv + ConvFFN1 来源边界的候选；不得预先选定 | 更换另一篇近三年局部时间模块 |
| SADR | b_lambda 更负；k/d_a；正则 | ASTGRN global adaptive graph 与 DTW 的残差融合 | 更换另一篇近三年空间图模块；退回静态双图只算排障结果 |
| SC-SimGCA | rho 初始化；层数；SimAM lambda | 保留 G-STAN 层融合，移除 Graph-SimAM，改名 SC-GCF | 若仍失败，更换另一篇近三年空间传播模块 |

本次确认以§9.5的新冻结前置替代旧“单模块正式通过线由M5执行”：不再要求冻结前新增EV＋外部域单模块效果矩阵或锁定新practical-effect，不把旧0.5%安全线、多域改善/组合稳定不退化要求记为已满足。M5以正式协议、必要工程准备、既有开发证据与明确风险接受提请冻结；两模块有效性、组合交互与代价在M6的UrbanEV完整消融和六域整体模型比较中如实评价。负向结果须保留并收缩主张，不为凑数量新增候选、不按dataset/H关模块冒充同一模型；改变结构/任务须另行决定，不能看test后救援。

formal seed list=[2024]、std=N/A，原三seed方向一致性暂缓且Not evaluated，不记Passed；两个来源模块“经消融有效”是待证目标，结构冻结不是其证明。M4所有来源、停止线、原gate Not passed、H192及1%阈值原样保留，失败历史路线和第四章来源合同不改。

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
  + Sonnet-inspired S2 MVCA target-residual development 候选（development adequacy gate 已 Passed，仅为 M4 leading development candidate）
（TimeXer-inspired TEB 与 CrossLinear-inspired CCE 均作为失败历史证据保留；最终内部结构仍只由 M5 冻结）
数据：UrbanEV + EPF-PJM + ETTh1 + Weather + ECL + Exchange
任务：统一多变量历史输入→指定单目标MS精确合同已User confirmed；未决正式元数据/接口仍blocked
当前：MS文档、A/B接口及P2生产实现review/Git closure均已完成；C production implementation complete、engineering gate=Passed，有限工程范围及skip见M4 §51–52。原24-run完成：ETTm1两个安全gate Passed；UrbanEV完整性Passed、两个主效果gate Not passed；P2 development adequacy=Not passed，序列停止（M4 §55）。有限1/2/4路及4路×2线程预检已完成并获接受，S4T4仅为UrbanEV h3/AMD-Concat负载优先候选，效率探索暂停扩展；当前已完成24-run曲线及16份C权重CPU只读统计，原有限validation激活/旁路诊断已获用户批准并完成：8份权重、504次batch前向、0优化步骤，有限检查通过且结果已获ChatGPT接受（M4 §58）；当前THLS §59结构/身份、工程验收及A/N八个新run/80 run-epochs、独立160步兼容提案已获用户确认；模块、入口/汇总和18项永久测试已写入；reconstructed工具及当前精确限制已启用，4项当前批准不冒充历史恢复。§60.7历史新增验收7 passed、1 failure、10 unexecuted，停止于CPU T512关闭A/冻结AMD输入梯度逐位相等断言；继承314项及真实probe未执行，THLS engineering gate=Not passed。§60.8历史诊断及第二组缺失数值原样保留。§60.9累计102/24/0保留；本轮用户批准的共同DDI CPU GELU兼容patch已实现，身份ddi_cpu_gelu_cotangent_contiguous_v1，局部布局回归Passed，M0/tag不变、前向及CUDA路径不变。24-file source更新为7bade9b11541eb5d5c53c95910276af45282dfd60da2d4320944312155b7c3c9，后续训练仍须绑定修后source和实际commit，旧checkpoint不获跨source恢复豁免。§60.10的333 ID为1 passed、1 failure、331 unexecuted，保留历史。§§60.11–60.12固定参考超限、修订后的weight前置通过及CUDA bias旧逐位失败保留历史。§60.13仿射组参考和关闭等价结果已获接受，11 passed/1 error/321 unexecuted与累计253/70/0保留历史。§60.14已完成单行导入修复，未获预算时仅做静态检查。最新§60.20用户明确批准新增合成累计上限824/193/49，携入621/153/33，临时fixture修复后的唯一封存验收序列已完整通过：GELU1＋THLS18全部Passed；继承314为293 passed＋21原精确策略skip。共333 ID=312 passed＋21 skipped，failure/error/blocked/unexecuted均0，不能称333项全部通过。CPU/A800 CUDA实际覆盖原必需case；原36 forward/4 backward/0 Adam真实UrbanEV单批probe完成，四horizon A/N的train/validation前缀、CPU/CUDA batch2及h3/h12 CUDA batch128检查通过，test读取越界/解析/构造/迭代/评价哨兵均0。全文件读取仅为既定字节指纹，实际观测解析限原3909时间点前缀；无历史checkpoint读取、完整评价或development产物发布。三个业务阶段非预期受禁访问均0。新增本轮179/40/16、累计800/193/49，剩24/0/0；继承992/298/148及真实36/4/0另计，历史失败成本保留。未新增生产/测试repair；旧CCE可移植性断言在原字节下Passed。仅原指定float32 RevIN仿射参数组保留对称1e-7/1e-6数值规则，其他严格断言不变；报告整理的双方None字段错误从完整原记录修正，未重算模型。THLS implementation complete、engineering/implementation review Passed，统一代码closure=f13dd26dfa72e140c8e4c05b146ea39d6643f889；UrbanEV已提交文档/训练来源为e93f008f9f82546e13bf4c7f2522b1728d81cbdf，24-file source=16343e9298a9c4a1eb477c061f7b58552ab04b0706f751b8b83e42749b703a7f。§61独立N160步兼容和训练前方案已获审；随后按用户明确授权执行A四H→N四H两波，8/8、80 run-epochs、594040有效Adam步骤、墙钟22573.999078273773秒。最新ChatGPT接受完整性Passed、UrbanEV五项gate Passed及严格限本合同的positive development signal：N/A主MSE macro−0.873577%、MAE−0.715574%，四H及全部leave-one-out改善；旧中断A-h3成本与审计脚本修正保留（M4 §62.1–62.4）。ETTm1最小接入incremental engineering及ChatGPT implementation review均Passed，统一closure/训练HEAD=667a1b82bdf799a10a6ce7d813b424c0f223a5cf，24-file source=4d707b1d878d3490a55243e060a3d43258c60a70414edf23d2ff3fc4acedbc9a。已批336/320/320兼容检查完成后采用A四H→N四H两波；8/8、80 run-epochs、84260 Adam步骤、实际墙钟3142.443908929825秒。ChatGPT本轮接受完整性与§62.7四项安全线全部Passed：主validation MSE macro +0.112646595%、development-test MSE −0.032258338%、MAE −0.160544776%，最差H720 test MSE +0.466123118%；两种macro和逐H原值见M4 §64.2，不能称所有H均提升。结合UrbanEV五项gate Passed，THLS单模块本轮开发验证收口，ETTm1仍是development-only test；旧source和审计Pending快照不追写。§66–67双数据集N/S/J比较已完成统一接入、工程/review/closure及兼容，训练来源为4abb099d69db04532f066986fde0c43aedf2465a、25-file source=6a11d88f6d22861e49db0122bf685b642b197af2796ab8de4b7a424b4548a97a。24run按E-S→U-S→E-N→U-N→E-J→U-J六波完成，240epochs/1017450 Adam，墙钟10小时59分43.71秒。§68本轮结果review Passed；UrbanEV J/S与J/N各5/5、ETTm1 J/S 4/4，J/N因H192 test MSE +1.542951806%超过1%而3/4；原18项总gate Not passed、科学序列停止。ETTm1两组validation MSE macro安全项分别−0.049560234%和−0.133132443%，均Passed，不因摘要只列test而省略。有限十二run曲线未发现具体训练/选择异常，不自动重训、改模型或准入。以下§65失败/修复为已被§67续验取代的历史记录，不是当前工程状态：用户当时明确确认§64唯一结构/独立身份、8 fresh runs/80 run-epochs、五项效果gate及三笔独立额度；§65四文件组合接入和八项新增验收已执行至6 passed、1 summary负例failure、1 unexecuted。source负例参数传递错误已最小修复并静态检查；累计123/45/24、剩37/19/8，不足以按原要求在修后版本完整复验，当时engineering Not passed/未完成、implementation review/closure Pending，真实20/8/0未执行。当时25-file source=75458a103145f60b98314bd206330e45442ce65043c3161362208cecb10207e8；原工具/skip/模型数学及数值边界不改，当轮320步兼容和八run未启动。当前已完成新24-run且总gate Not passed；S2 Passed/leading、THLS单模块已接受结果、P2原Not passed、M4 In Progress保持，正式结构/variant未冻结，不启动M5/M6/M7。

第四章：EL-AMD + HSTGCN-core + ASTGRN-inspired SADR + G-STAN-inspired SC-SimGCA
数据：UrbanEV + CHARGED 六城市 + PEMS04 + PEMS08

所有正式比较/消融当前固定seed=2024（std=N/A），不追加或择优；
所有数据域独立训练；时空模型使用全部节点的完整图窗口；空间模块只输出 residual；
产物按 variant/dataset/task_mode/target/horizon/fold/seed/run_id 隔离。
```

## 2026-10-01 用户新增：独立七baseline M实验（待审分支，未合入生产）

本次用户决定新增独立`m-baselines-v1`批次，父生产commit为`5341fbcb7c9f4f97658728d79b1af5487f7d38c3`。本节只存在于独立分支`m6/m-baselines-v1`及worktree `AMD-m-baselines-v1`的待审字节，未合入生产分支；生产MS队列、原552目标与历史执行账保持。ECL仅在本次M批次暂缓，不取消原ECL任务或删除结果。

模型顺序为AMD→DLinear→PatchTST→iTransformer→TimeMixer→ModernTCN→TimeXer；每模型ETTh1→Weather→Exchange，每域H96/192/336/720、单fit/seed2024，共84个fresh run、最多840 run-epochs。无J/N/S、UrbanEV、PJM、NP/BE/FR/DE、ECL，也不新增Informer/Autoformer/TiDE、J-M-anchor或J-M-parallel。M与MS分别标识，不去重、不合并排名。

每个M profile从该父版本实际resolved MS profile派生，T、batch/eval_batch、固定LR、epoch、optimizer及其他训练/结构参数逐项保持。TimeMixer三域LR为0.001/0.001/0.0003；无scheduler/warmup。ETTh1 T512/C7/端点8640,11520,14400；Weather T512/C21/36887,42157,52696；Exchange T96/C8/5311,6071,7588。使用已接受前缀元数据和train-only scaler来源，不重新下载/解析真实观测；Weather重复记录政策不变。这是现有配置下的M对照，不是论文最佳M recipe或超参数搜索。

输入[B,T,C]，标签/输出[B,H,C]，全部C通道参与训练及validation。MSE/MAE在train-standardized空间累计全样本/H/C误差和元素数后归一；按全变量validation MSE选择best，相等保留更早epoch，结束后仅best一次test。同次test遍历记录逐变量与旧MS目标诊断：ETTh1 OT/index6、Weather T (degC)/index1、Exchange OT/index7；诊断不用于另选checkpoint。保留原始列顺序和逐通道归一化，不提供未来真实输入。

AMD使用既有MDM→DDI→AMS全通道路径，关闭全部增强、保留aux定义/系数；其余模型使用自己的锁定作者全通道输出，不复制目标充当C通道。TimeXer使用commit `76011909357972bd55a27adba2e1be994d81b327`、TimeXer.py SHA `b334d7869544d0a5de7a35501d0342d7bd0be4ebda06ccc045855a42819728c0`的原生features=M/forecast_multi、n_vars=C及C个global tokens，保留全输入cross embedding，不使用MS目标末列重排或自制parallel结构。

结果根为原E下`m-tasks/m-baselines-v1/formal-<MODEL>/<MODEL>-<DATASET>-M-f1-h<H>-s2024/`；总控制器位于其`queues/m-baselines-v1`。一个tmux内同步等待每模型真实退出与技术交接，再启动下一模型；STOP/技术失败停止后续，不自动fresh重试、不依效果跳过模型。组内q须来自新M准入，不能继承MS Passed。重复启动及已有失败/staging/输出冲突拒绝。

新M probe真实生产路径为preflight→精确scope/config→已有受限worker/监测→6步完整状态/梯度/Adam比较→来源绑定报告。21组，每组四H串行参考再q4，名义168 worker、1008 Adam/1344前向/1008反向。仅明确resource失败允许一次q2两波，最坏额外84 worker、504/672/504；总GPU技术计划上限1512/2016/1512，独立待审核，不挪旧余额。q2资源失败仅可使用同条件已通过的串行q1；数值/身份/guard/业务/未知失败立即停止，无机械GPU重试。21份数值政策逐模型/域注册为Proposed，按同scope父依据限定拟用规则，不全局扩大旧容差，不能将MS规则或证据当作M通过。

隔离CPU准备已完成54次合成完整前向（上限64），Adam/反向/GPU初始化均0；七模型各独立作者进程，AMD和TimeXer覆盖所有H，其他模型H96/720。无模型验收及专用合成tmux生命周期证据、失败与局部复验账在现有P的唯一`m-baselines-v1`准备包内。没有真实数据/test解析、正式权重加载或新正式run；不能从CPU输出检查推定M资源/并发Passed。

GPU执行须经本轮实际字节审核、独立分支clean closure、排他许可绑定及旧总队列TimeXer/N/S全部结束的技术边界核验。旧结果科学审核仍待审时如实保留，M批次不豁免旧统一完成审计。当前不后台等待、不GPU执行、不部署、不stage/commit/push。正式长训练最终由用户一次启动，原作者源码、环境、AGENTS、Closed M0–M5、baseline tag均保持。

汇总按各域四H主表/宏平均、逐通道及旧MS目标诊断分别输出；不跨域平均绝对误差，不把M/MS混排，没有J时不声称验证J竞争力。父72份MS runtime是估时参照，TimeXer当前运行中的结果不读；M总工期及最终q在M准入后修正，六步包不整体除6作整批实测。完成后按精确84任务/168 best-last权重CPU检查、history/budget/best/final-test证据和源数据状态统一审计，不重评test；technical-complete与result-review-pending分开。

本次审核repair仅修launcher身份、报告逐H参数表和新增handoff调度；84份科学profile、数值规则及独立预算保持。wrapper以noclobber创建日志，绑定本次scope、许可SHA、log inode/owner及一次性launch token；真正start只豁免已核验的本次日志/本次tmux pane，旧日志、错token、串scope、旧controller/result/session及重复claim仍拒绝。直接裸Python start没有对应launch身份时拒绝。

新增start_m_handoff.sh仅Prepared、未启动。后续审核及独立closure后，用户可一次启动独立handoff tmux：WAIT_OLD_QUEUE→OLD_BOUNDARY_SEALED→WAIT_PROBE_APPROVAL→PROBE_RUNNING→WAIT_FORMAL_REVIEW→FORMAL_RUNNING→COMPLETE。旧全队运行时仅60秒控制器身份/完成/failure/STOP轮询；异常消失或身份/完成范围不符即停。边界排他seal且科学review仍pending；等待实际probe-review后同步执行新M probe，等待ChatGPT真实probe审核和formal-review后才同步84-run。不自行签发许可、不改reviewed、不跳过probe审核；等待期间不占GPU。STOP只通知本handoff当前自有child的PID/start_ticks，不操作旧MS/其他进程。本轮无等待器、模型或GPU执行，修后字节仍待审核。


### M批次后续执行政策：预授权自动技术准入（2026-10-01）

用户最新决定只改变本批执行准入方式，取代上述等待probe-review/formal-review的handoff运行快照；旧段落保留其当时事实。未来用户一次启动handoff：WAIT_OLD_QUEUE→OLD_BOUNDARY_SEALED→PROBE_RUNNING→AUTO_PROBE_AUDIT→FORMAL_RUNNING→COMPLETE。旧生产启动锚点、PID/start_ticks、failure/STOP和完整旧队列技术边界保持。仅在本次后续实现实际字节获审、独立分支clean closure、固定计划/来源/环境/硬件及资源条件匹配后，程序排他生成仅限M probe的预授权许可。

probe结束后从实际21组、串行/并行状态与监测、原预登记数值规则、finite/初始化/RNG/batch、资源/RSS/收益、全部SHA和实际预扣/消耗账重新复核。全部gate自动Passed，才可基于实际probe complete SHA、21项decision及最终q，排他生成formal execution permit并在同一handoff同步继续84-run；正常成功路径无需用户、ChatGPT或Codex再次运行时签发许可。自动许可明确review_mode=preauthorized_machine_gate、manual_review=false、reviewed=false，不冒称ChatGPT人工审核probe或预测效果Passed。原probe报告及科学result_review保持pending。

任一工程gate失败、STOP、未知失败或需要改变合同/预算/容差/参数时停止人工裁决；不自动retry、删证据、搜索或择优。仅明确resource失败可沿原q4→q2→合法串行q1规则处理，预算不退款。84任务/840 run-epochs/294790更新、7模型与三域四H、seed2024、全部profile/模型数学/M监督与指标/test合同、ECL仅本批暂缓、独立probe上限1512/2016/1512及原numeric policies均不变。正式组同步等待真实退出及技术交接后进入下一模型；全队COMPLETE只表示technical complete，result_review=pending，用户随后报告“训练好了”再统一结果审计。

STOP只通知本handoff拥有的probe/formal PID/start_ticks链；旧队列等待期只停止等待器，自动审计期不持有GPU且STOP禁止formal签发/派发。该政策仅在独立W分支实现，未合入生产R；本轮不GPU、不probe、不训练、不启动handoff、不stage/commit/push。新字节仍待ChatGPT审核。


### 用户最新执行决定：native_time_mark_v1 与统一 successor chain（2026-10-01，隔离分支待审）

本节取代此前“旧 MS 直接接续 M handoff”的运行快照，保留此前记录的历史事实。原仅等待的 M supervisor 已按用户授权正常 safe-stop；其日志、STOP/failure 与执行目录保留，此次退休为协议替代而非科学失败。生产分支和旧 MS 总队列不改动。

只有 iTransformer、TimeMixer、TimeXer 恢复作者原生 historical x_mark_enc 接口。mark 使用已绑定 timestamp 的同一历史输入窗口，hourly timeF 的 HourOfDay/DayOfWeek/DayOfMonth/DayOfYear；不使用未来业务变量/target，不增加业务通道 C，不对 mark 拟合 scaler。UrbanEV F4 的 11 个业务输入（含五个既有日历列）全部保留。其余模型调用不变，TimeMixer 的 use_future_temporal_feature 不变且为 0，不新增 x_mark_dec。

三个模型的 UrbanEV F4 六 fold×四 H 加 PJM/NP/BE/FR/DE 各一项，共 87 fresh replacement：最多 1020 run-epochs、3,687,530 optimizer updates。新 revision/run ID 与结果根独立，旧 common-input/x_mark=None artifact 保留。新协议完整执行后这 87 格固定采用 native-time-mark-v1，不依效果择优。旧 ETTh1/Weather/ECL/Exchange-MS 不新增 replacement；通用 benchmark 后续比较采用 M。

M 仍为七 baseline×ETTh1/Weather/Exchange×四 H＝84 fresh runs、840 run-epochs、294,790 updates。仅上述三模型增加 historical x_mark；所有 frozen profile 的 freq 保持原值（包括 Weather/Exchange 当前 h），结构、T/H、batch、LR、seed、监督、metric/test 合同不变，ECL 本批仍暂缓。M protocol 重新绑定，旧许可不复用。

新独立 scope 为 m6-native-tmark-chain-v1。未来用户一次启动：old MS 全部 technical complete → 自动 seal → native mark probe/落盘技术审核 → 87 replacement → replacement technical boundary → 新 21-group M probe/落盘技术审核 → 84 M formal → COMPLETE。M 同时绑定 old MS 与 87 replacement 两条边界。机器许可 manual_review=false、reviewed=false、review_mode=preauthorized_machine_gate，正常成功路径无需中途人工许可；任一 gate 失败则停止、保留成本/现场，不改容差/参数/科学合同，不自动 retry。COMPLETE 仅为技术完成，scientific result_review=pending，科学结果审计在用户最终回复“训练好了”后进行。

原 M probe 上限 1512 Adam/2016 forward/1512 backward 不变且尚未执行。native mark probe 单列待审计划：16 个计算身份、7 个调度组、28 worker 名义；仅 resource q4 失败允许一次 q2，两档资源失败且串行有效才能采用 q1。TimeMixer/UrbanEV 保留已批准两点评价和阈值，单 worker 为 6/10/6；其余为 6/8/6。最大 40 worker、240 Adam/392 forward/240 backward。本轮不执行 GPU，也不启动新 handoff；实际字节审核、独立分支 closure 与匹配新许可是未来执行前条件。新代码/文档尚未合并生产 R。


### 用户最新决定：N/S 退休与 native successor v2（2026-10-02，工作区待审）

本节替代前节要求原552全部完成后才接续的运行快照；此前方案及失败记录保留。用户授权立即退休剩余N与全部S，两次既有safe-stop已停止旧MS队列自有N执行链及仍处WAIT_OLD_MS的v1等待器。生产R源码未改；停止产生的STOP/failure、已完成结果、partial checkpoint/history/staging/budget均保留，预算不退款，此次为用户协议退休，不是模型效果失败。

旧552实际完成520项；N完成16项、4项partial、4项未启动，S24项全部未执行，合计32项退休。边界kind=old_MS_user_authorized_retirement_boundary_v1，绑定原552精确集合、520完成与32退休集合、partial状态、用户退休回执完整SHA、原controller/PID/start_ticks退出、STOP/failure及原生产source/data身份；original batch technical_complete=false、retirement_accepted=true、scientific_failure=false、result_review=pending。不得写成552/552 Passed。N持久化消耗575959 Adam/646897 forward/575959 backward、172条已完成epoch记录；partial epoch未强行取整，历史已删除TimeMixer尝试成本不重建也不退款。

新scope固定m6-native-tmark-chain-v2，独立准备根native-time-mark-chain-v2、session ch3-native-time-mark-chain-v2，结果执行根revisions/native-time-mark-v2与m-tasks/m-baselines-native-time-mark-v2。输入科学协议及87项revision/run身份仍native-time-mark-v1；执行namespace v2不改变profile。v1 controller/STOP/failure/launcher均保留且不能授权v2。

新状态顺序为WAIT_OLD_MS_RETIREMENT → OLD_MS_RETIREMENT_BOUNDARY_SEALED → TMARK_PROBE_RUNNING → TMARK_AUTO_AUDIT → TMARK_FORMAL_RUNNING → TMARK_BOUNDARY_SEALED → M_PROBE_RUNNING → M_AUTO_PROBE_AUDIT → M_FORMAL_RUNNING → COMPLETE。启动时直接验证closure-bound精确退休回执，不再等待旧552 complete。M仍同时绑定退休边界与87 replacement技术边界。机器许可仍reviewed=false、manual_review=false、review_mode=preauthorized_machine_gate；总COMPLETE只表示后续技术链完成，保留original_old_batch_technical_complete=false和科学result_review=pending。

replacement增量probe只覆盖实际计划并发：三模型UrbanEV各fold1四H q4；iTransformer与TimeMixer分别四个兼容EPF市场q4，剩余第五市场固定单路；TimeXer的PJM/BE/FR batch16三任务使用q4容量（实际宽度3），NP/DE batch4两任务直接q2。正式任务按相同兼容组合派发，batch16与batch4不合并。UrbanEV其他fold及EPF同构singleton继承仅限已登记计算身份；市场/label/fold/data来源继续逐任务绑定。

真实计算身份由旧16收敛为7种、实际并发组合7个（不是沿用旧7组单路EPF方案）。为保留每个并发成员自己的串行数值与同任务耗时基线，需25个代表任务、50个名义worker，300 Adam/416 forward/300 backward；六个q4容量组最多各一次q2 resource复验，共最多73个worker、438/608/438。初始两任务q2组资源失败时仅可采用已测合法q1，不重复同一并发。旧16/7/240-392-240方案保留为历史；由于现在覆盖此前未测的跨市场并发，此待审上限高于旧方案，不能称为计算额度减少、挪用旧probe余额或自动启用。numeric/finite/identity/data/guard/未知失败立即停止，不作为resource降级。

87 fresh replacement（1020 run-epochs、3687530 updates）与84 fresh M（840 run-epochs、294790 updates）全部科学profile、数据、seed、模型数学、time-mark、M监督、metric/test及数值阈值不变，不加入N/S；M21组probe及1512/2016/1512独立上限不变。已完成N/J等旧来源保留，论文是否使用N/S留给最终审计。本轮只做退休、隔离实现与无GPU验收，不启动v2/probe/formal，不stage/commit/push；新字节待ChatGPT审核，尚未合入生产canonical。

### 用户最新决定：replacement v3 统一有限绝对阈值（2026-10-02，候选待审）

v2 实际在 TMARK_PROBE_RUNNING 因 TimeMixer-EPF-combination-0 的 serial/parallel 数值差异停止；iTransformer-UrbanEV-native、iTransformer-EPF-combination-0、TimeMixer-UrbanEV-native 三组已通过原 gate。失败组 worker exit0、finite=true、resource admission=true，不能据资源通过替代数值通过。原 v2 failure、permit、trajectory、budget 与所有现场永久保留，不能改为 Passed、覆盖或用于授权 v3。v2 已消耗 192 Adam / 272 forward / 192 backward，budget_refund=false。

已有落盘合成状态的只读量化：iTransformer 两组 state_max_abs=0；TimeMixer UrbanEV 最大 2.60770320892334e-6，TimeMixer EPF 最大 2.2735408720109263e-6；EPF loss_max_abs=1.1920928955078125e-7，归一 validation metric 最大 3.409995175118752e-9。这是已观察的 GPU 并发浮点扰动，不能据此断言唯一算子根因或正式效果。用户批准针对 replacement 的窄绝对上界，不是作者论文标准，不按结果效果择优。

v3 全部 18 个 replacement model/domain（iTransformer、TimeMixer、TimeXer × UrbanEV/PJM/NP/BE/FR/DE）统一 kind=full_float_state，state_atol=1e-5、loss_atol=1e-6、metric_atol=1e-6、loss_rtol=rtol=0、equal_nan=false。完整浮点参数、buffer、梯度、Adam exp_avg/exp_avg_sq 使用该界；初始化/RNG/合成 batch/order、step、非浮点/非张量 optimizer/model 状态、param groups、shape/dtype 与任务/profile/source/data 身份继续 exact。finite 必须通过。此前 replacement TimeMixer/UrbanEV 的 1e-4 在 v3 收紧到 1e-5；历史 1e-4 政策与证据保留，不修改旧报告。第2/6步评价及原 6/10/6 计算顺序保持，其他 replacement worker 仍 6/8/6。

新 scope m6-native-tmark-chain-v3 使用独立 package/session/log/permit/probe 与 revisions/native-time-mark-v3、m-tasks/m-baselines-native-time-mark-v3；科学输入协议仍 native_time_mark_v1。必须从头独立运行全部七组，各代表先重新 serial reference，再实际 parallel：三模型 UrbanEV fold1四H q4；iTransformer/TimeMixer 各四兼容市场 q4、第五同身份市场 formal 后续单路；TimeXer PJM/BE/FR batch16 实际三并发、NP/DE batch4 q2。共7计算身份、25代表、50名义/73最大 worker，名义 300/416/300，本次独立 caps 438/608/438、起始 debit=0；与已消费 v2 合计历史 actual+v3 worst cap=630/880/630。不得退款、搬用 v2 余额或从第四组续跑。

任何 state/loss/normalized metric 超差、finite、身份、数据、guard、业务或未知失败立即停止，不能 resource fallback 或运行时改阈值。仅明确 resource failure 允许 q4→一次 q2→合法已测 serial q1；TimeXer planned q2 资源失败只能采用合法 serial q1。成功链仍为精确用户 N/S retirement boundary → 七组 replacement probe/落盘自动 audit →87 replacement → replacement boundary →21-group M probe/落盘自动 audit →84 M →COMPLETE，无中途人工许可。机器许可 reviewed=false/manual_review=false，COMPLETE 仅技术完成，result_review=pending。

87/1020/3687530 与84/840/294790科学任务、训练参数、seed、数据/time-mark/M监督/test合同全部不变。M 的21项原数值政策及1512/2016/1512预算保持原字节，不将 replacement 阈值扩到 M。此轮仅 W 实现与无模型验收，模型构造/forward/backward/Adam/GPU均0；不 commit/push/start。新字节待 ChatGPT 实际审核及精确 closure，生产 R 未修改。

### 用户最新决定：replacement v4 恢复既有 full-state 准入尺度（2026-10-02，隔离工作区待审）

v3 的 state_atol=1e-5 是用户临时统一窄阈值。v3 已通过 iTransformer-UrbanEV-native 和 iTransformer-EPF-combination-0 q4，随后在 TimeMixer-UrbanEV-native 的 H12 代表失败：step5 normalize_layers.0.affine_bias 差 1.0162340231545386e-5，step6/overall state_max_abs=2.0210838556522503e-5。六步 training loss 逐项相同，第2步原 validation 完全相同；第6步 endpoint evaluation 有微小差异，MSE/normalized SSE 分别 1.5663930819087568e-8 / 1.5663930825898752e-8，MAE/normalized SAE 分别 1.1427270241171072e-9 / 1.142727085417765e-9，均低于既定 metric_atol=1e-6。finite、初始化/RNG/batch exact 与资源检查通过；原失败项仍仅是浮点状态超过 v3 的 1e-5，不能将 failure 改为 Passed。

用户随后明确要求“按之前实验阈值设置”。历史已审核 reviewed-resource-report 及原 TimeMixer/UrbanEV 确认计划证明此前项目正式使用 state_atol=1e-4、loss_atol=metric_atol=1e-6、rtol=0；原 full-state 比较器强制 finite，拒绝非有限状态。旧计划未显式单列 equal_nan/kind/loss_rtol，原字节保持；v4 将 full_float_state、equal_nan=false、loss_rtol=0 明确登记。此决定恢复项目已使用的 full-state 准入尺度，不是继续试探新阈值，不是作者论文标准，不根据正式效果择优。

新 m6-native-tmark-chain-v4 将全部18个 replacement scope（iTransformer、TimeMixer、TimeXer × UrbanEV/PJM/NP/BE/FR/DE）统一为 native-tmark-replacement-fullfloat-atol1e-4-v4：state_atol=1e-4、loss_atol=metric_atol=1e-6、loss_rtol=rtol=0、equal_nan=false。全量浮点 parameters/buffers、gradients、Adam exp_avg/exp_avg_sq 使用绝对界；initialization、RNG、batch/order、optimizer step、非浮点/非张量状态、param groups、shape/dtype、task/profile/source/data 仍 exact，finite 必须 Passed。不引入 loss_rtol=1e-5 或 ModernTCN-ECL 的1e-3特殊规则，不修改M的21项数值政策。

v1/v2/v3 package、failure、permit、trajectory/full-state、budget、controller/progress/launcher 永久保留，只作历史依据，不能授权v4或退款。v2实耗192/272/192，v3实耗144/208/144。v4独立package/session/result根为 native-time-mark-chain-v4、ch3-native-time-mark-chain-v4、revisions/native-time-mark-v4、m-tasks/m-baselines-native-time-mark-v4；科学协议仍 native_time_mark_v1。v4 debit=0，nominal300/416/300、caps438/608/438；历史实际加v4 worst cap=774/1088/774，仅为成本账，不是v4 cap。

v4从头重新生成全部七组独立 serial reference，再运行原实际并发组合：三模型UrbanEV各四代表q4；iTransformer/TimeMixer EPF各四代表q4；TimeXer batch16三代表、q4容量实际宽度3；TimeXer batch4两代表q2。保持7计算身份、25代表、50名义/73最大worker、原formal waves。只有明确resource failure允许q4→一次q2→合法已测serial q1；planned q2仅可退合法serial q1。numeric/finite/identity/RNG/batch/data/source/guard/业务或未知失败立即停止，不能降级掩盖、改阈值或自动retry。

87 replacement（1020 run-epochs / 3,687,530 updates）与84 M（840 run-epochs / 294,790 updates）、seed2024、科学profile、结构、LR/batch、数据/time-mark、监督及test合同全部不变。configs/ch3_formal_profiles.json保持原字节，M protocol仍831163583475c86604286db561cc309b10b31a6d4df10f3b57c0a52f6c74b3fa，M 21-group probe caps1512/2016/1512不变。

正常成功链仍为精确N/S retirement boundary→v4七组probe/落盘机器audit→87 replacement/boundary→原21组M probe/机器audit→84 M→COMPLETE，无中途人工许可；机器准入manual_review=false/reviewed=false，COMPLETE仅技术完成、result_review=pending。本轮只在W实现与无GPU验收，不stage/commit/push/start，不修改生产R；新字节待ChatGPT实际审核及精确closure，不能宣称当前Ready。

### 用户授权：v4 formal admission / recovery execution repair（2026-10-03，隔离字节待审）

v4七组replacement probe已技术通过，formal已开始；现场发现formal准入链反复回放已通过的probe完整数值证据，属于执行效率和职责耦合问题，不是科学结果失败。用户授权调用既有safe-stop中断自有v4链。实际中断在iTransformer第三个UrbanEV wave：8个完整结果保持原字节，4个fold3任务保有epoch7的last/best/history，75个任务未启动；87项分类A/B/C/D/E=8/4/0/75/0。所有STOP/failure、partial、staging、checkpoint和日志保留，原因登记user_authorized_execution_repair_stop，scientific_failure=false、budget_refund=false、result_review=pending。

科学持久化进度与实际成本分账。4项B在checkpoint后已消耗8407 Adam、8408 backward、8410 forward，未保存partial work不退款、不视作已保存epoch。历史实际259281 Adam/259282 backward/288464 forward；从合法checkpoint继续的未来最坏成本3436656/3436656/3898201，已包含必要重复工作，不再次相加。总最坏3695937 Adam/3695938 backward/4186665 forward；相对原optimizer上限3687530，Adam缺口8407。forward的原4178255仅为冻结算术派生上界，不冒称原许可独立授予该forward cap。当前恢复不可执行，必须另有精确额外执行预算授权；不能用可能early stop解释通过。

恢复scope为m6-native-tmark-chain-v4-recovery1，是execution identity，不是新科学variant。science_baseline_commit=0734f3e91f15854f75907c30d10d43e942f69267；原A保持真实旧执行commit，B记录旧checkpoint前缀与新worker执行版本，C/D记录新closure的实际worker版本。controller_execution_commit和worker_execution_commit在后续实际closure后取得，本轮不预填。science computation fingerprint绑定未变模型/作者源码、adapter、数据/time-mark、profile、更新/评价/optimizer/BestState和训练epoch循环。

原固定v4 probe complete是派生artifact manifest的信任根：先验原报告SHA，再提取其expected artifact SHA、校验当前文件并登记size；紧凑admission-summary由固定原报告确定性投影，不能现场自签Passed。准备投影核验与未来每个formal监督器生命周期一次SHA完整性scan分别记账；formal group/wave/task/worker只消费紧凑许可与跨进程、当前自有启动链认证的runtime-admission，不回放probe数值或读取raw sidecar。M仍保留各组即时gate，M_AUTO_AUDIT完整审计一次，M formal新生命周期扫描其manifest一次；M的16项None政策仍完整exact检查，5项full_float_state不再对相同generic payload重复比较，TimeMixer UrbanEV专属endpoint gate保留。

A只读继承、不train/test；B严格read-old/write-new，恢复model/optimizer/RNG/generator/BestState，从最后完整epoch+1继续，历史是不可变prefix与新suffix，旧best仍胜出时使用其精确字节来源；C必须精确0模型工作并保留旧准备材料，新路径fresh；D新路径fresh。若某任务无法确认正式test是否已经访问，或其执行身份无法确认，则该任务必须标记为E类；不得自动恢复，也不得再次执行正式test，必须停止该任务的自动恢复流程并返回后续明确处理。有效87格来源按inventory固定，不按效果选择；只在原wave内跳过A，不跨wave补位、不提高q。87成功closeout/seal只做一次，未成功seal的中断校验可在后续合法生命周期重做；进入M后只轻量消费sealed boundary。

drain-stop持久写DRAIN_STOP，不signal当前wave；当前wave技术完成后停止下一wave/group，登记DRAINED、technical_failure=false、resume_eligible=true，恢复仍需inventory/预算/版本/来源/资源/STOP核验。emergency safe-stop保持owned PID/start_ticks清理。原87/1020/3687530科学合同与84/840/294790 M合同、21组M probe/caps1512/2016/1512、各数值政策完全不变。本轮只完成repair、受限CPU checkpoint审查和无模型fixture验收，没有重跑v4 probe、没有恢复训练、没有修改生产R或stage/commit/push；新字节尚待ChatGPT实际审核。

### M6 最新用户协议修订：baseline-unified96-onecycle001-v1（隔离 W 待审候选）

用户本轮明确取代此前 replacement/recovery 剩余执行安排：七个 baseline 按 AMD、DLinear、PatchTST、iTransformer、TimeMixer、ModernTCN、TimeXer 顺序，在新统一协议中全部 fresh 重训。MS 是 UrbanEV F4 六 fold×四 label horizon 加 PJM/NP/BE/FR/DE 各 H24，203 项；M 是 ETTh1/Weather/Exchange×H96/192/336/720，84 项，共287项。J 保持既有冻结身份及结果，不改结构/L/LR，不重训或重新 test；N/S 不加入，ECL 仍不进入 M。这里统一的是七 baseline 新训练协议，不能声称 J 与 baseline 使用完全相同超参数。

UrbanEV 仍 T=12、pred_len=1、C=11、label H3/6/9/12，既有 F4 业务列/target/aux、split/scaler/metric/test-once 不变；其它 MS/M 全部 T=96。MS 中 iTransformer/TimeMixer/TimeXer 从已审核 native v4 profile 继承，其余四模型从生产552矩阵对应 F4/EPF profile 继承；M 从当前已审核 M profile 继承。结构、batch/eval_batch、epochs、dropout、seed、原生 historical time-mark/freq 和其它冻结参数不变。只有 T、training.lr 及 scheduler 变更，287份 old→new 全字段对照存于外置新包。

统一 scheduler 取自锁定 TimeMixer 作者 commit e24610583b36fdd8c76cc17a8df4e65759a5f460 的 scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_unify.sh、run.py 和 exp/exp_long_term_forecasting.py；正式 PyTorch2.0.1 签名已读取。显式冻结 OneCycleLR(max_lr=0.01,pct_start=0.2,anneal_strategy=cos,cycle_momentum=true,base_momentum=0.85,max_momentum=0.95,div_factor=25.0,final_div_factor=10000.0,three_phase=false,last_epoch=-1,verbose=false,total_steps=None,epochs=该任务冻结上限,steps_per_epoch=该任务 train_batches)。Adam 的 beta2/eps/weight_decay 等父值保留，beta1 按标准 cycle_momentum 更新；不是 fixed LR0.01。UrbanEV/M 继承10epoch、EPF继承20epoch，不能把作者 ETTh1 的10epoch套到所有域。这是用户选择的统一公平/诊断协议，不是各 baseline 作者原始推荐设置。

scheduler 仅在成功 Adam update 后 step 一次，validation/test 不 step。新 best/last 使用 ch3-state-v2-onecycle 并保存 scheduler config/state/count/LR/betas；resume严格校验身份并恢复，旧 ch3-state-v1 仍供原协议读取，禁止跨协议恢复。PyTorch2.0.1 state_dict 中的 anneal_func 绑定方法以冻结 cosine 身份处理，其余实际状态完整序列化，避免将带 optimizer 的 callable 写入 probe JSON。

当前旧 recovery1 已使用自身 safe-stop 退休：停时 REMAINING_TMARK_FORMAL/iTransformer，16项有效完成、4项 fold5 partial（四H history到epoch2）、67未启动，原 supervisor/group/四 worker 与其监测子实例均退出。原因 user_authorized_protocol_supersession，不是科学/模型/效果失败。原 v4＋recovery formal实耗 Adam/backward/forward=573259/573261/641201，全部checkpoint/history/result/STOP/failure保留，不退款；旧8407 allowance不转给新实验。退休回执 baseline-unified-protocol-retirement.json SHA256=64869e3735c49334dfd2e127691c1c2c958670c0e3d5f3d3b59f5e696aa26604。旧完成/partial/checkpoint均不得进入新287格。

新 scope m6-baseline-unified96-oc01-v1、新结果根 baseline-unified96-onecycle001-v1/{MS,M,probe,queue} 独立。单次用户启动正常链：协议 preflight→MS probe→MS AUTO_AUDIT→MS七模型串行正式→seal MS boundary→M probe→M AUTO_AUDIT→M七模型串行正式→COMPLETE。技术失败停止且保留证据，不自动重试/调参，不按中途效果取消模型。每组即时数值/资源 gate 保留，AUTO_AUDIT完整落盘审计一次；formal 每生命周期仅一次 probe manifest SHA scan，worker只消费紧凑许可/runtime refs，不重放 raw numeric。COMPLETE仅技术完成，result_review=pending。

新预算为待审提案：MS203/2380 run-epochs/7974050 Adam、backward/9019739 forward；M84/840/300150 Adam、backward/359040 forward；合计287/3220/8274200 Adam、backward/9378779 forward。MS最小代表计划15组63代表（UrbanEV每模型fold1四H，EPF按真实兼容结构/batch分组，因各市场 scheduler 步数不同，各市场保留独立短轨迹）：名义126 worker、756/1024/756，cap187 worker、1122/1520/1122。M21组84代表：名义168 worker、1008/1344/1008，cap252 worker、1512/2016/1512（Adam/forward/backward）。数值政策逐scope继承，未扩大容差；额外核 scheduler state/LR/beta1/steps exact。仅资源失败允许登记的q4→一次q2→合法serial q1；原q2组仅可退合法q1。

旧结果永久保留，新协议若最终用作论文 baseline，主结果整体采用新协议，禁止按 test 在 old/new 间逐格择优。J仍如实使用自身冻结协议。工期依据已绑定历史同模型/域 worker runtime、精确新窗口/batch/epoch 和固定 wave估算，非新协议实测；probe后须用初始化/稳定 update/validation 分项修正，不将六步包整体除6外推。本轮只实现、CPU/无负载验收、dry-run与模板preflight检查；未stage/commit/push、未GPU/probe/formal/start，生产 R 原字节不动，W尚未合并生产，新字节待 ChatGPT 实际审核和新预算批准。

### M6 最新用户门禁修订：baseline-unified96-onecycle001-v2（2026-10-03，隔离 W 字节待审）

v1 closure 为 22b6447c4e62078d903446690225504f26e7a3ed。MS probe 在11/15组Passed后，于ModernTCN-EPF-compatible-0因exact numeric gate失败自动停止，未进入正式训练。四个已完成q4成员的state/loss/normalized metric最大差分别为：PJM 1.3441592454910278e-4 / 1.1920928955078125e-7 / 1.0502029601511254e-7；NP 2.063065767288208e-4 / 1.1920928955078125e-7 / 2.1928189686271082e-7；BE 2.4513527750968933e-4 / 2.384185791015625e-7 / 2.3932049497688013e-7；FR 1.7508119344711304e-4 / 2.384185791015625e-7 / 3.2853599907234354e-7。v1 failure/progress/trajectory/full-state/budget与所有历史现场保持原字节，实际642 Adam / 642 backward / 872 forward，budget_refund=false，不resume、不改成Passed。

用户明确批准仅MS ModernTCN-PJM/NP/BE/FR/DE五scope从None/exact改为full_float_state：state_atol=5e-4、loss_atol=metric_atol=1e-6、loss_rtol=rtol=0、equal_nan=false。全部浮点parameters/buffers、gradients、Adam exp_avg/exp_avg_sq按有限绝对阈值比较；初始化/RNG/batch/order、optimizer step、非浮点状态、param groups、shape/dtype、task/profile/source/data、scheduler配置/step/LR/beta1仍exact，finite必须Passed。ModernTCN-UrbanEV仍exact，其余模型MS与全部M政策不变。这是用户批准的M6 technical numeric admission threshold revision，不是作者论文标准。

v2协议/结果身份baseline-unified96-onecycle001-v2，scope m6-baseline-unified96-oc01-v2，package/result/probe/queue/launcher/session与v1隔离。原v1配置保留，新v2配置逐项证明287份正式训练profile完全一致，仅任务/协议namespace、数据元数据索引绑定及上述五scope政策变化；UrbanEV T12，其它T96，OneCycle、batch/epochs/结构/data/seed2024/metric/test/native mark均不变。七baseline、MS203＋M84、J/N/S/ECL排除保持，不能以v1的11组Passed拼接v2准入。v2必须fresh重跑完整15组MS probe，后续仍为MS_AUTO_AUDIT→203 MS正式→seal MS→21组M probe→M_AUTO_AUDIT→84 M正式→COMPLETE；正常成功路径一次用户start，科学result_review=pending。

用户独立批准v2 MS probe从0计账，cap1122 Adam / 1122 backward / 1520 forward；v1 actual＋v2 worst为1764/1764/2392，v1不退款、不转移余额。M probe cap1512/1512/2016不增加。正式预算保持MS203/2380/7974050 Adam及backward/9019739 forward、M84/840/300150 Adam及backward/359040 forward；total287/3220/8274200 Adam及backward/9378779 forward。仅明确resource失败允许q4→一次q2→合法已测q1，原q2组仅可退合法q1；numeric/finite/identity/scheduler/data/guard失败立即停止，不自动放宽或重试。

本轮只做无模型验收及v1保存payload的只读离线新政策重放；重放不构成v2实测admission，DE不能据四个旧q4成员重放冒称已通过。真实dry-run/模板preflight仍拒绝不可执行模板及尚未closure的新字节；未生成可执行start authorization，未GPU/probe/formal/start、未stage/commit/push，生产R与旧artifact不改。新实现与文档仍待ChatGPT实际字节审核，未合并生产canonical。

### 用户最新协议切换：baseline-unified96-onecycle001-v3（2026-10-03，隔离 W 新字节待审）

本小节取代 v2 剩余执行计划，以上正文保留历史时点。用户明确要求安全中止 v2，并以全新 v3 fresh 重跑完整链。开发基线为 b1a88bff143e4513761ac02895efd683fe32b499，生产 R 仍为 5341fbcb7c9f4f97658728d79b1af5487f7d38c3，不合并、不部署到 R。停止前实读为 MS_FORMAL_ALL_BASELINES、AMD、已完成 wave1；当时 wave2 为 fold3 四H。只调用旧唯一 safe-stop：入口先持久化 STOP，随后因旧 command_has 要求 start 而拒绝合法 group-child，返回 exit1；运行 controller 继续按既有 STOP 逻辑清理 owned tree，未人工追加 signal。supervisor 16699/169639165、group-child 31619/169810770、workers 54584/170649808、54585/170649816、54586/170649822、54587/170649835、监测子 14310/171196207 原实例全部退出，原 tmux 已退出，GPU计算进程为空。原 STOP/failure/controller/progress/result/checkpoint/history/budget/log/probe 保留；v3 同时以实际完整命令＋PID/start_ticks＋parent 校验修正该停止入口的 group/probe-child 判定，不扩大信号范围。

退休性质 user_authorized_protocol_supersession、scientific_failure=false、budget_refund=false、result_review=pending、resume_into_v3=false。MS实际为8 completed、4 partial/staging、191 not-started；M为0/0/84。四个partial为 AMD UrbanEV fold3 H3/6/9/12，history均完成epoch6；持久预算 Adam/backward/forward 分别为24295/24296/26930、24519/24519/27112、24411/24411/26968、24514/24514/27028。MS正式持久累计244649/244650/272116，MS probe实际756/756/1024；正式持久计数作为可证明下界，未持久化瞬时消耗明确unknown，不猜测为0、不退款、不转移。退休回执 native package v3/v2-retirement-evidence.json SHA256=f4a85296ece0cadb9f3835e0c3252517d0b18f9d89150cb81bd6b1675d6cf20e。v1/v2所有结果只作历史证据，禁止resume、copy/hardlink/symlink、旧best初始化、旧test填格或按效果逐格择优。

新 scientific_protocol baseline-unified96-onecycle001-v3、scope m6-baseline-unified96-oc01-v3、tmux ch3-baseline-unified96-oc01-v3，package/result/probe/queue/permit/launcher/STOP全部隔离。七baseline固定 AMD→DLinear→PatchTST→iTransformer→TimeMixer→ModernTCN→TimeXer；MS为UrbanEV七模型六fold四H共168＋五EPF共35=203；M为ETTh1/Weather/Exchange四H共84；total287 fresh，J/N/S/ECL排除。J冻结身份、结构、来源、结果及test不动。

直接从审核 v2 config 的固定path/SHA继承。唯一新训练科学变化是168项 UrbanEV training.epochs 10→20、patience None→5、training.scheduler.epochs 10→20；每任务 steps_per_epoch不变。UrbanEV仍T12、pred_len1、C11、F4、batch/eval128、Adam、weight_decay1e-7、seed2024；EPF仍20/5，M仍10/None；T/H/结构/dropout、optimizer、数据/split/train-only scaler、feature/target/aux、native historical time-mark、metric/test/source全部保持。OneCycle其余显式字段全部与v2一致：max_lr=.01、pct_start=.2、cos、cycle_momentum=true、base/max_momentum=.85/.95、div_factor25、final_div_factor10000、three_phase=false、last_epoch=-1、verbose=false、total_steps=None。最大UrbanEV schedule为20×各任务train_batches；沿用BestState严格validation MSE改善、tie留早best、连续5次未改善在下一epoch前停，正式test仍仅对validation-selected best.pt一次；不得test驱动停止或取消后续任务。

MS/M numeric registry逐项继承v2且diff=0：MS registry SHA d29aaf90b9db5dfa8625500b114ee2f580bfa01e453dce31064bbee40d1be811，M registry SHA 3f20ba5a36d921f1b4178b98648421774469d861b9ab6f794d4e3c0385cfc084。ModernTCN五EPF仍5e-4 full_float_state、loss/metric1e-6、rtol/loss_rtol0、equal_nan=false，UrbanEV仍exact；其它policy不改。数据metadata科学值不变，仅task-id索引投影到v3。旧probe不授权v3：MS fresh15组63代表、M fresh21组84代表，serial→planned parallel→numeric/resource gate→AUTO_AUDIT；仅明确resource允许原q4→一次q2→合法已测serial q1、原q2→合法q1，numeric/finite/identity/scheduler/RNG/batch/data/source/guard/unknown失败立即停止。

用户本次已明确授权全部新独立预算，按正式helper独立算术复核一致（以下顺序均Adam/backward/forward）。MS正式203 runs、4060 max run-epochs、15274280/15274280/17173829；M正式84/840、300150/300150/359040；正式total287/4900、15574430/15574430/17532869。MS probe nominal756/756/1024、cap1122/1122/1520；M probe nominal1008/1008/1344、cap1512/1512/2016；两阶段probe worst2634/2634/3536。v3 debit=0，各旧消费不转移；additional_search=0、seed2024，不因early stop挪用预算、不自动扩cap或追加候选。

单次成功链仍为BASELINE_PROTOCOL_PREFLIGHT→MS_RESOURCE_NUMERIC_PROBE→MS_AUTO_AUDIT→MS_FORMAL_ALL_BASELINES→SEAL_MS_BOUNDARY→M_RESOURCE_NUMERIC_PROBE→M_AUTO_AUDIT→M_FORMAL_ALL_BASELINES→COMPLETE，正常技术Passed路径无需逐模型/阶段人工介入，COMPLETE仅technical complete、result_review=pending。76/76无模型方法验收通过，0 skip；torch/models导入被禁止，模型构造/forward/backward/Adam/GPU initialization均0。ch3_runner仅code_binding新增v3登记，训练/update/evaluate/checkpoint/test数学函数AST保持开发父版本。模板所有执行授权布尔值仍false、closure_commit=null，真实模板preflight必须拒绝；用户预算已登记并不等于未来未知closure已审核。本轮不stage/commit/push，不生成start-review、不启动v3/GPU/probe/formal/test，新实现字节待ChatGPT实际审核。

#### v3 审核后历史测试版本隔离修复（2026-10-03，新修复字节待审）

原 Codex targeted 76/76 Passed 证据继续保留，其范围为当前 unified＋v3，未包含历史 v2 模块。ChatGPT 服务器实读审核的 broader v2＋v3 regression 实际为48项、5 failures、2 errors：历史 v2 测试错误使用当前 v3 mutable globals，取得了 v3 package/session/profile/budget/probe task IDs。这是 test harness regression，不是科学、模型或效果失败；此前失败事实和上一版交付材料留存，不改写为从未失败。

本次仅在 tests/test_m6_baseline_unified_v2.py 将18个历史测试绑定冻结 v2 commit b1a88bff143e4513761ac02895efd683fe32b499、固定 SHA 的 v2 config/start-review/plan及准备证据。历史许可拒绝行为只提取旧源中的校验函数，不加载或运行旧链；过去准备时的fresh状态按留存证据审核，不要求已经退休的v2现场重新变成fresh。测试方法、原断言语义及数值容差保留，无skip/expectedFailure。正式Python、CUDA_VISIBLE_DEVICES=''、PYTHONDONTWRITEBYTECODE=1及指定restricted_regression PYTHONPATH下，v2＋v3复验48/48、统一三组复验94/94均Passed，0 failure/error/skip；仅纯数据fixture及已保存payload只读比较，模型构造/forward/backward/Adam/GPU initialization均0。

相对于修前v3候选，科学配置/profile/numeric registry/预算和全部科学及执行代码字节均未变；本次只增加历史测试修改与两份文档的审核记录。MS203、M84、total287，UrbanEV168项20/5、EPF35项20/5、M84项10/None，numeric diff=0和全部已授权预算保持。未stage/commit/push、未生成start-review、未启动v3；新测试/文档及刷新交付证据仍待ChatGPT最终实际字节审核。


### M6 用户新增：baseline-type1-followup-v1（2026-10-05，独立 N 分支候选，未审核/未启动）

本轮在 `m6/type1-followup-v1`、`AMD-type1-followup-v1` 从运行基线 `df6a16403e10d51097db8c88829909c533d15652` 创建独立 worktree。运行 W 的 v3 与生产 R 的 `5341fbcb7c9f4f97658728d79b1af5487f7d38c3` 不编辑、不停止；锁定作者仓库与环境不改。新分支仅为候选权威增量，未合并生产，未stage/commit/push，ChatGPT尚未审核本轮新字节。

新协议 `baseline-type1-followup-v1`，scope `m6-baseline-type1-followup-v1`，session `ch3-baseline-type1-followup-v1`；package/result/probe/queue/log/permit/STOP 全部独立。七模型固定顺序 AMD→DLinear→PatchTST→iTransformer→TimeMixer→ModernTCN→TimeXer。第一环 UrbanEV F4 只选 fold1、6 × label H3、12，共28项，覆盖训练规模/跨度两端，非按test挑选；这是首批28格正式补充实验，覆盖28/168而非完整六fold，也不是validation-only开发。T12/pred_len1/C11、20epoch/patience5、batch/eval128、结构、输入11列、target/aux、原生historical marks、split、train-only scaler、seed2024及其它父值不变，只改优化调度。第二环 PJM/NP/BE/FR/DE ×七模型、H=pred_len24，共35项；T由96恢复168，只再改变优化调度，20/5及各模型/市场自己的batch/eval/结构和其它profile保持。T依赖的train窗口/batch与metadata按固定endpoints重算，validation/test目标边界和scaler不变，无test标签预算读取。合计63 fresh run/1260 max run-epochs；J/N/S/ECL不入队，无新M，不补齐其余140个UrbanEV格。

学习率仅借锁定TimeXer commit `76011909357972bd55a27adba2e1be994d81b327` 的type1规则，来源run.py、utils/tools.py、exp/exp_long_term_forecasting.py、scripts/forecast_exogenous/EPF/TimeXer.sh逐份SHA绑定；tools SHA `64048f154f899221d202ee740d06a32d30b5ba01425c9d69a3f20af8e43c15b8`，exp SHA `c235090c91d22275ba871ed6972a9e4521c6b5b53239ef38b4a757b742adcacc`。不执行作者训练脚本，不搬入作者每epoch访问test或默认10/3。两环Adam initial_lr=1e-4，基础betas固定，eps/weight_decay保留，移除OneCycle动态beta1。第e轮实际LR为 `1e-4*0.5**max(e-2,0)`：前两轮均1e-4，第3轮5e-5，第4轮2.5e-5。epoch末validation与BestState更新后、若继续训练才调整下一轮；严格val MSE `<`、tie保留早best、连续5次无改善停止不变。独立type1 checkpoint schema保存updates/completed_epoch/LR位置/current-nextLR/固定betas/config，完整恢复避免重复/漏衰减；旧OneCycle路径保持兼容，跨科学协议checkpoint禁止恢复。

用户明确授权每个新run在validation-selected best锁定后一次正式test，两环均如此。probe不访问正式test；训练/早停不依据test。已有v3及T168旧formal/补做（包括supplements/epf4-timemixer-v1）对照只读落盘指标与身份，不重训、不重test、不逐格择优、不自动替换论文主表。历史test曾被查看如实登记，单seed std=N/A。未来同协议扩展需另行授权且精确继承28项，不因队列改名重训；本轮不实施扩展。若无法确认test是否已访问，停止该项自动恢复，不再次test；checkpoint恢复须审计scheduler/optimizer/RNG/best/patience/history一致性，本链fresh-only，不构建无限resume平台。

新等待器未来一次用户start/arm即可持久tmux等待。固定上游v3 start-review SHA `4c39ec5e025ca8ea492129d459ba395c70e68d1d2a6a698b2b142518bf8e632d`、controller SHA `1b8e6098450326577a3f42091ac09237a2c84caac12d2aff7a9b6403645dda79`、owner36799/start_ticks171530843及scope/commit/path进入代码和许可绑定。等待期间不初始化GPU、不占组GPU锁、不启动probe/worker；旧链running是合法arm状态。只有合法顶层complete覆盖203 MS＋84 M、MS/M sealed边界和完整唯一结果索引/来源/expected refs一致、无旧STOP/failure、原owner及登记owned实例全部退出，才能启动新计算。只完成MS、owner消失无complete、伪造/错误SHA/缺失/重复/来源变化均不能启用。旧链不做test/probe数值回放；旧checkpoint只继承其sealed索引引用，不反序列化。新handoff完成核验后由当前受控生命周期MAC封存，不能以现场JSON自签READY。新的safe-stop只影响新scope，不向旧v3发信号。

正常链：WAIT_V3_COMPLETE_AND_RELEASED→协议preflight→Urban probe→Urban AUTO_AUDIT→28 formal/test→Urban sealed boundary→EPF probe→EPF AUTO_AUDIT→35 formal/test→COMPLETE。第一环效果不控制第二环；技术完整就按固定计划继续，无中途人工许可。技术失败保留现场并停后续，不自动调参/改阈值/fresh retry。COMPLETE仅technical complete，result_review=pending。

必要技术准入共15组63代表，各自独立serial六步，再实际parallel。Urban每模型f1H3/f1H12/f6H3/f6H12同波候选q4；非TimeXer六模型EPF五市场q4候选4+1；TimeXer PJM/BE/FR为batch16实际3、q4容量，NP/DE为batch4 q2，不跨模型并发，不改batch。跨epoch减半/20轮/resume以无模型合成测试证明，不把六步称作完整epoch。全部numeric逐scope继承v3；ModernTCN五EPF仍5e-4 full-state、Urban exact，浮点完整state/gradient/Adam moments与finite/初始化/RNG/batch/order/step/身份检查保留，scheduler/LR/betas exact，TimeMixer Urban step2/6 endpoint gate保留。仅resource可一次q4→q2→合法serial q1、原q2→serial q1；numeric/finite/identity/scheduler/data/guard失败不得fallback。每环AUTO_AUDIT完整审计一次，随后compact summary/manifest/permit/runtime refs；每formal监督器生命周期一次manifest SHA scan，group/wave/task/worker零probe原张量回放，generic payload不重复比较，第二环轻量消费第一环sealed边界。GPU组锁由实际执行子组持有，等待supervisor不自锁。

用户授权固定预算已由helper独立复算，数字顺序Adam/backward/forward：Urban正式28/560，2414440/2414440/2711114；EPF正式35/700，672860/672860/778932；总63/1260，3087300/3087300/3490046。Urban probe 7/28，nominal336/336/464、cap504/504/696；EPF probe8/35，nominal420/420/560、cap618/618/824；总nominal756/756/1024、cap1122/1122/1520。cap含预登记resource回退，不含无限重试；所有旧消费保留、不退款、不抵扣，技术失败不抹账，效果不佳/正常早停不增加预算，不新增seed/search/candidate。预算本轮已经由用户批准，模板flags仍为false，仅表示尚待新字节审核/closure/start-record绑定，不是再次请求预算授权。

192/192无模型验收Passed，0 failure/error/skip；真实模型构造/forward/backward/Adam update/GPU initialization均0，GPU未初始化。原OneCycle和历史v2/v3回归保留；历史配置/来源显式版本隔离，所有fixture输出移到新P，不让新worktree或旧链实时结果误作历史fresh状态。初轮80项1failure、后续111项2error、187项5failure及187项1failure/3error均保留，属于测试/执行身份接线发现，不删断言/skip/改容差。送审前纯metadata复核发现15项EPF native mark形状仍继承T96，已按T168更正并增加逐项shape/window绑定验收；scaler/prefix/source内容和所有预算保持，未发生任何真实模型尝试。合成tmux首轮因缺start argv被身份门禁拒绝，重做的受控fixture已证明SSH启动器返回后等待存活、owned safe-stop退出、old_chain_signal_sent=false；没有启动真实新等待器。

工期为历史同model/域runtime、精确窗口/batch/max20及实际wave的估算，不是新协议实测：候选q通过且无其它GPU负载时，Urban约27.8–55.6小时、EPF约2.9–5.8小时；新probe名义约0.24–1.07小时，总新计算约31–63小时，另加旧v3剩余等待。资源退为serial时正式约2.95–5.90天；early stop可能减少actual但不扩大cap。当前只生成不可执行模板，dry-run exit0展示计划且blocked、模板preflight exit2按预期拒绝未review/closure/start-record绑定。READY_TO_ARM_HANDOFF与READY_FOR_GPU_EXECUTION分别登记；本轮二者false，没有start-review、真实session、结果根、GPU/probe/formal/test。外置P存完整task/profile/source/budget/调用链/验收失败账/diff/SHA与操作命令，N最新canonical/M6为候选，W/R原文和历史artifact保持。
#### baseline-type1-followup-v1 审核后完整审计职责修复（2026-10-05，N候选修后待复审）

ChatGPT服务器实际字节/代码路径审核确认：原type1候选许可只有type1_scope，run_probe尾部仅排除recovery_scope/unified_scope，因此probe全部成功后仍调用validate_probe_completion，随后对应AUTO_AUDIT再次调用。原192/192无模型证据继续对其原覆盖范围有效，但未覆盖这条双调用路径；该发现来自代码审核，不是已启动GPU后观察到的耗时、数值或科学结果失败。

本轮仅在utils/ch3_native_execution.py的尾部归属条件加入type1_scope排除，与既有recovery/unified保持一致；不伪造unified_scope，不删除legacy完整审计，不删除serial自检及逐组numeric/finite/resource/identity即时gate，不延迟失败组检查，不删除AUTO_AUDIT。新增tests/test_m6_type1_followup.py中的7项调用位置回归，用合成fixture驱动实际run_probe及audit_probe入口；仅模型/进程计算和完整数值回放使用可计数替身，没有整体mock这两个待验证入口。

实际成功调度证据：URBAN_SUBSET的run_probe尾部完整审计0、AUTO_AUDIT完整审计1；EPF_ALL同为0/1；两环合计2而非4。逐组实际compare和串行自检继续执行，numeric/finite/identity失败均不派发后续组、不走resource fallback。legacy尾部仍1，recovery/unified尾部仍0。formal permit/config/worker/group/wave的完整probe审计及原始张量比较均0；formal监督器manifest扫描仍每生命周期1次，子路径不再次扫描，跨进程轻量绑定、第一环sealed边界消费方式不变。

本轮原192项覆盖加新增7项实际共199/199 Passed，0 failure/error/skip；模型构造、真实forward/backward/Adam、GPU初始化全部0。首轮实际199项出现1个error：recovery兼容fixture错误读取旧退休scope的STOP，已将该fixture的CONTROL显式绑定其临时目录，保留原STOP检查和全部断言；失败日志及旧送审材料完整归档，不加skip或放宽容差。

两份type1 config、63任务/profile、numeric registry、数据metadata、TimeXer来源recipe、type1 scheduler实现、正式/probe计划与预算逐份SHA保持原值；W/v3与生产R、作者仓库及环境不改，不signal或停止旧链。候选变化仍为原24项（12 modified+12 untracked），本轮相对上次候选仅执行文件、测试文件和两份最新文档4项变化。外置P刷新调用次数、验收/不变证明、inventory/完整patch/report/evidence-index及受代码绑定影响的非执行fixture；保留原192和前一版审核材料。未stage/commit/push，未物化可执行许可、未启动真实等待器/probe/formal/test；修后新字节仍待ChatGPT审核。

### M6 用户最新协议：baseline-type1-followup-v2 三环准备（2026-10-05，N候选待实际字节审核）

本轮在同一N分支 `m6/type1-followup-v1` 从真实closure `9341e4eb44ed9225a3965b102b2504b1fecfe830` 开发v2，不新建worktree。先复核N/W/R的HEAD、tracking、live remote、0/0、clean和原owner身份，再仅调用原v1的safe-stop。停止时v1仍为WAIT_V3_COMPLETE_AND_RELEASED，owner PID22206/start_ticks187020263，没有owned计算子进程、probe或正式输出；停止后原实例和原tmux均退出。实际completed/partial/not-started为0/0/63，模型、GPU、forward/backward/Adam和正式test消耗均0；旧STOP/failure/controller/log及许可保留。退休性质为user_authorized_protocol_supersession，scientific_failure=false、budget_refund=false、old_v3_signal_sent=false、resume_into_v2=false。外置新包的 `v1-retirement-receipt.json` SHA为 `b472ac18d204d4d668e4c55ed2ae3b848f8d420888e997e2ade67f08df97a5de`。原v3 owner36799/start_ticks171530843停止前后及交付核验仍存活，进度可自然推进；W的df6a164...和R的5341fb...、作者仓库、环境和数据均不编辑、不signal。

新协议 `baseline-type1-followup-v2`，scope `m6-baseline-type1-followup-v2`，session `ch3-baseline-type1-followup-v2`；package/result/probe/queue/permit/log/STOP全部fresh隔离。七模型固定顺序仍AMD→DLinear→PatchTST→iTransformer→TimeMixer→ModernTCN→TimeXer。URBAN_SUBSET为F4 fold1、2×H3、12，共28项：T12/pred_len1/C11、train/eval batch128、20epoch/patience5；它只覆盖28/168，不是完整六fold，也不补其余140格。EPF_ALL为五市场PJM/NP/BE/FR/DE×七模型，共35项：T168/H=pred_len24、所有模型实际train/eval batch32、10epoch/patience3。M_ALL为ETTh1/Weather/Exchange×H96/192/336/720×七模型，共84项：T96、所有模型实际train/eval batch128、10epoch/patience=None、全通道监督和全通道评价。合计147 fresh run/1750 max run-epochs（560/350/840）；J/N/S/ECL、新seed/候选/搜索均排除。

batch来源绑定冻结v3的TimeMixer M profile：ETTh1/Weather四H原128，Exchange四H原512由用户明确覆盖为128；不引用在线最新脚本或TimeXer batch。accumulation=1，不缩小microbatch、增积累、开AMP或修改BatchNorm来绕过资源。逐项parent→new profile差异仅包含本次批准的抽样/身份、T、batch/eval_batch、epochs/patience、lr/scheduler；模型结构、目标及有序特征、native historical marks、split/scaler、seed2024、Adam基础betas/eps/weight_decay、dtype/threads/workers/shuffle/drop_last保持。T/batch相关窗口、marks形状、批次尾部和预算重算；评价仍按全元素SSE/SAE加权，eval_drop_last=false，未读正式test标签做预算。M正式closeout验证真实all-channel身份、输出顺序与元素总数，不把MS换标签冒充M。

调度身份为 `type1_horizon_scaled_v1`：Adam initial_lr=1e-4、基础betas固定，无OneCycle动态beta1。第e轮实际LR为 `1e-4*0.5**(max(e-2,0)*c(E))`，E<=10时c=1，E>10时c=8/(E-2)。E10逐轮等价锁定TimeXer type1；E20的拉长是本项目本轮用户明确修改，不是作者原配置。前两轮均1e-4；E20第3轮约7.348672461377994e-5、第10轮约8.504937501089857e-6、第20轮3.90625e-7。E预先固定，不随早停/暂停/恢复重新缩放，不按validation临时调速。epoch末validation/BestState之后、若继续训练才用绝对epoch闭式设置下一轮LR，再保存边界；独立state/checkpoint schema绑定类型、公式/E、updates、completed_epoch、LR位置、current/next LR与固定betas。旧type1及OneCycle实现兼容，新协议拒绝旧checkpoint，模型前向、损失、数据和评价数学未改。

上游仍绑定完整unified-v3的df6a164...、固定start-review SHA `4c39ec5e025ca8ea492129d459ba395c70e68d1d2a6a698b2b142518bf8e632d`、controller SHA `1b8e6098450326577a3f42091ac09237a2c84caac12d2aff7a9b6403645dda79`及原PID/start_ticks。未来用户只arm一次；合法旧链running允许READY_TO_ARM_HANDOFF，等待期间READY_FOR_GPU_EXECUTION=false、不初始化GPU、不占训练组锁。只有合法顶层完成覆盖203 MS＋84 M、完整结果索引和sealed边界一致、无旧STOP/failure且登记owned实例释放，才自动启用新三环。链为等待→新preflight→Urban probe/AUTO_AUDIT/formal/test/seal→EPF probe/AUTO_AUDIT/formal/test/seal→M probe/AUTO_AUDIT/formal/test/seal→COMPLETE。每环技术完整就继续下一环，效果不控制派发，无中途人工许可。M只消费本新协议28＋35项前序边界，不复用旧M许可，不残留87/203/287项新边界硬编码。

必要probe为Urban7组28代表、EPF8组35代表、M21组84代表，共36组147代表，各自独立serial六步。Urban实际跨fold1/2的四成员候选q4；非TimeXer EPF五市场固定4+1，TimeXer保留PJM/BE/FR三成员q4容量及NP/DE两成员q2，统一batch32不等于结构等价；M每模型×数据集四H候选q4。不超过q4、不跨模型补位。numeric registry逐scope继承v3，diff=0；ModernTCN五EPF仍5e-4 full-state、Urban仍exact，M原21项政策不变，TimeMixer Urban endpoint gate保留。仅resource可预登记降q，q1也放不下固定batch则阻塞，numeric/finite/identity/scheduler/RNG/batch/data/source/guard失败即停、不放宽。三环实际调度分支验证run_probe尾部完整审计0、AUTO_AUDIT各1，合计3而非6；逐组即时gate保留。formal零完整probe回放、零逐task原始probe张量读取，每环formal监督器生命周期一次manifest scan，子进程用紧凑summary/permit/runtime refs；后环轻量消费前序sealed边界。

用户已授权固定147项和必要技术计划，helper独立复算预算（Adam/backward/forward）：Urban正式28/560为1028440/1028440/1143128；EPF正式35/350为399000/399000/467845；M正式84/840为108080/108080/128877；总147/1750为1535520/1535520/1739850。Urban probe nominal336/336/464、cap504/504/696；EPF nominal420/420/560、cap618/618/824；M nominal1008/1008/1344、cap1512/1512/2016；总nominal1764/1764/2368、cap2634/2634/3536。probe/formal独立核账，cap含固定resource回退而非无限重试，旧消费不退款、不转移。效果不佳和正常早停不构成补做理由。

三环均由validation严格MSE改善、tie保留早best和既定patience选best，锁定后正式test一次；probe无正式test。若无法确认某任务正式test是否已访问，停止其自动恢复，不再次test。全部147项保留，不按旧新test逐格择优，不自动替换论文主表；对照只读合法既有指标，包括EPF supplements/epf4-timemixer-v1，不重训/重test旧run。保留此前test曾被查看的研究历史；本次同时改变batch/scheduler/EPF轮数/Urban抽样，不能把效果差异单独归因于batch，单seed std=N/A。

保留原199/199证据及其覆盖范围；本轮相关历史版本隔离94项、OneCycle11项和新增v2 81项，实际186/186 Passed，0 failure/error/skip。全程禁止计算钩子记录真实model construction/forward/backward/Adam/GPU initialization均0。旧测试方法/断言保留，无skip或容差放宽；新增实际M完整落盘审计及M closeout合成fixture。初次72项3failure、182项1error、合成文件语法/缩进及启动材料辅助脚本失败均保留，修正fixture/期望/准备接线后183/185和最终186证据分别保留。真实模板dry-run另发现source_states按数据集名筛选路径键而成为空记录，已改为冻结v3预期路径记录的确定性投影，再校验实际stat；数据文件/profile/numeric/预算均未改变，初始配置和CLI证据归档并增加回归。

AST/JSON/bash-n/restricted bundle SHA/git diff --check通过。修后真实模板dry-run exit0，preflight exit2；blocked只剩未review、未closure和未物化真实start-record三项，来源/数据/环境检查通过，模板五flags=false、closure_commit=null。本轮READY_TO_ARM_HANDOFF=false、READY_FOR_GPU_EXECUTION=false，无真实新许可、结果根、tmux、等待器或计算。工期仅工程估计，按同模型/域runtime、精确批次和实际wave重构，不用六步除6外推正式epoch：候选q通过的新正式约10–55小时，含必要probe和检查的规划区间约11–60小时；退q2约16–97小时，全串行约30–182小时。M batch128和M输出开销尚无新实测，代理估计保留较宽不确定性。旧v3剩余等待单独记录，不纳入新计算估计，也不作启用门禁。本轮未stage/commit/push，W/R原字节不变，N增量及完整P材料待ChatGPT实际字节审核。

#### baseline-type1-followup-v2：M新增ETTh2/ETTm1/ETTm2（2026-10-05，231项完整候选待实际字节审核）

用户在未commit、未物化许可、未启动的同一v2候选中新增ETTh2、ETTm1、ETTm2，不新增协议/worktree或第四环，不重新退休旧队列。扩展前147项送审材料及18份候选原字节已保留于外置P的reviewed-147-candidate；原186/186和历史失败账继续按原覆盖范围保留。W/unified-v3不编辑、不停止、不signal；原type1-v1退休事实和全部证据不变，生产R、作者仓库、环境与数据原文件不改。

新增参数来源只采用固定论文TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting，ICLR2024，arXiv 2405.14616v1 PDF第14页Table7（URL https://arxiv.org/pdf/2405.14616v1 ，PDF SHA256 a599b338e44af70d8e9c87be3c5417bde7864b2c92074e1346703f3e2b641e3d）。表中ETTh1/ETTh2/ETTm1/ETTm2均为train batch128、epochs10；本轮仅据此对齐新增ETT的这两个字段，eval_batch128及patience=None继承项目M合同，不声称来自Table7。initial_lr=1e-4、type1_horizon_scaled_v1、seed2024不改；不迁移论文LR、宽度、层数或重复次数。Weather继续10/None/batch128，明确没有对齐论文Table7的20轮。

三环现在精确为Urban28/560、EPF35/350、M168/1680，共231 fresh run/2590 max run-epochs。M包含ETTh1/Weather/Exchange后追加ETTh2/ETTm1/ETTm2，七模型固定顺序，每模型每数据集H96/192/336/720；保留原147项相对顺序和全部resolved scientific profile（逐项diff=0），Urban及EPF两份配置字节不变。新增84项以当前v2同模型同H的ETTh1 M profile为固定模板，保留各自结构、正则化、优化器、全通道路由、split/scaler政策、seed和训练设置，仅更新数据集/任务身份、实际采样/time-mark字段及派生窗口/批次/scheduler steps。此为固定跨数据集迁移，不声称复现各模型在新数据集的全部作者超参数，不按test选择模板。T96、pred_len=H、train/eval128、10/None、accumulation1、eval_drop_last=false；不加入ECL/J/N/S或额外Urban格、seed/候选/search。

三份本地CSV已与官方ETDataset锁定提交1d16c8f4f943005d613b5bc962e9eeb06058cf07的对象逐字节一致。ETTh2 SHA a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b，17420行、小时采样、固定endpoints8640/11520/14400；ETTm1 SHA 6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e，ETTm2 SHA db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd，均69680行、15分钟采样、固定endpoints34560/46080/57600。七通道顺序HUFL/HULL/MUFL/MULL/LUFL/LULL/OT核验一致；未使用的尾行保留。各自新source_admission绑定真实官方对象、本地SHA/stat、作者hour/minute reader与timeF证据，不复制ETTh1 data SHA/scaler或冒称已通过旧M5。新scaler仅fit各自train prefix，validation/test保留T96历史上下文和固定目标边界；准备只做原始checksum、schema/时间元数据及train/validation数值前缀，不执行正式test推理或用test标签调参。

ETTh2沿小时freq=h、原4维历史timeF；ETTm1/2显式freq=t、实际间隔15min、5维MinuteOfHour/HourOfDay/DayOfWeek/DayOfMonth/DayOfYear，与锁定作者time_features逐值一致，不把小时clock盲复制给分钟数据。N仅补充time-mark生成/形状门禁与adapter K检查，未改作者模型或model forward、loss、scaler、window切片和evaluate数学；原小时marks输出及原147项profile/metadata保持。新增模型接口验证使用非训练合成调用替身，不构造真实模型。

原有每个numeric scope diff=0；新增21个model×dataset条目显式继承同模型ETTh1 M规则，None/exact仍exact，full_float_state保持原阈值与覆盖，只更新dataset/id/适用范围，不把MS/ModernTCN EPF5e-4套给M。新增政策不是旧准入Passed，必须在本链真正probe时验证。必要probe为Urban7/28、EPF8/35、M42/168，共57组231代表，独立serial六步及实际四H组候选q<=4，逐组即时gate/finite/完整state/gradients/Adam/RNG/batch/scheduler/LR/betas与资源检查保留；只允许预登记resource降q，固定batch128若q1仍容纳不了则阻塞。probe尾部完整审计0、每环AUTO_AUDIT1，三环合计3；formal零完整probe回放、零逐task raw sidecar读取，manifest scan仍每环监督器生命周期1次。

上游等待仍严格为原unified-v3的203 MS＋84 M=287项，不变成168个M；新链完成必须包含28＋35＋168=231项，旧84项M边界不足以通过新COMPLETE。新来源/profile/policy/metadata/plan纳入config及许可/runtime/worker/guard绑定；旧147项模板不能授权扩展后的231项。用户未来一次arm即可等待上游完整技术完成并释放owned计算进程后自动三环，无中途人工许可，效果不影响后环派发。三环每项只用validation选best/早停、锁定后正式test一次；无法确认test已访问状态时停止该项自动恢复，不再次test。结果全部保留，不逐格择优、不自动替换论文主表、Urban仍仅28/168，result_review=pending。

独立budget helper复算（Adam/backward/forward）：原147项预算不变，新增84/840为166880/166880/227325；扩展后Urban28/560为1028440/1028440/1143128，EPF35/350为399000/399000/467845，M168/1680为274960/274960/356202，总231/2590为1702400/1702400/1967175。probe Urban nominal336/336/464 cap504/504/696；EPF nominal420/420/560 cap618/618/824；M nominal2016/2016/2688 cap3024/3024/4032；总nominal2772/2772/3712 cap4146/4146/5552。固定范围由本轮用户授权，预算只作运行保护及真实记账，不无限重试，旧消费不退款/转移/抵扣，additional_search=0。

本轮复验相关历史type1/OneCycle/三环M与新ETT范围，实际221/221 Passed，0 failure/error/skip；真实model construction/forward/backward/Adam/GPU initialization全部0，GPU未初始化。原147项数据合同测试绑定保留候选快照，保留原方法与断言；新增35项覆盖84格迁移、数据来源/reader/scaler/分钟marks、全通道接口、窗口尾batch、政策、新旧完成边界与许可、budget及科学数学AST。辅助来源获取的GitHub API403已保留，改用只读git ls-remote锁定提交及原始对象获取；来源脚本语法错误及一次原子patch上下文拒绝记录保留，未产生模型计算或科学尝试。没有删除断言、skip或容差放宽。

AST/JSON/bash-n/bundle/diff-check通过；真实不可执行模板dry-run exit0，preflight exit2，blocked仅为未实际字节审核/未closure/未物化start record三项；五授权flags仍false、closure_commit=null。本轮READY_TO_ARM_HANDOFF=false、READY_FOR_GPU_EXECUTION=false，没有新start-review、真实等待器/tmux、probe、formal/test或结果根。工期重新按231项实际固定波次及同模型/H的ETTh1代理、分钟数据实际train/val/test workload比计算，未以run数等比例或六步除6外推：候选并发新正式约10.1–78.7小时，降q2约15.1–131.7小时，串行约29.4–248.0小时；M的新增数据/batch128缺同任务实测，区间为较宽代理估计而非置信区间，另加未实测probe/完整性检查开销及旧v3剩余等待，不复用旧147项11–60小时为新工期。本轮只在N/P形成候选，未stage/commit/push，扩展后字节待ChatGPT审核，不宣称已审核或已同步。
