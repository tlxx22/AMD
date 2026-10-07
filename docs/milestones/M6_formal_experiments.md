# M6：第三章正式实验与定稿

**In Progress — N实际closure/旧r2执行版本为760b9dd7162d200c11b8836a7c9ece41822dfcf1。M_BASE 21/21技术准入、M84及混合来源基础287已完成；M_AMEND首个AMD–ETTh2 serial worker因M声明旧三域允许名单拒绝而退出，补做正式0/112、第三轮正式0/231。当前为新增ETT声明窄修及完成前缀采用候选，只继续剩余343项；统一133政策、五阶段科学profile和原预算保持，旧来源/失败/许可/结果保留，result_review仍pending。真实CPU构造/合成输出检查已完成；本轮未stage/commit/push、未生成可执行新许可、未arm/start或执行新GPU probe/正式训练，候选待ChatGPT实际字节审核。三轮合同见[§0总览](#m6-three-rounds)，最新增量见[新增ETT声明repair](#m6-amend-ett-identity-repair-v1)。**

原开篇“尚未运行任何正式训练或正式test评价”及各历史小节的“当前”均属于各自记录时点，不代表当前实际进度。

<a id="m6-three-rounds"></a>
## 0. 三轮正式实验组织与最终结果采用规则

### 0.1 名称、用途及状态含义

M6按三个主要正式实验批次组织讨论、结果表和复现材料。

| 名称 | 技术范围 | 结果组织 |
|---|---|---|
| 第一轮正式实验：初始全量方案 | 最初正式实验及归属于该方案的补充覆盖、已接受修订 | 原有效结果加已接受补做；TimeMixer指定16格采用fixedlr-v2修订来源 |
| 第二轮正式实验：v3统一OneCycle方案 | 原baseline-unified96-onecycle001-v3，加baseline-unified96-onecycle001-v3-amend1 | 修订完成后的默认有效结果为371格 |
| 第三轮正式实验：衰减学习率续接方案 | baseline-type1-followup-v3，承接原type1-followup-v2的231项并修改Weather训练长度 | 231项独立fresh正式实验 |

“三轮”是主要实验批次的报告名称，不是只启动过三次、只访问过三次test，也不等同于工程目录的v1/v2/v3。补做、暂停、失败、修订和退休保留真实身份与记录；第二轮amend1属于第二轮，不另称第四轮。

批准的最终结果采用规则与结果已经全部产生分开。未完成、未审核或没有合法来源的格子不能因写入总览而变成完成；技术完成、结果记录审核和效果达标分别报告。此前实现、文档及两次窄修已closure，本轮从N实际`1134611cd52e4cedb7418d8a9f8a7f6e76512617`继续开发。原v3的203项MS已训练/test完成但封存失败，原M84未开始；旧type1-followup-v3等待器因上游failure退出，没有补做或第三轮计算。旧type1-followup-v2退休记录保持。本轮修后候选尚未审核/closure/启动。

<a id="m6-round-one"></a>
### 0.2 第一轮：范围及最终结果来源

第一轮最初登记495项，后经EPF四市场及TimeMixer覆盖补做扩展到552个计划任务身份。这些数字描述历史计划范围，不自动等于全部完成或全部被接受；各任务实际完成、修订和退休状态依据既有审核记录。

第一轮最终结果必须包含已注册的PJM、NP、BE、FR、DE五个EPF市场。七baseline与第一轮已注册的J分别保留自身任务身份，八个主表模型的完整EPF目标范围为40格。不能因后两轮只跑七baseline就从第一轮历史中删掉J，也不为N/S增加EPF任务。

原PJM结果与`E/supplements/epf4-timemixer-v1`补充结果按固定来源合并。57项补做主要用于补齐覆盖，不是按指标替换已有结果。工程完整性与结果审核状态分别依据收据，不能仅见complete.json就宣称全部review Passed；缺失列出，本轮不自动补跑。本次只读核验40个EPF result/manifest全部存在且任务身份相符，无缺项；其中20项可绑定明确reviewed收据（原AMD/J/PatchTST PJM3项及41-run补做中的EPF17项），其余DLinear/iTransformer/ModernTCN/TimeXer各五市场共20项未在本次定位到单独结果review收据。本条不将这些20项追认为review Passed，也不据此否定可能存在的其他历史审核；逐项来源检查见下方记录。

TimeMixer原四标准域16项最终采用`timemixer-fixedlr-v2/attempt2`：

| 数据集 | 原固定LR | 修订固定LR | 正式修订范围 |
|---|---:|---:|---|
| ETTh1 | 0.01 | 0.001 | 四H |
| Weather | 0.01 | 0.001 | 四H |
| ECL | 0.01 | 0.001 | 四H |
| Exchange | 0.0003 | 0.0003 | 四H，随修订独立fresh重跑 |

四H为96/192/336/720，原任务为MS，不能与后两轮M全通道指标混为同一任务。仅前三个域12个profile改变LR；Exchange不是从0.01改为0.001。TimeMixer的UrbanEV和EPF补做不属于本次固定LR替换范围。

这16项按事先确定的修订来源采用，不按新旧test逐格择优。修订发生在观察过旧正式结果后，是项目批准的事后协议调整，不能描述为首次盲测，也不能因旧目录在新训练前退休就称整个调整未受旧test信息影响。替换理由按当时记录区分用户改变协议、异常现象和已证实实现错误；无对应证据时不写“旧代码已证实有bug”“ECL根因已确定为LR”或“固定0.001全面优于0.01”。本修订不是TimeMixer作者OneCycle的完整复现。

#### TimeMixer修订前后表现与证据状态

修订后读取已审核`timemixer-fixedlr-v2-completed-review-v1/reviewed-result-receipts.json`；16项result及manifest当前SHA与收据绑定值一致，指标直接来自合法result，不重新test。修订前检查了删除前接受索引、退役索引/摘要、删除计划/回执和既有导出/早期比较包，未定位到可核验的旧TimeMixer MSE/MAE；删除前摘要保留了身份与退役事实，不能据此补造旧指标。旧值及变化百分比均记N/A，代表证据未找到，不代表旧训练未发生或旧值为0。

百分比定义为`100×(新值/旧值−1)`，负数为误差下降；旧值为0时不计算普通百分比。下表新值显示到小数点后9位，外置来源记录保留原JSON精度；R16为上述reviewed收据，按dataset/H精确定位，每行独立result/manifest路径及SHA见来源检查记录。

| 数据集 | H | 旧MSE | 新MSE | MSE变化% | 旧MAE | 新MAE | MAE变化% | 来源 |
|---|---:|---|---:|---|---|---:|---|---|
| ETTh1 | 96 | N/A | 0.087243289 | N/A | N/A | 0.229839473 | N/A | R16：ETTh1/H96，旧值未核验 |
| ETTh1 | 192 | N/A | 0.112136047 | N/A | N/A | 0.266859509 | N/A | R16：ETTh1/H192，旧值未核验 |
| ETTh1 | 336 | N/A | 0.100225454 | N/A | N/A | 0.250576811 | N/A | R16：ETTh1/H336，旧值未核验 |
| ETTh1 | 720 | N/A | 0.121722273 | N/A | N/A | 0.280806657 | N/A | R16：ETTh1/H720，旧值未核验 |
| Weather | 96 | N/A | 0.091616175 | N/A | N/A | 0.222349706 | N/A | R16：Weather/H96，旧值未核验 |
| Weather | 192 | N/A | 0.148864001 | N/A | N/A | 0.285876258 | N/A | R16：Weather/H192，旧值未核验 |
| Weather | 336 | N/A | 0.212876268 | N/A | N/A | 0.339612209 | N/A | R16：Weather/H336，旧值未核验 |
| Weather | 720 | N/A | 0.365029224 | N/A | N/A | 0.456327601 | N/A | R16：Weather/H720，旧值未核验 |
| ECL | 96 | N/A | 0.239910976 | N/A | N/A | 0.344974410 | N/A | R16：ECL/H96，旧值未核验 |
| ECL | 192 | N/A | 0.316621938 | N/A | N/A | 0.402006865 | N/A | R16：ECL/H192，旧值未核验 |
| ECL | 336 | N/A | 0.385504480 | N/A | N/A | 0.435793028 | N/A | R16：ECL/H336，旧值未核验 |
| ECL | 720 | N/A | 0.566518442 | N/A | N/A | 0.540629301 | N/A | R16：ECL/H720，旧值未核验 |
| Exchange | 96 | N/A | 0.101263750 | N/A | N/A | 0.240706488 | N/A | R16：Exchange/H96，旧值未核验 |
| Exchange | 192 | N/A | 0.309439363 | N/A | N/A | 0.423410461 | N/A | R16：Exchange/H192，旧值未核验 |
| Exchange | 336 | N/A | 0.484237818 | N/A | N/A | 0.533336686 | N/A | R16：Exchange/H336，旧值未核验 |
| Exchange | 720 | N/A | 0.923061728 | N/A | N/A | 0.749124548 | N/A | R16：Exchange/H720，旧值未核验 |

| 数据集 | 四H旧MSE均值 | 四H新MSE均值 | MSE变化% | 四H旧MAE均值 | 四H新MAE均值 | MAE变化% |
|---|---|---:|---|---|---:|---|
| ETTh1 | N/A | 0.105331766 | N/A | N/A | 0.257020613 | N/A |
| Weather | N/A | 0.204596417 | N/A | N/A | 0.326041444 | N/A |
| ECL | N/A | 0.377138959 | N/A | N/A | 0.430850901 | N/A |
| Exchange | N/A | 0.454500665 | N/A | N/A | 0.486644546 | N/A |

只计算各数据集内部四H算术均值，不跨数据集平均原始MSE/MAE。该review包的`comparison.json`比较修订后TimeMixer与AMD/J/PatchTST，不能作为TimeMixer修订前后对照；AMD/J的ECL诊断不作为TimeMixer故障根因。未完成并退休的native-time-mark replacement不零散拼入第一轮最终矩阵；后续协议采用其接口修订，不意味着历史replacement结果完整可用。

来源与核验记录：

- [TimeMixer16项reviewed收据](../../../amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/timemixer-fixedlr-v2-completed-review-v1/reviewed-result-receipts.json)。
- [41项补做reviewed收据](../../../amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/catchup41-completed-review-v1/reviewed-result-receipts.json)及[原已接受结果收据](../../../amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/boundary-index-v1/accepted-receipts.json)。
- [本次TimeMixer指标来源检查](../../../amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/baseline-unified96-onecycle001-v3-amend1/documentation-review-v1/timemixer-metrics-source-check.json)与[第一轮40格EPF来源检查](../../../amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/baseline-unified96-onecycle001-v3-amend1/documentation-review-v1/first-round-epf-source-check.json)。

<a id="m6-round-two"></a>
### 0.3 第二轮：v3修订后完整范围

第二轮默认采用统一OneCycle体系。原v3登记287项，amend1追加84项ETT及28项Weather20修订训练。

| 范围 | 数据集/任务 | L | 最大epoch/patience | 有效格数 |
|---|---|---:|---|---:|
| UrbanEV MS | F4六fold×H3/6/9/12×七模型 | 12 | 20/5 | 168 |
| EPF MS | 五市场各H24×七模型 | 96 | 20/5 | 35 |
| 原有M | ETTh1、Exchange各四H×七模型 | 96 | 10/None | 56 |
| 新增M | ETTh2、ETTm1、ETTm2各四H×七模型 | 96 | 10/None | 84 |
| Weather M修订 | Weather四H×七模型 | 96 | 20/None | 28 |
| 合计 | MS203＋M168 | — | — | 371 |

MS任务batch/eval_batch继续采用原冻结effective profile。本次用户明确批准原未执行M84全部train/eval batch128，57项训练batch改变、27项原本128；独立M128配置重算批次数、OneCycle steps_per_epoch及执行计量，不改原绑定配置。新增三个ETT84项已是128/128，科学配置保持；Weather20继承本次M128的同模型/H Weather配置，再只增加epochs/scheduler.epochs至20。因此第二轮修订后所有M域train/eval batch128；模型结构、数据与其他冻结字段不随统一batch改变。

新增ETT从同模型同H的v3 ETTh1-M profile迁移，保留各模型结构，采用各自独立数据、split、train-only scaler及适用历史marks。ETTh2为小时，ETTm1/ETTm2为15分钟及适用的5维历史timeF；marks按各模型原生接口提供，不新增七模型统一业务输入通道。训练长度/batch对齐限于已绑定TimeMixer论文Table7与指定字段，eval_batch128和patience=None属于项目合同；OneCycle采用绑定作者实现。已知结构差异保留：TimeMixer迁移ETTm2的d_model=16，Table7为32，不为追齐该表修改冻结结构，不声称完整作者配置复现。

第二轮OneCycle：max_lr=0.01、pct_start=0.2、cos，cycle_momentum=true，base/max_momentum=0.85/0.95，div_factor=25，final_div_factor=10000，three_phase=false；其余初始化和推进参数按冻结配置。max_lr不是固定LR；调度初始化LR为0.01/25=0.0004。每次成功optimizer update后推进scheduler，validation/test不推进。

Weather20相对本次M128 Weather10只改training.epochs与training.scheduler.epochs；相对先前amend1候选同步新增batch/eval_batch128及派生steps_per_epoch变化。必须fresh按20轮长度执行完整OneCycle，不从Weather10 checkpoint续接或拼入旧轨迹。batch改变会改变每轮更新次数及按step展开的LR/beta1轨迹，是用户批准的科学修订，不是修复旧batch的bug。

### 0.4 第二轮最终结果采用规则

原UrbanEV168＋原EPF35＋本次M128的ETTh1/Exchange56＋新Weather20/batch128的28＋新增ETT84＝371有效格。

新Weather28项全部技术完成后统一切换来源，即使部分变差，也不在10/20轮间逐格挑选。完成之前显示修订Pending，不把10轮值标为20轮。原MS203及原failure/未封存现场保持；恢复版基础287明确组合旧MS203与fresh新M128的84项，不扩张原失败controller的完成证明。先完成的新M128 Weather10保留原位置和引用，最终不进入修订后的默认主表。日常只采用一个第二轮修订入口，引用保留真实路径、config/训练commit、封存执行commit、revision、替代关系及时间，不移动改写旧artifact或伪造原执行。

371是有效格数；287＋112＝399是含Weather替换训练的计划正式运行次数，实际是否执行以完成记录为准。probe和失败attempt另计，不混入正式格数，也不要求用户逐次审批底层step。

<a id="m6-round-three"></a>
### 0.5 第三轮：最新231项矩阵

第三轮采用baseline-type1-followup-v3；七模型顺序为AMD→DLinear→PatchTST→iTransformer→TimeMixer→ModernTCN→TimeXer，seed2024，全部fresh，additional_search=0。不含J/N/S/ECL，不继承第二轮checkpoint。

| 环节 | 数据集/任务 | L | train/eval batch | 最大epoch | patience | runs |
|---|---|---:|---|---:|---|---:|
| UrbanEV精简MS | F4 fold1/2，各H3/H12 | 12 | 128/128 | 20 | 5 | 28 |
| EPF MS | PJM/NP/BE/FR/DE，各H24 | 168 | 32/32 | 10 | 3 | 35 |
| 其他M | ETTh1/ETTh2/ETTm1/ETTm2/Exchange，各四H | 96 | 128/128 | 10 | None | 140 |
| Weather M | 四H | 96 | 128/128 | 20 | None | 28 |
| 合计 | — | — | — | — | — | 231 |

最大2870 run-epochs＝Urban560＋EPF350＋其他M1400＋Weather560。UrbanEV的H是单点标签偏移，model pred_len=1；EPF pred_len=24；M pred_len=H，全通道监督及评价。patience=None是不按早停规则提前结束，不是取消validation或best选择。

UrbanEV只覆盖28/168，不称完整六折四H。第二轮修订包含三个新增ETT，不能继续概括“第三轮新增ETT没有第二轮对应任务”；补做完成后这84项有对应结果，完成前不伪造对照。

### 0.6 第三轮学习率与来源边界

Adam、initial_lr=1e-4，基础betas/eps/weight_decay按各冻结profile，不使用OneCycle动态beta1，accumulation=1。

`type1_horizon_scaled_v1`：`lr(e,E)=1e-4×0.5**(max(e−2,0)×c(E))`；E<=10时c(E)=1，E>10时c(E)=8/(E−2)。

| 最大长度E | 适用任务 | epoch1/2 | epoch3 | epoch10 | epoch20 |
|---|---|---:|---:|---:|---:|
| 10 | EPF及其他M | 1e-4 | 5e-5 | 3.90625e-7 | 不适用 |
| 20 | UrbanEV及Weather | 1e-4 | 约7.348672461378e-5 | 约8.504937501090e-6 | 3.90625e-7 |

E在训练前固定，不随早停、暂停、恢复或validation重新缩放。10轮时序借鉴锁定TimeXer官方type1；20轮拉长是项目修改，不是作者原配方。Weather只继承20轮调度，不继承UrbanEV的patience5；第二轮Weather20仍是OneCycle，与本轮衰减方式区分。

### 0.7 自动执行顺序与评价边界

只读验证旧MS203、精确封存失败身份与owned进程释放→恢复MS封存→M128的84项必要probe/AUTO_AUDIT/formal/test→旧MS203＋新M12884的基础287来源封存→第二轮amend1的112项必要准入、训练/test及完整性检查→371格来源封存→第三轮UrbanEV→EPF→M→总链技术完成，结果审核另行进行。不得继续等待已失败退出的旧owner生成原complete，不重派MS203，不等待监督器自身退出。

文档审核、最终closure和启动准备完成后，由用户一次启动恢复总监督器。导入封存不占训练组锁、不初始化GPU；实际计算组持锁并等待真实子进程退出释放资源后衔接，不因效果好坏改任务或等待中间人工许可。M128及第三轮沿各自冻结模型/任务顺序；补做每模型ETTh2四H→ETTm1四H→ETTm2四H→Weather四H。只允许登记的重复键故障在完整来源证明下导入，不提供通用ignore_failure。

M128为21组/84代表，补做28组/112代表，第三轮57组/231代表，共106组427代表，不计作额外427个正式结果。并发最多q4，仅在真实batch/结构/调度条件验证后用；仅明确resource失败按预登记降q，单任务batch128不适合时停止，不改batch、精度、结构、accumulation或阈值。numeric/finite/identity/data/scheduler/guard失败停止后续派发。逐组即时gate保留，完整审计仅由各阶段AUTO_AUDIT执行一次，五阶段合计5次；formal轻量消费凭据，监督器生命周期manifest扫描每阶段1次，不逐任务回放完整probe，不重新回放原MS数值轨迹。

每项正式训练仅validation选best及执行既定早停，锁定后按本轮授权最终test一次；历史test访问如实保留，某run只test一次不代表整个研究从未多次观察同一测试集。test状态不明停止自动恢复，不重复test试探。

<a id="m6-result-expression"></a>
### 0.8 统一结果表达与复现定位

分别采用第一轮有效来源、第二轮修订来源及第三轮来源，不跨轮按test逐格择优拼一张“最优表”。配对核对模型、dataset、MS/M、fold、H、input variant、target/输出范围及seed；第一轮MS与后两轮M不是同监督同评价任务。多个训练/输入条件同时变化，不把差异单独归因于LR或batch。

UrbanEV保留逐fold/H和明确域内均值，EPF逐市场报告；不跨数据集平均原始MSE/MAE。单seed2024，std=N/A，多seed稳定性未评价。论文正文选哪些域与执行范围分开，不因此删结果或取消任务。

现有复现根：

```text
E=/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-formal-launch-dhozikhu
P=/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1
第二轮原协议：E/baseline-unified96-onecycle001-v3，P/baseline-unified96-onecycle001-v3
第二轮补做：E/baseline-unified96-onecycle001-v3-amend1，P/baseline-unified96-onecycle001-v3-amend1
第三轮当前：E/baseline-type1-followup-v3，P/baseline-type1-followup-v3
```

P中的两个新准备包已存在，E中对应新结果根尚未创建。路径是冻结规划，不表示已经运行；不重命名历史分支、协议或artifact。第二轮默认主结果命令为N的`start_type1_followup.sh second-round`，第三轮为`third-round`，未封存时返回Pending；本轮不为说明命令创建新结果或许可。

预算详情见[已绑定budget.json](../../../amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/baseline-unified96-onecycle001-v3-amend1/budget.json)，来源/字段差异见[来源证明](../../../amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/baseline-unified96-onecycle001-v3-amend1/source-alignment-proof.json)和[逐项profile diff](../../../amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/baseline-unified96-onecycle001-v3-amend1/effective-profile-diff.json)，实现及验收见[既有实现小节](#m6-amend-implementation)与[84项验收证据](../../../amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/baseline-unified96-onecycle001-v3-amend1/acceptance.json)。本轮只整理文档，不递归把文档自身SHA或不存在的新commit写入正文。

---

**以下§1及其后小节保留各自记录时点；其中“当前”“最新”“待审核”“未提交”等描述只对应当时，不能与上方真正当前总览竞争。旧计划、失败、删除授权、退休、执行日期和test历史不倒写。**

## 1. 授权、冻结身份与来源

本次用户明确授权J最终冻结、结束M5、进入M6，并要求每个实验的启动命令。最终模型族EL-AMD，结构ID `el-amd-s2-thls-v1`，代码与run逻辑身份仍为J=AMD＋Sonnet/MVCA S2 target residual＋THLS。冻结不等于正式效果Passed；M4原gate失败及H192风险完整保留。起点Git为`2f6a5d8cc9faf9e33373a34471be8fd78fe9dec0`、三端0/0 clean。M5封存文档只在本次结束登记后提交，不后续追加。

原技术probe：`../amd-execution-evidence/m5/m5-rss-kernel-t9a2njbt/probe/complete.json`，SHA `99c3a6c0d19e36a52189e2c499e3304aa7d89448cc6f5c90d3d48183b1cd5a6d`。54/54 Passed；q4=44组、q2=3组、q1=7组。既有报告不改写。启动接线改变不涉及model/build/update/evaluate/init_training/save/restore/formal_worker计算；以父报告、495个相同profile及AST等价证明生成明确标注runtime carry-forward的准入文件，不冒充重做probe。

## 2. 最小正式启动接线与验收

核对发现旧正式入口会为同一波每个run分别启动资源监测进程；现改为每个固定波次统一调用已有run_configs，一次监测全部自有worker，保留已验证PID/退出处理和余量线。正式任务的monitor按流式保留完整日志及在线峰值/间隔统计，仅保留一个实时sample，避免长训练控制器无界累积sample对象。任一正式worker非零退出即停止本波自有进程和后续调度；不影响无关进程。GPU互斥锁取得前不创建模型run输出；失败/staging保留，不能原命令覆盖重启。

新增M6前缀绑定purpose仅允许各文件既定train/validation端点；不构造或评价当前run的test，不运行模型/optimizer。29个dataset/fold/input前缀组合实际读取并检查，各组加载一次后按相同Windows数学派生H的metadata，绑定全部495个任务。UrbanEV按六个rolling fold各自train-only scaler和split-local标签；跨fold同一timestamp的角色按既定rolling合同保留，不将后折训练前缀误称所有折都未见的时间段。前后source stat相同，audit deny=0，GPU/模型/Adam/前向/反向=0。

16个精确CPU方法首验通过；静态收口检查补充完整test元素数校验与正式日志路径后，同16项复验通过，无失败/skip/方法互调。验证了单个波次协调器、失败阻止下一波、自有子进程停止、固定任务覆盖、data-binding/身份及模型计算不变；没有重新运行54组probe，也没有GPU计算。现有模型保存恢复工程证据按不变函数复用，本轮不声称执行了新的模型动态回归。

## 3. 正式矩阵与固定执行规则

495 fresh runs / 最多5340 run-epochs；formal seed=[2024]、std=N/A、随机初始化稳定性Not evaluated。所有模型同域T统一；AMD/J/N/S原设置不变，外部baseline仅按冻结来源层迁移batch/eval batch/LR；禁止test后调整参数/结构/seed或追加搜索。TiDE继续Deferred；TimeMixer仅四个标准域，不临时增加UrbanEV/PJM。

AMD/J各自包含六域主表和UrbanEV F1–F4输入消融，F4 run直接被主表/消融复用；N/S仅UrbanEV F4六fold×四H。不存在F0、J/F0或PJM TargetOnly。所有组/配置须在首个正式test出现前锁定。

| 模型入口 | fresh runs | 最多run-epochs | 已验证调度 |
|---|---:|---:|---|
| AMD | 113 | 1180 | PJM单路，其余四路 |
| J | 113 | 1180 | PJM单路，其余四路 |
| DLinear | 41 | 460 | PJM单路，其余四路 |
| PatchTST | 41 | 460 | ECL两路，PJM单路，其余四路 |
| iTransformer | 41 | 460 | PJM单路，其余四路 |
| TimeMixer | 16 | 200 | ECL两路，其余标准域四路 |
| ModernTCN | 41 | 460 | ECL两路，PJM单路，其余四路 |
| TimeXer | 41 | 460 | PJM单路，其余四路 |
| N | 24 | 240 | UrbanEV四路 |
| S | 24 | 240 | UrbanEV四路 |

原独立启动规则：用户每次只启动一个模型；该模型内部按dataset→input_variant→固定H/fold波次自动完成，不跨组补位，不自动进入下一模型。该规则对§5.14本次明确授权的剩余六组由一次总启动、自动技术交接取代，不倒写此前历史。已验证并发无需再次申请；硬件/shape/batch/线程/workers或资源条件改变时不外推。长训练异常保留全部文件，先只读审计后才能给合法resume，不自动fresh重跑。正式test只在每run训练完成后对validation-selected best执行一次；不得逐epoch看test或用test选模型。

## 4. 版本、许可与启动材料

证据根：`/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-formal-launch-dhozikhu`。

`formal-approval.json`由实际closure成功后生成，绑定实际commit、code/config/环境/硬件、冻结ID、全部模型、495个data metadata摘要和source stat；`runtime-probe-admission.json`明确继承原54组报告及独立不变性证据，不覆盖原报告。无负载preflight逐十模型必须通过后才给用户启动命令。本轮不代启任何正式模型。

启动方式沿`bash scripts/ch3/start_model.sh --model MODEL start --approval "$E/formal-approval.json" --probe-report "$E/runtime-probe-admission.json"`。`status/logs/complete/safe-stop`使用同一模型入口；日志在每run目录的worker.log、history.jsonl和每波memory.jsonl，模型controller日志位于本证据根。全部具体命令保存在本证据根`launch-commands.md`。

在正式任务运行期间不修改仓库/已锁定配置，不升级依赖或作者仓库；结果保存在外置run目录。组间先做只读完整性审计，不为状态文字递归提交导致旧许可失效。`complete=true`仍需核对run数、best/last/history/metrics及预算/finite，不能仅凭tmux消失判成功。

## 5. 2026-09-27 EPF四市场及TimeMixer新增覆盖：外置准备，未部署

### 5.1 授权、身份与保护

这是用户明确批准的M6事后覆盖扩展，不是M5重筛选，不由既有结果择优调整参数。父commit `033913ff00f3d09a0ed1eff26baf2352ffb27bc4`；父protocol `22a237321f8333d0eb32d9c513a339541c95ccd40c105e7c2912004125f18b78`；批次`epf4-timemixer-v1`，注册时间与57个ID见本包`plans/registration.json`。§1–4保留原准备时点。原495个profile逐项摘要一致，模型源码及训练、评价、初始化、保存恢复、probe计算函数AST不变。原TimeMixer16-run队列不调整、不重启、不读正在生成的checkpoint/test结果；缺少已审核安全结束边界，因此仅外置候选/patch，不修改生产Git工作树。ECL机制调查不重做。

### 5.2 任务、固定参数及队列

新增32个四市场主表run（AMD/J/DLinear/PatchTST/iTransformer/TimeMixer/ModernTCN/TimeXer各4）＋TimeMixer/PJM 1＋TimeMixer/UrbanEV F4 24＝57，增加最多900 run-epochs，总552/6240。原495结果保留。N/S仍仅UrbanEV F4，TiDE Deferred，无F0、TargetOnly、新seed/H/fit。新批次不借原许可。

新增EPF统一T168/H24/20epoch/patience5，Adam/固定LR/原精度及数据评价合同保持。AMD/J、DLinear、PatchTST、iTransformer、ModernTCN为B128/eval128/LR5e-5，来自各自冻结PJM项目迁移；无证据称为该域原论文参数。TimeXer自身锁定脚本明确NP/DE B4、BE/FR B16，eval同batch；LR1e-4来自同commit `run.py:94`隐式默认，原PJM结构保留，不引入脚本其他结构。出处/commit/脚本SHA/行号见config覆盖层及`plans/parameter-source-search.json`。

TimeMixer新增EPF五域为B128/eval128/LR1e-4；新增UrbanEV为B128/eval128/LR1e-3、T12/pred_len1、10epoch无早停。两者d64/ff128/e2、avg降采样window2、CI0、dropout0.1、use_norm1、不用未来时间特征；EPF down_layers3/moving_avg25，UrbanEV down_layers2/moving_avg3。其他实际构造参数全部展开在完整profile，都是用户本次固定配置，不是搜索。原四标准域不变。原生CI0保持enc_in/dec_in/c_out=C、全通道denorm后选目标，UrbanEV275节点仍走原时间接口，变量维为F4十一字段。

补做入口41任务按AMD→J→PatchTST→TimeMixer；各模型NP→BE→FR→DE，TimeMixer再PJM→UrbanEV F4六fold四H。其余四模型每组尾各追加四市场，合计16，与41互斥。EPF q1；TimeMixer/UrbanEV并发尚未验证，不能填q4 Passed。固定波次与失败停止复用原执行器；不跨数据集补位。新增结果根为原E内`supplements/epf4-timemixer-v1/formal-<MODEL>/<run_id>/`，新controllers独立，旧路径与许可不动。同一index入口关联552个身份，重复身份或未审核success拒绝；本轮未读原TimeMixer结果，注册索引未核验状态不表示否定历史完成审计。

### 5.3 数据来源、边界及实际前缀

四市场已存在于TimeXer固定commit `76011909357972bd55a27adba2e1be994d81b327`的`dataset/EPF/{NP,BE,FR,DE}.csv`。本轮从固定Git blob导出至独立版本目录，并与现有文件逐字节比较；没有重复下载或覆盖数据。各自记录边界独立核验得52416条，端点36691/41933/52416，不因PJM相同而猜填。原始SHA和Git blob见`data-sources.json`。

NP列为Grid load forecast/Wind power forecast/OT；BE/FR为Generation forecast/System load forecast/OT；DE为Wind power forecast/Ampirion zonal load forecast/OT。原header空格及拼写保留；目标OT index2、aux[0,1]。每域各解析一次41933条前缀、仅36691条拟合scaler，首batch与尾窗通过；train36500窗、validation5219窗。test10460窗仅端点算术，未解析test数值。B128 train285批/丢20、validation尾99；TimeXer B4/B16的步数和尾批由profile独立计算。新增最大optimizer步数1664430，与新增900 run-epochs分开记账，不推导运行工期。旧PJM/UrbanEV metadata按相同输入/fold/H从原许可继承，不重复真实前缀读取。

直接文件身份不能消除来源解释差异：原EPF文献BE使用法国预测量，而TimeXer描述存在地理不一致；FR年份叙述不同；DE原文聚合wind+solar与文件Wind标签不完全同义。本项目固定文件/列位，不换真实负荷或猜转换。d−1可得性只标论文级来源假设，逐条vintage及完整转换链未核验。TimeXer固定commit未找到根LICENSE，Zenodo记录为other-open，不伪称MIT或CC BY。参考逐页材料/发布记录见reference；业务解释和再分发许可不扩张。原始文件完整字节已用于身份/记录边界检查，承认包含test区间字节；数值解析仅前缀。

### 5.4 验收、失败成本与探针停止点

18项精确CPU首验全部通过，其中TimeMixer两方法实际5次CPU前向（UrbanEV四H与EPF一个形状），0Adam/反向/GPU。随后身份收紧：扩展路径仍检查既有RSS/数值政策，receipt不许覆盖run字段/路径，缺许可时也显式报告原活动controller；6项受影响/新增CPU方法通过，无模型复验。子case及源码绑定分别见`cpu-result.json`、`cpu-hardening-result.json`，不拼成同一首验版本。完整scope、5次前向预算及不变路径说明见plans。

准备脚本首次使用已有字节扫描器错误返回键失败；一次机械修复后数据比较完成，但因固定仓库不存在LICENSE结束失败。两日志保留。随后独立只读来源盘点如实登记无许可证文件并封存已取得CSV，无下载/模型重试，不能把失败写成初次全程Passed。

新增34组/37代表；其中26个EPF q1计算profile只在dataset/字段名称不同，可提交原自身PJM资源证据继承审查；数据身份仍单独核验。另8组/11代表须新测：TimeXer NP/DE、TimeMixer五EPF及UrbanEV。继承获接受后核心90Adam/120前向/90反向；全部重测名义246/328/246。UrbanEV必要q2/q1各至多24Adam，机械复验至多24Adam，均受本包总360Adam/360反向/512完整前向约束；封存前计数器复验后累计计入10CPU前向。全名义＋两级回退＋局部机械＋CPU为318Adam/434前向/318反向，RSS长窗不得超总额。新域不继承旧TimeMixer其他域的数值例外，不新增宽松容差。

GPU探针未执行、未发新并发许可。原队列安全结束审计未取得；目前GPU部署/实际scope-bound探针入口联动仍Blocked，不能调用旧start_probe.sh当成新范围。`plans/probe-coverage.json`仅是精确准备清单。候选补做入口经shell语法、dry-run、缺许可拒绝及tmux自有标准库占位生命周期检查；这不等于正式启动Ready。完整运行命令、源SHA、候选patch、原现场保护与剩余项见本包`report.md`和`commands.sh`，正式start必须经部署审核、统一closure、新许可与对应资源准入后由用户执行。

### 5.5 封存前计量自检失败：不追加复验

首验驱动完整前向counter错误记录0，与两模型方法固定4＋1次target_prediction调用不符。一次外置驱动修复尝试仅更换类识别为源码路径，复验两方法断言通过，但结束断言counter=5失败（进程exit1，原result中success只代表unittest，不代表驱动整体成功）。静态定位当前PyTorch `nn/modules/module.py:1575` 的`__call__ = _call_impl`类定义别名；仅替换`_call_impl`不会拦截`__call__`，之前模块名不匹配假设不成立。没有继续第二次机械复验，没有改torch或生产模型。

实际累计26方法断言通过/1次计量驱动自检失败，完整模型调用按固定测试源码与两轮成功方法保守记10次（不是已通过的运行时计数器读数），Adam/反向/GPU仍0。旧结果及失败日志不覆盖，`counter-incident.json`和`execution-ledger-final.json`明确订正。剩硬上限360Adam/360反向/502前向。本包整体工程验收Not passed；新增数据/任务/形状检查保持各自证据边界。待审最小修复建议是正确绑定实际`Module.__call__`或原生forward并先验计量，需后续授权复验；本轮停止所有动态负载。

### 5.6 用户新授权：计量修复复验与隔离接线收尾（生产未切换）

沿用同一包，新执行记录`meter-closeout-v1/`，旧report/index/patch及候选修前字节保存于该记录的before目录。§5.4–5.5所述计量失败和当时probe入口未接完是历史快照；本节为最新外置候选状态。原495/新增57、552/6240及所有科学profile不变。

正式Python实际torch源码确认`Module.__call__`为类定义时的`_call_impl`别名。本次只在外置进程替换实际`Module.__call__`，按已登记根实例计数，内部子模块不重复计；finally恢复。微型方法3次进入（2完成、1预期entered异常）、第4次在额外计算前拒绝；之后原TimeMixer两方法5次完整调用均完成。精确3 ID、调用序列和源码SHA在fixture前登记。驱动与unittest均通过，exit0、墙钟2.586秒、RSS采样峰值336244736字节、torch/interop各1线程；单进程nice10，CUDA不可见。原10次不退款，本轮8次计入总18，剩494/512完整前向，Adam/反向/GPU仍0。计量当前阻塞解除；旧错误过滤器及两轮旧成本不改写。

新增`utils/ch3_extension_probe.py`、`m6_probe_entry.py`及`start_extension_probe.sh`，复用原make_config/run_configs、六步probe_worker、GPU互斥锁、NVML/退出监测、RSS与数值比较。候选m5_formal_entry仅为新scope配置单独session/fixture/STOP及前置ID边界；旧scope原路径保持，旧probe总入口对扩展配置明确拒绝。原54组、新34组、26待审继承和8待新测精确关联；每worker6/8/6及全包360/512/360封顶，预扣准备18前向。未来失败保留实际账与保守预留，无自动退款/重试；新域不取得旧TimeMixer数值例外。模板reviewed/execution_permitted均false、commit/hardware为空，不是许可。无模型13精确方法首验通过；外置candidate不能冒充clean生产，scope错误/旧许可/未完成安全审计均在负载前拒绝。没有GPU执行或有效资源报告，也没有后台等待任务。

同一index入口已接收据文件，`commands.sh index [receipt路径]`实际可用，默认指向本轮有来源绑定的预览收据。只读三处获准审核目录的必要JSON，逐份对固定evidence-index摘要；不直接读取正式result或checkpoint，不接触TimeMixer正在生成的结果。AMD113/PatchTST41从缓存checkpoint-plan JSON取得实际逐run身份，与审核run-audit及J复核来源摘要交叉一致；J113的完成/指标/result及manifest摘要保留，但获准摘要未序列化逐run实际profile_sha/data_sha/commit/protocol_sha，字段留空，expected值与实际值分开。索引为154 success、113 accepted-complete-binding-incomplete、228 unverified、57 not-run；267个已接受完成事实没有否定，未运行和缺审核绑定的任务不填假成功。需补的唯一数据是J113份已核验manifest的上述四字段摘录，不需要再评价test或读权重，本轮未扩大读取。

当前停止点：计量修复与无模型接线Passed；GPU执行、生产部署、closure和正式补做仍未执行。原TimeMixer仍优先，只读controller身份/完成标志；即使结束也不自动审计/切换。原生产文档、AGENTS、Closed文档、源仓库、环境和原许可/产物零修改。详细差异、操作命令、前后SHA、机器账和当前索引见同包report及完整evidence-index；新字节仍待ChatGPT审核。


### 5.7 新增probe证明/失败边界与J身份摘录（外置候选，零模型负载）

本次用户授权沿同包`boundary-index-v1/`增量。§5.6计量Passed及累计18/512、剩494继续有效，未重跑计量或形状方法。ChatGPT已接受26个q1组的资源转移依据（仅dataset/features不同、窗口/批次数学一致、父PJM Passed/q1）；原54组、新26转移和8待新测仍分开。552任务/57新增/41与16/6240及全部profile不变；接受转移依据不等于实际新许可。

scope和不可执行模板现绑定`plans/probe-coverage.json`完整SHA；父组映射或差异证明任何变更均改变许可scope，执行前再次核对用于转移的同一计划字节。候选并行失败只在failure_kind明确resource时允许q4→q2；None、未知、业务、数值、身份、守卫/测量未确认失败立即向外抛出，保留成本/产物并停止后续批次，不以降低并发掩盖。既有预算预扣、资源阈值和数值规则未改。12个精确无模型CPU方法全部Passed，负例均拒绝，无新增模型计算；合成测试不构成GPU准入。

本轮获准按已接受J审核index/source-sha256锁定113个manifest路径与SHA。每份读取前后SHA一致，摘录实际identity的run_id/profile_sha/data_sha/commit/protocol_sha；复算profile/metadata JSON摘要，核对task、冻结profile、原许可data_bindings/code/commit/protocol。113份全部通过。原manifest只读，缓存原始JSON字节及摘录写在本包；索引查询从缓存字节对已接受manifest SHA重建实际身份，不再打开原formal-J。未读best/last或真实CSV，未重算test；AMD/PatchTST154条完整收据直接继承已接受旧收据字节，不重审。

统一索引现267 success（实际身份完整、结果沿已接受审核）、228 unverified、57 not-run；J新摘录的identity_extract_review独立标pending ChatGPT，不能把新摘录写成已被审核。旧J指标、best epoch、result来源/SHA保持。重复run、缺实际字段、摘要/actual-expected冲突及修改指标均拒绝；不手填expected到actual。默认index入口已指向本轮有来源链的收据，旧113字段缺失为§5.6历史时点。

本轮CPU/前向/反向/Adam/GPU负载中的模型计数均0，原18成本不变。候选及当前模板仍未部署/许可，TimeMixer安全结束审计、Git closure及8组GPU准入均未执行，不后台等待、不启动任何正式模型。原生产Git、文档、许可及artifact不改。新字节停于ChatGPT审核。


### 5.8 用户授权TimeMixer旧组无备份退役与固定LR重跑准备（尚未部署）

本次用户明确授权删除旧16任务整个formal-TimeMixer、准确绑定controller日志及旧专用启动证据目录，不需重复索取删除许可；这不授权删除其他模型/共享许可/旧技术准入，不改写此前训练或已见test事实。删除前只做任务/complete摘要、PID/start_ticks、项目锁、路径与共享占用核验，不读旧权重/CSV，不重算test或新建旧指标汇总/备份。旧189个组内文件、1个绑定空controller日志和8个启动材料共198个文件为目标；删除dry-run在同用户活动SSH进程PID64747的/proc/fd权限处停止，无法排除共享占用，实际删除0。未杀进程/提权/绕过权限，不能把仍存在的旧组写成已删除。相应回执仅保存路径/数量/大小/状态和必要身份依据，不复制旧性能记录。

原四域16任务按独立timemixer-fixedlr-v2/attempt2从头重跑：ETTh1、Weather、ECL固定LR0.001；Exchange维持0.0003及T96，其他三域T512。四H/seed2024及batch、epoch/patience、结构、固定LR无scheduler、精度/optimizer其余项全部保持。12个profile仅LR改变，483原profile及57新增profile不变。来源是用户批准的项目修订，不冒充作者原值，不是首次盲测，不做失败后自动换LR。执行额度新增16/最多200epoch，覆盖仍552/6240，含重复的计划尝试上限568/6440；旧实际消耗不退款。TimeMixer新增UrbanEV与5个EPF的既定参数及41/16分配均不改。

候选新入口start_timemixer_revision.sh固定ETTh1四路→Weather四路→ECL两路两波→Exchange四路，仅该16任务，不自动衔接其他组或新增29任务。原模型入口TimeMixer路由至同一revision；旧路径即使消失也不自动重建。worker在创建目录前核对新许可、revision/attempt、fresh及精确输出，禁止旧权重resume；manifest保存独立执行身份。重复start、已有staging或失败必须拒绝；失败停止后续，safe-stop只认本组PID/start_ticks。目标目录原E/revisions/timemixer-fixedlr-v2/formal-TimeMixer，原共享许可不能授权新LR。模板不可执行、不猜新commit。

退役边界与完整性审计分开：只有真实删除回执、所有目标已移除、用户退役身份才能准入；没有checkpoint完整性审计Passed。当前没有删除回执，preflight真实Blocked。三个改LR组从不变原组继承54中排除，现51不变＋3单列；26个新q1转移依据继续获接受，8新组尚未GPU准入。三域12代表的新LR串行/既定并发六步数值复核为单独提案（144Adam/192前向/144反向），尚未批准/执行，不使用57任务包剩494额度；Exchange仅按有效同profile证据审查复用，不伪造不同科学参数。既有数值规则/资源阈值不改。

本轮7删除合成方法＋11revision方法＋5受影响旧断言方法=23 Passed，均无模型；9项无负载CLI检查通过，模板preflight拒绝、旧入口dry-run正确路由、index保持其他267结果来源。一次真实删除dry-run失败保留，不能用CPU Passed覆盖。0模型/前向/反向/Adam/GPU；计量累计仍18/512、剩494。准备材料沿同包timemixer-retire-v2，旧结果没有新建备份；原共享审核材料保持，新当前索引解除旧组路径引用，16重跑为not-run。生产仓库/AGENTS/Closed/数据/作者/环境/原许可未改，不stage/commit/push，不部署/启动/后台等待。新字节待ChatGPT审核，当前实际删除仍Blocked。


### 5.9 删除前进程检查收窄复验：实际目标占用仍阻塞（外置候选）

沿同一timemixer-retire-v2，按本次明确授权把“所有同UID的fd/cwd/root均可读”改为旧controller/16worker原实例停止、既有项目锁、针对相关活动的占用检查及精确清单一致。普通非目标进程权限限制分开记录；关键原实例身份、锁、目录盘点或相关进程无法核实时仍拒绝。进程名/PID没有白名单；可见相关子进程也纳入判断。目录盘点用显式scandir传播权限错误，保留越界/符号链接/硬链接/共享材料保护。没有修改系统权限、杀进程或读取权重。

原7项删除边界断言原样保留，加10项受影响合成方法，17/17 Passed、exit0、0失败/error/skip，子case逐项落账；未重跑23项整包或9项CLI。真实dry-run一次exit1：31个同UID进程快照中，Desktop Commander的node PID64404/start_ticks103420214有17个fd实际指向旧TimeMixer的memory/process/config/manifest。其可见父链按相关活动登记；旧PID64747的cwd/root/fd不可读仅是非目标检查局限，不再作为删除阻塞。只证明当次可见元信息，未检查字段不冒称没有句柄，项目锁也不证明全系统无读者。原controller和16worker身份均无存活实例、项目锁已取得，complete及三目标归属仍匹配；重新盘点198文件/1,009,496,210字节与原快照一致。

因实际目标引用，按本轮明确停止线没有执行execute，三目标仍在、删除0、无deletion_completed回执，未创建备份或后台等候。须使用方释放已识别目标句柄后，再沿既有授权复核和删除；本次未操作Desktop Commander/SSH进程。上轮PermissionError日志和本轮实际占用日志均保留。本轮模型构造/前向/反向/Adam/GPU均0；累计18/512、剩494不变。新revision仍not-run，267个其他已接受结果来源不变；没有修改LR/任务/许可/索引实现，没有生产部署、closure、GPU准入或训练。候选审核仍独立，本段新字节未声称ChatGPT已审核。


### 5.10 独立Desktop Commander维护、精确删除与服务恢复（外置候选）

本次用户明确授权同轮暂停独立desktop socket服务、沿原脚本删除、再恢复。Codex执行进程不属于该服务进程树或PGID，生产仍033913ff且clean。原pane shell PID62975/start_ticks103285744保留；原remote/npm/sh/MCP链与前台PGID64363逐项复核。首次操作预检发现额外tee，尚未发信号即拒绝；进一步核对其PID64364/start_ticks103419896，stdin与npm的stdout/stderr同为pipe307256179，确属原服务日志管道，无无关负载。保留首次exit1记录，未重复领取测试/模型额度。

语法检查通过后，仅向desktop/%0发送一次C-c，原持有者64404及整条服务/日志链退出，shell取得前台；没有SIGTERM/SIGKILL、关闭SSH或tmux server。delete_exact.py保持SHA fd6f49ef29622667331ad164f4d582b7bdcabccdce563bef99ff6c3fbe9583c8，17项专项证据原样复用，不重跑。它自行取得项目锁，两步dry-run/execute均exit0；原controller和16worker无原存活实例，无相关目标引用，非目标权限局限仍记录。实际删除198文件/1,009,496,210字节、103目录，恰为原formal-TimeMixer、绑定空controller日志和旧专用launch目录；三目标均不存在，无残留/备份。deletion-receipt标user_authorized_retirement与deletion_completed=true，不是checkpoint_integrity_passed。未读权重/CSV或重算test，历史成本不退款。

finally路径在原pane按缓存Node22.16.0和0.2.50入口恢复：remote PID19308/start_ticks131841321，MCP PID19329/start_ticks131841821；一个remote及一个MCP，本地恢复标记之后见Connected日志，未再次打开目标。未更改认证/目录限制/安装文件；ChatGPT实际重连尚未代验。操作总信号为一次pane正常中断，不能把删除检查自身signals_sent=0解释为本轮没有信号。

其他267项索引与来源、共享许可、生产文档/代码/AGENTS/Closed/baseline保护核验保持。新revision16项仍not-run，原57任务准备、固定LR、覆盖与预算均不变。0模型构造/前向/反向/Adam/GPU，累计前向18/512、剩494；本轮只做操作脚本语法、真实删除和恢复核验，不运行新准入或训练。退役完成不等于重跑Ready；候选审核、部署/closure、新许可及数值/资源准入仍为独立停止条件。详细回执沿同一timemixer-retire-v2，旧失败证据保留，没有后台等待任务。


### 5.11 共用工程接入R工作区、两个独立入口与准入绑定（待审核/closure）

本次用户明确授权从P/candidate累计补丁接入R。起点三端033913ff00f3d09a0ed1eff26baf2352ffb27bc4、0/0、clean，22个逐文件生产修前SHA和候选修后SHA均吻合；无活动正式controller/probe/冲突维护，项目锁可取得。只做一次退役回执/服务操作回执SHA与三目标不存在核验，接受user_authorized_retirement/deletion_completed事实，不读已删除旧组、不重跑删除脚本/17测试、不再操作插件。22文件以git apply精确接入而非目录覆盖，随后仅改共用入口/身份测试与文档；R是全部业务import及code_binding来源，P/candidate保留历史准备用途，不是另一生产实现。Git工作区有8 modified＋14 untracked，index空，无stage/commit/push。

执行顺序：工程审核/closure＋必要准入→用户启动TimeMixer fixedlr-v2/attempt2的16次fresh/最多200epoch（ETTh1 4→Weather 4→ECL 2＋2→Exchange 4）→必要完成审计→用户另行启动41补做。41内部AMD四市场→J四市场→PatchTST四市场→TimeMixer NP/BE/FR/DE/PJM及UrbanEV F4的24任务；不得自动从16入口转入41。四个未跑模型各在原41任务后追加四市场，独立各45任务入口，不并入41补做。三域LR0.001、Exchange0.0003/T96及57新增配置均保持，552/6240目标和568/6440尝试上限不变。

接线修正：共用preflight/approval按实际授权task集合区分TimeMixer原四域16和补做29，AMD/J/PatchTST补做各4，不再混套原模型整组许可。41入口新增revision_completion_audit引用：新许可绑定审计path/SHA；审计必须是reviewed、当前protocol/commit、revision/attempt2和精确16任务，绑定当前修订目录complete及每run的manifest/result SHA与实际profile/data/commit身份；不读取被删旧目录，不以手填完成标记替代源链。无审计即阻断41，其他四个独立模型尾部追加不继承此队列顺序例外。模板保持false/commit=null，既有冻结科学事实不因此重新否定或自动变成新执行授权。

资源分区从现有清单核对51＋26＋8＋3=88互斥全覆盖；原三个0.01域不在不变继承内。26仅接受q1资源转移依据，8新组及3改LR组尚无新GPU结果。完整PLAN SHA、scope、当前code/config、环境/硬件、退役回执、父证据与未来新资源报告接线保持；新修订资源报告还必须与执行许可中的code/environment/hardware一致，不能只靠reviewed字段。原8组沿360Adam/512前向/360反向总上限，累计18前向/剩494；三个改LR组144/192/144独立待批，不挪用余额。未更改数值容差、资源余量、模型数学、数据/训练/指标合同。

受影响无模型CPU方法19＋7＋3=29次均Passed，正常unittest生命周期，精确ID和源码SHA在fixture前约束，子case单列；低优先级单进程/单线程，CUDA不可见，模型/optimizer/autograd/GPU入口禁止。未重跑计量、数据前缀、删除、旧模型smoke或整套probe。首次CLI索引比对发现16个pending行缺少revision/attempt输出，而旧预览还带删除前Blocked注记：现由生产索引生成计划revision/attempt2，状态仍not-run、无mse；历史Blocked不复制成当前状态。失败材料保留，局部3项复验及最终CLI index通过，267已接受行和来源逐行不变。

两入口dry-run/status/logs正常；complete因未运行返回2；使用不可执行模板调用preflight/start均返回2，未查询硬件、创建目录/训练session或负载。4个追加模型dry-run各45；shell语法和当前R导入/源码指纹通过；已有tmux守护/互斥/失败停止证据按未改路径复用，不新建占位守护。累计模型前向仍18/512，剩494，0新GPU/Adam/反向/正式run。

完整命令、22文件前后SHA、候选应用原补丁及当前工作区精确diff见同包workspace-integration-v1与report.md。当前启动仍Blocked于新版本审核/closure、相应技术准入和实际许可；41另等待16-run完成审计。P中的模板不是许可，未猜未来commit。生产AGENTS、Closed M4/M5、baseline、作者/环境/数据、共享原许可及其他模型原artifact保持；本段停在工作区实现/证据待ChatGPT审核。


### 5.12 三个改LR组独立受限probe接线（无模型验收；预算仍待批）

本次在§5.11已接受的22文件工作区上增量接线。起点三端033913ff00f3d09a0ed1eff26baf2352ffb27bc4、0/0，8 modified＋14 untracked、index空；22份字节及workspace-integration-v1清单/补丁摘要吻合。原链的确存在自依赖：extension probe授权要求revision_resource_report，旧make_config又统一使用extension-probe-v1，因此不能生成三组改LR所需报告。

新增`utils/ch3_revision_probe.py`、`m6_revision_probe_entry.py`及`scripts/ch3/start_revision_probe.sh`。`probe_scope=timemixer-revision-numeric-v1`经共用preflight、make_config、validate_config、worker边界、run_configs与STOP连接到既有受限六步worker、监测器、计量和逐域数值比较；不直接调用裸worker、不新建执行平台。扩展scope保持独立，不改变原8组授权。新scope不要求自己的未来完成报告，但仍要求预算明确批准、reviewed/execution_permitted、实际clean closure、code/config/environment/hardware、完整计划SHA、精确任务与退役回执绑定。模板为false、commit=null、hardware=null；不是许可。P中admission-plan当前“退役未完成”字段改为已实际完成、不是checkpoint完整性审计；旧失败记录未改。

固定12任务仅TimeMixer的ETTh1/Weather/ECL，各H96/192/336/720；串行参考各6步，再按q4/q4/q2两波各6步。独立144 Adam/192完整前向/144反向仍为待批准提案，起始debit=0，没有重试余额、不借用扩展包剩494。预算在创建worker前预扣；失败不退款，数值、资源、业务、身份、守卫或未知失败均停止后续，本scope不降并发补救。GPU余量仍max(8GiB,10%)，并发前结合单路峰值保守核验；原RSS/数值判据原样复用，需要额外长窗时本scope停止，不暗增步数。

真实执行以后，报告将绑定24次worker的配置、实际计数、轨迹、监测及必要数值sidecar SHA。三组决策由实际监测/既有比较函数及同任务短包收益产生；Exchange只绑定同profile的原q4已接受证据，不增加其probe。`complete.json`区分execution_complete与reviewed=false；`revision_report_reasons`消费新报告时复核其来源及决策，一次执行完成不自动等于审核。无模型合成fixture仅用于定向测试，不输出真实许可或真实准入结果。全覆盖仍51＋26＋8＋3=88；原51排除三个旧0.01域，26仅既定q1资源转移，8及3未进行GPU新测。

本轮15个精确CPU方法、25个子case全部Passed，正常unittest生命周期，源码SHA/完整ID在fixture前限制；低优先级单进程单线程、CUDA不可见，模型/optimizer/autograd/GPU入口被禁止。覆盖scope互斥、自报告依赖消除、跨许可/旧profile/错误源码/路径拒绝、STOP、固定波次、预算不退款、报告待审及Exchange合并、正式队列和267项来源索引不变。未重跑旧19＋7＋3、计量、删除、前缀或任何模型测试。14项shell语法/CLI检查符合预期：两入口dry-run/status/logs成功，preflight/start因模板未获准返回2，complete因未执行返回2；两技术输出根仍不存在。累计前向18/512、剩494，本轮模型/前向/反向/Adam/GPU均0。

全部科学profile、552目标任务/6240目标run-epochs、568尝试/6440尝试上限、16与41入口顺序及其他16追加任务保持。新三组必须先获预算批准、修后审核/统一closure和匹配许可，执行结果再审核；扩展probe消费该报告，正式16重跑和41补做的独立门禁继续有效。本轮未stage/commit/push、未GPU probe/训练。完整命令、不可执行模板、增量diff、前后SHA及精确账本见同包`revision-numeric-wiring-v1/`，不声称ChatGPT已审核本段新字节。


### 5.13 TimeMixer/UrbanEV限定数值确认与88组准入合并接线（本轮A；待字节审核）

本次用户批准限定政策与一次48 Adam/80完整前向/48反向预算；不直接授权尚未生成字节的commit或GPU执行。已接受诊断：`urban-numeric-diagnostic-v1/execution/complete.json` SHA `771eccd79de30375f8e07923262ac858db8639abe2260df00131d2edbe589207`；量化汇总 SHA `de2640b26c25cc34c5d0c3f68727a587aa5cd0892faf47187fb854add744fd3d`。接受诊断不等于准入，原 `admission_granted=false` 和不带捕获的UrbanEV q4 exact失败均保留。

新scope `urban-numeric-confirmation-v1` 仅作用于TimeMixer/UrbanEV/F4、h3/6/9/12、冻结T12/pred_len1/C11/B128/eval128/LR0.001/seed2024及既有结构/线程/精度/硬件。初始化、任务/profile、RNG、batch与形状精确；浮点参数/buffer、梯度、Adam moments最大绝对差≤1e-4；Adam step、非浮点、非张量状态及参数组精确。逐步loss绝对差≤1e-6；第2与第6步MSE/MAE、同元素数归一SSE/SAE差≤1e-6；全部rtol=0。finite、RSS、显存余量、归属和同任务并发收益原规则不变。该政策是用户批准的项目范围决定，不是作者配置、唯一算子根因或全局宽松容差。

四H各一次串行，然后一次q4：8 worker、**5次实际派发**。指令“6波”与固定顺序算术不符，按明确的4串行＋1并发执行，不增加波次/worker。每worker6/10/6：第2步按原顺序生成并评价两个合成CPU batch，缓存并记摘要；六步训练及原状态/RNG记录结束后，eval/no_grad复用这两个batch新增两次前向，核对RNG、持久模型/optimizer及batch不变并恢复模式。其他scope保持6/8/6。局部无重试、无q2/q1回退、无额外smoke。扩展总额360/512/360，起点144/210/144，完整执行后192/290/192、余额168/222/168；三个改LR组的独立144/192/144已经耗尽，不可借用。

准入来源拆分为51原不变＋26已接受q1转移＋7已接受q1＋3已接受改LR＝87；第88个UrbanEV必须来自本scope实际测量。新 `merged-admission.json` 逐组保留来源路径/SHA、真实旧commit/protocol/code和转移证明，不改旧失败目录，不要求旧extension complete变成功。新报告由实际状态/计量/监测重放生成，保持reviewed=false/admission_granted=false；正式消费必须另有对原未审核总报告SHA的明确审核登记。代码继承证明逐文件绑定本轮after字节，全部552科学profile、原政策及生产训练函数保持；仅新增scope启用终点评价，未启用时六步计算路径保留。

新入口为 `scripts/ch3/start_urban_confirmation.sh`，支持dry-run/preflight/start/logs/status/complete/safe-stop，使用原tmux方式和受限worker。不可执行模板及固定计划位于既有P的 `urban-numeric-confirmation-v1/`。A仅定向无模型验收；实际测试账、diff和完整前后SHA见该目录，不倒写旧证据。B须ChatGPT实际字节审核、精确closure、clean且三端一致以及匹配的新许可；本轮不stage/commit/push、不GPU、不正式开训。

确认通过并经结果审核后，才准备TimeMixer原四域16-run正式许可；仍由用户启动。41-run补做总队列必须等16-run完成审计后由用户另启，不自动跨接；267项已接受正式结果、552任务、6240目标run-epochs和568/6440尝试上限不变。带捕获/额外评价的短包耗时不冒称正式训练加速比。

### 5.14 2026-09-30：41-run完成审计与剩余六组顺序总队列（未提交、未启动）

用户本次批准跨模型一次启动，未缩减训练：DLinear→iTransformer→ModernTCN→TimeXer→N→S，45/45/45/45/24/24＝228唯一任务，最多2640 run-epochs、7,438,600 optimizer steps。按已接受88组decisions复算波数15/15/16/15/6/6＝73；前四模型的NP/BE/FR/DE在原六域41任务后追加，N/S仍仅UrbanEV F4。PJM及四市场单路；ModernTCN ECL q2两波，其余已接受q4。不跨模型/域补位，不提高并发、不按效果跳任务或改参数。全矩阵552/6240及含TimeMixer重复尝试上限568/6440不变。

已接受正文范围同步合入：UrbanEV/PJM/Weather/Exchange；UrbanEV为核心EV充电需求，PJM为price↔OT/index2电价预测，Weather/Exchange考察跨领域适用性。ETTh1/ECL/NP/BE/FR/DE不进入正文主结果表，但实验及结果保留，不缩减到180。不同域不求总平均；域内fold/H汇总、UrbanEV F1–F4及F4 A/N/S/J保持。无F0或新增消融，Informer/Autoformer未接入，J冻结与第四章路线不变。“每章不超过四个”未核实，不登记硬性校规；这是当前正文决定，不倒写原全部评测范围。

41-run在旧commit `22b968e741bf45e02c1f7c8ba96a8a91d53e7fb7` 下审计后才开始工作区修改。41任务、23波，实际541 run-epochs/1,128,675 Adam与反向/1,276,641前向；history按末条累计steps核账，validation严格best与early-stop、一次final test的计量/冻结源码/guard日志证据一致。controller与记录worker已退出，23波资源/退出摘要通过；未重扫高频遥测。82份本组best/last逐份受限CPU反序列化，检查SHA、ZIP/pickle允许类型、实际身份、epoch/steps、finite、optimizer映射/RNG、best/last状态结构一致性及同epoch关系；Python/system摘要一致。没有构造模型或重新评价test，未解析CSV。检查不等于完整constructor schema重建或完整系统调用审计。

审计脚本两次机械失败保留：metadata使用不存在的BestState.stop，修正为stopped；首份权重检查将DDI BN计数误设为steps，依据common.py循环修正为steps×(T/patch−1)，未放宽阈值。实际CPU反序列化83次、82份唯一权重，复验RSS峰值364,515,328 bytes，模型/前向/反向/Adam/GPU均0。新审计来源、收据、边界与失败日志见P/catchup41-completed-review-v1；工程Passed，结果及新审计仍待ChatGPT审核，不改原complete的Pending。

新增窄总入口start_remaining_models.sh和Python控制器，只同步调用现有m6_extension_entry --model（前四组）及ch3_runner --model（N/S），不调用裸worker、不启动六个独立tmux后立即返回。一个总tmux，子组真实退出后进行JSON技术交接：精确任务/身份/实际许可、finite、history/计量/早停/best、波次资源与进程退出、源码/环境/数据stat。通过后写technical-complete/result-review-pending并自动进入下一组；完整权重及科学结果统一在总队列结束后审计，不冒称逐组ChatGPT审核。

总运行元数据限定E/queues/remaining-models-v1；原任务仍E/formal-MODEL，四市场仍原supplements路径。独立队列锁与既有GPU组锁不嵌套同锁，冲突拒绝。STOP先持久记录，dispatch前后和交接前后检查，再按PID/start_ticks通知自有子controller，复用原受限波次清理worker；未知失败停止后缀，保留现场，不自动fresh重试/退款。活动、失败、完成、staging或已有任务输出均拒绝重复start。

许可分总许可与六个精确子许可；N/S不带extension_batch，不能混入catchup-41。新88组carry-forward保留原报告/lineage/真实旧commit/code/SHA，以552同profile、限定路由diff及原计算函数AST不变证明连接新版本；需要本轮实现实际字节审核，不重做88组probe，不改数值政策或预算。原扩展累计192/290/192、剩168/222/168保持。模板均不可执行、未来commit留空；必须完成41审计结果审核、修后closure和真实许可后，用户才可一次启动。当前实现与所有证据位于P/remaining-models-v1；精确测试账、tmux合成生命周期、patch与逐文件SHA见该目录。本轮不stage/commit/push、不真实start、不后台等待。

### 5.15 2026-10-01：隔离七baseline M批次准备（ECL暂缓仅限本批）

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

### 用户授权：v4执行修复与recovery1（2026-10-03，待审核/预算阻塞）

v4 probe已Passed，formal职责耦合导致重复full audit；这是工程执行问题，不是模型/效果失败。用户授权既有safe-stop，原7个chain/stage/group/worker实例全部退出，原8项完整结果88个文件未改写。停时实际为iTransformer UrbanEV fold3四H，完整checkpoint/history到epoch7；87 inventory为A8/B4/C0/D75/E0。停止原因user_authorized_execution_repair_stop，全部旧STOP/failure/staging/history/checkpoint/budget保留，result_review=pending，scientific_failure=false、budget_refund=false。test状态仅按现有guard、预算、checkpoint/结果证据核验，不声称逐次kernel访问审计。

四项B committed steps依H3/H6/H9/H12为26061/26012/25970/25921。实际Adam为28051/28211/27824/28285；checkpoint后Adam为1990/2199/1854/2364，合计8407，backward合计8408，forward合计8410。这些partial消耗不退款；恢复从epoch8开始，未保存partial需要重复时计入future成本中的说明项，不再次加总。历史实际Adam/backward/forward=259281/259282/288464；future worst=3436656/3436656/3898201；total worst=3695937/3695938/4186665。原optimizer授权3687530，Adam缺口8407；forward上界4178255为算术推导，不冒称独立原授权。目前preflight应明确additional execution budget authorization required，不能签发可执行恢复许可。

新package native-time-mark-chain-v4-recovery1与旧v4隔离，执行scope m6-native-tmark-chain-v4-recovery1，session ch3-native-tmark-chain-v4-recovery1；replacement/M新输出根分别revisions/native-time-mark-v4-recovery1、m-tasks/m-baselines-native-time-mark-v4-recovery1。science baseline仍0734f3e91f15854f75907c30d10d43e942f69267，controller/worker真实版本后续closure取得；A真实旧版本不改，B旧前缀+新后缀，C/D真实新版本，不能声称新fresh任务在0734实际执行。

原probe四个固定SHA引用保持，manifest由原complete expected SHA确定性投影，admission-summary紧凑且来源绑定。formal supervisor启动一次全量SHA scan，跨进程runtime-admission绑定实际owner PID/start_ticks及启动链认证；permit/config用refs，不复制巨大probe报告，formal worker不允许raw probe sidecar。87完成后一次完整closeout/seal，未成功seal允许合法重试；M permit/group/wave/task轻量消费sealed87边界。M probe即时gate保持，最终AUTO_AUDIT一次；M formal启动scan一次，此后raw replay=0。generic full_float_state按payload只比较一次，None仍全状态exact，未知policy拒绝，TM UrbanEV额外endpoint gate保留。

A不train/test，B不改旧artifact、严格恢复model/optimizer/RNG/best并read-old/write-new，C须0/0/0、旧staging保留，D fresh。若某任务无法确认正式test是否已经访问，或其执行身份无法确认，则该任务必须标记为E类；不得自动恢复，也不得再次执行正式test，必须停止该任务的自动恢复流程并返回后续明确处理。按原wave过滤A，不补位；effective-result-map固定科学格来源、不按效果选旧新。drain-stop等待当前wave落盘/技术清算再停，resume_eligible只表示候选，不能绕过后续门禁；safe-stop只停本scope owned tree。科学87/1020/3687530与M84/840/294790、M21组原policy/caps1512/2016/1512及所有模型/数据数学保持；本轮GPU/模型构造/forward/backward/Adam=0，checkpoint仅受限CPU审核，不重跑原probe或完整结果，不恢复、不commit/push。生产R不变，工作区新字节与外置回执待ChatGPT实际审核。

### 最新决定：退休 recovery1，准备 baseline-unified96-onecycle001-v1

本小节登记用户最新科学协议，取代旧 v4/recovery1 剩余执行要求，不改写以上历史快照。新 scope m6-baseline-unified96-oc01-v1；七baseline按 AMD→DLinear→PatchTST→iTransformer→TimeMixer→ModernTCN→TimeXer。MS=UrbanEV F4六fold四H＋PJM/NP/BE/FR/DE各H24，共203；M=ETTh1/Weather/Exchange四H96/192/336/720，共84；total287 fresh。J冻结不重训、不改L/LR/结构/结果、不重新test，N/S排除，ECL继续不进入M。

旧链停前实时为 REMAINING_TMARK_FORMAL/iTransformer：有效complete16、fold5四H partial4、not-started67；partial history末epoch2。仅执行自身 safe-stop，原 supervisor47335/start_ticks166770247、group47816/166772919、workers13925/167274780、13928/167274798、13934/167274822、13942/167274842及监测子22591/167415365原实例均退出。原v4＋recovery formal累计 Adam/backward/forward=573259/573261/641201，不退款；8407旧额外授权不转移。全部旧结果/checkpoint/history/budget/STOP/failure保留，退休性质 user_authorized_protocol_supersession、scientific_failure=false。回执 SHA256=64869e3735c49334dfd2e127691c1c2c958670c0e3d5f3d3b59f5e696aa26604；旧完成或checkpoint不填新格、不得resume进新协议。

UrbanEV T12/pred_len1/C11/F4六fold及label H3/6/9/12保持；其它任务T96。MS原生mark三模型继承已审核native-v4父profile，其它四模型继承生产552对应正式profile；M继承当前M resolved profile。只改T和LR/scheduler，batch/eval_batch、epochs、结构、dropout、seed、target/aux/feature order、split/train-only scaler、metric/test-once/native historical mark与freq全部保留。UrbanEV/M10epoch、EPF20epoch。两份新config与287份精确parent→new对照独立于冻结原config。

OneCycle来源为 TimeMixer作者e24610583b36fdd8c76cc17a8df4e65759a5f460的ETTh1_unify脚本、run.py、exp_long_term_forecasting.py以及正式PyTorch2.0.1签名。显式冻结 max_lr=.01,pct_start=.2,cos,cycle_momentum=true,base_momentum=.85,max_momentum=.95,div_factor25,final_div_factor10000,three_phase=false,last_epoch=-1,verbose=false,total_steps=None、各任务epochs/steps_per_epoch；Adam成功update后step，val/test不step。继承beta2/eps/weight_decay；beta1按标准OneCycle更新。新checkpoint v2包含scheduler，恢复LR/beta/steps连续；旧schema保持兼容，但跨协议checkpoint拒绝。该统一训练设置是用户选择的baseline公平/诊断协议，不能冒称各作者推荐设置，也不能声称J使用相同超参数。

正式预算待审：MS203/2380/7974050 Adam、backward/9019739 forward；M84/840/300150 Adam、backward/359040 forward；total287/3220/8274200 Adam、backward/9378779 forward。MS probe15组63代表，UrbanEV每模型fold1四H、EPF按真实batch/structure兼容性分组，每市场scheduler长度独立核对；名义126 worker、756/1024/756，最多187 worker、1122/1520/1122。M21组84代表，名义168 worker、1008/1344/1008，cap252 worker、1512/2016/1512（Adam/forward/backward）。沿现有对应scope数值政策，不扩大容差，scheduler/LR/beta1/nonfloat/RNG/batch/identity exact；仅明确resource失败允许原q4→一次q2→合法serial q1，原q2组只可退q1。

单tmux新链：BASELINE_PROTOCOL_PREFLIGHT→MS_RESOURCE_NUMERIC_PROBE→MS_AUTO_AUDIT→MS_FORMAL_ALL_BASELINES→SEAL_MS_BOUNDARY→M_RESOURCE_NUMERIC_PROBE→M_AUTO_AUDIT→M_FORMAL_ALL_BASELINES→COMPLETE。MS七组真实退出并技术seal后进入M，自动机器准入，不逐模型/阶段等人工。AUTO_AUDIT一次完整审计，formal生命周期一次manifest SHA扫描；group/wave/task/worker仅轻量许可/runtime refs，raw numeric sidecar不进formal worker allowlist。STOP持久登记，只signal新scope PID/start_ticks拥有的链，不碰旧实验；重复start/staging/failure拒绝。技术失败保留现场，不重试、不调LR，不按效果取消后续；COMPLETE=result_review pending。

新结果位于 baseline-unified96-onecycle001-v1/{MS,M,probe,queue}，旧结果不搬迁/删除/覆盖、不与新结果按test择优。若新协议最终作为论文baseline，七baseline主结果整体用新协议，J仍保持自身冻结结果。工期只是按父runtime、新窗口/batch、T和固定wave估计，probe后用初始化/update/validation分项校正。本轮停在隔离W新字节待ChatGPT审核：未stage/commit/push、未新start/GPU/probe/formal/test；生产R不变，预算与许可模板不可执行，尚未合并生产canonical。

### 最新用户政策修订：baseline-unified96-onecycle001-v2（2026-10-03，隔离 W 候选待审）

v1实际closure为22b6447c4e62078d903446690225504f26e7a3ed。MS probe共15组，前11组Passed，ModernTCN-EPF-compatible-0因exact numeric gate失败并自动停止；正式MS/M未开始。失败四个q4成员的state/loss/normalized metric差：PJM 1.3441592454910278e-4 / 1.1920928955078125e-7 / 1.0502029601511254e-7；NP 2.063065767288208e-4 / 1.1920928955078125e-7 / 2.1928189686271082e-7；BE 2.4513527750968933e-4 / 2.384185791015625e-7 / 2.3932049497688013e-7；FR 1.7508119344711304e-4 / 2.384185791015625e-7 / 3.2853599907234354e-7。原failure/progress/permit/trajectory/full-state/budget/launcher/controller全部保留，不resume、不改写Passed；历史actual为642 Adam / 642 backward / 872 forward，budget_refund=false。

仅MS ModernTCN-PJM/NP/BE/FR/DE五scope的numeric_policy由None/exact变为统一full_float_state，policy id baseline-unified-v2-moderntcn-epf-fullfloat-atol5e-4：state_atol=5e-4、loss_atol=metric_atol=1e-6、loss_rtol=rtol=0、equal_nan=false。全浮点参数/buffers、gradients、Adam exp_avg/exp_avg_sq比较；初始化/RNG/batch/order/step、非浮点state、param groups、shape/dtype、task/profile/source/data以及scheduler配置/step/LR/beta1继续exact，finite必须Passed。ModernTCN-UrbanEV仍exact，AMD/DLinear/PatchTST/iTransformer/TimeMixer/TimeXer与所有M政策完全保持。这是用户批准的M6 technical numeric admission threshold revision，不冒称作者标准。

新协议/结果identity baseline-unified96-onecycle001-v2、scope m6-baseline-unified96-oc01-v2、tmux ch3-baseline-unified96-oc01-v2；全部新package/result/probe/queue/launcher隔离。v1配置文件原字节保留；新v2 MS/M配置的287个正式profile逐项与v1相等，只有任务/协议namespace、数据元数据索引及明确五scope门禁变化。七baseline固定，MS UrbanEV+五EPF共203，M ETTh1/Weather/Exchange共84；total287/3220，J/N/S/ECL排除，UrbanEV T12、其它T96，OneCycle/batch/epoch/结构/data/seed/metric/test/time-mark均不变。

v2重新从独立serial开始运行完整15组MS probe，不借用v1的11组Passed。新MS probe cap1122 Adam/1122 backward/1520 forward，starting debit=0/0/0；旧actual＋新worst=1764/1764/2392。正式预算不新增：MS2380 run-epochs/7974050 Adam及backward/9019739 forward，M840/300150 Adam及backward/359040 forward，total8274200/8274200/9378779；M21组probe原cap1512/1512/2016保持。仅resource失败可q4→一次q2→合法已测q1、原q2→合法q1；numeric/finite/identity/scheduler/data/guard失败停止，不fallback、不退款、不扩额。

完整单启动链仍为BASELINE_PROTOCOL_PREFLIGHT→MS_RESOURCE_NUMERIC_PROBE→MS_AUTO_AUDIT→MS_FORMAL_ALL_BASELINES→SEAL_MS_BOUNDARY→M_RESOURCE_NUMERIC_PROBE→M_AUTO_AUDIT→M_FORMAL_ALL_BASELINES→COMPLETE，正常Passed路径不逐模型/阶段等人工。旧四成员payload只读重放只证明新阈值下的离线比较，不算v2实测Passed，也不授权DE或未来formal。本轮完成无GPU准备与模板拒绝路径检查，不生成可执行start authorization，不启动v2/GPU/probe/formal，不stage/commit/push，生产R不变；新字节待ChatGPT审核，COMPLETE仍仅technical complete、result_review=pending。

### 用户最新决定：baseline-unified96-onecycle001-v3（2026-10-03，隔离工作区待审）

用户明确安全中止v2，退休剩余计划，七baseline UrbanEV统一max20 epoch/patience5后完整fresh重跑。基线commit b1a88bff143e4513761ac02895efd683fe32b499；生产R 5341fbcb7c9f4f97658728d79b1af5487f7d38c3保持clean，本轮不合并生产。停止前state MS_FORMAL_ALL_BASELINES、model AMD，已落group progress wave1；活动wave2为fold3四H。现成safe-stop先持久化STOP，旧helper把合法group-child当start-only而返回PermissionError/exit1；controller随后按既有STOP清理owned tree，未手工扩大kill。原supervisor16699/169639165、group31619/169810770、workers54584/170649808、54585/170649816、54586/170649822、54587/170649835及子14310/171196207全部退出，tmux退出，GPU计算进程为空。旧failure为InterruptedError persistent STOP；性质user_authorized_protocol_supersession、scientific_failure=false、budget_refund=false、result_review=pending、resume_into_v3=false。入口错误与检查失败尝试如实保留；v3仅修正owned child完整命令/action与parent匹配，仍严格PID/start_ticks，只signal本scope。

MS真实8 completed、4 partial/staging、191 not-started；M 84全未开始。8 completed为AMD UrbanEV fold1/fold2四H，4 partial为fold3 H3/6/9/12，history均到epoch6。四partial持久Adam/backward/forward为24295/24296/26930、24519/24519/27112、24411/24411/26968、24514/24514/27028；MS正式累计244649/244650/272116，MS probe实际756/756/1024。持久formal计数为可证明lower-bound，未持久瞬时消耗unknown，不猜、不少报、不退款、不抵扣v3。2259份v2留存文件的metadata及关键SHA绑定；退休回执v3/v2-retirement-evidence.json SHA=f4a85296ece0cadb9f3835e0c3252517d0b18f9d89150cb81bd6b1675d6cf20e。所有旧completed/partial/checkpoint/history/budget/log/STOP/failure/probe继续保留，绝不resume/copy/hardlink/symlink/初始化或填入新格，也不按旧新test择优。

新protocol baseline-unified96-onecycle001-v3、scope m6-baseline-unified96-oc01-v3、tmux ch3-baseline-unified96-oc01-v3，全新package/result/probe/queue/permit/start/launcher/STOP隔离。七baseline顺序保持AMD→DLinear→PatchTST→iTransformer→TimeMixer→ModernTCN→TimeXer；MS UrbanEV168＋EPF35=203，M ETTh1/Weather/Exchange四H84，total287 fresh；J/N/S/ECL排除，J既有冻结结构/profile/source/result/test不改。

直接父来源是固定SHA的v2 MS/M配置。168 UrbanEV唯一profile diff为training.epochs10→20、training.patience None→5、training.scheduler.epochs10→20；steps_per_epoch逐任务保持，所有OneCycle其它参数不变。35 EPF科学diff=0且20/5；84 M科学diff=0且10/None。UrbanEV T12/pred_len1/C11/F4、batch/eval128、Adam、weight_decay1e-7/seed2024、fold/H、feature/target/aux、split/scaler/data/time-mark/source/metric/test保持。沿BestState validation MSE严格<改善、tie保留更早best、连续5次无改善后下一epoch前停；max20不强制满20。best仅validation选、正式test一次、不按test提前停或取消后续。

numeric-policy diff=0，MS registry SHA d29aaf90b9db5dfa8625500b114ee2f580bfa01e453dce31064bbee40d1be811，M registry SHA 3f20ba5a36d921f1b4178b98648421774469d861b9ab6f794d4e3c0385cfc084。ModernTCN五EPF仍5e-4 full_float_state、loss/metric1e-6、rtol/loss_rtol0、equal_nan=false，UrbanEV仍exact；其余MS/M逐项继承。data metadata仅task-id namespace索引变化，科学值不变。MS15组63代表和M21组84代表全部重新从serial开始，不拼接v1/v2 Passed。仅resource可原q4→一次q2→已合法serial q1、原q2→serial q1；numeric/finite/identity/scheduler/RNG/batch/data/source/guard/unknown失败停止。

用户已明确授权独立v3全预算，helper复核一致：MS203/4060 run-epochs，Adam/backward/forward=15274280/15274280/17173829；M84/840、300150/300150/359040；total287/4900、15574430/15574430/17532869。MS probe nominal756/756/1024、cap1122/1122/1520；M nominal1008/1008/1344、cap1512/1512/2016；probe合计worst2634/2634/3536。starting debit均0，旧成本永久保留、不转移。seed2024、additional_search0、无新增候选，不挪early-stop节省额度，不自动扩cap。

链保持BASELINE_PROTOCOL_PREFLIGHT→MS_RESOURCE_NUMERIC_PROBE→MS_AUTO_AUDIT→MS_FORMAL_ALL_BASELINES→SEAL_MS_BOUNDARY→M_RESOURCE_NUMERIC_PROBE→M_AUTO_AUDIT→M_FORMAL_ALL_BASELINES→COMPLETE，正常Passed无需中途人工许可。76/76 unified＋v3无模型方法验收通过，0 failure/error/skip，模型构造/forward/backward/Adam/GPU initialization均0；ch3_runner除v3 code_binding外所有训练数学函数AST与父版本相等。执行模板reviewed/execution_permitted/structure_frozen/m6_authorized/budget_authorized仍false、closure_commit=null；真实dry-run展示blocked计划，模板preflight拒绝尚未审核/closure字节。本轮只准备，未stage/commit/push、未生成可执行start-review、未启动v3/probe/formal/test。新字节待ChatGPT服务器实际审核，COMPLETE=result_review pending。

#### v3 审核后历史测试版本隔离修复（2026-10-03，新修复字节待审）

原 Codex targeted 76/76 Passed 为当前 unified＋v3 的窄范围证据，继续保留；它未覆盖历史 v2 测试。ChatGPT broader v2＋v3实测48项、5 failures、2 errors，原因是旧测试通过当前共享模块取得v3 package/session/profile/budget/probe IDs，属于test harness regression，不是v3科学/模型失败。原审核失败及上一版inventory/patch/report/evidence-index/acceptance全部保留。

本次只修 tests/test_m6_baseline_unified_v2.py：18个测试方法和原断言语义不删、不skip、不expectedFailure、不改容差，显式绑定v2冻结commit、config/package/result、固定SHA的start-review及plan。历史准备fresh按当时留存证据核验，历史拒绝行为只提取旧源校验函数，不加载旧执行链；v1只读payload重放不生成任何新admission。使用指定正式Python和无CUDA、无pyc环境，v2＋v3为48/48 Passed，统一三组为94/94 Passed，0 failure/error/skip；模型构造/forward/backward/Adam/GPU initialization均0。

相对修前v3候选，v3代码、config/profile/numeric registry/预算及计划科学字节均保持，MS203/UrbanEV168/EPF35、M84、total287与UrbanEV20/5、EPF20/5、M10/None不变，numeric diff=0；没有修模型/训练或扩大执行范围。新增工作区变化仅历史v2测试，候选合计8 modified＋3 untracked=11。两份文档只追加上述审核/复验记录。未stage/commit/push、start-review、tmux、GPU/probe/formal/test-set访问，生产R保持clean；修复后的新字节待ChatGPT最终审核。


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

N从真实closure `9341e4eb44ed9225a3965b102b2504b1fecfe830` 在原 `m6/type1-followup-v1` 分支继续开发，不新建worktree。先复核N/W/R完整Git现场与原owner，原v1仍为WAIT_V3_COMPLETE_AND_RELEASED；仅通过旧入口safe-stop停止PID22206/start_ticks187020263。原实例/tmux退出，无owned计算子进程；实际completed/partial/not-started=0/0/63，模型/GPU/forward/backward/Adam/test消耗0。原STOP/failure/controller/log/permit全部保留，退休原因user_authorized_protocol_supersession，scientific_failure=false、budget_refund=false、old_v3_signal_sent=false、resume_into_v2=false。新包 `v1-retirement-receipt.json` SHA `b472ac18d204d4d668e4c55ed2ae3b848f8d420888e997e2ade67f08df97a5de`。W/v3原owner36799/start_ticks171530843仍存活、自然推进；W df6a164...、R5341fb...、作者源码/环境/数据不编辑、不signal。

新protocol/scope/session为baseline-type1-followup-v2、m6-baseline-type1-followup-v2、ch3-baseline-type1-followup-v2；package/result/probe/queue/permit/log/STOP独立fresh。顺序固定AMD→DLinear→PatchTST→iTransformer→TimeMixer→ModernTCN→TimeXer。URBAN_SUBSET：F4 fold1/2×H3/12，28项，T12/pred_len1/C11，batch/eval128，20/5；仅28/168，不补其余140格、不冒充六fold。EPF_ALL：PJM/NP/BE/FR/DE×七模型，35项，T168/H=pred_len24，实际batch/eval32，10/3。M_ALL：ETTh1/Weather/Exchange×H96/192/336/720×七模型，84项，T96，batch/eval128，10/None，全通道监督/评价。合计147 fresh/1750 max run-epochs（560/350/840），J/N/S/ECL、新seed/search/candidate均排除。

batch来源为冻结v3 TimeMixer M profile，ETTh1/Weather原128，Exchange原512由用户覆盖128；accumulation=1，不按microbatch伪装，不自动缩LR、AMP、增积累或改BatchNorm。parent→new差异仅批准的抽样/身份、T、batch/eval、epochs/patience、lr/scheduler，模型结构、features/target/aux、historical marks、split/train-only scaler、seed2024、基础betas/eps/wd、线程/workers/dtype/shuffle/drop_last保持。T/batch窗口、marks形状、批次尾部和预算重算；评价全元素SSE/SAE加权、eval_drop_last=false，预算未读test标签。新增M是真实all-channel路由，closeout核验输出顺序与元素总数。

独立scheduler `type1_horizon_scaled_v1`，Adam initial_lr=1e-4，固定基础betas，无OneCycle动态beta1。实际第e轮 `lr=1e-4*0.5**(max(e-2,0)*c(E))`，E<=10为c=1，E>10为c=8/(E-2)。E10逐轮等价锁定TimeXer760119...的type1；E20拉长是用户批准的项目修改，不冒称作者原配置。首两轮1e-4，E20第三轮约7.348672461377994e-5，第10轮约8.504937501089857e-6，末轮3.90625e-7。E固定，不随早停/暂停/恢复重缩；epoch末val/BestState后、继续时按绝对epoch设置下一轮LR并保存，独立state/checkpoint schema含公式/E、updates、completed_epoch、LR位置/current-nextLR/固定betas。旧type1/OneCycle兼容；新协议拒绝旧checkpoint。模型forward/loss/data/evaluate数学与作者源码不改。

未来用户arm一次即可持久等待完整unified-v3，固定其Wcommit、start-review/controller SHA及PID/start_ticks，沿用原锚点和完整203 MS＋84 M sealed边界/唯一索引/来源检查。旧running允许arm但不启新GPU；等待不持训练组锁。无旧STOP/failure、原owner及登记owned实例释放后才计算。正常链等待→preflight→Urban probe/AUTO_AUDIT/formal/test/seal→EPF probe/AUTO_AUDIT/formal/test/seal→M probe/AUTO_AUDIT/formal/test/seal→COMPLETE，result_review=pending。第一/第二环效果不取消后环，只由技术完整自动续接，无中间人工许可。新M消费本协议28＋35项前序sealed边界，不用旧M许可或87/203/287硬编码。旧v3自然完成不会要求重训或再手工start。

probe共36组147代表：Urban7/28跨fold1/2同波q4；EPF8/35，非TimeXer六模型固定4+1，TimeXer保留PJM/BE/FR结构组实际3、q4容量及NP/DE实际2、q2，统一batch32不合并结构；M21/84每模型×dataset四H候选q4。各代表独立serial六步，跨完整epoch/全E/resume用无模型fixture。numeric逐scope继承v3，diff=0；ModernTCN五EPF仍5e-4/Urban exact，M21政策不变，TimeMixer Urban endpoint保留。仅明确resource可预登记降q，不改batch/AMP/accumulation，q1仍不容固定batch则阻塞；其它技术失败立即停。实际run_probe尾部完整审计每环0、AUTO_AUDIT每环1，合计3而非6；即时gate保留。formal完整probe回放与逐task原张量读取0，每环formal生命周期manifest scan1，子进程轻量消费compact refs；后环不逐任务重建前序边界。

用户授权的预算由helper独立复算（Adam/backward/forward）：Urban28/560=1028440/1028440/1143128；EPF35/350=399000/399000/467845；M84/840=108080/108080/128877；总147/1750=1535520/1535520/1739850。probe Urban nominal336/336/464、cap504/504/696；EPF nominal420/420/560、cap618/618/824；M nominal1008/1008/1344、cap1512/1512/2016；总nominal1764/1764/2368、cap2634/2634/3536。正式/probe独立计账，cap只含预登记resource回退，无无限重试；旧消费不退款、不转移，效果不佳/早停不增加预算或补做。

三环都是正式补充实验，val严格MSE `<`、tie保留早best及既定patience，best锁定后test一次；probe不访问test，派发和早停不由test控制。无法确认test是否已访问则停止该项自动恢复，不再次test。全部147结果保留，不按旧新test逐格择优、不自动替换论文主表；对照只读合法旧指标，包括EPF补做目录，不重训/重test。保留已看test的研究历史；多项合同同时变化，不将差异单独归因于batch，单seed std=N/A。

原199/199证据保留其原覆盖；本轮版本隔离历史94、OneCycle11、新v2 81合计186/186 Passed，0 failure/error/skip，禁止计算钩子记录真实模型构造/forward/backward/Adam/GPU初始化0。实际M完整审计和closeout合成fixture通过。初次72项3failure、182项1error及辅助脚本语法/缩进/启动失败均留存，修正fixture/期望/接线，不删断言、不skip、不放宽；183/185和最终186证据均保留。真实模板dry-run发现初始source_states用dataset名过滤路径键而为空，现按冻结v3预期记录投影所需文件路径、再校验实际stat；未修改数据/profile/numeric/预算，初始材料归档，新增回归覆盖。

AST/JSON/bash-n/bundle SHA/diff-check通过。修后模板dry-run exit0展示完整计划；preflight exit2按预期拒绝未review/closure/真实start-record绑定，blocked仅上述三项，来源/数据/环境检查通过。模板flags=false、closure=null，当前READY_TO_ARM_HANDOFF=false、READY_FOR_GPU_EXECUTION=false，没有start-review、结果根、真实tmux/等待器/GPU/probe/formal/test。估时按历史runtime、精确批次和实际wave，候选q通过新正式约10–55小时，含probe/检查规划约11–60小时；q2约16–97小时，serial约30–182小时。M batch128和输出开销尚未实测，保留宽代理区间；旧v3剩余等待独立登记，不用六步÷6外推正式epoch，也不把估计代替启用门禁。本轮未stage/commit/push，外置P完整SHA/证明/失败账/计划/操作命令及N候选增量待ChatGPT实际字节审核，W/R原文不改。

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

<a id="m6-amend-implementation"></a>
#### 第二轮112项补齐/Weather20修订与第三轮Weather20续接（2026-10-06 UTC，N候选）

本次来自用户明确科学调整，不冒称已证明旧实现有科学bug。日常仍只有三轮：第一轮＝初始全量及已接受补做；第二轮＝unified-v3及本次amend1；第三轮＝type1-followup内部revision v3，不称第四轮。开发仅N，基线5bd62dc93d611b3271467a46fd16335b622e0af1。W原v3继续203 MS＋84 M=287，代码/config/docs/许可/历史artifact不改；R、作者、数据、环境、Closed milestones及baseline tag不改。

旧第三轮v2原owner46708/start_ticks189114285经scope、完整argv、parent/owned核验，用原safe-stop入口退出，session退出。停止时仍WAIT_V3_COMPLETE_AND_RELEASED，无probe/formal/test及消耗，Adam/backward/forward/test均0；退休性质user_authorized_protocol_supersession、scientific_failure=false、budget_refund=false、old_v3_signal_sent=false，原STOP/controller/日志/许可永久保留。W owner36799/start_ticks171530843未被发送信号，继续合法执行；自然进度不作版本异常。

第二轮补做独立baseline-unified96-onecycle001-v3-amend1，七模型AMD→DLinear→PatchTST→iTransformer→TimeMixer→ModernTCN→TimeXer，每模型ETTh2四H→ETTm1四H→ETTm2四H→Weather四H，共112 fresh/1400 max run-epochs。新增ETT84项从冻结v3自身同H ETTh1-M实际profile迁移结构；L96、pred_len=H、train/eval128、10/None、seed2024、accumulation1，全通道监督/评价。独立数据SHA、split、train-only scaler、小时/15分钟marks继承现有已审核来源，不能套ETTh1数据SHA/scaler或小时marks。采用冻结v3 OneCycle：max_lr0.01、pct_start0.2、cos、cycle_momentum=true、base/max_momentum0.85/0.95、div25、final_div10000、three_phase=false、last_epoch=-1、verbose=false、total_steps=None、epochs10、steps_per_epoch按实际批次；每成功Adam update后推进，validation/test不推进，max_lr不等于固定/第一步LR。Weather28只改training.epochs和training.scheduler.epochs 10→20；patience=None、各模型自身原batch/eval_batch、steps_per_epoch、结构/数据/marks及其余OneCycle不变，fresh20，不加载10轮checkpoint或拼接旧轨迹。

来源复核固定论文2405.14616v1第14页Table7，PDF SHA a599b338e44af70d8e9c87be3c5417bde7864b2c92074e1346703f3e2b641e3d；TimeMixer作者commit e24610583b36fdd8c76cc17a8df4e65759a5f460与四份来源源码SHA一致。对齐限定指定L/batch/最大epoch及ETTh1-derived OneCycle安排，eval128/patience=None为项目合同，不能声称各模型作者最优或完整复现。差异明确：冻结TimeMixer迁移至ETTm2仍d_model16，Table7为32；本轮不扩大结构修改。

第二轮唯一日常主结果入口371有效格＝原MS203＋原ETTh1/Exchange56＋新Weather20的28＋新ETT84，原287边界不扩成371。新Weather28全部技术完成后按预先固定规则整批切换，效果更差也不逐格选旧结果；此前显示revision Pending，不冒充10轮值是20轮。原Weather10和其他旧artifact保留位置、真实时间/config/commit；索引只引用并保存来源ref/SHA、profile/protocol、实际执行版本、revision及supersedes，不搬移/重写。371格区别于原287＋112＝399次计划正式运行，其中Weather28是替换重复计算；probe与失败attempt另记。第三轮结果汇总消费封存的第二轮修订来源，旧v3记录另标历史，不自动按test替换主表。

第三轮新baseline-type1-followup-v3仍231 fresh，Urban28（f1/2、H3/12，T12、20/5）、EPF35（T168、batch32、10/3）、M168（六域/四H，L96、batch128）。只Weather28变20/None、scheduler.epochs20、coefficient4/9；其他203项resolved scientific profile diff=0，全numeric registry逐项不变，ModernTCN EPF5e-4不套给M。总max run-epochs2870。复用type1_horizon_scaled_v1，lr(e,E)=1e-4×0.5**(max(e-2,0)×c)，c=1(E<=10)或8/(E-2)；WeatherE20首两轮1e-4、e3约7.348672461378e-5、e10约8.504937501090e-6、e20=3.90625e-7；E固定，基础betas固定，不残留OneCycle动态beta1，不拼接旧10轮。无J/N/S/ECL、新seed/search或额外Urban格。

未来一次总arm：旧287完整技术完成＋owned释放→112项必要probe/AUTO_AUDIT/formal/test/seal→第二轮371来源封存→第三轮Urban→EPF→M各自probe/audit/formal/test/seal→COMPLETE，result_review=pending。第三轮不得仅等旧287绕过112；旧v3 running允许审核closure后arm等待，等待不初始化GPU、不占训练组锁。效果不控制后环，技术失败停止派发；仅明确resource按预登记降q，q<=4，不改batch/精度/结构/容差。即时gate不删，probe尾部完整审计0、每个AUTO_AUDIT1，四执行阶段合计4；formal compact引用贯通、零完整probe回放/逐task raw读取，manifest扫描每阶段监督器生命周期1次。补做仅新增scheduler类型接线到原OneCycle精确检查，legacy/type1行为保留。

helper预算（Adam/backward/forward）：补做112/1400=608000/608000/734673；第三轮Urban28/560=1028440/1028440/1143128，EPF35/350=399000/399000/467845，M168/1960=354480/354480/446642，第三轮231/2870=1781920/1781920/2057615；新增总343/4270=2389920/2389920/2792288。补做probe cap2016/2016/2688、第三轮4146/4146/5552，总6162/6162/8240；nominal总4116/4116/5504。用户批准固定科学范围，机器准入与step核账由工程层负责；技术失败成本保留、不退款，诊断修复和实际字节审核后的合法補做不扣减原任务完成额度，不盲重试，不把效果差/正常早停当故障或据节省做搜索。

每任务只validation严格MSE改善选best、既定早停，锁定best后正式test一次，probe无test，后环派发不依test；test状态无法确认则停止自动恢复，不重复访问。保留已看test及协议修订研究历史。runner的formal_worker/evaluate/save/restore/audit_resume/init_training函数级AST不变，只code_binding纳入新增文件；OneCycle/scaled/data/time-mark/adapter数学字节保持。历史测试绑定冻结C3/147版本，原方法与断言保留，不skip/放宽。

定向验收/最终复验与失败账绑定外置amend1包，原221项证据保留原覆盖，不机械全跑。首次79项1error发现OneCycle误送type1校验，修正窄路由；随后79项1failure为即时/最终比较次数混计，现分开验证；82项定向通过，真实model/forward/backward/Adam/GPU均0。辅助语法/白名单遗漏、patch上下文/重复操作拒绝保留，不形成科学计算。模板dry-run exit0展示112/28/35/168，preflight exit2只有未review/closure/start-record绑定三项，flags=false、closure=null。READY_TO_ARM_HANDOFF=false、READY_FOR_GPU_EXECUTION=false，无新真实许可、等待器/tmux、结果根、probe/formal/test；本轮未stage/commit/push，停在候选待ChatGPT实际字节审核，不宣称已审核、closure或训练完成。

最终定向复验84/84 Passed（此前82/82及83/83通过记录同时保留），0 failure/error/skip，真实model construction/forward/backward/Adam/GPU initialization全部0。实际调度分支证明四阶段probe尾部0、AUTO_AUDIT各1，formal完整审计/数值回放/raw读取0，runtime完整manifest扫描各1；合成tmux持久等待及owned safe-stop通过。JSON/AST/bash-n/bundle SHA/git diff --check通过，全部修后SHA、完整patch、固定来源/预算/调用次数及失败账由本轮外置包绑定；该Passed是CPU验收，不是实际GPU准入、科学结果review或ChatGPT对新候选的审核。

#### closure祖先关系窄修与定向验收（候选）

据用户本轮说明，三份文档已通过ChatGPT服务器实际字节审核；随后发现启动器要求`HEAD^ == BASE`，会拒绝实现closure之后的正常文档提交。本轮只将`closure()`改为冻结BASE必须是当前HEAD的严格祖先，并实际核验live remote；保留正确分支、worktree/index clean、HEAD/tracking一致及0/0要求。祖先关系不表示已审核或获准启动，`validate_start`仍要求许可完整`closure_commit`精确等于实际当前HEAD，scope/config/plan/来源等绑定保持。BASE不变，实际执行版本仍记录当次commit，不冒充原实现closure。

正式Python、无模型/GPU定向验收实际24/24 Passed：新增15项合成Git/许可回归及9项相关既有回归，0 failure/error/skip。验证后续文档提交可通过、BASE自身/非后代/错误分支/dirty/index/tracking/live remote异常拒绝、旧commit许可拒绝；精确绑定合成HEAD的许可驱动实际队列分支，仍须287完成释放→112补做→371封存→第三轮。真实model construction/forward/backward/Adam/GPU initialization全部0，无真实probe/test/checkpoint访问。原84/84及221项证据保留各自有效范围，未全量重跑；科学配置、任务、numeric policy、scheduler、预算及W/R均未修改。增量diff、验收和SHA绑定另存`P/baseline-unified96-onecycle001-v3-amend1/closure-lineage-repair-v1/`，原19项及`documentation-review-v1`证据不覆盖。本轮未stage/commit/push，未物化可执行许可、未arm/start，修后新字节停在ChatGPT review点。

#### 启动前远端核验与运行期本地核验（候选）

随后审核确认：前轮将`ls-remote`放入`closure()`，经`validate_start`/`validate_permit`进入逐任务和worker路径；前轮24项证明版本条件，未覆盖运行期联网次数。本轮将严格祖先/分支/clean/HEAD-tracking/0/0及精确commit许可检查留在本地，远端核验移至既有公开启动前`readiness_report`→`readiness(live_remote=True)`路径；原start/arm包装器仍先调用真实preflight，CLI无需修改。一次合法preflight仅查远端一次，30秒超时、不交互、不fetch/pull、不自动重试；远端不一致、缺失、不可达或超时明确阻止启动。已启动等待器的本地readiness、许可、任务配置及worker检查均不联网；正常文档后继、固定BASE、scope/config/plan/来源和287→112→371→第三轮顺序不变。

正式Python定向复验实际30/30 Passed，0 failure/error/skip：保留前轮相关语义，将远端拒绝断言迁移至实际preflight，并新增6项调用路径回归。真实公开CLI preflight远端查询1次；实际运行分支中35次permit校验、17次make_config、17次validate_worker及run_group/wave检查远端调用0次，完整四阶段队列也为0。preflight后远端不可用仍可作本地核验，本地代码/HEAD或许可不符仍拒绝。首次30项1error为合成Git根误用于读取真实代码清单，测试夹具分离版本操作与N代码读取后复验通过；原失败日志保留。真实model construction/forward/backward/Adam/GPU initialization全部0，无真实probe/test/checkpoint访问。前轮证据、科学材料及W/R保持，增量另存`P/baseline-unified96-onecycle001-v3-amend1/preflight-remote-repair-v1/`；本轮未stage/commit/push、未物化可执行许可、未arm/start，修后候选待ChatGPT实际字节审核。

<a id="m6-ms-seal-m128-recovery"></a>
#### MS封存重复键修复与M128恢复候选

本次故障与科学修订分开登记。原W版本`df6a16403e10d51097db8c88829909c533d15652`在SEAL_MS_BOUNDARY因`TypeError("dict() got multiple values for keyword argument 'protocol_sha'")`退出：统一链与type1链的seal同时显式传入protocol_sha并展开已含此字段的dynamic。N修后dynamic只取一次，验证其中protocol_sha精确等于配置digest，再完整写入；错误digest仍拒绝，不删除字段、静默覆盖或catch后当成功。W原失败字节、failure/controller/许可/probe/日志及203项MS产物均保留；本次不重新训练、validation或test这些MS项。

旧MS来源核验实际203项、七组各29项、1421个标准artifact引用，绑定原训练commit、完整任务/配置、manifest/history/budget/runtime、原probe/admission及记录的validation-selected best/test一次。checkpoint仅checksum，不反序列化；不重新完整数值回放。原owner36799/start_ticks171530843及续接owner64037/start_ticks198463305退出，登记owned实例共661条（含原监控metadata_unavailable样本：ticks未知且当前PID不存在；不猜ticks，PID重新出现即拒绝）。原M probe/queue/result及补做、第三轮计算不存在。旧MS正式实际Adam/backward=8,065,706、forward=9,115,302；原MS probe实际756/756/1024。消耗保留，不退款、不抵扣原任务完成额度。原失败不是科学/效果失败；续接失败保留真实上游failure原因，没有发信号或重新退休旧链。

用户在本条明确批准未执行M84全部train/eval batch128。逐项effective profile独立复算：57项训练batch改变，27项原本128；科学diff白名单仅training.batch、training.eval_batch及必然派生scheduler.steps_per_epoch，结构/L96/H/epochs10/patienceNone/seed2024/Adam基础设置/数据/marks/评价/numeric policy不变。OneCycle按新train_batches展开，update数量与LR/beta1轨迹因此变化，不能称仅显示值修改。新增ETT84与第三轮231科学diff均0；Weather20相对新M128 Weather10仅epoch/scheduler.epochs至20，相对前版amend同步batch及派生steps变化，不能加载10轮checkpoint续训。

隔离准备包为`P/baseline-unified-v3-ms-seal-m128-recovery1/`，未来结果根为`E/baseline-unified-v3-ms-seal-m128-recovery1/`，session为`ch3-baseline-type1-followup-v3-recovery1`；总scope为`m6-baseline-type1-followup-v3-recovery1`。M基础执行身份为`baseline-unified96-onecycle001-v3-mbatch128-recovery1`，补做为`baseline-unified96-onecycle001-v3-amend1-m128-recovery1`，第三轮科学方案仍是冻结type1-followup-v3，仅恢复执行revision改变。仍是第二轮修订与第三轮，不新增第四轮。两条旧失败根不复用、不覆盖，也不清除failure/STOP。唯一MS故障入口只接受登记failure、完整来源与退出证明，不提供通用ignore_failure。

实际队列顺序：MS203只读导入/恢复封存→M12884 probe/AUTO_AUDIT/formal/test→混合来源基础287封存→112补做→371来源封存→第三轮Urban28→EPF35→M168→COMPLETE。导入seal保存旧training_commit/protocol/code及新的seal_execution_commit/code；新M、补做、第三轮记录未来实际closure和自身config，不将全部来源强标成旧commit。第二轮计划399、有效371、第三轮231不变，余下新正式427；不额外重跑ETTh1。Weather28统一采用20轮batch128，保留新Weather10引用，不按指标选择。用户未来只arm一次，各技术通过自动继续，效果不控制后环。

正式helper独立预算如下，数字为最坏上界，早停actual另记；未读取test标签作算术。

| 阶段 | runs | max run-epochs | Adam | backward | forward |
|---|---:|---:|---:|---:|---:|
| M_BASE | 84 | 840 | 108080 | 108080 | 128877 |
| M_AMEND | 112 | 1400 | 325920 | 325920 | 410445 |
| URBAN_SUBSET | 28 | 560 | 1028440 | 1028440 | 1143128 |
| EPF_ALL | 35 | 350 | 399000 | 399000 | 467845 |
| M_ALL | 168 | 1960 | 354480 | 354480 | 446642 |
| 余下总链 | 427 | 5110 | 2215920 | 2215920 | 2596937 |

probe cap按上述阶段分别为1512/1512/2016、2016/2016/2688、504/504/696、618/618/824、3024/3024/4032，合计7674/7674/10256（Adam/backward/forward）；含原预登记resource回退，不含无限重试。共106组427代表。各阶段probe尾部完整审计0、AUTO_AUDIT1，formal不完整回放；原即时numeric/finite/identity等gate保留。q<=4，仅resource降q，不自动降batch/改精度/结构/阈值；单任务batch128无法容纳时停止。实际资源准入仍待未来GPU probe，不将CPU验收冒充Passed准入。

本轮正式Python定向最终62/62 Passed，0 failure/error/skip，含54项新恢复回归及8项受影响的既有Git/许可/远端保护回归。合成fixture驱动实际seal、MS导入、混合287来源、371固定替换、队列、probe/AUTO_AUDIT及permit/config/worker/group路径；验证full dynamic字段集合、防错digest、缺失/重复/错来源/篡改/存活worker/未知test拒绝，旧许可不能授权新配置。五阶段完整审计各1，runtime启动manifest扫描1，formal完整probe回放和运行期联网0；公开preflight远端1次且30秒有界超时。合成tmux及owned safe-stop通过，旧scope信号0。runner函数级AST仅code_binding增量，模型forward/loss/evaluate/数据处理、scheduler实现及全部模型源码保持；测试禁止实际torch/models/layers导入，真实model construction/forward/backward/Adam/GPU初始化全部0。

失败账保留：首次来源核验遇原样本缺start_ticks，改为明确未知且PID不存在的保守退出证明；首轮42项2error为尚未物化plan材料，第二轮48项1error为合成probe缺approval.json，补齐fixture后复验。随后probe职责2/2、runtime路径3/3及最终62/62实际结果独立保留，不拼接成Passed、不删断言/skip/放宽阈值。原84/221/30证据保留各自有效覆盖，不机械重跑全历史。

模板flags全部false、closure=null；实际dry-run exit0展示完整计划，模板preflight exit2拒绝未审核/未closure候选，READY_TO_ARM_HANDOFF与READY_FOR_GPU_EXECUTION均false。新结果根、可执行start-review、launcher日志及真实session均未生成，未stage/commit/push，未真实启动恢复/probe/formal/test。完整diff、SHA、effective profile差异、来源核验、预算、测试/失败账及未来操作命令集中绑定上述新准备包；本轮新字节仍待ChatGPT服务器实际直读审核。

<a id="m6-pid-identity-repair"></a>
#### 旧进程身份归并窄修（2026-10-07 UTC+8，未提交候选）

上述19项实现随后审核并closure。2026-10-07 06:39 UTC+8的实际公开preflight返回exit2：`old owned instance live or unverified PID reappeared`；许可及原始输出保留在恢复包的`start-review.json`和`start-preparation-v1/`。当次未记录命中PID及即时身份，不能据后续PID均不存在改写为通过，也不能认定当时必然由某个Codex/preflight进程复用造成。

原`ms203-source-verification.json`保持原SHA和661条历史采样登记，其中332条有ticks、329条缺ticks。对56个formal wave和84个probe wave的原始`process.json`/`memory.jsonl`逐wave核验后，340次缺字段采样涉及的329个PID均在各自同scope、同wave的注册与完整采样中唯一关联到ticks、namespace和host PID；未发现未关联项或冲突。归并后是332个唯一进程实例，包括原controller、group child和已失败续接controller。缺字段记录仍保留为当时监控事实，不作为独立、永久未知的进程身份，也不跨wave按PID全局补ticks。

窄修仅在`ch3_type1_upstream.py`与`ch3_ms_seal_recovery.py`统一身份判断：轻量检查消费固定SHA绑定的小型归并证明，完整snapshot重新核对注册/采样引用及原历史投影，再调用同一存活判断。两次有界`/proc`读取须给出稳定身份或稳定不存在；同一旧PID/ticks仍活跃则拒绝，可确定的不同ticks新实例不误判成旧实例；无可靠关联、多候选、namespace冲突、权限不足或读取中身份变化仍阻塞。诊断只给PID、历史/当前身份、scope/wave、来源及拒绝理由，不打印环境变量、token或无关命令。不发送信号。

增量证据位于原恢复包的`pid-identity-repair-v1/`，归并证明绑定原SOURCE_REF及140个wave的注册/采样来源。旧SOURCE_REF、MS1421项完整性证据、两条旧failure、19项送审证据和启动准备exit2原文均保留；未再次checksum真实checkpoint、反序列化或回放probe。旧许可仍绑定原closure，不能授权未来修后commit；本轮不生成新可执行许可。

正式Python最终无模型定向验收45/45 Passed，0 failure/error/skip，含新增18项身份测试及27项相关来源、许可、worker、队列、来源索引和审计职责回归；补充冲突诊断最多4个候选的上界后复验最终字节，此前44/44通过日志独立保留。合成收据与进程观察驱动真实公开preflight及snapshot/verify_source，验证PID复用场景下轻量与完整导入结论一致；未整体mock这些入口。测试限制真实torch/models/layers导入和信号操作，真实model construction/GPU/forward/backward/Adam/validation/test/恢复封存/信号均0。原62项仍保留其有效覆盖，本次不将它们拼成新的全量Passed，也没有重试真实失败preflight。

M84 batch128、补做112、第三轮231、各自scheduler、numeric policy、科学配置及正式/probe预算不变；MS→M84→基础287→112→有效371→第三轮顺序不变。canonical与AGENTS保持；本轮仅4文件增量，未stage/commit/push、未arm/start，修后字节待ChatGPT实际审核。

<a id="m6-probe-schema-repair"></a>
#### M128 validation schema兼容与精确probe断点恢复候选

PID窄修已closure于`fe3def16a3e19928e7c538e223d766751721cbe2`并经用户启动；MS203已成功导入封存。新的故障是`M_BASE-probe`子进程在第13组TimeMixer-ETTh1-H96的serial六步轨迹自比较时抛出`ValueError('validation schema changed')`，不是已证实的数值或效果失败。实际旧现场保留12组局部Passed、97份trajectory（前12组96份与TimeMixer单份serial），预算实际582 Adam／582 backward／776 forward；所属supervisor、probe-child及97个有PID/start_ticks的worker实例均退出。M基础84、补做112、第三轮231正式训练/test仍未开始；原failure、日志、许可、MS封存和probe文件不改写。本轮冻结旧现场SHA作为明确恢复来源，不通过删除failure或重复arm旧根恢复。

比较器按已验证task/resolved profile的`task=='M'`识别，与现有evaluate路由一致；核对任务归属、MS/M一致性、全通道metric_scope、supervised_channels、output_order、C、诊断字段、通道名称/顺序/元素数、finite及聚合一致性。M的七个validation字段全部保留落盘，仅比较器内部投影全通道聚合指标；MS_target_diagnostic仍为诊断，不新增逐通道效果gate。聚合一致性的binary64累加舍入检查不改变串并发numeric阈值；exact、历史named_tensor及full_float_state规则、完整state/gradient/Adam覆盖和scheduler/LR/beta exact规则不变。五份科学配置、任务profile、数据metadata、numeric registry、scheduler实现及正式/probe预算原字节保持。

准备阶段用原保存证据只读复现旧schema错误，再对精确旧前缀做一次有界离线核验：48次serial/q4比较结果保持有效，TimeMixer-H96作1次自检查通过，共49次。完整97份轨迹/预算/runtime/guard、61份波次资源与makespan证据及1419个probe artifact引用核对；没有模型计算、训练checkpoint读取或正式test。TimeMixer自检查不等于该组q4 Passed，M阶段仍须21组完整通过。证据增量在`P/baseline-unified-v3-ms-seal-m128-recovery1/probe-schema-repair-v1/`，保留原生产commit/config/scope/path，另绑定新比较器AST、来源及采用关系。

恢复入口为N的`scripts/ch3/start_probe_schema_recovery.sh`，计算输出为`E/baseline-unified-v3-ms-seal-m128-recovery1-probe-schema-r1/`，session为`ch3-baseline-type1-followup-v3-recovery1-probe-schema-r1`；日志、launch/claim、队列锁和fixture也隔离。科学protocol/task ID及427正式任务集合不改，这是同一恢复链的新attempt，不是第四轮或新的科学协议。旧许可不能授权修后worker；新许可仍须审核closure后精确绑定实际HEAD，当前只生成不可执行模板。旧数值sidecar只允许精确来源目录和SHA目录表，不能任意跨目录读取或通用忽略failure。

实际接线复用旧MS封存，通过新的owned生命周期记录引用，原训练与封存执行版本都保持；不机械复查MS checkpoint或重训/retest。probe预算以历史实际582/582/776初始化，另记new_actual；12组不再派发，已完整TimeMixer-H96 serial不再派发，名义仅71份新增短轨迹（426/426/568）。原12组及该serial的离线复核在最终AUTO_AUDIT消费绑定结论，不重复数值比较；其余新/混合成员按原比较器验证。probe尾部完整审计0、最终AUTO_AUDIT1，manifest及每阶段生命周期完整性扫描保持，formal仅消费紧凑summary/permit/runtime引用。原caps1512/1512/2016不增加，预登记resource-only回退不改；本次可证最坏新增107份worker（642/642/856），合旧最多1224/1224/1632，技术失败仍停止，不自动降低batch/精度/模型或放宽容差。

顺序保持：复用MS203封存→补齐M_BASE probe→21组AUTO_AUDIT/准入→M84→基础287封存→补做112→有效371封存→第三轮231。未完成21组不能生成整个M阶段准入，效果不改变后续派发。正式Python最终定向45/45 Passed、0 failure/error/skip：25项schema/恢复/准入接线及20项既有PID/结果来源回归；合成现场驱动实际run_probe、AUTO_AUDIT、manifest、permit/runtime/config/worker、封存与整链入口，验证97份旧证据不派发、71份名义新测量、21组完整报告、AUTO_AUDIT一次、formal数值回放0，及numeric/finite/identity失败立即停止。新入口marker的合成tmux生命周期与owned安全停止仅操作本轮夹具进程，未signal历史或无关实例。首次23项2error为测试夹具的历史named_tensor调用层和type1状态字段错误；第二次23项1error由测试期间文件清单变化触发worker来源保护；第一次45项1failure为历史preflight夹具误读当前真实失败输出。只修测试夹具并隔离历史输出，旧断言不删、不skip、不放宽容差；所有失败日志保留，最终稳定字节独立复验。原62/45等证据保留其未受影响范围，不拼接成新版本Passed。AST/JSON/bash-n/bundle/diff-check通过；不可执行模板dry-run exit0展示427项，preflight exit2按未审核/未closure保护拒绝，两个READY均false。本轮未stage/commit/push、未物化可执行新许可、未arm/start、无新真实模型/GPU/probe/formal/validation/test，候选待ChatGPT实际字节审核。


<a id="m6-moderntcn-etth1-numeric"></a>
#### ModernTCN–ETTh1–M数值失败诊断与2e-4政策候选

本次父版本为N `e73545ecfb717284023bc14f00993b91b2c75bf1`。schema恢复已经用户启动，MS203封存保持；M_BASE前15组实际Passed，第16组ModernTCN–ETTh1的四H串并发比较在原full_float_state、state_atol=1e-4下真实失败，未改称resource failure。当前失败链的128份短轨迹与768 Adam／768 backward／1024 forward全部保留，owner/child及登记worker共130个进程实例以PID/start_ticks/namespace核验退出，没有向旧链发送信号。M84、补做112和第三轮231正式训练未开始。

独立逐张量复核四H全部六步浮点state、gradient、Adam moments及exact项：初始/RNG/batch/order/OneCycle配置、step、LR/beta1和非浮点状态一致，均finite。原最大state差依H96/192/336/720为1.2704776600003242e-4、9.125238284468651e-5、1.0902760550379753e-4、3.3357180655002594e-5；H96及H336仅第6步stem Conv1d bias第63号元素超原阈值。第1步差异限于stem weight及其gradient/Adam moments。普通probe的保存validation来自第2步，不能当作第6步末端评价。原失败路径、坐标、值和差值在`moderntcn-etth1-numeric-diagnosis-v1/offline-state-review.json`逐项保存。

本次用户明确授权固定算子诊断及一次短确认。仅在隔离诊断worker中使用FrozenConvReplay：输入[896,1,100]、stride[100,100,1]、weight[64,1,8]、上游梯度[896,64,24]和dtype/逐次operand SHA固定；当前cuDNN enabled=true、benchmark=false、deterministic=false、allow_tf32=true。当前模式4次反向weight梯度最大差5.587935447692871e-9、forward完全相同；仅改deterministic=true的对照4次梯度完全相同。flags全部恢复，训练模型/optimizer/RNG before==after；不将后端改动写回生产设置。该固定输入实测证实此Conv1d反向存在有限浮点非确定性，不仅依据警告推断。保存的近零bias梯度、weight_decay=1e-7及原Adam moments按原eps=1e-8、OneCycle动态beta1作binary64重建，bias更新残差最多约2.32e-10，与差异放大相符；重建不是独立 optimizer运行，也不宣称证明所有后续差异的唯一原因。

独立确认只有一次固定seed2024、四H serial+q4，共8个六步worker。第6步评价重用第2步两份合成batch，核验model/optimizer/RNG和batch SHA及train/eval模式均保持；没有真实数据validation/test、正式checkpoint访问或额外整epoch。四H完整state/gradient/Adam、原loss界和第2/6步全通道指标得到：

| H | state最大绝对差 | loss最大绝对差 | 第2步MSE差 | 第6步MSE差 |
|---:|---:|---:|---:|---:|
| 96 | 1.591164618730545e-4 | 1.1920928955078125e-7 | 1.2218957334830804e-7 | 1.5537624253880722e-7 |
| 192 | 6.704777479171753e-5 | 1.1920928955078125e-7 | 3.745490140261154e-8 | 5.9583270539675937e-8 |
| 336 | 6.018020212650299e-5 | 0 | 9.267338008100978e-10 | 4.547595899850876e-8 |
| 720 | 6.423145532608032e-5 | 1.1920928955078125e-7 | 5.7828080102240165e-9 | 9.255777078109872e-8 |

MAE及归一化聚合检查同样在1e-6内；对应资源、进程归属、退出和q4收益通过。q4为73.325秒，即使保守排除含算子重放的H96 serial，其余三个serial合计139.486秒仍更长。短检查不能外推全训练逐位一致或科学效果。既有restricted guard、SharedBudget、组锁和资源监控实际执行；新增48 Adam／56 backward／88 forward（含8次隔离卷积反向和8次卷积forward）独立记账，8个worker已退出，无重试或择优。

条件成立后只生成M_BASE/ModernTCN/ETTh1/M四H的项目技术政策候选`moderntcn-etth1-M-full-state-2e4-diagnostic-r1`。state_atol仅1e-4→2e-4；loss/metric_atol=1e-6、rtol/loss_rtol=0、equal_nan=false，其他finite/exact/覆盖要求不变。这不是作者推荐阈值。独立配置`configs/ch3_round2_m_batch128_etth1_numeric_r1.json`保留84项全部effective科学profile、数据metadata、来源、batch128、OneCycle、seed及预算；原配置原字节不改。M_AMEND、第三轮M_ALL、新三个ETT及所有其他numeric政策diff=0，不经ETTh1模板传播。

采用证据精确保留15组原决定、ModernTCN原8份及独立确认8份，不从两组测量中挑最好值；原采集policy_sha/policy_id/producer commit原样保存，候选采用记录另列新评价policy及确认来源。前12组复用已有效48次离线复核；其余3组及当前组完成必要源码/来源/完整状态核验，数值复核本轮40次，原failure仍为失败。原MS203封存只读引用，不重训/retest或重复checksum其全部checkpoint。

新的受控恢复入口是N `scripts/ch3/start_moderntcn_etth1_recovery.sh`；准备包仍在原P的`moderntcn-etth1-numeric-diagnosis-v1/`，新结果根为E `baseline-unified-v3-ms-seal-m128-recovery1-moderntcn-etth1-numeric-r1/`，session `ch3-m6-m128-moderntcn-etth1-numeric-r1`。这是同一恢复链的执行attempt，非新实验轮次。旧许可不能授权它，当前仅不可执行模板；审核closure后才生成绑定实际HEAD和新来源/政策/计划的许可。

128份旧probe不再派发；原16组采用候选仍不等于M阶段准入，剩余5组名义40份六步轨迹必须实际完成。历史probe768/768/1024与独立诊断48/56/88不清零；种子实际816/824/1112，名义补齐后1056/1064/1432，计量继续受既有1512/1512/2016 cap保护。resource-only预登记回退、batch/精度/梯度累积/结构不改。最终AUTO_AUDIT一次，紧凑summary/manifest/runtime-admission串联formal，逐任务不重放旧张量或联网。全链仍MS203引用→21组M准入→M84→基础287→补做112→有效371→第三轮231；新正式427、第二轮计划399及有效371不变，技术完成与result_review=pending分开。

无模型定向复验最终57/57 Passed，0 failure/error/skip：覆盖范围与旧阈值拒绝、2e-4界内/超界、loss/metric/NaN/identity、128份复用与40份真实合成派发、21组完整审计/混合manifest、正式runtime零数值回放、阶段封存、精确permit、PID与运行期零联网、总链顺序及新scope合成tmux退出。CPU测试真实model/GPU/forward/backward/Adam为0，和上述已授权短诊断消耗分开。首次56项1failure+1error为历史AST测试误读current runner和None exact-policy反例夹具错误；历史断言保留并固定原closure版本，当前数学另用AST及真实调用路径验证，未删断言/skip/放宽容差，失败日志保留。证据准备首尝试把已继承字典键顺序当成执行顺序而拒绝；按精确组集合、任务/wave记录及来源核验后通过，原失败记录保留。canonical/AGENTS及W/R保持。本轮未stage/commit/push、未生成可执行正式许可、未arm新恢复总链、无新正式训练/test，修后字节尚待ChatGPT实际审核。


<a id="m6-moderntcn-M-policy-extension"></a>
#### 用户授权ModernTCN M政策扩展，Exchange保持exact（未提交候选）

用户在上述ETTh1诊断候选之后要求所有ModernTCN M任务统一新阈值，随后明确纠正：Exchange保持exact不变。本次最新采用范围因此为ETTh1、ETTh2、ETTm1、ETTm2、Weather及H96/192/336/720，覆盖M_BASE 8项、M_AMEND 16项、第三轮M_ALL 20项，共44项、11个stage/model/dataset政策条目。M_BASE及M_ALL的ModernTCN–Exchange共8项继续原exact，不转换为full_float_state。此范围取代上节仅M_BASE/ETTh1的候选采用范围，旧文按当时事实保留。

五域统一kind=full_float_state、state_atol=2e-4、loss_atol=metric_atol=1e-6、loss_rtol=rtol=0、equal_nan=false，完整浮点state/gradient/Adam moments及初始化/RNG/batch/scheduler/LR/beta1/非浮点exact要求保持。ModernTCN–UrbanEV仍exact，五EPF市场仍原5e-4；其他模型政策不变。统一阈值是用户明确批准的项目技术政策选择，不冒称作者阈值或已在全部五域证明同一算子原因。已有一次短确认只覆盖M_BASE/ETTh1；Weather、新ETT及第三轮对应组仍须在自己的数据/调度/实际并发条件下通过准入，未被登记为Passed。

在N继续同一未启动候选，M_BASE候选更新政策，M_AMEND及M_ALL使用两个独立numeric-r1配置以保留旧冻结配置和历史许可绑定。effective科学profile、task ID/集合、data/metadata/source、batch/optimizer/scheduler/seed/test合同与formal/probe计量均逐项不变。总链427、第二轮计划399及有效371保持。公共validator仅为固定SHA登记的M政策修订增加明确分支，Exchange/MS/其他模型或超2e-4变更仍拒绝，未扩大通用ignore_failure或版本平台。

新绑定及不可执行模板集中于原诊断包内`moderntcn-all-m-policy-v1/`。原诊断、前57项验收、旧failure、128份轨迹、独立8份确认与原15组决定保持；新采用证据仅刷新config/policy/计划引用，另绑定上版采用证明，不重新数值回放、GPU派发或把旧数据标成新producer。当前入口/session/未来结果根沿原候选；未来许可需在审核closure后精确绑定新HEAD和三份M配置/新政策，原仅ETTh1模板不能授权扩展候选。仍须21组完整准入，probe尾部完整审计0、AUTO_AUDIT1、formal完整probe回放0，MS不训练/retest。

正式Python无模型定向复验本版63/63 Passed，0 failure/error/skip，108.096秒；覆盖11政策/44项矩阵、Exchange两处exact与非法传播拒绝、各新域真实共享比较器1.5e-4接受及2.1e-4拒绝、原loss界、legacy配置/旧许可保护、compact worker绑定、实际合成40worker断点恢复/21组审计/manifest/正式准入/顺序及owned生命周期。前版57/57及其失败记录保留各自版本范围，不拼接新Passed。本轮新增真实model/GPU/forward/backward/Adam/validation/test全部0，未重跑已完成短确认、旧MS或原probe。本次范围较前候选增加两个M配置和两个既有validator文件，理由仅为三阶段政策绑定及旧合同保留。canonical/AGENTS、W/R不改；未stage/commit/push、未物化可执行许可、未arm/start，修后完整候选待ChatGPT实际字节审核。


<a id="m6-moderntcn-M-Exchange-alignment"></a>
#### 用户最新决定：Exchange也与ETTh1对齐（未提交候选）

用户在上节明确保留Exchange exact后，再次明确要求Exchange阈值也与ETTh1等域对齐。本次最新采用范围为ModernTCN全部M数据集：ETTh1、ETTh2、ETTm1、ETTm2、Weather、Exchange及四H。M_BASE 12项、M_AMEND 16项、第三轮M_ALL 24项，共52项、13个stage/model/dataset政策条目。本次新增适用范围是M_BASE/M_ALL的Exchange共8项：从原exact改为full_float_state，state_atol=2e-4，loss_atol=metric_atol=1e-6，loss_rtol=rtol=0，equal_nan=false；其他44项保持上版规则。完整参数/buffer、gradient及Adam moments覆盖，初始化/RNG/batch、optimizer step、scheduler配置/推进/LR/beta1、非浮点exact及finite要求保持。ModernTCN MS UrbanEV/五EPF及其他模型政策不改，不将M政策传播到MS。

旧冻结配置及上节Exchange-exact候选/日志/证据完整保留。本次在原诊断包内新增`moderntcn-all-m-policy-v2/`增量，刷新三份M候选配置及相应policy/config/plan/不可执行许可模板绑定；没有新科学协议或worktree。M_AMEND没有Exchange任务，它的数值规则不再变化，仅修订证据引用随最新总链绑定更新。原全部effective scientific profile、task ID/集合/顺序、data/metadata/source、batch/epoch/optimizer/OneCycle/type1/seed/best/test合同及formal/probe预算不变；427个新正式任务、第二轮399次计划正式运行和371个有效格保持。旧仅ETTh1及Exchange-exact五域模板不能授权新候选。

阈值统一是本次用户批准的项目技术准入政策，不冒称作者阈值或已证实所有域都因同一算子非确定导致差异。既有算子诊断和一次独立短确认只实测M_BASE/ETTh1；Exchange及其余未实测适用组仍须通过自己的独立serial reference、实际并发、numeric/finite/identity/scheduler和资源gate，不登记为Passed。原15组Passed决定、原128份短轨迹、独立8份ETTh1确认、原1e-4失败及生产版本/policy_sha/消耗均不改写；本次仅刷新明确采用关系的不可变绑定，没有重放真实数值sidecar或再执行GPU诊断。

正式Python无模型定向复验本版64/64 Passed，0 failure/error/skip，112.174秒。新增Exchange两个阶段四H及旧例外来源断言；复验13政策/52项矩阵、真实共享比较器1.5e-4接受/2.1e-4拒绝/loss超限拒绝、MS隔离、旧冻结配置和旧模板拒绝、科学字段及预算不变、compact worker绑定、合成真实40worker断点恢复/21组最终AUTO_AUDIT/manifest/准入/顺序、封存、PID身份、运行期不联网及合成owned生命周期。上版63/63、原57/57及历史失败继续保留各自证据范围，不拼接为新Passed；本轮测试没有失败尝试。完整审计仍probe尾部0、AUTO_AUDIT1、formal完整probe回放0，全部21组完成才可M84。

本轮真实model construction、GPU初始化、forward/backward/Adam、validation/test均0；未读取训练checkpoint、未重训/retest MS203。仓库增量仅三份M候选配置、恢复绑定模块、对应测试及本M6共六文件；完整候选仍13 modified+9 untracked=22、index空，HEAD仍e73545ecfb717284023bc14f00993b91b2c75bf1。canonical/AGENTS、W/R和历史artifact保持。AST/JSON/bash-n/bundle/diff-check及完整/增量patch检查见新增验收材料。未stage/commit/push、未物化可执行正式许可、未arm/start、未创建真实新session/结果根，修后完整候选待ChatGPT实际字节审核。

<a id="m6-baseline-numeric-admission-v1"></a>
#### 用户批准全部baseline常规数值准入统一默认政策（未提交候选）

本次用户将上一节ModernTCN全部M候选扩展为全部七baseline常规probe默认政策，原None/exact条目也必须转换，不以先出现exact失败或逐模型算子诊断作为采用容差的前置。默认表及唯一当前loss例外见[canonical常规数值准入政策](../AMD_EV_Thesis_Final_Implementation_Plan_v2.1.md#baseline-numeric-admission-defaults)，统一实现由`utils/ch3_contract.py`物化并校验。项目技术政策不冒称作者推荐阈值或全训练长期稳定证明；旧原始政策、采集schema/producer、失败/诊断和Closed历史原样保留。

| 当前阶段 | 正式任务数 | stage/model/dataset政策数 | 实际probe组数 |
|---|---:|---:|---:|
| M_BASE | 84 | 21 | 21 |
| M_AMEND | 112 | 28 | 28 |
| URBAN_SUBSET | 28 | 7 | 7 |
| EPF_ALL | 35 | 35 | 8 |
| M_ALL | 168 | 42 | 42 |
| 合计 | 427 | 133 | 106 |

133条全部full_float_state，M类91条、UrbanEV-MS7条、EPF-MS35条；TimeMixer–Weather相对loss例外只在三个M阶段保留，ModernTCN五EPF的例外不传播到M或UrbanEV。相对本次起点22项候选的84条None/exact全部转换；从五份未修订冻结配置审计则为86条（两条ModernTCN–Exchange已在上一候选转换），两个比较口径分别登记。resolved科学profile、task ID/集合/顺序、data/metadata/source、batch/epoch/LR/seed/best/test合同及全部formal/probe预算逐项diff=0。额外两个MS配置为必要numeric revision，只改政策及绑定，不覆盖已执行/历史许可引用的原配置。

证据增量集中于原诊断包`baseline-numeric-admission-v1/`；上版`moderntcn-all-m-policy-v2/`的inventory/patch/acceptance及全部原诊断保留。按登记wave引用只读核验原128份trajectory和完整schema：参数/buffer、gradient、Adam moments覆盖和严格初始化/RNG/batch/order/scheduler/LR/beta1/非浮点身份无缺口；前15组旧exact或更严格浮点界包含于新默认，建立`policy-inclusion.json`，无需再读数值payload比较。ModernTCN–ETTh1原1e-4失败仍保留，采用原完整轨迹在2e-4下的有效复核及独立8份确认两份依据，不从多次测量择优。新采用记录分别绑定旧collection和新evaluation政策/引用，未改写producer、原trajectory或policy_sha。本轮没有再运行算子重放、短确认或旧MS checkpoint全面checksum。

已有16组采用依据仍非整个M准入；恢复只补后5组名义40份六步测量，全部21组完成后一次AUTO_AUDIT，probe尾部0次完整审计、formal零完整回放。运行期closure/permit/config/group/worker只做本地绑定，公开preflight远端核验一次；必要runtime manifest扫描按生命周期一次。严格schema、finite、Adam step、非浮点/参数组和初始化/RNG/batch/scheduler等保护保持；界内非零浮点差可以Passed且bitwise_equal=false，下游不将bitwise_equal作为隐含通过条件。同一generic/full M payload只比较一次，TimeMixer–UrbanEV已有第2/6步端点保护保留，不增加其他模型评价次数。

正式Python本次最终定向71/71 Passed，0 failure/error/skip，156.834秒：统一政策专门20项（17个新增方法及3个继承真实formal/runtime入口回归），相关51项覆盖旧政策/科学字段/来源、真实合成断点恢复和21组AUTO_AUDIT、compact summary/manifest/permit/runtime/group/wave/worker、MS封存、PID实例、整链顺序和历史exact/named/full兼容。实际run_probe保留128份不派发、只派40份；新q4夹具用界内非零state差且bitwise_equal=false仍通过最终21组审计；正式运行路径审计0、远端调用0、runtime扫描1。实际writer用纯合成tensor/container/optimizer数据替身验证完整capture及结构/非finite拒绝，无真实模型或Adam计算。

本轮失败完整保留：首轮40项1failure/2error分别为冻结配置与当前候选exact计数口径、NaN异常类型和测试夹具schema修改未登记；第二轮71项1failure为历史PID公共preflight夹具仍指向真实旧失败输出，生产fresh保护正确拒绝。只修夹具隔离和准确断言，保留所有原方法/保护断言，无skip/删断言/再放宽阈值。一次单项诊断复现相同输出阻塞后纳入修正，最终同一版本71项重新完整通过，不拼接不同版本Passed。来源审计还有wave定位同时命中独立确认、以及将Markdown/patch引用误作JSON的两次辅助脚本失败，校正为精确登记wave和文件类型后通过，未改科学/原artifact。之前57/63/64及更早证据保持各自有效范围，未机械全量重跑。

全部正式仍427 fresh/最多5110 run-epochs，Adam/backward/forward为2215920/2215920/2596937；五阶段必要probe总cap7674/7674/10256，均与原合同逐项相等。历史实际816/824/1112不清零；名义新增240/240/320，补齐后1056/1064/1432，预登记resource回退保持，本轮真实新增消耗0。顺序仍MS203来源复用→M_BASE剩余probe/AUTO_AUDIT→M84→基础287→补做112→第二轮有效371→第三轮231；第二轮399次计划运行不与427个余下正式任务混淆，效果不控制派发，result_review另行。

完整候选当前17 modified+12 untracked=29、index空，沿N同一HEAD/分支，不先closure旧22项。新增七项相对原范围为统一合同模块、既有schema测试夹具、专门测试、两个MS配置和canonical/AGENTS；其余文件继承既有候选或在授权范围增量。未来入口/session/结果根沿上一候选，只使用新统一政策包内经后续审核closure物化的许可；目前只有不可执行模板。dry-run exit0展示427项但blocked，模板preflight exit2正确拒绝未审核/dirty/未closure，两READY为false。AST/JSON/bash-n/bundle/diff及patch检查、全文件修前/修后SHA见本增量送审材料。W/R、作者源码/数据/环境/tag及旧artifact不改，本轮真实model/GPU/forward/backward/Adam/正式validation/test均0；未stage/commit/push、未生成可执行新许可、未arm/start或创建真实新结果/session，候选尚待ChatGPT实际字节审核。

<a id="m6-import-path-repair-v1"></a>
#### 统一政策closure后r1父控制器包导入故障与r2窄修候选

上节29项统一政策候选已实际closure为N `3c837ba6dd82043cb2dcffdd5263e55d5588444a`。用户随后首次启动r1，在2026-10-07 19:21:40（UTC+8）停于`VERIFY_IMPORT_MS203_AND_SEAL`，错误为`ModuleNotFoundError("No module named 'run_restricted'")`。本次失败新增M_BASE测量0/40、新正式任务0/427；不是numeric、resource或科学失败。旧MS203封存、原128份短轨迹与独立8份确认未改写。r1 controller、wrapper、pane的PID/start_ticks登记与当前/proc观测核验均已退出，原session不存在，未发送信号；旧failure/progress/controller/log/launch/claimed/result及许可永久保留。

根因为`verify_production_inheritance()`使用顶层`from run_restricted import verify_bundle`，实际模块位于`tools/restricted_regression/run_restricted.py`；父控制器没有probe-child的restricted目录sys.path注入。仅改为`from tools.restricted_regression.run_restricted import verify_bundle`，不改run_restricted、sys.path、bundle封印或计算生产端。`status()`在只读`verify_registered_source()`之后增加同一生产继承检查，使公开dry-run/preflight提前覆盖父控制器实际第一步。ImportError/RuntimeError转为保留原cause的有界ValueError诊断，由既有preflight返回blocked；生产delta错误仍拒绝，不捕获后视为成功，不调用会物化边界的import_ms。

新技术attempt为`M_BASE-ModernTCN-ETTh1-numeric-r2`，结果根为E `baseline-unified-v3-ms-seal-m128-recovery1-moderntcn-etth1-numeric-r2/`，session为`ch3-m6-m128-moderntcn-etth1-numeric-r2`。日志及本次增量位于P `moderntcn-etth1-numeric-diagnosis-v1/baseline-numeric-admission-v1/import-path-repair-v1/`。科学scope仍为`m6-baseline-type1-followup-v3-recovery1`，scientific_protocol仍为`baseline-type1-followup-v3-recovery1`，不是第四轮。启动模板增加精确execution_attempt与`r1-technical-failure-anchors.json`引用，绑定r1七份保留产物及失败身份；其余source/config/plan/政策/预算不变。r1许可不能授权r2，未来许可须绑定实际修后closure，不覆盖旧许可或原送审索引。

正式Python从repo root、去掉PYTHONPATH且未手工注入restricted目录的独立进程中，实际包导入与verify_bundle返回`e3ff46b98d4752f2f94cb1b6822e37b4b08b7585942fa9ca2ad55d03ac8096d9`，torch未导入；真实冻结来源的status只读检查通过。新增11项永久回归加11项受影响既有定向回归，共22/22 Passed，0 failure/error/skip，115.420秒，本轮无失败测试尝试。真实公开CLI由合成来源/许可驱动，实际status、生产继承、bundle、readiness和validate_start均执行；bundle导入失败、错误封印和非法producer delta均在preflight blocked。检查没有创建result/controller/upstream boundary/probe/formal，真实恢复run和MS checkpoint扫描禁止。原71项及更早证据保留有效范围，不机械全量重跑。

定向回归同时核对五阶段133政策、427任务、TimeMixer–Weather loss例外、全部effective科学profile/data/metadata/source/顺序和预算diff=0；原生产数学AST保持。合成实际恢复入口不再派发128份旧轨迹或8份确认，只补5组名义40份，21组全部通过后一次AUTO_AUDIT，formal完整probe回放0、运行期远端调用0；继续M84→基础287→补做112→第二轮有效371→第三轮231。原失败和历史实际消耗保留，不因新attempt清零或扣减原任务完成额度。

本轮仅四份已跟踪文件及一份新增测试的候选变更，index空、HEAD仍为上述3c837版本。AGENTS、W/R、作者源码、数据、环境、tag、统一政策及旧原始证据保持；完整前后SHA、只读保护核验、测试日志及patch见本次repair增量。不stage/commit/push，不物化可执行start-review，不arm/start，不创建真实r2结果根/controller/session；真实model/GPU/forward/backward/Adam/validation/test和信号操作均0。当前修后字节待ChatGPT服务器实际直读审核，未声称整个M准入、正式结果或效果通过。


<a id="m6-amend-ett-identity-repair-v1"></a>
## AMD新增ETT的M声明准入及基础287后恢复候选（2026-10-07）

上节导入路径repair已closure为`760b9dd7162d200c11b8836a7c9ece41822dfcf1`并由用户启动r2。实际M_BASE complete有21个Passed组，七份正式模型组完成收据各12项，共84项；基础287封存将原MS203与该版本M84明确分开引用。随后补做首个`AMD-ETTh2-M-oc01-v3-amend1-m128-recovery1-f1-h96-s2024`在真实adapter→AMDEnhanced→validate_amd_declaration入口失败：旧允许名单仅ETTh1/Weather/Exchange，遗漏已批准ETTh2/ETTm1/ETTm2。该声明的L96/C7、H96、patch16、layernorm、target_idx6、aux空、norm开启、parallel_multivariate及S2/THLS关闭均符合合同；它是固定声明准入漏接，不是numeric/resource失败，也不是增加科学任务。

仅扩充该M域元组到六域，其余contract AST逐节点保持。全部新增AMD声明24项（M_AMEND12＋M_ALL12）与既有28项AMD M声明定向检查；未授权域/模型、错误shape/pred_len/patch/target_idx/模式/aux/norm/layernorm、启用S2/THLS继续拒绝，历史MS分支保持。公开准备readiness单次检查52个AMD M声明，不构造模型；本地worker许可检查不重新扫描声明矩阵。数值默认表、special exact合同及模型/训练/评价数学未改。

真实CPU工程smoke经过现有models.ch3_adapter.build与AMDEnhanced，不mock构造入口：AMD三个新增ETT×四H共12次构造/前向；其余六baseline×三域各H96最小代表共18次构造/前向，作者模型独立子进程导入。合计30次构造、30次合成前向；小时/分钟历史marks维度、七通道输出及两阶段build profile映射适用。小batch2只用于形状检查，正式batch128未改，不作为q4证据；backward、Adam、GPU初始化、真实数据validation/test和checkpoint加载均0。TimeMixer ETTm2仍为已冻结d_model16，未顺带追齐论文结构。

恢复增量集中于P的`moderntcn-etth1-numeric-diagnosis-v1/baseline-numeric-admission-v1/amend-ett-identity-repair-v1/`。旧r2 controller/probe-child/所属worker退出证据按PID＋start_ticks核验，旧session不存在，不发送信号；旧failure/controller/progress/log/launch/claimed/许可及所有完成结果永久保留。首个失败worker保留runtime、构造尝试和日志，记录Adam/backward/forward均0，不抹除真实尝试。

新技术attempt为`M_AMEND-AMD-ETT-identity-r1`，result根为E的`baseline-unified-v3-ms-seal-m128-recovery1-amend-ett-identity-r1/`，session为`ch3-m6-amend-ett-identity-r1`，log位于本增量`followup-launcher.log`。继续沿现有共享入口、permit/AUTO_AUDIT/runtime/summary机制；科学scope/protocol保持，不复制五份配置，不另设实验轮或工作区。启动后一次有界采用核验检查旧七份M收据、精确84任务和非checkpoint产物SHA、历史best/test调用事实；checkpoint只核对既有sealed SHA引用和已登记stat，不重新全面checksum、不反序列化。MS203复用原封存及来源核验，不重训/retest，不重放probe。

新的M_BASE采用记录保留旧训练commit、protocol、code及receipts，另记新adoption commit/owner/MAC；旧MAC与许可不充当新生命周期授权。生产继承证明精确绑定旧760b9dd字节与本次修后字节，AMD contract只允许上述域元组AST差异，其他计算生产文件不得变化。公开preflight检查旧来源/退出与该精确生产继承、qualified verify_bundle，纯只读；完整前缀采用在受控启动路径执行一次，不进入逐任务checkpoint扫描。运行期无远端查询，启动前保持一次实际live remote检查。

新总链固定为：基础287来源采用→M_AMEND必要probe及一次AUTO_AUDIT→112项fresh正式训练/test→371有效格来源封存→第三轮Urban28→EPF35→M168→technical_complete。不重新派发M_BASE的21组或已完成40份测量，不重训M84、不等待旧失败owner；本次输出343项与已完成84项合计原批准427。第二轮计划399、有效371及第三轮231不变；M84 Weather10保留，Weather20全部28项完成后整批替换默认来源，不按指标择优。第二轮索引分别记录MS旧版本、M84实际760b9dd版本及补做的新实际版本；complete不要求新目录重复产生427份结果。

合成混合来源索引检查还定位到旧M父配置digest与实际统一政策M84配置digest不同的接线遗漏。修后从封存绑定解析实际M_BASE配置，并要求tasks、resolved_profiles、datasets和sources逐项保持；M84与补做分别核对各自封存的真实producer commit，不将采用版本冒作训练版本。此项仅修正已批准来源的索引绑定，不改科学profile或数值政策。

最终当前版本无模型定向套件22/22 Passed（419.204秒），覆盖前缀采用、混合来源371、真实permit/runtime/worker接线、四个剩余阶段各一次AUTO_AUDIT、formal完整probe replay为0、运行期远端查询为0、公开preflight查询一次、失败停止及新入口合成tmux owned safe-stop。适用的真实CPU smoke3/3及受影响既有保护50/50分别登记；三项历史四阶段fixture另在冻结1134611版本3/3核验，不拼为当前单次验收。六次失败测试尝试及修正依据保留在failure-ledger与原日志，没有删除断言、加skip或放宽容差。五阶段427项scientific profile和133政策逐项不变；剩余343项最多4270 run-epochs、85个准入组，正式及probe派生上限见scientific-invariance，原预算不改。

当前修后只是12文件候选（8 modified＋4 untracked），尚未stage/commit/push、生成新可执行start-review或启动新链；已有效历史证据按适用范围复用，不机械重跑全部历史milestone。后续需ChatGPT实际字节审核、精确closure及实际许可/preflight，然后由用户最后一次arm。未来操作材料见本增量operations.sh，已语法检查并标明审核closure及许可就绪后启用，不预填未来commit或许可SHA。
