# TimeMixer基线配对来源与训练接线核对

绑定起点：`491ce9848575f744125def9950e994a9bc5a65e8`，branch=`M6-Timemixer`，worktree=`/public/home/yueweiting/大论文/TimeMixer-U+V-v1`。这是M6内既定训练准备的技术证据，不是另一份状态文档、正式训练配置或启动许可。首对实现与迁移审核Passed由用户本次通知；本轮新增材料尚待ChatGPT审核。第一段精确26文件closure的三端0/0及当时clean边界独立保存于外置`M6-Timemixer-closure-r1/closure-result.json`；第二段未提交材料不改变该历史事实。

## 来源范围和可用性

[baseline-sources.json](baseline-sources.json)逐格保存37个B对照的真实任务身份、来源ref/SHA、训练commit、config/profile/protocol SHA、完整结构/训练元数据、history轮数、best、保存指标和两轮配对候选。范围为UrbanEV-F4 fold1/4×四H的8格、六M域×四H的24格、五EPF市场H24的5格。另16个Urban格只保留索引定位，没有打开它们的run文件。

采用链从当前合法第三轮科学合同及实际upstream boundary定位到`round2-main-results.patchtst-enc2-dm4-selected-v1.json`（SHA `e6a5729c0856882f237c9f53c989a6b98557900794f15e4b27deb5fe7cb3c876`）。该371格索引的53个TimeMixer单元与此前固定enc2索引逐项相同；本轮需要的37格为original_v3 13、base_m128 8、amend1 16，不按指标选来源。dm4/enc2只替换预先决定的PatchTST来源，没有把它们登记成TimeMixer变体。

第二轮37格均有采用索引绑定的manifest/result/runtime以及对应history/budget；本轮核对了引用SHA、task/seed/fold/H/M或MS、精确profile、canonical config digest、执行commit、runtime完成、连续history、严格validation MSE选best、合法早停/跑满上限和保存的test-once记录。37格均可进入后续配对审查；采用索引`technical_complete=true`，`result_review=pending`。本轮结论仅为**结构化元数据与采用链核对通过**，不升级成全artifact或科学结果审核Passed。

实际训练commit为：MS13格`df6a16403e10d51097db8c88829909c533d15652`，基础M128的8格`760b9dd7162d200c11b8836a7c9ece41822dfcf1`，amend1的16格`b97392a179b55b4bffb2bf0e3b85267bb0b07750`。每个commit的六份关键runner/adapter/data/marks/contract/M评价源码Git blob SHA与manifest source对应项一致；完整明细见JSON。

amend1的16份manifest source没有直接列出最新Weather-patience配置的原始文件SHA。本轮如实保留该字段为null：实际读取配置的完整SHA另行绑定，其canonical JSON digest严格等于manifest/result/采用索引的protocol_sha，精确resolved profile也相同。不能将这个内容身份核对写成“原manifest已直接记录该原始配置文件SHA”。

第三轮核对的是当前合法attempt `THIRD-PatchTST-dm4-Urban6-four-H-reviewed-q-r1`：controller/progress scope、实际upstream adoption commit、选定main/boundary、三个科学config ref及两个STOP旧attempt信用0均一致。读取时为`URBAN_SUBSET_FORMAL`；按37个确切任务路径查验，TimeMixer结果尚不存在，故37格全部Pending。未等待其完成，未把旧两H/停止四H、展示隔离区或其他被替换结果纳入来源。

两轮均绑定作者TimeMixer commit `e24610583b36fdd8c76cc17a8df4e65759a5f460`，入口SHA `817d62f4aac54c8566e560f6d3785856e31c8ee51460279ee3d0a4823f11d4be`；只读验证实际作者文件及依赖SHA。没有重取论文、修改作者代码或追齐论文推荐结构。所有实际读取文件的SHA、角色和引用校验记录见[source-bindings.json](source-bindings.json)。

## TimeMixer专属实际协议差异

下表取第二轮实际采用run manifest及第三轮最新有效科学config的resolved profile。`E/p`为最大epoch/早停patience，None仍评价validation并选best。所有train/eval batch相同，均为seed2024、fresh、Adam、accumulation1、float32、workers0；这些不是本候选已经选择的训练参数。

| 范围 | 第二轮L；batch；E/p | 第三轮L；batch；E/p | 原结构D/尺度数S；主要差异 |
|---|---|---|---|
| UrbanEV-F4（开发f1/4，全H） | 12；128；20/5 | 12；128；20/5 | 64/3；主要为OneCycle与type1调度差异，P始终1 |
| EPF PJM/NP/BE/FR/DE，MS H24 | 96；128；20/5 | 168；32；10/3 | 64/4；L、batch、上限、早停及调度均不同 |
| ETTh1，M四H | 96；128；10/None | 96；128；10/None | 16/4；调度差异 |
| ETTh2，M四H | 96；128；10/None | 96；128；10/None | 16/4；调度差异 |
| ETTm1，M四H | 96；128；10/None | 96；128；10/None | 16/4；15分钟历史marks，调度差异 |
| ETTm2，M四H | 96；128；10/None | 96；128；10/None | 16/4；保留冻结D16，不改成论文Table7的32 |
| Weather，M四H | 96；128；20/10 | 96；128；20/10 | 16/4；最新patience10，两轮调度不同 |
| Exchange，M四H | 96；128；10/None | 96；128；10/None | 16/4；调度差异 |

第二轮实际OneCycle：profile lr/max_lr=0.01，初始化LR=0.0004，pct_start0.2、cos、cycle_momentum=true、base/max momentum0.85/0.95、div25/final_div10000、three_phase=false；每次成功optimizer update后推进。每格steps_per_epoch保留真实profile值，没有统一替代。第三轮initial_lr=1e-4、固定基础betas；`lr(e,E)=1e-4*0.5**(max(e-2,0)*c(E))`，E<=10时c=1，否则8/(E-2)，在validation/best更新后且继续训练时调整。E20的c=4/9，调度长度不因早停重新缩放。两轮weight_decay=1e-7、betas基础[0.9,0.999]、eps1e-8；OneCycle的beta1轨迹与type1不同。

Weather实际history为H96完成20轮/best11，H192完成13轮/best3，H336完成15轮/best5，H720完成16轮/best6；后3格按patience10合法早停，不能全部写“实际训练20轮”。旧正文20/None、旧Weather10和早期第三轮231/287格范围只作历史；当前第三轮为Urban六fold全四H、EPF五市场及六M域371格。本轮不会从旧正文抄默认值或复用被替换Weather10。

两轮所需37格的features、target/aux顺序、native marks规则及structure字段逐格相同；EPF实际序列长度由各自协议继承。ETTm1/2历史timeF为K5/freq=t，其余所需域K4/freq=h；marks与业务C分开。Urban C11为特征数，275节点在样本维展开；所读fold1端点576/648/720、fold4端点2342/2634/2928，属于累计月份rolling折，不能写成两个独立月份。

## 评价空间和未完成审核

保存MSE/MAE均为`train-standardized`空间。Urban volume按各fold训练段逐节点标准化，价格用train min-max，天气按训练段标准化，calendar与历史native marks保留；M/EPF按各自训练段StandardScaler，split/window及节点顺序来自保存metadata和绑定data源码。本轮没有重新读取真实文件拟合scaler、验证数据SHA或重做test；配置声明的数据SHA及保存scaler元数据以引用身份记录，不能冒称重新完成原始数据审计。

MS计目标通道全部预测元素的SSE/SAE除以元素数；M计所有变量×预测位置×样本元素的总误差，另有逐通道指标及MS-target诊断，诊断不替代M主结果。已有单run MSE/MAE不是原物理空间，也不是先对fold/H等权的结果。开发Urban每臂/输入方案8格等权；后续正式24格等权重判；六个M域各自四H等权，五EPF各自H24。不得跨域原始MSE平均，也不把报告两项指标升级成双硬门槛。主评价指标保持未决定；既有B checkpoint按validation MSE选择是既有协议事实。

本轮读取了已保存的37格test指标与validation history，记录为历史test信息访问，不是独立盲测，也没有新评价test。二进制checkpoint/数值sidecar、全artifact完整性及独立result review不在本轮核验范围；没有加载、扫描或重新计算checkpoint SHA。下一步须统一协议并定向完成配对/结果审核；不得按某轮B较弱、按数据集/H效果择优或混用协议解释U/V收益。当前没有选择第二轮或第三轮，也没有批准两套训练。

## 非执行计划与训练接线

[开发计划JSON](../../../../../configs/timemixer_uv_development_plan_v1.json)只有77项覆盖：目标一F4的U/V/UV 24项，目标三UV的F1/F2/F3 24项，目标二UV的M24+EPF5共29项。目标三F4复用目标一同身份的8个UV结果；没有B重训、没有模型×输入的笛卡尔积、没有其余Urban四fold任务。后者96项只是三个开发目标达标审核后补齐的条件说明。EPF L显式未选，保留round2=96/round3=168两种来源选项，不假装已选可执行结构。[metadata-validation.json](metadata-validation.json)记录16项纯元数据/有理权重核对通过，没有模型构造、梯度、训练或真实数据评价。

| 现有能力/位置 | 可直接复用的事实 | 后续具体接线缺口（本轮未实现） |
|---|---|---|
| `timemixer_dev.adapter` | build_from_task解析既有TimeMixer profile，prediction明确MS切片/M全通道，loss拒绝广播；四臂和合成strict恢复已审核 | 训练入口需独立candidate ID/四臂路由与结构spec；当前build_from_task只接受原TimeMixer任务作结构来源，不能把增强对象登记成B |
| `ch3_runner.init_training/update/evaluate`、`utils.ch3_m_tasks.evaluate_all` | Adam、finite检查、同形监督、MS/M误差计数均存在 | 当前构造/预测硬接`models.ch3_adapter`，仍构造原B；新增candidate需要薄接线覆盖构造、更新、validation/test四处，保留原B路径 |
| `utils.ch3_data.load/Windows`、FeatureSchema、native marks | train-only scaler、Urban节点展开/H偏移/P1、标准多步窗口及历史marks已有 | 为F1/F2/F3按既有顺序派生独立candidate任务/profile/数据身份；不能直接用现有F4-only执行配置，不能增加未来真实输入 |
| 既有OneCycle/type1、BestState与formal worker | 已有scheduler、严格validation-best、早停、best-selected test-once和预算记录 | 统一选定协议/指标后绑定candidate计算及参数组，不改已有B科学profile，不生成正式许可或假artifact |
| 既有save_state/restore_state、manifest与source binding | 有模型/optimizer/scheduler/RNG/identity保存恢复及strict机制 | 本包仅有合成state_dict合同；真实artifact必须加入candidate/use_u/use_v/特征/L/P/H/来源及实现SHA，防止跨臂/输入/协议恢复，不能直接以synthetic checkpoint helper冒充正式恢复 |
| 现有固定q4适用历史证据、启动与审计设施 | 能作为后续资源/并发审核依据 | 需审查新U的成对张量及V位置权重资源、具体训练入口与独立授权；缺新probe不自动阻塞，也不能凭本计划取得执行许可。本轮不扩建队列/runner/许可平台 |

CI=1作者marks按C再b重复，与变量b再C展开有已登记排列限制：b>1且样本marks不同会改变样本/变量的对应关系。当前已审核候选与历史B均保留该行为，四臂需同等提供marks；本轮不修。若后续希望纠正，必须先裁决新实现/基线身份及旧B可复用性，不能把修后模型直接配对旧B并仍称同版本收益；修复或重训都需另行授权。S=1的作者PDM限制对现有S3/S4 profile不触发，V单模块S1零修正证据保留，不新增阻塞或暗中修B。

## 工具诊断与边界

首次元数据脚本误把保存的final_test字典当布尔值；实际字段为calls=1、selected=best.pt、epoch=best_epoch，已按真实schema核对。第二次发现amend1 manifest没有直接raw-config source项，现明确记录缺项及canonical digest/精确profile证明，不伪造该项。计划生成器首次误将AST Dict.values列表当方法，改正后16项元数据核对通过。失败日志、最终来源核对及计划生成日志均在外置`M6-Timemixer-training-preparation-r1/`保留；不是模型测试失败，没有修改源码/测试断言或降低来源SHA核验。

本轮没有改U/V公式、模型源码、既有测试、canonical、生产/二轮/共享来源，未重跑281项或16项模型测试。没有GPU/probe/真实train/validation/test、依赖安装或真实checkpoint读取。第二段材料全部unstaged，停在ChatGPT送审；协议、主指标、配对结果审核及真实启动授权仍待统一决定，三个效果目标Not evaluated。
