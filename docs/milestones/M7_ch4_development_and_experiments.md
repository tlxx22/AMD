# 大论文 07｜M7 第四章开发与实验

状态：**联合探针r2全覆盖完成；用户已批准正式q1/T2，资源选择固定，新接线及正式启动材料待ChatGPT工程审核。** HEAD `3e387a9a3b52af4f0ab02763977ac399c475a6f9`，本轮不stage/commit/push、无可启动AUTH、未启动正式训练/test。详见§5.P。

## 1. 权威分工、现场与保护

[canonical](../AMD_EV_Thesis_Final_Implementation_Plan_v2.1.md)§0/§1.2/§4.2/§10–16规定研究合同；本文件是唯一M7执行/审核账；docs/README只导航，第三章实验索引CURRENT_ADOPTED_RESULTS保持原结果入口。本轮不会创建第二milestone或平行current_status。论文围绕城市EV充电需求，第三章时间建模→第四章空间扩展；六城CHARGED独立域和PEMS跨域比较不改变主研究对象。M8正文/图表/分析，M9终稿/复现/答辩。

hostname=b82673a648b4。工作区`/public/home/yueweiting/大论文/CH4-v10-spatial`，branch=`M7-v10-spatial-v1`，初始parent=`ee8ea3f32e87ed48608d21991a4fc60d06891c23`。79路径closure后原基点为`75158a60956fadea9ad6d1c08caf8dbd4e3a284c`；r4精确9路径closure后的当时HEAD/tracking/实际origin均为`ee9110140a7cd2ee8c0cadbc766ea9d156ead8be`、0/0，upstream为origin/M7-v10-spatial-v1。上述为此前closure现场；本轮入口HEAD为3e387a9a3b52af4f0ab02763977ac399c475a6f9，仅canonical/M7已有未提交修改，当前新增q1/T2接线待审，index保持空。原创建前M6基点及无upstream状态仅为历史。`amd_reproduced_baseline_v1`保持`fa9665627e6fcfb1d0c2bc22d943ca9666304fd6`。

第三章工作区、v1/v3/v9/v10冻结实现、正式配置、models/tsAMD.py、M6和采用产物不修改。前后SHA/import绑定核验见外置初始身份和最终检查。集中包原producer、v9/P1/V3复用链、54缓存/235逻辑来源、Goal1/2 Met与Goal3 Not met以及test开发反馈限制保持，不因新会话失效。

## 2. 本轮批准范围与最新修订

完整v10 U/V、G5/R4、D64单block，patch1/stride1、L12/K12、large7/small3、time dropout.1/head0、RevIN affine0。新模型与v10参照fresh联合训练，不读第三章权重。独立wrapper在最终target H之后空间交互，原head与U读取同一H*，单norm/denorm；U只修正输出，不回写H。geo/sem四头×16维、geo偏移0/1和sem0、接收端82→32→8条件低秩r8、e16及Wo零初始化等按canonical精确公式。

最新用户答疑取代首版228项和各模型专属训练配方。当前18模型：七时间baseline AMD/DLinear/ModernTCN/PatchTST/TimeMixer/TimeXer/iTransformer；九空间baseline PDFormer/HimNet/TESTAM/D2STGNN/MegaCRN/STDN/STAEformer/Graph WaveNet/AGCRN；完整v10和新模型。PatchSTG全局退出，已下载来源、早期adapter副本和CPU历史证据保留，不补其他模型。

18×12×seed2024=216fresh，Urban72×50+CHARGED108×50+PEMS36×200=16200 run-epochs上限。当前候选batch对应6667200 updates上限，容量准入后按冻结B重算；patience改变不会减少预登记最大预算。正式指标仅train-standardized MSE/MAE，同一strict validation MSE-best；原单位辅助单列，无RMSE。全部Adam/OneCycle统一，max_lr.001、pct_start.1、div25/finaldiv10000、momentum .85/.95，beta2.999，梯度clip5，FP32/AMP关闭，drop_last=False/accumulation1。每epoch完整validation，严格改善重置，连续达到20/40轮无改善停止，OneCycle总步数不随早停重算。

第一轮即完整批准目标，关闭horizon curriculum和未来真值回灌；保留TESTAM专家/路由、MegaCRN记忆及AMD选择器辅助项并分账。TESTAM保留12槽并同步选择预测/专家/门控，Urban选H-1、CHARGED0、PEMS全12，不额外监督其它槽；此前将历史索引+L误判为必须新建P1头的判断已纠正，不新增该头。Graph WaveNet使用README完整图配置，不使用裸默认关闭图模块的组合。

## 3. 数据身份与实际准备

仅读取train/validation数值前缀，没有解析原始test。UrbanEV原fold6端点3475/3909/4344、275区域、F4 C11/L12、H3/6/9/12；原loader/scaler及split-local窗口复用。CHARGED取得Zenodo正式v3 Hourly，268720500 bytes，官方MD5 `099a592653ca74fa1f1051e9c23d6456`通过；六城4392小时，端点3513/3952/4392，C4/L12/P1。AMS2449、JHB47、LOA229、MEL63、SPO47、SZH1445，完整站点/城市独立。MEL/SPO站点ID列实际为site，其余site_id，已核对volume/距离/坐标ID，不按全期统计筛站。上游IQR/零序列插值限制明确披露。

PEMS采用原基准发布者wanhuaiyu/ASTGCN commit `86efe627f8d7a1d461be9446a3c9d0960c301d0b`四数据文件及README/utils来源。NPZ键data，实际shape分别[16992,307,3]/[17856,170,3]，flow/occupancy/speed有README依据；只解析至validation端点13593/14284。train端点10195/10713，5分钟L12/P12。distance.csv的from/to/cost整数索引与数组轴对应，仅称发布距离权重，未确认单位或真实道路长度，无GPS推造。

主模型geo/sem已按各域train及真实distance生成，原生baseline关系不能直接复用这些top-k/cosine缓存。当前数据目录在外置包data下；图和scaler身份见每域metadata.json。九域原生固定图与位置编码保持，PDFormer train-only DTW/模式缓存现已全部技术完成并获产物审核，按§5.K显式采用；缓存就绪不等于资源冻结或正式训练准入通过。

## 4. 来源与实现

| 模型 | 固定来源commit |
|---|---|
| PDFormer | f8c8f6ad007a04fad3baee958b89504711852ce9 |
| HimNet | 2a9aa44d1864f6fb60bdf9d14bafd6c1681e81a5 |
| TESTAM | 20ee7815a053c4a7b64dca676581d88a68ae5078 |
| D2STGNN | 82c2d38ba098a6aa87e8099c1ee8da58480692ec |
| MegaCRN | 5158190948de7e8628f84780a429eb9189f5908f |
| STDN | e24a60fa9485fb9805df9c171d67ab60840d757d |
| STAEformer | fc49d39b2f1a8e3cf37b6289d7240680e1690f3f |
| Graph WaveNet | 6b162e80c59a1d494809252eca055cff93dc66b1 |
| AGCRN | 7fbbf2aeb099242098a3cf482b55cd45d7295c28 |
| DLinear | 0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6 |
| ModernTCN | 56a9a2c018385cd5acef015378cae7f084d1b11c |
| PatchTST | 204c21efe0b39603ad6e2ca640ef5896646ab1a9 |
| TimeMixer | e24610583b36fdd8c76cc17a8df4e65759a5f460 |
| TimeXer | 76011909357972bd55a27adba2e1be994d81b327 |
| iTransformer | c2426e68ca13f74aaec08045c5c724d8ad328124 |

AMD及七时间profile实际生产身份来自CURRENT_ADOPTED_RESULTS来源账；ch4/temporal_sources.json保留当前采用的第二轮UrbanEV MS结构，历史训练字段不作为本轮训练参数。最新裁决明确七时间baseline跨全部第四章数据域沿用该结构，不适用作者默认回退；PatchTST e_layers=2，ModernTCN保持B_cfg。只有九个新增空间baseline按作者当前数据集结构→推荐完整默认入口→构造器默认解析，结构与training字段分开；HimNet EV默认METRLA，D2 EV默认METR-LA，STAEformer EV默认入口PEMS08，PDFormer EV模型默认config，MegaCRN/AGCRN/STDN按实际默认入口。完整v10及新模型不适用回退。保留原生机制，不读取作者逐epoch test训练入口。

代码范围ch4/及本轮tests。source固定原件在外置sources，必要适配放ch4/vendor：字段/日历频率/P输出、Graph WaveNet旧Conv1d四维输入兼容为Conv2d、CPU设备跟随输入、STDN P1在原calendar decoder之前作12→1输出长度适配；这些差异须送审，未宣称作者原配方复现。完整来源config/核心SHA/适配入口在comparison-plan.json及外置vendor diff。独立环境amd-execution-envs/m7-v10-spatial-r1继承已验证数值库，仅本环境新增来源依赖，未升级第三章环境或驱动/CUDA。

## 5. 验收、队列及尚未闭合事项

主模型7组CPU检查通过（P1/P12/P24 exact、真实空间执行ΔH0、一次head/U/RevIN、非target、稀疏dense、mask/边界、置换、两读出梯度与时间参数冻结）。新版协议4组通过（20/40 patience、同域一致、OneCycle固定总步及恢复、TESTAM跨午夜/时间槽/专家路由与监督范围）。CPU fixture worker已跑通训练/validation保存→完整best冻结→一次合成评价和重复claim拒绝；使用临时合成数据，不是真实训练/test或第三章权重。具体逐baseline结果及失败保留在checks；不能把先前源配置的通过自动套成修订后配置全部通过。

总启动器ch4.queue/worker/reporting组织全部train/validation→freeze→test-once→seal，复用owner PID/start、锁、STOP和receipt；真实启动前要求工程缺口清零、资源冻结、clean实际commit/tracking/远端、正式AUTH。现有源码尚未closure，无正式AUTH，不能启动。恢复helper保存optimizer/OneCycle/RNG/best/patience/epoch/update，不自动技术重试。失败日志/staging保留。

资源探针候选5准备+20计时更新/配置、必要计时最多重复一次、总4GPU小时上限；实现先审再执行。并发仅以实测显存/安全余量/OOM风险、单步及端到端吞吐、GPU忙碌与资源争用选择，q8须足够余量和实际总吞吐收益；选择后冻结batch/q/设备/线程/任务顺序。串并行参数、梯度、Adam、逐步loss及预测张量的exact/allclose/最大误差等价门禁取消，选择前后不执行，不改名为smoke/暖机/稳定性检查。资源准备更新只服务计时。资源准入和正式训练不默认保存numeric-full-step-*.bin或等价逐步完整状态转储；finite/OOM/进程归属/恢复身份/数据/test/产物校验及正常best/last保持。并发数值等价登记“本轮按用户批准不执行；基于任务独立执行机制、既有证据及差异风险评估接受”，不得登记新版实测Passed。候选B：Urban32、AMS4、SZH8、小城32、PEMS16。同域独占OOM按预登记减半序列统一降B；仅并发OOM先降q；B1仍失败阻断，不删站/缩结构/AMP。q从1比较2/4，8需实测余量与吞吐；本轮没有容量/并发实测。资源差异报告显示AMS B4时PDFormer单层geo/sem logits约6.91GB，非完整峰值。历史第三轮成本仅用户转交ChatGPT核验记录，不冒称本轮重新读取1484文件或获得新GPU实测。

七时间结构适用范围和CHARGED固定图迁移已获明确裁决，不再列为待用户决定。实际落实：逐城市float64完整D总体std（含对角零）→exp(-(D/sigma)^2)→严格<0.1置零→float32共享基础A；核对原始行列ID后显式重排，A自然含self。GWN缓存[R(A),R(A.T)]，D2缓存两者各自转置；STDN非PEMS二值图排除self；PDFormer不重复高斯化。Urban原adj保持，PEMS发布有向连接与cost分离，PDFormer来源bidir仅作用于其副本。主模型geo/sem及其它不需固定图的模型保持原机制。图身份/每模型缓存SHA和排除自环后的诊断在data/native-*-r1/identity.json。

九域固定图及8/32维位置编码已生成，维度和finite检查通过。新增graph-contract-r1四项CPU验收通过，覆盖阈值边界/合法零距离、ID与非法距离、非对称/对称不均匀度图方向、零度行、self、hop不可达、PD不重复高斯化及七时间profile保留。graph-consumers-r1中D2/GWN各P1/P12的实际support方向、前向/反向/mask验收通过。MEL有1个孤立节点、Urban有4个；这些事实未触发补边/调阈值/删站。

上轮历史：曾启动PDFormer训练期DTW/KShape的CPU预处理：仅UrbanEV train前缀，DTW完成，KShape未完成；为遵守本次固定图修订后送审停止点，核验自有PID/start及日志后SIGINT停止，未继续后续域。发出停止信号时记录进程墙钟261.84秒、user CPU278.26秒、system CPU128.77秒；拟合函数最终退出记录305.58秒。CPU读数仅截至信号时，不冒称最终总量，并行CPU秒不称GPU小时。日志、DTW、pattern-start及停止回执原样保留，没有pattern-identity完成凭据，未把该缓存写成就绪。后续需在明确准备范围中完成该缺项，不用主模型cosine替代、不减聚类数或按结果重选。独立环境增加Cython并编译同版fastdtw以支持原生预处理，依赖身份见environment-r2.json；未修改第三章环境。

当前未闭合：九域模式已按用户批准采用，12项PDFormer模式缺项已解除，本次接线增量已Review Passed；整包工程审核收口、资源探针/冻结、Git closure及正式AUTH仍待完成，没有正式训练启动命令。旧UrbanEV前缀模式及旧AMS失败、后续未执行队列继续保留历史身份；不修改旧results、启动计划或报告来制造成功。当前采用来源仅为native-cache-launch-r1的四域和five-domain-cache-launch-r1的五域，详见§5.K。

外置证据：`/public/home/yueweiting/大论文/amd-execution-evidence/m7/M7-v10-spatial-implementation-r1`，本次送审回执`REPORT.md`及`review-files.json`给出实际修改、SHA、验收和缺项。源码获取超时/API限流、CHARGED首次部分下载及分段恢复、数据列名/依赖/适配/测试工具失败均保留，不能把最终通过写成无失败。SIGINT停止模式拟合时Numba表现为SystemError，原pattern-failure及traceback保留，不冒称缓存完成。资源与效果未验，不宣布M7 Closed。

## 5.R 送审后定向工程修复（本轮，未作审核通过宣告）

新证据目录：`amd-execution-evidence/m7/M7-v10-spatial-engineering-repair-r2`；保留上轮REPORT SHA `636916e89f92a2ed67ca3e20ffc3a3c66bbf8dae49317fb5d5eff7fa83547970`及全部失败材料。分支HEAD仍ee8ea3f32e87ed48608d21991a4fc60d06891c23，无upstream，index空；本轮没有重新联网查询远端。

依赖修复：M7本地runtime_io承接小型proc_start/JSON辅助，去掉单函数导入旧launcher执行链。静态局部import闭包含包初始化及函数/条件内导入，177个Python文件是保守覆盖而非全部实际执行；另43项运行资产，实际import-only观察101个项目/来源文件均被覆盖，无跨第三章worktree混载。运行绑定、作者来源快照和退休PatchSTG历史参考分列；临时副本漏绑/篡改负例拒绝，未改真实冻结文件。

最终封存：完整任务/收据/claim/best/指标/文件集合核对，新增汇总表和来源账纳入manifest；Python完整身份验证→系统sha256sum -c→最后原子发布complete。manifest不含自身/checksums/complete，checksums含manifest但不含自身/complete，complete绑定两控制文件SHA，无循环自哈希。合成监督器实跑少量子进程，覆盖全部train→freeze→一次评价→seal、技术失败停后续、重复/不明确claim拒绝、缺失/篡改/实际checksum失败不发布成功；生产216断言保留。测试夹具首轮缺environment失败及修复后日志均保存。

定向CPU验证：repair 4项（含P1/P12同优化器与时间主干/V3/U/空间梯度及参数更新）；supervisor原4项通过，新增真实checksum负例1项通过；pattern identity/reuse 2项通过；CPU准备守护2项通过。未机械重跑原时间等价、图方向、稀疏/dense或历史回归。P1/P12受控非零Wo用于验证联合训练，未要求零初始化首步全部内部梯度非零。

UrbanEV新attempt `patterns/UrbanEV-r1`：train-only KShape16/长度3/max_iter5/seed2024，336合法历史×275节点=92400候选，实际n_iter3，16簇均非空，中心[16,3,1]有限；墙钟836.248804秒，user CPU878.344663秒，system CPU436.299158秒。未新增GPU小时或真实test访问。旧DTW仅引用，旧中断不改写为算法失败；其余AMS/JHB/LOA/MEL/SPO/SZH/PEMS04/PEMS08均未执行本轮拟合。

依赖差异披露：来源requirements指定tslearn0.5.2，原M7环境实际0.6.3。核验KShape主类源码SHA相同；底层相关计算由Cython变为Numba（精度/并行实现不同），不宣称位级复现作者环境。保留当前既有环境和KShape方法，未安装/升级依赖；environment-r3增加实际实现字节绑定，审计源码与差异见library-audit。该差异提交工程审阅，不以缓存生成成功代替审核通过。

comparison-plan显式引用固定图与模式两个identity；只解除UrbanEV四项PDFormer的模式缺项，其余八项保持阻塞。216/16200及20/40 patience不变；72原固定图、9原identity和70冻结时间依赖最终按SHA复核，详见本轮REPORT。CPU长任务命令、日志、PID/start、状态/完成及仅本scope STOP见本轮CPU_PREPARATION.md；未自行启动该八域队列。

## 5.S 并发资源准入政策修订（2026-11-14，本轮）

用户取消第四章串并行数值等价验收；§5资源段及canonical §15为有效协议。ch4/protocol.py和comparison-plan将资源选择与“不执行”结论机器登记，冻结的资源计划必须保持该结论及禁止逐步状态转储；旧第三章容差/探针不再形成第四章并发门禁。现有队列原本没有串并行等价探针入口，本次未删除模型正确性测试或放宽容差。资源差异计划以新resource-difference-r2.json承接原形状估计，仍executed=false，历史r1不改。

外置修订证据：`amd-execution-evidence/m7/M7-concurrency-resource-policy-r1`。本轮只做JSON/源码静态及协议拒绝检查，不运行模型/数值探针/GPU/训练/test。18模型、216项、16200 run-epochs、Adam/OneCycle、50/200及patience20/40不变。无stage/commit/push或正式AUTH，M7不Closed。

附带现场事实：CPU缓存监督器PID64476已退出，AMS失败为“source KShape exhausted its native initialization attempts without a successful fit”。AMS墙钟12140.760329秒、user CPU12786.245508秒、system CPU6242.498447秒；后续七域未执行。仅登记现状，不将本并发政策授权扩大为缓存修复或重跑。旧CPU计划不更新、不自动重用。

## 5.D PDFormer方案B：训练期候选覆盖适配（前轮送审历史；候选规则继续有效）

用户选择方案B；现行精确定义见canonical §16。L_legal按完整L12历史和目标都位于train计算，K=min(原天数×每日步数,L_legal)，i_k=k×(L_legal−1)//(K−1)，K1取0。固定数量覆盖共同合法范围，不按非零率/聚类质量/validation选择。Urban四H采用H12交集3452个合法起点中336个；六城各3501中336个；PEMS04为10172中4032个，PEMS08为10690中6048个。保留固定node order、真实零、原KShape16/长度3/max_iter5/seed2024和随机初始化；无额外归一化、方案A或搜索。

新证据目录：`amd-execution-evidence/m7/M7-PDFormer-train-coverage-adaptation-r1`。候选identity分别绑定索引/规则/源码、元数据和输入文件SHA；fit实际使用同一整数索引构造，模式采用校验拒绝旧前缀身份。新CPU计划按UrbanEV→AMS→JHB→LOA→MEL→SPO→SZH→PEMS04→PEMS08固定顺序，首失败停止，无自动重试，不改变216项训练预算。旧成功/失败attempt不覆盖，固定图仍引用原九域72文件。AMS及UrbanEV旧DTW来源核验后复用，不重新计算。

AMS先前失败诊断见`M7-AMS-PDFormer-KShape-diagnosis-r1/REPORT.md`（SHA `99b62ce017caeb43d8a1cc7badf8d16de47fa48efe3145de55ae9dc0d2e1044e`）：原822864候选中795545个全零三点；固定seed的10次原生初始化各16个中心均取到相同全零向量，按tslearn执行条件可推出首次分配空簇。这是保留现场加静态/确定性索引分析所得，不冒称日志逐次记录了内部状态。旧AMS墙钟12140.760329秒、user CPU12786.245508秒、system CPU6242.498447秒完整保留，DTW阶段包含在其中，CPU秒不算GPU小时。旧Urban成功墙钟836.248804秒不抹除。

本轮8项定向CPU检查通过：公式边界/确定性、共同合法范围/时间频率、真实fit入口的候选张量与顺序（mock fit，未聚类）、节点置换/真实零保留、候选篡改/旧前缀拒绝、完整模式采用、DTW来源拒绝、队列候选参数传递及首失败停止。未重跑与本改动无关的模型/GPU验收。AMS新候选仅做一次不拟合聚合检查，结果及实际成本在本轮诊断JSON，不依据统计调整规则、不保证后续KShape成功。九域新模式均未执行；本轮停在送审，无GPU/正式训练/test/commit/AUTH。

## 5.E PDFormer确定性初始化适配（本轮送审）

用户批准唯一新增适配为初始中心选择。现行规则`native-sbd-farthest-first-v1`见canonical §16：原顺序最早合格候选起始，随后最大化到已选集合的最小原生SBD，tau=1e-12并列按最小展平索引；中心必须来自原池且非零/非常量，形状等价不能重复选入。数值判定只控制中心资格，不过滤fit候选、不额外归一化。通过tslearn原生init=array接入，保留16簇/长度3/max_iter5/random_state2024、原更新和空簇检查，不进行其他初始化方案或次数搜索。

证据目录`amd-execution-evidence/m7/M7-PDFormer-deterministic-init-r1`：九域候选仍引用前轮完整身份，方案B源码/索引/数量不变；每域只执行两次初始化用于精确复现核验及16点原生归属，KShape.fit被测试护栏禁止。中心索引、[16,3,1]实际值、库/代码SHA、原生距离矩阵/归属见证和CPU成本分别保留。初始化身份不作为pattern fit成功身份；真正fit前保存initialization-identity，最终采用验证还要求真实fit成功及最终pattern keys。

旧AMS重复常量中心用原生_assign的小型16点检查保留精确EmptyClusterError证据，新的16点见证检查确保首次分配每簇至少一池内样本。后续更新仍可能空簇；显式中心下原生内部重试重复相同起点，不能写成可消除所有确定性失败。原AMS wall12140.760329秒、user12786.245508秒、system6242.498447秒和旧Urban成功成本全部保持；DTW不重算。前轮方案B仅禁止初始化变更的条款已被本次批准取代，其历史回执不倒改。

实际九域均为initialization_passed_only：两次选点结果逐元素及索引相同；16见证样本原生归属计数全部为[1]×16，AMS最小归属余量0.1489674408794509，九域最小值为SPO的0.13947276198680414。九域检查墙钟36.206158秒、user CPU33.514743秒、system CPU6.663725秒，峰值RSS1103312 KiB，2线程；真实fit/DTW/GPU调用均0。10项定向合成检查通过；最终门禁增加错规则、越界及非池内中心身份的拒绝检查，日志和最终配置核验另存新scope。初始化通过不解除12项PDFormer模式缺项，也不等于整包审核通过。

仍阻塞真实九域拟合、资源准入、工程审核、Git closure和正式AUTH。前轮CPU启动计划的源码绑定已过时，不执行旧命令；审核后另绑定本版并交付用户最后启动。18模型/216项/16200 run-epochs、50/20与200/40、Adam/OneCycle及test边界不变，本轮不运行GPU/训练/test、不提交Git。

## 5.F 九域CPU缓存启动准备（初始化审核已通过，尚未启动）

用户明确转交`M7-PDFormer-deterministic-init-r1`的ChatGPT服务器工程审核：确定性初始化实现与九域初始化验收Review Passed，审核HEAD ee8ea3f32e87ed48608d21991a4fc60d06891c23，初始化源码SHA `4c1251ae8c11d540777f1f30a4a933808005b366e37beb90324f699641670c5f`。实际78份审核文件及291项比较计划绑定核对一致，没有未审核源码或科学配置变化。本轮不修改候选、初始化算法、KShape参数、第三章/主模型/baseline或216项训练合同。

新scope：`amd-execution-evidence/m7/M7-PDFormer-native-cache-launch-r1`。计划`cpu-pattern-plan-r1.json` SHA `6b152f0d4f3fd2ebd3ed003f0c4798e4bb5971087f302c8b22df5f36ce5a9c0e`，绑定280项实际依赖/输入/固定图/方案B候选/初始化审核证据及独立环境身份。UrbanEV→AMS→JHB→LOA→MEL→SPO→SZH→PEMS04→PEMS08串行，每进程2线程、CUDA不可见；首技术失败停止、无自动重试。UrbanEV与AMS沿原身份链复用DTW，其余七域按原合同生成DTW和模式；新输出分别为本scope的patterns/<domain>-r1，队列queue-r1，日志supervisor.log及queue-r1/logs/<domain>.log。旧成功、失败与送审产物均不覆盖。

轻量准入实际通过：源码/配置/环境SHA、九域候选及训练时间身份、DTW引用/生产快照、固定图身份、当前分支/HEAD、目标目录/日志不存在；未发现同用户缓存执行冲突进程，目标tmux session不存在，启动脚本bash -n通过。没有重跑九域初始化、真实fit、DTW重算或任何GPU/训练/test。实际执行前启动脚本再次校验计划SHA和同样的只读准入，复用已审核pattern_batch守护/owner PID-start/STOP/receipt；无需持续SSH或代理。

用户最后启动前停住。本轮只交付完整启动/状态/日志/安全停止命令；新计划不是正式训练AUTH，不预设Git closure。九域技术完成须all_attempted/all_caches_valid均true、九域completed和完整身份核验；随后仍提交ChatGPT审计，通过后才显式采用新缓存。后台不更新工作区注册或自动开训；前轮旧CPU命令继续作废，不重启旧r2队列。

## 5.G MEL模式失败：低成本诊断与最小观测提案

实际队列`M7-PDFormer-native-cache-launch-r1/queue-r1`已结束：UrbanEV/AMS/JHB/LOA为completed，MEL为failed，其余SPO/SZH/PEMS04/PEMS08为not_executed；all_attempted=false、all_caches_valid=false。四域模式身份、fit成功/n_iter/16簇占用、文件SHA及来源链已独立核验，不重新拟合或覆盖；技术完成后仍待缓存采用审核。四域n_iter分别2/2/5/4，墙钟分别737.352459/7840.758839/180.682501/884.499807秒。

MEL候选21168×3×1，有限、float32，沿用336起点×63节点和3513行train；保存的16个显式中心与已审核初始化身份及池内样本相同，复用原生16点见证（最小归属余量0.15218452633332213），本轮不再计算九域初始化。日志显示DTW在0.725809秒完成、initial_centers在8.939586秒保存；最终是native_graphs.validate_pattern_fit本地成功门禁抛出ValueError，日志没有原生迭代、空簇编号或返回模型内部状态。原KShape.fit已返回，而不是日志直接抛出未捕获EmptyClusterError或eigh异常。

静态控制流支持：匹配输入与已有见证排除“重复初始中心导致首次空簇”；若原生EmptyClusterError在fit内部被捕获，应定位更新后或回滚归属路径。fit只捕获EmptyClusterError；也存在未接受finite inertia/best导致_post_fit清空_X_fit的门禁路径。现有统一文案不能唯一确定真实迭代、空簇编号或数值状态，不能把合成失败当作MEL根因。暂不裁决需修改聚类方法。

MEL原成本完整保留：wall818.402638秒、user CPU863.772010秒、system CPU425.886139秒、max RSS582536 KiB；DTW包含于其中，不重复加算。MEL DTW文件SHA `55829997ffc2bce2044dc14b51134f2559c03516d8e48b0c03b4b248394be759`由本轮失败诊断首次登记，63×63 finite/非负/对称/零对角检查通过，不伪称旧失败attempt已有最终pattern manifest。原DTW、中心、失败、日志和receipt保持原字节。整队列wall10485.494330秒不作为单域或GPU小时。

新证据：`amd-execution-evidence/m7/M7-PDFormer-MEL-failure-diagnosis-r1`。现场核验墙钟2.033087秒，3项小型合成观测器检查14.387787秒；仅32条合成序列、4次原生内部尝试，用于验证只读追踪不改变原方法输出/异常，并确认可分辨首次归属与更新后空簇。真实KShape.fit/真实MEL尝试/DTW计算均0。合成例在第一次更新后出现空簇，仅证明机制可发生，不归因MEL。

待ChatGPT审核的最小观测：仅MEL原21168候选与保存中心、2 CPU线程，调用原生_fit_one_init一次（max_iter仍5），不调用完整fit的10次内部尝试循环；只读sys.settrace记录初始/更新后/回滚/post-fit阶段、迭代、空簇编号、簇计数、finite/inertia和小型中心/eigen状态。原更新、空簇检查及计算值不改；结果不是模式缓存，不采用为训练输入。提案脚本与计划已准备并通过语法/合成检查，真实观测未启动，须审核后用户最后启动。后续完成准备优先复用已成功四域和MEL现有DTW，仅处理受影响MEL及未执行四域；不将失败作为重跑择优或新增科学额度。

## 5.I 更新后空簇回滚与提前终止（本轮新增，待ChatGPT工程审核）

用户批准在方案B与native-sbd-farthest-first-v1之上新增`post-update-empty-rollback-stop-v1`，不改变候选/零值/节点、16簇、长度3、max_iter5、seed2024、原生距离及中心更新公式。唯一新增处理位于更新后的_assign空簇异常：先恢复中心、范数、标签、inertia和迭代属性完整前态，再验证16簇非空与finite；至少完成一次合法更新才提前成功返回，首次更新即空簇仍失败。初始化失败、更新函数异常、原生inertia回滚归属异常、非finite及其他异常均不转换为成功；正常原生收敛/inertia回滚保持。未被策略接受的EmptyClusterError沿用原生失败尝试上限，不搜索新中心。源码为M7局部子类，第三方安装未改。

MEL观测事件SHA `d981446aefd6b5c1301c0fd17278bdcc8e3d3f924b7073424c7e85e41cfa0ff4`：簇7初始478、第一次更新后508、第二次更新后0，原生_assign在空簇检查处抛EmptyClusterError，相关距离finite。用户审核指出第二次更新后簇7/5的SBD约1.98e-10，支持中心趋同。其触发阶段适用本策略，但本轮未实际运行新MEL fit，不预先宣布成功。原失败wall818.402638秒与单次观测wall86.642033秒分别保留，均为CPU技术成本；原DTW SHA `55829997ffc2bce2044dc14b51134f2559c03516d8e48b0c03b4b248394be759`、中心/日志/失败回执不变。前节“观测尚未执行”是当时提案记录，由本节新增实际结果承接，不倒改历史。

策略sidecar记录版本/源码、每次尝试、触发迭代与空簇、更新前/失败后/恢复后摘要、完成合法更新和最终保留更新数、最终占用/inertia/中心shape与finite、提前终止及原因。原生inertia回滚完成次数与保留次数分列、n_iter原语义保持；新空簇提前终止n_iter只计合法完成次数。采用门禁验证sidecar与fit及实际pattern_keys一致；旧四域按精确pattern-identity SHA识别为历史原生更新来源，不补写新策略、不重新拟合、不自动采用。比较计划补入新依赖，216项任务定义/来源/训练字段保持逐项一致。

新证据scope为`amd-execution-evidence/m7/M7-PDFormer-empty-cluster-rollback-r1`。18个定向测试用例最终覆盖通过：11项回滚状态机（含真实原生公式的小型2簇对照、16簇受控更新/实际原生归属、首次失败、正常max_iter/inertia回滚、非finite及其他异常、重复确定性），4项方案B合成范围/接线，3项身份/历史来源/DTW复用门禁。累计19次用例执行，曾因合成负例文件与policy侧车重名出现1次FileExistsError；只修复fixture路径后定向复验，没有删除断言、skip或放宽容差。测试累计wall43.747808秒、user CPU46.092560秒、system CPU3.491554秒，峰值RSS546820 KiB；2线程、CUDA不可见。只读审计脚本另有工作区导入路径和annotated tag对象/commit区分两处工具错误，已修正工具并保留记录，失败审计的完整CPU计时未采集，不伪造成本。正常/故障fixture通过不等于真实九域fit成功。

冻结文件、九域72份固定图、方案B九域候选、初始化源码及旧成功/失败/观测按SHA保护核验；未执行模型forward/backward、真实fit、DTW、GPU或test。18模型/216项/16200 run-epochs、候选batch预算6667200 updates、Adam＋OneCycle及20/40 patience不变。源码变更使旧CPU队列计划绑定过时；不能原路径重启。待本轮工程审核后，仅为MEL及后续四域另建新attempt/队列和实际绑定，复用MEL DTW，用户最后启动，首技术失败停止；完成后仍须审计再显式采用。资源准入、Git closure及正式AUTH继续阻塞。

## 5.J 回滚审核通过后的五域CPU补做启动准备

用户明确转交上一scope Engineering Review Passed；审核HEAD仍为ee8ea3f32e87ed48608d21991a4fc60d06891c23，canonical审核SHA `7f04b78bb575cf412bbd0fc622ab61dd0523e8a53f36e39643ffcd507a199bc1`，本M7审核前版SHA `a1fd4ce67683d3ffa028f57c6a5bc7cedf5c975746cf08e56dfc5bfc68f3c3c6`。执行前审核包字节、全部292项比较依赖及实际Git一致；没有未审核源码/科学配置变化。本轮工作区只更新本执行记录，canonical和已审核源码保持原字节，不改聚类或216项训练合同。

新scope：`amd-execution-evidence/m7/M7-PDFormer-five-domain-cache-launch-r1`。`cpu-pattern-plan-r1.json` SHA `6daa1a8b48099bdb116edcf5b9ef6a86e6bd95fd8854bdf20de01f7690dad918`，284项绑定覆盖实际执行/准入/完成检查依赖、环境、五域输入/候选/固定图、确定性初始化、回滚策略、DTW来源、旧四域成功及审核证据。MEL→SPO→SZH→PEMS04→PEMS08串行，每进程2 CPU线程，CUDA不可见；首技术失败停止，无自动重试。MEL复用原DTW `55829997ffc2bce2044dc14b51134f2559c03516d8e48b0c03b4b248394be759`，原输入/节点/参数/生产快照核对一致；其余四域按原批准范围生成DTW和模式。旧四域只按原pattern-identity核验，不重新选中心、拟合或补写策略身份。

轻量准入通过：环境/依赖SHA、五域候选及train时间身份、固定图节点、MEL DTW链、旧四域不可变身份一致；无冲突同用户CPU缓存/观测进程，目标tmux `m7-pdcache-rollback-r1`不存在，新queue-r1、supervisor.log与patterns/<domain>-r1均不存在；启动脚本bash -n通过。复用既有监督器和owner PID/start/STOP机制，没有重跑18项测试、九域初始化或增加资源探针。启动脚本先验证固定plan，再重复只读准入，后台移除代理环境变量，无需持续SSH连接。

停止点为用户最后启动。每域保存独立日志、PID/start、初始化身份、kshape-policy、实际fit与CPU成本、pattern-identity或failure及receipt；只读完成检查要求五域全部completed、结束记录/results/receipt/SHA一致、完整新模式门禁及回滚记录有效，并再次核验旧四域。仅有preparation-finished.json不等于成功；合并九域来源时分列历史四域与新五域。技术完成后仍送ChatGPT产物审计再显式采用，不自动更新注册、启动GPU资源准入或正式训练。旧失败和观测成本、全部历史目录保持。

## 5.K 九域模式缓存审核通过与正式采用接线

用户明确转交ChatGPT服务器直读审核：九域技术完成，Engineering / Artifact Review Passed。本轮在HEAD ee8ea3f32e87ed48608d21991a4fc60d06891c23的现有M7工作区采用，不变更算法或科学设置。新证据scope：`amd-execution-evidence/m7/M7-PDFormer-cache-adoption-r1`；九域精确身份、CPU成本和允许差异验收见strict-completion.json、adoption-validation.json及REPORT.md。canonical现行合同允许审计后显式采用，未重复改动。

后续状态同步（2026-11-15）：用户明确转交ChatGPT直读审核结论，`M7-PDFormer-cache-adoption-r1` **Review Passed**；已独立核验九域身份9/9、12项PDFormer接入、36项共享身份元数据变化、其余168项不变，以及216项/16200 run-epochs/342项绑定。审核版本：HEAD `ee8ea3f32e87ed48608d21991a4fc60d06891c23`；REPORT SHA `ccd37a2c996e12effa1b9b2977b9ec9f9afc9954d3e800a8375d1441baa5473d`；本M7审核前版SHA `5ec55ef3420f68cd7126abdf23a692b1c08ee077c9175bf97e8c5a628209d0b6`；comparison-plan SHA `317466dd7f40aa490b2fdef59c4bc91b12ede36e5b89f2b6e94ebf308ae03460`；pattern_registry SHA `7e032bf6a94cce5a947a1cc0a6e1d07ded56d627d6390ea781b4d0526f812ac9`。本轮仅更新唯一M7状态，不改canonical、计划、注册或绑定，不运行任何测试或模型。

全局`preparation_blockers`中的`engineering review`保留：ch4/prepare_plan.py:189在任务缺项之外无条件附加该项，canonical现行整包审核条款及本M7§5/§5.R未记录整包工程审核最终收口。其剩余范围是M7主模型及18模型适配、数据/训练评估执行链、依赖绑定、监督器→freeze→test-once→seal和已有修复验收证据的整体审核确认；不再包含已经通过的PDFormer缓存或本次采用接线增量审核。此为审核范围澄清，不认定上述组件测试失败，不要求重跑有效证据、不新增测试或科学门槛。GPU容量/吞吐及最终batch/concurrency审核、finite resource probe/review、Git closure及实际AUTH三项阻塞保持，resource_frozen=false、execution_authorized=false；comparison-plan完整字节不变。

现有ch4/pattern_registry.json仅将九个pattern_identity从null替换为已审核路径/SHA，candidate_identity及fixed_graph_identity完全不变。comparison-plan的12项PDFormer新增各域dtw_matrix/pattern_keys引用并删除模式未完成阻塞。既有生成器为D2STGNN/STDN/GraphWaveNet的36项任务复制同域完整native_identities，因此这些任务仅同步共享元数据中的pattern_identity；实际native_artifacts、模型输入/结构及训练均未改变，未向这些模型接入PDFormer模式。其余168项任务完全相同。216项所有科学字段、结构、训练配置、seed2024、窗口/节点、20/40 patience和test边界逐项相等；16200 run-epochs、候选batch的6667200 updates上限保持。

采用轮绑定292→342，仅原registry文件SHA更新，新增50项经审核的模式/DTW/初始化中心/策略文件绑定；执行Python源码未修改。采用增量审核现已通过；整包工程审核收口、GPU容量/吞吐和最终batch/concurrency审核、资源探针/冻结、Git closure及实际AUTH阻塞保留，resource_frozen=false、execution_authorized=false。采用轮仅执行必要身份/完整性/接线验收，包括严格完成工具和生成器保留的小型16中心身份见证；本次状态同步不重跑这些检查。

原四域来自`M7-PDFormer-native-cache-launch-r1`，保留原生产源码及无新增回滚策略的identity；新五域来自`M7-PDFormer-five-domain-cache-launch-r1`，统一绑定post-update-empty-rollback-stop-v1。MEL触发第2次更新空簇（簇7），恢复第1次合法更新，n_iter=1、簇7占用508、最终16簇非空，inertia=0.22457809428071107，停止原因post_update_empty_rollback。SPO/SZH/PEMS04/PEMS08未触发新空簇回滚，停止原因均native_inertia_rollback；不得将两类停止原因混称正常收敛。

当前九份成功producer自身记录的成本如下，单位均为秒；复用的历史DTW成本不再次分摊，不作为整个开发历史总成本或GPU小时：

|域|wall|user CPU|system CPU|
|---|---:|---:|---:|
|UrbanEV|737.352459|773.706163|375.319940|
|AMS|7840.758839|8212.480881|4113.369167|
|JHB|180.682501|190.496688|89.929395|
|LOA|884.499807|936.922433|455.979782|
|MEL|76.443721|81.416961|33.839253|
|SPO|104.438800|110.233166|46.440267|
|SZH|4626.109228|4827.248424|2372.788885|
|PEMS04|9845.275198|10274.727045|5070.123770|
|PEMS08|7217.253527|7513.666989|3873.378578|
|成功producer求和|31512.814081|32920.898750|16431.169037|

五域监督器实际wall21896.097165秒（约6.08小时），包含调度/验证等开销，与各producer计时口径分开。原MEL失败818.402638秒、单次观测86.642033秒、旧AMS失败和所有此前成本继续保留，不从训练额度扣减或抹除。新五域results和preparation-finished均完整有效；旧结束记录scientific_or_engineering_review_passed=false保留其生成时事实，本轮用户转交的后续审核结论在本节承接，不改历史收据。

采用轮停止在接线送审，其后审核通过由本节状态同步承接；仍不进行GPU资源探针或216项正式训练/test，不stage/commit/push、不生成正式AUTH。采用后旧CPU计划的registry绑定自然过时，旧队列不再用于启动或恢复；采用轮strict-completion保存修改前的严格验证结果及当前注册的独立接线验收，不倒改旧计划绑定。

## 5.L 整包工程审核E1/C1/C2定向补充（2026-11-15，待ChatGPT审核）

前轮`M7-whole-package-engineering-review-r1`为只读工程证据汇集，未判整包Passed。用户本轮批准三项最小处理，基点HEAD仍ee8ea3f32e87ed48608d21991a4fc60d06891c23、M7无upstream、index空，保留所有既有未提交实现。新证据`amd-execution-evidence/m7/M7-engineering-three-fixes-r1`独立保存改前字节、增量diff、SHA、定向验收及closure候选清单，不覆盖旧报告。

E1：M7局部lifecycle辅助由supervisor跟踪每个已启动worker，记录UTC/Unix起止、monotonic墙钟、PID/proc_start、启动时access/history偏移、退出码、日志SHA和config/runtime绑定。失败回执独立放failures，不污染成功receipt精确集合；成功worker无失败回执，原train-result及成功receipt结构保留。worker仅将optimizer_completed持久化移动至optimizer/scheduler完成后紧邻位置，随后才做累计统计，未改变训练计算及更新顺序。计账区分本次完整事件条数与累计update最大值；完整history才算完成epoch，不反序列化权重。损坏/截断日志标记accounting_complete=false；结构化异常、STOP和负signal退出码明确区分。supervisor自身失败等待已启动子进程退出，完成失败回执后再发布带自身成本及回执SHA的failure.json。墙钟含轮询观察延迟，不称GPU小时；SIGKILL、掉电、强制终止不保证最终落盘，未持久化事件不冒称确认更新。

本轮6项定向CPU测试通过，9.671秒：成功完整监督器→freeze→合成一次评价→seal且无失败回执；技术异常及后续停止；STOP、损坏日志、缺失成功receipt、SIGTERM负退出码；同wave失败使同伴STOP；监督器启动后异常；启动偏移及损坏history/截断负例。随后将“完整JSON但缺失事件类型”也明确判为损坏，新增该负例并仅复验对应日志解析用例通过，共6项独立用例、7次执行，无失败。仅少量合成子进程，无真实模型、数据或checkpoint。旧checksum/claim等既有验收复用，不机械重跑。生产216断言、原封存代码不变。

C1：根.gitattributes仅追加ch4/vendor/STDN/model.py -text -eol；STDN原SHA a9c63db994fdda47bdc3292b80b2f6b4638afd13a12e45569b16327bbe43a992、577行CRLF/0裸LF不变。属性均unset，filtered/no-filters Git对象同为7de2b59a124a25d95443901fbdee162271945f29；临时独立仓库clean/checkout过滤往返保持原SHA，未触碰当前index或修改全局Git配置。

C2：再次核对全部worktree后，仅当前M7存在两份退出PatchSTG文件；共享AMD/.git/info/exclude只追加/ch4/vendor/PatchSTG/LICENSE及/ch4/vendor/PatchSTG/models/model.py两条精确规则，原文件原路径/SHA保持。无整目录忽略，无.gitignore修改；已忽略的lib/utils.py保持原政策，不处理prunable worktree。exclude仅服务器本地，不入closure；.gitattributes和lifecycle纳入精确候选清单。

重新生成执行闭包和comparison-plan，仅同步worker/queue/lifecycle源码绑定及依赖元数据：342→343项绑定，仅两个既有SHA变化及新增lifecycle；Python闭包180→181，运行资产43不变。逐项核对216任务及训练字段完全相同，16200 run-epochs和候选6667200 updates不变，pattern_registry、九域模式/固定图、seed/预算/20与40 patience/test边界不变。资源、engineering review、Git closure及AUTH阻塞继续保留，resource_frozen=false、execution_authorized=false。B1证据仍仅恢复helper与worker接口，不宣称生产队列断点续跑已验收，不扩建resume。本轮实现完成不代表Review Passed，停在重新送审。

## 5.M E1失败计账最后两项边界修复（2026-11-16，待ChatGPT整包复审）

用户转交ChatGPT直读复审：`M7-engineering-three-fixes-r1`的C1（STDN CRLF）和C2（PatchSTG精确本地排除）Review Passed；E1主体及六项CPU用例通过，但需补receipt结构及序号缺口。本轮未修改.gitattributes、STDN、PatchSTG或共享info/exclude，不重做其已通过验收。工作区/分支/HEAD仍为CH4-v10-spatial / M7-v10-spatial-v1 / ee8ea3f32e87ed48608d21991a4fc60d06891c23；无upstream，index空。

`lifecycle.finish()`在遍历前明确要求receipt根对象和files均为dict；非法类型统一受控ValueError并归类invalid_success_receipt，保留真实returncode=0，正常生成独立失败回执和Supervisor failure.json，不重复finalize同一异常、不发布complete或调度后续。正常成功receipt及seal合同、queue/worker本体保持原字节。

`progress()`在重复/回退等既有校验之后检查相邻新updates或epoch恰好加1；遇到缺口仍计入两端已确认完整记录，并记录缺失的起止序号，使最终accounting_complete=false。不要求offset之后第一条从1开始，不跨启动偏移推断旧累计状态、不补算缺失操作、不读取checkpoint。updates101/103及epoch7/9分别保留2条、累计max103和最后epoch9，准确记缺102及8。已有canonical §16的损坏/不完整计账合同已覆盖本次工程校验，canonical保持原SHA不重复增写。

本轮仅运行两个定向CPU测试方法：receipt的files list/null及根对象list/null四个子场景均正常退出0后拒绝；序号缺口的失败回执与非1起点/非零offset合法连续记录核对通过。无真实模型/数据/权重/GPU；原成功、STOP、异常、checksum和恢复helper证据复用，不重跑无关验收。外置证据`amd-execution-evidence/m7/M7-E1-accounting-boundary-repair-r1`保留日志、定向diff、schema说明、改前/改后SHA及计划差异，不覆盖旧报告。

343项绑定只更新lifecycle源码及对应闭包身份；216任务逐项完全不变，模型/训练字段/seed/20与40 patience/16200 run-epochs/test边界不变。closure候选仍为原79路径，无新增路径，排除info/exclude及PatchSTG；所有工程/资源/Git/AUTH阻塞及resource_frozen=false、execution_authorized=false保持。B1继续仅指helper与worker接口证据，不宣称生产队列恢复已验收。本轮不stage/commit/push、不创建正式AUTH，停止于ChatGPT最终整包复审前。

### 5.M 后续最终审核裁决与状态同步（2026-11-16）

用户明确转交ChatGPT服务器只读最终结论：**M7 Whole-package Engineering Review：Passed**。通过范围为E1失败生命周期、receipt根对象/files结构与update/epoch缺口计账，C1 STDN原始CRLF保护、C2两份PatchSTG精确本地排除，完整v10空间扩展与18模型适配、数据/训练执行链、342→343绑定（181 Python/43资产/169来源快照）、九域PDFormer缓存与216项计划、Supervisor/test-once及最终seal。此前§5.L/§5.M的“待审核”保留对应送审时点，由本次最终裁决承接，不倒改历史失败和成本。

审核版本：HEAD ee8ea3f32e87ed48608d21991a4fc60d06891c23；canonical SHA `4d8d08e14e05bf7637315cfcf56b31890fb3d9d4206266ded6957bba4247109c`；本M7审核前SHA `943241b7ba66ac4f799f0306429f54907ac008e8e53ce871dce9bc276d6fd0aa`；comparison-plan SHA `616e6f85375033e02cdb14faf312f620a229127130fed5b73e32ada5b494766c`；lifecycle SHA `2e219601c83c6b307f1cfabc97563a1177a6fd89a3e31b4aaa979214d84dda03`。对应修订证据为M7-E1-accounting-boundary-repair-r1/review.json，SHA `940467444229959de339eb9e4cfb40397d42e38cc72ed0f29f59300a0433852d`。以上为本次状态同步之前已审核字节，不把同步后的新文档/计划字节冒称已完成closure复核。

本轮只将canonical §16和本文件当前状态同步为整包Passed，从comparison-plan全局preparation_blockers仅移除engineering review。其余三项阻塞（GPU capacity/throughput and final batch/concurrency not yet reviewed、finite resource probe/review、Git closure and actual AUTH）及两个false标志保持；216任务、训练字段、343绑定和科学输入完全不变。刷新原79路径allowlist的实际SHA，集合与当前Git status逐路径相等，不纳入info/exclude或PatchSTG历史文件。外置同步证据：amd-execution-evidence/m7/M7-whole-package-review-passed-sync-r1。

下一步是精确Git closure，仍待最终字节复核及实际执行；GPU资源审核/冻结、正式AUTH、训练/test未获准。本轮不运行测试或模型，不stage/commit/push，不创建AUTH。B1恢复证据边界及§15资源/失败成本合同均不变。

## 5.N 进程生命周期完整修复与最后配置准备（本轮待审）

本轮新增外置scope：`amd-execution-evidence/m7/M7-resource-probe-preparation-r4`，旧r1/r2/r3原字节保护，不补写旧complete、不修改失败收据。r2五项历史采用、r3十八项原completed，以及UrbanEV-AMD-q4原wave failed但四worker完整5+20更新、退出0、GO归属/资源及产物通过离线核验，形成独立history-adoption.json；合计24项引用，producer仍是旧commit。唯一未执行是`PEMS04-ch4_v10_pdformer_him_v1-q4`。原失败仍为failed，不将引用计为r4新测量。

r3旧错误将“PID/start变化或进程退出”合为同一文案，异常后即时poll=None即拒绝。已观察到四份完整结果、真实退出0和归属样本；源码确定缺少有界退出过渡。旧日志不足以证明具体内核分支为Z状态或PID替换，不为微时序重跑模型。占卡成本绑定原attempt收据：r1=6.6386255687102675秒，r2=101.2724538249895秒，r3 elapsed=289.0257544461638秒；沿用r3较保守的cumulative=396.9368384378031秒，原字段因不同monotonic采样与elapsed求和略有差异，cost-ledger.json分列。4小时剩余14003.063161562197秒，保留60秒收尾后有效deadline额度13943.063161562197秒；不再加wave/worker时间，不将整个r3记为无效失败成本。

外置probe拆分identity_conflict、exit_hint、identity_unavailable，保存实际/预期PID/start、state、读取阶段和异常、单调时间、前后poll。已登记Popen的明确退出迹象进入2秒固定窗口，受原wave/全局deadline限制；PID/start/namespace/boot矛盾不等待。逐slot推进，q4其余存活slot仍受实时核验；退出0验产物，非零拒绝，旧GPU PID刷新消失限2秒，未知/残留到限拒绝，GO仍要求全部q存活并匹配。正式队列只使用所属进程身份和独立session/group，不复制NVML监测器。

正式queue/lifecycle/runtime_io：异常、STOP、可处理SIGTERM/SIGINT统一合作停止10秒→TERM5秒→KILL5秒，各slot共同推进；身份不符拒绝发信号。未确认退出写明null退出码、有限日志快照及不完整计账，不能成功或继续调度。原始异常与清理/计账异常分列，SIGKILL/掉电/不可中断内核状态不能保证回收或落盘。test按独立日志/offset计before b与after b+1完整对，分列已开始/已完成/未确认batch及同步origin样本数，成功receipt总数须一致；claim及访问保留，不重测。worker.py计算及日志原定义未改。

CPU定向证据：cpu-probe-r2.log中14项Passed（含5项有效旧fixture复用，并补清理单slot身份错误不阻塞其它slot）；cpu-formal-r4.log中11项Passed（7项新增边界、4项受影响旧验收）。早期cpu-formal-r1.log保留旧STOP断言不适用的失败，r3日志保留两项测试类名拼写导致的加载错误；均不伪装Passed。没有真实GPU/模型/test。正式历史模型、初始化和数值一致性测试未重跑。全部证据及新增绑定、计算路径不变证明、精确closure候选见本scope REPORT和清单。

comparison-plan只刷新三份执行源码的SHA、相应依赖原因行号，并增加本次修复待审阻塞；216项完整任务对象及其余科学字段保持。343绑定路径集合不扩大，181 Python/43资产/169来源快照边界保持。外置续接草案commit=null，启动校验明确拒绝；须ChatGPT审核、一次精确closure后绑定实际commit和SHA，再交用户完整启动/日志/完成/停止命令。历史producer不改写，24+1覆盖和原计时重复政策不变，不自动进入正式训练。

## 5.P 上轮五文档基础合同的历史收口

此前五文档以完整v10必承接、空间及具体用法未定完成审核并closure至ee8ea3f32e87ed48608d21991a4fc60d06891c23；上游24c45c9c八路径与16e45afe等历史证据保留。本轮具体开发授权及2026-11-14协议修订取代上轮只读停止点，不倒改旧回执。

### 5.H 原closure之前的静态记录（历史时点，不是当前Git状态）

工作区仍为`/public/home/yueweiting/大论文/ModernTCN-U+V`，分支`M6-ModernTCN-UV-v1`。本轮起点HEAD/tracking/实际远端`16e45afe658bc5fdf0ec1ab2801fc893b7ed3866`、0/0；8路径unstaged/index空，不是clean，不能当作未来M7 parent。不可变baseline tag保持原目标。

以上旧时点由已完成的24c45c9c…closure承接，保留其事实，不重放其任务；本轮实际状态以前文和新外置回执为准。

Project injection维持项目级通用规则；本任务不复制injection到服务器，不修改Project设置，也不重写用户的会话首条消息。会话标题保持“大论文 07｜M7 第四章开发与实验”。


## 5.O 并发效率补充探针（2026-11-18，首版历史时点；现行修订见§5.O.1）

用户转交确认：原25配置完整覆盖，24历史引用＋r4最后一项，且r4退出待确认状态实际触发后成功清场；工程审核与9路径closure有效。旧r2/r3 failed波次、producer及失败成本不改。继而明确暂停“全q1、原216顺序”的资源冻结决定；旧`M7-resource-adoption-and-formal-preparation-r1`已有资源表、容量表和耗时估计全部保留，仅历史草案，不继续引用其User-approved q1作为现行依据。正式仓库尚未写入该冻结，comparison-plan和343绑定本轮原字节保留。

新scope：`amd-execution-evidence/m7/M7-concurrency-efficiency-probe-r1`。现行精确方法见canonical §15及resource-plan.json：162单元、486基础q1/q2/q4候选，q3/q6/q8条件登记且扩展最多12波；Urban H6形状映射其它H但不合并科学任务。先全单元q1基础覆盖，再按正式任务成本推进q2/q4；预算不足逐项标未覆盖，不冒称已优化。旧25项仅复用于容量/成本及历史参考，与新block协议吞吐分开。

新协议formal-overhead-block-timing-v1保留正式训练步日志、标量取值及计算顺序；5准备＋20校准，固定公式选20–100计时更新，同单元比较q不改变工作量或OneCycle。块边界同步，包含最慢slot与退出前排空尾部。最多18池、当前6个类别的确定性真实四任务组合，串行/两q2/单q4按同池工作量比较完整makespan；资源近优5%带、最多3独立波次，中位数和范围，无法稳定区分须明示。不得解释为论文数值容差，不做串并行数值一致性或完整状态转储。

CPU affinity/cpuset为8-11,40-43（8逻辑CPU）；本层quota=-1、period=100000，宿主父层被cgroup子树挂载遮蔽，未证明父层有效8核额度。q8和>4 CPU扩展因此保留条件缺口；不修改系统或以可见CPU数冒充完整quota证据。主搜索每worker1线程，现有loader workers0，不增线程全笛卡尔积。

旧成本保守415.89825103618205秒，剩余13984.101748963818秒；8400/1800/1800秒分配和至少60秒最终预留，均是硬预算下有限搜索，不是完成保证。原r4 cost与complete采样毫秒差分别保留，不相加为两次执行。所有实际技术失败停止后续；容量预判排除、预算未覆盖、技术失败与实测分列。

本轮仅CPU回调/静态绑定验收，实际数量与日志SHA以外置preparation-receipt/REPORT为准；不机械重跑旧14/11验收，不运行真实模型、优化更新或GPU。补充探针尚未启动，未生成正式AUTH；审批通过后用户一次启动，结束再审资源选择，未预填最终q。正式216任务及6667200更新/16200run-epochs上限、模型/数据/训练字段/seed/test边界保持。


## 5.O.1 补充探针选型逻辑修订（2026-11-18，待ChatGPT复审）

用户转交首版只读审核：暂不批准启动，保留正式步开销、块同步、逐slot任务/来源绑定、train-only与r4归属/退出/清理机制。本轮在原scope内保留`revisions/selection-r2/before`及改前SHA，再作明确增量；未覆盖历史运行证据、旧q1草案或r1–r4封存包。§5.O中的父quota缺口及原20–100计时描述是首版历史，不再作为永久拒绝扩展或充分确认的现行依据。

现行精确定义见canonical §15和新版resource-plan：同构按完整排空计时块总更新吞吐；混合按同池同工作量各实际波排空计时块耗时之和的中位数。初始化/校准/清理及完整墙钟另留作预算，不以短波启动摊销排名。新增`selection_rules.py`纯CPU规则，统一排名、5%候选带、确认对象和扩展收益；偶数重复先取耗时中位数再倒数表示排名，不能与倒数中位数混用。

初筛保留5准备/20校准/20–100计时；不足3秒或波动不能支持5%判断时仅粗筛。较长确认按3秒/最快初筛步时向上取整，至少原计时次数和20、最多4096且不超过原OneCycle余量；跨q同组工作量完全相同，独立seed2024初始化，source/组SHA绑定，不同协议/工作量不混合统计。所需时限超过600秒或预算不够则明确确认不足。每候选总计最多3波。至少两个充分计时重复及保守范围支持的具体对照才可标局部确认；初筛落后未复测不否定其它已确认对照，也不冒充全候选最优。

父quota不可见继续标未知，但在affinity和已知本层/平台限制内允许原登记q6/q8条件候选；不改系统配置。共享确认预算先给至多4个触发单元有限扩展机会（首次准入子预算900秒），再按扩展影响优先和正式成本排序联合安排同构/混合对照，保护首个非扩展机会450秒，第二轮交错反序；扩展首次及确认合计≤12实际波。q4池容量排除时保留serial/q2同任务同工作量比较。8400/1800/1800阶段、累计4小时、旧415.89825103618205秒和剩余13984.101748963818秒均不变，至少60秒最终收尾。

定向CPU新增15项通过（`revisions/selection-r2/cpu-final.log`），受影响既有7项通过（`affected-existing-r1.log`）；未重跑全部19项或r4保护测试。首次合成测试暴露排除记录函数的参数名与reason详情冲突，已窄修并保留失败日志`cpu-r1.log`，不写成真实GPU故障。全部检查使用纯CPU回调/合成身份；无真实模型导入、GPU查询、optimizer、训练/validation/test或checkpoint读取。原进程状态机、正式训练步计算顺序和train-only边界复用，变化仅选型/确认元数据及调度；实际diff/AST核对、SHA和绑定检查见本轮外置报告。

正式comparison-plan及343绑定、216项科学任务/原顺序、模型/数据/batch/seed/OneCycle、50/200 epoch与20/40 patience、16200 run-epochs/6667200 updates保持原字节。本轮仅canonical/M7两份仓库文档变化，index空，不stage/commit/push；未生成批准记录、正式AUTH、arm或运行attempt。资源仍未冻结，修订包完成仅表示准备送审，ChatGPT尚未审核本轮新字节；审核通过后用户最后启动，测量完成再裁决并发。


## 5.O.2 确认预算闭合修复（2026-11-18，待ChatGPT增量复审）

用户转交selection-r2复审：主排名/成本分列、长块确认/组SHA、分组统计与波动边界、父quota及扩展12波、混合部分可行比较、r4生命周期/数据保护通过，直接复用。唯一剩余阻塞为旧调度“全部组第一轮→第二轮”仅预留一轮，可能耗尽预算却无完整对照。本轮仅修订controller确认调度、对应计划条款及定向CPU证据；selection-r2历史和审核字节保存在`revisions/selection-r3/before`，不倒改§5.O.1的当时记录。

新准入先计所选对照所有尚缺的两次独立确认波次（含每波20秒清理裕量），通过后连续完成两轮、第二轮反向。全集合放不下则依次闭合赢家/相邻q、赢家/必要参照、赢家/其余；同组已完成候选结果可复用，不为另一对照重复执行，不混入初筛。混合三模式两轮最小1960秒不能进入1800秒预算，但q2/q4模式对最小840秒可进入；再补serial只预留其尚缺的1120秒。未覆盖模式保留，工作量完成不是自动确认通过，真实失败仍终止后续。

新增5项推进假时钟的controller CPU用例全部通过：48组×q1/q2/q4、每波120+20秒预留/实际模拟10秒，闭合21组共126确认波，阶段实际模拟1260秒（旧轮次优先算法在该fixture闭合0组）；混合先闭合具体模式对并复用已完成模式，实际模拟120秒/波的分支保留serial预算未覆盖；第二轮首波失败保留首轮3波和确认阶段40秒模拟成本、不封存/不继续；另验证全局截止、每候选3波及扩展总12波。受影响既有3项controller用例通过，未重跑其它已审测试。日志为`revisions/selection-r3/cpu-final.log`、`affected-existing.log`，这些秒数为合成假时钟，不是实测GPU成本。

正式comparison-plan、216任务、343绑定、模型/数据/训练字段及16200 run-epochs保持原字节；1800秒确认阶段、累计4小时、4096更新/600秒单波、原OneCycle及其它门禁不变。无GPU查询、真实模型/optimizer/训练/validation/test、运行attempt或approved=true记录；index空，无stage/commit/push。仅本增量待审核，不代称ChatGPT已审新字节。


## 5.O.3 补充探针r1 JSON控制发布失败及r2续接准备（2026-11-18，待ChatGPT审核）

selection-r3经用户转交ChatGPT Review Passed后，已仅外置exclusive-create批准资源探针，未授权正式训练/资源冻结。用户启动r1后完成162项q1和51项q2，另5项q2容量预判排除；在SZH-GraphWaveNet-q2-attempt1，worker1读取measurement-GO抛JSONDecodeError并退出1，worker0按原策略清理退出−15。failure SHA为361ce933e630ed1590e8076185fa30c9bed5993b244e6d09325db4850881ee23，wave receipt SHA为febb276624c469d62a44da03be0935320f6c2131f942138aeb065a38f81c205a。旧open('x')先创建最终路径后写JSON是确定的竞态窗口；日志没有捕获读取瞬间字节，不能把具体内核微时序写成已证明。

r1 attempt墙钟2999.2392212999985秒；原累计415.89825103618205秒；失败回执保守累计3415.1374728847295秒，剩余10984.86252711527秒。基础阶段已用2998.604080704972秒、剩余5401.395919295028秒；混合/确认各1800秒上限不变，最终至少60秒清理。累计字段与分项直接相加的微秒级采样差照实保留，成本不加各wave/worker求和。原r1全部4799文件、267个登记进程身份检查（无同start存活者）保留；失败波部分更新不得采用completed，r1不补写complete。

独立scope为amd-execution-evidence/m7/M7-concurrency-efficiency-probe-r2。new()只在同目录完整关闭fsync临时JSON后原子hardlink发布，不覆盖已有路径；仅清理自有inode临时文件并同步目录。GO增加阶段/slots/wave-spec SHA及精确calibration/measurement次数验证，ready身份相应检查；r4退出/归属/清理、模型step、CUDA块同步及统计选型保持。

213项历史成功逐项核验receipt、任务/工作量、原GO/归属、退出0及所有文件SHA；5排除核对原显存容量公式，仍为预判，不称实测Passed。r2引用真实r1结果路径，统一用于q2/q4固定次数、容量、混合成员和确认来源；不复制成fresh或重跑q1。原972候选保留，218历史已处理项不再初筛执行；基础待处理268项（106q2含失败任务＋162q4），首项SZH-GraphWaveNet-q2在新scope标attempt2，另486条件候选及原混合/确认仍按原触发与预算。失败实际波保留在每候选最多3波计数内，故该失败候选若再次初筛成功仅余一次确认机会；不能自动扩额或称已充分确认。

CPU定向验收、真实少量CPU子进程、历史采用、正式队列只读风险与完整SHA以本轮外置REPORT/receipt为准。正式comparison-plan、343绑定、216任务/训练字段及16200run-epochs不改；本轮仅工程条款/成本同步，index空，不stage/commit/push。无GPU查询/负载、真实模型/optimizer/训练/validation/test/checkpoint；不生成r2 approved=true或正式AUTH，不创建运行attempt。审核后用户最后启动，首失败停止，不自动恢复或进入216项正式训练。


## 5.O.4 补充r2资源审计、调度候选及正式JSON发布窄修复（2026-11-19，待复审）

用户转交ChatGPT确认r2正常完成，plan SHA为1431ea17f132b5418d019bc14cd4ab6e228c4f5218a0ed8d34b823a5c6b8d2d7，complete SHA为cf92ff5117029acb15db6e86b36268f3e8da3f1aded43d91cc27943da85bbaaa。本轮从真实producer重核11590 payload/11591 checksum、296新成功波＋213历史成功波、162单元/972候选及6真实混合池；保留r1失败部分更新、原producer/commit和技术成本。r1/r2共16443文件已建立保护清单。8单元有局部重复确认，不能推广q8或把62预算未覆盖等负向状态删掉。资源正式冻结仍未批准。

外置新scope：`amd-execution-evidence/m7/M7-concurrency-resource-audit-and-runtime-repair-r1`。逐单元CSV/JSON保留q1/q2/q4及条件q的原结果SHA、计时工作量分组、完整排空吞吐、显存、时长、CPU可见信息及覆盖状态；混合按同任务/同工作量分别重算，PEMS池确认q2/q4排序仍处边界，单次领先不称稳定确认。原r1失败与本轮审计脚本错误日志均保留，不把工具解包/结构映射断言错误归因于producer；TESTAM的H切片与PEMS P12按真实协议区别核对。

保守候选原序216个q1波；效率候选198波（184q1、12q2、2q4），仅采用有相应证据的精确混合池及Urban同模型四H形状复用，任务集合/模型/数据/batch/seed/训练/指标/test边界不变。按正式6667200最大更新逐项估算，训练块外推分别190.51与189.48小时；不是整轮实测，validation/checkpoint/I/O单列且未实测。50%/75%/100%轮数敏感性情景均提供，效率全上限宽区间165–397小时，不预设提前停止或宣称全局最优。候选明确新顺序与waves对应；正式配置尚未采用，不新增AUTH。

后续补测并非上述候选正确执行的必需门禁。若目标是显著缩短长任务，可供用户裁决一个AMS的HimNet/MegaCRN/AGCRN/STDN真实池serial/q2/q4对照：四项全上限串行训练约16.20小时，零干扰理论最优尾部5.55小时，0–10.65小时仅机会区间并非预测收益；现有同构副本不能证明该混合收益。建议仅最小一次筛查、原阶段与4小时余量内另行明确计划，不能本轮自动增池或重分配预算；确认预算不足时仍标初筛。

r2成本文件累计10439.124434339814秒，complete稍晚采样对应保守累计10439.34133428149秒，0.216899941675秒差是一次收尾采样差，不是第二次执行。全局余3960.6586657185107秒，阶段余base277.55403361003846/mixed1217.6791812898591/confirm484.95573597215116秒，仍留60秒；没有新增占卡成本。

正式修复仅runtime_io.write_new：同目录唯一临时文件完整写入、flush/fsync/关闭后原子hardlink，不覆盖已有文件，仅清理自有inode；owner/runtime/STOP字段及所有调用方保持。新增5项CPU原子发布测试通过（0.076秒），受影响既有成功监督/封存/claim及STOP/损坏日志/信号成本2项合成测试通过（3.926秒），未重跑模型/r4保护测试。comparison-plan仅source_binding及dependency_identity两处runtime_io SHA变更，仍343绑定、216任务、16200run-epochs、6667200updates。未查询GPU、未运行真实模型/optimizer/训练/validation/test、未读取checkpoint；正式源码修复尚未审核/closure，不stage/commit/push。


## 5.O.5 原子发布精确closure与q1/q2/q4缺项补测准备（2026-11-19，补测包待审核）

用户明确批准原子发布修复closure，不代表采用§5.O.4的189.48小时候选。allowlist SHA `d70256d4b2555842b7505d9e74579829d10c2488d11c6c6bb07be633f4f38ab3`的精确5路径已逐字节核验、stage、提交并正常推送：`3e387a9a3b52af4f0ab02763977ac399c475a6f9`，父`ee9110140a7cd2ee8c0cadbc766ea9d156ead8be`。提交blob与审核SHA一致；当时HEAD=tracking=实时origin、0/0、worktree/index clean，M6及不可变tag不变。外置closure-receipt保存完整证据；复用5项新增CPU与2项受影响队列测试，不重跑模型或探针。此后本节及canonical为任务二未提交的新修改，不夹带入已审closure。

用户最新范围固定162单元×q1/q2/q4=486条件，取消新增高q、混合和强制重复确认。重新核验q1 162、q2 156、q4 91项同工作量初筛，共409原生产者引用；余6q2＋71q4共77，其中15旧容量投影排除没有OOM实测，重新列待处理。保留旧高q/混合/确认历史，不用于三档选档。单元按既有顺序、内按q1→q2→q4，成功不重测。原协议5准备/20校准/20–100计时，跨q复用同单元q1准确次数与工作量，完整OneCycle/seed/batch/训练步开销不变。

新外置scope为`amd-execution-evidence/m7/M7-q124-coverage-completion-r1`。仅明确结构化CUDA OOM可使当前波失败、清理/退出/归属及空闲确认后局部淘汰并继续；原失败收据与部分进度保留，不将部分成功采用整波成功。其它技术错误、安全余量违规或清理未确认仍停止；不降batch、不重试、不以137或模糊字符串判OOM。最终表独立列实测最快档与5%内较小建议档，单次短测不称统计显著/全局最优，未重复不自动退q1。

按r2较晚complete采样，历史保守占卡10439.34133428149秒、余额3960.6586657185107秒，至少60秒收尾。旧阶段剩余额度明确撤销隔离，统一用于缺项；同一次cost/complete采样差0.216899941675秒不相加。逐项最近同单元实测q墙钟外推，77项低/中/高情景约1759.13/3973.16/6729.74秒（不是置信区间；并发争用/OOM/启动/身份哈希开销未知），中间情景已逼近余额，不能保证补齐。按波超时及清理准入，预算不足正常列未覆盖；本轮无新增GPU成本。

定向CPU测试与完整采用、绑定、SHA及保护结果见新scope REPORT和preparation-receipt；只测新增复用/局部OOM/预算逻辑，不重新运行模型、GPU或既有r4全回归。正式comparison-plan、216任务/343绑定/16200run-epochs/6667200updates及训练字段保持closure字节。本轮不创建approved=true、运行attempt或正式AUTH；仅为审核后用户一次启动准备命令草案。不stage/commit/push，不代称ChatGPT已审新补测实现。原§5.O.1–4为生产时点历史，不倒改。


## 5.O.6 真实任务组端到端三档对照及4线程补充（2026-11-19，待审）

用户撤换§5.O.5尚未启动的同构缺项路线，保留`M7-q124-coverage-completion-r1`原包与49项校验清单，不再用其启动命令。当前五路径closure不重做；入场HEAD 3e387a9a3b52af4f0ab02763977ac399c475a6f9、index空，仅canonical/M7原未提交两份修改；canonical入场SHA9b5e05f8f056cae65a0f0eb1c1da4951793f76c0693d77e3adf74c8100085359，M7入场SHA194297880797e1f2cda5e15e4aaaae4f3ff67f1925d186cfbce0b83037e84250。先保存改前文件再更新，不reset/stash/覆盖旧证据。

新scope `amd-execution-evidence/m7/M7-real-task-q124-probe-r1`。按当前216实际task固定54个四任务组，18组Urban同模型四H，36组非Urban跨域同模型；AMS/SZH分开，依据旧1线程q1显存与成本离线轻重搭配，q2两对预先确定。精确task ID/顺序/来源/评分与分组表保存在group-table-54.json，比较成员和任务配置在q1四波、q2两波、q4一波完全相同。满覆盖162组级条件、378波、648worker、25×648=16200探针更新，不是新增正式实验或旧162单元。

用户最新明确每worker PyTorch/OMP/MKL=4，覆盖上一指令的1；interop=1，OPENBLAS/NUMBA/NUMEXPR/VECLIB=1，loader=0。已核对第三章v10 training_config.py threads=4及worker实际设置、launcher继承环境；不假称其所有辅助库均显式设置。可见8逻辑CPU、父quota未知，q4最多16配置计算线程，允许超额订阅而不声称16独立核；所有q均保持单worker4线程。正式216配置线程尚不改，后续冻结再同步。旧1线程仅辅助，不充当新主对照。

协议real-task-e2e-25-t4-v1：fresh同seed/数据RNG/完整OneCycle，只5准备+20计时，删旧校准和自适应确认，step公式/辅助loss/clip/必要标量日志保持。整包核验只在controller准入一次且计总成本，worker核当前controller准入和逐slot任务/数据/GO身份；单task原loader及实际import身份检查仍保留。主量是四任务完整端到端T1/T2/T4及S2/S4，并列实际训练块/最慢尾部/CPU和GPU采样；不把短测加速比直接外推正式天数。新主对照直接复用0项。

组q内任一波明确CUDA OOM后拒绝整组该q成功，停止其剩余波，清场/身份/资源恢复通过才继续其它q；原失败、部分更新和成本保留。非OOM继续硬失败。单列待审差异：运行期8GiB由异常改为预警，启动及OOM清场后的空闲要求、外来进程/归属和预算硬门禁保持；批准文件必须显式认可此差异，不会在未审情况下运行。

旧保守累计10439.34133428149秒、余3960.6586657185107秒、最终60秒保留不变；新估算低/中/高约5269.22/7218.71/13365.60秒，均含一次准入情景30秒；4线程与并发争用未知，非统计保证。低情景也超余额；按高情景+清理建议另批9480秒（158分钟），建议累计23880秒，仅建议未授权。完整54组计划已准备，按组高情景+20秒准入，实际累计原14400秒硬线不变；预算不足不开始下一组，不能漏档伪装覆盖。没有为塞进余额删任务、改batch或改线程。

定向CPU验收、源码差异/前后SHA、历史保护、固定环境和完整操作草案以本scope REPORT和preparation-receipt为准。测试只用回调/假时钟及普通CPU子进程环境，不构造真实模型、不查询GPU、不执行optimizer/validation/test或读取checkpoint，不重跑既有r4回归。343正式绑定、216科学字段/16200run-epochs/6667200updates保持，comparison-plan SHA35587af552c39c6d5f5083f17335f426a8c4b6152d53159efa0b87d35e5a1771不变。无approved=true、正式AUTH、run attempt或新GPU成本；本轮不stage/commit/push，待ChatGPT审核及必要预算裁决。


## 5.O.7 双线程真实任务组联合准备（用户最新决定，待审）

用户明确批准增加每worker2线程，与4线程完整同工作量比较；不再询问加入范围，不代表增加预算或批准运行。新scope `amd-execution-evidence/m7/M7-real-task-thread24-q124-probe-r1`；§5.O.6及其未获批9480秒追加建议保留为历史。旧4线程包52文件原字节保护，原计划SHA `3ffc06e49dbb433082da0bbe2c5a49a8a4d45593efaa3a13c8fb3042ed83f52d`、harness SHA `97e02b4559a1287372910c3e41f90a08ef16ee43a7d28679c77ac923cb4cd08e`核对一致。本轮文档改前canonical SHA `05b60f0a768b36f577fb2cf62f5c4eae66867cfe53989b3da7c2691dd5b83910`，M7 SHA `26ffa13224d9598de0e48842a6227fe73cd8120decb95067e76553f9f333ff7a`，已另存before。

同一参数化harness，精确继承54组216真实task与q2配对；每组T2/T4各q1/q2/q4，奇数组T2先、偶数组T4先，同q相邻，条件串行。324条件/756波/1296短worker/32400更新（6480准备+25920计时）。每task六次均原task SHA/batch/seed/train窗口/RNG/full OneCycle/5+20更新，只线程override不同。torch/OMP/MKL=T，interop与辅助池1，loader0；真实环境/torch getter校验，线程贯穿GO/slot/目录/结果/OOM/成本身份，不能跨T混用。正式training.threads不改。T4/q4可配置16线程但可见8逻辑CPU、父quota未知；超额订阅与运行期8GiB预警仍待审，不宣称新实测。

汇总按同q配对R=T4/T2及共同成功组总时间比，再分别列全T2/全T4各组选最快成功q的完整任务集成本；缺组时总时间为null、明确覆盖不完整，胜负数仅辅助，不用均值比选线程。运行结束自动产生coverage与joint-summary；仅候选供ChatGPT资源采用审查，不能自动冻结线程/并发或开216训练。

历史占卡10439.34133428149秒只计一次，现余3960.6586657185107秒，60秒收尾不变。T2低/中/高5558.71/6826.01/14386.44秒，T4 5239.22/7188.71/13335.60秒，共享准入30秒后合计10827.92/14044.72/27752.04秒；均为基于旧1线程证据的情景，不是新实测或置信区间。T2可能较快或较慢；完整追加建议23880秒，累计建议38280秒，仅待裁决。实际代码硬限仍14400秒，组准入覆盖其全部六条件，不先跑完所有q1。任何预算/技术中止保留未覆盖，不称线程选择完成。

增量CPU使用stdlib回调、假时钟及普通CPU环境子进程；覆盖计数/顺序、task与25步一致、线程设置与实际环境、错线程GO/结果、局部OOM、非OOM停止、全组六条件预算和配对/完整覆盖汇总。本次15项增量CPU验收全部通过，最终源码对应日志用时1.291秒；结果、日志、源码对应及保护检查以本scope REPORT/preparation-receipt为准；不导入真实模型/torch、不查询GPU，不重跑不受影响的原15项及r4保护测试。训练step、两块同步、原子发布、进程身份/有界退出/所属清理/数据边界保留AST证据。正式comparison-plan SHA `35587af552c39c6d5f5083f17335f426a8c4b6152d53159efa0b87d35e5a1771`及343绑定不变；无approved=true/正式AUTH/运行attempt，不stage/commit/push。停止于ChatGPT工程审核和具体预算裁决；不代称修订字节已审核。


## 5.O.8 已批准资源合同与最终成本计账修订（待增量工程复审）

用户本次完整授权替代未发送的最小修订：原14400秒追加23880秒，累计38280秒；已用10439.34133428149秒，余额27840.65866571851秒，至少60秒最终收尾。旧T4单方案9480秒不叠加。独占归属核验后的运行期8GiB仅预警/记录、T4/q4可见8逻辑CPU上16线程对照均已批准；启动/清场8GiB及外来进程/归属/nonfinite/其它技术错误/总预算继续硬门禁。不再将资源决定写作待用户裁决；新代码整体工程审核仍Pending。

本轮起点plan SHA `954b304d1f299819f48b9cc89c61f9d4f8e9e4be3355c3f16ecc4ba7df810a05`、harness `aa4571c1a295afc02ba94834c8afdc6be914bde1618eac476a76f046dba3248b`、REPORT `12c1835d185e07c4ee708d7381c8b83a4b6dd0cd64a06f94f70c3136bf9b07d2`一致。已将原包55文件及两份当前文档共57份逐字节保存于本scope revisions/final-r2-before，原T4包及历史运行目录不改。旧plan SHA与旧启动命令作废，新的绝对路径命令仍只能增量工程审核后使用。

新增窄职责resource_accounting.py共享获批资源身份核对、deadline及终端成本校验。资源决定JSON只表用户批准，不是overall工程批准或正式AUTH；未来resource-review须精确绑定plan、harness、accounting、资源决定SHA及38280秒。运行期预警、原子发布、进程归属/退出/清理、线程身份和train边界计算机制保持；正式comparison-plan及343绑定不变，54组/324条件/756波/1296worker/32400更新、奇偶顺序、每task5+20与完整OneCycle均不变。

成功成本在payload封存、系统checksum及result-seal身份检查之后采样，最终complete.json内final_cost为唯一成功成本来源；原cumulative-cost改标非最终执行快照。失败路径保留原异常，计入部分汇总/失败记录耗时，最终final-cost.json绑定原失败；整理错误、退出未确认明确计账不完整，不覆盖原异常或制造成功。最后回执自身发布不递归计时，历史一次计入，不累计wave、不直接加预留、不裁剪真实超限。成功inspect读取新最终口径，不能再用封存前快照。

本轮只运行受修改影响的CPU假时钟/小文件/合成回执及必要普通CPU子进程检查；非零seal和失败整理耗时、新额度/批准身份/截止、原异常保护、低显存预警及硬归属/清场保持的结果记录于本scope cpu-final-r2日志与修订REPORT。原15项及r4证据按未变代码复用，不机械重跑历史模型/GPU验收。无GPU查询、模型/optimizer/validation/test或checkpoint解码；实际新增占卡0秒。预算和资源规则已批准，工程复审后用户一次启动，之后仍仅生成资源采用候选，不自动进入216正式训练。


## 5.P q1/T2正式资源采用与启动准备（2026-11-19，新字节待审）

用户明确结束资源选型，统一q1、torch intra-op/OMP/MKL=2，interop及辅助池1、loader0，各域已批batch/eval_batch不变。前§5.O系列是原时点历史；旧189.48小时候选未采用，本次不是倒改旧q1草案。原子发布五路径closure实际HEAD=tracking=实时origin=3e387a9a3b52af4f0ab02763977ac399c475a6f9、0/0，index空；入口仅canonical/M7 modified，两份改前SHA为0794629967d99cee78b0dd893010cb213445d43e4709de57daeda8de836bfdc6及14b75581e04c21b6d4f689417f001144152e44408dc8b29537193426a016b9c2，保存在新scope before。

r2导入修复经ChatGPT增量审核后用户启动：resource-plan SHA dd4adb2b99b865c700276f3eeb773723b8756fd728e9dd1318262dfdcc605b23；group-results SHA afa59ea9a7b8f342cdfc5ba1e9437ce4334f174f8c905732adc2de29b1e8b5ef；complete SHA 310b37128ed71066b0c14b02af900d8dbff7cb21d6ec590ecd16c4874bd4ceb3。封存完整性只读核验通过：54组216真实task、324条件756波1296worker、32400更新均完成，无OOM/技术失败/预算未覆盖。本次terminal-card-cost-v2为8271.264713955112秒、累计18718.58358510118秒，包含此前r1导入失败7.977536864578724秒一次；原失败、not_admitted、producer、日志和成本不改。独立资源采用记录在M7-q1-t2-resource-adoption-and-formal-preparation-r1；不将探针38280秒预算移植为正式运行截止。

216原task ID和原顺序精确保留，train波次每项一个；全训练/validation→全best冻结→原序test各一次→seal。逐task除training.threads=2和已解决readiness外科学字段零变化；16200最大run-epochs/6667200更新、50/20与200/40保持。resource_frozen=true、execution_authorized=false；本轮新工程审核、精确closure、实际AUTH未完成。已完成探针、生命周期及原子发布修复不再保留机械Pending。外置AUTH绑定未来实际commit，不预填，不通过修改已提交文件回填。

新增窄职责ch4/resources.py及formal_resources.json：启动环境、实际torch getter、q1顺序、batch/资源身份、合法单GPU分配/UUID-minor与启动空闲检查；queue在原-m ch4.queue/cwd工作区的独立进程中调用，train/test同线程入口，记录各phase线程身份。生成器使用T2并绑定采用证据。原worker训练计算、optimizer/scheduler、模型、数据、test claim和首失败停止逻辑不改；正式OOM不局部跳档。源绑定343→349：新增resources.py、formal_resources.json、独立adoption和原探针plan/group-results/complete共6项，不删旧项。资源接线/初始GPU身份查询的代码是新增待审字节；本轮未实际查询GPU。

定向CPU六项通过（7.249秒）：同task科学与预算/q1覆盖、负向资源/线程、train/test真实空载CPU子进程线程getter及包导入、缺少/未审核AUTH、合成GPU分配拒绝、真实supervisor少量合成任务全训练/冻结/test-once/seal及环境传递。不构造真实模型、不调用CUDA、optimizer或真实观测数组，不重跑无关模型与历史探针。旧恢复证据仍仅helper/worker接口。新日志、diff、SHA及closure候选清单以本scope材料为准，不代称新字节已审。

逐task同T2/q1的20步直接证据映射全部216任务，按6667200最大更新外推纯训练约192.995小时；50%/75%/100%更新量只是排期情景，不假定某固定epoch早停。validation训练量10–30%、checkpoint/I/O 5–20%、test全上限训练量1–3%均明确为未实测假设；加一次性初始化/收尾后最大情景约224–295小时，非保证区间或硬截止，不因此补探针。q1/T2是用户选择，不宣称baseline普遍无并发收益或GPU必满载。

启动草案使用固定环境和绝对路径tmux，queue.start仅初始联网准入一次，后台worker不逐task联网；返回前确认owner/runtime及首worker线程身份。未生成可启动正式AUTH，不创建运行attempt。下一步ChatGPT工程审核→精确closure→实际commit绑定及AUTH→轻量preflight/用户一次启动，不另加机械回填审批。
