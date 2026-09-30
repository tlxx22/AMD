# M6：第三章正式实验与定稿

**In Progress — 原267项及TimeMixer修订16项已接受来源保持；41-run补做已完成，本轮旧绑定版本完整性审计通过、结果待ChatGPT审核。剩余六模型228任务总队列为未提交待审核实现，未启动。当前决定与边界见§5.14。**

原开篇“尚未运行任何正式训练或正式test评价”属于初次启动准备时的历史快照，不代表当前实际进度。

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
