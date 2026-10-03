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
