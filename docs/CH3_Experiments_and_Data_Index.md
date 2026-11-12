# 第三章实验说明与数据索引

## 全覆盖数据采集：工程已审，进入closure与正式启动材料交接

用户确认ChatGPT已通过Desktop Commander实际直读本轮13项目路径、161依赖、206外置成员/207校验，工程review Passed；8独立CPU用例/8执行通过，合成1构造/3前向/5呈现，真实负载尚未启动。旧四H/NP的41receipt/29波/24claim、198轮/1163164更新核账仍限封存、相关缓存及权重metadata，不升级完整state/Adam/RNG审计或CUDA恢复。新39冻结test评估＋O按40串行波/q1覆盖54单元、6600区域指标；15整组与5部分组保留真实producer/test来源，不混validation。资源“基于既有证据复用与差异评估接受；本轮未重新实测”，不宣称新GPU峰值/速度/真实恢复或科学效果Passed。

原准备包`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-full-region-five-market-preparation-r1`及其Pending历史字节保持。此次closure入口为`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-fullcoverage-review-closure-r1`；条件成功后正式许可入口为`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-fullcoverage-launch-authorization-r1/AUTH.json`，须绑定实际新commit并通过轻量preflight，是否已生成以外置回执为准。配置仍`configs/m6_ch3_fullcoverage_v10.json`，两个参数启动脚本`scripts/ch3/start_m6_ch3_fullcoverage_v10.sh`。规划结果`/public/home/yueweiting/大论文/amd-execution-evidence/m6/moderntcn-uv-v10-fullcoverage-reporting-runs/all275-4h-epf5-sixmodels-first-r1`执行前尚不存在，不能写为完成；正式负载由用户最后启动，执行期间不改Git绑定工作区。

只采集全量数据，不选区域或按指标增删曲线。逐位置组合由用户预先规划与可读性决定，具体名单未传入不表示结果驱动选择，也不阻塞数据覆盖。区域全表仅内部产物，不要求正文/附录展示；最终名单、曲线与论文叙事后置，不固定“八次优”配额。原四图/旧结果保留；Goal1/2 Met、Goal3 Not met、单seed与开发test边界不变，新结果review在执行完成后进行。本指南继续单份维护，不替代canonical/唯一M6，不新增平行索引。

## 最新资料：旧四H/NP批次已完成；六模型全覆盖数据采集实现待审、未执行

本指南仍为派生材料导航，不替代canonical与唯一M6。以下旧“未启动/规划”段落保留历史时点；现已完成来源为`/public/home/yueweiting/大论文/amd-execution-evidence/m6/moderntcn-uv-v10-ch3-amendment-runs/urbanev4h-np-amd-timexer-first-r1`。其`complete.json`、`runtime.json`、41 receipt/29波/24 claim、`plots/sensitivity-means.json`、`plots/five-model-position-summary.json`、`plots/source-ledger.json`及原四区域图均保留。训练198轮/1163164更新，test17218 batch/1480760呈现/3886560元素；本轮核账为文本、343非checkpoint payload、32权重metadata与20个所需旧缓存，不是完整state/Adam/RNG/CUDA恢复。先转FP64相减的历史异序差异与按生产顺序24项exact的ChatGPT复核声明并列保存。

已完成敏感性主口径为UrbanEV四H等权＋NP：默认G5/R4在UrbanEV五点内test MSE/MAE最低（0.42970876852184436/0.4417005394048928）；NP默认MSE最低0.2623656241180711，G3/R4 MAE 0.29216226998993505略低于默认0.2933803950496397。只描述固定点集、单seed及已有test开发反馈，不据test换主模型、不称全局最优。Goal1/2 Met、Goal3 Not met及PJM历史补充保持。

新已生成准备入口：`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-full-region-five-market-preparation-r1`。`source/sources.json`逐54逻辑单元绑定原数据/protocol/profile/scaler/metrics/spec/strict-best/checksum与真实producer，`source/main-45-bindings.json`对应当前统一第二轮MS主表45baseline单元（PatchTST UrbanEV2层、EPF3层），UV九源沿批准v9→v10 P1链/原v10。`audit/reuse.json`导航NP/BE/DE15整组与H6五部分组的原receipt/runtime/cache，不能假定全部产于同一目录；`plan/urban-H3-all275.json`等四H文件及`urban-H6-remaining265.json`给出原metadata推导的样本键与SHA。原数据路径在每行`data_ref`、`data_protocol`及`test_arithmetic`，本轮不读取新原始test。

新范围：ModernTCN B_cfg、TimeMixer、TimeXer、AMD、PatchTST、默认完整UV六模型；UrbanEV全部275节点×四H×共同2023-02-13至02-19目标周，以及五EPF完整test/P24。39项新增冻结test评估＋O、40串行波，10973 batch/1257300呈现/4866000元素；15整组及5部分组引用不计新增前向，无训练。H6原十区域每模型1680条缓存保留，只补265区域；H3/H9/H12分别推导各自origin/index，不能复用H6预测。全部54单元保留标准化/物理单位、时间/node/H、真实生产者与缓存血缘。

规划输出（当前尚不存在）：`/public/home/yueweiting/大论文/amd-execution-evidence/m6/moderntcn-uv-v10-fullcoverage-reporting-runs/all275-4h-epf5-sixmodels-first-r1`。各job的`predictions.npz`、`source.json`、`result.json`、`access.jsonl`及claim/receipt供追溯；`data/source-ledger.json`连接新39评估和旧20引用，`data/merged/`仅保存H6五组无交无漏的合并缓存；`data/region-metrics-6600.csv/.json`、`region-wide-1100.csv`、`EPF-position-all.csv/.json`与五份`EPF-市场-six-model-data.json`为全量机器可读实验数据，不要求论文展示全区域表。每市场数据含MSE/MAE两套24位置序列，位置不是不同horizon训练。

本轮不选区域、排名/阈值或最终3/4曲线名单。每H两例/八图只是后续意向；原四展示图与选择事实保留，不能冒称新最终方案或总体随机代表。数据完成后用`m6_ch3_fullcoverage_v10.entry plot-region`显式给H/node_ids/模型列表，或`plot-epf`显式给市场/模型列表，仅读缓存生成SVG/PNG，不再次推理。最终区域主图意向truth＋UV＋ModernTCN＋TimeXer，可选PatchTST/AMD以后决定；五市场最终曲线名单尚未指定。数据采集照样保留全部六模型，不按成绩增删、截不利尾部或凑八次优，不写论文正文。

新代码8独立CPU用例/8执行通过，合成1构造/3前向/5呈现，旧缓存离线读20份单列；真实权重反序列化/原始test/模型负载/GPU查询0。合成全54单元汇总及3份SVG/PNG仅验收，不是科学图稿。161依赖、资源按既有证据与差异评估接受，本轮未重新实测。`REPORT.md`、`file-manifest.json`、`manifest.json`、`non-execution-preparation/OPERATIONS.md`和false/null模板统一待ChatGPT审核；未commit/push、未正式preflight/AUTH/attempt，真实负载由用户后续启动。执行中不改绑定Git工作区，完成审计后继续维护本指南，不另建状态索引。

## 四H/NP与五模型扩展：工程已审，正式执行材料定版、待用户启动

ChatGPT已实际直读审核本scope，工程review Passed限实现、来源绑定、有限CPU与控制逻辑。152依赖、14路径、323成员/324校验；12独立用例最终通过，13执行12成功1历史fixture失败，来源检查失败及typed/篡改拒绝记录保持。UrbanEV四H等权＋NP完整25格采用原H6四点与五中心引用，PJM敏感性保留历史；五模型逐位置与十区域/锁定四图规则不变。q4训练/q1评估按既有证据和差异评估接受，本轮未重新实测。历史内嵌指针差异仅作已核实执行链范围内限定接受，不改旧protocol，完整协议/active data_ref/profile/stat继续严格绑定。Goal1/2 Met、Goal3 Not met不变；新训练/test/GPU恢复及科学效果尚未执行，不把规划输出标为已完成。

closure回执入口：`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-urbanev4h-np-baseline-extension-review-closure-r1`。closure成功后正式启动材料入口：`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-amendment-launch-authorization-r1`；正式AUTH绑定实际新commit，原false配置和模板不改。规划结果仍为`/public/home/yueweiting/大论文/amd-execution-evidence/m6/moderntcn-uv-v10-ch3-amendment-runs/urbanev4h-np-amd-timexer-first-r1`，尚未创建或启动。实际三文档SHA/commit及记录层限定在外置回执保存，本文不递归回填；运行中不修改绑定工作区，完成后审计并更新同一指南。

## 当前采用口径增量：四H/NP敏感性与五模型对照（实现待审，正式负载未执行）

最新用户决定将主敏感性改为UrbanEV H3/6/9/12四H等权与NP；旧H6单任务/PJM仍有真实结果，保留为历史补充，不能将其最优点写成UrbanEV总体或NP结论。完整覆盖为25格：16新非默认训练＋9引用；训练上限320轮/1804640更新，原strict validation MSE-best及同checkpoint双指标不变，新训练封存后16独立首次test，不用test选配置/seed/epoch或替换主模型。

新逐位置对照为NP/BE/DE×ModernTCN B_cfg/TimeMixer/TimeXer/AMD/完整UV，共15格；旧9结果复用，只新增6次test。区域仍UrbanEV H6，原十节点208/330/333/570/969/1083/1113/1120/1162/1173、2023-02-13至02-19测试周；复用旧三模型缓存，只新增AMD/TimeXer各一次。四图节点固定1162/1083/1120/969，不按新增baseline重排，图中真实值＋五模型，十区域全表不能删除不利行。四图是原两baseline比较后选出的展示子集，不是预先随机四区域或全275区域代表。新24 test＝16首次＋8后评估，17218 batches/1480760呈现/3886560元素；41 jobs/29波，评估q1、训练四波q4。

现有真实结果入口：`/public/home/yueweiting/大论文/amd-execution-evidence/m6/moderntcn-uv-v10-test-reporting-runs/test-reporting-v10-import-retry1/complete.json`，其`plots/sensitivity-all-points.json`、`case-plan.json`、`plots/display-selection.json`和原source-ledger保留。旧S8、9逐位置和3案例的真实producer/receipt/预测缓存不能改名为本批前向；旧失败test/权重读取成本和旧validation图不抹除。本轮仅封存文本、权重stat和必要已存缓存核对，没有新原始test或权重加载，也未补状态级/CUDA恢复审计。

新资料入口（均服务器端实际已生成）：`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-urbanev4h-np-baseline-extension-preparation-r1`。`source/sources.json`导航真实metrics/spec/profile/config/data/weights-checksum；`source/sensitivity-25-cells.json`列每个H/G/R的引用或计划生产者；`source/main-index-eight-bindings.json`核对当前统一MS主表八baseline单元；`audit/reuse.json`记录旧引用与缓存核验；`non-execution-preparation/jobs-waves-budget.json`及`OPERATIONS.md`为41任务/29波及不可执行命令草案。历史protocol内嵌时间标记SHA差异见`audit/historical-inner-reference-difference.json`，原文件未改，当前执行data_ref/profile/stat及完整protocol身份另核对。新配置`configs/m6_ch3_scope_amendment_v10.json`、独立入口`m6_ch3_scope_amendment_v10.entry`复用原模型，source实际152项。

规划输出（当前尚不存在）：`/public/home/yueweiting/大论文/amd-execution-evidence/m6/moderntcn-uv-v10-ch3-amendment-runs/urbanev4h-np-amd-timexer-first-r1`。未来`plots/sensitivity-25-cells.csv`与`sensitivity-means.csv`保存逐H及四H等权/NP指标，`five-model-position-all.csv`覆盖全部24位置，`ten-region-five-model.csv`保存十区域完整对照；SVG/PNG与同源JSON/CSV一起生成，`plots/source-ledger.json`链接每项真实producer/receipt/result/cache。这些新科学图稿当前未生成，CPU合成图稿只用于模板验收，不是结果。

新实现12独立CPU用例最终通过，13执行12成功1保留fixture失败；另保留来源检查失败及定向校验。合成4构造/6前向/10呈现、1合成state写读，backward/Adam0；真实模型/权重/raw test/训练/GPU0。仅false/null模板、无正式AUTH/attempt/preflight/启动，无commit/push；全部新字节待ChatGPT统一审核。资源按既有证据和差异评估接受，本轮未重新实测；Goal1/2 Met、Goal3 Not met、单seed与开发test限制不变。下文旧未启动表述保留历史时点，本指南不替代canonical/唯一M6。

**冻结test报告技术续办已实现；用户明确授权验收后由Codex启动。** 用户本轮答复“完成续办接线、验收并启动”，范围保持原20模型评估＋O，不新增科学任务。此前导入时序修复的3个CPU场景复用；本轮6独立控制/缓存用例最终通过，7次用例执行为6成功＋1测试异常类型写错，另一次定向调用在测试导入前因缩进错误失败，全部日志保留，未放宽保护。已接通原8份S-test真实receipt引用、剩余12模型评估/13波及两来源O汇总；旧99文件/失败/test访问成本保留，新attempt不伪造旧receipt，第一项固定PT-NP-ModernTCN。新配置为`configs/m6_ch3_test_reporting_v10_import_retry1.json`，规划attempt为`test-reporting-v10-import-retry1`；剩余780 batch/99180呈现/2264400元素，零训练/backward/optimizer。新控制验收只用CPU空回调和已存合成缓存，真实权重/模型计算/test/GPU均0；启动后真实评估消费另记。代码依赖142项，原模型/guard/数值政策、固定十节点/测试周/选后四图、Goal1/2 Met和Goal3 Not met不变。续办证据入口`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-test-reporting-import-continuation-r1`；新正式许可入口`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-test-reporting-import-retry1-launch-authorization-r1`，须绑定本次实际提交并通过preflight后启动。此段是提交前时点，实际启动/receipt在外置证据登记；不声称ChatGPT已读本次新实现字节，不在运行中改Git绑定文档。下文局部修复未接通/旧待启动文字保留各自历史时点。

**冻结test报告导入时序局部修复（尚未续跑）。** 首批S-test8已生成有效receipt，3924前向batch/501640呈现/1463960元素；原99个attempt文件核对保持。第9项PT-NP-ModernTCN已读取test10460窗口、恢复权重并加载CUDA，在首次前向前因作者baseline切换utils搜索路径后才导入项目batch_parts而失败，不能记为零test/GPU消费。本轮仅将evaluate.py的batch_parts导入提前至模块级，模型/数学/guard/旧配置及产物不改。独立CPU子进程3场景最终通过：旧序复现拒绝，ModernTCN与TimeMixer的有/无marks路径在utils切换前后输出/状态/BN/RNG exact；新合成模型3构造、12前向/20呈现，真实checkpoint/test/GPU及backward/Adam均0。首轮测试工具取错source.root的3次失败发生于模型构造前，修正为原protocol.sources路径并保留日志，未放宽断言。本轮修复证据位于`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-test-reporting-import-order-repair-r1`。现scope明确拒绝既有attempt且汇总只认同目录20份结果，安全续跑还需8项原结果引用＋剩余12评估/13波及两来源汇总接线，当前未实现或启动；旧AUTH不能消费修后源码。原Goal1/2 Met、Goal3 Not met不变，不自动重测8成功项、不覆盖原失败，不预称新修复字节已获ChatGPT审核。

## 冻结test报告执行材料入口：工程已审、待用户启动

用户确认ChatGPT已实际直读审核新scope，工程review Passed（限实现/来源/权限/CPU验收/控制），9独立用例/11执行及工具错误、定向补强记录保留。旧47任务核账只覆盖文本/非checkpoint payload/权重metadata；S8实际117轮/369113 updates、原test0，未补完整state/Adam/RNG/CUDA审计。新20项test评估＋2中心旧结果引用按21波q1执行，600820呈现/4704 batch/3728360元素；零训练/backward/optimizer。137候选固定随机10、十区域全表及选后4图规则不变，四图为典型/优势展示子集，不称全部UrbanEV随机代表；敏感性test不用于重新选G/R/seed/checkpoint或替换主表，Goal1/2 Met、Goal3 Not met保持。资源基于既有证据及差异评估接受，本轮未重新实测；尚无新test结果，不能宣称本包GPU恢复或数值一致性已实测通过。

本次closure材料：`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-test-reporting-review-closure-r1`。closure成功后正式授权及preflight材料入口：`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-test-reporting-launch-authorization-r1`，其中`AUTH.json`由本轮已审false/null模板绑定实际新commit后派生，不能使用模板启动。配置为`/public/home/yueweiting/大论文/ModernTCN-U+V/configs/m6_ch3_test_reporting_v10.json`，启动入口为同W下`scripts/ch3/start_m6_ch3_test_reporting_v10.sh`（配置、正式AUTH两个参数）。规划结果目录`/public/home/yueweiting/大论文/amd-execution-evidence/m6/moderntcn-uv-v10-test-reporting-runs/test-reporting-v10-first-r1`当前尚不存在，待用户启动；原E报告/patch/Pending和旧validation/test0产物不改写。正式运行期间不修改Git绑定工作区，完成后再审计结果与更新本指南，不另建索引。

## 当前增量：旧validation补充包已完成，新test报告只完成实现准备

原补充数据现已存在于`/public/home/yueweiting/大论文/amd-execution-evidence/m6/moderntcn-uv-v10-ch3-supplement-runs/ch3-supplement-v10-retry2`：`complete.json`及`plots/source-ledger.json`保留47逻辑job的12继承＋35新执行来源。E32性能、原105/408区域validation案例、PJM validation逐位置及S8 validation结果均保留，不改标test。S8实际117轮/369113 updates，原批次test_calls=0。此次完成核账不包含checkpoint反序列化或完整状态级审计；旧未启动段落继续作为历史记录，不代表当前未完成。

新证据目录`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-test-reporting-preparation-r1`的`audit/completion.json`、`source/sources.json`、`source/preselection.json`及`non-execution-preparation/arithmetic.json`分别导航至旧完成成本、20评估＋2中心引用真实producer/权重封存位置、实际137→随机10名单、逐job样本/元素/batch。权重仅定位，不复制或加载。新代码`m6_ch3_test_reporting_v10/`及`configs/m6_ch3_test_reporting_v10.json`是独立test scope，旧许可保持validation-only；本轮9独立CPU用例/11执行通过，新字节待ChatGPT审核。

用户本轮新增test授权为S8首次test、NP/BE/DE各B_cfg/TimeMixer/UV共9次完整test、UrbanEV同三模型共3次固定test周案例；总20模型评估＋离线汇总，零训练。两个默认中心只引用既有正式test；全部strict validation MSE-best，同checkpoint双指标，不改变主表或选新G/R。预计4704 batch/600820呈现/3728360元素；案例3×10×168＝5040呈现。真实test/GPU/权重恢复尚未执行，尚无新增test结果；CPU fixture图稿明确标合成，不是科学结果。

资格集合实为137而非142，来自旧train统计中等需求范围。PCG64(2024)随机无放回十节点，显示序208、330、333、570、969、1083、1113、1120、1162、1173。十区域全表必须保留；共同test周、H6滚动P1，再按四项胜数/有效相对改善/同单位绝对差额/node_order选4展示，0分母NA且不重抽。最终4图须标“所抽10区域中的典型/优势展示案例”，不称总体随机代表，不删不利行。选周仅时间metadata，正式预测前固定精确样本键；所有模型同周同节点。

规划（尚不存在）新结果：`/public/home/yueweiting/大论文/amd-execution-evidence/m6/moderntcn-uv-v10-test-reporting-runs/test-reporting-v10-first-r1`。将生成各job的`access.jsonl`/`predictions.npz`/`result.json`和独立claim/receipt，`plots/`内敏感性validation/test分列、三市场24位置、十区域全表和四案例SVG/PNG及source-ledger。只能缓存重绘，不反复前向；原数据路径沿source ledger和原protocol定位，不能把旧validation缓存当test。当前仅false/null模板与操作草案，无正式AUTH或启动。Goal1/2 Met、Goal3严格链Not met和单seed/test开发反馈限制不变；本指南仍为派生资料入口，不替代canonical/唯一M6，也不开始论文正文。

**GPU可见性绑定修正，准备材料待审。** 旧正式preflight因原身份生成时隐藏GPU、启动时`CUDA_VISIBLE_DEVICES=0`而拒绝；同一torch_build由10行变14行，精确新增CUDA Runtime 11.8、完整NVCC架构flags、CuDNN 8.7、Magma 2.6.1。旧/新文本SHA分别为`b384ae4b0a6c7e141a83e78638b2d557a7f3b843b4d63a9104672e65ee239205`、`8c454fd3f128eca1154bd540123adafcf6b7111ea3a165b5ec433d4baf937a93`；同一torch._C SHA为`1277e4be87e0b15987f796117a7422c7a445b4df9213ecf2ddc3de718b0d66fc`，CUDA未初始化。新证据入口为`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-cuda-visibility-binding-r1/environment/target-visible-gpu0.json`，SHA `b29414000d07ee0fdc570fe3b513b0ebcaf3805473dfc3b642602998ae4ec41d`；仅retry2配置environment_ref改指此处，仍严格比较全部身份，不改变启动器GPU可见性或任何模型/科学参数。该目录REPORT、environment与checks提供四行比对、8项窄范围断言和不可执行模板；原target和AUTH-r1/preflight失败材料保留，旧AUTH不能用于新配置。本轮不运行正式preflight，不创建AUTH-r2或真实attempt，未执行新模型/test/GPU负载。旧retry1的12项成功仍为原生产数据，剩余35项/29波和S8尚未执行，retry2目录仍为规划。Goal1/2 Met、Goal3 Not met及单seed/历史test限制不变；本指南原数据与未来结果入口沿下文保留，不能将此次身份准备当作新增实验完成。新字节尚待ChatGPT审核。

**2026-11-11审核裁决与数据位置。** ChatGPT已实际直读环境修复包声明字节，环境接线/12项继承工程review Passed。原12项E测量正式保留在`/public/home/yueweiting/大论文/amd-execution-evidence/m6/moderntcn-uv-v10-ch3-supplement-runs/ch3-supplement-v10-retry1`，真实producer/runtime/venv及SHA不改写。剩余35项/29波计划目录为同一result_root下`ch3-supplement-v10-retry2`，当前准备交接、未启动；未来完整数据通过受限来源索引合并，不能把35项写成47项新执行。复用依据为数值库路径/所查字节、UUID/flags/计时协议及差异评估，不是跨环境逐位或GPU恢复实测证明；TimeXer真实恢复、剩余性能、案例/PJM和S8仍未完成。旧两次失败/成本、Goal1/2 Met、Goal3 Not met及单seed/开发test限制保持。本指南仍为唯一资料入口，旧Pending历史不倒改，新增closure/AUTH字节不预称已被ChatGPT读取。

**2026-11-11增量：已产生12项性能测量，整批补充实验尚未完成。** 实际旧attempt为`/public/home/yueweiting/大论文/amd-execution-evidence/m6/moderntcn-uv-v10-ch3-supplement-runs/ch3-supplement-v10-retry1`；其`receipts/`和各job/result.json保留UrbanEV下UV/AMD/DLinear/ModernTCN/PatchTST/TimeMixer各b1/b16的12项结果（6600调用/56100呈现）。首TimeXer依赖导入失败已封存；后续案例、PJM逐位置与敏感性仍未完成，不能把测试fixture的图稿当正式图。原模型成绩与Goal1/2 Met、Goal3严格链Not met保持。

当前环境修复包`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-environment-repair-r1`提供`audit/inherited-receipts.json`（12旧receipt/60payload及真实producer）、`environment/target.json`（指定既有venv身份）、`non-execution-preparation/queue.json`（35剩余job/29执行wave）及REPORT。原数值核心库相同、计时元数据一致支持有限保留，未保存的历史全量模块/瞬时负载不补造；本轮只独立导入和CPU控制验收，真实GPU修后整批成功尚未验证。封包工具另保留1次返回码处理失败：git diff --no-index --check对新增文件返回1且诊断为空，工具原误要求0；已按实际差异返回码修正封包判定，不修改实现字节或放宽格式检查。最终CSV/plots/source-ledger.json将通过受限来源解析器记录每行真实旧/新producer、runtime、environment和SHA，不把继承数据改名为新执行。

规划完整结果目录：`/public/home/yueweiting/大论文/amd-execution-evidence/m6/moderntcn-uv-v10-ch3-supplement-runs/ch3-supplement-v10-retry2`，**规划、尚不存在、尚未启动**。新配置为`configs/m6_ch3_supplement_v10_retry2.json`，来源包模板commit=null/review_approved=false，不是正式许可。原47job/41wave由12继承＋35执行闭合，S8仍从头训练、全部新test0，原case不重新选择。本轮新控制/配置及三文档统一待ChatGPT实质审核，不宣称已读新字节；原指南入口和历史时点原样保留。

**PID修复工程审核裁决与retry1启动交接。** 用户确认ChatGPT已通过Remote Desktop Commander实际直读审核：固定3.10平台PID归属、拒绝与证据保存、技术retry1绑定工程review Passed；范围限已读源码/来源及11独立CPU用例最终通过、14执行12成功2历史失败，mtime取证选择修复及原失败保留。CPU验收未运行真实GPU；旧first-r1至少发生一次权重恢复/模型构造/CUDA加载，0预热/计时/训练/test，历史GPU PID列表缺失，仍不能排除当时外部负载。修复只白名单映射后的一个driver PID，未知/额外PID/缺自身仍拒绝，证据先保存，管理检查不进入计时区；修后真实GPU执行尚未验证。原Goal1/2 Met、Goal3严格链Not met不变，原送审E/Pending及旧失败产物封存不改。本次进入精确closure及用户启动交接，新技术attempt `ch3-supplement-v10-retry1`准备就绪、未启动；原47job/41wave、S8/160轮617120更新/test0及固定case-selection保持。规划路径为`/public/home/yueweiting/大论文/amd-execution-evidence/m6/moderntcn-uv-v10-ch3-supplement-runs/ch3-supplement-v10-retry1`，不能称已创建或已有新效果。配置execution_authorized=false及preparation_head保持，正式AUTH仅在closure成功后另绑定实际commit；真实负载由用户最后启动。运行期间不修改绑定工作区，完成后再登记实际结果。

**补充包首次启动技术失败及PID归属修复（待审）。** 原输入消融review Passed、Goal1/2 Met及Goal3严格链Not met不变。补充队列已由用户实际尝试启动，旧`ch3-supplement-v10-first-r1`完成0/47、S完成0/8；首E项已发生至少一次真实权重恢复、模型构造及CUDA加载，在预热/计时前被GPU归属保护拒绝，0预热/计时/训练/新test，显存与耗时未保存不补造，不能称旧尝试GPU消费0。旧runtime/failure/AUTH/config/日志/case-selection及空job目录保持。当前3.10.0-1127.el7.x86_64平台已实证local PID与/proc/self/sched首行p->pid不同；原事故GPU PID列表缺失，不能百分百排除事发外部负载。修复只针对新补充控制代码：同一存活leader的status Pid/Tgid、stat start、namespace和sched原文前后绑定，目标GPU仅白名单一个driver PID，未知/额外PID/缺自身均拒绝，原始管理查询与判定在抛错前落盘；启动前/模型加载前/计时前后有界检查不进入计时区。原模型/guard/benchmark科学算术不改。11独立CPU控制用例最终通过，14执行12成功2失败；两次失败来自测试mtime取证歧义，改为本次唯一新增文件定位并定向复验，原失败保留。仅CPU控制及本机自身/parent-child /proc，真实模型/权重加载/CUDA/真实GPU查询/forward/backward/Adam均0；550次成功计时控制空回调两次共1100次，不是模型前向。新技术attempt仅规划`ch3-supplement-v10-retry1`，47job/41wave及S8/160轮617120更新/test0完全不变，固定沿原case nodes=['105','408']/indices=[2,54]、2023-01-30至02-05目标周；不恢复checkpoint、不增加科学额度或择优重跑。新配置/不可执行模板commit=null、review_approved=false，未创建新runtime/attempt、未正式preflight或启动。修后证据在`M6-ModernTCN-UV-v10-ch3-supplement-gpu-pid-repair-r1`，统一待ChatGPT审核，旧Pending历史不倒改。

本指南是面向查阅、复现及后续图表的派生材料导航，不是第二份milestone或第三章论文正文。权威合同见[canonical](AMD_EV_Thesis_Final_Implementation_Plan_v2.1.md)，进度和审核见[唯一M6](milestones/M6_formal_experiments.md)及实际证据。原报告的Pending/准备文字保留原历史时点，不倒改。所有下列绝对路径均在服务器 **b82673a648b4**；当前维护工作区为 `/public/home/yueweiting/大论文/ModernTCN-U+V`。外部AMD等路径是原数据/producer或来源地址，不是本指南另一个维护工作区。

## 阅读口径

当前主模型为 **ModernTCN-U+V v10**，B为`cfg_adapted_v1`；C.1未采用，不能把本模型称为旧AMD/J/EL-AMD。七baseline为AMD、DLinear、ModernTCN、PatchTST、TimeMixer、TimeXer、iTransformer。UrbanEV固定MS、F4、fold6、275节点、split[3475,3909,4344]、L12/P1、标签间隔H3/6/9/12；P1代表单个输出值，H代表该值距预测原点的间隔。EPF五市场PJM/NP/BE/FR/DE为MS、L96/P24，H24对应连续24个输出位置，不是UrbanEV式四个独立H任务。

原train-only scaler/schema/特征顺序、seed2024、float32、MSE、Adam/OneCycle、max20/patience5、严格validation MSE选best及test-once保持；同一MSE-best checkpoint报告MSE和MAE，不另选MAE最佳轮。当前UV batch/eval128；baseline以每条原profile为准，TimeXer既有训练batch差异保留，性能共同batch不回填训练配置。单seed及test曾用于开发的限制不变，不称显著稳定或独立盲测。原始数据单位、dataset scaler后的评价单位及RevIN模型内部单位必须区分。

所有等价引用保留真实producer commit/task/config/revision。v10的Urban P1 U/UV来自批准v9链，V来自V3；不是v10重新训练或checkpoint迁移。不同输入C导致native参数量变化，不用补层凑参数。下述表行数是评价覆盖，不能相加为独立训练次数。

## 已完成且当前采用的实验

|类别|研究问题、对象与控制|覆盖及统计|结论与边界|
|---|---|---|---|
|A 整体性能|七baseline与完整UV；同MS任务/数据/target，保留各自真实实现profile|九任务×八模型=72行；UrbanEV四H等权，EPF五市场分列，不混平均跨域原误差|展示完整当前主表，不将Goal2写成胜全部baseline；单seed比较不证明稳定领先|
|B 模块消融|B/U/V/UV；九任务，数据与训练合同固定|36行=9个B引用＋27增强臂评价格；27由10项EPF U/UV fresh、9V3引用、8v9 Urban U/UV引用组成|Goal1 Met：Urban双指标等权均值胜两单臂，EPF仅NP同市场四比较通过。Goal2 Met 7/10仅相对B_cfg：MSE NP/BE/DE，MAE NP/BE/FR/DE；PJM双指标和FR MSE未胜B保留|
|C 输入消融|完整UV始终开启，F1/F2/F3/F4各四H；除schema/C及相应参数shape外训练/窗口/标度固定|16格=12fresh＋4F4等价引用；分别按双指标四H未舍入等权|保存结果review Passed；Goal3完整覆盖但严格链Not met，F4平均最佳，F2/F3未胜F1；不改阈值或重跑修饰|

F1=目标volume＋日历（C6），F2=F1＋价格（C8），F3=F1＋天气（C9），F4全输入（C11）；F1不是纯目标历史，F3不是F2加天气。schema以`utils/feature_schema.py`为准。Goal3要求每指标F4<F2<F1且F4<F3<F1，不要求F2/F3排序、逐H全胜或B链；缺项NE、平局不通过。输入消融原成本154 run-epochs/1143574 updates/12原test，F4新增0。v10 EPF批原成本130轮/37050 updates/10原test；17引用另保留原生产成本，不计本批新增。

百分比统一为`100*(比较值/参照值-1)`，负值表示误差下降；严格胜负用未舍入值，展示舍入不改变gate。MSE/MAE由原SSE/SAE/elements得到；UrbanEV按四H等权，不按不同H窗口数重新加权。Goal1/2已审结果与Goal3负向结果同时保留，不按市场拼接旧候选。

## 数据、配置与产物的精确导航

以下“已存在”仅表示本轮静态路径核验，不代表重新加载权重或执行模型。每个文件所在机器均为上面的服务器；producer/consumer的连接由指定键及ledger保存，目录路径不代替封存文件SHA。

### 原始输入位置

|用途|精确路径|存在状态/逻辑键|SHA或定位|
|---|---|---|---|
|BE原始输入|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/data/timexer-76011909357972bd55a27adba2e1be994d81b327/BE.csv`|已存在；protocol.datasets.BE|见封存ledger/manifest；目录不计算SHA|
|DE原始输入|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/data/timexer-76011909357972bd55a27adba2e1be994d81b327/DE.csv`|已存在；protocol.datasets.DE|见封存ledger/manifest；目录不计算SHA|
|FR原始输入|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/data/timexer-76011909357972bd55a27adba2e1be994d81b327/FR.csv`|已存在；protocol.datasets.FR|见封存ledger/manifest；目录不计算SHA|
|NP原始输入|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/data/timexer-76011909357972bd55a27adba2e1be994d81b327/NP.csv`|已存在；protocol.datasets.NP|见封存ledger/manifest；目录不计算SHA|
|PJM原始输入|`/public/home/yueweiting/大论文/TimeXer/dataset/EPF/PJM.csv`|已存在；protocol.datasets.PJM|见封存ledger/manifest；目录不计算SHA|
|UrbanEV原始输入|`/public/home/yueweiting/大论文/AMD/data/UrbanEV/data`|已存在；protocol.datasets.UrbanEV|见封存ledger/manifest；目录不计算SHA|

UrbanEV目录内读取volume.csv、e_price.csv、s_price.csv、weather_central.csv；日历按原管线生成。它们不是运行输出或validation缓存。原数据边界、scaler、node_order与prefix fingerprint在每条data_ref/profile中；不把prefix hash误称全文件hash，不重新fit或读取新test观测。

### 配置、入口及审核材料

|用途|精确路径|存在状态/逻辑键|SHA或定位|
|---|---|---|---|
|冻结v10科学配置|`/public/home/yueweiting/大论文/ModernTCN-U+V/configs/moderntcn_uv_fold6_epf_horizon_coupled_spline_v10.json`|已存在；文件本身|`24eb43ad65b9a105416ab6dab1a865d8bfab4ce9a49209756907788e585bb96d`|
|原v10入口，仅定位不重跑|`/public/home/yueweiting/大论文/ModernTCN-U+V/moderntcn_train_v10_entry.py`|已存在；文件本身|`657e13b58adce3ac245fd7df2b407ded304ef907555b83354af6c2be743a4c03`|
|输入scope配置|`/public/home/yueweiting/大论文/ModernTCN-U+V/configs/moderntcn_uv_fold6_input_ablation_v10.json`|已存在；文件本身|`f06f75f4e9ab1dc4a1ef36d12c76c9f2b7626cb3830ba1df692839d6c62cf984`|
|输入scope入口，仅定位|`/public/home/yueweiting/大论文/ModernTCN-U+V/moderntcn_input_ablation_v10_entry.py`|已存在；文件本身|`55146a5dbad9c158e6dea970310b41c520ea57265fb67507915880223b6f3791`|
|F1—F4有序schema|`/public/home/yueweiting/大论文/ModernTCN-U+V/utils/feature_schema.py`|已存在；文件本身|`1db7c6bd5d9d47fdeb8ff9df74812cd665b78733704d31abb5de94efcced64ae`|
|最新统一第二轮主入口|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-formal-launch-dhozikhu/baseline-unified-v3-ms-seal-m128-recovery1-patchtst-depth-urban6-r6-no-telemetry/round2-amendment/queue/round2-main-results.patchtst-enc2-v1.json`|已存在；文件本身|`5ecec35a4e719104e2f11a79484660707ad8e993c4b0c94bdc06ee82bca36792`|
|90条来源到原run映射|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/source/source-ledger.final.json`|已存在；文件本身|`c6246b10403ddfb869a915101858f195fdba3aedfb95d24e4949b77d22b0be2d`|
|历史producer与当前桥接差异|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/source/current-execution-compatibility.json`|已存在；文件本身|`6036e06fdde31f277ddfea1a980ef78fe6f8f914639fe24edb427c09eeb78d3f`|
|输入保存审计|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/audit/result.json`|已存在；文件本身|`8a483c61ba9f65bc644a7c1e2ffed9b926d944fa1fd5b58dba43ba509e80d978`|
|输入Goal3计算|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/audit/goal3.json`|已存在；文件本身|`74464776e2936d3c6831223a70e41c79deadccc56445feae5bd00b96cfc86ad1`|
|输入逐run结果与原路径|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/audit/saved-results.json`|已存在；文件本身|`ccc244a5be13747f57edf8c3bb3440d37329e9880f5dba5c85fa33c7c93e5196`|
|F4等价来源链|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/audit/F4-reuse.json`|已存在；文件本身|`c782641ba4a9b634e58fdbaa2eb6dcdbf58c68f627c2d3004d0615bd30802d6c`|
|已审v10模块结果包|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-postrun-review-r1/REPORT.md`|已存在；文件本身|`16124656bdb2d979d56c656bee3a1bff5cc2895147ec9025af44097f9f1eeab0`|
|补充包送审历史报告|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/REPORT.md`|已存在；文件本身|`1332dc9e6ab872b8eeb1844cfea671e60481fd8aaac828b6166d284b03273320`|
|该送审包逐文件checksum索引|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/manifest.json`|已存在；文件本身|`0647273fb6114d075b49af419226d17fb4803ff9cf48fc67e2533e2f0a712dfc`|

最新入口包含M历史范围，当前只消费ledger选定的63个MS baseline来源；M的PatchTST enc2决定不能移植到MS。旧`ch3_baseline_ms_u96_oc01_v3.json`仅作为每条producer绑定protocol，不绕过最新主入口。source-ledger的90条是63baseline＋27增强臂来源，不是90次本批训练。

从汇总到单run的操作：main-results/module-ablation按`model,dataset,H,fold`选择行，读取`metrics_path,metrics_sha,producer_commit`；在ledger.entries按相同metrics_ref.path匹配。baseline consumer为如`MS|AMD|UrbanEV|6|3|2024`的字符串，增强臂consumer为dataset/H/fold/arm/candidate对象。继续读取该entry的metrics_ref、manifest_ref、spec_ref（可能是manifest内JSON pointer）、data_ref（含scaler/prefix）、config_ref/protocol_ref、history_ref、weights_ref/checksum_ref；seal_ref存在时照原条目使用。原task/identity/producer字段保留真实身份，不用逻辑consumer覆盖。

输入表按`variant,H`取`source`目录，fresh字段区分12fresh与4引用；在该目录读取实际metrics/manifest/spec/data-metadata/history及checksum/seal，F4链另由audit/F4-reuse.json与原P1 proof绑定。checkpoint仅按封存checksum与stat定位，不迁移、复制或反序列化。

下面给出实际ledger样例，其他run照同字段查询，避免手抄几十个权重hash：

|用途|精确路径|存在状态/逻辑键|SHA或定位|
|---|---|---|---|
|metrics_ref|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-formal-launch-dhozikhu/baseline-unified96-onecycle001-v3/MS/formal-AMD/AMD-UrbanEV-F4-u96-oc01-v3-f6-h3-s2024/result.json`|已存在；"MS|AMD|UrbanEV|6|3|2024"|`8b62816c99e5571ecf3a239e7bc4c489557b9911f43a361544272e0912008041`|
|manifest_ref|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-formal-launch-dhozikhu/baseline-unified96-onecycle001-v3/MS/formal-AMD/AMD-UrbanEV-F4-u96-oc01-v3-f6-h3-s2024/manifest.json`|已存在；"MS|AMD|UrbanEV|6|3|2024"|`1ec7662ec1d22bb87ed3e11f95c4247993c2618371b55a089fa4ef4d21224a2b`|
|spec_ref|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-formal-launch-dhozikhu/baseline-unified96-onecycle001-v3/MS/formal-AMD/AMD-UrbanEV-F4-u96-oc01-v3-f6-h3-s2024/manifest.json`|已存在；"MS|AMD|UrbanEV|6|3|2024"|`1ec7662ec1d22bb87ed3e11f95c4247993c2618371b55a089fa4ef4d21224a2b`|
|data_ref|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-formal-launch-dhozikhu/baseline-unified96-onecycle001-v3/MS/formal-AMD/AMD-UrbanEV-F4-u96-oc01-v3-f6-h3-s2024/manifest.json`|已存在；"MS|AMD|UrbanEV|6|3|2024"|`1ec7662ec1d22bb87ed3e11f95c4247993c2618371b55a089fa4ef4d21224a2b`|
|history_ref|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-formal-launch-dhozikhu/baseline-unified96-onecycle001-v3/MS/formal-AMD/AMD-UrbanEV-F4-u96-oc01-v3-f6-h3-s2024/history.jsonl`|已存在；"MS|AMD|UrbanEV|6|3|2024"|`1f5f36cfe80037ece81de3b74358448deee5104a4c37199ce35f0bfb35575526`|
|weights_ref|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-formal-launch-dhozikhu/baseline-unified96-onecycle001-v3/MS/formal-AMD/AMD-UrbanEV-F4-u96-oc01-v3-f6-h3-s2024/best.pt`|已存在；"MS|AMD|UrbanEV|6|3|2024"|见封存ledger/manifest；目录不计算SHA|
|checksum_ref|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-formal-launch-dhozikhu/baseline-unified96-onecycle001-v3/MS/formal-AMD/AMD-UrbanEV-F4-u96-oc01-v3-f6-h3-s2024/result.json`|已存在；"MS|AMD|UrbanEV|6|3|2024"|`8b62816c99e5571ecf3a239e7bc4c489557b9911f43a361544272e0912008041`|
|metrics_ref|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/moderntcn-uv-v9-development-runs/urbanev-f6-epf-u-uv-v9-first-batch-v1/urbanev-f6-epf-moderntcn-uv-v9-first-r1/runs/UrbanEV-F4-f6-h9-UV-c5eff016173b/metrics.json`|已存在；{"H": 9, "arm": "UV", "candidate": "moderntcn-uv-dev-v10", "dataset": "UrbanEV", "fold": 6}|`5a987f20524ec60571001439f94b4dc16a50aa17ddb5d9977cf63fdb2ca0f4fb`|
|manifest_ref|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/moderntcn-uv-v9-development-runs/urbanev-f6-epf-u-uv-v9-first-batch-v1/urbanev-f6-epf-moderntcn-uv-v9-first-r1/runs/UrbanEV-F4-f6-h9-UV-c5eff016173b/manifest.json`|已存在；{"H": 9, "arm": "UV", "candidate": "moderntcn-uv-dev-v10", "dataset": "UrbanEV", "fold": 6}|`442aa8d9af09cb69b245ee023b6377a35140d6e0b979170190fcc1efba9604a1`|
|spec_ref|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/moderntcn-uv-v9-development-runs/urbanev-f6-epf-u-uv-v9-first-batch-v1/urbanev-f6-epf-moderntcn-uv-v9-first-r1/runs/UrbanEV-F4-f6-h9-UV-c5eff016173b/spec.json`|已存在；{"H": 9, "arm": "UV", "candidate": "moderntcn-uv-dev-v10", "dataset": "UrbanEV", "fold": 6}|`e095a225dd30b510714a2e26079fd69b9861dcf5aedc3afcec87610a7401d4eb`|
|data_ref|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/moderntcn-uv-v9-development-runs/urbanev-f6-epf-u-uv-v9-first-batch-v1/urbanev-f6-epf-moderntcn-uv-v9-first-r1/runs/UrbanEV-F4-f6-h9-UV-c5eff016173b/data-metadata.json`|已存在；{"H": 9, "arm": "UV", "candidate": "moderntcn-uv-dev-v10", "dataset": "UrbanEV", "fold": 6}|`14dc04f5cf9b7156a98563c8e820721fb0c9b1bb6fac2f37ecd8380601d9dc78`|
|history_ref|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/moderntcn-uv-v9-development-runs/urbanev-f6-epf-u-uv-v9-first-batch-v1/urbanev-f6-epf-moderntcn-uv-v9-first-r1/runs/UrbanEV-F4-f6-h9-UV-c5eff016173b/history.jsonl`|已存在；{"H": 9, "arm": "UV", "candidate": "moderntcn-uv-dev-v10", "dataset": "UrbanEV", "fold": 6}|`15572cf8bc18416310ddf746bb81514ee80712299299f14e9344a55a26067d0e`|
|weights_ref|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/moderntcn-uv-v9-development-runs/urbanev-f6-epf-u-uv-v9-first-batch-v1/urbanev-f6-epf-moderntcn-uv-v9-first-r1/runs/UrbanEV-F4-f6-h9-UV-c5eff016173b/best.pt`|已存在；{"H": 9, "arm": "UV", "candidate": "moderntcn-uv-dev-v10", "dataset": "UrbanEV", "fold": 6}|见封存ledger/manifest；目录不计算SHA|
|checksum_ref|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/moderntcn-uv-v9-development-runs/urbanev-f6-epf-u-uv-v9-first-batch-v1/urbanev-f6-epf-moderntcn-uv-v9-first-r1/runs/UrbanEV-F4-f6-h9-UV-c5eff016173b/best.pt.sha256`|已存在；{"H": 9, "arm": "UV", "candidate": "moderntcn-uv-dev-v10", "dataset": "UrbanEV", "fold": 6}|`b59488a48c9275b892592d5e981e566c75d113b6465f81688feac23f74adfa40`|
|原更新日志|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/moderntcn-uv-v9-development-runs/urbanev-f6-epf-u-uv-v9-first-batch-v1/urbanev-f6-epf-moderntcn-uv-v9-first-r1/runs/UrbanEV-F4-f6-h9-UV-c5eff016173b/updates.jsonl`|已存在；{"H": 9, "arm": "UV", "candidate": "moderntcn-uv-dev-v10", "dataset": "UrbanEV", "fold": 6}|`903e6a568da9526e801c3d0a4006fb7bb3370ed0b0df4e215554be919cbc87ea`|

本指南不保证所有旧baseline都保存训练loss、逐步journal或逐点test预测。ledger没有预测缓存入口时，不得把diagnostic validation缓存当正式test预测。只使用实际存在的history字段；缺失项不补造，不为绘图重新test。weights_ref定位strict-best，不按图像效果换checkpoint。

## 已有图表、数据和复现方式

|用途|精确路径|存在状态/逻辑键|SHA或定位|
|---|---|---|---|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/EPF-UV-relative-mae_percent.png`|已存在；EPF-UV-relative-mae_percent|`be163e41ef981c8346e0e0407dc94500e73fa7884c5f583b47d3dbb5f2e3eb85`|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/EPF-UV-relative-mae_percent.svg`|已存在；EPF-UV-relative-mae_percent|`aacac1f45bd50811c0bed1a58ba9f8a8d8d683cc1c11c915b545a2a4603c5bea`|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/EPF-UV-relative-mse_percent.png`|已存在；EPF-UV-relative-mse_percent|`66878f9262a18d453a9361c021cebdd2c7539f0a0032d9e02d3ce60a4a2b61c1`|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/EPF-UV-relative-mse_percent.svg`|已存在；EPF-UV-relative-mse_percent|`a2b6b50f0aceb09b04224327e358f1636ad2064222d9308bd9a6abdf6de011c2`|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/PJM-convergence.csv`|已存在；PJM-convergence|`d49eb31c82451ac0aebdebd88798c00cf8630891648bbb9f344ecb5b4ca5e7d8`|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/PJM-convergence.png`|已存在；PJM-convergence|`c87c37a7938dd59963a8f68edfe95f0b975ef8be5810d144d321713b73d6b56f`|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/PJM-convergence.svg`|已存在；PJM-convergence|`778ec7600f2ffb42d4790804517d7e0cfea549da8021ba1c999a896c703f5dbc`|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/UrbanEV-UV-relative-mae_percent.png`|已存在；UrbanEV-UV-relative-mae_percent|`ade2229a19178117b041173b46dbf26f0b515b5c4b1cff6f3662176e08f51fef`|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/UrbanEV-UV-relative-mae_percent.svg`|已存在；UrbanEV-UV-relative-mae_percent|`4dea0bdbb41101f8016daa077ca5ccc0e0aa7e095205ddb06d2ba3aba1237aea`|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/UrbanEV-UV-relative-mse_percent.png`|已存在；UrbanEV-UV-relative-mse_percent|`fab2a08151de2557792dba3f1fc3d2c420d1de92c948856832448c64b5261ef1`|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/UrbanEV-UV-relative-mse_percent.svg`|已存在；UrbanEV-UV-relative-mse_percent|`6106c7068d7e0c3d756f0ecba36c3c085e6516259d55beb6c12787d1b9043889`|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/UrbanEV-convergence.csv`|已存在；UrbanEV-convergence|`1d53a5c45ef17f04a802ba25782fa56cf8fcf8b97b7e675f0bad38f1ef399313`|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/UrbanEV-convergence.png`|已存在；UrbanEV-convergence|`b81f320b9e1431a026a027bdac871f432f3ab39b8329c3cf8bf95b7b878f17d8`|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/UrbanEV-convergence.svg`|已存在；UrbanEV-convergence|`7bc8f212d624e0fbce9e91c4290e540916962c2ac8383b1dae8a941c8ccc2e13`|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/UrbanEV-input-mae.png`|已存在；UrbanEV-input-mae|`8bdee0901887d561f0c3f93bddd6cf2668b6f639fbd1a770117345af086711ca`|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/UrbanEV-input-mae.svg`|已存在；UrbanEV-input-mae|`8d5124fe4c60d9656d1e034f2e2d268bfee9afe17a29c2fbe8c422f916871921`|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/UrbanEV-input-mse.png`|已存在；UrbanEV-input-mse|`470ec5fb4e72877376b45f9a1aa9d8880309cc07a0b84f0540bdc8dc3e2317d3`|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/UrbanEV-input-mse.svg`|已存在；UrbanEV-input-mse|`39660b023fbf11a64a513ae90c2f78e56e69501d11b4303867bab614ac71954e`|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/UrbanEV-main-mae.png`|已存在；UrbanEV-main-mae|`fb96c5da963e698b80affe8f2243879c49c73834e49a849aad61ddd7d7c4fa4d`|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/UrbanEV-main-mae.svg`|已存在；UrbanEV-main-mae|`cc2dcef12236aee2feeb9c6da5335cac8b83d580d8314bfa27162631a93523a4`|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/UrbanEV-main-means.csv`|已存在；UrbanEV-main-means|`58bd6b4ad9efa0f792b7cc1ab2b8f1ceba116f331691a940e9f0b23ad4ac671d`|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/UrbanEV-main-mse.png`|已存在；UrbanEV-main-mse|`84e05527407c5c8f2e766a6f05343e0ca7cb90292a57a3426f052a0321400a1a`|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/UrbanEV-main-mse.svg`|已存在；UrbanEV-main-mse|`d77b981e097e97dba3d02aea1d35063bf8dbe5195ca0156effa739d6a9c21166`|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/UrbanEV-module-mae.png`|已存在；UrbanEV-module-mae|`7514d921d6ab28ead68fd34c1a1c223a2d387e44d26f8466ba833b6a47ab1e9b`|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/UrbanEV-module-mae.svg`|已存在；UrbanEV-module-mae|`f038cea84544e25d3b1b2c48183e8e5298602050fa9b755255f6ff90c30083de`|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/UrbanEV-module-means.csv`|已存在；UrbanEV-module-means|`c3adbc545bb36436d4dc79566a02e22e5c83a46fd0ecf567c3b51d4a184d162e`|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/UrbanEV-module-mse.png`|已存在；UrbanEV-module-mse|`242fba9ccc4a7d209e52531eaa05400004b7418e3d46bad03fb2a0246bff3660`|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/UrbanEV-module-mse.svg`|已存在；UrbanEV-module-mse|`c5c8ffa33f8015a982e2c4dfea623c5b2c188147c0a1cedb547096953d6e788c`|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/input-ablation.csv`|已存在；input-ablation|`db1421111249701fed5573ea85295b4cf4125d1bf1d64684bb8d937d560fb5ce`|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/input-means.csv`|已存在；input-means|`f557285f9f8800a21695b9239bd47f469cc00d656c1040766989781eb42c39b4`|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/main-results.csv`|已存在；main-results|`6d89d12df5dd410d498f14ac2f7755c53273f579ae57bf688bd11f2abd277867`|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/main-results.md`|已存在；main-results|`f00584daf23a08a465dfc1489dea3f862c9970a09ef9111af11254643cb3f49e`|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/module-ablation.csv`|已存在；module-ablation|`8d20756afff40518375acb7cee647324904ce3a34a3e1277b1b5868306422822`|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/module-ablation.md`|已存在；module-ablation|`00168424f8c024da131c2d78b0573c42762cc81c555c4add8d8911a009b95eff`|
|已有保存成绩派生图表/数据|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/plots/completed-r2/module-relative.csv`|已存在；module-relative|`9b426fc67d2b1dae1856a2b620e26e6948fa98d3449d6e552d39aa587a70497a`|
|既有数据/图稿生成脚本|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/tools/prepare_tables.py`|已存在；原送审生成步骤|`5c2f9d15c442d67efc4421119af2acad7063c8b57d8b8a64b58bcdb4ac1d23ef`|
|既有数据/图稿生成脚本|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/tools/prepare_tables_r2.py`|已存在；原送审生成步骤|`8eb2aedee493be4a5701fe7c009c62fe2bb85695a8e53e04c52df13f7b307201`|
|补充阶段离线绘图实现|`/public/home/yueweiting/大论文/ModernTCN-U+V/m6_ch3_supplement_v10/plots.py`|已存在；O phase|`15d792ebe0f4424082ed63ddefcfdbb9f6342f872efe80438c0e81ed537a9be7`|

上述CSV/JSON和SVG/PNG同源，图稿存在不等于科学效果通过。main-results为72行；module-ablation36行；input-ablation16行，input-means为4个等权均值。UrbanEV-main曲线对应四个独立H；PJM未来逐位置图对应同一P24内部k=1…24。convergence只消费原history/updates，训练batch平均loss与epoch末validation不是同一固定模型上的train-eval差。

复现时先核验封存manifest与来源键，再按已保存脚本读取原文本表。脚本中的既有输出路径属于封存包，不直接重跑覆盖；后续O只写新attempt内产物。此指南不授权重新模型推理来补同一张图。正式论文图号未冻结。

## 已实现但未执行的补充实验

统一状态：**工程已审，正式负载未启动**。工程review限冻结合同、实现、来源桥接、限定CPU验收和控制逻辑；旧baseline真实权重恢复、正式GPU执行、性能数字与敏感性效果尚未实测，不能称Passed。

|阶段|问题与对象|固定范围/预算|未来产物|
|---|---|---|---|
|S|完整UV对G/R共享能力的敏感性，不替换主模型|UrbanEV F4 H6、PJM各G3/R4、G7/R4、G5/R2、G5/R6，8fresh，两波q4；中心G5/R4各1引用；最多160轮617120更新；validation-only/test0|各job history/best/last/validation-result及全部参数点图|
|E|精度与真实推理成本|UV＋七baseline，两任务batch1/16共32条件，q1；50预热＋5×100计时，共17600调用149600呈现；backward/Adam0|每组时间/吞吐/allocated/reserved、参数及误差—延迟图|
|C|两个固定UrbanEV区域案例|H6、B_cfg/TimeMixer/UV；训练统计选2节点，共同validation168小时目标时间，1008呈现|选择证据、原单位预测/真值、两张曲线及局部指标|
|P|PJM内部24位置误差|B_cfg/TimeMixer/UV各一次完整validation，strict-best不变|逐位置SSE/SAE/count、预测缓存、MSE/MAE曲线|
|O|同源离线汇总|不新增模型计算；E→C/P→S Urban→S PJM→O，共47job/41执行wave|CSV/JSON/SVG/PNG和覆盖汇总|

R6而非R8保留Urban Kv7行正交语义；G5/R4默认非零等价支持中心引用。8项新训练不增加seed、不搜K，不以结果好坏取消/追加；全批新test0。q4按既有证据与激活/反向/Adam/workspace差异接受，本轮未重新实测；E/C/P q1是串行公平测量，不是训练降并发。

规划result_root：`/public/home/yueweiting/大论文/amd-execution-evidence/m6/moderntcn-uv-v10-ch3-supplement-runs`；规划attempt：`/public/home/yueweiting/大论文/amd-execution-evidence/m6/moderntcn-uv-v10-ch3-supplement-runs/ch3-supplement-v10-first-r1`。**规划；当前尚不存在**，不得当作已完成产物。实际job路径由原queue写入该attempt的runtime/receipts/logs及各job目录，不能提前猜runtime UUID。

|用途|精确路径|存在状态/逻辑键|SHA或定位|
|---|---|---|---|
|补充scope固定配置|`/public/home/yueweiting/大论文/ModernTCN-U+V/configs/m6_ch3_supplement_v10.json`|已存在；文件本身|`db49c99e1c30f97b0599cc3a8d64d9fba923b43d8bee847064af6353591e6ea0`|
|补充入口|`/public/home/yueweiting/大论文/ModernTCN-U+V/m6_ch3_supplement_v10/entry.py`|已存在；文件本身|`b91c3d56e7fc3d050a2ce9e29cff4cc3a2be518474ea4cfbeb599f58931f7f92`|
|job与产物路径/阶段队列|`/public/home/yueweiting/大论文/ModernTCN-U+V/m6_ch3_supplement_v10/queue.py`|已存在；文件本身|`dd146f3fa1e4867922bcc0e65f59b2f72f4a313aa889a4a4ba2c85d225136a1a`|
|两参数总启动脚本|`/public/home/yueweiting/大论文/ModernTCN-U+V/scripts/ch3/start_m6_ch3_supplement_v10.sh`|已存在；文件本身|`5befd35e9aeda50d9fa53cfa856497de640e2cd2344ed1fff80968db3695fb10`|
|不可执行历史模板|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/non-execution-preparation/launch-binding.template.json`|已存在；文件本身|`697b9488656480ffeed029028e37d08886a19289bc9ce363c8328ad99b54ddf5`|
|历史操作草案|`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/non-execution-preparation/OPERATIONS.md`|已存在；文件本身|`f3325b443c70dd70e348c92fbd309184c6486be1b6f8b62a00f53309b0d8eee1`|

以后正式执行完成并审计后更新本指南，不另建平行索引。运行中不修改Git绑定工作区；结果先写外置attempt，完成后随实质收口更新。正式AUTH只能在精确closure和实际远端核验成功后另行绑定，模板不是许可。初始Git联网依赖不自动代表后台需要联网。

## 历史未采用实验导航

历史/未纳入当前采用主表：旧AMD/J与相关命名、旧TimeMixer候选、ModernTCN旧候选、旧fold1/4、C.1及M历史。仅从[唯一M6](milestones/M6_formal_experiments.md)原有标题/时间记录及其原证据链接追溯；可按这些原名称查找，不重新命名第一/二/三轮入口，不展开重审或混入当前数据。统一round2入口的M记录保留原名但不被本MS ledger消费。v9成功候选及所有旧负向结果保持；v10采用引用不改变其生产者。

## 审核状态说明

ChatGPT已服务器直读审核输入消融保存结果和补充包工程，分别review Passed；Goal3仍Not met。12独立用例最终通过、19执行17成功2失败，原失败/修复/消费均封存，不写成19次全通过。本指南仅整理已审事实与用户冻结合同，其新字节尚未被宣称由ChatGPT读取。它不授予真实训练或替代正式AUTH，不开始M7。
