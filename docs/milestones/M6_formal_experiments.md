# M6：第三章正式实验与定稿

**In Progress — 正式协议/结构已冻结，启动准备就绪后由用户逐模型启动；尚未运行任何正式训练或正式test评价。**

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

用户每次只启动一个模型；该模型内部按dataset→input_variant→固定H/fold波次自动完成，不跨组补位，不自动进入下一模型。已验证并发无需再次申请；硬件/shape/batch/线程/workers或资源条件改变时不外推。长训练异常保留全部文件，先只读审计后才能给合法resume，不自动fresh重跑。正式test只在每run训练完成后对validation-selected best执行一次；不得逐epoch看test或用test选模型。

## 4. 版本、许可与启动材料

证据根：`/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-formal-launch-dhozikhu`。

`formal-approval.json`由实际closure成功后生成，绑定实际commit、code/config/环境/硬件、冻结ID、全部模型、495个data metadata摘要和source stat；`runtime-probe-admission.json`明确继承原54组报告及独立不变性证据，不覆盖原报告。无负载preflight逐十模型必须通过后才给用户启动命令。本轮不代启任何正式模型。

启动方式沿`bash scripts/ch3/start_model.sh --model MODEL start --approval "$E/formal-approval.json" --probe-report "$E/runtime-probe-admission.json"`。`status/logs/complete/safe-stop`使用同一模型入口；日志在每run目录的worker.log、history.jsonl和每波memory.jsonl，模型controller日志位于本证据根。全部具体命令保存在本证据根`launch-commands.md`。

在正式任务运行期间不修改仓库/已锁定配置，不升级依赖或作者仓库；结果保存在外置run目录。组间先做只读完整性审计，不为状态文字递归提交导致旧许可失效。`complete=true`仍需核对run数、best/last/history/metrics及预算/finite，不能仅凭tmux消失判成功。
