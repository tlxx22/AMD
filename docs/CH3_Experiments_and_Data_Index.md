# 第三章实验说明与数据索引

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
