# M6：第三章正式实验与定稿

**当前ModernTCN v4工程审核与启动交接（本轮新增裁决）**：按用户转达的ChatGPT通过Desktop Commander实际直读审核，v4固定实现、版本隔离及限定CPU验收工程review Passed。已审REPORT SHA-256=`f22c3f53eec66e74139de0c8038b777794026c425069330e9e8b55b9deab18e8`，file-manifest SHA-256=`08ce8d3ace77118e4c6855c7747e92c81cc94d0bbd925481f7d893f92e766e35`。九格v3 V-only等价单臂引用批准，保留原task/producer/config/source/identity及封存SHA，不计v4 fresh或新增test；旧U不能冒充新U，新U/UV十八项必须fresh，禁止旧权重warm-start。

本批Urban/EPF q4基于既有证据复用与差异评估接受；本轮未重新实测GPU峰值、CUDA恢复或训练数值轨迹，不补探针。v4仍为待验证科学方案，三个效果目标全部Not evaluated，不继承v3的8/10。原计数漏拦截、错误日志、按完成控制流补账及失败成本保留。B_ref=B_cfg=cfg_adapted_v1，C.1历史不采用；单fold6及五EPF、原训练/评价合同均不改。

本轮仅两文档增量，23个已审新增文件保持原字节，配置execution_authorized=false及preparation_head不改。25路径精确closure的实际commit和文档SHA由外置`M6-ModernTCN-UV-v4-review-closure-r1`记录；随后`M6-ModernTCN-UV-v4-launch-authorization-r1`绑定正式AUTH及轻量元数据preflight。十八项fresh U/UV按六波4/4/1/4/4/1、q_max=4，最多360 run-epochs、1,245,080 updates，另九格V显式引用；不执行额外任务或按效果取消后续波次。用户最后启动真实长训练，Codex本轮不执行start/supervise/worker、不创建真实runtime/attempt。前台preflight和监督器初始检查仍需Git网络；用户保持SSH/17890，确认匹配AUTH/实际commit的runtime、首波worker身份与正常日志后再断开，后续波次沿既有本地核验。以下待审文字保留原历史时点。

**当前ModernTCN v4辅助内容实现（本轮新字节统一待ChatGPT审核）**：用户转达的服务器直读裁决确认v3保存产物/结果及限定机制观察review Passed；范围不含CUDA恢复、训练重演或唯一根因证明。v3目标一覆盖完成但双删除对照平均正向贡献未支持，目标二对B_ref=B_cfg=cfg_adapted_v1为8/10数值条件满足，目标三未评价；原结果、失败成本和历史待审时点保留。

本轮只实施ChatGPT已选定的TFT历史变量选择核心改造：原FFN1一次产生h，先h_v=h+V3(h,Z,X)，再以h_v目标latent为context、pre-RevIN有序aux原生patch为Value输入新U，target-only写入后继续原FFN2/head/denorm。旧U不注册，V3公式/类及训练协议不变；这不是v3技术故障修复，也未证明新方案有效。新候选moderntcn-uv-dev-v4三个目标全部Not evaluated。

11项定向CPU用例通过。实测U参数Urban F1/F2/F3/F4为3557/4471/4928/5842、EPF为2298；V仍572/2076，F4/EPF联合增量6414/4374。V-only与v3非零路径train/eval、状态/BN/梯度、两步Adam/OneCycle及RNG exact；准备九格原v3 V显式等价引用＋十八项fresh U/UV，复用方向不再是旧U。六波4/4/1/4/4/1，360 run-epochs、1,245,080 updates仅为非执行上限，未生成真实AUTH/runtime/attempt。

实际合成消费24次完整模型构造、55次完整forward/110次样本呈现、42次完整backward、14次Adam（6次optimizer构造），另39次局部模块调用、4次局部autograd及1/1次合成state保存/读取。记录器漏拦截forward/backward的工具缺陷及原输出保留，依完成用例的固定控制流补账，无为计数重跑；11用例无数值失败/skip/容差放宽。真实checkpoint、真实数据模型前向、新test、真实训练、GPU查询/负载均0。证据在`M6-ModernTCN-UV-v4-auxiliary-content-implementation-r1/`；新代码与两文档增量全部unstaged、index空，不commit/push，等待新字节实质审核。

**当前ModernTCN v3完成后审计与增量机制诊断（2026-10-25，统一待ChatGPT审核）**：实际训练commit为`683d6725ee710c4d894ee0124c34cc0d434402ca`；18fresh、18回执、六波技术完成，228 run-epochs、700626 updates、18次原test，另9格U为获准v1等价引用。本轮493个attempt文件、414个封存payload及36份CPU best/last保存状态自检通过；不等于CUDA恢复或训练重演。工程review沿用Passed，新结果/机制review待ChatGPT审核。

目标一单fold6三臂覆盖完成，但v3 UV对U四H等权test MSE/MAE为+2.6599%/+1.2238%，对v3 V为+2.0247%/+0.6697%；完整validation均值也弱于两单臂，两删除对照均未支持完整模型的平均正向贡献。V对v2 V改善11.0619%/2.9763%，独立U/V正向证据保留。目标二对B_ref=B_cfg=cfg_adapted_v1为8/10（MSE4/5、MAE4/5）数值条件满足，NP/BE/FR/DE获益、PJM未胜；目标三Not evaluated。C.1仅保留已执行未采用历史。

固定train/validation诊断显示context作用分化：UV关闭context后抽样validation MSE在H6下降3.31%，H12上升18.96%、BE上升2.52%，PJM仅上升0.18%；不能认定目标状态普遍有害。H12同权重关U或V均变差，却仍存在独立UV弱于单臂的结果；没有证明历史梯度冲突或唯一训练根因。建议保留U/V独立有效能力，联合功能分工交ChatGPT重新评估；当前证据不足以冻结v4小修公式，未实施下一模型。实际新增36次CPU加载、10构造、276完整forward/15840次呈现，另96次缓存滤波重建与4次局部context校准；新test/backward/optimizer/训练/GPU均0，无模型/状态审计失败或复验重读。外置证据为`M6-ModernTCN-UV-v3-postrun-review-diagnosis-r1/`；仅本两文档增量、unstaged/index空，不commit/push。以下“未启动/待审”等保留旧时点。

**当前ModernTCN v3工程审核与启动交接（2026-10-24，本轮新增裁决）**：按用户转达的ChatGPT通过Desktop Commander实际直读审核，v3实现、版本隔离及定向CPU工程review Passed。审核包为`M6-ModernTCN-UV-v3-target-state-routing-implementation-r1`，REPORT SHA-256=`fa6f0b5d71efa1677cc38ac6d6b7655dd169300af9faf1497f285509b3ccbc63`，file-manifest SHA-256=`6c87611710f28189bc603fada9d8bb7e10f46707b0acd8bbb36e424cbbb51040`。九格v1 U-only显式等价单臂复用获准，保留原task/producer/config/source及封存结果SHA，不计为v3 fresh或新增test，不冒称CUDA轨迹重演。十八项V/UV必须fresh，六波4/4/1/4/4/1，q_max=4，最多360 run-epochs、1,245,080 updates；不重训或重新test B/U。

本批Urban/EPF q4基于既有证据复用与差异评估接受；本轮未重新实测GPU峰值、CUDA恢复或数值重演，不补探针。模型、训练和评价合同不改，B_ref=B_cfg=cfg_adapted_v1，C.1已执行未采用、历史保留。v3仍是增加V目标条件的待验证结构假设，不是已查明根因的技术修复；三个效果目标仍Not evaluated，不能继承v2结果。原历史、v2审计增量及失败成本保留；以下待审文字保留当时时点。

本轮只对两份权威文档增加裁决，22个新增文件保持审核字节；配置`execution_authorized=false`及`preparation_head`不变。24路径精确closure的实际commit和文档SHA由外置`M6-ModernTCN-UV-v3-review-closure-r1`记录；随后`M6-ModernTCN-UV-v3-launch-authorization-r1`绑定正式AUTH（review_approved=true、恢复关闭）与轻量元数据preflight，不递归回填本文自身commit。最后真实长训练由用户启动，Codex不执行start/supervise/worker、不创建真实attempt。本批为立即启动六波：前台preflight和监督器初始授权检查仍需Git网络，用户保持SSH/17890代理，确认匹配AUTH/commit的runtime、首波worker身份和日志正常后再断开；tmux返回或空日志不足以判启动成功。后续波次沿既有本地核验，不新增延迟等待链或离线准入平台。

**当前ModernTCN v3目标状态路由实现（2026-10-24，新字节统一待ChatGPT审核）**：用户确认唯一方案，仅在v2 V路由中加入本次pre-RevIN目标历史的`[mean,log1p(population variance)]`经`P_s[16,2]`投影，所有变量共享；`P_s=0`且不消耗初始化RNG。U、同层post-FFN1并联、原生主干和科学协议保持。13项定向CPU用例通过；29次完整模型构造、68次完整合成forward/135次呈现、46次backward、18次Adam，另11次局部模块forward、8次局部autograd、13次局部conv及1/1次合成checkpoint保存/读取。真实checkpoint、真实数据模型前向、新test、真实训练和GPU均0。非零条件响应与零P_s退化exact均验证；U-only与v1/v2非零输出/状态/梯度/两步Adam/RNG exact，准备九格原U显式等价引用＋十八格V/UV fresh，未生成正式AUTH或attempt。v3是待验证结构假设，不是已确认根因的技术修复；新效果全部Not evaluated。

本次据用户转达的服务器直读裁决，v2的18fresh/6波、9U引用、保存产物与结果review Passed；有限train/validation机制报告仅在声明范围Passed，不含CUDA恢复、训练重演或唯一根因证明。原244轮/812310更新/18test、诊断51读取/15构造/1121前向/28359呈现及全部失败成本保留。v2目标一覆盖完成但V平均正向贡献未支持；目标二对B_cfg为7/10（MSE3、MAE4）；目标三未评价。B_ref=B_cfg=cfg_adapted_v1，C.1已执行未采用。详细v3证据位于`M6-ModernTCN-UV-v3-target-state-routing-implementation-r1/`；仅新版本文件和本两文档增量，unstaged/index空，不commit/push。下列待审、未启动等为旧历史时点。

**当前ModernTCN v2完成后审计与机制诊断（2026-10-24，统一待ChatGPT审核）**：实际训练/closure commit为`170c27e685686ed1932849090b65c9c79de36e8c`；18项fresh V/UV、六波已技术完成，244 run-epochs、812310 updates、18次原test，另9格U为获准v1等价引用。493个attempt文件/414封存payload及36份CPU best/last保存状态审计自检通过；不等于CUDA恢复或训练重演。目标一单fold6消融覆盖完成，但v2 UV对U四H等权test MSE/MAE为+1.1447%/+0.8905%，V的平均正向贡献未获支持。目标二对B_ref=B_cfg=cfg_adapted_v1为7/10（MSE3/5、MAE4/5），数值条件满足、新结果review待审；目标三未评价。C.1已执行未采用，历史保留。

同权重观察确认两支路共享h、局部并联和q/直接分支不变；后续FFN2/head仍有任务相关交互，不能称完全解耦。H12/NP固定train均值route在所测validation上更好，BE动态route更好；不以动态性、秩或gate单独判有效。建议保留U，不再仅靠换位/加rank/调温度修补V；一条目标状态可见的路由重构假设仅交ChatGPT裁决，未实现/未冻结新候选。本轮实际51次CPU状态读取（36唯一＋工具失败后15必要重读）、15构造、1121前向/28359次呈现；5699条任务窗口、4498个不同物理历史区间；新test/backward/optimizer/训练/GPU均0。外置完整证据为`M6-ModernTCN-UV-v2-postrun-review-diagnosis-r1/`；仅两文档增量、unstaged/index空，不commit/push。以下未训练/待审等保留原历史时点。


**ModernTCN-U+V v2工程审核与启动交接（本轮新增裁决）**：按用户转达的ChatGPT服务器直读审核，同层并联实现、版本隔离和定向CPU验收工程review Passed；审核REPORT SHA-256为`0e9168352a02465f21dee55564e4ffef27726a8fdaea7d742fdc7123d79a5f15`，file-manifest为`854dabc9e12708495db443cd232edb6545367ac19fc5b170c9456db87df539a8`。九格v1 U-only的显式等价复用获准，保留原task/producer/config/source/结果SHA，不冒称CUDA轨迹重演、不记作v2 fresh完成。十八项V/UV必须fresh，六波4/4/1/4/4/1、q_max=4，最大360 run-epochs与1,245,080 updates；不重训或重新test U/B，不执行fold1/4、M、额外seed、搜索及F1/F2/F3输入消融。

本批Urban/EPF q4依据既有正式q4、工程验收与V反向依赖变化的差异评估接受，本轮不新增GPU/OOM/数值探针，不声称本版峰值或CUDA数值轨迹实测。B_ref=B_cfg=cfg_adapted_v1；C.1已执行但不采用、历史保留。v2效果与三个目标仍Not evaluated，不能继承v1效果。目标一继续单fold6独立模块消融，无U<B/V<B链；目标二十格至少七格严格胜配对B_cfg，缺项不缩分母。

本轮仅两文档追加裁决，其余22个新增文件保持审核字节；配置execution_authorized=false与preparation_head不改。按24路径普通closure后，由独立外置`M6-ModernTCN-UV-v2-review-closure-r1`记录实际commit/逐文件SHA，`M6-ModernTCN-UV-v2-launch-authorization-r1`绑定正式AUTH及元数据preflight；不递归回填本文自身commit。用户已授权review_approved=true，恢复字段关闭；实际commit从Git取得，Codex不执行start/supervise/真实worker、不创建真实attempt，最后长训练由用户启动。

这是用户立即启动六波的队列，不是延迟等待链。前台preflight与监督器初始授权检查仍需Git网络；用户应保持SSH/代理可用，确认匹配AUTH/commit的真实runtime及首波worker正常出现后再断开，不能只凭tmux命令返回判启动成功。六波启动后的worker、波次衔接及汇总沿本地身份保护；不新建联网凭证平台，不改源码绕过检查。下列历史待审/未授权文字保留其原时点。

**当前ModernTCN v2同层并联实现（2026-10-23，工程自检完成、新字节统一待审）**：用户批准唯一方案：原dw→block.norm→FFN1一次得到h，U(h,X)与V_filter(h,Z)读取同一未修改h，再相加进入原FFN2；原生RevIN/head及训练协议不变。13项定向CPU用例最终覆盖通过、14次执行，30次模型构造、90次完整合成forward/179次呈现、47次backward、27次Adam、6次局部autograd及6/6次合成checkpoint保存/读取；失败和计数补账保留。U-only非零路径输出/状态/梯度/Adam/RNG与v1 exact，准备显式引用九格原U；18项V/UV必须fresh，六波4/4/1/4/4/1，360轮/1245080更新上限，尚未执行。B_ref=B_cfg恢复cfg_adapted_v1，C.1为已执行未采用历史；单fold6合同与v1目标二7/10（MSE4、MAE3）沿用最新裁决。v2效果Not evaluated；无真实AUTH/attempt、无真实checkpoint/数据模型前向/test/训练/GPU。以下旧时点原样保留。

**当前ModernTCN v1单fold6＋EPF审计诊断（2026-10-23，统一待审）**：扩展27项/9波已完成，独立核账382 run-epochs、1,272,765更新、27次原test；736原文件/621封存payload及54份best/last CPU状态自检通过，不含CUDA恢复或训练重演。按用户最新决定，增强Urban模块/输入消融改为现有fold6单切分，旧fold1/4保留历史、旧未执行输入消融superseded。当前消融覆盖完成；UV相对U的四H test MSE/MAE为+1.1473%/+0.6637%，双指标额外收益未获支持。B_cfg保留适配原B；现场C.1已53/53封存，B_ref五EPF齐全，UV严格胜7/10格（MSE3/5、MAE4/5），新目标二数值满足、结果review仍待ChatGPT；目标三Not evaluated。只读有限CPU机制诊断支持保留U、重审V与U的直接接口，未实施下一结构。详情见同一ModernTCN专节；下列旧Pending/未完成/旧目标文字保留其当时时点。

**ModernTCN v1 首批完成审计与fold6＋EPF三臂扩展（2026-10-22，统一送审）**：原首批24项/6波技术完成，独立产物及CPU保存状态自检Passed（655文件、48份best/last只读反序列化）；核账370 run-epochs、1,038,487 updates、24次原正式test。目标一Not met，目标二/三Not evaluated；新结果与扩展实现仍待ChatGPT审核。用户批准新增批次27项：Urban F4 fold6四H×U/V/UV 12项＋EPF五市场三臂15项，9波q4，最多540 run-epochs/1,867,620 updates。22个新科学身份＋原5个EPF UV身份，原53项历史及24项输入消融保留，合并75个唯一身份；扩展批次完成0。模型/模块及原53项配置不改，仅新增精确scope与队列/汇总分派。17项CPU元数据/保存指标fixture通过，无模型前向、Adam或GPU。新字节统一待审，未commit、无可消费AUTH/新attempt，下一任务经实质审核后连续closure/实际commit绑定/AUTH/preflight/用户启动。

**ModernTCN v1 免重复探针准入与启动交接（2026-10-22）**：用户最新决定取代此前待探针流程。原ModernTCN-UrbanEV-F4 q4准入Passed、正式队列技术完成及本版已审CPU主干/初始化/梯度/有限性/接线证据，结合相同Urban L12/P1/C11/batch128/float32与U/V额外激活、反向及优化器开销的差异评估，接受Urban首批q4；不声称本版GPU峰值、数值串并行或加速比实测。16.5MiB仅为V共享基中间量。原探针计划撤销且未执行、旧包及Pending历史保留；本次接受记录为外置 `M6-ModernTCN-UV-v1-launch-authorization-r1/resource-reuse-acceptance.json`，不另设机械审批。工程Passed沿用，效果三个目标Not evaluated。

先执行用户授权的TimeMixer v6 r4 owned STOP：监督器及四worker退出、共享锁释放，12完成/4中止staging/8未启动，history保存195 epochs、journal确认554011 updates下限、尾部未知，12份test claim/result保留。没有补test/恢复，原failure文字保留并在 `M6-Timemixer-v6-r4-user-stop-r1` 声明用户主动停止语义，不伪造全批效果结论。此次仅本工作区两文档增量普通commit/push，21已审文件及29依赖不改；实际新commit外置绑定正式AUTH和元数据preflight。首批24项/六波q4/480 run-epochs/1485300更新，其他29MS保留，B_ref=B_cfg=cfg_adapted_v1。用户最后启动，Codex不运行start/supervise/worker、不创建真实attempt，不改启动时UUID/空闲门槛/锁/STOP保护。

**ModernTCN-U+V v1 工程审核闭环（2026-10-22）**：ChatGPT已直接核验23路径、来源/配置、CPU证据及训练/恢复/汇总/门禁，实施与CPU工程review Passed；GPU资源review Pending，三个效果目标Not evaluated，正式完成0。原送审包 `M6-ModernTCN-UV-v1-implementation-r1` 的REPORT SHA-256=`8e3bbd0aa004c635d11e4c4a4b5738d88a2f970787516944f7ac506d5611088e`，file-manifest SHA-256=`c21f44feee56e7767ea648d899905d89532c63d8c48ad2bd5cab195aae98598f`。本轮不重跑15项CPU模型检查，仅增量文档、精确23路径普通commit/push；21个非文档文件保持已审核字节，原失败/计数/报告不改。实际提交与SHA记录在独立 `M6-ModernTCN-UV-v1-review-closure-r1`，不把工程通过写为真实效果或CUDA恢复通过。

53项MS、13格B、8格同版F4复用及三个严格效果目标不变；B_ref=B_cfg=`cfg_adapted_v1`，`paper_C1_r8_v1`仍未执行。首批24项、U-f1/U-f4/V-f1/V-f4/UV-f1/UV-f4六波q4、最大480 run-epochs／1,485,300更新；其余29项保留。资源探针仅在TimeMixer自然结束、设备无须避让负载及原共享mutex非阻塞取得后允许；本次发现其监督器与四worker仍活跃，故探针未执行，GPU查询/锁取得尝试均0，不自动等待。本轮仅准备外置探针与非执行绑定，不创建真实AUTH/attempt，不操作TimeMixer。新增GPU证据须待ChatGPT审核，真实长训练仍由用户最后启动。

来源与证据边界沿用ChatGPT已核对的[FiLM](https://arxiv.org/pdf/1709.07871)、[Non-stationary Transformers](https://arxiv.org/pdf/2205.14415)、[CondConv](https://arxiv.org/pdf/1904.04971)和[Dynamic Convolution](https://arxiv.org/pdf/1912.03458)：U是条件特征调制/统计回注加目标条件辅助汇聚、低秩跨特征调制；直接FiLM可适配，当前U未证明更优，同q下其三个投影比直接FiLM两个投影多512参数。Non-stationary Transformers Table 6已有DeFF统计回注。V是条件卷积的受约束专家改造，参数下降伴随容量限制，不等于已证明计算/显存/延迟降低。本次不实现或运行新对照，不改模型与训练科学合同；以下历史待审状态保留原时点。

**当前分支新增 ModernTCN-U+V v1（2026-10-22，工程统一送审）**：在 `ModernTCN-U+V`／`M6-ModernTCN-UV-v1` 独立工作区，从源项目实际已提交 `0fe117517a3dff6bc4f3c66a9228a72dda06f525` 派生；不操作仍在进行的TimeMixer训练、源码、输出、队列或锁。按用户冻结合同完成U/V两处内部接入、独立53项MS配置、原生loader/Adam/OneCycle/严格best/一次test/恢复与原子发布接线。15项合成CPU定向检查最终覆盖通过（17次用例执行）；48/0主/子Adam、130次完整合成forward／259次呈现、58次backward及6次局部autograd.grad，失败与计数器补账保留。新实现待ChatGPT审核，三个效果目标Not evaluated，完成数0；无真实AUTH/attempt、训练、真实数据模型前向、真实checkpoint读取或GPU操作。详见 [ModernTCN v1增量](#m6-moderntcn-uv-v1)。以下TimeMixer等记录保留各自时点。

**最新v6 r4工程审核闭环与首批启动交接（2026-10-21）**：用户确认ChatGPT已直接读取服务器实际源码、配置、测试及执行证据，r4实现与25项定向合成验收工程review Passed；原送审REPORT SHA为`99ae63eb0e4895a778469e0d9109b47eca85e09c24a6b82f8bb55b16e6d2f631`，file-manifest SHA为`5d30aec2d6d6435283b12869b26ff5d83d5a85471553de786f5f23bacbb91c14`。本轮保持20个新增实现文件的审核字节与科学合同，仅增量登记裁决并办理22路径普通commit/push、实际commit绑定、正式AUTH和元数据preflight；实际commit及结果由独立外置闭环/授权回执记录，不递归回填本文。r4仍0项完成，三个效果目标Not evaluated；53项MS及目标二10格至少7格严格胜配对B的合同不变。只授权UrbanEV F4首批24项、原顺序6波q4，最大480 run-epochs/1485300更新；其余29项MS保留待后续安排，原B只复用。真实训练由用户最后启动，本轮不运行start/supervise/worker、模型或GPU查询。以下待审/无AUTH等文字保留原历史时点；详见同一[r4专节](#m6-timemixer-v6-r4-revision)。

**当前v6 r4定向修订（2026-10-21，统一送审）**：ChatGPT已审核r3首批结果审计及限定CPU机制观察，review Passed，原REPORT SHA为`d5e7d4793ff2eb02fe2582fcc83ec7ffc00f75e99c478c0b45aa3ff349d9441b`；r3目标一Not met、目标二/三Not evaluated。按本轮唯一冻结方案新增独立`timemixer_dev_v6_r4`及入口/配置：U保留目标条件辅助控制器，只改为沿D范数保持的方向接口；V只改稳定真L轴MA3差分和逐token/head的Q/K RMS规范化。新增参数0，科学任务只保留53项MS，r4从0完成开始；r3的24完成/29保留未执行及全部历史成本不回写。25项定向合成用例最终覆盖通过，42次用例执行及初始化/夹具故障与复验保留；主/子Adam56/0、完整合成forward189/1473次呈现、反向68次、合成checkpoint反序列化16次；真实checkpoint反序列化/真实数据模型前向/真实训练/GPU均0。新字节待ChatGPT审核，r4三个效果目标均Not evaluated，无正式训练AUTH或attempt。详见同一[v6 r4增量](#m6-timemixer-v6-r4-revision)。

**当前v6 r3首批审计、机制诊断与MS范围修订（2026-10-21，统一待审）**：实际工程/训练commit为`d29cc0e63a3f741605a65db1d2208a0baa55b8d7`，正式AUTH已按用户授权物化并由用户启动；首批24项/6波完成，独立核账400 run-epochs、1207328 updates、24次原正式test，655文件/48份best-last CPU保存状态自检Passed。目标一Not met，目标二/三Not evaluated；本轮新结果/机制尚待ChatGPT审核。登记后新增CPU有限诊断560次完整前向/32960次样本呈现，复用4096条固定train/validation窗口记录及匹配原B缓存；新test/backward/optimizer/训练/GPU均0。U有完整收益但出现小门/大carrier的共适配，V注意力高度集中且独立收益不足；建议审查接口/关系几何，不自动实施或换源。用户最新决定仅保留53项MS（24完成、29未执行），24项M退出后续；新目标二只与同任务原B比，五EPF市场10指标格至少7格严格胜B，不附加每指标各70%门槛。目标一/三保留，模块形式与来源抽取限制放开，最终新机制仍由ChatGPT选择。原77项resolved、AUTH/runtime和历史时点文字不改，详见同一[r3专节](#m6-timemixer-v6-r3-submodules)。

**最新v6 r3工程裁决与精确closure（2026-10-20）**：ChatGPT已通过服务器直读审核当前r3送审字节，工程review Passed，范围为固定来源、冻结公式、完整接线、版本身份、定向合成检查、失败修复、消费核账及已声明的静态资源分析；不包含真实效果、CUDA恢复或GPU峰值实测。当前执行24路径工程closure及首批24项启动材料定版，实际新commit与逐文件SHA由外置`M6-Timemixer-v6-submodule-r3-closure-r1/`绑定，不回写旧训练身份或递归修改本文自身commit。r3真实结果尚无，目标一/二/三均Not evaluated；r2原工程/结果审核Passed、目标一Not met、目标二/三Not evaluated及历史成本原样保留。本轮不运行模型、checkpoint反序列化、GPU或真实训练；启动绑定仅为review_approved=false的待审草稿，未创建真实attempt。以下原送审/Pending文字保留历史时点，详见同一[r3专节](#m6-timemixer-v6-r3-submodules)。

**最新v6 r3固定子模块实现（2026-10-20，统一送审）**：已审核r2接口诊断文档以`007ea7ec7161c0ecdc597c9c9e4868efa5e1a811`闭环，原122文件诊断包保全。按ChatGPT选定的XLinear/TimeBridge来源与冻结合同，新增独立`timemixer_dev_v6_r3`、配置/入口/启动脚本/测试；candidate仍v6，revision为r3，r2资产不改。U是目标条件辅助混合基控制的native-head门，V是真L轴关系Q/K与原坐标Value时间加工；原TimeMixer完整路径/科学字段保留，不声称已解决r2异常。25项定向合成检查覆盖通过（首轮23/25、仅修两处错误断言并复验2项），主/子合成Adam44/0、初始工程额度120/0，逐step预记账并保留失败；真实checkpoint/数据模型前向/训练/GPU均0。21种登记张量形状验证，Urban B/U/V/UV参数70885/71838/74982/75935，V最大M单份注意力静态378 MiB而非峰值实测。29项目执行文件/固定作者依赖与两论文绑定齐全；见现有[v6 r3专节](#m6-timemixer-v6-r3-submodules)。r3效果Not evaluated、工程待ChatGPT审核，未生成真实AUTH或attempt；新实现/文档未暂存、index空，不做实现closure。

**最新v6 r2接口诊断裁决与文档closure（2026-10-20）**：用户通知ChatGPT已审核`interface-diagnostic-r1`并review Passed，范围为四份状态、登记样本、CPU eval及已实施开关干预：4次checkpoint加载/4次构造、32次完整前向/800次呈现，128条任务窗口记录/120个不同物理历史区间；test/backward/optimizer/训练/GPU均0，不包含CUDA恢复或训练重演。U消息/PDM配对差异增大已有直接证据，所测端点target gamma趋零的主因解释被削弱；offU/offBoth仍异常但不是独立B，尚未定位最初训练根因，不能推出冻结主干；V-last即时作用小不证明训练期间完全无影响。r2工程Passed、首批result review Passed、目标一Not met及目标二/三Not evaluated分别保留。既定任务内必要诊断/技术修复/复验/重试额度由ChatGPT/Codex合理安排，预先登记、实际核账并保留失败成本，不要求用户逐次批准数字；不授权效果择优重跑或本轮真实训练。以下Pending文字保留原时点，新结构按新revision独立登记。见[现有v6专节](#m6-timemixer-v6-implementation-r1)。

**最新v6 r2独立有限接口诊断（2026-10-20，新分析统一送审）**：已审核结果文档以`78ae3c3ad2fd11f84ca0d62e1de14e96fed3b319`普通commit/push闭环，原训练commit仍为`2825c690261c0c785ed401ec38c3bb1480351b8c`。随后独立读取指定4份best/last、构造4模型，完成32次CPU前向/800次呈现，固定128条train/validation窗口记录；观测器输出/持久状态/RNG exact，test/backward/optimizer/训练/GPU均0。实测last的U更新/原finest embedding RMS比约63.31和102.11，两层PDM配对差异累计倍率约7659.53和13565.77；目标gamma远离0，denorm并非所测极端放大的位置。关U/双支路仍有大输出，UV-last的V额外即时变化相对很小；不由此推断训练根因、主干损坏或freeze必要性。详见现有[v6专节](#m6-timemixer-v6-implementation-r1)及结果根`interface-diagnostic-r1/REPORT.md`。本次新诊断待ChatGPT审核；原结果review Passed范围、目标一Not met、目标二/三未评价及旧0前向账不改。新增文档保持unstaged/index空，无新结构、来源或训练许可。

**最新v6 r2首批结果裁决与文档closure（2026-10-20）**：用户通知ChatGPT已审核`first-batch-review-diagnosis-r1`，首批产物与结果review Passed，限身份、24项/6波、248 run-epochs/737045 updates/24次原正式test、封存核账及CPU保存状态审计；不包括CUDA恢复、训练重演或机制因果证明。既有r2工程Passed沿用，目标一Not met、目标二/三Not evaluated。本次只closure已审核结果文档；原训练commit仍为`2825c690261c0c785ed401ec38c3bb1480351b8c`，不改旧AUTH/runtime/checkpoint。当前r2配对不继续小修或原样重跑；保留U目标条件辅助交互需求并重审接口，V能力/位置/来源交回ChatGPT。随后授权的四状态有限接口诊断独立记账，新机制结果待审核，新来源/公式/组合尚未冻结；不回填旧审计的“0真实前向”。详见现有[v6专节](#m6-timemixer-v6-implementation-r1)。以下送审/Pending等均保留各自历史时点。

**最新v6 r2首批完成后审计与可行性分析（2026-10-20，统一送审）**：工程已closure到`2825c690261c0c785ed401ec38c3bb1480351b8c`，用户批准的r2首批24项/6波已完成，独立核账248 run-epochs、737045 updates、24次原正式test；655文件/552封存payload及48份best/last CPU状态自检Passed。目标一逐fold双指标Not met，目标二/三Not evaluated。新的关键证据是U8/8、UV8/8在best后均出现过≥100倍best validation MSE的恶化，B/V各0/8；U/V参数均确有学习，不能归因于零初始化失活。U有局部完整收益但需较大表示接口改造，当前V末端时间子算子的继续投入依据不足，建议交ChatGPT重新考虑分工/位置/来源，不机械小修。详见[现有v6专节](#m6-timemixer-v6-implementation-r1)本轮增量。当前新result/机制报告待ChatGPT审核；本轮无真实CSV读取、新前向/梯度/optimizer/GPU操作，无新候选、许可或训练。以下送审/未训练/Pending保留原时点。

**最新v6子模块边界修订（2026-10-20北京时间，r2统一送审）**：用户新增“来源block内部子模块”抽取边界后，同候选`timemixer-uv-dev-v6`收窄为目标条件化router聚合—分发核心U与串联非线性深度时间算子V；替代原完整TSA/Block送审版，不是原实现技术失败，不新建v7。新revision、执行来源/配置/模板同步；32项本轮定向CPU检查通过，新增合成Adam主87/子0（新上限120/0），旧39项/46次及主99/子8证据原字节保留。v5未恢复，仍4 completed/4 staging、79保存epochs/138120 journal更新且尾部未知；不重复STOP、不判中断批次Not met。v6效果Not evaluated，新字节待ChatGPT审核、真实训练未授权；全部unstaged/index空，无closure或真实许可。见[现有v6专节](#m6-timemixer-v6-implementation-r1)r2增量；以下旧时点文字保留历史。

**最新v5按用户中止、v6实施送审（2026-10-19）**：先最小核验归属并调用原STOP，监督器/4worker已退出、无下一波、锁释放。停止原因`USER_REQUESTED_STOP_FOR_DESIGN_REVISION`，原泛化failure措辞保留。v5仅4个completed＋4staging，保存79 run-epochs、journal确认138120 updates、4次正式test；未持久化尾部未知，批次未完成、目标一Not evaluated，不判效果失败。见[v6统一专节](#m6-timemixer-v6-implementation-r1)：原DSW/完整TSA只改目标条件化合法来源router，原ModernTCN Block只改串联非线性时间算子；完整v6接线与39项CPU覆盖已完成，合成主99/子8，上限200/8。新字节待ChatGPT审核，v6效果Not evaluated、真实训练未授权，不恢复v5或启动v6。旧实现/启动/未训练/Pending保留原时点。

**最新v5原论文来源补齐（2026-10-19）**：现有[v5专节](#m6-timemixer-v5-implementation-r1)新增用户原件核验与方法对照。Crossformer 21页原件已齐；用户通知ChatGPT实际阅读后确认U无需结构返工，模型/接线及原工程证据review Passed。本轮来源/元数据10项无模型复验Passed，未重跑38项；新绑定及文档字节待审核，效果Not evaluated、真实训练未授权。原HTTP403/Pending/缺论文是历史时点，原记录不覆盖；无新候选、训练、closure或真实许可。

**最新v5内部表示模块联合送审（2026-10-19）**：见[v5完整实施专节](#m6-timemixer-v5-implementation-r1)。固定Crossformer-inspired U和ModernTCN-inspired V已完成两处内部接入、完整训练/恢复/评价/封存/汇总及38项定向CPU验收，效果Not evaluated、新实施字节待ChatGPT审核。Crossformer源码/许可已固定，官方PDF403，论文材料未齐但按用户明确例外完成独立公式与CPU验收，真实准入保持材料检查；ModernTCN43页正式PDF的3.1/3.2/附录G已对照。合成Adam主251/子24（含全部失败复验和初始预算超2记录），真实训练/真实test新增预测/GPU均0。首批24/6波仅非执行模板，无grant/attempt，不启动其余53项。v4结果及机制在声明范围已获用户通知接受，不冒称ChatGPT重加载checkpoint或CUDA恢复；旧负向结论与各时点文字保留。

**最新v4完成审计与采用机制/完整split诊断（2026-10-19）**：见[v4完成后统一专节](#m6-timemixer-v4-postrun-diagnosis-r1)。实际训练`831f6192ff48ecd84edaf2302496f48edf941109`完成24项/6波、310 run-epochs/963215 updates；655文件及48份best/last CPU状态自检Passed。额外完整validation/test各24次、48项新scope数值核验通过，旧B/v3完整预测复用；所有原正式test/封存记录不改。目标一Not met（fold1 MSE排序满足、MAE不满足），目标二/三未评价。U缓解fold4低mean代价而fold1收益减弱，V/UV状态内MAE问题与共训补偿仍在。新result/机制包待ChatGPT综合判断；允许讨论非残差/内部接口需求，具体来源和设计必须回ChatGPT。未实施v5、选替代来源、训练或单独closure；以下旧状态保留各自时点。

**最新v4联合实现送审（2026-10-18）**：用户采用ChatGPT给定的原v3提案＋各自采用分数＋固定“不修正”竞争方案，`timemixer-uv-dev-v4`完整模型/训练接线及40项CPU合成验收通过，见[v4专节](#m6-timemixer-v4-implementation-r1)。主/子合成Adam81/8，Urban参数B/U/V/UV为70885/75773/71598/76486；旧实现与科学训练配置保持。本轮效果Not evaluated、新字节review Pending、无真实训练/closure；原完整split报告方法及主要结论按用户交接在声明范围内获接受，不伪称ChatGPT独立复算全包SHA或网络。以下各时点未训练/Pending/停止事实保留，与本轮统一送审。

**最新完整split续接/定位（2026-10-18）**：用户明确批准停止后的诊断专用`cpu_gpu_fullsplit_diagnostic_v2`；见[精确续接与完整结果](#m6-timemixer-v3-full-split-error-continuation-r1)。旧1项复用＋新63项，64过程全部完成并通过新界，旧false与停止包不改。累计validation32/test32、22584batch/2886400呈现、33次CPU best加载，backward/Adam/GPU均0。完整fold4证据支持低历史mean/std状态占比变化与组内相对误差变化并存，输出修正正偏及near-constant状态的残差代价可见；不支持直接推断主干损坏或仅按事后低误差/零标签做门。正式成绩仍为原GPU，目标一Not met、目标二/三未评价；新包待ChatGPT审核，不实现v4/训练/单独closure。旧停止记录保留原时点。

**最新完整split定位停止（2026-10-18）**：见[本轮完整split诊断](#m6-timemixer-v3-full-split-error-review-r1)。32份既有best/64个model-split已预登记；首个B fold1/H3 validation重算MSE差`2.0188995366599016e-6`，超出预登记metric_atol=1e-6，按合同停止剩余63个过程，不放宽容差或重推。实际125次CPU batch/15950呈现，完整validation模型评价1、完整test模型评价0；loader对齐已读取该格test前缀，另记数据访问，不称test未接触。尚未得到fold4完整split定位，不改原结果或训练身份。本轮仅两份文档和外置证据待审，无closure/训练/v4。

**最新v3完成后统一审计与诊断（2026-10-18）**：见[v3审计诊断专节](#m6-timemixer-v3-postrun-diagnosis-r1)。实际训练commit `33e92b150a439ad5dd3d3298e0244835afa9ccd5`；24项/6波、318 run-epochs、931047 updates。655文件、48份best/last的CPU审计自检Passed，目标一逐fold双指标Not met，目标二/三Not evaluated。仅补固定旧样本的v3观测，证实支路作用与主干补偿、典型/大误差权衡；没有足够依据冻结单点v4小修，回ChatGPT综合判断。新result/机制材料待审核；本轮只两份权威文档及外置证据增量、不commit、不训练。以下实现送审/未训练/Pending均保留原时点，不推翻合法完成身份。

**最新v3完整实现送审（2026-10-17，M6-Timemixer）**：用户已接受v2结果/机制报告的声明范围，并由ChatGPT选定SOFTS/APT新输出结构。`timemixer-uv-dev-v3`已实际完成来源审计、原B后并行输出支路、独立训练/恢复/汇总接线及39项CPU验收；主/子合成Adam更新80/8。见[v3统一专节](#m6-timemixer-v3-implementation-r1)。v1/v2目标一Not met保留；v3工程自检Passed、ChatGPT新字节review Pending、Git closure和真实训练未执行、效果Not evaluated。非执行启动模板不构成训练许可；以下“待选择来源/v2新结果Pending/未训练”等保留各自时点，最新状态由本节覆盖。

**最新v2首批完成审计与机制对照（2026-10-17）**：见[统一审计诊断专节](#m6-timemixer-v2-postrun-diagnosis-r1)。v2实现审核及closure已完成，实际训练commit为`553332e5cf73a66a7859f4d51b26905667d21c63`；24项合法完成、318 run-epochs/915691 updates，本轮655文件及48份best/last的CPU审计自检Passed。目标一逐fold双硬Not met，目标二/三Not evaluated。固定旧样本的v2受限诊断已完成，U限幅有效但head放大仍在，tau已进入V路由却无稳定联合收益；首选保留U v2正向信号，V交回ChatGPT综合判断，不冻结v3或新来源。新artifact/result/机制材料待ChatGPT审核，未宣称修后字节已读取。以下“v2未训练/仅实现/Pending”保留旧时点，本轮不单独closure文档、不自动启动后续任务。

**现行同源v2实现送审（2026-10-16）**：机制证据及建议按用户交接已review Passed，用户批准U实际残差限幅和V目标趋势上下文。独立`timemixer-uv-dev-v2`已一次接完模型、配置、worker/恢复/汇总及操作草案，32项本轮同字节CPU复验通过，见[唯一v2专节](#m6-timemixer-v2-implementation-r1)。本轮及此前两份文档增量统一送审、全部unstaged、index空，无单独closure；v1全字节和负向结果保留。v2实现review Pending，真实训练未执行、效果Not evaluated。以下较早“未实现第二版/建议待审”等仍为历史快照。

**现行机制诊断送审（2026-10-16，替代指令）**：见[U/V受限诊断与改造可行性](#m6-timemixer-mechanism-diagnosis-r1)。已完成32份best的CPU只读诊断、4096条固定窗口、9路径及640次完整模型调用；梯度、Adam更新、test新增预测、GPU查询均0。U有明显实际作用，V在UV中出现近均匀/近固定尺度及主要趋势修正的不同状态；仅提出一组最小改造建议，未实现第二版。ChatGPT负责综合裁决及任何替代来源选择，本轮材料待审。首批原/新目标一Not met、目标二/三Not evaluated不变，旧结果和审计原字节保留。

**现行首批完成与结果诊断（2026-10-16）**：见[首批24项审计及结果后逐fold口径](#m6-timemixer-first-batch-review-r1)。训练commit为`437b43b5ace68329ceb5a18cf75952baaa9103c4`，24/24、297 run-epochs、909136更新；本轮48份checkpoint CPU内容及完整artifact自检Passed，ChatGPT result review Pending。原八格/新逐fold目标一数值均Not met，目标二/三Not evaluated；不原样重跑或自动推进53项/正式补fold。下列旧“未启动/未closure/Pending”等定位均保留其历史时点，现行状态以本节为准。

**现行查询修复定位（2026-10-16）**：见[GPU管理查询60秒修复](#m6-timemixer-gpu-timeout-repair-r1)。双指标review Passed、精确12路径已closure到`f48c9104913ca9ab9f38bff605fc6cf3657126f9`。本轮实际查询13.073秒成功，原10秒不足；仅管理查询timeout改60秒，9项无模型复验通过，修后待审核/未closure。用户撤回新serial/q4短轨迹/makespan强制前置，按历史batch128 q4、结构及当前资源快照准备首批，不恢复强制probe。旧“GPU查询Unresolved/联系管理员/需补probe”属于先前时点，保留原字节；无真实训练或可消费许可。

**现行定位（2026-10-15，M6-Timemixer双指标）**：用户已确认MSE/MAE双硬要求、目标二方案A及首批24任务，见[现行专节](#m6-timemixer-dual-gate-r1)。此前runner修复已review Passed并closure至`05d254341c9b5e63c3f5cd0499e6ffd7933d1440`，本轮新双指标接线待审核、未closure。第二轮训练和原模型不变；q4资源/真实启动仍未获准，无真实实验，效果Not evaluated。以下旧单指标、待选批次、Needs repair/未提交文字保留为当时快照，现行合同以本节为准；不修改生产分支文档。

**最新修复定位（2026-10-15）**：见[M6-Timemixer runner三项修复](#m6-timemixer-runner-repair-r1)。上一轮18文件与21/13通过证据已由ChatGPT直读，但runner review Needs repair；首对U/V模型review仍Passed。本轮针对所有权拒绝副作用、恢复前完整校验和精确完成核账修复，第二轮科学合同不变；修后待复审、不closure、不启动真实训练。原自检通过及失败记录保留。

**最新定位（2026-10-15）**：见[M6-Timemixer实际训练接入](#m6-timemixer-runner-integration-r1)。第二轮完整协议已由用户选定；第三轮371/371、5083 epochs已完成，两轮原始review获接受，范围按下节登记。候选worker/恢复/artifact/固定波次启动器已实际接通，本轮CPU证据及接线字节待ChatGPT审核，全部unstaged；未启动真实训练。主判指标、具体启动批次与资源绑定待最终启动合同确认。以下此前“协议未选/第三轮Pending/接线缺口”仅是历史快照。

**最新训练准备定位（2026-10-11）**：见[M6-Timemixer基线与计划记录](#m6-timemixer-baseline-plan-r1)。首对实现及迁移review Passed；精确26文件已closure并单分支push，三端0/0及当时clean边界已独立保存。本轮随后新增来源核对与77项非执行计划，全部unstaged、待ChatGPT审核；协议/主指标未选，无真实训练启动授权。以下此前定位及送审快照保留当时事实，当前以本专节最新准备记录为准。

**最新隔离开发定位（2026-10-11）**：见[M6-Timemixer专节](#m6-timemixer-implementation-r1)。J暂停新开发；依据本次用户通知，首对TimeMixer U/V implementation review Passed、281项CPU验收证据已审核。本轮工作区迁移自检及16项新路径复验通过，迁移增量待ChatGPT审核；无真实开发实验/Git closure。生产第三轮仍与本分支隔离。以下旧文件头及历史正文保留当时“未提交/未启动”事实，不代表当前生产停止。

<a id="m6-timemixer-implementation-r1"></a>
## M6-Timemixer：首对U/V实现与CPU验收送审

<a id="m6-timemixer-runner-integration-r1"></a>
### 第二轮完整协议与训练链路接入送审（2026-10-15）

<a id="m6-timemixer-v4-implementation-r1"></a>
<a id="m6-timemixer-v6-implementation-r1"></a>
<a id="m6-timemixer-v6-r3-submodules"></a>

<a id="m6-timemixer-v6-r4-revision"></a>
**2026-10-21工程审核裁决、真实绑定与启动边界增量**：本轮用户传达ChatGPT对上述实际送审字节的工程review Passed，适用实现、完整接线、版本隔离、定向验收及保留的失败/消费证据，不包含真实效果、CUDA恢复或GPU峰值实测。沿用25项定向验收及原消费账，本轮不重新执行模型验收，不改变U沿D范数保持接口、V稳定MA3差分/逐token-head Q/K RMS及局部float64统计和float32点积/softmax；不改训练、数据、选best、早停或test-once政策。目标一逐fold四H等权双指标严格排序、目标二五EPF市场10指标格至少7格严格UV<B、目标三同r4八格双指标两条路径均沿用；r4效果仍Not evaluated，不借用r3成绩。

本次只提交已审核20个新增文件及两份权威文档增量。原实现包原字节保留；实际closure commit、逐路径SHA和三端/clean检查置`M6-Timemixer-v6-submodule-r4-review-closure-r1/`，正式AUTH、preflight及操作说明置`M6-Timemixer-v6-submodule-r4-launch-authorization-r1/`。用户已授权物化review_approved=true的首批真实许可：UrbanEV F4、fold1/4、H3/6/9/12、U/V/UV，按fold1 U、fold4 U、fold1 V、fold4 V、fold1 UV、fold4 UV六波，每波最多q4；最大480轮/1485300更新。fresh seed2024、L12/P1/batch128、原B配对不变，默认不恢复；29项其余MS仍保留但未纳入本批授权，24项M历史不删。正式元数据preflight仅核验Git/配置/来源/授权/任务/data stat，不能称GPU准入或真实训练；未来启动仍由原监督器检查UUID/余量/锁和STOP归属，GPU查询60秒、最多3次、仅TimeoutExpired重试。启动命令交由用户执行，本轮不创建真实attempt。后续若需换模块，仍回ChatGPT作结果分析、文献比较和选型。本条是新增裁决，以下原送审文字保留历史事实。

**2026-10-21：r3审核裁决与单一r4方向保持/关系规范化修订**

用户传达ChatGPT已实际读取r3统一REPORT（SHA `d5e7d4793ff2eb02fe2582fcc83ec7ffc00f75e99c478c0b45aa3ff349d9441b`），首批结果审计和限定状态/固定train-validation样本的机制观察review Passed；不包含CUDA恢复、训练重演、test逐样本或训练根因证明。原r3为400 run-epochs/1207328 updates/24次正式test，目标一Not met，目标二/三Not evaluated。以下旧待审文字保留当时事实。本次不单独closure文档；与r4完整接线共同送审，HEAD仍为原`d29cc0e63a3f741605a65db1d2208a0baa55b8d7`。

r4 candidate仍`timemixer-uv-dev-v6`，revision分别为`xlinear_vgm_direction_preserving_native_carrier_v6_r4`、`timebridge_stable_time_residual_head_rms_qk_v6_r4`，fusion仍`finest_post_pdm_v_then_native_head_u`。r3/r2源码、配置、AUTH/runtime、训练产物和证据保留；新包/入口/启动脚本/配置独立命名r4。U控制器初始化/aux basis的AST与r3一致，K8/有序aux/显式target/跨P共享不改；只在finest时间映射后、原native projection前计算`v=m*z; z_star=||z||2*v/||v||2`，零z恒等。范数仅沿D；bias、趋势和其他尺度不门控。用float64局部`logsigmoid`共同尺度抵消及缩放范数，原参数/主干仍float32；居中等价写法保留初始exact及门梯度，无gate==1旁路、无epsilon/门下界/额外增益/detach。非零z若出现数值零分母或不可表示输出明确异常。相对重标门与原m观测分别标明。它取消U整体调幅，不限制carrier绝对范数、不消除网络或门全部尺度自由度；可能删除有益幅度信息。

V保持原位置、4×8无bias Q/K、raw Value、`alpha=.5*tanh(a)`及初始化，只将关系输入改为`((G-Gprev1)+(G-Gprev2))/3`后detach，并在每token/head沿8维用`sqrt(mean(q²)+1e-6)`规范化。RMS统计局部float64避免平方溢出，点积/softmax及Value聚合在原训练float32；无可学习温度/affine/Value-output projection/FFN/BN。首行和常值残差exact零，首query均匀；L12/96的通用pmax界约0.962994/0.750818，仅数学性质。差分式改变浮点舍入，明确属于r4，不冒称与r3逐位等价；soft attention或熵增加不等于独立V有效。U范数保持是项目修订，V规范化仅借鉴机制依据，不称原QKNorm或完整Swin V2。固定XLinear/TimeBridge/TimeMixer来源与许可继续绑定，未选择替代模块。

外置`M6-Timemixer-v6-submodule-r4-implementation-r1/`保存来源分类diff、几何证据、测试/失败/消费和启动方案。16组旧U/UV validation数组独立复算平均cos(z,mz)范围0.608615–0.715824，g样本CV范围0.145529–0.252952，并保存cos/g/1g和W(mz)、gWz、差值分布及数组/weights SHA；只做数组运算，不加载旧模型，不证明g无用或导致泛化失败。25独立工程用例最终覆盖通过、42次执行、0 skip；预算主Adam96，实际56，子0；47次合成构造/189次完整前向/1473呈现、222次U/V模块调用、66次Tensor backward+2次autograd.grad、28个optimizer实例、16次合成checkpoint反序列化。首轮新夹具路径漏迁移在读取保护前拦截（已发生16更新保留），只将本轮误置新夹具移入r4失败证据，原r3文件未覆盖；float64初始恒等舍入用居中等价式修复；恢复fixture漏fold由身份保护拒绝，补齐后复用保存合成状态复验，未追加训练。失败日志、执行快照和实际消费保留，未放宽断言/容差或修改科学公式。

新resolved只保留53项MS：目标一24、Urban UV F1/F2/F3目标三24、五EPF目标二5；13格原B和8格同r4 F4复用；总规划上限按保留任务重算1060轮/2999100更新，顶层预算漂移明确拒绝，未改子任务参数。逐任务原profile/data/scaler/marks/L/P/H/seed/batch/Adam/OneCycle/MSE/strict-validation-best/patience/test政策不改，原77项r3配置不改；24项M退出后续但历史不删。新目标二仍10指标格中至少7格严格UV<B，平局不计、缺项不缩分母，分别报告MSE/MAE比例但无各自70%门槛；不与全部baseline比。目标一逐fold四H等权双指标`UV<U<B且UV<V<B`；目标三原八格双路径，不能拿r3 F4填r4。r3历史已完成24/保留29未执行与r4零完成分别核账，r4三个目标均Not evaluated。

参数与r3相同：Urban F4 B/U/V/UV为70885/71838/74982/75935；U953、V4097，新增加参数0。U接口额外O(BRPD)，不物化BRPDK；V注意力按真实MS B×4×L²核账，局部float64增加开销，静态张量量不称GPU峰值。仅准备首批24项/原6波q4/480轮/1485300更新的非执行模板（commit=null、review_approved=false、恢复false）；配置与29执行依赖绑定，未来GPU UUID/余量/全局锁/STOP和60秒最多3次仅超时重试不变。只做waves/语法/未授权拒绝等静态检查，不称正式授权或GPU preflight通过。新实现review后，后续同一工作任务完成精确closure、实际commit绑定、正式AUTH/preflight及用户最终启动，不再拆机械二次送审；本轮不实现其他方案、不长训练、不执行其余29项或M7。

**2026-10-21首批完成、独立审计与机制证据（新内容待ChatGPT审核）**

实际closure/训练commit为`d29cc0e63a3f741605a65db1d2208a0baa55b8d7`；真实授权`M6-Timemixer-v6-submodule-r3-launch-authorization-r1/AUTH.json` SHA为`e03eeb5bacaeef889c77dc995ec6a87b2fbb4d0aa5ca9eba67cb1fe7648797f3`，已由用户启动。结果根`timemixer-uv-v6-r3-development-runs/urbanev-f4-f1f4-uv-v6-r3-first-batch-v1`，attempt为`urbanev-f4-f1f4-v6-r3-first-r1`。本批24项/6波、400 run-epochs、1207328更新、24次原正式test已独立复算；655文件Python/系统checksum、552封存payload、48份best/last CPU状态与Adam/OneCycle/严格best/journal/RNG自检通过。owner及末波worker退出，staging空，无failure/STOP；progress末次RUNNING与77项summary Partial均保留历史语义。不是CUDA恢复或训练重演。既有工程review Passed沿用，本轮结果与机制review待ChatGPT；r2原结果/负向证据及248轮/737045更新等历史成本不变。

原GPU保存未舍入成绩按逐fold四H等权：fold1 B/U/V/UV MSE为0.8799985234060477/0.7206958863319872/0.8924394982140433/0.7119601142900644，MAE为0.6812829584677625/0.6287908757276733/0.6883908202425839/0.6214121662545837；fold4 MSE为0.44521339011192423/0.4349178033873237/0.4509637957904126/0.4295493141718626，MAE为0.4707588790697806/0.4790179365940449/0.47552954185157986/0.47754034993244965。UV平均两fold均优于U，但V均值未胜B，fold4增强臂MAE未胜B，故目标一Not met。UV在f1/H6、f4/H3存在双指标局部代价，不被均值掩盖。目标二/三未执行，仍Not evaluated。r3 U/V/UV在best之后保存的validation MSE最高/best仅1.119965/1.066015/1.075966，没有复现r2百倍恶化；不排除未保存中途瞬态。

新外置包为该结果根`first-batch-review-diagnosis-r1/`。预登记后只读反序列化审计48份状态，另诊断24份best，合计72次；模型构造48次。复用已核验旧固定16节点×16时间grid的train/validation缓存，共4096任务窗口记录、992不同物理历史区间；匹配原B16份trace复用，另8份B best仅SHA扫描、0反序列化/0前向。核心及观测一致性合计560完整CPU前向/32960呈现，非32960独立样本；新test/backward/optimizer/Adam/训练/GPU/真实CSV解析/scaler拟合均0。24模型观测器输出/输入/持久状态/RNG exact；原Normalize临时缓存与持久状态区分。7项工具检查、32组U/32组V保存张量原float32图exact及分组算术通过。首轮双精度重排式与float32输出之差的检查超界保留；修正比较对象为原运算顺序exact，没有提高容差、改模型或重推。另一次展示环境缺numpy成本0，改用既定amd环境，记录保留。

本批实际B已有C→D混合、历史marks、两层PDM、12/6/3多尺度及完整时间映射，不能说没有变量/时间建模；L12/P1/H偏移、target-only及节点样本轴不混淆。固定样本中U门很小但随样本/latent变化、辅助局部输出敏感性非零，同时carrier很大；关U后的路径异常说明共训状态依赖幅度补偿，不是独立B损坏的证明。目标gamma远离0，当前denorm不是奇异放大主因。V多数非首query近硬选择，Q/K/alpha确有学习和非零输出作用；UV有部分即时增益，但完整V单臂证据弱。大关系尺度也使首MA3数学零行的float32残余在部分UV产生非均匀注意力，保留现象，不判作已定位训练根因。没有测到梯度冲突或r3 test尾部；所测validation低历史状态多改善，不能照搬旧候选低需求正偏解释。U值得保留目标条件交互需求并重审幅度接口，V需重审关系几何/预测端作用；外置报告仅给一组未实施、未冻结、可反驳的机制建议，最终由ChatGPT选择。

**用户最新科学决定，替代较早范围和目标二/模块形式约束**：逐原ID映射77项为53项保留MS＝Urban F4目标一24＋Urban UV F1/F2/F3目标三24＋PJM/NP/BE/FR/DE MS H24五项；24已完成、29保留未执行。六M域×四H的24项退出后续执行，历史记录/成本不删、不转为新MS，已绑定77项resolved不原地改写。新目标二“除UrbanEV以外的数据集有70%以上的指标最优”，用户补充明确只与同任务原B比：dataset×H×metric为格，MSE/MAE各一格，当前10格至少7格严格UV<B；相等不计、缺项不从分母删除；分别报MSE/MAE比例但不追加两者各自≥70%的硬门槛。五格B采用身份已核对，不与全部baseline比赛、不跨轮择优；原11域/各8域目标二退休。目标一仍逐fold四H双指标UV<U<B且UV<V<B；目标三仍两fold八格双指标F4<F2<F1且F4<F3<F1，同版F4才可复用。未执行29项不自动获准。

新U/V不强制或禁止残差；取消魔改程度、必须顶会来源小改及必须抽取block内部子模块的普遍前提，允许有依据的原创轻量模块。每项仍须解释任务缺口、计算、输入输出/位置、复杂度与验证，不能以堆层/参数代替机制。TimeMixer增强主线及可追溯原B保留；不因此修改数据、loss/LR/seed/选best/评价或test开发反馈政策。新来源、原创机制及最终组合仍回ChatGPT；本轮没有选源、实施新结构、生成AUTH、训练剩余29项或进入M7。

以下2026-10-20送审/草稿/尚未训练等文字保留当时事实；本次只修改两份权威文档并保持unstaged/index空，原源码、配置、成功artifact与历史证据保持，完整报告/消费/样本/激活/范围映射及前后SHA/patch集中于上述唯一外置包。

**2026-10-20工程审核裁决与closure增量**：用户明确传达ChatGPT服务器直读统一裁决：v6 r3工程review Passed。已审核原包`M6-Timemixer-v6-submodule-r3-implementation-r1`的REPORT SHA为`f449ffb33f473960f9ce4a8f4e00df7f9740baf1f0b804c9f8303b4a2cca2837`，file-manifest SHA为`8441222cf7370da62d0feb034d92590209fcee4661e086e238e1d19247dc8461`，complete/incremental.patch SHA均为`8131829734c28e7a3c4e6a6e2639cb9a0756d73f75211ab9284a15fdbecd8819`，evidence-manifest SHA为`f1ef0d80ecb39d1c4c8a5cd688a1d656767008e7c406d3f551303a8e2e404921`。范围为固定来源/冻结公式/完整训练恢复评价接线/版本身份/定向合成检查/失败修复/消费核账/已声明静态资源；不是效果、CUDA恢复或GPU峰值验证。25独立用例最终覆盖通过、27次执行，首轮两处断言失败与对应修复/复验成本保持；主/子合成Adam44/0、完整合成forward193/backward44，真实checkpoint反序列化/真实数据模型前向/真实训练/GPU均0。本轮沿用上述证据，不重跑模型或旧套件。

资源字段说明：原封存资源表的`K`来自`time_mark.K`（历史native marks维数），不是U混合基数或V头数；本版明确`u_basis_count=8`、`v_heads=4`。这只是closure说明，不改原资源表、模型或测试，也不将最大M的batch1检查外推成batch128/q4峰值通过。已审核22个新增实现文件保持原字节，仅本文及唯一M6增量登记审核裁决。24路径普通提交/push后，实际H3、三端0/0/clean、逐文件SHA与本轮前后文档快照保存在独立closure根；原送审包封存不改。

首批启动准备限UrbanEV F4、fold1/4、H3/6/9/12、U/V/UV、seed2024，按fold1-U/fold4-U/fold1-V/fold4-V/fold1-UV/fold4-UV共6波q4，最多480 run-epochs/1485300更新，B只复用原8格。独立`M6-Timemixer-v6-submodule-r3-launch-preparation-r1/`仅存绑定实际H3的AUTH待审草稿（review_approved=false、恢复关闭/空列表）、任务/资源/操作材料；resolved.execution_authorized=false及历史preparation_head不改。真实启动仍须核验UUID/余量/全局锁、60秒最多3次且仅超时重试；本轮不查询GPU、不启动监督器/worker、不生成真实许可/runtime/owner/children或attempt。r3三目标Not evaluated，r2结果/负向证据保留，其余53项不执行。

#### v6 r3：XLinear辅助基门控与TimeBridge真时间关系子算子（2026-10-20，实施待审）

**身份与来源**：candidate=`timemixer-uv-dev-v6`；u=`xlinear_vgm_target_conditioned_aux_basis_v6_r3`，v=`timebridge_temporal_qk_raw_value_operator_v6_r3`，fusion=`finest_post_pdm_v_then_native_head_u`。来源由ChatGPT本轮选定，XLinear固定`8398e656e945da84a79e8b2f64af8f3f68842e19`（MIT），TimeBridge固定`0f9a83fbc3e1260c9ddd527c522dff0ce4b9554b`（MIT）；TimeMixer仍固定`e24610583b36fdd8c76cc17a8df4e65759a5f460`（Apache-2.0）。XLinear用户现有11页PDF SHA `0c092bda5acb441009bab8b62b091edd256a5ca8492df963378cd8ae3f69c9bb`，TimeBridge正式26页PDF SHA `4dc544641cd166346b8f1a26103612bb4f0d0d605d80c02244e74d71c16df94a`，pypdf实际提取并对照相关方法/源码。未换源、运行作者训练器或安装依赖。固定源码/论文/许可/29项目执行依赖及六作者执行文件的完整SHA见外置`M6-Timemixer-v6-submodule-r3-implementation-r1/`。

**U冻结计算与接口**：raw X是原loader预处理后的[B,L,C]，不改原X、不调用B Normalize或增加统计分支。shared Linear(L,D,bias)得E[B,C,D]，q=显式receiver。K=8个混合基，MS只登记ordered aux槽位，M登记C槽位且以exclusive prefix/suffix在聚合前排自身，不使用全量和减自身或C×C receiver-source张量。`S=sum_j(A_mix[k,j]*E_j)/n_r`；`kappa=2*sigmoid(a_k*q+b_kappa)`；`h=ReLU(kappa*S+c_k*q+b_h)`；`m=2*sigmoid(sum_k(w_k*h)+b_o)`，m[B,R,D]沿P共享。实质改造是`diag(kappa(q))*A_mix`有效辅助基条件化；原XLinear已有内生global token条件，不能声称其无目标条件。无合法aux时整controller跳过、门1，F1按实际日历aux判定。只取Gating_Block两层门控核心与历史Linear，不搬TGM、独立global token、完整模型或作者预测头；不宣称各profile都严格低秩。

U仅在finest原时间Linear(L,P)之后、native变量投影之前：CI0按`sum_d(W_native[r,d]*m[b,r,d]*z[b,p,d])+bias[r]`；MS只改目标行，其他行保留native投影语义。代码用等价`native(z)+W*((m-1)*z)`保留原投影调用和未缩放bias，以矩阵乘避免BRPDK/BRPD物化；不是完整Y0后追加自由aux Value或另建预测头，z不反馈给门控制器。CI1门按b-major/variable-minor对齐B*C，接原共享D→1与reshape。原trend和其他尺度原路径保留，MS仍target-only loss（含电价target2），M全C；V内部变化可影响原生非目标输出，不套旧输出残差模块的全模型非目标不变要求。

**V冻结计算与接口**：只处理最后PDM finest G[B_eff,L,D]（MS B_eff=B；M B*C）。真实L轴左复制首值2位、stride1 MA3，`R=(G-MA3_time(G)).detach()`；无bias Q/K各D→32，4头×8，沿L×L key_time softmax/√8。Value为原G，D按原坐标顺序分四组再拼回；`alpha=0.5*tanh(a)`，`G_new=G+alpha*(A*G-G)`。显式D%4检查，无Value/output projection、FFN、额外BN/Block LN/伪业务C轴/预测头。原TimeBridge PeriodNorm在该路径末轴D计算并处于no_grad；改真L轴是项目改造。detach只阻断关系内容到G，不冻结Q/K参数、Value或backbone。关系可利用全部L，MA3不代表整个算子只看3点。给定有限G的局部无穷范数≤2倍，负alpha不保证凸包，不能外推全模型/训练稳定。

**顺序、初始化与保护**：原全部PDM→V finest→原时间Linear→U门/native投影→原trend/其他尺度→原求和→单次denorm；off/off直接作者forecast。共同B初参/key/dropout RNG保留，新模块各自隔离CPU RNG/dropout0。U历史Linear、A_mix、kappa/目标项沿登记的常规Linear初始化，仅最终w/b_o置0使m初始1；V Q/K常规非零、单scalar a=0使初始恒等。首步部分上游零梯度合法，后续必要权重通路已测finite/有效。原CI1 marks排列/S1等限制不修B。新r3任务ID标记、结果命名空间和spec/config/source/runtime/grant/receipt/checkpoint完整身份隔离，旧r2即使同candidate/shape也在live写入前拒绝；不动r2源码包/配置/许可/产物。

**科学字段、资源与验收**：从SHA锁定的r2 resolved派生`configs/timemixer_uv_round2_resolved_v6_r3.json`，77项逐字段仅id/revision变动，训练/数据/profile/scaler/marks/L/P/H/seed/窗口完全不变；root仅结构/来源/派生与产物身份变更。保留24/24/29、37原B、8同r3 F4、1340最大epochs及逐fold双指标/目标二A/目标三八格严格政策，旧r2 F4不冒充r3。首批原顺序24/6波/480最大epochs/1485300更新只有不可执行计划，真实训练待授权，无AUTH/attempt，不执行其余53项。原B不重训/重测，r2原248 epochs/737045 updates/24次test及负向结果保留。

本轮25项定向合成检查当前全部覆盖通过、27次执行、skip0；首轮两错均为测试对象混淆（非零alpha/L7常值归约数学恒等误判bitwise、把应变化的denorm前q误判不变），只修测试并复验两项，未改模型/初始化/公式或关闭/恢复exact，未放宽预登记方程容差。原始日志/快照及44次主进程合成Adam成本全部保留，子0；120/0为委托下初始工程安排，不写成用户逐项批准数字，未需扩额。新r3合成worker四臂、合法epoch恢复/完整状态exact/test-once/封存/旧revision拒绝已测；21登记张量形状以batch1检查，max-M batch128仅静态资源核算。累计r3根模型构造40+原B合成参照3、完整模型forward193、backward44、autograd.grad6、新合成checkpoint读取18，真实checkpoint/数据前向/训练/GPU均0；冷CLI只测help/waves与bash语法，不称真实GPU preflight Passed，未重跑旧套件或双波。Urban参数B/U/V/UV=70885/71838/74982/75935；U新增953，V新增4097；M C21/L96/D16/P720/batch128的V分数或权重单份378 MiB，不是峰值，反向/临时量/backbone另计，未减batch或执行GPU probe。

r3工程自检完成但新字节待ChatGPT审核，效果Not evaluated，未实施实现closure、真实训练、v7或M7。用户已审核的r2接口报告与本次新工程材料分别保存，不把旧Passed扩成新结构效果或新实现review Passed。

#### v6 r2有限接口诊断已审核裁决（2026-10-20，文档closure）

用户明确通知`interface-diagnostic-r1`的ChatGPT review Passed，限所绑定四份best/last、固定train/validation样本、CPU eval模式与normal/offU/offV/offBoth既定干预。实际4次反序列化/4次构造、32次完整模型forward/800次呈现，128条task-window记录、120个不同原node/历史区间；test/backward/optimizer/训练/GPU为0，未验证CUDA恢复或训练重演。原REPORT SHA `276b9a71060e9ca6f8b0241462f3a0df298c5fd9095d43de15dbc4ac98892852`及122文件诊断包保持封存，closure回执另置同级`interface-diagnostic-closure-r1/`。本轮不追加真实诊断，不把新增审核写回原AUTH/runtime/checkpoint或旧0前向审计。

接受的证据是U消息及PDM有限配对差异显著增大、所测target gamma趋零主因解释被削弱、关U/双支路仍有大输出；共训后主干不等于独立B，未定位最初训练根因，不据此冻结主干。V-last即时作用小也不等于训练全程无影响。r2工程Passed、首批结果review Passed、目标一Not met分别保留，目标二/三Not evaluated，248 run-epochs/737045 updates/24次原正式test及所有历史成本不改。

用户已委托ChatGPT/Codex合理安排既定任务内必要诊断/技术修复/复验/重试额度，执行前登记、实际核账、保留失败成本，无需逐次向用户申请数字批准；不允许效果择优重跑，正常早停/Not met保留。后续新结构按独立revision，不覆盖r2。用户本轮选定XLinear/TimeBridge的v6 r3冻结合同，仅授权独立实现/接线/必要合成检查，未授权真实训练或新AUTH；新实施字节须另送审。

#### v6 r2有限接口诊断：独立授权、实际观察与限制（2026-10-20，待审）

第一段仅两份权威文档closure，实际commit `78ae3c3ad2fd11f84ca0d62e1de14e96fed3b319`，当时本地/tracking/network一致、0/0且clean；外置回执在结果根`first-batch-review-closure-r1/`。这是文档commit，不写回原训练AUTH/runtime/checkpoint。第二段新证据只在`interface-diagnostic-r1/`，先固定计划/绑定/样本，再逐份CPU读取f1/H3 U的best5/last10与f4/H12 UV的best1/last6。实际4次反序列化（旧48份审计外新增读取）、4次构造、24核心+8观测一致性=32完整forward、800次呈现，固定128条task/split/window记录；无失败或重试forward。原fold-train scaler直接复用、fit0，四载体每任务仅解析到validation末端（648/2634条），总8次前缀解析/13128条数值记录；共享文件缓冲与fold间日期重叠如实记录，未构建test或取test标签。CPU线程1/eval/inference_mode、原float32；test/backward/optimizer构造与step/训练/GPU均0。

四状态观测器均输出exact，参数/持久buffer/CPU及Python/NumPy RNG、输入/marks和开关恢复保全；Normalize临时mean/stdev缓存与持久状态区分。24核心路径finite，原float32 denorm重构与尺度求和exact。validation固定32条中，U更新/原embedding RMS比best为1.014/1.638、last为63.309/102.113；last消息RMS19.940/15.392、桥接更新83.360/106.236，两层PDM后normal-offU的finest差异RMS为638500.84/1441179.32。其累计有限差异倍率约7659.53/13565.77，支持实际增益链疑点，不能称Jacobian或梯度冲突。原趋势旁路同状态开关间exact不变，跨best/last共同训练后也变大，不能确定谁先改变。

目标Normalize gamma四状态约2.7144/4.6035/1.1153/4.4256，原eps=1e-5、分母gamma+eps²无零/非finite；配对恒等式`Δy=stdev/(gamma+eps²)*Δq`已按保存张量核对，float32舍入残差另存。validation normal-offU的最终/denorm前差异RMS比约0.1592/0.1474/0.3217/0.1668，削弱所测端点“目标gamma趋零导致极端逆放大”，不排除中途瞬态、其他通道或输入Normalize的间接作用。UV-best少数窗口gain略大于1，不宣称永不放大。U-last关U的validation MSE仍约228988.50，UV-last关双支路仍约64837.06；这是增强模型自己的共训主干，不是独立B，不能据此判主干损坏或自动freeze。UV-last V更新/输入RMS约1.30e-6，normal-offV目标最终差异RMS约0.1521；两组BN统计/affine/counter完整且eval冻结，离线批统计与running比较不构成train反事实，不证明BN因果。

新证据支持重审消息/bridge/PDM的表示坐标与增益链、削弱“只改denorm或只关闭一次U就能解释全部”的说法；并未证明改吃normalized seasonal X就是修复，raw_x已被loader预处理且U已有DSW/pre_norm，可能包含有用辅助水平信息。当前r2配对仍不小修/原样重跑，U保留功能需求、V能力/位置/来源交回ChatGPT；新来源/公式/组合未冻结。7项纯数组/元数据自检及实际观察一致性通过，两处前置外置元数据格式错误保留、模型消费0；未重跑原32项或历史套件。全部小样本误差仅解释本次输入，不替代原成绩/gate，不宣称已测梯度冲突、变量因果、test尾部或全部24项根因。原655文件与执行来源保全另有SHA核验；本段新文档未提交，机制报告待ChatGPT审核。

#### v6 r2已审核结果裁决与独立诊断边界（2026-10-20）

用户明确传达ChatGPT裁决：既有r2工程Passed沿用；`first-batch-review-diagnosis-r1`首批产物与结果review Passed，范围为身份、24项/6波、248 run-epochs、737045 updates、24次原正式test、保存结果/封存完整性和CPU状态审计证据，不包含CUDA恢复、训练重演或机制因果证明。目标一Not met，目标二/三Not evaluated。原REPORT、清单、消费记录与文档快照不改，文档closure新回执置于结果根同级`first-batch-review-closure-r1/`；实际文档commit与原训练commit分开记录，不写回原训练身份。

当前r2配对不继续小修或原样重跑。保留U目标条件辅助交互需求，但需重新审视表示接口；V完整效果不支持继续沿当前配对调试，其能力/位置/来源交回ChatGPT。新U/V可非残差，仍须抽取来源block内部具体子模块并作实质改造；新来源、公式、接入与组合由ChatGPT综合决定，Codex不自行选源或实现下一候选。

本次另授权独立`interface-diagnostic-r1/`：仅f1/H3 U和f4/H12 UV的best/last四状态、固定train/validation各32窗口及规定开关矩阵，CPU读取≤4次、构造≤4次、完整forward≤32次/呈现≤800；test/backward/optimizer/训练/GPU均0。先完成本次文档closure，再固定计划与绑定后执行。新增诊断不属于旧“0真实前向”审计，不更改原test次数/训练成本；其新结果须另送ChatGPT审核，不生成AUTH或启动训练。

#### v6 r2首批完成后：审计、TimeMixer任务适配与模块可行性（2026-10-20，当前结果状态）

**实际身份与完成审计**：同W/M6-Timemixer，工程/训练commit `2825c690261c0c785ed401ec38c3bb1480351b8c`；u=`crossformer_router_core_target_conditioned_v6_r2`、v=`moderntcn_temporal_operator_only_serial_v6_r2`、fusion为两个既定内部表示位置。AUTH SHA `7532a7e7394d81659b37b16e1267634c6e27bc23f5f9f33ca8f43e46a11e8655`、实际runtime SHA `c615d2bed4e30396c0b1bb00e29613cf63f4c9f6e9652119c6a860527880a6f9`、resolved SHA `ff886f7a97b59f05259798a7f1d846aaa593e362fba8caa7c086fe303c2df210`匹配；旧同candidate完整block版未混入。24精确ID/6波、24 runs/receipts、registry与producer/profile/source/config/data/environment一致，staging空、无failure/STOP，owner/末波PID-start对应进程均退出。progress末波RUNNING和summary的77项Partial保持历史/全局含义，不改原文件、不补其余53项。

独立复算248保存run-epochs、737045成功Adam更新、24次完整正式test。实际655文件、552不同封存payload的Python/系统SHA一致；48份best/last只读CPU加载各一次，key/shape/dtype/finite、活动Adam/moments/step、scheduler LR/beta1/进度、strict validation-MSE best/早停/cap、BestState/journal/RNG格式通过；V两组BN running_mean/var/counter/affine完整、counter与steps相符。12个作者CI0未调用参数没有假moment，合法初始化不误判。test按validation-best、一次claim/封存、SSE/SAE/elements/目标0/尾batch/标准化空间一致。B复用原第二轮8格，核对保存result/history/profile/scaler/node/marks/边界/元素数；不读B权重或重测。B旧schema未保存单独test_data_sha，不补造字段；当前v6 test_metadata/test_data_sha与同任务已审v4记录一致。此为Codex保存状态/artifact自检，不等于CUDA恢复、数值轨迹重演或ChatGPT重新加载checkpoint。

**完整保存成绩，四H等权与严格双指标**：fold1 B MSE/MAE=0.8799985234060477/0.6812829584677625，U=0.7995004858105131/0.6726365681555820，V=0.8865957696406738/0.6838856602109945，UV=0.8122661206775458/0.6773886987433739；fold4 B=0.4452133901119242/0.4707588790697806，U=0.4339876186622237/0.4802907220886016，V=0.4542638871443666/0.4757689000633452，UV=0.4466555464397650/0.4897209345121413。未舍入数值与逐H/差值保存外置。U相对B：fold1 MSE−9.148%/MAE−1.269%，fold4 MSE−2.521%/MAE+2.025%；V两fold均值双指标未胜B，UV两fold均值双指标均未胜U。fold4 U仅H9双指标胜B，H12仅MSE；V仅H9双指标胜B，不把局部正向说成全无效。目标一每fold各要求UV<U<B且UV<V<B，两fold均Not met；不跨fold抵消、不增加逐H门槛或固定U/V顺序。目标二方案A、目标三原八格两路径本批Not evaluated。

**当前B能力与限制**：L12/P1、独立H3/6/9/12、CI0、实际12/6/3尺度、moving_avg3、两层PDM。原loader预处理X有目标/价格2/天气3/日历5，另有历史native marks4；每尺度Normalize→pre_enc season/trend→season的C11→D64 kernel3 TokenConv与marks→PDM的latent分解/非线性特征和时间尺度混合→原预测映射+原trend旁路→求和→一次denorm→MS target0监督。B已有变量/辅助利用与时间加工；D不是原变量轴，各尺度Y不是独立监督专家。只有12点实际历史、无未来辅助/node-ID/图；多分辨率不能补出周/月目标观测，不把单P输出判为设计错误。B八格validation最大/best比约≤1.162，七格best在LR下降段，fold4平均MAE胜全部增强臂，不能预设B很差。仍有U可改善的完整指标空间，但不能据此证明某个外生源的因果增量。

**最强的新诊断证据与边界**：U八项best为fold1[5,5,2,4]、fold4[2,2,2,3]，UV为[3,2,2,1]/[3,2,3,1]；含U的16项在best后全部出现过≥100倍best validation MSE，train epoch均值也恶化，B与独立V均无同类轨迹。U/UV bridge最大奇异值best约2.62–4.78、last约11.17–20.70；这支持原协议下表示/优化适配与增益链的具体疑点，不证明静态范数就是实际激活放大。UV八个best均在OneCycle上升段；高LR阶段有关联，但早best本身不等于训练不足，B fold4/H9也best2却不暴走。原严格选点与patience合法，不将效果差或合法早停改成技术失败，不自动延长/重跑/改LR。

U的23个参数tensor、V的6个参数tensor在各自8个best及UV best中均有初始化偏移与活动Adam，削弱“零初始化导致完全没学到”；不等于真实来源利用/因果证明。U本批L12无左pad，只在同patch完成目标条件路由，经bridge直接加共享season embedding；raw X与原Normalize/pre_enc坐标的关系、两个router压缩、后续CI0 PDM未使用定义中的LN，是需要实际激活区分的接口疑点。V只有最后finest的depthwise时间算子，不直接改原trend或其他尺度这层预测，可能与已有PDM时间加工重复；L12下4个位置有完整9点卷积支持、8个位置受零边界影响，但BN train沿batch×time统计，整个训练计算不只九点。两组BN静态gain有大值，但U-only无V BN也恶化、V-only有BN却平稳，削弱“BN是全部失败唯一原因”。未测量激活/attention/梯度、来源贡献、BN真实train/validation分布或同权重旁路，不能说已证明梯度冲突或路由饱和。

另一个可核验限制是：fold4 U四个H的validation MAE最小轮均与所选MSE-best轮相同。因此不能仅靠“按MSE选点而非MAE”解释这些任务的test MAE问题，更不能据此用test换checkpoint。

当前没有逐样本v6预测/残差。fold4 U平均best-validation双指标较B好但test MAE较差，不代表每H验证MAE都改善（H3原本较差，H6/H12出现相对反转，H9保持收益）。MSE/MAE分歧只能提示误差权衡，不能把旧v3/v4低状态偏差/尾部发现套成本版。当前增强模型own-backbone也不等于独立训练B，不由其差异自动推出freeze/stop-gradient或辅助loss。

**明确可行性与职责**：分析已完成，不将整个报告挂Pending。U保留“目标条件辅助交互”的需求，但当前结构需较大表示接口改造/改变接入方式，不支持原样重跑或无机制证据的小修；唯一优先方向是router-to-backbone坐标/条件与目标-来源分工对齐，具体公式与位置尚不足冻结。V本版末端时间/BN子算子两fold完整增益均未成立，当前证据不支持沿此机制继续投入，建议ChatGPT重新考虑能力、位置或来源；不等于证明任何同源改造不可能。组合不继续当前配对的小修，不能原样保留fold4 MAE失败的U、仅换V就保证目标一。新模块不必残差，但仍须抽取来源block内部具体子模块、实质改造、保持完整TimeMixer backbone和原MS/M/F1–F4/数据协议约束。具体新论文/可核验实现的检索、比较、入选/排除、来源选择与组合设计均回ChatGPT，本轮未选源、命名新候选、改源码或冻结新revision。

外置附一份**未执行**最小诊断建议：UV fold1/H3与fold4/H12各best/last、固定train/validation各32窗口，正常/offU/offV及观测一致性，最多32次CPU完整forward/800呈现，0test/梯度/Adam/GPU；仅用于区分消息增益与共训/BN/denorm变化，不生成许可、不自动执行。即使不批准，也已有充分证据支持“不继续当前r2配对小修”的判断。L/P/H、输入、loss/LR/batch/patience/seed/冻结/选点/test政策变化均是另行科学方案，不能混入纯模块收益。

**本轮工具/消费/停止点**：计划先存，外置日志解析器首轮因CuDNN warning前缀格式失败，24项均在checkpoint加载/模型构造前停止，首代码/日志/错误证据保留；仅修外置解析器并4项纯文本检查通过，再完成48份唯一CPU状态审计。系统checksum共48次run级核验（首轮24+复验24），不同payload552，不隐藏重复读取。分析另8项纯合成/保存指标fixture通过，无工程32项或历史套件重跑。CPU schema模型构造24、checkpoint反序列化48；真实CSV/旧版或B checkpoint读取、新真实train/validation/test前向、关分支干预、backward/optimizer构造/step、GPU/下载/安装均0。未因缺逐样本观测将全部分析写Pending；仅ChatGPT新result/机制review待审。

统一包在`timemixer-uv-v6-development-runs/urbanev-f4-f1f4-uv-v6-first-batch-v1/first-batch-review-diagnosis-r1`，含REPORT、完整结果/轨迹/48状态/BN静态证据、假设表、明确可行性和待决事项、工具/消费/来源/完整SHA及文档patch。原655文件、AUTH/runtime/claim/checkpoint/日志、v5中止记录、旧负向证据及共享来源/保护仓库/tag保持原字节。最后仅canonical/M6两份文档增量，unstaged/index空，不stage/commit/push/merge，不建current_status/平行milestone、不实现v7或训练/补其余53项/M7。工程自检Passed、效果Not met与ChatGPT review待审分别登记，返回综合判断。

以下为此前v6 r2实施与来源边界的历史时点，原文保留。

#### v6 r2增量：仅抽取router核心与深度时间子算子（2026-10-20北京时间，当前活动结构）

**合同与历史身份**：本次由用户新增抽取边界，收窄尚未提交、未真实训练的完整block版；不是前版实现错误、不是v7，也不批准旧版closure或训练。候选仍`timemixer-uv-dev-v6`，当前`structural_revision.u=crossformer_router_core_target_conditioned_v6_r2`、`.v=moderntcn_temporal_operator_only_serial_v6_r2`，fusion仍`two_internal_finest_representation_residuals`。旧同candidate完整block revision在checkpoint/config/grant/runtime/receipt/恢复前校验中不能冒充r2；测试使用相同张量shape、仅替换revision，确认拒绝且live参数/optimizer/scheduler/RNG不变，不依赖shape mismatch。

**改前保全与停止事实**：设备b82673a648b4，HEAD/local/tracking/network均`573177153c14b64e21594ad7a0527113a4c7d0fc`、0/0；原28路径unstaged/index空。已把原manifest中28文件完整字节、SHA/Git mode和当前diff保存到`M6-Timemixer-v6-implementation-r1/submodule-boundary-r2/before/`；原顶层送审包、39项/46次日志、合成产物、主99/子8成本与停止回执不覆盖。仅复核既有停止回执、191份v5原文件stat及所属PID未恢复，不再调用过期STOP、不读真实checkpoint/补test。v5的4 completed/4 staging、79已保存epochs、138120 journal确认updates和未知未持久化尾部保持原解释，主动中止不判完整目标一失败。

**来源与真实执行边界**：沿固定TimeMixer e2461058、Crossformer c10c8ead、ModernTCN 56a9a2c0/PDF/许可，未下载或改选来源。活动`author_components.py`仅留`DSW_embedding/FullAttention/AttentionLayer/get_conv1d/get_bn/conv_bn`，未改叶子源码AST；完整TSA、ModernTCN Block、ReparamLargeKernelConv、fuse_bn/merge实现已从活动执行包移除，没有通过继承、实例化或flag绕过。完整原实现可从改前快照及固定只读作者仓库恢复；许可和改编说明保留。已有axis_ops未扩建，当前DSW只使用两种字面轴变换。抽取范围收窄/降参数/桥接不是主要创新；主要变化仍为目标条件化合法来源router和非线性串联时间算子。

**U当前公式/接口**：p=stride3、32维、4heads×8、router2、dropout0。原预处理X左复制首值补3倍数，经来源DSW Linear3→32＋position[1,C,N,32]＋pre_norm得E[b,N,C,32]。每receiver r：`target=E[:,:,r,:]`，`R_cond=router[None]+target[:,:,None,:]`，`buffer=dim_sender(R_cond,E_A,E_A)`，`message=dim_receiver(target,buffer,buffer)`。MS合法来源为ordered aux、M为除自身的C；聚合前排自身，无源严格零且不调用空attention，无null/手工统计/时间attention。只输出message，不做target+message或任何norm1–4/MLP1–2/完整TSA残差链。接收者共享sender/receiver、固定顺序循环，不物化[b,N,R,C,32]；专属路由最坏二次C，不宣称原共享router线性复杂度。零bridge32→3D恢复[b,R,L,D]并裁左pad，只注入finest embedding/首PDM之前；MS唯一receiver，M按b-major/variable-minor，不按历史marks.repeat定义变量排列。

**V当前公式/接口**：直接G[B_eff,L,D]转[B_eff,D,L]，`Q=conv_bn(k3,d3,pad3)(GELU(conv_bn(k3,d1,pad1)(G.transpose(1,2))))`，`G_new=G+Q.transpose(1,2)`。两卷积in/out/groups=D、stride1/biasFalse、原零padding；两BN eps1e-5/momentum0.1/affine/track。不构造Block/SerialTemporalBlock、Block.norm、ConvFFN1/2、nvars伪轴、D→2D→D投影。仅第二BN weight/bias置0，两个conv/第一BN标准初始化、running统计原默认，初始Q严格0；不再ffn2零初始化。train更新/eval冻结，两组running_mean/var/counter完整保存恢复，B_eff*L=1 train按原BN限制拒绝；批统计依赖与卷积局部覆盖分开，不宣称整个train只依赖九个时间点。非线性串联不可merge，不新增特征通道混合。仍只更新最后PDM finest，再走原head/趋势旁路/求和/一次denorm，不在最终预测后校正。

**接线、独立身份与科学保护**：原model.py、adapter、worker、训练状态/队列/loader/调度/BestState/封存/指标逻辑保持改前字节；通过现有版本化接口接入，BN校验动态覆盖两组，无三组BN硬编码。off/off原作者路径、共同B初参/key/dropout RNG及模块独立CPU RNG、原X/pre_enc/趋势输入和两个插入点不变；内部MS原生非目标C输出允许变化，监督仍显式target-only，M仍原C。77项除structural_revision及source_binding技术身份外逐字段不变；原24/24/29、37B、8同r2 v6 F4、1340epochs、原数据stat/scaler/marks/L/P/H/fresh2024/Adam/OneCycle/MSE/validation严格best/patience及test政策保持。首批24 ID与顺序、6波q4、480epochs/1485300updates不变，其他53项不执行。三目标严格双指标口径和缺项/跨版本拒绝保持，不传播CPU/GPU诊断容差。所有权、成功不可覆盖、test一次/不明claim拒绝、STOP/child/wave保护不删除。

**本轮定向复验与消费**：14项模型/叶子边界与独立公式、8项选定实际合成worker/恢复保护、7项配置/指标元数据、3项来源/非执行交付/BN损坏检查，共32项32次，全部通过、0失败/0skip。验证实际Module树/key/调用路径没有父block，DSW/AttentionLayer/conv_bn对源，R+target确实影响sender buffer，聚合前排自身/MS非0target/M/左补短L/空源，V独立卷积BN公式/零第二BN/两组BN训练冻结及恢复、合法延迟梯度、关闭与少量更新/RNG exact。四臂与F1–F4/EPF/M合成CSV实际worker、合法epoch全状态/批序exact、test复用/未知claim拒绝/封存/重复身份与旧revision拒绝均通过。独立重排公式float64仅沿预登记rtol=atol=1e-12，关闭/恢复exact未放宽；无删安全断言/增skip。新合成Adam主87/子0，上限120/0逐step前检查；旧主99/子8单列不倒改。原双波fork/runpy及其8 child更新只读复用，未重跑旧39/38/40/281套件或双波，不冒称新冷exec/GPU链路通过。真实CSV/旧真实checkpoint读取、真实train/validation/test新预测、GPU/下载/安装/v5恢复均0。

**资源、不可执行材料与停止点**：实算U=`8640+64N+32CN+96D`，V=`10D`；Urban N4/C11/D64的B/U/V/UV为70885/87333/71525/87973。V持久buffer为256float32＋2int64（1040bytes）。已按77个完整profile实算参数和shape；U只计DSW/专属路由/桥接，不再计Cross-Time/MLP，V只计两层depthwise/BN而非Block/FFN。最大M Weather C21/L96/D16/P720单列，静态工作量不等于实测峰值或收益保证。沿既有A800/batch128 q4历史证据保留首批q4评估，不机械降q1/恢复强制probe；未来真实启动仍UUID/余量/锁/STOP和60秒/最多3次/仅超时重试。本轮`submodule-boundary-r2/launch-preparation`为新revision不可执行模板：commit=null、review_approved=false、resume=false，任务/policy与旧清单逐字节一致；无真实attempt/grant/训练。项目仍28路径unstaged/index空，无stage/commit/push/merge，旧v1–v5/共享来源/保护仓库/baseline不变。v6工程自检Passed、效果Not evaluated、新字节ChatGPT待审，交回统一审核；不自动closure、恢复v5或训练v6。

以下为原完整block版的历史送审时点，已由上述用户新合同取代，原日志和失败成本保持原身份。

#### v6：轻改完整来源子模块并保持两处内部接入（2026-10-19，统一送审）

**v5停止优先及真实基点**：本轮先核设备/AGENTS停止规则、owner PID/start/runtime SHA和children任务/PPID/root/attempt，再立即调用用户指定的旧`source-completion-r1/launch-preparation/commands.sh stop`，exit0、stdout null。原监督器19375及其当时四个worker均退出、无后续匹配进程，STOP存在，failure记录next_wave_dispatched=false；全局锁非阻塞检查已自由并立即释放。无pkill/额外kill/GPU reset或无关进程操作，原监督器按已实现路径结束所属子进程。progress最后RUNNING保留历史快照，不修原artifact。外置原因明确`USER_REQUESTED_STOP_FOR_DESIGN_REVISION`，不是技术失败或自动技术补做授权。当前实际HEAD/runtime/closure均为`573177153c14b64e21594ad7a0527113a4c7d0fc`，起点三端0/0、clean；tag解引用`fa9665627e6fcfb1d0c2bc22d943ca9666304fd6`匹配。

v5保留4个fold1 U完成run与4个fold4 U staging，共79个保存epoch、journal确认138120次更新，4个保存正式test；当前中断epoch尾部未持久化step未知，不声称确认数为实际执行精确总数。原191文件成员/大小/mtime经停止后再核对不变；没有加载真实checkpoint、补test/训练、删除/覆盖/迁移/封装artifact。已完成四H U的保存指标可在停止回执汇总，但不替代完整目标一，v5本批Not evaluated（不写24完成/Not met）。所有历史许可、claim、日志、负向证据和成本保留。

**固定来源与实质变化边界**：TimeMixer e2461058、Crossformer c10c8ead、ModernTCN 56a9a2c0只读。Crossformer用户21页原件SHA `a5fcad16cd810492124af3efad2fa18f7c5f9556fb303bb88360cceaf396183a`与已核验ModernTCN PDF/许可复用，无新下载或选源。按Apache-2.0/MIT抽取原DSW、完整AttentionLayer/TSA、ModernTCN Block及helpers，源码方法体AST一致，许可落在新包。只有U router gather和V self.dw是两项主要改动；桥接/维度/段长为固定接口，不算主要创新。不新增手工统计/log1p/null采用门/原型/kappa/额外loss或业务特征。

**U公式与来源保留**：p=stride3，d_u32、4heads×8、d_ff128、router2、单TSA/dropout0。X是原loader预处理历史，左复制首值补齐；原DSW线性3→32＋位置[1,C,N,32]＋pre_norm，不保留v5 mu/sigma/z/statistics。完整time_attention/out projection、norm1/2、MLP1与两残差产生T[b,N,C,32]。每receiver r的`R_cond=R+T_r`，在sender聚合阶段即排除非法自身K/V（MS ordered aux/M其余C），再receiver(T_r,buffer,buffer)，原norm3/4、MLP2与残差保留。无源主干更新严格零，无null或全mask softmax。接收者共享权重、确定原变量顺序循环，无成对D维拼接；M专属router最坏二次C工作量。仅bridge32→3D无bias零初始化，恢复L/crop左补后加finest embedding；MS显式目标、M b-major/variable-minor。完整HED/decoder/head没抽取是子模块边界，不是额外科学变化。

**V公式与BN边界**：原PDM最后finest [B_eff,L,D]按[B_eff,1,D,L]使用原Block nvars1、dff2D/drop0，norm BN、ConvFFN1/2、原reshape/permute/group和residual主体未改。只换self.dw为`conv_bn3(d1,pad1) → GELU → conv_bn3(d3,pad3)`，in/out/groups=D、无bias、原零填充；每层BN eps1e-5/momentum0.1/affine/track原样。仅ffn2pw2 weight/bias0，FFN1不另零；新算子不能做原线性大小核merge，不调用merge_kernel。nvars1的FFN2是逐特征非线性，不宣称业务C交互。train BN依赖批内样本，eval冻结并验证独立计算，B_eff*L=1 train按原限制拒绝；running_mean/var/num_batches_tracked完整保存/恢复exact、缺失/非finite/错dtype先验拒绝且live状态不写。

**主干、初始化及接线**：保留原两个内部位置：embedding0后U→原全部PDM→最后finest V→原future/head/趋势旁路/求和/一次denorm。forecast必要副本按v5仅命名迁移，off/off直接作者。B key/共同初参/dropout RNG不变；新增模块独立CPU RNG、原Linear/Conv/BN/LN标准初始化、位置/router randn，末层0导致合法延迟梯度，冗余bias不强求非零。MS原C输出/target-only loss且非目标可变化，M全C与原marks/CI1限制不改。完整配置→原loader→v6→Adam/OneCycle→严格validation-best/早停→best完整test一次→checksum/原子封存/汇总及epoch恢复已用合成CSV跑通；所有权拒绝无副作用、活动Adam/RNG/进度、成功不可覆盖、不明claim拒绝、精确child/wave/STOP保护保留。旧v5许可/状态/F4不能冒充v6，更不恢复v5。

**科学合同与范围**：从实际审核v5 resolved SHA `0bd254b0812d47bf9414e1a9b761c083f1b7fe2132583f7abcfc4699fdb8672d`派生。77项逐任务除id/model/candidate/revision外一致，原profile/data/scaler/marks/L/P/H、fresh2024/float32/原batch/Adam/OneCycle/MSE/validation严格best/patience及test政策不改。31项source覆盖原只读依赖、抽取代码、轴兼容和许可。24/24/29、37格原B、8格同v6 F4、1340epoch；首批按实际v5许可映射24ID原顺序/6波q4/480epochs/1485300updates，B复用8格，其他53项不授权。目标一逐fold独立四H双指标、目标二方案A各≥8/11域且集合可不同、目标三原八格两路径/F2-F3不排序；严格未舍入、精确覆盖与缺项拒绝不改，不使用诊断容差放宽exact或效果。

**定向验收与真实消费**：39项覆盖14模型/来源公式、12真实合成worker/恢复保护、10指标配置、3交付/BN。原B关闭输出/loss/梯度/少量Adam-OneCycle/RNG exact、四臂初始输出、DSW/pre_norm/time stage对源、receiver条件sender实际变化与聚合前排自身、V原Block/非线性串联/BN/FFN、短L/非0target/MS/M/F1–F4/EPF/长P、完整恢复和拒绝负例均通过。首次source导入发现amd缺einops，按原6种固定布局用本进程局部torch轴等价实现，不安装或全局monkeypatch；独立布局公式验证，作者方法体保持。模型首轮13过1失败，原因是exact同源time阶段参照误传独立补算输入；改为同一实际embedding后该项exact单独通过。保留首日志/受影响复验，不删断言/skip或加容差，不改模型迎合测试。

主99/子8累计合成Adam，上限200/8先登记并每step前锁内检查；不倒改旧v5成本。原38/40/281套件未重跑。双波只声明CPU fork/runpy，冷CLI仅元数据，45秒边界不变，无新的强制GPU/serial-q4/makespan probe。未启动新的真实train/validation/test前向、旧真实checkpoint读取或GPU查询；已在途v5实际成本单列，不将其混为合成更新或v6科学run。

**参数/资源/非执行材料**：Urban B/U/V/UV=70885/108517/88677/126309；U=29824+64N+32CN+96D，V=4D²+22D。V BN持久buffer为6D个float32＋3个int64，Urban1560 bytes。77完整profile已构造静态计数、无forward；U明确时间阶段[b*C,4,N,N]及receiver专属gather[b*N,4,2,|A_r|]，不能只套v5常数。最大M Weather C21/L96/D16/N32、P720，时间attention score约44MB、各receiver gather合计约55MB、单FFN hidden约33MB（单张量静态量，不是峰值/梯度全图）。首批Urban增量按既有A800/batch128 q4证据评估保留q4候选，无GPU可用性/峰值声称；实际未来仍查UUID/余量/锁、管理查询60秒/最多3次/仅超时重试。

唯一外置根`M6-Timemixer-v6-implementation-r1`包含停止回执、源码抽取/核心diff、科学资源、测试失败复验/消费、精确manifest及完整/增量patch。不可执行模板commit=null/review_approved=false、默认不恢复、无v6真实attempt，操作命令按启动/日志/状态/完成/逐fold汇总/STOP分开，tmux优先、真实长训练用户最后启动。当前仅新v6文件与两份权威文档、unstaged/index空，不add/commit/push/merge，不单独closure停止记录、不恢复v5/训练v6/M7。工程自检Passed，ChatGPT新字节review Pending，v6效果Not evaluated、训练未授权，统一交回审核。

**最终字节窄复验补记**：抽取文件4处源尾空白清理后AST完全一致；v6 preparation_head已改为实际基点5731771，不沿用源v5继承的历史491ce98准备标记。更新仅影响技术来源/配置身份，任务科学字段不变。随后只复验来源轴变换、合法epoch恢复、BN损坏保护及配置/交付6项，全部通过，增加主10次合成更新至99/子8；未重跑双波或全套。首轮13个未受影响模型项、单项修复及其余已通过检查继续适用，39项不同用例累计46次执行，原失败日志保留。

<a id="m6-timemixer-v5-implementation-r1"></a>
#### v5：Crossformer分段变量交互U＋ModernTCN潜在时间细化V（2026-10-19，联合送审）

**原论文来源补齐增量（同日）**：本轮只使用用户原件`/public/home/yueweiting/大论文/paper/crossformer.pdf`，SHA-256 `a5fcad16cd810492124af3efad2fa18f7c5f9556fb303bb88360cceaf396183a`、18,941,392 bytes、21页，第一页标识ICLR 2023 conference paper。ChatGPT实际读取并对照、U无需结构返工及原v5工程review Passed由用户明确交接；Codex本轮另核对本地SHA/页数并提取文本、读取§3.1式1/2、§3.2式3/4/图2、§3.3、D.1式9、D.2式10/Table7。全文侧文件与阅读对照保存`M6-Timemixer-v5-implementation-r1/source-completion-r1/`，不联网、不换论文、不动原件。Crossformer.paper更新真实path/SHA/pages和`verified_method_read`，原acquisition_ref保持HTTP403历史身份，响应/下载记录/旧source-material-status和全部失败成本保留原字节。

原DSW分段变量轴与左复制补齐保留；删除原位置嵌入、TSA时间阶段/router/完整残差归一化FFN、层级encoder-decoder、head/trainer。现有形态＋水平统计、target/aux或M自排除、null、32→3D恢复、最细embedding接入均是项目改造，不称Crossformer/TSA/router完整复现或线性变量复杂度。同段交互不独立覆盖所有跨段时滞；交给TimeMixer/PDM/V是项目设计，非原TSA数学等价替代。原D.2确有协变量实验，但ETTh1四种日历、五种预测长度范围不能外推UrbanEV价格/天气。无需修改U结构、插入点或初始化，也不新增未来协变量/位置表/时间attention。model.py/modules.py/source.py及真实材料检查原SHA保持，ModernTCN和TimeMixer绑定不变。

本轮只改source_binding、resolved执行SHA、delivery测试和两份权威文档；resolved除source_binding.json摘要外逐字段不变，保存配置与实际28项source一致。77任务/24-24-29、37B、8同v5 F4、训练/data/结构/gate政策、首批24ID原顺序/6波不变。4项delivery＋6项来源/元数据共10项通过，无失败/skip；真实两篇论文材料正常通过，mock Pending/缺path/缺sha/错SHA与旧grant/未审核grant仍拒绝，纯元数据fixture不触发worker/supervise/GPU。原38项计算逻辑证据复用，未把新绑定写成重新执行38项；旧主251/子24、初始主超2成本不改。本轮forward/backward/optimizer创建/Adam/真实train-validation-test/GPU/checkpoint加载全部0。

新template/commands/operations在`source-completion-r1/launch-preparation`，旧launch-preparation原字节不动。template仅同步source_binding/config_sha256，task/policy字节一致、commit=null/review_approved=false/默认不恢复；notice为严格两字段合同保留旧草案原文，新operations明确其缺论文句子是历史状态，不能将其当当前缺口或训练许可。无真实attempt/grant或真实preflight Passed。最终相对HEAD联合24路径与本轮5路径增量统一送审，仍unstaged/index空；不add/commit/push/merge、不重新开发模型/接线、不进入真实训练。当前最终来源绑定新字节待ChatGPT审核，v5效果Not evaluated，不能提前声称ChatGPT已读取新输出。

**版本/来源**：独立候选`timemixer-uv-dev-v5`，基于HEAD `831f6192ff48ecd84edaf2302496f48edf941109`的新未提交实现，不把基点说成已包含v5。旧v1–v4源码/入口/配置/测试和真实产物均冻结。TimeMixer原`e24610583b36fdd8c76cc17a8df4e65759a5f460`、Model SHA `817d62f4aac54c8566e560f6d3785856e31c8ee51460279ee3d0a4823f11d4be`不变。Crossformer固定`c10c8eadb153d1dd9798250967747ca3ebb81383`、Apache-2.0；ModernTCN固定`56a9a2c018385cd5acef015378cae7f084d1b11c`、MIT。实际clean checkout在外置`M6-Timemixer-v5-references-r1`，只读参考，不运行原训练器/安装依赖/更换来源。各关键代码、依赖与LICENSE实际SHA见source-audit和v5 source_binding。

**论文真实状态**：Crossformer官方PDF一次HTTP403（约0.817秒），原响应存bin，未伪造PDF/阅读Passed。按用户明确例外继续独立冻结公式实现和CPU验收；真实训练前必须补齐该来源阅读，`verify_training_materials`即使收到其他字段有效的未来grant也拒绝当前缺口。ModernTCN复用服务器43页正式PDF，SHA `2e1681606501b715185e9ffa78dbda9ba2e829799e1ed4cea29fa5c1f8f13944`，原件/副本一致并提取可读文本，重点核对3.1、3.2及附录G。SOFTS/APT/Pathformer仅保留历史身份；本版不是复活PMCR/P2/THLS。

**U精确实施**：p=stride=3采样点，pad=(-L)%3，左侧复制首值到3倍数，raw_patch[b,N,C,3]；mu、sigma=sqrt(var+1e-5)、z按段/变量求。`e=LN32(Linear3→32(z))+Linear2→32([mu,log1p(sigma)])`。Q/K/V为各自有bias32→32，4head×8；同一段目标query与合法辅助K/V交互，MS ordered aux排除target，M所有接收变量排自身，固定score0/value0的null沿source轴stable softmax。context[b,N,R,32]经无bias、零初始化32→3D bridge恢复[b,R,3N,D]再裁左补，得到[b,R,L,D]。MS加A[:,0]到最细embedding，M以b-major/variable-minor展平[b*C,L,D]，不按作者marks.repeat错误排列。无源严格零，不新增scaler/detach/ID/跨时间attention或逐对32维拼接。保留DSW与片段级变量交互思想，删原TSA时间阶段/router/层级/head/trainer；统计分支、mask/null、位置恢复和TimeMixer接口是本项目改造，不称完整复现或线性通道复杂度。

**V精确实施**：所有原PDM之后仅取enc_out_list[0]，LN(D)后转[B_eff,D,L]；DW7复制边界3/3与DW3复制边界1/1相加，均stride/dilation1、groups=D、padding0、无bias。转回后`G_new=G+Linear2D→D(GELU(LinearD→2D(Q)))`；末层weight/bias0，其余Xavier/零bias，LN1/0。无BN/运行统计/dropout/门/额外loss，不混有效样本。保留时间DW/特征PW及大小核思想；删stem/stages/下采样/ConvFFN2/RevIN/原head/BN/重参数合并。D是原profile latent，不是业务变量C，不全域硬编码64。

**两个内部位置与B保护**：原embedding→U只直接更新finest→原全部PDM→V只直接更新最后finest→原future_multi_mixing/趋势旁路/head/求和/一次denorm。U读取原X的独立变量视图，不改X、Normalize/pre_enc或原趋势输入，PDM对粗尺度传播允许；V自然读取受U影响后的PDM，不沿用v4只读Y0规则。off/off直接super().forecast；开启路径保留作者forecast副本，仅增加原X保存和两处插入，调用真实继承私有`_Model__multi_scale_process_inputs`，逐段diff在外置。没有旧输出Delta/kappa/原型/采用softmax或第三融合，两个跳连不保证输出/梯度/误差有界。CI1 marks repeat与S1 PDM限制未修，L1/非整除仅新模块边界验收。

**初始化/输出语义**：B seed2024，模块各自隔离CPU RNG后恢复，全关B初参/key/RNG与作者一致，同模块单臂/UV初参相同；末层零使初始预测一致，无新dropout消耗B RNG。合法首步上游零梯度不算缺陷，非退化合成4步后新通路finite且接通，未给unused作者参数造假moment。MS原生C保留、target-only loss，内部共同latent使非目标原生预测也可能改变，不保留v3/v4非目标逐位不变断言；M全部C。F1～F4 C6/8/9/11和既定特征、Urban L12/P1/H3/6/9/12、EPF96/24/target2、M原P/C/marks均保持，不额外喂H/未来marks。

**训练/身份/范围**：从审核v4 resolved SHA `3f255d04b5750f0770ef6f0321a67c809f2a07fccfe2cc3fbac9f5ef5e7c5fec`派生，新科学对照逐任务证明除id/model/candidate/revision外所有字段相等：profile/data/split/scaler/marks/L/P/H/seed/batch、fresh float32 Adam/OneCycle/MSE、validation MSE严格best/原patience等不改。v5完整配置→原loader→模型→训练/选best/早停→完整test一次→精确checksum/原子发布/汇总已用合成CSV跑通。保持先验全身份/活动Adam/RNG/进度恢复、成功不可覆盖、test不明claim拒绝自动重测、所有权拒绝无副作用、逐child/wave核账、STOP和失败停队列。技术schema不机械升号，旧v1–v4状态/grant/F4拒绝冒充新身份，未来仅fresh。

77项仍24/24/29、37格B、8格同v5 UV F4复用、1340run-epochs上限；首批按原v4真实24-ID清单逐项映射顺序，6波q4、480run-epochs/1485300updates，仅非执行准备。配置存储顺序不替代批准执行清单；不重训/重test有效B。目标一每fold四H等权、MSE/MAE各严格UV<U<B且UV<V<B；目标二各指标≥8/11域、集合可不同；目标三原两fold八格F4<F2<F1且F4<F3<F1。精确覆盖/缺项/重复/错版本/B配对拒绝，旧诊断容差不传播到exact或效果门槛。

**验收及失败复验**：最终38项通过：12模型/独立公式/接入位置/作者关闭与更新RNG exact/真实MS-M接口，12实际worker/恢复/封存与保护，10指标/科学配置，4来源/材料阻断/交付。原loader合成CSV覆盖四臂、F1～F4、H/P分离、EPF和M；合法epoch恢复的参数/optimizer/scheduler/RNG/批序/best/步数exact，完整test仅一次、保存test复用、坏身份拒绝且live不写。CPU fork/runpy实际v5双波通过，冷CLI仅元数据；未改45秒双波deadline或再次故意制造冷exec超时，不重跑历史整套。

首轮fixture的nonleaf deepcopy/缺task取值、迁移中的旧v4合成根/空负载根、测试正向身份旧字符串及准备工具错用config遍历顺序均定位并修正，原日志保留；未改模型公式、断言、skip或容差。累计合成Adam主251/子24，总275，含所有失败/复验。最初自定主160/子16预算错误地放在套件结束检查，主超2至162后发现；未倒改原计划或日志，进一步更新前登记工程复验增量主256/子24，并用每step前锁内检查，最终251/24在增量界内。该成本不是科学run或训练授权；不得声称从未超原计划。真实CSV仅stat/保存身份，真实checkpoint读取/真实train-validation-test新forward/GPU查询与负载均0，未安装升级。

**参数/资源/非执行操作**：U=3456+96D，V=4D²+15D，Urban D64实算B/U/V/UV=70885/80485/88229/97829。全部77完整profile已静态计数；M注意力[b,N,4,R,C+1]为O(N*C²)，V按B_eff/L/D，不能外推Urban或称GPU峰值实测。沿用历史A800/Urban batch128 q4及本版增量，保留首批q4候选、无强制serial/q4/梯度/makespan probe、不机械退q1；真实启动仍查UUID/余量/全局锁，60秒/最多3次/仅超时重试、STOP不变。本轮没有GPU查询。非执行模板commit=null/review_approved=false、恢复默认false；tmux优先，v5子入口，启动/日志/状态/精确24完成/逐fold汇总/STOP分开给出，真实attempt未创建，waves元数据通过不等于真实preflight许可Passed。

**送审停止点**：唯一实施根`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-Timemixer-v5-implementation-r1/`提供来源状态、作者函数diff、版本化接线分类、28项source与保存resolved、科学/资源证明、最终38项/失败/主子消费、不可执行操作、联合/增量patch及manifest。v4结果/机制在报告原声明范围被用户通知接受，不冒称ChatGPT重载checkpoint或CUDA恢复；旧目标一Not met、目标二/三未评价、单seed与test开发访问保留。v5工程自检Passed、新字节review Pending、效果Not evaluated。当前两份文档既有增量与v5统一送审、全部unstaged/index空，不add/commit/push/merge，不单独closure，不启动24/77/其余53项、新seed/候选或M7，不宣称内部结构已解决fold4或TimeMixer原无变量/时间能力。

<a id="m6-timemixer-v4-postrun-diagnosis-r1"></a>
#### v4首批完成后审计、采用机制与完整误差定位（2026-10-19，统一送审）

**身份及审计**：实际commit `831f6192ff48ecd84edaf2302496f48edf941109`、候选`timemixer-uv-dev-v4`；原许可SHA `87efe60ac37b68aced753052f20ad62634c2af89e2e530724d8674712d364212`，原resolved SHA `3f255d04b5750f0770ef6f0321a67c809f2a07fccfe2cc3fbac9f5ef5e7c5fec`。独立复算24/24、6波、310 run-epochs/963215 updates，19项合法patience早停、5项20轮上限。实际655文件的Python/系统checksum、552内部payload及48份best/last CPU状态核对通过，含活动Adam、scheduler/LR/beta1、严格validation MSE best、journal及RNG格式；无optimizer构造、CUDA恢复或训练重演。owner及末波worker退出，历史progress RUNNING未改；早波PID历史未保留，不冒称重建全部OS执行轨迹。旧B/v1/v2/v3审计复用，只核对本轮结构化引用，不读旧checkpoint。

**完整结果**：原GPU保存指标按每fold四H等权复算。fold1 U/V/UV MSE约0.771090/0.733788/0.726149，MAE约0.637115/0.637120/0.639290；只MSE组合条件成立。fold4 B MAE约0.470759，U/V/UV约0.474066/0.485583/0.495107，三臂MSE均低于B但UV不如两单臂。目标一仍Not met，不能用接近或CPU/GPU诊断容差判效果通过；目标二方案A、目标三八格/两条路径/同版F4复用保留且Not evaluated。全部B/v1/v2/v3/v4逐H、validation/test、best/stop/LR记录见外置完整表。

**本轮新增访问与限定数值scope**：仅24份v4 best各一次完整validation/test，共48过程；沿用原batch128、CPU float32、线程1、eval/inference_mode、Python -B、CUDA不可见。`v4_cpu_gpu_fullsplit_diagnostic_r1`由本轮用户明确采用：`abs(cpu-saved)<=1e-6+1e-5*abs(saved)`，仅MSE/MAE及同elements归一后的SSE/SAE、finite/equal_nan=false，48项全部通过，最大MSE差约8.55601e-6。身份/数据/窗口/标签/节点/scaler/marks/尾batch必须精确；常规准入表、关闭/恢复exact与科学门槛不改。只复用D3索引B/v3数组，首个旧B跨目录引用保留；CPU分组一致使用CPU预测，不宣称每条或每组与GPU等价。

实际16938完整batch/2164800呈现，额外仅fold1/H3各4条validation的观测/真实开关检查11次forward/44呈现；总16949/2164844。新增完整test24次、validation24次，原每run正式test_calls=1不改。审计checkpoint48读＋推理best24读＝72次本批CPU读取；loader8次/CSV前缀解析32次，时间行58368、其中test前缀5856，分别记账。旧模型前向、新train/backward/Adam/GPU均0；推理墙钟约487.742秒，不冒称CPU core-seconds。中间量仅截取原固定validation交集，24组各256条，共6144，随正常前向取得，无新小窗口套件。

**U采用改变了什么**：fold4最低历史mean四分位条件ΔMAE（对B）由v3+0.074304降到v4+0.003125；第二四分位仍+0.023134，最低std四分位已改善-0.010885，近常数组代价只贡献全体MAE约+0.000164，不能概括为全部低状态退化或单一近常数原因。U的fold4正式MAE由v3约0.504957降至0.474066，但fold1由0.615971升至0.637115。alpha0随样本明显变化（fold4 test均值约0.72310、cell内std均值0.29518），并非门未工作。v3/v4的own-Y0偏差及修正方向不同，不能等同于对同一提案固定缩放；own-Y0较差只支持共训分工，不直接证明主干损坏。

**V和UV的剩余问题**：fold4 V加权修正均值从validation约+0.041468到test+0.141363，最终bias约+0.104810；预定均匀采用在validation劣于正常、test优于正常但仍未胜B，支持选择关系有split依赖，不授权据test固定0.5。UV test平均加权U/V约-0.183505/+0.338558，净+0.155053；alpha0约0.10050。正常MAE约0.495107，合法offU约0.742569、offV约0.537241；固定原权重仅删U约0.551111，与合法offU不能混称。即时依赖与独立训练收益不同，不能把UV-offV称独立U。54/22分项贡献和`alpha*(proposal-c)`灵敏度已保存，未将参数范数、权重或局部导数当因果/完整梯度证明。

**范围限制与完整状态定位**：fold4 test标签落入已学提案区间的比例U/V/UV约40.22%/30.96%/65.18%，区间外距离仅作事后几何，未构造oracle新成绩，不证明整个模型类不可能拟合。null是共训Y0，不是历史独立B，有限logit也非硬关闭。完整35896行边际覆盖预定历史mean/std组、近常数、小时/周末、全部节点和连续日期；不做全部交叉或按误差选组。固定validation mean权重下，U/V/UV test ΔMAE约-0.006599/-0.003370/+0.006339，对应实际+0.003306/+0.014825/+0.024348，组成变化和组内变化并存。节点组成恒定、小时标准化几乎不消除反转，不能解释成已解决区域机制。B低误差、真实零标签及凸包成员是事后描述，不可直接作部署条件；未拟合任何test偏置/阈值/权重/kappa。

**改造判断与边界**：本轮没有足够依据继续默认增加采用门、温度或kappa小修。U的目标条件化辅助选择思路仍有信号，但需由ChatGPT判断表示/接入；V历史时间—状态条件化有作用，却未稳定解决MAE迁移，组合分工亦需重审。新模块不再限定残差或最终输出层，可以讨论保留原变量轴的表示、联合embedding/PDM或预测映射之前的功能接口；TimeMixer backbone、MS/M及F1～F4真实输入、第二轮协议不变，latent D不得当原变量轴。保留来源改接口、只换一个或重新设计两者由ChatGPT综合决定；Codex不选论文/仓库、不冻结新公式/插入点、不实现下一版。freeze/stop-gradient/主干loss或LR/loss/patience/seed/数据/选点变化是另行科学取舍。原样保留失败单臂不能保证仅换另一模块达到完整目标一。

**证据/停止点**：唯一新增外置根`/public/home/yueweiting/大论文/amd-execution-evidence/m6/timemixer-uv-v4-development-runs/urbanev-f4-f1f4-uv-v4-first-batch-v1/first-batch-review-diagnosis-r1/`，含REPORT、完整数组/记录引用、分组/采用/区间/旁路、10＋5项工具自检、真实计数及SHA。fixture首次失败和复验保留，无删断言/skip/容差放宽、无推理重试。技术完成与artifact自检Passed；本轮ChatGPT result/机制review Pending；目标一Not met、目标二/三未评价。本轮仅两份权威文档及外置增量、unstaged/index空，不单独closure、不训练/v5/新来源/其余53项或M7。旧实现送审/未训练/Pending、失败结果与原运行身份保留原时点。

#### v4：U/V输出采用与固定不修正选项竞争（2026-10-18）

**现场与科学决定。** 工作区/分支不变、起点及三端为`33e92b150a439ad5dd3d3298e0244835afa9ccd5`，原仅canonical/M6文档unstaged、index空。用户本轮接受ChatGPT新结构并授权实现、完整接线、CPU合成与非执行材料；没有v4真实训练、旧权重微调、新来源或参数搜索授权。此前64过程完整split的声明范围内方法与主要结论获ChatGPT内容接受，非独立重算全部SHA/网络；本轮核对实际引用manifest `bbe1a5af64fdf332248777e733c5680f393340076bb9798c8168e89c1546a6e3`而不重推64项。旧1e-6 false、专用v2跨设备政策/额外32 test访问、v1/v2/v3目标一Not met及负向证据原样保留。v4不采用near-constant硬mask，不预称已解决fold4。

**模型实现。** 作者B完整一次前向/一次denorm得到原Y0；v3 U/V提案、8×16原型、kappa=1、最终投影零初始化及独立CPU RNG不变。新增U Linear54(55参数)、V Linear22(23参数)，均零初始化。每预测位置的5维历史/原Y0状态为`[mu,log1p(sigma),last-mu,mad,Y0-mu]`，U拼e32/core16/状态5/u1，V拼context16/状态5/v1。代码以同一Linear的静态52/20维投影＋两标量广播实现，不物化[b,R,P,54/22]。V提案和分数不读U；U分数允许原Y0，U提案仍不读Y0。无detach/freeze/辅助loss/新数据/未来输入或新可学习融合模块。

固定null logit=0；稳定softmax仅在启用有效的null/U/V选择上归一化，`Yhat=Y0+alpha_U*u+alpha_V*v`。无合法来源U严格从分母排除，F1日历来源不漏；MS按显式target/aux、M排自身value，非目标C输出不改。V自身不随U开关变、竞争权重可变；off/off原B直接路径。初始提案0、单臂1/2/双臂1/3，合法延迟梯度；直接修正对own-Y0绝对值≤1，不推出对历史B/误差/梯度的保证或可靠性概率。

**身份、来源与接线。** 独立v4包、entry/start脚本、resolved及5份测试/fixture，不在旧扫描目录加文件；v1/v2/v3全套保持。SOFTS/MIT、APT固定版本无明确许可、TimeMixer/Apache-2.0和已有论文原绑定复用，17项关键作者源码校验，0下载/新选源/APT代码移植。分数/竞争是项目改造，不声称原APT机制或凸组合原创。28项执行SHA覆盖实际依赖，底层loader/Adam/OneCycle/BestState/checkpoint/artifact保护只读复用；命名硬绑定建立版本化副本，真实逻辑差异集中在modules/model/spec及配置派生。

从原v3 resolved SHA `c8428a04a4c31d13d73aa154ebb8f0beaff9e49be7e6301636516bf22a31c357`逐任务派生/保存v4；77项训练/数据/结构来源、scaler/marks/node/L/P/H/seed/窗口算术完全一致。仍第二轮fresh2024、MSE loss/validation严格best、原OneCycle/patience/batch；不改历史训练协议字符串。37格B文本result SHA及15个数据路径stat核对，无真实CSV/旧真实checkpoint读取。24/24/29、8格同v4 F4、1340 run-epochs保留；v3 F4、checkpoint、grant和registry均不能冒作v4。

真实worker合成CSV已验证配置→原loader/scaler/window→v4 train/validation/早停→best完整test→checksum/原子封存/汇总；epoch恢复前完整身份/活动Adam/RNG/进度、无副作用拒绝所有权、成功不可覆盖、不明test拒绝重测、child/wave精确核账、STOP/失败停后续仍在。目标一逐fold双指标、目标二方案A独立≥8/11、目标三两fold八格/F4同版复用和严格未舍入比较均不改。

**本轮验收与成本。** 40/40 CPU测试，模型/公式14、worker/恢复12、指标/配置10、交付4；0 errors/failures/skips、无非预期失败或复验，预定故障注入不删。主进程81次成功Adam（含class fixture2）、fork子进程8次，预登记上限100/8；所有验证/恢复exact及测试断言保持，未借专用诊断界放宽。涵盖54/22独立float64公式、长P实际静态输入形状、无效集合/极端logit、MS非0/M/L1/marks、独立V和竞争、初始零与更新后各新增通路、公共B/dropout RNG、真实CSV四臂/F1–F4/EPF/M、合法恢复/保护负例、双指标/同版覆盖。CPU双波仅fork/runpy fixture实测、45秒边界不改；冷CLI只waves元数据，不伪称冷exec训练或GPU验证。旧39/32/281套件未重跑；真实train/validation/test新评价与GPU查询/负载/CUDA初始化均0。

交付期外置diff检查曾误认no-index正常差异退出1，并查出测试EOF多一空行；修正检查语义/删该空行后，通过AST精确等同、语法和patch复核，未改断言或模型、未增加Adam。首次静态检查与前后SHA另存，不混作模型40项失败或复验。

Urban完整profile参数B/U/V/UV实算`70885/75773/71598/76486`；77项对应23种结构构造核算，资源核算forward0。U长M的O(C²)条件张量最大约7,225,344 bytes，V[b,R,8]及输出[b,R,P]，长P避免54/22大拼接；静态估计不等于GPU峰值。首批不可执行清单按已审v3顺序映射24个新ID、6波q4、480 run-epochs/1,485,300 updates，其余53项保留不执行。沿用A800/Urban batch128 q4历史依据提出本批计划，无强制serial/q4/makespan probe；管理查询60秒/最多3次/仅超时重试及UUID/余量/锁/STOP不变，真实启动须现场复查。

**统一送审/停止。** 外置根`amd-execution-evidence/m6/M6-Timemixer-v4-implementation-r1/`集中来源分类/公式/科学对照、参数、40项日志/消耗、完整与增量patch/manifest及分开的start/log/status/complete/summary/STOP/受限恢复操作说明。template为`commit=null`、`review_approved=false`，无真实attempt/grant/自动恢复。工程自检Passed、ChatGPT实施新字节review Pending、Git closure未执行、真实训练未执行、v4效果Not evaluated。此前两份文档与本轮24路径联合unstaged送审、index空；不单独closure文档、不启动其他任务/fold/seed或M7。旧产物和64项诊断输出保持原字节，不将内容接受写成ChatGPT独立重验全包SHA。

<a id="m6-timemixer-v3-full-split-error-continuation-r1"></a>
#### 诊断专用跨设备政策v2、剩余63项及完整误差定位（2026-10-18）

**授权/旧证据。** 依据用户本轮通知，ChatGPT已直接核验旧停止报告、48证据、39登记来源、原算术/数组/Git/文档SHA并接受停止与访问账；原1e-6检查仍为false，此前只完成1项。本次是用户观察超界后的明确新授权，仅固定32 best/64 model-split采用`cpu_gpu_fullsplit_diagnostic_v2`：`abs(cpu-saved)<=1e-6+1e-5*abs(saved)`，equal_nan=false，finite及checkpoint/profile/source/窗口/标签/节点/通道/scaler/marks/eval batch/尾batch精确要求不变。SSE/SAE除同elements后应用，不改全局工程表、关闭/恢复exact或科学训练/效果政策。新政策、旧plan引用及全部新材料只在`full-split-error-review-r1/continuation-r1/`。

**精确续接/代价。** 核对旧B fold1/H3 validation的数组SHA `6fd478614d4cb52bcc2dfdaecd8da77efe7c0ce6bd0b0169d9fd7960faaccf09`、完整15950 cursor/125batch/尾78、原best和记录；只复算新政策判断，不重推。用旧64key求差得到31validation+32test共63，实际新增22459batch/2870450呈现；累计validation32/test32、22584batch/2886400呈现。新增best加载32、累计33（首个B重载只接test），非重复原validation。累计18次loader/72次CSV前缀解析/122496时间记录，其中test时段12288条；数据解析和32次额外完整test推理分别记账。推理665.421秒，CPU/eval/inference_mode/线程1/batch128/Python -B，无train forward/backward/optimizer/Adam/CUDA/GPU查询。旧51文件/原false/失败日志保持原字节；新过程一次完成，无重试择优。

**来源/数值。** HEAD及训练commit仍`33e92b150a439ad5dd3d3298e0244835afa9ccd5`；v3 resolved `c8428a04a4c31d13d73aa154ebb8f0beaff9e49be7e6301636516bf22a31c357`、28项执行绑定及作者来源/数据stat不变，B为原第二轮8个正式best。全64项新政策通过，最大MSE差`6.079013415982715e-6`（B fold1/H9 test，允许`1.1644625553792394e-5`），最大MAE差`3.0288857740812247e-6`。24个fold/split/臂/指标相对B方向与原GPU一致且超过实际总体数值差异尺度。新CPU分组只与同批CPU B配对，不能用总体界证明逐样本/每组GPU等价；正式成绩和科学gate仍取未舍入原GPU。

**完整相对反转。** 下表为CPU逐H完整split后四H等权，非跨H池化；原GPU逐H/均值与CPU64项对照完整保存。

| fold4臂 | validation MAE | test MAE | validation对B差 | test对B差 | test有符号bias |
|---|---:|---:|---:|---:|---:|
| B | 0.609781 | 0.470759 | 0 | 0 | -0.032381 |
| U | 0.575326 | 0.504957 | -0.034455 | +0.034198 | +0.151038 |
| V | 0.571123 | 0.484500 | -0.038658 | +0.013740 | +0.081341 |
| UV | 0.574737 | 0.497267 | -0.035044 | +0.026508 | +0.114654 |

所有臂test绝对MAE比自身validation低，但B降幅更大；反转指相对收益，不能说test整体更难。原GPU四H×三臂的12项test MAE仍全部未胜B。fold1两split仍有相对B收益，组合未超过独立U。目标一逐fold双指标Not met，目标二/三Not evaluated。

**历史状态定位。** 固定完整train合法窗口阈值，fold4mean最低四分位占比20.74%→32.90%、最高组29.30%→13.47%；std最低组23.35%→41.57%、最高组27.12%→10.68%；near-constant（历史方差<=1e-8）2.57%→5.65%。最低两mean组test对ΔMAE贡献U/V/UV为+0.03973/+0.02355/+0.03343，最高mean组仍有负贡献收益。全量15528行边际组统计保留小时/周末/mean/std四分位/近常数、全部275节点和连续日期，并单列事后组；无按误差选样、无全部笛卡尔积。test正MAE贡献小时/节点/日期分别为U21/171/11、V16/150/8、UV18/170/9，故不是单一H或少数位置的普遍解释；节点定位不证明空间机制。

**组成与组内分解。** mean分组的val→test相对MAE变化，U/V/UV组内项+0.04669/+0.03432/+0.04222，组成项+0.02197/+0.01808/+0.01933。按validation占比标准化test差为+0.01223/−0.00434/+0.00718；V的mean状态占比变化对跨零点重要，不能称固定占比后三臂仍全部失败。小时/周末组成项很小或有利，不能仅用日历混合解释；node组成0是全节点等权设计结果。以上都是共同支持上的边际描述，不称因果分解或新正式成绩，fold1不完整支持显式报coverage。不同分组贡献不能跨分组相加。

**支路与共训分工。** fold4 test中U/V/UV各自own-Y0 MAE约0.515903/0.500568/0.517207；最终修正仍降低其整体MAE约0.01095/0.01607/0.01994，但远少于validation约0.12408/0.09804/0.12926。own-Y0与独立B的差异是不同共训分工，不是主干损坏的证明。实际float32加法重构精确，float64分解舍入剩余另报。mean最低两组中，残差在test对自身Y0增加MAE；near-constant组own-Y0与B贡献近0，而残差在validation/test都增加MAE，test对全体贡献+0.01011/+0.00482/+0.00644。V/UV整体MSE仍受益于误差方差下降，bias平方代价同时增加；U的bias代价已超过方差收益，三臂test绝对误差恶化样本比例约55.03%/53.22%/54.35%。不从2uv或相消单独判因果/组合优劣。

**事后标签/误差组。** 原每node scaler物理零点匹配的标签比例4.43%→7.25%，test零点组对ΔMAE贡献+0.02549/+0.01627/+0.02014，非零组+0.00871/−0.00253/+0.00637。不是把标准化y<=0当无充电，也不能用未来零标签做门。旧B_typical准确解释为事后B低误差组，即便阈值来自train，成员仍依赖未来真值，不能当作推理状态或残差关闭条件；原统计原字节保留。

**待裁决而非v4。** 最小待审方向仅为历史可见的增量风险控制，near-constant输出保护有直接作用证据；未实现、未测试新后处理、未生成v4配置/许可。它只覆盖5.65%，非近常数组仍有MAE代价约+0.02410/+0.00892/+0.02009，不能保证完整目标一，更不能原样保留失败单臂后只换另一个。硬旁路可能丢掉平稳历史之后由外生信息预示的变化；任何新结构须ChatGPT综合选择并完整训练验证，不能用test调偏置/阈值或改loss/LR/patience冒称模块收益。新U2/V2论文和结构选择仍回ChatGPT，Codex未选替代来源。

**工程与状态。** 新6项政策/续接及4项统计纯合成测试Passed、旧7项复用；没有模型套件重跑/合成Adam。两次临时只读索引列名检查错误已记录并修正，不增加模型前向；统计和图表只读导出数组。原训练产物技术完成及既有审计继续有效，本轮64项诊断数值/统计自检完成、ChatGPT新输出review Pending、效果仍Not met。本轮所有新材料和两份文档全量/增量patch统一送审；只有canonical/M6两份项目文档unstaged、index空，不commit/push/merge或单独closure，不执行其余53项/新seed/fold/M7。

<a id="m6-timemixer-v3-full-split-error-review-r1"></a>
#### 完整validation/test逐样本定位与数值停止回执（2026-10-18）

**授权/来源。** 本轮允许24个v3 best及第二轮严格配对8个B best各一次完整validation/test，只读CPU，不更新/重新选点/重新封存。三端训练HEAD `33e92b150a439ad5dd3d3298e0244835afa9ccd5`、0/0及仅两份已知文档modified/index空核对一致；原resolved `c8428a04a4c31d13d73aa154ebb8f0beaff9e49be7e6301636516bf22a31c357`和32个best文件SHA、来源引用通过，未重复旧完整审计。新根为v3结果根下`full-split-error-review-r1/`，不写原attempt或旧诊断。ChatGPT此前读过报告/相关文档，远程工具未完成独立SHA/Git复核，不记录为完整字节审核/closure批准。

**预登记与工具。** 64过程=validation32+test32，2886400样本呈现/22584完整batch，batch128、float32、eval/inference_mode、CPU线程1、CUDA不可见。固定forecast-origin小时/工作日周末、train历史mean/std四分位、近常数方差<=1e-8、全部节点/连续日期边际分组；标签物理零点按每node train scaler正确还原，不能用标准化y<=0代替。7项纯合成工具检查Passed，覆盖键/标签错配拒绝、尾batch/游标续接、scaler零点、组贡献还原、四H等权/缺项拒绝、原文件不写入及误差算术。第二轮原loader/adapter/marks固定文件在外置目录保存，当前新增resolved-profile/分钟支持不影响本格K4小时路径；完整x/y/marks、metadata/scaler/窗口/节点精确对应。初版工具误要求两历史wrapper整文件同SHA、后续脚本语法错误均保留，修工具后才开始一次真实推理；未改项目源码或断言界限。

**实际唯一评价。** B fold1/H3 best为原第二轮正式来源，SHA `8078e0429edac5ca2e833de42a9663a0ac9e0f5fcd76cb97269dd80fc65302a6`；CPU strict加载原作者Model，完整validation 15950元素/125batch/尾78。保存MSE `0.5888678177135626`、CPU `0.5888698366130992`，绝对差 `2.0188995366599016e-6`；保存MAE `0.5326005399124084`、CPU `0.532600610864883`，差 `7.095247456234688e-8`。事前登记的归一化metric界1e-6/rtol0在MSE超界，停止后续63过程，不重复本split，不加界/改精度/换checkpoint。CPU/GPU口径适用性与差异原因仍待ChatGPT判断，不能直接认定原B损坏/技术失败；本轮fold4反转机制及组内/组成分解Not evaluated。

**访问与成本。** 32份best做来源SHA核对，实际CPU反序列化1份；模型完整validation1/test0、125forward/15950呈现，train forward/backward/optimizer/Adam/GPU均0。不能概称“无test访问”：对齐检查的两个loader各读至720行，含test时间区间[648,720)，4份CSV各两次前缀解析；未计算任何test模型预测。保存独立访问账、完整validation预测与共同validation/test键/历史统计；原test claim、test_calls、metrics、registry、许可与所有artifact不写入。7项工具自检有效，原模型套件未重跑。

**解释与下一步。** own-Y0相对独立B变化支持共训分工，不能直接称主干破坏或推出freeze/stop-gradient/辅助loss。旧B_typical按abs(B-y)定义，准确称事后B低误差组，成员依赖未来真值，不能当作预测可见状态或残差关闭条件；原统计保留。缺少完整split可采信导出前不分析fold4反转，也不给v4处方。先返回ChatGPT裁决CPU/GPU数值核验及必要受限定位，之后仅按获准口径续接，不因绘图/分析失败重跑已有预测。原24项技术完成与目标一Not met、目标二/三Not evaluated均不变；本轮诊断Blocked at numeric consistency，ChatGPT新字节/结果review Pending。全部文档增量unstaged、index空，不单独closure、不训练、不实现新候选/选源。

<a id="m6-timemixer-v3-postrun-diagnosis-r1"></a>
#### v3首批完成后审计、跨版本机制对照及可改造性（2026-10-18）

**原身份及审计。** 用户批准的v3首批UrbanEV-F4 fold1/4×四H×U/V/UV共24项/6波已完成，训练及当前commit `33e92b150a439ad5dd3d3298e0244835afa9ccd5`；合法许可为`M6-Timemixer-v3-implementation-r1/launch-authorized-r1/launch-binding.json`，SHA `30132c7ebc5f9707d1b3db5b0fe839393d231f8963d0fc1865535e989bad1382`。本轮先在clean训练字节上只读审计，再诊断，最后才更新两份文档。655文件全成员Python/系统checksum、552发布payload、24任务/回执/注册表/日志、6波及原source/config/data/environment/producer匹配；staging空、无failure/STOP、owner和末波4个进程已退出。原progress最后RUNNING保留快照。此前波次PID记录未完整保留，以精确回执/child输出/runtime/complete核账，不伪造旧PID退出证据。

本批48份best/last CPU读取完成：身份、key/shape/dtype/finite、活动Adam集合/moment/step、scheduler/LR/beta1、BestState、epoch/update/journal前缀与RNG格式均通过；不是CUDA恢复/轨迹重演。逐931047条update核验OneCycle推进，318 run-epochs、严格validation MSE best、原patience/20轮上限、每项一次保存的完整test/尾batch/SSE/SAE/元素数一致。零初始化支路合法保存，未要求作者unused参数有假moment。旧v1/v2审计及第二轮8格B引用复用，只核对实际引用及相同profile/scaler/marks/test_data_sha/元素数，不加载其checkpoint、不重新test。没有已保存的逐样本test预测/残差。本轮首次外置审计工具将无扩展名registry当Path导致失败，实际JSON回执相等；修工具读取并复验通过，失败日志/首版工具保留，未修改模型或artifact。

**完整结果（train-standardized，每fold四H等权）。** 使用未舍入值判断；B/v1/v2/v3共80行逐H、20行均值、前版与B变化、validation/test选点对照在外置JSON/CSV。

|fold|模型|MSE|MAE|
|---|---|---:|---:|
|1|B|0.8799985234060477|0.6812829584677624|
|1|v3 U|0.6955358649384490|0.6159710746769994|
|1|v3 V|0.7210320927999367|0.6315038607143635|
|1|v3 UV|0.7189510727369660|0.6330696495845880|
|4|B|0.4452133901119243|0.47075887906978064|
|4|v3 U|0.4490491604843632|0.5049574194262557|
|4|v3 V|0.42669823023309217|0.48449937176648356|
|4|v3 UV|0.44078735464061325|0.49726755742833556|

目标一每fold双指标分别`UV<U<B`且`UV<V<B`：Not met。U fold1有效，fold4均值双指标未胜B；V MSE比旧V改善但fold4 MAE仍失败；UV fold1不如U、fold4不如V。fold4四H的U/V/UV共12项test MAE全部未优于B，不能笼统说所有单格MSE无效，也不能跨fold抵消。目标二仍双指标方案A、目标三仍两fold八格与同版F4复用，均Not evaluated；不运行其余53项/其他fold/seed。

**固定机制样本及无侵入性。** 原sample-manifest SHA `b0d0aa8a6c49846b1a0707d6b2b60181f56bba08c8f8504a79e897a9e05faf74`，16组×256=4096记录；直接复用旧输入/marks/标签、B/v1/v2 trace，逐SHA、flat index、scaler与数据前缀核对。CPU线程1/batch64/eval/inference_mode，只读本批24 best。192正常+48观测+40真实开关=280次完整forward/12640呈现；24模型预测、输入、参数、持久buffer、RNG和开关恢复精确检查通过。新旁路均按原float32加法顺序代数重构，不机械重跑旧8路径。局部来源屏蔽和原型/仿射干预768调用/49152呈现，正常公式核对256/16384另记；不加扫描。

**主干和残差必须分开。** `Yhat-B=(Y0-B)+Delta_U+Delta_V`逐样本保存；Y0为该增强checkpoint共训主干，不是独立B，offUV不等于B。固定validation四H平均MSE：fold1 B0.82850，U/V/UV自己的Y0为0.88891/0.88878/0.98851，正常为0.69481/0.69415/0.69938；fold4 B0.91726，own-Y0为1.02026/1.02386/1.06980，正常为0.71049/0.67821/0.68446。支路不是没作用，而是在明显补偿own-Y0；跨checkpoint差异仍不证明训练某一步的因果。v3残差已位于原head/denorm之后，不再用旧趋势head放大解释Delta。

**U的新观察。** U-only validation残差RMS约0.530/0.532，跨样本std约0.523/0.514；形态/统计表示量级相近，alpha真实变化，非普遍固定偏移。|Delta|≥.95约8.6%/8.5%，UV约3.8%/0.3%，不支持全体饱和。屏蔽日历使U-only validation MSE增加约0.101/0.176；价格/天气局部有利/不利不一致，权重不是变量因果或正式F消融。按2/3/5通道数同时给每通道和组总贡献，null/gate/core/dU/Delta、低波动/近常数历史状态及过冲均保存；mu/sigma属于既有预处理输入。

**V的新观察。** 原型pi跨样本std均值约0.079–0.087，原型中心化有效秩约6.26–6.61；固定同checkpoint train均值pi令validation MSE增加约0.087–0.341，动态状态被实际利用。幅度项/偏移项同时有用，去偏移项损害更大；相消约27%–35%。固定X/M的解析`J=1+aV*(1-tanh(dV)^2)`在4096条V/UV validation记录范围[-0.192,1.881]，6条为负、|J|>2为0；未支持广泛失控，且非全模型Jacobian/训练梯度验证。不能用旧尺度pi/P/Q解释本版原型。

**组合与误差权衡。** UV的validation `2uv`均值约+0.0858/+0.0939，有重叠成本；同checkpoint组合仍优于任一旁路，相消约17.4%/13.1%，非“相消即失败”。旧B train边界定义的validation典型组MAE：fold1 B0.1796→U0.3432/V0.4031/UV0.4080；fold4 B0.1824→0.3044/0.3071/0.3254；大误差组两fold均明显下降。共同主干在典型组已变差，不能只怪最终残差。完整fold4的12项validation MSE/MAE全胜B却没保持到test MAE，best为7项OneCycle上升/17下降，8项到20轮；不认定合法早停为技术失败，不改loss/LR/patience或用test选点。局部样本排序不同于完整split，不能声称解释了未观察的test尾部。

**可行性判断及停止点。** U选择与V状态原型均有实效，但现有两个单臂都未达fold4 MAE要求，不能原样保留一个并保证只换另一个即可通过目标一。当前证据不足以冻结单点v4公式，不机械调整kappa/温度/原型数、屏蔽某组、加融合门/正交loss或冻结B。交回ChatGPT重审辅助增量、典型误差保护和共同适配分工；替代需求仍为history-only X/M、MS显式target/M全通道、F1–F4同规则、独立开关及第二轮纯模块比较。Codex没有选新论文/来源/名称或实现新候选，训练政策改变另行审批。

**送审与保护。** 唯一外置根：`amd-execution-evidence/m6/timemixer-uv-v3-development-runs/urbanev-f4-f1f4-uv-v3-first-batch-v1/first-batch-review-diagnosis-r1/`，核心为REPORT、audit/binary-audit、全量对照、逐样本trace/旁路/局部干预、假设表/可行性决定、3图、消费账及manifest/patch。最终13项工具自检通过（此前12项/补2项保留，共27执行）、真实24模型观测精确一致；CPU checkpoint读取72=48审计+24诊断，旧权重0。合成完整B forward0、独立支路5次/20呈现；新增test/backward/optimizer/Adam/GPU均0。v1/v2/v3源码/配置/入口/测试、原产物、共享来源及tag保持原字节；仅canonical/M6两份文档unstaged、index空。技术完成、Codex artifact自检Passed、本轮ChatGPT result/机制review Pending、目标一Not met、目标二/三Not evaluated分别登记；不commit/push、不单独closure、不自动实施或训练下一候选。

<a id="m6-timemixer-v3-implementation-r1"></a>
#### v3全链路实现、CPU验收及非执行启动材料（2026-10-17）

**继承及新身份。** 用户交接确认v2结果/机制review在原报告声明范围内接受，非效果Passed；v1/v2目标一Not met、目标二/三Not evaluated原结果与负向证据保留。本轮ChatGPT已经选定SOFTS/APT及完整设计，授权Codex一次完成来源、模型、runner和CPU验收，未授权真实训练。基点仍`553332e5cf73a66a7859f4d51b26905667d21c63`，起点预期只有此前两份权威文档unstaged/index空，未为旧文档单独closure。新`timemixer-uv-dev-v3`并非v2小修；v1/v2源码、配置、测试、许可、结果及共享作者/生产/第二轮保持原字节。

**来源核验及许可。** 新参考根`amd-execution-evidence/m6/M6-Timemixer-v3-references-r1/`，APT固定checkout `98a4c9c017666207b02029f842eff92818f0eab8`；关键APT.py SHA `7b28659d60baab8347b123e0ee20beed98617eb803c3e8bd28b6181b69f79ed9`、normal_tsf_runner.py SHA `a1232dd25ffe3a09d0ad08daeba2448172a45d406f49cf886ccc06c15353d37f`均匹配。AAAI正式PDF9页，SHA `7a82364c598d6c835cd15dd3c5df22f9e9ae54f4256881ca47a29936e8e62658`，已验证PDF、pypdf实际提取方法/公式/训练文本；paper/APT-AAAI2026.pdf为同字节副本。固定源码树与README未找到明确代码许可，不宣称MIT；未运行其脚本/安装依赖/复制APT执行代码。SOFTS原commit/关键SHA/31页PDF/MIT绑定复用；TimeMixer原commit/主文件SHA和20项作者来源绑定不变。source-audit包含STAR调用链及训练随机/推理加权差异，APT runner→ModelWithNormalization→APT、式4–14/仿射/匹配/freeze损失与本候选的逐项对照。新V不再登记Pathformer。

**落地公式和位置。** 独立包直接继承绑定原作者Model，完整作者forecast及一次denorm后得到原C通道Y0，再并行`Y0+Delta_U+Delta_V`；MS仅target，M全变量。没有旧历史改写或尺度重权，也不再分解/额外denorm。U自有无状态窗口z/mu/sigma→32维变量表示→16维接收条件masked core（null0、MS aux/M对角屏蔽）→P步门控输出→tanh，固定kappa1、Wu无bias零初始化，参数4833（Urban）。V用原登记M首尾及自身4维状态→16维query→8×16 dense prototypes→16维MLP→先aV后bV，`Delta_V=tanh(aV*(Y0-mu)+bV)`，最终Wf全零、参数690（Urban）。V独立于U，只看Y0；完整层规格与MS/M、F1–F4和来源保留/删除/新增见canonical本轮合同。没有未来终点marks/ID/新scaler、APT预训练冻结/辅助loss、第三融合层或非负裁剪。

**训练接通及独立保护。** v3配置→原真实loader/scaler/window→v3模型→Adam/OneCycle→validation严格MSE best/原patience→best完整test一次→manifest/history/metrics/校验→原子发布→汇总已接通。复用文件和命名迁移/真实逻辑差异由source-change-classification及v2-v3-source.diff逐份登记；28项实际本地执行依赖包含只读复用的底层恢复/loader/adapter/finite组件，作者文件另行SHA绑定。旧包仅只读复用技术helper，不以TimeMixer身份伪装增强模型。v3候选/revision/spec/task/config/source/runtime/checkpoint/receipt/completed registry相符；旧版本、跨臂/F/H/协议/source/data/结构、活动Adam缺失/moment/step异常先拒绝且live状态不变。所有权拒绝无副作用、child精确ID/路径/producer核账、STOP/失败停队列、test一次/不明claim停止、成功不可覆盖继续有效。

**科学配置实证。** 已保存`configs/timemixer_uv_round2_resolved_v3.json`与现场source binding、derive结果一致，逐77任务除身份四字段外所有字段与固定v2一致；structural_source_profile/baseline_training、数据/预处理/scaler/marks/L/P/H/seed和窗口/OneCycle算术没有改变。15个既定数据文件只stat，37格B result JSON SHA及原指标只读匹配，不加载旧checkpoint或真实观测。77=目标一24+目标三24+目标二29，37B重训0、8个同v3 F4复用，最大1340 run-epochs；第一批按v2实际清单逐项映射24个v3 ID、同臂同fold四H的6波，480 epochs/1485300 updates，其余53项不启动。指标政策完整继承v2逐fold目标一双硬、目标二方案A、目标三八格；精确任务覆盖和未舍入严格小于，不足范围保持Not evaluated。

**实际CPU验收。** 固定amd Python、-B、CUDA不可见/禁止bytecode、线程1；先登记200主进程+16子进程合成更新上限。首轮35项、交付元数据补4项，共39项全部通过，无skip或放宽断言；主进程实际80次、fork子进程8次Adam更新，合计88。原B关闭初始化/RNG/key/输出/loss/梯度和两次Adam/OneCycle逐项精确；两模块初始0/同子流、不额外消耗B dropout RNG、原forecast/denorm一次、X/M/cache未被支路改变。U/V独立float64公式、来源/null/receiver/原型轴及bounds，常数/近常数/大输入、缺marks/错K、非零target/M及EPF/M长P接口，零头首次梯度为0而后续通路接通均通过。真实合成CSV四臂/Urban F1–F4/H12-P1、EPF L96/P24 target2、M L96/P96全通道训练评价通过；完整epoch resume与不中断参数/optimizer/scheduler/RNG/批序/best/step精确一致，损坏状态共享fixture派生不另训。封存Python/系统checksum、test一次/不明claim、duplicate/半目录/receipt错配/跨attempt、所有权和失败停后续通过。真实双波走fork/runpy v3入口，冷CLI仅waves元数据，准确保留范围，不冒称冷exec双波/GPU训练/恢复验收。没有重跑281/32等历史套件；真实train/validation/test新评价、旧权重载入、GPU查询/负载均0。

**资源及非执行启动材料。** Urban全profile参数B/U/V/UV=70885/75718/71575/76408，匹配批准公式；U的M成对O(C²)/长P及V [b,R,8]/[b,R,P]静态张量量级已列，不称峰值实测。沿用既有A800/Urban batch128 q4证据评估6波，不恢复强制serial/q4/梯度/makespan probe，也不机械退q1。管理查询60秒/最多3次/仅超时重试、UUID/余量/全局锁不降级。v3入口/子进程及tmux优先守护脚本、精确24项技术检查、逐fold汇总/安全STOP/受限epoch恢复说明均备齐，命令分项列出。template commit为空、review_approved=false、resume默认false，未创建真实attempt/可消费许可；waves元数据通过，preflight对未审核template按预期拒绝，不能称真实授权preflight Passed。

**统一证据与状态。** 本轮全部材料在`amd-execution-evidence/m6/M6-Timemixer-v3-implementation-r1/`：来源对照、acceptance-plan/日志/主子消费、合成artifact、source/config/science proof、参数资源、launch-preparation、完整及增量patch/manifest。旧日志/机制/结果原字节保留；本节与此前两份文档增量统一送审。v3 implementation工程自检Passed；ChatGPT本轮实施/runner/文档review Pending，Git closure未执行、真实训练未执行、三个效果目标Not evaluated。全部unstaged、index空，不创建第二milestone/平行状态或修改AGENTS；停在完整实现、接线及CPU验收就绪，统一交ChatGPT审核。

<a id="m6-timemixer-v2-postrun-diagnosis-r1"></a>
#### v2首批完成后审计、v1/v2机制对照及可改造性（2026-10-17）

**原训练身份与授权。** 分支M6-Timemixer，工作区`/public/home/yueweiting/大论文/TimeMixer-U+V-v1`，设备b82673a648b4；起点三端0/0、clean，HEAD及实际v2训练commit均为`553332e5cf73a66a7859f4d51b26905667d21c63`。v2 resolved SHA为`8c774165a928cd2a06052237477c808929a77619225e884eb166020e58f0e730`。v2实现/32项CPU及精确25路径closure已获用户交接确认；用户随后明确批准首批24个fresh seed2024、6波q4、最大480 run-epochs/1485300 updates，B复用第二轮8格。实际可消费grant为`M6-Timemixer-v2-implementation-r1/launch-authorized-r1/launch-binding.json`，SHA `56d64e5cd3cff5389e1f09ad5b1a32d7cdba1178fedbc41602f4e3284089f076`；旧launch-ready-r1草案未被当作许可。本轮没有新增启动或恢复。

**统一证据位置与原样保护。** R2=`amd-execution-evidence/m6/timemixer-uv-v2-development-runs/urbanev-f4-f1f4-uv-v2-first-batch-v1`，attempt=`urbanev-f4-f1f4-v2-first-r1`；所有本轮工具/结果仅在R2/`first-batch-review-diagnosis-r1/`。v1机制evidence-manifest SHA `1d5420fddf2b2a08b6630d62958240bd7bb83d4fc7436e3bb04211ac563d92af`、固定sample-manifest SHA `b0d0aa8a6c49846b1a0707d6b2b60181f56bba08c8f8504a79e897a9e05faf74`均匹配；核对178份本次复用缓存/trace/工具引用，不重审v1全包。模型/训练器/loader/resolved/共享来源及旧许可/runtime/gate_policy、v1/v2产物保持原字节；不写入原attempt、不改生产/第二轮/主仓库或baseline tag。

**先clean审计，后诊断/文档。** 精确24 ID、24 receipts、24成功registry、6波对应；staging为空，无failure/STOP，owner及最后4worker均退出；progress最后RUNNING为历史快照，未修写。实际655个attempt文件经Python/系统SHA复核一致，其中552个run payload通过原artifact audit、24个seal/成员/claim/metric/manifest自洽。CPU读取48份本批best/last，核对identity/producer/key/shape/dtype/finite、实际活动Adam/moment/step/组、scheduler/LR/beta1、BestState、epoch/journal前缀及Python/NumPy/Torch/DataLoader generator/保存CUDA RNG元数据与格式；未创建optimizer、恢复CUDA或重演轨迹。318 run-epochs、915691更新逐条OneCycle、严格validation MSE best/合法patience、每run一次best完整test、MS目标0、SSE/SAE/elements和eval尾batch均匹配。原环境/argv/线程/配置/source/data绑定一致。v1/v2同任务profile、train/test metadata/scaler/marks/test_data_sha及元素数相同；B仅复核第二轮结构化result/history/manifest/config，不加载B checkpoint。本批没有保存逐样本test预测/残差，不补新test forward；不是全CSV重审、CUDA恢复数值Passed或独立OS I/O跟踪。

**完整效果。** 目标一训练前逐fold四H等权、MSE与MAE各需UV<U<B且UV<V<B；原值严格判断，不跨fold抵消、不要求每H排序，也不固定U/V先后。下表仅展示6位小数，逐H及validation/test/相对变化见full-per-H.csv、full-results.json和history-phase.csv；独立求和与原aggregate的math.fsum有末位浮点差异，结论一致。

| fold | 模型 | 四H平均test MSE | 四H平均test MAE |
|---|---|---:|---:|
| 1 | B | 0.879999 | 0.681283 |
| 1 | U v2 | 0.700376 | 0.614660 |
| 1 | V v2 | 0.874765 | 0.677156 |
| 1 | UV v2 | 0.707976 | 0.618337 |
| 4 | B | 0.445213 | 0.470759 |
| 4 | U v2 | 0.429769 | 0.478436 |
| 4 | V v2 | 0.450309 | 0.473690 |
| 4 | UV v2 | 0.437302 | 0.486024 |

两fold目标一均Not met。U v2较v1两个fold平均双指标均改善，fold4 MAE仍比B高1.631%；V v2 fold1改善、fold4较v1更差；UV v2两fold均不如同版U（MSE高1.085%/1.753%，MAE高0.598%/1.586%）。目标二仍方案A、目标三仍八格平均/同版F4复用，均Not evaluated；未执行53项不删除、不取消、不自动运行。v1八格启动policy、结果后逐fold口径及v2训练前逐foldpolicy分别保留。

**固定样本与真实计数。** 直接复用v1 train/validation输入/marks/标签及144组B/v1 trace，16组×256窗口共4096记录，跨H共同合法origin及顺序不变，不重读CSV或重新拟合scaler。只加载本批24个v2 best；CPU/eval/no_grad、线程1/batch64，正常/同权重旁路8路径产生128组trace，主完整forward512次/32768呈现；24模型各4个train样本的有无观测48次/192呈现，合计560次/32960呈现，约25.37秒。预测逐位一致、state hash/持久buffer/开关恢复通过，Normalize缓存未重复覆盖、最终denorm一次。另416次U/V局部诊断重算、208次V表示分解、32次原tau重构及32次固定train均值tau局部干预；不新增完整前向。工具8项+保存/局部检查5项首轮全通过，合成完整forward12次/24呈现（上限20）；实际/合成合计572次/32984呈现。CPU checkpoint加载共72次（48审计+24诊断，48个不同文件），新test/B/v1 forward、optimizer实例、backward、Adam更新、GPU查询/probe/负载均0。绘图首次在amd因缺matplotlib失败，原日志保留，使用已有/usr/bin/python3绘图环境完成；不安装依赖或重跑模型套件。

**U限幅的支持与限制。** validation实际改写RMS/自身历史RMS从v1约0.91–2.86降至v2约0.27–0.65，绝对r/s>1约占5.44%–42.09%，固定s局部敏感性中位数约0.640–0.999，限幅实际激活而非全面饱和。近常量目标每组fold1为11、fold4为9，残差RMS约0.0028–0.0032，校正被抑制但该小组v2并未稳定比v1差，暂不能据此加std下限。粗尺度输入normal/offU逐元素相同；主要输出作用仍可经原trend head。fold4/H6 U共同评价空间Q旁路变化RMS约0.768→0.868，远大于PDM各尺度0.0154/0.00475/0.00183，最终预测变化RMS0.8681、均值+0.3466；UV该格Q从约1.545降到0.755。Normalize affine与denorm有部分尺度抵消，不能仅看权重或不同checkpoint内部绝对RMS宣称绕过限幅。gamma负值仍合法。gate/null、每通道及2/3/5组权重/有效贡献、形态/统计分支和train/validation差异完整保存，不把权重当因果重要性。 本轮按mechanism-statistics中全16组validation U/UV记录修正该范围下限，原数值证据不变；详见本轮doc-number-correction.json。

**V上下文确有作用，稳定收益未成立。** 捕获实际d/tau、W_d d/W_tau tau、h/logits/pi/w、P/Q/Y和修正。validation投影RMS比W_tau/W_d约0.68%–8.05%，固定同checkpoint各尺度train样本tau均值后，pi变化RMS约0.000378–0.028066、共同评价空间预测变化RMS约0.000300–0.024887，误差方向混合。此局部干预保留PDM状态及Y_s，不取validation/test均值，不是v1重现或独立重训消融。UV fold4/H6第三尺度pi均值由v1约0.999997降至v2约0.620605，熵由0.0000375升至0.826812；H9也更动态，不能再说UV总固定。H12仍偏第三尺度约0.950736，但实际加权第一尺度RMS15.829、第三尺度2.191，pi不代表输出贡献。w=.5+1.5*pi且各Y只受最终求和监督，V在校准共训贡献/相消，常主要调节U已经影响的Q路径；不能由高相关或单项幅度直接判无用。

三项趋势摘要的线性信息损失也有直接诊断：fold4/H6 V细尺度以完整train趋势识别既有Q线性关系、在validation残差RMS约1.77e-7，三统计线性解释残差约2.319、Q标准差2.281；UV对应完整残差7.54e-7、三统计1.408/Q标准差9.945。仅为只读中间量关系识别，不是训练新预测模型，也不反证GELU可以非线性利用tau；尚不足以证明再补统计能过完整效果门槛。

**组合、优化与误差边界。** 同权重UV旁路V在固定validation两fold均变差，但UV normal仍不如独立训练U；即时依赖不等于独立收益，offUV不是B。fold4/H6按旧B train q50/q90定义的validation低误差组125条，B MAE0.21085→U v2 0.32494/UV v2 0.31732；高误差组39条，B2.13608→U1.53100/UV1.51986。只说明该样本的误差再分配，不能宣称test尾部改善或因果。完整history显示fold4/H6 B best/停止20/20，v1 U/V/UV=2/7、1/6、6/11，v2=16/20、3/8、4/9；v2 U已经有下降后期best，合法早停不等于训练不足。validation双指标改善未稳定保持到test，不能只用loss=MSE解释或改用test选best。

**一组首选判断，待ChatGPT综合。** U v2已有正向依据，建议保留原公式/位置/开关，暂不再凭小窗口冻结一项U结构改动，fold4 MAE硬失败继续保留。V优先返回ChatGPT重新考虑其目标相关作用接口及共同校准，当前同源小修的边际依据不足，不自动再推加统计/温度/rho，也不声称所有改造永远失败。minimum-proposal.md只列保留U和替代V所需能力/既有PDM/目标趋势/P/Q/Y张量接口、MS/M/F1～F4/资源及纯第二轮配对约束；没有选新论文、模块名称、公式或融合。若需新U2/V2，检索/阅读/选源/改造与组合必须回ChatGPT；Codex不实现v3、不换训练协议、不训练或发许可。后续任何新候选的完整四臂验证范围仍须另行审核，不能把建议当本轮新增run授权。

本轮状态：v2技术完成24/24；artifact audit自检Passed（上述限定范围）；新机制review与ChatGPT result review Pending；目标一Not met，目标二/三Not evaluated。原实现审核继续有效，不代替新结果审核。只存在本分支canonical/M6的unstaged增量、index空及指定外置材料；不单独closure，不add/commit/push/merge，不执行其余53项、补fold/新seed或进入M7，统一交回ChatGPT综合判断。

<a id="m6-timemixer-v2-implementation-r1"></a>
#### 同源v2：U残差限幅、V目标趋势上下文及完整训练接入（2026-10-16）

本轮起点仍为`437b43b5ace68329ceb5a18cf75952baaa9103c4`，设备`b82673a648b4`、M6-Timemixer及当前工作区不变，三端0/0。进入时仅canonical/M6两份前轮文档unstaged，SHA分别d74ab97dff084c7592d50a1e5965352f71f1a1abb2a6e0f5b9464fbd22938af5、fc9e28cc03626231dd2ab3d729aa4a93d408554b20ae7fe23615e47c6a1e88a8，index空。用户通知机制review Passed并批准两项同源改造；这不是原24项效果Passed，也不是已批准启动新批次。前轮文档与本轮实质实现统一交审，不先提交文档。

**模型合同。** 独立ID`timemixer-uv-dev-v2`，新包/入口/脚本/配置均明确v2；不在`timemixer_dev/`添加文件，不改旧candidate_id。U仍在最细尺度Normalize后、CI展开前，只把`gamma_U*delta`改为`s*tanh(gamma_U*delta/s)`，s沿修改前接收历史[b,R,L]的L轴计算标准差尺度加eps1e-5，形状[b,R,1]。无新增参数、detach/clamp/温度，gamma=.01及负gamma合法性、mask/null/空来源/零core、MS仅目标/M同时更新保留。可选诊断显式返回r/s/r_limited及gate/core/alpha，默认不保存激活。限幅不是最终输出界限。

V的CI0上下文严格来自本次pre_enc实际送到原out_res的趋势[b,Ls,C]，按明确target_idx取mean/last/meanabsdiff，[b,1,S,3]，Ls1差分为0；与原PDM描述拼成3D+3，h新增48个权重。U启用时T0使用实际已改写的历史，不重复Normalize/分解，不detach、不改head输入。M/CI1仍原3D描述与原投影尺寸，数学/同参数输出和梯度与v1 V精确一致。eta/hbar/f/p、rho=.5、原尺度求和修正、原PDM/FMM/head、单次denorm及MS/M语义不变，CI1 marks排列和S1 PDM限制保留，不静默修B。

**接线、身份与保护。** `timemixer_train_v2_entry.py`、`scripts/ch3/start_timemixer_uv_v2.sh`实际调用v2 worker/adapter，train/validation/test不退回旧模型适配。source binding覆盖28份实际依赖（包括新入口及只读复用的v1 checkpoint helper），不含文档/日志/resolved自身摘要；作者绑定、许可说明原样保留。稳定训练状态验证/恢复只读复用，外加v2 candidate/revision/live spec/trend-mode前置检查；相同shape不接受v1。原活动Adam/RNG完整校验、owner/start_ticks拒绝无副作用、逐child assigned ID核账、STOP/失败停队列、成功身份保护/test一次/不明claim拒绝与原子发布保持。技术schema足够时沿用，不机械全改版本号；科学candidate和结构身份严格独立。

**科学配置与聚合。** 已存v2 resolved从SHA为4839ccdaf2c2e67edcf2bfda86cf6786c96f2e7a64b17599bd7d3879aaf618a5的原配置复制派生。77项逐任务structural_source_profile、训练参数、data/split/scaler/marks、L/P/H、seed、窗口/批次数与optimizer上限均保持，只变candidate/revision/task ID和必要技术来源；不重审历史全包或读取真实CSV/checkpoint。24/24/29、37 B/0 B重训、8格同v2 F4复用、1340上限不变，旧UV不得充当新F4。首批24个v2 ID为U-f1/U-f4/V-f1/V-f4/UV-f1/UV-f4六波各四H，最大480 run-epochs，只准备、不启动；其余53项保留，不加其他fold/seed或混合旧新臂。

新政策`timemixer-uv-dual-gate-v2`分别定义：目标一每fold内四H等权，fold1和4的MSE/MAE各自均需UV<U<B且UV<V<B，不得用八格平均抵消；目标三仍八格等权、两指标各需F4<F2<F1且F4<F3<F1，F4精确复用本v2八个UV；目标二仍11域各域原H平均、两指标各至少8域严格胜出、集合可不同且不设交集门槛。未舍入严格比较，完整24只判断目标一，精确缺项/重复/错版本不得凑数。数值条件、工程完成和result review仍独立。

**本轮CPU验收。** 原amd Python 3.11、torch2.0.1/numpy1.24.3/pandas2.0.3，-B/禁bytecode/CUDA不可见，CPU与interop线程1；仅生成合成CSV及checkpoint。最终同字节32/32、0skip：10项模型/关闭/初始化/U/V，10项元数据gate/配置/版本，10项worker/loader/恢复/队列与2项共享checkpoint保护。覆盖off/off初参、RNG、输出/loss/梯度及Adam/OneCycle精确等价；v2 U/V子流、限幅公式及常数/负gamma/饱和/空来源/零core、非饱和梯度；真实pre_enc传递、非零target、CI1数学/梯度、一次denorm；F1～F4/H12/P1、EPF96/24及M全通道合成loader；四臂完整worker、合法epoch精确恢复/顺序/RNG/best、跨版本/臂/F/H/协议/趋势/source/data/Adam损坏前拒绝；test claim/封存/重复身份/STOP/owner/receipt及正常双波。

首轮31个唯一用例中27通过、4个用例失败，日志包含2条failure及3条error（含清理次生异常）。原因是EPF S被误写死3、精确参照strided/contiguous浮点归约布局不一致、mock拦截Git子进程，以及冷exec双波45秒超时/fixture清理仍有写入。修测试工具而未改模型或训练源码，未删断言/增skip/放宽容差或deadline；复用已审核CPU fork/runpy实际入口双波fixture，另补冷CLI元数据。准确通过范围是fork双波及冷元数据，不把冷exec双波超时写成Passed。失败子进程随后退出，8次更新从保留journal独立核回；无关进程未启停。首轮counter中的child0为异常路径未收账的原记录，保留原字节，以外置补充核账为准。配置对照工具另有遗漏tasks顶层白名单的断言失败，补齐已逐任务核对的身份变更后通过，原日志保留。

更新预登记为首轮主≤128/子≤8；失败后预登记修复复验新增主≤80/子≤8（不占科学run额度）。实际首轮77+8、复验77+8，累计主154/子16/共170次合成Adam更新，包含等价/调度参照的真实step。技术失败和修复成本单列，不改历史769–781等旧消耗；真实CSV读取、旧真实checkpoint读取、真实train/validation/test、GPU查询/probe及CUDA负载均0。没有机械重跑旧281/21/15/22套件、机制诊断或完整来源审计。

**资源与操作草案。** Urban F4完整profile参数量B70885/U76434/V74645/UV80194；U相对v1新增0、V CI0新增48、CI1新增0。新增r/r_limited各[128,1,12]、s[128,1,1]、tau[128,1,3,3]仅作静态量级说明，不冒称GPU峰值/并发数值/makespan已测。复用原Urban batch128 q4历史及v1结构评估，准备q4上限6波，不因缺新probe退q1，也不恢复强制probe。GPU管理仍60秒/最多3次/仅超时重试，UUID/余量、全局锁及运行保护不降级；本轮不查GPU，现场余量留到审核closure后的真实启动检查。

统一外置`amd-execution-evidence/m6/M6-Timemixer-v2-implementation-r1/`保存before、版本复用/diff分类、科学配置不变证明、32项日志与首轮失败/成本、合成artifact、资源增量及非执行launch-preparation。template的commit=null、review_approved=false，未创建真实attempt，无可消费许可；child入口确认v2，操作命令已语法/路径核对，守护方式复用既有证据、未另做空跑。当前implementation自检Passed、ChatGPT v2 review Pending、Git closure未执行、v2真实训练未执行、三个效果目标Not evaluated。v1原/新目标一Not met及297 epochs/909136更新、旧成果/负向记录保留。本轮所有变更unstaged、index空，统一送审后再按授权精确closure；不自动启动新批次或进入M7。

<a id="m6-timemixer-mechanism-diagnosis-r1"></a>
#### 现有U/V受限机制诊断与最小改造可行性（2026-10-16，替代合同）

本轮新用户指令完整替代已撤销的诊断指令。Codex只分析现有实现/保存实验并执行受限诊断；ChatGPT先综合回执，再决定是否改U/V。如需新U2/V2，论文检索、阅读、来源选择、组合设计及结构裁决均回ChatGPT；Codex未搜索/下载替代来源、未实现新候选或训练。继续原worktree/分支，HEAD与首批训练commit均`437b43b5ace68329ceb5a18cf75952baaa9103c4`，前置三端0/0，仅上一轮两份文档modified、index空；已知dirty不伪造clean或绕过正式启动器。

证据唯一增量：首批结果根下`mechanism-diagnosis-r1/`，先登记hypotheses.md与diagnostic-plan.json，再执行。32份best包括原24个U/V/UV及精确配对第二轮8个B，按原manifest/result/checksum/schema/profile/任务严格追踪与加载真实作者Model；没有用UV双旁路冒充B，没有加载last。复用上一轮审计，不重跑48/655完整包；只核对实际读取来源、checkpoint、32份history及24个best对应真实update行。原first-batch-review-r1、24项产物、B引用保持原字节。

样本：fold1/4×H3/6/9/12×train/validation×256，共4096条记录。每格固定16个等距node order节点×16个共同H12合法起点，跨H对齐forecast origin，标签在对应split，四臂同输入/时点/标签；未按误差选样或取flatten前256。原loader仅解析至validation终点，scaler按完整原train算法复建并与保存统计逐项一致，未用小样本/validation/test拟合。保存输入、样本node/历史/标签时点/原索引及SHA。CPU eval/no_grad、线程1、batch64，不改正式batch128协议。

正常B/U/V/UV及U-only offU、V-only offV、UV offU/offV/offUV共9路径，主前向576次/36864次样本呈现；32模型观测一致性参考64次/256次，总640次/37120次。416次U/V局部诊断补算与真实模块输出逐位相同；32组真实输入观测前后预测逐位相同，参数/持久buffer不变、开关恢复，每尺度norm一次、最终denorm一次。8项合成工具首轮通过（11次完整模型forward/44次呈现、7次局部模块forward，另记），4项保存证据核账首轮通过；没有删断言/skip/容差放宽。backward=0、optimizer实例/step=0、真实test新增forward=0、GPU查询/probe=0；真实前向段约50.43秒。

实际TimeMixer路径：Urban L12/P1/C11/CI0，原尺度12/6/3均覆盖同一12小时；label H独立于P。Normalize→pre_enc season→11→64 token卷积+独立native marks→两层PDM→各尺度预测，与pre_enc原始trend经变量独立out_res/regression的旁路相加，再求和/V、单次denorm和MS目标切片。B已有变量/latent混合，PDM也有latent趋势；各尺度只通过最终target loss共同训练，不是独立监督专家。源码/公式推导表明辅助窗口常数水平经Normalize/season未直接送入MS目标，而U能用既有预处理mu/sigma补此信息；不是未处理物理量，也不声称B无任何交互。日历native marks另有绝对时点路径。

新增直接观察：

- U/UV validation有效改写RMS/自身历史RMS约0.914–2.862，含明显均值/时间变化，不能称小残差。统计/历史嵌入全局RMS比0.383–1.249，削弱“全局统计分支一直压倒历史”的解释，但Ta等单通道可偏大。alpha每核心维/通道及null、分组加权核心贡献完整保存，样本权重有变化；本轮不隔离query/key，不能把alpha当因果重要性。组权重按2/3/5通道同时报总量和每通道量，不用日历总量判断更有用。
- 价格近常数在fold4/H6固定validation窗口约68.36%/67.97%，is_weekend87.50%；U统计来自Normalize前的既有预处理数据。offU配对中粗尺度输入逐元素不变，细尺度改写经season/trend和PDM向多尺度传递；fold4/H6 U最细trend预测变化RMS2.868、PDM预测变化0.0606，UV对应18.281/0.2929，主要输出作用可走原始趋势捷径。RMS不是正交能量占比。
- V-only权重动态且修正明显，不是未工作。UV fold4/H3近均匀，修正RMS0.000819，而尺度中心化离散1.919；小修正由近均匀权重解释，不能归因所有Y相同。UV fold4/H6 train/validation第三尺度pi均值约0.9999965/0.9999972、validation熵0.0000341，修正RMS9.9666，其中原始trend修正9.8079、PDM修正0.2991；H9/H12也偏第三尺度，H9两类分量都大。V读取PDM却调PDM+原始trend，存在显式上下文缺口；近固定偏好不自动等于有害或证明test退化因果。
- 9路径逐样本预测、偏差、误差分位数及配对结果均保存。U-only/V-only八格validation正常相对各自同权重旁路的MSE/MAE均更低；UV H3旁路V几乎无影响，其他若干格影响明显。fold4/H6 UV正常MSE/MAE约0.7760/0.6508，offV约1.6908/1.0137。共训权重的即时依赖不能替代独立训练臂效果；UV offUV不是旧B。
- 误差分组边界只取B train样本q50/q90。fold4/H6 validation低组125样本MAE从B0.21085到UV0.31818变差，高组39样本从2.13608到1.52736改善；这只是同样本误差再分配，不能宣称真实test尾部改善。该validation样本格总体两指标仍改善，不能据此解释完整test MAE。抽样排序可与全量相反，完整效果结论不改。

OneCycle/history：fold4/H6 B/U/V/UV best/停止epoch为20/20、2/7、1/6、6/11；U/V best在上升段，停止时已进入下降段，之后五轮均未严格改善，属于合法早停。V best LR0.0018060208795、beta1 0.9353539492，UV best LR0.0096193239668、beta1 0.8538067756；保存update行与checkpoint/原recipe一致。不能仅因best早就说训练不足，不能把改LR/loss/patience/选点混入纯模块收益解释。本次train/validation确有局部输入分布差异（如选样Ta均值），但不证明未额外检查的test分布机制。

**仅一组待审最小建议，未实现**：U保留核心选择/统计分支/插入点，只把`r=gamma*delta`改为`s*tanh(r/s)`，s取原目标历史波动，新增参数0；有利于限制过大历史改写，但可能裁掉有用校正或产生饱和。V保留原softmax/rho/FMM公式，仅在CI0路由描述增加实际原始目标trend的mean/last/meanabsdiff，Linear_h从3D到3D+3，增加48权重；CI1没有该旁路则保留原3D，不捏造新trend分支。完整公式、张量/target/MS/M/F1～F4/开关、来源身份、资源、可反驳标准及风险见minimum-proposal.md。U保留改造依据较强；V补上下文有合理希望但未证明能消除静态偏好或改善正式目标，不直接决定换来源。

如ChatGPT接受，最小完整效果验证建议仍为第二轮协议、fold1/4×四H×三个新臂24项、配对B复用8格，旧24项保留；新结构必须新身份、fresh训练，旧权重不得冒充。此为未来建议，不是本轮实现/训练许可，不自动派发其余53项、正式补fold或新seed。若决定换来源，仅返回能力/接口/限制需求，由ChatGPT检索选择；Codex未指定替代论文或模型名称。

首批报告声明范围内的结果审计按用户通知已获接受，不冒称ChatGPT重做全部二进制审计。本轮机制材料待ChatGPT分析/review Pending，不升级为本批最终result review Passed；原/新目标一Not met、目标二/三Not evaluated不变。原八格运行policy与结果后逐fold规则均保持历史身份。只修改本分支canonical/M6，模型/训练器/loader/resolved/作者来源不变，全部unstaged、index空，不commit/push/merge、不训练、不生成新启动许可。

<a id="m6-timemixer-first-batch-review-r1"></a>
#### 首批24项完成后只读审计与结果诊断（2026-10-16）

实际训练及本轮起点`437b43b5ace68329ceb5a18cf75952baaa9103c4`；分支/工作区为M6-Timemixer、`/public/home/yueweiting/大论文/TimeMixer-U+V-v1`，前置三端0/0、clean及指定canonical/M6/resolved SHA均一致。60秒修复review/closure及最终绑定已完成，首批UrbanEV-F4/fold1、4×四H×U/V/UV已按第二轮完整协议实际完成。原模型实现、runner及双指标review继续有效，但不把它们替代本批结果审核。

结果根：`/public/home/yueweiting/大论文/amd-execution-evidence/m6/timemixer-uv-development-runs/urbanev-f4-f1f4-uv-first-batch-v1`；attempt `urbanev-f4-f1f4-first-r1`；本轮证据唯一增量目录为该根下`first-batch-review-r1/`。原启动resolved SHA为`4839ccdaf2c2e67edcf2bfda86cf6786c96f2e7a64b17599bd7d3879aaf618a5`；runtime SHA为`6e2280658b2a97c41f3d8303ad3e5712b1f13fe9dd0ecd1c9b88fd61e9afa68d`，实际grant SHA为`042f7b419def0b7b450463e5c9cf75984f5b54004a30968a54ef0c0251dba804`，均保留原字节。本轮不修改入口、原policy或任何训练产物。

审计先于文档/分析修改：精确24 IDs、24 receipts、6波；staging空、无failure；owner与最后4子进程均已退出，worker.log结束/complete/回执一致。progress.json保留最后一波RUNNING历史快照，未修写状态。Python和系统核对552个run payload，并补充核对655个attempt文件；实际CPU加载本批48份best/last，检查模型key/shape/dtype/finite、实际活动Adam参数及moment/step/参数组、scheduler/LR/beta1、epoch/BestState、journal前缀SHA、Python/NumPy/Torch/generator RNG及保存CUDA RNG元数据/格式。909136条update日志与原OneCycle逐条一致，validation严格best/早停与上限合法，test记录每项一次且eval尾batch完整，MSE/MAE与SSE/SAE/elements算术一致。source/config/data/environment/producer与原训练身份一致。

累计消耗为297 run-epochs、909136 optimizer updates、24次保存的validation-best完整test。本轮新增更新0、真实test评价0、CUDA恢复0、GPU查询/probe0。B仅复核第二轮8份保存result/history/manifest/config SHA与profile、split/windows/node order/marks/scaler，不加载B checkpoint；本批二进制审计不等于CUDA恢复数值Passed，真实CSV仅stat及保存绑定核对，不冒称全文件内容重新审计。655文件原SHA复核未变。

**本轮用户新口径（结果后采用）**：目标一每fold独立对H3/6/9/12等权，MSE、MAE各自均需UV<U<B且UV<V<B；严格使用未舍入值，不跨fold抵消、不要求每H排序或U/V固定先后。只评价已有fold1、4。原launch/runtime/config中的双指标v1八格平均保留，新口径外置版本为`timemixer-goal1-post-result-foldwise-v1`，不得称为训练前冻结。目标二方案A、目标三八格平均/F4复用不变；未来fold1/2与fold3/4拟展示仍每fold独立，本轮不启动2/3或取消5/6范围。

| fold | 模型 | 四H平均MSE | 四H平均MAE |
|---|---|---:|---:|
| 1 | B | 0.879998523406 | 0.681282958468 |
| 1 | U | 0.722107437593 | 0.629086303238 |
| 1 | V | 0.885728381345 | 0.684625165919 |
| 1 | UV | 0.725389827037 | 0.630776588619 |
| 4 | B | 0.445213390112 | 0.470758879070 |
| 4 | U | 0.447931483401 | 0.495911153911 |
| 4 | V | 0.446925487056 | 0.472155860690 |
| 4 | UV | 0.440766613767 | 0.490795610682 |

独立复算与交接初算一致；展示舍入不参与判断。原八格B/U/V/UV的MSE为0.6626059567589859/0.5850194604967792/0.6663269342003673/0.5830782204021093，MAE为0.5760209187687716/0.5624987285745319/0.5783905133043834/0.5607860996505887；两指标均UV<U<B成立而V<B不成立。新逐fold两fold均不能满足完整目标，原/新目标一数值Not met；目标二、三Not evaluated。

三条诊断：U fold1平均MSE/MAE降低17.942199%/7.661524%，fold4升高0.610515%/5.342921%，后者四H MAE均差；V fold1/4两指标平均均较差，fold4 H6的3.925866%/2.443114%恶化抵消其他三H小幅改善；UV fold4 MSE降低0.998797%、MAE升高4.256262%，H6/H9 MSE改善而H3/H12变差，四H MAE均差。UV相对U在fold1升高0.454557%/0.268689%，fold4降低1.599546%/1.031544%；相对V在fold1降低18.102452%/7.865410%，fold4 MSE降低1.378054%但MAE升高3.947796%。完整每H/未舍入值见CSV/JSON。

History核对显示fold4 U/UV四H best validation两指标均优于B，test MAE却均较差；B/U/V/UV的best epoch四H均值在fold1为13.50/6.75/12.50/4.75、fold4为15.25/5.25/9.25/8.25。V fold4 H6在epoch1最佳、epoch6正常早停，B该格在epoch20最佳。已有证据可排除本轮检查覆盖的错配、调度、选择、元素数/聚合或封存错误，但不能从单seed聚合误差确定分布变化、V尺度饱和或变量因果。gamma_U允许变负，实际有限；V参数统计及Adam活动不能替代pi/修正量/残差分布。外置review分别列事实、排除范围、假设和下一轮最小train/validation诊断建议；未授权额外forward/seed/搜索/候选/训练，不补真实test预测，不因效果差原样重跑。

证据包括audit.json、audit-supplement.json、binary-audit.json、原产物655文件SHA、来源/审计记录、post-result-policy.json、results.json、per-H.csv、history-comparison.csv、module-parameter-stats.csv、fold-means图和可复算脚本。8项纯保存指标fixture覆盖每fold四H等权、跨fold不能抵消、双指标联合、缺H、F4去重、错B配对、相等不胜出及其他目标未评价。审计检查器命令日志字段假设、渲染器summary字段定位及amd环境缺绘图库的失败日志均保留；修正工具后通过，仅绘图使用既有系统Python，无安装。训练代码/产物未修、数值容差未放宽。

状态：技术完成24/24；artifact audit为Codex自检Passed；ChatGPT result review Pending；目标一原/新数值Not met；目标二/三Not evaluated。本轮只改本分支两份权威文档，外置分析送审，保持unstaged/index空，不commit/push，不执行其余53项、正式补fold或M7。生产/二轮/共享TimeMixer/历史B产物不写入。

<a id="m6-timemixer-gpu-timeout-repair-r1"></a>
#### 管理查询超时窄修复与复验（2026-10-16）

起点/当前HEAD `f48c9104913ca9ab9f38bff605fc6cf3657126f9`，M6-Timemixer/new worktree不变，前置三端0/0/clean，launcher旧SHA `b8c6a4680a911955c3bd017277e3358a7b129d83bf45ccd6145ba842adff5e8b`匹配。用户SSH观察19.796秒与ChatGPT Desktop Commander观察13.166773566秒由用户交接提供，分别登记，不伪称Codex执行。Codex本轮独立一次查询13.073184480秒，exit0、stderr空，返回index0/UUID `GPU-3d365efd-300b-f527-e8fe-703fb0cfb738`/总81920MiB/余81222MiB；必要进程查询一次12.781181385秒、exit0/无计算占用记录，全局锁可取得并释放。仅当时快照，不冒称CUDA负载或GPU恢复数值验收。

唯一生产逻辑差异：launcher.gpu_startup管理查询timeout10→60。最多3次、仅超时重试、UUID与余量门槛、全局锁、所有权、STOP及失败不派后续波保持。9项无模型mock复验全通过：timeout实参60、成功解析、UUID拒绝、绝对/比例显存门槛、边界、超时后成功/三次最终失败、非超时不重试、supervise查询失败既有attempt不变/新attempt不创建/无worker。CPU模型计算/Adam更新/GPU负载0，不改模型测试deadline或数值容差，不重跑281/21/15/22；原9模型文件SHA不变。

resolved仅更新launcher来源SHA，全部77 task/profile/data/optimizer/训练上限/gate_policy与上一审核字节语义完全相同；24/24/29、37 B/0重训、8 F4复用、1340上限保持。物理minor2、NVML index0和CUDA本地ordinal分开；本轮不改cuda:0、可见性或设备映射。GPU管理成功和耗时确认10秒上限不足，不继续宣称硬件故障或强制管理员前置，旧失败日志与诊断说明仍保留原字节。

最新用户已撤回serial/q4短轨迹、梯度/makespan强制安排；按已有效的UrbanEV/TimeMixer batch128 q4正式历史、U/V静态结构及当前资源快照评估固定6波q4。不新增probe，不因缺新probe阻塞，也不称增强模型峰值已测。首批24已确认，B复用8，余53保留，第二轮OneCycle、MSE/MAE双硬指标及目标二方案A不变。本轮不训练、不新建科学任务或seed。

现行外置材料见`M6-Timemixer-first-batch-admission-r1/timeout-repair-r1/`，包括来源区分、查询原输出、定向日志、配置不变证明、资源/操作草案和最新manifest/patch。管理查询及当前快照自检通过，代码和新材料ChatGPT review Pending；未来实际修后commit未填，v2模板review_approved=false，审核closure后仍由用户最后启动。全部变更unstaged、index空，不commit/push，不修改生产/第二轮/共享作者或历史artifact。真实训练未执行，效果Not evaluated。

<a id="m6-timemixer-dual-gate-r1"></a>
#### 双指标硬验收、方案A及已确认首批（2026-10-15）

起点`05d254341c9b5e63c3f5cd0499e6ffd7933d1440`，M6-Timemixer、新路径不变；用户通知此前模型和runner修复review Passed，19路径已精确单提交push、三端0/0/clean，外置closure边界保留。本轮新字节尚未由ChatGPT读取审核，不倒改旧Pending、自检或失败记录。

用户新科学决定：MSE和MAE各自均为硬要求，未舍入严格小于，相等不胜。目标一fold1/4×H3/6/9/12的8格等权，两指标各自满足`UV<U<B`且`UV<V<B`。目标三完整UV输入消融同8格等权，两指标各自满足`F4<F2<F1`且`F4<F3<F1`；F2/F3不排序，F4复用目标一同身份8个UV结果。目标二方案A：六M域各自四H等权、五EPF各H24，MSE胜出域≥8且MAE胜出域≥8，胜出集合允许不同、交集不设门槛，不跨域平均原始误差。仍继承第二轮原始TimeMixer B，不重训有效B；原review接受范围为来源/执行/原始指标/封存完整性，不冒称checkpoint tensor/optimizer/RNG重新加载审查。

训练不变：第二轮完整MSE loss、validation MSE严格best、fresh seed2024/Adam/OneCycle/早停；MAE只增加效果硬验收。所有77项task/profile/结构/训练/data/stat/窗口/步数、37 B/0 B重训、8 F4复用、24/24/29及1340上限与上一审核resolved逐项相同，仅政策与必要源码技术绑定更新。U/V九份原模型文件不修改，CI1 marks排列与S1 PDM限制保留。

首批24项已确认，沿用现存精确ID和顺序，6个候选q4波次为U-f1/U-f4/V-f1/V-f4/UV-f1/UV-f4，各含H3/6/9/12；每run最多20 epochs，首批最大480 run-epochs。其余53项保留，不执行、不取消；不新增其余Urban四fold96项正式任务。q4资源最终准入仍Pending，上轮GPU0查询10秒超时，UUID/余量未知；本轮无GPU查询、probe或真实负载。新外置模板不填未来审核commit/GPU UUID、review_approved=false，不授予真实启动。

交付版本化`timemixer-uv-dual-gate-v1`，launch/run-identity/owned-runtime v2，配置/worker/runtime/回执/CLI/汇总一致验证，拒绝单指标旧绑定或“both”字符串。报告政策与成功训练身份分离，完整运行/恢复源绑定不放宽；政策或报告封装变更不能重新训练已成功任务。checkpoint技术schema v2及原恢复/test-once行为保留，不修改旧checkpoint/结果/历史review。

每目标按精确ID与B配对覆盖独立判断；首批24齐全只评价目标一，目标二/三仍Not evaluated。完整77项输出goal1及goal3的MSE/MAE条件、goal2两个胜出计数/集合和各目标联合数值结论；工程完成、数值条件、结果审核分离，自动报告不称result review Passed。现有completed registry负责跨批聚合，重复/错ID/错配拒绝，F4不重计。定向测试、真实合成更新记账、科学不变证明、全量/增量patch及完整SHA集中在既有送审根新增`dual-gate-r1/`，见该处acceptance.md；旧21/13、修后15、281及失败材料原字节保留。

当前状态：双指标实现与定向自检以外置实际结果为准；ChatGPT新字节review Pending；Git closure未执行；真实训练未执行；三个效果目标Not evaluated。全部修改保持unstaged，index空；只维护本分支canonical/本唯一M6，不合并生产、不进入M7。审核与closure后仍须资源/许可绑定，由用户最后启动。

<a id="m6-timemixer-runner-repair-r1"></a>
#### runner定向修复与复验（2026-10-15，repair-r1）

起点HEAD仍`491ce9848575f744125def9950e994a9bc5a65e8`，M6-Timemixer工作区不变；前置18路径/SHA/mode及空index核对一致。ChatGPT发现：旧supervise在锁占用/live owner/已完成恢复拒绝时进入异常块写STOP/failure；Adam state只做合法ID子集检查，允许实际活动状态丢失，CUDA RNG缺设备/长度检查；合法自洽回执未绑定child预期task及当前attempt，complete直接复制传入ids。已在本轮临时锁、空负载owner与新合成checkpoint/artifact复现，未用真实生产负例，原U/V数学9文件未改。

修复一：先在全局锁与恢复前置只读检查通过后转移owner，归档的是旧控制快照；只有本次claim的PID/start_ticks/runtime_ref仍与owner相符，运行失败才写STOP/failure并处理所属进程。锁占用、旧owner活着、完成attempt、恢复授权/来源不符等拒绝不改原文件、不移历史、不发信号。修复二：checkpoint技术schema=v2，显式保存Adam活动参数名/step，完整匹配state与参数组/moment/进度，原作者unused参数允许无状态；严格设备/RNG格式/形状/长度检查与私有对象验证先于live写入。当前Torch2.0.1无GPU UUID属性，设备UUID从已核验启动runtime绑定，CUDA_VISIBLE_DEVICES=0及完整可见CUDA状态必需；CPU仅测试设备元数据拒绝，不声称GPU恢复数值Passed。

修复三：每child保存expected task ID，回执与完整task/arm/F/fold/H/seed、实际config/source/producer、结构spec/runtime/device绑定并限制当前root/attempt合法runs路径；每波精确集合核对后才推进，整批complete按实际校验集合生成。明确恢复子集只核本次执行项，旧已完成结果保留/汇总但不重训。相关源码绑定及实际保存resolved JSON同步；除技术schema/来源外逐任务科学配置严格不变，77/24+24+29、37 B/0 B重训、8 F4复用及1340上限不变，不增seed或任务。

全部本轮证据在`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-Timemixer-runner-integration-r1/repair-r1/`；acceptance.md登记复现/复验/首次失败、main与owned child真实合成消费及选择原测试的理由，operations.md是修后操作说明，最新manifest/完整patch及相对原18文件增量独立保存。旧r1 manifest/patch、21/13记录、原失败日志和769–781消费均保留，不回填未知尾部；技术修复消费另账，不是新增科学训练额度。无281/16/历史全量回归，无GPU/真实train/validation/test、旧checkpoint或真实许可；所有变更unstaged，修后ChatGPT review Pending，真实训练与效果目标仍Not evaluated，禁止closure/启动。

最终repair-attempt7-final同字节15/15、42.506秒，通过7项新增（19种恢复损坏副本、8种CUDA纯元数据负例、5种supervise拒绝、8种回执负例及恢复子集/实际resolved核对）与8项原受影响测试；合法epoch恢复全状态/batch顺序精确相等，test-once/不明claim、两波12 updates审计推进及失败1 update停队列均验证。最终suite main42/child13=55；本轮含复现/首次失败/三份CPU诊断共main308/child70=378（本轮预登记总上限438）。旧769–781原账不改，累计1147–1159。首次共享夹具重复身份、测试进程替身/STDOUT适配错误和45秒冷启动等待均保留失败；不删断言/增skip/放宽deadline或容差。最终双波测试恢复原batch16/窗口/12 updates，只以CPU专属预热fork创建真实CLI监督器/worker，生产exec不变；冷CLI完整worker的三份诊断保留。CPU interop限1为合成线程约束，未改真实训练线程/模型数学。当前工程自检Passed，ChatGPT修后review Pending。

起点为实际closure commit `491ce9848575f744125def9950e994a9bc5a65e8`，branch=`M6-Timemixer`，工作区`/public/home/yueweiting/大论文/TimeMixer-U+V-v1`。首对U/V实现review、迁移3文档增量/16项review Passed；前轮6文件准备范围亦已审核，仍未提交。本轮保留旧准备包的manifest/patch与原始字节，新增范围联合送审；不stage/commit/push/merge，不更改候选`timemixer-uv-dev-v1`，不进入M7。

**状态与B采用**：用户确认第三轮371/371技术完成、5083 epochs；两轮原始review已接受，覆盖来源、执行、原始指标与封存完整性，不能冒称新读取checkpoint tensor/optimizer/RNG或效果gate Passed。本轮对需用37个第二轮TimeMixer单元与新task审计逐项核对producer/profile/protocol/data SHA、原始MSE/MAE与result/manifest引用一致，不重审742任务/5373文件，不读取旧checkpoint、不重新评价真实test。采用绑定`/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-original-round2-round3-result-review-v1-20261015/`：round2-raw-result-review.json SHA=`b69ebfce815d029bb2819f4ea33fa7c60f13ca74ccbe3f432dce3e3937bd4783`；round3-raw-result-review.json SHA=`cbee1e91b61b63d0bd3a0dfa890c1ca28b3987e4d7d66b0a5a02b3ada21563dd`；task-audit.v2.json SHA=`1dbc648f6e9a9243f06937578622561d0fb92ad3fb1e35a8766472be22bfd74f`。原JSON pending保留历史，采用依据是本次用户通知与生产`2a910ba83ef560d9eaafb1087979dd38c080df90`文档绑定，不回写artifact。实际生产HEAD已为`a6a373c96ec9ed553b6a4d47282fabe974936086`，新增仅PatchTST文档政策；只读保护其当前字节，不合并或复制生产文档，不采用展示值。

**协议已定**：全部增强任务逐项继承已审核baseline-sources中第二轮精确profile及配置引用；新[完整resolved配置](../../configs/timemixer_uv_round2_resolved_v1.json)只含第二轮。Urban L12/P1/b128/20/pat5，EPF L96/P24/b128/20/pat5，五个非Weather M L96/b128/10/None，Weather L96/b128/20/pat10。batch/eval128、fresh seed2024、Adam float32/accum1/wd1e-7/betas0.9,0.999/eps1e-8，真实OneCycle max_lr0.01、pct_start0.2、cos、momentum0.85/0.95、div25/final10000/three_phase=false，每个成功update推进一次；初始化LR名义0.0004，eval不推进，早停不重新缩放。结构D/S/PDM/head及全部其他profile字段继承，Weather B实际20/13/15/16不充当新任务epoch上限；不加载B旧权重，不冻结backbone，不修改CI1 marks或S1 PDM行为。

**任务核账**：77项＝目标一24＋目标三24＋目标二29；37个已有B配对键、B重训0、F4精确复用8个目标一UV结果、最大1340 run-epochs。每项完整保存schema/target/aux、独立marks、结构来源、训练参数、split窗口/batch/steps_per_epoch及epoch/update上限。F1/F2/F3只改同fold/H F4的schema与必需输入维度/数据身份，训练规则不变；实际loader按真实F定义构造，不能用F4内容冒充。旧非执行计划和五份来源材料保留，未创建其余四fold的96项，也未选择实际启动的77项批次。新科学配置execution_authorized=false；运行许可独立于结构spec和范围计划。

**实际接线**：`timemixer_train_entry.py`、`timemixer_dev/{training_config,worker,training_state,artifacts,launcher,summary}.py`及`scripts/ch3/start_timemixer_uv.sh`交付真实worker与守护链。构造、train/update、validation/test全部使用增强适配独立身份；`utils/ch3_data.load`仅增加显式resolved_profile可选入口，baseline默认逻辑不变，复用原loader/scaler/windows/marks、Schedule与BestState/finite。完整保存config/spec/source/data/environment/真实argv、command与launch历史、worker日志、history/update计数、best/last状态、一次完整test角色/结果、metrics/manifest与checksum。Python及系统SHA核对后原子发布，完成后不可覆盖；失败保留staging/checkpoint和真实消耗。不长期monkeypatch原runner/作者，不修改已审核U/V模型数学、原adapter或共享作者仓库。

恢复限定完整同身份epoch边界，先检查checksum/journal与全task/arm/F/L/P/H/seed/训练/data/source/schema，再检查key/shape/dtype、optimizer参数组/moment、OneCycle LR/beta1/updates、BestState及全部RNG，之后才写live模型/optimizer。恢复Python/NumPy/Torch/CUDA RNG、loader generator、真实epoch/step；不支持未封存batch回滚，不清零加训。test claim不明时拒绝自动重测；已合法保存test-result在封存恢复中复用。validation严格MSE选best及patience，test完整访问一次、M全通道/MS目标按SSE/SAE/元素数聚合且尾batch不丢。主判指标仍待用户明确，不能把训练MSE或报告双指标当成三个效果目标已批准硬门槛；汇总缺项/主指标未知不报科学Passed，完整数值条件也与result review分开。

**固定批次与资源**：总启动器只消费明确task ID子集和独立审核后binding，不默认派发77项；同批固定波次仅在技术审计通过后推进，失败/STOP停止后续，按owner PID/start_ticks只清理所属进程。q4上限方案复用第二轮Urban/M真实TimeMixer并发记录；第二轮EPF L96/b128所选记录为单路，新增U/V张量静态估计下的EPF q4/4+1仍属待审推断，不能借第三轮b32当b128实测。保留全局GPU锁、启动时GPU UUID/余量、clean三端commit/config/source/data/q绑定、STOP；无强制probe链、无运行期GPU轮询。tmux空负载守护/STOP与真实CPU子进程双波推进/失败停止定向验收，systemd/nohup仅语法准备。未创建真实训练根、未生成真实可消费许可、未查询GPU或运行真实模型。

单一外置增量为`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-Timemixer-runner-integration-r1/`：acceptance.md登记各组CPU结果、失败/修复、实际合成更新及有限回归；operations.md给出语法核验的启动/日志/状态/完成/安全STOP/受限恢复命令，非执行binding模板不预填未来commit。完整patch相对HEAD含上一轮6文件，incremental.patch相对本轮开始快照，最新manifest绑定当前全部字节；旧包不覆盖。最终完整21/21同字节CPU验收及相关loader回归13/13通过，最终suite精确190次Adam update；全部尝试769次已封账，早期超时子进程另有0–12次无法追溯，累计769–781（预先上限880）。首次LR名义值断言、重复身份fixture及命令导入问题已修正；双波用例原45秒deadline/全部断言保留，最终复验通过，原失败日志不删除。原281/16证据沿用、不重跑全套。真实数据仅必要stat/既有绑定核对，工程测试CSV均合成并走真实loader；checkpoint测试仅本轮合成状态。当前工程自检Passed，ChatGPT本轮review Pending、Git closure未执行、真实训练未执行、效果目标Not evaluated；审核与closure后仍由用户最后一次启动，待定主指标/具体批次/资源绑定不阻塞本轮代码实现。

<a id="m6-timemixer-baseline-plan-r1"></a>
### 基线来源核验与非执行开发计划（2026-10-11）

依据本次用户通知，首对U/V implementation review、worktree迁移/3文档增量/16项新路径验收review均Passed；针对绑定作者行为，不表示原作者无缺陷。第一段只对已审核manifest的26个精确文件stage/commit/push：真实commit=`491ce9848575f744125def9950e994a9bc5a65e8`，parent=`5a1f34dbea9e906d3525f79b094b82efad2159b0`。本地HEAD=origin/M6-Timemixer=网络端同名ref、ahead/behind=0/0、worktree/index clean，26个blob及Git mode、三份审核文档SHA保持不变；保护工作区及不可变baseline tag核验无本轮范围变化。独立closure证据在`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-Timemixer-closure-r1/`。没有重跑281/16项模型测试，没有为closure改写送审快照。

全部closure启用条件成立后，从这个实际clean commit开始第二段。当前工作区仍`/public/home/yueweiting/大论文/TimeMixer-U+V-v1`、branch=`M6-Timemixer`、candidate=`timemixer-uv-dev-v1`；M6内既定训练准备，不新建阶段、不改U/V/原B/共享作者、不重取论文、不进入M7。

**B来源定向核验**：需37格＝UrbanEV-F4 fold1/4×H3/6/9/12的8＋六M域四H的24＋五EPF H24的5。当前采用链指向选定dm4第二轮371格索引，53个TimeMixer单元与此前固定enc2索引完全相同；本轮37格按原采用关系为MS13、基础M128 8、amend1 16。保存manifest/result/runtime引用SHA、config canonical digest/精确profile、实际训练commit、history/budget、严格validation MSE选best、合法早停及test-once记录均核对；实际训练commit分别为`df6a16403e10d51097db8c88829909c533d15652`、`760b9dd7162d200c11b8836a7c9ece41822dfcf1`、`b97392a179b55b4bffb2bf0e3b85267bb0b07750`，各六份关键源码Git blob与manifest绑定一致。37格可进入后续配对审查；采用索引technical_complete=true、result_review=pending，本轮只称“保存结构化元数据与采用链核对通过”，不冒称全artifact审核Passed。amend1 16格没有直接raw-config source项，明确记录该缺项，以实际读取文件SHA、canonical protocol digest和精确profile绑定，不伪造原manifest。

第三轮核验当前合法attempt=`THIRD-PatchTST-dm4-Urban6-four-H-reviewed-q-r1`的controller/upstream/config/main/boundary及停止队列信用0；读取时URBAN_SUBSET_FORMAL，37个确切TimeMixer任务路径尚无结果，全部Pending。不等待其完成，不借用旧停止attempt或展示值；其余16个Urban格仅保留索引定位，正式补齐前另审。两轮均绑定TimeMixer作者`e24610583b36fdd8c76cc17a8df4e65759a5f460`及相同入口SHA，未改变结构以追齐论文。

**实际协议差异**：TimeMixer第二轮均OneCycle(max_lr0.01、pct_start0.2、cos及动态beta1)；第三轮initial_lr1e-4、type1_horizon_scaled_v1及固定基础betas。Urban两轮L12/batch128/20/5、P1；EPF第二轮L96/batch128/20/5，对比第三轮L168/batch32/10/3；六M域均L96/batch128，非Weather10/None、Weather最新有效20/10。Weather第二轮四H实际完成20/13/15/16轮，best11/3/5/6，不能从旧20/None或Weather10快照抄值。ETTm1/2保留K5/freq=t历史marks，其他域K4/h；各格features/target/aux/structure及marks规则两轮相同。详表及完整调度参数见[协议与接线核对](../evidence/M6/timemixer-v1/training-preparation-r1/protocol-comparison.md)。

保存MSE/MAE均train-standardized；MS目标元素聚合，M全通道元素聚合，MS-target诊断不替代M。开发Urban8格、正式24格等权；六M域各四H平均、五EPF各H24，不跨域原始误差平均，不双指标硬门槛。主指标、统一二/三轮协议及B配对/result review仍待决定，不逐域/H择优、不因某轮B较弱而选协议。本轮只访问已保存test指标/history，没有新test评价；不称盲测。真实数据SHA/scaler不重算，checkpoint/二进制payload不读取、不扫描、不补SHA；缺失审核范围明确保留。

**非执行计划**：[configs/timemixer_uv_development_plan_v1.json](../../configs/timemixer_uv_development_plan_v1.json)绑定上述实际closure commit，training_protocol/main_metric=undecided、execution_authorized=false。目标一F4三新增臂24，目标三UV F1/F2/F3 24，目标二UV M24+EPF5=29，合计77；无B重训。目标三F4精确复用目标一8个UV同身份结果，输入与模型消融不做笛卡尔积。77是候选完整覆盖，不是每轮套餐或本轮获准启动批次；其余四Urban fold的96项未创建，仅保留三目标达标审核后补齐条件。EPF L保持未选、来源选项96/168，避免假装已选训练协议。16项纯元数据/有理权重核对通过，包括任务唯一性、24/24/29、37个B配对键、8个F4复用、11域各自平均和范围防扩张；没有模型构造或真实数据评价。

当前薄适配已支持profile→model→prediction→同形loss及合成strict恢复；真实runner的构造/update/validation/test仍接旧adapter，候选独立ID路由、F1–F3任务/profile派生、选定scheduler/BestState与正式artifact/spec/optimizer/RNG身份绑定、资源审核及独立启动授权仍需具体接线。可复用既有data/marks/scaler/window、Adam/scheduler/BestState、MS/M指标与审计设施；本轮没有实现训练入口或扩建runner/队列/许可平台。CI1 marks排列及S1 PDM限制保留；若后续要修，先裁决修后身份与B可复用性，不自动重训B，也不凭缺新probe阻塞当前准备。

本轮新增证据集中于`docs/evidence/M6/timemixer-v1/training-preparation-r1/`：baseline-sources.json、source-bindings.json、protocol-comparison.md、metadata-validation.json；外置执行/失败记录、clean起点、完整增量patch与SHA在`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-Timemixer-training-preparation-r1/`。canonical和此前implementation/迁移/closure材料保持原字节；只在本M6专节登记新实质发现。元数据工具的schema假设与AST列表调用错误已按实际内容修正并留失败记录，未修改模型测试断言或放宽SHA核验。

**第二段停止点**：B配对来源与非执行计划就绪，新增材料未stage/commit/push、ChatGPT review Pending；第一段26文件closure仍为已成功事实。真实开发实验未执行，三个效果目标Not evaluated；统一协议/主指标及真实训练启动授权未取得，不启动GPU/probe/真实train/validation/test，不加载真实checkpoint。以下首对实现与迁移专节保留此前送审时事实。

### 首对实现与迁移送审快照（保留原文）

首对候选实施授权已实际执行，设备`b82673a648b4`；本分支当前工作区=`/public/home/yueweiting/大论文/TimeMixer-U+V-v1`，branch=`M6-Timemixer`，第三轮基点/当前HEAD=`5a1f34dbea9e906d3525f79b094b82efad2159b0`。首次创建前曾核验主仓库/二轮/三轮/TimeMixer完整HEAD及交接文档SHA、本地tracking与必要网络ref一致，同名本地/远端分支及初始目录不存在后才创建；这段创建记录不把当前路径伪造为最初创建路径。分支仍无upstream，未push。实际读取适用AGENTS，AGENTS未改。

本轮按用户授权执行Git worktree正式移动，当前工作区已统一为新路径，迁移记录见外置送审根的`worktree-rename-r1/`。移动后26文件相对路径/SHA/mode、分支/HEAD/index均核对不变，旧目录不存在；原Git管理ID保留且gitdir已正确连接新路径。随后仅修改canonical、本专节及既有验收说明；模型、测试、结构配置及资源辅助脚本SHA不变。指定16项既有CPU合成测试全部通过、退出0，无skip或放宽断言；覆盖真实作者导入/子进程cwd、profile/任务、strict磁盘合成checkpoint及Urban C11/H3、M C7/P96、EPF L168带marks四臂forward。原281项日志/失败证据和已审核patch/manifest不改；最新送审必须使用`worktree-rename-r1/file-manifest.json`与`worktree-rename-r1/complete.patch`，本轮不重跑全套、不修CI1 marks排列或S1 PDM限制。

最新科学与公式合同以[canonical新专节](../AMD_EV_Thesis_Final_Implementation_Plan_v2.1.md#m6-timemixer-uv-dev-v1)为准。仍M6、不重走阶段、不进入M7/空间；J暂停新开发，旧J/AMD/baseline及负向证据不失效。工程ID=`timemixer-uv-dev-v1`为首个候选，不是最终论文模型或效果通过。B是原TimeMixer，四开关臂独立于J/N/S身份；第四章接入未定。

**多分支文档治理补充**：本分支与第三轮生产分支分别维护自身canonical及本唯一M6 milestone，属于同一逻辑文档的不同Git分支版本，不允许长期存在互相冲突的最终权威文档。本轮只补本分支新增专节，保留共同基点和历史正文，不改生产文档、不跨分支合并、不新建平行状态文档。未来用户决定整合时，必须先核验双方最新完整Git SHA、文档SHA、共同祖先及各自相对祖先的实际文档diff，同时保留双方有效开发/实验记录，禁止整份覆盖；无Git文本冲突不等于无科学合同/状态语义冲突。授权、实验身份、结果采用或当前状态冲突先交ChatGPT审核和用户裁决，不擅自选择、删除、覆盖。是否/何时合并及最终时间模型仍由用户后续决定；本次修改前后SHA与增量diff保存在外置送审证据的`document-governance-r1/`，支持未来整合，不代表已审核或已整合。

开发目标一为UrbanEV-F4四臂，目标三为固定UV的F1/F2/F3/F4输入消融；二者均只fold1、4×H3/6/9/12的8格等权平均，不能做笛卡尔积。目标一UV分别优于U/V、U/V分别优于B；目标三F4分别优于F2/F3、F2/F3分别优于F1；均不要求U/V或F2/F3排序及每格严格成立。目标二只B/UV：六M域各四P平均、五EPF各P24，共11域最多3个例外，主指标未定、不双指标硬门槛、不跨域原始误差平均。三个开发目标满足后才补其余Urban四fold并按24格重判正式效果；77/173为工作量参考，没有自动执行授权。

Urban 275区域在样本维展开，L12/P1及H独立身份保留；FeatureSchema核对F1–F4为C6/8/9/11、volume/idx0，不重排；独立historical native marks、split/train-only scaler/价格天气/节点窗口继承，无空间交互。未来本分支允许完整train/validation/test开发，test反馈须记访问与影响、不能声称盲测，不参与梯度或scaler拟合。原B复用统一选定轮次同任务有效正式来源，不逐格择优；二轮/三轮训练协议及主指标在实质训练前统一决定。第三轮代码基点不构成第三轮协议选择。

实际文件：新增`timemixer_dev/{modules,source,spec,model,adapter}.py`、包入口、`source_binding.json`、来源notice；独立`configs/timemixer_uv_structural_v1.json`无执行许可；新增定向验收和旧回归合成wrapper。没有改通用runner/loader、旧baseline配置、生产许可、AGENTS或Closed milestone。真实作者Model继承、原key保留；off/off直接作者forecast；启用仅最细norm后U和原FMM求和处V。独立seed子流、完整profile→model→prediction→同形loss和strict合成恢复已接通。

来源：先实际获取/解析正式SOFTS NeurIPS2024和Pathformer ICLR2024 PDF、完整作者shallow clone，再实现；PDF额外复制到大论文`paper/`。外置只读参考目录为`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-Timemixer-references-v1/`，不入本Git。SOFTS@`f5d35fd7c3e716b6383ce6d3cc42c131e32c3c44`/MIT；Pathformer@`ea85d82932215e171357da47b3bc82d502344758`/未找到许可文件，U/V独立实现不复制其代码。论文公式、源码调用链/依赖及保留/删除/改造见[来源审计](../evidence/M6/timemixer-v1/implementation-r1/source-audit.md)，完整PDF/文件SHA见绑定；不用裸main。

CPU最终完整实现268/268，分组A关闭身份4、B/U8、C/V6、D任务与接入244（含240形状组合）、E工程6；另13/13相关旧合成回归，共281项有效通过。关闭初参/输出/loss/梯度/Adam更新/RNG零差异；新增逻辑参数finite非零梯度，9个启用fixture各3 Adam step为27，A组3对关闭等价另6步，最终完整套件共33；三次完整实现套件累计99个CPU合成Adam step。测试float32，关键代数补float64；无真实数据/旧checkpoint。初次误选旧回归入口出现两个缺文件error（未读到文件），日志保留；改用显式生成4344×275 CSV、生产parser和原测试断言，13项复验全部通过。资源脚本语法错误已修正，源PDF sidecar首次surrogate写入错误已改用可审计转义；未削弱断言/增skip/放宽容差。初轮266、补充后268日志保留，最终以同一实际字节下`acceptance-final-r2.log`为准，不拼接Passed。

原CI1 marks.repeat的b/C排列限制原样保留；原PDM S1会IndexError也未修，当前profile S3/S4；V单模块S1零修正通过。两项均有具体来源及原B/增强对照，不能隐去或以此声称旧B已修。Urban F4参数B/U/V/UV=70,885/76,434/74,597/80,146；更多profile及主要张量形状见[资源JSON](../evidence/M6/timemixer-v1/implementation-r1/resources.json)。M Weather/b128/P720静态例中U pair[128,21,21,32]约6.891MiB，V pi[128,21,4,720]约29.531MiB；只是单张量float32算术，非GPU实测峰值、并发准入或正式batch选择。后续ChatGPT结合已有适用证据评估、优先q4，缺新probe不自动阻塞。

验收环境为`/public/home/yueweiting/miniconda/envs/amd/bin/python`，Python3.11.15/torch2.0.1/numpy1.24.3/pypdf6.16.2；所有验收CUDA不可见、-B/禁止bytecode、OMP/MKL/torch线程1，未安装升级依赖、未生成共享作者pycache。必要保护复核不递归扫描实验产物，正常生产日志/进程变化不回滚；本轮无GPU查询/负载、真实train/validation/test、旧权重读取、生产启停或任何Git closure动作。

首轮实现证据集中于`docs/evidence/M6/timemixer-v1/implementation-r1/`；完整含新增文件patch和逐文件SHA另存`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-Timemixer-implementation-r1/`，首轮获准参考/PDF副本保留。本轮仅移动本工作区、更新本分支文档及外置迁移增量证据，不修改生产/二轮/共享来源/真实产物；全部Git变更unstaged。**implementation自检通过；依据本次用户通知，首对U/V ChatGPT implementation review Passed、281项CPU证据已审核（针对绑定作者行为，不代表原作者无缺陷）；迁移自检通过，迁移后字节的ChatGPT review Pending；Git closure未执行；真实开发实验未执行；三个效果目标Not evaluated。** 本轮迁移增量交回审核，不声称ChatGPT已读取迁移后版本；审核后与首对实现统一Git收口，本轮不自动closure。以下为生产基点原文。

**In Progress — N审核起点0a0049a8fbe3505c7f3ca719b8c392df071a0e2c；四H attempt已按用户指令safe-stop，owned进程/session退出且该attempt未来信用0。最新用户政策：未来新建EPF七baseline通用默认4＋1，当前56组全部固定q4；probe按需诊断，历史证据评估固定q计划直接正式调度；371项/5670最大run-epochs、Weather128等科学字段不变。本轮未提交候选、CPU/合成验收，无新许可/GPU/训练。**

## EPF通用默认4＋1与当前56组固定q4候选（本轮用户新决定）

用户决定：TimeXer在后续所有新建EPF实验中，与其它六个baseline永久统一默认4＋1，策略版本`epf_all_baselines_4_plus_1_v1`。通用`ch3_type1_tasks.probe_groups()`按PJM→NP→BE→FR→DE排序，每个baseline一个q4容量组：第一波PJM/NP/BE/FR实际四路，第二波DE实际单路。该尾波不是probe、OOM回退或新增任务；不跨baseline补位。当前B没有局部分组特例，直接消费通用默认及新版本固定计划。

历史3＋2保持原身份和字节：真实旧MS worker的PJM/BE/FR为batch16/q4容量，NP/DE为batch4/q2；原工具、来源报告、Passed及负向记录只读保留。冻结旧type1 EPF协议有显式`historical_epf_probe_groups_v1()`只读兼容入口，按精确protocol digest识别，不能成为新任务默认。新协议或字段变更不能冒充该旧身份。未来明显不同的batch/模型形状会在通用分组中标记resource_review_required与风险，等待资源评估；不静默恢复3＋2。

当前TimeXer五市场均T168、pred_len24、C3、train/eval batch32、float32且网络结构一致。统一四市场并发是用户批准、历史资源证据支持的工程推断，不登记新组合numerical Passed、Measured或GPU准入成功。当前计划Urban7＋EPF7＋M_ALL42＝56组，全部固定q4；EPF14个正式波次、35项。Urban与M_ALL执行分组保持，任务身份、三份科学配置SHA、Weather128/非Weather PatchTST dm4、seed/数值/test边界不变。

第三轮仍371项，最大3360＋350＋1960＝5670 run-epochs；已审核的5670启动合同修复与3990拒绝保护继续保留。正常队列直接FORMAL，不强制PROBE/AUTO_AUDIT；保留GPU0启动前身份和max(8GiB,10%)余量、TimeoutExpired最多3×10秒/间隔2秒、q≤4、全局锁、STOP/owned停止与正式失败停队列，运行期GPU遥测0。旧两H和四H STOP、678/679/905最后持久化实际成本及未知尾部保持，不供给新完成/准入信用。

本轮证据集中于`reviewed-concurrency-reuse-v1/timexer-epf-four-plus-one-v1/`。旧57组计划、5670修复包及纠正范围前的非执行版本不覆盖；新计划、科学/启动合同、producer证明、非执行许可模板与准备脚本逐份版本化。永久回归覆盖通用新协议默认、显式历史读取、TimeXer两波、56组/371项/5670、错误版本/顺序/q/SHA、真实permit/runtime/worker/manifest/group complete以及Python3.11/3.13的66项保护；结果与所有失败诊断见本轮acceptance/evidence-index。当前未stage/commit/push，无可执行新许可、GPU preflight或arm，新增字节待ChatGPT审核；technical_complete、result_review与效果gate分开。

补审确认：原25项最终执行清单遗漏`test_complete371_real_scheduler_permits_audits_boundaries_no_probe`，生产`completion_record()`因未定义`v`触发NameError；原25/25不代表完整COMPLETE链已验证。本轮局部修复从已验证`contract()`取得EPF分组版本，并在`validate_complete()`显式核对该版本、56组计划、371/371计数、两次停止队列信用0及result_review=pending。实际CPU合成`q.run()`已执行源采用→Urban168→EPF35→M_ALL168的permit/runtime、正式产物审计与三阶段封存，最终complete.json在独立夹具落盘且真实完成验证通过；版本/旧57组计划/SHA/旧信用/计数/review状态9类篡改拒绝。该夹具没有真实GPU、模型训练或test，不提供科学完成信用。Python3.11/3.13真实source-check与66项AST保护通过，新proof-v11、非执行template-v7与准备脚本版本化；旧proof、失败及送审字节保留。最终15文件完整patch必须独立包含13份tracked修改及两份新增文件；本次增量位于`timexer-epf-four-plus-one-v1/completion-chain-repair-v1/`，实际日志、SHA及审计边界以该包为准。仍未stage/commit/push、无新可执行许可/GPU preflight/arm，修后字节待ChatGPT审核。

## 并发probe按需诊断、固定计划直接正式调度候选（2026-10-10，北京时间）

用户正式修改M6并发执行政策：并发probe改为按需工程诊断，历史成功证据优先复用；ChatGPT训练前按模型实际结构、batch、T/pred_len/C/token/patch、dtype、GPU/driver、threads/workers、OOM与数值/RNG/order/吞吐证据评估固定q1/q2/q4。不仅因dataset/fold/H不同机械重测，不把缺少新probe等同于只能q1。相同或相近形状且历史依据充分时正常推荐q4；估计与实测分别记录，不伪造未测Passed、numeric equivalence或Measured。最终并发计划仍待ChatGPT本轮字节审核。

按用户指令于北京时间2026-10-10 01:24:37（UTC2026-10-09 17:24:37，STOP落盘时间）停止四H attempt `THIRD-PatchTST-enc2-dm4-Urban6-four-H-r1`。核对scope、许可b3608cfcd276cff959eef7f16d97a64676fb8a7c7718aeb79e0b43b9fb92405d、controller22784/start_ticks220162060、probe child23072/start_ticks220163886及worker46233/start_ticks220765449后，既有safe-stop退出0并落盘STOP；上述进程和tmux均退出。原controller.failure保留InterruptedError('persistent unified STOP; no retry')。正式训练0，EPF/M未派发；本次四H的所有probe/validation/正式产物只作停止历史，不作为未来新队列的准入或completed信用，禁止恢复原attempt。更早被废止的两H B记录及原第二轮371、196项、层数48和宽度40合法来源保持。

停止时AMD与DLinear各完成24 serial及24 q4 worker，PatchTST完成17 serial、另1 worker中断；共113完成/114启动、2组完成/1组部分/4组未启动。波次总账持久化actual Adam678/backward678/forward904；逐worker预算另核实中断worker已有0/1/1，故全部worker最后持久化actual合计678/679/905。原波次总账不倒改，独立cost-reconciliation-v2补充差异；reservation684/684/912不是实际消耗。113个完整六步短轨迹中训练forward678、合成validation forward226；不是226次真实数据验证或正式test，正式test0。中断worker未落盘尾部的实际消耗仍未知，最后持久化数只作下界，不退款、不扣减科学任务完成额度。已完成波次elapsed为AMD serial1955.531512秒/q4 510.809978秒、DLinear serial1613.879306秒/q4 440.505540秒、PatchTST serial1281.616084秒；这是监控入口墙钟记录，不是GPU忙时，不能相加解释利用率。event-driven运行期GPU显存遥测本来不采集。独立停止/撤销、成本及1683份保留文件inventory存于`urban6-four-H-third-v1/reviewed-concurrency-reuse-v1/`；二进制数值payload与checkpoint只记录stat，不读取内容，SHA明确为null。

新政策身份为`reviewed_concurrency_reuse_without_mandatory_probe_v1`，新attempt `THIRD-PatchTST-dm4-Urban6-four-H-reviewed-q-r1`、session `ch3-m6-third-fourH-reviewed-q-r1`，结果根为同级`...third-selected-dm4-urban6-four-H-reviewed-q-r1`。独立合同、固定并发计划、producer证明、非执行许可模板和operations均在本增量目录。正常调度从采用已接受旧来源后直接进入Urban168正式→EPF35→M168正式及各自技术封存，不触发PROBE/AUTO_AUDIT，不创建probe permit、假admission-summary或probe manifest；正常启动模板probe预算为0，原手动诊断工具保留但须用户明确请求和适用独立授权，不暗中派发。新formal permit、当前owner/HMAC runtime、worker配置与manifest、波次/技术边界、summary/complete均绑定固定计划及真实科学来源；无条件True不能替代完成审计。

候选计划57组：56组推荐q4；TimeXer EPF NP/DE结构组按原范围推荐q2。历史原始来源82组，包含原MS203、M84/M_AMEND112、r6 Urban七组、enc2层数及dm4宽度实验的成功短波/正式收据。49组当前主要张量/结构形状在历史记录中有匹配，8组为相近形状推断；这里的“历史实测”只描述来源，不表示新队列已实测。Urban H6/H9在T12/pred_len1/C11/batch128下按同模型形状评估；第二轮M/宽度实验与第三轮type1的优化协议差异明确保留，所有57组均未在新attempt实测并发数值等价。历史显存峰值只按原采样范围解释，event-driven缺失值不补0，不声称持续余量或无人占用由工具证明。被停止四H及两H B均不供给新并发信用。真实正式OOM、finite/数值/业务失败按原技术失败协议停止，不自动降q、变batch、清空重训或增加test。

科学配置原字节不动：Urban六fold×H3/6/9/12×七模型168，EPF35，M_ALL168，总371/最大5670 run-epochs。非Weather PatchTST M20项dm4/encoder2；Weather4完整128/heads16/encoder2/ff256、20/10及其余全部科学字段保持；新371采用索引仍20替换/351保留。保留GPU0及启动前max(8GiB,10%)余量、TimeoutExpired最多3×10秒/间隔2秒、有界查询失败阻塞、全局GPU锁、q≤4、环境/CPU亲和性、来源/config/data SHA、真实初始化/RNG/order、预算/finite、validation-best/test-once、STOP和合法owned清理；运行期GPU查询0。原restricted guard和bundle不改，66项跨Python AST登记集合保留；必要的12项调度/许可控制函数差异明确登记，其余54项保持，模型训练数学及数据函数不变。

本轮仅代码/CPU合成与只读来源验收：当前政策39项、受影响科学/隔离回归24项、独立数值比较8项，共71项有效通过，全部在同一生产字节下完成。首次10项套件有3个error（group receipt漏传mandatory_probe=false及两处夹具问题），已修正并保留日志；后续47项混合套件为39项通过、8项历史数值夹具在比较前因全局lifecycle混用报错，不冒称47/47。8项原数值测试不改断言、不改容差，独立解释器复验8/8。真实371项CPU合成正式驱动未派发probe，实际permit/runtime、受限guard、正式结果审计、三阶段封存和complete均覆盖；真实只读source-check通过、66项跨Python校验通过、GPU查询0、未生成许可。详细通过数、源字节绑定和全部失败日志见本增量acceptance/evidence-index。CPU夹具的虚拟closure、独立路径、合成checkpoint占位与结果记录不可用作正式科学来源。未stage/commit/push，未生成新可执行许可、未GPU preflight、未arm或启动训练；新字节尚待ChatGPT审核。后续审核固定q计划和代码→精确closure→独立许可与启动前检查→用户一次arm，正常阶段自动衔接。technical_complete、result_review、效果gate分别审查。

<a id="m6-urban6-four-H-third-v1"></a>
## 旧B停止与Urban六fold四H独立第三轮候选

旧attempt=`THIRD-PatchTST-enc2-dm4-selected-startup-r2`当时处于URBAN_SUBSET_FORMAL，受保护safe-stop退出0，STOP持久化；controller43798/start_ticks219633778、group44176及四个owned worker均退出，tmux session消失，EPF/M和后续Urban模型未启动。停止时间北京时间22:33:31（UTC14:33:31），来源为用户撤销H3/H12第三轮有效实验资格的明确指令。旧H3/H12是此前批准的精简范围；本次用户新四H决定替代旧287项合同，不倒改历史为当时未获批准。

独立停止回执记录正式完成0、用户中断4、当前运行0、尚未派发283；四个staging与上述四个中断任务重叠，不是额外四次运行。history保存19个完整epoch/validation记录；最后持久化预算Adam33997/backward34000/forward37244，停止时未最终封账，尾部未落盘消耗未知，不冒称精确最终值。四worker启动至STOP请求的约1939–1941秒是墙钟估计，不能相加冒充GPU忙时；保留原RSS记录，运行期GPU峰值/利用率本就不采集。restricted audit仅有train/validation prefix解析，正式test记录0；中断中的validation调用数不能从完整history推断。全部日志、manifest、history、budget、metrics/process及负向记录原字节保留；8个best/last checkpoint本轮只保留路径、size、mtime/mode，不读取内容或计算新SHA（SHA记null并说明），不能声称已完成checkpoint checksum审计。旧B任何正式/probe结果不作为新第三轮准入、completed、result_review或效果gate来源，不能resume或重复arm；第二轮371、此前196、层数48和宽度40的合法来源不受影响。

父配置`configs/ch3_type1_urban6_h3_h12_v1.json`（SHA b962ca4557720d1223e14fd0decfe2be0660fb543f8049d767d315436a0ff459）只读。新`configs/ch3_type1_urban6_four_H_v1.json`具有168个唯一task/profile身份，按AMD→DLinear→PatchTST→iTransformer→TimeMixer→ModernTCN→TimeXer、fold1–6、H3/6/9/12固定顺序。原84项H3/H12科学profile完全继承；新增84项H6/H9由冻结`ch3_baseline_ms_u96_oc01_v3.json`对应真实fold/H的数据metadata、月份边界、窗口、train-only scaler与节点顺序唯一派生，再应用已批准第三轮type1合同，不读取真实test。Urban仍F4/T12/C11/MS、单点pred_len1，batch/eval128、epochs最多20/patience5、seed2024、Adam初始lr1e-4、type1_horizon_scaled_v1最大E20/coefficient4/9。数值表仅扩展适用H集合，容差和全部比较器不变；不把第二轮OneCycle带入第三轮。

新执行固定Urban168→EPF35→M_ALL168＝371正式任务，最大run-epochs3360+350+1960＝5670。EPF与选定dm4 M_ALL两个effective配置原字节不动；M_ALL非Weather20项dm4及其它148项（含Weather4完整128/heads16/encoder2/ff256、20/10）保持。新四H合同对应重新派生正式和probe保护上界，旧B实际消耗单列、不扣减科学completed也不退款，不把旧reservation计成actual。

独立attempt=`THIRD-PatchTST-enc2-dm4-Urban6-four-H-r1`、session=`ch3-m6-third-dm4-urban6-fourH-r1`，result为同级`...third-selected-dm4-urban6-four-H-r1`；新许可、log、launch/claimed、owner/MAC/STOP均隔离。增量包为原`ast-proof-repair-v1/urban6-four-H-third-v1/`，旧B及旧送审包不覆盖。七模型每组24个fold/H独立serial参照，覆盖全部168项后按原q4/q2/q1数值、finite、资源与同模式makespan门禁准入；不整体采用旧r6两H或被停止B的probe。新生命周期完成Urban AUTO_AUDIT后才派发168正式，封存后自动进入EPF35、M168；不因中途效果改计划。资源仍exclusive_gpu_event_driven_v1，仅启动前真实GPU0有界检查，运行期显存查询0，余量和明确CUDA OOM分类/STOP/预算/test-once不变。公开preflight与实际controller单一采样阶段分离、TimeoutExpired最多3×10秒/间隔2秒规则保留。

66函数的原AST登记集合及跨Python稳定算法不删：本次明确批准四H范围、独立来源与计数所需的9项控制函数AST差异逐项登记，其余57项与父版本完全一致；模型数学、guard、bundle和运行期资源/数值函数继续精确保护。summary仅增加完整168/168四H报告，独立before/after SHA登记。候选CPU/合成定向验收及所有失败日志保存于本增量acceptance/evidence-index，覆盖真实调度、受限guard、完整新Urban probe/AUTO_AUDIT、manifest、旧来源隔离、STOP、启动重试与66项证明；合成硬件/CPU短进程和数值payload不能冒充真实GPU准入或正式结果。当前未生成可执行许可、未创建新结果根/session、未运行真实GPU、模型、validation/test或checkpoint加载。后续ChatGPT审核→精确closure→新许可/source-check/公开preflight→用户一次arm；technical complete、result_review和效果gate分别审查。

本轮有效CPU验收合计80项通过，另有8项数值子测试通过：此前71项套件中70项成功记录在同一生产字节下沿用，失败的完整Urban合成夹具修正后单独通过，新增371项实际封存/完成判定1项通过，资源/数值定向8项通过；不把曾有error的71项套件写成全通过。完整Urban测试以336个CPU短worker覆盖168个serial与168个q4代表，真实执行run_configs、数值比较、资源审计、AUTO_AUDIT及紧凑formal准入；未运行真实模型或GPU。初次数据绑定误含冻结来源中5个无关EPF source_states，已只投影4个真实Urban源文件，168项metadata/scaler未变；TimeMixer夹具补足原门禁要求的七点诊断轨迹，371封存夹具补齐自有MAC secret，均未弱化生产保护。首次及中间失败日志原样保留；真实只读source-check通过、GPU查询0、未生成许可，Python3.11/3.13的66项AST验证一致。


<a id="m6-B-AST-proof-repair-v1"></a>
## B启动候选AST证明修订与真实来源验证

原11文件候选和63/63有效历史测试保留。本轮从已知10 modified＋1 untracked、index空的候选继续；不closure旧证明、不生成B许可、不GPU preflight或arm。旧producer-delta-proof SHA `9bf2b2cedb36fbe6a37ba6c4742b69364311403180170594b1b682e84723d5b9`及旧送审索引原字节保留。独立增量为`startup-timeout-recovery-v1/ast-proof-repair-v1/`，包含相对HEAD完整diff、相对原候选增量和新的逐文件SHA/mode。

根因可复现为解释器序列化口径：旧生成器与旧生产验证器均用ast.dump(include_attributes=False)；正式Python3.11.15匹配66/66且真实验证器返回，默认Python3.13.5因show_empty=False及新增type_params字段失配66/66。两个解释器的当前源码/Git父版本比较均0差异。不是66个受保护函数被修改，也不是正式3.11已运行验证失败；旧证明依赖未显式冻结的解析器行为。

修订算法full-field-ast-json-v1由生成器与生产验证器共用，显式包含所有语义字段、空列表、None、标量类型和值；旧解析器缺失type_params仅规范为空列表，非空类型参数仍受保护。原66函数集合与protected_exact、bundle、guard检查不删不放宽；新证明逐项绑定当前AST、Git父版本、原producer和旧启动证明。未来非执行模板/操作材料与默认日志位于新增量目录，startup-r2 attempt/session和原未创建结果根不变，旧许可/token不能消费。

定向CPU回归实际执行verify_production_inheritance与verify_source（真实工作目录、无整函数mock），双Python验证66项一致；故意篡改函数、proof指纹及删除登记函数继续拒绝。新prepare-permission的source-check复用未来main的完整只读来源/配置/模板检查，但不生成许可、不查询GPU，READY_TO_ARM_HANDOFF=false。未closure的main仍拒绝。原启动去重、三次超时上限与非超时立即拒绝测试继续执行；首轮新增测试中中文路径JSON转义导致的断言失败及修正后的同字节日志全部保留，最终数量以新acceptance为准。

旧送审索引2718项加索引自身共2719份文件逐份保护；旧B许可/日志/launch SHA及PRE_CONTROLLER_START_FAILED、零计算事实保持。选定371索引仍20替换/351保留，M_ALL仅20宽度修订/148保持，Weather4完整128/heads16/encoder2/ff256/epochs20/patience10、type1 scheduler/数据/评价不变。旧196、层数48、宽度40及Urban七组来源不重训不retest。287项/3990 run-epochs、独占event资源模式、数值/finite/RNG/order/STOP和运行期零GPU查询政策不变。当前候选未获新一轮ChatGPT字节审核；无新许可、真实结果根、launcher或session，不执行后段。

<a id="m6-B-startup-timeout-recovery-v1"></a>
## B首次启动前GPU查询超时与独立startup-r2候选

起点N为`8de112a2f1ff2117e94c22e5119d86ccaf0bdec7`，初始N/W/R各自HEAD/tracking/live remote一致、0/0且clean，AGENTS、冻结baseline tag、作者源码、数据及环境不变。真实旧失败发生于北京时间2026-10-09 19:21:59（UTC 11:21:59）。旧owner64326/start_ticks218806679及session已退出，旧claimed不存在；第三轮controller/结果根尚不存在，无probe/formal/validation/test计算。

旧许可SHA `be6da901ba58f24a73f2656a0467b849c63c49cdfc050ba671c1e32193ede802`、旧日志SHA `3b8f1adc66153453662ab14a32cde82f0c193bea6f98ff384f04c4445ccc8772`、旧launch SHA `121396fb9aee1bb500568c571c87b018a3362a3762f7c9b9cd8df26604f5db8a`逐份核验并保留。旧P中144份文件建立只读SHA保护记录，原准备收据、失败和来源证明不覆盖。来源科学scope、287项/3990 run-epochs、旧196/层数48/宽度40来源保持，查询超时不计正式任务或模型计算成本，不改成OOM/resource fallback。

旧arm路径为Shell preflight第一次查询→新建launcher/launch/tmux→controller.start/readiness第二次查询；第二次10秒超时触发启动失败。本轮Shell使用无GPU的launch-check，核对真实许可、当前closure/live remote、source/data/环境、STOP和排他身份；controller仍执行完整原检查与一次新的逻辑采样阶段，绝不复用公开preflight旧余量。仅TimeoutExpired可额外重试2次，固定间隔2秒，每次10秒、最多3次，查询加等待约34秒。合法响应仍核对UUID/型号/driver、显存会计/有限非负及原余量；非法响应/命令错误/来源或STOP等立即拒绝，不靠重试碰碰运气。单次耗时、重试和最终状态可追溯，超时查询子进程由subprocess终止并回收。

增量目录`dm4-selection-third-round-v1/startup-timeout-recovery-v1/`保留非执行contract/template、未来prepare-permission/operations及新的精确producer proof。新attempt=`THIRD-PatchTST-enc2-dm4-selected-startup-r2`，session=`ch3-m6-third-selected-dm4-startup-r2`；未来AUTH/launch/claimed/失败标记独立，日志为`B-startup-r2-launcher.log`。因旧结果从未创建，沿用批准结果路径但禁止重用旧许可/token/MAC。启动失败清除有效grant/_LAST，并留下拒绝再次消费token的标记；status只读显示当前Prepared与前次PRE_CONTROLLER_START_FAILED，附时间、错误及SHA来源，不物化controller或查询GPU。

定向测试执行真实公开readiness/资源判定、Shell arm控制分支、prepare_launch/verify_launch、controller.start及真实MAC生成，仅GPU输出、未来closure和tmux进程边界用明确合成夹具；controller.run在派发前截停，无真实任务。1次成功、1/2次超时后成功、3次超时、非超时错误、身份/余量/会计拒绝、STOP、旧许可隔离、token失效及CPU子进程回收分别验证。旧Urban来源/AUTO_AUDIT采用、20/351、20/148、Weather完整profile、287调度、guard与q1/q2/q4运行期零GPU查询另作必要回归；最终同字节数量和完整日志见本增量acceptance/evidence-index。

测试首次Shell夹具未正确替换日志目标，产生本增量一份0字节followup-launcher.log，随后prepare-launch因合成目标日志缺失拒绝；没有GPU查询、controller或真实许可。该空文件、失败日志及隔离说明原样保留，修正夹具后完整链通过；未来真实日志采用独立B-startup-r2-launcher.log。此负向夹具不得混为新技术启动或旧失败补写成功。运行期回归的旧r6夹具另曾因历史producer证明不匹配及真实旧日志存在而出现1 failure/1 error；失败日志保留。现仅将夹具日志隔离、用本轮真实精确差异证明验证当前字节，生产r6校验器及旧证明未改；没有无条件通过或放宽科学断言。

本轮所有effective配置、科学profile、数值容差/比较器、RNG/order/scheduler、资源门槛、guard及bundle字节保持；旧成功probe照原scope/producer采用，旧失败始终失败。新查询策略只适用于绑定本B合同，其他执行不静默增加重试；进入controller/worker后的GPU自动查询仍0。真实GPU超时恢复能力尚未实测，工程CPU验收不替代实际GPU准入。当前停在未stage/commit/push审核点，无实际新B许可、GPU preflight、arm或结果根。后续ChatGPT审核→精确closure→新B许可与公开preflight→用户一次arm，正常Urban84→EPF35→M168自动衔接，technical_complete不代表result_review或效果Passed。下方历史正文保留当时时点事实。

<a id="m6-selected-dm4-third-round-v1"></a>
## 用户正式选择dm4与第三轮B独立恢复候选

本轮起点`7d9fec8c5f66e9f38dbf396bf8d1136276b8e78e`，N/W/R初始各自HEAD/tracking/live remote一致、0/0、clean。用户选择五个非Weather域各四H的dm4，不依后续效果重新选择；作者4/8线索仍是未确认唯一配置的用户转述。

A的dm4/dm8各20项完成，controller、两阶段probe、formal组收据与技术boundary核对；280份产物文件Python/system checksum一致，实际400 run-epochs、Adam55840/backward55840/forward75292、40次正式test均为validation-selected best、finite及计数/预算保持。本轮未反序列化checkpoint、未调用新test、未运行模型/GPU。source/identity/completion/metric provenance审查通过，不替代整体result_review或效果gate。

选定来源包`patchtst-width-urban-audit-v1/separate-entrypoints-v1/dm4-selection-third-round-v1/`新增独立371索引与来源boundary；旧r6索引SHA `5ecec35a4e719104e2f11a79484660707ad8e993c4b0c94bdc06ee82bca36792`和旧boundary保留。仅非Weather20格切换dm4，351格逐字段相同（Weather4+其他347），真实训练producer保持7d9fec8...；dm8、dm16、enc1/enc3及所有历史产物保留。新来源记录不携带旧owner MAC，不自行变成执行许可。

M_ALL新版本`configs/ch3_type1_m_all_patchtst_enc2_dm4_selected_v1.json`从父SHA `f26ac0cab3cb2a2c47a20553b1f1e8e6b7d3951f5e33ae3dcdcbcb8c80a68ed2`派生，仅20项PatchTST非Weather d_model16→4及必要身份引用；其余148项完整任务/profile不变。Weather4原128/16 heads/2层/ff256、dropouts0.2、20/10、第三轮type1 scheduler、数据及评价全部保持。其他numeric policy、seed、batch、lr/Adam、监督/输出、scaler/window、validation-best/test-once无差异；第二轮OneCycle不迁入第三轮。

物理B入口`scripts/ch3/start_third_round_after_selection.sh`与`m6_third_round_after_selection_entry.py`使用独立`THIRD-PatchTST-enc2-dm4-selected-r1` / `ch3-m6-third-selected-dm4-r1`，不调用A、不接受A许可、不重派旧196或层数48。先签本生命周期的来源采用记录，再从Urban AUTO_AUDIT边界恢复：保存r6原scope/producer下实际审核七组数值/资源；仅精确path/SHA只读登记，错误task/stage/producer、篡改引用、越界与symlink继续拒绝。旧r6 failure保持失败；新B的采用complete、manifest、admission HMAC只表示新生命周期合法采用，不改旧历史。Urban84→EPF35→M168自动衔接共287，最大3990 run-epochs；原531/4550与独立宽度40/400分别记录。A的q4不外推到第三轮新协议/形状，M_ALL仍按自己的serial/q4/q2/q1准入，formal不回放全量probe。

本轮定向验收包含20/351来源、20/148配置、Weather完整不变、错误宽度/许可拒绝、manifest/完整AUTO_AUDIT接线、失败/STOP与287调度、当前owner HMAC隔离和A完成不触发B；真实来源审计与独立合成生命周期分开标记，失败测试日志保留。详细最终计数及逐份SHA以本增量包acceptance/evidence-index为准。未执行GPU查询、模型、probe/formal或validation/test；无可执行B许可、结果根、日志或session。下一步为ChatGPT字节审核→精确closure→新B许可/公开preflight→用户一次arm；本轮不执行这些后段。历史小节保持当时时点事实。


原开篇“尚未运行任何正式训练或正式test评价”及各历史小节的“当前”均属于各自记录时点，不代表当前实际进度。

下方三轮总览中Weather20的20/None保留此前批准时点；最新采用规则为两阶段20/10，见[本次启动前早停修订](#m6-weather20-patience10-v1)。其他历史事实与参数不倒改。

下方第三轮231项/Urban28及旧瞬时状态亦为历史登记。本次六fold扩展、独立层数补做与真正当前完成状态统一以[最新增量](#m6-patchtst-depth-urban6-repair-v1)为准；不把历史技术失败补做混作新增科学任务。


<a id="m6-width40-urban-audit-v1"></a>
## r6 Urban AUTO_AUDIT窄修与非Weather PatchTST宽度4/8复现实验

N实际起点`bd4b63dfcce78f38cc233e74f3c50271b020d8de`，N/W/R各自HEAD/tracking/live remote一致、0/0、初始clean，tag/AGENTS/作者代码/数据/环境保持。r6controller及worker/session退出，progress停在URBAN_AUTO_AUDIT，failure仍为`ValueError('probe artifact path/symlink outside exact scope')`。enc1/enc2分别24项边界与真实组收据/manifest/result/budget/runtime引用已只读核验，不反序列化或重新checksum历史checkpoint、不重新test。旧371与r6固定enc2新版371并存，均不覆盖；技术完成和result_review pending分别记录。

根因是`urban-probe-reuse.json`已接受AMD/fold1/H3的process `fc56e22a37ae93fe3f2ce351d55ed8824739291d20d1d2ff8fd73770a73a23d5`，但retained_refs只投影artifacts，manifest_projection合并process_refs后正确触发scope拒绝。窄修只登记该精确process，联查task/stage/producer、原许可/profile与config/runtime/trajectory/budget成功证据；旧H12失败不进入集合。原symlink/越界拒绝和guard保留，不给任意历史目录授权；真实r6manifest/failure不补写Passed。合法/错路径/SHA/task/stage/producer/资源失败/篡改/symlink在独立合成目录验收，真实数值完成校验、历史资源收据、manifest及AUTO_AUDIT另有完整合成路径证据；不重跑真实Urban probe。

本次科学线索仅为用户转述作者邮件，d_model4/8仍不确定。Codex实际核验TimeMixer ICLR2024 PDF Table2及四H均值说明，四ETT参考为ETTh1 0.516/0.484、ETTh2 0.391/0.411、ETTm1 0.406/0.407、ETTm2 0.290/0.334（MSE/MAE）；Exchange未列入，不虚构。原论文主实验三次重复与本项目seed2024单次分别披露于本服务器记录，不据此新增重复。

父源逐域逐H来自第二轮真实enc2 resolved_profiles。ETTh1/ETTh2/ETTm1/ETTm2/Exchange均d_model16/n_heads4/e_layers2/d_ff128、dropout和fc_dropout0.3；两候选只改d_model为4或8，各20任务、唯一ID及独立fresh来源。其他structure、L96、batch/eval_batch128、seed2024、Adam/OneCycle/lr0.01/pct_start0.2、epochs10/patienceNone、split/scaler/全通道监督评价、validation-best/test-once全等。第三轮原type1配置此时不改，不与本轮OneCycle混用。

Weather四H不进入新增任务：第二/第三轮真实enc2结构均128/16 heads/2 layers/256 ff、dropout/fc_dropout0.2、20/10及原scheduler；8份完整profile前后全等，配置文件SHA不变，已有Weather结果不改不重训。dm4每头维度1及dm8每头维度2经真实PatchTST适配器CPU构造/输出前向，覆盖五域四H两候选40次构造、40次前向；小batch2仅形状/finite工程检查，正式batch128不变，不作为并发证据。CPU smoke无Adam/backward/GPU/真实validation/test/checkpoint读取。

新增40项单列最大400 run-epochs、formal Adam/backward各55840、forward75292、最多40次test-once；两候选probe nominal各240/240/320、caps各360/360/480，共上限720/720/960，新形状独立准入。同模式serial/makespan与数值门禁保持，不能拿dm16并发Passed顶替；q4/q2/q1按原资源性顺序，明确CUDA OOM之外失败停止。沿用独占event模式，只有启动前10秒内固定卡查询/余量检查，运行期不采集GPU，生命周期、数值/finite/RNG/order/scheduler/guard/预算不变。

原531及4550科学合同不改写：原完成196＋本次r6已完成层数48分别登记，第三轮287暂停。新40是新增复现实验，不冒充技术失败补做，不混进旧531完成计数。r6真实probe、失败成本和旧serial/许可一律保留，reservation不当actual，另有actual账本。不为本轮重新访问已完成任务test或加载其权重。

候选复用共享签名许可、native probe→一次AUTO_AUDIT→紧凑formal审计/封存及tmux入口；独立`PatchTST-enc2-width4-8-reproduction-r1`结果/日志/许可，固定dm4→dm8。40项完成后technical_complete且awaiting_user_width_selection=true，绝不自动生成采用索引或进入Urban/EPF/M_ALL。对照报告按五域各四H列同一run的MSE/MAE、域均值、四ETT论文差距、各指标接近者及冲突；不自动择优/择差/选width，作者线索不变成确认。层数/宽度尝试和来源只要求M6及服务器记录可追溯，不新增论文板块要求。

用户统一选4或8并完成结果/采用审核后，才另建新版371：相对r6版20个非WeatherPatchTST格切换、351格保持，包括Weather128的4格；第三轮M_ALL只改20个非WeatherPatchTST的d_model。其他347格、Weather完整结构与Urban/EPF/MS配置保持。修后Urban有效证据须合法审计后采用，旧r6失败不变Passed。本轮只准备该未来边界，不提前实现选择或第三轮自动恢复。

最终同字节定向验收、各次合成失败日志、真实CPU构造捕获、原source保护、精确diff/inventory和非执行模板集中于`runtime-no-telemetry-v1/patchtst-width-urban-audit-v1/`。测试以实际受限guard安装/真实绑定路径、共享数值/资源判定、AUTO_AUDIT、来源拒绝与队列停止为边界，所有合成产物隔离。当前未stage/commit/push、未生成可执行许可、未运行真实GPU/probe/formal/test；修后字节尚未获ChatGPT审核。审核closure后再物化实际commit许可及公开preflight，由用户最后一次启动40项，日志/status/complete/width-results/safe-stop使用同一许可，正常阶段自动衔接，40项后必须停止。

### 本轮补充：A四十项队列与B第三轮恢复物理隔离

入口A为`scripts/ch3/start_patchtst_width_reproduction.sh`及其独立Python入口，原40项/400 run-epochs、科学配置与停止语义不变。A一次启动两候选，完成后只封存技术结果和对照报告；无B调用、无第三轮调度、无自动采用。审核closure后使用本次最新非执行模板排他准备A许可，公开preflight后由用户最后arm；日志/tail/status/complete/width-results/safe-stop固定同一A许可。

入口B为`scripts/ch3/start_third_round_after_selection.sh`及`m6_third_round_after_selection_entry.py`。当前是非执行准备态，允许只读plan/status；所有启动/许可/prepare-launch/GPU preflight明确exit2，旧A/r6许可和命令参数不能补成选择，不读取其许可。计划绑定r6已完成196＋层数48、原固定enc2新版371及Urban七组probe/ref，未来先完成选择后的20格替换/351格保持和M_ALL非Weather20项配置审核/closure，再生成独立B许可。受控采用在修复后的Urban AUTO_AUDIT边界执行，随后仅Urban84→EPF35→M168；48项层数实验和已完成有效probe不派发。原Weather128/heads16/ff256/20-10及完整profile保持；Weather不属于本次修改范围。

用户已明确纠正补充指令中的Weather16笔误：Weather的d_model始终为128，本次修改不涉及Weather，第三轮PatchTST Weather的任何现有配置均不修改。该字段已澄清；A仍为40项非Weather任务，B仍不可启动，继续等待非Weather统一宽度4/8选择、结果及配置审核、closure与独立许可。

当前B没有选择、可执行配置、许可或新371；不会GPU查询、调度、发信号或创建真实结果/log/session。未来独立B操作框架有arm/logs/tail/status/complete/second-round/third-round/safe-stop；这些启动动作在本轮候选中继续拒绝，须在用户选择与后续审核closure后才接入真实执行，不能用A的complete自动开启。增量`separate-entrypoints-v1/`同时保留上一版15文件送审包，提供相对HEAD完整最终diff及相对上一候选的增量、两入口权限/来源框架、CPU拒绝与隔离测试。当前无stage/commit/push、无真实许可、无GPU或第三轮启动，新字节待ChatGPT审核。

<a id="m6-runtime-no-telemetry-v1"></a>
## r5四H serial成功、q4查询超时后的运行期无遥测候选

本轮N起点`6f525489cff125248cbc0b256e9744acf67590da`，N/W/R各自HEAD/tracking/live remote一致、0/0且初始clean；r5 owner、probe-child、worker及session已退出。旧失败process SHA `2ed6d8af03233fb756df937c5c0987d4949c39d60cc636c59f430414a92cb7fb`、snapshot SHA `ebe2ea508395e69a45c6a9f0460b06a4a2ade9eeeb6951e107dc39df480641eb`精确核验。失败源为整卡查询10秒TimeoutExpired，四worker被清理、resource_admission=false、实际0/0/0；不据此认定CUDA OOM，也不离线追认为Passed。受限worker旧MS重复校验修复保持。

r5 H96独立短验收6/6/8和主链H192/H336/H720各6/6/8均成功。已只读核对原process、runtime、budget、trajectory、schema及绑定artifact SHA/身份，原producer仍为6f525489...；没有重放旧数值sidecar或读取checkpoint。数值参照覆盖完整state/gradient/Adam等原schema，继续按原比较入口使用；本轮不把历史门禁改名或把旧q4改成通过。旧serial波耗时81–83秒含监控，worker自身约10–11秒不等于原波次计时口径，不能直接与新q4比较。`r5-source.json`预登记一次仅ETTh1四H的同模式serial计时补测24/24/32；不是新增H96成功收据前置，不全量重测，不挑最快一次。其余组照原准入顺序，收益gate不改。

资源合同集中于canonical，当前明确模式`exclusive_gpu_event_driven_v1`。只有启动前固定GPU0/环境/会计/余量查询，10秒上限；真实启动将硬件证据与许可、closure、自有控制生命周期及MAC绑定。运行期generic hardware_binding、worker bootstrap、预算sample、轨迹捕获、波次/正式审计、summary和complete全部消费静态证据，不查询GPU。原分配限制机制/公式保留，仅复用已核验启动前free/total；CPU/RSS与Adam/backward/forward硬门禁保持。GPU峰值、持续余量、采样间隔及退出新样本均null/N/A，不造Measured或零峰值。正常退出和任务/产物完整性仍是必要条件。

OOM判定保留原probe q候选顺序，但原始明确CUDA异常与随后自有cleanup分开保存。有独立业务异常、普通MemoryError、信号退出、其他CUDA错误、访问/数值失败时不能资源回退；q1 OOM及formal OOM按原停止合同处理。无监控查询能在计算中终止正常worker，安全停止/STOP/超时只处理合法自有实例，不以GPU进程表批量发信号。

新身份为`PATCHTST-depth-Urban6-exit-r6-no-telemetry`，session同后缀；独立result/log/未来许可均在新包绑定，旧r1–r5字节保留。实际相关合计42/42/56＝旧负向18/18/24＋r5成功24/24/32；r4与本次失败q4为0。PATCH_ENC1 caps432/432/576原样保留，本次计时补测占原额度，reservation不当actual；旧成本不扣减科学任务完成额度。完成196、待执行335、总531、正式4550 run-epochs、七份effective科学配置及原数值表全部零差异。

无模型CPU/合成复验使用真实run_configs、资源收据判定、guard安装、共享比较、AUTO_AUDIT、formal审计和汇总入口；计算产物与硬件查询仅为显式合成夹具，所有输出位于独立合成目录。运行期GPU查询设为一旦调用即失败，覆盖q1/q2/q4、启动前错误、OOM/cleanup与混合异常、预算/STOP/超时、缺项/错身份和旧证据采用。各次失败日志保留，最终同字节结果见本包acceptance；不拼接旧版本Passed，不执行真实模型/GPU/forward/backward/Adam/validation/test或checkpoint load。

审核closure及新许可就绪后，由用户一次启动：采用旧371/196→ETTh1必要同模式计时及缺失q4准入→enc1 M24→enc2 M24→固定enc2新版371（24格切换、347保留）→Urban84→EPF35→M168。并发须实际适用证据，不凭本修订批准q4、不将正式队列统一退回串行。技术完成与result_review/效果gate分开；当前为未提交送审候选，不声称ChatGPT已审核新字节。

<a id="m6-worker-guard-binding-fix-v1"></a>
## r4真实短验收失败：受限worker重复配置验证修复候选

起点N为`805656e9da7d780330cddbf79995f234deac90d6`，parent为`0abc95542f28c000d0366ad43a29e0b85867e38c`；三仓库HEAD/tracking/live remote各自一致、0/0、初始clean，冻结tag保持。r4 owner/probe-child/worker及旧session退出；不发信号。原whole-card包、许可、日志、14份整卡采样和所有负向记录原字节保留。

真实故障链为`probe_worker → probe_resource_binding → chain.configs → validate_revision → bound(旧ms-data-bindings.json) → ForbiddenAccess`。当前guard安装已完成validate_config/validate_worker，当前stage及资源合同在metadata授权集合，旧baseline-unified96-onecycle001-v3绑定不在其中。错误发生在模型构造与计算前，不是数值失败、资源不足或保护规则错误。r4新增0/0/0；旧r1–r3累计18/18/24不退款、不变Passed，PATCH_ENC1 caps仍432/432/576。

修复仅在已安装guard的worker路径消费真实受保护配置/许可，严格核对purpose、stage/task/protocol、scope和精确resource_mode/ref；控制器及安装前的全阶段校验保留。另将已安装worker的当前配置重读改为复用同一校验结果，digest与访问钩子仍检查绑定字节，拒绝未安装配置副本。没有修改guard、monitor或bundle，没有授权、复制或改写旧MS数据绑定；probe_worker除传入原task外的计算AST保持。

无模型夹具执行真实guard安装、真实worker入口及故障绑定路径，仅在init_training前使用明确的无计算终止点。首轮被授权文字保护拒绝，且发现RESULT隔离遗漏：三份合成采用JSON留在未启动的r5目录，closure为夹具值，永久保留为测试失败证据，不代表真实生命周期或完成结果。随后隔离全部输出至本包tests；未来真实短验收/主链使用独立`PATCHTST-depth-Urban6-exit-r5-worker-guard`身份。旧r4及合成目录不删除、不覆盖。

本包`worker-guard-binding-fix-v1/`保存各次测试与最终同字节验收、精确producer差异证明、配置零差异、保护核验及非执行模板。CPU边界覆盖到真实资源绑定，未运行模型构造、六步落盘或真实GPU；后者必须在ChatGPT审核closure和新许可后通过既有单serial入口验收。成功收据仍精确采用一次，主链不重复H96 serial；失败停止，不循环重试。后续371/196来源采用→enc1 M24→enc2 M24→固定enc2新版371→Urban84→EPF35→M168顺序、335项及原上限保持。当前停在未stage/commit/push的送审点，不声称修后字节已被ChatGPT审核。

<a id="m6-exclusive-whole-card-v1"></a>
## r4未提交候选按用户独占决定改用整卡资源监控

本轮起点仍为N `0abc95542f28c000d0366ad43a29e0b85867e38c`，已知12 modified＋1 untracked、index空；13份字节及父包inventory/patch/acceptance/evidence-index匹配。N/W/R三端0/0，W/R clean，旧实例/session退出，r4主链及短验收没有AUTH、result、log或launch/claimed。旧父包437份索引材料和95项保护引用核验保持，原85/85属于旧方向，不能当作新模式验收。

用户独占声明是新模式的适用前提，不冒称工具核实没有其他进程。资源合同集中于canonical，新候选、完整相对HEAD diff及相对旧r4增量集中于`patch-enc1-identity-settle-repair-v1/whole-card-monitor-v1/`。沿用r4主链及独立serial-check身份，日志与未来许可移至该增量；不另开r5、不再开发恢复器或另一套短验收。

采样器/真实run_configs在q1/q2/q4不调用worker GPU归属、namespace/sched或GPU进程表，不创建ExitObservation。只记录固定整卡数据、耗时和裁决；逐进程峰值、Measured与外部归属为null/N/A。最少Popen/PID/start_ticks/任务配对及poll/wait/退出码仍属于生命周期管理。退出后的独立整卡采样、原余量/会计/finite/UUID/10秒查询约束全部保留；不要求显存归零。查询异常、STOP、任务超时和业务失败按真实类型拒绝；既有明确资源不足/CUDA OOM规则保持，数值与身份故障不能资源回退。

完整消费链接入当前资源合同：start模板/permit/config→单串行成功收据及采用→真实共享compare/probe审计→manifest/summary/runtime→formal波次和完成审计→固定enc2来源索引/结果汇总/complete。审计同时要求任务完整性、合法自有生命周期和真实整卡证据，不能无条件Passed。新probe不调用PyTorch逐worker allocated/reserved/peak统计，轨迹字段为null/N/A；CPU/RSS检查保留，旧模式仍按原字段和规则审计。单串行成功收据明确保存控制PID与task配对。formal资源记录流式保存/审计；AUTO_AUDIT仍每阶段一次，formal不回放完整数值probe。当前bundle与精确producer delta同步更新，旧训练producer和旧封印保持。

无模型定向复验使用真实监控与资源审计本体、实际CPU短子进程及合成整卡查询；GPU归属/namespace/进程表调用设为一旦调用即失败。覆盖q1/q2/q4、正式审计、错误绑定、资源不足/非finite/UUID/会计/查询失败、worker/STOP/超时/重复启动、安全停止只处理自有fixture，以及单serial成功后主链少派一次serial。失败尝试和修正依据完整保留；NaN失败快照以非有限值描述保存，不能作为有效资源测量。最终同版本结果见本增量acceptance，不拼接旧85/85；真实模型构造、GPU、forward/backward/Adam、真实validation/test和训练checkpoint读取均0。

全部七份effective配置/任务/profile、原133数值政策、额外两层数政策和formal/probe上限保持。旧371和196项完成来源、r1–r3负向资源收据与18/18/24成本不变；仅整卡模式和执行引用变化。审核closure后先物化本增量的短验收许可，公开preflight后做一次原batch128六步真实验收；失败停止，成功精确复用。然后按原顺序由用户一次arm剩余335项：enc1 M24→enc2 M24→固定enc2新版371（24/347）→Urban84→EPF35→M168。真实短验收不代表q4、整个probe或效果通过。本轮仍为未提交review点，新字节尚未获ChatGPT审核。

<a id="m6-patch-enc1-identity-settle-repair-v1"></a>
## r3失败后的统一有界身份核实与r4候选（此前未提交方向）

本轮技术修复起点为N `0abc95542f28c000d0366ad43a29e0b85867e38c`；N/W/R实际HEAD、tracking和live remote一致、0/0且clean。r3 owner40928/212082645、probe-child41139/212091657、worker41212/212094890已退出，tmux不存在；不发信号。process SHA `94a1f00581764cf98639c425a93980c5099c4321421582ea7452e229202272a0`、失败快照SHA `4592a5a77d4ec343c5317c6add2b807b265a6a78624b2db18ca17ca524b6a62a`与用户锚点一致。

r3 worker host49401、namespace pid:[4026534934]，查询前完整身份可靠；查询后PermissionError发生在namespace读取，stat-before ticks仍匹配，active_before=active_after=[41212]、poll仍未退出、NVML仍有49401而owned_host_pids为空。最终worker exit0、六步finite、runtime.error=None只证明短计算完成，不能替代当时资源准入。旧代码仅接受查询后已退出的特殊组合，本轮修正完整待核实策略及资源归属接线；底层errno当时未保存，内核权限原因未证实。r2活跃→退出快照及两份第二轮DLinear成功波memory已只读核对：旧波含metadata_unavailable和后续清晰样本，但缺少失读阶段，不能倒推同一种权限根因或改写历史收据。

新状态机要求可信历史身份及同一个受控Popen，当前可读host/ticks/namespace不得冲突。namespace暂时PermissionError或退出一致缺失，查询前/后及活跃/退出统一进入pending，不只接单次例外。首次失读起60秒期限不重置；同一段待核实中的跨worker交接也共享首次截止，所有对象核实后才结束该段。NVML每次查询最多10秒且受剩余期限约束。pending期间不启动下一波、不授予准入、不为取证主动结束正常worker；继续检查STOP、worker错误、GPU UUID、整卡余量和未知占用。独立新样本完整确认同一活跃实例，或poll确认终止后双重PID消失并核清NVML残留，才解除不确定；最终正常退出码、资源及数值gate仍不可缺。

资源评估保留raw unknown_pids，仅可信原生命周期的host列为pending_owned_unverified；不补零显存、不改成Measured。其他GPU进程、归属冲突、PID复用、读解析/其他阶段权限异常和不可靠查询仍拒绝；临时失读到期保持观测失败，不伪装resource-only回退。记录errno、读取阶段、单调时刻、proc状态/ticks、前后poll、待核实对象、无效范围及新鲜最终裁决；错误采样本身永不变Passed。

r4主attempt `PATCHTST-depth-Urban6-exit-r4`，session `ch3-m6-patchtst-depth-urban6-r4`，独立同级result及本包followup-launcher.log。r1/r2/r3每次6/6/8，累计18/18/24，原资源收据保持失败，PATCH_ENC1硬上限432/432/576不改；工程补做不扣减原科学任务完成额度。MS203/M84/M_AMEND112及原371字节/真实producer不改，无重训、retest或额外seed。

现有入口只能完整stage probe，本轮必要范围扩展为专用`m6_patch_enc1_serial_check_entry.py`及共享worker激活/任务guard、native probe的首serial结束分支。`ch3_native_recovery_records`同时将精确scope及单serial来源引用从manifest传至完整性扫描，包含旧完整permit和process SHA，不扩大任意路径访问；派发前确定复用身份，避免共享字典把新采集误认作旧证据而省略自检查。不是新调度平台。审核closure后先使用独立r4-serial-check生命周期、专属非执行模板生成新许可，仍通过clean精确HEAD、live remote、source、bundle、token/tmux、Popen和GPU锁保护；只执行相同PatchTST enc1/ETTh1/H96、原batch128和原六步/8 forward。成功后停止，不物化整阶段complete，不运行formal。失败保留并阻塞主链，不连续重试。成功收据的commit/config/schema/policy、实际resource/runtime、artifact SHA及退出身份完全核验后，主链只读采用这份serial；该阶段名义新增47份，短验收6/6/8另列为已采用成功成本，三次旧失败18/18/24继续单独保留。不能以短验收证明q4或全阶段Passed。

仍采用旧196项，剩余335、总531，待执行最大4550 run-epochs；顺序371采用→enc1 M24→enc2 M24→固定enc2新版371→Urban84→EPF35→M168。统一数值政策、PatchTST M固定enc2、Urban六fold/H3/H12、Weather20/10、scheduler/seed/test-once及原profile/任务/顺序不变；每阶段一次AUTO_AUDIT，formal不完整回放probe或重复联网。

本轮CPU/合成验收、保留失败日志和最终同字节结果见本包acceptance；真实模型构造、GPU probe、forward/backward/Adam、正式validation/test均0。候选未stage/commit/push、未生成可执行许可、未启动真实短验收或r4总链，尚待ChatGPT实际字节审核。

<a id="m6-patch-enc1-namespace-exit-repair-v1"></a>
## r2首个PATCH_ENC1串行波：退出窗口namespace读取窄修候选

开发起点N `d852398c7949a2feea9b572ee793c1e793ecd06c`，N/W/R三端0/0且clean，旧r2 controller22479/211655344、probe-child22687/211664339、worker22760/211667661均已退出，session不存在；不发送信号。只读核验process SHA `898c2495e079b7a3e575d04e377d1f4e6b9cca3f56110915c34efa0a70d43664`及失败快照SHA `1aec6160c3d423f7f99cd2dba848edc220f8d5d4ca58b666183cc0587aa730f3`与用户锚点一致。

八份成功采样及查询前记录稳定绑定PID22760/host13636/start_ticks211667661/namespace `pid:[4026534934]`。失败样本active_before=[22760]、active_after=[]，仅post-query namespace读取PermissionError，stat_before ticks仍一致；记录失败时poll=0、/proc条目缺失，NVML表为空且可靠。旧实现对metadata_unavailable统一立即拒绝，未进入后续新鲜采样；这是可定位的观测接线问题。快照未保存底层errno，不能进一步断言内核权限原因，更不能以事后消失或空GPU表改记当次Passed。

修后仅这个精确类型允许记录为未解决过渡。必须有早前完整可信身份和本次查询前一致身份，同一受控worker查询后poll确认退出，namespace阶段PermissionError且stat-before ticks相同；错误样本仍不可信，标记admission_deferred=true、assessment.admission=false。后续独立新鲜采样须在两次身份读取中均确认/proc已消失，时间严格晚于过渡，并核对无新生命周期、可靠GPU UUID/进程表和资源状态；NVML短暂残留继续有界等待。权限持续、活跃进程权限错误、其他阶段/解析异常、ticks/host/namespace冲突、PID复用、缺历史身份、未知占用和查询失败均拒绝。原60秒尾部和单次10秒查询上限、显存余量与failure分类保持；观测失败不按resource-only回退。

成功的过渡样本记录前后poll、原失败分类、历史身份及读取阶段；后续采样记录新鲜缺失证据和最终裁决，process追加状态快照。失败仍保留observation-failure.json的SHA引用，不把旧snapshot或负向资源收据改成Passed。restricted bundle只为监测字节更新，生产继承证明确认训练/评价数学未变；旧封印及producer保持。

r3身份为`PATCHTST-depth-Urban6-exit-r3`，session `ch3-m6-patchtst-depth-urban6-r3`，result为原r2同级`...patchtst-depth-urban6-r3`；log和未来AUTH集中于`patchtst-enc2-adoption-v1/patch-enc1-namespace-exit-repair-v1/`。r1/r2许可、launch/claimed、失败、采用边界、日志、八份r2成功采样、失败快照与两份六步轨迹保留原SHA。两次各Adam6/backward6/forward8，累计12/12/16、不退款，PATCH_ENC1 caps仍432/432/576；仅导入历史成本，不生成Passed种子，新worker必须绑定未来r3许可。

第二轮371及MS203/M84/M_AMEND112来源保持，不重训/retest。采用196＋待执行335＝531，待执行最大4550 run-epochs；顺序仍为371来源采用→enc1 M24准入/正式→enc2 M24准入/正式→固定enc2新版371→Urban84→EPF35→M168。PatchTST M固定enc2、Urban六fold/H3/H12、Weather20/10、133原数值政策、额外12个层数政策及全部配置/任务/seed/test边界不改；Urban有效H3按原精确证据复用。每阶段AUTO_AUDIT一次，formal不完整回放probe或重复联网。

最终同一代码版本定向70/70 Passed（新包`tests/attempt-2.log`），无skip。旧状态机从保存r2快照复现原PermissionError拒绝；修后该样本仅defer，真实CPU短子进程在namespace故障注入后必须经历新鲜双重缺失采样才能完成资源准入，持续权限异常仍保留SHA绑定失败快照。覆盖原身份/未知占用/60秒与查询上限反例、335项混合来源恢复、许可隔离、一次AUTO_AUDIT及formal紧凑准入。首轮70项的4处错误均为新r3包尚未物化PATCH_ENC1计划；日志和当次绑定保留，补齐独立计划后复验，不删除断言或降低门槛。

本轮仅CPU合成子进程、真实监控状态机和无模型恢复夹具，未导入torch。真实模型构造/GPU/probe/正式训练/validation/test为0，不读训练checkpoint或重新比较数值sidecar。语法、bundle、diff、范围及SHA检查见同包acceptance/inventory。候选停在未提交、未生成可执行许可、未arm/start的ChatGPT审核点，不声称修后字节已获审核。

<a id="m6-patch-enc1-observation-repair-v1"></a>
## PATCH_ENC1首个串行波的身份观测修复候选

固定enc2/Urban六fold候选已closure于`e051b96e416433b8940d9ef6b71192033fa07a89`；真实r1 attempt为`PATCHTST-depth-Urban6-exit-r1`。本轮只修该批准任务中的观测故障，不新增科学任务，也不改baseline_numeric_admission_v1、模型、训练/评价或test合同。

旧失败位于enc1/ETTh1/H96首个serial wave-0。worker PID4385/start_ticks211021095、host PID26191及namespace `pid:[4026534934]`在最后成功采样中可靠；owner4106/211008825和probe-child4313/211017804均已退出，旧session不存在。worker退出码0、runtime.error=None，六步finite及实际Adam6/backward6/forward8保持。process记录`failure_kind=observation`、`resource_admission=false`，没有任何组Passed。失败采样未落盘，现有七份成功采样和异常字符串不足以区分退出竞态、读取异常或解析失败；不能凭后来PID不存在认定当时退出已核验，也不能把本故障称OOM或数值超限。

本轮在受限监控器内增加读取阶段/失败类别及稳定start_ticks检查，并核对查询前后poll和必要namespace身份。可靠旧身份与明确退出中间态仅允许最长60秒的新鲜复采样；查询每次最多10秒且受剩余期限约束。不因退出码0直接放行，也不按缓存身份授予资源许可。权限/解析失败、身份冲突/复用、未知占用、UUID变化、不可靠采样、查询失败和超时保持阻塞。失败时独立落盘有SHA引用的观测快照，包含失败样本（如已取得）、查询前后状态、读取阶段及状态机状态；其不算成功采样或Passed资源证据。

新attempt为`PATCHTST-depth-Urban6-exit-r2`，session为`ch3-m6-patchtst-depth-urban6-r2`，result为原r1结果根同级的`...patchtst-depth-urban6-r2`；日志及本轮增量集中于`patchtst-enc2-adoption-v1/patch-enc1-observation-repair-v1/`。旧r1许可、日志、launch/claimed、controller/failure、七份采样及六步产物全部只读保留。新许可必须绑定修后实际closure及精确父失败引用；旧许可不能启动r2。本轮不生成可执行新许可。

r2只采用已完成第二轮来源，保留各自原producer；不重训MS203/M84/M_AMEND112、不增加test访问。失败enc1轨迹不作为有效资源波种子或Passed决定，仅将真实6/6/8历史成本带入账本，和本次新增量分开。补足首个缺失测量并执行原未完成准入；enc1当前没有可跳过的Passed组。原Urban有效H3来源复用，H12负向证据保留。PATCH_ENC1 probe硬上限仍Adam432/backward432/forward576，历史消费不清零也不自动扩大；正式任务完成额度不因技术失败扣减。

正常顺序仍为第二轮371来源采用→enc1 M24准入/正式→enc2 M24准入/正式→固定enc2新版371（24格切换、347格保持）→Urban84→EPF35→M168。已采用196、待执行335、总531及原4550最大run-epochs保持；按预冻结enc2采用，不按指标改选。每阶段AUTO_AUDIT一次，正式任务不完整回放probe或重复联网。

本轮最终同一代码版本定向56/56通过（`tests/attempt-4.log`），覆盖真实受限状态机、CPU子进程退出、慢采样/残留、读取与解析失败快照、严格身份反例、旧失败计量和恢复审计接线；首轮6处测试错误及后续复验日志保留。完整候选SHA见同包的`acceptance.json`、`workspace-change-inventory.json`与`workspace.patch`。真实模型/GPU/probe/正式训练/validation/test为0；CPU合成子进程和保存证据的只读核验另记。技术通过不表示新GPU准入或科学效果Passed。当前停在未提交、待ChatGPT实际字节审核点，不声称修后候选已经审核。

<a id="m6-patchtst-depth-urban6-repair-v1"></a>
## 当前增量：PatchTST固定层数补做、Urban六fold与退出观测修复

本节对应用户明确批准的新科学范围和原第三轮技术故障修复，两者分别登记。开发起点为N的`b97392a179b55b4bffb2bf0e3b85267bb0b07750`，本节仍是未提交候选，不表示已获ChatGPT审核或已启动新链。W/R、AGENTS、作者实现、原始数据、环境、Closed milestones及不可变baseline tag保持。

### 完成来源与原故障

Weather20/patience10版本已closure。旧`amend-ett-identity-r1`链的MS203、M84、M_AMEND112及第二轮371格均技术完成，Weather20已按整批规则采用；结果review与效果分析仍pending。原371主索引SHA为`8a9ccbe8c0574e0cb82a1b2d129b1528516d63772b5e3c23b96e5c84ecd43aa9`，边界SHA为`2cc3a7ac538369272f79e4b46d12fbe8f57348d88d4571d207db782d521b65fc`，本次保持原字节和各自生产commit。

第三轮已经执行AMD fold1的H3/H12两份六步串行probe，原实际消耗为Adam12、backward12、forward16，正式任务为0；不能写成第三轮从未启动。H3具备完整退出和资源收据，可按固定绑定复用。H12 worker退出码0、runtime.error=None，但资源收据失败：`owned exit observation did not settle within 3 seconds`。该波实际采样间隔最长9.732354781823233秒；末尾子进程metadata不可用，NVML仍登记host PID14956，对应worker PID14330、start_ticks208056410及其原PID namespace。现有材料支持退出观测与慢采样衔接故障，不能据此认定OOM或将资源Passed手填为true。

当前窄修仍要求原实例退出、GPU UUID/namespace/start_ticks身份可靠、无未知占用且获得新鲜清洁尾样本；尾部总等待最多60秒，每次NVML查询最多10秒，临近截止时进一步收紧两次查询的合计时间。短暂NVML残留仅延迟裁决；长期残留、PID复用/归属冲突、读取不确定、权限或查询失败、UUID改变均拒绝。查询/观测故障不进入resource-only降并发回退。资源余量、原数值政策和模型计算数学不变，当前restricted bundle仅更新监测文件的封印，旧producer封印保留。

H12原6/6/8失败消耗及原轨迹继续保留，但缺少的资源退出证明不能用离线数值自比较代替。因此只重新测量这一受影响serial；H3不再派发，其余新层数、新fold和未开始任务仍需按真实形状完成准入。并发最多q4，仅使用实际验证范围，不将3层或fold1/2的旧准入直接外推。

### 新科学任务及结果采用边界

PatchTST新增encoder1和encoder2各24项，共48项，依次完成enc1固定组、enc2固定组。六域为ETTh1、ETTh2、ETTm1、ETTm2、Weather、Exchange，H96/192/336/720，seed2024。逐任务从第二轮已封存的对应PatchTST有效profile派生：ETTh1/Exchange取M_BASE，新增三个ETT及Weather取M_AMEND；Weather明确取20轮/patience10来源。唯一结构差异为`e_layers=3→1/2`，variant/task-ID/配置与产物路径独立；作者PatchTST实现及其余结构、输入、监督、数据处理、Adam、batch/eval_batch128和完整OneCycle参数保持。四ETT及Exchange为10/None，Weather为20/10，scheduler仍按最大轮数计划。每项独立fresh，不取原3层权重，不warm-start。

M数值准入沿`baseline_numeric_admission_v1`，full_float_state、state_atol2e-4、metric_atol1e-6，PatchTST原loss及所有exact身份要求保持。当前五阶段原133条政策不变，两层补做增加12个对应model/dataset政策条目；政策条目不等同于probe组。

48项仍作为第二轮协议下的独立层数实验，原399次计划运行事实及旧371封存、3层结果索引不变。用户随后在新增enc1/2结果产生前明确冻结第三章M主表PatchTST为enc2，并澄清“固定enc2，不因结果改选”；这一科学身份不再pending。enc1完整执行和保留，但不参与正式主表；enc2全部24项技术完成并封存后，建立独立`round2-main-results.patchtst-enc2-v1.json`和对应边界，24个PatchTST M格统一切换，其他347格保持原cell全部来源字段。新版记录原索引引用、逐格previous/new来源、预先采用政策与采用身份，原Weather10→Weather20来源链保留；旧文件不覆盖、不修改、不删除。MSE/MAE来自同一run的validation-selected best，不按test、指标、相对排名、论文接近度或enc1更好/enc2更差改选，不逐H拼接。

第三轮M_ALL新增配置`configs/ch3_type1_m_all_patchtst_enc2_v1.json`从第三轮原对应profile派生：仅PatchTST24项`e_layers=3→2`，配套新variant/task-ID、metadata身份映射和来源绑定，其余144项科学profile与task-ID保持。原type1学习率、seed、batch、epochs、patience（含Weather20/10）、数据及test合同不变，M_ALL仍168项；全部MS任务（Urban六fold和EPF）的PatchTST配置保持。新形状必须完成本阶段适用probe，不继承3层并发Passed。

`depth-results`完整呈现enc1/2/3逐域四H及域内均值，三种配置、结果和结果产生前的固定采用规则仅要求在M6与服务器实验artifact完整留存，不要求增加论文实验设置、附录或其他板块；正式M主表展示固定enc2。TimeMixer Table7的Layers属于其自身模型，通用run.py默认不能证明主表PatchTST逐任务配置；本次未确认该作者来源，48项不称精确论文主表复现。

第三轮Urban扩为七baseline×fold1–6×H3/H12共84项，只有两个预测偏移，不是六折四H矩阵。原fold1/2的28个科学profile不变，新增56项逐一采用冻结的真实fold月份边界、训练窗口、train-only scaler、节点顺序及metadata，重新派生batch数、steps_per_epoch和计量。F4/T12/C11、target=volume、pred_len1、batch128、epochs20/patience5、seed2024、Adam初始lr1e-4及type1系数4/9保持。EPF35保持；M168除上文PatchTST24项结构和身份修订外保持。第三轮为84＋35＋168＝287项，最大3990 run-epochs。

### 接线、计量与启动边界

新技术attempt为`PATCHTST-depth-Urban6-exit-r1`，session为`ch3-m6-patchtst-depth-urban6-r1`，scope仍`m6-baseline-type1-followup-v3-recovery1`。独立结果根为现有formal evidence下的`baseline-unified-v3-ms-seal-m128-recovery1-patchtst-depth-urban6-r1`，日志在本增量包`followup-launcher.log`；旧许可、失败、日志、结果和两份短轨迹不覆盖。

执行顺序固定为：采用原第二轮371来源及新生命周期记录→enc1 M24准入/正式/封存→enc2 M24准入/正式/封存→另封存固定enc2新版371来源→Urban84准入/正式/封存→EPF35→M168（PatchTST enc2）→technical_complete。不重新派发MS203、M_BASE21组/旧40份、M84、M_AMEND28组/112项或其test，不等待已退出的旧owner。旧训练来源与新采用commit分开；新许可需在实际review/closure后绑定当前HEAD及本次新M_ALL配置/plan/固定政策。第三轮许可须消费新371边界，旧371仍保留为历史来源；运行期只核验紧凑边界/MAC和索引SHA，不逐worker重放371个结果或probe。各阶段完整AUTO_AUDIT一次，正常阶段自动衔接，层数效果不控制后续执行。

| 新执行阶段 | 正式任务 | 最大run-epochs | 最大Adam/backward各 | 最大forward |
|---|---:|---:|---:|---:|
| PatchTST enc1 | 24 | 280 | 50640 | 63806 |
| PatchTST enc2 | 24 | 280 | 50640 | 63806 |
| Urban六fold、H3/H12 | 84 | 1680 | 7300300 | 8197182 |
| EPF | 35 | 350 | 399000 | 467845 |
| M_ALL | 168 | 1960 | 354480 | 446642 |
| 合计 | 335 | 4550 | 8155060 | 9239281 |

已完成196项新正式任务加本次335项，共531；较原427新增104＝层数48＋fold扩展56。正常早停如实记实际轮数，不新增run/搜索额度。新五阶段共69个准入组、335个代表；含两份历史串行及失败成本的名义probe计量为4026/4026/5416，保护上界6024/6024/8104（Adam/backward/forward）；本次新计算分别为4014/4014/5400及6012/6012/8088，名义669个新短worker、保护上界1002。原成本不清零，不将probe轨迹计作正式结果。

基于第二轮同模型/域/fold/H的已记录elapsed与forward计数，仅作计划代理：按本轮最大计算量折算串行等价约261.2小时，理想q4算术约65.3小时，未计采用/probe/审计/尾观测开销。该估算不是新形状实测工期或时限保证；enc1/2借用3层吞吐参考、batch/LR/早停及共享负载变化均可能影响实际耗时，并发仍待实际准入。逐任务计量及来源见外置预算记录。

### 当前定向复验与送审材料

固定enc2决定前的19文件候选定向30/30通过，原日志、inventory/patch/acceptance/index完整保留。其退出观测、来源完整性及未变化的模型计算证据继续按适用范围使用；该30/30不冒充当前增量新字节的验收结论。

固定enc2增量定向23/23通过（独立`tests/attempt2.log`），覆盖M_ALL仅24个结构差异、MS/数值政策/计量不变、真实371索引生成与边界封存、347格逐字段保持、enc2更差仍固定采用、旧enc3及Weather替换历史保留、缺项/重复/篡改/错producer/test次数拒绝、新旧配置/plan/政策/closure许可隔离、第三轮必须消费新371边界、完整335项顺序与五次阶段AUTO_AUDIT、compact formal准入及运行期联网0。第一次23项验收有两项新夹具断言/收据注入错误，失败日志保留；修正夹具后同一最终代码和绑定证据完成23/23，不删除保护断言、不放宽比较容差。

第三轮PatchTST enc2的24项实际构造输入签名与既有enc2 CPU capture一致，逐项绑定复用32次构造/32次最小合成前向中的适用证据，本次增量新增真实构造/模型前向0；这只证明构造与输出接口，不证明第三轮新形状的并发或数值准入。新增GPU、backward/Adam、真实validation/test、checkpoint字节读取/反序列化和信号操作均0。原128份轨迹等更早材料未在本次重放，原371结果及旧失败现场保持。

真实PatchTST适配器CPU构造32次、最小合成前向32次，确认1/2层实际encoder block数及全部48任务的结构签名映射；hour/minute ETT配置差异独立覆盖。旧完整数值轨迹只作有界只读复核；本轮backward/Adam、GPU初始化/probe、正式训练、真实validation/test与checkpoint读取/反序列化均0。旧完成前缀只核验非checkpoint产物及checkpoint stat，保留原已验证checksum引用，不重算checkpoint checksum。

本轮失败验收日志和证明迭代保留，最终Passed仅对应最终同一候选字节，不拼接不同版本Passed。一次合成夹具遗漏结果根隔离，曾创建新候选根下六个空queue目录；已确认无文件、无进程、无许可/计算并仅用rmdir移除，夹具现已全部使用临时根。该事实独立保留，不声称新结果路径从未存在；当前路径再次为空缺状态。

增量集中于[patchtst-depth-urban6-repair-v1](../../../amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/baseline-unified-v3-ms-seal-m128-recovery1/moderntcn-etth1-numeric-diagnosis-v1/baseline-numeric-admission-v1/patchtst-depth-urban6-repair-v1/)：`contract.json`登记科学增量与父profile；`source-anchors.json`、`prefix-verification.json`登记371来源和退出/完整性；`urban-probe-reuse.json`区分H3复用与H12失败；`producer-delta-proof.json`精确绑定监测/接线差异和新bundle；`cpu-smoke/`与`tests/attempt5.log`为实际验收；`budget-and-timing.json`保留派生计量及工期来源；完整inventory/patch/acceptance/evidence-index供直读审核。`start-approval.template.json`只有未授权模板，`operations.sh`是review/closure及许可就绪后的操作方案，当前未生成可执行AUTH。

当前固定enc2送审增量位于上述包的`patchtst-enc2-adoption-v1/`：`fixed-encoder-policy.json`记录用户结果前决定及仅M6/服务器完整留存；新`contract.json`、`m_all-data-bindings.json`、`m_all-plan.json`及`producer-delta-proof.json`绑定第三轮24项修订；`effective-profile-diff.json`、`cpu-interface-reuse.json`、`budget-verification.json`保留逐项差异、CPU映射和计量复核。新的完整20文件inventory/patch/acceptance/index合并原19候选及本次增量，原19包不覆盖。当前未授权模板和经bash-n核验的`operations.sh`均在该子目录；未来许可须排他生成该子目录`start-review.json`，显式指定新许可路径及实际SHA。入口/session/结果根沿用本候选身份，日志仍在父包`followup-launcher.log`，不误写到子目录；旧父包模板/操作记录只保留其当时准备事实。

当前停止在未stage/commit/push、未arm/start、未启动真实恢复监督器或新GPU计算的review点。技术完成、结果review与效果结论始终分开；不宣称ChatGPT已审核本轮新字节。

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

<a id="m6-weather20-patience10-v1"></a>
## 启动前Weather20 patience10修订候选

上一节候选已真实closure于`08a79f250c4982201f4ae065b812a18caa3d5dc4`，旧配置许可/preflight准备通过。2026-10-08 01:25 UTC+8只读核验N/W/R三端一致0/0及clean，新attempt没有所属入口进程、session、result/log/launch/claimed；条件授权成立。本次是用户在启动前明确改变早停协议，不称旧None为bug、技术失败或效果失败。旧许可`start-review.json`及原准备输出完整保留，不授权未来修订commit。

M_AMEND与M_ALL独立新配置分别为`configs/ch3_round2_m_amend1_weather20_patience10_v1.json`和`configs/ch3_type1_m_all_weather20_patience10_v1.json`。严格按dataset=Weather、resolved epochs=20筛选，两阶段七模型各四H，共56项，仅training.patience None→10；其余科学profile、task/顺序、source/data/marks、batch128、seed、133政策及TimeMixer–Weather loss例外不变。最大epochs仍20，第二轮OneCycle20完整合同、第三轮type1固定4/9均不改。已完成M_BASE Weather10仍None，MS203/M84/基础287不重训/retest或倒改。

现有BestState严格MSE改善重置bad；相等不改善，保留较早best，连续10次未改善后不启动下一epoch。scheduler仍绑定20轮最大计划，保存/恢复保留真实epoch、bad、best与scheduler更新数；当前统一执行仍fresh-only，本次没有开放真实checkpoint resume。完成审计根据实际history和合法停止条件核验，可接受合法少于20轮的结果，拒绝无停止依据的截断、停止后再追加epoch及重复test。最终test仍只访问validation-selected best一次；正常早停不重跑。实际训练轮数从封存history保留，不能写成20。

配置生成与校验共用精确Weather修订投影；先核对旧完整配置及56处允许差异，再沿原数值合同验证，不把新默认动态套到历史配置。当前恢复入口只替换两个待执行config/plan引用，原SOURCE_REF、prefix-verification及numeric.CONFIG_REFS继续表示旧生产来源。新许可模板绑定Weather合同、两份新config/plan、完整427任务上限及精确producer delta；旧许可/配置不能串用。生产继承检查新增的限定变更是profile生成/校验及绑定，不改训练/评价数学；restricted bundle仍完整验证。

顺序保持基础287采用→M_AMEND probe/一次AUTO_AUDIT/112正式→第二轮371封存→Urban28→EPF35→M168。已完成84＋剩余343仍为原427；第二轮399计划运行、371有效格、第三轮231不变。M_BASE21组、旧40份测量和M84不再派发；Weather20全部28项合法完成后整批替换Weather10来源，不按指标择优。最大轮数与计算上限不增加，实际早停成本照实记录，不转为搜索额度。

本次增量集中于原准备包`weather20-patience10-v1/`，只提供非执行许可模板与审核closure后的操作方案；原许可、历史失败和完成来源保留。本轮不执行真实模型构造、GPU、forward/backward/Adam、真实数据validation/test，不读取真实训练checkpoint、重放数值sidecar或重新采用完成前缀。定向验收及完整候选字节清单由该目录acceptance/inventory记录；纯合成checkpoint与scheduler事件明确区别于业务训练消耗。

最终同一代码版本定向套件29/29 Passed（419.892秒），覆盖56项差异、133政策和上限保持、两种scheduler的真实formal循环/BestState/合成checkpoint保存恢复、合法早停与跑满20轮、截断/停止后多训/test重复拒绝、新旧许可隔离、完整混合来源恢复顺序及四个剩余阶段各一次AUTO_AUDIT；formal不回放完整probe，运行期远端查询为0。7份Python内存语法、JSON、bash-n、tracked diff-check、三份新增文件EOF/whitespace及qualified bundle验证通过。父准备上下文没有导入torch；合成scheduler/checkpoint套件使用CPU PyTorch事件替身，禁止业务模型、Adam构造和CUDA初始化，真实数据读取/训练/评价均0。现有入口fresh-only不变。

首轮28项中的两处失败/一处错误及日志保留：worker夹具缺少实际子组专用导入路径、无torch断言受测试顺序污染、旧全配置相等断言需绑定本条批准的Weather差异；保护断言保留，未加skip或放宽政策。外置机械检查曾误解no-index的差异退出码及annotated tag对象/commit区别，诊断记录保留，未改文件权限或tag。最大剩余343项仍4270 run-epochs，正式Adam/backward/forward上限2107840/2107840/2468060，probe上限6162/6162/8240，不因patience改变增加额度。最终候选为8 modified＋3 untracked共11文件，index为空，未stage/commit/push或生成可执行许可；新result/log/session仍未创建，待ChatGPT实际字节审核。

<a id="m6-moderntcn-uv-v1"></a>
**本轮：v3审核裁决与v4 TFT辅助内容实现（统一送审，尚无真实效果）**

v3已审REPORT SHA-256=`e141bddc8a9a95eb01050ee4aadd5c1f5e882ffe7d8984e0a3d4dc92b1c870f0`，manifest=`79b9f890a5f728b0f69a549f21636ad42fc8c23d2d0d8d06e86dc56dda7c1c7b`。接受范围是保存产物/结果与原声明的有限机制观察；独立U/V有效证据及组合退化并存，原因未唯一确认。本次保留V3、替换U是用户委托ChatGPT后的唯一科学选型，不回写旧报告/许可/结果，也不继承v3的8/10为v4通过。

新身份为`moderntcn-uv-dev-v4`；revision为`u=tft_target_conditioned_auxiliary_patch_selection_v4`、`v=target_state_conditioned_low_rank_temporal_filter_post_ffn1_v3`、`fusion=post_ffn1_v_then_auxiliary_content_target_update_pre_ffn2`。U取有序aux的pre-RevIN历史原生patch：Urban p1/stride1/N12，EPF p8/stride4/N24，右端仅复制最后已观测值4次。各aux独立p→8 embedding及GRN；selector输入aux-major展平J×8并以当前h_v目标D64 latent作context，GRN输出8维后score8→J、沿aux softmax，加权内容由零初始化W_out8→64只写目标槽。target不进入辅助Value，其他槽字节不变，无虚拟token或未来观测。

GRN为ELU(L_in(a)+L_ctx(c))→L_hidden→sigmoid(L_gate)⊙L_value→skip+LayerNorm8；LN eps1e-3/affine/float32，context投影无bias，其余四Linear有bias，selector有独立J8→8 skip。U内部dropout0；除W_out直接零外Linear均Xavier/bias0，LN gamma1/beta0，独立CPU子流17011，V子流17021不变。V继续过滤原h，U context来自h+ΔV；没有冻结、detach、额外loss/归一化或预测端修正。出口初始零仍执行真实计算图，内部loss梯度在出口打开后激活；不把weight decay当loss梯度。

来源为TFT arXiv1912.09363v3及官方google-research固定commit `8bedbbd982e7dcf5c5280e6ae2f0769295c3369e`。官方`tft_model.py` SHA=`53f0046e12b9b79ffea096516b2278774e65d760df684f97600341b9abef9a2c`，Apache-2.0 LICENSE SHA=`cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`，PDF SHA=`79580118987a028db9c83c9bfedbc5b8bc126d40c357bf0284dcf044299db5b5`。保留版权/许可，记录apply_gating_layer/add_and_norm/GRN及lstm_combine_and_mask变量选择函数；本次没有搬LSTM/attention/future/static encoder/quantile loss或TFT训练协议。scalar→patch、static context→当前target latent、aux-only内容、8维selector后score、target-only零出口均为本项目改造，不是整套TFT复现或效果保证。细粒度exo未必更好的风险保留，不声称原创性或因果互补已证。

参数公式U=J×(449+8p)+1272实测一致，联合较v3多4032（Urban F4）/688（EPF）。新U主要激活BNJ8、selector和target delta需反向保存，V的BCD4N basis不变；参数/梯度/Adam、原workspace与carrier开销分别列明，不以单张量或参数量冒充GPU峰值。已有正式q4、CPU证据及差异评估支持准备q4复用；本轮未重新GPU实测、不安排探针或强制降q1。

非执行配置`configs/moderntcn_uv_fold6_epf_aux_content_v4.json`绑定40项实际运行依赖；九格V3保存文本保留原task/producer/config/source/manifest/metrics/spec/seal与SHA，只是等价引用，不是v4 fresh或新test。新U和UV共18项必须fresh，旧U和旧UV不能代替新成绩，也不能warm-start。B_cfg只读复用；MS单fold6、EPF五市场、7/10严格胜B_cfg、目标一无B链/逐H全胜及目标三同revision合同不变。F1/F2/F3只保留接口/scope，不执行。所有新结果Not evaluated。

完整/本轮增量diff、来源逐项迁移、11项验收、计数修复与消费、V-only等价、资源和命令模板集中外置。模板commit=null/review_approved=false，配置execution_authorized=false；本轮没有可消费AUTH/真实attempt。新字节审核通过后再同轮精确closure→实际commit/source绑定→正式AUTH/轻量preflight→用户启动交接，不为回填字段机械再审批。本轮停止于实现和定向验收齐备、待审。以下旧待审记录保留历史时点。

**2026-10-25：v3完成后审计与“单臂有效、组合退化”诊断（新材料统一待审）**

训练/closure commit `683d6725ee710c4d894ee0124c34cc0d434402ca`，candidate `moderntcn-uv-dev-v3`，原配置/AUTH/runtime与40项执行依赖保持。18fresh V/UV、六波、228轮/700626更新/18次原test；9格U仅来源引用，不计新训练或test。保存产物、strict-best/patience、每步LR/beta1与Adam时刻、BN/RNG/finite、checksum/seal/claim/receipt/registry及数据/scaler/profile自检通过；36份best/last各CPU反序列化一次，未恢复训练。owner和末波child退出、staging空，无failure/STOP；progress末尾RUNNING保留其快照含义。中间波逐PID/start历史未完整留存，波次由runtime、逐run回执及固定派发源码支持，不伪造调度时间线。

Urban仅fold6四H等权：U test=0.4292047424030744/0.44044568609070744，v3 V=0.4318767256921122/0.4428699701890627，UV=0.4406210605788158/0.44583574625479233。目标一覆盖完成，UV两删除对照的平均正向贡献均未支持；H6/H12占UV−U净MSE差额77.9074%，两格UV均best1/总6，MSE/MAE最小轮均1，最后完整validation MSE只较best差0.732%/1.114%，不称技术失败、不补轮。H3/H9行为不同。目标二对B_cfg为8/10（MSE4、MAE4），四个获益市场UV亦优于两单臂，PJM相反；目标三未执行。test作为开发反馈如实登记，不称盲测或单seed显著。

主干已有DW时间加工、FFN1 latent混合、FFN2变量交互和全历史head；Urban适配N12/K7/r1、EPF N24/K51/r1，不套用C.1。固定样本提示状态相关残差和独立U/V校正重叠，也存在各自独有改善；不能预先指定“U管状态、V管时间”。context的ctx0/独立train均值替换在H6改善、H12/BE恶化，PJM即时作用小。真实kernel有动态变化，EPF route高饱和在失败/成功市场均存在，不能据此调温度。合法mask分解显示V同时具有位置增益与邻域差分，二者抵消且非正交；没有证据直接删除增益、改zero-DC或加正交loss。96次原序缓存重建exact，代数重排按预登记舍入界核验，不倒改旧失败或用于放宽效果比较。

H12固定UV关U、关V均恶化，独立UV却较两单臂差；off路径不等于独立训练消融。H6/H12 last端点中P_s、BN和核谱发生共同变化，但有限eval端点不能证明梯度冲突、BN错误、冻结主干或延长训练有效。BE抽样MSE排序与完整test不同，保持固定样本并明确不外推。结论分别为：U保留独立有效能力；V保留本版单臂改善证据；组合需要重新审视功能分工，继续机械加状态/换位/调门或rank依据不足。只交付能力需求、反证与验证边界，由ChatGPT比较机制后选择，不强行产生v4公式。

本轮预登记核心H6/H12/PJM/BE共8个best模型，条件触发H6/H12 UV last共10构造；36次唯一CPU加载、276完整forward、15840次呈现，另96次缓存局部滤波与4次局部context校准（806窗口）。任务窗口3518条、不同物理历史区间3188，呈现数非独立样本数。观察器输入/输出/参数/持久buffer/RNG保全；旧B/U/v1/v2/C.1 checkpoint读取0、新CSV读取0，新test、完整backward、局部autograd、optimizer、训练、GPU均0；无模型/状态审计失败或复验重读。报告及逐份SHA位于`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v3-postrun-review-diagnosis-r1/`。工程Passed、新结果review待审、科学数值条件分开登记；本两文档与材料统一送审，不单独closure，不生成新AUTH。

收口元数据工具故障（shell提示前缀的JSON解析、补丁块顺序及文件用途映射缺项）均保留失败输出与修复记录；失败未改实验产物；已定位修复并保留记录，未增加模型/状态读取/GPU消费、未改代理或环境。

**2026-10-24：v2结果审核裁决与v3目标状态路由实现（新实现待审）**

用户转达ChatGPT已直读审核v2统一报告`e6182f54839ca46bfa92607d7383ba62817bf73d456d02590033d071c1081841`及manifest `b78911186752f1e6feefd65de979f2bcd103f1ac11053adb75543b8119858759`，保存产物/结果和限定机制观察review Passed，边界不包含CUDA恢复/训练重演/唯一训练根因。18fresh/9U引用、原训练244epochs/812310updates/18test与诊断失败成本原样保留；目标一V正向平均贡献未支持、目标二B_cfg 7/10、目标三未评价，不继承为v3结果。

v3冻结图为原RevIN→stem→DW/BN/FFN1/dropout一次得到h，`s=U(h,X)+V_new(h,Z,X_y)`→原FFN2/H skip/head/denorm。唯一改动：`s_y=[mean_L X_y,log1p(mean_L((X_y-mean_L X_y)^2))]`，`a_j=tanh(W2 GELU(W1 Z_j+P_s s_y))`。状态float32、总体方差，target由schema读取Urban0/EPF2；无epsilon/clamp/detach/新增状态，非finite明确拒绝。V-only独立计算本次目标状态，显式传X，无跨batch缓存；V滤波仍读h，U与V不读取对方更新。`P_s[16,2]`直接零Parameter，原B/U/V初始化顺序与CPU子流17011/17021不变。A零导致首步P_s合法零loss梯度；出口开启后P_s/router有有效梯度，不以weight decay代替证据。

身份`moderntcn-uv-dev-v3`；revision为`target_conditioned_window_state_modulation_v1` / `target_state_conditioned_low_rank_temporal_filter_post_ffn1_v3` / `first_block_post_ffn1_parallel_u_v_pre_ffn2`。独立`moderntcn_dev_v3`、配置`configs/moderntcn_uv_fold6_epf_target_state_v3.json`及入口；40项执行依赖覆盖真实v1/v2只读复用，旧字节不改。训练循环、Adam/OneCycle、loss、strict-best/patience、恢复、test-once、封存、owner/STOP等沿已审逻辑，仅版本身份分派。配置execution_authorized=false，非执行模板commit=null/review_approved=false。

13项定向CPU检查均通过，无模型验收失败或容差放宽；两次只读路径定位错误无模型消费，保存记录；两处v2/v3说明文字修正经AST主体一致核对并刷新来源绑定。零P_s且旧分支非零时train/eval公共输出/梯度/BN/RNG与v2 exact；U-only与v1/v2两步Adam/scheduler/state exact；非零P_s/A fixture证明目标状态改变route/实际核/ΔV，非目标均值变化在固定Z下不误触发。覆盖F1–F4、EPF P24/target2、无跨batchcontext、条件核独立参照、版本拒绝及观察器保全。实际完整/局部消费见文件头及外置events，不重跑历史训练/恢复大全套。

静态训练体比对首次遗漏独立证据目录名的映射而失败，已保存原工具/输出并仅补齐该精确路径映射，复验通过；没有模型或训练逻辑修复、额外数值消费或容差调整。启动脚本实际0755，继承core.filemode=false引发的patch mode提示用单命令filemode核验处理，不修改Git配置或index。

实测U不变；Urban F4 V=572、UV增量2382；EPF V=2076、UV增量3686，比v2 V/UV各加32，U-only加0。新增O(BL)目标统计、32B投影乘加、16BC广播加法；主滤波张量shape不变，但反向保存和输入引用可能变化，静态小张量不是完整GPU峰值。结合v1/v2正式q4与定向工程证据准备q4复用，分别记录OOM和数值风险，本轮未重新GPU实测/无探针。router显式获得目标状态不等于必然互补，可能与U重复、发生饱和或损失已有EPF收益；不承诺目标改善。

科学范围仍单fold6 `[3475,3909,4344]`/275节点/L12/P1/H3/6/9/12及五EPF L96/P24；B_ref=B_cfg，C.1历史保留。非执行映射为九格v1 U来源引用（不计v3 fresh/test）＋十八V/UV fresh，六波4/4/1/4/4/1、q4、360轮/1245080更新。v2 V/UV不复用、不warm-start。目标一无旧B链/逐H全胜，目标二10格至少7格严格胜B_cfg，目标三同revision fold6 F4；F1/F2/F3仅scope，不执行。

来源如实登记：现有CondConv-inspired受约束条件滤波；Non-stationary Transformers为统计回注到算子条件的机制参考，不复制De-stationary Attention、不声称其理论保证。按用户转达，本次选型已比较CondConv、Dynamic Convolution、Non-stationary Transformers、DUET、TimeXer；未引入DUET/TimeXer执行模块，本轮不重新选源/下载。32参数本身不构成独立原创贡献。全部新字节统一交ChatGPT审核；通过后再同轮完成精确closure→实际commit绑定→正式AUTH/preflight→用户最后启动，不为填commit/许可另加机械审批，本轮不跳过实质审核。

**2026-10-24：v2首批完成后审计、v1→v2定向机制诊断（新结果统一待审）**

当前科学口径：增强仅MS、Urban只用fold6端点[3475,3909,4344]，275节点、L12/P1、H标签3/6/9/12；目标一独立UV/V/等价U逐H及四H等权，不恢复U<B/V<B链或逐H全胜。B_ref=B_cfg=cfg_adapted_v1；目标二五EPF双指标十格至少七格严格胜，平局不胜/缺项不缩分母，无各指标70%。目标三本轮未做F1/F2/F3，未来F4必须同v2/revision。C.1保留历史、未采用，不加载其checkpoint或更换主表。

实际runtime为`runtime-4fccd9d6bd2f43f6a49cdd00684e51c3.json`（SHA `aa531f9eb3604e643caa77c4f40a835816c9048eaf1d8e681c5dd0a4566cc144`），AUTH SHA `19872f55fdfd1391fad877859e4a43a6aed4ce8a002bd3f6dfeab9d8e29c6077`，配置SHA `6cd579123469000206729b9d4ad3fe27722fc6774e2443eb2eb8ac9abc8ac7a7`，训练commit `170c27e685686ed1932849090b65c9c79de36e8c`。18fresh/18回执/六波、staging空、无failure/STOP、owner及末波child退出；末progress RUNNING是历史快照。Python/系统checksum与493文件/414payload、严格best、完整LR/beta1、Adam/scheduler/BN/RNG、test-once/receipt/seal/registry及36状态CPU审计自检通过。244轮/812310更新/18次原test，9U保留原producer/来源身份且不算fresh；18项data-metadata与同格原U完整相等。未重审v1的54状态或进行CUDA恢复、训练重演。

Urban U/v1UV/v2UV test MSE为0.4292047424030744/0.4341288578374556/0.43411778425054237，MAE为0.44044568609070744/0.44336884940291527/0.44436801109805873。v2 UV对U test +1.1447%/+0.8905%，validation +1.7869%/+1.0666%；V单臂对v1 V改善约0.9169%/0.5524%。H12 UV best1/总6，validation双指标最小均在1，正确早停；占四H净MSE差额81.8%，不因此补轮、改LR/patience或换best。EPF v2对B_cfg 7/10（MSE3、MAE4），BE对v1 UV差8.2023%/10.4933%，NP联合弱于两单臂，PJM/FR/DE仍有联合正向证据。工程Passed沿用、新结果review待ChatGPT，数值条件与机制价值分开。

主干已具备DW时间加工、FFN1特征混合、FFN2变量交互及全历史head；不能声称没有时间/变量建模。U独立有正向证据，联合状态中BE关闭U却改善，不能默认U总正确。v2核有实际动态变化；部分Urban原子能量集中，但单位方向有效秩仍近4，H12 UV/BE并无四基塌缩。H12/NP固定train均值route优于动态route，BE相反。非零训练状态确认两支路同h/直接输出独立、原序合并exact，预测端交互仍存在；offV不是独立U、offBoth不是B，未测梯度冲突或唯一训练根因。有限样本H12/BE较高历史状态组是具体反例，未把它外推为test尾部事实。

本轮15个CPU eval模型，1121次完整前向、28359次呈现；复用E1样本/预测，无CSV读取/refit，5699条任务窗口对应4498个不同历史区间，非统计独立样本。36状态审计后外置event字段冲突、构造前失败，保留失败与预计数校正，增量登记重读15必需状态，累计51读取；另两次纯文本字段定位错误及修复保留，新增模型成本0。观察器output/input/持久state/RNG exact；原序局部合并exact，重排式浮点残差如实登记，未放宽旧断言。新test、完整backward、optimizer构造/更新、训练、GPU查询/负载均0；3项缓存dropout解析不冒称训练因果。

裁决建议：保留U；V条件时间加工需求合理，但不推荐继续仅换位或调rank/温度；一条未实施、待ChatGPT选择的假设为在原router hidden加入目标历史状态项P[mean(X_y),log1p(var(X_y))]，P为16×2，增32参数，其余核/插点/U不动。依据是具体状态分组残差与缺少显式目标状态的接口，最强反证是U已供状态、BE动态V即时有利及优化/选点解释仍未排除；不承诺有效，可能损失现有EPF收益。若ChatGPT不接受该证据，应返回多机制/文献比较，不能直接当v3执行。完整REPORT、配对数值/轨迹、审计、缓存/干预/失败/消费及SHA清单统一位于`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v2-postrun-review-diagnosis-r1/`。模型/源码/配置/许可/原产物/旧证据原字节保留，建议未实施，无新候选或训练许可。本轮仅两权威文档修改、未暂存、index空，统一交ChatGPT审核。

**2026-10-23：v1审核裁决与同层并联v2统一送审**

用户转达ChatGPT对v1扩展27项产物/结果及限定机制诊断的review Passed，原REPORT SHA-256为`692e5a64cb5c0c793d8eb1935f454528130dbd2a0b1fb83407659e6bcc0c5985`。两项float64展开式比较仍记Failed；原序float32缓存重建exact，不倒改失败、不扩称CUDA恢复或训练重演。目标一消融覆盖完成，但V加入U的平均正向贡献未获支持；目标三未评价。最新决定将B_ref固定回B_cfg=`cfg_adapted_v1`，v1目标二对该B严格胜7/10格（MSE4/5、MAE3/5），不与C.1的历史分项混用。C.1实验保留为已执行、未采用，不改主表、不追加工作或删除其结果。

增强Urban模块/输入消融只用现有fold6，端点[3475,3909,4344]、275节点、L12/P1和H标签3/6/9/12、train-only scaler不变。fold1/4是历史，旧未执行输入消融superseded。目标一报告独立UV、w/o U=V、w/o V=U逐H与四H等权双指标，不恢复U<B/V<B、逐H全胜或U/V排序；覆盖与正向贡献分开。目标二五EPF/H=P24共十指标格，UV对配对适配B至少七格严格胜、平局不胜、缺项不缩分母，不加各指标70%。目标三仍为fold6四H等权的双指标F4<F2<F1、F4<F3<F1，F4必须同候选/revision。

当前只实现用户批准的`moderntcn-uv-dev-v2`：revision U=`target_conditioned_window_state_modulation_v1`，V=`input_conditioned_low_rank_temporal_filter_post_ffn1_v2`，fusion=`first_block_post_ffn1_parallel_u_v_pre_ffn2`。X是loader已缩放的RevIN前历史，Z是唯一一次原生RevIN后的历史；H为stem输出。原dw及内部BN、block.norm、FFN1/drop2原序各执行一次得到h，先计算ΔV=V_filter(h,Z)，再h_u=U(h,X)，s=h_u+ΔV，最后原FFN2和H残差/head/一次denorm。两支路都读同一未修改h；没有把旧V(H,Z)搬到U后，也没有新增目标状态router。去掉V扰动直接通过U乘性算子的路径，不等于共享主干、FFN2/head或优化梯度完全解耦，不保证泛化改善。

v1的U/V公式、初始化和模块源码直接只读复用，主干构造/RevIN/stem/head适配继承原实现；新增版本只覆盖首block接线。Urban/EPF适配结构、CPU RNG偏移17011/17021、loss/Adam/OneCycle/seed/batch128/选模/test-once保持。v2独立spec/config/入口/worker/封存/汇总身份，真实依赖37项全部绑定，v1/v2同shape状态及旧许可不能串用。原训练循环和恢复/封存主体按namespace版本化，差异逐项外置；v1执行文件及旧配置不改。

13项必要CPU用例最终覆盖通过，累计14次执行、无skip；非零fixture证明U和V的滤波输入确为同一个h且router只读Z，双分支局部独立且梯度回h，Urban/EPF中性输出/公共state/dropout RNG/BN/梯度exact。U-only同v1非零train/eval/梯度/少量Adam更新exact；共享基与显式核float64参照按预登记1e-10/1e-9通过。实际合成worker完成严格best、一次合成test、原子封存和epoch恢复exact，成功不可覆盖，身份负例保留。首轮aux元组/列表断言失败仅修测试容器比较；forward计数截获遗漏按保存routes及精确调用流补账，未为补数重跑模型。累计30构造、90完整forward/179呈现、47backward、27主进程Adam（子进程0）、6局部autograd、6/6合成checkpoint保存/读取；真实checkpoint、真实数据模型前向、真实test、真实训练、GPU查询/负载全部0。

实测新增参数相对v1为0：UrbanF4 U/V/UV=1810/540/2350，EPF=1610/2044/3654，作者注册但未执行参数保留。V的[B,C,D,4,N]基滤波临时量在batch128下Urban16.5MiB、EPF9MiB，仅为单个forward中间量，不是总GPU峰值；V现在也经FFN1/norm/dw反传，激活存活/反向调度可能变化。资源说明复用既有v1真实q4、历史Urban/EPF及有效CPU证据和形状差异，未新做GPU/OOM/数值/恢复探针，也未声称延迟或峰值实测。

非执行配置`configs/moderntcn_uv_fold6_epf_parallel_v2.json`精确18项fresh V/UV（Urban八项＋EPF十项），九格U以原task/producer/config/source/result SHA显式引用，标记等价单臂复用，不能冒称v2重训完成。六波依次V Urban4、V PJM/NP/BE/FR4、V DE1、UV同序，预算360 run-epochs/1245080 updates；v2 fresh完成0。另保留fold6 F1/F2/F3十二格scope，不可启动；未来F4仅引用v2 UV四格。B只读复用，不重训/重test。三个v2效果目标Not evaluated。

统一外置证据`M6-ModernTCN-UV-v2-same-level-parallel-implementation-r1`保存起点快照、相对HEAD完整diff及相对本轮起点增量diff、来源/代码差异、失败/消费、参数资源、九U映射和18任务非执行模板。模板commit=null/review_approved=false，无真实AUTH/runtime/attempt。本轮仅实现及自检，新字节待ChatGPT实质审核；不stage/commit/push，不创建新worktree、不改C.1/AMD/作者源码/AGENTS，也不恢复撤销的网络准入实现。通过审核后下一任务连续closure→实际commit/source绑定→正式AUTH/轻量preflight→用户启动交接，本轮不越过审核训练。


**2026-10-23：单fold6科学范围与27项完成后诊断（本轮新结果/机制待ChatGPT审核）**

实际训练及现场HEAD为`900fb19c792251ca549ef8c4cebcc3b725d5fe22`。扩展配置/runtime/AUTH分别保持`fd0bdca9637ea19858ba60ec7f63136fc881846f3e288ce59392dd7feeb418b4`、`cfa6ec00e2466c6ac96cf5d9b9df093f35eaa51854e6e43825f218ec6ef6e525`、`23ad4385142af999ff46142ac875b29934c6d1ce47b4e7af3359c051517f6991`。27项/9波/回执/registry、全部736文件与621 payload双checksum、54份CPU保存状态、history/journal/OneCycle/严格best/test-once自检通过，382轮/1272765更新/27test。一次旧launcher远端Git失败发生于worker创建之前；当前27项均fresh一次，无训练恢复/重试。原失败日志保留，不计成第二次科学训练。CPU状态审计不等于CUDA恢复或轨迹重演。

用户最新决定替代旧增强Urban双fold合同：今后模块及输入消融仅用现有fold6，切分[3475,3909,4344]、275节点、L12/P1、H标签偏移3/6/9/12、train-only scaler及seed2024不变；不把validation/test并入训练。旧24项fold1/4 F4完成记录保留历史，旧24项未执行输入消融superseded；当前fold6三臂12＋EPF三臂15已完成，未来fold6 F1/F2/F3四H共12项只登记scope，不执行、不建许可。F4只能复用同候选/revision四格。C.1六fold和M基线计划不受影响，原配置/AUTH/runtime的旧目标不回写。

当前目标一按独立UV、w/o U=V、w/o V=U逐H及四H等权报告双指标，不再要求U<B/V<B，也不增加逐H全胜。fold6 U test MSE/MAE=`0.4292047424030744/0.44044568609070744`，UV=`0.4341288578374556/0.44336884940291527`；UV对U为+1.1473%/+0.6637%，validation为−0.0305%/+0.1238%，H3/6/12 test较差、H9较好。覆盖完成，但V加入U后的均值正向贡献未获支持；U加入V有正向结果。B_cfg=旧cfg_adapted_v1用于归因；B_ref=paper_C1_r8_v1用于目标二。本轮现场C.1已53/53封存，五EPF UV严格胜新B_ref 7/10（MSE3、MAE4），数值满足，不跨市场平均误差、不加每指标70%或其他baseline门槛；新结果review待审。目标三fold6输入消融Not evaluated。

外置`M6-ModernTCN-UV-v1-fold6-epf-review-diagnosis-r1`预登记后执行CPU单线程eval诊断，Urban四H及PJM/BE/NP同格配对；6652条task-window含5168分析＋1484独立train校准，共5094不同物理历史区间。实际87次checkpoint读取（61不同文件、工具故障后26次重复）、29次模型构造、1971次完整forward/49737次呈现；test/backward/optimizer/训练/GPU均0，C.1 checkpoint访问0。原scaler直接复用，无refit、无test样本。观察器输出/输入/持久状态/RNG exact；736原文件及执行源码最终保全。外置导出上下文和零余弦分母工具错误保留修复成本；70项局部公式比较68通过，H6两组float64展开式超原2e-5界仍记Failed，原float32顺序缓存重建四路径exact，未放宽容差或改模型。

直接证据支持U实际工作，V也非静态/四基塌缩：UV固定validation核中心化/总RMS约0.878–0.987，有效秩约2.954–3.664；V关支路或固定train均值路由的作用依任务而异，BE/PJM有保留价值、NP未显示一致动态收益。同X关V时q exact，U的(I+M(q))对目标扰动RMS倍率约1.19–2.89；这是局部耦合证据，不能推断梯度冲突或训练根因。有限样本在部分格与完整成绩排序不同，不能外推test尾部/节点偏差。27项best后最坏validation MSE约为best的1.355倍，未见严重失控。建议保留U，优先交ChatGPT审查V移到U后、FFN2前以解除直接乘性传递的接口假设；次选为V补充目标窗口状态条件，两者均未实施、未冻结。EPF收益风险及反证条件见统一REPORT。仅两文档增量未暂存，index空，不单独closure、不选新来源、不启动下一训练。

**2026-10-22首批审计与27项扩展增量**

首批实际训练commit为`5ff5ea17732891556270789aeb504e2ab356ecfb`。24项身份/回执、runtime六波、成功registry、封存成员及Python/系统checksum、history/update journal、严格validation MSE best/早停/OneCycle LR和beta1、一次正式test与SSE/SAE/elements完成核查；48份best/last在CPU只读检查保存参数/BN buffer、活动Adam、scheduler、BestState、进度与RNG格式。未构造模型/optimizer、未执行恢复或CUDA恢复/训练重演；key/shape按保存参数名与同臂/同profile状态核对，不能冒称新模型载入验证。655原文件保持。首批370轮/1038487更新/24test；原53计划Partial不等于首批失败。fold1四H MSE B/U/V/UV=0.8650273498189662/0.6931539555005402/0.8830968997368811/0.6979381201822326；fold4=0.4522708881943221/0.45604676476825784/0.44917289620612294/0.4267214645241454。双指标逐fold目标一Not met，目标二/三Not evaluated；不把效果差判作技术失败，新结果review Pending。

独立配置`configs/moderntcn_uv_extension_f6_epf_3arm_v1.json`使用`task_scope=development_extension_f6_epf_3arm_v1`，仍走受保护development链路。冻结27项完整科学字典/参考SHA，明确九波：U Urban4、U PJM/NP/BE/FR4、U DE1；V同序；UV同序。原53项精确网格继续校验，原配置字节不改；五EPF UV完整task身份复用、fresh尚未执行，另增22身份，不覆盖原24完成任务。fold6继续M1累计六个月切分端点[3475,3909,4344]，train-only scaler/节点275/特征/标签不变；四H每epoch batches=7435/7429/7422/7416。EPF各285 batches/epoch，C3/target2/有序aux[0,1]/L96/P24保持。预算独立复算Urban1782120＋EPF85500=1867620 updates，27×20=540 epochs。9格B只读引用第二轮cfg_adapted_v1，不重训/重test、不切换paper结构。

原目标一仅fold1/4，fold6只是扩展比较；新汇总分列fold6各H/四H均值及EPF B/U/V/UV。目标二仅完整五市场UV十指标格至少七格严格胜B，不因缺项缩分母，不加单指标70%门槛；U/V仅解释独立贡献。目标三原八格与输入消融保留，本批不判目标一/三，也不把27项技术完成误判为全局覆盖失败。checkpoint/成功registry/test-once/恢复/STOP保护不改；旧AUTH/runtime仍绑定旧执行字节，不能用于新配置。五EPF UV保留成功身份键，防止将同一科学任务重命名择优重跑；正常恢复还须原独立恢复许可及检查。

资源沿历史ModernTCN-UrbanEV-F4与ModernTCN-EPF-compatible-0 q4 Passed、首批实际q4完成及已审CPU证据进行差异评估，免新增探针。fold6每步形状不变；EPF共享基激活约9MiB、Urban约16.5MiB，EPF动态基卷积MAC为110886912、Urban30277632，计入反向/优化器风险，均非GPU总峰值实测。正式启动仍保留UUID、余量、共享锁和owned保护。17项定向元数据/fixture通过；两项外置交付工具问题（schema字段名、模板继承旧路径/额度）修复并保留前件/日志，追加模型消费0。原15项模型验收复用，不重跑；本轮48次已训练状态CPU读取之外，模型构造/forward/backward/optimizer/真实新test/GPU均0。

统一外置证据：`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v1-extension-f6-epf-implementation-r1`。含审计、精确27配置/31执行依赖、9B、9波、来源与预算、资源复用、测试失败修复及patch/SHA。规划新结果根`moderntcn-uv-v1-development-runs/urbanev-f6-epf-3arm-uv-v1-extension-v1`、attempt=`urbanev-f6-epf-3arm-moderntcn-uv-v1-extension-r1`，均未创建。非执行模板commit=null/review_approved=false；本轮只送新范围实质审核，不生成可消费AUTH，不stage/commit/push。以下历史记录保留原时点。
## ModernTCN-U+V v1：实现、接线与有限CPU送审（2026-10-22）

本分支独立工作区 `ModernTCN-U+V`，分支 `M6-ModernTCN-UV-v1`，父HEAD `0fe117517a3dff6bc4f3c66a9228a72dda06f525`。起点读取源TimeMixer当前已提交HEAD及clean状态后创建；没有checkout/reset/stash/clean原工作区，没有操作TimeMixer进程、队列、锁、环境和输出。共享作者ModernTCN固定 `56a9a2c018385cd5acef015378cae7f084d1b11c`，只读继承实际Model并版本化最小forward适配，原生公共state key和注册但未执行的参数保留。

科学合同及公式见 [canonical对应专节](../AMD_EV_Thesis_Final_Implementation_Plan_v2.1.md#m6-moderntcn-uv-v1)。U只在首stage首block的FFN1 drop2后、FFN2前作用目标槽；V只在原native_dw与block.norm之间增加输入条件低秩时间滤波，router只读一次RevIN后的Z。原RevIN、两个原核内部BN、block.norm、FFN顺序、残差、head与denorm均保留。实际B宽度为dims[0]=64，禁止用无关d_model512解释模型。初始化先完整B，再独立CPU子流；关闭模块不注册该模块参数。默认关闭的诊断只返回统计，U分开乘性项/状态项/实际Delta，V分开router/DeltaK/DeltaH及eval下BN仿射后的Delta，不追加模型前向；train下不把局部BN affine式冒充批统计反事实。

新 `configs/moderntcn_uv_round2_resolved_v1.json` 逐项绑定53项MS、13格ModernTCN B与原第二轮profile/结果SHA、8格同版F4复用，从0完成开始。目标一24、目标二5、目标三新增24；其余29保留，不继承TimeMixer完成数。目标二是五EPF十指标格至少七格严格胜B，缺项不缩分母，平局不算胜；目标一逐fold四H、目标三八格口径不变。B_ref=B_cfg=adapted；`configs/moderntcn_paper_C1_r8_v1_pending.json` 仅登记13格未来备选B结构，switch_approved=false，保持第二轮训练与正则化，不执行、不切B_ref、不改主表。

15项独立定向用例最终通过，累计17次执行、0 skip。核心源数学与接线在首轮已通过；一项off/off更新测试误调Schedule方法名产生1次失败更新，改测试调用后复验，未放宽exact。四臂合成CSV经真实worker完成train/validation/严格best/一次test/校验封存；另一UV合成任务在合法epoch边界恢复，模型/Adam/scheduler/RNG/BestState/进度与连续执行exact。合成test共5次，不是正式test。新增schema、target/aux、候选/source身份拒绝在live写入前完成，成功产物不可覆盖。原训练器保护逻辑只作独立namespace版本派生，未重跑历史工程套餐或真实模型。

实际消费：30次完整模型构造；130完整合成forward、259次样本呈现；58backward和6次局部autograd.grad；48主进程Adam、子进程0（上限64）；合成save/load为15/9。初始外置计数工具有截获缺口，原日志不覆盖，`initial-counter-reconciliation.json`用五份保存routes和精确测试控制流补齐；后续使用调用profiler。另保留首次元数据profile字段定位错误及global_lock/EPF前缀元数据修正。真实checkpoint、真实数据模型前向、真实训练、正式test及GPU均0。

实测新增U参数：F1/F2/F3/F4=1685/1735/1760/1810；EPF=1610。V：Urban540、EPF2044；UV UrbanF4=2350、EPF=3654。包括作者未执行注册参数的UrbanF4 B/U/V/UV总数169929/171739/170469/172279，EPF126268/127878/128312/129922。共享基卷积的[B,C,D,4,N]临时张量与额外MAC单独统计；参数少不代表无代价，未测GPU峰值/并发性能。paper_C1_r8_v1的UrbanN3/Kv5及EPFN24/Kv47仅合成shape通过。

统一外置包：`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v1-implementation-r1/`；含REPORT、源/科学配置对照、任务/B引用、完整patch与manifest、失败与消费、合成产物、参数/资源、非执行模板及操作命令。非执行首批24任务按U fold1/4、V fold1/4、UV fold1/4分六波，最大480 run-epochs/1485300更新；q4只是待审分组，资源review Pending，commit=null、review_approved=false、resume=false，无真实AUTH/attempt。工作区变化全部未暂存、index空，不commit/push；新工程字节和三个效果目标分别为待ChatGPT审核／Not evaluated。
