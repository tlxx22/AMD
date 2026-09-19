# M5：模型筛选与结构冻结

2026-09-19当前状态（M5 §24）：RSS复合判据与后端条件数值准入后的10组补测已完成并审核，54/54组技术准入全部Passed（44继承＋10本轮新通过）。本轮实际480 Adam/640前向/480反向，低于594硬上限；80条worker轨迹均finite、0 OOM、0资源归属失败、0受限审计deny。Passed并发为q4 44组、q2 3组、q1 7组。报告与当前protocol/code/environment/hardware/许可、kernel admission均匹配。该结论仅为资源/并发/短轨迹数值技术gate Passed，不是效果gate，不冻结J；M5仍未Closed，M6未开始。以下§23及更早为历史时点。

2026-09-19当前状态（M5 §23）：按用户本次RSS复合判据及条件性数值授权完成接入。RSS需连续增长且净增>32MiB、平均>1MiB/step，并经24步窗口（前6步warm-up）末8步无平台才作CPU持续增长阻塞；显著短窗增长先标NeedsLongWindow，不冒充泄漏。三组24步代表均平台。四个数值作用域的固定Conv1d输入/权重/上游梯度重放均实测默认梯度非逐位、仅诊断切换cuDNN确定性后重复exact；生产确定性设置未改。TimeMixer/Exchange、ModernTCN/Weather采用全浮点1e-4及指标1e-6；ModernTCN/ECL初始界失败保留，条件证据成立后前瞻性采用state1e-3、validation1e-6、loss abs1e-6+rel1e-5并经新轨迹通过；原RSS阻塞还掩盖TimeMixer/Weather数值差异，已在原10组范围内补2串行验证，state1e-4、validation1e-6、loss abs1e-6+rel1e-5通过，正式并发仍待probe。32次CPU方法通过；21诊断worker实际180 Adam/294前向/252反向，机械余108；原48次隔离grad漏计单列补账、钩子修复，旧记录不改。44组155代表继承核验，10组40代表补测只用594既有余额，核心<=480，资源性候选回退按固定顺序限余额，不增加预算。495/5340、全部正式profile与训练数学不变；尚未启动新probe/J冻结/M5关闭/M6。以下§22及更早为历史时点。

2026-09-19当前状态（M5 §22）：20组补测已完成并审核，54组终态44 Passed/10 Blocked（34继承＋10本轮新通过）。实际846 Adam/1128前向/846反向，低于1440上限，新增144额度未实际动用；0 OOM、0资源归属失败。剩余10组中7组由CPU RSS四点严格递增触发，GPU allocated均稳定；3组为数值准入失败：TimeMixer/Exchange超过原具名单张量1e-7或白名单外exact，ModernTCN/Weather与ModernTCN/ECL仍exact不一致。完整report与当前protocol/code/environment/hardware/许可均匹配。结果review Not passed；J未冻结、M5未Closed、M6未开始。以下§21及更早为历史时点。

2026-09-19当前状态（M5 §21）：用户明确确认三组限定数值等价及额外144次补测Adam。仅TimeMixer/ETTh1、TimeMixer/ECL、ModernTCN/ETTh1四H采用全浮点state/gradient/Adam moment atol=1e-4、rtol=0，loss与归一化validation atol=1e-6；初始身份/RNG/batch、optimizer step、非浮点状态和param-group结构继续exact，非finite拒绝。TimeMixer/Exchange原具名单张量1e-7规则保持。实现使用预分配CPU缓冲区写原始sidecar，正式训练数学/profile不变。CPU首验12 passed/1 error/3 unexecuted，唯一机械清理修复后16/16；三组H96串行参考＋两路并发＋独立串行重复共72 Adam/96前向/72反向均通过，最大state差分别1.91e-6、1.91e-6、4.32e-5，最大指标差2.38e-7，均在预注册界内；机械余额288。34组/115代表继承已核对，20组/80代表补测上限1440=原余额1296+新批144，尚未启动。J未冻结、M5未Closed、M6未开始。以下§20及更早为历史时点。

2026-09-19当前状态（M5 §20）：用户授权直接修复并准备补测；已审核第二轮54组终态34 Passed/20 Blocked（10继承＋24新通过），本轮原probe实际1740 Adam/2320前向/1740反向，无已尝试worker OOM或资源归属失败，不外推未执行H。新增固定CPU摘要缓冲区，三个原RSS失败代表复验通过且计算轨迹与原版exact相同；没有删除四点规则或放宽门槛。CPU16+16方法通过；15个合成诊断worker累计90 Adam/120前向/90反向，机械余额360。三组数值诊断初始/RNG/batch相同但串行重复也不exact，最大模型状态差约1.55e-6、7.08e-8、5.25e-5，不能直接套TimeMixer/Exchange单张量1e-7白名单，未新增容差。计划34组继承/20组80代表补测上限1440，原补测剩1296，额外144仅Proposed。工程repair可收口，完整补测仍Blocked；J未冻结，M5未Closed，M6未开始。以下§19及更早为历史时点。

2026-09-19当前状态（M5 §19）：用户明确选择数值等价并继续授权ChatGPT直接执行至工程closure。仅对TimeMixer/Exchange既定四H的enc_embedding.value_embedding.tokenConv.weight及其梯度/Adam两动量，预登记atol=1e-7、rtol=0、NaN/Inf拒绝；训练loss与归一化validation误差同绝对界，初始/随机/批身份、步数及白名单外模型/梯度/optimizer状态继续exact，其他模型/域不放宽。新JSON逐步数值证据与精确残余摘要已接实际probe比较及报告检查。CPU16/16首次通过；H192新六步串行参考、两路各六步及独立串行重复共24 Adam/32前向/24反向，全部在该固定界内，最大梯度差7.450580596923828e-08、动量差1.4901161193847656e-08、参数和loss差0，旧exact Not passed不倒改。机械余额450；3036补测额度未消耗，10组25代表继承已核对，44组170代表仍待用户一次启动。495/5340、全部profile/T/结构/训练/数据/确定性设置不变。技术阻塞已按新合同处理，本轮工程审核closure后可生成实际版本probe许可；不代启补测，不冻结J、不关闭M5、不进入M6。以下§18及更早为历史时点。

2026-09-19当前状态（M5 §18）：用户明确授权ChatGPT接手服务器执行上一轮确认方案及检查工程Git closure。本轮57个来源待决run全部落实，iTransformer/ECL明确采用本模型官方脚本B16；A/J/N/S的274个profile、495任务/5340 run-epochs、54组/195worker及统一T等不变。最终最大optimizer步数算术16,482,750；新增709次补测Adam获批，2327原余额＋709=3036，未消耗。双时点RSS测量保留四点严格增长规则和前序哈希影响说明；CPU16/16首次通过、无复验。限定5worker均正常完成，共30 Adam/40前向/30反向，机械余额474；AMD/ETTh1六步未触发旧增长检查。TimeMixer/Exchange H192初始/RNG/batch及loss一致，但重复串行也有Conv1d权重梯度/Adam微小非逐位差异，exact仍Not passed，不擅调容差或确定性设置。10组25代表继承profile核对通过，44组170代表补测计划保留但全队启动仍Blocked；本轮仅工程版本收口，不生成完整probe许可、不冻结J/关闭M5/进入M6。以下§17及更早为历史时点。

最新状态（§16）：**本次用户明确批准的数据准入已落地；Weather/PJM各一次受限train/validation前缀连通通过，修后精确CPU 16/16通过。** Weather保留作者原记录，允许非唯一但非递减时间；ECL/Exchange以已核验benchmark身份作标准化评价并披露未知解释；PJM锁定TimeXer、n=52416及既定端点，原EPF d−1可得性作为documented source assumption。细分事实/接受范围/限制，未一键改成Passed。495/5340、54组/195worker、模型训练配置与NVML门禁保持；probe端点/尾批技术缺口已收口，dry-run/preflight只剩审核许可、统一closure/clean及用户启动，完整probe尚未执行。首轮CPU7 passed/1 error/8 unexecuted，唯一机械夹具修复后完整16通过，累计24次方法调用；§14额外调用偏差保留。0GPU/模型/Adam/前向/反向/checkpoint，未重哈希原数据。J未冻结、M5未关闭、M6未启动；以下§15及以前均为历史时点。

最新状态（§15）：**来源/版本直接核验已完成五组主比对：PJM与ETTh1为A类原始字节一致；Weather/ECL/Exchange为B类，仅CRLF→LF后文本一致。作者分发的Weather同样含19043/19044重复时间；PJM n=52416，既定端点36691/41933/52416。** UrbanEV复用Closed M1审计并核对Git对象，不重读数据。EPF原论文已明确day-ahead外生预测在d−1可得，不能继续写成完全没有来源说明；但这不是逐条vintage审计。另取得原EPF Zenodo发布物，发现其header/列序与TimeXer转换版不同（C类），转换链仍有独立缺口。本轮不落地§14.4降级政策、不改loader/配置/mandatory；全部新事实待审核后联动生产准入。495/5340、54组/195worker不变；0模型/GPU/训练/checkpoint，8项合成审计方法首次通过，无复验。完整字节比较实际遍历test字节，test数值解析/统计/评价为0。论文表格的文本冲突未获页面图像复核，保留限制，不据此裁定。§14额外test调用偏差原样保留。

最新状态（§14）：**NVML命名空间PID映射和未登记CUDA时序缺口已在正常权限下实际解决。** 自有`/proc/<pid>/sched`内核PID与NVML逐一匹配；1→2→4个16MiB张量进程实际准入通过，首次27.87秒、4实例，0模型/Adam/前向/反向。CPU首次16/16；另1次嵌套旧test断言调用造成严格首验调用计数17（上限16），偏差待审。同步实测资源状态、移除嵌套test调用后唯一复验16/16，未重跑CUDA。此结论只覆盖监测链，不发正式模型并发许可。495/5340、54组/195worker及所有科学配置保持；四域数据仍mandatory blocked，精确合并待决方案见§14.4。当前可提交整体审核并在获准后统一closure；完整probe仍需修后审核、实际closure/clean、用户启动及PJM尾批等数据边界处置。J未冻结、M6未启动，以下§13及更早内容均保留历史时点。

最新状态（§13）：**用户重新决定恢复A/J双方UrbanEV F1–F4输入消融；本次只恢复AMD/F1–F3的72个计划。当前495 runs/最多5340 run-epochs，54组/195代表worker。** 双方F4复用主表；F0、J/F0、PJM TargetOnly均为0。schema v2及`input_variant`保持，完整配置指纹更新。本轮16个精确CPU方法首次全通过、无复验，0 CUDA/Adam/前向/反向，生产模型和训练路径不改。数据mandatory与NVML归属Not verified沿用§12；已将资源路径缺口加入完整probe启动前阻塞，不能先跑全队再重复发现。完整probe/正式训练未启动，J未冻结、M6未开始。以下§12及更早摘要均为当时事实；当前范围、命令及停止线见§13。

最新状态（§12）：**In Progress；当前423 runs/最多4620 run-epochs，51组/183代表worker。全部消融仅UrbanEV：F4模块A/N/S/J只新增N/S；输入仅J/F1–F4，F4引用主表。撤下97个未执行计划，J/F0始终非法。** 新正式schema统一`input_variant`；修后CPU25/25及一个J/ECL CUDA方法1/1通过，本轮累计CPU50调用、2 Adam/6 forward/2 backward。Weather重复原始时间、PJM版本端点/单位/as-of、ECL/Exchange映射与单位缺口保持Blocked；NVML进程归属仍Not verified，无资源许可。完整probe/正式训练未执行，J未冻结、M6未开始。以下§11摘要是历史快照，当前规则与操作命令以§12为准；旧证据未覆盖。

历史实施登记（§11）：用户当时授权“按模型分组＋固定配置轻量资源准入”替代三包启动及未执行旧B/probe补丁，额度不叠加。起点99be14ce840e67ecd72720860f5d4ab0247418a1，三端0/0、worktree/index clean。实施前精确写入清单：本文件、canonical、configs/ch3_formal_profiles.json、utils/ch3_contract.py、utils/ch3_data.py、models/ch3_adapter.py、models/tsAMD_enhanced.py、ch3_runner.py、tests/test_ch3_formal.py、tools/restricted_regression/m5_formal_entry.py、tools/restricted_regression/restricted_io_guard.py、tools/restricted_regression/sitecustomize.py、tools/restricted_regression/bundle.sha256、scripts/ch3/start_model.sh、scripts/ch3/start_probe.sh。保留旧loader及旧M4入口/构造合同，新增正式声明和共享管线；不改作者仓库/环境/Closed文档/空间方案。证据仅amd-execution-evidence/m5/m5-formal-admission-bizsxh0v，合成fixture仅/tmp/amd-m5-formal-jef2lnmg。该轮完整资源probe未启动；§11收尾为当时事实。

历史§11状态：**当时按模型分组520 runs/最多5600 run-epochs；正式声明/adapter/前缀loader/共享runner及入口已实现。最终CPU清单25/25；四个代表性CUDA方法首次及唯一模型复验均4/4。模型累计16 Adam/48 forward/16 backward。Weather重复时间戳及既有数据映射/as-of缺口仍blocked，完整资源probe未执行，J未冻结、M6未启动。** §10八源smoke、ETTh1固定13项及原失败成本保留，未重跑。下列三包与旧筛选方案均为历史记录；当前规则见§12。

开始日期：2026-09-16（UTC）。canonical：`docs/AMD_EV_Thesis_Final_Implementation_Plan_v2.1.md`，v2.1-R1。

历史§10时点：2026-09-16阶段启动与§§1–9各轮提案/授权及旧工具Blocked保留历史。该轮只授权§10来源准备、限定工具增量和实测验收，正式模型/runner等当时未获授权。**本轮§11已经授权必要正式接入及短验收，但正式训练、完整probe现场启动、M6与最终结构冻结仍未授权。** 本轮空间方案/源码/数据/环境均不修改或准备。

## 1. 启动门槛、版本与继承事实

- A段实际统一closure commit：`3e43a55327eac641445ab5a43d06424d7fc5ed3f`，提交题为`docs: seal M4 diagnostics and candidate decisions`；父提交`4abb099d69db04532f066986fde0c43aedf2465a`。本轮不另做closure。
- B段写入前现场回核：HEAD、`origin/AMD-paper-repro-custom-modules-v1`、`git ls-remote origin refs/heads/AMD-paper-repro-custom-modules-v1`均为上述closure commit；ahead/behind=`0/0`；worktree/index clean、untracked none。三份closure文件的工作树字节均与HEAD blob一致。此为本轮对A段完成现场的回核，不冒称本轮重新执行了A段提交/推送或历史动态审计。
- 已封存M4 SHA-256：`3b29d43624de290ebdec6a1701066f6dded33f8e788d8b183cc1fa416df4c11e`，与用户给定审核字节完全一致。**M4已Closed**；其文件头与§69最后的待审/未closure文字是提交前历史快照，结合实际closure解释，本轮不回填、不修改M4。
- canonical修改前SHA-256：`600979d52b6077e49a38b867d3c2b39a3dcd9f3480adaaac4830878148bd0d14`。适用仓库`AGENTS.md`已读，SHA-256=`4dfbc7161b731e20674d28eb1b4759dd1ffbc403ae5217add26efd7433091e6c`；同时遵守本次消息提供的服务器规则。限定检查`docs/milestones`，原无M5文件，故新建本文件，不建立第二份摘要。
- 继承M4 §§67–69：incremental engineering及既有implementation/result review的Passed只覆盖原合同；原18项总效果gate仍 **Not passed**。唯一失败ETTm1 H192 J/N development-test MSE退化`1.542951806%`，超过原`1%`安全线；原24-run科学序列停止。J带已知风险进入候选审查，不是已冻结EL-AMD。N/S保留，S2与THLS单模块已接受开发结论、P2原Not passed及失败来源历史不改。

读取顺序为canonical指定条款→封存M4必要小节→已登记来源绑定→限定生产源码/测试定义。未重读论文、参考仓库全集、历史artifact/曲线；未读取或哈希checkpoint、真实CSV/观测。本轮没有import训练入口、构造模型/Dataset、运行测试、forward/backward/Adam、训练或新并发探针。正文中的历史Passed均有文档出处，不是本轮运行Passed。

## 2. 固定候选数学及四臂解释

本轮概念标签：**A=同输入完整AMD/AMD-Concat；N=A+THLS；S=A+Sonnet S2 target residual；J=A+S2+THLS**。不注册这些标签，不重定义历史A/B/C、U/M、`M4_NSJ_*`，不创建run。

来源继续绑定canonical §1.1.1及M4 §§43–44：Sonnet，AAAI 2026，DOI `10.1609/aaai.v40i30.39736`；PDF SHA `b076e6fed68448d3c3382c96f6f6985a988ea019ef3c470353780385c4011079`；官方仓库commit `bf3d4801d34c5e7261718490f287c6fb15cadfdb`，许可记录`license_text_missing_classifier_only`。复用已审paper-defined MVCA与项目独立实现边界，不重新择源或复制官方实现。S2保留joint embedding、learnable wavelet、latent轴FFT coherence、no-Koopman reconstruction及target residual；固定d64/K8/alpha=.5/gamma=.001，无额外norm/head/parallel路径。

THLS沿M4 §§59.2–59.5、§60.20、§63及§67：项目修改为原RevIN目标历史的有符号`[y,d1,d2]`→3→8投影→双DW→feature-LN→8→16→8 FFN→8→1投影→eta=.001。底层`ReparamLargeKernelDWConv`来源边界继续沿ModernTCN-inspired局部卷积；THLS项目形状特征/直达hidden支路不冒称论文原模块，也不改名成PMCR v1/P2。不存在新的来源审计结论。

`models/tsAMD_enhanced.py::forward`（1109起）静态确认：一次AMD RevIN后先clone目标history且不detach；S2先写目标输入；唯一MDM→DDI；THLS用保存的S2前history修正DDI目标hidden；AMS selector取该MDM输出；完整通道denorm后切目标。state的第三段保持context32零占位，不实现M7。A关闭两模块且不实例化；四臂均关闭PMCR/P2、TEB与CCE。S/J必须非空有序aux、target_exogenous；THLS要求norm/layernorm、T≥3及显式合法奇数核。每组公共AMD初值、S/J共有S2、N/J共有THLS和独立train generator须匹配；关闭模块不消耗其初始化子流。

比较问题固定为N/A局部模块独立贡献、S/A外生模块独立贡献、J/A组合总收益、J/S局部条件增量、J/N外生条件增量。完整四臂才允许报告同一任务上的交互描述量`MSE_J−MSE_S−MSE_N+MSE_A`；不把它当因果证明，不跨dataset直接相加误差。旧NSJ没有同批A，不足以回答全部独立贡献或该交互量。

## 3. 正式MS静态接入审计

证据标签：**已有工程证据**=封存milestone所记已执行结果，仅继承其范围；**仅静态可见**=本轮读到的代码/测试定义；**数据事实尚未核验**=保留Not verified；**缺接口/缺验收**=拟议M5尚无获验收端到端能力。同一项可以同时具有前两类证据并仍有接入缺口。

### 3.1 六个既定数据任务的入口定位

共同合同已确认（canonical §§5.1/5.5/9.1，M4 §§49.3/50）：保留全部历史变量及ordered aux；日期列只作索引，UrbanEV保留原F4历史日历；无未来真实协变量。目标loss/完整预测区间、train-standardized指标、formal seed=[2024]均不重新征询。下列索引为去时间列后0-based；数据版本/SHA沿M4 §49.3登记，本轮未重新核验真实数据文件。

| 既定任务与信息集 | 静态入口/继承证据 | 具体接入阻塞与状态 |
|---|---|---|
| UrbanEV：volume/0；F4 C11；aux=[e_price,s_price,Ta,P,h,hour_sin,hour_cos,weekday_sin,weekday_cos,is_weekend]；T12，label H=3/6/9/12、pred_len=1，fold1–6、全部275区 | `utils/feature_schema.py`；`UrbanEVFoldPreprocessor.fit_transform`、`TemporalRegionDataset`；`main::_validate_urbanev_protocol`允许六fold；M4 §67继承fold6三臂证据 | **缺接口/缺验收**：THLS/NSJ公开合同在`_prepare_thls_contract`锁fold6；`_build_urbanev_runtime_data`也固定`train_validation_fold=6`；`UrbanEVRawData.load`拒绝restricted fold≠6。一般六fold能力不等于六fold冻结前前缀隔离。需逐fold受限读、四臂新身份及六fold宏汇总；本轮无新六fold数据/模型证据 |
| EPF-PJM：price↔OT/2；C3源顺序及aux=`[" System load forecast"," Zonal COMED load foecast"]`；T168→完整24步；F_PJM待闭环 | `CustomDataLoader`可按字段构造通用MS；M4 §50.2已有TimeXer来源脚本/历史输入说明 | **数据事实尚未核验＋缺接口/缺验收**：市场/结算、单位转换、forecast发布时间/vintage/as-of、正式split/retraining/fit政策及项目公开管线缺失。历史输入不证明当时可得；不能用通用70/10/20或来源itr=1锁定项目F_PJM。四臂不接受该dataset，J构造也无该shape声明 |
| ETTh1：OT/6；C7；aux=[HUFL,HULL,MUFL,MULL,LUFL,LULL]；T512，H96/192/336/720，official单split | `utils/dataloader.py::_dataset_kind/_compute_split_endpoints`有ETTh固定8640/11520/14400；MS唯一目标解析、train-fit scaler及完整H取窗静态存在；M4 §49.3保留版本证据 | **缺接口/缺验收**：generic读取全表/test并构造test loader；S/N/J公开入口仅UrbanEV/ETTm1。ETTh1虽与ETTm1同C/T/target，不能借用其development身份/test许可或把shape吻合当Passed。新四臂、正式身份、test隔离尚缺 |
| Weather：T (degC)/1；C21，ordered aux为M4 §49.3精确源序去index1（含原OT及原U+FFFD列名）；T512，四H，official单split | generic MS按名称定位非末列目标、保存原顺序，静态可见；已登记端点36887/42157/52696 | **数据事实尚未核验**：版本时间粒度仍Not verified，原OT辅助含义限制保留。**缺接口/缺验收**：冻结前隔离、C21/目标1的四臂schema/构造/恢复与评价未接入；不因字段确认改Passed |
| ECL：OT/320；C321；aux=["0",…, "319"]；T512，四H，official单split | generic MS保留全部历史输入；已登记端点18412/21044/26304；仅静态路径与历史元数据记录 | **数据事实尚未核验**：原客户/序列映射、单位转换链缺；匿名OT不补造业务身份。**缺接口/缺验收**：C321负载、四臂新合同/隔离均未验收；不能由小C工程证据外推显存/耗时 |
| Exchange：OT/7；C8；aux=["0",…, "6"]；T96，四H，official单split | generic MS取完整H；已登记端点5311/6071/7588；仅静态可见 | **数据事实尚未核验**：币种、基准币、报价方向及原列映射缺。**缺接口/缺验收**：T96新四臂/隔离/配置尚缺；不靠列序或shape核销Not verified |

Weather/ECL/Exchange/PJM正式任务继续blocked。上表端点/列名是已登记事实的引用，不是本轮读取真实CSV获得；不下载、迁移、补观测或改数据。标准任务val只借训练尾部T点context、标签全部位于val；UrbanEV各split独立起窗，两种语义不互换。

### 3.2 四臂、指标、访问和身份的具体缺口

| 能力 | 本轮源码/测试定义定位与已有证据 | M5尚缺 |
|---|---|---|
| 公共AMD与开关 | `AMDEnhanced.__init__/forward`、`main::_build_model`；M4 §67.4有旧N/S/J关闭、初始化、梯度/RNG/首batch工程证据 | A的数学关闭路径存在，但`sonnet_thls_contract.COMPARISON_ARMS`只有M4_NSJ_N/S/J，没有同批A；不得往旧三臂协议塞A或换formal purpose |
| S2/THLS构造差异 | `models/modules/sonnet_thls_contract.py::comparison_configuration`仅两dataset；enhanced构造176–212行只允许旧UrbanEV与(512,7)形状；`main::_prepare_sonnet_thls_contract/_prepare_thls_contract`有精确dataset/协议/seed/旧预算限制 | 新声明及新dataset/shape验收待批；单S2模块数学不限数据名不表示runner已放开。`main::_thls_ms_interface_contract/_build_model`按ETTm1选5/31、其余3/7，不能直接将else当正式数据配置。PJM patch、ETTh1/其他域THLS核须显式绑定，不隐式迁移 |
| 指定目标loss/严格best | `main::_prediction_for_loss/train_one_epoch/evaluate/should_update_best`；预测仅[B,H,1]，接受对应[B,H]或[B,H,1]标签；MSE＋单列AMD固有aux，validation严格有限下降、等值早epoch | 目标监督基础已有；正式训练/停止政策及全链身份未注册。`tests/test_runner.py::test_prediction_loss_adapter_accepts_only_two_exact_target_shapes/test_train_and_evaluate_share_the_strict_target_adapter/test_best_selection_is_strict`是可复用定义，本轮未跑 |
| 尾批/聚合 | generic `get_train` shuffle/drop_last=True，val/test shuffle=False/drop_last=False；`_accumulate_errors/_finalize_errors`用全目标元素SSE/SAE/Q；M4 §67生命周期继承旧任务验收 | `summarize_results.aggregate_runs`主要按科学比较hash聚合seed，不实现canonical六fold→四H宏平均、跨任务配对完整性与新purpose。单seedN/A现仅特定M4 variant分支，不可泛称正式汇总已支持；需逐fold/H未舍入表、完整矩阵/duplicate拒绝 |
| train-only scaler与test隔离 | generic `_read_data`仅train fit，但`_read_raw_dataframe`无界`pd.read_csv`，随后transform test；`main::_build_generic_runtime_data`无条件`get_test()`。UrbanEV `fit_transform`按fold train fit，受限前缀只fold6 | **确定静态阻塞**：train-only fit不等于test从未读取。只设evaluation_policy或不调用final test不够，必须在解析/transform/Dataset/DataLoader之前隔离。正式test禁令不继承ETTm1例外；逐fold访问语义须保持原累计月切分，不另改六fold |
| config/科学身份 | `parse_args` artifact_purpose choices仅`m4_development_candidate`；`_scientific_config/_resolved_config/_training_protocol_block/_checkpoint_common`已有目标/来源/初始化/训练字段基础 | `ch3-ms-specified-target-full-horizon-v1`仅文档已确认，尚无M5 purpose/四臂/正式任务合同闭环。需绑定原始业务身份、完整映射、split/scaler/T/H、loss/metric/macro、test policy、seed/初始化/训练停止/配置机会等，config→hash→manifest→checkpoint一致；不是只换目录 |
| 恢复与汇总拒绝 | `main::_load_resume_checkpoint`先读manifest/resolved，核variant/config/data/schema/candidate/评估合同后才`torch.load`，再严验tensor；M4 §67有旧协议原子拒绝/恢复证据；`summarize_results::_validate_sonnet_thls_variant_contract/_validate_sonnet_evaluation_artifact`等硬绑旧M4 | 新身份的跨purpose/task/target/H/metric/split/预算/early-stop拒绝及四臂合法resume仍缺验收；外部metadata不一致须先于反序列化，tensor不一致先于写参。summarizer会读取权重，故本轮只读其源码，不运行summarizer；旧artifact不改名、不跨协议恢复 |
| 新早停与运行预算 | `_main_impl`按train_epochs循环、每epoch保存严格best，既有标准协议是fixed-budget；未见拟议patience停止状态 | 若采纳§4新早停，需新增停止状态/完成原因/best与last分离、resume及summary验收；不能仅改CLI epoch就声称可用 |

总结：**已有M4工程能力可复用，拟议M5四臂正式任务尚不可直接执行**。最先必须补数据访问隔离；随后仍需新身份/构造声明、正式元数据、训练协议和聚合验收。源码可见不记为动态Passed；六数据集正式baseline适配仍沿M4 §49.5，不把另外六个baseline全量接入作为本首包任务。

## 4. 唯一推荐筛选协议（Proposed，未批准执行）

### 4.1 最小集合、比较及覆盖解释

推荐UrbanEV F4 **全部六fold×四H**、EPF-PJM **原T168/H24×F_PJM**、ETTh1 **official×四H**，每项同批新A/N/S/J，单配置、seed2024。四臂同数据版本/信息集、split、scaler、label、batch、训练/选择政策及公共初始化；不按已见结果删除H192，也不按dataset/H换用N/S/J。

此为最小**数据域集合**，并不把UrbanEV缩成fold6：canonical §9.5外生模块“UrbanEV/EPF中至少两项”按两个数据域解释，S/A必须在UrbanEV及PJM成立；局部模块“UrbanEV＋Weather/ECL/ETTh1至少两项”由N/A在UrbanEV及ETTh1覆盖。§21的EV＋外部域要求亦保留。不得把两个H/fold算作两个域。若“至少两项”被要求作更广覆盖解释，应显式解决合同解释，不能静默扩大结论或删条件。

这只是M5筛选集合提案，**不删Weather/ECL/Exchange正式任务，不缩减§9.3六数据集模块消融或§9.4输入消融**。未纳入本筛选集合的正式任务必须在相应M6任务前闭环；若其接入迫使改变已选结构/训练合同，返回M5审议，不在test之后调整。PJM仍blocked，不能省略它而宣称S2已满足外生模块条件。ETTm1仍development-only，不纳入本矩阵或M6未见test主表。

### 4.2 复用与必须新取得的结果

复用M4 §67工程/初始化/恢复/关闭/路由证据、§68已接受结果、§69风险披露及同形状并发的有限结论；完整333/339、旧八项、smoke、兼容包与24-run不默认重跑。对实际新增接口另作增量验收，旧skip/受限证据不扩大为新全覆盖。

必须新取得：同批A/N/S/J三域指定任务的train/validation结果、严格best与完整聚合；新任务的初始化公平、隔离/恢复/身份验收；参数与执行成本；按预先锁定规则作模块/组合判定。旧S2/A、THLS/A开发结果及无A的NSJ不能拼成正式四臂或升级为主表，不用旧checkpoint warm-start。不得为了让best epoch更晚而调参。

### 4.3 安全底线、practical-effect与组合判定

**既有合同不变**：相对同输入AMD验证平均退化≤0.5%仅是安全线；需要上述跨域独立改善、代价收益匹配和预先锁定practical-effect；组合不因交互产生稳定退化。formal seed=[2024]、std=N/A；随机初始化稳定性 **Not evaluated**，不能由fold/H一致性替代。M4的1%原阈值与H192失败永远保留，以下新规则只用于未来M5。

推荐新增操作定义（全部Proposed）：令每数据域的MSE/MAE宏均值按canonical §5.5聚合，`Δ_d(X/R)=100×(macro_d(X)/macro_d(R)−1)`；不平均不同数据集原始误差，不用逐H相对变化均值替换主gate。PJM若多fit，先逐fit列报，再按预登记fit等权宏平均；重叠验证日期/权重须在PJM合同闭环时说明。

1. N/A、S/A、J/A在三个域各自的validation MSE与MAE Δ均≤+0.5%；逐fold/H（PJM逐fit）MSE退化均≤+1%。前者MSE源自既有安全要求，逐域落实、MAE及逐单元1%是新的前瞻性操作提案，不能声称已获批准。
2. **practical-effect推荐0.5%**：S/A在UrbanEV和PJM、N/A在UrbanEV和ETTh1各自MSE Δ≤−0.5%，MAE Δ≤0；U/ETTh1至少3/4个H改善，任去一个H后的宏MSE仍<0。UrbanEV另要求六fold中至少4折的四H宏改善；这是单seed跨任务一致性，不是统计显著性/seed稳定性。PJM F=1只可支持该split；若F>1要求多数fit方向改善并列全表，不伪称多seed。
3. J须在三个域各自相对A满足MSE Δ≤−0.5%、MAE Δ≤0。相对N及S，每域MSE/MAE Δ均≤0，并保留逐单元MSE≤+1%的安全检查；对每一个control至少在UrbanEV和一个外部域达到MSE Δ≤−0.1%的条件增益。0.1%是新增组合实用增量提案。完整列报交互量与负向任务，不能仅凭平均提升称不存在交互风险。
4. 技术或任一已锁定科学gate失败即停止当前序列并报告，不自动加epoch/LR/seed、删除任务或新结构。正常完成整套矩阵后统一裁决，避免依中途效果削减弱臂；单项非改善不足以证明跨seed“稳定退化”，但未达推荐操作gate也不能自动冻结。若只剩N/S/A合格，先返回用户按§21决定，不把关掉模块写成已满足两模块目标。

上述阈值是一套供锁定的保守方案，尚无未来M5数据支持其已通过；不追认M4失败通过，不针对H192设置特例。

### 4.4 训练/选择、公平机会与代价（新增值均Proposed）

- 推荐每run **最多30 epochs，patience=5**：每完整epoch后按目标validation MSE严格下降更新best；任何严格下降均重置耐心，等值不更新best且计未改善，连续5轮未改善则停；不设epoch0候选、warm-start或额外最小epoch。早停和达到上限均为明确完成原因；未满上限不等于训练失败。该30/5不是继承M4固定10轮，须先实现恢复/汇总支持并另获训练批准。
- 每dataset/arm/fold/H只给 **1套配置、1次from-scratch机会、0额外搜索run**。推荐固定Adam lr=3e-5、weight_decay=1e-7、无LR schedule，四臂相同；S2与THLS数学/初始化沿已登记规格。常规最佳checkpoint依validation选择，不能择run/seed或在见到结果后延长某臂预算。
- 推荐UrbanEV batch128/patch12/THLS核3和7；ETTh1 batch32/patch16/THLS核5和31；PJM batch32/patch24/THLS核5和31。后两域是待批配置迁移及新shape验收，尤其PJM不能以当前`else`分支自动落3/7。其余共同AMD参数推荐沿NSJ（n_block1、mix_layer_num3、mix_layer_scale2、dropout.1、alpha0、norm/layernorm=true），明确登记而非借旧M4身份。train drop_last=True、val keep_all；不增加AMP/TF32/compile/workers优化。
- early-stop只优化既定validation选择与成本，不以best位置晚/曲线好看为目标。模型间参数不同如实计数；四臂优化机会相同，不将这些AMD家族设置自动强加给M6其他六baseline，后者仍需公平搜索机会合同。
- 代价收益推荐：先满足科学gate，再逐域列参数、完整run墙钟、实际epochs/steps、峰值显存与推理吞吐；J相对A的完整run串行等效成本比和峰值显存比均建议≤2，超出则停在代价审查，不自动冻结/扩预算。2倍是待批实用上限，不是实测。参数增长逐项解释，不用参数/MAC代替时延。精确独占效率/新探针预算另审，不为本轮制造测量；若无法得到可比成本，代价gate为Not evaluated。

### 4.5 run数与预算组成（Scenario；当前授权为0）

| dataset | arm×fold/fit×H×配置×seed | runs | 30 epoch上限 |
|---|---|---:|---:|
| UrbanEV | 4×6×4×1×1 | 96 | 2880 run-epochs |
| ETTh1 | 4×1×4×1×1 | 16 | 480 run-epochs |
| EPF-PJM | 4×F_PJM×1×1×1 | 4F_PJM | 120F_PJM run-epochs |
| 合计 | 按不同dataset任务数求和 | **112+4F_PJM** | **3360+120F_PJM** |

F_PJM=1仅名义Scenario：**116 fresh runs、最多3480 run-epochs**；不是已锁定fit政策/新预算。PJM推荐先论证单固定train/validation/test划分、无rolling retrain是否符合既有任务；比例可沿来源70/10/20作为Proposed起点，但须有版本/as-of依据、精确边界和明确批准后才能代入1，不能把这条建议当成既定fold修改。事实不支持时不跳过PJM，重新核算F。正式其他五数据集已有fold/H不变。

每任务训练步上限为`30×floor(n_train_windows/batch)`；总上限对四臂、全部fold/H/fit求和。UrbanEV窗口还乘275区域，标准任务标签涵盖完整H。真实窗口数、有效prefix边界验收、PJM F、各任务每步/每epoch耗时和峰值资源 **Unknown/待核验**；此轮不据旧run均值编造GPU小时或精确Adam数。canonical §9.6.3的历史窗口算术只作来源记录，不能代替新管线核账。

预算分账：本轮文档静态工作0模型/测试/训练；首包合成loader验收提案见§6，与训练独立；后续四臂工程/真实smoke/新形状并发预算待单列；训练按上述上限；额外搜索、自动失败重跑均0；中断先审计合法resume，成本保留且不自动扩额。旧实验余额不转为M5额度。

调度推荐默认串行。UrbanEV fold6、原C/T/H/batch/线程/环境等确实相同的负载可在接入等价及协议变更影响经审核后参考M4已验证最多四路范围；不是立即复用放行。其他fold、ETTh1、PJM不由ETTm1/UrbanEV旧短测推成四路，最长30轮/早停也未实测。无本轮并发/效率探针、无新launcher；物理GPU占用区间并集、run时长和任务组makespan分列，不能以并发数换算实际加速。

## 5. M5与M6的训练去重（Proposed流程，未获执行批准）

M5只为A/N/S/J完成筛选，不提前训练M6八模型全矩阵。推荐在将来M5首run前锁定正式训练/选择协议与新M5身份；结构保持候选直至全部筛选及用户冻结决定完成。拟在M6复用的是**将来新M5合格训练产物**，不包括任何M4开发artifact。

复用须逐run同时满足：

1. 数据版本/业务目标/源列与模型列双向映射/ordered aux、T/H/fold、节点、split/scaler和完整目标指标完全一致；元数据blocked已解除，有访问证明。
2. 架构及初始化、run seed2024、训练配置/上限/早停/best规则、搜索机会均在训练前锁定；M5选择出的正式结构与该产物完全相同。仅M5之后才新定的训练合同不能追认先前run符合冻结协议。
3. 独立M5 purpose/protocol、source/科学hash、manifest、checkpoint、完整性及resume链可验证；M4旧身份拒绝。冻结时另立明确批准记录，将已选M5产物映射到正式模型/消融角色，不改原manifest或目录冒充从未筛选过。
4. M5仅train/validation，正式test从未用于选择；M6冻结后用独立只读评价记录关联原best checkpoint及冻结记录，再按批准清单开放该任务test，不重新优化、不改变原M5 test_access_policy。此“冻结后关联评价”接口和发布/汇总流程目前不存在，需另行审核、实施、验收，不能靠换purpose恢复完成。
5. 原训练协议或最终结构改变则该run不可扣除；事先定义哪些改变要求新训练，不能看test后决定。M6继续补齐未覆盖的三个数据集、其余baseline、TargetOnly及UrbanEV F1–F3原范围消融和效率。

条件全部成立时，未来M5矩阵A/J可覆盖对应主表格，N/S覆盖对应独立模块消融；同一训练计一次。名义F=1且最终冻结正是J、全部116可合法复用时，canonical原579-run完整情景的剩余新训练为`579−116=463`（一般为`456+7F_PJM`）；这是条件算术，**不是已节省116次或批准463次**，也不含新增工程、搜索、评价/效率成本。不重新训练只为把M5标签换成M6，不把ETTm1 test纳入正式表。

## 6. 首个最小实施包：ETTh1受限loader能力（仅设计，Proposed）

推荐下一笔只解决§3中“generic在模型构造前就解析/transform test”的首要阻塞。限定 **ETTh1的train/validation-only loader能力＋合成验收**；不会单独使M5四臂可训练，也不解除PJM等元数据阻塞。首包不增加模型结构、不注册正式variant/run，不修改旧M4合同。

### 6.1 精确文件/接口范围

| 拟修改文件 | 推荐增量及边界 |
|---|---|
| `utils/dataloader.py` | 在`CustomDataLoader.__init__/_read_raw_dataframe/_read_data/get_test/metadata`增显式受限访问参数，默认保留旧全split行为。首包只允许ETTh1固定已登记split；解析止于val_end=11520，train-fit止于8640，val借T历史并覆盖全部H标签；不得先全表read再slice，不transform/构造test dataframe或test dataset；受限`get_test`先于任何访问抛错。metadata区分完整版本声明、已读prefix与未访问test，不能把prefix长度重新按比例切分，不能声称验证过未读区域。保留原命名目标/ordered feature、MS标签与inverse_transform行为 |
| `tests/test_dataloader.py` | 仅新增受限ETTh1合成测试类/方法，复用现有临时CSV与MS/scaler断言；不依赖真实data存在，不因真实资产skip跳过新测试。旧test定义/断言保留 |

不在本首包改`main.py`、模型、summarizer、工具/访问guard、launcher或环境。后续主入口必须在新M5独立身份批准后显式接入受限能力，保持RuntimeData.test_data=None；这项runner wiring、新四臂声明、正式元数据/早停/summary仍是后续阻塞，不能把首包验收记成完整正式MS Passed。推荐精确新增keyword-only参数`access_policy="train_validation_test"`，受限值`"train_validation_only"`；这两个值及metadata同名字段均为本首包待批接口，只描述访问，不暗授M5 purpose。

数据访问政策扩展须用户单独批准：本包动态验证只允许独立`/tmp`合成CSV，真实CSV/历史artifact/checkpoint/正式test读取和哈希均0；模型/optimizer构造与forward/backward/Adam均0。完整真实数据版本hash将来只在另批数据审计中核验，首包使用fixture自己的版本身份，不伪装成正式数据已核验。不得借首包放宽既有restricted工具政策。

### 6.2 必需增量验收、预算与停止线

推荐 **8个新合成方法ID，各执行一次**（实施时固定完整ID）：①受限读取边界/毒化test行不会被解析；②固定端点与四H首末窗口、完整目标和历史context；③train-only scaler及改动val不影响fit；④指定非末列目标/有序aux/目标inverse_transform，证明不依赖末列猜测；⑤val尾批完整、train drop_last及generator状态；⑥受限get_test在访问前拒绝且metadata无伪造test观测；⑦错误dataset/policy/短prefix/列映射早拒绝；⑧默认旧模式与显式旧模式的合成ETTh1/ETTm1数据、metadata、窗口一致。

复用执行的旧合成方法仅建议5项：`test_public_split_endpoints_are_unchanged`、`test_real_window_counts_and_loader_policies`（其helper为合成CSV）、`test_inverse_transform_accepts_full_or_s_ms_target_width`、`test_explicit_train_generator_can_be_saved_and_restored`、`test_missing_target_is_rejected_for_s_and_ms_but_not_m`。合计 **13方法调用，CPU loader-only，0 GPU、0模型、0反向、0Adam、0真实访问**；这是新增待批测试预算，不是本轮已执行或从旧333/339额度扣除。合成每CSV≤20000行×8特征、临时占用≤64MiB、单进程；首次技术失败即停，不含自动重跑余量。fixture可有合成test值，仅用于验证拒绝/旧默认兼容，不接触正式test。

验收同时要求：新受限路径不读取/解析val_end之后的观测、不隐式调用get_test；四H标签与列顺序正确、scaler完全train-fit、metadata如实；旧默认兼容未变；新八项及指定五项各有完整终态，不能以AST/编译或部分通过替代。纯loader构造与取batch属于下一包待批动态工作，**本轮没有执行**。

如需改模型数学、旧身份、默认split/scaler、真实数据访问、工具政策或扩大文件/预算，立即停在该缺口报告；不能以M5已启动自动放行。成功也仅登记ETTh1受限loader增量完成，回到M5审核，不生成训练ready、run或自动转下一包。13项通过不证明六域四臂可执行、效果Passed或M5结构冻结。

## 7. 本轮静态验收、待锁定项与停止位置

本轮允许写入仅canonical的最新状态/必要状态引用和本M5文件。封存M4、AGENTS、M0–M3、代码/tests/工具均保持原字节；未stage/commit/push，不另做小文档closure。静态验收只核对文档范围、关键授权/风险措辞、矩阵算术、SHA与`git diff --check`，不运行训练入口或测试。

待用户锁定的是新内容：§4的三域四臂筛选提案/覆盖解释、practical-effect与组合阈值、30/5训练/选择和单配置机会、PJM split/fit方案及所需事实闭环、新域配置/代价线/预算；§5未来产物关联评价/去重流程；§6首包两文件接口/合成访问和13项预算。已定目标、输入/H/六fold、正式六数据集/模型名单、seed政策及本次M5阶段启动不重问。

本轮文档静态核验已完成：canonical仅§0.1增加最新状态、§19更新状态引用；本文件为唯一M5；矩阵116/3480及条件去重463的算术、末尾换行/空白、受保护M4/AGENTS SHA和两文件范围均已核对。B段实际为canonical modified＋本M5 untracked、index空，HEAD仍为A段closure commit；这是授权文档dirty，不是A段closure失败。最终diff检查/范围/SHA在回执中列报，避免在文件内循环写自身hash。M5首轮静态材料待ChatGPT服务器直读审核、科学提案待批准；没有自动实施、测试、训练或冻结。默认Project操作为“无需上传（ChatGPT通过服务器直读核验）”，该渠道说明不代表ChatGPT已经读取本轮修后版本。

## 8. 正式协议优先：供一次确认的合同修订稿（2026-09-18，Proposed）

### 8.1 起点与拟替换的阶段分工

本轮branch=`AMD-paper-repro-custom-modules-v1`；HEAD/tracking/live remote均为`3e43a55327eac641445ab5a43d06424d7fc5ed3f`、0/0，index空，canonical modified＋本M5 untracked，与预期一致。修前SHA：canonical=`b8df30e4d32bf4a9baff6cc807316b3aa575377fd1f585057c62af6b4debcd81`；M5=`95c9dee9b7b80795321d416cf62487dd1757dbfd27904bf66befb8e5fb31943c`。适用AGENTS及Closed M4 SHA保持§1记录；baseline解析到commit `fa9665627e6fcfb1d0c2bc22d943ca9666304fd6`。没有重做源码/数据/历史artifact审计。

**推荐分工**：M5锁定正式协议与执行配置，完成必要工程接入和针对实际缺口的验收，以已接受开发证据、明确风险接受和准备结果为依据，提请用户作最终结构冻结决定；不默认新增`112+4F_PJM`完整性能筛选矩阵。M6按冻结结构/协议执行六域主表、下列选定消融及成本报告，负向结果完整保留。这里取消的是拟议额外筛选前置，不取消正式证据取得，也不把工程验收当性能实验。

**推荐冻结前置（待替代旧合同）**：①用户明确确认本节对范围及旧多域效果前置的修订；②正式任务/配置来源、同信息集与公平选择、身份/恢复/汇总/test隔离及预算合同锁定，所需工程能力有相应验收，数据事实缺口按任务闭环；涉及J输入、模块适用性或配置能否成立的未决问题须在冻结前解决，其余未闭环任务继续blocked，不称六域已可执行；③向用户提交完整工程/协议准备回执和已有开发风险，用户另行作明确结构冻结决定。**本次确认协议本身不满足②③，不提前冻结J。** 冻结表示停止结构选择、进入预定评价，不表示效果Passed或“两个模块已正式有效”。M4总gate Not passed、H192 J/N退化1.542951806%超过原1%、单seed及ETTm1 development-test限制均保留；不因冻结改判，不按dataset/H切换模型，不新增候选。

### 8.2 推荐正式矩阵、主张与限制

以下精确范围全部Proposed；A/N/S/J沿§2概念定义。J推荐占未来EL-AMD模型族的正式位置，**以用户后续明确冻结J为条件**，不将推荐写成已入选。

| 组成 | 推荐范围与复用关系 |
|---|---|
| 六数据集主表 | UrbanEV、EPF-PJM、ETTh1、Weather、ECL、Exchange均保留同输入A和J。UrbanEV/PJM仍为DLinear、PatchTST、iTransformer、TiDE、TimeXer、ModernTCN、AMD-Concat、未来EL-AMD；另四域仍为DLinear、PatchTST、iTransformer、TimeMixer、ModernTCN、TimeXer、AMD、未来EL-AMD。每域仍8个训练模型，不新增或删除baseline |
| 完整模块消融 | 集中UrbanEV F4：A/N/S/J，全部6 fold×4 H；A/J引用兼容主表run，只新增N/S。其他五域不默认追加N/S；外部域单模块验证若以后确需开展须另定范围/预算，不以隐藏前置方式恢复旧筛选包 |
| TargetOnly | 保留UrbanEV F0的A-TargetOnly，全部6 fold×4 H；保留PJM price-only，原T168→24及F_PJM。其他四域不扩增TargetOnly；F0上的J仍N/A，不自动试空aux S2 |
| 输入消融 | 保留UrbanEV F1–F3各自同输入A/J、每项6 fold×4 H；F4已计主表、F0已计TargetOnly，不重复。J在各preset的新接入仍须必要验收，不能只按名称认定可用 |
| 无训练对照/边界 | UrbanEV Last Observation单列评价；ETTm1继续仅development，不进入M6正式未见test主表。其他可选扩展不自动纳入 |

所有既定字段/有序输入、T/H、fold、目标loss与metric space保持canonical §§5.5/9.1：UrbanEV T12、偏移3/6/9/12、pred_len1；PJM T168/H24；ETTh1/Weather/ECL T512、Exchange T96，四H=96/192/336/720。formal seed=[2024]、std=N/A、随机初始化稳定性Not evaluated，不重复征询。

**可支持的研究主张**：六域主表检验完整J相对同输入AMD及原baseline的跨数据集表现；UrbanEV四臂回答局部/外生独立贡献、条件增量及组合交互，输入消融回答辅助历史的作用。**不能据此声称**N、S各自在六域有效、已满足旧外生UrbanEV＋PJM/局部UrbanEV＋外部域效果线、跨seed稳定或普遍正交互。正式消融若不支持模块贡献，须如实收缩论文主张；“两个来源模块经消融验证有效”是待证研究目标，不是冻结时已成立事实。负向test结果不触发自动调参、删任务、替换某H模型或追加实验。

### 8.3 正式训练/选择：真正待锁定的内容

推荐先形成一份按dataset/model列出的配置清单再申请工程/执行授权，不要求用户重选已确认目标/T/H/seed。待锁定项集中如下，不新增效果门槛：

- **AMD家族消融控制配置**：共同AMD参数及初始化、optimizer/LR、batch/precision、epoch/停止与验证选择政策保持成组一致；N/J共有THLS、S/J共有S2采用既定数学和匹配初始化。推荐同任务A/J主表配置直接作为四臂控制配置，不单独给J或单模块更大的搜索机会；原开发参数只是配置来源，不自动成为正式批准值。
- **不同baseline的公平配置**：优先沿已登记的原论文/官方实现与项目MS适配合同，逐项注明出处及必要适配；允许各模型不同LR、层数、patch等原生参数，不强行统一AMD的3e-5等值。推荐首先比较“每模型一套有出处的固定配置、无额外验证搜索”的最小方案；若需搜索，在看正式结果前统一登记各模型的配置数/计算额度、验证选择和失败处置机会，另算成本，不能暗中只调J。固定配置比较的局限须披露，不宣称各baseline已充分调优。
- **best/early-stop与预算**：目标validation MSE有限且严格下降才更新best、等值保留早epoch为既有合同。是否early-stop、最大epoch、patience、LR schedule、train/eval batch和精度等仍待锁定；旧§4的30/5仅是待比较推荐值，不自动批准、也不机械施加所有baseline。若采纳早停，计数/重置、完成原因及resume须统一定义并验收；不为best更晚而调参。旧0.5% practical-effect、0.1%组合增量及2倍成本线均不由本稿采纳为新冻结前置。
- **test与成本**：结构及选择协议先冻结，正式test只在M6按批准任务用于最终评价，不能每epoch读test或据test挑配置；train-only scaler、完整目标元素/尾批/宏汇总保持。记录实际epochs/steps、参数、完整run时长、峰值显存、串行等效工作量、物理GPU占用与队列makespan，计入准备/验证/保存/失败成本；独占模型效率与并发调度效率分开，旧有限并发不自动扩域。PJM split/retraining/F_PJM及既有元数据缺口仍待闭环。

### 8.4 同一训练只计一次与条件预算

推荐预先把dataset/version、target/input/order、fold/H、split/scaler、模型结构/初始化、训练/选择配置、seed、source及科学身份绑定为同一run；表格/阶段只是引用用途，不构成新训练身份。主表A/J与UrbanEV F4模块消融完全匹配时引用同一run，不为换表/换阶段重训。若未来另批验证配置搜索，同一合格最终配置run仅在训练前协议一致且冻结/test边界可核验时计入；不匹配则不扣除。M5默认不预跑主表，必要工程fixture/smoke不作正式性能run。M4旧artifact不升级、不改purpose/manifest，不跨协议恢复；旧§5的116-run迁移方案不成为新默认前置。

按上述矩阵、每格一套最终配置、单seed的**训练次数Scenario**为：

| 分账 | 次数 | 依据 |
|---|---:|---|
| 六域主表 | `8×(40+F_PJM)` | UrbanEV24＋四标准域16＋PJM实际fit数；含A/J |
| UrbanEV N/S | `2×24=48` | A/J已计主表，四臂合计96但只新增48 |
| UrbanEV/PJM TargetOnly | `24+F_PJM` | 原覆盖保留 |
| UrbanEV F1–F3 A/J | `3×2×24=144` | F4/F0不重复 |
| 合计 | **`536+9F_PJM`** | F_PJM=1时545，仅名义情景 |

相对旧六域模块消融情景`568+11F_PJM`少`32+2F_PJM`（F=1为34），来自其他四标准域与PJM不默认追加N/S；不是删主表、TargetOnly或输入消融。**不再额外相加旧M5的112+4F_PJM，也不把旧463剩余数套到新矩阵。** PJM fit、实际窗口/Adam步、不同baseline训练上限、搜索/工程/效率及耗时均待核验/待批，不能据545乘旧时长估实测成本，不能统一乘未批准的30轮。该算术不是训练预算批准。

### 8.5 必须联动的canonical替换语义（本轮不生效）

| 定位及原要求 | 拟替换语义（Proposed） | 保留约束 |
|---|---|---|
| §0.1中“其他多数据集、独立模块及practical-effect要求保持”“M5正式筛选前锁定阈值”及对§9.5/§21的前置引用 | M5依正式协议/必要工程准备、既有开发证据及明确风险接受提请冻结，不默认增加完整多域效果筛选；正式效果及模块贡献在M6按批准矩阵评价。冻结与性能裁决分别登记 | M4原gate/H192/1%与来源、六域主表、test隔离和单seed限制不改；不是认定旧前置已满足 |
| §9.3“仅外生”“仅局部”均覆盖六域 | 完整A/N/S/J消融改为UrbanEV F4全6折×4H；其他五域保留主表A/J，不默认N/S | 原baseline、TargetOnly两域及§9.4输入消融保留，旧U/M/开发identity不重定义 |
| §9.5第1/2/4/5项及末段：0.5%安全、多域独立改善、代价匹配、预锁practical-effect作为M5筛选保留条件 | 推荐不再作为结构冻结的性能通过前置；改为M6逐任务效果/代价及UrbanEV模块贡献的报告责任，不承诺正收益，不另设新量化通过线。旧0.5%可作为历史合同参照，不能写成新合同已Passed；旧多域单模块要求明确被替代，不能声称用主表J/A满足它 | §9.5第3项稳定性Not evaluated及正式证据不得由M4升级保留；负向结果必须报告，失去的外部单模块证明能力明确披露 |
| §9.6.1的六域单模块展开`568+11F_PJM`、§9.6.3以579为基数的情景及§9.6.4阶段接口 | 正式矩阵计数改为本节`536+9F_PJM`条件式，取消额外默认M5筛选账；M5确认协议/配置与必要工程，M6执行批准清单；后续据实际配置另核工程/搜索/效率成本 | 所有历史Measured记录不改，旧Scenario只作历史；F_PJM未知、单seed、并发适用边界与预算授权分离；第四章计数/职责不顺带更改 |
| §19 M5“公平多数据集与独立模块筛选…practical-effect冻结”、M6“原范围消融” | M5完成正式协议/必要接入验收并提请用户结构冻结，采用§8.1新前置；M6承担六域主表、UrbanEV模块消融及保留的输入/TargetOnly消融和效率报告 | 不凭文档把工程能力记Passed，不因协议确认启动M6，不提前作最终冻结决定 |
| §21“单模块正式通过线由M5执行”第1/2/4/5项、首段“两个…通过消融” | 移除其作为冻结前多域性能通过条件的职责；两个模块仍须在UrbanEV独立/组合消融中接受正式评价，有效性由结果决定。M6无收益或负交互须收缩论断，若要改变结构/矩阵须另行决定，不自动救援 | 不新增候选、不复活失败路线、不改来源或历史gate；第3项seed限制不变；不能关模块后仍声称两模块已有效 |

这是**实质合同修订提案**，不是解释旧合同已自然允许省略。用户确认前canonical旧科学条款仍有效，本轮不会依据新语义冻结；旧§4–6暂停且未获批准，旧§7待决清单以本§8最新审议范围为准。无需修改Closed M4来实现修订。

### 8.6 一次确认范围与停止点

建议用户一次审议：**采用§8.1/§8.5的新阶段职责及冻结前置；采用§8.2精确矩阵并接受其研究主张限制；采用§8.4同run引用和条件计数；按§8.3锁定配置来源/公平选择原则，并明确具体配置、early-stop/预算与PJM fit仍待后续成表批准。** 可以确认这套协议方向并保留上述数值未决项，不将其解释成已批准30/5、545次训练、13项验收或J冻结。材料保持可供一次确认，后续必要工程与最终冻结各需对应授权。

本轮仅修改canonical审议摘要及本M5文件头/新增§8；旧§§1–7（含§4–6）正文保持原字节。零模型、零训练、零动态测试、零真实观测/checkpoint读取或哈希、零历史artifact重审；未改loader/代码/工具/环境，未stage/commit/push或另做closure。文档diff、范围/空白、前后SHA静态核验结果随回执列报；ChatGPT尚未读取/审核本轮新字节。

## 9. 统一正式协议确认、结构来源展开与ETTh1 loader实施（2026-09-18）

### 9.1 本次授权与合同替代

起点与用户指定一致：branch=`AMD-paper-repro-custom-modules-v1`，HEAD/tracking/live remote=`3e43a55327eac641445ab5a43d06424d7fc5ed3f`、0/0，index空、canonical modified＋唯一M5 untracked。修前SHA：canonical=`9cc0b25f85265def042b6da99c3bd08de142341d8dc58b17e165b5b8a5aeb5fe`，M5=`cc2fb9d3e6bb7ebfd717eb7810fc213946a2922fcee4698c4a1b237ec614873b`；AGENTS/Closed M4沿§1原SHA，baseline仍指`fa9665627e6fcfb1d0c2bc22d943ca9666304fd6`。本轮内容修改M5，但它在Git仍是新增未跟踪文件。

用户本次明确采用正式协议优先：M5完成正式协议/必要工程准备，依据既有开发证据与明确风险接受提请用户作最终结构冻结；M6执行正式效果评价。canonical §0.1、§5.5/§9.2、§9.3、§9.5、§9.6、§19、§21已联动，§17训练公平引用仅对第三章消歧。旧多域独立模块效果和新practical-effect冻结前置已被替代，**不是旧门槛已满足**；旧§4–8正文及Closed M4不改。原18项gate Not passed、H192 J/N退化1.542951806%超过原1%、原24-run停止和真实来源保留，J未冻结、不按dataset/H混用模型。

确认六域原8模型主表、UrbanEV F4全6fold×4H完整A/N/S/J模块消融；其他域不默认N/S。原UrbanEV/PJM TargetOnly、UrbanEV F1–F3 A/J和Last Observation保留，F0的J为N/A。A/J主表与消融训练配置/科学身份兼容时引用同一run，不为换阶段/表格重训；不升级M4旧artifact、不改purpose/manifest或跨协议恢复。六域主表只能检验完整模型跨域表现，UrbanEV模块结论不能扩成六域单模块有效；负向结果照实报告并收缩论断。结构冻结不等于效果Passed。

### 9.2 已确认训练项与规划账

同一数据集全部baseline、AMD/J和实际消融严格统一：

| dataset | train/eval batch | epochs | LR | early-stop | 唯一runs | 最多run-epochs |
|---|---|---|---|---|---:|---:|
| UrbanEV | 128/128 | fixed10 | 3e-5 | off | 408 | 4080 |
| PJM | 128/128 | max20 | 5e-5 | patience5 | 9 | 180 |
| ETTh1 | 128/128 | fixed10 | 5e-5 | off | 32 | 320 |
| Weather | 128/128 | fixed10 | 5e-5 | off | 32 | 320 |
| ECL | 128/128 | fixed20 | 3e-4 | off | 32 | 640 |
| Exchange | 512/512 | fixed10 | 3e-4 | off | 32 | 320 |
| 合计 | — | — | — | — | **545** | **5860** |

Adam betas=(.9,.999)、eps=1e-8、wd=1e-7；固定LR，无scheduler/warmup/OneCycle；float32、梯度累积1、workers0、每进程4线程。train shuffle/drop_last=True，eval不shuffle/保留尾批；每任务每模型单配置、0额外搜索、from-scratch。seed=[2024]、std=N/A、稳定性Not evaluated；既定匹配初始化子流保持。不更改TF32/确定性等环境设置，不加AMP/compile。有限目标validation MSE严格下降才更新best，相等留早epoch；PJM严格改善重置耐心，连续5轮无改善完成后停止（最多20轮）。fixed任务无early-stop；旧30/5与其他效果/成本提案不采用。论文称统一训练协议比较，不声称各模型最优，也不猜测其他baseline曾搜索。

PJM F=1已确认，时间顺序train=floor(.7n)、test=floor(.2n)、val=n−train−test；不rolling、不train+val重训。批准的是政策，不核销实际市场/as-of/单位/版本事实缺口。其他目标/有序输入/T/H/MS/split/scaler与全目标元素/宏指标口径不变，不新增未来信息。

计数分解：主表328＋UrbanEV N/S 48＋两域TargetOnly 25＋UrbanEV F1–F3 A/J 144=545；UrbanEV为192＋48＋24＋144=408，PJM8＋1=9，四标准域各8×4=32。不额外相加旧112+4F筛选。545/5860是规划，**本轮正式训练授权为0**；未知窗口/steps/实测耗时仍Unknown，旧时长不是新工期，1.4仅并发假设。并发配置没有本轮新增验证或扩域授权。

### 9.3 原生结构、来源与未闭环项（只读，无模型运行）

训练项以§9.2为唯一现行值；以下原生结构值与其分开。共同AMD沿现有合同n_block=1、mix_layer_num=3、mix_layer_scale=2、alpha=0、dropout=.1、RevIN=True、context32零占位；A关闭新增模块。S2沿canonical §1.1.1：d64/K8/alpha=.5、epsilon1e-6、attention dropout.1、gamma=.001、aux先target后、latent FFT、no-Koopman目标残差。THLS沿§2有符号三路形状、hidden8/FFN16、feature-LN eps1e-5、dropout.1、eta=.001及既定初始化。共有AMD/S2/THLS和train generator用既定2024匹配子流；不放宽旧M4构造/恢复identity。

| 域 | AMD patch | AMD骨干layernorm | THLS大/小核 | 状态 |
|---|---:|---|---|---|
| UrbanEV | 12 | True | 7/3 | 用户本次确认 |
| PJM | 24 | True | 31/5 | 用户本次确认；新任务接入仍缺 |
| ETTh1/Weather | 16 | True | 31/5 | 用户本次确认 |
| ECL | 16 | **False** | 31/5 | 用户本次确认；THLS内部feature-LN仍开启 |
| Exchange | 4 | True | 31/5 | 用户本次确认 |

AMD标准域核对来源为本仓库`scripts/ETTh1.sh`、`scripts/Weather.sh`、`scripts/ECL.sh`、`scripts/Exchange.sh`，commit为本轮HEAD；其骨干patch/layernorm与上表相符。来源旧M输出与训练项不搬入正式MS。本轮只读既有两个参考仓库：TimeXer commit=`76011909357972bd55a27adba2e1be994d81b327`、ModernTCN commit=`56a9a2c018385cd5acef015378cae7f084d1b11c`，各自Git clean；没有重新择源/调优。

**上下文缺失不补造**：本次消息所指“ChatGPT正文UrbanEV/PJM结构表”和“上一轮已选定标准域来源对应关系”未附在当前可见正文；旧M5仅有AMD家族建议和来源定位，不能推出全部baseline的已选结构。已请求只补该缺失文本，不重新征询训练参数。下表登记实际静态解析到的来源值，**不把它们自动登记成缺失的项目固定结构表**；完整逐H defaults＋显式覆盖、路径/commit/SHA见本包`source-config.json`。

| 已读来源（相对各参考仓库） | 实际静态展开；非新性能选择 |
|---|---|
| TimeXer `run.py`及`models/DLinear.py` | moving_avg25、individual=False |
| TimeXer `run.py`及`models/PatchTST.py`、`models/iTransformer.py` | 默认d_model512、d_ff2048、n_heads8、e_layers2、factor1、dropout.1、activation=gelu；PatchTST构造默认patch_len16/stride8。不能据此自动给UrbanEV T12选patch16 |
| TimeXer `run.py`及`models/TimeMixer.py` | 默认e_layers2/d_model512/d_ff2048、channel_independence1、moving_avg25、decomp_method=moving_avg、down_sampling_layers0/window1/method=None、use_norm1；不是已确认的每标准域配置 |
| TimeXer `models/TiDE.py`及`run.py` | 默认hidden/res_hidden512、encoder2/decoder1、temporalDecoderHidden2048、feature_encode_dim2、dropout.1；动态特征宽度由freq映射，缺失正文的信息集适配不能由默认补齐，不引入未来真实协变量 |
| TimeXer `scripts/forecast_exogenous/ETTh1/TimeXer.sh`＋`run.py` | H96/192/336/720分别d_model512/128/512/512，d_ff512/128/512/2048；e_layers2、factor3，继承heads8/patch16/dropout.1/use_norm1。H720缺d_ff显式值，故实际为默认2048，不能补成512 |
| TimeXer `scripts/forecast_exogenous/Weather/TimeXer.sh`＋`run.py` | 四H均e_layers1/d_model128/d_ff2048、factor3、heads8/patch16/dropout.1/use_norm1 |
| TimeXer `scripts/forecast_exogenous/ECL/TimeXer.sh`＋`run.py` | e_layers四H=1/1/1/3，d_model512/d_ff2048、factor3、heads8/patch16/dropout.1/use_norm1 |
| TimeXer `scripts/forecast_exogenous/EPF/TimeXer.sh`中PJM命令＋`run.py` | e_layers3、d_model512、d_ff默认2048、patch24、heads8、factor默认1、dropout.1/use_norm1；不是邻近NP命令显式d_ff512。source batch16不覆盖本次128 |
| ModernTCN `ModernTCN-Long-term-forecasting/scripts/{ETTh1,weather,ECL,Exchange}.sh`＋同目录`run.py` | 四域num_blocks=[1]、dims=[64,64,64,64]、large=[51]/small=[5]、use_multi_scale=False/small_kernel_merged=False；前三域patch_size8/stride4，Exchange1/1；ffn_ratio依次1/8/8/1，dropout .3/.4/.9/.2，head_dropout 0/0/0/.6；其余默认revin1/affine0/subtract_last0、decomposition0、individual0、downsample_ratio2、stem_ratio6、dw_dims=[256,256,256,256]。记录原列表而不猜改stage宽度 |

来源脚本的T（如TimeXer标准域96、ModernTCN标准域336/720、Exchange6）、M标签、batch/LR/epoch/scheduler均不得覆盖已定项目T/MS及§9.2；没有Exchange专属TimeXer外生脚本的已选绑定，不借别域数值冒充。所有结构缺项保持Missing，已见官方值不是项目已选值。

**真实实现冲突**：`models/tsAMD_enhanced.py::__init__`旧THLS/NSJ守卫要求norm且layernorm=True，NSJ仅旧UrbanEV/ETTm1形状；本次ECL骨干False与旧守卫不兼容，但THLS自身feature-LN必须保留。`main::_thls_ms_interface_contract/_build_model`又按ETTm1选5/31、其余3/7，不能承担本次“其余31/5”的正式域映射。两项均为后续正式接入blocker，本轮不改模型/main或旧identity；不能为绕过守卫把ECL设置改回True、关THLS LN或冒用ETTm1身份。

### 9.4 两文件静态实现与精确验收清单

`utils/dataloader.py`新增keyword-only `access_policy`，默认`train_validation_test`保留旧路径/metadata；`train_validation_only`仅允许dataset_id=`ETTh1`。受限分支用CSV record迭代只解析header及11520条前缀记录，不让pandas先解析整表再slice；固定train8640/val11520，val借T历史、标签覆盖完整H。仅train-fit scaler和train/val transform，不构造test dataframe/Dataset；`get_test`在访问前抛PermissionError。metadata的raw_rows=None、parsed/used_rows=11520、full_file_verified=False，已读端点与声明的test_end14400分开，test观测未核验；不重算比例、不读取/哈希真实文件。命名目标、输入顺序/MS/目标inverse_transform继续复用原路径。

`tests/test_dataloader.py`新增独立`ETTh1RestrictedLoaderTests`，不继承旧测试方法，复用合成TemporaryDirectory/CSV helper；不增skip或放宽旧断言。以下8个完整ID已静态冻结，前缀为`test_dataloader.ETTh1RestrictedLoaderTests.`：

1. `test_prefix_stops_before_poisoned_test_records`
2. `test_four_horizons_exact_windows_context_and_target`
3. `test_scaler_fits_train_only_and_never_transforms_test`
4. `test_named_nonlast_target_order_and_inverse_transform`
5. `test_tail_batches_and_training_generator_are_preserved`
6. `test_forbidden_test_access_and_truthful_prefix_metadata`
7. `test_invalid_policy_dataset_prefix_and_schema_are_rejected`
8. `test_legacy_default_matches_explicit_full_access`

追加的五个旧完整ID前缀为`test_dataloader.DataLoaderTestCase.`：`test_public_split_endpoints_are_unchanged`、`test_real_window_counts_and_loader_policies`、`test_inverse_transform_accepts_full_or_s_ms_target_width`、`test_explicit_train_generator_can_be_saved_and_restored`、`test_missing_target_is_rejected_for_s_and_ms_but_not_m`。逐方法静态确认均为纯split算术或由`make_generic_loader/write_csv`创建的合成fixture；带real字样的方法也只用tiny.csv，不读取真实data。机器可读13个完整ID见`acceptance-result.json`。

新CSV最大14400行、3特征＋date，均低于每CSV20000行×8特征；ETTm1默认兼容子例仅用内存合成57600×3 frame替代raw-read返回，不生成超行数CSV，不接触真实ETTm1。正常unittest setUp/tearDown生命周期保留；将来获支持的受限入口必须在import/fixture之前绑定唯一/tmp根及64MiB上限，子进程继承，不能直接运行这些测试替代guard。

### 9.5 工具阻塞、实际账目与审核停止

**动态验收Blocked，未运行，不记Passed。** 静态查明现有`tools/restricted_regression/acceptance_driver.py::all_ids/validate_inputs`固定1 repair＋18 new＋314 inherited；`worker`发现全测试并按固定阶段选组，无本包13 ID选择入口；`validate_execution_authorization`绑定M4旧批准文件与24-file源码指纹；`current_policy.require_scope`也绑定THLS旧policy/approval。新增8方法/loader源码后不能以旧binding承担本包。直接调用底层execute_cases、改inventory/policy或另建driver都将绕过/放宽现有绑定，故均未做；也没有运行会读取旧M4预算证据的preflight。

依用户“工具绑定不支持时停止”要求停在业务导入/fixture前：首次方法调用0/13；passed=0、failed=0、error=0、skipped=0、unexecuted=13；机械修复复验未启动、调用0，累计0/26。这是执行前阻塞，不是测试失败或13项通过。未创建/tmp fixture、0 CSV、0 GPU/模型/optimizer/forward/backward/Adam、0真实观测/test/checkpoint读取及哈希、0历史artifact重审。唯一canonical补丁第一次因末尾上下文不匹配原子拒绝，随后修正文档hunk成功，无动态调用或复验额度消耗。

永久证据仅本包目录：`/public/home/yueweiting/大论文/amd-execution-evidence/m5/m5-etth1-prefix-6qg_f8j0/`。`session.log`登记操作与阻塞；`acceptance-result.json`含完整13 ID/首次与复验账/逐ID unexecuted及静态依据；`source-config.json`含两参考仓库commit、defaults与逐H显式解析值/源码SHA；最终`static-review.json`、`changes.patch`和`sha256.json`保存静态差异及范围/SHA核验。没有通用新执行器、第二milestone或旧证据修改。

本轮仅做标准库AST/compile语法检查（未import业务）、文档/差异/范围/SHA检查，不能替代动态验收。合同落地完成；结构来源部分展开，缺失正文绑定及ECL/正式域构造冲突保留；loader代码和测试定义完成、engineering acceptance未完成。结束停在修后四文件与首包证据待ChatGPT审核，不推进下一接入包、不训练、不冻结J、不启动M6；未stage/commit/push，不另做closure。ChatGPT尚未读取本轮修后字节。

## 10. 独立作者来源实测、受限验收解阻与TiDE暂缓（2026-09-18）

### 10.1 授权、起点和证据边界

本节依据用户最新完整指令执行；此前未执行的“来源准备＋解除测试阻塞”指令整体作废，不叠加范围。起点HEAD/local/tracking/live remote均为`3e43a55327eac641445ab5a43d06424d7fc5ed3f`，ahead/behind=0/0、index空；canonical/loader/tests modified、唯一M5 untracked，与指定起点一致。修前canonical SHA=`af143b44de002c4df1f4f6b9a4bb68cec3d8cb8e74e63ee2055ed981efa61f66`，M5 SHA=`1ae4e7335ca5c1bf4995b8b3e7869282fcf966b0adc5a46ae3f85b3e912abedf`；loader/tests沿§10.5所列SHA。本轮允许在既有dirty状态验收，不把它误写clean或要求先closure。

本包唯一永久证据目录：`/public/home/yueweiting/大论文/amd-execution-evidence/m5/m5-source-smoke-8sr2d3d_/`。旧`m5-etth1-prefix-6qg_f8j0`未修改。适用AGENTS已核对；其SHA保持`4dfbc7161b731e20674d28eb1b4759dd1ffbc403ae5217add26efd7433091e6c`，Closed M4保持`3b29d43624de290ebdec6a1701066f6dded33f8e788d8b183cc1fa416df4c11e`，baseline仍指`fa9665627e6fcfb1d0c2bc22d943ca9666304fd6`。没有重审旧artifact/曲线或读取/哈希真实观测、历史/正式checkpoint。

### 10.2 八个独立作者来源与环境

以下目录均在`/public/home/yueweiting/大论文/`。五个缺失仓库按用户指定URL实际clone；已有三仓库只核对，不pull。所有作者源码工作区结束仍clean，无作者源码兼容补丁；Sonnet兼容改动仅在外置调用适配器的Adam参数，不改作者模型或Lightning基类。

| 来源/目录 | remote | 完整commit | 许可证与本轮状态 |
|---|---|---|---|
| AMD-upstream（新增clone） | https://github.com/TROUBADOUR000/AMD.git | `000d377a1ed8946aa817ff357cdf1de64b99abb9` | MIT；clean |
| DLinear / LTSF-Linear（新增clone） | https://github.com/cure-lab/LTSF-Linear.git | `0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6` | Apache-2.0；clean |
| PatchTST（新增clone） | https://github.com/yuqinie98/PatchTST.git | `204c21efe0b39603ad6e2ca640ef5896646ab1a9` | Apache-2.0；clean；supervised |
| iTransformer（新增clone） | https://github.com/thuml/iTransformer.git | `c2426e68ca13f74aaec08045c5c724d8ad328124` | MIT；clean |
| TimeMixer（新增clone） | https://github.com/kwuking/TimeMixer.git | `e24610583b36fdd8c76cc17a8df4e65759a5f460` | Apache-2.0；clean；原TimeMixer预测模型 |
| ModernTCN（既有） | https://github.com/luodhhh/ModernTCN.git | `56a9a2c018385cd5acef015378cae7f084d1b11c` | MIT；clean |
| TimeXer（既有） | https://github.com/thuml/TimeXer.git | `76011909357972bd55a27adba2e1be994d81b327` | tracked tree无LICENSE文件；不推断许可；clean |
| Sonnet（既有） | https://github.com/ClaudiaShu/Sonnet.git | `bf3d4801d34c5e7261718490f287c6fb15cadfdb` | setup.py有MIT classifier，tracked tree无LICENSE正文；不补造许可文本；clean |

模型/训练入口分别为：AMD `models/tsAMD.py::AMD`/`main.py`；DLinear `models/DLinear.py::Model`/`run_longExp.py`；PatchTST `PatchTST_supervised/models/PatchTST.py::Model`/同目录`run_longExp.py`；iTransformer `model/iTransformer.py::Model`/`run.py`；TimeMixer `models/TimeMixer.py::Model`/`run.py`；ModernTCN `ModernTCN-Long-term-forecasting/models/ModernTCN.py::Model`/同目录`run.py`；TimeXer `models/TimeXer.py::Model`/`run.py`；Sonnet `sonnet/mts_model/models/Sonnet.py::Model`/`scripts/run_experiment.py`。入口均按各自来源使用，未借其他作者仓库附带baseline。

必要导入及源码闭包先静态解析、SHA绑定；`source-inventory.json`、工具`m5_authorization.json`记录URL/commit/license/dependency文件、完整配置与预绑定源码路径/SHA，每个成功尝试`trainability.json::imported_sources`记录实际导入路径/SHA，`source-results.json`记录其确定序列化指纹。AMD-upstream不替换项目实际A或baseline；完整Sonnet仅核验S2来源，不新增候选，也不替代项目S2/J证据。

原`/public/home/yueweiting/miniconda/envs/amd`只读使用。必要依赖在新venv `/public/home/yueweiting/大论文/amd-execution-envs/m5-source-smoke-8sr2d3d_/`（system-site-packages）安装，先保存dry-run解析，确认无torch/NumPy/CUDA替换再安装。实用栈保留torch2.0.1、numpy1.24.3、pandas2.0.3、scikit-learn1.3.2、scipy1.11.4；新增einops0.8.1、reformer-pytorch1.4.4、lightning/pytorch-lightning2.0.9、torchmetrics0.11.4、gluonts0.14.4及已解析依赖。它是兼容的smoke环境，不冒称Sonnet requirements中的torch2.5.1/lightning2.6.0环境复现。完整安装、解析与freeze日志在本包；`pip check`仅报告继承triton2.0.0缺cmake，未为未使用的compile路径扩装。未升级/卸载原环境、未改系统/驱动/TF32/确定性/MIG/MPS，无AMP/compile或空间/TiDE环境。

### 10.3 窄用途绑定与固定smoke配置

新增一个`tools/restricted_regression/m5_entry.py`入口、一个`m5_model_adapter.py`调用适配器和精确授权/来源清单`m5_authorization.json`；guard/sitecustomize仅增加M5分支，更新bundle。旧M4 policy、inventory、授权文件、预算逻辑及证据不改、不读取旧授权来放行M5。复用`run_restricted.execute_cases/LedgerResult`的正常unittest生命周期与原文件访问保护，不discover整套再筛选。业务导入前校验purpose、精确ID、来源SHA、bundle和唯一fixture根，绑定源码加载器不执行旁边旧pyc；Python子进程继承配置。该保护是可信代码的Python访问/预算instrumentation，不宣称内核级审计或通用系统沙箱。

`m5_etth1_loader_only`禁模型/optimizer构造与CUDA初始化；`m5_author_source_trainability`仅允许八个源闭包和本包自产checkpoint。其他作者根、非绑定项目文件、外部观测/权重扩展名和历史证据读取被拒绝；写入限定本包/fixture根。四项标准库toy各一次Passed（共4、0业务forward/backward/Adam）：精确ID前置拒绝、来源/用途错误拒绝、合成保护路径拒绝、正常生命周期及Python子进程继承。业务审计中保护拒绝为0；只有两次预期toy拒绝，不使用真实资产作探针。

首次运行前已静态固化如下小配置（完整字段在机器清单；均非正式结构选择、无搜索）：

| 原生模型 | 工程smoke结构与约束 |
|---|---|
| AMD-upstream | n_block1、patch16、k3/c2、alpha0、dropout.1、norm/layernorm=True、原生固定MoE hidden2048/8 experts；全7输出保留 |
| DLinear | individual=False；原生moving_avg25、96→24线性头 |
| PatchTST supervised | e_layers1、d_model16/d_ff32、heads4、patch16/stride8、padding=end、RevIN1/affine0、decomposition0、dropout.1、fc/head dropout0 |
| iTransformer | e_layers1、d_model16/d_ff32、heads4、factor1、gelu、dropout.1、use_norm1、class_strategy=projection |
| TimeMixer | e_layers1、d_model16/d_ff32、channel_independence1、moving_avg25、下采样1层/window2/avg、use_norm1、无未来时间特征；不使用TimeMixer++ |
| ModernTCN | num_blocks=[1]、dims/dw_dims=[8,8,8,8]、核7/3、ffn_ratio1、patch8/stride4、stem_ratio6/downsample2、dropout.1/head0、RevIN1/affine0、无多尺度融合/分解/核合并 |
| TimeXer | features=MS、patch16、e_layers1、d_model16/d_ff32、heads4、factor1、dropout.1、use_norm1 |
| Sonnet | 完整Model、d_model16、n_atoms4、alpha.5、RevIN=True、downsample_factors=[1]、c_out1；原生complex64 Koopman参数/计算保留，不做丢弃虚部的float强转 |

共同seed2024、float32实数输入/参数、B4/T96/H24/7列、固定合成train28/validation8、workers0/threads4；Adam lr5e-5、wd1e-7、betas(.9,.999)、eps1e-8。两轮各7更新，加自产last恢复后2更新，最多16；每轮/恢复后各2 validation batch，eval一致性仅同一batch。保留原生多输出，选合成最后一列目标MSE；AMD原生辅助返回检查finite，不纳入本次目标MSE。无真实时间/未来协变量，四参数接口的mark/decoder入参为None。无scheduler/AMP/compile，不要求loss单调或达到效果阈值。

### 10.4 实际原生训练链结果与一次机械修复

八源均在独立CUDA:0子进程串行检查，启动前A800可用显存81222MiB；未启动并发探针、未终止无关进程。每尝试父进程300秒上限、方法290秒；PyTorch allocator设3GiB分配上限并检查reserved≤4GiB。以下峰值是**PyTorch reserved**，不是NVML整进程峰值；整进程4GiB峰值未单独采集核验，不能将allocator检查冒充这一完整资源验收。未以小smoke推断正式任务/完整epoch显存或工期，也不为补峰值记录自动增加复验。

| 来源 | 首次Adam扣账 | 复验扣账 | 完整通过步骤 | 成功尝试墙钟秒 | reserved峰值MiB | 最终训练链 |
|---|---:|---:|---:|---:|---:|---|
| AMD-upstream | 16 | 0 | 16 | 18.490 | 110 | Passed，CUDA:0 |
| DLinear | 16 | 0 | 16 | 6.676 | 22 | Passed，CUDA:0 |
| PatchTST | 16 | 0 | 16 | 8.027 | 22 | Passed，CUDA:0 |
| iTransformer | 16 | 0 | 16 | 10.032 | 150 | Passed，CUDA:0 |
| TimeMixer | 16 | 0 | 16 | 8.728 | 26 | Passed，CUDA:0 |
| ModernTCN | 16 | 0 | 16 | 9.579 | 22 | Passed，CUDA:0 |
| TimeXer | 16 | 0 | 16 | 8.285 | 152 | Passed，CUDA:0 |
| Sonnet | 1（失败调用） | 16 | 16（独立复验） | 14.535 | 26 | Passed，CUDA:0，外置API兼容 |

每个Passed均独立完成forward/目标loss/backward/Adam、全输出/loss/已产生梯度/参数finite、每步至少一个目标非零梯度及实际参数变化、实际validation、严格validation-best保存、last含optimizer/RNG/步数、本包last严格恢复、同batch eval逐元素完全相等及2步续训；不要求未参与目标的所有参数有梯度。自产best/last/resumed-last只放本包，不进正式artifacts。每模型完整证据来自一个成功尝试，没有拼接失败/不同版本结果。

Sonnet首次在第1次Adam调用触发torch2.0.1 `_multi_tensor_adam` 对complex参数weight decay的shape错误（2 vs 4）；此前forward/backward已发生，但可能存在部分参数组更新，故**保守计1次调用，不声称完成一次可验证更新**。原journal、stderr、空`result.json`均保留；后者因Lightning导入后的locale造成默认ASCII结果编码失败，终态由完整journal另存`Sonnet-first-terminal-recovered.json`，不覆盖失败文件。一次机械修复仅：外置Sonnet Adam显式`foreach=False`使用等价单张量API，以及入口read/write/journal显式UTF-8。无模型数学修改、删模块、放宽finite/equality检查或新配置/seed。`mechanical-compatibility.patch`及`first-tool-bytes/`保存精确差异/原工具字节；guard/bootstrap/四toy方法未改，未重复toy。七源首次计算路径不受Sonnet专属API修复影响，分别保留首次完整证据；Sonnet第二次从头完成全链。

总账：首次113/128 Adam调用，唯一复验16，累计**129/256**；完整成功链共128次已验证更新，失败成本1次保守扣账，0额外搜索。原生CLI/完整论文runner八源均**Not run**：本轮实际测的是外置受限驱动直接调用作者原生Model的训练链，不仅import/help，也不冒称完整runner已复现。可核对实际命令在各尝试`command.json`和launch日志：独立venv的`bin/python tools/restricted_regression/m5_entry.py --source <清单名>`；Sonnet另有`--attempt 2 --repair-reason ...`。已用包/尝试不可重复执行，入口会拒绝覆盖证据或超额；这些是复现记录，不是再次运行授权。

### 10.5 ETTh1 loader正式工程验收账

§9.4八个`ETTh1RestrictedLoaderTests`及五个`DataLoaderTestCase`完整ID原样绑定机器清单，五旧方法再次静态核对为split算术/合成CSV。CPU loader-only首次**13 passed、0 failed、0 error、0 skipped、0 unexecuted**，正常setUp/tearDown，墙钟13.483秒；复验0，累计**13/26**。loader用途forward/backward/Adam均0、CUDA reserved0，禁止模型/optimizer/GPU钩子在fixture前生效。

唯一合成fixture根`/tmp/amd-m5-8sr2d3d_-7f0p1cul`，CSV最多14400观测行×3特征（另date），受20000×8及64MiB边界；逐行写与方法前后检查磁盘大小，没有另测全进程瞬时磁盘峰值。ETTm1旧兼容子例仅合成内存frame。测试覆盖受限解析到11520/train-fit到8640、四H与context/完整标签、命名目标与有序aux/MS/inverse_transform、尾批与generator、test前置拒绝/metadata、坏策略/schema/短前缀拒绝和默认旧路径等价。没有真实CSV/test或checkpoint访问。

本轮loader/tests无需机械修复、内容未变：`utils/dataloader.py` SHA=`0200ee5a1c739ef507220ce9c93203a3ff26e8cb420b4830cd3d9eb7021e44f6`；`tests/test_dataloader.py` SHA=`30ef9bf880ced6fef944f9562fa2b8dd12c97fb9530ff7718a725b70680051f1`。它们仍因上一轮修改在Git显示modified，不能记为本轮新改。逐ID/事件/终态和独立预算见`loader-attempt1/{ledger.jsonl,result.json,budget.json,process.json}`与`loader-acceptance.json`。没有重跑旧333/339、旧八项、smoke兼容包或24-run。

### 10.6 当前正式规划、空间例外及剩余阻塞

TiDE **Deferred**：本轮不clone Google Research、不装TensorFlow、不检查/迁移/用其他仓库替代；不永久删除。原545 runs/最多5860 run-epochs是含TiDE完整规划；当前520/最多5600，TiDE25/最多260单列暂缓，不计已完成或消耗。独立用户启动的三个正式包为：包1 A/J六域主表82/最多920；包2其余非TiDE baseline主表221/最多2500；包3剩余消融217/最多2180。包内自动队列，包间不得自动启动；后续结构/配置/评价规则须在包1正式test结果出现前锁定，不据中途效果改模型、删任务或加搜索。同run跨表引用，不升级M4旧artifact。

§9.2正式训练表及共同合同原样保持：UrbanEV128/128固定10、3e-5、early-stop off；PJM128/128最多20、5e-5、patience5；ETTh1/Weather128/128固定10、5e-5、off；ECL128/128固定20、3e-4、off；Exchange512/512固定10、3e-4、off。Adam固定LR/wd1e-7/float32/seed2024/0额外搜索，std=N/A、稳定性Not evaluated；T/H/目标/信息集/MS/split/指标不变。本包smoke不构成任何正式训练授权，未知窗口/PJM fit/耗时仍Unknown，1.4仅并发假设。

空间仅联动canonical §0.1/§1.2/§10：HSTGCN-core←HSTGCN、SADR←ASTGRN、SC-SimGCA←G-STAN三个已批期刊例外保留。当前/未来backbone先在论文1–2个可取得数据集、可比任务/处理/指标/原训练协议上复现原版，再用/改；HSTGCN Boulder公开、3h/6h是一个数据集两任务，北京不可得如实登记。原版与EL-AMD+改造core实验分开，core不另加原论文指标门槛；模块只需来源/工程/项目消融，不要求整篇来源模型指标复现。空间阶段执行前锁预算/参照值/容差，有限排障失败记原因/成本再提替代；新替代backbone/模块要求2023+顶会、优先作者代码。结论仅指定任务通过/未通过/条件不足，不延伸为造假判断。本轮空间clone/源码/数据/环境/模型/toy graph/复现均0，无M7/M8或新候选。

**仍未闭环**：正式六域target/MS/信息集与purpose/manifest/checkpoint/resume/summary接入、已登记Weather/ECL/Exchange/PJM数据及as-of/单位事实、各模型逐正式任务结构配置锁定与适配、公平early-stop/best/汇总、正式预算/冻结及用户启动仍需后续包处理；不把本轮smoke配置当正式结构。§9.3的TimeXer捆绑baseline表仅是旧轮次静态历史，不再作为这些模型当前作者来源。ECL骨干layernorm=False与项目旧J守卫、正式域THLS核映射冲突保持blocked，THLS自身feature-LN不能随骨干关闭；本轮不改main.py/tsAMD_enhanced.py解除守卫。许可证文件缺口和环境pip-check的cmake提示分别如实登记，不影响已实际完成的有限链结论，也不宣称完整环境/正式runner通过。

原M4总gate Not passed、H192失败/原1%线、单seed及development-test限制、原科学序列停止与真实来源不改。源码smoke Passed与loader工程Passed均不等于六域可执行、正式效果Passed或J冻结。收尾仅精确diff、范围、AST/格式/SHA/Git核验；无stage/commit/push/closure或下一包启动。最终仍是本轮授权的dirty工作区，修后字节与本包证据待ChatGPT审核，未声称ChatGPT已读。

## 11. 按模型分组、正式接入与轻量资源probe准备（本轮实际实现）

### 11.1 起点、授权与唯一机器清单

起点实际核验为`99be14ce840e67ecd72720860f5d4ab0247418a1`，local/tracking/live remote一致、0/0、worktree/index clean、untracked none。适用AGENTS及Closed M4分别为用户指定SHA `4dfbc7161b731e20674d28eb1b4759dd1ffbc403ae5217add26efd7433091e6c`、`3b29d43624de290ebdec6a1701066f6dded33f8e788d8b183cc1fa416df4c11e`；baseline仍指向`fa9665627e6fcfb1d0c2bc22d943ca9666304fd6`。本轮先登记文件头15路径再实施，未增加第二milestone/状态摘要。证据根为`/public/home/yueweiting/大论文/amd-execution-evidence/m5/m5-formal-admission-bizsxh0v/`，合成fixture根为`/tmp/amd-m5-formal-jef2lnmg`；不改旧证据、作者仓库、环境、Closed文档、原loader/tests或空间方案。

用户本轮授权必要接入/修复、短验收、真实train/validation前缀连通、固定配置资源probe准备和模型入口；全部未执行旧B/probe补丁作废，额度不叠加。**没有完整probe现场启动、正式训练、J冻结或M6授权。** 旧§4筛选矩阵、§10三包启动不执行；§§1–10保留历史事实。§10.6“PJM fit Unknown”更正：**F=1和train=floor(.7n)、test=floor(.2n)、val=n−train−test已确认**；缺的是版本n/端点及单位、as-of等事实，不rolling、不train+val重训。

`configs/ch3_formal_profiles.json`是唯一任务/配置/执行清单，`utils/ch3_contract.py::generate_tasks/generate_groups`重算并拒绝重复、遗漏或清单偏差。任务ID明确model/dataset/preset/fold/H/seed，A/J跨主表和消融引用同一ID；完整Sonnet、AMD-upstream不加入正式矩阵。

| 模型组 | runs | 最多run-epochs | 内容 |
|---|---:|---:|---|
| AMD/A | 138 | 1440 | 六域主表、UrbanEV F0–F4、PJM TargetOnly |
| J | 113 | 1180 | 六域主表及UrbanEV F1–F3；F0 N/A |
| DLinear / PatchTST / iTransformer / ModernTCN / TimeXer，每组 | 41 | 460 | 原各域主表 |
| TimeMixer | 16 | 200 | 四标准域主表 |
| N / S，每组 | 24 | 240 | 仅UrbanEV F4 |
| 当前合计 | **520** | **5600** | 单配置、seed2024、0搜索 |

TiDE25 runs/最多260继续Deferred，含TiDE545/5860完整规划不删除。Last Observation非训练评价仍保留，不计上述run；本包不宣称已执行其正式评价。

### 11.2 结构来源与实施边界

统一训练项沿§9.2，不用作者脚本的batch/LR/epochs/scheduler覆盖。AMD patch为U12/PJM24/ETTh1-Weather-ECL16/Exchange4；MDM3/倍率2、DDI1、alpha0/dropout.1、AMS8/TopK2/hidden2048及原辅助loss系数1、RevIN开启；仅ECL骨干LN=False。S2 d64/K8/alpha.5/dropout.1/gamma.001与THLS hidden8/ratio2/dropout.1/eta.001保持原数学和匹配初始化；THLS显式U7/3、其余31/5，内部feature-LN保持。独立`ch3-target-ms-formal-v1`声明允许新正式shape，旧M4构造/恢复身份不放宽。

原生结构已从各自作者来源实际静态展开到JSON `sources/defaults_source/structures`，完整路径、commit、文件SHA和解析值逐项登记；外置`resolved-structures.json`保留解析记录。不是从TimeXer附带baseline代取，也不是§10 small-smoke配置。U/PJM使用本次用户精确结构表＋该模型自身默认项。标准域选择下列已选脚本，H相关覆盖分别保存：

| 模型 / 作者commit | 标准域结构来源（相对各自作者根） |
|---|---|
| DLinear `0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6` | 自身`models/DLinear.py`/`run_longExp.py`默认；individual=False、moving_avg25；不改作者初始化 |
| PatchTST `204c21efe0b39603ad6e2ca640ef5896646ab1a9` | supervised `scripts/PatchTST/{etth1,weather,electricity}.sh`；Exchange用自身ETTh1模板 |
| iTransformer `c2426e68ca13f74aaec08045c5c724d8ad328124` | `scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh`及Weather/ECL/Exchange各自脚本；保留脚本逐H层宽/层数 |
| TimeMixer `e24610583b36fdd8c76cc17a8df4e65759a5f460` | 原TimeMixer的`long_term_forecast/*_script/*_unify.sh`，Exchange用自身ETTh1；不换TimeMixer++ |
| ModernTCN `56a9a2c018385cd5acef015378cae7f084d1b11c` | `ModernTCN-Long-term-forecasting/scripts/{ETTh1,weather,ECL,Exchange}.sh`；不启用未用部署分支 |
| TimeXer `76011909357972bd55a27adba2e1be994d81b327` | `scripts/forecast_exogenous/{ETTh1,Weather,ECL}/TimeXer.sh`；Exchange固定2层/d128/FF128/8头/patch16/dropout.1 |

`models/ch3_adapter.py`每baseline只导入自身绑定源码；每worker独立进程，避免同名models/layers污染。TimeMixer保持C输出宽度再选择目标；TimeXer真正features=MS并将指定目标可逆移到末列，其他原生多输出只切指定目标监督。生产A仍为本项目AMD关闭模块路径。未修改作者源码、模型主体数学、环境或旧baseline。

`utils/ch3_data.py`按固定端点逐记录解析前缀，不先整表读后slice；scaler仅fit训练；标准域validation借T历史context但完整H标签均在val，UrbanEV各split独立取窗。数据始终留CPU，batch才传GPU。`ch3_runner.py`实现目标MSE＋原AMD辅助loss、严格有限validation-MSE best（相等留早epoch）、PJM连续5轮无改善停止、完整尾批流式SSE/SAE、fold/H宏汇总、唯一ID拒重、purpose/config/source/data/RNG/optimizer/epoch/步数状态。保存CPU状态，恢复用CPU反序列化；final选择best前释放optimizer，不保存GPU预测全集或带图loss列表。恢复入口要求已审manifest/checkpoint/history指纹与合法本次身份；已完成run不自动重跑，失败目录不删除、不自动fresh。

正式数据及未来test能力单独受purpose/approval约束；正式入口要求实际closure commit/clean、源和环境/硬件绑定、全配置已审、对应任务无mandatory blocker、结构冻结/M6授权以及已审probe报告。当前approval模板全为false/null；不会猜未来commit或自动批准。

### 11.3 实际前缀核验与尚存数据阻塞

最终文本证据：`prefix-final/prefix-results.json`和同目录audit/config/log。只解析获准记录前缀，未解析/统计/评价本任务正式test，未读取/哈希任何旧checkpoint或历史artifact；record-prefix SHA明确不是全文件SHA，不把逻辑记录审计冒充内核逐字节读取追踪。

| 域 | 本轮实际范围与结论 | 仍需保留的限制 |
|---|---|---|
| UrbanEV | F0–F4×6fold共30项，分别止于648/1317/1965/2634/3304/3909；每项实际h3的train/val首batch连通。275节点同序，小时索引；六fold四H标签端点另以合成索引核验 | 其他H真实batch未冒称逐项动态跑过；其窗口数逐项算出。每fold按自己的授权前缀，不把较早fold测试期自然进入后续fold训练误称跨任务test评价 |
| ETTh1 | 读至11520、fit8640；首H96 train8033/val2785窗，小时索引；四H窗口数登记 | 全文件/正式test未核验；不是重跑旧13项 |
| Weather | 读至42157；header保留原U+FFFD字符，ISO时间格式已确认。前缀有1处重复时间戳：0-based行19044与前行均为2020-05-12 06:00:00；loader拒绝继续 | **Blocked**。不删行、去重、排序或改split；频率/版本解释仍未闭环，OT辅助身份限制保留。不得把时间字段可解析写成版本Passed |
| ECL | 读至21044、fit18412，小时索引；H96 train17805/val2537窗，321字段/目标OT保持 | 客户ID、单位转换/处理链仍Not verified，mandatory阻塞未解除 |
| Exchange | 读至6071、fit5311，每日索引；H96 train5120/val665窗 | 匿名列币种、基准币/报价方向与原映射仍Not verified；不猜OT币种 |
| PJM | 仅header＋首行，原3字段/前导空格/拼写保留；首时刻2013/1/1 0:00 | F=1和70/10/20政策Confirmed；版本n/端点、单位、forecast as-of仍Blocked。未为了求n扫描整表，未以时间戳证明forecast当时已发布，也不据缺证据断言泄漏 |

UrbanEV每fold/preset记录CPU数组字节及当时RSS；最终前缀审计进程累计VmHWM=738361344 bytes，这是该审计进程峰值，不是六fold同时常驻或正式多worker RAM上界。数据/元数据缺口不由工程Passed核销。

### 11.4 验收终态、失败成本与复用范围

固定ID见JSON `acceptance`，通过正常unittest生命周期和LedgerResult执行，未discover旧套件再过滤。25个`ContractTests/DataTests`覆盖任务/预算/波次、字段/核映射/声明负例、严格best/耐心、汇总身份/缺项、前缀毒化行/训练fit/尾批、六fold四H标签、目标流式指标、权限/精确ID及恢复拒绝。首次25 passed、修复后25 passed、最终CPU诊断补充后25 passed；每次failure/error/skipped/unexecuted均0。CPU unittest首次25/160、机械累计50/160，未删断言或增skip。

四个模型ID为`test_ch3_formal.ModelTests.test_amd_target_only_resume`、`test_j_ecl_new_declaration`、`test_time_mixer_native_width`、`test_time_xer_true_ms`。分别绑定AMD-U-F0/h3、J-ECL/H96、TimeMixer-Exchange/H96、TimeXer-U-F4/h3；固定正式结构，短接入batch2，**不是正式batch资源测试**。每项CUDA执行2 Adam/6 forward/2 backward，实际参数变化、有限值、目标validation/尾批、严格best/last保存、本包自产恢复/继续更新和跨purpose拒绝；首次4/4、唯一模型复验4/4。两次合计**16 Adam/48 forward/16 backward**，各轮8/24/8低于32/128/64；未重跑八源smoke、旧13项/333/339/24-run。

首个真实前缀流程因UrbanEV原始时间字符串格式不同而失败，已按既有语义改为解析时间比较并新增合成断言。第二流程因Weather解析格式误配失败，其回溯查找不存在的`AMD/strptime.pyx`被guard拒绝；拒绝4次、受保护文件成功读取0，日志保留，未增加pyx白名单。纠正为实际ISO格式后第三流程发现真实重复时间戳而停止；最终诊断流程保留Weather Blocked并完成其他独立域。共4次前缀流程，前三次失败成本不抹除，最终流程exit0只代表报告完成，不代表六域均Passed。guard/身份异常仍停止动态业务，不绕过保护。

占位标准库验收2次：tmux 3.2a实际脱离终端运行，固定波次、日志、第二GPU锁拒绝、波次间STOP通过；另一次SIGTERM验证活动子进程被回收、后续波次未启动、Python子进程继承相同guard SHA。5个入口检查通过：probe/AMD dry-run；未审probe start、未冻结正式start、未知model均前置拒绝，未启动真实负载。若将前缀流程及这些入口/占位检查也保守各计一个CPU方法，则首次33、机械53，仍均≤160。完整流水账见`execution-ledger.json`；CPU/模型/probe额度分别计。

最终CPU诊断只改loader错误信息、按域保留Blocked和Weather时间格式；其后probe准入估计还静态补为取torch peak reserved与NVML采样峰值的最大值，避免忽略allocator峰值。该probe分支本轮未动态执行。**没有追加第二次模型复验**。`model-evidence-reuse.json`核对全部520计算profile不变，明确列出与模型复验时不同的CPU/probe工具seal字节；只复用未变化的实际模型/optimizer/恢复路径证据，不声称最终整个包字节被再次CUDA运行。

修后模型短验收NVML进程PID归属仍Not verified（不再将未匹配记录成0占用）；整卡采样峰值约0.53/1.05/0.55/0.68GiB，实际采样间隔约0.21–1.43s，建议100ms不冒称已达到。CPU affinity=8个可见逻辑CPU，cgroup v1 quota=-1、period=100000；每进程threads4。新probe记录原始NVML进程表和整卡采样；不能归属进程时仅用串行整卡峰值作保守上界，另列Not verified。历史smoke“整进程4GiB Not verified”保持；本包未通过正式batch显存/并发/完整epoch资源验收。

### 11.5 固定组probe：已准备、尚未启动

56个去重代表组、Q=200（48组q4＋8个PJM q1），520 run→profile→组/固定波次均在唯一JSON。标准域真实四H组合；UrbanEV以fold1四H代表，其他fold同T/C/输出/B/结构，标签跨度不改变GPU形状。validation按代表任务实际余数尾批；较小尾批仅沿模型eval中batch独立、无跨batch聚合的结构依据复用，不假称每fold尾批均实测。PJM版本端点未知使实际尾批未知，八个PJM worker目前预登记Blocked，完整probe也不虚构n或尾批；不从Q/正式矩阵删除它们。

每worker：2 Adam→完整validation batch＋实际尾批（整除时登记无余数、第二批为完整batch）→4 Adam；计6 Adam/8 forward/6 backward/2 validation调用。先串行取初始参数/RNG/batch/六步轨迹/最终状态及资源；直接测获准q路，必要时两路分波，禁止任意混合补位。按最大Q预算，三阶段首验上限3600 Adam/4800 forward/3600 backward/1200 validation；额外机械/诊断预留512 Adam，本轮使用0。实际因q1不并行或Blocked而少执行时记真实账，不凑数、不自动重试相同失败配置。

资源预留max(8GiB,10%总显存)，整卡外部占用计入。监测构造、首次Adam、持续更新及验证切换；采样峰值不称连续上界。identity/初始RNG不一致停止全probe；模型数值/生命周期异常阻塞该配置，资源不足才允许下调并发。四路静态不准入记ResourceRejected，短包无收益与资源安全分开；只有对应组实际安全、严格短轨迹且同q任务makespan有收益才采用并发，其余不循环搜索。持续allocated/RSS增长会阻塞组，不默认追加32步。

`m5_formal_entry.py`复用原文件guard/bootstrap、LedgerResult、预算hook和bundle seal，限定CPU/前缀/模型验收/probe/正式worker/占位目的，业务前校验ID/代码/配置/作者来源；旧M4清单/预算逻辑保持。probe的共同源码/权限失败停止全部；局部资源失败日志保留。完整probe、本轮正式训练均为**0次启动/0 Adam**，暂无可靠ETA；用户启动后按首批真实耗时评估，不承诺4–8小时。

### 11.6 可执行入口与停止位置

以下均在仓库根执行。`probe-review-request.json`/`formal-review-request.json`是本包外置待审模板（false/null），**不是许可**；ChatGPT审核并统一closure后，须由相应批准形成实际`probe-approval.json`，绑定真实commit、当前protocol/code/environment/hardware。缺失或不匹配时start拒绝。不能把待审模板的false静默改成true，也不在本轮生成猜测commit的许可。

```bash
cd /public/home/yueweiting/大论文/AMD
CH3_EVIDENCE=/public/home/yueweiting/大论文/amd-execution-evidence/m5/m5-formal-admission-bizsxh0v

# 现在可看计划；当前preflight/start拒绝是预期，未授权完整probe。
bash scripts/ch3/start_probe.sh dry-run
bash scripts/ch3/start_probe.sh preflight --approval "$CH3_EVIDENCE/probe-approval.json"

# 仅修后审核/closure并取得实际许可后，由用户一次启动；tmux后台。
bash scripts/ch3/start_probe.sh start --approval "$CH3_EVIDENCE/probe-approval.json"
bash scripts/ch3/start_probe.sh status
bash scripts/ch3/start_probe.sh logs
tail -n 60 "$CH3_EVIDENCE/probe-controller.log"
bash scripts/ch3/start_probe.sh complete
bash scripts/ch3/start_probe.sh safe-stop
```

probe完成以`probe/complete.json`存在并逐组核对终态/预算/Blocked为准，文件存在不代表所有组Passed；进度为`probe/progress.json`，每worker日志/配置/预算在其组子目录，安全停止只终止本入口子进程并保留产物。没有为全包异常或未执行项伪造complete；未做任何resource probe时不宣称并发度已核验。

```bash
# 后续另获结构冻结/M6执行许可、解决对应mandatory blocker、完成probe审核之后：
CH3_MODEL=AMD  # 也可逐次选 J/DLinear/PatchTST/iTransformer/TimeMixer/ModernTCN/TimeXer/N/S
bash scripts/ch3/start_model.sh --model "$CH3_MODEL" dry-run
bash scripts/ch3/start_model.sh --model "$CH3_MODEL" preflight \
  --approval "$CH3_EVIDENCE/formal-approval.json" --probe-report "$CH3_EVIDENCE/probe/complete.json"
bash scripts/ch3/start_model.sh --model "$CH3_MODEL" start \
  --approval "$CH3_EVIDENCE/formal-approval.json" --probe-report "$CH3_EVIDENCE/probe/complete.json"
bash scripts/ch3/start_model.sh --model "$CH3_MODEL" status
bash scripts/ch3/start_model.sh --model "$CH3_MODEL" logs
bash scripts/ch3/start_model.sh --model "$CH3_MODEL" complete
bash scripts/ch3/start_model.sh --model "$CH3_MODEL" safe-stop
```

默认正式输出为本证据根`formal-<模型>/`，controller日志在根目录`model-<模型>-controller-<时间标识>.log`；组内任务、history、manifest及checkpoint独立。`complete.json`要求该组全部唯一run与合法summary，不能用历史artifact补缺。失败先审计：恢复许可需逐run的`resume_audits`，区分complete/resume/not_started，并绑定manifest/result或last/best/history的SHA；批准后在同一start命令追加`--resume`。没有这种审计记录便拒绝，不覆盖旧日志/产物、不自动fresh。不同模型用户逐次独立启动，永不自动进入下一个模型。

**收尾状态**：已完成本包接入及限定工程证据，仍有真实数据阻塞、NVML进程归属限制和完整probe/正式训练未执行项；不写“六域正式准备全部Passed”。正式结构/配置审阅、M5冻结决定、M6授权及未来实际训练各保持独立。原M4总gate Not passed、H192失败、1%线、单seed/development-test及来源历史不改。代码/文档/证据待整体审核，本轮不stage/commit/push/closure、不关闭M5。全部前后SHA、精确diff及最终Git检查保存在本包`change-inventory.json`、`review.diff`、`static-verification.json`；未声称ChatGPT已读取本轮新字节。

## 12. UrbanEV消融范围收敛、input_variant及最小复验（2026-09-18）

### 12.1 授权、起点与精确增量

用户本次明确取消AMD额外输入消融及PJM TargetOnly；模块消融只做UrbanEV F4 A/N/S/J，输入消融只做J/F1–F4，两表的A/J主表run直接引用。J/F0非法，不关闭S2保留J身份，不造零辅助/复制目标；不新增N/F0或其他替代实验。447/4860是含非法J/F0的错误建议，**不是用户曾批准的合同**。TiDE Deferred、按模型分组、统一训练表与首次正式test前锁定全部配置的规则不变。

起点HEAD/local/tracking/live remote均为`99be14ce840e67ecd72720860f5d4ab0247418a1`，ahead/behind 0/0，index空，6 modified＋9 untracked路径项，**不是clean**。canonical修前`2d7a22956d020caff48d6f685b8107712be687b4220b7fb17d91ff377cc8a387`；本文件修前`ea094ad188c8ae695b4dc9e7311c67024e55a3569cb5d9e7514018132edd000a`。其余13文件与旧`m5-formal-admission-bizsxh0v/change-inventory.json`一致，该清单SHA=`dedc22ca6046d6e3e9026a993ba7890895d361911276d79399649fd79c919ac8`。AGENTS与Closed M4的固定SHA及baseline解引用commit=`fa9665627e6fcfb1d0c2bc22d943ca9666304fd6`核对一致。

写入前以`scope.json`登记本轮精确9文件：canonical、本文件、`configs/ch3_formal_profiles.json`、`utils/ch3_contract.py`、`utils/ch3_data.py`、`ch3_runner.py`、`tests/test_ch3_formal.py`、`tools/restricted_regression/m5_formal_entry.py`、`tools/restricted_regression/bundle.sha256`。增量复用原有管线，不新建runner/adapter/入口或平台。两启动脚本、adapter、tsAMD_enhanced、guard/bootstrap保持§11字节；作者仓库、环境、Closed文档、baseline及旧证据不改。

本包永久证据唯一根：`/public/home/yueweiting/大论文/amd-execution-evidence/m5/m5-input-scope-8ah0wzk_`（下称E）；合成fixture唯一根：`/tmp/amd-m5-input-37zlykjd`。旧200份证据未覆盖、未重新校验整包checkpoint。当前15文件累计dirty修改继续保留；本轮未stage/commit/push、未closure。

### 12.2 由清单生成的任务、输入方案与probe核账

`plan-difference.json`逐项保存97个撤下ID及423个保留ID；逐项静态比较旧/新profile，除contract版本、`input_variant`字段名外，保留任务ID、模型、目标、有序输入、fold/H、结构与训练设置完全一致。撤下AMD-UrbanEV-F0/F1/F2/F3共96和AMD-PJM-TargetOnly 1，全部为未执行计划；不删历史产物或失败日志。独立算术见`independent-accounting.json`，不以手填计数替代生成验证。

| 当前组成 | 唯一runs | 最多run-epochs |
|---|---:|---:|
| AMD/A，仅六域主表 | 41 | 460 |
| J，六域主表及UrbanEV F1–F3 | 113 | 1180 |
| DLinear/PatchTST/iTransformer/ModernTCN/TimeXer，各组 | 各41 | 各460 |
| TimeMixer，四标准域 | 16 | 200 |
| N/S，各自UrbanEV F4 | 各24 | 各240 |
| 当前合计 | **423** | **4620** |
| TiDE Deferred | 25 | 260 |
| 含暂缓的完整规划 | **448** | **4880** |

另一核算为六域主表303＋N/S额外48＋J/F1–F3额外72；UrbanEV288×10＋PJM7×最多20＋ETTh1/Weather/Exchange各32×10＋ECL32×20＝最多4620。§11的520/5600和更早545/5860仅为历史快照，不再作当前合计。

F1=volume＋calendar（C6/aux5）；F2=F1＋e_price＋s_price（C8/aux7）；F3=F1＋weather（C9/aux8）；F4=volume＋calendar＋price＋weather（C11/aux10）。F2/F3并列。F0字段及历史能力保留，但正式清单、代表worker和正式表行中的F0、J/F0、PJM TargetOnly、AMD额外输入消融数量均为0。当前schema `ch3-target-ms-formal-v2` 的生产者、profile/声明、metadata、manifest/恢复身份、result/summary和入口读到同一`input_variant`；历史日志/配置/checkpoint不追改。`profile`和AMD声明均前置拒绝J/S空aux。旧520协议指纹保留在`supersedes`作拒绝依据，不是兼容许可。

| 实际覆盖 | 组 | 代表worker |
|---|---:|---:|
| 四标准域，每域8模型×四H | 32 | 128 |
| UrbanEV F4九模型 | 9 | 36 |
| UrbanEV J/F1–F3 | 3 | 12 |
| PJM七个主表MS单fit | 7 | 7 |
| 合计 | **51** | **183** |

44组四任务、7组单任务；每组固定原有H顺序与fold波次，代表fold仍1，不跨模型/数据集/输入方案补位。共享GPU形状不声称六fold数据、标签或host内存相同；沿用§11六fold前缀及标签证据，本轮不重读30个UrbanEV前缀。每worker仍6 Adam/8 forward/6 backward/2 validation batch；串行1098 Adam，候选四路至多1056，必要两路回退至多1056，固定流程合计至多3210（保守授权18Q=3294）。相应至多4280 forward/3210 backward/1070 validation batch。额外机械/诊断512 Adam只为上限、未使用，不叠加旧余额；单fit不重复并发。完整probe本轮0次。

### 12.3 工程验收、修复与证据复用

执行前固定`exact-cpu-ids.json`完整25 ID（8个受影响ContractTests、12个IncrementalTests、5个DataTests），在业务import/fixture前校验精确ID、bundle、源码和新配置指纹，正常unittest/LedgerResult生命周期，无discover扩大范围。首次CPU25 passed，failure/error/skipped/unexecuted均0，0 Adam/模型/GPU。覆盖423/4620、51/183、固定波次/完整覆盖、消融复用、空aux/J-F0拒绝、schema、旧许可/报告拒绝、身份/汇总/恢复前元数据拒绝，以及Weather诊断与NVML空/部分/未知PID、N/A显存、整卡降级。

首次CPU结束后的只读资源检查发现NVML另列驱动保留显存：total81920 MiB、reserved699、used1、free81222。首次实现遗漏reserved会错误拒绝准入；仅作一次机械修复，纳入`memory.reserved`守恒项，保留原整数查询2 MiB舍入界限及max(8GiB,10%)整卡余量，不改科学参数、删断言或放宽测试。记录`mechanical-repair.json`。修后对同25 ID完整复验：**25 passed、0 failure/error/skipped/unexecuted**。本轮CPU调用50≤32＋32；不拼接修前/修后结果。首次Passed证据仍保留其旧监控实现局限。

因正式声明字段及恢复身份接入变化，只执行`test_ch3_formal.ModelTests.test_j_ecl_new_declaration`一个限定CUDA方法；绑定`J-ECL-MS-f1-h96-s2024`，合成batch2，使用原正式结构，**1 passed，2 Adam/6 forward/2 backward**。检查目标训练、参数更新、完整/尾批validation、best/last自产保存、异purpose拒绝、optimizer/RNG恢复及eval一致性后继续更新；模型复验0次。它不是正式batch128资源probe，也不验证六域正式准入。方法采用本包新formal_identity后改为合成purpose，未读取旧权重。

`evidence-reuse.json`证明：adapter/tsAMD_enhanced、shell入口/guard/bootstrap字节未变；save_state/restore_state、init_training/update/evaluate、probe_worker、固定波次/dispatch/锁、Windows/prefix_csv/batches的AST未变。保留任务科学profile逐项一致；旧TimeMixer输出宽度、TimeXer真实MS、模型数学、六fold标签等沿用已接受的§11限定证据，不重跑八作者smoke、旧13项或旧333/339。§11 CPU75、模型16/48/16、前缀失败及占位检查成本全部保留；§11＋§12累计CPU125方法调用，模型18 Adam/54 forward/18 backward，不是新授权余额。

受影响入口仅做6次无负载CLI检查：probe及AMD的dry-run/preflight/complete；预期退出分别0/2/0，两组均符合，preflight因未审核closure/无许可拒绝，complete为false。另计于`entry-checks.json`，不冒充CPU测试方法；无tmux/互斥锁重复负载，未启动下一个模型。

### 12.4 数据与NVML真实状态、集中待决项

详细来源URL、已证/未证与建议见`metadata-findings.json`；以下建议均Proposed，不改变现行mandatory条款，也不把补证不足解释为模型不适用。

| 项目 | 本次查到的事实 | 剩余阻塞与一个推荐 |
|---|---|---|
| Weather | 仅沿既有prefix权限遍历到19048记录，输出零基19041–19047的原始时间：05:40、05:50、06:00、06:00、06:10、06:20、06:30，日期2020-05-12；数值观测解析0、test读取0。直接字符串已重复，排除pandas转换碰撞。 | 不能证明重复来自原发布版本还是本地转换；不下载档案、不删/去重/排序/重采样。推荐补齐发布版本绑定后，另行明确批准“按原记录顺序预测、保留重复行”：T/H按记录计而不保证每步严格10分钟；批准前保留unique-index断言和具体机器阻塞项。 |
| PJM | F=1、70/10/20取整已确认；仅7个主表MS任务。作者epftoolbox文档给2013-01-01至2018-12-24。 | 公开日期不能认证本地n；旧材料未提供可绑定本地CSV的行数，endpoints仍null。缺本地版本/行数与转换清单、价格/负荷单位、forecast发行时刻与修订/as-of说明。推荐保留任务blocked并补这份版本说明，不扫描test求n，不由历史行推定已发布或反推已泄漏。 |
| ECL | 作者基准README说明321客户、2012–2014、小时消费；UCI原始说明为15分钟kW、除4转kWh。 | 本地CSV如何选列、改名和聚合/换算尚未闭环；不能将原始单位直接移植。推荐把客户映射要求明确为“发布版本与基准列槽位对应”，不要求真实客户身份；处理链/单位仍mandatory。若不可补，应另审仅标准化基准量、不解释原单位的条款修订，不自动optional。 |
| Exchange | 作者README列8国日汇率及1990–2016；项目OT第7通道及输入顺序保持。 | 国名列表不是本地匿名列字典，报价分子/分母未证。推荐若找不到可靠字典，另审“固定版本匿名基准列＋标准化指标、无币种/报价单位解释”的窄修订；现mandatory不解除、不猜币种。 |
| NVML | 实测GPU UUID=`GPU-3d365efd-300b-f527-e8fe-703fb0cfb738`；容器PID/NSpid=49030，而NVML列PID=46399，正常权限无法对应；驱动保留显存已纳入。 | process_attribution=Not verified，进程峰值null，不填0；外部竞争不能可靠归属时不发并发许可。仅整卡可靠、外部占用可判断、余量满足时允许整卡口径；空/部分/未知PID和坏采样已用合成方法覆盖。不提权、不改驱动/环境。 |

Weather名义10分钟聚合及归档组织有[站点说明](https://www.bgc-jena.mpg.de/wetter/Weatherstation.pdf)和[2020归档目录](https://www.bgc-jena.mpg.de/wetter/weather_data.html)支持，但不证明本地版本和重复行的来源。PJM仅采用[作者数据提取说明](https://epftoolbox.readthedocs.io/en/latest/modules/data_extract.html)的时期元数据；其数据库链接本次不可访问，未另下观测。ECL/Exchange依据[作者基准README](https://github.com/laiguokun/multivariate-time-series-data)及[UCI原始单位说明](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)，不从shape或匿名列吻合升级Passed。

本次J/ECL短验收整卡采样峰值1127219200 bytes（约1.050 GiB）、CPU RSS峰值1219395584 bytes；采样间隔实测0.284224812–1.388134556秒，100ms只是请求目标。仅采样峰值，非严格连续峰值，也不是正式batch资源证明。`resource_admission=false`，进程映射缺口继续存在；旧整进程4GiB Not verified不倒改。

### 12.5 启动前停止点与完整操作命令

**新版probe入口/清单已完成增量修正并通过上述限定验收，但不能称51组只剩按键启动即可全部准入。** 审核/统一closure/用户启动仍缺；PJM7组还缺版本端点与实际尾批，入口会保留Blocked。其余44组有合成计算覆盖定义，不保证实测资源Passed；当前NVML进程归属/外部竞争限制可能得到ResourceNotVerified，不能冒发并发许可。正式数据完整准入方面，Weather/ECL/Exchange各8组及PJM7组共31组有mandatory阻塞；UrbanEV12组＋ETTh1 8组没有新增mandatory数据项，但仍须正式资源/冻结/M6批准。无新许可、无猜测未来commit，模板reviewed=false/commit=null；旧520摘要/报告不能通过当前协议指纹与组覆盖校验。

以下仅为将来的可执行操作命令，**本轮未执行start**；许可文件须由后续审核/实际closure形成，模板不改true来绕过审批。当前先用dry-run/preflight查看拒绝原因：

```bash
cd /public/home/yueweiting/大论文/AMD
CH3_EVIDENCE=/public/home/yueweiting/大论文/amd-execution-evidence/m5/m5-input-scope-8ah0wzk_
bash scripts/ch3/start_probe.sh dry-run
bash scripts/ch3/start_probe.sh preflight
# 后续已审实际closure、clean及用户明确启动后：
bash scripts/ch3/start_probe.sh preflight --approval "$CH3_EVIDENCE/probe-approval.json"
bash scripts/ch3/start_probe.sh start --approval "$CH3_EVIDENCE/probe-approval.json"
bash scripts/ch3/start_probe.sh status
bash scripts/ch3/start_probe.sh logs
tail -f "$CH3_EVIDENCE/probe-controller.log"
bash scripts/ch3/start_probe.sh complete
bash scripts/ch3/start_probe.sh safe-stop
```

tmux仍为§11已测守护；会话`ch3-resource-probe`。`probe/progress.json`与各组worker/process日志保留完整终态；`complete`校验当前协议、Q及51组覆盖，不以文件存在或部分Passed冒充全包准入。安全停止只停止本入口子进程/后续波次，不杀无关进程。ETA须将来按首批实测给出。失败保留现场，先审计、不自动fresh或覆盖。

```bash
# 后续另获结构冻结、M6及该模型正式启动许可，解决mandatory并审核probe后：
CH3_MODEL=AMD  # 一次只选AMD/J/DLinear/PatchTST/iTransformer/TimeMixer/ModernTCN/TimeXer/N/S之一
bash scripts/ch3/start_model.sh --model "$CH3_MODEL" dry-run
bash scripts/ch3/start_model.sh --model "$CH3_MODEL" preflight \
  --approval "$CH3_EVIDENCE/formal-approval.json" --probe-report "$CH3_EVIDENCE/probe/complete.json"
bash scripts/ch3/start_model.sh --model "$CH3_MODEL" start \
  --approval "$CH3_EVIDENCE/formal-approval.json" --probe-report "$CH3_EVIDENCE/probe/complete.json"
bash scripts/ch3/start_model.sh --model "$CH3_MODEL" status
bash scripts/ch3/start_model.sh --model "$CH3_MODEL" logs
tail -f "$CH3_EVIDENCE"/model-"$CH3_MODEL"-controller-*.log
bash scripts/ch3/start_model.sh --model "$CH3_MODEL" complete
bash scripts/ch3/start_model.sh --model "$CH3_MODEL" safe-stop
```

输出仍为E下`formal-<model>`，逐run隔离。恢复须审核本次产物与`resume_audits`，再在同一start命令追加`--resume`；新协议/input_variant不符在读取checkpoint前拒绝，合法恢复沿用原CPU反序列化与RNG流程。不用M4或§11旧purpose/配置跨协议恢复。模型组互斥，用户决定下一个模型，入口不自动衔接。

本轮停在修后代码/文档/证据待整体审核；完整probe、正式训练、J冻结与M6均未启动/批准。AGENTS、Closed M0–M4、八作者仓库、环境、baseline及空间方案保持。`change-inventory.json`给9文件本轮前后SHA；`review.diff`是相对§11已审dirty字节的精确增量，`cumulative-review.diff`为相对HEAD累计15文件；`static-verification.json`和`evidence.sha256`登记终态，后者不包含旧checkpoint。本文件本轮为内容修改且Git tracked modified，不能沿用早期M5 untracked状态。canonical/current M5无需上传（ChatGPT通过服务器直读核验）；该渠道说明不代表ChatGPT已读取本次修后字节。

## 13. 恢复UrbanEV A/J同输入消融的限定增量（2026-09-18）

### 13.1 用户决定、起点及文件范围

用户本次重新决定：UrbanEV输入消融由§12的仅J改为A/J双方F1/F2/F3/F4，两者F4引用各自主表同run；只恢复AMD/F1–F3的72个未执行计划。§12当时取消AMD输入消融的授权与事实完整保留，不倒写为本次授权。不恢复任何F0/PJM TargetOnly，不新增N/F0或其他模型/数据集消融；N/S仍仅UrbanEV F4。447/4860从未获批；当前含TiDE的520规划也不是§11旧520集合。

起点分支`AMD-paper-repro-custom-modules-v1`；HEAD/local/tracking/live remote=`99be14ce840e67ecd72720860f5d4ab0247418a1`，ahead/behind 0/0，index空、6 modified＋9 untracked路径项，非clean。canonical修前SHA=`8a15448e5ab9a1ae7d2ee100a32fcfad8a4d85f2f43f34ccff0e012018303663`；M5修前SHA=`705dcf1a83e68f58246973750740833a91faec54b941e0de6e882ca4a15cb199`。§12的`change-inventory.json` SHA=`9c7040049f6df16446105cc0ba6473b371bcacc1cef539f4b195630fffda0cac`及所列字节匹配；AGENTS、Closed M4及baseline解引用commit保持原值。

写入前`scope.json`登记精确六文件：canonical、本文件、`configs/ch3_formal_profiles.json`、`utils/ch3_contract.py`、`tests/test_ch3_formal.py`、`ch3_runner.py`。其中runner只改报告Q从当前组清单派生，并在probe preflight报告当前机器配置的mandatory资源阻塞；不改训练/恢复/调度实现。`m5_formal_entry.py`、bundle、两启动脚本、guard/bootstrap、loader、adapter、模型本体均无需修改。无重建runner/平台、clone/pull或环境安装。

本包永久证据唯一目录：`/public/home/yueweiting/大论文/amd-execution-evidence/m5/m5-aj-input-7rzf5m90`（E）；唯一合成fixture根：`/tmp/amd-m5-aj-5zperqe6`。§11/§12证据不覆盖；只复制必要旧计划JSON作版本对照，不读取旧checkpoint或运行曲线。本轮无真实前缀、网站补证或原始数据修改。

### 13.2 495任务及同输入身份核账

现有generate_tasks/generate_groups按UrbanEV AMD/J双方F1–F4生成，其他主表模型和N/S在UrbanEV只生成F4，其他域只生成MS。`plan-difference.json`保存423保留ID＋72新增ID，删除0；新增集合严格为AMD×UrbanEV×F1/F2/F3×六fold×h3/6/9/12。全部423旧任务字典及profile逐项完全相等。72新增ID与§11计划对应，科学profile在字段名/历史contract表示归一后匹配；当前实际身份仍用schema v2及新完整配置指纹，不借旧计划许可恢复。

| 模型组 | 唯一runs | 最多run-epochs |
|---|---:|---:|
| AMD/A，主表＋UrbanEV F1–F3 | 113 | 1180 |
| J，主表＋UrbanEV F1–F3 | 113 | 1180 |
| DLinear/PatchTST/iTransformer/ModernTCN/TimeXer，各组 | 各41 | 各460 |
| TimeMixer | 16 | 200 |
| N/S，各组 | 各24 | 各240 |
| 当前合计 | **495** | **5340** |
| TiDE Deferred | 25 | 260 |
| 含Deferred完整规划 | **520** | **5600** |

独立算术：303主表＋48 N/S模块消融＋144 A/J输入消融＝495；UrbanEV360×10＋PJM7×最多20＋ETTh1/Weather/Exchange各32×10＋ECL32×20＝最多5340。含TiDE的520包含暂缓TiDE且没有F0/TargetOnly；§11旧520包含AMD/F0及PJM TargetOnly而没有TiDE，两集合不同。按完整指纹与任务/组内容验证，不以某旧数字自动判真伪。F0定义/旧能力及J/S非空aux前置拒绝不变，正式清单F0/J-F0/PJM TargetOnly全部0。

每个F1–F4内A/J字段和顺序、目标、scaler、fold、标签、训练/评价完全一致，profile除model身份外相等；A关闭S2/THLS，J同时启用。F2价格和F3天气并列。共有AMD初始化复用原`build`路径；THLS与S2分别在`torch.random.fork_rng`隔离子流内构造，退出恢复公共RNG，关闭模块不构造它；训练数据使用`init_training`独立返回的generator并传给`batches`。这比“seed相同”更强，且相关源码未变，本轮未新增数值证明。F4每个A/J run仅出现一次，主表/模块表/输入表引用同ID；原summarize可同时输出同输入方案A/J组，不重新训练或择优挑选统计。

schema保持`ch3-target-ms-formal-v2`，本轮无字段语义改名。旧423及旧520完整配置指纹作为负例保留；许可/报告用当前digest、代码及完整组覆盖核对，恢复元数据使用当前digest/profile/input_variant，旧版本在checkpoint读取前拒绝。当前495的协议SHA见`plan-difference.json`，模板不授予实际许可。

### 13.3 54组、195worker及资源停止线

| 覆盖 | 组 | 代表worker |
|---|---:|---:|
| 四标准域 | 32 | 128 |
| UrbanEV F4九模型 | 9 | 36 |
| UrbanEV A/J F1–F3 | 6 | 24 |
| PJM七个单fit | 7 | 7 |
| 合计 | **54** | **195** |

仅新增AMD-UrbanEV-F1/F2/F3三个组；47组四任务＋7组单任务。保留fold1代表、同模型/数据集/input_variant的原固定H与fold波次，不补位或自动启动下一模型。共享GPU形状不等于六fold数据/标签/host内存相等；沿用已有分fold工程证据，不再读取UrbanEV前缀。

每worker仍6 Adam/8 forward/6 backward/2 validation；生成式核账：串行195×6=1170，候选四路188×6=1128，必要两路回退188×6=最多1128，实际固定流程至多3426 Adam、4568 forward、3426 backward、1142 validation batch。保守授权18×195=3510 Adam；额外机械/诊断余量512不变且不叠加旧额度。没有新增恢复矩阵、长测或调度搜索，195以外profile不自动扩额。本轮完整probe为0。

沿用§12：进程归属Not verified；当前`resource_assessment`依赖每个活动worker在正常权限下匹配PID/NSpid和NVML记录，才能排除未知外部竞争。§12实测环境未满足，整卡记录有效也不能据此发许可。本轮仅将`active_worker_pid_attribution_unresolved_reviewed_admission_path_required`登记到配置并在preflight提前拒绝；不改PID匹配、NVML采样或余量规则，不新建监测平台。**完整probe启动前必须先有已审核且实际可通过的资源准入路径，当前不能启动全队去逐组重复发现此阻塞。** 后续解决时须更新真实配置/源码绑定并重新审核，不把字段删掉或模板置true冒充解决。

数据阻塞集中沿用§12.4：Weather重复原始时间与发布版本/按行预测政策；PJM F=1已定但版本n/端点、单位/as-of未闭环；ECL转换/单位与基准列对应；Exchange列与报价对应。Weather按行、匿名标准化基准口径仍Proposed，现mandatory不变。PJM7组资源尾批仍Blocked；Weather/ECL/Exchange各8组正式数据仍Blocked。增加AMD三个UrbanEV组不解除这些31组的数据限制；UrbanEV15组＋ETTh1 8组也仍需资源、冻结/M6等正式许可。旧4GiB Not verified、M4 gate Not passed/H192/1%线及单seed/development-test限制不改。

### 13.4 精确16方法、成本与证据复用

执行前固定`exact-cpu-ids.json`的16个完整ID，使用既有ch3_cpu入口在业务import/fixture前绑定，正常unittest/LedgerResult生命周期、无discover。首次**16 passed，failure/error/skipped/unexecuted均0**；机械修复/复验0，模型/optimizer/GPU/Adam/forward/backward均0，未构造新CUDA模型。首次16≤16，未使用额外额度。

方法内参数化循环显式记入`parameterized-case-plan.json`：10模型计数、148个固定波次case、495训练profile、423保留任务/profile子case、72新增ID集合、96组A/J同输入fold/H子case与192行合成summary、2个旧许可指纹、4个报告负例、3个恢复元数据负例、4个输入宽度、2个空aux臂。均为CPU断言，不将子case隐去当作未耗费，也不伪算成新增模型实验。

覆盖新增/保留集合、495/5340与54/195、A/J F4重复拒绝及四输入方案汇总、禁止F0/TargetOnly、旧423/旧520清单/许可/报告/恢复身份拒绝、相同计数的外来内容拒绝以及全队启动前监测阻塞。6个CLI检查另计：probe及AMD的dry-run/preflight/complete，退出0/2/0符合预期；AMD显示113任务，probe显示54/195和监测阻塞，complete=false，无start/占位worker负载。

`reuse-proof.json`核验训练/结构/来源/数据政策配置区块不变，profile/AMD声明/summary、保存恢复/初始化/目标训练评价/固定波次与锁的AST不变，adapter/模型/loader及受限工具字节不变。模型结构、映射和生产训练路径没有改变；新72配置与已登记旧计划一致，故无需AMD/F1 CUDA方法。这个依赖证明不冒充新正式batch或shape动态Passed。§11/§12旧CPU125及模型18 Adam/54 forward/18 backward与失败成本保持；本轮新增CPU16后累计**141方法调用**，模型累计仍18/54/18。未重跑八作者smoke、旧13项、旧整套回归或真实训练。

### 13.5 操作命令、审核与停止

本轮已完成清单/合同增量及限定验收，停在整体审核；当前资源阻塞没有解决，**不是只剩closure便可启动完整probe**。两启动脚本仍原版本，不自动启动下一模型。当前查看命令：

```bash
cd /public/home/yueweiting/大论文/AMD
CH3_EVIDENCE=/public/home/yueweiting/大论文/amd-execution-evidence/m5/m5-aj-input-7rzf5m90
bash scripts/ch3/start_probe.sh dry-run
bash scripts/ch3/start_probe.sh preflight
bash scripts/ch3/start_probe.sh status
bash scripts/ch3/start_probe.sh logs
bash scripts/ch3/start_probe.sh complete
```

以下命令仅在资源路径真实解决并获审、数据/尾批限制明确、修后整体review/实际closure/clean及用户启动许可完成后使用。本轮没有生成猜测未来commit的许可；`probe-approval.template.json`为reviewed=false/commit=null，不能用于启动。

```bash
bash scripts/ch3/start_probe.sh preflight --approval "$CH3_EVIDENCE/probe-approval.json"
bash scripts/ch3/start_probe.sh start --approval "$CH3_EVIDENCE/probe-approval.json"
bash scripts/ch3/start_probe.sh status
bash scripts/ch3/start_probe.sh logs
tail -f "$CH3_EVIDENCE/probe-controller.log"
bash scripts/ch3/start_probe.sh complete
bash scripts/ch3/start_probe.sh safe-stop
```

沿已测tmux会话`ch3-resource-probe`；进度`probe/progress.json`，完成报告`probe/complete.json`逐组核验当前协议、195worker和54组内容，存在文件不等于全Passed。日志/预算/失败产物保持；safe-stop只停止本入口拥有的子进程/后续波次。未运行不得产生假complete；首批实测前无可靠ETA。

正式入口仍`bash scripts/ch3/start_model.sh --model AMD dry-run`，模型名可逐次独立选AMD/J/DLinear/PatchTST/iTransformer/TimeMixer/ModernTCN/TimeXer/N/S；后续start仍需独立冻结/M6/对应模型授权、无mandatory数据阻塞、审核后的资源报告。操作形式沿§12.5，但证据根用本节E，旧许可不得复用；本轮无正式启动。恢复只允许审核过的本次合法产物，不读旧权重、不自动fresh。

终态：本轮只改上述六文件，保留累计15文件dirty现场，index空；不stage/commit/push或单独closure。AGENTS/Closed M4、baseline、作者仓库/环境及空间路线均未修改。`change-inventory.json`列六文件前后完整SHA，`review.diff`为相对§12增量，`cumulative-review.diff`为相对HEAD累计范围，`static-verification.json`及`evidence.sha256`给终态与新证据指纹。本轮canonical/M5均为内容修改、Git tracked modified；新字节待审核，不声称ChatGPT已读。

## 14. 有限NVML现场修复与四域数据决策收敛

### 14.1 起点、授权与不变项

2026-09-18，用户确认§13已实际直读审核接受；本轮不重做495任务清单或此前验收。起点HEAD/local/tracking/live remote=`99be14ce840e67ecd72720860f5d4ab0247418a1`，0/0、index空、6 modified＋9 untracked文件，非clean。canonical/M5修前SHA分别为`8140b800767ec4e8abf2ebc6a291a1c5ee4f8e8da617cad0b109c7c9ae433f0f`、`6ee9c68621f9bbe9e10bb734fea1146b96a81d9827e247a5b3f896156720a052`；config/runner及§11–13累计inventory绑定均匹配。适用AGENTS、Closed M4及baseline保持登记字节/commit。

写入前登记六文件：canonical、本文件、`tools/restricted_regression/m5_formal_entry.py`、`tests/test_ch3_formal.py`、`configs/ch3_formal_profiles.json`、`tools/restricted_regression/bundle.sha256`。runner、adapter、模型、loader、guard/bootstrap及两启动脚本字节不变；不扩命令白名单、不新建监测平台。仅在现有ch3分支添加`ch3_resource_diagnostic`窄用途，仍于业务/fixture前校验精确ID、当前源码/配置SHA、路径和0 Adam/forward/backward额度；禁止模型、optimizer和autograd，真实数据/checkpoint无权限。完整probe的review/closure/clean门禁未解除。

唯一永久证据E=`/public/home/yueweiting/大论文/amd-execution-evidence/m5/m5-nvml-admission-dl_e40l_`；临时fixture=`/tmp/amd-m5-nvml-2iwrai5e`。旧证据不覆盖。任务、groups、模型/训练/数据/source配置逐字段与§13相同；只更新本包执行位置、CPU IDs、诊断额度和已验证资源状态。495唯一runs/最多5340、54组/195worker、A/J F1–F4、无F0/PJM TargetOnly、TiDE Deferred全部不变。无模型小smoke、完整probe、真实数据读取/哈希、旧checkpoint访问或新来源调查。

### 14.2 根因、修复与现场正例

§12只读取status/NSpid，容器可见PID不能对应NVML的内核PID。本轮正常权限读取**本包自有**进程的status、stat、sched及PID namespace：本机sched首行提供另一PID，实际与NVML表逐项相符。修后只使用该直接内核元信息映射，不以“新出现的PID”或显存增量猜归属；采样前后stat启动时刻必须相同，缺失、权限不足、PID生命周期变化或重复host映射均不能变成有效归属。未读取其他用户/proc文件，不提权或调整容器、驱动/环境。

旧`resource_assessment`要求活动worker必须已有CUDA表记录，CPU初始化时因此误报外部占用未知。修后可靠自有PID元信息足以识别归属边界；尚无CUDA记录标`alive_not_registered`，进程显存保持缺失/Not verified，不能断言GPU显存为0。CPU-ready消息在本次诊断中额外证明尚未开始CUDA；生产采样仅报告可观察的未登记状态。NVML中未归属且不在基线的进程仍拒绝；UUID变化、坏采样、非有限/负显存、driver_reserved不自洽和余量不足仍拒绝。采样间退出的自有进程单列生命周期事件并重新取活动集合；不把退出/采样失败伪造成0显存。

实际诊断使用共享GPU锁，先确认无外部compute进程及安全余量，再逐步持有1→2→4个独立进程。每实例通过CPU-ready、GO、CUDA-ready、RELEASE消息，只创建一个float32 CUDA张量（4,194,304元素=16MiB）、填充并同步，最后正常退出。此16MiB不限制整个CUDA进程上下文显存。

| 容器PID | 自有sched中的内核PID＝NVML PID | CUDA-ready后NVML进程采样值 |
|---:|---:|---:|
| 62415 | 5409 | 506MiB |
| 62491 | 5618 | 506MiB |
| 62567 | 5804 | 506MiB |
| 62653 | 6019 | 506MiB |

GPU UUID=`GPU-3d365efd-300b-f527-e8fe-703fb0cfb738`；命名空间`pid:[4026534934]`；本次NSpid字段未暴露外层PID，原样保存为空。4次CPU-ready观测、1/2/4存活规模各3次CUDA观测全部`admission=true`，CUDA稳态`process_attribution=Measured`；全部4个returncode=0。首验27.869539秒、4实例，诊断复验0，未用其余4实例额度。整卡采样峰值2041MiB，驱动保留699MiB，预留8192MiB；单进程506MiB包含上下文等开销，不写成16MiB整进程。全事件采样间隔0.664153–4.584536秒（包含子进程初始化等待），相同CUDA稳态相邻采样0.664153–0.774722秒；100ms仅请求目标，非实测达成、非连续峰值。就绪/退出时间、原始NVML表与映射元信息均保存在`resource-diagnostic-attempt1/`。

因此可以将原`active_worker_pid_attribution_unresolved_reviewed_admission_path_required`标记为本机监测链已解决；配置保留结果路径/SHA及“仅16MiB自有张量进程通过，正式组未测”的限定状态，未设置正式审批为true。该结论不证明正式batch/模型构造/Adam/validation峰值、四路数值一致或加速，不授予任何54组并发许可。旧smoke“整进程4GiB Not verified”与§12当时Not verified不倒改。不需要采用削弱归属要求的整卡保守替代提案。

### 14.3 精确验收、修复复验和成本

`exact-cpu-ids.json`在fixture前固定`test_ch3_formal.ResourceAdmissionTests`下16个完整方法，正常unittest/LedgerResult生命周期，无discover。覆盖known/unknown PID、CPU初始化、未映射、退出、采样失败、驱动保留、8GiB/10%余量、UUID、N/A显存、sched、PID复用、解析、非有限/负值、重复映射及用途/完整probe门禁。两种非法显存、两种设备总量余量为显式子case；首验parser case复用一次既有无fixture的断言方法，不计作另一个独立旧验收通过。随后将其内联，避免跨test方法调用；同步经实际诊断确认的资源状态并修正永久测试中旧“缺口未解决”断言，记录于`cpu-repair-reason.json`。

首验16 passed；上述机械测试/状态修正后的唯一CPU复验16 passed，最终同一版本全部16通过；两次failure/error/skipped/blocked/unexecuted均0，不拼接版本。新增正常unittest生命周期32方法；§11–13累计141保留，至本轮累计173。**额外调用偏差必须单列：首验另有1次旧`test_nvml_parser_namespace_and_na`作为断言helper的直接调用，没有独立生命周期。若按所有test方法实际函数调用计，首验为17而非16，超出本轮首验上限1次；本轮合计33。不能仅凭ledger的16/16宣称调用额度完全合规。该调用已移除，唯一修后复验16方法无此额外调用；偏差及原证据保留，待整体审核，不自动追认授权。** GPU诊断4实例/27.87秒；0模型、0optimizer、0Adam、0forward/backward、0checkpoint，未重跑模型来源链。历史模型18 Adam/54 forward/18 backward不变。诊断入口、映射和准入源码在实测后没有变化；末次CPU复验仅涉及测试断言和配置资源状态，科学配置不变，故无需消耗CUDA复验。诊断config保存实际执行时完整SHA绑定，不冒充执行时已经使用修后状态字段。

### 14.4 四域合并待决稿——全部Proposed，批准前不生效

沿§12.4证据，不重复前缀、网页概述或全文件观测读取。本轮没有新定位而尚未读过的生成脚本/版本说明，因而没有新来源事实可登记Passed。共同拟保留的硬要求：目标/有序字段、原split与train-only scaler、完整H、train-standardized MSE/MAE、seed及任务数量不变；**可复现本地版本绑定仍为mandatory**。完整字节SHA只能绑定本地artifact，不单独证明其等同作者发布版本或证明转换链；未证的作者版本等价必须披露，不能隐去。

| 域／现行阻塞条款 | 已有事实 | 单一推荐、拟改的准确语义与验收条件 | 任务/论文影响及额外访问 |
|---|---|---|---|
| Weather；canonical §5.1末段、§9.1 Weather行；机器`version_frequency`与`duplicate_timestamp_19043_19044_requires_explicit_row_index_policy`；`utils/ch3_data.py::load`唯一时间断言 | 原始字符串在零基19043/19044均为2020-05-12 06:00:00，不是pandas解析碰撞。名义10分钟资料不证明本地发布版本相同。固定端点36887/42157/52696。 | **保留当前文件全部行和原顺序，按记录建窗。** 将“唯一时间索引及时间粒度核验门槛”窄改为“Weather允许非唯一、非递减时间；T/H按原记录数，时间只作索引，不删行/排序/重采样”。拟将`version_frequency`拆为“冻结本地字节版本＋名义频率/异常披露”；不要求补造官方等价结论。验收：字节版本与已登记本地版本核对、既有header/目标/端点不变；后续获准前缀验收报告重复/不等间隔，逆序仍拒绝；synthetic重复时间覆盖。批准前本轮不删unique断言。 | 32个主表run不变；H不再被表述为严格H×10分钟，指标仍按记录目标标准化。推荐补一次只读全文字节SHA与历史已登记版本比对；不解析test，不据SHA宣称官方发布一致。 |
| PJM；canonical §5.1/§5.5/§9.1 PJM行；`version_row_count`、`market_units`、`forecast_as_of` | F=1与70/10/20取整已定；只有7个MS主表任务。现材料不能确定绑定本地CSV的n；公开2013–2018日期不足。本地forecast单位及发行/vintage/as-of未证。 | **先做最小字节级版本/行界核验，但保持单位/as-of独立硬门槛。** 不修改F=1政策。建议只流式计SHA、物理行界及CSV引号/行界结构，不解码数值/时间/列；确认无引号内换行且末行处理明确后，n=非空数据记录行数（扣一个header），若格式不符合一记录一行即停止，不猜n。按n算train=floor(.7n)、test=floor(.2n)、val=n−train−test及累计端点，登记metadata；不能借此解除`market_units`/`forecast_as_of`。二者验收仍需绑定此数据版本的列单位/转换说明和forecast发行时刻、修订/vintage相对预测起点的可得性证据；资料不足维持Blocked，不换特征。 | 7 runs及指标不变，版本端点只解决窗口/尾批，不证明forecast当时已发布，也不证明泄漏。**全文字节计数/哈希会遍历test区域字节，超出此前前缀权限，必须另获明确批准。** 即使计数通过，正式PJM仍不能开始。 |
| ECL；canonical §5.1末段、§9.1 ECL行与表后“匿名字段不能替代业务身份”；`processing_chain_customer_map_units` | 已确认C321、OT槽位320及原字段顺序，端点18412/21044/26304。公开原始UCI为15分钟kW；基准说明为小时消费，但本地选择/聚合/单位换算链未证。真实客户身份不是评价必要条件。 | **冻结本地匿名基准版本，限定报告标准化指标。** 拟将客户实名及原单位转换闭环从运行mandatory改为披露限制；替换为“必须绑定不可变本地版本、基准列槽位及有序字段/split/scaler；不主张原kW/kWh、客户业务身份或与官方发布逐字节等价。无法证明的转换链明确未知”。验收：当前完整字节SHA与已登记版本核对、已有header/OT320/端点证据一致；仍不能绑定本地版本则继续Blocked，不用shape过关。准确替换上述两处canonical及机器复合blocker，需用户批准后另行实现，不在本轮optional化。 | 32 runs不变；仅train-standardized目标MSE/MAE，不反变换报告物理用电量或解释具体客户。需本地CSV完整字节SHA的额外只读授权；不读test观测、无需识别真实客户。 |
| Exchange；canonical §5.1末段、§9.1 Exchange行与表后业务身份约束；`currency_column_quote_map` | C8，目标OT槽位7，端点5311/6071/7588。公开8国列表没有证明本地列的国家顺序、基准币或报价方向。 | **冻结本地匿名汇率基准槽位，仅报告标准化预测误差。** 拟将币种/基准币/报价方向的解释闭环改为强制披露限制；新mandatory为“本地版本可复现、原8列顺序及OT7、既定split/scaler可核验”；不得猜币种，不主张官方等价或经济收益。验收同样要求当前完整字节SHA与已登记版本核对及已有字段/端点一致；版本自身无法绑定仍Blocked。上述canonical和机器条款批准后才能替换。 | 32 runs、目标与输入不变；不报告币值误差、汇率方向结论或交易收益。需当前CSV完整字节SHA授权；无需识别真实币种，test解析仍禁止。 |

以上是一次可确认的**数据政策修订＋窄只读核验**组合，尚未批准/执行。本地固定基准方案的研究主张为“明确版本和列槽位上的同协议标准化预测比较”；它不声称恢复了无法证明的原单位/作者处理链。不能以当前SHA和历史SHA一致推导官方等价；若不同必须停下报告版本差异，不覆盖数据或自动接受新版本。

最小额外访问请求，路径仅四个：`AMD/data/weather.csv`、`AMD/data/electricity.csv`、`AMD/data/exchange_rate.csv`各一次完整字节SHA；`TimeXer/dataset/EPF/PJM.csv`一次流式SHA＋物理行界/引号结构计数。均相对`/public/home/yueweiting/大论文/`，只输出摘要/计数/格式状态，不解码或统计test观测，不输出test内容。承认会遍历整个文件（含test）的原始字节；这是需要用户新增授权的范围，本轮访问量0。其余既有header/前缀证据复用，不再请用户自行查资料。PJM单位/as-of没有已定位未读的新材料，仍为独立待补证事实，不能通过政策批准或行数自动核销。

### 14.5 审核、closure与启动命令

工程状态：正常权限监测链已实际通过，**可进入本轮整体审核，获准后统一closure**；本轮没有stage/commit/push。不是现在可以启动完整probe或训练：probe仍要求修后源码/config审批、实际closure commit及clean、无产物冲突和用户启动；正式组并发必须由完整轻量probe实测产生。PJM7组尚缺n/真实validation尾批，仍不能完整资源准入；Weather/ECL/Exchange各32正式run共96，以及PJM7正式run共103均继续数据mandatory Blocked。其余正式run也仍需要冻结/M6/资源审批。数据决策与许可必须在正式test出现前锁定。M4 gate Not passed、H192、1%线及单seed限制保持。

当前安全查看（不会启动worker）：

```bash
cd /public/home/yueweiting/大论文/AMD
CH3_EVIDENCE=/public/home/yueweiting/大论文/amd-execution-evidence/m5/m5-nvml-admission-dl_e40l_
bash scripts/ch3/start_probe.sh dry-run
bash scripts/ch3/start_probe.sh preflight
bash scripts/ch3/start_probe.sh status
bash scripts/ch3/start_probe.sh logs
bash scripts/ch3/start_probe.sh complete
```

本轮实际诊断命令为下列命令（首次目录已存在，**不得作为下一步重复执行**）；日志在`$CH3_EVIDENCE/diagnostic-command.log`，完整逐进程数据在`resource-diagnostic-attempt1/`：

```bash
PYTHONDONTWRITEBYTECODE=1 /public/home/yueweiting/大论文/amd-execution-envs/m5-source-smoke-8sr2d3d_/bin/python tools/restricted_regression/m5_formal_entry.py resource-diagnostic
```

以下仅在审核/closure/clean、PJM等范围阻塞有明确处置及用户启动许可均完成后使用；本轮不生成未来commit审批，也不将模板设true：

```bash
bash scripts/ch3/start_probe.sh preflight --approval "$CH3_EVIDENCE/probe-approval.json"
bash scripts/ch3/start_probe.sh start --approval "$CH3_EVIDENCE/probe-approval.json"
bash scripts/ch3/start_probe.sh status
bash scripts/ch3/start_probe.sh logs
tail -f "$CH3_EVIDENCE/probe-controller.log"
bash scripts/ch3/start_probe.sh complete
bash scripts/ch3/start_probe.sh safe-stop
```

复用已测tmux方式、共享GPU锁和固定波次。状态文件存在不等于Passed；采样失效、UUID/源码/数据边界错误、未知外部竞争或余量不足必须拒绝/安全停本包进程。仍只有54组/195worker原额度，无新shape/并发搜索，不自动启动正式模型组或下一模型。首批probe未执行，无可靠ETA。新版待ChatGPT审核，不声称已被读取；精确增量/累计diff、六文件前后SHA、绑定复用说明与最终Git核验见本包`change-inventory.json`、`review.diff`、`static-verification.json`、`evidence.sha256`。

## 15. 论文—作者发布物—本地文件的只读来源核验（2026-09-18）

### 15.1 起点、权限和实际取得的对象

起点HEAD/local/tracking/live remote均为`99be14ce840e67ecd72720860f5d4ab0247418a1`，0/0、index空、6 modified＋9 untracked文件，非clean。canonical修前SHA=`1a54e48753003b512f4f2b66376aabbccee27c3c1a05f6027535d9affe393cc5`；本文件修前SHA=`eb8cccaba0b77f96ff886e7ccc5056bd53a8a6ee016bcb5ce0ee65bca21e2336`；适用AGENTS、Closed M4、baseline及§14累计版本匹配。本轮只改两文档并新增`tools/m5_source_version_audit.py`，不修改训练/资源工具或生产数据。

唯一证据E=`/public/home/yueweiting/大论文/amd-execution-evidence/m5/m5-source-version-_rkt_o7f`，发布副本只在`E/reference/`。用户本轮明确允许五个本地文件完整字节比较及对应作者发布物下载；这包括test区域原始字节，不等于批准test字段数值解析/统计/预测/评价。没有替换AMD/data、TimeXer/dataset或当前训练文件，没有运行下载/预处理脚本，没有新装环境、clone、pull或fetch。

八作者仓库HEAD与§10.2逐一完全相同，tracked状态clean；版本完整记录于各`*-git.json`。实际读到的数据入口为AMD-upstream `000d377a1ed8946aa817ff357cdf1de64b99abb9:readme.md`第24–37行、iTransformer `c2426e68ca13f74aaec08045c5c724d8ad328124:README.md`第65行、PatchTST `204c21efe0b39603ad6e2ca640ef5896646ab1a9:README.md`第58行、TimeXer `76011909357972bd55a27adba2e1be994d81b327:README.md`第21行。其余仓库仅核对版本，不遍历数据。

主参照选用AMD-upstream明确指向的iTransformer数据入口：[Drive文件ID `1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot`](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view)，显示名`iTransformer_datasets.zip`。本次取得171600194 bytes，原包SHA=`01d7016eaa3eb571b77542612219d116dfc6a9f5399586dea1f98261fc4d80c4`；仅检查归档路径并提取electricity、exchange_rate、weather三个CSV，不执行包内代码，不提取其他benchmark。完整原包留存，是因为该入口提供整包；无不可变历史版本号，只认证2026-09-18本次取得的副本，不称论文发表时原始字节。未为匹配本地遍历其他网盘或镜像。

PJM主参照为TimeXer固定提交的`dataset/EPF/PJM.csv`，Git blob=`75be301851b30429ad2629df77114b2a98efacd1`。使用`git cat-file blob <commit>:<path>`导出到`reference/PJM-git.csv`，未checkout。blob大小1960359，解析到CSV行界/header，不是LFS指针；该路径filter/text/eol均unspecified，工作文件与原blob字节完全一致，不能仅以Git clean替代本次结果。

ETTh1没有从当前必要记录中找到“双方固定发布对象实比”的充分证据，故取得ETDataset作者固定对象：[`11ab373cf9c9f5be7698e219a5a170e1b1c8a930:ETT-small/ETTh1.csv`](https://raw.githubusercontent.com/zhouhaoyi/ETDataset/11ab373cf9c9f5be7698e219a5a170e1b1c8a930/ETT-small/ETTh1.csv)。仅取此对象及提交元数据，不clone整个仓库；取得时间/响应和SHA见`reference/ETTh1-source.json`。

### 15.2 论文与补充来源的实际版本

既有PDF均只读。下表页码为PDF物理页（1-based）；定位时保存必要页文本，未重新分析模型数学或论文性能。现有amd环境的pypdf可用，没有安装依赖。

| PDF／版本 | 完整SHA-256 | 实际相关位置 |
|---|---|---|
| `paper/amd-paper.pdf`，服务器既存14页版；metadata创建2025-04-16 | `02aad9b5b876e3b53d6c34114d6e658eb557898d57f20402baf512b3e264b05e` | p10 Detailed Dataset Descriptions；p11表5，字段数/频率/分割样本描述 |
| `paper/ICLR-2024-moderntcn-a-modern-pure-convolution-structure-for-general-time-series-analysis-Paper-Conference.pdf`，ICLR 2024、43页 | `2e1681606501b715185e9ffa78dbda9ba2e829799e1ed4cea29fa5c1f8f13944` | p14表6、数据介绍及脚注1/3/4/6，指向MPI、UCI、LSTNet数据、ETDataset |
| `paper/NeurIPS-2024-timexer-empowering-transformers-for-time-series-forecasting-with-exogenous-variables-Paper-Conference.pdf`，NeurIPS 2024、30页 | `c2ec27241da87e0559c4f797f5a23d20f725c215c1decb8f007afeb5bfd85964` | p14 A.1及表6；p11参考文献[15]；p19 H.1外生预测情景 |
| `paper/Li 等 - 2025 - UrbanEV An Open Benchmark Dataset for Urban Electric Vehicle Charging Demand Prediction.pdf`，Scientific Data 12:523、10页 | `728eefeaa8934dc598799c786ec5ef1ca02cac0e6d6ee257211b262cefbb176b` | p2时期；p4生成/清洗及表2；p5聚合、275区、Data Records和kWh说明 |
| 新取得`E/reference/epf-2008.08004v2.pdf`，[arXiv:2008.08004v2](https://arxiv.org/pdf/2008.08004v2)，26页 | `8532301359bf3b1efe97ed1eab088e507139de2f09b553727ae2202b52ded6b1` | p6–7 §§3.1/3.3；p8图2及§4.1，单位和前一天可得性 |

补充作者说明也固定到具体对象而非只记首页：LSTNet数据README为`7f402f185cc2435b5e66aed13a3b560ed142e023:README.md`（SHA `fa5e75b4638f607e3a0ac1de6c8b3862dd5883ad5eb26441c59e3449ef626253`）；epftoolbox数据读取说明为`6657f207c047ffb9ca3a1a675d8221690b1bdd0a:epftoolbox/data/_datasets.py`（SHA `7473cb96a15237519f91126e6f09dbe94fbb7b37882799805b43e6df7b7b2a6f`），第112行指向Zenodo record 4624805。仅阅读脚本，未执行它的pandas/下载路径。Zenodo API元数据实际取得，明确PJM是COMED zonal price、system/COMED两种day-ahead load forecast、Eastern Time，以及DST春季插值/秋季重复小时均值处理；不是逐记录发行日志。

**文本冲突和图像核验限制：** ModernTCN p14表6文本写Exchange 7207，而本次两份CSV均7588记录；TimeXer p14正文写Weather目标Wet Bulb，表6文本却写CO2-Concentration。两项并列保留，不静默选择其一或改项目T(degC)/OT7。TimeXer页面文本经pypdf及Desktop Commander两次提取一致，但这不替代图像核验。现有工具未返回可检查的对应页面图像（会议PDF网络读取有大小限制，本地无可用页面栅格化器）；**图像复核未完成，疑似表格矛盾不作最终裁定，不作为数据替换或政策放宽依据**。不以断裂表格文本推算本地n；下述n来自实际字节结构检查。TimeXer README第49行的全球3850站NCEI/ERA5大气象集未下载，也未代替本项目MPI Weather；Sonnet的其他同名数据未使用。

### 15.3 六域对应表与直接比对级别

证据标签固定为【论文明确说明】【作者README/脚本说明】【实际发布文件核验】【本地文件核验】【尚未找到】。A=原始字节一致；B=只按明确格式规则文本一致；C=超出这些规则的实际内容不同；D=参照不可取得/来源不明。本轮五个主比对没有C/D；另一个原始EPF发布参照为C，见§15.4。所有本地及主参照文件读前后device/inode/size/mtime/ctime稳定。

下表本地路径相对`/public/home/yueweiting/大论文/`；L/R分别为本地/发布副本完整SHA。iTransformer包内前缀均为`iTransformer_datasets/`。双方完整header/列序包含在比较范围，没有去字段空格、改header、重排列、数值舍入或填补。

| 数据集｜本地文件 | 作者发布对象 | 论文页码/表号｜README/脚本位置 | 双方摘要及级别 | 已证语义｜真正剩余缺口 |
|---|---|---|---|---|
| Weather｜`AMD/data/weather.csv` | 主Drive对象内`weather/weather.csv` | AMD p11表5；ModernTCN p14介绍1/表6；TimeXer p14 A.1；AMD README24–25→iTransformer README65 | **B**；L `9b2b19b13e342d9c9b9c51620d42b4be232df7fba30e7f5c4968b7c79ab138ad`；R `34ee981d07313e51da2a50bb600072c8ae4a69cb4b0651f4cb93a069d7a2ba63`。L7182728/R7235425 bytes；52696记录 | 【发布＋本地核验】仅CRLF/LF不同，21字段及原顺序一致；两副本零基19043/19044均2020-05-12 06:00:00，不是只有本地才发生。【论文】MPI、2020、名义10分钟。【剩余】链接历史不可变版本未证；重复时间与unique断言的政策冲突仍需决定，未证明原始采集时间严格等间隔。 |
| ECL｜`AMD/data/electricity.csv` | 主Drive对象内`electricity/electricity.csv` | AMD p11表5；ModernTCN p14介绍3；TimeXer p14 A.1；iTransformer README65；LSTNet数据README第11行 | **B**；L `10b4ddd095548839274ce4d55d3c7d6842b4bfa69ac2a4321a7608622da7b4eb`；R `7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad`。L95555457/R95581762 bytes；26304记录 | 【发布＋本地核验】321通道和原header/列序（OT320）与该作者分发文件一致。【论文】小时级、321序列；ModernTCN说明2012–2014。【README】LSTNet数据作者明确写kWh及转为小时消费。【剩余】本包benchmark与原UCI逐步转换/列重命名代码未找到；不能把UCI原始15分钟kW直接移植。作者kWh声明与既有UCI原说明的差异须披露，不以客户姓名未知否定已证基准身份。 |
| Exchange｜`AMD/data/exchange_rate.csv` | 主Drive对象内`exchange_rate/exchange_rate.csv` | AMD p11表5；ModernTCN p14介绍4/脚注4；iTransformer README65；LSTNet数据README第21行 | **B**；L `d55e7aa2641009814a18ba3279431b13f6d413b0eab195b9ff21988d8cf94e97`；R `48b4d9d3d508f5104162e85b9a6042e3557fde11aa9f2944eba8c0d0efc89842`。L630212/R637800 bytes；7588记录 | 【发布＋本地核验】8列与OT7确实对应本次作者包同一列槽位和顺序。【论文/README】日频、1990–2016及8国名单。【尚未找到】明确的“本地列槽位→币种/基准币/报价方向”字典；国家列举顺序不当作字典。ModernTCN表格长度冲突保留，不能据此删381行或换数据。 |
| PJM｜`TimeXer/dataset/EPF/PJM.csv` | TimeXer固定commit/blob，见§15.1 | TimeXer p14 A.1/表6、参考[15]；EPF p6–8 §§3.1/3.3/4.1和图2；TimeXer README21 | **A**；L=R=`58cc0ad32e22e61d9e183b0dc8201747f8c2ef822e469832c5beea8d88251cc0`；双方1960359 bytes；52416记录 | 【发布＋本地核验】锁定作者CSV及n闭环。【论文】COMED价格、系统/COMED两forecast角色、2013-01-01至2018-12-24；原EPF图2标USD/MWh，§4.1明确day-d预测在d−1可得。【剩余】TimeXer转换版与原EPF发布文件非字节/格式等价；两forecast明确单位、转换/数值精度链、逐条vintage/issue-time仍未闭环。不能将论文级可得性降成“完全没说明”，也不能升成逐条已审。 |
| ETTh1｜`AMD/data/ETTh1.csv` | ETDataset固定`11ab373cf9c9f5be7698e219a5a170e1b1c8a930:ETT-small/ETTh1.csv` | AMD p11表5；ModernTCN p14介绍6/脚注6；TimeXer p14 A.1 | **A**；L=R=`f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`；双方2589657 bytes；17420记录 | 【发布＋本地核验】对应固定ETDataset发布对象，7通道、OT6；完整文件17420行，不是任务test_end14400。【论文】小时级、油温/6负荷。项目任务端点8640/11520/14400、T512/四H/scaler保持，不将论文样本计数当原始长度。 |
| UrbanEV｜`AMD/data/UrbanEV/data/{volume,e_price,s_price,weather_central}.csv` | `IntelligentSystemsLab/UrbanEV@44f2aa0c8d89f192bce00bafb0def74a21b39c68`，tracked对象与M1所记一致 | UrbanEV p2/p4/p5；Closed M1 §§4–7/11–13，README89/106–122 | **复用M1既有审计，非本轮重比A/B。** M1 volume SHA=`a55a095ce75af33c59aece2643d5d71b5cd5a0dc73bb97bc553f0a48f40ace32`；e_price=`0076d03b8e400c3e911789e2c7ffb7dd0d44a4414247ead676b508def95bcef4`；s_price=`d125783e042024157f38d1749232696ea2aa893c61fc31672a3c54374498d3dc`；weather_central=`da8c16dcc6a25eadc97ca062998b5dbb01efbb4569efdd693ac98fb5bbc6d065` | 【M1实测复用＋本轮Git元信息】官方repo HEAD/index对象仍对应；没有新冲突，未重读/重哈希当前四份数据。【论文】2022-09至2023-02、275区、小时聚合，volume为额定功率估计的kWh，非直接充电表计真值。目标volume、split-local标签/六fold/scaler与官方默认occupancy及其预处理差异继续保留。 |

B类三文件的独立`sed 's/\r$//' 发布副本 | cmp --silent 本地 -`均退出0；本地均LF、参考CRLF，BOM均无，双方末尾换行状态相同（Weather/ECL有，Exchange无）。因此本次实际只需CRLF→LF，不需要末尾换行/BOM规则才匹配。归一化SHA依次为Weather `6f0d06550a3e10e127157ea5b7c396d79795324ee767abcff981661b3132b006`、ECL `33317c3aefc0d68bc9c3a4e0d686923f00a6b21876dd138199c0f1672d008598`、Exchange `d55e7aa2641009814a18ba3279431b13f6d413b0eab195b9ff21988d8cf94e97`。A类两文件另经原始`cmp --silent`退出0。结果见`comparison-results.json`、`independent-cmp.json`，不叫三份B文件“原始字节一致”。

关于Exchange表述的限定推断：7207恰好等于AMD表5所列5120＋665＋1422三个split样本数之和，因此可能涉及窗口样本数与原始记录数的口径区别；数值相等不证明ModernTCN采用该口径，更不能据此认定数据被删改或论文错误。原始7588记录的直接核验不受该解释未决影响。

### 15.4 PJM记录数、算术与原EPF发布链的界限

PJM CSV逐字节检查引号、转义、CRLF/LF及末行，52417记录含1个header，0空记录/引号内换行/字段数量异常，故n=52416，不是用`wc -l`猜值。既定F=1、floor(.7n)/floor(.2n)得train=36691、validation=5242、test=10483；累计端点36691/41933/52416。T168、H24下，仅作整数算术：

| split | 窗口数 | B128批数 | 余数/处理 |
|---|---:|---:|---|
| train | 36500 | 285完整批 | 20，既定drop_last=True |
| validation | 5219 | 41 | 尾批99，保留 |
| test | 10460 | 82 | 尾批92，保留；只是长度算术，未解析test值 |

与TimeXer表6列示的样本数相符，但证据方向是“本地实测n→既定窗口算术”，不是由表格反推n。生产配置的PJM endpoints本轮仍为null；此事实足以供审核后接入端点/尾批，不自动解除其他阻塞或启动负载。

为核清被TimeXer引用的原EPF发布链，沿作者`_datasets.py:112`取得[Zenodo record 4624805](https://zenodo.org/records/4624805)元数据；它报告PJM文件2530311 bytes，与TimeXer版本1960359不同，这是取得**第二个PJM可信参照**的具体理由，而非搜索相同镜像。只下载该record的PJM.csv；MD5与发布值`bf78d77746b0331c46ceecfbd6bc9592`相符，SHA=`8596cff7a87cf47f9ee2215bd2af9fc8d6950d57339de9e06c87bad2efd407c6`。

原发布header为`Date, Zonal COMED price, System load forecast, Zonal COMED load foecast`；TimeXer header为`date, System load forecast, Zonal COMED load foecast,OT`。在byte1/line1已不同，列序/header超出B类许可，故此辅助参照为**C类**。只核对header，不对test重排列、取数值、舍入或统计近似匹配；不声称只是列移位而全部值相同。锁定TimeXer的相关py/README中未找到把原EPF发布物生成当前CSV的专用转换脚本；原库reader的`Price/Exogenous`重命名也不等于TimeXer这个处理过程。

【论文明确说明】EPF p8 §4.1将day-d两个外生预测列为在d−1可取得的输入；p6–7说明本地时区和DST插值/均值处理，Zenodo元数据明确PJM采用Eastern Time。这提供**基准设计/发行可得性的论文级依据**，不认证每条当前CSV的发行时间、修订版本或日内更新。USD/MWh为原EPF价格图示单位；TimeXer到原EPF数值转换未证明时，只按来源声明记录，不能把两个负荷forecast单位凭常识写成MW已测通过。当前历史输入政策不扩为使用未来forecast。

唯一建议：保留已锁定TimeXer版本，先审核接纳其A类身份、n及端点事实；转换链问题在现有已批准train/validation前缀内作下一笔独立、限定字段/精度核对，必要时向作者求转换说明。即使前缀核对一致也不得外推全部test值或逐条vintage。当前不换数据、不以原Zenodo替代TimeXer、不放宽单位/as-of条款。没有证据支持“已经泄漏”的结论。

### 15.5 已消除疑问、剩余停止线与执行账

可提交审核解除的**事实疑问**：Weather是否仅本地出现重复（本次作者副本也有）、三标准域是否对应明确作者分发文件（B类已证）、PJM是否对应锁定作者Git对象及其n/端点、ETTh1是否对应固定作者发布对象及完整行数。ECL的基准列身份不再因为未知真实客户姓名而写成未知；Exchange的OT7基准槽位身份与具体币种解释分开。以上不自动修改现行复合mandatory字段。

真正剩余：Weather保留重复记录与unique断言之间仍须明确数据协议决定；ECL原UCI→processed benchmark精确转换/单位链；Exchange币种/基准币/报价列字典；PJM TimeXer转换链、两forecast单位和逐条vintage审计范围。主Drive链接的历史版本不能从本次内容一致推定。图像复核限制见§15.2。**§14.4全部仍Proposed**，不能先落地匿名标准化/按记录政策再倒找证明；用户可以基于本次明确发布身份重新判断需要修订的最小条款，不再泛化为“所有版本资料均未知”。

仅新增一份固定路径/对象的标准库审计脚本。先执行8个精确unittest方法（ID写入`exact-test-ids.json`，正常生命周期、方法不互调）；首次8 passed，failure/error/skipped=0，未用机械复验。显式子case为4种非法引号/行界、5个合法路径绑定＋1个拒绝；全部合成，不import业务/model或用真实数据fixture。真实比对在合成验收后运行，脚本无读取任意CSV的CLI；固定五对路径及唯一reference根，扫描器仅处理字节/分隔符，不将数值/日期转换成模型数据。

直接主比对：五个本地文件各完整读取2次（SHA/结构扫描一次＋独立cmp一次），共215836826 bytes；五个主参照比较读取216010006 bytes。Weather另各读到19045记录前缀，逻辑读取本地2564392/reference2583438 bytes，只输出4个时间字段。PJM辅助参照只输出header和首个差异位置；cmp内部缓冲的实际读取量未追踪，上界为已授权完整文件，未虚称只读1 byte。下载/提取/hash的reference读取另记，不冒充模型实验。完整原包171600194 bytes，仅三目标CSV提取；另取得ETTh1固定对象、PJM Git副本、PJM Zenodo副本、EPF论文和必要公开元数据；无空间/大规模气象数据或其他benchmark提取。所有比对前后原件状态稳定。

测试与执行证据：`synthetic-tests.json/log`、`comparison-results.json`、`byte-access-ledger.json`、`independent-cmp.json`、`weather-prefix-times.json`、`pjm-original-release-check.json`、`paper-index.json`、`source-documents.json`、`audit-facts.json`。0模型/GPU/Adam/训练/checkpoint读取或哈希，0正式test数值解析/统计/评价；承认完整test字节被比较。未重跑§14诊断或旧loader/model/probe。§14首验多1次test调用偏差保留，不能用本轮8项抵账。

终态只增加上述审计工具并修改两文档，保留此前15文件累计dirty修改，index空；不stage/commit/push、closure、冻结J、关闭M5或进入M6。原495/5340、54组/195worker，生产配置/loader/mandatory、作者仓库/环境、Closed M1/M4及AGENTS、baseline和空间路线均不变。精确diff、前后SHA与最终Git状态见本包inventory；修后版本尚待ChatGPT审核。

## 16. 用户明确确认后的数据准入收口与限定复验

### 16.1 授权、版本继承与精确变更范围

本轮用户以“我作为用户确认授权”明确批准Weather原记录建窗、ECL/Exchange已核验benchmark标准化评价及解释限制、PJM锁定TimeXer与原EPF论文级可得性依据，以及对应CPU和两域限定前缀验收；不是因收到§15回执自动放行。本节是对§14.4 Proposed的本次精确确认，不倒写§14–15当时授权。

起点HEAD/local/tracking/live remote均为`99be14ce840e67ecd72720860f5d4ab0247418a1`，0/0，index空、6 modified＋10 untracked，非clean。canonical修前`974a2326c404a66ad3ee1dda730a0cd87b6b894a282785707f82ff6a5b4a47c7`；本文件修前`0ffe42dafbde2ae13b7c6ebfe6f2cc5d9016ff7d8a83f935456dc3f95425938b`。§11–15累计16文件的修前摘要逐项一致。AGENTS、Closed M4、不可变baseline、八来源/环境及空间路线不变。

新证据唯一目录`/public/home/yueweiting/大论文/amd-execution-evidence/m5/m5-data-admission-ozab6blq/`（下称E）；合成fixture唯一根`/tmp/amd-m5-data-axaq0mkd`。本轮实际修改8个既有工作文件：canonical、本M5、`configs/ch3_formal_profiles.json`、`utils/ch3_data.py`、`tests/test_ch3_formal.py`、`tools/restricted_regression/m5_formal_entry.py`、其`bundle.sha256`、`ch3_runner.py`。未新建平台、任务清单或milestone。M5在当前HEAD已tracked，本轮为modified；配置、runner等原untracked文件是本轮内容修改，不是本轮首次创建。

继承§15证据根`m5-source-version-_rkt_o7f`，本轮仅读取文本证据及五文件stat，不完整扫描/哈希数据、不重新下载/提取。绑定：

| §15文本证据 | SHA-256 |
|---|---|
| comparison-results.json | bfcfc2adb157750bbbab434b4f8cba7f65b0fcb48ed14d2fa9a2f8bf57ab8ae1 |
| independent-cmp.json | 4c1f01ff4ce871f973ccabcac62fd021547af714a0af0ca5a6de117c34dfa88d |
| published-object-manifest.json | ea9fa63671080ae1dc4d45defb57ba8106495bd6f548852819468a99f021fdbc |

五个本地文件device/inode/size/mtime_ns/ctime_ns均与§15读后记录一致。配置登记继承的原始SHA、细分来源状态及文件状态；loader读前/读后核对stat，preflight只作stat复核，变化则拒绝，不自动接受版本。stat继承并非本轮重新证明完整字节相同；发布物及原始SHA仍以§15为证据，不用规范化SHA替换本地文件SHA。

### 16.2 生效数据政策与仍须披露的限制

| 域 | 已证事实及本次接受范围 | 保留的解释限制与硬边界 |
|---|---|---|
| Weather | §15作者分发副本与本地仅CRLF/LF不同；两者均有零基19043/19044重复时间。本次仅该域允许非唯一但非递减时间，按原记录T/H造窗；36887/42157/52696、T512、T (degC)/1不变。 | 不删/去重/排序/插值/补时间/重采样；逆序仍拒绝，其他域唯一时间要求不变。名义10分钟不等于每步严格10分钟；重复不是新增样本扩增。云盘当前副本不能证明所有历史版本。 |
| ECL | 作者benchmark文本一致，321列原顺序、OT320、18412/21044/26304与train-only scaler保持；仅既定train-standardized指标。 | 原UCI至benchmark的完整转换、客户业务身份和物理单位恢复未独立核实，改为必须披露限制，不要求客户实名，不推导物理用电量误差。不能把未知转换写成已验证。 |
| Exchange | 作者benchmark文本一致，8列及OT7槽位已证；5311/6071/7588及scaler不变。 | 币种、基准币、报价方向未知仍披露，不再作本次标准化基准运行前置；不猜币种、不解释交易收益。 |
| PJM | TimeXer固定commit `76011909357972bd55a27adba2e1be994d81b327`与§15 A类CSV；n52416、端点36691/41933/52416。原EPF §4.1 d−1可得性为documented source assumption。 | 非record-level vintage audit Passed；两forecast单位、完整TimeXer→原EPF转换和逐条发行审计仍未独立核验，不再是当前运行前置；潜在源数据风险仍披露。不换Zenodo、不搜索全值转换、不引入未来forecast。 |

PJM价格及系统/COMED负荷forecast角色按TimeXer/原EPF登记；原EPF价格图标USD/MWh是来源声明，不由此猜两个forecast的单位。版本/字段/端点、用户接受的来源依据与任务解释范围、仍未核验的业务解释在配置`source_admission`内分别记录；四个旧复合mandatory以本节用户精确授权替代，未把所有metadata标Passed。ETTh1/UrbanEV及其他训练、test隔离、历史信息集、目标/有序aux、split/scaler政策不变。

### 16.3 同SHA页面图像复核与两种规范化口径

本次按用户转达登记“**ChatGPT同SHA页面图像复核**”：ModernTCN PDF SHA `2e1681606501b715185e9ffa78dbda9ba2e829799e1ed4cea29fa5c1f8f13944`，物理第14页表6 Exchange Dataset Size确为7207；只确认印刷内容，不裁定原因、不改已核验7588记录。TimeXer PDF SHA `c2ec27241da87e0559c4f797f5a23d20f725c215c1decb8f007afeb5bfd85964`，物理第14页正文Wet Bulb与表6 CO2-Concentration确实不同，保留不一致，不改本项目T (degC)/1。Codex本轮未渲染/查看这些页面、未安装环境或改PDF；§15当时未取得页面图像的事实保持。

§15脚本`normalized_sha256`使用其完整声明规则：去开头UTF-8 BOM、CRLF→LF、去一个末尾LF；这不是独立cmp的口径。独立cmp仅CRLF→LF即通过，三文件无BOM且双方末尾换行状态相同。本轮仅澄清口径，不重新扫描计算。

### 16.4 精确CPU验收与失败计账

执行前`E/exact-test-ids.json`固定16个完整ID：10个`test_ch3_formal.DataAdmissionTests`方法，以及6个既有`DataTests`方法（prefix_stops_before_poison、scaler_train_only_and_val_context、tail_and_shuffle_policy、no_implicit_test_capability、ordered_header_rejected、short_prefix_rejected）。受限bootstrap在业务import/fixture前核对ID/源码/协议摘要，正常unittest生命周期；方法不互调，普通fixture helper独立。未discover整套、未调用模型路径。新方法覆盖Weather重复保留/逆序拒绝/其他域严格、state变化拒绝、PJM算术、分层来源状态、旧许可/报告拒绝、窄前缀能力、495任务/54组及全部495个原profile逐项相等。

首验执行8方法：7 passed、1 error、0 failure/skip、8 unexecuted。错误是旧协议许可负例夹具误带真实作者来源，preflight试图读未绑定DLinear源码，被guard拒绝；未读到源码内容、不放宽guard。唯一机械修复把该负例来源集合设为空的合成fixture，保留旧protocol mismatch断言。修后同一源码完整复验16方法：16 passed、0 failure/error/skip/unexecuted。合计24次方法调用，未超16＋16上限；不能把首验7项拼接成终态。完整日志、实际启动/终态ID及失败trace保留于两次`cpu-tests-attempt*/{ledger.jsonl,result.json,audit.jsonl}`。

显式参数化负例/核对：其他域唯一时间4子case、来源状态4域、前缀错误3子case、时间策略拒绝2项、495 profile/54组配置相等检查；这些是已列方法内部的固定检查，不运行模型。`acceptance-ledger.json`单列，不隐藏方法互调。§11–15已有成本及§14额外1次test调用偏差保持，不由本轮通过抵账。

### 16.5 两域各一次真实前缀连通

窄`ch3_prefix`能力只接受`ch3.prefix.Weather.train_validation`或`ch3.prefix.PJM.train_validation`，路径/42157或41933记录上限、用途及当前源码SHA在import/fixture前绑定；禁止model/optimizer/GPU，0 Adam/forward/backward。没有扩大旧M4或全局访问策略。

| 域 | 解析／scaler-fit止点 | 实际CPU首batch及目标 | 窗口／尾批与时间核对 |
|---|---|---|---|
| Weather | 42157／36887；无test对象 | train/validation各[128,512,21]→[128,96,1]，finite；另取实际尾批55 | H96 train36280、validation5175。前缀42156个间隔中42154个10分钟、1个0分钟（后重复索引19044）、1个100分钟；无逆序。原记录/值保留，未修补间隔。其他H192/336/720 validation窗5079/4935/4551、尾87/71/71为同一前缀长度算术。 |
| PJM | 41933／36691；无test对象 | train/validation各[128,168,3]→[128,24,1]，finite；另取实际尾批99 | train36500窗，285完整批、drop20；validation5219窗、40完整批＋99。41932个前缀间隔均1小时，无重复/逆序。test10460窗、尾92仅整数算术，未解析test。 |

各域首次通过，0前缀复验；各只调用一次`load`，完整header/字段原序、前缀finite、指定目标及train-only scaler经同一生产路径核对。实际尾批用该validation Dataset最后余数索引与同一collation取得，不遍历完整validation做评价。日志为`weather-prefix-attempt1/`、`pjm-prefix-attempt1/`下的`prefix-result.json`、config/audit/budget和进程日志。prefix摘要只认证已读记录的规范化文本，不是全文件SHA；没有新完整数据SHA。解析止于批准端点，不向模型送数据、不解析/统计/评价test；不把文本I/O缓冲说成逐字节操作系统隔离。

Weather的100分钟间隔作为实际异常明确披露，不与本次已批准的非递减按记录合同冲突，不自动补时间。ECL/Exchange/ETTh1/UrbanEV复用§11–15有效证据，未重读其前缀或UrbanEV全部文件。本轮0GPU、0模型/optimizer、0Adam、0前向/反向、0checkpoint读取/哈希。

### 16.6 54组技术条件、真正停止线与完整操作命令

`E/probe-tail-readiness.json`从未改变的495任务独立核账：495/5340、54组/195代表worker；所有组端点/validation尾批可计算且窗口为正。47组四任务＋7组PJM单fit保持。Exchange H336/H720的validation只有425/41窗、小于B512，真实validation仅尾批；既定probe的合成完整B512是保守资源形状检查，不冒称实际存在整批。静态算术整理时曾误设n≥eval_batch假设并停止，纠正该报告假设后登记真实完整批数0；不是模型/测试额外调用或科学参数变更。

前置工程/数据阻塞在本次范围内已收口；`start_probe.sh dry-run/preflight`实测54/195，阻塞只为`reviewed clean closure required`及`explicit review/closure authorization missing`（preflight预期exit2）。没有生成许可、猜未来commit或把模板置true。继承§14已测PID映射路径，未知外部PID/坏采样/UUID不符/余量不足仍拒绝；正式模型单路及并发资源尚未测，旧smoke整进程4GiB Not verified保持。因此可交ChatGPT整体审核；统一closure/clean及实际许可后由用户启动完整probe，不是在此认定54组资源Passed。完整probe仍3510保守首验Adam上限、实际流程≤3426及额外机械上限512，不追加预算。

以下是修后审核、统一closure后使用的完整命令；`probe-review.json`目前不存在，须绑定实际closure commit、修后完整配置/代码/源/环境/硬件及reviewed用途，不得用旧配置许可。当前直接preflight仍会拒绝。日志/status/complete/safe-stop仅在用户实际启动后使用：

```bash
cd /public/home/yueweiting/大论文/AMD
CH3_E=/public/home/yueweiting/大论文/amd-execution-evidence/m5/m5-data-admission-ozab6blq
bash scripts/ch3/start_probe.sh dry-run
bash scripts/ch3/start_probe.sh preflight --approval "$CH3_E/probe-review.json"
bash scripts/ch3/start_probe.sh start --approval "$CH3_E/probe-review.json"
bash scripts/ch3/start_probe.sh logs
tail -n 100 "$CH3_E/probe-controller.log"
bash scripts/ch3/start_probe.sh status
bash scripts/ch3/start_probe.sh complete
bash scripts/ch3/start_probe.sh safe-stop
```

`start`沿既有tmux守护，`safe-stop`沿自有STOP标志，不杀无关进程；`complete`读取/核验当前协议覆盖的终态，文件存在不自动等于所有组Passed。失败保留全部现场，不能自动fresh重跑。正式`start_model.sh --model <名称>`另外要求结构冻结、M6授权、数据绑定及已审核资源结果，本轮均不代批。模型按组启动规则、同run跨表引用、F0/PJM TargetOnly=0、TiDE Deferred不变。

本轮不stage/commit/push、不closure、不跑完整probe/正式训练、不冻结J/关闭M5/进入M6。精确本轮diff、修前/修后SHA、最终Git状态与范围见`E/change-inventory.json`、`E/changes.patch`和`E/final-verification.json`；只是可审阅材料，不声称ChatGPT已经读取本轮新字节。


## 17. 外部baseline来源batch/LR迁移、首轮probe复核与有限机械修复

2026-09-19当前限定增量（M5 §17）：用户批准外部六baseline仅按对应论文/明确委托的本模型官方代码迁移batch、eval batch及初始LR；A/J及N/S原训练设置、全模型T/epoch/停止/优化器与信息合同保持。唯一profile覆盖层已接入，未决来源字段保持待审并前置拒绝；这是有限参数迁移，不是完整作者recipe复现、不声称各模型最优或纯结构差异完全隔离。原54组probe已完成：13 Passed、22 ResourceNotVerified、19 Blocked，206worker、1183 Adam调用/1580前向/1184反向，失败不退款。当前CPU24+24方法均通过；CUDA两个代表首次及唯一机械复验累计8 Adam/18前向/8反向，修后iTransformer小形状与TimeMixer ECL B32/T512/H96连通、自然退出监测通过，首次退出监测失败仍保留。九组RSS增长旧阻塞及TimeMixer/Exchange旧数值不一致不倒改。495/5340、54组/195worker不变；10组可提交有条件继承审核，44组170worker补测上限3036 Adam，原首验余额2327，额外709仅Proposed；机械512余额504。来源、内存判据及额度待决，修后review/closure和用户启动仍必需，本轮未启动完整补测/正式训练，J未冻结、M5未关闭、M6未开始。以下§16及更早均为历史时点。

### 17.1 起点、授权及不可变项

起点实际HEAD/local/tracking/live remote均为`9e71fd572f866797376584f93cf4b25bd6840c34`，0/0、clean；本轮仅dirty增量，不stage/commit/push。canonical/M5/config修前SHA分别为`1a2e9f6300c558275d051adfb7a107de5ea0874262155dff91e880a55e1aebdc`、`f30dfaff329056770172af8763eff02bedf776ae97d7ecd52c124d56db0958f2`、`c54392cacdbd18cb1627238d1c4c61cb7c3e321fa3e8eadb6955bb2abaa0a5ba`。AGENTS、Closed M4、baseline与作者repo/env不改。新证据唯一目录：`/public/home/yueweiting/大论文/amd-execution-evidence/m5/m5-baseline-followup-0r6ra554`；fixture=`/tmp/amd-m5-followup-zob_s493`。原§9–16和旧54组文件全部保留。

A/J及N/S全部274个run的完整profile与修前逐项相等；495个逻辑ID、54组/固定波次、模型结构/目标/输入/初始化、dataset T/epoch/patience/优化器/精度及数据政策不变。统一T而外部batch/LR部分不同，是有限参数迁移比较，不是完整作者recipe或纯结构隔离实验。未决值不以旧默认值偷偷启动。source表按H完整展开121条，57个正式run涉及未决来源字段；不是57个新增任务。单seed2024/std=N/A/稳定性Not evaluated、M4总gate Not passed/H192/1%与科学停止保持。

### 17.2 论文、作者代码及逐模型数据集解析

先读论文相关实验/附录，再读各自锁定作者脚本/defaults。未搜索、未择优，未读作者真实数据或旧checkpoint。四份缺失论文从对应arXiv入口取得本包reference副本；ModernTCN/TimeXer用服务器既有PDF。页码均物理PDF页；表7文本行明确，无歧义数字未凭其他模型补造。本轮没有新增PDF页面图像复核。

- P1 Are Transformers Effective for Time Series Forecasting?；版本 `['arXiv:2205.13504v3  [cs.AI]  17 Aug 2022']`；PDF `/public/home/yueweiting/大论文/amd-execution-evidence/m5/m5-baseline-followup-0r6ra554/reference/DLinear-2205.13504.pdf`；SHA-256 `97abddd1821cc72942c8d7ddde7e99466bb91f1bddc37c2b54e0e97be7b5be1b`。
- P2 A Time Series is Worth 64 Words: Long-term Forecasting with Transformers；版本 `['arXiv:2211.14730v2  [cs.LG]  5 Mar 2023']`；PDF `/public/home/yueweiting/大论文/amd-execution-evidence/m5/m5-baseline-followup-0r6ra554/reference/PatchTST-2211.14730.pdf`；SHA-256 `ffd4021d25b4959883242f256b0fe4ec42f477f66db78c61bfacf7baa7848b0e`。
- P3 iTransformer: Inverted Transformers Are Effective for Time Series Forecasting；版本 `['arXiv:2310.06625v4  [cs.LG]  14 Mar 2024']`；PDF `/public/home/yueweiting/大论文/amd-execution-evidence/m5/m5-baseline-followup-0r6ra554/reference/iTransformer-2310.06625.pdf`；SHA-256 `83e889795af52c672090255493266e3ea93e7b2a32c528f58074f2f9f757ffc5`。
- P4 TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting；版本 `['arXiv:2405.14616v1  [cs.LG]  23 May 2024']`；PDF `/public/home/yueweiting/大论文/amd-execution-evidence/m5/m5-baseline-followup-0r6ra554/reference/TimeMixer-2405.14616.pdf`；SHA-256 `a599b338e44af70d8e9c87be3c5417bde7864b2c92074e1346703f3e2b641e3d`。
- P5 ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis；版本 `locked conference PDF, metadata in paper-index-with-versions.json`；PDF `/public/home/yueweiting/大论文/paper/ICLR-2024-moderntcn-a-modern-pure-convolution-structure-for-general-time-series-analysis-Paper-Conference.pdf`；SHA-256 `2e1681606501b715185e9ffa78dbda9ba2e829799e1ed4cea29fa5c1f8f13944`。
- P6 TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables；版本 `locked conference PDF, metadata in paper-index-with-versions.json`；PDF `/public/home/yueweiting/大论文/paper/NeurIPS-2024-timexer-empowering-transformers-for-time-series-forecasting-with-exogenous-variables-Paper-Conference.pdf`；SHA-256 `c2ec27241da87e0559c4f797f5a23d20f725c215c1decb8f007afeb5bfd85964`。

| 模型/数据集 | H | 旧B/eval/LR | 当前生效B/eval/LR | 未决字段的唯一推荐（未实施） | 来源 |
|---|---|---|---|---|---|
| DLinear/UrbanEV | 3,6,9,12 | 128/128 / 3e-05 | 128/128 / 3e-05 | 无新增待决值 | P1: p9 Appendix B.2 (delegates hyperparameters to code)；project-approved transfer（不是该域paper setting） |
| DLinear/PJM | 24 | 128/128 / 5e-05 | 128/128 / 5e-05 | 无新增待决值 | P1: p9 Appendix B.2 (delegates hyperparameters to code)；project-approved transfer（不是该域paper setting） |
| DLinear/ETTh1 | 96,192,336,720 | 128/128 / 5e-05 | 32/32 / 0.005 | 无新增待决值 | P1: p9 Appendix B.2 (delegates hyperparameters to code)；S1 + S2 |
| DLinear/Weather | 96,192,336,720 | 128/128 / 5e-05 | 16/16 / 0.0001 | 无新增待决值 | P1: p9 Appendix B.2 (delegates hyperparameters to code)；S3 + S2 |
| DLinear/ECL | 96,192,336,720 | 128/128 / 0.0003 | 16/16 / 0.001 | 无新增待决值 | P1: p9 Appendix B.2 (delegates hyperparameters to code)；S4 + S2 |
| DLinear/Exchange | 96,192 | 512/512 / 0.0003 | 8/8 / 0.0005 | 无新增待决值 | P1: p9 Appendix B.2 (delegates hyperparameters to code)；S5 + S2 |
| DLinear/Exchange | 336,720 | 512/512 / 0.0003 | 32/32 / 0.0005 | 无新增待决值 | P1: p9 Appendix B.2 (delegates hyperparameters to code)；S5 + S2 |
| PatchTST/UrbanEV | 3,6,9,12 | 128/128 / 3e-05 | 128/128 / 3e-05 | 无新增待决值 | P2: pp13–14 A.1 supervised protocol; no unique batch/LR；project-approved transfer（不是该域paper setting） |
| PatchTST/PJM | 24 | 128/128 / 5e-05 | 128/128 / 5e-05 | 无新增待决值 | P2: pp13–14 A.1 supervised protocol; no unique batch/LR；project-approved transfer（不是该域paper setting） |
| PatchTST/ETTh1 | 96,192,336,720 | 128/128 / 5e-05 | 128/128 / 5e-05 | 128/128 / 0.0001；batch,eval_batch,lr待审 | P2: pp13–14 A.1 supervised protocol; no unique batch/LR；S6 + S7 |
| PatchTST/Weather | 96,192,336,720 | 128/128 / 5e-05 | 128/128 / 5e-05 | 128/128 / 0.0001；batch,eval_batch,lr待审 | P2: pp13–14 A.1 supervised protocol; no unique batch/LR；S8 + S7 |
| PatchTST/ECL | 96,192,336,720 | 128/128 / 0.0003 | 128/128 / 0.0003 | 32/32 / 0.0001；batch,eval_batch,lr待审 | P2: pp13–14 A.1 supervised protocol; no unique batch/LR；S9 + S7 |
| PatchTST/Exchange | 96,192,336,720 | 512/512 / 0.0003 | 512/512 / 0.0003 | 无新增待决值 | P2: pp13–14 A.1 supervised protocol; no unique batch/LR；project-approved transfer（不是该域paper setting） |
| iTransformer/UrbanEV | 3,6,9,12 | 128/128 / 3e-05 | 128/128 / 3e-05 | 无新增待决值 | P3: p14 implementation: B32, LR search set; ECL code conflicts；project-approved transfer（不是该域paper setting） |
| iTransformer/PJM | 24 | 128/128 / 5e-05 | 128/128 / 5e-05 | 无新增待决值 | P3: p14 implementation: B32, LR search set; ECL code conflicts；project-approved transfer（不是该域paper setting） |
| iTransformer/ETTh1 | 96,192,336,720 | 128/128 / 5e-05 | 32/32 / 5e-05 | 32/32 / 0.0001；lr待审 | P3: p14 implementation: B32, LR search set; ECL code conflicts；S10 + S11 |
| iTransformer/Weather | 96,192,336,720 | 128/128 / 5e-05 | 32/32 / 5e-05 | 32/32 / 0.0001；lr待审 | P3: p14 implementation: B32, LR search set; ECL code conflicts；S12 + S11 |
| iTransformer/ECL | 96,192,336,720 | 128/128 / 0.0003 | 128/128 / 0.0003 | 16/16 / 0.0005；batch,eval_batch,lr待审 | P3: p14 implementation: B32, LR search set; ECL code conflicts；S13 + S11 |
| iTransformer/Exchange | 96,192,336,720 | 512/512 / 0.0003 | 32/32 / 0.0003 | 32/32 / 0.0001；lr待审 | P3: p14 implementation: B32, LR search set; ECL code conflicts；S14 + S11 |
| TimeMixer/ETTh1 | 96,192,336,720 | 128/128 / 5e-05 | 128/128 / 0.01 | 无新增待决值 | P4: p14 Table7 long-term forecasting；S15 + S16 |
| TimeMixer/Weather | 96,192,336,720 | 128/128 / 5e-05 | 128/128 / 0.01 | 无新增待决值 | P4: p14 Table7 long-term forecasting；S17 + S16 |
| TimeMixer/ECL | 96,192,336,720 | 128/128 / 0.0003 | 32/32 / 0.01 | 无新增待决值 | P4: p14 Table7 long-term forecasting；S18 + S16 |
| TimeMixer/Exchange | 96,192,336,720 | 512/512 / 0.0003 | 512/512 / 0.0003 | 无新增待决值 | P4: p14 Table7 long-term forecasting；project-approved transfer（不是该域paper setting） |
| ModernTCN/UrbanEV | 3,6,9,12 | 128/128 / 3e-05 | 128/128 / 3e-05 | 无新增待决值 | P5: pp17–18 C.1 long-term: LR1e-4; batch unspecified；project-approved transfer（不是该域paper setting） |
| ModernTCN/PJM | 24 | 128/128 / 5e-05 | 128/128 / 5e-05 | 无新增待决值 | P5: pp17–18 C.1 long-term: LR1e-4; batch unspecified；project-approved transfer（不是该域paper setting） |
| ModernTCN/ETTh1 | 96,192,336,720 | 128/128 / 5e-05 | 128/128 / 0.0001 | 512/512 / 0.0001；batch,eval_batch待审 | P5: pp17–18 C.1 long-term: LR1e-4; batch unspecified；S19 + S20 |
| ModernTCN/Weather | 96,192 | 128/128 / 5e-05 | 128/128 / 0.0001 | 256/256 / 0.0001；batch,eval_batch待审 | P5: pp17–18 C.1 long-term: LR1e-4; batch unspecified；S21 + S20 |
| ModernTCN/Weather | 336,720 | 128/128 / 5e-05 | 128/128 / 0.0001 | 512/512 / 0.0001；batch,eval_batch待审 | P5: pp17–18 C.1 long-term: LR1e-4; batch unspecified；S21 + S20 |
| ModernTCN/ECL | 96,192,336,720 | 128/128 / 0.0003 | 128/128 / 0.0001 | 32/32 / 0.0001；batch,eval_batch待审 | P5: pp17–18 C.1 long-term: LR1e-4; batch unspecified；S22 + S20 |
| ModernTCN/Exchange | 96,192 | 512/512 / 0.0003 | 512/512 / 0.0001 | 128/128 / 0.0001；batch,eval_batch待审 | P5: pp17–18 C.1 long-term: LR1e-4; batch unspecified；S23 + S20 |
| ModernTCN/Exchange | 336,720 | 512/512 / 0.0003 | 512/512 / 0.0001 | 512/512 / 0.0001；batch,eval_batch待审 | P5: pp17–18 C.1 long-term: LR1e-4; batch unspecified；S23 + S20 |
| TimeXer/UrbanEV | 3,6,9,12 | 128/128 / 3e-05 | 128/128 / 3e-05 | 无新增待决值 | P6: p14 A.2: initial LR1e-4; batch unspecified；project-approved transfer（不是该域paper setting） |
| TimeXer/PJM | 24 | 128/128 / 5e-05 | 128/128 / 0.0001 | 16/16 / 0.0001；batch,eval_batch待审 | P6: p14 A.2: initial LR1e-4; batch unspecified；S28 + S25 |
| TimeXer/ETTh1 | 96,336 | 128/128 / 5e-05 | 128/128 / 0.0001 | 32/32 / 0.0001；batch,eval_batch待审 | P6: p14 A.2: initial LR1e-4; batch unspecified；S24 + S25 |
| TimeXer/ETTh1 | 192 | 128/128 / 5e-05 | 128/128 / 0.0001 | 4/4 / 0.0001；batch,eval_batch待审 | P6: p14 A.2: initial LR1e-4; batch unspecified；S24 + S25 |
| TimeXer/ETTh1 | 720 | 128/128 / 5e-05 | 128/128 / 0.0001 | 128/128 / 0.0001；batch,eval_batch待审 | P6: p14 A.2: initial LR1e-4; batch unspecified；S24 + S25 |
| TimeXer/Weather | 96,192,336,720 | 128/128 / 5e-05 | 128/128 / 0.0001 | 32/32 / 0.0001；batch,eval_batch待审 | P6: p14 A.2: initial LR1e-4; batch unspecified；S26 + S25 |
| TimeXer/ECL | 96 | 128/128 / 0.0003 | 128/128 / 0.0001 | 4/4 / 0.0001；batch,eval_batch待审 | P6: p14 A.2: initial LR1e-4; batch unspecified；S27 + S25 |
| TimeXer/ECL | 192,336,720 | 128/128 / 0.0003 | 128/128 / 0.0001 | 32/32 / 0.0001；batch,eval_batch待审 | P6: p14 A.2: initial LR1e-4; batch unspecified；S27 + S25 |
| TimeXer/Exchange | 96,192,336,720 | 512/512 / 0.0003 | 512/512 / 0.0003 | 无新增待决值 | P6: p14 A.2: initial LR1e-4; batch unspecified；project-approved transfer（不是该域paper setting） |

**来源裁决**：DLinear P1 p9 B.2明确委托代码，四标准域按S索引实施；Weather LR为自身run_longExp.py默认1e-4。TimeMixer P4 p14表7给ETTh1/Weather B128与ECL B32、初始LR1e-2；同作者unify脚本ETTh1/Weather命令行128覆盖未使用的shell变量16，不误取16。iTransformer P3 p14 LR仅搜索集合{1e-3,5e-4,1e-4}，不得任选；ETTh1/Weather/Exchange明确batch32已落地；ECL论文通用32与自身脚本16冲突，推荐脚本16/LR5e-4，但三字段待审。PatchTST没有找到足以唯一确定数值的论文设置/明确超参数委托，其supervised脚本B128/128/32、LR1e-4只作一次集中推荐。ModernTCN P5 pp17–18 long-term明确LR1e-4已落地，batch仍为各H脚本待审；不误用p18 M4 short-term LR5e-4。TimeXer P6 p14 A.2 LR1e-4已落地（含PJM）；batch代码补充未冒称论文值，PJM脚本真实16，不拿NP的4。UrbanEV及其他论文缺失域保留已批准迁移值；TimeMixer/PatchTST/TimeXer的Exchange尤其不套其他域的新batch/LR。未发现上述所读论文中独立eval batch数值；本项目默认eval=train，非作者全部runner复现。

作者路径/摘要/显式及继承行号的完整机器记录为`source-code-index.json`；下列S标识对应同模型自身仓库，commit未更新：

| S | 文件 | commit | SHA-256 |
|---|---|---|---|
| S1 | `/public/home/yueweiting/大论文/LTSF-Linear/scripts/EXP-LongForecasting/Linear/etth1.sh` | `0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6` | `cdd402ec2f78dc525485d278b3360bb705995b2d72d7ae2da8e77c3fc6094ff5` |
| S2 | `/public/home/yueweiting/大论文/LTSF-Linear/run_longExp.py` | `0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6` | `7519b8f95f06768da5f2faae5c10ae23855015df764298e79c7d85cd1396ab2b` |
| S3 | `/public/home/yueweiting/大论文/LTSF-Linear/scripts/EXP-LongForecasting/Linear/weather.sh` | `0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6` | `b67d06322415e368e970c18919aaec78952e50687c3c1373912151cedda82016` |
| S4 | `/public/home/yueweiting/大论文/LTSF-Linear/scripts/EXP-LongForecasting/Linear/electricity.sh` | `0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6` | `26ba11e76db043dbed35ced79b642253c7c6750572918828a124662da45551da` |
| S5 | `/public/home/yueweiting/大论文/LTSF-Linear/scripts/EXP-LongForecasting/Linear/exchange_rate.sh` | `0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6` | `01f6734a7bf2b888113b95f45343d93dd7238891049d6bb760faafcdfb6cd18b` |
| S6 | `/public/home/yueweiting/大论文/PatchTST/PatchTST_supervised/scripts/PatchTST/etth1.sh` | `204c21efe0b39603ad6e2ca640ef5896646ab1a9` | `556c87cd7dd795094ff6cb73465e84c90686017e1535e0ba23ed43683b9b0897` |
| S7 | `/public/home/yueweiting/大论文/PatchTST/PatchTST_supervised/run_longExp.py` | `204c21efe0b39603ad6e2ca640ef5896646ab1a9` | `44345cb6c6a5e7a6ff95b8e95c988aef3386d9fff7ddd23cb5fee5d3a151bdfc` |
| S8 | `/public/home/yueweiting/大论文/PatchTST/PatchTST_supervised/scripts/PatchTST/weather.sh` | `204c21efe0b39603ad6e2ca640ef5896646ab1a9` | `3f9e2d58fa3f3cb725f7d10b98e99fa1226c07899c2d4a20f7c9eae98a7f386b` |
| S9 | `/public/home/yueweiting/大论文/PatchTST/PatchTST_supervised/scripts/PatchTST/electricity.sh` | `204c21efe0b39603ad6e2ca640ef5896646ab1a9` | `e20aefa2c3ebda862dab6eb9f970b2f139c8b30b8d294fd3f8ed355816808920` |
| S10 | `/public/home/yueweiting/大论文/iTransformer/scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh` | `c2426e68ca13f74aaec08045c5c724d8ad328124` | `81f8409cff60b754b45ed2e76133c7e3b6e585a50c8361f04fa3daf3b6997cc0` |
| S11 | `/public/home/yueweiting/大论文/iTransformer/run.py` | `c2426e68ca13f74aaec08045c5c724d8ad328124` | `650873cfd6fee1f0415d4e8fd69943ad736fd1bab6495b640aa27b1a769f59f9` |
| S12 | `/public/home/yueweiting/大论文/iTransformer/scripts/multivariate_forecasting/Weather/iTransformer.sh` | `c2426e68ca13f74aaec08045c5c724d8ad328124` | `7cbdad479bb1a000b5c5bab82635a4bb5b700fd3e1efe922b265309b9331d054` |
| S13 | `/public/home/yueweiting/大论文/iTransformer/scripts/multivariate_forecasting/ECL/iTransformer.sh` | `c2426e68ca13f74aaec08045c5c724d8ad328124` | `44c695c21a4ac03d440e09ddcd6e39cb7cb0c09095df085f423281461a983014` |
| S14 | `/public/home/yueweiting/大论文/iTransformer/scripts/multivariate_forecasting/Exchange/iTransformer.sh` | `c2426e68ca13f74aaec08045c5c724d8ad328124` | `834aa1a46dcd67fc7aae98c97b441ede0e311a7b848ca3d6e1a66ac0476204d8` |
| S15 | `/public/home/yueweiting/大论文/TimeMixer/scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_unify.sh` | `e24610583b36fdd8c76cc17a8df4e65759a5f460` | `13086d899b992b67cdd4409d26f686c1ac7e15e9ad4fd03dd53a2e0728554fc4` |
| S16 | `/public/home/yueweiting/大论文/TimeMixer/run.py` | `e24610583b36fdd8c76cc17a8df4e65759a5f460` | `2d945cae47154e08e2a0aa8f794e2a25fb1bd5ce8ff3e070d2692326fe197024` |
| S17 | `/public/home/yueweiting/大论文/TimeMixer/scripts/long_term_forecast/Weather_script/TimeMixer_unify.sh` | `e24610583b36fdd8c76cc17a8df4e65759a5f460` | `24638b0358519d62024bc7d4e42ad76e0cb71f4b85b74482d32bd15d3816f8d2` |
| S18 | `/public/home/yueweiting/大论文/TimeMixer/scripts/long_term_forecast/ECL_script/TimeMixer_unify.sh` | `e24610583b36fdd8c76cc17a8df4e65759a5f460` | `89b0c34325def762e3764e89ca093f15a36188f8e93705fe3cea02d66e675123` |
| S19 | `/public/home/yueweiting/大论文/ModernTCN/ModernTCN-Long-term-forecasting/scripts/ETTh1.sh` | `56a9a2c018385cd5acef015378cae7f084d1b11c` | `3b70d51a3be4f2875dc5032fae1a03334144391410a2876812bcdf07adcef3fb` |
| S20 | `/public/home/yueweiting/大论文/ModernTCN/ModernTCN-Long-term-forecasting/run.py` | `56a9a2c018385cd5acef015378cae7f084d1b11c` | `e08dc1fd5fe38851a085c3fe6f54085cc0d207233910fa5d71f00341cb724e75` |
| S21 | `/public/home/yueweiting/大论文/ModernTCN/ModernTCN-Long-term-forecasting/scripts/weather.sh` | `56a9a2c018385cd5acef015378cae7f084d1b11c` | `ab689e89b1907f4fdf67438d4b9ba5307dc48a8d03d7bc1463a6f3db6986852a` |
| S22 | `/public/home/yueweiting/大论文/ModernTCN/ModernTCN-Long-term-forecasting/scripts/ECL.sh` | `56a9a2c018385cd5acef015378cae7f084d1b11c` | `776d1f1e325fa40cf7dee19655ddd37b4a200639ca4e0bd7184fb1fa5795c3f1` |
| S23 | `/public/home/yueweiting/大论文/ModernTCN/ModernTCN-Long-term-forecasting/scripts/Exchange.sh` | `56a9a2c018385cd5acef015378cae7f084d1b11c` | `a0b9bde526f2e53115a10d02f8aec036291eddbf11ec44010dbd837a475cb4d6` |
| S24 | `/public/home/yueweiting/大论文/TimeXer/scripts/forecast_exogenous/ETTh1/TimeXer.sh` | `76011909357972bd55a27adba2e1be994d81b327` | `ea1f533f05f1efad9ba8711e5d4c0ecb5d1c61ef1101c88c244c57e06407cdda` |
| S25 | `/public/home/yueweiting/大论文/TimeXer/run.py` | `76011909357972bd55a27adba2e1be994d81b327` | `55e3ec417d876049d39eda74e48108076ac145be2b2999834eeae53500e85b1e` |
| S26 | `/public/home/yueweiting/大论文/TimeXer/scripts/forecast_exogenous/Weather/TimeXer.sh` | `76011909357972bd55a27adba2e1be994d81b327` | `b92af18050d96c5db4810fe6bbfda75786d815df701702144cccb4828c60d7a5` |
| S27 | `/public/home/yueweiting/大论文/TimeXer/scripts/forecast_exogenous/ECL/TimeXer.sh` | `76011909357972bd55a27adba2e1be994d81b327` | `d507c4142dd8c946c469199daf7c7af0dbd6b9729de945f6b87bdc9054e6512c` |
| S28 | `/public/home/yueweiting/大论文/TimeXer/scripts/forecast_exogenous/EPF/TimeXer.sh` | `76011909357972bd55a27adba2e1be994d81b327` | `4510275fcd7734128459d71f81db90b135bf54bde76980435796d1eb49840855` |

### 17.3 实际实现与算术

`utils.ch3_contract.baseline_training/profile`按共同→dataset→external层合并，只允许batch/eval_batch/lr、正有限值/整数batch和明确source/pending；AMD/J/N/S任何外部覆盖层被拒绝。`training_blockers`由preflight和模型验收配置在业务fixture前拒绝未决来源。`step_arithmetic`不读数据，为495run导出train窗口/整批/丢尾/最大optimizer steps及validation/test尾批；test仅端点算术。`effective-profiles.json`逐run记录新旧profile摘要、训练值、来源待决项；完整协议摘要与旧审批/报告不兼容，新artifact根为本包，旧文件不覆盖。

逐run求和：原最大optimizer调用规划15,883,010，当前部分落地、含未决占位值的算术为16,140,950，增257,940；这不是最终已锁定步骤预算，57个未决run锁定后须重算。495/5340不变，旧耗时不沿用，ETA Unknown。PJM A/J仍36500训练窗/285整批/丢20，validation5219窗尾99、test10460尾92（算术）。

### 17.4 首轮结果与明确机械问题

原`/public/home/yueweiting/大论文/amd-execution-evidence/m5/m5-data-admission-ozab6blq/probe/complete.json` SHA=`bae5e732a1d41d8b3cba53c11e55fdd75b6809ee4b36db8828addd46d85993ef`；54组13 Passed/22 ResourceNotVerified/19 Blocked；206worker、1183 Adam调用、1580 forward、1184 backward。失败调用不等于成功更新、不退款。首轮失败分为iTransformer六组缺output_attention、九组RSS单调增长、PatchTST/TimeMixer/ModernTCN ECL B128三组单路OOM、TimeMixer/Exchange数值轨迹不一致，以及资源归属退出时序（RNV另外计）。

iTransformer adapter的纯参数构造`native_options`补作者run.py默认output_attention=False，六域必要键静态/CPU核验；不改作者主体。NVML `ExitObservation`只保存本波曾稳定核验的host PID/start_ticks，UUID或生命周期变化拒绝；退出或元信息短暂不可得时最多3秒等待，该窗口永不发许可，须后续稳定同生命周期映射恢复或NVML条目消失，未知外部PID仍立即拒绝。首验TimeMixer自有7425→host28137退出窗口触发拒绝，唯一机械修复覆盖metadata_unavailable短暂状态，修后真实8581→host33215等待并清除成功。无无限缓存归属、无未知内存填0、无CPU/GPU环境变更。

九组memory.json/trajectory证据最后4点CUDA allocated均不增长，RSS小幅单调增长（`memory-growth-review.json`）。原hash对每个GPU参数/Adam状态先复制CPU，再numpy.tobytes额外分配bytes；已用memoryview消除第二份临时bytes，四种tensor布局/类型/空张量摘要与旧实现一致。此修复不证明原九组host增长已消除；原allocated/RSS四点严格单调停止线原样保留，不删断言/不加empty_cache。**待审唯一建议**：未来在固定update后、任何CPU哈希前测RSS，同时保留哈希后RSS作归因，预分配六个日志槽；相同四点规则仍适用于模型阶段RSS及allocated。模型阶段真实单调增长仍拒绝；只有哈希后增长不能冒称GPU泄漏。因改变测量相位会影响准入，该方案本轮未实施/未批准；需正例（两阶段增长）与反例（只额外CPU摘要分配）验收后才替代旧判据。

TimeMixer/Exchange已逐H比较旧serial/q4/q2 JSON：initial/RNG/batch相等，但trajectory内optimizer/参数摘要及部分final不同，详见`timemixer-exchange-review.json`；未解析旧checkpoint、未测差异幅度、未放宽exact。该域论文无新batch/LR唯一值、保留现有project-approved transfer，尚未证明可机械修复；本轮不重复旧整套诊断，完整六步一致性仍Blocked，需在最终经审核profile补测中保留严格对照。ECL旧B128 OOM事实不改；A/J ECL保持128；仅TimeMixer ECL新B32/T512/H96本轮两步短验收通过，不外推其他H或四路，PatchTST/ModernTCN的ECL来源batch仍待审未负载。

### 17.5 固定验收、实际成本及证据复用

精确24个CPU ID先固化于`frozen-execution-order.json`及配置acceptance，fixture前验证代码/配置/用途，正常unittest生命周期，test方法不互调。24首次passed；因上述监测机械修复，同24方法修后再次passed，合计48方法调用，0fail/error/skip/unexecuted；参数化子case定义由同SHA源码及495清单冻结，单列`subcase-inventory.json`（每轮1420个subcase，不隐藏为零）。没有重跑旧13项/333/339/8源smoke。

| CUDA方法 | 首次 | 唯一机械复验 | 每次实际调用 |
|---|---|---|---|
| ModelTests.test_itransformer_defaults：U/F4、batch2 | 计算及监测通过 | 同修后监测通过 | 2 Adam /6 forward /2 backward；自产best/last及拒绝跨身份恢复、预测复现 |
| ModelTests.test_time_mixer_ecl_batch：T512/B32/H96/LR.01 | 计算passed、退出监测失败 | 计算/监测通过 | 2 Adam /3 forward /2 backward；1 validation、finite/参数更新 |

当前两次合计8 Adam/18 forward/8 backward，从机械512扣8，余额504；本轮64/96/64额度尚余56/78/56，但不自动使用。首次资源失败成本不退款、不拼接两个版本。修后自然退出样本见`natural-exit-verification.json`；两方法实际采样间隔分别0.314–0.650s、0.298–0.597s，不把100ms目标写实测；TimeMixer进程采样峰值29379002368 bytes，约27.36 GiB。旧4GiB缺项不倒改；本轮只获这两个有限路径的证据，不授予54组资源Passed。未新做张量诊断或新并发；CPU最终24/24和GPU最终两方法在同一修后代码/bundle上通过。

### 17.6 精确补测/继承与剩余待决

`followup-plan.json`从54组生成完整划分：10组（25代表worker）仅列有条件继承审核，其余44组170worker补测；每组明确changed_profiles、pending_source、parent_status、代表ID。继承必须在后续审核核对父证据profile/源/模型数学/摘要等价及环境硬件，不能把相同seed当证明，不能直接用旧全局protocol报告。无原Passed的组不自动改Passed。

可提交继承的10组：`AMD-UrbanEV-F2`、`AMD-PJM-MS`、`J-UrbanEV-F4`、`J-PJM-MS`、`DLinear-PJM-MS`、`PatchTST-PJM-MS`、`PatchTST-Exchange-MS`、`ModernTCN-UrbanEV-F4`、`ModernTCN-PJM-MS`、`S-UrbanEV-F4`。

补测固定流程上限：170代表串行1020 Adam；其中168代表为四任务组，候选q4最多1008，必要q2回退最多1008；合计3036 Adam/4048 forward/3036 backward/1012 validation batch。原首验3510已耗1183，剩2327；**额外首验709 Adam仅Proposed，当前批准额外为0**，不与504机械余额混账。改变batch/LR不重置预算。参数未决若后续批准会更新协议/计划摘要；该3036是完整待补集合的保守规划，并非本轮授权启动量。

集中待决只有：①表17.2的PatchTST数值、iTransformer LR/ECL冲突、ModernTCN与TimeXer batch推荐；②上文RSS测量相位方案，当前旧停止线不动；③TimeMixer/Exchange保持exact的最终profile补测，若仍不一致停止，不自动容差救援；④补测首验额度709及10组具体继承依赖审定。数据§16政策没有新增缺口，不重读真实前缀/CSV；来源未决不是数据缺陷或模型不适用。

共享`start_probe.sh`不复制/不重建，资源入口增加followup精确划分和父complete SHA、许可followup摘要检查，每组启动前保守预扣最大消耗，避免末尾才发现越额。当前dry-run为54/195；preflight退出2，真实拒绝来源待决/当前mandatory、dirty closure及无新许可；未生成未来commit或review=true模板。因预算与来源尚未批准，不能声称只剩审核/closure。完整正式模型入口仍独立需结构冻结/M6授权，不自动启动下个模型。

### 17.7 操作命令与停止点

下列start仅供后续来源/判据/额度及继承条件落实、修后审核和统一closure、绑定实际commit的新review文件生成后，由用户一次执行；**当前preflight预期拒绝，Codex未执行start**。新review须包含完整protocol/code/environment/hardware与`followup_sha=digest(execution.followup)`，不得沿用旧probe-review。后续若需要改配置以采纳待决项，须同一整体审核更新effective profiles/计划/许可，不直接手改review绕过。

```bash
cd /public/home/yueweiting/大论文/AMD
export PATH="/public/home/yueweiting/大论文/amd-execution-envs/m5-source-smoke-8sr2d3d_/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
E="/public/home/yueweiting/大论文/amd-execution-evidence/m5/m5-baseline-followup-0r6ra554"
bash scripts/ch3/start_probe.sh dry-run --approval "$E/probe-review.json"
bash scripts/ch3/start_probe.sh preflight --approval "$E/probe-review.json"
# 仅全部前置通过后由用户执行；本轮未创建此许可
bash scripts/ch3/start_probe.sh start --approval "$E/probe-review.json"
tail -n 80 -F "$E/probe-controller.log"
bash scripts/ch3/start_probe.sh status
bash scripts/ch3/start_probe.sh logs
bash scripts/ch3/start_probe.sh complete
# complete=true只表示文件完成，仍须审核逐组status、继承和预算
bash scripts/ch3/start_probe.sh safe-stop
```

停止时只写本包STOP、结束本包worker，保留logs/staging/自产checkpoint，不自动fresh重跑。日志逐组位于`$E/probe/<group>/.../worker.log`；机器状态`$E/probe/progress.json`、终态`complete.json`。本轮不重演tmux/互斥锁，占位入口未改；启动脚本字节保持。当前交付仅修后代码、源表、实际短验收及补测准备待ChatGPT审核，不声称ChatGPT已读新字节。


## 18. 用户授权ChatGPT直执行：来源定值、测量诊断及工程收口

### 18.1 授权与执行边界

用户在本会话明确提出Codex额度用尽，授权ChatGPT通过Remote Desktop Commander直接执行上一轮方案并检查closure。本次直接接手仅限该方案，不更改长期职责规则。起点HEAD/local/tracking/live remote=`9e71fd572f866797376584f93cf4b25bd6840c34`、0/0，保留§17九项modified；未使用reset/clean/stash。修前canonical/M5 SHA分别为`ca4ddf3fa8d17828049ae8a7e67704ea3d92f569e664436efe3835e57a5c899c`、`ed685eadadbcfe94313448f71de53f01c3a087aa3ec24dfe4119b4863adcb2b0`。

本轮独立证据根：`/public/home/yueweiting/大论文/amd-execution-evidence/m5/m5-approved-source-direct-veaod729`；fixture=`/tmp/amd-m5-direct-1swdep9l`。`authorization.json`记录用户授权、执行者及32 Adam/48前向/32反向合并诊断上限。作者仓库/环境、原观测、旧checkpoint、AGENTS、Closed M4、baseline与空间方案保持；完整probe/正式训练均未启动。

### 18.2 最终来源定值与范围

按用户本次选择落实§17.2全部57个未决run，标记为user-approved official-code supplement。原论文未给唯一值的字段不冒称paper-explicit；iTransformer/ECL脚本B16与论文通用B32的差异保留，采用16是本次明确选择。eval batch等于train batch；以下四元组顺序为H96/192/336/720。

| 模型 | 数据集 | train/eval batch | 固定LR |
|---|---|---|---|
| PatchTST | ETTh1、Weather | 128 | 1e-4 |
| PatchTST | ECL | 32 | 1e-4 |
| iTransformer | ETTh1、Weather、Exchange | 32 | 1e-4 |
| iTransformer | ECL | 16 | 5e-4 |
| ModernTCN | ETTh1 | 512 | 1e-4 |
| ModernTCN | Weather | 256/256/512/512 | 1e-4 |
| ModernTCN | ECL | 32 | 1e-4 |
| ModernTCN | Exchange | 128/128/512/512 | 1e-4 |
| TimeXer | PJM | 16 | 1e-4 |
| TimeXer | ETTh1 | 32/4/32/128 | 1e-4 |
| TimeXer | Weather | 32 | 1e-4 |
| TimeXer | ECL | 4/32/32/32 | 1e-4 |

其他已落实值和项目迁移值不变，未搜索T/batch/LR、不增加scheduler/AMP/seed；来源路径/commit/SHA继承§17并逐条列于`source-decisions.json`。495逻辑任务、全部T/epochs/patience/结构/信息合同不变，274个AMD家族完整profile与§17逐项相等；F0与PJM TargetOnly仍为0。最终max_optimizer_steps算术为16,482,750，非正式训练授权或耗时实测。`effective-profiles.json`逐run记录参数和窗口/尾批，TimeXer/PJM新B16为train2281整批/丢尾4、validation尾批3。

### 18.3 测量实现及预算前置

`probe_worker`预分配六个trajectory/batch/memory槽，在每次update后、CPU状态哈希之前记录RSS及CUDA allocated，再记录哈希后RSS。保留四点严格递增判定（allocated或rss_before_hash），哈希后增长单独可见。明确本步哈希前RSS仍可能受上一步哈希/allocator历史影响，不把它宣称为纯模型内存，也不自动将旧九组改Passed。摘要byte语义、update/evaluate/init_training/save/restore/formal_worker数学代码不变；未设置empty_cache或新数值容差。

新增纯合同函数`followup_limits`同时检查54组精确划分、170代表、3036/4048/3036/1012上限以及2327＋709的授权；缺少额外额度或匹配followup摘要必须在创建probe目录/worker前拒绝。父complete SHA和25条继承代表profile在创建输出前再次核对。新许可还需绑定`approved_extra_adam=709`，未生成许可、未消耗额外709。

限定诊断使用单独`ch3_step_diagnostic`目的，仅允许当前包固定AMD/ETTh1 H96与TimeMixer/Exchange H192，6/8/6每worker、180秒、真实配置/源码绑定，禁止观测及旧权重访问；这不是解除dirty状态下完整probe审批。诊断保存的CPU参数/梯度/Adam状态只属于本包合成证据，不得转为正式产物或资源许可。

### 18.4 验收、实际诊断与未通过项

16个SourceClosureTests首次完整通过：16 passed、0 failure/error/skip/unexecuted，无复验、无test互调。覆盖57定值、274家族不变、495/5340、继承profile、违规覆盖、端点尾批、旧报告拒绝、四点增长正反例、709前置拒绝与有限诊断入口。测试ID及顺序在`frozen-execution-order.json`先行固定。参数化子case按日志保留，不冒称为0。

五个真实CUDA诊断worker均exit0、finite，进程监测与退出观察无失败：AMD/ETTh1一次6步；TimeMixer/Exchange H192串行1、两路各6步，以及出现摘要差异后唯一串行重复。合计30 Adam、40 forward、30 backward，未做额外机械复验，未超32/48/32。机械池原504减30=474，首验补测池与机械池不混账；原§17及更早失败/嵌套test额度偏差保持。

AMD/ETTh1原增长触发代表的末四步allocated/RSS无持续递增，新测量检查通过。这只证明本次代表路径，不代表原九组已全修复或长期无泄漏。

**TimeMixer/Exchange仍不满足exact。** 四个worker初始状态/RNG/batch相同；本轮loss、validation、final RNG一致，但两次独立串行也自第1步出现`enc_embedding.value_embedding.tokenConv.weight`梯度及Adam moment非逐位差异。所有本轮对照中最大梯度绝对差`4.470348358154297e-08`、最大Adam状态绝对差`1.1175870895385742e-08`；一份并发副本第6步仅1/48个卷积权重出现绝对差`1.4901161193847656e-08`。对应张量、步次和幅度见`numeric-state-comparison.json`；不由小误差自动判等。

该证据说明问题不是仅在并发时出现，已定位首个可见分歧的参数/梯度路径；仍不能唯一归因某个CUDA kernel或推出长期误差无害。没有改变确定性设置、容差、seed、模型、batch或LR。按现行exact规则仍Not passed，完整补测preflight保持明确阻塞，不通过临时改为单路掩盖串行自身非逐位重复性。

新增28份自产CPU状态文件仅用于本次比较；没有读取或哈希历史checkpoint。本轮0真实观测读取、0test解析/评价，合成诊断不属于正式效果实验。

### 18.5 源码审核、证据复用与下一步

57参数和709额度均已落地。10组/25代表继承profile与父证据实际核对一致，计算/作者源/环境依赖延用已有审核，原Passed记录不倒改；其余44组/170代表计划仍保留。新范围及父来源见`inheritance-verification.json`，完整补测没有运行。

动态CPU与诊断完成后，唯一配置增量为`execution.probe.mandatory_blockers`从“待诊断”改成实际TimeMixer exact阻塞说明；其他参数/来源/额度/test ID/生产代码保持相同。该纯状态文本差异的复用依据见`post-test-binding-delta.json`，不宣称对最终全局摘要又重复执行过验收。

本轮可以对已验收实现和真实未通过记录进行工程版本closure，但这不是M5阶段关闭，也不意味着TimeMixer数值验收通过。准确closure commit、三端/clean、保护对象及逐文件最终SHA记录在外置`closure-verification.json`，不为文档自写commit递归修改/提交。

下一步需用户明确选择针对TimeMixer该路径的确定性执行验证或限定数值准入修订；本轮不擅自执行任一选择。源参数和709额度无需再次确认。未知来源/数据政策不重开，完整补测不得带已知exact阻塞启动。

操作（本轮不执行start）：
```bash
cd /public/home/yueweiting/大论文/AMD
export PATH="/public/home/yueweiting/大论文/amd-execution-envs/m5-source-smoke-8sr2d3d_/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
E=/public/home/yueweiting/大论文/amd-execution-evidence/m5/m5-approved-source-direct-veaod729
bash scripts/ch3/start_probe.sh dry-run
bash scripts/ch3/start_probe.sh preflight
# preflight当前应拒绝TimeMixer exact阻塞及缺少新review许可。
# 不创建review=true，不运行start，不删除/覆盖旧probe结果。
```

## 19. 用户确认数值等价：限定TimeMixer/Exchange、现场复验与工程closure

### 19.1 本次授权和预登记标准

用户原话“改成第二种吧. 然后继续直到完成closure”，延续本会话已明确授予ChatGPT的服务器执行权限。本轮只处理此数值准入修订和必要复验/提交，不修改长期职责、模型数学或确定性设置。起点三端commit为`0a446ac7f0b29dc8e7a85c67a529f473e0888fae`，0/0、clean；原canonical SHA=`aa29e877050bf048a5e6223351097edf254bfba55f490e7c027e8deb60a81d15`，原M5 SHA=`fdf954510225ec8934342eb4cea9c7a5851e40f243cdbcb121fc1e4e39674892`。

独立证据根：`/public/home/yueweiting/大论文/amd-execution-evidence/m5/m5-numeric-equivalence-80dirugd`；fixture=`/tmp/amd-m5-numeric-0ln4445u`。在运行新轨迹前写入`authorization.json`（SHA `495cec7f05357e3c0229fd63631d99c6df76f580c4a91ac6f87365e6f8bfd46c`），固定绝对界，不依据本轮结果调阈值。

规则ID=`timemixer-exchange-tokenconv-atol1e-7-v1`，完整policy digest=`400210ef6c5e1f6104b2e1b90406e75a1d360d63bcf53465a201ffd02ded7d82`。适用范围仅TimeMixer/Exchange、H96/192/336/720、当前锁定source/structure。具名张量为`enc_embedding.value_embedding.tokenConv.weight`，float32、[16,1,3]。仅其权重、梯度、Adam exp_avg和exp_avg_sq逐元素采用`abs(a-b)<=1e-7`，rtol=0、非finite无条件拒绝。不默用torch默认allclose容差。

每步loss和validation MSE/MAE采用同一绝对界；SSE/SAE先用严格相同元素数归一化再比较，验证自身聚合关系不变。原始统计和摘要均保留。初始模型/RNG、输入batch摘要、最终RNG、任务/profile、元素数、步次完全一致；Adam step/param_groups、全部白名单外模型与梯度/optimizer状态仍exact。没有关闭或豁免其他参数，其他模型/数据集仍采用原exact规则。该标准是本次用户选择后的前瞻性工程准入规则，不证明原exact通过或长期误差无害；根本CUDA内核原因仍未唯一确定。

### 19.2 实际接入及验收

`numeric_probe_policy`校验完整固定规则，非法模型/范围/阈值变更拒绝；`numeric_probe_snapshot`每步只输出四组各48个数及形状/dtype，同时保存白名单外精确摘要，按真实optimizer参数映射而非写死state编号。`compare_probe_trajectories`在严格身份后做逐步数值比较，返回最大差与失败字段；新证据字段缺失时拒绝，不能用旧哈希猜差异。完整probe已调用同一比较器，报告标bounded_numeric而非exact，正式报告验证要求对应policy及数值审核摘要。没有新增保存旧checkpoint或改变训练算子。

16个NumericEquivalenceTests首次16 passed、0fail/error/skip/unexecuted、无复验。覆盖1e-7边界、越界拒绝、禁用相对容差、NaN/Inf拒绝、身份/RNG/batch、形状/dtype、非白名单状态/步数、其他模型exact、统计归一化、快照映射和报告许可绑定。test方法不互调，参数化subcase按日志留存；没有重跑旧套件。详细ID在`frozen-execution-order.json`。

新现场只用TimeMixer/Exchange H192既定T96/B512/LR3e-4及固定seed2024合成数据，依次串行参考→两路各六步→独立串行重复。每worker6 Adam/8前向/6反向，4worker合计24/32/24（本轮合并上限32/48/32）。所有workerexit0、finite、监测/退出通过、memory_review未阻塞。没有新增CUDA设置、AMP、seed、batch或LR。

相对于串行参考，三份对照全部在预登记数值界内；本轮max abs：权重0；梯度`7.450580596923828e-08`；exp_avg `1.4901161193847656e-08`；exp_avg_sq `3.637978807091713e-11`；loss和归一化validation差0。初始/RNG/batch和残余状态exact。每份对照检查1152个白名单数值元素，原hash仍可不同且bitwise_equal=false，不能将新数值通过写成逐位通过。

该H192有限正例证明生产比较入口能正确执行新规则；不是四H和四并发全部通过，其他H仍在完整补测中逐项核验。原首轮及§18的exact失败不改。此次无test数值/真实观测读取、无checkpoint读取/反序列化，只读本轮JSON数值证据；没有完整epoch或性能实验。

### 19.3 预算、复用和前置条件

原机械余额474减24=450；§17的8和§18的30消耗、原首验1183及历史失败/额外test调用均保留。原首验剩2327＋已批准709=3036完整补测上限不变，本轮未消耗补测池。CPU16次不挪用或重置历史账。

495任务/5340 run-epochs、54组195代表、全部495个有效profile和274个AMD家族profile与上个closure一致；max_optimizer_steps=16,482,750。`update/evaluate/init_training/save_state/restore_state/formal_worker/memory_growth_review` AST不变，作者源码和结构表不变。10组/25继承代表逐一核对父trajectory的profile摘要和原Passed，未继承TimeMixer；44组/170代表待补测。继承证据见`inheritance-verification.json`，不将旧报告原地升级。

验收后仅将配置中的待验收blocker改为空及更新监测状态说明；预注册policy、阈值、source、预算和计算字节未再变化。该纯状态文本差异见`post-test-binding-delta.json`，不宣称最终全局digest重新跑过16项。本轮整体源代码/配置审核后精确stage/commit/push，实际commit、三端/clean和保护对象写外置`closure-verification.json`，不为记自身commit递归提交。

### 19.4 closure成功后的操作

本轮不启动完整补测。closure成功、三端0/0/clean、文档字节一致后生成只适用于当前版本的外置`probe-review.json`，包含实际commit/code/config/environment/hardware、followup digest、approved_extra_adam=709及numeric policy摘要。不得将structure_frozen/m6_authorized设true；无负载preflight应blocked=[]后才交用户启动。工具和原tmux入口不重建，仍一个补测入口、固定波次、无模型间自动正式启动。

```bash
cd /public/home/yueweiting/大论文/AMD
export PATH="/public/home/yueweiting/大论文/amd-execution-envs/m5-source-smoke-8sr2d3d_/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
E=/public/home/yueweiting/大论文/amd-execution-evidence/m5/m5-numeric-equivalence-80dirugd
bash scripts/ch3/start_probe.sh dry-run --approval "$E/probe-review.json"
bash scripts/ch3/start_probe.sh preflight --approval "$E/probe-review.json"
# 仅实际closure/新许可/preflight成功后，由用户一次启动：
bash scripts/ch3/start_probe.sh start --approval "$E/probe-review.json"
tail -n 100 -F "$E/probe-controller.log"
bash scripts/ch3/start_probe.sh status
bash scripts/ch3/start_probe.sh complete
bash scripts/ch3/start_probe.sh safe-stop
```

complete只表示终态覆盖，不等于全部准入通过；若模型/数值/资源仍有失败，保留证据审计，不能再自动放宽容差或重跑。工程closure不等于M5 Closed；J未冻结、M6和正式训练均未启动。数据政策/原论文限制及空间路线不变。


## 20. 第二轮补测审计、复用CPU摘要缓冲区及三组数值诊断

### 20.1 授权、现场与第二轮结果

用户本轮“修复然后给出补测指令”承接已授权的ChatGPT服务器执行。本轮不自动扩大容差、改变确定性或科学配置。起点三端`6805a910d1ed7fcfb2b0726155bca3c3cd1cca22`、0/0、clean。证据根`/public/home/yueweiting/大论文/amd-execution-evidence/m5/m5-buffer-repair-zwqjvgf_`；fixture=`/tmp/amd-m5-buffer-grj1mrp3`。开始前授权清单SHA=`241552d9c367e3e79d4d2a27b911f706553ae33a826dc05b6c94d63977dd3711`。

第二轮parent为`m5-numeric-equivalence-80dirugd/probe/complete.json`，SHA=`def4e11ed39b4df9b1cd3aa26f597f2a0e1b5bb0d81b1fa75a986f99c6c0af7f`。54组完整终态：34 Passed（10继承＋24新通过）、20 Blocked；Passed中27组四路、7组单路。290个worker实例共1740 Adam/2320 forward/1740 backward；17组因RSS增长检查、3组因exact数值差异阻塞。本次已尝试负载没有OOM和NVML不准入，不等于未执行H或正式长训练也已验证。旧complete/trajectory/log和失败事实不改，结果登记可接受与总体资源gate未通过分开。

### 20.2 摘要机械修复与实际复验

`ReusableTensorDigest`为每个dtype一次分配可容纳最大已登记state张量的CPU缓冲区，构造期预触页并缓存各shape和字节视图。每次hash使用同步copy_复用，不再对每个GPU参数/Adam状态调用.cpu()分配临时CPU tensor。未知dtype/shape拒绝，不静默扩容；包含Adam懒建立的float32 step标量。保留原递归排序、dtype/shape头和字节语义、原RNG/训练/指标/保存恢复函数。GPU state不修改；不设置empty_cache或新CUDA选项。TimeMixer/Exchange原白名单残余摘要也使用同一复用接口，规则本身不变。

三个专用、无状态捕获的原RSS失败代表：AMD/ECL/H192、PatchTST/ECL/H336、TimeXer/ECL/H96，各6更新＋2validation，通过原四点检查；与原trajectory的initial/RNG/batch/逐步参数及optimizer hash/loss/validation/final全部一致。CPU摘要缓冲区分别4,194,312 /11,010,056 /10,240,000 bytes。其结果仅证明这三条修复路径，不把原17组批量追改Passed、不保证长期无泄漏。详见`rss-repair-verification.json`。

### 20.3 限定数值诊断：不是均在1e-7内

针对TimeMixer/ETTh1、TimeMixer/ECL、ModernTCN/ETTh1，均取当前H96正式profile；每组串行参考、两路各一份、独立串行重复，每worker六步。参数/梯度/Adam状态只保存于本轮自产合成文件。初始state、RNG和batch身份均exact，三组独立串行重复也不exact；没有证据把差异只归于并发。首个可见梯度差异分别在TimeMixer tokenConv和ModernTCN downsample卷积路径，未唯一定位底层kernel。

| 诊断组 | 模型state最大绝对差 | 梯度最大绝对差 | Adam state最大绝对差 | validation MSE最大绝对差 |
|---|---:|---:|---:|---:|
| TimeMixer/ETTh1/H96 | 1.5497207641601562e-6 | 1.9073486328125e-6 | 2.384185791015625e-7 | 6.763589599501074e-8 |
| TimeMixer/ECL/H96 | 7.078051567077637e-8 | 9.5367431640625e-7 | 1.1920928955078125e-7 | 0 |
| ModernTCN/ETTh1/H96 | 5.246791988611221e-5 | 2.824526745826006e-8 | 3.958120942115784e-9 | 2.5322343333300523e-8 |

ModernTCN最大差位于第6步`model.downsample_layers.0.0.bias`，独立串行差达到上述最大值；差异并非仅单张量或统一1e-8量级。不能据合成指标很接近推断长期无害，也不将原exact改为Passed。`new-numeric-state-comparison.json`逐步给出实际路径、差异元素、绝对/相对幅度；本輪共读取84份自产状态，不读任何历史/正式checkpoint。候选的三组浮点state atol1e-4、loss/归一化指标atol1e-6、rtol0方案仅列于`pending-decision.json`，是诊断后的工程政策提案，未经用户确认，不实施、不追认旧结果。

TimeMixer/ECL四个带状态捕获的诊断中3个在完成六步和状态保存后仍触发RSS判据、exit1；监测均通过，无OOM。逐步torch.save/cpu_tree额外分配是诊断流程事实，但本轮没有做去掉捕获的配对归因，不宣称已证明所有增长来源。捕获诊断不是资源准入通过；原失败与完整成本保留。

### 20.4 验收与实际计账

BufferRepairTests首验16/16通过。收口检查发现新父报告中10个祖先继承组没有直接serial目录，最小修复`inherited_trajectory_path`：只解析父报告已绑定path/SHA的JSON引用，范围/文件名/身份/摘要不符拒绝，不搜索其他artifact。随后同16个方法完整复验通过并增加相关子case；不存在首验失败被隐藏或test方法互调。方法合计32，子case按同SHA源码与日志登记。未重跑作者smoke、旧loader或其他回归。

15个诊断worker合计90 Adam/120 forward/90 backward，在预先固定96/128/96、最多16实例内，无额外模型复验。12 exit0、3 exit1（上述capture诊断RSS），所有六步操作已计费；不可将driver exit0解释成所有worker Passed。机械池450−90=360，补测池不混用。一条离线汇总命令最初缺tools模块路径，导入前退出；补充固定sys.path后成功，模型/测试调用为0，见`audit-command-errors.json`。

495个effective profile和274个AMD家族profile与起点完全一致，预算仍5340 run-epochs、最大16,482,750 optimizer步骤。所有T/batch/LR/结构/数据/seed与原numeric policy均不改变。未读取真实观测或test内容，无完整补测/正式训练。

### 20.5 精确后续计划与停止点

父报告34组/115代表profile与直接或祖先引用SHA逐一核对；仍为原Passed，不冒称修后重跑。其余20组/80代表保留，最多80串行＋80四路＋80两路回退worker，即1440 Adam/1920 forward/1440 backward/480 validation。上包3036−1740=1296剩余，新缺144仅Proposed；不得从机械360借用，不清零历史成本。

曾检查50条个别成功serial的参考复用方案，但未执行；旧serial耗时含旧临时分配，不能直接与新buffer并发耗时计算同实现加速比。因此未启用该执行捷径，保留仅同profile的34个既有完整组继承，其余组从匹配实现建立新对照。详见`serial-reuse-proposal-not-executed.json`与正式`followup-plan.json`。

当前只提交本轮机械修复、证据和未通过事实；工程closure不代表M5结束。三组数值准入扩展及144额度未获批准，完整队列preflight必须拒绝，不生成reviewed=true许可或启动start。下一步用户集中裁决后才能落实新数值验收、确认有效预算、必要复验并生成实际新版本许可。其他34组不重跑，不恢复54组整包。

```bash
cd /public/home/yueweiting/大论文/AMD
export PATH="/public/home/yueweiting/大论文/amd-execution-envs/m5-source-smoke-8sr2d3d_/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
bash scripts/ch3/start_probe.sh dry-run
bash scripts/ch3/start_probe.sh preflight
# 当前应拒绝三组数值政策、额度和缺少新许可；不要运行start。
```

准确commit/三端0/0/clean与文件SHA在本轮外置closure-verification.json登记，不为写入文档自身commit递归提交。AGENTS、Closed M4、baseline、作者源码/环境、空间路线保持；J未冻结、M5未Closed、M6未开始。

## 21. 用户确认三组全浮点数值等价、追加144额度与启动前收口

### 21.1 授权与固定准入规则

用户本轮明确回复“确认”，批准§20提出的三组限定数值等价及额外144次补测Adam。新增范围仅为TimeMixer/ETTh1、TimeMixer/ECL、ModernTCN/ETTh1的H96/192/336/720；全部浮点模型参数与buffer、现存梯度、Adam浮点moment逐元素`atol=1e-4, rtol=0`，loss及归一化validation MSE/MAE/SSE/SAE采用`atol=1e-6, rtol=0`。初始模型、RNG、输入batch、optimizer step、非浮点buffer/状态、param-group结构继续exact；NaN/Inf无条件拒绝。TimeMixer/Exchange §19具名单张量`1e-7`规则保持不变，其他模型/域仍exact。阈值在任何本轮CUDA轨迹运行前写入`authorization.json`。

科学合同未改变：495 runs/5340 run-epochs、全部T/batch/LR/epoch/seed/数据与模型结构、max optimizer steps 16,482,750保持。34组/115代表继承profile及祖先trajectory引用重新核对一致；20组/80代表补测上限1440，由原余额1296＋本轮新批144覆盖，尚未消耗。

### 21.2 实现与CPU验收

为避免再次引入state-sized CPU分配，全浮点比较沿§20的`ReusableTensorDigest`固定buffer，在worker内按固定schema逐tensor复用同一CPU storage写raw sidecar；每步只保留路径/dtype/shape/mode/offset及SHA。浮点条目后续分块比较绝对差，optimizer step和非浮点条目按原始字节exact；optimizer group和非tensor结构写入schema并exact。sidecar缺失、SHA/长度/schema变化、未知dtype、非finite均fail closed。训练`update/evaluate/init_training/formal_worker`数学字节未改。

16个新CPU方法首验在第13项按预期注入NaN时暴露比较器清理机械错误：异常路径仍持有mmap导出view，导致预期ValueError被BufferError覆盖；当次为12 passed/1 error/3 unexecuted。唯一机械修复改为有界分块文件读取，不改阈值/模型/判断；第二次同16方法16/16 passed、0fail/error/skip/unexecuted。

### 21.3 三组现场数值复验

本轮仅合成H96，按每组“串行参考→两路并发→独立串行重复”运行，共12 worker、72 Adam/96 forward/72 backward，全部exit0、finite、资源归属通过，未读真实CSV/test或历史checkpoint。相对串行参考，三份对照均通过预注册规则：TimeMixer/ETTh1最大full-state绝对差`1.9073486328125e-06`、最大标量差`2.384185791015625e-07`；TimeMixer/ECL分别`1.9073486328125e-06`与`2.384185791015625e-07`；ModernTCN/ETTh1分别`4.315376281738281e-05`与`1.1920928955078125e-07`。exact残余状态全部通过。该有限正例不证明完整四H或长期训练稳定，只证明生产比较入口按新规则可执行。机械诊断余额360−72=288；补测池1440未消费。

### 21.4 收口与后续启动边界

本节实现/验收通过后仅允许精确stage/commit/push及三端0/0 clean核验；生成的新`probe-review.json`必须绑定实际closure commit、当前protocol/code/environment/hardware、34/20 followup digest、`approved_extra_adam=144`及四条numeric policy摘要。无负载preflight必须`blocked=[]`。本轮不代用户启动20组补测，不冻结J、不关闭M5、不进入M6或正式训练。补测完成后仍须逐组审计，不能把complete=true等同于全Passed。

## 22. 20组补测完成审计：44 Passed / 10 Blocked

### 22.1 完整性与版本绑定

本轮用户回复“结束了”后执行完成审计。`probe/complete.json`与`progress.json`字节一致，SHA-256=`4e9ceb09b6631b1086e3502915d88e836510ecd7f59f976be312c652ffaa49f2`；54组/195 Q完整覆盖。报告protocol=`a6f0d6daa99294807fe00a21ae3b48265e957a011b0e5ac14f8511c004068217`，与当前配置、code binding、环境和A800硬件均一致；`probe-review.json`仍绑定closure commit `4f27160a4980c9cb110c69093c80e2b867b1306e`、34/20 followup digest与`approved_extra_adam=144`。完成时Git三端一致、0/0、clean；baseline tag仍`fa9665627e6fcfb1d0c2bc22d943ca9666304fd6`。无残留probe worker。

### 22.2 终态与成本

54组最终为44 Passed、10 Blocked；44个Passed中34组来自审核继承，10组为本轮新通过。Passed并发记录为q4 36组、q2 1组、q1 7组。20组补测实际消耗846 Adam/1128 forward/846 backward，低于1440上限，未进入原1296余额之外，因此本轮新批144未实际消耗。0 OOM、0 NVML/外部PID归属失败；完整补测没有正式epoch训练、test评价或正式checkpoint。

本轮10个新Passed组为：AMD/ETTh1、AMD/Weather、AMD/Exchange、J/UrbanEV-F1、J/UrbanEV-F2、TimeMixer/ETTh1、TimeMixer/ECL、ModernTCN/ETTh1、TimeXer/ETTh1、TimeXer/ECL。已批准的三组全浮点数值等价在实际补测中通过，说明§21生产比较入口可用于对应组，但不外推其他模型/域。

### 22.3 剩余七组RSS阻塞

7组由原四点`rss_before_hash`严格递增规则阻塞：AMD/ECL、J/ETTh1、J/Exchange、PatchTST/Weather、PatchTST/ECL、iTransformer/ECL、TimeMixer/Weather。所有触发轨迹的CUDA `allocated`末四步均稳定，未出现GPU allocated连续增长；典型CPU RSS增量多为4–65 KiB级，但PatchTST/Weather含单次约5.55 MiB，iTransformer/ECL含连续约10.53/10.49 MiB。当前证据因此不能将其写成GPU泄漏，也不能在未改合同前直接判Passed；四点规则仍原样保留。

### 22.4 剩余三组数值阻塞

TimeMixer/Exchange四路实际补测在原§19具名单张量`atol=1e-7`规则下仍Blocked：至少一个代表出现parameter `1.1920928955078125e-07`、gradient `1.4901161193847656e-07`超过1e-7；另一个代表虽然具名张量在界内，但第5–6步白名单外model/optimizer/gradient exact摘要变化，因此`exact_residual_state=false`。原H192有限诊断通过不被倒改，但不能覆盖当前四H实际并发结果。

ModernTCN/Weather和ModernTCN/ECL的串行参考与并发worker均正常完成、资源准入通过，但当前仍使用exact规则，四个代表均出现bitwise numerical mismatch，因此Blocked。当前probe只保存摘要，不能从该报告推出具体最大绝对差；若要改变准入，需另做限定数值诊断并由用户明确决定，不得直接套用ModernTCN/ETTh1的1e-4范围。

### 22.5 当前裁决与停止点

本轮完整性/版本/资源审查Passed，但技术准入为44/54而非全通过，故本轮结果review=`Not passed`。这不是效果gate，也不涉及J的预测性能优劣；不能据此冻结J、关闭M5或进入M6。下一步若继续M5，只允许针对7组RSS判据和3组数值准入做限定诊断/政策决策；34组继承和10组本轮新Passed均不应重跑。

## 23. RSS复合判据、固定算子因果对照及10组补测准备

### 23.1 本次授权与预登记

用户要求RSS不能仅因连续增长而失败，还应有累计/平均增量下限与较长窗口无平台；数值重新检验，确为底层算子误差才可降低准入，并要求给probe启动指令。本轮起点三端`47bce39e79a585e46da4362ceeea0ef6236a3fff`、0/0、clean，直接执行权限沿本会话。证据根为`../amd-execution-evidence/m5/m5-rss-kernel-t9a2njbt`，fixture为`/tmp/amd-m5-rss-kernel-yuep_3nw`。授权/预登记、ECL修订和隐藏Weather增量分别存authorization.json、ecl-policy-preregistered-amendment.json、latent-weather-preregister.json。没有新模型、数据集、seed或正式run。

### 23.2 RSS新规则及24步实测

短窗取末4点，GPU allocated的连续增长停止线保持。CPU RSS同时满足连续增长、累计净增>32MiB、净增/step>1MiB才升级长窗核验。少于24个更新时，显著增长标needs_long_window，不将它判为已证明泄漏，也不直接发资源许可。长窗24更新，排除前6步；末8步range<=8MiB且绝对端点平均变化<=256KiB/step即视为平台。只有复合增长条件成立且无平台时CPU增长blocked。未达到显著门槛的六步screen可以通过，但不声称已观察长期平台。出现needs_long_window时先停止该组进入明确长窗诊断，不偷偷把额外更新混进六步makespan。

实际对AMD/ECL/H336、J/Exchange/H720、iTransformer/ECL/H336分别作24更新合成诊断，各26 forward/24 backward；不逐步复制状态做哈希，观测训练路径本身。三个窗口均达到平台，排除warm-up后RSS净增分别12288、86016、45056 bytes。数据和模型/optimizer数学不变；仅为代表证据，不将旧7组全部倒改Passed。

### 23.3 后端误差的直接证据和限定准入

每个目标诊断捕获一次实际输入卷积的输入、权重、bias及上游梯度，固定这些操作数并重放同一Conv1d反向。默认设置重复梯度有差异；仅在诊断上下文切换cuDNN deterministic=True后重复梯度exact；离开上下文恢复全部原flag且操作数SHA保持。TimeMixer/Exchange、ModernTCN/Weather、ModernTCN/ECL各8次默认/8次确定性；后来补入的TimeMixer/Weather各4次。此为特定形状算子非确定性的实测证据，并非仅引用文档或从小误差猜测；也不声称所有未来误差的唯一来源都已排除。

三组原数值阻塞均进行了6步串行参考、两路并发与独立串行重复。TimeMixer/Exchange全状态最大差1.4901161193847656e-7，指标差0；ModernTCN/Weather最大state差6.179045885801315e-5、最大标量差9.920845345234852e-8，接受固定state atol1e-4/metric atol1e-6、rtol0。TimeMixer/Exchange旧单张量1e-7规则及其失败保持历史事实，当前在后端条件成立后替换为全浮点范围。

ModernTCN/ECL初始state1e-4与绝对loss1e-6候选未通过：最大state约4.67e-4，raw loss约9.35e-5，独立串行也有差异；旧结果保留。依据已成立的后端条件及loss约11的实际量纲，在新复验前登记该域state绝对界1e-3、validation绝对界1e-6，loss采用abs(a-b)<=1e-6+1e-5*max(abs(a),abs(b))。新独立4worker复验通过：最大state4.5868754386901855e-4、validation3.166413207189578e-7、loss9.059906005859375e-5，最高loss allowance占比0.8019067079293531。不能把此后验工程修订写成原界Passed，或当作正式效果/长期稳定性证据。

本轮只读重审另外7个RSS组已存在的六步JSON，发现TimeMixer/Weather四H存在先前被RSS返回码遮住的exact差异，其他已取得并发JSON的RSS组未见该现象。它仍在原10个任务组内；追加2个独立串行H96诊断，算子重放同样证实默认非确定、确定性重复exact；state最大4.842877388000488e-6、validation3.9419564501486093e-7，接受state1e-4/validation1e-6、loss abs1e-6+rel1e-5。这里只完成串行重复对照，四路仍由补测检查，不能声称已验证该域并发。实例规划19补为21，但所有操作计数仍在先登记192/296/256及既有机械余额内。

初始模型/RNG/batch、非浮点状态、optimizer step与param-group结构始终exact；任何NaN/Inf拒绝。此前TimeMixer/ETTh1、TimeMixer/ECL、ModernTCN/ETTh1三域规则保持。其余模型不降低数值门槛。conditional-kernel-admission.json绑定四域证据SHA，未满足条件或证据字节变化则preflight拒绝。

### 23.4 验收、反传计数修复与复用边界

CPU两轮各16个已登记方法passed，无fail/error/skip；共32方法。第二轮核对了隔离梯度计数及ECL规则；之后TimeMixer/Weather为作用域/诊断重复次数的增量，采用两份实际guarded worker与静态依赖检查，不虚称第三轮16/16。全495个profile与初始逐项一致，update/evaluate/init_training/formal_worker/save/restore的AST保持；作者代码、原数据、AGENTS、Closed M4及不可变tag不变。

计账核对发现CH3既有bootstrap只包装autograd.backward，未包装autograd.grad，原3个隔离核验共48次grad调用未出现在budget.json。已保留原raw ledger并独立补账；钩子仅对本次授权的kernel诊断加计autograd.grad，新的ModernTCN/ECL首worker实测6 Adam/24 forward/22 backward，与操作逐项对应。没有删除或重写旧budget.json，也未将漏记视为免费。

本轮合计21个worker实际180 Adam/294 forward/252 backward，其中72次为独立卷积梯度重放；均在192/296/256上限内。机械余额288−180=108。初始ECL数值候选失败、首轮48次漏记事实及历史失败全部保留。无真实观测读取、无正式test计算或旧checkpoint读取；仅本轮自产raw sidecar用于比较。

### 23.5 44继承/10补测及预算纪律

最新父complete为`m5-three-scope-numeric-znYE96/probe/complete.json`，SHA=`4e9ceb09b6631b1086e3502915d88e836510ecd7f59f976be312c652ffaa49f2`。44组155代表的profile与父/祖先引用逐一核验；不能将父Blocked改Passed。重测仅余10组40代表，保留44组已获准结果，不重跑54组。

只使用594的既有补测余额，不追加126或其它额度。每组串行参考和一个可行候选共至多48 Adam，10组核心<=480；所有可选资源回退受总594硬上限和后续组核心预算约束。四路放不下时按已登记形状先两路，无论选哪种每个H仍只运行自身任务；发生模型/数值失败不能降并发掩盖。若继续探测并发会占用后续组核心额度，只保留已验证单路并标该候选未测，不把未测数值或并发说Passed。六步计算预算和模型/数据参数均不改。

### 23.6 closure及启动停止点

本轮完成实现/证据审核后只对八个已审核文件精确stage/commit/push。实际commit和SHA写closure-verification.json；不为记录自身commit递归改文档。closure成功且三端一致/0/0/clean后生成绑定当前code/protocol/环境/硬件、父证据、594余额和kernel证据的新probe-review.json，无负载preflight须blocked=[]。长时probe仍由用户一次启动；不代启，不冻结J、不关闭M5、不进入M6。计数/完整性审计通过不等于44/54历史技术gate追认全通过。

## 24. RSS/数值修订后10组补测完成：54 / 54 技术准入Passed

### 24.1 完整性与版本绑定

用户回复“结束了”后执行最终完成审计。`probe/progress.json`与`probe/complete.json`字节一致，SHA-256=`99c3a6c0d19e36a52189e2c499e3304aa7d89448cc6f5c90d3d48183b1cd5a6d`；54组/195 Q完整覆盖。报告protocol=`b415cec95ab03e547594cc39bda42992fc25c2b67c87759cb51cbb4f78ea471b`，与当前配置、code binding、环境、A800硬件及closure许可全部一致；`conditional-kernel-admission.json` SHA=`64b4e836ca20f57931f8cce714408d8acc09d654895536fd6a831005bbf5bd02`仍匹配四个条件作用域。完成时Git三端一致、0/0、clean；不可变tag仍`fa9665627e6fcfb1d0c2bc22d943ca9666304fd6`。无正式训练输出。

### 24.2 终态、成本与并发

最终54/54组均为Passed：44组来自已审核继承，10组为本轮新通过。并发决策分布为q4 44组、q2 3组、q1 7组。本轮10组实际80条worker轨迹，全部`finite=true`且各自6步/batch身份完整；52份controller/process记录均返回0、无failure且资源准入为true，80份受限audit日志无deny事件。实际消耗480 Adam/640 forward/480 backward，低于594硬上限，剩余114；无OOM、无NVML/外部PID归属失败，也未触发长窗追加worker。

10个本轮新Passed组为AMD/ECL、J/ETTh1、J/Exchange、PatchTST/Weather、PatchTST/ECL、iTransformer/ECL、TimeMixer/Weather、TimeMixer/Exchange、ModernTCN/Weather、ModernTCN/ECL。AMD/ECL、J/ETTh1、J/Exchange、PatchTST/Weather、PatchTST/ECL、iTransformer/ECL按exact短轨迹数值通过；其余四域按§23条件数值规则通过。

### 24.3 RSS复合判据实际表现

本轮所有新worker的`memory_review`均`blocked=false`且`needs_long_window=false`，因此没有在probe内启动24更新长窗。该结果并不表示取消长窗机制：短窗若同时达到连续增长、累计>32MiB和平均>1MiB/step仍会进入`NeedsLongWindow`，再按24步平台标准判断。此前三个24步代表的平台正例保留为规则验收证据；本轮新10组没有达到显著增长触发线。GPU allocated增长检查仍有效。

### 24.4 数值条件准入在实际四H补测中的结果

TimeMixer/Exchange q4四代表均通过当前全浮点条件规则，最大state绝对差`2.384185791015625e-07`；ModernTCN/Weather q4最大state差`7.723458111286163e-05`；ModernTCN/ECL按资源选择q2，最大state差`4.7803670167922974e-04`；TimeMixer/Weather q4最大state差`7.62939453125e-06`。四个作用域的非浮点/optimizer step/结构等exact残余状态全部满足当前policy，production comparator及报告验证均通过。这些是短轨迹并发工程准入，不代表完整训练轨迹逐位一致或效果等价证明。

### 24.5 当前裁决与M5停止点

因此，本轮**资源/并发/短轨迹数值技术gate = Passed（54/54）**。这只解决M5进入正式实验前的工程准入，不改写M4效果事实，也不是预测效果gate。J仍只是M5候选，没有因本次资源probe自动冻结；M5未Closed，M6未授权/未启动，495个正式run仍未执行。下一步若要冻结结构或进入M6，必须按用户新的明确决定执行，不能由本节自动推进。
