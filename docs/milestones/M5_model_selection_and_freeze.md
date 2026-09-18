# M5：模型筛选与结构冻结

状态：**In Progress — 最新§10：八个独立作者原生模型均完成CUDA合成训练链smoke；Sonnet首次Adam API失败后唯一机械复验通过，合计保守扣账129次Adam调用。ETTh1受限loader固定13项首次全Passed，累计13/26。旧工具绑定阻塞已由本次授权的两个窄用途解除。TiDE Deferred，正式520 runs/最多5600 run-epochs按三包规划，未启动。旧筛选包不执行，正式结构配置/六域接入缺口仍在，J未冻结。**

开始日期：2026-09-16（UTC）。canonical：`docs/AMD_EV_Thesis_Final_Implementation_Plan_v2.1.md`，v2.1-R1。

2026-09-16阶段启动与§§1–9各轮提案/授权及旧工具Blocked保留历史时点。最新用户指令整体取代未执行的“来源准备＋解除测试阻塞”指令，仅按§10执行来源准备、限定工具增量和实测验收；不重复征询已确认参数/职责/消融范围，不复活旧四臂筛选。**正式训练、正式模型/runner/身份/早停/汇总接入、M6与最终结构冻结均未授权。** 空间本轮只有方案文档修订。

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
