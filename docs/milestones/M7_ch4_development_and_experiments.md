# 大论文 07｜M7 第四章开发与实验

状态：**Planning：组织方式已定，第四章科学技术方案未定。** 本轮仅修正文档；未选择/下载模型或论文，未实现模型、图、状态wrapper或数据处理，未生成AUTH/任务队列，未创建M7 worktree，未执行训练/推理/test/checkpoint读取/GPU操作。新字节统一待ChatGPT审核。

## 1. 已定组织方式

唯一权威合同见[canonical](../AMD_EV_Thesis_Final_Implementation_Plan_v2.1.md)现行§0、§1.2、§4.2–4.3、§10–21。本记录统一承担第四章研究选型、获批baseline比较、候选开发/迭代、消融、诊断和以后批准的补充实验，不逐模块切milestone。

先研究、比较和讨论，确认本批baseline比较协议与范围，再完成获批比较；随后对选定且获批的backbone开展有依据的改进与消融。baseline-first不是已有模型名单或训练授权，也不要求未选型时先创建接口或强制提交完整开训矩阵。组织流程不是某种技术路线的隐性前置条件。

旧M7–M11工作合并到新M7；旧M12职责归“大论文 08｜M8 论文正文、图表与结果分析”，旧M13归“大论文 09｜M9 终稿审校、复现材料与答辩”。M0–M6历史编号、名称和产物保持，不建M7-A/B或第二份M7。

## 2. 尚未选择的科学事项

以下全部未确定，不填默认、首选或隐性约束，留待新会话研究和用户决定：

| 事项 | 待研究内容 |
|---|---|
| backbone与baseline | 模型身份、名单、比较适用范围 |
| 新增模块 | 身份、数量、来源、插入/替换方式和公式 |
| 信息交互 | 时间与空间信息是否及如何交互；串联、并联、残差、不同层融合或端到端联合均未选 |
| 第三章资产 | 是否使用模型、权重、表示或预测以及如何使用；不是自动指定的S0、time encoder或预测旁路 |
| 图与接口 | 是否需要图、Graph Mode、固定/自适应邻接、状态提取或StateAdapter；全部维度、状态宽度与关闭分支行为 |
| 数据与协议 | 数据集/城市/任务/L/H/P/输入/目标/划分/fold/seed/损失/指标/超参数/训练冻结或联合策略/best选择/test访问政策 |
| 复现、预算与标准 | 原论文复现是否需要及范围，训练/算力预算、定量效果标准/停止线/正式采用标准、比较与消融之外的补充实验 |

当前没有架构公式、接口需求或批准矩阵。已排除的旧技术规划不列为现行方案、候选池或优先项，也不借来源名称恢复资格。一般数据完整性、公平性、无泄漏和真实成本原则保持，但不能据此预设架构或参数。

## 3. 可查询的第三章资产

[第三章实验索引](../CH3_Experiments_and_Data_Index.md)的唯一CURRENT_ADOPTED_RESULTS是当前采用入口；[M6原记录](M6_formal_experiments.md)保存历史实验和审核。集中包为`/public/home/yueweiting/大论文/amd-execution-evidence/m6/CH3_Results_Consolidated_v10_r1`，七类数据归集/完整性/采用口径review Passed，不升级为完整state/Adam/RNG/CUDA恢复。

第三章已完成模型、结果、环境与工具保留；v10普通时间执行事实见canonical §8.0，仅作资产说明。是否复用须后续选择适用，不自动成为第四章组件、监督或参数。Goal1/2 Met、Goal3 Not met及单seed/test开发反馈边界保持，M6暂停新增实验但非正式Closed。

C内README/handoff和旧文档中的第四章规划是组织与技术路线均过时的快照，不约束M7。封存字节、原结果与review保持，不重开第三章科学审核或重跑归集。

## 4. 下一会话：研究选型与讨论

第一步是研究问题、比较可能方案并与用户讨论，不直接创建graph wrapper、状态适配器或训练矩阵。科学事项按第2节集中讨论；选定适用方案后再确认本批比较协议与范围，落实获批工作。不把历史论文、旧代码可用性或静态检查Passed当作选型授权，不凭空补齐未定公式、模块数量或预算。

本轮纠正上一版文档擅自继承的技术假设，不表述为用户更换已批准backbone或路线。上一包真实静态检查与报告保留，但不作为错误科学条款的放行依据。未定标准不能见结果后伪称预登记，不保证必胜、不无限追加或择优重跑。

## 5. 本轮边界与送审

工作区仍为`/public/home/yueweiting/大论文/ModernTCN-U+V`，分支`M6-ModernTCN-UV-v1`。本轮起点HEAD/tracking/实际远端`16e45afe658bc5fdf0ec1ab2801fc893b7ed3866`、0/0；8路径unstaged/index空，不是clean，不能当作未来M7 parent。不可变baseline tag保持原目标。

证据：`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-M7-open-design-contract-correction-r1`，仅本轮before/diff/文件SHA与文档检查。原三份repair、归集包、旧证据和执行代码不改，无模型测试或正式负载，无stage/commit/push。此次新字节审核通过后，再与原已审repair/归集一次精确closure，随后新会话研究选型；不先提交错误方案或为文字纠偏追加实验。

Project injection与聊天交接由ChatGPT重新交付、用户替换；本工具未修改或同步Project设置。会话标题保持“大论文 07｜M7 第四章开发与实验”。
