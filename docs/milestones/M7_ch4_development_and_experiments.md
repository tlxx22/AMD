# 大论文 07｜M7 第四章开发与实验

状态：**Planning：完整ModernTCN-U+V v10时间基础已定，新增空间／时空方案待定。** 本轮仅修正文档；未选择/下载模型或论文，未实现模型、图、状态wrapper或数据处理，未生成AUTH/任务队列，未创建M7 worktree，未执行训练/推理/test/checkpoint读取/GPU操作。新字节统一待ChatGPT审核。

## 1. 既定时间基础与组织方式

唯一权威合同见[canonical](../AMD_EV_Thesis_Final_Implementation_Plan_v2.1.md)现行§0、§1.2、§4.2–4.3、§8.0及§10–21。第四章以第三章已经完成的完整ModernTCN-U+V v10为时间建模基础，继续研究引入空间信息的扩展。U、V成果及第三章默认G5/R4基础身份已定，不是重新待选的时间模型，也不降格为可有可无的对比baseline；未经批准不得退回裸ModernTCN、删U/V或更换时间骨干。

本记录统一承担第四章开发实验，不逐模块切milestone。后续确认适用比较协议与范围、完成获批baseline比较，再在选定空间方案和既定时间基础上迭代改进与消融/诊断。比较baseline可有各自原生时间组件，不改变我们的改进模型以v10为基础。这是后续工作方式，当前接收仅限背景阅读、只读核验和理解回执；具体研究与执行任务由用户随后明确下发。

旧M7–M11工作合并到新M7；旧M12职责归“大论文 08｜M8 论文正文、图表与结果分析”，旧M13归“大论文 09｜M9 终稿审校、复现材料与答辩”。M0–M6历史编号、名称和产物保持，不建M7-A/B或第二份M7。

## 2. 待定的空间方案、具体使用方式与实验合同

以下事项待后续研究与用户决定；它们不包含重新选择第三章时间基础：

| 事项 | 待定内容 |
|---|---|
| 新增空间／时空backbone或框架 | 新增建模部分及其与v10的组织方式；不重选时间骨干 |
| 新增模块 | 身份、数量、来源、插入/替换方式和公式 |
| 信息交互 | 时间—空间信息在什么层、怎样交互；残差/串联/并联/分层融合/联合建模均未选 |
| 具体权重与训练 | 是否加载某个预训练checkpoint、冻结/微调/联合训练、表示暴露、原head保留或适配、数据域适配；采用v10基础本身已定 |
| 图与接口 | 是否需要图、邻接、Graph Mode、StateAdapter及状态维度等；不预先实现 |
| 实验合同 | baseline名单、首批数据任务、L/H/P/输入输出/划分、seed/超参数/损失/评价/best选择/test边界、资源预算与定量标准 |
| 复现与补充实验 | 是否需要及其范围/额度、比较与消融之外的实验 |

以v10为基础不等于空间残差终式、固定S0实验身份、独立预测旁路或某个图接口。原权重/数据/超参数/test经历不自动成为第四章完整协议。已排除的空间骨干/模块不恢复候选资格，也不借来源名称恢复；模块数量未定。若未来需要时间实现调整，必须给出明确方案/范围，保留第三章原代码和产物，本轮没有调整授权。

## 3. 可查询的第三章资产

[第三章实验索引](../CH3_Experiments_and_Data_Index.md)的唯一CURRENT_ADOPTED_RESULTS是当前采用入口；[M6原记录](M6_formal_experiments.md)保存历史实验和审核。集中包为`/public/home/yueweiting/大论文/amd-execution-evidence/m6/CH3_Results_Consolidated_v10_r1`，七类数据归集/完整性/采用口径review Passed，不升级为完整state/Adam/RNG/CUDA恢复。

第三章模型、结果、环境与工具保留，完整v10已是第四章时间基础。普通时间执行事实见canonical §8.0，用于后续接口论证；具体checkpoint、表示/输出、训练和域适配仍待方案明确，不把尚未实现的state API写成现成接口。Goal1/2 Met、Goal3 Not met及单seed/test开发反馈边界保持，M6暂停新增实验但非正式Closed。

C内README/handoff和旧文档中的旧第四章规划不作为现行空间方案指令，也不能反向否定当前已明确的v10基础。封存字节、原结果与review保持，不重开第三章科学审核或重跑归集。

## 4. 当前接收边界：背景阅读与只读承接

ChatGPT第四章会话及Codex第四章会话均先完成背景阅读、只读核对实际Git/现行合同和理解回执。收到交接消息不自动开展研究选型，更不直接实现graph wrapper、状态适配器、模型或训练矩阵。具体研究和执行任务由用户随后明确下发；后续方案基于第1节已明确的时间基础和第2节待定空间内容讨论。

此前将“是否复用第三章模型”整体列为未定，是ChatGPT扩大未定范围的错误。本次恢复论文既有立意，不登记为用户更换时间模型路线；旧空间路线也不因此恢复。过去报告与审核保留真实时点，当前科学边界以本条修订为准，不倒改原报告。

## 5. 本轮起点、保护与送审

工作区`/public/home/yueweiting/大论文/ModernTCN-U+V`、分支`M6-ModernTCN-UV-v1`保持。原8路径closure已完成：实际HEAD/tracking/实时远端`24c45c9cb907148ac78cd438b4c953e6b244f0ac`，0/0，起点worktree/index clean；其父为16e45afe658bc5fdf0ec1ab2801fc893b7ed3866。原closure报告及三文件repair不修改，不重做或回退旧收口。该commit是本轮实际起点，不预填未来M7 worktree parent或训练commit。

本轮证据：`/public/home/yueweiting/大论文/amd-execution-evidence/m7/M7-v10-foundation-contract-alignment-r1`。仅五份既有文档新修改，before/diff/SHA和静态检查统一待审；无stage/commit/push。原数据/模型/配置/测试/环境/封存证据保持；未创建worktree、AUTH、队列/attempt，未研究/选择/下载模型论文，未实现接口或执行模型/训练/test/checkpoint/GPU。本轮新字节须另经审核，不能沿用原8路径许可提交。

### 5.H 原closure之前的静态记录（历史时点，不是当前Git状态）

工作区仍为`/public/home/yueweiting/大论文/ModernTCN-U+V`，分支`M6-ModernTCN-UV-v1`。本轮起点HEAD/tracking/实际远端`16e45afe658bc5fdf0ec1ab2801fc893b7ed3866`、0/0；8路径unstaged/index空，不是clean，不能当作未来M7 parent。不可变baseline tag保持原目标。

以上旧时点由已完成的24c45c9c…closure承接，保留其事实，不重放其任务；本轮实际状态以前文和新外置回执为准。

Project injection维持项目级通用规则；本任务不复制injection到服务器，不修改Project设置，也不重写用户的会话首条消息。会话标题保持“大论文 07｜M7 第四章开发与实验”。
