# M6-Timemixer 首次来源审计（2026-10-11，北京时间）

先取得、解析并读取正式论文的方法段及下列真实作者调用链，再编写 U/V。
没有运行来源仓库安装/下载/训练入口，没有导入 SOFTS/Pathformer 作为本项目运行依赖。
原始 PDF、完整 shallow clone、pypdf 全页文本 sidecar 和 acquisition manifest 位于
`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-Timemixer-references-v1/`。
两个作者 checkout 和原 PDF 已去除写权限；完整第三方仓库不进入 AMD Git。
`paper/SOFTS-NeurIPS2024.pdf`、`paper/Pathformer-ICLR2024.pdf` 位于大论文根，复制字节一致。

## 正式来源与版本

| 来源 | 完整版本 / SHA-256 | 许可核验 |
|---|---|---|
| [SOFTS 正式 PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/754612bde73a8b65ad8743f1f6d8ddf6-Paper-Conference.pdf)，31页 | `1cd4341b064d41b6fca6f2c17a33b538b3110fce5f2bbba1a98198814f36e62e` | NeurIPS 2024公开论文；论文再分发许可未另行裁定，不能套用代码MIT |
| [SOFTS 源码](https://github.com/Secilia-Cxy/SOFTS/tree/f5d35fd7c3e716b6383ce6d3cc42c131e32c3c44) | `f5d35fd7c3e716b6383ce6d3cc42c131e32c3c44` | 根 LICENSE 明确 MIT，copyright 2024 Xu-Yang Chen |
| SOFTS `models/SOFTS.py` | `71cd0b1d72055ff71a313139bf1e9801351f7798c7b392f679a3befd7c2f16e1` | 只读参考，无源码复制 |
| [Pathformer 正式 PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/2be6705de7412adf107900add727a795-Paper-Conference.pdf)，19页 | `6c865054161f7e94fecc533b21dbc652818315ef3ac3c057644ba8464691609f` | ICLR 2024公开论文；论文再分发许可未另行裁定 |
| [Pathformer 源码](https://github.com/decisionintelligence/pathformer/tree/ea85d82932215e171357da47b3bc82d502344758) | `ea85d82932215e171357da47b3bc82d502344758` | tracked tree未找到LICENSE/LICENCE/COPYING/NOTICE；许可不明，无源码复制 |
| Pathformer `layers/AMS.py` | `4cd8462587e071f141d1ceae0371bcfc91d5c17f8418454969e517944c43883e` | 只读参考 |
| Pathformer `models/PathFormer.py` | `281136623b21804d0236b0caa5757f69e9e7aba00cfa472c7c1ef1d7625c3920` | 实际文件大小写为 PathFormer.py |
| TimeMixer真实backbone | `e24610583b36fdd8c76cc17a8df4e65759a5f460`；入口 `817d62f4aac54c8566e560f6d3785856e31c8ee51460279ee3d0a4823f11d4be` | Apache-2.0；本地forecast局部适配已注明来源 |

完整依赖文件SHA见 `timemixer_dev/source_binding.json` 与外置 `source-manifest.json`。
绑定完整commit而非裸main；shallow clone不代表复核了全部历史版本。

## SOFTS：论文 → 实际调用链 → U

核对PDF §3.1/3.2，式(1)–(5)，PDF页3–5；Appendix B.1 Algorithm 1和B.2式(6)–(8)，页14–16。
真实调用链：`run.py → exp/exp_long_term_forecasting.py::_build_model/模型调用 → models/SOFTS.py::Model.forward/forecast → DataEmbedding_inverted → Encoder → EncoderLayer → STAR.forward → gen1/gen2 → pooling → gen3/gen4`。
读取 `layers/Embed.py`、`layers/Transformer_EncDec.py`；STAR返回`(output,None)`，残差由EncoderLayer在STAR外执行，随后LN和conv FFN；不能把孤立STAR输出误当成论文完整残差层。

| 机制 | 论文/作者代码事实 | 本项目U合同 |
|---|---|---|
| 每变量历史表示 | [b,C,L]沿L嵌入为[b,C,d]；marks可额外拼成token | 保留每变量表示；`LN(Linear_z(Z0)) + Linear_a([mu,log1p(sigma)])`，统计项在LN外；不移植marks token |
| 值投影 | 两层GELU MLP得[b,C,d_core] | 保留 `V2(GELU(V1(e_j)))`，32→32→16 |
| 核心聚合 | 全局单一core；每core维沿C softmax；训练multinomial抽取、eval加权求和 | 改为接收r条件化成对打分 `Ao(GELU(Aq(e_r)+Ak(e_j)+bA))`；训练/eval均稠密masked softmax |
| 来源与空值 | 所有通道参与，全局core无接收者mask/null | MS显式ordered aux且排target；M diagonal mask；固定0 value/0 logit null，空来源直接旁路 |
| 分发融合 | Repeat_Concat到每变量，再gen3/gen4；外层residual | 接收core→sigmoid历史gate，`delta=gate*Linear_out(core)`；无bias保证core0→delta0；`z'=z+gamma*delta`，gamma初始0.01 |
| 插入/主干 | 多层STAR、SOFTS embedding/head及独立norm | 全部删除；仅TimeMixer最细尺度既有norm后、CI reshape前；其余尺度不改 |
| 复杂度 | 原STAR通道线性 | U有[b,R,C,32]成对中间量；M为通道二次，不能继承线性宣传 |

## Pathformer：论文 → 实际调用链 → V

核对PDF §3.1/3.2，Figure 2/3，式(1)–(7)，PDF页3–6。
真实调用链：`run.py → exp/exp_main.py::_build_model/模型调用 → models/PathFormer.py::Model.forward → optional RevIN → start_fc → AMS_lists → AMS.forward → seasonality_and_trend_decompose → noisy_top_k_gating → SparseDispatcher.dispatch → Transformer_Layer.forward（patch division / intra+inter attention / FFN）→ SparseDispatcher.combine → residual → projection → denorm`。
读取 `layers/Layer.py`、`layers/Embedding.py`相关入口及`utils/Other.py`的Dispatcher/分解实现；训练loss另加balance_loss。
作者模型中[b,L,C]先变[b,L,C,D]；router实际取latent第0分量，先做趋势和Fourier，再start_linear将C→1，以L→experts的线性层生成每样本路由。

| 机制 | 论文/作者代码事实 | 本项目V合同 |
|---|---|---|
| 多尺度来源 | 不同patch size专家Transformer，dual attention | 不移植；只读TimeMixer最后PDM状态及原FMM每尺度输出，原D/S/PDM/head不变 |
| 输入条件化 | 历史动态经额外季节/趋势分解驱动router | 保留思想，改读既有PDM的mean/last/mean(abs(diff))；MS为联合latent，M恢复b,C轴，不detach |
| 状态投影 | router按样本生成expert logits | `LN(GELU(Linear_h([m,last,q])))+eta_s`；共享层，尺度eta初始0；拼尺度均值，经f/p输出每位置P logit |
| 选择 | train noisy top-k；eval clean top-k；balance loss | 全部删除；沿S稠密softmax得到[b,R,S,P]，无noise/top-k/辅助loss |
| 融合 | 论文加权专家融合；实际Dispatcher对expert output exp后加权再log | 不移植Dispatcher；`Y_base=sum_s(Y_s)`；加`0.5*sum_s((S*pi-1)*Y_s)`；保留原CI=0趋势残差 |
| 权重及归一化 | 作者规范归一化及专家空间输出 | `w=.5+.5*S*pi`、sum(w)=S；均匀pi恢复原求和；S1修正0；只在求和及修正后原denorm一次 |
| 预测位置 | 原模型projection统一预测 | V共享Linear_p输出P个位置logit，无horizon embedding；Urban label_horizon不输入，P恒1 |

发现并报告的论文/代码差异：论文式(5)先从X减seasonality再求trend；代码trend_model直接读x。论文式(6)/(7)描述全softmax再top-k，代码对选中top-k logits重新softmax；代码combine为exp/log域，不能描述为普通线性求和复现。上述均是本合同明确删除的机制，不改变本轮科学设计。

## TimeMixer继承限制和局部复制对照

原forecast多尺度→逐尺度norm→CI展开/marks.repeat→pre_enc→embedding→最后PDM→FMM（含CI=0 out_res）→尺度求和→denorm。
`model.py::forecast`只加入最细norm后U、FMM后V两处逻辑；future-feature原分支在当前profile恒关闭，结构spec拒绝打开；off/off直接super调用。没有全局monkeypatch、永久重定向models/utils路径或改变原参数key。

两个继承限制照实保留：

- CI=1变量展开按b再C，作者marks用`repeat(C,1,1)`按C再b重复；b>1且各样本marks不同时存在排列限制。本轮保留原语义并用no-op插入与原B bitwise对照，不修B后声称能配对旧结果。未来如修须独立身份及重新基线裁决。
- 原PDM的season/trend mixing无条件读取第二尺度，S=1真实主干会IndexError；现有profile为S3/S4。本轮单独证明V在S1修正0，同时原B和增强入口仍同样拒绝执行该继承缺陷路径，不声称修好了原主干。

读取/解析问题：首次pypdf提取成功但写UTF-8 sidecar遇lone-surrogate `UnicodeEncodeError`；随后只在写sidecar时使用backslashreplace保留可审计转义，重新提取全部31/19页。未OCR、未strings、未猜测缺失公式。没有来源获取失败，没有模型验收失败。论文实验表不为本候选提供任何效果信用。
