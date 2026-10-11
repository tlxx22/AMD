# M6-Timemixer implementation-r1 验收证据

状态：implementation自检通过；依据本次用户通知，首对U/V的ChatGPT implementation review **Passed**、281项CPU验收证据已审核（针对当前绑定作者行为，不代表原作者无缺陷）。本轮工作区迁移及16项新路径复验自检通过；迁移后字节的ChatGPT review **Pending**。Git closure **未执行**；真实开发实验 **未执行**；三个效果目标 **Not evaluated**。

本文件是执行证据索引。科学/结构合同只以canonical新专节及唯一M6 milestone为准，不创建另一份当前状态文档。来源逐项审计见[source-audit.md](source-audit.md)，运行源码绑定见`timemixer_dev/source_binding.json`，新增实现自身SHA另列于spec及外置逐文件manifest，不能拿旧B source hash冒充增强模型hash。

## 设备、基点与范围

设备`b82673a648b4`；本分支当前cwd/worktree=`/public/home/yueweiting/大论文/TimeMixer-U+V-v1`；branch=`M6-Timemixer`；base/当前HEAD=`5a1f34dbea9e906d3525f79b094b82efad2159b0`；Git common-dir=`/public/home/yueweiting/大论文/AMD/.git`。
当前工作区已统一为新路径，Git正式迁移记录见外置送审根`worktree-rename-r1/`。本说明改前原字节已保存到该目录的`before/`；原测试/失败日志仍记录实际执行时的路径，未作替换，也不把下列当前复现命令当作历史执行命令。
主仓库HEAD=`5341fbcb7c9f4f97658728d79b1af5487f7d38c3`；二轮=`df6a16403e10d51097db8c88829909c533d15652`；第三轮=`5a1f34dbea9e906d3525f79b094b82efad2159b0`；TimeMixer=`e24610583b36fdd8c76cc17a8df4e65759a5f460`。
不可变tag的tag object=`a32c522377ae6d8598bd173f29ac02e54d140b00`，peeled commit=`fa9665627e6fcfb1d0c2bc22d943ca9666304fd6`。

实际检查祖先指导文件（不存在的路径未冒称已加载），读取主仓库、生产第三轮和新worktree的AGENTS；新worktree tracked指导文件仅根`AGENTS.md`。用户本轮AGENTS指令补充生效：patch helper实际成功，无bwrap错误，无需fallback；PDF使用现有pypdf，没有尝试缺失命令/OCR。未修改AGENTS。
创建前所有保护仓库worktree/index干净；本地tracking和必要`ls-remote`分支与交接一致，新目录/本地及远端M6-Timemixer均不存在。只新增worktree/branch元数据，既有分支/tag未移动；新branch无upstream属正常，不push。

允许写入仅此worktree、外置参考`.../amd-execution-evidence/m6/M6-Timemixer-references-v1/`、外置送审`.../amd-execution-evidence/m6/M6-Timemixer-implementation-r1/`、两份指定`大论文/paper/` PDF。没有读取真实数据/旧checkpoint；初次回归真实路径缺文件检查不等于读到了真实文件。合成CSV/checkpoint全在此worktree下TemporaryDirectory，完成后清理，证据长期保存。

## 最终验收（完整同字节版本）

| 组 | 最终数 | 主要验证 |
|---|---:|---|
| A | 4 | 同seed原作者及off/off的key/shape/参数、eval、train RNG/输出/loss/梯度/Adam state及更新bitwise一致；U/V独立子流、启用no-op复制路径；实际旧adapter在独立解释器构造的B同参数 |
| B/U | 8 | target0/非零、ordered非连续aux及非法索引；MS其他通道逐元素不变；M diagonal/simultaneous/float64 reference；空来源/C1、常数与真实近零方差、null/core0、接收与来源历史响应、统计投影位置 |
| C/V | 6 | CI0联合latent/CI1 b再C视图；mean/last/abs-diff历史轴及Ls1；尺度softmax/权重和与边界、状态响应；均匀恢复求和/S1零修正、目标/全通道及float64代数；历史梯度与无诊断计算图缓存 |
| D | 244 | 240个四臂形状组合：Urban4C×4H×2marks×4臂=128；M3C×4P×2marks×4臂=96；EPF2L×2marks×4臂=16。另3域×3启用臂各3Adam step、norm缓存/一次denorm、目标/loss拒绝广播、ETTm1/2五维marks |
| E | 6 | 四臂合成checkpoint真实磁盘strict往返；开关/任务/fold/H/来源/实现等12类身份篡改及tensor shape拒绝；实际作者路径/命名空间恢复；spec无训练许可/覆盖新增实现；既有task/profile结构逐字段相同；继承S1缺陷对照 |
| 相关旧回归 | 13 | 5项fold scaler、3项target offset（含72格窗口计数）、5项native marks/旧adapter；明确合成CSV夹具，生产parser和原断言未改 |

最终268+13=**281项有效通过**。`acceptance-final-r2.log`为最终完整实现结果，`regression-synthetic-final.log`为最终相关旧回归；不把初轮或失败套件拼成Passed。
启用梯度验收的9个fixture各3步，共27个CPU Adam step；A组3对关闭等价fixture另有6个单步更新，因此最终完整套件实际33个Adam step。三次完整实现套件（266/268/268）累计99个CPU Adam step，均为合成数据，无正式训练信用。所有新增参数组在每步具有finite、有效非零梯度并发生更新；不要求作者已有unused参数或退化fixture每元素非零。单独V历史state梯度非零，不detach主干；norm hook证明输入缓存来自同一原窗口且denorm只一次。
float32为主；U/V关键逐接收/逐输出代数reference使用float64，`atol=rtol=1e-12`。关闭身份使用既有`tests.test_runner::_assert_nested_equal`逐项torch.equal，零差异；未调用并发容差政策。

环境：`/public/home/yueweiting/miniconda/envs/amd/bin/python`，Python3.11.15、torch2.0.1、numpy1.24.3、pypdf6.16.2。
所有测试均CUDA_VISIBLE_DEVICES=""、PYTHONDONTWRITEBYTECODE=1、Python -B、OMP/MKL/OPENBLAS及torch线程1；没有安装/升级依赖、GPU初始化/查询或自动discover。

当前验收索引：本轮只执行已存在的16项路径敏感测试，结果16/16通过、退出0、无skip；精确ID见外置`worktree-rename-r1/test-ids.json`，原日志见`path-tests.log`，环境/命令/退出码见`run-command.json`，结果见`test-result.json`。包括AIdentity的旧adapter子进程、EEngineering的profile/任务/作者命名空间/strict磁盘checkpoint，以及Urban C11/H3、M C7/P96、EPF L168带marks的四臂形状。未重跑完整281项，模型/测试/结构配置/资源脚本SHA不变，合成临时fixture正常清理，继承CI1 marks排列和S1 PDM限制原样保留。

当前路径复验命令（CPU合成，非正式训练入口）：

```bash
cd '/public/home/yueweiting/大论文/TimeMixer-U+V-v1'
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /public/home/yueweiting/miniconda/envs/amd/bin/python -B '/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-Timemixer-implementation-r1/worktree-rename-r1/run-path-checks.py'
```

完整套件的当前复现命令（以后必要时使用；本轮未执行，不因改名要求重跑）：

```bash
cd '/public/home/yueweiting/大论文/TimeMixer-U+V-v1'
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /public/home/yueweiting/miniconda/envs/amd/bin/python -B -m unittest tests.test_timemixer_uv -v
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /public/home/yueweiting/miniconda/envs/amd/bin/python -B -m tests.timemixer_uv_regression
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /public/home/yueweiting/miniconda/envs/amd/bin/python -B -m docs.evidence.M6.timemixer-v1.implementation-r1.resource_snapshot
```

## 失败、修复与复验

1. 正式PDF获取成功，首次pypdf全页提取后的sidecar写入因lone-surrogate `UnicodeEncodeError`失败；仅改写出方式为backslashreplace并重提31/19页。源PDF不变、无OCR/猜测。该异常为工具输出记录，此处转录；没有把失败当作完成审计。
2. 首次完整实现266/266通过，补充task工厂、S1继承限制及数据域拒绝后268/268通过；再加强近零方差float32 fixture（17附近1e-4扰动，避免1e-9被舍入成常数）和S1非零随机logit断言，最终完整268/268通过。三份原日志保留，没有实现测试失败。
3. 初次旧回归选择时误含两个依赖真实文件的入口：`test_target_transform_inverse_round_trip_numpy_and_torch`及TargetOffsetTests.setUpClass。新worktree无`data/UrbanEV/data/volume.csv`，在存在性检查即抛FileNotFoundError。`regression.log`记录10个实际测试、两个error（另一个class未启动），不能说10/10。没有链接/读取生产数据来消除错误。新增显式wrapper，用既有生成器造4344×275 CSV并让原production parser读取该fixture，原13项断言全部通过；不删除/skip断言、不放宽容差，不冒称真实数据回归。
4. 资源脚本首次line65的print列表表达式缺一个`]`，解释器报`SyntaxError: closing parenthesis ')' does not match opening parenthesis '['`，未执行模型。修正列表表达式后`resources-final.log`退出0，30个继承profile参数量及公式核验通过；异常原工具输出在此转录。

## 参数与静态资源

| 结构实例 | B | B+U | B+V | B+U+V |
|---|---:|---:|---:|---:|
| UrbanEV F4 L12/C11/D64/S3/P1 | 70,885 | 76,434 | 74,597 | 80,146 |
| ETTh1 L96/C7/D16/S4/P96 | 75,497 | 89,194 | 78,441 | 92,138 |
| Weather L96/C21/D16/S4/P720 | 219,249 | 232,946 | 232,177 | 245,874 |
| EPF-PJM L168/C3/D64/S4/P24 | 289,185 | 309,866 | 293,281 | 313,962 |

U新增参数=`97L+4385`；V=`48D+16S+16P+576`，30个metadata结构逐项实算核对，见[resources.json](resources.json)。不改变profile D/S/downsample/PDM/head。
主要张量：U e[b,C,32]、pair[b,R,C,32]、alpha[b,R,C+1,16]、core[b,R,16]；V state视图[b,R,Ls,D]、desc[b,R,S,3D]、pi[b,R,S,P]，原尺度预测stack[b,P,C,S]。
静态M/Weather/b128/P720例：U pair6.891MiB、alpha3.609MiB；V pi与原预测stack各29.531MiB。b128只是待审核的算术情景，不批准真实batch。未计autograd/临时拷贝/完整主干/优化器，不能当作峰值相加或GPU实测。M下U通道二次；未跑GPU probe/吞吐/并发准入，后续ChatGPT优先结合适用历史q4证据评估，不因缺新probe自动阻塞。

## 送审边界与保护

原CI1 marks.repeat排列限制和原PDM S1错误见来源审计，有作者源码和真实原B/增强对照，未修B、未冒称旧B为修后实现。结构设计没有改变；局部forecast删除的未来特征分支由spec明确拒绝打开，所有当前profile原本为0。
本轮未修改通用runner、data loader、旧baseline配置/许可/共享作者仓库；未创建训练队列或启动GPU/真实train/validation/test，未读旧checkpoint。生产观察仅/proc下该cwd的进程comm与脚本名，观察到第三轮入口仍在；未读命令token、未遥测GPU、未停止/重启。
首轮保护记录、变化清单、前后文档SHA及含新增文件完整patch保留于外置送审目录`/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-Timemixer-implementation-r1/`，根下已审核`complete.patch`、manifest及旧SHA清单不覆盖。本轮最新送审使用其`worktree-rename-r1/file-manifest.json`与`worktree-rename-r1/complete.patch`；迁移增量diff、前后SHA、旧名分类、保护复核和新路径复验日志同在该增量目录，不能混用新旧版SHA。完整patch排除外置包本身以免递归；manifest对当前worktree全部26个变更文件绑定。index保持空，HEAD仍为基点，无stage/commit/push/merge/cherry-pick。
训练前仍须统一二/三轮协议、主指标、同任务有效B来源review及并发/正式启动合同。用户已通知首对实现审核Passed；本轮迁移自检通过不等于ChatGPT已读取、同步或审核迁移后版本。
