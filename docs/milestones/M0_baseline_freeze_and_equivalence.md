# M0：基准冻结与增强空壳等价性

状态：Closed
baseline tag：`amd_reproduced_baseline_v1` → `fa9665627e6fcfb1d0c2bc22d943ca9666304fd6`
M0-A commit：`1f76f7904198297beade6c5ad2b34fc917281624`
M0-B commit：`7ba48697a417f8b517e03d16ed7493b7bbebea27`
G0 commit：`2ca729f7454ec4634c2e4d7c90a4a3eb08984128`
开始日期：2026-08-15（UTC）
结束日期：2026-08-15（UTC）

## 1. 阶段目标

M0 的目标是冻结唯一可执行 AMD 基准、审计其来源和 smoke artifact，建立不改变预测行为的最小 `AMDEnhanced` 空壳，并以数值等价和完整回归关闭 G0。M0 不实现 UrbanEV DataLoader、PMCR、TEB、`H_time`、`StateProjection` 或任何空间模块，也不进入 M1。

阶段边界：

- M0-A 冻结 tag、完成累计 upstream diff、artifact/环境/数据/工作区审计并闭合文档与备份；
- M0-B 仅增加 pass-through `AMDEnhanced` 和 `return_state_source`；
- G0 对 baseline/current 的 LayerNorm 已知失败做定性，维持 AMD/AMDEnhanced 的 `1e-6` 等价门槛，并要求完整回归通过；
- 本报告合并原 `baseline_audit.md` 与 `m0a_closure.md` 的有效内容。重复 closure 文件不再保留，机器证据统一位于 `docs/evidence/M0/`。

## 2. M0-A：基准冻结与审计

### 2.1 结论摘要

1. 唯一正式 baseline tag 为 amd_reproduced_baseline_v1，已在本地和 origin 验证；它解析到完整提交 fa9665627e6fcfb1d0c2bc22d943ca9666304fd6。没有创建、移动、覆盖或 force-update 任何同义 tag。
2. 当前开发分支保持 AMD-paper-repro-custom-modules-v1。
3. docs/AMD_EV_Thesis_Final_Implementation_Plan_v2.1.md 已成为唯一权威方案；旧稿保留在 docs/archive/plans/ 并明确失效，不参与实现决策。
4. 当前 models/tsAMD.py 的真实 paper-close forward 是 MDM 输出同时进入 DDI 主干和 AMS selector：u_mdm = MDM(x_ch)，v = DDI(u_mdm)，pred = AMS(v, u_mdm)。不存在 h = x_ch。
5. 相对 upstream/main 的审计是完整累计 diff，不是只看最后一个提交。upstream/main 是当前基准的直接祖先；当前基准领先 4 个提交、落后 0 个。
6. 已定位的 ETTm1 四个 horizon、seed 2024 运行归类为“完整、可追踪的 M0 冒烟复现产物候选”。它不是第三章正式 AMD baseline。
7. 本轮没有重跑训练，没有重组历史 artifact，没有修改任何 Python 模型、训练脚本或测试代码，也没有进入 M0-B/M1。

### 2.2 Git 基准、tag 与分支

| 项目 | 审计值 |
|---|---|
| 开发分支 | AMD-paper-repro-custom-modules-v1 |
| branch HEAD | fa9665627e6fcfb1d0c2bc22d943ca9666304fd6 |
| origin 分支关系 | +0/-0 |
| 唯一正式 tag | amd_reproduced_baseline_v1 |
| tag 类型 | annotated tag |
| tag object | a32c522377ae6d8598bd173f29ac02e54d140b00 |
| tag peeled commit | fa9665627e6fcfb1d0c2bc22d943ca9666304fd6 |
| origin tag 状态 | 已推送；远端 peeled commit 与目标完整 SHA 一致 |
| upstream/main | 000d377a1ed8946aa817ff357cdf1de64b99abb9 |
| merge-base | 000d377a1ed8946aa817ff357cdf1de64b99abb9 |
| baseline 相对 upstream | ahead 4 / behind 0 |

执行前本地和 origin 均不存在该正式 tag，因此创建 annotated tag 后只推送：

~~~text
refs/tags/amd_reproduced_baseline_v1
~~~

没有创建 amd-paper-baseline-v1 或其他同义 tag；没有覆盖、移动或强制更新现有 tag。

### 2.3 唯一权威方案与归档状态

唯一权威方案：

~~~text
docs/AMD_EV_Thesis_Final_Implementation_Plan_v2.1.md
~~~

replacement 源文件保留在仓库外一层，未移动、未删除：

~~~text
/public/home/yueweiting/大论文/AMD_EV_Thesis_Final_Implementation_Plan_v2.1_replacement.md
~~~

最终状态核验时曾在 Git 顶层发现一个同名 untracked 副本。M0-A 收尾获得明确授权后，已将该重复副本移出仓库到：

~~~text
/public/home/yueweiting/大论文/m0a_external_backups/document_sources/AMD_EV_Thesis_Final_Implementation_Plan_v2.1_replacement_repo_duplicate_20260815.md
~~~

父目录源文件继续保留。移动前后两份内容大小相同、SHA-256 相同，没有覆盖现有文件。

文档指纹：

| 文件 | SHA-256 |
|---|---|
| 父目录 replacement 源文件 | 1f952f13f50b5717e9fa1dda4b1a1436414c2f2765aaf0a1a0a3621b28be2dd5 |
| 已移出仓库的重复副本 | 1f952f13f50b5717e9fa1dda4b1a1436414c2f2765aaf0a1a0a3621b28be2dd5 |
| canonical v2.1（M0-A closure 快照） | de89194462d88a8f9d837bf8e48ee04f27142e263cbf701660c7c6bd5da118b7 |
| canonical v2.1（G0 合同/路径核对后） | 36b443f200bc1361614a0c990ae40ffe3c75feccfaf530bb6dde5ac4ad5e6858 |
| 归档 v2.0 | beddb865a84cdf5d178800d8f859c9add29adbd25c184165cf71b0e3031e1780 |
| 归档早期方案 | af364a1fa0b62f753b88e7792c5c801781f03f0fbbc0b15c9fe0e59d618343f2 |

canonical 首先按 replacement 逐字节复制，复制后两者哈希完全相同；随后只按本轮明确要求做了以下 M0-A 文档修订，因此最终哈希不同：

- 归档目录写法由 AMD/docs/archive/ 统一为仓库内 docs/archive/。
- artifact 路径加入 task_mode。
- resume 核验字段加入 task_mode 和 run_id。
- 实验登记字段加入 task_mode。
- 文末隔离合同加入 task_mode。

两份旧方案最终统一保留在 docs/archive/plans/，其归档/失效声明没有删除；只把其中的唯一权威路径修正为 docs/AMD_EV_Thesis_Final_Implementation_Plan_v2.1.md。

在 /public/home/yueweiting/大论文 全范围和 /public/home/yueweiting 最大六层深度内未定位到用户提到的 DOCX。本轮只记录“未定位”，没有推断、移动或删除任何 DOCX。

### 2.4 冻结基准真实 forward

当前真实实现位于 models/tsAMD.py。忽略变量命名差异后的数据流为：

~~~text
x_norm = RevIN_norm(x)             # norm=True 时
x_ch = transpose(x_norm, 1, 2)
u_mdm = MDM(x_ch)
v = u_mdm
for DDI block:
    v = DDI(v)
pred_all_norm, moe_loss = AMS(v, u_mdm)
pred_all = RevIN_denorm(pred_all_norm)
pred = optional_target_slice(pred_all)
return pred, moe_loss
~~~

源码中实际局部变量名为 u 和 ddi_output，但语义严格等价于：

~~~text
u_mdm = MDM(x_ch)
v = u_mdm
v = DDI(v)
pred = AMS(v, u_mdm)
~~~

关键结论：

- DDI 输入是 MDM 输出，不是原始 x_ch。
- AMS 第一个输入是 DDI 后的表示，第二个 selector/time embedding 输入是原始 u_mdm。
- h = x_ch 不是当前实现，也不得作为增强模型设计依据。
- M0-A 冻结点的 AMD 还没有 PMCR、TEB、`return_state_source` 或 AMDEnhanced；M0-A 没有添加这些实现。
- M0-A 已锁定：未来 M0-B 在 TEB 尚未实现时，`exo_context` 必须初始化为 dtype/device 正确的固定零向量；M0-B 随后按此合同实现，但不生成 `H_time`。

关键源码现场 SHA-256：

| 文件 | SHA-256 |
|---|---|
| models/tsAMD.py | fa72cdbe34348364344c0d9c0755668a82d22f6a37ee061c7ece93ecfaf90ba1 |
| models/common.py | 570f47c3a7db3b5156e4e95df65b81aa13c5a0a741a61f1bb0798ab1ec1a3afb |
| models/tsmoe.py | d6c7888410dc64c3514c76cf4f2720b99c11773b0011780afaac76ca98aee0f1 |
| main.py | 4e7df0fd2eb246d8aeea86a50d484378cc5102ae48817dd6665b45fcde5bb13c |
| utils/dataloader.py | 27b68d051180d14f0399181f12e802d07cd567adafd5e24eb5370af58bccc87c |
| scripts/ETTm1.sh | 986ec89016c7e74a03c2ef86f9cf150b73b179be5df15602df9eef88084cba7f |
| summarize_results.py | e7b2226e87d7958538abd28bbef927a570818b4a6023f3aab26c515075a6a037 |

### 2.5 相对 upstream/main 的累计 diff

权威比较范围：

~~~text
000d377a1ed8946aa817ff357cdf1de64b99abb9
..
fa9665627e6fcfb1d0c2bc22d943ca9666304fd6
~~~

merge-base 就是 upstream/main，故不存在当前基准尚未包含的 upstream 独立提交。当前基准累计包含 4 个提交：

| 顺序 | Commit | 说明 |
|---:|---|---|
| 1 | e6eed2e36b3d42788ce72c47eaa9c2da429b8627 | Implement AMD MDM-U-to-DDI experiment variant |
| 2 | 68c2afdbfce4cb2f67898f2e7221bda0f4d3a351 | 文档 |
| 3 | 5a718d5309b5bec628047a11c3859a5ae0a5805a | Implement paper-close AMD normalization and DDI variant |
| 4 | fa9665627e6fcfb1d0c2bc22d943ca9666304fd6 | Normalize repository text files to LF |

累计统计：

| 统计口径 | 文件数 | 新增 | 删除 |
|---|---:|---:|---:|
| 原始 Git diff | 28 | 5228 | 970 |
| 忽略行尾空白 | 28 | 4778 | 520 |

最后一个提交对 common.py、tsAMD.py、tsmoe.py 做了 LF 统一，原始统计含行尾噪声；忽略行尾后仍有大量实质变更。完整 patch 证据：

~~~text
docs/evidence/M0/upstream-main-000d377_to_fa96656.full.patch
SHA-256: 790d4455f87991aed61521626245196f06e3c23c45dcb71b27366882907f003f
~~~

#### 2.5.1 models/tsAMD.py

upstream 数据流是 MDM(x) 只作为 AMS time embedding，而 DDI 仍消费原始 x。paper-close 基准改为：

~~~text
u = MDM(x)
v = DDI(u)
pred = AMS(v, u)
~~~

还增加了输入形状、超参数、空 batch 及 BatchNorm 单样本训练检查。

#### 2.5.2 models/common.py

实质变化：

- upstream 的 layernorm=True 实际是展平后的 BatchNorm1d(seq_len × feature_num)。
- 当前基准改为在 [batch, channel, sequence] 最后一维使用 LayerNorm(seq_len)，并用于 MDM、DDI 入口。
- DDI 内部 norm1/norm2 仍保留公开实现的 BatchNorm1d。
- alpha > 0 时 DDI hidden width 使用 max(32, 2^ceil(log2(feature_count)))。

因此当前 paper-close 基准不是未经修改的 upstream 公开代码。

#### 2.5.3 models/tsmoe.py

- 增加输入、shape、dtype、device、Top-K 与超参数检查。
- 输出初始化改为 x.new_zeros，使 dtype/device 正确继承。
- AMS selector/expert 的核心结构未重新设计。

#### 2.5.4 main.py

runner 的实质变化包括：

- 修复 CLI 布尔解析。
- device 显式化；正式脚本固定 cuda:0。
- Adam weight decay 从 1e-9 改为 1e-7。
- val/test 指标由 batch mean 再平均，改为全元素 SSE/SAE 聚合后的全局 MSE/MAE。
- 最后一轮不再覆盖更早的 best；最终测试从磁盘加载严格按 validation MSE 选出的 best.pt。
- 新增源码、数据、环境、预处理和科学配置指纹。
- 新增原子 JSON/checkpoint/history、run lock、失败 manifest 和 epoch 级 resume。
- upstream 中未实际写结果的 result_path 被弃用，正式结果改为 metrics.json。

#### 2.5.5 utils/dataloader.py

保持的协议：

- scaler 只用训练集拟合。
- ETT 固定 train/val/test 边界不变。
- val/test 保留前序 seq_len 上下文，预测目标仍位于各自 split。

实质变化：

- reader/split 由 dataset_id 决定，配置错误可能选择错误协议。
- 修复 Solar headerless 首行丢失。
- validation 改为 shuffle=False、drop_last=False。
- train 保持 shuffle 与 drop_last=True，但加入显式 Generator 以支持精确 resume。
- 补全目标列解析、窗口计数、inverse transform 与 preprocessing metadata。
- 增加 NaN/Inf、列、长度和空 split 等早期检查。

#### 2.5.6 scripts/ 与 artifact/result

9 个公开脚本统一增加严格 shell 选项、project-root 解析、PYTHON_BIN、SEEDS、多 seed、显式 variant/dataset_id/device/artifact root/weight decay，并停止写 legacy result.csv/checkpoints。

upstream 只有单一 best.pt，缺少结构化配置、指纹、manifest、history、安全 resume 和可信汇总。当前基准使用：

~~~text
artifacts/
  AMD-paper-norm-wd-ddi-v1/
    dataset/
      sl<seq>_pl<pred>/
        seed<seed>/
          run_id/
            manifest.json
            config.resolved.json
            best.pt
            last.pt
            history.jsonl
            metrics.json
~~~

新增 summarizer 只读取 completed 且内部一致的 run，并拒绝为同一科学配置/seed 的多个成功 run静默择一。

风险：当前目录结构缺少显式 task_mode、target、fold 层级，horizon 还和 seq_len 合并为 sl*_pl*；它不能直接作为 v2.1 最终合同。本轮只记录映射，不迁移历史产物。

### 2.6 ETTm1 smoke-reproduction artifact 审计

#### 2.6.1 分类

审计分类：

**完整、可追踪的 M0 冒烟复现产物候选。**

明确边界：

- 它不是第三章正式 AMD baseline。
- 它不是“只有部分日志的初步复现证据”。
- 它也不是“来源无法确认的不完整产物”。
- 完整性结论不等于正式实验范围完整；当前只有 ETTm1 和 seed 2024。

产物根目录：

~~~text
artifacts/AMD-paper-norm-wd-ddi-v1/ETTm1
~~~

每个 horizon 都有完成态 manifest、resolved config、10 条 epoch history、metrics、best checkpoint、last checkpoint 和 run lock。四组 checkpoint 内的 variant、config hash、data hash、epoch 与外部 JSON 一致。

#### 2.6.2 命令证据

四个 manifest 都没有 command、argv 或 invocation 字段，也没有 stdout/stderr 原始控制台日志。因此无法证明父 shell 的逐字命令。

可以根据未修改的 scripts/ETTm1.sh 和 resolved config 高置信重建等价调用：

~~~text
python -u main.py
  --implementation_variant AMD-paper-norm-wd-ddi-v1
  --seed 2024
  --dataset_id ETTm1
  --data data/ETTm1.csv
  --feature_type M
  --target OT
  --artifact_root artifacts
  --name ETTm1
  --device cuda:0
  --seq_len 512
  --pred_len <96|192|336|720>
  --n_block 1
  --alpha 0.0
  --mix_layer_num 3
  --mix_layer_scale 2
  --patch 16
  --norm True
  --layernorm True
  --dropout 0.1
  --train_epochs 10
  --batch_size 128
  --learning_rate 0.00003
  --weight_decay 0.0000001
~~~

本次 M0-A 接受“未修改脚本 + resolved config”作为该 smoke run 的等价命令证据，但仍不得表述为已保存的原始 argv。脚本的默认 seed 列表为 2024，horizon 顺序为 96、192、336、720；四个 run 的创建时间连续且顺序吻合。

所有未来正式 run 必须原生保存逐项 `sys.argv`、可重放的完整 command、完整 `stdout.log`、`stderr.log` 与 `train.log`，不得以事后重建替代。

#### 2.6.3 共同配置与结果

共同配置：

| 字段 | 值 |
|---|---|
| commit | fa9665627e6fcfb1d0c2bc22d943ca9666304fd6 |
| variant | AMD-paper-norm-wd-ddi-v1 |
| dataset | ETTm1 |
| feature_type | M |
| 实际输出/评价通道 | 全部 7 个变量 |
| seq_len | 512 |
| seed | 2024 |
| epochs | 10 |
| batch size | 128 |
| learning rate | 3e-5 |
| weight decay | 1e-7 |
| device | NVIDIA A800 80GB PCIe / cuda:0 |
| metric space | train-standardized |

结果：

| Horizon | Run ID | Best epoch | Test MSE | Test MAE |
|---:|---|---:|---:|---:|
| 96 | 20260815T095505.154937Z-352d581a | 10 | 0.29053000 | 0.34479168 |
| 192 | 20260815T095816.693056Z-b7e4863f | 10 | 0.32905525 | 0.36706965 |
| 336 | 20260815T100109.227685Z-c8bfd64a | 9 | 0.36377269 | 0.38649264 |
| 720 | 20260815T100404.645317Z-5f125d34 | 6 | 0.42211783 | 0.41598824 |

target 字段存在语义歧义：CLI/config 写 OT，但 feature_type=M、target_indices=0..6，实际评价覆盖全部 7 个变量。因此 artifact 合同中不应将其误记为“仅 OT”。

#### 2.6.4 Artifact 文件指纹

| Horizon | best.pt SHA-256 | last.pt SHA-256 |
|---:|---|---|
| 96 | 9fdffa9afad8654761af1ecad1264a8bde745722601472ce8513431910c70272 | ddf9d6358694191e176571ad83487853129991eeecc741ce3b32730ef94d46b1 |
| 192 | 6c6d7790c5ecbf7364edecc69c99325e9f96835734b203ef1d85e3c07ce0dc88 | 873b1762ba79ca1f3ee0d302ad580b05a5522b1553caea2f70c140ccda302dd6 |
| 336 | 176b78babbca03b0ef20b47752d783a14e4f515124061300e75c8e935dc76542 | f73228b75eee6c758c1a475baccea1fd3ea58fe3ea8278ed51e0b719ee0dcdb4 |
| 720 | d9caef669ee9cefac9f4c2864a4269b712461b4c053242c2e9a2e0b1a3ef7ede | f1d051216a0300c1c3e9773530ad56c668da00dec82a05cb0562ebf879348360 |

结构化文件指纹：

| Horizon | config.resolved.json | history.jsonl | manifest.json | metrics.json |
|---:|---|---|---|---|
| 96 | 26a6432bbe6c3847118a2c3ed96269bce871ff044f7ed1709f43321309a38d37 | cf59c2e06cb76d569702b38d0206872f3ad1745aaa4c95c2b3e528e73b39d85b | a834a0ef9dd00d9cc6827dfdf1eab762fae1d1aceb27dc7ac25a63da926a8263 | 02818e156a935c9476dfef8a9bff33d6f13572e3e1bac5754f9eb8489884b60e |
| 192 | 67cc1129536bffba30589231cf065491b729698f8a08f0fe4f59d708c03023de | d718f4c3b65e994a36f0da0d5d7d367d8d4d9060a1450a5006a49adb49b9f16a | 2f938e8a9545268b1debf4ec8002a1e307aaa41257078f1038c4b7f10b4e1e86 | dc76d7427bf02fa42e5ed0235ab22e1f83cc211deebc3c09fff11943c7842d4e |
| 336 | a262408306efff8a7923c538957002a13dbbbf483d85e918145552df1e39b59b | 02110aeae45b3a3804345e2721665cfda93342af485f0d7c6fae32ae8057b83b | 7aa6d60ce58b1e23df21535098f3bf05fd1d5ea238857b1af7e47daeec5fa570 | f477b7bd8b236a2a0ae58cae3e07cfc776a7d2866bda3e9bbbe2e3686014fbef |
| 720 | 4e33671bee8fc7ce2936f7c70ebb989a11bb3988fc07a96e69af214927e67381 | 29e940db435d1dcdf0432c72bbbd6d25900d09e84c22b1e69783af4b39611931 | 738c720e6db6b127d0f4cab0eb024304138d510ddbf79ef7f6038f1768e49eb1 | 7dd3ae32ac6cf2f536c35755773d639f0a2256eb888950c00488e2ac6b0e5b0d |

checkpoint SHA-256 是本次现场审计计算值，原 manifest 未保存这些哈希。

#### 2.6.5 来源状态与缺失项

四个 config 记录的 Git 状态相同：

~~~text
commit = fa9665627e6fcfb1d0c2bc22d943ca9666304fd6
dirty = true
status = ["?? docs/"]
~~~

关键 Python、dataloader、ETTm1 脚本相对该 commit 没有工作区修改；runner 记录的聚合源码指纹已成功现场复算。因此 dirty 状态不会使本次来源失去可追踪性，但必须保留这一事实。

缺失或风险：

- 未保存原始 argv/父 shell 命令。
- 未保存 stdout/stderr 原始日志。
- 未保存逐样本 prediction/ground-truth 数组。
- task_mode 与 fold 不是显式元数据字段。
- target=OT 与实际全变量评价存在语义歧义。
- artifacts 被 .gitignore 忽略，只在本机保存时存在单点丢失风险。
- 原 manifest 未记录 checkpoint 文件哈希。
- 当前产物路径不满足 v2.1 完整合同。

本轮默认不重跑，只记录以上缺失项。

### 2.7 环境与数据指纹

#### 2.7.1 数据

| 字段 | 值 |
|---|---|
| 数据文件 | data/ETTm1.csv |
| SHA-256 | 6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e |
| 原始行数 | 69,680 |
| 本协议使用行数 | 57,600 |
| train_end | 34,560 |
| val_end | 46,080 |
| test_end | 57,600 |
| 列数 | 7 |
| scaler | 仅由训练切片拟合；mean/scale 已保存于各 resolved config |

每个 horizon 的 split context start、窗口数、列名、scaler 参数和数据哈希均在 config.resolved.json 中。固定 ETT endpoint 会忽略标准长度之后的尾部行；数据哈希和 used_rows 必须与结果共同报告。

#### 2.7.2 运行环境

四个 run 记录的环境一致：

| 组件 | 版本/状态 |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.0.1 |
| torch CUDA | 11.8 |
| cuDNN | 8700 |
| NumPy | 1.24.3 |
| Pandas | 2.0.3 |
| SciPy | 1.11.4 |
| scikit-learn | 1.3.2 |
| torchvision | 0.15.2 |
| torchaudio | 2.0.2 |
| GPU | NVIDIA A800 80GB PCIe |
| CUDA driver | 550.163.01 |
| cudnn deterministic | true |
| cudnn benchmark | false |
| CUDA matmul TF32 | false |
| cuDNN TF32 | true |
| deterministic algorithms | false |

环境规范文件指纹：

| 文件/记录 | SHA-256 |
|---|---|
| environment.yml | 22c75c7c8772288477e120cc45fff8d0cbb6f16d5e2e95d83db7ed281bda427c |
| requirements.txt | decf601fa4e839455ed04d50ea03ebb1930875b707a3fecfda12addacc148889 |
| runner 记录并复算匹配的聚合源码指纹 | c54d35cfd393e315b7f1ff3e920df203abbf70258b0623dd2d222f25c6ba1380 |

### 2.8 Artifact 合同与现有映射

v2.1 的最低 artifact 身份合同固定为：

~~~text
variant / dataset / task_mode / target /
horizon / fold / seed / run_id
~~~

未来建议路径：

~~~text
artifacts/<variant>/<dataset>/<task_mode>/<target>/
  horizon_<h>/fold_<fold>/seed_<seed>/<run_id>/
~~~

本次 ETTm1 历史产物的建议映射：

| 合同字段 | 建议值 | 证据状态 |
|---|---|---|
| variant | AMD-paper-norm-wd-ddi-v1 | 已显式记录 |
| dataset | ETTm1 | 已显式记录 |
| task_mode | parallel_multivariate | 已锁定映射；历史 metadata 未显式记录 |
| target | all | 已锁定映射；当前 CLI 的 OT 容易误导 |
| horizon | 96 / 192 / 336 / 720 | 已显式记录为 pred_len |
| fold | official | 已锁定映射；历史 metadata 未显式记录 |
| seed | 2024 | 已显式记录 |
| run_id | 各现有时间戳-ID | 已显式记录 |

现有路径：

~~~text
variant/dataset/sl512_pl<horizon>/seed2024/run_id
~~~

它缺少 task_mode、target、fold，并额外把 seq_len 与 horizon 合并。本轮不重组、不迁移历史产物；只记录映射和风险。

正式增强模型 variant 保持：

- 时间模型：el-amd-pmcr-teb-v1
- 时空模型：st-el-amd-hst-sadr-sc-simgca-v1

这两个 variant 尚未创建运行产物。

### 2.9 公平性与图构造边界

后续实验的“公平统一”含义固定为：同一数据集内统一输入变量、数据划分、scaler、seed 列表和评价流程；不要求不同模型使用完全相同的模型专属超参数。模型专属学习率、hidden、layer、patch、kernel 等可以不同，但验证搜索空间、预算和选择依据必须留档。

图构造边界固定为：

- DTW、相关性等由需求时间序列统计得到的图，必须按 fold 且只使用该 fold 的训练集构建。
- 官方地理邻接、距离或坐标图可以固定使用，但必须记录来源、节点顺序和哈希。
- 本轮未构图，也未修改任何图代码。

### 2.10 工作区清理审计

整理动作前已经先生成：

~~~text
docs/evidence/M0/m0a_workspace_inventory.md
SHA-256: bdac55d54d51a8d76c48f95816282309a92ac87d0088282bea07a88a82c29dea
~~~

清点时：

| 状态 | 数量 |
|---|---:|
| tracked 文件 | 33 |
| staged/unstaged tracked diff | 0 / 0 |
| untracked | 4 |
| ignored | 82 |

4 个初始 untracked 文件是两份归档旧方案和两份 diff patch。ignored 分组为 artifacts 32、checkpoints 4、data 9、summaries 2、各处 Python cache 35。

ETTm1 smoke artifact 仓库外备份：

~~~text
/public/home/yueweiting/大论文/m0a_external_backups/ETTm1_smoke-reproduction_fa966562_20260815
~~~

- 原 artifact 保持原路径，未迁移、未覆盖。
- 备份文件数：28。
- 备份校验文件：`checksums.sha256`。
- `checksums.sha256` 自身 SHA-256：`01b9fff9e96396d1e80b589edbae46844c69060a5bb4e6a6084a7d4b7bbbc13a`。
- `sha256sum -c`：28/28 通过。
- 原目录与备份 `diff -qr`：无差异。

本轮清理决策：

- 保留全部 artifacts、checkpoints、data、summaries、日志和 cache。
- 保留两份 diff patch。
- 保留父目录 replacement 源文件。
- 经明确授权，将 Git 顶层同名 replacement 重复副本移至仓库外文档源备份目录。
- 不处理其他用途不明或可能为唯一副本的文件。
- 未执行 git clean -fd、git reset --hard 或 rm -rf。
- 未删除任何 material 文件。

### 2.11 M0-A 已完成与当时未完成的复现范围

#### M0-A 已完成

- tag 目标与本地/远端状态闭环。
- 当前真实 forward 审计。
- 相对 upstream/main 的完整累计 diff 审计。
- 工作区清单与非破坏性整理建议。
- 唯一权威 v2.1 文档落定，旧稿继续归档失效。
- ETTm1 四个标准 horizon、seed 2024 的 smoke artifact 完整性、配置、结果、环境、数据、源码和 checkpoint 指纹审计。
- Artifact 合同建议映射与风险登记。
- ETTm1 smoke artifacts 已完整复制到仓库外，原 artifact 未迁移或覆盖。
- 仓库外备份的 28 个文件已由 `checksums.sha256` 覆盖，`sha256sum -c` 全部通过，且与原目录 `diff -qr` 无差异。

#### M0-A 当时未完成或不在范围

- 第三章正式 AMD baseline 的完整 dataset × horizon × seed 范围。
- 原始 argv、stdout/stderr 与逐样本 prediction/ground-truth 的补录。
- 历史 artifact 的合同迁移；本次仅完成原路径不变的仓库外备份。
- 任何同协议重跑。
- AMDEnhanced、return_state、固定零 exo_context、PMCR、TEB。
- AMDEnhanced 全关闭数值等价测试。
- 训练行为或测试代码修改。
- M0-B 和 M1。

### 2.12 M0-A 收尾已锁定决策

1. 本次 ETTm1 四 horizon 结果固定归类为“完整、可追踪的 M0 smoke-reproduction artifact”，不升级为第三章正式 AMD baseline。
2. 本次 smoke run 接受“未修改脚本 + resolved config”作为等价命令证据。
3. 标准 parallel multivariate 固定为 `task_mode=parallel_multivariate`、`target=all`、`fold=official`。
4. UrbanEV/CHARGED 纯时间固定为 `task_mode=target_exogenous`。
5. 第四章固定为 `task_mode=graph_spatiotemporal`。
6. 所有未来正式 run 必须原生保存 `sys.argv`、完整 command、`stdout.log`、`stderr.log` 和 `train.log`。
7. 所有未来正式 run 必须生成独立 `checksums.sha256`，至少覆盖 best、last、config、history、metrics、manifest 和 train.log。
8. 当前 ETTm1 smoke artifacts 已建立仓库外完整备份与校验清单；原 artifact 保持不动。
9. canonical v2.1 保持唯一权威；Git 顶层重复 replacement 已移出仓库，父目录源文件继续保留。
10. baseline tag 保持不动。M0-A 文档提交并达到 clean worktree 后，才允许进入已授权的 M0-B。

第三章正式 baseline 的完整 dataset、horizon、seed 范围仍未由本次 smoke artifact 自动定义，需在正式实验登记前另行锁定。

### 2.13 M0-A 实际执行的命令记录

以下列出本轮实际执行的核心命令；审计子任务还对多个 horizon 文件重复执行了同类只读检查。

#### 2.13.1 Git、tag 与 upstream

~~~bash
pwd
git status --short --branch --untracked-files=all
git status --porcelain=v2 --branch --untracked-files=all
git rev-parse --show-toplevel
git remote -v
git rev-parse HEAD
git rev-parse fa9665627e6fcfb1d0c2bc22d943ca9666304fd6
git cat-file -t fa9665627e6fcfb1d0c2bc22d943ca9666304fd6
git tag -l amd_reproduced_baseline_v1
git ls-remote --tags origin refs/tags/amd_reproduced_baseline_v1
git tag -a amd_reproduced_baseline_v1 fa9665627e6fcfb1d0c2bc22d943ca9666304fd6 -m "AMD reproduced baseline v1 at paper-close commit fa9665627e6fcfb1d0c2bc22d943ca9666304fd6"
git rev-parse amd_reproduced_baseline_v1^{}
git cat-file -t amd_reproduced_baseline_v1
git push origin refs/tags/amd_reproduced_baseline_v1
git ls-remote --tags origin refs/tags/amd_reproduced_baseline_v1 refs/tags/amd_reproduced_baseline_v1^{}
git fetch upstream main
git ls-remote --heads upstream refs/heads/main
git rev-parse upstream/main
git merge-base fa9665627e6fcfb1d0c2bc22d943ca9666304fd6 upstream/main
git rev-list --left-right --count fa9665627e6fcfb1d0c2bc22d943ca9666304fd6...000d377a1ed8946aa817ff357cdf1de64b99abb9
git log --reverse --format="%H %ad %s" --date=iso-strict upstream/main..fa9665627e6fcfb1d0c2bc22d943ca9666304fd6
~~~

#### 2.13.2 累计 diff

~~~bash
git diff --stat upstream/main..fa9665627e6fcfb1d0c2bc22d943ca9666304fd6
git diff --summary upstream/main..fa9665627e6fcfb1d0c2bc22d943ca9666304fd6
git diff --name-status upstream/main..fa9665627e6fcfb1d0c2bc22d943ca9666304fd6
git diff --numstat upstream/main..fa9665627e6fcfb1d0c2bc22d943ca9666304fd6
git diff --ignore-space-at-eol --stat upstream/main..fa9665627e6fcfb1d0c2bc22d943ca9666304fd6
git diff --ignore-space-at-eol --numstat upstream/main..fa9665627e6fcfb1d0c2bc22d943ca9666304fd6
git diff --ignore-space-at-eol upstream/main..fa9665627e6fcfb1d0c2bc22d943ca9666304fd6 -- models/tsAMD.py
git diff --ignore-space-at-eol upstream/main..fa9665627e6fcfb1d0c2bc22d943ca9666304fd6 -- models/common.py
git diff --ignore-space-at-eol upstream/main..fa9665627e6fcfb1d0c2bc22d943ca9666304fd6 -- models/tsmoe.py
git diff --ignore-space-at-eol upstream/main..fa9665627e6fcfb1d0c2bc22d943ca9666304fd6 -- main.py
git diff --ignore-space-at-eol upstream/main..fa9665627e6fcfb1d0c2bc22d943ca9666304fd6 -- utils/dataloader.py
git diff --ignore-space-at-eol upstream/main..fa9665627e6fcfb1d0c2bc22d943ca9666304fd6 -- scripts
git grep -n -I -E "artifact|result_path|result.csv|checkpoint|metrics.json|summar" upstream/main
git grep -n -I -E "artifact|result_path|result.csv|checkpoint|metrics.json|summar" fa9665627e6fcfb1d0c2bc22d943ca9666304fd6
~~~

#### 2.13.3 工作区、文档与指纹

~~~bash
git status --short --ignored --untracked-files=all
git ls-files
find docs -type f -printf ...
find . -path ./.git -prune -o -type f -print
find /public/home/yueweiting/大论文 -type f -name "*.docx" -printf ...
find /public/home/yueweiting -maxdepth 6 -type f -name "*.docx" -printf ...
find /public/home/yueweiting/大论文 -maxdepth 2 -type f -name "AMD_EV_Thesis_Final_Implementation_Plan_v2.1_replacement.md" -printf ...
grep -R -n -F "AMD_EV_Thesis" . --include="*.md" --exclude-dir=.git
sed -n "1,1278p" ../AMD_EV_Thesis_Final_Implementation_Plan_v2.1_replacement.md
sed -n "1,260p" models/tsAMD.py
sed -n "1,240p" scripts/ETTm1.sh
sha256sum ../AMD_EV_Thesis_Final_Implementation_Plan_v2.1_replacement.md docs/AMD_EV_Thesis_Final_Implementation_Plan_v2.1.md
sha256sum environment.yml requirements.txt data/ETTm1.csv models/tsAMD.py models/common.py models/tsmoe.py main.py utils/dataloader.py scripts/ETTm1.sh summarize_results.py
diff -u docs/AMD_EV_Thesis_Final_Implementation_Plan_v2.1.md docs/.AMD_EV_Thesis_Final_Implementation_Plan_v2.1.m0a-new.md
diff -u docs/archive/AMD_EV_Thesis_Final_Implementation_Plan_v2.0.md docs/archive/.AMD_EV_Thesis_Final_Implementation_Plan_v2.0.m0a-new.md
diff -u docs/archive/AMD_EV_Thesis_Modification_and_Experiment_Plan.md docs/archive/.AMD_EV_Thesis_Modification_and_Experiment_Plan.m0a-new.md
mv -- docs/.AMD_EV_Thesis_Final_Implementation_Plan_v2.1.m0a-new.md docs/AMD_EV_Thesis_Final_Implementation_Plan_v2.1.md
mv -- docs/archive/.AMD_EV_Thesis_Final_Implementation_Plan_v2.0.m0a-new.md docs/archive/AMD_EV_Thesis_Final_Implementation_Plan_v2.0.md
mv -- docs/archive/.AMD_EV_Thesis_Modification_and_Experiment_Plan.m0a-new.md docs/archive/AMD_EV_Thesis_Modification_and_Experiment_Plan.md
~~~

文档内容均由 apply_patch 生成。由于宿主机不支持 bwrap user namespace，apply_patch 对已存在文件的直接 Update File 验证失败；因此先用 apply_patch 生成临时替换稿，使用 diff -u 精确确认只含预期行，再以显式路径原子改名到正式位置。

#### 2.13.4 ETTm1 artifact 只读审计

~~~bash
find artifacts/AMD-paper-norm-wd-ddi-v1/ETTm1 -type f -printf ...
sed -n "1,260p" artifacts/.../manifest.json
sed -n "1,320p" artifacts/.../config.resolved.json
sed -n "1,240p" artifacts/.../metrics.json
sed -n "1,240p" artifacts/.../history.jsonl
grep -RIn "ETTm1|AMD-paper-norm-wd-ddi-v1" artifacts scripts main.py
sha256sum data/ETTm1.csv
sha256sum artifacts/.../best.pt artifacts/.../last.pt
sha256sum artifacts/.../config.resolved.json artifacts/.../history.jsonl artifacts/.../manifest.json artifacts/.../metrics.json
git ls-tree -r fa9665627e6fcfb1d0c2bc22d943ca9666304fd6
git diff fa9665627e6fcfb1d0c2bc22d943ca9666304fd6 -- models main.py utils/dataloader.py scripts/ETTm1.sh
git check-ignore -v artifacts/AMD-paper-norm-wd-ddi-v1/ETTm1/...
wc -l artifacts/.../history.jsonl
od -An -tx1 artifacts/.../.run.lock
stat -c ... data/ETTm1.csv artifacts/.../manifest.json
~~~

为核验 checkpoint 顶层元数据、runner 的 config/source fingerprint 算法，审计子任务还在现有 amd 环境中执行了只读 Python 单行命令；没有导入训练入口、没有训练或改写文件。

rg、jq 和 file 的首次只读调用因系统未安装而返回 command not found，随后分别使用 find/grep、sed 和 checkpoint 元数据读取完成等价审计；没有安装任何工具。若干最初命令因宿主机 bwrap/user namespace 不可用失败，随后在用户批准的受控权限下以相同只读目标重试。

### 2.14 M0-A 停止点与闭合

M0-A 于 commit `1f76f7904198297beade6c5ad2b34fc917281624` 闭合并推送。闭合时尚未创建 AMDEnhanced，尚未实现 `return_state_source`、PMCR 或 TEB，未修改训练行为、未启动训练，也未进入 M1；随后才在独立 commit 中执行 M0-B。`m0a_closure.md` 中未重复的备份、校验、replacement 副本与门控信息已合并到本节及 2.3、2.10、2.12，不再保留重复 closure 文件。

## 3. M0-B：AMDEnhanced 空壳

M0-B 在独立 commit `7ba48697a417f8b517e03d16ed7493b7bbebea27` 中完成，只新增：

- `models/tsAMD_enhanced.py`：128 行；
- `tests/test_tsAMD_enhanced.py`：216 行。

没有修改 `models/tsAMD.py`、LayerNorm 实现、训练入口、训练行为或 artifact 写入逻辑。

### 3.1 真实 forward

`AMDEnhanced` 继承冻结 AMD，并逐句保持 baseline tag 的主路径：

```python
u_mdm = MDM(x_ch)
v = u_mdm
for block in DDI_blocks:
    v = block(v)
pred_all_norm, moe_loss = AMS(v, u_mdm)
```

完整默认路径仍执行原 AMD 的输入检查、可选 RevIN norm/denorm、transpose 和 `target_slice`。新增模块全部关闭时，不在预测路径中插入任何变换。默认调用：

```python
pred, moe_loss = model(x)
```

返回合同与原 AMD 相同。

### 3.2 return_state_source 合同

公开参数名固定为 `return_state_source`，不提供含义模糊的 `return_state`。`target_idx` 是 keyword-only 构造参数，必须显式提供且满足：

```text
isinstance(target_idx, int)
target_idx 不是 bool
0 <= target_idx < feature_num
```

`teb_context_dim` 同样必须显式提供且为正整数。非法值在构造阶段拒绝。开启 `return_state_source=True` 时返回：

```python
exo_context = v.new_zeros((v.shape[0], teb_context_dim))
state_source = torch.cat(
    (
        v[:, target_idx, :],
        u_mdm[:, target_idx, :],
        exo_context,
    ),
    dim=-1,
)
return pred, moe_loss, state_source
```

固定合同：

```text
state_source shape = [B, 2 * seq_len + teb_context_dim]
zero_exo_context shape = [B, teb_context_dim]
```

零 context 由 `v.new_zeros` 构造，因此 dtype/device 与 `v` 一致且值确定。`return_state_source=True` 只增加返回值，不改变预测或 MoE loss。M0-B 没有实现或返回 `H_time`，没有创建 `StateProjection`/`StateAdapter`，也没有实现 PMCR、TEB、UrbanEV DataLoader 或任何空间模块。

## 4. 数值等价与回归测试

### 4.1 AMD 与 AMDEnhanced 等价性

等价测试采用：

- 相同 AMD 权重并以 `strict=True` 加载；
- base/enhanced 均为 `eval()`；
- 相同输入；
- 每条 forward 前恢复相同 CPU RNG 状态和所有可用 CUDA RNG 状态；
- 同时覆盖 `return_state_source=False` 与 `True`；
- 预测和 MoE loss 的最大绝对误差门槛保持严格小于 `1e-6`。

实测结果：

| Device | 默认路径 pred | 默认路径 MoE | state 路径 pred | state 路径 MoE | 结论 |
|---|---:|---:|---:|---:|---|
| CPU | 0 | 0 | 0 | 0 | 通过 |
| CUDA（NVIDIA A800 80GB PCIe） | 0 | 0 | 0 | 0 | 通过 |

权重键严格兼容，无 missing/unexpected keys；state_source 顺序、shape、dtype、device 和固定零尾部均由独立测试锁定。

### 4.2 LayerNorm baseline/current 五次对照

唯一已知失败测试：

```text
tests.test_public_architecture.ArchitectureContractTests.
test_layernorm_uses_each_channels_last_sequence_dimension
```

对照版本：

- baseline：`amd_reproduced_baseline_v1` → `fa9665627e6fcfb1d0c2bc22d943ca9666304fd6`；
- current：M0-B commit `7ba48697a417f8b517e03d16ed7493b7bbebea27`。

baseline 使用 detached 临时 worktree，未移动 tag、未覆盖当前工作树、未改写历史；每次运行均为新的 amd Python 进程。CUDA 对照通过 `torch.set_default_device("cuda")` 运行同一个未修改测试。记录值是失败断言 `mdm_output.mean(dim=-1)` 相对零张量的最大绝对误差：

| 版本 | Device | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | 原 3e-6 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| baseline tag | CPU | 3.09944152832e-6 | 3.09944152832e-6 | 3.09944152832e-6 | 3.09944152832e-6 | 3.09944152832e-6 | 5/5 失败 |
| M0-B commit | CPU | 3.09944152832e-6 | 3.09944152832e-6 | 3.09944152832e-6 | 3.09944152832e-6 | 3.09944152832e-6 | 5/5 失败 |
| baseline tag | CUDA | 0 | 0 | 0 | 0 | 0 | 5/5 通过 |
| M0-B commit | CUDA | 0 | 0 | 0 | 0 | 0 | 5/5 通过 |

baseline 与 M0-B 在各 device 上完全一致；`tests/test_public_architecture.py`、`models/common.py` 和 `models/tsAMD.py` 在两版本间也无差异。因此 CPU 失败定性为：

```text
pre-existing numerical-tolerance failure
```

CUDA 的零误差不改变该定性，反而说明该断言存在后端相关的浮点舍入差异。未发现 M0-B import、副作用、默认 dtype、RNG 或测试顺序回归。

### 4.3 测试维护与完整回归

独立 test-maintenance commit：

```text
2ca729f7454ec4634c2e4d7c90a4a3eb08984128
```

唯一修改：

```python
torch.testing.assert_close(
    mdm_output.mean(dim=-1),
    torch.zeros(1, 2),
    rtol=0,
    atol=5e-6,
)
```

即仅将该测试的绝对容差从 `3e-6` 调整为 `5e-6`。没有修改 LayerNorm 实现或任何模型代码，也没有放宽 AMD/AMDEnhanced 的 `1e-6` 等价门槛。维护后该唯一测试在 CPU、CUDA 各单独运行一次均通过。

完整回归命令：

```bash
/public/home/yueweiting/miniconda/envs/amd/bin/python   -m unittest discover -s tests -p "test_*.py" -v
```

| 测试状态 | 总数 | 通过 | 失败 |
|---|---:|---:|---:|
| M0-B commit，维护前 | 57 | 56 | 1（上述既有 CPU 容差失败） |
| G0 commit，维护后 | 57 | 57 | 0 |

维护后完整回归耗时 1.437 秒并返回 `OK`。完整回归同时实际执行 M0-B 的 CPU/CUDA 等价测试，四项最大误差在两个 device 上均打印为 0。

## 5. 已知问题

1. LayerNorm 均值为零的 float32 断言对执行后端敏感：本机 CPU 为 `3.09944152832e-6`，CUDA 为 0。`atol=5e-6, rtol=0` 只用于该结构测试，不是模型等价门槛。
2. ETTm1 四 horizon 仍只是完整、可追踪的 M0 smoke-reproduction artifact，不是第三章正式 AMD baseline；正式实验范围尚需在相应阶段锁定。
3. 历史 smoke run 没有原生保存 argv、完整 command、stdout/stderr/train.log，也没有按新合同组织 task_mode/target/fold；这些缺口已记录，不追溯伪造、不默认重跑。
4. 当前宿主机禁用非特权 user namespace，部分 Codex 沙箱命令会触发 bwrap 错误；本轮以逐条批准的受控沙箱外命令执行。这是开发环境限制，不影响仓库数值结论。
5. M0 没有训练 `H_time` 或任何状态投影；后续如需图状态，只能在相应里程碑中从 `state_source` 建立并训练明确的适配器，不能把未训练随机投影当作第三章输出。

以上问题均已定性并有边界，不阻塞 G0；M1 仍未开始。

## 6. G0 门禁结论

状态：Closed。

关闭依据：

- baseline tag 的本地/远端目标固定为 `fa9665627e6fcfb1d0c2bc22d943ca9666304fd6`，没有移动或 force-update；
- M0-A commit 已推送，基准、upstream 累计 diff、ETTm1 smoke、环境、数据和工作区均可追踪；
- M0-B 保留真实 `MDM → DDI → AMS(v, u_mdm)` forward，默认返回合同与 AMD 相同；
- `return_state_source`、显式合法 `target_idx`、固定零 exo_context 与 shape 合同均有测试；
- CPU/CUDA 的预测与 MoE loss 最大绝对误差均为 0，严格满足 `<1e-6`；
- LayerNorm 对照证明唯一失败为基准中已存在的数值容差问题，且独立维护 commit 只修改测试断言；
- G0 技术门禁 commit 上完整回归 57/57 通过；
- 文档结构归一为 Plan / Milestone / Evidence / Archive，M0 只有本报告一份阶段报告；
- 未启动训练，未进入 M1，未实现 UrbanEV DataLoader、PMCR、TEB、`H_time` 或空间模块。

`2ca729f7454ec4634c2e4d7c90a4a3eb08984128` 是完成数值门禁并通过完整回归的 G0 技术 commit；随后独立 docs commit 只记录和整理本结论，不改变代码或测试行为。

## 7. 交付物和 artifact 路径

| 类型 | 路径/标识 |
|---|---|
| 唯一权威方案 | `docs/AMD_EV_Thesis_Final_Implementation_Plan_v2.1.md` |
| M0 唯一阶段报告 | `docs/milestones/M0_baseline_freeze_and_equivalence.md` |
| M0 机器证据 | `docs/evidence/M0/` |
| 失效旧方案 | `docs/archive/plans/` |
| AMDEnhanced | `models/tsAMD_enhanced.py` |
| M0-B 测试 | `tests/test_tsAMD_enhanced.py` |
| LayerNorm 维护测试 | `tests/test_public_architecture.py` |
| ETTm1 smoke 原产物 | `artifacts/AMD-paper-norm-wd-ddi-v1/ETTm1/` |
| ETTm1 仓库外备份 | `/public/home/yueweiting/大论文/m0a_external_backups/ETTm1_smoke-reproduction_fa966562_20260815/` |
| 备份校验清单 | 上述备份根目录的 `checksums.sha256`；清单 SHA-256 `01b9fff9e96396d1e80b589edbae46844c69060a5bb4e6a6084a7d4b7bbbc13a` |
| 父目录 replacement 源备份 | `/public/home/yueweiting/大论文/AMD_EV_Thesis_Final_Implementation_Plan_v2.1_replacement.md` |
| 移出的仓库重复副本 | `/public/home/yueweiting/大论文/m0a_external_backups/document_sources/AMD_EV_Thesis_Final_Implementation_Plan_v2.1_replacement_repo_duplicate_20260815.md` |

`docs/evidence/M0/` 当前包含：

- `upstream-main-000d377_to_fa96656.full.patch`；
- `5a718d5_to_fa96656.ignore-eol.patch`；
- `m0a_workspace_inventory.md`。

原始训练日志、checkpoint、metrics、manifest 继续留在 artifacts/仓库外备份，不复制进 docs。M0 报告至此冻结；后续阶段不得继续向本报告追加新的阶段结果。
