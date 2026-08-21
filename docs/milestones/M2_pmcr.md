# M2：Peak-preserving Modern Convolution Refinement（PMCR）

状态：Closed
日期：2026-08-21（UTC）
结束日期：2026-08-21（UTC）
canonical 内部版本：v2.1-R1
implementation gate：Passed
implementation review：Passed
Git closure：Authorized by user

## 1. 范围与阶段边界

本里程碑只实现 variable-independent PMCR、AMDEnhanced 条件接入、冻结 AMD checkpoint 专用 importer、递归模型源码 fingerprint，以及对应单元测试。未实现 TEB、parallel TEB、StateAdapter、`H_time`、图模式或任何空间模块；未接入正式增强 runner，未进入 M3。

## 2. 继承状态

```text
repository: /public/home/yueweiting/大论文/AMD
branch: AMD-paper-repro-custom-modules-v1
starting HEAD: f906a969f6fbd250f1d4520ec4db8be0d4f3d0df
origin tracking HEAD: f906a969f6fbd250f1d4520ec4db8be0d4f3d0df
baseline tag: amd_reproduced_baseline_v1
baseline commit: fa9665627e6fcfb1d0c2bc22d943ca9666304fd6
M0/G0: Closed
M1: Closed
```

本轮未 checkout、reset、stage、commit、push 或移动 tag。

## 3. canonical 更新

```text
before SHA-256: e5492be514df8c61f12855eb6198368139b3856b8e653ea97aefdae57cce555e
after SHA-256:  79b9bebec6591f9334d174900e2a2e8b799b7e83aad9760029ce8a8cab9ff45d
internal version: v2.1-R1（未改变）
```

仅消歧 canonical 第 6 节 PMCR 合同：固定 feature-wise LayerNorm、ConvFFN1 ratio 2、可学习全局 `gamma_pmcr=1e-3`、初始化、same padding、显式 deploy API、条件实例化与 checkpoint allowlist。EL-AMD 总体路线、TEB、M1 数据合同、StateAdapter、第四章图结构、实验门槛和 variant 命名均未改变。

## 4. ModernTCN 来源边界

只读参考服务器已有官方仓库：

```text
path: /public/home/yueweiting/大论文/ModernTCN
remote: https://github.com/luodhhh/ModernTCN.git
commit: 56a9a2c018385cd5acef015378cae7f084d1b11c
branch: main
status: clean
license: MIT（LICENSE）
```

AMD 仓库未复制 ModernTCN 源文件。PMCR 按本项目锁定公式独立实现，只保留 variable-independent temporal processing、large/small depthwise convolution、结构重参数化、ConvFFN1 和外层 residual；删除 patch stem、多 stage、ConvFFN2 和预测头。本轮未联网或下载。

## 5. 锁定模块定义

归一化与 FFN：

```text
large DWConv ----\
                  + -> feature-wise LayerNorm(d, eps=1e-5)
small DWConv ----/  -> Conv1d(d,2d,1) -> GELU -> Dropout
                     -> Conv1d(2d,d,1) -> Dropout
```

LayerNorm 在 `[B*C,T,d]` 的最后一维执行，不使用 BatchNorm、跨 batch 统计或分支内 norm。ConvFFN1 无内部 residual，dropout 默认 0.1。

`gamma_pmcr` 是所有变量共享的无约束标量 `nn.Parameter`，初始化 `1e-3`，不使用零初始化。input/output projection 和两个 FFN projection 使用 Xavier uniform、bias 0；两个 DWConv 使用 Kaiming fan-in linear 后各乘 `1/sqrt(2)`、bias 0；LayerNorm weight 1、bias 0。

kernel 固定 stride 1、dilation 1、zeros padding、`padding=k//2`；两个 kernel 必须为正奇数且 `k_large > k_small`，AMDEnhanced 额外要求 `k_large <= seq_len`。

## 6. 真实张量流

```text
H [B,C,T]
 -> reshape [B*C,1,T]
 -> Conv1d(1,d,1)
 -> large/small temporal DWConv sum [B*C,d,T]
 -> transpose [B*C,T,d]
 -> LayerNorm(d)
 -> transpose [B*C,d,T]
 -> ConvFFN1 d -> 2d -> d
 -> Conv1d(d,1,1)
 -> delta [B,C,T]
 -> H_out = H + gamma_pmcr * delta
```

`compute_delta(H)` 只返回未乘 gamma 的 `delta`；`forward(H)` 返回外层 residual。按 `B*C` 展开保证变量之间没有数据混合，全部变量共享参数。

AMDEnhanced 的实际路径：

```text
u_mdm = MDM(x_ch)
v = DDI_blocks(u_mdm)
v = PMCR(v)                    # 仅 use_pmcr=True
pred_all_norm, moe_loss = AMS(v, u_mdm)
```

AMS experts 使用 PMCR 后的 `v`，selector 使用原始 `u_mdm`。`state_source` 第一段使用 PMCR 后 `v_target`，第二段使用原始 `u_mdm_target`，第三段继续使用 dtype/device 正确的固定零 `exo_context`。

## 7. 重参数化合同

`ReparamLargeKernelDWConv` 训练形态包含 large/small 两个纯 depthwise Conv1d 分支。small kernel 居中补零后：

```text
K_eq = K_large + center_pad(K_small)
b_eq = b_large + b_small
```

公开 API：

- `get_equivalent_kernel_bias()`：只计算，不修改模块；
- `switch_to_deploy()`：显式原地融合、幂等，不由普通 forward 自动触发；
- `to_deploy()`：返回 eval 深拷贝，不修改原模块。

部署只融合 temporal branches；LayerNorm、ConvFFN1、input/output projection 和 residual 保留。转换保持 dtype、device、groups、stride、padding 和 dilation，部署 state dict 只保留 fused temporal convolution key。

## 8. `use_pmcr=False` baseline 保护

关闭时 `self.pmcr = None`，不产生 `pmcr.*` state dict key，forward 不执行 PMCR。AMD 与 AMDEnhanced key 集合完全相同，冻结 AMD checkpoint 可继续 `strict=True` 加载。CPU/CUDA 上 prediction、MoE loss，以及 `return_state_source=False/True` 路径最大误差均为 0。

## 9. checkpoint importer

`AMDEnhanced.load_amd_backbone_state_dict()` 在修改参数前比较完整 key 集合：

```text
allowed missing keys == 当前模型全部 pmcr.* keys
allowed unexpected keys == 空集
```

任何非 PMCR missing key、任何 unexpected key、或混入部分 PMCR key都会拒绝。PMCR 完整 checkpoint 恢复仍使用普通 `load_state_dict(..., strict=True)`；未引入全局静默 `strict=False`。

## 10. source fingerprint 递归修复

`main.source_fingerprint()` 原先只覆盖 `models/*.py`，无法纳入嵌套 PMCR 源码。现在递归覆盖 `models/**/*.py`，只收集 Python 文件，并按仓库相对 POSIX 路径确定性排序；仍覆盖 `main.py` 和既有 `utils/*.py`。

当前递归 source fingerprint：

```text
cef34929fd00c192e958ab01fe680287fe831d05ade673fe0089d64c1e4790f9
```

回归测试验证不同创建顺序哈希相同、修改嵌套 Python 文件会改变哈希、非 Python 文件不会改变哈希。未增加 PMCR CLI、implementation variant、artifact 路径或正式 runner 模型选择。

## 11. 实际代码与测试文件

新增代码：

- `models/modules/__init__.py`
- `models/modules/modern_conv_refinement.py`

修改代码：

- `models/tsAMD_enhanced.py`
- `main.py`（仅 source fingerprint）

新增测试：

- `tests/test_pmcr.py`
- `tests/test_pmcr_no_cross_variable.py`
- `tests/test_pmcr_reparameterization.py`

修改测试：

- `tests/test_tsAMD_enhanced.py`
- `tests/test_runner.py`

文档：

- 修改 `docs/AMD_EV_Thesis_Final_Implementation_Plan_v2.1.md`
- 新增本文件 `docs/milestones/M2_pmcr.md`

## 12. M2 十项门禁与测试映射

| # | 门禁 | 状态 | 主要证据 |
|---:|---|---|---|
| 1 | shape | Passed | `test_pmcr.py` 的短/长序列 shape |
| 2 | dtype/device | Passed | CPU float32/float64、CUDA float32 |
| 3 | `use_pmcr=False` 严格旁路 | Passed | `test_tsAMD_enhanced.py`，PMCR 为 None、key 相同 |
| 4 | PMCR 参数梯度非零 | Passed | input/branches/LN/FFN/output/gamma 全逻辑组 |
| 5 | PMCR 输入梯度非零 | Passed | 非对称加权 loss 输入梯度 |
| 6 | 无跨变量混合 | Passed | 单变量扰动与变量置换等变性 |
| 7 | T=12 same padding | Passed | k=3/7 的分支和最终长度均为 12 |
| 8 | 双分支/fused 等价 | Passed | CPU float64、CPU/CUDA float32 |
| 9 | 关闭时 AMD/AMDEnhanced 等价 | Passed | prediction 与 MoE loss 最大误差 0 |
| 10 | 完整回归 | Passed | 98/98 passed |

## 13. 定向测试结果

PMCR 核心：

```text
command: python -B -m unittest -v tests/test_pmcr.py tests/test_pmcr_no_cross_variable.py tests/test_pmcr_reparameterization.py
result: 15/15 passed, failed=0, skipped=0
unittest elapsed: 0.440 s（最终重跑）
```

AMDEnhanced、公共架构与 runner：

```text
command: python -B -m unittest -v tests/test_tsAMD_enhanced.py tests/test_public_architecture.py tests/test_runner.py
result: 38/38 passed, failed=0, skipped=0
unittest elapsed: 1.737 s
```

所有命令实际使用 `/public/home/yueweiting/miniconda/envs/amd/bin/python`，并设置 `PYTHONDONTWRITEBYTECODE=1`。

## 14. M1 保护性测试

```text
files: test_target_offset.py, test_fold_scaler_no_leakage.py,
       test_temporal_graph_loader_consistency.py,
       test_state_restore_node_order.py, test_urbanev_data_contract.py
result: 21/21 passed, failed=0, skipped=0
unittest elapsed: 2.882 s
raw x/y, y_time, state_source, moe_loss parity max_abs_error: 0
```

UrbanEV DataLoader、源 CSV 和 M1 milestone 均未修改。

## 15. 完整回归与数值误差

```text
command: python -B -m unittest discover -s tests -p 'test_*.py' -v
result: 98/98 passed, failed=0, skipped=0
unittest elapsed: 4.315 s（最终重跑）
```

重参数化最大绝对误差（temporal/delta/forward 三者取最大）：

| device/dtype | max_abs_error |
|---|---:|
| CPU float64 | 6.66133814775e-16 |
| CPU float32 | 4.17232513428e-07 |
| CUDA float32 | 3.57627868652e-07 |

AMD/AMDEnhanced 关闭等价最大误差：CPU 0，CUDA 0。未放宽既有 `1e-6` 门槛。

## 16. 冻结资产保护

```text
models/tsAMD.py SHA-256 before/after:
fa72cdbe34348364344c0d9c0755668a82d22f6a37ee061c7ece93ecfaf90ba1

M0 milestone SHA-256:
e2a20131664391752340e92a9d9a5302b0078cac48d7dcdbb4a4841a16f62cdd

M1 milestone SHA-256:
bcd9b3e3d821a1cf609423a3e8b58ecb4129bc202238e9fe1f8d7cf3a361c70b

baseline tag before/after:
fa9665627e6fcfb1d0c2bc22d943ca9666304fd6
```

M0/M1 milestone、冻结 AMD、baseline tag、UrbanEV 嵌套仓库和 ModernTCN 参考仓库均未修改。

## 17. 实现轮结束时尚未完成事项（Git closure 前历史快照）

本节记录 implementation review 结束、Git closure 开始前的现场；当前最终状态以第 19 节 Git closure 为准。

- 尚未获得 Git closure 授权，未 stage、commit 或 push；
- 未启动任何真实数据或正式实验训练；完整回归中既有 runner 集成测试仅在系统临时目录运行 tiny synthetic fixture，并由 `TemporaryDirectory` 回收；
- 未实现 TEB、parallel TEB、StateAdapter、`H_time` 或空间模块；
- 未接入正式增强 runner，`el-amd-pmcr-teb-v1` 延期到 M3 完整 EL-AMD 后统一接入；
- 未进入 M3；
- M1 延期项保持不变：`adj.csv` 不对称延至 M7，POI 无 TAZID 延至第四章可选静态特征工作。

## 18. Git closure 前工作区与下一步门禁（历史快照）

本节记录 implementation review 结束、Git closure 开始前的现场；当前最终状态以第 19 节 Git closure 为准。

当前 HEAD 仍为 `f906a969f6fbd250f1d4520ec4db8be0d4f3d0df`，与 origin tracking ref 一致；本轮修改只包含第 11 节列出的代码、测试和两份文档。M2 implementation gate 已通过，但 M2 尚未 Closed。

下一步必须由用户审核本实现、更新后的 canonical 与本 milestone，并另行授权 Git closure；未获授权前不得 commit、push 或进入 M3。

## 19. Git closure

- 用户已于 2026-08-21（UTC）完成 M2 implementation review，并明确授权 M2 Git closure。
- M2 implementation review：Passed。
- M2 implementation gate：Passed。
- M2 状态由 `Implementation complete; awaiting review and Git closure` 更新为 `Closed`。
- closure 范围严格限于本 milestone 第 11 节记录的 11 个代码、测试与文档文件。
- closure 前 PMCR 定向测试、AMDEnhanced/runner 定向测试、M1 保护性测试及完整回归全部通过；完整回归为 98/98 passed。
- `models/tsAMD.py`、M0 milestone、M1 milestone、UrbanEV、ModernTCN 和不可变 baseline tag 均未修改。
- 本次 closure 未启动训练，未实现 TEB、parallel TEB、StateAdapter、H_time 或空间模块，未接入正式增强 runner，未进入 M3。
- 本次 closure 使用一个 M2 commit，并推送到 `origin/AMD-paper-repro-custom-modules-v1`。
- 最终 closure commit 的完整 SHA、远端核验和最终 clean 状态由 Codex 最终回复报告；本文档不写入尚未存在的、自身所属 commit SHA，也不为记录自身 SHA 创建递归补充提交。
- M2 在 closure 后冻结，后续阶段不得继续向本 milestone 追加 M3 或更晚阶段结果。
