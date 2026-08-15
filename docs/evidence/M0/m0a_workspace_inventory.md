# M0-A 工作区文件清单与建议动作

> 快照日期：2026-08-15（UTC）
> 本清单在文档整理前生成。清点过程未移动、删除或覆盖任何文件。

## 1. Git 工作区快照

- 开发分支：`AMD-paper-repro-custom-modules-v1`
- `HEAD`：`fa9665627e6fcfb1d0c2bc22d943ca9666304fd6`
- 相对 `origin/AMD-paper-repro-custom-modules-v1`：`+0/-0`
- tracked 文件：33
- staged/unstaged tracked diff：均为空
- untracked 文件：4
- ignored 文件：82

清点时的 4 个 untracked 文件：

1. `docs/archive/AMD_EV_Thesis_Final_Implementation_Plan_v2.0.md`
2. `docs/archive/AMD_EV_Thesis_Modification_and_Experiment_Plan.md`
3. `docs/audit/5a718d5_to_fa96656.ignore-eol.patch`
4. `docs/audit/upstream-main-000d377_to_fa96656.full.patch`

其中两份 Markdown 是已经归档的旧方案；两份 patch 是基线/上游差异审计证据，不属于旧方案，不应移入 `docs/archive/`。

## 2. ignored 文件分组

| 分组 | 文件数 | 建议 |
|---|---:|---|
| `artifacts/` | 32 | 保留原位；包含 ETTm1 冒烟复现候选产物，不迁移、不删除 |
| `checkpoints/` | 4 | 保留原位，不删除 |
| `data/` | 9 | 保留原位，不删除 |
| `summaries/` | 2 | 保留原位，不删除 |
| 各目录 `__pycache__/` | 35 | 本轮无需处理；不为追求状态整洁而删除 |

ignored 文件不妨碍 tracked 工作区保持干净。它们中可能存在唯一副本，因此 M0-A 不做移动或清理。

## 3. 权威方案与旧方案

已定位 replacement 源文件：

```text
/public/home/yueweiting/大论文/AMD_EV_Thesis_Final_Implementation_Plan_v2.1_replacement.md
```

- 大小：36,181 bytes
- SHA-256：`1f952f13f50b5717e9fa1dda4b1a1436414c2f2765aaf0a1a0a3621b28be2dd5`
- 清点时它是唯一定位到的 v2.1 replacement 源副本。

两份旧方案已位于 `docs/archive/`，且正文标明“已归档/失效、不得参与决策”。其权威文档路径仍指向工作区根目录，建议仅修正为 `docs/AMD_EV_Thesis_Final_Implementation_Plan_v2.1.md`，继续保留旧稿及失效声明。

在 `/public/home/yueweiting/大论文` 全范围和 `/public/home/yueweiting` 最大六层深度内均未定位到 `.docx`。本轮只记录“未定位”，不推断用途，也不执行文件操作。

## 4. 建议动作

1. 将 replacement 内容落为 `docs/AMD_EV_Thesis_Final_Implementation_Plan_v2.1.md`，作为唯一权威方案。
2. 落盘后核验源文件和 canonical 副本的 SHA-256；如需按用户明确修正补入 `task_mode`，应单独记录该文档差异。
3. 保留仓库外 replacement 源文件；本轮不移动、不删除。
4. 只修正两份归档旧稿中的权威文档路径，继续保留其失效声明。
5. 保留 `docs/audit/` 下两份 patch，作为累计 diff 的审计证据。
6. ETTm1 产物只做完整性与来源审计；本轮不重组、不迁移、不重跑。
7. 不处理 artifacts、checkpoints、data、summaries、日志或 Python 缓存。
8. 对未定位的 DOCX 和任何用途不明/可能为唯一副本的文件，只报告，等待用户决定。

## 5. 本轮明确禁止的清理动作

- 不执行 `git clean -fd`。
- 不执行 `git reset --hard`。
- 不执行 `rm -rf`。
- 不删除、覆盖或移动用途不明或唯一性未确认的文件。
- 不修改任何 Python 模型、训练脚本或测试代码。
- 不启动训练或复现重跑。
