# M0-A 收尾记录

> 日期：2026-08-15（UTC）
> 阶段：M0-A closure；本记录生成时 M0-B 尚未开始

## 1. 已锁定合同

- 标准 parallel multivariate：`task_mode=parallel_multivariate`、`target=all`、`fold=official`。
- UrbanEV/CHARGED 纯时间：`task_mode=target_exogenous`。
- 第四章：`task_mode=graph_spatiotemporal`。
- 当前 ETTm1 四 horizon 结果是完整、可追踪的 M0 smoke-reproduction artifact，不是第三章正式 AMD baseline。
- 本次 smoke run 接受未修改脚本与 resolved config 作为等价命令证据。
- 所有未来正式 run 必须原生保存 `sys.argv`、完整 command、`stdout.log`、`stderr.log` 和 `train.log`。
- 所有未来正式 run 必须生成独立 `checksums.sha256`，至少覆盖 best、last、config、history、metrics、manifest 和 train.log。

## 2. ETTm1 仓库外备份

原 artifact：

~~~text
/public/home/yueweiting/大论文/AMD/artifacts/AMD-paper-norm-wd-ddi-v1/ETTm1
~~~

仓库外备份：

~~~text
/public/home/yueweiting/大论文/m0a_external_backups/ETTm1_smoke-reproduction_fa966562_20260815
~~~

- 原 artifact 未迁移、未覆盖。
- 复制文件数：28。
- 备份校验文件：`checksums.sha256`。
- 校验清单 SHA-256：`01b9fff9e96396d1e80b589edbae46844c69060a5bb4e6a6084a7d4b7bbbc13a`。
- `sha256sum -c checksums.sha256`：28/28 通过。
- 原目录与备份目录 `diff -qr`：无差异。

## 3. Replacement 重复副本

父目录权威源备份继续保留：

~~~text
/public/home/yueweiting/大论文/AMD_EV_Thesis_Final_Implementation_Plan_v2.1_replacement.md
~~~

Git 顶层同名重复副本已移出仓库到：

~~~text
/public/home/yueweiting/大论文/m0a_external_backups/document_sources/AMD_EV_Thesis_Final_Implementation_Plan_v2.1_replacement_repo_duplicate_20260815.md
~~~

两者 SHA-256 均为：

~~~text
1f952f13f50b5717e9fa1dda4b1a1436414c2f2765aaf0a1a0a3621b28be2dd5
~~~

移动没有覆盖现有文件。

## 4. 文档状态

- 唯一权威方案：`docs/AMD_EV_Thesis_Final_Implementation_Plan_v2.1.md`。
- canonical SHA-256：`de89194462d88a8f9d837bf8e48ee04f27142e263cbf701660c7c6bd5da118b7`。
- 两份旧方案继续保留在 `docs/archive/`，并明确失效。

## 5. Baseline tag

- tag：`amd_reproduced_baseline_v1`。
- peeled commit：`fa9665627e6fcfb1d0c2bc22d943ca9666304fd6`。
- tag 未移动、未覆盖、未 force-update。

## 6. 实际执行的收尾命令

~~~bash
test ! -e /public/home/yueweiting/大论文/m0a_external_backups/ETTm1_smoke-reproduction_fa966562_20260815
mkdir -p /public/home/yueweiting/大论文/m0a_external_backups/ETTm1_smoke-reproduction_fa966562_20260815
cp -a --parents artifacts/AMD-paper-norm-wd-ddi-v1/ETTm1 /public/home/yueweiting/大论文/m0a_external_backups/ETTm1_smoke-reproduction_fa966562_20260815

find artifacts/AMD-paper-norm-wd-ddi-v1/ETTm1 -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > checksums.sha256
sha256sum -c checksums.sha256
sha256sum checksums.sha256
diff -qr /public/home/yueweiting/大论文/AMD/artifacts/AMD-paper-norm-wd-ddi-v1/ETTm1 /public/home/yueweiting/大论文/m0a_external_backups/ETTm1_smoke-reproduction_fa966562_20260815/artifacts/AMD-paper-norm-wd-ddi-v1/ETTm1

mkdir -p /public/home/yueweiting/大论文/m0a_external_backups/document_sources
mv -- AMD_EV_Thesis_Final_Implementation_Plan_v2.1_replacement.md /public/home/yueweiting/大论文/m0a_external_backups/document_sources/AMD_EV_Thesis_Final_Implementation_Plan_v2.1_replacement_repo_duplicate_20260815.md
sha256sum ../AMD_EV_Thesis_Final_Implementation_Plan_v2.1_replacement.md /public/home/yueweiting/大论文/m0a_external_backups/document_sources/AMD_EV_Thesis_Final_Implementation_Plan_v2.1_replacement_repo_duplicate_20260815.md
~~~

文档内容通过 `apply_patch` 生成；已有文档因宿主机 bwrap 不支持 Update File，采用“apply_patch 生成临时稿 → diff 精确验证 → 显式路径原子改名”的方式落盘。

## 7. 阶段门控

M0-A 文档必须提交并推送，随后 `git status --porcelain` 必须为空。只有该门控通过，才开始已授权的 M0-B；baseline tag 始终保持不动。
