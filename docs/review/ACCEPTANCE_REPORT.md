# AI 技术审阅与阶段验收报告

> 历史快照：本文记录修复前运行 `dev-baseline-v0.1-contract-v1-r2` 的当时事实，不回写为已在公开提交上重新执行。修复后的证据与状态见 `REPAIR_REPORT.md` 和仓库根部 `EVIDENCE_STATUS.json`。

审阅日期：2026-09-06。当前结论：`REAL_DEVELOPMENT_EXECUTED_QUALITY_FAILED`。

- 审阅方式：self-review；没有把本次自审表述为独立审查。
- 代码状态：新初始化的独立 `main` 仓库，尚无 commit、全部工程文件未提交；源码/测试/公开配置快照摘要为 `7bac507576bfdf161345c742e702727d38995424e8635b3dd1ebdc1159f7cf1e`。
- 远端：仅在本地绑定 `https://github.com/Luqhhh/iron.git`，未 push。
- 冻结配置 SHA256：baseline `8bf46a0...a98c`；features `c041789...a13`；validation `8636f87...714f`；acceptance `082708c...de31`。
- 环境：WSL2 Linux，AMD Ryzen 9 7945HX（16 核/32 线程）；Python 3.12.12；CatBoost 1.2.8；NumPy 2.2.6；pandas 2.3.3；PyYAML 6.0.2；openpyxl 3.1.5；pytest 9.0.2。完整传递依赖和哈希在 `uv.lock`（SHA256 `bd804b8...d6dc`）。
- 实际命令：`uv sync --locked --extra dev --python .venv/bin/python`；`.venv/bin/pytest -q`；`.venv/bin/python -m bf_tap audit --data-config configs/data.local.yaml --output local/reports/structural_audit_v4.json`；`.venv/bin/python -m bf_tap validate --data-config configs/data.local.yaml --config configs/baseline.yaml --feature-config configs/features.yaml --split-config configs/validation.yaml --suite development --output local/runs/dev-baseline-v0.1-contract-v1-r2`。
- 合成测试：32 passed，0 failed，日志 `local/reports/pytest_precommit.xml`（SHA256 `2e75c86...a2f2`）。覆盖手算 WMAPE=0.1/Score=90、ID 对齐、0/100 分边界、schema、保护标签读取、fit/逐样本边界、动态过程源、未来扰动、历史冻结/当前铁次排除、窗口左右边界、24h 陈旧边界、同刻冲突、输入顺序不变、缓存隔离、B0/B1、提交/ZIP、固定 800 轮双模型 roundtrip 与端到端 smoke。
- 真实结构审计：`local/reports/structural_audit_v4.json`。训练 2754 行、A 335、B 322、C 548、operation 9360、burden 4021、history 2754；所有审计时间可解析，样本 ID 无重复。burden 有同一 `cal_time` 的 2 行，但两行全字段完全一致，依契约可记录后去重，无值冲突。
- 数据副本：train/test 中 operation、burden、history、dictionary 分别逐字节相同；数据不进入 Git。
- 保护集：没有读取或报告 2024 年 11 月两个目标值；结构审计从 CSV 解析层排除两个目标列。`holdout_consumed=false`。
- 真实字段状态：冻结 `competition-timestamp-contract-v1`：operation 使用 `clock`、burden 使用 `cal_time`、历史与目标使用 `tap_end_time` 作为 `available_at`。这是依据官方发布表唯一时间字段形成的可复现操作解释；官方若补充报告延迟则新建版本，不能覆盖本 run。
- DEV_LONG：训练 1180、评估 1244；CatBoost E=0.185646（Score=81.4354），B0 E=0.176148，B1 E=0.176101；综合门槛失败，time 相对 B1 的单目标护栏也失败。
- DEV_SHORT：训练 1803、评估 621；CatBoost E=0.171456（Score=82.8544），B0 E=0.178157，B1 E=0.178897；综合与两个单目标门槛均通过。
- 真实运行：`local/runs/dev-baseline-v0.1-contract-v1-r2`，执行 71.51 秒；metrics SHA256 `c03b5c8...8b17`，run manifest SHA256 `18e841e...0b54`。包含逐月、逐铁口、缺失/陈旧分组、Top-20、本地模型及全配置/环境快照。
- H1–H4：未运行，保护标签仍封存。
- 独立进程双推理、双重训、真实耗时/RSS/模型体积：属于 M5，未执行。
- G0：`NOT_MEASURED`；合成核心正确性与真实 DEV 执行通过，但 H1–H4、独立进程和双重训等 M5 复现证据未完成。
- G1：`FAIL_DEV_LONG`；全部六场景尚未完成，但 DEV_LONG 已违反必要条件，因此整体 G1 已不可能通过。本轮没有据此调参或换模型。
- official_baseline_status：`UNKNOWN`。
- 自动提交/付费资源/远程写入：均未发生。

## 方案审阅发现

冻结的 D0–D3、时间切分、固定双 CatBoost、特征族、指标与验收方向可以实施。原实施包有两类事实需要更正，但无需推翻路线：

1. 实际 CSV 和字典已经存在，原文的 `BLOCKED_DATA` 已过期；可用时刻缺列已由带版本的赛事时间戳操作契约修复。
2. 字典 `PackageFiles/Summary` 仍是两阶段旧汇总，和实际三阶段文件及 PDF 不一致。实现采用 PDF 与实际文件一致的 A/B/C 行数和 `prelim`/`round2`/`semifinal` 命名。

下一任务若继续 baseline 验收，应冻结当前代码/配置并进入 M5 的 H1–H4、独立进程复现和双重训；由于 DEV_LONG 已失败，后续状态最多为 `BASELINE_REPRODUCIBLE_QUALITY_FAILED`。优化需作为单独阶段，不能改写本轮证据。
