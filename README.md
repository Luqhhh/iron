# bf-tap-predict

[![locked-tests](https://github.com/Luqhhh/iron/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/Luqhhh/iron/actions/workflows/tests.yml)

高炉铁次双目标预测的防时间泄漏 baseline。项目提供从原始文件审计、时间切分、特征构造、训练、离线推理到提交打包的完整工程链路。

工程基线已冻结为 [`baseline-v0.1-reproducible`](https://github.com/Luqhhh/iron/tree/baseline-v0.1-reproducible)。后续模型质量工作必须进入独立的 optimization-v0.2 阶段，不回写 baseline 配置、结果或证据。

## 当前状态

optimization-v0.4 已完成 OPT-11 四臂解耦、OPT-12 两个固定历史候选、OPT-13 跨截止点组合、
18 格回溯验证与唯一胜出候选发布。**R2** 通过全部预登记门槛，J 改善 0.0013787；
132 项锁定测试通过。正式训练使用 2,754 行，截止点为 2024-12-01 01:44+08:00。

2026-09-09 用户回传 R2 test_a 为 **83.0207**，相对 E16 的 82.9918 提高 **0.0289**。
当前发布登记为 `configs/optimization_v0_4/active_release.yaml`；桌面同名提交包已核验为 R2。
成绩属于用户回传，未通过平台 API 或回执独立核验。November 始终是已消费的回溯开发数据。
详见 [v0.4 完整结果](docs/optimization_v0_4/RESULTS.md) 和
[当前发布与恢复方法](docs/optimization_v0_4/CURRENT_RELEASE.md)。本轮初赛工作包已收口；B/C 尚无本轮验收和平台结果。

E16 原包保留为回退，旧 v0.3/r2 发布配置和导出入口保留历史含义。
旧全量 E12-raw 的 82.5430、CB-FC-CVcal 的 82.2871 及失败证据均不覆盖。
下表及后续旧阶段结果仍是冻结 baseline / 历史记录，不作为当前发布指针。

| 项目 | 状态 |
| --- | --- |
| G0 工程正确性与可复现性 | `PASS_LOCAL_LOCKED_ENVIRONMENT` |
| G1 冻结模型质量 | `FAIL_DEV_LONG` |
| baseline 发布状态 | `BASELINE_REPRODUCIBLE_QUALITY_FAILED` |
| 时间可用性证据 | `competition-timestamp-contract-v1 / ASSUMED` |
| baseline 冻结时保护集 | 当时未消费；当前已由 r2 合法消费，见上方 |
| 冻结测试 | 50 passed，0 failed，0 skipped |
| test_a / prelim 平台成绩 | `81.4554`（用户报告，未独立核验） |

真实 DEV 结果：

| 场景 | CatBoost E | B0 | B1 | 质量 |
| --- | ---: | ---: | ---: | --- |
| DEV_LONG | 0.185646 | 0.176148 | 0.176101 | FAIL |
| DEV_SHORT | 0.171456 | 0.178157 | 0.178897 | PASS |

E 越低越好。这些是真实赛事数据上的授权本地运行证据，不是公共 GitHub 环境对私有数据的独立复现。两次干净重训的 raw prediction 完全一致；冻结后的 bundle v3 与此前 baseline 输出最大绝对差为 `0.0`。

用户使用冻结 baseline、2024-11-01 development 截止点生成的 `Luqhhh_bf_tap_predict_prelim.zip` 报告 test_a 平台成绩 `81.4554`。提交包 SHA-256 为 `44ac6ced2fe3b871f387500caed325183bd73a55f11c56efb7a0f84531957c8f`；平台回执未纳入仓库，因此该成绩只标记为用户报告。详见 [提交记录](docs/submission_log.md)。

## 冻结内容

模型路线保持为两个固定 CatBoost MAE 回归器。没有通过调参、改变窗口、放宽门槛或读取 11 月标签来制造质量通过。

`configs/baseline.yaml` 声明三份 canonical SHA-256：

| 契约 | SHA-256 |
| --- | --- |
| baseline | `798a44e4fb98a1f83d91f0077c32618117b9c25098562a73e6412e792911639c` |
| feature | `c2fa86c47b4ef906591530cd7fbb368de14bf48f1b1fb3bc4f1dba6bd51e3191` |
| semantic | `b54c7add8805094245548ac79ff9087d1d7183d68a79441b27a8a96604ad6a25` |

启动时会重新 canonicalize 完整配置并核对摘要。因此 operation/burden 字段、6/24 小时窗口、24/72 小时陈旧阈值、history 3/10 窗口、布尔开关、类别缺失标记或时间语义发生漂移时，都不能继续冒充 `baseline-v0.1`。

bundle v3 还包含 `inference_source_contract`，强绑定训练时公共 process source 的 SHA-256 和字节数。当前真实 source contract ID 为：

```text
public-process-sources-v1-2773c0f38f89ce32
```

本机路径不参与 source 身份：内容相同、路径不同可以恢复；内容变化即使列结构相同，也会在特征构建前失败。官方若更新公共表，必须显式训练新 bundle 并生成新 source contract/version。

## 环境安装

权威本地环境为 Python 3.12 和锁定的 `uv.lock`：

```bash
uv sync --locked --extra dev --python 3.12
uv run python -m bf_tap --help
uv run pytest
uv run python scripts/check_no_private_artifacts.py
```

CI 另有 Python 3.11 兼容性 job，但它不替代 Python 3.12 锁定环境证据。

## 数据准备

复制示例配置并填写本机路径：

```bash
cp configs/data.example.yaml configs/data.local.yaml
```

`configs/*.local.yaml` 已被 Git 忽略，只承载本机文件位置。可提交的字段映射和时间语义位于 `configs/data_contract.yaml`，保护边界位于 `configs/protection.yaml`；不要把私有路径复制进语义配置。

真实 CSV/XLSX、字段字典、模型、预测、报告、访问账本和提交包不得进入 Git。仓库公开不代表赛事资产获准外发。

推理可使用更小的 `configs/predict.local.yaml`：

```yaml
schema_version: 1
paths:
  test_a_samples: /absolute/path/to/test_a_samples.csv
  operation_hourly: /absolute/path/to/operation_hourly.csv
  burden_change: /absolute/path/to/burden_change.csv
```

推理不需要训练标签文件；历史授权快照、特征配置和时间语义都随 bundle 保存。

## 运行流程

### 1. 结构审计

结构审计不会统计受保护目标值：

```bash
uv run python -m bf_tap audit \
  --data-config configs/data.local.yaml \
  --output local/reports/structural-audit-<unique-id>.json
```

### 2. DEV 验证

```bash
uv run python -m bf_tap validate \
  --data-config configs/data.local.yaml \
  --data-contract configs/data_contract.yaml \
  --protection-policy configs/protection.yaml \
  --protection-ledger local/manifests/protected_access.json \
  --config configs/baseline.yaml \
  --feature-config configs/features.yaml \
  --split-config configs/validation.yaml \
  --suite development \
  --output local/runs/<unique-dev-run-id>
```

DEV 会在读取标签前检查独立保护边界、fold 时序、时区、分区非空和 train/eval ID 交集。run ID 已存在时直接失败；失败目录和证据不得删除或覆盖。

### 3. Development bundle 训练

```bash
uv run python -m bf_tap train \
  --data-config configs/data.local.yaml \
  --data-contract configs/data_contract.yaml \
  --protection-policy configs/protection.yaml \
  --config configs/baseline.yaml \
  --feature-config configs/features.yaml \
  --mode development \
  --train-start 2024-03-01T00:00:00+08:00 \
  --fit-cutoff 2024-11-01T00:00:00+08:00 \
  --output local/runs/<unique-train-run-id>
```

bundle v3 包含模型、特征 schema、完整配置、契约摘要、训练身份、授权历史快照、公共 source contract、环境和组件摘要。

### 4. 独立离线推理

```bash
uv run python -m bf_tap predict \
  --bundle local/runs/<train-run-id>/bundle \
  --data-config configs/predict.local.yaml \
  --stage test_a \
  --output local/predictions/<unique-predict-run-id>
```

`predict` 从 bundle 恢复语义并校验公共 source，不调用 `fit`。同一 bundle 的两次独立进程推理应满足 raw 最大绝对差不超过 `1e-9`，结果 CSV 字节一致。

### 5. 校验和打包

```bash
uv run python -m bf_tap check-submission \
  --data-config configs/predict.local.yaml \
  --stage test_a \
  --path local/predictions/<id>/result.csv

uv run python -m bf_tap pack \
  --data-config configs/predict.local.yaml \
  --stage test_a \
  --team-name <team> \
  --result local/predictions/<id>/result.csv \
  --output-dir local/submissions/<id>
```

内部生成器要求 ID 集合和官方输入顺序同时一致。`pack` 会重新检查字段、阶段 ID、顺序、行数、有限性和非负性，并在压缩后回读 ZIP、复核 payload 摘要；不依赖操作者预先运行 `check-submission`。

## 时间与保护集契约

当前时间映射是可复现的操作约定，不是官方确认的完整发布时间事实：

- operation：`event_time = available_at = clock`
- burden：`event_time = available_at = cal_time`
- history/target：`available_at = tap_end_time`

仍待官方确认：小时统计窗口边界、burden reporting delay、target reporting delay。不要擅自添加 1/6/24 小时滞后，也不要根据排行榜反馈修改时间含义；官方澄清后应创建新 contract ID，使相关旧产物失效。

development 流程不能读取 2024 年 11 月目标。`official-release` 需要冻结 manifest 和显式 `final_training` 访问账本；当前 H1–H4 与真实 protected lifecycle 均未执行。

## 项目结构

```text
configs/                 冻结模型、特征、验证、语义与保护配置
src/bf_tap/              审计、特征、训练、推理、验证和提交实现
tests/                   单元测试及 subprocess 文件级 E2E
scripts/                 环境证据、私有资产检查和复现比较工具
docs/                    当前契约、状态和审阅报告
md/                      原始实施方案归档，不是运行权威来源
初赛数据集/              用户明确授权纳入 Git 的赛事数据集
EVIDENCE_STATUS.json     机器可读项目状态
```

## 证据与报告

- [工程冻结报告](docs/review/FREEZE_REPORT.md)
- [修复报告](docs/review/REPAIR_REPORT.md)
- [历史验收报告](docs/review/ACCEPTANCE_REPORT.md)
- [数据契约](docs/data_contract.md)
- [发布身份](docs/release_identity.md)
- [机器可读状态](EVIDENCE_STATUS.json)
- [平台提交记录](docs/submission_log.md)

## 后续工作

以下 P2 已记录，但不阻塞 optimization-v0.2：

- 用纯合成数据贯通 `holdout_scoring` 与 `final_training` lifecycle；
- 正式候选生成 bundle 外部 `release_manifest.json`；
- 正式 candidate 强制 clean commit/tree；
- 为 GitHub main 配置 branch protection 和 required `locked-tests`。

下一阶段的首要建模问题是：为什么 DEV_LONG 上 CatBoost 弱于简单的 per-spout median？任何优化都应以冻结标签作为对照，使用新配置、run ID 和独立分支记录。

## optimization-v0.2

OPT-01/02 的独立入口、E00–E06 特征消融、origin×horizon 网格、候选登记和
收缩诊断位于 `src/bf_tap/optimization/` 与 `configs/optimization_v0_2/`。
冻结 baseline 文件及其行为保持不变。实施边界、决策和仅含汇总值的本地开发
结果见 [执行计划](docs/optimization_v0_2/PLAN.md)、[决策记录](docs/optimization_v0_2/DECISIONS.md)
与 [结果摘要](docs/optimization_v0_2/RESULTS_SUMMARY.md)。

截至 2026-09-07，OPT-01/02 已完成；OPT-03 冻结历史适配已实施并完成全网格，
G0 通过但 G1 未通过。test_a 探索候选的用户回传最高分仍为 `82.7046`；该序列
不是独立验证。用户明确要求将较优的预登记 OPT-03 候选 E07 作为一次探索性
平台探针，用户回传 `82.3610`，低于当前最高分 `82.7046`；该例外不改变失败
的开发门禁，也不重启依据平台分数的自适应调参。

OPT-04 已加入基于冻结 as-of 聚合的有符号过程变化量。E09 在完整网格中通过
全部 G1 门禁（J `0.173861`，相对 E00 改善 `0.009222`），用户回传 test_a
成绩 `82.8174`，比此前最高分提高 `0.1128`，现为平台 incumbent。

OPT-05 已将固定派生候选正式纳入完整门禁。`E12_BLEND_E09_E04_80_20` 的 J 为
`0.171713`，相对 E00 改善 `0.011370`，G0/G1 均通过；其 335 行 test_a 包已
冻结并替换桌面提交包。用户回传成绩 `82.9543`，比 E09 提高 `0.1369`，现为
平台 incumbent；该分数不用于回调 OPT-05 权重。

OPT-06 在 clean 预登记提交上比较目标级组合。`E16_TIMECAL_E12` 通过全部 G1
门禁，J `0.169902`，相对 E00 改善 `0.013182`；其提交包已替换桌面包，平台
回传成绩 `82.9918`，比 E12 提高 `0.0375`，现为 incumbent。OPT-06 已关闭，
不依据该分数继续调整残差或组合。
