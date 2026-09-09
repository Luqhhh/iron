# bf-tap-predict

[![locked-tests](https://github.com/Luqhhh/iron/actions/workflows/tests.yml/badge.svg?branch=optimization-v0.10)](https://github.com/Luqhhh/iron/actions/workflows/tests.yml)

高炉铁次铁量与时长预测项目，包含 as-of 特征、因果 OOF、训练、离线推理、质量门槛和提交包审计。
工程基线 `baseline-v0.1-reproducible` 保持冻结；后续优化使用独立阶段、配置与运行目录。

## 当前状态

截至 2026-09-09，最新完成阶段为 **optimization-v0.10 / OPT-23**，代码提交 `fcda5e0` 已推送。
当前活动候选及桌面包仍为 **V1_RATE_STRUCTURAL，test_a 用户回传 83.0319**；回退候选为 **R2，83.0207**。
平台成绩未独立核验，不代表开发结果能直接换算成排行榜收益。

| 项目 | 当前状态 |
| --- | --- |
| 当前发布登记 | [configs/optimization_v0_8/active_release.yaml](configs/optimization_v0_8/active_release.yaml) |
| 回退登记 | [configs/optimization_v0_4/active_release.yaml](configs/optimization_v0_4/active_release.yaml) |
| 最新工程验收 G0 | v0.10 执行及独立冷进程通过，六个 origin 预测最大差为 0 |
| 最新质量验收 G1 | V4 九项门槛全部失败，固定候选已关闭 |
| 锁定环境测试 | Python 3.12.12，221 passed；本地证据，不等同于远端 CI 状态 |
| 新待测包 | 无；v0.9/v0.10 均未生成 challenger |
| 保护标签状态 | November 已在授权生命周期消费；后续为已消费回溯开发 |
| 时间语义 | `competition-timestamp-contract-v1 / ASSUMED`，未新增官方确认 |

[当前发布与推理](docs/optimization_v0_8/CURRENT_RELEASE.md) · [最新实施报告](docs/report.md) ·
[文档索引与历史口径](docs/INDEX.md) · [机器可读状态](EVIDENCE_STATUS.json)

## 最近实验

| 阶段 | 结果 | 决策 |
| --- | --- | --- |
| [v0.10 / OPT-23](docs/optimization_v0_10/RESULTS.md) | 12 次低容量 residual fit；相对 V1，J 退化 0.0053128，H1 E 退化 0.0015335 | 关闭固定 V4；无 final fit、无平台包 |
| [v0.9 / OPT-21/22](docs/optimization_v0_9/RESULTS.md) | W0 保留 V1 开发收益 91.44%；8 次 q fit；V2/V3 J 改善仅 0.0001607 / 0.0000530 | 严格门槛失败，关闭 ratio 扩展 |
| [v0.8 / OPT-20](docs/optimization_v0_8/RESULTS.md) | V1 通过开发门槛；平台用户回传比 R2 提高 0.0112 分 | 保留当前 V1 |
| [v0.7 / OPT-19](docs/optimization_v0_7/RESULTS.md) | 0 fit；pseudo-history J 退化 0.0047625 | 关闭历史递归路线 |
| [v0.6](docs/optimization_v0_6/RESULTS.md) | S1 开发通过，但平台回传 82.7707，比 R2 低 0.2500 分 | 未晋级；关闭旧模型路由/融合路线 |

各阶段报告记录当时的候选、桌面包和训练状态，不能用其中的“当前”替代上方发布登记。
历史 baseline DEV_LONG 失败、50 项冻结测试等证据见 [冻结报告](docs/review/FREEZE_REPORT.md)，不作为最新项目状态。

## 环境与数据

```bash
uv sync --locked --extra dev --python 3.12
uv run --locked --python 3.12 pytest
uv run --locked --python 3.12 python scripts/check_no_private_artifacts.py
```

CI 另有 Python 3.11 兼容性检查。真实训练和推理的权威环境使用 Python 3.12 与 `uv.lock`。

用户已授权将 **`初赛数据集/`** 内赛事数据纳入 Git；此授权不扩大到其他目录。
模型、逐样本预测、本地报告、访问账本和提交 ZIP 仍保存在被忽略的 `local/`，不进入 Git。
新机器仅克隆源码及数据不会自动获得已保存的 V1/R2 模型包，需要恢复匹配摘要的本地产物。

本机路径配置可从 `configs/data.example.yaml` 复制为被忽略的 `configs/data.local.yaml`。
当前 V1 的独立推理配置只包含 `test_a_samples`、`operation_hourly`、`burden_change` 和 `data_dictionary`，
带 `schema_version: 1`；不包含训练标签或官方历史文件路径。使用与 bundle 内容身份匹配的公共源。

## 使用当前 V1

现成 test_a 原包：

```text
local/runs/optimization-v0.8-v1-challenger-r1/Luqhhh_bf_tap_predict_prelim.zip
SHA-256: fdcbe03e8ea31577bfeed0c45cd0a0013bda42ab88d29db703fad9c2f7e557aa
```

从已保存模型重新进行独立推理（输出目录须存在，文件名须未使用）：

```bash
mkdir -p local/predictions
uv run --locked --python 3.12 python scripts/optimization_v8_cold_predict.py \
  --bundle local/runs/optimization-v0.8-v1-challenger-r1/bundle \
  --data-config local/runs/optimization-v0.8-v1-challenger-r1/cold_data_repaired.yaml \
  --output local/predictions/v1-cold-UNIQUE.csv
```

该脚本验证原 R2 预测一致性和输入反序一致性，禁止推理 fit，输出内部预测及审计 JSON。
它不生成新的提交 ZIP，也不上传平台。提交使用已核验原包；内部预测列不应直接作为赛事 `result.csv` 上传。
通用 `python -m bf_tap predict` 是 baseline bundle 入口；当前 V1 复合模型使用上方专用入口。
V1 现有专用入口面向 test_a；R2 的 B/C 冷检查是历史工程证据，不等于当前 V1 的 B/C 质量验收。

R2 回退包 SHA-256：`e42602d3045e43b4b49dd1e1c104aa8e5c1f29c639ff5f892b3b07434ed9bbdf`。
恢复方式见 [R2 回退说明](docs/optimization_v0_4/CURRENT_RELEASE.md)。

## 因果与保护边界

- operation：`event_time = available_at = clock`；burden：`event_time = available_at = cal_time`。
- 历史目标：`available_at = tap_end_time`，只纳入参考时刻前已可用且符合场景 cutoff 的记录。
- 冻结 baseline 的 development 入口仍拒绝 November 目标。已授权的优化阶段使用自己的访问范围、冻结 manifest 和追加账本；November 已消费，不再称为未触碰 holdout。
- v0.8–v0.10 已评估六个 H1 origins、18-cell 和 DEV_LONG/SHORT；真实 holdout/final-training 生命周期在早期 r2 阶段已执行。
- 时间语义仍是条件性操作约定；官方若改变窗口或报送时点，应新建 contract ID，不能覆盖旧证据。

详细边界见 [数据契约](docs/data_contract.md) 和 [实施范围](docs/task_contract.md)。
平台“最后一次提交”与“最优成绩”口径尚无本项目独立确认记录，上传前须确认；本项目不自动上传。

## 文档与工程入口

```text
configs/                 冻结契约、分阶段配置与发布登记
src/bf_tap/              特征、模型、训练、推理、评估和审计
scripts/                 冷进程复现、环境证据和私有资产检查
tests/                   单元测试与合成端到端测试
docs/                    当前入口文档及各阶段冻结报告
md/                      原始实施包归档，不作为当前状态来源
初赛数据集/              已授权公开的赛事数据
local/                   本机模型、预测、报告、账本及 ZIP（不入 Git）
EVIDENCE_STATUS.json     current_status 为当前摘要，旧字段保留历史含义
```

[平台记录](docs/submission_log.md) · [发布身份](docs/release_identity.md) · [文档目录](docs/INDEX.md)

当前没有已登记的后续训练任务。后续实验须独立预注册候选、OOF 边界、预算与门槛；
本轮 V4 的失败不自动推导为所有 residual 方法都无效，也不授权继续参数扫描。
