# bf-tap-predict


高炉铁次铁量与时长预测项目，包含 as-of 特征、因果 OOF、训练、离线推理、质量门槛和提交包审计。
工程基线 `baseline-v0.1-reproducible` 保持冻结；后续优化使用独立阶段、配置与运行目录。

v0.14 / OPT-30–31 已完成 H2 匹配时长校准：G0 通过，G1 失败、关闭固定 V7；V1 与 R2 回退保持不变。

## 当前状态

截至 2026-09-12，最新本地阶段为 **optimization-v0.14 / OPT-30–31**，评分前登记 `4f855ed`。v0.13 已在用户授权后推送 `2db6d5f`，其 locked-tests run `34687452551` 成功；本轮新增 [维护观察](docs/optimization_v0_14/MAINTENANCE_20260912.md)，旧历史报告不改写。v0.14 本地提交、未公开推送。
当前活动候选仍为 **V1_RATE_STRUCTURAL，test_a 用户回传 83.0319**；回退候选为 **R2，83.0207**。
平台成绩未独立核验，不代表开发结果能直接换算成排行榜收益。

| 项目 | 当前状态 |
| --- | --- |
| 当前发布登记 | [configs/optimization_v0_8/active_release.yaml](configs/optimization_v0_8/active_release.yaml) |
| 回退登记 | [configs/optimization_v0_4/active_release.yaml](configs/optimization_v0_4/active_release.yaml) |
| 最新工程验收 G0 | v0.14 七份真实 H2/同日期 H1、六个 outer 原模型复现与独立冷审计通过；铁量 exact equality |
| 最新质量验收 G1 | FAIL_CLOSE_V7_RETAIN_V1；H2 delta E +0.00017521，delta J -0.00015855 |
| 锁定环境测试 | Python 3.12.12，342 passed；v0.14 本地证据，与 v0.13 远端 CI 分开 |
| 新待测包 | 无；v0.9–v0.14 均未生成 challenger |
| 正式复赛数据 | 尚未到 09-21 官方开放时间；旧 test_b 仅 PREVIEW_ENGINEERING_ONLY |
| 桌面副本 | 当前检查路径未找到；仓库 local 原包已核验，不重建或覆盖，见 [补充](docs/optimization_v0_13/DESKTOP_COPY_STATUS.md) |
| 平台有效提交证据 | 已自行检索，未获得账号回执；不宣称资格已确认 |
| 保护标签状态 | November 已在授权生命周期消费；后续为已消费回溯开发 |
| 时间语义 | `competition-timestamp-contract-v1 / ASSUMED`，未新增官方确认 |

[当前发布与推理](docs/optimization_v0_8/CURRENT_RELEASE.md) · [最新实施报告](docs/report.md) ·
[文档索引与历史口径](docs/INDEX.md) · [机器可读状态](EVIDENCE_STATUS.json)

## 最近实验

| 阶段 | 结果 | 决策 |
| --- | --- | --- |
| [v0.14 / OPT-30–31](docs/optimization_v0_14/RESULTS.md) | 0 新基础模型 fit、6+6 时长 LAD；H2 delta E +0.00017521，delta J -0.00015855；342 tests、独立冷审计通过 | FAIL_CLOSE_V7_RETAIN_V1；无 final fit/新包 |
| [v0.13 / OPT-27–29](docs/optimization_v0_13/RESULTS.md) | 0 fit；原 E/J 和 J 贡献重建，旧 A/B stage 预演通过，286 tests | 保留 V1；正式包/回执待核验，不新建候选 |
| [v0.12 / OPT-25–26](docs/optimization_v0_12/RESULTS.md) | 16 次直接目标 fit + 12 次 LAD；V6I/V6T/V6B 的 Delta J 为 +0.00022759 / -0.00062889 / -0.00040130 | 全部完整门槛失败；无 final fit、无新包；随后推送 62c62cc |
| [v0.11 / OPT-24](docs/optimization_v0_11/RESULTS.md) | 8 次直接时长 fit + 6 次 LAD；相对 V1，J 退化 0.00013836，H1 时长 WMAPE 退化 0.00024810 | 关闭固定 V5；无 final fit、无新包 |
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

**2026-09-12 维护状态：暂停公开发布。** 旧说明中的用户仓库授权不能替代赛事主办方的数据公开授权；当前历史含赛事数据，仓库仍为 public。v0.12 已按此前用户明确指令推送；v0.13 随后按用户明确指令推送；v0.14 在本地执行，数据历史处置完成前不公开推送。详见 [数据与发布边界核查](docs/optimization_v0_12/DATA_PUBLICATION_REVIEW.md)。
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
原 v8 专用入口面向 test_a；R2 的 B/C 冷检查是历史工程证据，不等于当前 V1 的 B/C 质量验收。

R2 回退包 SHA-256：`e42602d3045e43b4b49dd1e1c104aa8e5c1f29c639ff5f892b3b07434ed9bbdf`。
恢复方式见 [R2 回退说明](docs/optimization_v0_4/CURRENT_RELEASE.md)。

## V1 的复赛工程预演

新入口显式接收 stage，保留旧 v8 入口和原 bundle。已保存旧 test_b 的 322 行内部冷预测，只是工程预演；正式复赛包身份、质量和平台成绩均未核验。

```bash
uv run --locked --python 3.12 python scripts/optimization_v13_stage_predict.py \
  --stage test_b \
  --bundle local/runs/optimization-v0.8-v1-challenger-r1/bundle \
  --data-config local/runs/optimization-v0.13-opt28-preview-r1/test_b_cold.yaml \
  --output local/predictions/v1-test-b-preview-UNIQUE.csv
```

配置只允许 schema_version=1、test_b_samples、operation_hourly、burden_change、data_dictionary。源契约发生变化时阻断，不删掉检查；不移动旧 cutoff，不补造 December 真值或递归使用预测。入口输出内部三列 CSV 和 local 审计，不生成新 ZIP。
正式包到达后需按 [复赛接入清单](docs/optimization_v0_13/SECOND_ROUND_PROTOCOL.md) 新建身份 manifest，再做冷推理和独立发布验收。

## 因果与保护边界

- operation：`event_time = available_at = clock`；burden：`event_time = available_at = cal_time`。
- 历史目标：`available_at = tap_end_time`，只纳入参考时刻前已可用且符合场景 cutoff 的记录。
- 冻结 baseline 的 development 入口仍拒绝 November 目标。已授权的优化阶段使用自己的访问范围、冻结 manifest 和追加账本；November 已消费，不再称为未触碰 holdout。
- v0.8–v0.12 已评估六个 H1 origins、18-cell 和 DEV_LONG/SHORT；真实 holdout/final-training 生命周期在早期 r2 阶段已执行。
- 时间语义仍是条件性操作约定；官方若改变窗口或报送时点，应新建 contract ID，不能覆盖旧证据。

详细边界见 [数据契约](docs/data_contract.md) 和 [实施范围](docs/task_contract.md)。
[09-11 官方复赛通知](https://www.aicomp.cn/notice/notice-3/5248.html) 已明确复赛每日最多 5 次取最高成绩、算分延迟；不能推断初赛也取最高分。初赛有效回执仍未独立核验，本项目不自动上传。

## 文档与工程入口

```text
configs/                 冻结契约、分阶段配置与发布登记
src/bf_tap/              特征、模型、训练、推理、评估和审计
scripts/                 冷进程复现、环境证据和私有资产检查
tests/                   单元测试与合成端到端测试
docs/                    当前入口文档及各阶段冻结报告
md/                      原始实施包归档，不作为当前状态来源
初赛数据集/              历史含赛事数据，公开处置待单独实施
local/                   本机模型、预测、报告、账本及 ZIP（不入 Git）
EVIDENCE_STATUS.json     current_status 为当前摘要，旧字段保留历史含义
```

[平台记录](docs/submission_log.md) · [发布身份](docs/release_identity.md) · [文档目录](docs/INDEX.md)

本阶段固定 V7 已完成，未登记额外拟合或后续候选。后续实验须独立预注册候选、OOF 边界、预算与门槛；
本轮 V4 的失败不自动推导为所有 residual 方法都无效，也不授权继续参数扫描。
