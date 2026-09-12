# 文档索引与历史口径

截至 2026-09-12，最新本地阶段为 v0.15（FAIL_CLOSE_V8_RETAIN_V1，正式包与回执待核验），活动模型仍为 V1（用户回传 83.0319）。
随后按用户指定独立交付 V8 初赛实验包，用户回传 83.1636，比 V1 高 0.1317 分；原开发验收和发布指针保留。[独立平台反馈](../local/runs/optimization-v0.15-v8-user-test-a-r1/platform_feedback_r1.json) 与训练/封包证据仅存 local。v0.15 代码已按用户指令推送 `6d19a29`。
阅读当前状态时按以下入口；各阶段冻结计划和结果保留运行当时的含义。

## 当前维护文档

- [README](../README.md)：状态、环境、当前 V1 推理入口与数据权限。
- [实施报告](report.md)、[实施范围](task_contract.md)：最新 G0/G1、已消费状态和关闭决策。
- [当前 V1 发布](optimization_v0_8/CURRENT_RELEASE.md)、[R2 回退](optimization_v0_4/CURRENT_RELEASE.md)。
- [用户指定 V8 实验包平台反馈](../local/runs/optimization-v0.15-v8-user-test-a-r1/platform_feedback_r1.json)：包摘要、用户成绩、与 V1 差值及未核验回执状态。
- [桌面副本核查补充](optimization_v0_13/DESKTOP_COPY_STATUS.md)。
- [发布身份](release_identity.md)、[平台记录](submission_log.md)、[数据契约](data_contract.md)。
- [机器状态](../EVIDENCE_STATUS.json)：`current_status` 是当前摘要；其余旧顶层 baseline 字段和版本字段是阶段证据。

## 阶段结果

| 阶段 | 冻结结果 | 阅读口径 |
| --- | --- | --- |
| v0.15 | [OPT-32–33](optimization_v0_15/RESULTS.md) | 固定QRF时长分支6 forest/6 preprocessor、1536树；根364/worker25测试、独立冷审计；十项质量失败、关闭V8，无新包；[工程恢复](optimization_v0_15/ENGINEERING_REPAIR.md)、[维护记录](optimization_v0_15/MAINTENANCE_20260912.md) |
| v0.14 | [OPT-30–31](optimization_v0_14/RESULTS.md) | 原基础模型 0 fit；V7/D1 6+6 时长 LAD；342 tests、独立冷审计；FAIL_CLOSE_V7_RETAIN_V1；[维护观察](optimization_v0_14/MAINTENANCE_20260912.md) |
| v0.13 | [OPT-27–29](optimization_v0_13/RESULTS.md) | 0 fit；E/J/贡献重建、旧 A/B stage 预演通过；286 tests；[维护观察](optimization_v0_13/MAINTENANCE_20260912.md)、[正式接入待办](optimization_v0_13/SECOND_ROUND_PROTOCOL.md)；随后按用户指令推送 2db6d5f，run 34687452551 成功 |
| v0.12 | [OPT-25–26](optimization_v0_12/RESULTS.md) | 三个 recency60 候选完整质量门槛失败；255 tests，G0 冷审计通过；16+12 fits，无新包；随后推送 62c62cc、CI 成功；[发布暂停](optimization_v0_12/DATA_PUBLICATION_REVIEW.md) |
| v0.11 | [OPT-24](optimization_v0_11/RESULTS.md) | V5 四项质量门槛失败；235 tests，G0 修正后冷审计通过；8+6 fits，无新包；已推送 |
| v0.10 | [OPT-23](optimization_v0_10/RESULTS.md) | V4 失败；221 tests，G0 冷审计通过；无新包 |
| v0.9 | [OPT-21/22](optimization_v0_9/RESULTS.md) | ratio 扩展失败；“v0.10 尚未训练”仅描述该轮收口时 |
| v0.8 | [OPT-20](optimization_v0_8/RESULTS.md) | 开发交付时 R2 仍 active；随后 V1 平台回传 83.0319，见当前发布页 |
| v0.7 | [OPT-19](optimization_v0_7/RESULTS.md) | pseudo-history 失败，当时恢复 R2 |
| v0.6 | [S1 / OPT-18](optimization_v0_6/RESULTS.md) | S1 平台失败；“桌面仍 S1”是回传记录当时，后来已替换 |
| v0.5 | [OPT-14/15](optimization_v0_5/RESULTS.md) | 旧组合研究结束，R2 保留；B/C 冷检查不等于 V1 的 B/C 验收 |
| v0.4 | [R2](optimization_v0_4/RESULTS.md) | 当时 R2 晋级，当前回退 |
| v0.3/r2 | [生命周期与回退](optimization_v0_3/OPT10_EXECUTION_R2.md) | November 从此已消费；旧 E16 发布页仅供历史回退 |
| v0.3 | [研究汇总](optimization_v0_3/RESULTS_SUMMARY.md) | 当时的候选、未完成项和平台状态 |
| v0.2 | [OPT-01–06](optimization_v0_2/RESULTS_SUMMARY.md) | E09/E12/E16 的“incumbent”按当时解释 |
| baseline | [冻结报告](review/FREEZE_REPORT.md) | 历史测试数、DEV_LONG 失败和当时未消费状态 |

## 不回写的历史证据

`optimization_v*/PLAN.md`、冻结结果、决策记录、`docs/review/` 和 `md/` 是按阶段
保留的历史。部分文件摘要已经绑定本地 manifest 或完成回执；不能为了更新“当前”
而改写旧实验身份。本轮通过当前入口、阶段说明及回退页消除歧义，保留它们原文。
`AGENTS.md` 的旧 baseline 起点也按其“除非用户明确开启独立优化阶段”条件解释，
不表示项目尚未进入后续优化阶段。`md/` 是原始实施包，不作为执行权威来源。

当前没有已预注册的新模型任务。后续工作需新增阶段配置和预算；旧结果中的建议
或“下一阶段”不是自动重启已经关闭实验的授权。
