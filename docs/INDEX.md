# 文档索引与历史口径

后续按用户指令提交推送 platform-probes-r2，并已把原 V11 ZIP 字节一致地复制到桌面：`Luqhhh_bf_tap_predict_prelim_V11_V6I_IRON_QRF_MEAN_TIME.zip`，SHA-256 `f11dc1216c4c0742955a7222b16d65bc9ca4c1fd3b11d1c38fb7416f920b5efa`。原开发阶段桌面写入 0 的冻结记录保持原意；本次独立交付新增桌面副本 1，未重训、未重新封包、未上传平台。V11 成绩待反馈，当前最高用户回传仍是 V10 83.1951。[交付与推送回执](../local/runs/platform-probes-r2-publication-r1/publication_receipt.json)。

本轮 platform-probes-r2 已完成唯一 V11 零拟合探针的本地准备，以及 V10 独立模型推理：旧 test_a 335 行全精度分支和原 result.csv 字节精确复现；旧 test_b 322 行仅工程预演，全量/反序/分块/子集/单样本一致性及 root/worker 拟合保护通过。新模型、预处理、LAD 拟合均 0；V11 铁量原字段保持 V10，时长为同一最终 QRF 的加权均值，六位提交精度改变 335 行。根 Python 3.12 锁定测试 382 项、独立 worker 32 项分别通过。V11 未上传、无平台分数，桌面写入 0；当前最高用户回传仍是 V10 83.1951，旧 G1、发布指针与原包不变，V2/v0.16 暂停。平台剩余次数未知，仅本地准备；手动探针需剩余至少两次。[实施规格](platform_probes_r2/PLAN.md)、[本地完成记录](../local/runs/platform-probes-r2-r1/completion.json)。

截至 2026-09-12，最新本地阶段为 v0.15（FAIL_CLOSE_V8_RETAIN_V1，正式包与回执待核验），活动模型仍为 V1（用户回传 83.0319）。
当前初赛最高用户回传为 **V10（V6I 铁量＋V8 时长）83.1951**，比 V8 高 0.0315、比 V1 高 0.1632 分，与封包前预期在显示四位小数上一致。[V10 后续反馈](../local/runs/optimization-v0.15-v10-platform-feedback-r1/platform_feedback.json) 绑定原 335 行 ZIP，登记新增拟合为 0，原开发 FAIL 与发布指针不改。
V8 用户实验曾独立完成 1 forest＋1 preprocessor，回传 83.1636；三条回收实验另记 2 CatBoost＋3 LAD，回传 V6I 83.0634、V6T 82.9852、D1 82.9993。证据与原包仅存 local，v0.16 暂缓；不由初赛反馈外推复赛。v0.15 代码已按用户指令推送 6d19a29。
已按用户指令删除桌面五份提交 ZIP，local 原包与旧证据保留。[清理与平台测试复核](../local/runs/optimization-v0.15-feedback-push-cleanup-r1/cleanup_receipt.json)：无必须追加测试，D2 为零新增训练的第一可选对照，V2 为需单独注册拟合的第二梯队。
阅读当前状态时按以下入口；各阶段冻结计划和结果保留运行当时的含义。

## 当前维护文档

- [README](../README.md)：状态、环境、当前 V1 推理入口与数据权限。
- [实施报告](report.md)、[实施范围](task_contract.md)：最新 G0/G1、已消费状态和关闭决策。
- [当前 V1 发布](optimization_v0_8/CURRENT_RELEASE.md)、[R2 回退](optimization_v0_4/CURRENT_RELEASE.md)。
- [用户指定 V8 实验包平台反馈](../local/runs/optimization-v0.15-v8-user-test-a-r1/platform_feedback_r1.json)：包摘要、用户成绩、与 V1 差值及未核验回执状态。
- [V10 最高用户回传包反馈](../local/runs/optimization-v0.15-v10-platform-feedback-r1/platform_feedback.json)：83.1951、原包身份、相对 V8/V1 的增益和四位显示分数加法一致。
- [V6I 更正成绩与 V10 零拟合组合](../local/runs/optimization-v0.15-v10-target-composition-r1/completion.json)：V6I 83.0634、原列精确组合、桌面包与理论分数；[三包回收记录](../local/runs/optimization-v0.15-platform-recovery-r1/completion.json) 的 2 CatBoost＋3 LAD 单独记账。
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
