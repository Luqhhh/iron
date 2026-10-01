# 项目文档索引

更新日期：2026-10-01。执行规则见 [AGENTS.md](../AGENTS.md)；机器状态见 [EVIDENCE_STATUS.json](../EVIDENCE_STATUS.json)。本文的分数和交付状态是该日期的登记快照，后续以状态文件为准。README 按用户要求保持不变，不作为最新队列入口。

## 当前复赛状态

| 项目 | 最新登记 |
| --- | --- |
| 平台回传同分最佳 | **EMA_TIME_Q75 / EMA_TIME_Q100 = 96.3920**；用户回传，未独立平台核验 |
| 当前代表 | Q75，组件外推较小；未证明统计上独胜 |
| 目标 | **96.4 → 96.45 → 96.5**，尚差 0.0080 / 0.0580 / 0.1080 |
| 已完成的两项全量探索 | PTARL_TIME_Q20用户回传96.378，低于Q75；EMA_IRON_EMA_TIME已桌面交付，仍待用户反馈 |
| 替补 | **DE3_IRON_EMA_TIME_Q75_RESERVE 暂不平台测试**，等待信息量或收益更值得名额的候选 |
| 其他旧包 | 未回传不等于继续待测；Q25、旧 Q50 组合和更早包不因存在文件而自动恢复优先级 |
| 初赛历史最高 | V30A_OOB_BOTH_TARGETS = 83.3175；与复赛成绩分开，不是当前复赛参照 |

两项新探索的公式、包身份及 G0/G1 见 [PTaRL/EMA 交付](ptarl_ema_exploration_release/DELIVERY.md)。它们不是新增四切分正式晋级；不生成第三个双目标组合，用户自行上传。本文不推断剩余额度或规定新的上传顺序。

## 后续筛选与验证

**本地收益不能作为唯一指标。** 本地和平台的符号、排序、幅度、最佳权重可能不同；切分稳定性不等于独立数据泛化保证。SAM 时长本地 −0.01643、平台 +0.0031；EMA 时长本地 +0.00217、平台 +0.0168。Q75/Q100 相对 Q50 的本地两切分均负，平台均 +0.0025。

保留完整同协议评估、当前目标参照、增量融合收益和冻结门槛；正式晋级仍要求至少四个完整 split seed、各 seed 正收益及 seed 层配对 LCB95 > 0。平台探索另行登记理由与授权，不追溯改写失败决定。不用固定偏移、放大倍数或条件列加法冒充平台预测。

优化不设时间预算，运行任务按 600 秒定时观察。数据保护、无泄漏、资源/数值门槛、追加式证据、独立冷推理与未修改列字符串检查仍有效。旧暂停、小时监控、时间拒绝和队列仅是历史记录。

## 当前维护入口

- [实施报告](report.md)、[任务范围](task_contract.md)、[发布身份](release_identity.md)、[提交与反馈记录](submission_log.md)。
- [候选分类及正式晋级边界](candidate_tiers.md)、[数据契约范围](data_contract.md)、[待确认语义与平台口径](rule_questions.md)。
- [四项本地/平台诊断反馈](local_platform_diagnostic_release/DELIVERY.md)：四项均已回传，SAM 时长反转，EMA 时长获益。
- [EMA 稀疏权重反馈与替补安排](ema_time_followup/DELIVERY.md)：Q75/Q100 已回传，DE3＋Q75 仅作替补。
- [PTaRL 相对 Q75 的零拟合诊断](ema_evaluation_diagnostics/RESULTS.md)：两切分描述性证据，不预测平台分数。
- [PTaRL 时长 / EMA 铁量全量交付](ptarl_ema_exploration_release/DELIVERY.md)：PTaRL时长用户回传96.378，EMA铁量待反馈。
- [Q75误差地图](q75_error_relocation/RESULTS.md)及[压差时长校准完整开发](q75_error_relocation/CALIBRATION_RESULTS.md)：G0通过，压差两切分均负、全局一负一正，无确认候选及新平台包。
- [EMA时长平均跨度预登记](ema_average_span/PREREGISTRATION.md)及[执行记录](ema_average_span/EXECUTION.md)：1358项锁定检查通过，两个完整切分串行开发已启动，G1及新保存状态审计待完成。
- [DE3 铁量历史交付与回传](de3_user_release/DELIVERY.md)：96.3749，相对 V32 +0.0022。

`current_status` 是项目摘要，`round2_current_platform_best` 是平台最佳，`round2_current_candidate_queue` 是当前候选安排。旧初赛摘要和旧队列完整保存在各自的 `history_before_documentation_refresh_20261001`；其他阶段条目保持原运行时含义。

## 历史文档阅读范围

`optimization_v*/`、编号 `round2_v*/`、`round2_next_phase/`、`round2_final_top5/`、`round2_slots_20260930/`、`review/` 及 `md/` 记录阶段当时的计划、参照、结果或交付，不是实时队列。预登记、实验配置、模型配方和失败门槛保持原样；不要照旧文恢复暂停、提交顺序、旧预算或旧“当前最佳”。原始历史内容和身份保留，不以文档更新重新授予拟合或发布资格。

- [Top5 历史反馈](round2_final_top5/FEEDBACK.md)：AJ3 的首选身份限于 2026-09-23 批次。
- [V3.4 历史得分转移与门槛](round2_v3_4/SCORE_TRANSFER_AND_NEXT_TARGET.md)：96.25 是原阶段门槛，不是所有探索的统一否决线。
- [V5 结果及判读更正](round2_v5/RESULTS.md)：0.0005 是单次扰动效应，不是平台分辨率；方向/排序保证已撤回。
- [V6 历史线搜索及更正](round2_v6/RESULTS.md)：固定放大推出的“96.35 不可达”已作废；旧五包顺序不再作为当前队列。
- [初赛 V1 发布入口](optimization_v0_8/CURRENT_RELEASE.md)、[R2 历史回退](optimization_v0_4/CURRENT_RELEASE.md)、[编号与身份修复](round2_round_numbering.md)。

## 初赛及基线阶段归档


| 阶段 | 冻结结果 | 阅读口径 |
| --- | --- | --- |
| v0.32 | [同铁口优先的 OOB 响应条件化](optimization_v0_32/RESULTS.md) | 0 fit；固定 v0.29 members 与每树等权，仅在原集合内同铁口优先；A/B 相对 V30A 的历史 ΔJ +0.00006051/-0.00018334；G0 PASS；平台回传 83.2764/83.0910，预算 2/2，两项关闭并保留 V30A，agent 上传 0 |
| v0.31 | [两个目标隔离的 OOB 叶响应汇总实验](optimization_v0_31/RESULTS.md) | 0 fit；14 份 v0.29 OOB 附件复用；整数 occurrence 质量池化；G0 PASS；用户回传 I=83.3123、T=83.3116，均低于 V30A=83.3175，两项关闭并保留当前最高 V30A；平台预算 2/2、agent 上传 0 |
| v0.30 | [双目标 OOB 收益组合与时长森林固定扩容](optimization_v0_30/RESULTS.md) | G0 PASS；7 次追加 fit/5,376 棵新树（2 个已完成拟合显式恢复、0 re-fit）；A 列组合恒等式残差 ≤5.55e-17；平台用户回传 A=83.3175（晋级，与加性推算一致）、B=83.2654（关闭），预算 2/2、agent 上传 0 |
| v0.29 | [冻结森林 OOB 叶响应双实验](optimization_v0_29/RESULTS.md) | 0 fit；14 份 OOB 附件；G0 PASS；平台回传 A/B=83.2970/83.3141，预算 2/2，B 晋级为当前最高；离线排序不回写 |
| v0.28 | [固定等权、双目标隔离集成](optimization_v0_28/RESULTS.md) | 0 fit；A 平均 V26A/V27I 铁量，B 平均 V26A/V21 时长；G0 PASS，平台回传 83.2936/83.2604，预算 2/2，A 晋级、B 关闭 |
| v0.27 | [铁量 QRF 与时长叶内 recency](optimization_v0_27/RESULTS.md) | G0 PASS；A/B 用户回传 83.2480/83.2710，均低于 V26A=83.2828，固定候选关闭 |
| v0.26 | [时长森林分区双实验](optimization_v0_26/RESULTS.md) | G0 PASS；V26A 用户回传 83.2828 并晋级为当前最高，V26B 83.0240 关闭 |
| v0.25 | [历史基准中心化双目标实验](optimization_v0_25/RESULTS.md) | G0 PASS；7 centered CatBoost + 7 signed QRF、0 新预处理器/校准；A 相对 V21 的 J 退化 +0.00039378，B 改善 -0.00221680 但仍不及 V1；平台用户回传 83.1516/83.0117，预算 2/2，两项关闭并保留 V21 |
| v0.23 | [恢复、统一参照与复赛完整算法预演](optimization_v0_23/RESULTS.md) | 0 fit；V21 原 ZIP/payload 恢复；六个历史 replay 逐字节复验；四算法统一六位 scorecard；M-only 数值对照优于 V22；旧 test_b 的 V1/V21/V22 双进程冷推理一致，正式复赛身份仍待核验 |
| v0.24 | [双目标隔离变料历史特征实验](optimization_v0_24/RESULTS.md) | 7 CatBoost + 7 QRF/preprocessor；A/B 相对 V21_REPLAY 的 ΔJ 均小幅退化；平台用户回传 83.1902 / 83.2288，均未超过 V21，固定候选关闭 |
| v0.22 | [V22 因果 H2 QRF 支持度收缩](optimization_v0_22/RESULTS.md) | 2 个 warmup QRF、七份真实 H2 bank、12 个开发 lambda；全部预注册门槛通过；后续 test_a 包冷验通过，用户回传 83.1166 后关闭该候选，V21/V10 保留 |
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
