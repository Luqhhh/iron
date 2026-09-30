# 发布与运行身份

更新日期：2026-10-01。当前复赛平台代表与两项探索包如下；仅引用已登记身份，未在本次维护重训或改包。分数是用户回传，未独立平台核验；动态状态读取 `round2_current_platform_best` 与 `round2_current_candidate_queue`。

| 身份 | 状态 / ZIP SHA-256 |
| --- | --- |
| EMA_TIME_Q75 | 96.3920，同分最佳的代表；`41a046d5ce74e8a9c7c9acb124fa680cabf9e87e4a79edb50d81efb625bde825` |
| EMA_TIME_Q100 | 96.3920，同分最佳；`a102cbf123981ca3103b5ba24f78b8b36203bee29810c6f0b68c6aad8ca139cb` |
| PTARL_TIME_Q20 | 已审计、平台待反馈；`a6305ef143b81e19dc57f8422e360d751e0aa27280b4b955075b52b7083c4c6d` |
| EMA_IRON_EMA_TIME | 已审计、平台待反馈；`88a3602efe5145428693155896e2ce1ec029f5241e56f2ca1a90b942c177c4a0` |
| DE3_IRON_EMA_TIME_Q75_RESERVE | 替补、暂不平台测试；`86bf20d8cbe938f06b7550a3cf50e28fd100ba94a672c666d0aee576f25bcee3` |

准确路径、公式与保留列见 [EMA 交付](ema_time_followup/DELIVERY.md) 和 [PTaRL/EMA 交付](ptarl_ema_exploration_release/DELIVERY.md)。初赛 `active_release.yaml` 指针与基线标签仍保留原身份，不代表当前复赛平台最佳；不能用旧初赛推理脚本导出这些复赛包。

每次执行记录当时源码/配置/数据/模型/运行环境和预测身份，目录追加、不覆盖。推送分支不等于 CI 或平台质量验证；参考工程与模型质量分别报告。

## 历史初赛发布与基线身份（2026-09-09）

以下提交、活动指针、回退与检查仅适用于原阶段，保持历史含义。

截至 2026-09-09，最近已推送的实验结果提交为 `optimization-v0.10@fcda5e0`。
实验分支最新版与当前活动模型是两个不同身份：v0.10 失败，发布继续使用 v0.8 的 V1。

| 身份 | 当前值 |
| --- | --- |
| 当前模型 | V1_RATE_STRUCTURAL |
| 发布登记 | `configs/optimization_v0_8/active_release.yaml` |
| V1 ZIP SHA-256 | `fdcbe03e8ea31577bfeed0c45cd0a0013bda42ab88d29db703fad9c2f7e557aa` |
| 回退登记 | `configs/optimization_v0_4/active_release.yaml` |
| R2 ZIP SHA-256 | `e42602d3045e43b4b49dd1e1c104aa8e5c1f29c639ff5f892b3b07434ed9bbdf` |
| 最新质量决策 | v0.10 V4 失败，无新发布包 |

推理使用与发布登记匹配的 bundle 和专用入口，见 [V1 发布说明](optimization_v0_8/CURRENT_RELEASE.md)。
每次新实验冻结源码、配置、公共源、模型和预测身份；目录不可覆盖。历史 run 的
commit/tree/dirty 状态不因后续提交而回写。当前优化实现已使用 feature cache，
具体边界和键由各阶段 manifest/实现声明，不能把旧 baseline 的 cache 状态推广到全部入口。
远端分支推送不代表已经通过 CI；远端治理状态不由本地测试推断。

## 历史 baseline 修复与冻结身份

- 修复基线固定父提交：`3f94a9892bf5997746624672ea512ff0e7067495`。
- 修复工程提交：`4d6c2be4420d631e8383b212b6fdab036031bc5b`，已推送至 `origin/main`。
- baseline 工程冻结标签：`baseline-v0.1-reproducible`；该标签绑定两个 P1 收口后的不可变基线，后续优化不得移动或复用此标签。
- 原运行 `dev-baseline-v0.1-contract-v1-r2` 是该提交前工作树产生的本地记录；其报告中的 `uncommitted`/`not pushed` 是运行当时事实，不回写为“已在 3f94a98 上运行”。
- `3f94a98` 将原实现和原报告关联到首次公开版本，但不是原运行的事后伪造执行身份。
- 修复后的 run 必须使用新 ID，并记录当时 commit/tree、dirty 状态、源码快照、锁文件、输入、分区、历史授权集合、模型和预测摘要。
- 旧 baseline 修复时 `feature_cache_key` 只有纯函数与单测；DEV/训练/推理均未启用缓存 I/O，run manifest 必须写 `cache.enabled=false`。

GitHub `main` 的分支保护和必需检查应由仓库管理员在平台设置中启用；本次代码修复不把平台治理状态当作 G0 正确性证据。
