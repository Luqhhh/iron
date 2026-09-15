# 发布与运行身份

截至 2026-09-15，`platform-probes-r2` 的 V11 反馈收口提交 `468207c` 已推送；最近模型质量分支为 `optimization-v0.15-qrf-time@b01ab117`。V11 仅是后续零拟合工程探针，用户回传 83.1806 后已关闭；上述哈希是语义提交锚点，不是分支最新 HEAD 声明。
活动发布、模型质量结论与初赛实验最高回传是三个不同身份：活动发布继续使用 v0.8 的 V1；最近模型质量 G1 仍为 v0.15 失败；最高用户回传包为 V10 83.1951，未取得独立平台回执，也未改动发布登记。

| 身份 | 当前值 |
| --- | --- |
| 当前模型 | V1_RATE_STRUCTURAL |
| 发布登记 | `configs/optimization_v0_8/active_release.yaml` |
| V1 ZIP SHA-256 | `fdcbe03e8ea31577bfeed0c45cd0a0013bda42ab88d29db703fad9c2f7e557aa` |
| 回退登记 | `configs/optimization_v0_4/active_release.yaml` |
| R2 ZIP SHA-256 | `e42602d3045e43b4b49dd1e1c104aa8e5c1f29c639ff5f892b3b07434ed9bbdf` |
| 最近模型质量决策 G1 | v0.15 `FAIL_CLOSE_V8_RETAIN_V1`，无新发布包 |
| 最新工程验收 G0 | platform-probes-r2 通过；V10 独立复现、V11 封包，根 382 / worker 32 项锁定测试通过 |
| 最高用户回传实验包 | `V10_V6I_IRON_V8_TIME`，83.1951；ZIP SHA-256 `be83f1f623c12f4ef03479f2fcacdfb756920d41c4e09d1c4f6b55a2279761ac` |
| 最近探针 | `V11_V6I_IRON_QRF_MEAN_TIME`，83.1806；`CLOSE_V11_RETAIN_V10` |

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
