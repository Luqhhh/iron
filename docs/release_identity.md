# 发布与运行身份

- 修复基线固定父提交：`3f94a9892bf5997746624672ea512ff0e7067495`。
- 修复工程提交：`4d6c2be4420d631e8383b212b6fdab036031bc5b`，已推送至 `origin/main`。
- baseline 工程冻结标签：`baseline-v0.1-reproducible`；该标签绑定两个 P1 收口后的不可变基线，后续优化不得移动或复用此标签。
- 原运行 `dev-baseline-v0.1-contract-v1-r2` 是该提交前工作树产生的本地记录；其报告中的 `uncommitted`/`not pushed` 是运行当时事实，不回写为“已在 3f94a98 上运行”。
- `3f94a98` 将原实现和原报告关联到首次公开版本，但不是原运行的事后伪造执行身份。
- 修复后的 run 必须使用新 ID，并记录当时 commit/tree、dirty 状态、源码快照、锁文件、输入、分区、历史授权集合、模型和预测摘要。
- `feature_cache_key` 目前只有纯函数与单测；DEV/训练/推理均未启用缓存 I/O，run manifest 必须写 `cache.enabled=false`。

GitHub `main` 的分支保护和必需检查应由仓库管理员在平台设置中启用；本次代码修复不把平台治理状态当作 G0 正确性证据。
