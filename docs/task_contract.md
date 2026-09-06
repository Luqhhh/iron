# 当前实施范围

本轮在固定父提交 `3f94a9892bf5997746624672ea512ff0e7067495` 上完成 R1–R8 与不改变模型的 P2 工程收口。CatBoost 参数、800 轮、MAE、特征窗口、非负裁剪和质量门槛均未调整。

DEV_LONG/DEV_SHORT 已使用新 run ID 在锁定环境完成两次干净重训。H1–H4、保护集评分和正式最终训练均未执行；11 月目标仍封存。当前状态为本地 G0 工程通过、G1 因 DEV_LONG 失败，发布口径是 `BASELINE_REPRODUCIBLE_QUALITY_FAILED`，不是高质量 baseline 验收通过。

字段可用性使用带证据摘要的 `competition-timestamp-contract-v1`，状态为 `ASSUMED`。官方澄清若改变时点含义，必须创建新 contract ID 并使相关 bundle/run 失效，不覆盖历史记录。
