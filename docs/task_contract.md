# 当前实施范围

本轮在固定父提交 `3f94a9892bf5997746624672ea512ff0e7067495` 上完成 R1–R8 与不改变模型的 P2 工程收口。CatBoost 参数、800 轮、MAE、特征窗口、非负裁剪和质量门槛均未调整。

DEV_LONG/DEV_SHORT 已使用新 run ID 在锁定环境完成两次干净重训。H1–H4、保护集评分和正式最终训练均未执行；11 月目标仍封存。当前状态为本地 G0 工程通过、G1 因 DEV_LONG 失败，发布口径是 `BASELINE_REPRODUCIBLE_QUALITY_FAILED`，不是高质量 baseline 验收通过。

字段可用性使用带证据摘要的 `competition-timestamp-contract-v1`，状态为 `ASSUMED`。官方澄清若改变时点含义，必须创建新 contract ID 并使相关 bundle/run 失效，不覆盖历史记录。

两个后续 P1 已完成：公共 process source 内容与 bundle v3 强绑定，三份执行配置由 canonical digest 完整冻结。`baseline-v0.1-reproducible` 标签形成后 baseline-v0.1 不再原地修改；质量研究转入独立 optimization-v0.2。

synthetic protected lifecycle、外部 release manifest 与正式 candidate clean-tree gate 是已记录 P2，不阻塞优化，也不授权读取 11 月标签。
