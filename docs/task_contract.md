# 当前实施范围

本轮实现 M0–M3，以及 M4 固定 CatBoost 模型接口与合成 smoke test。禁止调参、特征搜索、融合、伪标签或读取保护集反馈。

真实数据的赛事时间戳操作口径已冻结为 `competition-timestamp-contract-v1`，允许执行 DEV_LONG/DEV_SHORT。HOLDOUT 仍须先完成冻结 manifest，且不得在开发流程读取 11 月标签。
