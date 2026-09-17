# optimization-v0.26：时长森林划分方式双实验

本阶段从 `cd2e06848f55b993cbb456ae89cbd084ba93f4b3` 分支实施，只改变原时长 QRF 的构树分区：A 使用 `RandomForestRegressor(criterion="absolute_error")`，B 使用 `ExtraTreesRegressor(criterion="squared_error", bootstrap=True)`。两者均固定 256 棵树、原始非负分钟标签、原 210 列 E09/R2 输入、原 v0.15 预处理器、full-original-training 叶分布、较小加权中位数、重新计算的旧 QRF 支持度 gate、V21 收缩和逐字符串不变的 V21 铁量。

P0 同时完成本地 registry 等价实验检索和 v0.25 聚合口径审计。主评价口径固定为 `macro_origin_mean_wmape`；`exposure_pooled_wmape` 只作单独诊断。训练预算为 A/B 各 6 个开发 fit 加 1 个最终 fit，共 3,584 棵树；新预处理 fit、CatBoost/E04/rate、LAD/beta/lambda/偏置 fit 均为零。

两候选的完整定义、模型、预测和 ZIP 必须在任何本轮平台反馈前同时冻结。平台上传、桌面写入和公开推送均需另行明确授权，本 runner 不自动执行。
