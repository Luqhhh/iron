# v0.33：时长 OOB-QRF 的特征抽样与单树样本抽样

状态：预登记实施规格；结果见同目录 `RESULTS.md`（执行完成后生成）。基点为 `optimization-v0.32-same-spout-oob-responses@f6eb91aebd0b827a0aac1d2c7f161a354f44c0fe`。

本轮同时冻结两个 time-only 候选。A `V33A_OOB_TIME_FEATURE_THIRD` 仅把 RandomForestRegressor 的 `max_features` 从 0.7 改为 binary64 的 `1/3`；B `V33B_OOB_TIME_HALF_BOOTSTRAP` 仅把 `max_samples` 从 `None` 改为 0.5。两项都从头训练 256 棵 `absolute_error`、`best` 树；A 不叠加半量抽样，B 不叠加 1/3 特征抽样。

输入严格复用 v0.15 的 210 列 handoff 与已拟合预处理器（213 列 float32），七个 cutoff 的 N 为 888、1180、1490、1803、2091、2424、2754。A 每树抽 N 次；B 实际 m 为 444、590、745、902、1046、1212、1377。总预算为 14 次森林 fit、3584 棵新树；预处理器、铁量、CatBoost、校准拟合均为 0。

新 `QRF_SAMPLED_FOREST_OOB_LEAF_RESPONSE_v033` 附件直接读取拟合模型的 `estimators_samples_`，分别登记 N 与 m。每棵树把全部 N 个唯一训练行投叶；非空 OOB 叶使用各唯一 OOB 行一次，空 OOB 叶才回退同一 full leaf。响应保持每树等权、叶内等权，并调用原 `distribution_weights` 与 `lower_median`。旧 v0.29 N 长协议不修改。

新 raw 中位数先六位 round-trip，再重放原 v0.15 support gate、V21 查询日前 60 日合法历史中位数和 0.25 收缩。两个候选的铁量均逐字符串复制 V30A。历史主对照为 V30A，同时报告 V1 与 V21_REPLAY；H1 为本轮主要观察项，H1–H4、J、DEV 均完整保存。历史标签已消费，离线结果不是平台 PASS。

执行顺序固定为 register → prepare → develop → score → finalize → cold。两项最终模型、OOB 附件、预测和包在任何新平台反馈前共同冻结。平台顺序 A→B，各一次；不得根据 A 反馈改 B，不生成组合或第三候选。自动上传、桌面写入、远端推送均为 0，除非用户之后另行明确授权。
