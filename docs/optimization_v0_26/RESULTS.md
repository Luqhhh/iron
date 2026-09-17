# optimization-v0.26 执行结果

执行日期为 2026-09-17，基点为 `cd2e06848f55b993cbb456ae89cbd084ba93f4b3`，冻结实现提交为 `78ba01933e8a0e411571a314222fae327ef13444`。两项时长森林划分实验均已完成，G0 工程状态为 **PASS**；G1 离线质量单独报告，两项均保留预登记的平台探索名额。两份 test_a 包已在任何本轮平台反馈前同时冻结。运行本身未上传平台、未写桌面、未推送远端；随后按用户分别明确授权完成桌面替换与 Git 分支推送，仍未由 agent 上传赛事平台。

## P0 与 v0.25 聚合勘误

P0 检索了本地 123 份 manifest，没有发现与 A/B 定义等价的已完成实验。七个 cutoff 均逐字节恢复原 v0.15 handoff：210 列 raw schema（209 个数值列和 `spout_no`），实际 transformed float32 schema 为 213 列；预处理器仅 restore/transform，新 fit 为 0。训练行数依次为 888、1180、1490、1803、2091、2424、2754。

旧 QRF 的预测与 `effective_neighbors` 均从可信原 forest 重新计算，没有复用 V21 的 gate 数组；重算预测与原 v0.15 保存输出逐元素一致。test_a 的旧 gate 为预期 32 行。06–12 cutoff 的 gate 行数依次为 472、452、174、121、43、3、32。

独立 v0.25 审计确认：旧报告表格中的 E 使用正确的 origin 宏平均，但正文的四个分目标 WMAPE 变化使用了跨 origin 合并分母的 exposure-pooled 口径。100 个 algorithm/cell 身份检查均通过；canonical summary E 与重算宏平均最大差为 `5.55e-17`，隔离目标恒等式最大残差为 `1.39e-16`。旧 manifest、预测与平台记录均未改写；勘误见 `docs/optimization_v0_25/AGGREGATION_ERRATA.md`。审计摘要 SHA-256 为 `d4700ac327bab7947b4e76c3a85eedc6ee52820c6b311e5c286df97f9ec262af`。

## 训练与模型身份

A 为 `RandomForestRegressor(criterion="absolute_error", splitter="best")`，B 为 `ExtraTreesRegressor(criterion="squared_error", splitter="random", bootstrap=True)`。两者均使用 256 棵树、`min_samples_leaf=10`、`max_features=0.7`、`random_state=2026` 和 `n_jobs=8`，其余参数按注册配置固定。

每棵树均把全部原始训练行重新投叶，最终预测使用原训练时长响应的跨树加权经验分布和较小加权中位数；没有使用 estimator 的均值预测。训练目标保持原始非负分钟，不裁剪、不中心化、不取对数、不加 recency 权重。两项各完成 6 个开发 fit 和 1 个最终 fit，总计 14/14 fit、3,584 棵树；CatBoost、E04、rate、LAD/beta/lambda/偏置 fit 均为 0。

A 的七次训练累计约 159.97 秒，B 累计约 17.37 秒；记录到的进程峰值分别为 191,940 KiB 与 190,084 KiB。A 的绝对误差划分成本明显高于 B，但没有因此改变树数或叶子参数。

## 离线结果（主口径）

下表为 `macro_origin_mean_wmape`，变化量均为候选减 V21_REPLAY，负数表示改善。预测先六位序列化，标签不舍入；J 为 H1–H4 的宏平均 E 等权平均，DEV 不进入 J。

| 候选 | H1 ΔE | H2 ΔE | H3 ΔE | H4 ΔE | J | J Δ vs V21 | DEV_LONG ΔE | DEV_SHORT ΔE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V26A_QRF_ABSOLUTE_SPLIT_TIME | +0.00073446 | +0.00046446 | +0.00051282 | -0.00041929 | 0.16931966 | +0.00032311 | -0.00008853 | +0.00071081 |
| V26B_EXTRA_RANDOM_SPLIT_TIME | -0.00015213 | -0.00025159 | +0.00024038 | +0.00046716 | 0.16907251 | +0.00007595 | -0.00069828 | +0.00104070 |

V21_REPLAY 的 J 为 0.16899655，V1 为 0.16573254。A/B 的 J 均略差于 V21，也均差于 V1；B 是两项中离线更接近 V21 的候选。两项的铁量数组和最终 CSV 字符串都与 V21 完全一致，所有 cell 与宏平均均满足 `ΔWMAPE_iron=0` 和 `ΔE=0.5ΔWMAPE_time`，最大残差不超过 `1e-12`。

`exposure_pooled_wmape` 已单独写入诊断 CSV，未与主口径混用。共享 calendar-week bootstrap 的有效 draw 为 975/1000；相对 V21 的 J ΔE 95% 描述区间为 A `[+0.00012906,+0.00051841]`、B `[-0.00048147,+0.00050480]`。这是已消费回溯的稳定性描述，不是独立显著性验证，也不取消平台名额。

## 最终包

| 顺序 | 候选 | result.csv SHA-256 | ZIP SHA-256 | 相对 V21 |
| --- | --- | --- | --- | --- |
| A | V26A_QRF_ABSOLUTE_SPLIT_TIME | `9b42d41778fa23dfb07e825ae2ad44461e4d9886c11ea08e23a2dc553428cbad` | `e3aaaacd426b4113e120d8e3e90dd3e41d6d7ca71ffac70c6883a26ade60e60f` | 铁量字符串全同；时长改变 248 行 |
| B | V26B_EXTRA_RANDOM_SPLIT_TIME | `d666ac4edb94260e944cdf5a316ae72bbc46ab8b14e2a1267d48adaa2b66b5f5` | `bf16b0fd1d7ce096a19abd9e23ae5e119d84cf43f553545bf8fbd9c173d2de74` | 铁量字符串全同；时长改变 277 行 |

两包各覆盖 335 个唯一 ID，ZIP 内仅含 `result.csv`，所有预测有限非负且为六位小数；A/B 的时长字段有 293 行不同。私有路径为：

- `local/runs/optimization-v0.26-qrf-partition-tests-r1/submissions/V26A_QRF_ABSOLUTE_SPLIT_TIME/Luqhhh_bf_tap_predict_prelim.zip`
- `local/runs/optimization-v0.26-qrf-partition-tests-r1/submissions/V26B_EXTRA_RANDOM_SPLIT_TIME/Luqhhh_bf_tap_predict_prelim.zip`

用户随后明确授权删除桌面旧提交包并替换。两份 v0.25 桌面副本已删除，其 private local 原包仍保留、可恢复；桌面当前只保留：

- `C:\Users\lqh22\Desktop\Luqhhh_bf_tap_predict_prelim_V26A_QRF_ABSOLUTE_SPLIT_TIME.zip`
- `C:\Users\lqh22\Desktop\Luqhhh_bf_tap_predict_prelim_V26B_EXTRA_RANDOM_SPLIT_TIME.zip`

桌面摘要与上表冻结 ZIP 完全一致，每包仅含 `result.csv`。本地交付回执 SHA-256 为 `f9de48d5dc28abb4d73be27e5958de67dd7779f6f5a001072d45e7bc21fa9b88`。该操作是文件交付，不是赛事平台上传。

平台顺序固定为 A→B，每项一个名额。没有生成第三个融合、组合、切片或校准候选。本运行不自动上传或安排恢复提交。

## 平台反馈与决策

用户按冻结的 A→B 顺序一次回传两个分数；未取得 submission ID 或账号原始回执，因此证据等级保持为用户回传、未独立核验。

| 顺序 | 候选 | 用户回传 | Δ vs V21 | Δ vs V10 | 决策 |
| --- | --- | ---: | ---: | ---: | --- |
| A | V26A_QRF_ABSOLUTE_SPLIT_TIME | 83.2828 | +0.0453 | +0.0877 | 提升为当前最高用户回传完整原包 |
| B | V26B_EXTRA_RANDOM_SPLIT_TIME | 83.0240 | -0.2135 | -0.1711 | 关闭，保留 A |

A 比 B 高 0.2588，且超过此前 V21 的 83.2375；本阶段据此保留 A 的完整冻结原包。平台预算按用户回传计为 2/2，剩余 0；agent 自动上传为 0。账号当前有效提交仍未知：如果初赛平台仍采用“最后一次提交生效”，最后测试的 B 可能是当前有效条目；恢复 A 前必须先核对账号状态和剩余提交额度。反馈记录 SHA-256 为 `73572a331e9db156d22519131a05b3379dc2573d1489969f32b343346757fcda`。

## 冷审计、测试与证据

冷审计覆盖旧 QRF 及 A/B 的全部 7 个 cutoff，共 21 组独立进程恢复；A/B 均完成全量、反序、分块、子集和单样本精确一致性检查。RandomForest、ExtraTrees、PartitionForest、旧 QRF、预处理器和校准 fit 尝试均为 0；最终 CSV 与 ZIP 回读一致，只反序列化本运行预登记摘要绑定的私有模型。

锁定 Python 3.12.12 root 测试为 **448 passed**，JUnit SHA-256 为 `e38ca37eccf18bad5bef7c34377d5f378b1e6eee5742b986badba06d663bb4d5`；worker 测试为 **40 passed**（原 v0.15 34 项、v0.25 1 项、v0.26 5 项），JUnit SHA-256 为 `40d7699144e006ed6d8591944fd5933028c9f5dffc16d098322971b3490548a4`。

权威私有运行目录为 `local/runs/optimization-v0.26-qrf-partition-tests-r1`。manifest SHA-256 为 `e703b869cd1cba40568e2951fc8ba504410a5840e1dd579255c59fcbd87a7f12`，completion SHA-256 为 `82e865ce88a5d7bd401ed7bc5006059f6f969d1950d068d2f5661690d12e3898`，冷审计 SHA-256 为 `7be840a294c072f75b42619ba41c60fd5a261d5dc0ddb2c41cc5e865b5567074`。冻结 completion 保持平台反馈前的 `READY_FOR_TWO_EXPLICIT_PLATFORM_SUBMISSIONS`；阶段当前状态为 `COMPLETE_PLATFORM_FEEDBACK_A_PROMOTED_B_CLOSED`。

用户明确授权提交与推送后，分支 `optimization-v0.26-qrf-partition-tests` 已发布到 `origin`；首次发布头为 `471ed27f9e2da6d60a2fb393c09111c3e608b287`。private local 模型、响应、预测、账本、反馈记录和 ZIP 均未进入 Git。
