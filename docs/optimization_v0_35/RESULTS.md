# optimization-v0.35 结果：时长叶点的两项独立聚合实验

执行日期：2026-09-20。分支：`optimization-v0.35-time-leaf-location-and-vote`。完整私有运行：`local/runs/optimization-v0.35-time-leaf-location-and-vote-r1`，manifest 绑定实现提交 `282685a270f17de8a7cd37da8f27e8afb80f6106`。

## 结论

两项事前冻结的 time-only 候选均完成，G0 工程状态 **PASS**，G1 历史质量相对当前父 V34T 均退化：

- A `V35A_OOB_MIDPOINT_LEAF_MEAN_TIME`：同一 selected leaf 保存中位区间 lower/upper，以 512 个 binary64 端点的精确有理和形成等树中点均值；
- B `V35B_OOB_LOWER_MEDIAN_OF_LEAF_POINTS_TIME`：完整复用 v0.34 lower 表，对 256 个树点取排序位置 127 的较小中位数。

两项均复用 V26A 的 256 树时长森林、v0.29 时长 OOB selected members、原 v0.15 gate 与 V21 收缩；铁量逐字符串复制 V34T。没有新增森林、树、预处理器、CatBoost 或校准 fit。A 新建 7 份监督中位区间表；B 新建 lower 表 0 份、只读复用父 lower 表 7 份。没有读取 test target、上传平台、写桌面或生成第三候选。

## 离线主口径

主口径为 `macro_origin_mean_wmape`；变化为候选减参照，负数表示历史误差改善：

| 候选 | 参照 | H1 ΔE | H2 ΔE | H3 ΔE | H4 ΔE | J ΔE | DEV_LONG ΔE | DEV_SHORT ΔE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V35A | V34T | +0.00162687 | +0.00239570 | +0.00327048 | +0.00350379 | +0.00269921 | +0.00233149 | +0.00260606 |
| V35B | V34T | +0.00037616 | +0.00034261 | +0.00028580 | +0.00060225 | +0.00040170 | +0.00033001 | +0.00100204 |
| V35A | V30A | -0.00038224 | -0.00019951 | -0.00005979 | -0.00013723 | -0.00019469 | +0.00004728 | -0.00090081 |
| V35B | V30A | -0.00163295 | -0.00225260 | -0.00304447 | -0.00303877 | -0.00249220 | -0.00195420 | -0.00250483 |
| V35A | V1 | +0.00123209 | +0.00289448 | +0.00500996 | +0.00610581 | +0.00381059 | +0.00204295 | +0.00493040 |
| V35B | V1 | -0.00001862 | +0.00084139 | +0.00202528 | +0.00320427 | +0.00151308 | +0.00004147 | +0.00332638 |

A/B 在 H1–H4、J 和两个 DEV 上都没有超过当前父 V34T。B 仍明显优于旧 V30A，但本轮决策基准已经是 V34T，不能把“超过旧父”登记为新胜出。历史标签已消费，结果不构成独立平台证据；按事前登记，两项仍保留各一个探索名额。

两项都是 time-only。目标隔离恒等式覆盖 40 个 cell 与 50 个宏平均摘要，最大绝对残差 `2.78e-17`，低于 `1e-12`：铁量 WMAPE 变化为 0，且 `Delta E = 0.5 × Delta W_time`。

## 最终诊断

最终 test_a 输入覆盖 335 行：

| 项目 | A：叶内 midpoint | B：跨树 lower vote |
| --- | ---: | ---: |
| 相对父 raw 增量范围 | +1.435547 至 +3.621094 min | -1.699219 至 +2.621094 min |
| raw 增量中位数 | +2.171875 min | -0.148438 min |
| 非零中位区间树数/查询（最小/中位/最大） | 98 / 125 / 143 | — |
| 不同 lower 树点数/查询（最小/中位/最大） | — | 42 / 52 / 69 |
| 最终时长改动行数 | 335 / 335 | 332 / 335 |
| 最终时长差值范围 | +1.166015 至 +3.521484 min | -1.699219 至 +2.621094 min |
| 最终时长差值中位数 | +2.121094 min | -0.140625 min |

A 的 raw 和最终时长均逐行不低于 V34T，满足登记的不变量。B 的方向可正可负，没有据方向选择样本。两者铁量字符串逐行与 V34T 相同，且两份最终输出彼此不同、均非父方案 NO-OP。

## 冷审计与测试

独立 cold 进程覆盖 A/B × 7 cutoff。A 从相同响应和 v0.29 members 重建 lower/upper 表；B 从相同来源重派生 lower 表并与 v0.34 父表逐数组核验。两项 parent raw 与最终后处理均精确恢复 V34T，并通过 full/reverse/chunk/subset/single 和 CSV/ZIP 回读。所有 fit 拦截计数为 0。

锁定 Python 3.12.12 root 套件为 **526 passed**；v0.35 worker 合成测试为 **18 passed**；v0.29/v0.34/v0.35 相关 worker 回归为 **20 + 17 + 18 passed**。外部附件的 39 项参考测试未计入仓库测试数。私有产物 guard 通过。

关键证据 SHA-256：

- `manifest.json`: `d73177b85a9c1f83d0f1e747389c641d8eca58edc5222d770cc8a16847b768f6`
- `offline_assessment.json`: `bc6fcaa575965c640c50522eeecf430f467bb3937c37e2decbdeb69e5d1d670e`
- `fit_counts.json`: `5ecc4b4774b24c947bcd8002970313e4e080782d21fc3d28b90932441665fd75`
- `cold_validation.json`: `6a1fea446669f1103cf9d4e26cf3722cd4d19f6589d2f20d13918e9dbb1438da`
- `completion.json`: `0f99a74ad2d1714f6eba3057c421ee7299ae4bbc992b770d99868ff48e0abb17`

## 冻结包与提交边界

| 槽 | 候选 | result.csv SHA-256 | ZIP SHA-256 |
| --- | --- | --- | --- |
| A | V35A_OOB_MIDPOINT_LEAF_MEAN_TIME | `34fc6c354f726dbbca41c32950ef6d22e8040ed1fded0e0444cea8e888999d84` | `faf3ed009904f9253d3bd178c3b3b5b27bed60f743d4f7bad1ec70875d7cb63e` |
| B | V35B_OOB_LOWER_MEDIAN_OF_LEAF_POINTS_TIME | `246336fd00a5774c4b2101150aca27430eba8b9e20e7ef6d94e92e5f749f55a4` | `11f9cd5a8dfb4eec846d4f60789ff1b211a3ed32ce3764d5642ed563f4b746a4` |

两份 ZIP 各只含 UTF-8 `result.csv`，覆盖 335 个唯一 ID，预测有限、非负、六位小数；已在任何 v0.35 平台反馈前同时冻结。当前最高用户回传完整包仍为父 V34T=83.3201，ZIP SHA-256 `e3f970fa96cad54e0a6c534cc473d3b19eeb88269cb8c2630411796bfa313924`。

状态为 `READY_FOR_EXPLICIT_PLATFORM_SUBMISSIONS`，顺序 A→B、预算 2；agent 上传、桌面写入、恢复上传均为 0。赛段和账号状态尚未由本次运行核验，不能把旧 test_a 包改名跨赛段提交。

## 桌面交付（2026-09-20 追加）

用户随后明确要求写入桌面。两份冻结 ZIP 已采用不覆盖模式复制到 `C:\Users\lqh22\Desktop`：

- `Luqhhh_bf_tap_predict_prelim_V35A_OOB_MIDPOINT_LEAF_MEAN_TIME.zip`
- `Luqhhh_bf_tap_predict_prelim_V35B_OOB_LOWER_MEDIAN_OF_LEAF_POINTS_TIME.zip`

桌面摘要分别为 `faf3ed009904f9253d3bd178c3b3b5b27bed60f743d4f7bad1ec70875d7cb63e`、`11f9cd5a8dfb4eec846d4f60789ff1b211a3ed32ce3764d5642ed563f4b746a4`，与冻结源包逐份一致；每个 ZIP 只包含 `result.csv`。没有覆盖或删除任何文件，平台上传仍为 0。私有交付回执为 `desktop_delivery_receipt.json`。

## 平台反馈（2026-09-20 追加）

用户明确标注 35B 并回传平台显示成绩 **83.2778**；未取得 submission ID 或账号原始回执，证据状态为 `USER_REPORTED_NOT_INDEPENDENTLY_VERIFIED`。该分数比当前父方案 V34T=83.3201 低 `0.0423`，比旧 V30A=83.3175 低 `0.0397`，因此关闭 B 并继续保留 V34T 原高分包。

实际反馈顺序为 B 先于 A，与预登记 A→B 不同；偏差已如实记录。两份候选在反馈前已经同时冻结，B 的定义、预测和 ZIP 没有因任何本轮平台反馈改变。A 截至本记录尚未提交，冻结包仍保留；考虑到 A 相对 V34T 的历史 J 退化 `+0.00269921`，约为 B 退化幅度的 6.7 倍，当前建议跳过 A，除非仅为补齐实验且已确认有安全恢复 V34T 的机会。这是提交建议，不把未测试的 A 伪记为平台关闭结果。

本轮用户回传平台测试计数为 1，agent 上传仍为 0；计划实验槽剩余 1，但账号实际额度未知。当前最高用户回传仍为 V34T=83.3201，最新用户回传为 V35B=83.2778，账号当前生效条目仍因缺少回执登记为未知。私有反馈记录 `platform_feedback_B_user_reported.json` 的 SHA-256 为 `88a9d4412d07472c222e49e9399c1238464f82e4632a9c78d98faeaa14576110`。
