# optimization-v0.32 结果：同铁口优先的 OOB 响应条件化

执行日期：2026-09-19。分支：`optimization-v0.32-same-spout-oob-responses`。完整授权本地运行：`local/runs/optimization-v0.32-same-spout-oob-responses-r2`，manifest 绑定实现提交 `1332da7bbb0a0cc4d121a722624e1d4469a85346`。

r1 已在完成预测、评价和封包后，于 cold A/slot 6 的保存数组比较入口失败：实现对字符串数组错误使用 `numpy.array_equal(equal_nan=True)`。没有观察到模型或预测差异、没有拟合或上传；r1 的 manifest、冻结包及 `failure.json` 保留且没有覆盖。修复新增了字符串与浮点 NaN 数组比较回归测试，r2 从新提交完整重跑。

## 结论

两项候选均按冻结定义完成，G0 工程状态 **PASS**：

- A `V32I_SAME_SPOUT_OOB_BLEND` 仅改变铁量，在原 v0.29 selected members 内优先使用同铁口响应，再与原 V26A 完整铁量端点做冻结的 50/50 `mean6`；V30A 时长逐字符串不变。
- B `V32T_SAME_SPOUT_OOB_TIME` 仅改变时长，同铁口条件化后重放原 v0.15 gate、V21 60 日中位数与 0.25 收缩；V30A 铁量逐字符串不变。

新增模型、树、预处理、CatBoost、LAD、beta、lambda、bias 与校准 fit 全部为 **0**。14 份 v0.29 OOB 附件被只读恢复，没有创建新附件；14 份训练铁口向量按模型与预处理器 ID 对齐。没有读取 test target、没有递归写回预测、没有平台上传、桌面写入或公开推送。

## 离线主口径

主口径为 `macro_origin_mean_wmape`，变化为候选减 V30A，负数表示历史误差改善：

| 候选 | H1 ΔE | H2 ΔE | H3 ΔE | H4 ΔE | J ΔE | DEV_LONG ΔE | DEV_SHORT ΔE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V32I_SAME_SPOUT_OOB_BLEND | +0.00010914 | -0.00001024 | +0.00006715 | +0.00007598 | +0.00006051 | +0.00003753 | -0.00001347 |
| V32T_SAME_SPOUT_OOB_TIME | +0.00004381 | -0.00055842 | -0.00032251 | +0.00010376 | -0.00018334 | -0.00016961 | -0.00061728 |

A 的 J 相对父方案小幅退化，B 小幅改善；两项相对 V1 的 J 仍分别退化 `+0.00406579`、`+0.00382194`。历史标签已消费，OOB 不是独立验证集，这些结果不构成平台 PASS，也不能把 B 的变化解释为跨月份泛化已改善。

目标隔离恒等式覆盖 40 个 cell 与 50 个 macro 摘要，最大绝对残差 `5.55e-17`，小于 `1e-12`：A 的 `delta_wmape_time=0` 且 `Delta E=0.5 Delta W_iron`；B 的 `delta_wmape_iron=0` 且 `Delta E=0.5 Delta W_time`。

## 最终同铁口诊断

最终 335 个查询均为冻结 vocabulary 中的已知铁口（1 号 166、2 号 169），缺失和未知查询均为 0。规则对两个铁口对称应用。

| 目标 | 原跨铁口质量均值 | 新跨铁口质量均值 | 同铁口非空树比例均值 | 空同铁口交集树数中位数 | 原 OOB 空叶回退树数中位数 | raw QRF 变化行 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 铁量 A | 0.494833 | 0.030854 | 0.969146 | 8 | 0 | 321 / 335 |
| 时长 B | 0.476683 | 0.030737 | 0.969263 | 8 | 1 | 298 / 335 |

新集合最小成员数始终可为 1；A/B 最终新集合最大成员数的全查询最大值分别为 19/20。跨铁口质量下降是成员规则的数学不变量，不是准确率证据；约 3% 的残余跨铁口质量来自“同铁口交集为空时保留原 selected set”的冻结回退，未转去 full leaf 寻找新成员。原 OOB 空叶回退与本轮空交集回退在证据中分别记录。

最终完整 CSV 在六位提交精度下，A 恰有 321 行铁量变化、0 行时长变化；B 恰有 298 行时长变化、0 行铁量变化。两项均不是 NO-OP。

## 工程、冷审计与测试

r2 在独立进程中对 A/B × 7 cutoff 全部重新恢复源森林、响应、预处理器、bootstrap 与 v0.29 attachment；完成 full/reverse/chunk/subset/single 一致性，禁用条件化时精确恢复 v0.29 端点与 V30A 完整输出。所有 fit 拦截计数为 0，CSV/ZIP 回读一致，非有限或负预测为 0。

| 项目 | 注册上限 | 实际 |
| --- | ---: | ---: |
| 新模型/树/预处理/校准 fit | 0 | 0 |
| A 条件化读取 | 7 | 7 |
| B 条件化读取 | 7 | 7 |
| v0.29 OOB 附件只读恢复 | 14 | 14 |
| 对齐训练铁口向量 | 14 | 14 |
| 新候选包 | 2 | 2 |
| 平台上传 | 2 | 0 |

锁定 Python 3.12.12 主仓测试为 **507 passed**；v0.32 worker 独立测试为 **41 passed**。私有产物检查通过。r2 的 `completion.json`、`cold_validation.json` 与 `fit_counts.json` SHA-256 分别为 `16e1df1ad64786f3fa3ba95bfe68ded16fe085acaa2fe28569ef9e0cf1966b65`、`a60ecce100c5235bd5f369401d8a89206d1c64498b1665cbf2fdd581de346d22`、`0710fb489016c5510b10054929e5d04d50c69f6becd6992cfddfb7804d73c8f6`。

## 冻结包与平台边界

两份 ZIP 都只含 `result.csv`，内部 payload 与冻结 CSV 逐字节一致，覆盖相同顺序的 335 个唯一 ID：

| 槽 | 候选 | result.csv SHA-256 | ZIP SHA-256 |
| --- | --- | --- | --- |
| A | V32I_SAME_SPOUT_OOB_BLEND | `fa4e92fffff9aae4045fa8da6e7452dae0294bc1b5605f52772f057820effbc2` | `acafbc9d08540a1322802e339ff79ae5164e87391e893ed2ce3aad233931d6cd` |
| B | V32T_SAME_SPOUT_OOB_TIME | `110d73543dc1829792a28efec93234dae245ed8059b6fae540b43a54f78fdc6a` | `3d4596b2ece3baca3d0a3529373ca5482d1db4c996fdb66533a40efc7cd4b620` |

保底 V30A 原 ZIP 已重新核验为 `fffcf23b04b3069cd71027682047eea764a2747c2dea48cca91320d67e113986`，用户回传最高分仍为 83.3175。v0.32 平台状态为 `AWAITING_USER_REPORTED_PLATFORM_FEEDBACK`，预算仍为 2/2，顺序 A→B；agent 自动上传为 0。提交前仍须由用户核对账号当前生效条目、当日额度和恢复 V30A 的机会。A 的平台反馈不得改变已冻结的 B，不追加第三个组合候选。

## 桌面交付（2026-09-19 追加）

用户在冻结后显式要求把提交包写入桌面。已使用不覆盖模式写入：

- `C:\Users\lqh22\Desktop\Luqhhh_bf_tap_predict_prelim_V32I_SAME_SPOUT_OOB_BLEND.zip`
- `C:\Users\lqh22\Desktop\Luqhhh_bf_tap_predict_prelim_V32T_SAME_SPOUT_OOB_TIME.zip`

桌面 SHA-256 分别为 `acafbc9d08540a1322802e339ff79ae5164e87391e893ed2ce3aad233931d6cd`、`3d4596b2ece3baca3d0a3529373ca5482d1db4c996fdb66533a40efc7cd4b620`，与冻结源包一致。两份 ZIP 均只含 `result.csv`，payload 逐字节一致、各 335 行。没有覆盖或删除桌面文件，平台上传仍为 0。回执保存在 `local/runs/optimization-v0.32-same-spout-oob-responses-r2/desktop_delivery_receipt.json`。
