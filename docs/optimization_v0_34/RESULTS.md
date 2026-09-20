# optimization-v0.34 结果：OOB 叶中位数的等树平均

执行日期：2026-09-20。分支：`optimization-v0.34-oob-leaf-median-bagging`。有效完整本地运行：`local/runs/optimization-v0.34-oob-leaf-median-bagging-r2`，manifest 绑定实现提交 `6be934c10a69a6072d6d37c9e6e0cb2b7d38929d`。

## 结论

两项事前冻结的零拟合候选均完成，G0 工程状态 **PASS**：

- A `V34I_OOB_LEAF_MEDIAN_BAGGING_BLEND` 只替换 V30A 铁量集成中的 QRF 端点；
- B `V34T_OOB_LEAF_MEDIAN_BAGGING_TIME` 只替换 V30A 时长 QRF 端点，并保留原 v0.15 gate 与 V21 收缩。

两项均复用冻结的 256 树森林和 v0.29 OOB selected members。每棵树先在 selected leaf 内取原响应的较小中位数，再对 256 个持久化 binary64 叶点做精确等树算术平均。没有重训模型、树、预处理器或校准器；共生成 14 份新的监督叶点表，只读复用 14 份 v0.29 OOB 附件。没有读取 test target，没有平台上传、桌面写入或第三候选。

## 离线主口径

主口径为 `macro_origin_mean_wmape`，变化为候选减参照，负数表示历史误差改善：

| 候选 | 参照 | H1 ΔE | H2 ΔE | H3 ΔE | H4 ΔE | J ΔE | DEV_LONG ΔE | DEV_SHORT ΔE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V34I_OOB_LEAF_MEDIAN_BAGGING_BLEND | V30A | -0.00019272 | -0.00060022 | -0.00114301 | -0.00117065 | -0.00077665 | -0.00113240 | -0.00025517 |
| V34T_OOB_LEAF_MEDIAN_BAGGING_TIME | V30A | -0.00200911 | -0.00259521 | -0.00333027 | -0.00364102 | -0.00289390 | -0.00228421 | -0.00350687 |
| V34I_OOB_LEAF_MEDIAN_BAGGING_BLEND | V1 | +0.00142160 | +0.00249377 | +0.00392674 | +0.00507239 | +0.00322863 | +0.00086326 | +0.00557604 |
| V34T_OOB_LEAF_MEDIAN_BAGGING_TIME | V1 | -0.00039479 | +0.00049878 | +0.00173948 | +0.00260203 | +0.00111138 | -0.00028855 | +0.00232435 |
| V34I_OOB_LEAF_MEDIAN_BAGGING_BLEND | V21_REPLAY | +0.00103861 | +0.00010387 | -0.00115944 | -0.00012460 | -0.00003539 | -0.00096233 | +0.00198051 |
| V34T_OOB_LEAF_MEDIAN_BAGGING_TIME | V21_REPLAY | -0.00077778 | -0.00189111 | -0.00334669 | -0.00259497 | -0.00215264 | -0.00211413 | -0.00127119 |

A/B 相对 V30A 的 H1–H4、J 和两个 DEV 均改善，其中 B 的历史改善更大；但两项 J 仍分别比强离线参照 V1 差 `+0.00322863`、`+0.00111138`。历史标签已经消费，OOB 不是独立时序验证集，这些结果不是平台 PASS，也不证明跨月份泛化。

目标隔离恒等式覆盖 40 个 cell 与 50 个宏平均摘要，最大绝对残差 `6.94e-17`，低于 `1e-12`。A 的时长 WMAPE 变化为 0，B 的铁量 WMAPE 变化为 0，且均满足 `Delta E = 0.5 × Delta W_changed`。

## 最终叶点与预测诊断

最终 cutoff 的 A/B 点表分别覆盖 35,177 / 35,205 个 tree-leaf 键；原 selected member 数范围分别为 1–34 / 1–28。一个成员仍合法，原 OOB 空叶回退规则没有改变。

| 项目 | A：铁量 | B：时长 |
| --- | ---: | ---: |
| test_a 行数 | 335 | 335 |
| 每行不同叶点数（最小/中位/最大） | 171 / 211 / 230 | 42 / 52 / 69 |
| 叶点范围中位数 | 342.71 tonne | 82 min |
| 相对 V30A 改动行数 | 335 | 335 |
| 新目标最终差值范围 | -11.359492 至 -3.146739 | -3.816406 至 -0.148438 |
| 新目标最终差值中位数 | -6.629824 | -1.925781 |
| 不变目标 | V30A 时长字符串逐行相同 | V30A 铁量字符串逐行相同 |

所有最终改变量同号只是本次固定算法的观测结果，没有据此增加偏置、裁剪、路由或第三个组合候选。树叶点分歧只作描述，不视为已校准置信区间。

## 冷审计、测试与失败证据

独立 cold 进程覆盖 A/B × 7 cutoff，重新恢复原 forest、响应、预处理器和 bootstrap/OOB 附件，逐数组重派生 14 份叶点表，并完成 full/reverse/chunk/subset/single 一致性。切回原 v0.29 合并响应分布中位数时，两条完整链都精确恢复 V30A；CSV/ZIP 回读一致；所有森林、QRF、预处理器、CatBoost 与校准 fit 拦截计数为 0。

锁定 Python 3.12.12 主仓测试为 **520 passed**；v0.34 worker 合成测试为 **17 passed**。相关 v0.15/v0.26/v0.27/v0.29/v0.31/v0.34 worker 回归在正确锁定路径下合计 **94 passed**，私有产物 guard 通过。外部提案附带的 40 项参考测试不是本仓本轮新增测试，未混入上述数字。

首次运行 r1 在开发后复核中发现 worker receipt 将特征字典字段数 `4` 错记为查询行数。预测数组覆盖正确，但证书行数错误，因此 r1 被保留为未完成失败证据，没有评分、最终封包或 completion。修复提交 `6be934c10a69a6072d6d37c9e6e0cb2b7d38929d` 增加父端点行数硬检查；r2 从头注册、审计和派生，没有复用 r1 叶点表或预测。

关键证据 SHA-256：

- `manifest.json`: `92a860029f30d2d4a3d19a432eb8e9e4e5e369e47ecc1ad2ac6d3dc51f3cc40e`
- `offline_assessment.json`: `5c7b6a06a549ca37186c68c8d15e8f1b893d6721f789908777ccb2c59535e30a`
- `fit_counts.json`: `066375f6184315b09c24b361d2bb5e8d0f5420d5ed5c97147f82f34311e6c8b5`
- `cold_validation.json`: `a9fbd33b8a311425ae546c60ea65a5c39d29b074cb083ef4ddaac996e540be94`
- `completion.json`: `dcd727fcd4ef76ec6a5e76053ea09fa4488514cc792107c5c08cec08e2cb3473`

## 冻结包与平台边界

两份 ZIP 均只包含 UTF-8 `result.csv`，覆盖相同顺序的 335 个唯一 ID，预测为有限、非负、六位小数：

| 槽 | 候选 | result.csv SHA-256 | ZIP SHA-256 |
| --- | --- | --- | --- |
| A | V34I_OOB_LEAF_MEDIAN_BAGGING_BLEND | `e0a6353b0b630acf92b6f72a07c5c4fdbb0d600f7f65d47348b9ab44f8842985` | `80011770c3abbd108da59338ec9eb5e4e1c29c6d45b75387005349905d86311f` |
| B | V34T_OOB_LEAF_MEDIAN_BAGGING_TIME | `80a74e687f74181ec962cc8a12328380706c0d6bbbc7d53dd932866a169d6466` | `e3f970fa96cad54e0a6c534cc473d3b19eeb88269cb8c2630411796bfa313924` |

V30A 保底原 CSV/ZIP 已核验为 `97b6c3e0c648a0c6454cc9b36f518625487c03d1efd7621746d049457c4aed9a` / `fffcf23b04b3069cd71027682047eea764a2747c2dea48cca91320d67e113986`；在取得本轮反馈前，最高用户回传为 83.3175。

冻结状态为 `READY_FOR_EXPLICIT_PLATFORM_SUBMISSIONS`，预登记顺序 A→B、预算 2。agent 平台上传为 0，桌面写入为 0；两份定义、叶点表、预测与 ZIP 已在任何 v0.34 平台反馈前同时冻结。本轮没有第三组合或恢复上传预算；初赛阶段是否仍开放及账号当前生效条目尚未由本次运行核验。

## Git 交付

公共实现先后以 `e594b108f49fab4e218e03a5c7e7a9a6c5c166ef` 和证书修复 `6be934c10a69a6072d6d37c9e6e0cb2b7d38929d` 提交，并普通推送到 `origin/optimization-v0.34-oob-leaf-median-bagging`。推送前私有产物 guard 通过；`local/` 中的模型、响应、叶点表、逐样本预测、ledger、回执与提交 ZIP 均未进入 Git。

## 桌面交付（2026-09-20 追加）

用户随后明确要求把提交包写入桌面。两份冻结 ZIP 已采用不覆盖模式复制到 `C:\Users\lqh22\Desktop`：

- `Luqhhh_bf_tap_predict_prelim_V34I_OOB_LEAF_MEDIAN_BAGGING_BLEND.zip`
- `Luqhhh_bf_tap_predict_prelim_V34T_OOB_LEAF_MEDIAN_BAGGING_TIME.zip`

桌面摘要分别为 `80011770c3abbd108da59338ec9eb5e4e1c29c6d45b75387005349905d86311f`、`e3f970fa96cad54e0a6c534cc473d3b19eeb88269cb8c2630411796bfa313924`，与冻结源包逐份一致；每个 ZIP 只包含 `result.csv`。没有覆盖或删除任何现有桌面文件，平台上传仍为 0。私有交付回执为 `desktop_delivery_receipt.json`，SHA-256 `9abbe82ea1c9c015c0895d38d2626fac1def99eaa99ca7507c0ddfc5ffb3166e`。

## 平台反馈（2026-09-20 追加）

用户按预登记 A→B 顺序一次回传两个平台显示成绩，均未取得账号原始回执，证据状态为 `USER_REPORTED_NOT_INDEPENDENTLY_VERIFIED`：

| 槽 | 候选 | 用户回传 | Δ vs V30A | Δ vs A | 决策 |
| --- | --- | ---: | ---: | ---: | --- |
| A | V34I_OOB_LEAF_MEDIAN_BAGGING_BLEND | 83.3104 | -0.0071 | — | 关闭 A |
| B | V34T_OOB_LEAF_MEDIAN_BAGGING_TIME | 83.3201 | +0.0026 | +0.0097 | 提升为最高用户回传完整包 |

A 虽然历史 H1/J 相对 V30A 改善，平台显示仍低 0.0071；不把铁量叶点平均登记为平台收益。B 的历史 H1/J 和平台显示均高于 V30A，但 `+0.0026` 只支持这个固定完整包的结果，不证明跨月份显著改善，也不启动现场树、权重或收缩扫描。

本轮两个预登记平台名额已经用完，不生成 A+B 或第三候选。当前最高用户回传完整包更新为 B `V34T_OOB_LEAF_MEDIAN_BAGGING_TIME` = **83.3201**，ZIP SHA-256 为 `e3f970fa96cad54e0a6c534cc473d3b19eeb88269cb8c2630411796bfa313924`。agent 平台上传仍为 0；账号当前生效条目因没有账号回执继续登记为未知。

完整私有记录 `platform_feedback_user_reported.json` 的 SHA-256 为 `ed12f2abdef703fc2c8e00b4f93bad0a04b9228df7f67042d9d6edb059249539`；摘要 `platform_feedback_complete.json` 的 SHA-256 为 `1b60d158c685ab6b0ecd4f9c72b308d1616d7325c648baa6c59af9126443a3e3`。
