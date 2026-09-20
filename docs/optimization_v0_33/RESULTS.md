# optimization-v0.33 结果：时长 QRF 特征抽样与单树样本抽样

执行日期：2026-09-19。分支：`optimization-v0.33-qrf-feature-and-row-sampling`。完整授权本地运行：`local/runs/optimization-v0.33-qrf-feature-and-row-sampling-r1`，manifest 绑定实现提交 `441b09ba95c613379611de00af7a2e4c0a87f4d9`。

## 结论

两项事前冻结的 time-only 候选均完成，G0 工程状态 **PASS**：

- A `V33A_OOB_TIME_FEATURE_THIRD` 仅将 `max_features` 从 0.7 改为 binary64 的 `1/3`，`max_samples=None`；
- B `V33B_OOB_TIME_HALF_BOOTSTRAP` 仅将 `max_samples` 从 `None` 改为 0.5，`max_features=0.7`。

两项各从头训练 7 个 256 树 `RandomForestRegressor`，共完成 14/14 次 fit、3584 棵新树和 14 份新 OOB 附件。预处理器、铁量森林、CatBoost/E04/rate/q、LAD、beta、lambda、bias 与其他校准 fit 均为 0。没有读取 test target，没有平台上传、桌面写入、远端推送或第三候选。

## 离线主口径

主口径为 `macro_origin_mean_wmape`，变化为候选减参照，负数表示历史误差改善：

| 候选 | 参照 | H1 ΔE | H2 ΔE | H3 ΔE | H4 ΔE | J ΔE | DEV_LONG ΔE | DEV_SHORT ΔE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V33A_OOB_TIME_FEATURE_THIRD | V30A | -0.00039955 | +0.00001103 | -0.00023180 | -0.00002904 | -0.00016234 | -0.00023331 | -0.00005272 |
| V33B_OOB_TIME_HALF_BOOTSTRAP | V30A | -0.00019621 | -0.00006895 | -0.00030489 | +0.00013791 | -0.00010804 | -0.00011666 | +0.00016325 |
| V33A_OOB_TIME_FEATURE_THIRD | V1 | +0.00121478 | +0.00310502 | +0.00483795 | +0.00621401 | +0.00384294 | +0.00176235 | +0.00577850 |
| V33B_OOB_TIME_HALF_BOOTSTRAP | V1 | +0.00141812 | +0.00302504 | +0.00476485 | +0.00638096 | +0.00389724 | +0.00187901 | +0.00599446 |

A/B 的 H1 与 J 均小幅优于 V30A，但相对强离线参照 V1 的 J 仍分别退化 `+0.00384294`、`+0.00389724`。历史标签已经消费，OOB 叶响应不是独立验证集；因此这些结果不是平台 PASS，也不能解释为跨月份泛化已得到证明。

两项都是 time-only。目标隔离恒等式覆盖 40 个 cell 与 50 个宏平均摘要，最大绝对残差 `5.55e-17`，小于 `1e-12`：铁量 WMAPE 变化为 0，且 `Delta E = 0.5 * Delta W_time`。

## 最终 N/m、OOB 与资源诊断

最终 cutoff 使用 N=2754。A 每树实际抽 2754 次，且 256 份 draw 与原 V26A 逐元素相同；B 每树实际抽 1377 次，不要求旧 draw 前缀或旧叶映射相同。两项均将全部 2754 个唯一训练行逐树投叶，并核验每叶 draw 重数与 `weighted_n_node_samples`、唯一 in-bag 行数与 `n_node_samples` 一致。

| 项目 | A：1/3 特征 | B：半量 bootstrap |
| --- | ---: | ---: |
| unique in-bag / tree（均值） | 1741.895 | 1084.164 |
| global OOB / tree（均值） | 1012.105 | 1669.836 |
| OOB 空叶回退比例（全部叶） | 0.4585% | 0.0137% |
| OOB selected size 范围 | 1–31 | 1–66 |
| test_a effective neighbors 中位数 | 452.391 | 691.386 |
| 最终 forest 文件 | 2,962,370 B | 2,283,659 B |
| 最终 attachment 文件 | 3,789,379 B | 3,541,721 B |
| 最终 fit 时间 | 5.400 s | 5.908 s |
| 最终附件生成时间 | 1.693 s | 1.676 s |

B 的 global OOB 行数确实更多，但它同时改变了构树信息量与分区，不能把后续结果单独归因于 OOB 数量。两项响应汇总都保持每树等权、叶内等权，没有使用 bootstrap 重数、同铁口筛选、occurrence pooling 或 recency 权重。

相对 V30A 的最终提交时长，A 改变 234/335 行，差值范围 -5 到 +4 分钟；B 改变 203/335 行，范围 -2 到 +3 分钟。两项铁量字符串均逐行与 V30A 完全一致，且都不是 NO-OP。

## 冷审计与测试

独立 cold 进程覆盖 A/B × 7 cutoff，逐项重新恢复模型与附件、从实际 `estimators_samples_` 重建并比对 N/m、draw、树结构、full/OOB/selected mappings，完成 full/reverse/chunk/subset/single 一致性。cold 内所有森林、QRF 与预处理器 fit 拦截计数为 0；CSV/ZIP 回读一致。

锁定 Python 3.12.12 主仓测试为 **510 passed**；v0.33 worker 合成测试为 **17 passed**；私有产物守卫通过。G0 与 G1 分开登记。

关键证据 SHA-256：

- `completion.json`: `a247fa5461c29be04b391b723ddfe1400f376a3b9318f5bcfb2bf725cbb52cf3`
- `cold_validation.json`: `534911b9caf92f01ddf730cb96d098b96735a090fb07a208c930663c24ad21a2`
- `fit_counts.json`: `cd8ab122d320414567ff44c1a0f36e2f4201eca264d74a61ef4442aff7f2fddc`

## 冻结包与平台边界

两份 ZIP 均只包含 UTF-8 `result.csv`，覆盖相同顺序的 335 个唯一 ID，预测为有限、非负、六位小数：

| 槽 | 候选 | result.csv SHA-256 | ZIP SHA-256 |
| --- | --- | --- | --- |
| A | V33A_OOB_TIME_FEATURE_THIRD | `f9834287a4fc34744fc49dd3ad0459b2ae5cb7247e518a12910ca549617625c8` | `e7fe9c8cb82521b384b2b8d46031220b8f15bd7a09f00db7f3ccc95bc923a850` |
| B | V33B_OOB_TIME_HALF_BOOTSTRAP | `aef939dec9df8a4ad3438589be14a00e7de25886cdc179bda720114e43d6964c` | `0a50865800cba7181db115f50e31598b81c0a4f589fbb0c75d1a79774c3408c2` |

V30A 保底原 CSV/ZIP 已核验为 `97b6c3e0c648a0c6454cc9b36f518625487c03d1efd7621746d049457c4aed9a` / `fffcf23b04b3069cd71027682047eea764a2747c2dea48cca91320d67e113986`，最高用户回传仍为 83.3175。

冻结完成时状态为 `READY_FOR_EXPLICIT_PLATFORM_SUBMISSIONS`，预登记顺序 A→B、预算 2；agent 平台上传为 0。两个定义、最终模型、附件、预测和包已在任何 v0.33 反馈前同时冻结。

## 桌面交付（2026-09-19 追加）

用户随后明确要求把新包写入桌面并删除旧提交包。已精确删除桌面上的两份 v0.32 ZIP；仓库 private local 中的 v0.32 冻结原件没有删除，仍可恢复。两份 v0.33 ZIP 以不覆盖模式写入：

- `C:\Users\lqh22\Desktop\Luqhhh_bf_tap_predict_prelim_V33A_OOB_TIME_FEATURE_THIRD.zip`
- `C:\Users\lqh22\Desktop\Luqhhh_bf_tap_predict_prelim_V33B_OOB_TIME_HALF_BOOTSTRAP.zip`

桌面摘要分别为 `e7fe9c8cb82521b384b2b8d46031220b8f15bd7a09f00db7f3ccc95bc923a850`、`0a50865800cba7181db115f50e31598b81c0a4f589fbb0c75d1a79774c3408c2`，与冻结源包一致。桌面同赛事前缀 ZIP 核验后仅余这两份；平台上传仍由用户操作，agent 上传为 0。回执保存在 `desktop_delivery_receipt.json`，SHA-256 为 `cd1c98b29b0d912c799ef9afe98958f6fed64fad02729afaf6025d01acb1301e`。

## 平台反馈（2026-09-19 至 2026-09-20 追加）

用户回传 A `V33A_OOB_TIME_FEATURE_THIRD` 的平台显示成绩为 **83.2241**，未取得账号原始回执，证据状态为 `USER_REPORTED_NOT_INDEPENDENTLY_VERIFIED`。

| 候选 | 用户回传 | Δ vs V30A | Δ vs V29T | Δ vs V29I | Δ vs V28I | Δ vs V1 | 决策 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V33A_OOB_TIME_FEATURE_THIRD | 83.2241 | -0.0934 | -0.0900 | -0.0729 | -0.0695 | +0.1922 | 关闭 A，保留 V30A |
| V33B_OOB_TIME_HALF_BOOTSTRAP | 83.2638 | -0.0537 | -0.0503 | -0.0332 | -0.0298 | +0.2319 | 关闭 B，保留 V30A |

A 的历史 H1/J 相对 V30A 小幅改善，但平台显示分数低 0.0934；这不支持把 1/3 特征抽样视为可泛化收益。历史离线表不回写。当前最高用户回传完整包仍为 V30A=83.3175。

B `V33B_OOB_TIME_HALF_BOOTSTRAP` 的定义、模型、预测和 ZIP 未根据 A 的成绩修改。2026-09-20 用户回传 B=**83.2638**：比 A 高 0.0397，但仍比 V30A 低 0.0537。B 的历史 H1/J 也曾小幅优于 V30A，平台仍未提分；这不支持继续围绕固定半量 bootstrap 扫描抽样比例。

本轮两个预登记平台名额已按 A→B 用完，A/B 均关闭，当前最高用户回传完整包保持 V30A=83.3175。不生成 A+B、第三候选或现场参数扫描。agent 平台上传为 0；未取得账号回执，因此账号当前生效条目仍登记为未知。A 的初始记录保持不改，B 使用新增 append-only 记录 `platform_feedback_B_user_reported.json`（SHA-256 `ff5cb35ec6175c64f4a575d954af644782b2162bc5e100decdc59ddc95256a72`）；完整摘要为 `platform_feedback_complete.json`（SHA-256 `9e862ff7f62ae5eac79b29c9dedad80e384842ed95c5a048a616d7f5f58569d0`）。

## 远端推送（2026-09-19 追加）

用户明确要求提交推送。`optimization-v0.33-qrf-feature-and-row-sampling` 已首次推送至 `origin` 并设置 upstream；首次远端头核验为 `1ac0b8538748da06f1ab7cf3ee01b9fab0443ee4`。推送内容仅包含 Git 已跟踪的代码、配置、测试与文档；`local/` 中的模型、附件、逐样本预测、ledger、平台反馈原始记录和提交 ZIP 均未推送。agent 平台上传仍为 0。发布回执保存在 `publication_receipt.json`，SHA-256 为 `cd047d330d6b9da73779020a0908e57b91e273ae0848f7c9457ed3bf80ae68e1`。
