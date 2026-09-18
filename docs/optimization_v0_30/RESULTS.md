# optimization-v0.30 结果：双目标 OOB 收益组合与时长森林固定扩容

执行日期：2026-09-18。分支：`optimization-v0.30-oob-compose-and-time-growth`。实现提交：`2e287c4`、`a8be772`、`9bb2c9d`、`7d642d3`、`e311254`、`1fcbf88`。权威本地运行：`local/runs/optimization-v0.30-oob-compose-and-time-growth-r6`。

## 结论

两个预登记候选均已按冻结定义完成并同时冻结，G0 工程状态为 **PASS**，没有任何 CatBoost、铁量森林、rate、q、LAD、预处理器或校准拟合。B 在本轮预算内完成固定的 256→1024 棵追加：`7` 次追加 fit（`5` 次在本轮成功运行内训练、`2` 次从被保留的前序尝试直接恢复）、`5,376` 棵新树、`1,792` 棵复用旧树、`7` 份 1024 棵 OOB 附件、两份 335 行包。冷审计对全部 7 个 cutoff 重新派生树前缀、bootstrap、full/OOB 成员并完成全量/反序/分块/子集/单样本一致性；所有 fit 入口计数为 0。用户随后按 A→B 各提交一次并回传 A=83.3175、B=83.2654；agent 平台上传为 0。

已消费历史上的 G1 风险如下，变化均为候选减参照，负数表示改善：

| 候选 | 参照 | H1 ΔE | H2 ΔE | H3 ΔE | H4 ΔE | J ΔE | DEV_LONG ΔE | DEV_SHORT ΔE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V30A_OOB_BOTH_TARGETS | V28I_PARENT | +0.00087749 | -0.00006186 | -0.00067782 | +0.00049602 | +0.00015846 | -0.00043277 | +0.00235252 |
| V30A_OOB_BOTH_TARGETS | V29T_OOB_LEAF_QRF_TIME | +0.00021632 | -0.00004194 | -0.00029260 | -0.00024866 | -0.00009172 | -0.00046090 | +0.00043436 |
| V30B_OOB_TIME_1024 | V28I_PARENT | +0.00082913 | -0.00017386 | -0.00071869 | +0.00053181 | +0.00011710 | -0.00042284 | +0.00234912 |
| V30B_OOB_TIME_1024 | V30A_OOB_BOTH_TARGETS | -0.00004835 | -0.00011200 | -0.00004086 | +0.00003579 | -0.00004136 | +0.00000993 | -0.00000340 |
| V30B_OOB_TIME_1024 | V1 | +0.00156597 | +0.00298199 | +0.00502889 | +0.00627884 | +0.00396392 | +0.00200559 | +0.00582781 |

A 相对 V28I 的 J 增量为 `+0.00015846`，恰好等于 V29I 与 V29T 各自相对 V28I 增量之和（`-0.00009172 + +0.00025018`）；A 的 J 与 H1、DEV_SHORT 都劣于 V28I，也明显劣于 V1。B 相对 A（本轮扩容主对照）的 J 为 `-0.00004136`，H1 为 `-0.00004835`，H2/H3 与 DEV_SHORT 小幅改善、H4 与 DEV_LONG 小幅退化；这类量级接近噪声，不能把 B 高于 V29T 全部归因于新增树。两项都报相对 V1 的退化，`+0.00396392`（B）与 `+0.00400528`（A）。历史排序与 09-18 平台回传（V29T 高于 V29I、均高于 V28I）继续不一致，离线评价保持不改写。

## 身份与合成恒等式

- A 铁量逐字符串复制 V29I 完整 CSV 列，时长逐字符串复制 V29T 完整 CSV 列；逐 cell 与宏平均核验 `iron(A)=iron(V29I)`、`time(A)=time(V29T)`、`iron(B)=iron(A)`，全部残差 ≤ 1e-12。
- A 相对 V28I 的目标加性：`ΔE(A,V28I) = ΔE(V29I,V28I) + ΔE(V29T,V28I)`，20 个 cell 与 25 个宏平均最大绝对残差 `5.55e-17`。
- `ΔE(A,V29T)=0.5·ΔWMAPE_iron` 与 `ΔE(B,A)=0.5·ΔWMAPE_time` 均通过（最大残差 `5.55e-17`）。
- 算术推算 `83.2970 + 83.3141 - 83.2936 = 83.3175` 只是显示值推算，不是实测分数，不构成泛化或平台成绩承诺。

## 1024 棵扩容与 OOB 回归

从已认证 V26A 256 棵绝对误差时长森林深复制，`n_estimators: 256→1024`、`warm_start: false→true`，对认证的同一 X/y/ID 顺序只调用一次 `fit`，追加 768 棵；其余参数（absolute_error、best、`min_samples_leaf=10`、`max_features=0.7`、`bootstrap=True`、`max_samples=None`、`random_state=2026`、`n_jobs=8`、`oob_score=False`）保持不变。`workers/qrf_v030/append_forest.py` 在拟合前后逐棵比较 seed、criterion、splitter、children、features、thresholds、impurity、sample counts、values 与 `estimators_samples_`，并拒绝 `clone` 未拟合对象、参数不匹配或前缀变化。合成测试同时验证了 toy 数据上 warm-start 追加与一次性 1024 棵训练逐树一致、序列化后 draw 可复原、非法输入被拒绝。

7 个 cutoff 的新附件都沿用 v0.29 逐树 OOB 协议：只使用本树 bootstrap 未抽中的原训练成员，空 OOB 叶回退该树完整叶，一个 OOB 成员合法；先汇总全部 1024 棵树的分布，再取原始时长较小加权中位数。前 256 棵的抽样身份、叶成员与回退证书与 v0.29 附件逐叶一致（members 覆盖 84,250–260,887 项，按 slot），把新森林限制到旧 256 棵时完整原始 QRF 时长（对 V26A 保存输出）与 OOB 响应、最终 V29T 时长全部精确复现。

最终 2,754 行模型的诊断：

| 项目 | 数值 |
| --- | ---: |
| 1024 棵树的总叶数 | 140,842 |
| 空 OOB 回退叶 | 673（0.4778%） |
| selected count 范围 | 1–30 |
| 每查询回退树均值 / 最大值 | 2.609 / 8 |
| 256→1024 原始中位数平均/最大绝对差 | 0.447–0.690 / 2–3 分钟（按 slot） |
| 模型字节 / 加载 / 附件 / 全量推理秒 | 12,738,425 / 3.018 / 3.418 / 1.597 |
| 峰值内存 | 378,344 KiB |

OOB 不是独立验证集，1024 棵也不增加唯一训练样本；上述差异不构成无偏保证。

## 最终包、预算与失败处理

| 槽 | 候选 | result.csv SHA-256 | ZIP SHA-256 |
| --- | --- | --- | --- |
| A | V30A_OOB_BOTH_TARGETS | `97b6c3e0c648a0c6454cc9b36f518625487c03d1efd7621746d049457c4aed9a` | `fffcf23b04b3069cd71027682047eea764a2747c2dea48cca91320d67e113986` |
| B | V30B_OOB_TIME_1024 | `5098c345a263f005a2b33f32215f803cd4f5ef29d2dff2f3f9b30fe096d4fd21` | `25ef5659559bd2b4532997cd05b0d8098ca4489c027b830b0900d48b695c9dda` |

两份 ZIP 均只含 `result.csv`，覆盖 335 个唯一 ID，有限非负且为六位小数。包路径分别为 `local/runs/optimization-v0.30-oob-compose-and-time-growth-r6/submissions/V30A_OOB_BOTH_TARGETS/Luqhhh_bf_tap_predict_prelim.zip` 与同根 `.../V30B_OOB_TIME_1024/Luqhhh_bf_tap_predict_prelim.zip`。

## 平台反馈与决策

用户依次回传 A=83.3175、B=83.2654；两项均未取得账号原始回执，仍为 `USER_REPORTED_NOT_INDEPENDENTLY_VERIFIED`。

| 槽 | 候选 | 分数 | 相对 A | 相对 V29T=83.3141 | 相对 V28I=83.2936 | 决策 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| A | V30A_OOB_BOTH_TARGETS | 83.3175 | — | +0.0034 | +0.0239 | 晋级为当前最高用户回传完整包 |
| B | V30B_OOB_TIME_1024 | 83.2654 | -0.0521 | -0.0487 | -0.0282 | 关闭，保留原包 |

A 的显示值与预登记的加性推算 `83.2970 + 83.3141 - 83.2936 = 83.3175` 在四位小数上完全一致，这是该恒等式第一次与本轮平台反馈对上；但仍只是显示值一致，不代表账号级核验，也不构成泛化证据。B 比同轮 A 低 0.0521，平台不支持把固定 1024 棵扩容视为提分；离线 B-A 的微小改善（J -0.00004136）与平台排序相反，历史评价保持不改写。两次平台名额已由用户侧 A→B 各消费一次（预算 2/2），V29T 83.3141 保留为原包后备，不生成第三个候选。反馈记录为 `local/runs/optimization-v0.30-oob-compose-and-time-growth-r6/platform_feedback_user_reported.json`，SHA-256 `082090b8…5667`。

追加 fit 预算在轮内恰好用满：`r3` 完成 slot 6、`r4` 完成 slot 7，随后因读取端证书字段/适配器报告问题被阻断；`r6` 显式**恢复**这两个已完成拟合（re-fit 计数 0，重新核验前缀、bootstrap、full-leaf partition 与训练身份），并训练 slots 8–11 与最终 cutoff，共 `7` 次真实追加 fit、`5,376` 棵新树；`r2` 的 slot-6 intent 在训练前被阻断（0 棵树）。`r1`（P0 前失败）、`r2`、`r3`、`r4`、`r5` 运行目录全部保留，未覆盖、未删除。冷审计中 append fit、sklearn、PartitionForest、原 QRF、预处理器与校准入口的 fit 尝试均为 0。

## 测试与边界

根锁定 Python 3.12.12 为 **484 passed**；独立 worker 共 **86 passed**（v0.15 34、v0.24 1、v0.25 1、v0.26 5、v0.27 8、v0.29 20、v0.30 17）。根套件中新增 9 项覆盖列复制、恒等式与注册契约；worker 新增 17 项覆盖追加前缀、与一次性 1024 棵对照、序列化复原、附件前缀切片、受限 256 视图、full-leaf 附件与非法输入拒绝。

冻结与冷审计阶段未登录平台、未上传、未写桌面、未公开推送。用户随后显式授权桌面替换：桌面旧 iron 提交包在替换时已不存在（递归检索为 0），A/B 两份 ZIP 以 `Luqhhh_bf_tap_predict_prelim_V30A_OOB_BOTH_TARGETS.zip` 与 `..._V30B_OOB_TIME_1024.zip` 写入桌面并逐字节核验，回执见 `local/runs/optimization-v0.30-oob-compose-and-time-growth-r6/desktop_delivery_receipt.json`；平台上传仍为 0。未生成第三个候选或阈值/权重搜索。最高用户回传仍为 V29T=83.3141，A 的 83.3175 只是显示值推算；A/B 各保留一个事前登记探索名额，默认顺序 A→B，A 的反馈不得改变 B。正式复赛数据改变时新建来源契约，不把 test_a 成绩外推为 test_b 结果。
