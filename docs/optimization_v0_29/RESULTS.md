# optimization-v0.29 结果：冻结森林的 OOB 叶响应估计

执行日期：2026-09-18。分支：`optimization-v0.29-oob-leaf-responses`。权威本地运行：`local/runs/optimization-v0.29-oob-leaf-responses-r1`。

## 结论

两个预登记候选均已按冻结定义完成，G0 工程状态为 **PASS**，没有任何森林、预处理器、CatBoost 或校准拟合。14 份 OOB 叶响应附件、两份 335 行提交包和独立冷审计均已完成。两项仍保留各一次显式平台探索名额；本阶段未自动上传、未写桌面、未公开推送。

已消费历史上的 G1 风险如下，变化均为候选减 V28I，负数表示改善：

| 候选 | H1 ΔE | H2 ΔE | H3 ΔE | H4 ΔE | J ΔE | DEV_LONG ΔE | DEV_SHORT ΔE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V29I_OOB_LEAF_QRF_BLEND | +0.00021632 | -0.00004194 | -0.00029260 | -0.00024866 | -0.00009172 | -0.00046090 | +0.00043436 |
| V29T_OOB_LEAF_QRF_TIME | +0.00066117 | -0.00001991 | -0.00038523 | +0.00074468 | +0.00025018 | +0.00002813 | +0.00191816 |

A 的 J 小幅改善但 H1 退化；B 的 J、H1 和两个 DEV 中 DEV_SHORT 明显退化。两项都没有超过历史强参照 V1/V21，不能把 OOB 响应称为已经证实的泛化改进。按预注册协议，已消费历史评价只报告风险，不取消两个完整算法的平台名额。

## OOB 身份与叶协议

每个 cutoff 都从已认证 sklearn 1.8.0 模型的 `estimators_samples_` 恢复真实 bootstrap 位置，没有用随机种子重建或调用 `fit`。训练 ID、float32 矩阵、原响应和模型顺序逐项绑定。每棵树的 draw 长度为 N；每叶 bootstrap 多重计数与 `weighted_n_node_samples` 一致，唯一 in-bag 位置数与 `n_node_samples` 一致；重新 `tree.apply` 与原 full-leaf mapping 一致。

OOB 叶非空时，每个未被该树抽中的原始位置计一次；空 OOB 叶只回退该树同一完整叶。所有 256 棵树仍贡献相同质量，叶内等权并取较小加权中位数。将成员替回完整叶时，七个 cutoff 的原 QRF 原始输出及完整端点均精确复现。

最终 2,754 行模型的诊断：

| 目标 | 总叶数 | 空 OOB 回退叶 | 回退比例 | selected count 范围 | 每查询回退树均值/最大值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 铁量 | 35,177 | 148 | 0.4207% | 1–34 | 0.5851 / 3 |
| 时长 | 35,205 | 163 | 0.4630% | 1–28 | 0.9284 / 4 |

一个 OOB 成员按预注册规则合法使用；上述分布只作诊断，没有据此增加阈值或样本路由。OOB 不是新验证集，也不构成严格 honest forest 或无偏保证。

## 目标隔离与最终包

- A 固定原 V26A 完整 CatBoost 铁量端点与新 OOB-QRF 铁量端点各 1/2，再用 v0.28 的整数微单位、ties-to-even 六位算法平均；时长逐字符串复制 V28I。最终铁量改变 329/335 行。
- B 对新 OOB-QRF 原始时长先作六位 round-trip，再重放原 v0.15 支持度 gate 和 V21 的 25% 收缩；铁量逐字符串复制 V28I。最终时长改变 246/335 行。

逐 cell 和宏平均均通过未改目标 ΔWMAPE=0、`ΔE=0.5×ΔWMAPE_changed` 的 1e-12 检查。两个 ZIP 均只含 `result.csv`，覆盖 335 个唯一 ID，有限非负且为六位小数。

| 槽 | 候选 | result.csv SHA-256 | ZIP SHA-256 |
| --- | --- | --- | --- |
| A | V29I_OOB_LEAF_QRF_BLEND | `2b5bb1284c25ba9548d7d0c91d167adee0cb34b1d658b10e778bedd35063cf6c` | `68320081ec34e8e1296ddff3ac75fdb5dfedffac5d07fff7be95bc74e1b3a581` |
| B | V29T_OOB_LEAF_QRF_TIME | `53bf201d6b524c10261364c530bd46a32dc794659b68cdb98a68764cc3ab3dcd` | `650765ff7b749eda2ff41c7f8c0e99a33e4171958ee79918a6e4f14e623f244b` |

两份包路径分别为：

- `local/runs/optimization-v0.29-oob-leaf-responses-r1/submissions/V29I_OOB_LEAF_QRF_BLEND/Luqhhh_bf_tap_predict_prelim.zip`
- `local/runs/optimization-v0.29-oob-leaf-responses-r1/submissions/V29T_OOB_LEAF_QRF_TIME/Luqhhh_bf_tap_predict_prelim.zip`

## 冷审计、预算与测试

独立冷进程重新恢复 v0.27 铁量森林、v0.26 A 时长森林、原 v0.15 gate、原响应和 14 份附件，重新派生 bootstrap/OOB mapping，并完成全量、反序、分块、子集和单样本一致性。源模型文件摘要及关键树结构不变，所有 fit guard 计数为 0，ZIP/CSV 回读一致。

预算实耗：0 个模型/树 fit、0 个预处理 fit、0 个校准 fit、14 份 OOB 附件、2 个 ZIP、0 次 agent 上传。根锁定 Python 3.12.12 测试为 **475 passed**；独立 worker 共 **68 passed**（v0.15 34、v0.25 1、v0.26 5、v0.27 8、v0.29 20）；private-artifact guard 通过。

关键证据摘要：manifest `18a40277…88c`，completion `f589a7e6…67e5`，cold validation `ffdbe8ab…42f6`，packages frozen `b80554e5…07b`。

## 平台边界

当前平台父参考仍是用户回传、未独立核验的 V28I=83.2936。A/B 定义与包已在任何 v0.29 平台反馈前同时冻结，顺序固定 A→B，各一次。本报告没有平台反馈；最高用户回传、最新反馈与账号当前有效条目仍分开登记。没有自动恢复位、第三个组合、阈值或切片候选。
