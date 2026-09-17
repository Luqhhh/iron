# optimization-v0.27 执行结果

执行日期为 2026-09-17，基点为 `193842e81385f51efd48792df652fc03e30cc812`，冻结实现提交为 `9bdcaa3a90ecde2bddbaefd6bb50a1faf0cf07b0`。A（铁量 absolute-error QRF）与 B（冻结 V26A 时长森林的叶内 recency60 响应加权）均已完成，G0 工程状态为 **PASS**。两份 test_a 包已在任何本轮平台反馈前同时冻结；运行本身没有上传平台、没有写桌面、没有推送远端，随后按用户明确授权完成桌面旧包替换，仍未上传平台或推送远端。

## P0、来源与训练身份

P0 没有发现等价 v0.27 实验。06–12 七个 cutoff 均逐字节恢复 v0.26 已认证的原 v0.15 handoff：210 列 raw schema（209 个数值列和 `spout_no`），实际 transformed float32 schema 为 213 列；原预处理器仅 restore/transform，新 fit 为 0。训练行数依次为 888、1180、1490、1803、2091、2424、2754。

原 v0.15 平方误差 QRF 的 gate 支持度由原 forest 重新计算，没有复用 test_a ID 名单；test_a 的 gate 仍为 32/335 行。V26A 父 ZIP 与 payload 分别核验为 `e3aaaacd426b4113e120d8e3e90dd3e41d6d7ca71ffac70c6883a26ade60e60f`、`9b42d41778fa23dfb07e825ae2ad44461e4d9886c11ea08e23a2dc553428cbad`。

开发铁量标签读取器按训练 ID 读取官方训练表前缀，并在取得各 cutoff 的全部 ID 后立即停止；06–11 开发没有读取 2024-11 受保护目标。最终 fit 在 `final_training` 受保护生命周期中执行，读取前先将完整源 SHA-256 核验为冻结证书 `7b01a08d84e0bd380768424ceef58388bd0bb1eb28a2ac6d1920ee3da8e39e78`，并追加第二条不可覆盖 ledger 记录。测试目标始终未读。

## 模型与预算

A 使用原始非负 `tap_iron`（吨）训练 256 棵 `RandomForestRegressor(criterion="absolute_error", splitter="best")`，其余参数与 V26A 固定规格一致。最终输出是所有原始训练铁量响应在跨树叶分布中的较小加权中位数，不使用 estimator 的均值预测，也不经过旧 CatBoost/E04/rate/beta 铁量链。模型协议单独登记 `target=tap_iron, unit=tonne`。

B 未重训森林。它恢复同 cutoff 的完整 V26A 时长森林，把每条训练记录按 `reference_time` 到模型 cutoff 的年龄设为 60 日半衰期 binary64 权重；每棵树的查询叶内单独归一化，再让各树贡献相同总质量。实现没有采用“先算原跨树权重、再全局乘 recency”的非等价捷径。全部权重相等时逐元素退回原叶分布；0.5 临界 CDF 以持久化 binary64 权重的 `Fraction` 比例复核。旧 v0.15 gate、V21 的 25% 收缩、查询日 60 日中位数和六位 round-trip 顺序保持不变。

预算实际使用如下：

| 项目 | 实际值 |
| --- | ---: |
| A 森林 fit | 7/7 |
| A 内部树 | 1,792 |
| B 森林 fit | 0/0 |
| B 权重附件 | 7/7 |
| 新预处理 fit | 0 |
| CatBoost/E04/rate/q fit | 0 |
| LAD/beta/lambda/偏置 fit | 0 |

A 七次 fit 累计约 55.60 秒，记录到的进程峰值为 192,460 KiB。B 的推理耗时不计作森林训练。

## 离线结果（主口径）

下表使用 `macro_origin_mean_wmape`；变化量为候选减 V26A_PARENT，负数表示改善。预测按六位序列化、标签不舍入；J 为 H1–H4 的宏平均 E 等权平均，DEV 不进入 J。

| 候选 | H1 ΔE | H2 ΔE | H3 ΔE | H4 ΔE | J | J Δ | DEV_LONG ΔE | DEV_SHORT ΔE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V27I_ABS_QRF_DIRECT_IRON | +0.00031148 | +0.00183682 | +0.00193932 | +0.00320245 | 0.17114218 | +0.00182252 | +0.00249334 | +0.00008505 |
| V27T_ABS_QRF_LEAF_RECENCY60 | -0.00056052 | -0.00058720 | -0.00047335 | -0.00037947 | 0.16881951 | -0.00050014 | -0.00039878 | -0.00071081 |

父 V26A 的 J 为 0.16931966，V21_REPLAY 为 0.16899655，V1 为 0.16573254。A 相对父方案和所有强参照均退化；B 在 H1–H4、J 与两个 DEV 上均优于父方案，J 也比 V21_REPLAY 低 0.00017704，但仍比 V1 高 0.00308697。因此，G1 的准确表述是：A 在已消费回溯上退化；B 相对平台父方案获得一致回溯改善，但没有超过强离线参照 V1。

A 的时长、B 的铁量在全部 cell 与宏平均中均保持不变；`ΔE=0.5ΔWMAPE_changed` 的最大残差不超过 `1e-12`。`exposure_pooled_wmape` 只写入独立诊断 CSV，没有与主口径混用。共享 calendar-week bootstrap 有效 975/1000 draws；相对父方案的 J ΔE 95% 描述区间为 A `[+0.00054112,+0.00303858]`、B `[-0.00065750,-0.00032993]`。这些标签已被历史开发消费，不是独立显著性确认，也不取消任何预登记平台名额。

## 最终包

| 顺序 | 候选 | result.csv SHA-256 | ZIP SHA-256 | 相对 V26A |
| --- | --- | --- | --- | --- |
| A | V27I_ABS_QRF_DIRECT_IRON | `a42df723b06ab714712d3e13a75af241af36751b61744bdbcee767e17fb50cbf` | `5c5d8366962613f649fb82ca94df1796e55da1360a0783a01e7c19822e0f3ba5` | 时长字符串全同；铁量改变 335 行 |
| B | V27T_ABS_QRF_LEAF_RECENCY60 | `3b3325107399b3a617c14d7a7c19f00f268a74846fc296d370f19e17b8360597` | `767e515be195cf7df0839a7b1ac37217cb119260f86c3ee7405ac21114f4dc0e` | 铁量字符串全同；时长改变 174 行 |

两包各覆盖 335 个唯一 test_a ID，ZIP 内仅含 `result.csv`，全部预测有限非负且为六位小数。私有路径为：

- `local/runs/optimization-v0.27-qrf-iron-and-leaf-recency-r1/submissions/V27I_ABS_QRF_DIRECT_IRON/Luqhhh_bf_tap_predict_prelim.zip`
- `local/runs/optimization-v0.27-qrf-iron-and-leaf-recency-r1/submissions/V27T_ABS_QRF_LEAF_RECENCY60/Luqhhh_bf_tap_predict_prelim.zip`

用户随后明确授权删除桌面旧提交包并写入新包。两份 V26 桌面副本已移入系统回收站，其 private local 原包仍保留；桌面当前只保留：

- `C:\Users\lqh22\Desktop\Luqhhh_bf_tap_predict_prelim_V27I_ABS_QRF_DIRECT_IRON.zip`
- `C:\Users\lqh22\Desktop\Luqhhh_bf_tap_predict_prelim_V27T_ABS_QRF_LEAF_RECENCY60.zip`

桌面摘要与上表冻结 ZIP 完全一致，每包仅含 `result.csv`。本地交付回执 SHA-256 为 `7baf0cb17035ac0a7af7698297a795a671220138492d8a3229b76db665c7125d`。该操作是文件交付，不是赛事平台上传。

平台次序固定为 A→B，每项一个名额，总预算 2；当前平台上传为 0，反馈为空。没有生成第三个组合、融合、切片或校准候选，也没有自动准备恢复上传。当前最高用户回传仍是父 V26A 的 83.2828；账号当前有效条目未知。

## 冷审计、测试与证据

冷审计对 A/B 的全部 7 个 cutoff 完成独立进程恢复及全量、反序、分块、子集、单样本精确一致性检查，并再次核验旧 v0.15 gate。RandomForest、ExtraTrees、IronTargetForest、PartitionForest、原 QRF、预处理器、CatBoost 和校准 fit 尝试均为 0；B 的父树、叶成员和原始时长响应身份保持不变，双 ZIP 与 CSV 回读一致。

锁定 Python 3.12.12 根环境为 **455 passed**，JUnit SHA-256 为 `1599da3ad2437d79246f6f5ba65668926d0e965381e7674f7d7276174fa6604d`。worker 合计 **48 passed**：原 v0.15 34 项、v0.25 1 项、v0.26 5 项、v0.27 8 项；对应 JUnit SHA-256 分别为 `2247d976f156006f01f8712f24201513625c533ddd9e3cea58cc8949173d5d6f`、`5a23bc70784cf5b764cc0c54f15f6073740b8b406ec9b061be4189ed6e43bfda`、`78c9b252d44e6c204550ffde4e8bf0ca7225969069014b3d83cee039a9b83a65`、`4014c44aeebfe3624f8bb2907fbd93256995639bebf0b9ac27398bda87b8dbfe`。

权威私有运行目录为 `local/runs/optimization-v0.27-qrf-iron-and-leaf-recency-r1`。manifest SHA-256 为 `6bf2f834f7b1edb7a7b94974260f5d722297d0103546975b70585c4f69884ea2`，completion SHA-256 为 `4fc9c228e75a216accdf4c8dab4cc456714375e6fd39fd4d0b5f826bd492db73`，冷审计 SHA-256 为 `e6654e7344b583990a34a8c8cfe832d57ad83cdc95ed8749c8c155358ad85f41`。当前阶段状态为 `READY_FOR_TWO_EXPLICIT_PLATFORM_SUBMISSIONS`；private 模型、响应、预测、账本和 ZIP 均未进入 Git。
