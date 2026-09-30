# 本地—平台对照：同名包不等于同口径（2026-09-30）

共14个已有反馈条目；不拟合、不读取目标。平台分均用户报告。每行明确参照与本地协议，不跨seed混合OOF，不按这份选择过的样本计算泛化命中率。

| 包 | 共同参照 | 本地增量 | 平台增量 | 本地证据口径 |
|---|---|---:|---:|---|
| V5_SEED_SWAP_S1000 | V36 | -0.004620 | +0.000500 | combined seed-sensitivity diagnostic; not four-split confirmation |
| V5_TIME_N0048_Q20 | V36 | +0.009720 | +0.040900 | four split seeds; alpha .20/.21/.20/.20, not strictly constant-release alpha |
| V6_PORT_TIME_A05 | V5_TIME_N0048_Q20 | -0.005710 | -0.029900 | complete two-seed fixed-endpoint alpha surface |
| V6_PORT_TIME_A35 | V5_TIME_N0048_Q20 | -0.005230 | +0.022300 | complete two-seed fixed-endpoint alpha surface |
| V6_PORT_TIME_A45 | V5_TIME_N0048_Q20 | -0.015059 | +0.029500 | complete two-seed fixed-endpoint alpha surface |
| V6_PORT_TIME_A60 | V5_TIME_N0048_Q20 | -0.037079 | +0.032200 | complete two-seed fixed-endpoint alpha surface |
| V6_PORT_TIME_A72 | V5_TIME_N0048_Q20 | -0.061274 | +0.028200 | complete two-seed fixed-endpoint alpha surface |
| V7_TIME_PLR001_A50 | V6_PORT_TIME_A35 | +0.015270 | +0.015300 | four-split nested selection; derived-seed alpha adaptations documented |
| V12_IRON_JOINT_PLR001_A50 | V6_PORT_TIME_A35 | +0.018159 | +0.016000 | two-seed development against A35; confirmation not substituted |
| V20_B0_PLLT_A325 | V18_B0_V12IRON_V7TIME | +0.004607 | -0.000300 | four-split complete design check |
| V21_TIME_LOCAL | V18_B0_V12IRON_V7TIME | +0.012728 | -0.011200 | four-split complete composition |
| V32_TIME_A60V7_50 | V18_B0_V12IRON_V7TIME | 未提取同口径值 | +0.004800 | exact same-parent complete local gain not extracted |
| DE3_IRON_USER_REQUESTED | V32_TIME_A60V7_50 | +0.003480 | +0.002200 | two-seed complete development; no four-seed confirmation |
| TODAY_V28_SHARED_TIME_A20 | V18_B0_V12IRON_V7TIME | -0.000027 | -0.000800 | own-chat complete two-seed fixed B0/A20 endpoint |

## 必须保留的边界

- V5、V7 的多seed本地选择端点未必逐一等于平台固定权重；只能作注明口径的对照。
- A35/A45/A60/A72 属同一模型端点的权重线，不能算四个独立模型家族。
- V32 暂不拼接其他协议本地参照，保留缺失项。
- V20_B0_PLLT 与本对话 V20 遮蔽重建是不同策略；V28 以实际探索包的 B0/A20 为准。
- A60 对 A20 的负本地/正平台，以及 V21 四seed正/平台负，是直接的校准警报。没有因此放宽任何冻结门槛。
- 不从观察到的微小平台分差推断平台分辨率，不拟合统一分数修正系数。

## 来源与身份

- 跨分支公开汇总：[EVIDENCE_STATUS.json](https://github.com/Luqhhh/iron/blob/e8b5760cbacab4c25ee861025e6e961c4d47f85b/EVIDENCE_STATUS.json)。
- [V6固定端点曲线](https://github.com/Luqhhh/iron/blob/e8b5760/docs/round2_v6/RESULTS.md)、[V7](https://github.com/Luqhhh/iron/blob/e8b5760/docs/round2_v7/RESULTS.md)、[V12](https://github.com/Luqhhh/iron/blob/e8b5760/docs/round2_v12/RESULTS.md)。
- [本对话V28实际探索包口径](https://github.com/Luqhhh/iron/blob/e339011/docs/round2_v28_edge_kan/RESULTS.md)。

这里只发布已经公开/用户报告的汇总数值及ZIP摘要；没有发布包字节、预测、标签或平台原始回执。

| 包 | 已登记ZIP SHA-256 |
|---|---|
| V5_SEED_SWAP_S1000 | 4ff97c033f05b7e686453e913460b548cd7231d3736b3f431bb3882e1a82460f |
| V5_TIME_N0048_Q20 | 5ed99b8014fb1650b88b6ae57cc3ed4dac378a20831eab5efc800de91ab2c982 |
| V6_PORT_TIME_A05 | 3f9b27f327e4304fe21e5c3fb5d739765821d118333fbd7b598726242e353f08 |
| V6_PORT_TIME_A35 | b4e1fc2d1287a213e9d88d1c30a7420ecc222235fcc6b7191e341dfd16924b06 |
| V6_PORT_TIME_A45 | 4c12bedae707c30628f6b74e404b63596dee4145883dd6d2c64c2a4553ae8cd7 |
| V6_PORT_TIME_A60 | d5092d400fb5cc62c2ff61f529a4dc32d9d3493a6d82cad2f2a591a3d4bc2147 |
| V6_PORT_TIME_A72 | 808f00a9a3385b62447961badc73e92bf5d6dc845bc5802cea1ce623ccda40bf |
| V7_TIME_PLR001_A50 | 4382523c7bd688974f87eab2f42502bf8f54b2b330008b36e797ae7672490299 |
| V12_IRON_JOINT_PLR001_A50 | a1c205a6722da3976a12e258458b649967c7c25130a2c55d840c5ecb2a1dc669 |
| V20_B0_PLLT_A325 | f14f39df5474c8904639c0768bf8fa678f645f46fb0f987fbb84690a983eef57 |
| V21_TIME_LOCAL | 32a74052899bd0b02d841d2e07e87645fdaf4b8030fcc69c7d7b12a654fc2016 |
| V32_TIME_A60V7_50 | 54864561c057ae71f7779b15099750f7caf0141c149d609ae281dbaf9ef3aaa7 |
| DE3_IRON_USER_REQUESTED | 81d9d1b12c3b4fa9f40770b2ac2fef0caef7b500c5ff0ab0d7cc2948aa679ee0 |
| TODAY_V28_SHARED_TIME_A20 | 1020e9dd36b09d7edc50180fb502e037a8d24cfc167d5c01a12a06825401a06a |

本表为反馈齐全的选择样本，不能估计一般模型本地/平台同号概率；权重探针彼此相关，更不能当独立样本。没有因此调整旧决策、门槛或授权平台发布。
