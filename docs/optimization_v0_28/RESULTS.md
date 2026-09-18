# optimization-v0.28 执行结果

执行日期为 2026-09-17。实现从最新 v0.27 发布头 `eec5c95bef392348c2d7597d64b01a488be1d71c` 建立；冻结实现提交为 `6f0bd85`，两项冷审计修复提交为 `7c7ddac`、`48b0e7c`。本阶段完成两个固定 50/50 端点集成，G0 工程状态为 **PASS**；G1 使用已消费历史单独报告，不作为两个预登记平台名额的取消条件。

两份 test_a 定义、预测和 ZIP 已在任何 v0.28 平台反馈前同时冻结。运行没有训练模型、没有拟合权重、没有读取测试目标或自动上传平台。用户随后明确授权替换桌面提交包，并按 A→B 顺序回传 83.2936、83.2604；尚未公开推送。

## 定义与来源

| 槽 | 候选 | 变化目标 | 固定算法 | 不变目标 |
| --- | --- | --- | --- | --- |
| A | `V28I_CB_QRF_EQUAL_BLEND` | 铁量 | 完整 V26A 铁量与完整 V27I 铁量各 50% | V26A 时长字符串 |
| B | `V28T_L1_L2_QRF_EQUAL_BLEND` | 时长 | 完整 V26A 时长与完整 V21 时长各 50% | V26A 铁量字符串 |

平均发生在各端点原有模型、结构修正、门控和六位序列化全部完成之后。端点六位值先转为整数微单位，再以 1/2 平均，半个微单位按 ties-to-even 处理。没有合并森林叶分布，没有再次执行 rate 或 V21 gate，也没有学习或按样本路由权重。

最终源文件摘要全部匹配预注册身份：V26A payload `9b42d41778fa23dfb07e825ae2ad44461e4d9886c11ea08e23a2dc553428cbad`，V27I payload `a42df723b06ab714712d3e13a75af241af36751b61744bdbcee767e17fb50cbf`，V21 payload `d5a2e118655107af39a76b99f3f1d053cc9469885d859113ba6f18ef20b9cece`。P0 检索没有发现已完成的等价 v0.28；r1/r2 是本阶段保留的未完成工程尝试，不是既有实验结果。

## 预算

| 项目 | 实际值 |
| --- | ---: |
| 新基础模型 fit | 0 |
| 新树 fit | 0 |
| 新预处理 fit | 0 |
| 新校准或权重 fit | 0 |
| 固定组合算法 | 2 |
| 最终候选 ZIP | 2 |
| agent 平台上传 | 0 |

## 离线结果

主口径为 `macro_origin_mean_wmape`，变化量为候选减参照，负数表示改善。预测使用六位精度、标签不舍入；J 为 H1--H4 宏平均 E 的等权平均，DEV 不进入 J。

| 候选 | 参照 | H1 ΔE | H2 ΔE | H3 ΔE | H4 ΔE | J ΔE | DEV_LONG ΔE | DEV_SHORT ΔE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | V26A_PARENT | -0.00038062 | +0.00030149 | +0.00014858 | +0.00096932 | +0.00025969 | +0.00069137 | -0.00082764 |
| A | V27I_ENDPOINT | -0.00069210 | -0.00153533 | -0.00179074 | -0.00223312 | -0.00156282 | -0.00180197 | -0.00091270 |
| B | V26A_PARENT | -0.00039447 | -0.00026235 | -0.00030866 | +0.00015986 | -0.00020140 | +0.00001696 | -0.00039111 |
| B | V21_REPLAY | +0.00033999 | +0.00020212 | +0.00020416 | -0.00025943 | +0.00012171 | -0.00007157 | +0.00031969 |

J 分别为：V1 `0.16573254`、V21_REPLAY `0.16899655`、V26A_PARENT `0.16931966`、V27I_ENDPOINT `0.17114218`、A `0.16957936`、B `0.16911826`。因此 A 在 H1 优于父方案、并明显优于较弱 V27I 端点，但整体 J 比父方案差；B 的 J 优于父方案，却没有超过历史更强的 V21，更没有超过 V1。

A 的时长、B 的铁量在每个 cell 和宏平均中均保持不变；`ΔE=0.5ΔWMAPE_changed` 的容差检查通过。`exposure_pooled_wmape` 单独输出，不与主口径混用。共享 calendar-week bootstrap 为 1000 次、seed 2026，有效 975 次；这些标签已经消费，不是独立确认。

误差抵消诊断覆盖 7,516 个 cell-row 暴露：A 的端点误差异号 626 次、累计绝对误差抵消量 `4846.420804`；B 为 73 次、抵消量 `64.0`。六位新舍入造成的累计绝对误差差异分别约 `2.2e-5`、`0`，均在逐行半微单位上界内。该诊断没有用于追加切片或修改 1/2 权重。

## 最终包

| 顺序 | 候选 | result.csv SHA-256 | ZIP SHA-256 | 相对 V26A |
| --- | --- | --- | --- | --- |
| A | `V28I_CB_QRF_EQUAL_BLEND` | `d97f13746eeaf7f00c916c52371234a9e9537f02d3b7ebec9407435bfb71b56d` | `f56db20cf96ba16d0238f3f4cb338fe818d8021326e2d439d011f699f73c9ca7` | 铁量改变 335 行；时长字符串全同 |
| B | `V28T_L1_L2_QRF_EQUAL_BLEND` | `61d4a99cd60a6d589f32cc68c9e2e13e7316571dc13eb165ba8c5f7230399f73` | `77ffb03efd1b1378d59df73eee890f0a556cf8a1fea44e241e08409de7039ecc` | 时长改变 248 行；铁量字符串全同 |

两包各覆盖 335 个唯一 test_a ID，ZIP 内仅含 `result.csv`，全部预测有限非负并严格六位。私有路径为：

- `local/runs/optimization-v0.28-fixed-equal-blends-r3/submissions/V28I_CB_QRF_EQUAL_BLEND/Luqhhh_bf_tap_predict_prelim.zip`
- `local/runs/optimization-v0.28-fixed-equal-blends-r3/submissions/V28T_L1_L2_QRF_EQUAL_BLEND/Luqhhh_bf_tap_predict_prelim.zip`

桌面两份 V27 包已移入系统回收站，private local 原包保留；两份 V28 ZIP 已复制到 `C:\Users\lqh22\Desktop`，摘要和 payload 与上表冻结源完全一致。交付回执 SHA-256 为 `7f81627c944a464271a4c3640c46ea3e0e62af47e770c74923fe859bf88093f5`。

## 平台反馈与决策

平台次序冻结为 A→B，每项一次，总预算 2。下表为用户回传，未取得 submission ID 或账号原始回执，不能当作独立平台核验。

| 顺序 | 候选 | 用户回传 | Δ vs V26A | Δ vs V21 | 决策 |
| --- | --- | ---: | ---: | ---: | --- |
| A | `V28I_CB_QRF_EQUAL_BLEND` | 83.2936 | +0.0108 | +0.0561 | 晋级为当前最高用户回传完整包 |
| B | `V28T_L1_L2_QRF_EQUAL_BLEND` | 83.2604 | -0.0224 | +0.0229 | 关闭，保留 A |

B 比 A 低 0.0332。两次平台预算按用户回传计为 2/2、剩余 0；agent 自动上传仍为 0。反馈没有改变第二候选定义，没有生成第三个权重、路由或双目标组合。反馈记录 SHA-256 为 `169649eaacfa3e54a0ec9b5d6d0e83d0a9a78aa714a37418d10f2b156185c508`。

当前最高用户回传为 V28I 的 83.2936。最后测试的是较低的 B；若初赛账号按最后一次提交生效，当前有效条目可能不是最高包。由于未取得账号回执，`best_user_reported` 与 `account_effective_submission` 继续分开登记。

## 冷审计、测试与尝试保留

权威运行目录为 `local/runs/optimization-v0.28-fixed-equal-blends-r3`。冷审计对 7 个 cutoff 分别恢复 V26A、V27I、V21 源端点，由冻结 worker 独立重算并核对原 NPZ，再验证新组合的全量、反序、分块、子集和单样本一致性；所有模型、森林、预处理器和校准 fit 尝试均为 0，双 ZIP/CSV 回读一致。

r1 在冷审计阶段错误地假设旧 worker 会额外保存 cold NPZ；实际 worker 只保存重算/比对回执。r2 修正该假设后，又发现反序辅助断言错误比较 DataFrame 索引。两个目录均没有 completion，分别保留 3 份冷回执；候选定义和 `result.csv` 预测内容未因这两次修复改变，r3 重新冻结最终 ZIP 身份。r3 使用“worker 冷重算核验冻结 NPZ，再读取被核验 NPZ”的正确接口和按字段内容的反序比较完成全流程。

锁定 Python 3.12.12 根环境为 **465 passed**，JUnit SHA-256 为 `e72767b4c1ef249494451953461c38b8d859dc788292ca1e6c59ee9111853b2e`。独立 worker 合计 **48 passed**：v0.15 34、v0.25 1、v0.26 5、v0.27 8；四份 JUnit SHA-256 为 `9d782f2920a4192dac54d49c8b7fd70ecf7a763bf90db4e03cb69fa4dde7eac1`、`b77730740672d80d41251601a5df98c5e4db87f7eaffd1a2e69a9cb6de212989`、`c72fc7f1951b563ef9a3b2a3ce031f11fddf32a8fd39df418556248a51d0e40e`、`c3cb5e7c60fd072ad0660e8e1f0c41a70c3ea755babd21c604b7985943951af6`。

r3 manifest SHA-256 为 `3a979cb5b5310d15f4df8adbe43af7935e3d9f573ada48cb041dba83d7d23430`，completion 为 `f1e47fd0263fcb94ae3117ca3441d428afee0941e4ab11d07430c4bab53cc7c6`，冷审计为 `7a2b157beacd4923c7ff2b33ffab8a520bae0ca973969ba584ec6587d524ccdf`。private 端点、逐样本预测、账本、报告和 ZIP 均未进入 Git。

用户明确授权提交与推送后，分支 `optimization-v0.28-fixed-equal-blends` 已发布到 `origin`，首次发布头为 `6b40b553921a2ac581232d050def6ad2ccc164b9`。推送前 private-artifact guard 为 PASS；private local 模型、端点、逐样本预测、账本、平台反馈文件和 ZIP 均未进入远端。
