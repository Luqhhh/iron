# 平台提交记录

本文件只记录实际提交及其来源，不将用户提供的平台结果表述为仓库或编码代理独立复现。

## 2026-09-06 · test_a / prelim

| 字段 | 记录 |
| --- | --- |
| 平台显示成绩 | `81.4554` |
| 证据状态 | `USER_REPORTED_NOT_INDEPENDENTLY_VERIFIED` |
| 提交操作者 | 用户手动上传 |
| 提交文件 | `Luqhhh_bf_tap_predict_prelim.zip` |
| ZIP SHA-256 | `44ac6ced2fe3b871f387500caed325183bd73a55f11c56efb7a0f84531957c8f` |
| ZIP payload | 仅 `result.csv`，335 行 |
| result.csv SHA-256 | `9ffe21249c736e5267bf11716ad2e87ef8b71ef11a248728b4ea94d0300967df` |
| 训练 run | `local/runs/baseline-v0.1-submission-test-a-r1` |
| 训练生命周期 | `development` |
| 训练截止点 | `2024-11-01T00:00:00+08:00` |
| eligible rows | 2424 |
| source contract | `public-process-sources-v1-2773c0f38f89ce32` |
| 保护集 | `holdout_consumed=false`；H1–H4 未运行 |

该结果关联到用户在本记录前收到的提交包。仓库没有平台回执截图或平台 API 原始响应，因此不记录排名、评测集标签、重复提交次数或其他未提供信息。

## 2026-09-07 · test_a / E02

| 字段 | 记录 |
| --- | --- |
| 平台显示成绩 | `82.3290` |
| 证据状态 | `USER_REPORTED_NOT_INDEPENDENTLY_VERIFIED` |
| 候选 | `E02`；删除历史年龄列 |
| ZIP SHA-256 | `f08e18ac9f757fbab1f1ee12d57a4005f70307a4ea126f8db9cbe47167363b7e` |
| result.csv SHA-256 | `2263da2dac43cf32038a715414874cd564b5ea20e1dcf7730346330aab80bb64` |
| 训练生命周期 | `development`；截止 `2024-11-01T00:00:00+08:00` |
| 保护标签 | 未读取、未消费 |

相对冻结 baseline 用户报告成绩提高 `0.8736` 分。该差值是同一平台阶段的用户回传，不是独立复现或未来平台承诺。

## 2026-09-07 · test_a / E02-E04 target-specific blend

| 字段 | 记录 |
| --- | --- |
| 平台显示成绩 | `82.4431` |
| 证据状态 | `USER_REPORTED_NOT_INDEPENDENTLY_VERIFIED` |
| 候选 | 铁量 `0.5×E02 + 0.5×E04`；时长 `1.0×E02` |
| ZIP SHA-256 | `bbbf4b11425f1508f3e3542fc92cf437f0b0c740b8e84566e8bd56587f55ee3a` |
| result.csv SHA-256 | `a80a70275010764feb91decc6f3cd65be9b54001d15820e60a67449524bec12a` |
| 训练生命周期 | 两个组件均为 `development`；截止 `2024-11-01T00:00:00+08:00` |
| 保护标签 | 未读取、未消费 |

相对 E02 提高 `0.1141` 分，相对冻结 baseline 用户报告成绩提高 `0.9877` 分。平台只返回总分，不能据此反推任一目标或样本标签。

## 2026-09-07 · test_a / E02-E04 uniform 75/25 blend

| 字段 | 记录 |
| --- | --- |
| 平台显示成绩 | `82.6221` |
| 证据状态 | `USER_REPORTED_NOT_INDEPENDENTLY_VERIFIED` |
| 候选 | 两目标均为 `0.75×E02 + 0.25×E04` |
| ZIP SHA-256 | `679fd648ae9f1508f3e3542fc92cf437f0b0c740b8e84566e8bd56587f55ee3a` |
| result.csv SHA-256 | `3dd1e55d66829ddc8cbb88ff7cf3e608065009676c8eda24d659aa61eca136ae` |
| 训练生命周期 | 两个组件均为 `development`；截止 `2024-11-01T00:00:00+08:00` |
| 保护标签 | 未读取、未消费 |

相对上一融合候选提高 `0.1790` 分，相对冻结 baseline 用户报告成绩提高 `1.1667` 分。重复回传的同一分数只记为一次提交结果。

## 2026-09-07 · test_a / 75/25 blend with time calibration

| 字段 | 记录 |
| --- | --- |
| 平台显示成绩 | `82.7046` |
| 证据状态 | `USER_REPORTED_NOT_INDEPENDENTLY_VERIFIED` |
| 候选 | 两目标 `0.75×E02 + 0.25×E04`；仅时长减去早期 OOF 中位残差 `1.68610975` |
| ZIP SHA-256 | `d38e7f2ae71fa88ae63ca7dd43ee8a176a8366776c69181b3797211ddf6bac36` |
| result.csv SHA-256 | `9ee69b9372b385fd85b8c50e1a3ebde9db312add1a5041e4c6ac5d8d724b6b7f` |
| 校准拟合/检查 | O202406–O202408 拟合；O202409–O202410 时序检查 |
| 保护标签 | 未读取、未消费 |

相对未校准 75/25 候选提高 `0.0825` 分，相对冻结 baseline 用户报告成绩提高 `1.2492` 分。该结果作为本轮最后一次自适应平台验证；后续候选回到 development 网格预先冻结，不继续利用平台总分搜索融合或校准参数。

## 2026-09-07 · test_a / OPT-03 E07 frozen-history probe

| 字段 | 记录 |
| --- | --- |
| 平台显示成绩 | `82.3610` |
| 证据状态 | `USER_REPORTED_NOT_INDEPENDENTLY_VERIFIED`；候选仍未通过 G1 |
| 候选 | `E07_FROZEN_E02`；历史视图年龄 `{0,7,30,60,90}` 天 |
| 开发门禁 | G0 通过；G1 失败（DEV_LONG 未击败更优中位数控制） |
| ZIP SHA-256 | `c3dede4eaed2b7b6730b7b5486da2278c06e39a5c1a8073d61da24020c2c932b` |
| result.csv SHA-256 | `a622c68d08cb2d0a520925755516e6390e2f1ed449b8f9b9e5a43c9fd00f861f` |
| 训练 run | `local/runs/optimization-v0.2-opt03-e07-test-a-r1` |
| 训练生命周期 | `development`；截止 `2024-11-01T00:00:00+08:00` |
| 训练样本/视图 | 2,424 个原样本；12,120 个视图；每原样本总权重 1 |
| 保护标签 | 未读取、未消费 |

该提交是用户明确要求的单次探索性例外，不表示 E07 通过 G1，也不授权依据该
回传总分继续调整年龄、权重、融合或校准。用户回传分数比冻结 baseline 高
`0.9056` 分、比 E02 高 `0.0320` 分，但比上一份最高分 `82.7046` 低 `0.3436`
分。上一份最高分桌面包仍保存在对应的 `local/submissions/` 运行目录中，可恢复。

## 2026-09-07 · test_a / OPT-04 E09 process-change

| 字段 | 记录 |
| --- | --- |
| 平台显示成绩 | `82.8174` |
| 成绩回传日期 | `2026-09-08` |
| 证据状态 | `USER_REPORTED_NOT_INDEPENDENTLY_VERIFIED`；开发 G1 已通过 |
| 候选 | `E09_PROCESS_CHANGE_E02`；16 个过程变量的三类有符号变化量 |
| 开发门禁 | G0 通过；G1 六项全部通过 |
| 开发 J | `0.173861`；相对 E00 改善 `0.009222` |
| ZIP SHA-256 | `059f2b73baa026aa5b78ad58a641b848d777497eb77ef5d667ff1878d8bd1527` |
| result.csv SHA-256 | `bf037c118a3ee55cdc95f5964868e312453d87139c4a0de2f8eb55f53cf46ac4` |
| 训练 run | `local/runs/optimization-v0.2-opt04-e09-test-a-r1` |
| 训练生命周期 | `development`；截止 `2024-11-01T00:00:00+08:00`；2,424 行 |
| 保护标签 | 未读取、未消费 |

该候选不使用 OPT-03 的平台回传分数选取特征、窗口或权重。用户回传成绩比
上一最佳 `82.7046` 高 `0.1128` 分、比 OPT-03 E07 高 `0.4564` 分、比冻结
baseline 高 `1.3620` 分。E09 成为当前平台 incumbent；平台总分仍不作为其已
通过开发门禁的替代证据，也不用于回调 F1 定义。
