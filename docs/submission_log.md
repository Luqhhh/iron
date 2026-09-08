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

## 2026-09-08 · test_a / OPT-05 E12 registered blend

| 字段 | 记录 |
| --- | --- |
| 平台显示成绩 | `82.9543` |
| 成绩回传日期 | `2026-09-08` |
| 证据状态 | `USER_REPORTED_NOT_INDEPENDENTLY_VERIFIED`；开发 G1 已通过 |
| 候选 | `E12_BLEND_E09_E04_80_20`；两目标固定 `0.8×E09 + 0.2×E04` |
| 开发门禁 | G0 通过；G1 六项全部通过 |
| 开发 J | `0.171713`；相对 E00 改善 `0.011370` |
| ZIP SHA-256 | `5e628c5bdc3403f44b7272498345d2727df28a87f2ab97419a4edf65c02b593e` |
| result.csv SHA-256 | `0f271c7bff4b327e54294f2e286594fd41cfc6302e9087cd71f2e2ca820ba75e` |
| 开发 run | `local/runs/optimization-v0.2-opt05-derived-grid-r1` |
| 派生预测 run | `local/predictions/optimization-v0.2-opt05-e12-test-a-r2` |
| 训练生命周期 | `development`；截止 `2024-11-01T00:00:00+08:00` |
| 保护标签 | 未读取、未消费 |

E12 按预登记的等权 horizon 聚合目标优于 E14 后冻结，未依据 `82.8174` 搜索
融合权重。桌面提交包已替换并通过本地/桌面 SHA-256 一致性校验。用户回传分数
比 E09 高 `0.1369`、比上一派生最高分 `82.7046` 高 `0.2497`、比冻结 baseline
高 `1.4989`；E12 成为平台 incumbent，但 OPT-05 不据此继续调权或叠加校准。

## 2026-09-08 · test_a / OPT-06 E16 post-blend calibration

| 字段 | 记录 |
| --- | --- |
| 平台显示成绩 | `82.9918` |
| 成绩回传日期 | `2026-09-08` |
| 证据状态 | `USER_REPORTED_NOT_INDEPENDENTLY_VERIFIED`；开发 G1 已通过 |
| 候选 | `E16_TIMECAL_E12`；E12 铁水不变、时长减 `1.68610975` |
| 开发门禁 | G0 通过；G1 六项全部通过 |
| 开发 J | `0.169902`；相对 E00 改善 `0.013182` |
| ZIP SHA-256 | `1200d4dd8dee6e797aeeba86db1ceb02293c78dd36796d2e8d75156ce77160aa` |
| result.csv SHA-256 | `608e3c72db6ca2aab0fcd0dd11805334b682c60b8c321813f0690578eeb546b3` |
| 开发 run | `local/runs/optimization-v0.2-opt06-targetwise-grid-r1` |
| 派生预测 run | `local/predictions/optimization-v0.2-opt06-e16-test-a-r1` |
| 源码身份 | clean commit `a343e96` |
| 保护标签 | 未读取、未消费 |

E16 按预登记 J 优于 E15 后冻结；E15 仅在 H1 略优，不替代聚合选择规则。桌面
提交包已替换并通过 SHA-256 一致性校验。用户回传分数比 E12 高 `0.0375`、比
E09 高 `0.1744`、比冻结 baseline 高 `1.5364`。E16 成为平台 incumbent，
OPT-06 到此关闭，不依据该分数继续调整残差或组合。

## 2026-09-08：v0.3 CB-FC-CVcal 待平台测试包（用户例外选择）

用户明确批准以当前优化阶段最好新增算法替换发布候选并生成测试包。
当前发布指针为 `configs/optimization_v0_3/active_release.yaml`。未自动上传平台，
没有平台成绩；开发 J=.1712251132，相对 C_ref 改善 .0004883073，原 .0010
门槛未通过，因此记录为 USER_OVERRIDE_G1_FAIL，不回写 G1。

训练 run：`local/runs/optimization-v0.3-cb-fc-cvcal-release-r1/`。
2024-11-01 development 截止点，2,424 条训练样本，铁量 CB08 115 轮、时长
CB02 220 轮，均使用 E09+F-C。时长校准使用截止前 299 个合法内部 OOF 样本，
重估为减 10.710930574878446 分钟；没有读取 November 保护目标。

test_a 共 335 行，对应 H2；新进程完整恢复及分批预测差异为 0，格式验证通过。
ZIP SHA-256：`7a18681495803b22f3a8a0156606f0297864be2cfced2fe0fe16339c65d3d36c`。
CSV SHA-256：`7d62409a4c433b1cab73de9593fb31cbb1042110a7ccd7b959a1b7fc6af73d91`。

桌面 `Luqhhh_bf_tap_predict_prelim.zip` 已替换为新包并核验摘要；旧包另存为
`Luqhhh_bf_tap_predict_prelim.E16-backup.zip`，原 local/submissions 的 E16 包
亦原样保留，可恢复。详细证据和使用方式见 optimization_v0_3/CURRENT_RELEASE.md。

### 后续用户成绩回传：82.2871

按最新交付的 CB-FC-CVcal / test_a 包记录，ZIP SHA-256 为
`7a18681495803b22f3a8a0156606f0297864be2cfced2fe0fe16339c65d3d36c`。
用户报告成绩 82.2871，证据状态 USER_REPORTED_NOT_INDEPENDENTLY_VERIFIED。
相对旧 E16 82.9918 下降 0.7047；E16 仍为已记录的最高平台成绩。
没有改写开发指标、G0 核验或原 G1_FAIL；没有根据平台总分继续调参、读取保护
目标或自动回退。当前发布指针及桌面新包暂时保留，回退需用户确认。

### 用户确认回退：恢复 E16

用户随后确认回退并要求提交推送。active_release.yaml 与桌面
`Luqhhh_bf_tap_predict_prelim.zip` 已恢复 E16 原包，SHA-256 为
`1200d4dd8dee6e797aeeba86db1ceb02293c78dd36796d2e8d75156ce77160aa`。
新包另存 `Luqhhh_bf_tap_predict_prelim.CB-FC-CVcal-backup.zip`，本地原运行
目录与全部失败证据保留。没有重新训练、读取保护目标或自动上传平台。
