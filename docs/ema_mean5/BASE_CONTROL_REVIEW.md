# 已有BASE三成员均值的当前参照复核

2026-10-03。用户回传EMAmean=96.3954后，独立复核原三训练seed配对实验中已固定的BASE_MEAN3。它是当时的机制控制，不是本次在跑EMA_MEAN5候选池的一员。原比较数字在本复核前已经可见；本报告为回顾性缓存核对，不称新盲测或独立确认，不读取五成员批次的局部质量。

保持训练seed42/1042/2042等权、原完整split42/3407各五fold、相同Q75背景与铁量，配方仍为`Q75+.75*(mean(BASE42,BASE1042,BASE2042)-EMA42)`。相对当前EMA三成员均值的总分增量为：

| split | BASE_MEAN3 − EMA_MEAN3 |
| --- | ---: |
| 42 | −0.000046354 |
| 3407 | +0.001077053 |

G0：1,436项输入/源码身份冻结，60个原BASE/EMA估计器的120个物理selector/refit状态及原冷审计收据绑定通过；全部2754行、同split/fold、训练/query ID和原成员预测一致。重建BASE均值与原保存列逐位一致；独立新进程以math.fsum和完整双目标分数复算，最大差5.68e−14。准备、评估、独立审计均实际exit0。本次0新拟合、模型predict调用、确认seed、全量模型、包或平台排程；没有将原冷审计冒称本次新冷推理。

G1：均值+0.000515350，但一个split负、5/10个fold改善，现行candidate_tiers为exploration，失败项为both_splits_improve与minimum_improved_folds。未取得两个开发切分均正的确认准入，也未正式晋级；原EMA_MEAN3相对匹配BASE均值失败的决定保持。

此控制若在平台测试，可以回答“相同三个训练随机程序的普通均值与EMA均值是否有不同收益”，与三到五成员扩展的问题不同。当前没有登记该控制的平台反馈；已有单成员EMA与EMA三成员的平台正例，不能证明BASE三成员必然差，也不能由本地均值小正推断它会更好。保留其信息问题，暂不新增全量拟合或占用名额；待五成员完整结果后，结合已有平台反证和其他候选比较优先级，不把这一项本地负值当作所有平台探索的一票否决规则。

私有来源`local/runs/ema-retraining-initialization-20261002/development-r1`；新复核目录`local/runs/mean3-base-control-review-20261003/review-r1`保留冻结脚本、清单、派生完整OOF和实际退出对账。manifest SHA256为`4ca12adf2d47b4fadc40d267649342df88f96eca91eeeaafca042b09433379ad`，报告为`d6871e3abbe053c6d5134722d58bfa8d778e85f85f8dad37c78c2f7f98a9223c`，独立审计为`a414539e5c4dceec3db8b0f0f23238931de5530ac0f36ec4e5db3450aea1468a`。
