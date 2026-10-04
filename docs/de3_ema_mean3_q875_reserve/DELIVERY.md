# DE3＋EMA三成员Q87.5候补包

2026-10-04。用户明确要求“生成这个提交包，写桌面，暂时候补”。已按[事前固定配方与预算](../../configs/de3_ema_mean3_q875_reserve/SPEC.json)生成 `DE3_IRON_EMA_MEAN3_Q875_RESERVE`，完成独立包审计、桌面复制和独立回读。**候补，暂不平台测试，不进入当前优先测试队列。** 当前用户回传最佳仍为Q100组合的96.3979，未经独立平台核验。

桌面文件：`C:\Users\lqh22\Desktop\submission-DE3-IRON-EMA-MEAN3-Q875-RESERVE-20261004\Luqhhh_bf_tap_predict_round2.zip`。同目录README明确标注候补与暂不平台测试。

- ZIP SHA256：`30f1f47fbc5656643236d7a76d846fe81e3d9c3d8c0337bba2cfc34a56ade187`。
- CSV SHA256：`e41463c1767faaee2b71d1b0ef6be6af4067b8a146c4985e62316086236784be`。
- 私有包：`local/runs/de3-ema-mean3-q875-reserve-20261004/release-r1/DE3_IRON_EMA_MEAN3_Q875_RESERVE/Luqhhh_bf_tap_predict_round2.zip`。

固定铁量直接复制已测Q100包的DE3原字段字符串，与Q75包及原DE3包逐字段一致；时长为已测Q75、Q100两包原字段数值的等权中点，即q=.875。使用80位Decimal计算并以十进制原值输出，不选择权重、不重新推理模型、无裁剪或额外后处理。Q75、Q100最佳包及历史候补均保留。

G0通过：Python3.12.12新进程冻结并复核12项输入/源码/依赖锁身份，绑定两个已测父包、原DE3包、父包审计和反馈。独立审计使用Fraction有理数重算322行中点，与Decimal生成结果精确一致；每行位于两端预测之间。铁量原字符串差异0，ZIP仅含result.csv、CRC正确，322个唯一ID按官方模板顺序，预测有限非负。桌面包与私有新包字节一致，候补标识复核通过。生成、独立包审计、桌面复制及独立桌面审计均实际exit0。使用既有父包工程证据，本次无新模型冷推理或拟合。

G1为用户指定、平台未测的中点候补，未正式晋级。两端回传96.3977/96.3979不证明中点优于当前最佳。绝对误差评分下的条件中点下界仅是数学界，不是预测或实测成绩，且依赖同批样本、原包/回传身份、相同评分及报分舍入等条件。保留本地历史决定，不因候补生成而安排上传、推算名额或追加其他权重。

本批0新拟合/optimizer/CV/确认seed、1新包、1桌面复制、0助手上传。私有证据位于 `local/runs/de3-ema-mean3-q875-reserve-20261004/release-r1/` 的manifest、release、package-audit、desktop-copy、desktop-audit与terminal-reconciliation收据；机器状态键为 `de3_ema_mean3_q875_reserve_20261004`。
