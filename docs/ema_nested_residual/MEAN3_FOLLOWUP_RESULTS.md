# 新EMA_MEAN3参照上的残差衔接结果

2026-10-03。按[独立衔接协议](MEAN3_FOLLOWUP_PREREGISTRATION.md)，原批次真实成功退出及完整独立审计后才激活输入。参照为用户回传、未经独立平台核验的EMA_MEAN3_FULL_Q75=96.3954；只修正训练seed42成员，固定`mean3+.25*.25*head`，两候选均完整评估，没有选权或新增拟合。

| 固定修正器 | split42总分增量 | split3407总分增量 | 平均增量 | 决定 |
| --- | ---: | ---: | ---: | --- |
| RIDGE | +0.000149646 | +0.000188731 | +0.000169189 | 取得另行冻结确认的准入，暂不消耗确认或平台资源 |
| GBM | −0.001201093 | −0.001647592 | −0.001424343 | 本固定配方不确认 |

G0通过：1,534项实际输入和源码身份冻结，prepare/evaluate/独立audit真实exit均为0；同seed的ID、fold、标签、铁量和Q75背景逐位一致，各2754行完整覆盖。独立进程先修正原seed42成员、再以math.fsum求三个成员均值与双目标评分，最大差5.68e−14。0新拟合、模型predict调用、确认seed、全量拟合、包、桌面写入或助手上传。原Q75结果与决定完整保留。

G1：RIDGE现行自动分类为exploration，5/10个fold改善，仍未达到minimum_improved_folds；GBM为not_shortlisted。两个开发切分正收益不满足四seed正式晋级门，也不保证平台正收益。鉴于修正幅度小、当前平台信息问题优先级有限，保留准入而暂不启动确认，不生成新待测包；这不是增加数值硬门槛，也不是因预计训练时长否决路线。既有EMA平均的平台正反馈支持继续研究其机制，不能直接外推本残差头的平台成绩。

私有证据在`local/runs/ema-nested-residual-mean3-20261003/review-r1/`：`manifest.json` SHA256为`35c13c04acb562d6642d364efc19a5abd64309211d4886e2d06c223cd92122cd`，`report.json`为`b2dc0591686746ae03c343daa517449fd187ba5c1e7b78e26b1fbb4cbd4d796d`，独立审计为`fb7cf22193fb9f2da9ab71c234534f0e230239f84c0605ed8a2495339aea58fb`；实际退出另存`terminal-reconciliation.json`。本地结果均为重复使用开发样本的切分证据，不称新独立样本或平台预测。
