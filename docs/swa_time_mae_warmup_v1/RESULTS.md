# MSE预热→时长MAE SWA：完整开发完成，平均机制近零偏负

固定时长端点0.8*Q75+0.2*member。前40完整epoch原joint两目标标准化MSE，第41epoch起每head 0.5iron MSE+0.5time MAE；保留原恒LR、jointstdMAE选轮、早停和十epoch等权参数窗口。

| split seed | 预热MAE对Q75增量 | 原MSE SWA对Q75增量 | 配对机制增量 |
|---|---:|---:|---:|
| 42 | +0.004404679007 | +0.004339763111 | +0.000064915896 |
| 3407 | +0.004441405960 | +0.004547392986 | -0.000105987026 |
| 平均 | +0.004423042484 | +0.004443578049 | -0.000020535565 |

G1：两个完整seed对Q75收益均为正。平均+0.004423042484，原MSE SWA+0.004443578049，仅低约0.000020535565；42略强于原控制、3407略弱。冻结平均机制>0门未满足，确认预约／states／optimizer均0，未正式晋级，也没有四seed确认或LCB结果。不能据这么小的本地差异认定平台必然下降，更不能推算平台成绩；保留近乎持平、损失机制不同的探索证据，本阶段未授权封包或安排平台测试。

相对从头MAE上一轮平均+0.002903135387，固定MSE40预热后平均恢复到+0.004423042484。但每seed对原MSE控制的机制效应仍一正一负、均值略负，尚未显示稳定的额外融合收益。两轮均在同一官方样本和已使用split上评估，不能把均值恢复当成新独立泛化确认，或证明初期表征是差异的唯一原因。结论仅限各冻结配方，不否定MAE/SWA家族。

G0：新生产selection/refit 2states／2optimizer真实完成，4新旧模型独立cold passed，max预测2.5296138659e−6、窗口4.7683715820e−7、RSS442.8671875MiB。开发40新旧冷模型passed，max预测1.2991926809e−5、窗口4.7683715820e−7、RSS443.5859375MiB。独立审计窗口、恒LR及40/41真实损失阶段、原joint选轮、fresh refit、分区／预处理和完整／逆序／chunk预测；无工程停止或重试。

原早停patience25和最早最佳jointstdMAE选轮保持，允许选定epoch<=40且不强迫MAE尾段；窗口跨切换不重置，梯度raw，初始化不计，fresh refit只训练至所选epoch。本轮10个开发refit确实全部进入MAE尾段，分别如下，不由设计假定：

| unit | 实际MAE refit epoch数 |
|---|---:|
| s3407-f0 | 56 |
| s3407-f1 | 60 |
| s3407-f2 | 72 |
| s3407-f3 | 97 |
| s3407-f4 | 88 |
| s42-f0 | 87 |
| s42-f1 | 62 |
| s42-f2 | 89 |
| s42-f3 | 85 |
| s42-f4 | 140 |

工程synthetic refit实际MAE200epochs。每个selector/refit逐epoch损失阶段与恒LR0.001已独立零拟合核对，模型原件和原审计未覆盖。架构为原V12随机初始化joint periodicTabM/frequency0.01，训练分区预处理、AdamW、max240、min_delta1e−5、joint标准化MAE校准及十个完整epoch等权raw参数窗口不变；没有新投影、预训练模型、周期LR或时长专用选轮。

新消耗G02／2、开发20／20、确认0／0，共10outer候选模型、22保存状态和22optimizer。原SWA_WINDOW10控制22states按真实source directory／split／unit只读复用，0新控制fit、0参照fit。两seed各2754唯一ID五折完整OOF、固定端点、原预测和模型身份、全部冻结source／runtime／参照哈希、实际账本、标量收益及决定通过独立零拟合复核；没有重复训练或冷模型推理。

生产前必要全测只一次：神经Python3.12.3 1436passed／23warnings；locked Python3.12.13 1031passed／82skipped／3warnings，跳过可选神经依赖等缺失项，不伪称locked有Torch。测试收据及源码／日志SHA复核通过。

Preflight SHA：4f1a01083333e5fbf1d6c3396dfb7fdbcf44929c871ac7bb5eb8c3f1755272d6。获授权复赛缓存读取有保护配置摘要及追加访问记录，未请求2024年11月初赛受保护目标。无外部数据或预训练权重、跨split向量平均或临时裁剪。singleworker／RSS1024MiB／四数值和Torch线程1，无时间预算。

Q75平台96.3920为用户回传、未独立核验；以上是时长分量对总分增量，不是完整当前包本地分数。同样本多split只说明切分稳定性。无fullfit、封包、上传或桌面写；模型／预测／标签／ledger私有local。完整结果发布后继续已批准的原连续末5epoch零fit窗口探索，不重做本配方。
