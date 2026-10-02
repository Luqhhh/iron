# SWA_TIME_MAE_WARMUP_V1 策略占位

用户已批准三优先级串行队列第二项。唯一候选SWA_TIME_MAE_WARMUP40：原V12随机初始化joint periodicTabM前40个完整epoch完全使用原joint两目标标准化等权MSE，从第41epoch改为每head 0.5iron MSE+0.5time MAE，再做head均值。固定切换点40，不按本轮结果调参，不使用外部数据或预训练权重。

原AdamW恒LR0.001、预处理、max240、patience25、min_delta1e−5、平均state上jointstdMAE选轮、fresh refit和最后10完整epoch等权raw参数窗口全部保留，初始化不计、梯度raw。原早停和最早最佳checkpoint可在40之前，fresh refit只到选定epoch；不强制额外尾段，不改变选轮候选池。若选定epoch<=40，该模型没有MAE refit尾段，结果按实记录，不冒称已切换。窗口允许真实MSE→MAE过渡epoch并存，不重置窗口。

新G02states2optimizer、dev42/3407五折20/20；复用原SWA_WINDOW10工程2/dev20/confirm20真实原件，0新控制／参照fit。两完整开发seed固定A20=.8Q75+.2member收益均正且平均机制vs原MSE SWA正才confirm271828/314159五折20/20。最终四seed各正、单侧seedtLCB95(df3)>0、机制均值正才正式晋级。各split仍同一官方样本，非新独立数据集。

原从头MAE两seedmean+.002903135387，但原SWAmean+.004443578049，平均机制−.001540442662，结果已推送02adb83；本配方检验MSE初期学习后MAE优化是否改善，而非重试原候选。结论只限各冻结配方，本地不预测平台。目标参照EMA_TIME_Q75用户报96.3920未经独立核验，目标96.4→96.45→96.5。

单worker/RSS1024MiB/四数值Torch线程1，无时间预算。唯一新run local/runs/swa-time-mae-warmup-v1；冻结source/runtime/data/protection/reference/control/oldledger，独立cold重算uniformwindow、真实损失阶段与恒LR、joint选轮、分区／预处理、完整／逆序／chunk预测及预算。零fit工程修复保留原失败和身份，不重复已消耗fit、不改冻结配方。

只新策略占位和完整结果push，中间实现本地commit，每commit/push私有guard。模型／预测／标签／ledger私有local，无fullfit、封平台包、上传或桌面写入。30分钟健康安静单次检查。真实终态发布后继续已批准第三零fit原连续末5epoch探索，不新增审核／代理评审／worktree。
