# SWA_UPDATE_WINDOW70_V1

继续用户要求的SWA优化训练，不生成提交包。单候选：原V12 joint periodic TabM、两目标标准化MSE、AdamW恒LR .001、随机/inner seed42、max240/patience25/min_delta1e-5/joint标准化MAE选轮/freshrefit不变；只将最近10完整epoch的等权参数平均改为最近70个完整optimizer.step后的raw参数平均。70由原开发内层10epoch×7updates机械确定，不扫描。初始化不进入窗口，梯度raw；每epoch评估平均状态，选中epoch后freshrefit相同更新规则，不强制额外轮数。前70更新用已有状态；selection/refit重置更新计数。

控制为原连续SWA_WINDOW10工程2/dev20/confirm20状态真实只读复用，零新控制/参照拟合。新G02states2optim；开发42/3407五折10outerfits20states20optim；两完整seed固定时长A20=.8Q75+.2member增量各正且机制vs原SWA均值正才确认271828/314159同10/20/20。四seed各正/seed单侧t-LCB95(df3)正/机制平均正才达本地正式门；已见split重用同官方样本，不是新独立数据，也不预测平台。Q75=96.3920用户报告未独立核验。

独立新进程零fit冷审计更新序号、更新数、窗口float64重算、选轮/分区/scaler/词表/顺序及chunk预测/预算；不得误用epoch窗口auditor核验70更新候选。单worker/RSS1024MiB/四数值Torch线程1，无时间预算。源码/环境/数据/参照/控制/旧账本冻结，原科学源和历史失败不改。每30分钟单次检查并汇报。只有新策略占位和完整结果push；模型/预测/标签/账本私有local，fullfit0/packages0/uploads0/desktop0。
