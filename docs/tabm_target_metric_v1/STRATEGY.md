# TABM_TARGET_METRIC_V1 占位

三个已批准优先级中的第二项。第一项门控在完整测试阶段工程停止，零正式训练，不能当负收益。此轮保留V12原joint periodic TabM、16heads、2×256、原MSE/AdamW/240maxepochs/patience25与freshrefit，与BASE配对，仅TARGET_METRIC增加0.02×监督表征距离损失。无无标签预训练、外部数据或预训练权重。

每个当前训练batch取最终输出层输入，在head维求均值并L2标准化(epsilon1e-8)，距离(1-cosine)/2；铁量用该拟合分区mean/std标准化，目标距离1-exp(-|yi-yj|)，两者非对角均方误差作为辅助目标。batch不足2时为零；无投影层/额外优化器/预测后处理。输入相近与标签相近并非必然，因此辅助项可能伤害精度；完整OOF及BASE机制对照检验此配方。

同原重复组安全内层fold0选择epoch，训练/校准/outer隔离；selector与refit各自拟合预处理和标签标准化，辅助只用当前fit标签，校准仅原监督MAE选轮，freshrefit不再调辅助系数。固定iron端点0.8×当前native+0.2×新member，无扫描。新生产G0一全尺寸合成pair=4states4optim；开发42/3407各5折两臂共40states40optim；两完整seed候选都正且平均机制正才同预算确认7777/12011。四seed均正、seedLCB95正且机制平均正才正式晋级。无绝对分或时间门槛。

参照EMA_TIME_Q75用户报告96.3920未独立核验；nativeiron=B0四seed已有核验，DE3辅助。缺完整Q75时长不能报完整包本地分。原始冷推理绝对5e-4、统计rtol1e-14/atol1e-12、校准MAE重算atol1e-6；单worker单数值线程Torch1、1024MiB、冻结hash、预算账本、独立冷审计。科学失败不重试、不改冻结源、不覆盖/删除证据。只占位和完整结果push当前upstream；无fullfit、封包、上传、桌面写入。第三项原生时长SWA继续排队，不重做队友EMA。
