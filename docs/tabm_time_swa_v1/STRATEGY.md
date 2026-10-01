# TABM_TIME_SWA_V1 占位

已批准三个优先级中的第三项。原V12joint periodic TabM使用最后10个完整epoch原始参数状态的等权算术平均，前10epoch使用已有状态，初始化状态不计入。每epoch以当前窗口均值状态按原校准std joint MAE选轮；新freshrefit至选定epoch并取同样窗口均值。训练梯度始终在原始参数上，优化器/固定LR/240maxepochs/patience25/预处理/两目标stdMSE不变。此处是有限窗口均匀轨迹平均，不冒称包含经典SWA循环学习率，也不重复队友按每update指数加权的EMA短跨度配方。保存窗口原始状态供独立重算参数均值，预测不加临时裁剪。

BASE工程2states与开发20states复用已完成TABM_TARGET_METRIC_V1原生BASE控制（机制权重0）；必须逐文件哈希、source directory/split/trial、输入/分区、模型原生设置及完整冷审计核验，不能重跑旧控制或把copy当新fit。新生产G0仅SWA selector/refit2states2optim，开发仅SWA42/3407全五折20/20；确认只有满足门槛后才新BASE+SWA7777/12011两臂40/40，原轮没有确认控制，不伪称复用。

固定时长端点0.8×当前Q75+0.2×member，唯一alpha0.2；两开发seed都正且机制平均正才确认，四seed各正、seedLCB95正及机制平均正才正式晋级。参照Q75=96.3920用户报告、非独立核验。当前缺Q75四seed完整OOF时长/ID/折号/冻结与审计，必须验证后才能正式开发准入；不以历史V32或新BASE冒充当前参照，不补拟合当前包。等待期间进行实现和零优化器工程检查。

单worker单数值线程Torch1、1024MiB、无时间预算；数据/源码/环境/模型/参照/预算冻结，失败保留证据、无重试/覆盖/删证据。冷审计新进程零fit，原预测atol5e-4，统计rtol1e-14/atol1e-12，选checkpointMAEatol1e-6。只占位及完整终态push，不fullfit/封包/上传/桌面写入。模型、预测、标签和账本始终local。
