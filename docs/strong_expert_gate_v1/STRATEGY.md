# STRONG_EXPERT_GATE_V1 占位

用户已批准三个方向按优先级顺序执行。本轮先检验强专家条件融合：新训练固定 CatBoost MAE 树专家与原 V12 joint periodic TabM 神经专家；它是新双专家配方，树专家不冒称完整 A35/V36 组合的精确复现。与受相同收缩约束的 GLOBAL 全局权重对照配对，LINEAR_GATE 仅增加输入条件性，不预测 incumbent 残差方向。

每个外层训练分区按原 group-safe inner_seed42 分为 inner fit 与 calibration fold0。TabM 保留 V12 的 joint 标准化 MSE、AdamW、PLR、内层选择及 fresh outer refit；树在 inner fit 和完整 outer train 分别固定600轮新拟合，无校准标签选树轮数。门控只在 calibration 学习两个 selector 专家的绝对误差融合，使用 inner-fit 数值标准化与铁口onehot。TabM 选择checkpoint与门控用同一训练内校准池，可能增加内部过拟合；外层查询始终隔离，外层完整OOF决定质量。无跨outer OOF作为门控训练集，无用其他fold模型预测外层训练行。

权重 w=.5+Zβ，数值列clip[-3,3]/3、onehot及截距均绝对值≤1，约束||β||1≤.25，故每行w∈[.25,.75]。固定目标 sum|y−wA−(1−w)B|/sum(y)+.01||β||1，由稀疏线性规划精确求解。GLOBAL仅开放截距，其余同约束与正则。权重不是行标签选择结果；推理只输入特征与专家预测。完整外层专家重新拟合，门控保持内层校准学到的参数和标准化，不重新读取outer标签。

唯一端点 incumbent_iron*.8+.2*expert_mix；没有权重搜索、额外裁剪或包。每unit新状态：树selector/refit2、TabM selector/refit2、两gate2；TabM优化器2、树fit2、LP solver2。生产G0一unit全资源路径6states/2optim/2tree/2solver。开发42/3407各完整五折，共60states/20optim/20tree/20solver；两seed候选对incumbent均正且平均优于GLOBAL才同预算确认7777/12011。四seed均正、seed LCB95正及平均机制正才正式晋级。没有绝对分或时间门槛，fold指标仅描述。

冻结参照 EMA_TIME_Q75 平台96.3920（用户报告，未经独立核验）；其铁量=B0已四seed核验，DE3辅助。本机缺完整Q75时长缓存，不能报告完整包本地分。保护清单、重复组隔离、单worker/单数值线程/Torch1、1024MiB；无外部数据或预训练权重。独立新进程冷推理、预算和每partition身份审计；失败不重试、不改冻结源、不覆盖或删除证据。只占位和完整结果push当前upstream。

新目录local/runs/strong-expert-gate-v1。后续监督表征与SWA在此轮终态审计发布后另行冻结，不提前启动；不重复队友EMA短跨度。所有模型/预测/labels/账本留local；无fullfit、封包、上传、桌面写入。
