# SWA_TIME_MAE_V1 策略占位

用户批准按三项优先级继续。本阶段回到原恒LR、十个完整epoch等权平均的V12随机初始化joint periodicTabM，只将时长标准化MSE改为MAE；铁量MSE辅助监督保留，显式系数各0.5、每head计算后均值。不是1:3权重、不是周期LR，也不引入外部数据或预训练模型。

原AdamW恒LR0.001、预处理、240epoch上限、patience25、min_delta1e−5、平均state上jointstdMAE选轮、fresh refit及十epoch窗口不变。原冷审SWA_WINDOW10控制2/20/20states只读复用，不补训控制或Q75参照。

新G0 selection/refit 2states/2optimizer；开发42/3407各全五折20/20；两完整seed固定时长A20=.8Q75+.2member收益均正且机制meanvs原SWA正才确认271828/314159五折20/20。最终四seed各正、单侧seed tLCB95(df3)>0、机制mean正才正式晋级。已用切分仍复用同批样本，不称四新独立数据集。参照EMA_TIME_Q75用户报96.3920未经独立核验；目标96.4→96.45→96.5，本地增量不预测平台。

单worker/1024MiB/四数值线程及Torch1，无时间预算。唯一目录local/runs/swa-time-mae-v1；冻结source/runtime/data/保护配置/reference/control/ledger，独立进程冷审重算窗口、选轮、分区、预处理及顺序/chunk预测。预算防重复，保留失败证据，工程错误优先零fit恢复已有模型；不修改已冻结配方或重训已完成fit。

只策略占位和完整结果push，中间实现本地commit；每commit/push私有guard。模型/预测/标签/ledger留local，不fullfit、封包、平台上传或桌面写。30分钟单次安静检查，完成后复核并发布真实G0/G1。后两项另阶段冻结，不提前启动、不追溯修改本阶段。
