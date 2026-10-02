# SWA_TIME_SELECT_V1：时长选轮

用户批准按优化优先级继续，第一项修正注入已完整完成并推送 a04e858，没有解决271828负增量。第二项只改变原SWA内层选轮指标：joint标准化MAE → 时长标准化MAE。原joint两目标标准化MSE训练、TabM结构、AdamW恒定LR、十完整epoch窗口、max240／patience25／min_delta1e-5和fresh refit不变。校准集仅来自相应外层训练集，不用outer query或Q75 OOF选轮。

固定端点0.8Q75+0.2member，当前登记Q75=96.3920仅用户回传。42／3407完整开发，两个seed收益各正、平均相对旧joint-selector SWA机制收益正才确认271828／314159；最终四seed各正、seed单侧t-LCB95>0、机制均值正才正式晋级。无绝对本地分或时间门槛，不据本地符号预测平台。旧四seed是反复使用的同一批样本，候选尚未评估；不可称四个新独立数据集。

复用原SWA所有已冷审控制，包括合成G0的2states、开发20states、确认20states，0新控制拟合。新候选预算G0=2states／2optimizer，开发20／20，条件确认20／20；先完整测试，再唯一G0，准入通过后正式开发。每fit保留selector＋fresh refit及窗口原状态，独立新进程零fit重算参数均值、时长选轮／原control联合选轮、分区、预处理、完整与chunk预测和账本。

原模型／窗口／控制／Q75参照／输入／账本／所有历史停止与原决定均保持不变。本策略使用主checkout新本地Git分支codex/swa-time-select-v1，不开worktree。不fullfit、封包、上传或桌面写入，只占位与完整结果push。单worker1024MiB，四数值线程／Torch1，600秒一次安静检查；科学冻结后不修改、重试、覆盖或删除失败证据。