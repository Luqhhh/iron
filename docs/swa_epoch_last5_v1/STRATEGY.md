# SWA_EPOCH_LAST5_V1 策略占位

用户已批准SWA三优先级串行队列第三项。前两项从头MAE和MSE40→MAE均完整开发完成并按平均机制门未确认，真实结果分别02adb83、7503aae。现在检验原joint-selector/constant-LR SWA最后5个完整epoch等权参数窗口，避免重复训练或修改原选择epoch。唯一候选EPOCH_LAST5，不扫描窗口或融合权重。

从原SWA_WINDOW10 fresh refit实际保存的连续末10epoch raw参数中取最后min(5,selected_epoch)个等权均值，无初始化，不伪造缺失状态；整数buffer须全窗口一致。不用周期尾段、新time-selector、从头MAE或warmupMAE轨迹代替原控制。原架构、训练、预处理、选轮、fresh refit不变，控制直接读取同source directory/split/unit原uniform10预测，0新控制推理。参数均值之后预测，不用逐state预测均值替代。

四个已使用split42/3407/271828/314159各全五折，20生产窗口平均预测和20独立新进程冷回放；NumPy float64独立均值重建并逆序chunk37验证，生产state恢复和原模型/window/分区/输入/audit/账本身份全冻结。完整2754唯一ID同seed闭合OOF、固定A20时长=.8Q75+.2member，不跨split混预测或临时裁剪。

本项四seed已用于设计，严格post-selection探索，不构成新独立确认，formal_promotion_allowed=false。完整四seedmean对Q75与机制mean对原uniform10都正才保留唯一探索；各seed符号、单侧t-LCB95(df3)只描述，不正式晋级、不预测平台、不自动封包或平台排程。Q75平台96.3920用户回传未独立核验，目标96.4→96.45→96.5；本地增量不是完整当前包分数。

原科学证据和失败历史均保留，0新fit/optimizer/参照fit/G0科学优化器，所有fit/optimizer入口在回放中禁止。必要双环境原语和相邻窗口回归，singleworker/RSS1024MiB/四数值Torch1，无时间预算。唯一run local/runs/swa-epoch-last5-v1；只占位/完整结果push，中间实现本地commit，每commit/push私有guard。模型/预测/labels/ledger私有local，不fullfit/封平台包/上传/桌面写，不worktree/代理review/新方案审核。

30分钟每次检查完成后都及时用户可见汇报，健康也汇报；无间隙轮询。第三完整终态审计、独立零fit复核及发布后，三队列结束，删除swa检查卡片，不另起未经授权配方，旧AMF/五小时仍暂停。
