# SWA_TIME_HUBER_V1

用户授权继续离线优化，无fullfit/封包/上传。单候选：原连续10完整epoch参数平均SWA，铁量仍MSE，时长采用delta=1的标准化Huber2：残差绝对值<=1时r²，超过1时2|r|−1，每head .5iron+.5time后mean。小误差值与梯度逐值保持原MSE，大误差限制time梯度。delta1为单一预登记阈值，不扫描，不用外层标签调阈值。

原V12 joint periodic TabM、随机初始化42/inner42、AdamW恒LR.001/max240/patience25/min_delta1e-5、jointstdMAE平均状态选轮、freshrefit、last10窗口/init排除/梯度raw全部不变。原MAE与warmupMAE未优于原SWA；本轮是中心MSE/尾部线性的新鲁棒损失，不能归因为已证明的离群机制。

复用原SWA_WINDOW10工程2/dev20/confirm20控制states，0新控制/参照fit。新G02states2optim；dev42/3407各5折10outerfits20states20optim；两完整seed时长端点.8Q75+.2member收益各正且vs原SWA机制mean正才confirm271828/314159同10/20/20。最终四seed各正/单侧seedtLCB95(df3)>0/机制mean正才正式本地门。历史切分重用官方样本，非新独立数据；本地不预测平台。两seed正但机制负的候选可单独保留平台探索，不追溯改门槛，不自动封包。

Q75=96.3920用户报告未独立核验。singleworker1024MiB/四数值和Torch线程1，无时间预算。源码/runtime/官方数据保护/四seed参照/控制及旧ledger冻结。独立cold零fit核对loss身份、原window/选轮/分区/预处理/完整顺序chunk预测/预算；终态独立零fit核对OOF与决定，原证据不覆盖。只占位和完整结果push，中间实现本地commit。30分钟单次检查后汇报，失败优先零fit排错，禁止重复fit。
