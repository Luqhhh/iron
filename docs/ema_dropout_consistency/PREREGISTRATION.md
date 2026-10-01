# EMA时长双dropout一致性配对开发（2026-10-02）

本阶段在用户持续优化授权内另建worktree，只检验一个固定机制，不重复已完整失败的局部Mixup、耦合表征、平均跨度或通用MAE/Huber配方。当前参照为用户回传、未独立核验的EMA_TIME_Q75/Q100=96.3920，代表Q75；目标96.45。实际评估前重新检查动态参照并冻结数据、源码、环境、旧缓存身份和精确训练分区。外部数据及外部预训练权重继续禁用。

[R-Drop原论文](https://arxiv.org/abs/2106.14448)约束两次dropout的输出分布。这里是从零训练TabM回归的机制改编：在训练标准化目标坐标中，对每个样本、相同成员索引的两个预测a/b，使用L=.5*(MSE(a,y)+MSE(b,y))+lambda*MSE(a,b)。PAIR_CONTROL的lambda=0，CONSISTENCY的lambda=.5；没有系数网格。两个臂都连续执行两次独立dropout前向，主Torch RNG前进两次，一次backward和一次AdamW更新，再执行beta=.99的EMA；控制臂用于隔离双前向训练与新增一致性惩罚。没有成员间差异目标、KL分类复现或论文效果保证。

仅替换Q75时长的EMA组件：time=Q75_time+.75*(new_EMA_time-old_EMA_time)，铁量保持原同split原预测。架构、原time训练配置、初始化42、group-safe inner seed42/fold0、训练内预处理、均值/标准差、验证均值成员预测MAE选轮、patience25/min_delta1e-5/max_epochs240、fresh全外层训练refit均沿用原机制程序。损失两次前向平均及指定惩罚是唯一科学变化。推理只用保留EMA模型的一次eval前向，无校正或裁剪；F/Q拟合身份按物理source directory、split seed、trial记录。只用复赛训练2754行；同名受保护初赛CSV及2024年11月目标不读取，不推断sample_id时间含义。

开发池冻结为外层42/3407各五fold，两个臂共20个新估计器、40个selector/refit optimizer、40个新保存冷状态。复用原10套EMA和参照单元，并分别核对旧模型/预测身份；不得跨split平均OOF向量。只有CONSISTENCY可进入确认，必须两个完整split对Q75均正，且两个split相对PAIR_CONTROL均正；控制臂不追溯晋级。自动分类单独报告，每轮最多一个探索推荐；不把本地门失败等同于所有平台探索禁止，但本阶段不自动生成探索包。后续确认另冻结；正式晋级仍要求至少四个完整split、各收益均正、seed层配对LCB95>0。

G0须先完成解析梯度、零惩罚双前向控制身份、独立dropout/RNG、原网络保存/冷加载、训练分区与选轮、查询标签拒绝、完整同seed OOF计分及实际新进程冷推理检查。小形状三epoch合成训练测试只证明接口与零惩罚对照，不冒充全形状学得性或正式准入。正式启动前补齐全形状合成学得性/冷回放、固定代码身份、控制器与独立审计。验证只在本地锁定Python3.12 CPU环境执行，不恢复远程CI或同步通用依赖。

全形状工程准入固定seed961052：2754行独立标准正态21数值特征、独立均匀spout1/2，time=100+6*x0+4*sin(x1)+2*x2*x3+.4*noise；前2204行训练、后550行query。两臂各执行原宽256、两block、16成员/频率/embedding的完整240epoch fresh refit，共2个工程optimizer；都须优于训练中位数常量MAE，候选惩罚须实际非零。不要求候选在合成数据上胜过控制，不依合成分数调参。两模型与预测保留，独立新进程逐项核对训练内预处理及完整/逆序/37行分块推理，沿用数值和内存门；工程计数与正式40optimizer分开。

串行单worker，OPENBLAS/OMP/MKL/NUMEXPR与torch numerical/interop线程均为1；worker峰值门1536MiB，full-batch冷回放必须完全一致，逆序/分块最大差仍<=5e-4。每600秒观察，完成事件立即审计，不在间隙轮询，不设置优化时间预算，不自动重试。保存原失败/中断证据，不覆盖旧目录和协议。当前阶段0新增确认seed、0全量拟合、0提交包、0桌面写入、0助手上传；原未测组合探针与DE3替补排程不变。
