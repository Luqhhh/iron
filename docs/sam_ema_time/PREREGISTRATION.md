# SAM训练轨迹叠加EMA：时长单配方开发

2026-10-02，持续优化授权内的新阶段。参照为冻结时最新登记的EMA_TIME_Q75，用户回传96.3920、未独立核验；目标96.45。此前width512已完成并未通过其确认门，本阶段不改旧决定。历史原SAM/EMA批次明确未测试SAM+EMA，其排除只适用于原批次；原源码、协议和产物保持。

机制是一次原SAM更新之后，对已恢复扰动且已完成AdamW更新的原参数作一次EMA更新；beta=.99、rho=.05、epsilon=1e−12。两次SAM前向使用相同dropout RNG，随后恢复原单前向的RNG推进；weight decay只施加一次，EMA不平均扰动参数。每batch两次backward、一次AdamW、一次EMA。验证与最终推理使用EMA状态；训练继续使用SAM原参数。EMA从初始化参数开始，缓冲区遵循原复制规则。

原periodic TabM width256、blocks2、k16、dropout=.1、embedding/frequencies16、frequency初始尺度.01、个体成员MSE、AdamW lr=.001/wd=.0001、batch256、初始化42、inner seed42/fold0、训练内预处理、标准化成员均值MAE选轮、max240/patience25/min_delta1e−5及fresh外层refit保持。没有参数网格、新损失、裁剪或新特征。

平台历史证据：同V32参照下原EMA时长+0.0168、SAM时长+0.0031，均为用户回传；两者合成效果未知，不能相加或预测可达分数。[原反馈](../local_platform_diagnostic_release/DELIVERY.md)。机制依据为[SAM原论文](https://arxiv.org/abs/2010.01412)与[PyTorch参数平均文档](https://docs.pytorch.org/docs/2.14/optim.html#stochastic-weight-averaging)；组合收益是待检验假设。

完整开发split42/3407各五折，1候选SAM_EMA_TIME，10个新估计器/20次selector-refit optimizer/20个新状态。复用原同训练分区EMA、SAM各20个状态及10套完整参照；旧缓存身份按物理source directory/split seed/trial固定，不新增基线拟合。候选仅替换时长：`Q75_time+.75*(SAM_EMA_time-old_EMA_time)`，铁量保留同seed参照。另报告同权重旧SAM组件替换的匹配比较，不将它追溯为新候选或发布包。OOF各seed独立闭合2754行，不跨split平均向量，outer query标签不进入拟合/选轮。

两个完整开发seed对Q75和匹配SAM均为正，才准备新增确认阶段；正式晋级仍须至少四个完整split seed、每seed对Q75正、seed层配对LCB95>0。自动分类沿用现有candidate_tiers，每轮最多一个探索推荐；fold层仅描述。科学阶段0新增确认seed、0全量拟合、0包、0桌面写入、0助手上传。G0通过但本地门未过时，保留两条平台正方向交互这一探索问题，结合完整比较另行登记决定；不把所有本地负候选一律提交，不自动封包或更改既有平台队列。

先完成锁定Python3.12定向检查，包括rho=0对原EMA、beta=0对原SAM的完整轨迹/选轮退化等价，SAM恢复/AdamW/EMA更新顺序、失败不更新EMA、冷状态、训练分区与预算、完整双seed门和实际进程reap。再使用固定seed961053的2754行21特征/两铁口合成数据，前2204训练、后550 query，目标 `100+6*x0+4*sin(x1)+2*x2*x3+.4*noise`，单次240epoch工程refit。须优于训练中位数常量并独立冷回放；不要求优于原SAM/EMA，不调合成配方。工程optimizer=1、官方CSV读取=0，与科学拟合分开计数。

单worker、BLAS/OMP/MKL/NUMEXPR及torch/interop各1；已核验CPU依赖、uv --locked --no-sync --python3.12，不同步依赖。RSS门1536MiB、full-batch冷预测完全一致、逆序/37行分块差≤5e−4。每600秒检查，实际完成事件立即审计；无优化时间预算、无自动重试。原20新状态与40旧状态独立冷审计、完整OOF/标量、分区与实际退出证据闭合后才判断G1。私有运行根 `local/runs/sam-ema-time-20261002`，配置与冻结清单绑定源码、数据、环境、缓存和授权。
