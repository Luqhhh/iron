# EMA＋TabM-packed 时长开发

2026-10-04，按用户对96.45目标的持续优化授权执行。新阶段参照DE3_IRON_EMA_MEAN3_Q100=96.3979为用户回传、未独立核验；原ZIP摘要528b8bf91102bea7ce120a71c560f6b021382132fb425a8f95cbfe79f9e3712c。前一mini批次G0通过、两个固定权重均两开发split负，原决定保留，不激活其条件确认。

## 问题与机制

[作者论文3.3节](https://arxiv.org/html/2410.24210v3)给出packed结构：各MLP骨干权重独立，同时训练并根据整体集成性能选轮。作者总体结果更支持权重共享，不预设packed有更好的质量；本次检验取消骨干共享后，是否提供当前EMA组件尚未覆盖的增量。先前独立小批量、外层训练seed扩展、mini及宽度实验不是此固定结构配方的测量，也不构成模型家族无效的证明。

已完成的[零拟合核对](../tabm_packed_research/REVIEW.md)证明当前tabm0.0.3支持原生packed，并确认带PLR的数值嵌入仍共享。本阶段调用原生`TabM.make(arch_type='tabm-packed', start_scaling_init=None)`，不实现自定义近似packed。固定21数值列、三类别例子的总参数为2,450,288，对照183,200；未训练forward的内存不是训练资源准入。结构带来的容量和初始化序列改变属于整体配方，不声称共同权重逐位匹配或单独识别多样性的因果效应。

## 固定训练与评估

只改变三个时长EMA估计器的骨干结构，逐项沿用原完整TabM EMA训练配置：width256、blocks2、k16、dropout.1、PLR频率.01、embedding/frequencies16、lite、AdamW lr.001/wd.0001、batch256、个体成员标准化MSE、EMA beta.99、每epoch共享一个完整无放回排列。inner seed42/fold0按EMA集成均值的标准化MAE选轮，max240、patience25、min_delta1e-5，随后从同训练seed的新初始化refit完整外层训练池。所有预处理与目标缩放只拟合相应训练分区。

训练seed42/1042/2042；开发split42/3407各完整五fold。真实旧模型按(source directory, split seed, trial id)复用，并在新进程重新检查训练/查询身份、预处理、选轮及冷推理。worker只接收训练帧与无标签query，不读取初赛2024年11月目标、外部数据或预训练权重，不把sample_id当作时间。

同split参照时长B=V32+M−V7，M是原三个EMA均值，M′是新packed三个EMA均值。候选与平局顺序固定：PACKED_A100=B+(M′−M)，PACKED_A20=B+.2*(M′−M)；铁量固定为该split的DE3铁量。完整30新估计器及所有新旧冷审计结束后才读取质量；同split内完整组装OOF，禁止跨split平均预测。无fold0/1筛选、权重扫描、成员挑选、裁剪或两列组合。

两个完整开发split各自增量均正才取得另行冻结确认的准入，最多选一项，按平均增量、再固定顺序。自动candidate_tiers沿用现有规则独立报告，不等同于四seed晋级。正式晋级要求至少四个完整split、每seed均正且seed层配对LCB95>0；fold仅描述。平台探索结合完整本地证据、平台正反例、机制、信息量与G0决定，不以本地符号为唯一标准。

## 预算与工程门

开发预算30个估计器、60次AdamW构造、60新selector/refit状态及60个原EMA状态冷审计。0新确认seed、0全量拟合、0包、0桌面写入、0助手上传。

工程仅1个2204行合成估计器、2次optimizer，selector/refit各最多2epoch，合成seed964507。其他检查无拟合：原生factory逐位输出、骨干成员梯度隔离与共享嵌入、错误架构/批量模式拒绝、查询标签隔离、完整候选/seed门、任务库存/失败停止/不可覆盖、终态事件和原生选轮步数故障注入。工程合成预算不包含此前已结束的三个未训练factory构造；二者收据独立。科学拟合须等新工程门通过，不以旧mini工程检查替代。

锁定Python3.12/torch2.14.0+cpu及原依赖；导入前OPENBLAS/OMP/MKL/NUMEXPR=1，torch/interop=1，单worker，RSS1536MiB，600秒观察，完成事件可立即审计，无时间预算/自动重试。失败保留原目录、身份和消耗，不默默减宽度、成员数或重跑。

原共享训练循环只读；新模型、审计器和控制器独立模块。新源代码提交并普通推送当前分支后，创建独立detached科学worktree `local/worktrees/ema-packed-development`，运行目录 `local/runs/ema-packed-20261004/development-r1`。冻结数据/分区、缓存、源码、配置、原生包文件、锁文件、父包和反馈，并绑定已闭合mini终态以保持阶段串行。

冷进程禁止训练，完整batch必须逐位相同，反序/37行分块最大差≤5e-4，独立融合与分数重算误差≤1e-9。核对真实optimizer构造/step、所选epoch与fresh refit以及全部子任务和两外层工具真实退出码后，分别报告G0与G1。此阶段冻结前仅进行了源码与历史元数据读取，尚无packed合成或科学拟合。
