# EMA时长宽度512：单配方开发预登记

2026-10-02。本阶段在持续优化授权内检验一个容量变化：原periodic TabM时长EMA的width从256改为512。blocks2、成员16、embedding/frequencies16、frequency初始尺度.01、dropout.1、原MSE、AdamW学习率.001/weight_decay.0001、batch256、初始化42及EMA beta=.99全部保持；每batch仍为原单前向/单backward/单更新。没有宽度网格或其他同时改动。

已有raw-TabM容量实验、铁量旧粗筛与本候选的表征/目标/EMA组合不同；查阅登记后未找到本具体配方的完整评估。假设更宽的网络能降低现有EMA组件的表示偏差，EMA继续保留原平均方式。此假设尚未测得收益，不用其预测平台成绩，也不把上一轮一致性惩罚失败推广到容量变化。

当前冻结参照为用户回传、未独立核验的Q75/Q100=96.3920，以Q75为代表；目标96.45。科学启动前重新读取动态参照，冻结源码、环境、获授权复赛训练数据与旧缓存身份。原baseline、其他worktree、已冻结协议、旧运行与包保持隔离；外部数据/预训练权重不用。

仅替换时长组件：`Q75_time+.75*(EMA_WIDTH512_time-old_EMA256_time)`；铁量取原同split参照。复用原两seed十套EMA256和参照缓存，按物理source directory/split/trial核验。原group-safe inner seed42/fold0、训练内预处理、均值成员验证MAE选轮、patience25/min_delta1e-5/max240、fresh全外层训练refit保持。禁止查询标签进入模型，禁止跨split平均OOF向量或临时裁剪。

开发池固定split42/3407各五折、一个候选：10估计器、20次selector/refit optimizer、20个新保存状态；旧20个EMA状态另计。完整两切分相对Q75均正后才准备确认。正式晋级仍需至少四个完整split、各seed均正、seed配对LCB95>0；fold层只描述。原自动分类另报，每轮最多一个自动探索，当前不自动封包；本地也不作为平台筛选的唯一指标。

工程准入单独固定：seed961052生成2754行独立标准正态21特征、均匀spout1/2，`time=100+6*x0+4*sin(x1)+2*x2*x3+.4*noise`，前2204训练、后550查询。完整width512、240epoch fresh refit仅1次工程optimizer，须优于训练中位数常量；不依合成分数改参数。保存模型与预测，独立新进程核验训练内预处理、full-batch完全一致、逆序/37行分块差<=5e-4及原生模型回读。0官方CSV读取；工程1与科学20严格分开。

只使用单独获授权的复赛训练2754行；受保护初赛CSV及2024年11月目标不读取，sample_id不作时间。串行单worker，BLAS/OMP/MKL/NUMEXPR及torch numerical/interop均1，峰值RSS门1536MiB。每600秒观察，实际完成立即审计，无时间预算/自动重试，不覆盖失败或已有目录。

当前只登记并实施工程准备；科学启动须再完成源码对应的控制器、独立完整OOF/模型/标量审计与精确测试收据，并冻结全批数据/缓存分区清单。工程通过不能冒充科学准入或G1。0新确认seed、0全量拟合、0包、0桌面写入、0助手上传；原组合探针待测与DE3替补安排保持。
