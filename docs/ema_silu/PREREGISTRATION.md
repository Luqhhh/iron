# EMA共享骨干＋SiLU隐藏激活开发

2026-10-04，按96.45目标内的持续优化授权冻结。当前参照为用户回传、未独立核验的DE3_IRON_EMA_MEAN3_Q100=96.3979，父包摘要528b8bf91102bea7ce120a71c560f6b021382132fb425a8f95cbfe79f9e3712c。前批packed已经实际终态闭合；其两权重两开发split负，不改判、不进入确认。

## 单一问题

保留原完整TabM的参数共享、周期数值嵌入和EMA训练，仅将两个隐藏骨干block的ReLU改为原生`torch.nn.SiLU`，即x·sigmoid(x)。[Elfwing等原始研究](https://arxiv.org/abs/1702.03118)给出这一激活；[Ramachandran等](https://arxiv.org/abs/1710.05941)研究相应Swish形式。它保留平滑的负值响应，改变隐藏表示和梯度传递；本文提出的是对当前时长EMA的可检验假设，不把强化学习/图像实验结果迁移为本数据正收益。

已安装tabm0.0.3的原生`TabM.make`支持`activation='SiLU'`；默认ReLU。当前原EMA工厂没有显式改动activation。根状态和根预登记搜索中，V33类SiLU还同时有不同主干宽度、损失和训练协议，RealMLP使用自己的完整训练程序；这些不是此配方的配对试验。根HPO筛选也没有隐藏SiLU这一臂。搜索范围不代表所有历史worktree全局不存在。

数值嵌入内部ReLU保持原样，只有隐藏骨干激活改变。SiLU没有新参数，要求工程验证同训练seed下参数名、形状、初始张量及RNG消耗与原ReLU工厂逐位相同，输出允许不同。这是初始化控制，不声称训练后路径仍相同，也不保证参数平滑化提高泛化。保存状态必须绑定激活元数据；两者state_dict形状相同，不能只靠strict加载判断架构身份。

## 训练、候选与门槛

固定原训练：21数值特征、训练池铁口词表、PLR频率.01/embedding16/frequencies16/lite；完整TabM arch_type=tabm、blocks2、width256、k16、dropout.1；AdamW .001/wd.0001、batch256、各成员标准化MSE、EMA beta.99、每epoch共享无放回顺序。inner seed42/fold0仅在内层训练拟合预处理和目标缩放，按EMA成员均值标准化MAE选首次严格最好轮，max240/patience25/min_delta1e-5；再从同seed新初始化fresh refit完整外层训练池所选轮数。原训练循环只读继承，不增加训练日程、损失、特征或后处理。

开发split42/3407各完整五fold，训练seed42/1042/2042。旧三个EMA按真实source directory/split/trial复用并重新独立冷核验；工人只接收对应训练帧和无标签query。禁止初赛2024年11月目标、外部数据或预训练权重，不将sample_id解释为时间。

每个split，M是原EMA三seed均值、M′是本批SiLU三seed均值，固定参照时长B=V32+M−V7；铁量固定该split的DE3。候选与平局顺序：SILU_A100=B+(M′−M)，SILU_A20=B+.2*(M′−M)。没有权重搜索、成员选择、裁剪或跨split向量平均。全部30估计器、新旧冷审计完成且真实子任务退出后才比较完整OOF质量，不能用fold0/1粗筛。

两完整开发split各增量均正才准入另行冻结确认；最多一项，先平均增量再固定顺序。自动candidate_tiers独立报告，不能代替正式至少四完整seed、每seed均正、seed层配对LCB95>0的门。平台排程结合完整本地、机制、历史反馈和信息量，不用本地符号单独预测平台结果。

## 工程和资源预算

科学预算30新估计器、60optimizer、60新selector/refit状态和60原状态冷审计。0新确认seed、0全量拟合、0包、0桌面写入、0助手上传。

工程预算1个2204行合成估计器、2次optimizer，合成seed964508，selector/refit各最多2epoch。先做无拟合检查：原生工厂输出、配对参数/RNG初始化、隐藏激活位置和嵌入保持、错误激活元数据拒绝、完整候选与切分门、查询标签/行身份隔离、不可覆盖、失败停止、原生步数与选轮和终态故障注入。合成阶段只执行一次，并独立新进程禁止训练地回读两状态。旧mini/packed工程通过不能替代本阶段准入。

锁定Python3.12/CPU依赖；导入前OPENBLAS/OMP/MKL/NUMEXPR=1，torch/interop=1，单worker，RSS1536MiB，每600秒观察，真实完成事件可触发审计；无时间预算或自动重试。完整batch冷推理逐位相同，反序/37行分块≤5e-4，独立融合/分数差≤1e-9。失败保持原目录与消耗，不偷偷调宽度或阈值。

工程通过、源码普通提交推送后，在`local/worktrees/ema-silu-development`冻结独立detached快照；科学目录`local/runs/ema-silu-20261004/development-r1`。绑定全部数据/分区、旧缓存、源码、原生包、锁文件、父包、反馈及packed已闭合终态。新阶段启动前重新核对最新参照，改变则拒绝按过期参照启动。所有实际子退出、控制器/跟进器工具退出及独立终态审计闭合后分别报告G0/G1。

冻结时本批尚无模型构造、合成或科学拟合；截至此刻仅检查历史文字、实现源码及原始文献。
