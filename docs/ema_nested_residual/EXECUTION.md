# 执行登记

2026-10-03。按[冻结协议](PREREGISTRATION.md)启动完整两个开发split的嵌套残差批次，尚未获得G1结果。现有平台测试队列保持，不能把在跑实验当作新增待测包。

源码提交`fe7c5d7732b441ccf1b7f97d5389155a38bd11b5`；13项锁定Python3.12检查通过，包括改变外层标签不改变训练输入、两层分组隔离、显式EMA基预测特征、两个头的新进程禁止拟合冷回读、Ridge独立正规方程及固定修正算术。工程新增2个合成头拟合、0神经optimizer，与科学预算分开。

科学目录`local/runs/ema-nested-residual-20261003/development-r1/`，577项冻结文件，manifest SHA256为`9bb704d31630756054dc4580a45dcfe6b1c23d253660d6062e867196370a84b2`。准备进程保存每个outer单位自己的T标签与无标签query，各worker不解析其他外层标签。内折模型沿用原EMA训练器与选轮，完整T应用模型复用原状态。

启动核验确认supervisor PID5373存活，子进程PID5385正在执行split42、outer fold0、inner fold1的新增拟合，OPENBLAS/OMP/MKL/NUMEXPR线程环境均为1。该PID仅是启动时的观察，后续是否存活以私有`execution/`追加式事件和实际进程为准。

串行计划40新基础估计器/80次AdamW、80新状态和40旧状态冷审计、20次头拟合；0确认seed、全量发布、ZIP或平台上传。监督程序按600秒记录自有子进程，子任务实际完成立即保存退出码与峰值RSS；失败停止后续执行并保留证据，不自动重试，没有时间预算。完成状态以`execution/terminal.json`及独立审计为准，当前不声称科学G0全部通过。

首次600秒观察及首个outer单位审计：split42/fold0已经完成4个新增基础模型、8次实际AdamW构造、12个新旧基础状态的冷审计和2个头拟合。其12个执行/审计子进程实际exit均为0，已完成子任务峰值RSS812.79MiB，头冷回读及独立算术最大差8.88e-16。577项冻结文件复核未变。`execution/first-outer-unit-audit.json`保存本次追加式核对；监督程序的`observation-001.json`在600秒点确认当时子进程存活，随后收到了该单位及下一单位复用模型的成功完成事件。这里只闭合一个outer单位的G0，尚无完整split收益；未计算局部fold质量、没有改变候选或平台顺序。
