# EMA＋SiLU执行记录

按[预登记](PREREGISTRATION.md)执行，目标96.45，参照DE3_IRON_EMA_MEAN3_Q100=96.3979为用户回传、未独立平台核验。实时状态见EVIDENCE_STATUS.json的ema_silu_development_20261004。

## 工程准入

锁定Python3.12.12及CPU依赖，导入前单数值线程。25项无拟合检查和唯一合成检查全部通过；两个pytest子进程真实exit0。隐藏骨干两层为原生SiLU，周期数值嵌入的ReLU保持；相同seed17的原生作者工厂输出逐位一致，候选和原ReLU工厂的初始state_dict及RNG状态逐位一致，但输出不同。错误/缺失激活、错误架构和独立batch设置被拒绝，保存状态不能仅靠相同张量形状冒充不同激活。

另对三个正式训练seed42/1042/2042完成无拟合初始化核对：每seed候选与原ReLU参数和RNG状态逐位一致，三类别例子各183200参数；额外6次未训练工厂构造、0optimizer。此核对不是额外科学模型或质量实验。原ComponentRegressor训练循环继承关系已检查，原源码未改。

唯一合成估计器为2204行、seed964508生成的合成数据，训练初始化仍原seed42；selector/refit各2epoch，2次optimizer、14/18次原生更新。训练PID72625、独立冷进程PID72648；两保存状态、训练内预处理和选轮通过审计，完整batch逐位一致，反序/37行分块最大差1.1888261042258819e-8，合成及冷过程峰值780.2578125MiB，低于1536MiB。没有重试或新确认切分。

追加式私有证据engineering-r1/r2是JUnit与唯一合成状态，r3为正式seed的无拟合初始化检查，r4为聚合收据。前批packed真实终态已闭合并绑定。最终checks路径`local/runs/ema-silu-20261004/engineering-r4/checks.json`，摘要`68950aeffeef2e7b94d9672ae96a50a0aaead189ffc786095803efe64d9de95d`；聚合程序真实exit0，不新增拟合。

## 科学批次准备

待本次源码提交推送后创建独立detached worktree并冻结全部输入。计划30估计器、60optimizer、60新状态和60原状态冷审计；完整两split后统一读取质量。单worker、数值/torch线程1、RSS1536MiB、600秒观察、无时间预算或自动重试。当前真实数据拟合0，G1未测；0确认seed、0全量拟合、0包、0桌面写入、0助手上传。
