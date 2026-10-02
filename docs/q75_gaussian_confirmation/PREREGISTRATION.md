# 单高斯时长：固定Q75端点的两切分接续确认

2026-10-02，用户继续96.45目标，按当前目标内持续优化授权执行。前置开发已公开于6405dcf：42/3407两个完整seed各2754行、五折，固定 `0.8*Q75_time+0.2*GAUSS1_time` 增量+.009442093875/+.010147210139，20旧状态冷回读及独立评分通过、实际exit0。原V33协议、旧GAUSS1仅作控制以及MDN3失败均保持。

本次只在271828/314159两个已有Q75参照的完整外层切分训练GAUSS1，十个outer估计器、selection加fresh refit共20次optimizer和20个新保存状态；不重跑开发或母模型。该配额与队友SWA确认完全独立，不重用或扩充SWA预算。固定.20权重；不扫描融合权重、方差阈值、架构或损失。

科学训练复用原 `v33_mixture.MixtureRegressor`：两层128 SiLU、原周期嵌入、Gaussian NLL、原尺度下限、AdamW、初始化42、batch256、max240和patience30均不变。每个外层训练池用原group-safe inner seed27001/五折held0；selector仅用F训练，C按原时长标准化MAE选轮，fresh refit在F∪C按选定轮数训练。原V33额外计算的B0校准融合权重从未影响模型参数；本阶段权重已固定，因此省略此无效计算，不需要重新拟合B0校准参照。query删除两目标，预处理只用对应fit分区。

Q75参照只读复用原SHORT确认中的OLD_EMA及匹配基底，通过既有四seed导出binding、原warm/cold/实际exit0收据、预测摘要及同seed组件算术校验；SHORT预测不能代替OLD_EMA。显式按ID重排到本机原生frame，折号严格一致；各seed独立闭合，不跨seed平均预测。

运行前冻结源码、环境、数据、所有原参照与开发摘要。单worker、数值及Torch线程1、RSS1024MiB，不设时间预算、不自动重试，失败产物和已消耗配额保留。NativeLedger记录真实Optimizer.__init__、F/C/query身份及训练源码；每单位恰好两次，禁止预算外native调用。G0仅允许一次合成全形状selector/refit共2次optimizer，另计于科学20次；同样保存状态并由新进程冷审计，不消费官方标签。必要锁定Python3.12定向检查在正式准入前完成。

十个单位全部完成后，新进程回读20个新状态，独立验证训练均值/方差/词表、selector轨迹和实际最佳checkpoint、fresh refit轮数、finite状态、query预测及反序/37行分块，保持原1e−8容差；原账本/输出/清单逐项回核。另一个新进程用math.fsum重算四seed完整端点、fold/铁口描述、paired均值和单侧Student-t LCB95(df=3)。正式门为四seed各严格正且LCB95严格正；两seed开发formal分类不替代最终门，不恢复原96.25旧阶段门或追溯改变旧结论。

本确认0全量、0包、0桌面写入、0助手上传，平台队列不因中间结果自动改变。四seed都重用同批样本，只说明切分稳定性；本地增量不换算平台分。当前平台Q75/Q100=96.3920为用户回传、未经独立核验，目标96.45尚未实现。正式结果后再按完整证据冻结下一阶段。
