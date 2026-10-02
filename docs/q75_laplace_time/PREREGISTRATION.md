# Laplace时长网络：固定尺度与可学习尺度的完整开发

2026-10-02，用户继续96.45目标，使用现有持续优化授权。当前平台参照仍为用户回传、未独立核验的Q75/Q100=96.3920，Q75代表包SHA256 `41a046d5ce74e8a9c7c9acb124fa680cabf9e87e4a79edb50d81efb625bde825`。已完成GAUSS1_A20四seed正式晋级及全量包，但没有平台成绩；不把它替换成平台最佳。

既有证据：V33的GAUSS1时长控制后来在独立Q75阶段四seed均正；原MDN3失败、GAUSS1铁量负收益均保持。V34 CRPS树的固定500轮失败伴有较多触顶，不能推广到全部条件分布方法。队友`origin/codex/joint-support-research-20261002`的0b384a0只有Laplace树合成G0，尚未官方开发。本阶段是同一V33神经网络的损失/尺度对照，不重跑队友树或SWA；公开分支已检索到当前提交，没有声称排除队友未公开路线。

假设：Gaussian NLL对当前组合的互补性未必意味着学习尺度本身有效。Laplace位置对应条件中位数，固定尺度下NLL与MAE仅差正常数和倍数；学习尺度可能重新分配残差权重，也可能损害位置精度。公式依据[PyTorch Laplace实现](https://github.com/pytorch/pytorch/blob/main/torch/distributions/laplace.py)，风险背景参考[Stirn等，AISTATS2023](https://proceedings.mlr.press/v206/stirn23a.html)。本阶段不声称复现该论文方法或预言平台收益。

冻结两臂、只改时长：`LAPLACE_FIXED`和`LAPLACE_SCALE`。两者复用V33 GAUSS1的原两层128 SiLU/周期嵌入、float32数值预处理再转float64网络、原三行head和相同初始化42、AdamW lr.001/wd.0001、batch256、梯度裁剪10、max240、patience30、min_delta1e-5。目标仍为对应fit分区的均值/总体标准差。损失为`mean(abs(y-mu)/b + log(b) + log(2))`；FIXED的b恒为.5个标准化目标单位，SCALE保留原softplus+.05尺度head、原初始尺度.5。FIXED的尺度/logit输出不影响损失及位置预测；两臂位置初始化相同。未增加后处理或改特征窗口。

每臂端点均固定`0.8*Q75_time+0.2*arm_time`，原Q75铁量不变，不扫描权重。两臂都允许晋级，SCALE−FIXED仅作配对机制描述，不因SCALE输给控制而删除控制资格。另报告相对已完成GAUSS1_A20的同seed端点增量；其参考向量只读复用原开发冷审计结果，不重复训练Gaussian或Q75。

完整开发为42/3407各五折、每折两臂，总20个outer估计器、40次selector/fresh-refit optimizer及40状态。沿用原group-safe outer folds和inner seed27001五折held0；F拟合预处理，C按标准化时长MAE选轮，fresh outer refit按选择轮数。query不含目标。每seed2754行完整OOF闭合，禁止跨seed平均预测。先冻结所有数据、源码、环境、参照身份及保护配置，再追加访问账本后读取授权复赛标签，禁止初赛保护标签。

G0先完成两臂各一个全形状合成outer程序：总2754行、21个连续特征加ID/铁口，外层fold0约551行query，原内层划分与全240/patience30规则，4次工程optimizer、4状态。两臂分别优于对应训练中位数常量才准入。合成数据含随输入变化的Laplace噪声，只是实现检查，不据合成优劣选择臂。单元测试禁止任何optimizer构造，不另外消费训练预算。

所有fit使用原生Optimizer构造账本，每unit恰好两次，身份含source directory、整数split_seed/fold和arm。全新进程冷回读所有新状态，校验训练均值/方差/词表、完整选轮轨迹、最佳checkpoint、fresh轮数、finite参数、Laplace arm标签、整批/逆序/37行分块预测，保持1e-8容差。位置预测与原单分量通用reader相同，另用带Laplace标签的loader回核。独立新进程用math.fsum重算端点、整体/fold/铁口WMAPE、配对收益、候选分类与决策。

两开发seed均对Q75正，且相对现有GAUSS1_A20的平均端点收益为正，才允许另阶段冻结确认；最多一臂，按平均Q75收益降序，完全同值时FIXED优先。现有candidate_tiers分类完整报告，每轮至多一个自动探索建议；分类不能代替本阶段确认条件。此处未满足正式四seed晋级，后续仍需至少四完整seed各正及seed配对LCB95>0。负结果保留机制与平台探索理由，但不自动封包。

单worker、OPENBLAS/OMP/MKL/NUMEXPR及Torch线程1、RSS1024MiB、锁定Python3.12 CPU环境，600秒观察、无时间预算、无自动重试。0确认拟合、0全量、0包、0桌面写入、0助手上传；若确认条件通过，再单独冻结下一阶段，不临时改本阶段门槛。原Gaussian和EMA_MEAN3待测包保持，DE3仅替补。
