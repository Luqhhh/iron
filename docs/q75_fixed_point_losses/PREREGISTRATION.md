# Q75固定点损失：补齐尺度对照与铁量目标

2026-10-02，用户明确“继续”；本96.45目标持续授权生效。用户说明当前有两个平台名额，倾向充分探索后分配。本阶段不排程平台测试、不写桌面或上传。当前参照仍为用户回传、未经独立核验的EMA_TIME_Q75=96.3920；已完成Laplace时长四seed本地晋级不替代平台参照。

## 问题与历史范围

原V33单Gaussian学习尺度，固定Laplace只有位置接受损失梯度。上一轮只完成Laplace的固定/学习尺度配对，不能将Gaussian与固定Laplace的差异仅归因于损失形式。现在补齐时长的固定Gaussian，使同网络上的Gaussian/Laplace × 固定/学习尺度四格闭合。只新拟合缺失一格，已有三格只读复用。

铁量不从Gaussian NLL的历史负结果推断固定MAE也必负。原V10的MAE/SmoothL1使用TabM时长表征，V3.6为其他原始/PLE网络；本阶段为V33两层128周期嵌入网络的固定Gaussian和固定Laplace铁量配对。队友e88947a的独立Laplace树两臂对Q75为负，使用树结构/更新机制，与本神经实验不同；不重复队友SWA或树训练。

机制依据是[Stirn等的异方差回归分析](https://proceedings.mlr.press/v206/stirn23a.html)：联合似然训练可能损害位置预测。这只支持安排匹配对照，不保证本任务收益；不声称复现论文的faithful算法。Gaussian公式参照[PyTorch官方Normal](https://github.com/pytorch/pytorch/blob/main/torch/distributions/normal.py)。

## 冻结配方、数据和预算

三个新候选顺序固定：TIME_GAUSS_FIXED、IRON_LAPLACE_FIXED、IRON_GAUSS_FIXED。所有端点只改对应目标，固定0.8*Q75+0.2*member，不扫描权重、组合两列或使用外层真值选轮。固定Gaussian损失为mean(.5*((y-mu)/.5)^2+log(.5)+.5*log(2*pi))；Laplace为mean(abs(y-mu)/.5+log(1))。这是标准化目标下缩放MSE/MAE，.5来自原V33初始尺度，不是观测收益后选参。保持原三行head及初始化，无梯度的logit/scale输出不参与损失；保存明确point_loss_arm标签。

保留V33两层128 SiLU、周期嵌入、float32预处理输出再转float64网络、初始化42、AdamW lr.001/wd.0001、batch256、梯度裁剪10、max240、patience30、min_delta1e-5。原Laplace训练循环逐字节复用，inner27001分组五折held0按对应目标stdMAE选择最早严格改善；预处理/目标统计仅拟合F，fresh外层训练池重训选定轮数。验证、query不参与拟合；无跨seed向量平均。

两个完整开发split42/3407 × 五折 × 三候选，共30估计器、60optimizer、60新selector/refit状态。Q75/原Gaussian和两Laplace时长控制从已审计同seed同ID同fold缓存复用，0参照fit。铁量父列从原Q75来源的0.5*V36+0.5*V12独立重组，并验证训练/query身份；两目标2754行官方标签与固定split逐元素核对。

先做全形状2754行合成工程，每候选一个selector/refit程序，共3估计器/6optimizer/6状态；必须优于训练中位数常数且独立冷回读通过，不从合成成绩挑臂。定向锁定Python3.12检查禁止Optimizer构造，测试optimizer预算0。工程与开发分目录，源哈希冻结后不改；无自动重试，不覆盖失败/中断证据。

单worker、OPENBLAS/OMP/MKL/NUMEXPR/Torch及interop线程1，CPU及依赖版本在重启后的当前环境重新核验；uv --locked --no-sync。OS峰值RSS≤1024MiB，冷/反序/chunk37≤1e-8、独立标量≤1e-10。真实退出事件立即审计，运行中只按600秒观察；无时间预算。保护配置、数据/源码/模型和缓存source-seed-trial身份入冻结清单；复赛读取追加本地访问账本，禁止初赛2024年11月目标。

## 决定及发布边界

全部三个候选完整覆盖后按原candidate_tiers报告分类。确认准入：两个开发seed各自对Q75严格正；时长候选还须对已就绪LAPLACE_FIXED_A20的平均端点收益严格正。铁量两臂互相对照，最多一个候选跨目标进入另行冻结的确认：平均Q75总分增量最大者，完全同值按上述顺序。额外对现有候选比较仅减少相近候选排程，不改任何旧阶段门槛。

本阶段确认seed、full fit、包、桌面和上传均0。后续正式晋级仍至少四个完整seed各正且seed层单侧t LCB95>0。机制、负结果和平台探索理由分开记录；不以本地收益给96.3920加分。两个平台名额等待关键对照及候选比较闭合；第一项反馈再决定第二项，不要求一次用完。
