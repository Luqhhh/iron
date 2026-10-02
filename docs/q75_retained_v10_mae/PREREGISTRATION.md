# 原V10 MAE缓存相对Q75的零训练复核

2026-10-02，用户96.45目标内持续授权，两个平台名额继续保留。当前EMA_TIME_Q75=96.3920为用户回传、未经独立核验；队友预热→MAE仅见90cbc65策略登记，未见新结果，不在本机重复其训练。

## 未充分核对的证据

V10单目标周期TabM MAE在原A35/V7_TIME参照下两个完整开发seed均有增量，但当时冻结排序只将更强的SmoothL1送入确认，后者失败。原MAE没有四seed确认；不把该历史决定改写为MAE本身四seed失败，也不把V10原96.25门或旧参照追溯删除。

小型V33固定MAE已成为当前就绪时长候选；原V10用width256、16个TabM头、dropout.1、不同周期嵌入和inner42，不能仅因两者都用MAE认为同一模型。这里在当前参照下核对旧预测的增量，先不消耗新optimizer。

## 冻结范围与证据边界

唯一候选V10_MAE_Q75_A20：`.8*Q75_time+.2*V10_MAE_member`，固定alpha.2，不复用旧跨seed选权结果、不扫描权重、不混合跨split向量。完整split42/3407各五fold，原V7 MSE同配方预测构成V7_MSE_Q75_A20机制控制；对已就绪LAPLACE_FIXED_A20另作完整同seed比较。铁量不改。SmoothL1只保留历史失败身份，不加入本阶段候选池。

原V10/V7仅保存预测.npy及fit metadata，没有这批逐fold模型checkpoint和逐轮selector trace。本阶段G0只能证明保留预测的来源、完整性及算术，不能宣称独立模型冷复现、原生optimizer重新计数或重建未保存的选轮。原单元测试的冷推理能力也不冒充这些科学模型的冷证据。即使本地收益为正，仍不能直接进入确认/发布；如有后续价值，需另冻带完整工程证据的新开发验证。

先冻结原manifest/audit/ledger/预测及相关源码、原配置、官方复赛数据/保护配置、Q75/Laplace证据和当前环境。复赛标签读取前追加访问账本。重算原完整frame digest、同seed fold digest；按原分区顺序拼接预测，逐fold对fit/inner ID digest、train-only特征均值和已保存选轮元信息，拒绝缺失/重复单元、哈希漂移、错fold长度、非有限数值。原训练器/预处理/分组代码必须与原manifest同哈希。禁止初赛2024年11月目标。

每个split只报告固定端点相对Q75、原MSE控制及就绪Laplace的总分增量、fold/spout描述和原candidate_tiers分类；独立进程用math.fsum重算向量组合与指标，误差≤1e-10。最多一个探索线索，原分类不替代四seed门。

仅在两个Q75增量各正、两个对就绪Laplace增量各正、对MSE控制平均增量正时，登记值得另行工程开发的线索；否则保留具体失败条件。不做确认准入或正式晋级，不用本地收益预测平台分数。

所有新fit/optimizer/state/冷模型/工程/确认/全量/包/桌面/上传预算均0。代码显式阻止Optimizer构造和原训练入口，测试同样阻止optimizer。锁定Python3.12、当前CPU依赖、OPENBLAS/OMP/MKL/NUMEXPR/Torch/interop均1，单worker、RSS≤1024MiB；600秒运行观察，无时间预算或自动重试。独立私有新目录，原缓存和失败记录不改。
