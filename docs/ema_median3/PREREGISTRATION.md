# EMA 三训练 seed 中位数开发复核

2026-10-03。在96.45持续优化授权内，以最新用户回传、未经独立平台核验的 `EMA_MEAN3_FULL_Q75=96.3954` 为参照，ZIP SHA256为 `016e7e9cb3f750c74509dbb51961c61fea205d0296bb233056da66005f69a396`。EMA三seed均值比旧Q75提升0.0034，支持继续研究该方向；不据此保证其他聚合方式的平台收益。本阶段仅检验一个固定候选 `EMA_MEDIAN3`，不改变仍在运行的嵌套残差批次。

复用原初始化批次 `local/runs/ema-retraining-initialization-20261002/development-r1` 的split42/3407、各完整五fold，以及各fold的训练seed42/1042/2042。成员不筛选，原基础训练、选轮、预处理和checkpoint不改。每行的三个EMA预测取中位数；最终时长为 `Q75+.75*(median(EMA42,EMA1042,EMA2042)-EMA42)`，当前均值参照为相同背景下的三成员均值。等价增量为 `.75*(median-mean)`，不扫描中位数/均值混合权重，不调整其他组合系数。铁量仍取对应同seed Q75列。

在计算新候选收益前，冻结原manifest、成功终态、原报告和OOF摘要、30个EMA估计器的60个selector/refit状态及warm/cold见证、当前平台反馈与参照包、执行脚本、协议、分类配置和锁定环境。以原physical checkpoint路径、split、训练seed、fold和trial身份核对复用，缓存别名不增加拟合数。逐fold核对训练/query ID与完整OOF，禁止跨split平均向量；仅在完整覆盖两个split后评分。当前mean3重建须与原保存列一致，中位数由独立标量排序复核，包分差用两目标总分相减和 `50*(old_WMAPE-new_WMAPE)` 交叉核对。

候选池与平局次序均只含EMA_MEDIAN3。报告相对最新mean3参照及旧Q75的两个完整split增量、fold和铁口描述，以及现行candidate_tiers。只有对最新参照两seed均正，才有另行冻结确认的资格；正式晋级仍需至少四个完整split各正、seed层配对LCB95>0及完整G0。该批标签已经反复用于开发，两seed复核不是独立新样本。分类和平台排程分开：本地失败保留具体反证，若保留人工探索必须另记其信息问题和理由，不自动发布或安排名额。

固定预算：0新训练/预处理拟合、optimizer、基础模型predict调用、确认seed、全量训练、ZIP、桌面写入或助手上传；只从已审计数组计算两个完整派生OOF及报告。工程检查使用合成小数组，0拟合。初赛及受保护月份标签不读；复赛评分标签只取原冻结OOF，0原始训练CSV读取。所有数组与机器报告保存在私有新目录 `local/runs/ema-median3-20261003/development-r1`，失败目录保留，不覆盖缓存。

锁定Python3.12、原CPU依赖，OPENBLAS/OMP/MKL/NUMEXPR线程均1。无时间预算。此项为短时零拟合复核，若持续运行则按600秒观察，实际完成事件立即审计；不启动第二个训练worker。必要检查与源码提交后才能冻结执行，独立新进程回读和实际退出码通过后才报告G0完成。
