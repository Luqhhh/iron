# 独立MAE铁量三训练seed等权均值：配对已就绪单seed

2026-10-02，96.45优化持续授权有效，用户要求先充分探索再分配两个平台名额。Q75=96.3920仍为用户回传、未经独立核验的当前参照。SEPARATE_MAE_IRON_A20已完成四外层seed确认与全量包审计，但四个切分均使用training seed42。本阶段检验预先固定的训练随机性平均是否进一步增加收益；不把外层切分稳定性当作初始化稳定性。

唯一候选SEPARATE_MAE_IRON_MEAN3_A20：铁量=.8*Q75_iron+.2*mean(SEPARATE_MAE42_iron,SEPARATE_MAE1042_iron,SEPARATE_MAE2042_iron)，同一外层split内三个成员按固定顺序等权。时长不进入本候选，不生成双目标组合。training seed同时改变初始化和原numpy批次顺序，不声称纯初始化因果效应，不扫描seed或择优选成员。

JointMAERegressor、V33网络和预处理源码逐字节不变，直接使用SEPARATE_MAE两独立trunk原程序：两训练标准化目标，求和固定b=.5 MAE，原全局clip10、float64、2×128 SiLU周期网络、AdamW lr.001/wd.0001、batch256/max240/patience30/min_delta1e−5。target_order固定铁量、时长；时长仍参与原训练和joint标准化MAE选轮，不能把程序误述为单目标铁量训练。每个成员仅random_seed改为1042或2042；inner group-safe seed27001/held0固定，F训练、C按原joint stdMAE选最早严格最好checkpoint，再fresh外层训练池重训所选轮数。全部统计仅对应训练分区拟合。

完整开发split42、3407各五折×2新training seed=20新双输出估计器/40optimizer/40状态。原training42十个同折模型及20selector/refit状态只读复用，按source directory/split/trial保存原身份，0重复fit、0参照fit。原joint工程4optimizer/4状态的冻结源、manifest与真实终态核验后复用，0新工程optimizer。零拟合定向检查阻断Optimizer构造。Python3.12锁定CPU依赖，四数值线程及Torch/interop1，单worker，RSS≤1024MiB，600秒观察与真实完成事件触发审计，无时间预算或自动重试。

读取官方复赛训练前冻结保护配置、数据、源码、旧缓存和模型身份清单并追加访问账本；不访问初赛2024年11月目标。两seed各2754唯一ID、五折完整OOF，原铁量/时长标签及fold逐值对齐。新增40状态和复用20状态全部在独立新进程阻断拟合后冷回读：双目标全输出、训练ID、均值方差/类别词表、实际选轮、fresh refit轮数、完整/逆序/chunk37与独立NumPy前向≤1e−8。完整双输出回读通过后才取铁量列组成候选；禁止跨split平均向量。另用独立标量算术验证三均值、融合、WMAPE/分折/分出铁口/增量与决定≤1e−10。

确认准入须两个完整开发split分别对Q75均正，且分别对已就绪SEPARATE_MAE_IRON_A20均正。唯一三成员平均候选可进入后续另冻确认；单成员结果仅描述，不能改配方选幸运成员。正式晋级须四个完整split各对Q75为正且seed层单侧t LCB95>0；新增全量发布还须四个split相对原单seed均正且对应LCB95>0。candidate_tiers按原政策登记但不覆盖额外配对门；fold层仅描述，多split仍重用相同样本。

本阶段0新确认seed、0全量、0包、0桌面、0上传。原Laplace MEAN3的额外发布门失败和EMA MEAN3配对失败保留，不推广到此铁量程序。结果决定下一阶段，不把本地增量加到96.3920推算平台得分。两个平台名额继续保留。
