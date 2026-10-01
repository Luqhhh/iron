# ModernNCA：MAE相对原MSE的单因素开发

2026-10-02，在用户“授权，以及之后不需要申请”的持续授权内执行。本阶段冻结参照为用户回传、未经独立核验的Q75/Q100=96.3920，代表Q75；目标96.45。原MSE完整结果见[原批结果](../modernnca_q75_preparation/RESULTS.md)，其失败决定不变。

唯一科学变化是学习编码器训练损失：标准化训练目标的MSE改为MAE。逐目标WMAPE的分母固定，因此训练分区MAE与该分区WMAPE同向；这不保证外层或平台增益。默认MSE仍是源码默认值，新阶段显式选择MAE。其他设置沿用原冻结：PLR频率77、嵌入34、表示128、频率尺度0.04431360576139521、非平方欧氏距离、温度1、邻居采样0.5、AdamW学习率0.01/weight_decay0.0002、训练seed42、batch256、max_epoch240、patience25、min_delta0、float64，无额外网络块或后处理。

两个目标分别单列替换，权重固定`0.8*Q75_target+0.2*ModernNCA_target`，其他列保持Q75。科学候选仅MAE学习编码器；同初始化冻结编码器保留机制对照。原MSE的完整同分折OOF作为额外损失对照，冻结其原source-directory/seed/trial、完整审计及原始预测文件哈希，独立重算MAE−MSE增量；固定编码器预测应在1e-8内一致。该比较不追溯改变原MSE门槛或分类。

外层split seed42/3407各五折，两个目标各完整2754行OOF，设计矩阵不跨seed混合。内层分组安全五折fold0校准MAE，首次严格最佳选轮；重新初始化后仅在外层训练池refit。查询自身及同特征组标签排除，外部支持池排除整个batch同组标签；校准和外层验证不入邻居库。引用原已审计Q75缓存，无新参照拟合，不读取初赛受保护目标或把sample_id解释为时间。

完整预算20配对、80个selector/refit估计器/预处理状态、40次optimizer；单worker、BLAS/OMP/MKL/NUMEXPR及torch线程各1。0新增确认seed、0全量发布拟合、0包、0桌面写入、0助手上传。运行每600秒观察，真实终态可立即审计；RSS上限1536MiB，无时间预算、自动重试或配方缩减。

开发门沿用原批：两个完整seed各正、平均分数增量≥0.01、平均冻结编码器优势>0、平均本地包分≥96.25。96.25是继承的局部门，不是平台参照。自动分类沿用candidate_tiers.yaml；同值次序沿用mean/minimum gain及target/candidate排序，每轮跨目标最多一个探索推荐。完整覆盖、原门和分类分别报告，不以若干fold粗筛或MAE优于MSE替代相对Q75的门。

只有通过开发门的目标才准备新的确认阶段。正式晋级仍需≥4个完整seed、每seed收益正、seed层配对LCB95>0，fold层仅描述；本阶段不消耗确认seed。G0必须有训练/选轮身份、ledger闭合、独立新进程禁止拟合的NumPy冷推理、完整OOF独立标量评分及实际exit/RSS证据。G1单独报告，本地提升不写成平台成绩。

源码隔离于`local/worktrees/modernnca-mae-development`，基于原MSE源码f4b0c9d，新提交78a965890a71ee08b01ce22f539d7f048d320a99。锁定Python3.12/CPU环境42项定向测试通过，包含实际损失分派及无拟合冷状态回读。科学拟合前冻结私有manifest、数据/源码/环境、Q75参照、MSE对照和资源探针。实时状态键`modernnca_mae_q75_development_20261002`；原始预测、报告、账本及运行文件仅在私有local中。
