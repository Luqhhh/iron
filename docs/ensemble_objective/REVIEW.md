# 集成训练目标零拟合核对

2026-10-04，目标96.45；当前参照DE3_IRON_EMA_MEAN3_Q100=96.3979为用户回传、未独立平台核验。SiLU仍按原冻结批次执行，本核对不读取其进度、预测或质量。

[TabM作者论文§5.1](https://arxiv.org/html/2410.24210v3#S5.SS1)区分成员损失均值与集成预测的损失。当前`v12_joint.joint_loss`计算前者，`ComponentRegressor._train`仍以EMA成员均值的校准MAE选轮。[Jeffares等NeurIPS2023论文](https://papers.neurips.cc/paper_files/paper/2023/file/2bde8fef08f7ebe42b584266cbcfc909-Paper-Conference.pdf)分析直接集成目标下成员相互抵消偏差的现象及泛化风险；其结果不能直接认定本数据上必败。

以下为标量输出空间的代数，按行、目标平均后也成立。设成员输出p_j、成员数k、均值m、目标y：

\[
 I=\frac1k\sum_j(p_j-y)^2=(m-y)^2+V,
 \qquad V=\frac1k\sum_j(p_j-m)^2.
\]

因此插值目标L_b=(1−b)I+b(m−y)²=(m−y)²+(1−b)V，其输出梯度为2/k·[(m−y)+(1−b)(p_j−m)]。b=1时，任何保持均值的成员偏移都不影响损失；b<1仍惩罚这种差异，b>1在无约束输出空间沿该方向无下界。该恒等式没有证明参数空间一定出现这样的轨迹，也不证明差异越小越好。这里b是损失插值系数，和当前EMA衰减beta=.99不同。

例如k=2、m=y，成员从(y,y)移至(y+10,y−10)，集成MSE仍为0，成员MSE变为100。不能把成员离散度增大当作可泛化信息增加。平方损失分解也不能直接替代MAE分解；直接对均值计算MAE同样无法区分这些保持均值的偏移。

这与已失败的[双dropout一致性配方](../ema_dropout_consistency/PREREGISTRATION.md)不同：后者约束同一成员在两次dropout前向中的差异，没有奖励成员间离散度。[最终融合选轮](../ema_fusion_selection/PREREGISTRATION.md)改变校准选择指标，也没有改变训练目标。根docs/configs/src/tests的限定关键词检索未找到同名成员负相关目标，不据此声称所有历史worktree均未尝试。

决策：直接均值损失暂不进入科学候选池，理由是已有具体机制反证，尚无本数据增量证据；“推理用均值”不足以单独支持启动。保留中间系数为未测研究问题，不宣称整类方法无效，也不预选系数、改门槛或占平台名额。SiLU完整结果后再综合选择下一阶段。

私有范围及证据位于`local/runs/ensemble-objective-review-20261004/review-r1`。先冻结scope再执行锁定Python3.12标准库Fraction核对：k=2/3/16、b=0/1⁄2/1/5⁄4共12个合成例子、84项解析梯度与精确中心差分核对全部通过，实际exit0。例子只核对代数，不是候选权重搜索。原论文及本地源码摘要在manifest中；0模型构造、0拟合、0optimizer、0官方数据/运行批次读取、0确认seed、0全量拟合、0包、0桌面写入、0上传。G0仅为代数和来源核对，G1无测量。
