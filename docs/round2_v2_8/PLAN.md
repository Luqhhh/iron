# V2.8 核距离约束实验（冻结）

上一轮高C SVR训练约0.96%而验证5.70%/6.50%，不支持继续增大容量。本轮检查更宽的RBF平滑范围以及训练折内特征权重，沿用两组五折开发验证，不能声称独立确认。

KSM：独立目标 KernelRidge(alpha=0.1, kernel=rbf, gamma=0.005)，数值训练折StandardScaler、铁口训练折OneHotEncoder。相对V2.7 KR1只改gamma（多输出KR1本来也分别求解各目标）。

KWT：与KSM相同，但数值标准化后乘sqrt(w_j)，其中w_j=0.1+0.9*p*a_j/sum(a)，p=21，a是同验证折对应的原C2种子42模型PredictionValuesChange数值特征重要性。数值权重均值1且至少0.1；不删除特征；铁口独热权重保持1。必须检查原模型摘要、原折分配、恢复原C2 OOF；不使用验证标签计算重要性。该权重是假设，不是因果解释或已验证的最佳距离。

归一化标签：(y-s)/s，s只用训练折均值，预测恢复原单位。独立目标各20次，两路线40次真实拟合。固定AKSM/AKWT为当前目标参照与相应模型原单位等权融合，不增加拟合。候选池固定，不追加gamma/alpha/权重扫描。

正式分类仍使用candidate-tiers-v1，相对C2铁量和完整B3时长；另外报告相对DJ铁量和AORD时长差值。成本平局顺序KSM,KWT,AKSM,AKWT。推荐候选执行固定2000次配对重采样。只有候选在两组切分均超过现有最佳，才值得另行实施发布；本评估不做全量训练或封包。

新目录保存数据/原模型/配置/代码摘要、目标尺度、数值权重、解析参数、40次拟合账本和独立进程回读。反序、分批和单样本只含输入推理一致；旧模型、失败记录和五个待测包均保留。工程合成拟合单独计数，运行完整锁定Python3.12测试。平台最高仍用户回传96.0982，用户自行上传。

技术依据：[CatBoost重要性接口](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_get_feature_importance)允许从保留叶权重的模型提取PredictionValuesChange；[KernelRidge接口](https://scikit-learn.org/1.5/modules/generated/sklearn.kernel_ridge.KernelRidge.html)定义alpha和RBF gamma。这里的固定加权方案由本实验提出。
