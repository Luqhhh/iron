## Material Passport

- Origin Skill / Mode：academic-research-suite / experiment-agent / run。
- Owner：本聊天xjn；日期：2026-10-03；Verification Status：UNVERIFIED，执行前协议。
- 既有GMR方法的独立具体配方，不称原创算法，也不保证队友未公开研究不重合。

## 独立问题与复盘依据

旧树铁量四阶段完整配方未补足Q75；继续加epoch、深度或换损失近邻的信息价值降低。
当前团队97772dc及已拉取分支的源码/配置/文档检索未见非神经联合密度条件化。
V33的神经MDN/Gauss属于同一混合分布大类，但直接学p(y|X)，不同于本轮EM学p(X,y)后解析条件化。
旧输入乘积/比值、时长链、流率标签也不是该学习机制；没有把换名字当原创。

采用[Gaussian mixture regression原方法与开源实现](https://joss.theoj.org/papers/10.21105/joss.03054)的条件化思路，
使用本机既有[sklearn 1.8 GaussianMixture](https://scikit-learn.org/1.8/modules/generated/sklearn.mixture.GaussianMixture.html)拟合；不下载代码/权重、不升级依赖、不上传数据。
条件中位数适合绝对误差决策，但联合似然训练本身不等于优化比赛WMAPE，需完整检验。
潜在机制是软工况分区下的相关协方差与局部仿射时长，不是发现了真实工况或证明外推更好。

吸收用户PDF：先完成强配方和全部切分再比较，不靠首折、训练下降、耗时或方法名称判断；
多初始化为了减少固定局部极值失配，不因预计时间缩减，不先扫大网格再挑最好。
当前Q75是已部署强参照，但团队新审计指出历史融合选择复用了全部OOF标签；
本轮用其固定完整向量比较，不把旧确认LCB解释成独立未见标签证据。

## 唯一候选及机制控制

GMR8_TIME_A20：21个numeric加spout1/2两个onehot，训练标签只time；共同标准化后24维联合Gaussian mixture。
K8为唯一待提名候选，K1为全局正则仿射控制，不能因K1得分更好事后转提名。
两spout列共线：正定对角reg .01保障计算，但这改变密度模型，可能影响soft工况，应披露而非宣称唯一真实高斯。
所有归一化只由当前outer-training计算；零std置1。禁止合并query/test拟合预处理或EM。
本阶段实际spout必须只有1/2，遇到3/4拒绝，不能静默编码为(0,0)。

full covariance，reg_covar=.01，max_iter=2000，tol=1e−5，kmeans初始化，seed42–46五独立原生EM拟合。
按最终fit平均loglikelihood最大选start，完全相等选最早；不得按校准/query分数选start。
不设内层selector：全部超参数/候选已预冻，直接每fold完整outer-training fresh fit。
任何start未收敛/非有限/协方差无效都保留失败、不丢start或自动延长重试。
2000是EM迭代上限，原生tol收敛可结束；不冒充固定epoch完整训练，报告每start实际迭代与所有开始/完成。

query只X。使用Schur条件均值/方差及仅基于X的后验权重，解混合CDF=0.5。
不是混合均值、不是各component中位数的加权和，不输入query真time/iron；不clip无效预测。
数值层使用正负log-tail平衡求根，避免宽分离/近等权混合的浮点CDF=.5平台伪根。
生产为scipy log_ndtr尾概率/brentq；独立冷路径为math.erfc及z≥26的12项Mills展开、scalar二分。
两路门控各自实现max-shift/math.exp/math.fsum归一化；不因近等权的1ULP差异静默放宽raw容差。
宽分离、近等权纯数组反例先过定向测试；实际cold超差仍失败并保留，不掩盖不稳定推断。
数值依据：[SciPy log_ndtr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.log_ndtr.html)、
[NIST DLMF 7.12.1](https://dlmf.nist.gov/7.12#E1)的erfc渐近式；有限精度实现仍须实际冷核验。
单模型允许有限负输出，固定融合/发布必须有限非负。
K1/K8比较控制完整生成式配方容量，不声称只隔离某一个参数的因果作用。

## 完整评价与计数

沿原Q75公开V2的2754行/重复特征group-safe外层seed42/3407各5fold。
每seed独立完整OOF、每行一次；不跨seed平均向量。与原材料的outer-training/query ID逐项一致。
固定时长融合=.8*Q75_time+.2*candidate，铁量不改；不得观察OOF后扫alpha/成分数/reg或以弱K1胜利替代强参照收益。
记录Q75增量、K8−K1增量、每seed/fold/spout及原单模型WMAPE；评分是50*(父WMAPE−融合WMAPE)，不映射平台分。

工程合成240行/23特征/180fit，其中4个变化数值、17个常量数值及2个spout onehot，检验零std归一化。
K1/K8共2procedure、10native fits；不读官方标签做工程筛选。
工程两个臂的合成query MAE均须严格低于fit中位数常数，检查模型确实学习而非只输出常数；
冷审重新核验此工程门，不能据该合成优势宣称官方提分。科学manifest绑定整个工程原件与cold凭证。
科学2K×2seed×5fold=20procedure、100native EM fits，最多200000次实际EM迭代。
每native start状态都保存纯数值NPZ；20procedure不是100native fits的替代计数。
每native EM start另有一次KMeans初始化：工程10、科学100原生KMeans拟合，独立开始/完成事件核验，不隐藏计数。
预定工程2+科学20procedure、10+100native fits不等于已经完成。

先合成工程再新进程冷审计，全部通过后执行完整科学两seed，再新进程审计。
冷审计仅NPZ、无pickle/新拟合，核对源/data/partition、fit-only归一化、每start最终fit likelihood、选start、真实账本与conditionals。
独立math.erf+二分与生产brentq预测raw绝对容差1e−8；独立fit likelihood容差1e−8。
不宣称重新执行原生EM，源码身份/成功输出并不等于模型质量。

## 决策与发布边界

仅K8两完整seed相对Q75均严格正，才获得另冻确认资格；K1只是机制控制。
正式四完整seed各正、seed配对LCB95>0及其他门不变，复用开发标签的选择偏差仍披露。
Q75确认271828/314159原件未到本机：开发符合资格时再请求，不能伪造确认或重跑队友已完成开发冒充原件。
本协议不发布包、不安排自动平台探索；本地负不代表平台必负，但探索例外必须另冻具体信息问题和发布范围。
不把用户要包解释为必需把未通过的配方封包；不改变历史负结果、队友队列或其他运行任务。

单worker/线程1，RSS1536MiB；每600秒观察自有任务，无时间预算/自动重试。
仅private local新目录写产物；公开只有源码/协议/摘要，助手平台上传0。
