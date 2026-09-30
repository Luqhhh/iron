# AMF_FULL_V1：完整算术交互配方的发现实验

日期：2026-09-30。状态：**新策略占位／书面设计待审阅，尚未实现或运行**。
策略编号采用独立标识 `AMF_FULL_V1`，避免与队友 V 数字编号撞号。
用户已授权按差距报告继续研究；本文件是具体方案，不把该授权写成尚未发生的规格审阅或训练许可。

## 1. 目标与选择理由

在项目当前跨分支参考 DE3 铁量＋V32 时长之上，寻找一个新的完整预测方向，面向用户报告的榜首 96.5299。
先判断整套配方是否有竞争力，再分析算术分支贡献；不承诺弥补当前 0.1550 分差。
本轮不用外部预训练权重，不下载外部数据，不使用隐藏测试目标，不封包、不上传。

比较三条可行工作路径：

| 路径 | 当前证据与限制 | 决定 |
|---|---|---|
| 完成 T2G_GRAPH_V1 | 本机既有批准及 G0；完整 DE3 私有参照尚未共享 | 保留已冻结计划，缓存到齐后继续；不新拟合参照 |
| 继续 PTaRL／ModernNCA | 队友已启动 PTaRL 恢复；ModernNCA 分区实现已占位 | 不重复队友 fits，不认领这些策略 |
| 完整算术交互配方 | 已有论文及静态研究，未找到同名已登记实现；存在必须显式适配的数值/梯度问题 | 推荐作为本对话下一条独立研究路线 |

名称去重仅基于可见 Git 路径历史、远端名称及先前研究记录；不保证不存在异名或未发布的队友实现。
原理来源是 AMFormer 的加法／乘法并行交互与可学习提示。完整模型使用所有已有特征，不是对现任残差作回归。
作者工具箱的成绩不构成本赛题 WMAPE 收益证据。

## 2. 本轮与旧实验的区别

- ADD_ADD 与 ADD_PROD **都是可入围的完整候选**，不把加法臂设为永远不能晋级的控制。
- 算术臂减加法臂只用于机制诊断；机制对照不作为完整配方入围的否决门。
- 单模型与固定 A20 增量分别报告。**正式主端点仍固定 A20，不根据开发曲线扫描/更换权重**；本轮不是搜索所有组合的最佳权重。
- 新实验开发阶段以同参照相对增益选路线，96.25 只作描述；不自动改写任何旧实验、全局候选分层或发布门槛。
- 通过本实验确认仅形成新的候选池证据；实际打包仍需满足当时有效发布规则或用户明确特批，本次不授权发布包。

因此本轮并非“多组件因果证明”：即使一个完整候选成功，也只能先确认该整套适配配方有增量。
若 ADD_ADD 胜出，不将其收益解释为乘法交互有效。

## 3. 来源与必须声明的适配

来源：[AAAI 2024 论文](https://ojs.aaai.org/index.php/AAAI/article/view/29033)、
[作者仓库](https://github.com/aigc-apps/AMFormer)，固定审阅提交
`afd31a30aa0942ee8274fe2e0e0fb044ea9c21ee`。

静态读取的关键源码：
[MemoryBlock](https://github.com/aigc-apps/AMFormer/blob/afd31a30aa0942ee8274fe2e0e0fb044ea9c21ee/models/Attention/blocks.py)、
[模型](https://github.com/aigc-apps/AMFormer/blob/afd31a30aa0942ee8274fe2e0e0fb044ea9c21ee/models/AM_Former.py)、
[模型默认配置](https://github.com/aigc-apps/AMFormer/blob/afd31a30aa0942ee8274fe2e0e0fb044ea9c21ee/config/models/base.yaml)。

两个边界来自源码静态推导，**未执行原作者故障复现**：

1. 原乘法分支对整个批次张量取最小/最大值；查询集合改变可改变预测，零范围可能产生非有限值。
2. 正 top-k 路径用注意力排序索引收集 V，再用可学习聚合；没有把选中 softmax 权重乘入聚合。
   在该分支无其他损失路径时，索引操作不向 Q/K/提示反传，不能以此宣称提示参数已被正常优化。

本轮使用作者已有的全邻居选项（`num_per_group=-1`），让 softmax 权重参与聚合；明确替换全批次归一化为逐样本归一化。
这是**完整适配配方**，不是默认作者逐位复现，G0 不宣称论文完整基准复现。

实现依据公开论文独立编写，GPL-3.0 作者文件仅作静态来源参考，不直接复制或修改后导入公开实现。
G0 数值见证独立重建本节数学，而不是调用同一生产函数验证自身。

## 4. 模型、数据与训练的具体约定

### 4.1 公共结构

两个臂分别训练铁量、时长；不做联合目标头。21 个数值特征＋铁口类别 token＋readout token，总 23 个 token。
sample_id 只用于对齐与审计，禁止成为模型输入；无时间戳，不构造样本序号/lag特征。

- token 宽 192，3 层，8 个注意力头（每头 24 维）。
- 数值 token 为逐列可学习仿射嵌入；类别词表只从拟合分区建立，未知值用保留 index 0。
- readout 为可学习 CLS；最终 LayerNorm、ReLU、单输出线性层。
- 每层两个同容量分支，token 拼接后经 46→23 的线性 token 混合，再残差相加。
- FFN：LayerNorm → 192→1536 线性 → GEGLU（输出768）→dropout → 768→192 线性 →残差。
- attention dropout .2，FFN dropout .1，float32，CPU；关闭混合精度。
- 两臂所有参数形状相同；同 target/seed/fold 的初始化和样本次序配对。每臂分支参数互不共享。

第一分支为常规全连接注意力（先 LayerNorm，再 Q/K/V 投影与 softmax 聚合）。
第二分支为提示注意力：22 个可学习提示与23个输入 token 拼接，在 token 轴用45→23线性映射形成查询，
然后 Q/K/V、缩放点积 softmax、全邻居响应聚合及输出投影。

ADD_ADD 的第二分支输入保持原 token，响应不经过算术变换。
ADD_PROD 的第二分支先使用 `log1p(ReLU(token))`，其聚合响应 Z 按**每条样本**的所有头/查询/通道归一化：

`u = (Z - min_row(Z)) / max(max_row(Z) - min_row(Z), 1e-6)`，输出 `exp(u)` 后合并多头/线性投影。

零范围时 u=0，指数输出1；任何非有限输入或训练状态拒绝，不修补查询标签、不跨查询样本归约。
参数布局固定，层数、维数、分支数、提示数和算术规则不在正式数据上扫描。

### 4.2 拟合边界与训练

- 数值使用训练分区 mean/std 标准化；复用 NumericPreprocessor 的 raw 路径，常数/非有限列按既有契约拒绝。
- y 按本拟合分区 mean/std 标准化，优化标准化 MSE，**用原单位内层 MAE 选 epoch**。
- AdamW，lr=1e-4，weight_decay=5e-4，batch_size=128，固定 lr。
- max_epochs=600，patience=60，min_delta=0；首个严格最优 epoch 胜出。
- inner 划分采用已有 group-safe inner seed42/fold0；outer 使用现任四seed已冻结折向量。
- 每个 outer fit：内层选择器拟合后，重新初始化模型/统计，在完整 outer train 重拟合到选定 epoch。
- fit seed=42，所有 selector/refit 分别显式重新设为42；两臂初始化与数据装载器 seed 相同。split seed/target/fold 只区分台账身份，不隐式平移模型seed。
- 训练/选择绝不读取 outer query 标签；原有21列重复组隔离不变。
- 推理还原原目标尺度，对候选输出固定 `max(0,pred)`，再计算固定 .8参考+.2候选；
  生预测、截断预测、最终端点三者分别审计，不根据表现选择后处理。

600/60、AdamW、batch128 为本赛题事前训练选择，不声称作者默认调度。
选中 epoch 若达到600或其它边界，保留 cap-hit；不自动增加预算或重跑。

## 5. 参照身份与 G0 前置条件

平台参考 DE3_IRON_USER_REQUESTED=96.3749（用户报告），包身份
`81d9d1b12c3b4fa9f40770b2ac2fef0caef7b500c5ff0ab0d7cc2948aa679ee0`。
其时长为 V32 原列；铁量为 DE3 列。开发和确认必须用相同 seed/fold 的 DE3/V32 参考，
不能只因原分支 EVIDENCE_STATUS 较旧就退回历史 B0。

本机完整 DE3 参照尚未共享，**正式 fits 保持0，禁止另拟合参照代替缺失缓存**。
实现和合成工程见证可在设计/计划获批后进行，正式预检必须固定代码、数据、环境、控制器、参照与四seed折身份。

G0 需要：

- 独立 NumPy/有限差分核对第二分支响应与梯度；Q/K/提示梯度实际非零，常数/零范围有限。
- ADD/PROD 各一次合成可学性：固定seed2026，训练2048/查询1024，数值服从标准正态，铁口1/2等概率；y=8*x0+3*x1*x2+2*sin(x3)+1.5*(spout==2)，不加噪声。查询MAE≤训练目标中位数常数基线的0.75倍，两臂各最多240 epochs，不能仅证明参数有梯度。
- 完整资源路径使用独立seed2027的2754行合成表，outer query为首551行、其余2203为outer train，内层再从outer train的末441行留出；资源探针仅在合成数据上关闭早停，两阶段各执行600 epochs，作完整路径上界见证；这项探针设置不改变正式内层选择规则。只报告实测时间，不按早停短路径承诺最大预算速度。
- full-path resource 包括 selector、完整outer refit、保存/冷推理；峰值 RSS 每 worker≤1GiB。
- workers最多4，按实时可用 RAM 计算，至少保留1GiB余量；数值线程与 torch intra/inter-op 全部1。
- OPENBLAS_NUM_THREADS、OMP_NUM_THREADS、MKL_NUM_THREADS、NUMEXPR_NUM_THREADS 都为1。
- 没有预计耗时拒绝门，但 max_epochs、fit/optimizer 总预算以及内存安全界固定；时间照实记录。
- G0 优化器预算**最多6次**：两臂各1次合成可学性（2次），各1次完整合成资源路径的 selector＋fresh refit（4次）；失败消耗预算，不另起目录重试。
- 不执行大规模真实目标资源探针。保存状态与参考由外部哈希锚定；独立冷审计不拟合、不重新初始化模型。
- 梯度见证使用不退化的固定合成输入；零范围见证允许梯度为零但须有限。
- float32 原单位冷推理容差事前固定为 `1e-5*max(1,train_target_std)`；保存与singleton/reverse/chunk全部核验。
- 标签访问仍需原保护配置、冻结清单摘要及追加私有台账，不允许读取保护月份目标。

这些是工程准入，不是G1收益。若6次预算无法完成完整见证，停在G0并保留全部失败证据。

## 6. 正式预算、选优与停止规则（待本书面设计审阅后冻结）

开发：2 targets ×2 eligible recipes ×2 split seeds(42,3407) ×5 folds = **40 outer fits /80优化器**。
内层选择器与outer refit各一次，失败也计费；完整覆盖后才能决策，不做fold0/1筛选。

每 target 的候选需：两seed固定A20增量都>0，均值≥0.01；然后按增量最高的臂选**最多一个**。
并列在1e-10以内时按 ADD_ADD、ADD_PROD 的事前顺序。没有机制对照差>0的附加门。
同时报告单模型WMAPE、A20端点、分出铁口/折/seed表现及历史B0差值；后者绝不代替当前参照判决。
绝对本地分96.25只描述；旧实验门槛及现有发布规则不变。

确认仅对入围 target 的选定臂运行7777/12011，各完整5折：
每 target **10 outer fits /20优化器**，两target最大**20 /40**。
不重跑开发臂，不额外训练落选臂；确认只能说明该完整候选的增量，不能提供四seed机制因果证明。
开发若无入围者，确认为0，登记该冻结配方负结果，不追加规格。

四seed确认要求：每seed A20增量>0、seed-level paired LCB95>0（mean−2.353363434801827*sample_std(ddof=1)/2，df3），折符号只描述。
每target只报告独立确认结果；两target都入围不自动产生双列组合包。
即使上述通过，也不声称平台>96.5299、不以本地增量乘固定倍数预测平台，不自动启动全量训练或发布。
多seed仍复用同一训练人群；不能当作隐藏测试代表性证明。

正式总上限：开发40＋条件确认最多20=**60 outer fits /120正式优化器**，另G0最多6合成优化器。
正式唯一运行根为 `local/runs/amf-full-v1`，工程/开发/确认仅用 `engineering-r1/development-r1/confirmation-r1`；phase claim为全实验唯一，确认必须锚定独立开发审计并重新计算选优。

所有目录一次性claim；stage失败保留，禁止覆盖、删除、重跑、换目录绕预算。

## 7. 交付、复用与执行顺序

先完成书面设计审阅，再写实施计划；尽量复用现有数据/参照/独立审计/预算控制契约，不为每条路线另发明框架。
实施计划需要明确共享代码冻结边界，不得修改T2G已冻结源或队友活跃分支。
设计及实现使用当前任务隔离工作区，不在其他训练的checkout中修改源。
正式阶段启动前确认完整参照到位、进程资源安全及预算身份；不自动恢复已暂停自动任务。
训练开始后才配置每30分钟安静检查，正常运行不通知，仅终态/失败/需要用户处理通知。

本次公开提交仅是新策略占位：规格、依据和0-fit状态；中间实现检查留本地，
成功跑完后再集中推送结果，遵守用户最新推送节奏。私有源下载件、模型、预测、原始报告/回执、ledger和ZIP留忽略目录。

## 8. 来源与当前证据

- 当前项目参考与PTaRL恢复：
  [e8b5760 EVIDENCE_STATUS](https://github.com/Luqhhh/iron/blob/e8b5760/EVIDENCE_STATUS.json)。
- 原差距报告：[LEADERBOARD_GAP_ANALYSIS.md](../research_gap_20260930/LEADERBOARD_GAP_ANALYSIS.md)。
- 校准矩阵：[LOCAL_PLATFORM_CALIBRATION.md](LOCAL_PLATFORM_CALIBRATION.md)。
- 先前AMFormer研究：[WAITING_PERIOD_NEW_ALGORITHMS_20260930.md](https://github.com/Luqhhh/iron/blob/e339011/docs/research/WAITING_PERIOD_NEW_ALGORITHMS_20260930.md)。
- AMFormer源码仅静态审阅；没有执行作者模型、安装依赖或下载权重。
- 新模型工程G0：未测；真实数据G1：未测；新增目标读取0、fits0、优化器0、包0、上传0。
