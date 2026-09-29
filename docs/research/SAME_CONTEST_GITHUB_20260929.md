# 同赛题 GitHub 检索与结构策略参考（2026-09-29）

本次按用户要求扩大研究范围，先核对赛题文件，再检索公开实现与核心源码。结论：当前检索确认 **1 个同比赛、同预测目标的公开项目**，不是已验证的高分方案。最有用的参考是目标级联和机理表示，不能照搬它的历史特征、训练流程或经验裁剪。本文是研究记录，不是实验预登记、晋级证据或运行授权。

## 1. 赛题与数据版本决定可迁移范围

赛题源文件：`初赛数据集/高炉铁次预测-1.pdf`；官方页面：[2026 AIC·AI＋钢铁·高炉铁次预测](https://www.aicomp.cn/tracks/tracks-6/4177.html)。在开口时刻预测该铁次的铁量和时长，两个目标以等权 WMAPE 评分。原任务提供小时炉况、变料事件和历史铁次，历史实绩必须在预测时刻已经可用。

当前控制性数据说明是 `复赛_test/README.txt`：版本 `synthetic_round2_v2`，2754 条训练、322 条测试，21 个数值输入和铁口号。匿名编号和行号**不表示时间顺序**；不再做初赛多表时间连接。此次只读取数据说明和 CSV 表头，没有读取目标数值。

因此，初赛多表项目和当前任务的模型接口可以相同，信息集并不相同。相同样本数不能证明逐行映射；不得把原始历史表接到 V2 匿名样本，不得按 ID 构造伪时间序列。当前平台参照仍为用户报告的 V32_TIME_A60V7_50=96.3727，目标 >96.4；外部项目的本地成绩不能与之直接比较。

## 2. 检索范围与分类

GitHub repository search 使用比赛全称、`高炉铁次预测`、`高炉`＋`铁次`、`AI＋钢铁`、`tap_history_train`、`pred_tap_iron`，并扩展 `blast furnace`＋`tapping`。另检索代码字段 `tap_time_len`、`tap_history_train.csv`。查询和原始响应保留在忽略的 local 研究目录。

| 项目 | 分类及实际内容 | 处理 |
|---|---|---|
| [bigbigstrange/Blast-Furnace-Iron-Prediction](https://github.com/bigbigstrange/Blast-Furnace-Iron-Prediction) | README 明确同比赛、同铁量/时长目标；初赛多表 LightGBM baseline | 核心源码已静态核对 |
| [rahulgondhali/RIST-Modelling-for-Blast-Furnace-Ironmaking](https://github.com/rahulgondhali/RIST-Modelling-for-Blast-Furnace-Ironmaking) | RIST/RAFT 热量与物料平衡求解，独立高炉机理项目 | 参考状态表示；不是同比赛解法 |
| [nav0225/Adv-BF-eta-pred](https://github.com/nav0225/Adv-BF-eta-pred) | 炉渣黏度预测，显式机理通道＋神经网络 | 参考结构；输入化学组分与目标不匹配 |
| [yingtaoluo/Ironmaking-Sequence-Learning](https://github.com/yingtaoluo/Ironmaking-Sequence-Learning) | 硅含量/炉温时序，ICA＋LS-SVM | 不作为同赛题方案；当前缺少时序信息 |
| [YoggyZhangzhen/Quantitative-Forecasting-of-Ferrous-Production-A-Time-Series-Stacking-Approach-](https://github.com/YoggyZhangzhen/Quantitative-Forecasting-of-Ferrous-Production-A-Time-Series-Stacking-Approach-) | 宏观日均铁水产量，XGBoost＋Ridge stacking | 目标和粒度不同；不移植其成绩 |
| [WHGD666/SteelPowerForecast](https://github.com/WHGD666/SteelPowerForecast) | 同钢铁相关命题，15 分钟煤气发电量多步预测 | 不是高炉铁次任务 |
| lacsar712 的 tapforge01–10 / smeltpot01–10 | Go 出铁口调度/指令投递应用，共20仓库 | 名称命中，全部排除 |

这不是“全网只有一个”的证明。代码字段检索甚至未返回已确认 baseline，说明索引覆盖有限；未公开、未索引、名称不同的参赛方案仍可能存在。没有发现可核验的同赛题获奖成绩或当前 V2 高分复现。

## 3. 同赛题 baseline：可参考机制和实现风险

检视版本固定为 [`c8f38ab3c6e32e9e73a7fdb0c221296b7711098e`](https://github.com/bigbigstrange/Blast-Furnace-Iron-Prediction/tree/c8f38ab3c6e32e9e73a7fdb0c221296b7711098e)。17 个 README/Python/requirements 文件已核对 Git blob SHA；未执行外部代码，未安装依赖，未取模型权重或数据。

| 方法 | 源文件 | 当前 V2 的价值 |
|---|---|---|
| 先预测铁量，再将 iron_hint 输入时长模型 | [train.py](https://github.com/bigbigstrange/Blast-Furnace-Iron-Prediction/blob/c8f38ab3c6e32e9e73a7fdb0c221296b7711098e/code/src/train.py) | 可研究有正确嵌套交叉拟合的目标级联 |
| 风量×温度、压差/风量、煤气利用相关乘积 | [process.py](https://github.com/bigbigstrange/Blast-Furnace-Iron-Prediction/blob/c8f38ab3c6e32e9e73a7fdb0c221296b7711098e/code/src/features/process.py) | 可研究按字段单位构造的交互坐标，不能移植6小时窗口或任意系数 |
| 变料差值与间隔、料制组合 | [burden.py](https://github.com/bigbigstrange/Blast-Furnace-Iron-Prediction/blob/c8f38ab3c6e32e9e73a7fdb0c221296b7711098e/code/src/features/burden.py) | 静态料制组合可研究；变化量与时间间隔不可用 |
| 历史铁量/时长、同铁口 lag、累积铁量 | [history.py](https://github.com/bigbigstrange/Blast-Furnace-Iron-Prediction/blob/c8f38ab3c6e32e9e73a7fdb0c221296b7711098e/code/src/features/history.py) | 当前缺少时间与映射，不能迁移 |
| 序列递推和比例裁剪 | [predict.py](https://github.com/bigbigstrange/Blast-Furnace-Iron-Prediction/blob/c8f38ab3c6e32e9e73a7fdb0c221296b7711098e/code/src/predict.py) | 不照搬：数据无顺序，裁剪没有同协议增益证据 |

静态审查发现以下限制，不能把 README 的“严格无未来信息”直接当作审计结论：

1. 时长训练输入是真实铁量，验证/推理输入是预测铁量，存在训练和部署输入分布差异；这种 teacher forcing 不构成正确的 OOF 级联。
2. `data_loader.py` 虽解析 `tap_end_time`，基础历史选择列却丢弃它；历史仅按开口时间筛选。可能把尚未完成铁次的最终实绩作为已知信息。此次没有读取原目标来证明具体受影响样本，只确认检查缺失。
3. 验证先对全历史造特征，较晚验证样本可使用较早验证样本的真实结果；测试递推追加预测结果。两种信息条件没有在同一个验证过程里复现。
4. 物理范围/铁量时长比例后处理仅在测试推理执行，验证评分没有对应步骤；裁剪常数并非本项目批准的参数。

仓库公开 [model_meta.json](https://github.com/bigbigstrange/Blast-Furnace-Iron-Prediction/blob/c8f38ab3c6e32e9e73a7fdb0c221296b7711098e/code/models/model_meta.json) 记录本地时间切分分数 **85.0854511**，验证起点2024-11-01；它不是平台成绩，也不是当前 V2 对照。没有据此认定该方案强于当前系统。仓库未提供标准开源许可证，本文只记录方法和链接，不把外部源码复制入公开实现。

## 4. 与我们已测路线对照，避免重复包装

- V12 已做联合双目标学习；“换成多任务”本身不新。V15 的 MMoE 也已完成四 seed 确认并失败，不能把普通专家混合当作新方向。
- V17 的 P-LL 是双频率初始化控制，不是流率或物理目标分解。但初赛 optimization-v0.8/v0.9 已测 rate/逆 rate，并登记负面结论；简单的铁量÷时长分解不因外部 repo 出现比例约束就重新开放。
- V4.2 已真实执行32个 PySR 槽位，最好固定四分之一增量为 −0.2165。普通短公式搜索已测，不能把 [PySR](https://github.com/astroautomata/PySR) 当新模型再跑一遍。
- V2.11 已诊断四个机理数值组合的残差；这是零拟合诊断，不是一个端到端机理表示实验，也不证明该表示会有效。
- 最新远端已有 `codex/strong-component-regularization` 的 BASE/EMA/SAM，以及 `codex/component-augmentation-representation` 的局部 Mixup、固定/可学跨特征周期表示。已读取其预登记，本文不把这些重新命名为本对话新策略。机制是否有质量收益，以各分支最终审计为准。
- 历史残差校正、旧库 stacking、NODE/ODST 与 TabR 等负面证据继续保留；GitHub 检索不推翻已有结论。

## 5. 值得进入下一次设计的结构问题

### 优先：嵌套交叉拟合的目标级联

由同赛题项目直接引出，但必须重新设计。在每个外层训练分区内部，为每一训练行生成没有看过该行目标的第一阶段预测；第二阶段同时看到原始输入和预测提示。外层查询提示只由外层训练分区产生。独立比较铁量→时长与时长→铁量，保留同主干无提示控制。

当前仓库检索未发现这个完整流程的 `iron_hint`/RegressorChain 实现。它不同于已失败的“强模型残差再拟合”，也不同于只共享隐藏层的 V12。然而提示仍是原输入的函数，**不增加测试时可见信息**；可期待的是训练表示/函数分解改善，不能凭两个目标相关就声称有大增益。还须控制第一阶段训练数据量和 OOF/查询误差分布，不能声称交叉拟合自动消除了所有分布差异。

设计阶段需先计算真实成本：每个外层单元还包含多个内部第一阶段拟合，不可只计第二阶段的20或40个 fits。停止规则、参照、预算、独立审计和发布范围须在新运行前冻结。本次没有启动该流程。

### 次优先：机理分组的乘性交互主干

从同赛题静态交互和 RIST/黏度项目的显式机理通道得到启发。将21个字段按送风/氧气、压力、温度、料制等语义分组，用受控的跨组乘积或比值坐标作为模型内部表示，与同容量原始表示做对照；原始通道保留。重点是**如何表示跨组关系**，不是再换一个 MAE/NLL 损失，也不是测试时给现有预测加偏置。

这是任务假设，不是已验证物理定律。当前数据是合成表，缺少完整化学组分、炉体状态和边界条件；不能直接运行 RIST 求解器或黏度公式，更不能填造常数。具体坐标需核对字典单位和既有冗余关系，避免把已有 `air_press_ratio` 等关系重复计作新增信息。V2.11 诊断不能用来事后挑选最有利组合；候选应在下一次设计时有限冻结。

此方向与队友的跨特征周期投影都改变表示，但具体约束不同；只有明确新增机制和必要对照后才值得运行。普通 PySR、普通 MMoE、历史 lag、比例裁剪都不列入新候选。

## 6. 本次结果与边界

研究完成；源码审查没有执行训练或目标评估，因此不能称模型 G0 已通过，也没有新的 G1 增益。新增模型 fits、完整重拟合、submission.zip、桌面写入及平台上传均为0。没有改变当前实验阈值、历史决定、参照或队友分支。

这次检索支持把下一轮设计转向表示和目标结构，**没有找到可直接替换当前方案的同赛题高分代码**。在新假设和完整成本冻结前，不启动大批训练。外部代码、检索原始响应和本地检查记录保持在忽略目录；仅本研究摘要公开。
