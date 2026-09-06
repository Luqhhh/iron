# 高炉铁次预测：AI 主导 baseline 完整实施方案 v2

> 2026-09-06。此版本替代上一版“等待用户 D1/D2/D3 判断”的方案。技术决策已由 AI 作出；当前交付是设计、配置与执行约定，不是已训练的模型或可运行代码仓库。缺少真实 CSV 和 data_dictionary.xlsx，不能给出实测成绩。

## 1. 任务事实与来源边界

附件 S1/S2 规定：在每个目标铁次开铁的 reference_time，预测最终 tap_iron（吨）和 tap_time_len（分钟）；仅使用该时刻及之前已产生的信息。训练为 2024-03 至 2024-11，A 为 2024-12，B 为 2025-01，C 为 2025-02-01 至 2025-03-26。只有训练标签被提供。时间统一为 UTC+8。

历史实绩表已经是铁次级聚合表，不重复聚合成更粗粒度。实际主键和字段时间口径查 data_dictionary.xlsx，样本数以实际文件为准。最终 result.csv 严格三列 sample_id、pred_tap_iron、pred_tap_time_len；ZIP 根目录仅含该 CSV。（S1/S2）

目录、模型、验证、窗口与验收阈值均为本项目 AI 设计，不是主办方指定的方法或已验证的工业结论。当前仅有赛题 PDF/README，没有训练结果。

## 2. 范围与仓库

仓库名 bf-tap-predict，默认私有；但私有不意味着赛事数据获准上传。原始数据放授权本地目录、只读；真实模型、预测、审计与实验报告默认不进入 Git。当前不搭 Web 服务、数据库、分布式平台或多 Agent 调度基础设施。

```text
bf-tap-predict/
├── README.md
├── AGENTS.md
├── pyproject.toml
├── requirements.txt                 # 实际试装后从单一声明源锁定
├── requirements-dev.txt
├── .gitignore
├── .env.example
├── configs/
│   ├── data.example.yaml
│   ├── validation.yaml
│   ├── features.yaml
│   ├── baseline.yaml
│   └── acceptance.yaml
├── docs/
│   ├── task_contract.md
│   ├── data_contract.md
│   ├── rule_questions.md
│   ├── report.md
│   └── decisions/                   # 拷入本包 decisions/
├── src/bf_tap/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── io.py
│   ├── schema.py
│   ├── audit.py
│   ├── availability.py
│   ├── features/{build,sample,operation,burden,history}.py
│   ├── splits.py
│   ├── metrics.py
│   ├── models/{sanity,baseline}.py
│   ├── train.py
│   ├── predict.py
│   ├── submission.py
│   └── artifacts.py
├── tests/
│   ├── fixtures/synthetic/
│   ├── test_metrics.py
│   ├── test_schema.py
│   ├── test_availability.py
│   ├── test_history_visibility.py
│   ├── test_splits.py
│   ├── test_feature_contract.py
│   ├── test_model_roundtrip.py
│   ├── test_submission.py
│   └── test_end_to_end.py
├── .github/workflows/tests.yml       # 只运行代码与独立合成 fixtures
└── local/                           # 不进入 Git
    ├── manifests/
    ├── cache/
    ├── runs/
    ├── submissions/
    └── reports/
```

工程采用 Python 3.11 + pandas/NumPy + CatBoost + PyYAML + pytest。需要读取字段字典时使用 openpyxl。基础 baseline 不要求 GPU。依赖精确版本在实际环境安装及 smoke test 后锁定，本包没有编造 requirements 锁定结果。

## 3. 数据、特征和执行契约

文件 manifest 至少含路径来源、SHA256、字节数、行列、类型、时间范围、空值、解析失败、主键重复及冲突。sample_id 以字符串保存前导零；test 模板不能替代各阶段样本主表。公共表重名文件必须明确版本和位置，不取搜索命中的第一个文件。

available_at 为统一的可用性入口；不机械地拿任意 timestamp 当作可用时刻。来源修订、小时统计左右边界和质量报告滞后须有官方依据。训练/验证/推理调用同一个 feature builder，X/y/审计元数据分离。

缓存键包含代码/特征版本、数据身份、配置、场景、模型拟合起点、历史授权集合和样本集合。不能让 DEV、HOLDOUT 和正式推理共用一个未分区的 features.parquet。每行每类特征保存窗口、来源、最大可用时间、匹配数和缺失/陈旧统计，审计列不自动进入 X。

## 4. 已冻结的授权、验证、模型与验收

# D0：AI 主导的技术决策权

状态：已由用户授权，2026-09-06 生效；范围仅限高炉铁次预测竞赛。

用户最新明确授权：“我让出这个项目的核心判断权，由ai负责”。

本项目不再要求用户先提交验证设计、模型路线、特征方案或验收判断。AI 技术负责人负责设计、任务拆分、问题取舍、代码审查与技术验收；编码 Agent 按本包实施。旧版要求用户先填写 D1/D2/D3 的条款在本项目中废止，不扩展至用户其他科研项目。

当前授权不改变阶段：先完成可信 baseline，不自动进入调参、特征搜索、模型融合、排行榜试探或资源扩容。

技术自主决策与事实来源分离。AI 可以决定保守禁用含义不明的可选字段，但不能发明数据字典、官方规则、运行结果或外部服务权限。缺少真实数据只阻塞数据相关任务，不再以“等用户技术判断”为阻塞项。

AI 负责验收判断，但必须引用实际测试及产物。编码、审阅应为分离的工作阶段；同一 AI 执行两阶段时注明 self-review，不能冒充独立复核。远程仓库写入、赛事提交、付费资源、公开发布仍须有对应操作授权与工具条件，本授权不代表这些动作已执行。

# D1：固定起点、多预测跨度的时间验证

状态：AI 已冻结设计；真实运行等待数据与字段字典。所有边界使用 Asia/Shanghai（UTC+8），左闭右开，禁止按用户所在时区解释生产时间。

## 开发期场景

| 场景 | 拟合起点 C | 训练参考时间范围 | 评估范围 | 月份距标签起点 |
|---|---|---|---|---|
| DEV_LONG | 2024-07-01 00:00 | [2024-03-01, 2024-07-01) | [2024-07-01, 2024-11-01) | 7/8/9/10 月分别为 H1/H2/H3/H4 |
| DEV_SHORT | 2024-09-01 00:00 | [2024-03-01, 2024-09-01) | [2024-09-01, 2024-11-01) | 9/10 月分别为 H1/H2 |

DEV_LONG 在整个评估区间只训练一次、冻结一次历史标签集合；不能每月加入新的真实标签。DEV_SHORT 独立模拟较近起点。按场景报告全区间指标和逐月指标。两个场景在 9/10 月重叠，禁止合并成去重概念不清的 OOF 总分。

## 受保护报告集：2024 年 11 月

| 场景 | 拟合起点 C | 训练参考时间范围 | 评估范围 | 用途 |
|---|---|---|---|---|
| HOLDOUT_H1 | 2024-11-01 | [2024-03-01, 2024-11-01) | [2024-11-01, 2024-12-01) | 最近起点，近似 test_a 跨度 |
| HOLDOUT_H2 | 2024-10-01 | [2024-03-01, 2024-10-01) | 同上 | 近似 test_b 跨度 |
| HOLDOUT_H3 | 2024-09-01 | [2024-03-01, 2024-09-01) | 同上 | 近似 test_c 第一个月跨度 |
| HOLDOUT_H4 | 2024-08-01 | [2024-03-01, 2024-08-01) | 同上 | 近似 test_c 第二个月跨度 |

这些是不同信息起点下预测同一个 11 月的场景，不是四个独立样本集，也不是正式测试月季节工况的完整复刻。报告主结果为 H1；H2–H4 展示标签历史陈旧时的退化。不能通过把四份结果拼接来扩大样本量或声称获得四份独立证据。

开发时，11 月标签及历史表中的同批结果由独立评分入口保管；编码/训练/特征流程不读取标签统计、样本级误差或据此选择字段、参数、轮数。冻结代码、D2、列清单及依赖锁后一次生成四场景预测，再统一解封评分。完整审计阶段可检查结构与记录数；11 月目标分布审计结果也应先封存。

第一次解封就登记 holdout_consumed=true。根据 11 月结果修改模型后，后续 11 月结果只能叫复用留出集结果，不能恢复为“未见测试”。发现正确性缺陷可以修复，但保留旧证据与消费记录。

## 拟合、特征、历史集合的不同边界

对每个拟合起点 C：

1. 拟合样本必须来自训练主表、reference_time < C，且两个监督目标的已确认可用时刻均 <= C。边界时尚未结束或尚未报告的铁次不得进入拟合。两个模型使用相同的有效拟合样本集合。
2. 冻结历史结果集合 H_C：来自授权历史实绩、reference_time < C、对应结果 available_at <= C。验证月结果及历史侧表中的副本均排除。额外历史键不在训练主表中时，必须由数据契约证明其授权与时间口径，不能仅因未匹配主表就当成可用。
3. 对每个训练或评估样本 i，历史特征还须满足记录 available_at <= reference_time_i，并排除当前 tap_key。H_C 只是上界，不允许训练早期样本看到 C 前的所有结果。
4. 已结束不等于已报告；逐字段 available_at 来自字典或官方答疑。未知规则不擅自替换为开铁时刻或固定一小时延迟。
5. 小时/事件过程表不是标签库。对后续预测仍允许使用在其 reference_time 前已可用的官方过程记录；不能把这些记录也冻结在 C。
6. 所有拟合式处理仅使用该场景拟合部分；本版无监督目标编码、跨折标准化、数据驱动异常删除或全数据特征选择。

## 正式训练和 A/B/C 推理

首次正式 bundle 的 fit_asof 定为 test_a 主表中最早的 reference_time。只使用官方 train_samples 中、参考时间位于官方训练范围且两个标签在 fit_asof 前已可用的记录。训练范围末端跨过拟合时刻的记录剔除并登记，不能笼统声称使用了所有 11 月标签。

H_release 采用同一起点冻结的授权训练期历史结果集合。A/B/C 复用这一个固定参数训练 bundle；允许各预测时刻的官方过程源更新，不添加 A/B/C 真值，不递归回填预测作为真值，不因进入 B/C 就重新获得测试标签。B/C 不提供全阶段数据时，相关推理测试标为未测，不用空文件冒充完成。

规则日后明确新增标签或更新任务时，由 AI 新建决策版本及 run，不能覆盖本版。

# D2：唯一正式 baseline 与固定特征族

状态：AI 已冻结 baseline-v0.1 设计；不是模型效果结论。参数是事先指定的起点，不是调优结果，也不声称最优。

## 模型与目标

正式模型为两个相互独立的 CatBoostRegressor：分别预测 tap_iron 和 tap_time_len。均在原始单位上训练，loss_function=MAE，样本等权，无目标变换，无相互目标输入，无多任务网络、融合、伪标签或残差校准。

评分的每个目标分母在固定评估集合上是常数，因此降低绝对误差总和与降低该目标 WMAPE 的方向一致。两个独立模型可分别优化 MAE；这不保证训练误差下降会转化为时间外验证收益。不要用 CatBoost 的 MAPE 目标代替官方 WMAPE，也不要按 1/y 给每条样本加权。

固定参数见 configs/baseline.yaml：CPU、800 棵树、depth=5、learning_rate=0.03、l2_leaf_reg=5、seed=2026、8 线程、bootstrap_type=No、random_strength=0、rsm=1、boosting_type=Plain、has_time=true、one_hot_max_size=64、nan_mode=Min。

每次 fit 前按 reference_time、稳定 tap_key/sample_id 排序。has_time 不负责过滤未来数据，也不能代替 D1。唯一类别输入为 spout_no，以字符串表示；缺失用保留且无碰撞的 __MISSING__ token。训练类别基数必须 <=64，使用 CatBoost 内部 one-hot，避免引入类别目标统计。发现实际基数违反契约就报告结构问题，不悄悄改为目标编码。

固定训练 800 轮，use_best_model=false，不传 eval_set，不启用 early stopping，不从验证结果选轮数。两个目标及所有场景参数一致。仅允许为验证 API/工程兼容性做合成 smoke test；本实施包尚未执行这些训练测试。

## 两个固定对照

B0：按各场景有效拟合标签计算两个全局中位数。

B1：按铁口分别计算两个中位数；对应铁口训练样本少于 20 或未见类别时，退回该场景全局中位数。阈值 20 是固定规则，不搜索。对照不调用过程数据，不使用验证标签，训练与本场景主模型使用相同有效行。

它们是诊断对照，不是第二、第三条参赛候选模型。不能在 11 月看完结果后改成挑选其中更好者提交并仍声称原方案未选择。

## 特征族：先固定语义规则，再映射真实列

本段是 AI 工程设计，不是官方声明。实际列名、单位、空值和 available_at 仍须逐项引用数据字典；configs/data.example.yaml 里的 null 表示缺资料，不表示缺用户授权。

| 特征族 | 本版固定内容 |
|---|---|
| sample | spout_no；开铁时刻的小时 sin/cos（包含分钟秒）；weekday（0–6）；不纳入 sample_id、数值化 tap_no、原始时间戳或全局行号 |
| operation | 对赛题说明中的炉况语义白名单：最近可用值、事件年龄；6h/24h 窗口的 mean、std(ddof=0)、valid_count；每个最近值的缺失标记 |
| burden | 对赛题说明中的铁量、综合品位、矿耗、燃料比、焦比：最近可用值、事件年龄、缺失标记；不做窗口搜索或业务公式组合 |
| history | 全炉与同铁口各自：最近可用铁量/时长、最近 3 和 10 个可用铁次的均值、窗口实际计数、距最近结果可用的分钟数；不能使用当前铁次实绩 |

operation 的窗口定义为 t-W < event_time <= t，同时要求 available_at <= t；不是按 CSV 物理行号 rolling。最近值按已可用记录中 event_time 最新者选取；同键同刻冲突须依据官方修订语义解决，不能凭行顺序选最后一条。最近事件超过 24h 则最近值置缺失，但保留年龄/过期标记。24h 是保守工程陈旧阈值，不是已证实的高炉物理时间常数。

burden 同理，最近值陈旧阈值为 72h；不是把化验最终值倒填到事件发生时刻。72h 同样只是冻结的 baseline 规则。字段名“铁量”需字典确认其含义，不能误接为当前目标铁次的最终标签。

history 只按 D1 的授权、可用集合构造。没有足够 K 个记录时使用已有记录均值，并记录实际 n；没有记录则均值缺失、计数 0。不同目标的可用时刻不能混用；本版保守要求一条历史记录两个结果均已可用后再进入历史窗口。长预测跨度内历史值可能保持不变、年龄持续增加，这是必须报告的分布偏移风险，不人为补真实标签解决。

数值保留 NaN 交给 CatBoost；不 bfill、不居中插值、不用全表均值填补。模型输入为显式最终白名单与稳定列序。字典未证实的可选字段禁用并记录；如目标可用时刻或关键主键无法确认，阻塞正式验证。任何降低特征覆盖的结果需在报告中披露。

## 输出与后处理

先检查原始预测有限，不能用裁剪掩盖 NaN/Inf。数据契约要求真实目标为非负；遇到负目标须阻塞核对而不是静默删除。在该契约成立时，对全部样本统一应用 max(pred, 0)。对 y>=0，该投影不增加绝对误差；这不是逐行人工修正。保存裁剪前后结果与数量，不设置上界，不用分位数剪裁测试预测。

官方指标用全精度后处理预测计算，同时报告写出 CSV 回读后的分数。result.csv 固定保留 6 位小数。两个目标不对数化、不缩放、不做测试结果人工修改。

# D3：证据驱动的 AI 验收

状态：AI 已冻结项目内部验收规则；以下不是官方得分阈值或成绩保证。

## G0：工程正确性与可复现性

必须同时满足：

- 完成所有必需字段/主键/时间可用性契约；未知可选字段被禁用且显式披露，无未处理的数据冲突。
- 手算评分器、未来扰动、历史副本隔离、跨起点隔离、当前铁次排除、边界时刻、缓存隔离、训练/推理一致性及提交反例测试全部真实通过。未运行、依赖错误、数据错误、算法失败分开记录。
- 每个实际提供阶段的 sample_id 一对一完整覆盖，严格三列，预测有限且非负；前导零保真；result.csv 为 UTF-8；结果 ZIP 根目录只有该文件，回读检查通过。
- 同一 bundle 在两个独立推理进程中，原始预测 max_abs_diff <= 1e-9，最终 CSV 字节相同。
- 同一锁定依赖、CPU/线程/数据/配置下，两次干净重训，原始预测 max_abs_diff <= 1e-6。超过即不通过并解释；不能改阈值伪装通过。不预先保证跨硬件逐字节一致；换环境单独认证。
- 真实本地离线流程完成，保存代码、配置、数据哈希、环境、所有模型及预测。依赖锁和离线 wheelhouse 根据实际环境生成；仅写 requirements 而未试装不算通过。
- 每个已提供阶段，从新进程启动、读数据/模型、构建特征到写出 CSV 的墙钟时间 / N <=1s，并且总时间 <=1800s；记录测量硬件、N、峰值 RSS、模型体积。此处用作暂定内部运行门槛；官方最终细则变化时另立版本。
- 所有规定验证场景均有完整原始指标和分子/分母；缺场景不以平均分替代。不上传真实数据或产物至未经授权的远端。

G0 通过可标记 BASELINE_REPRODUCIBLE，只证明参考实现可复现，不证明效果强。

## G1：预测质量门槛

定义 E = 0.5*WMAPE_iron + 0.5*WMAPE_time；它是不截断的综合相对误差，越低越好。

对 DEV_LONG、DEV_SHORT、HOLDOUT_H1/H2/H3/H4 六个预注册场景，分别要求：

1. E_catboost <= min(E_B0, E_B1) - 1e-6，即超过两个固定对照中较好者；1e-6 是比较裕量而非统计显著性门槛。
2. 对每个目标，WMAPE_catboost <= WMAPE_B1 + 0.01。0.01 是绝对 WMAPE 的 1 个百分点，不是相对增长 1%。它用于避免综合分掩盖某个目标退化，是事前工程政策，不是推断出来的工业容忍限。
3. 逐月、逐铁口、缺失/陈旧分组的误差与样本数必须披露；分组不足不做显著性或普遍性声明。Top-20 绝对误差和训练集确定阈值的极端标签组仅用于报告，不据此手工改预测。

DEV 场景按各自全区间汇总分子/分母；逐月结果另列。四个 HOLDOUT 使用相同 11 月样本，不合并当作独立证据。满足 G1 也不能推出正式测试排名或已优于官方 baseline。

## 通过、失败和停止

- G0 未通过：BLOCKED_CORRECTNESS，不发布可复现标签，修复缺陷并保留失败证据。
- G0 通过，G1 未测：BASELINE_REPRODUCIBLE_QUALITY_UNMEASURED。
- G0 通过，G1 不通过：BASELINE_REPRODUCIBLE_QUALITY_FAILED。保存这个真实基准，但不称强 baseline，不自动搜索参数或更换路线来刷过门槛。
- G0、G1 均通过：BASELINE_OFFLINE_ACCEPTED，可生成待提交的离线候选；不代表已经提交排行榜。
- 缺少官方参考分数：official_baseline_status=UNKNOWN。禁止用本地 60 分或两个中位数对照替代官方有效成绩判定。

技术审阅与验收由 AI 负责，不再等待用户先判断。审阅报告必须指向真实代码版本、执行命令、测试日志和产物；仅审阅设计时写 DESIGN_REVIEW_ONLY。下一阶段的分数优化需要单独进入优化阶段，当前任务到 baseline 状态报告结束。

## 5. 评分器与测试矩阵

官方评分（S1）：

```text
WMAPE_iron = sum(abs(pred_iron - iron)) / sum(iron)
WMAPE_time = sum(abs(pred_time - time)) / sum(time)
Score_main = max(0, 100 * (1 - 0.5*WMAPE_iron - 0.5*WMAPE_time))
```

先按 sample_id 一对一对齐；非有限值、缺/多/重复 ID、分母 <=0 显式失败，禁止加 epsilon、跳过样本或改成平均逐样本百分比误差。记录每目标绝对误差和、真实值和、MAE、WMAPE、样本数以及未截断 E。

手算测试：iron 真值 [100,200]、预测 [110,180]；time 真值 [50,100]、预测 [55,90]。两项 WMAPE 均为 0.1，Score=90。另测完美 100、极差截断 0、打乱 ID 顺序不改变结果。

必须真实执行的反例：

| 测试 | 正确行为 |
|---|---|
| 未来值扰动/删除 | 固定模型与当前样本，仅改变未来源的值或移除其记录，不改变当前特征及预测 |
| 未结束历史 | 09:00开铁、10:30结束的结果不能影响09:40预测 |
| history副本污染 | 固定授权集合，将验证标签及历史副本改值，不改变拟合/特征/预测，只影响评分 |
| 跨场景污染 | 起点之后新增标签即使在评估时已结束，也不能进入冻结历史集合 |
| fit边界 | ref<C但label_available>C的训练行必须被排除并报告 |
| 逐样本边界 | 训练早期样本不能使用虽然在C前可用、却在该样本之后才可用的记录 |
| 过程源动态 | C之后但当前t之前已可用的过程记录可以正确参与特征 |
| 时间窗口/陈旧 | 测同刻、跨日月、空窗、部分K、24h/72h边界、事件与可用时刻不同 |
| 序列不变性 | 打乱输入物理行顺序后，稳定排序结果与预测一致 |
| 缓存 | 改数据、配置、起点或授权集合不得命中不相容旧缓存 |
| 模型/提交 | 单样本与批量一致；独立进程load一致；错列错ID错阶段ZIP多文件均失败 |
| 留出集账本 | 未冻结不得解封，解封后永久记录已消费，不冒充新的盲测 |

未来扰动测试不要求更改数据并重新训练后的模型仍保持相同预测。污染测试不得通过更改可用性字段导致换了测试条件；输入身份和可用性条件另设独立测试。

## 6. CLI 契约（待编码，不是本包已有程序）

```bash
python -m bf_tap audit --data-config configs/data.local.yaml
python -m bf_tap validate --config configs/baseline.yaml --split-config configs/validation.yaml --suite development
python -m bf_tap freeze --run-id <id> --output local/manifests/frozen_release.json
python -m bf_tap validate --config configs/baseline.yaml --split-config configs/validation.yaml --suite holdout --frozen-manifest local/manifests/frozen_release.json
python -m bf_tap train --config configs/baseline.yaml --mode official-release --output local/runs/<id>
python -m bf_tap predict --bundle local/runs/<id>/bundle --stage test_a --output local/submissions/<id>/result.csv
python -m bf_tap check-submission --stage test_a --path local/submissions/<id>/result.csv
python -m bf_tap pack --stage test_a --team-name <team> --result local/submissions/<id>/result.csv
```

predict 不调用 fit，不访问评分标签；可以读取明确授权的训练期历史快照。holdout 子命令内部先在标签隔离条件生成全部预测，再由评分入口解封评分。配置中“frozen_by_ai”说明设计决策完成，不等于数据已确认；未知数据映射仍失败关闭。

## 7. 产物、报告与离线复现

每次 run 使用独立目录，保存 resolved_config、decision_snapshot、data/environment/split/feature_manifest、metrics、按场景 predictions、raw/postprocessed predictions、model bundle、日志和审阅报告。训练失败也写 status，禁止用下一次运行覆盖。

bundle 至少含两个 .cbm 模型、特征名/顺序/类型、类别规则、as-of规则与所需历史快照身份、固定后处理和训练数据身份。历史快照可以随本地 bundle保存或从授权路径按 manifest 重建；二者选择写清，保证推理无需重训练或临时找数据。

JSON/YAML 元数据保存 Python/依赖/OS/硬件、8线程与seed、Git commit和dirty状态、首读holdout状态。数据哈希用于身份而非备份；保留授权原始版本。报告实测模型大小、峰值RSS和计时范围，准备实际验证过的离线依赖包。

result.csv 按阶段主表原顺序输出，UTF-8、无索引、6位小数。包装为 <teamname>_bf_tap_predict_prelim.zip / round2.zip / semifinal.zip 对应的完整官方命名，即第二、第三项分别为 <teamname>_bf_tap_predict_round2.zip、<teamname>_bf_tap_predict_semifinal.zip。ZIP仅含根目录result.csv。日志、报告、manifest与模型属于工程包，不进入结果ZIP。

报告从现在维护数据/方法/验证/误差/复现/限制，不编造收益或因果机理。后续总决赛还评价工程、技术材料和答辩（S1），但当前不制作空白“高分结论”。

## 8. 实施顺序

# 实施任务与依赖

所有状态初始为 NOT_STARTED。本包只有设计与配置，没有训练实现或真实测试成绩。

| 工作项 | 具体实现 | 可执行前提 | 验收证据 |
|---|---|---|---|
| M0 | src 布局、pyproject、CLI help、配置解析、日志、.gitignore、测试 CI | 当前即可 | 干净安装；合成测试；无真实数据被 Git 跟踪 |
| M1 | schema、只读 IO、manifest、审计、字典映射契约 | 通用实现当前即可；真实映射等数据 | SHA256/行列/主键/时区/冲突/字段依据；11 月目标审计封存 |
| M2 | WMAPE/Score、B0/B1、提交校验、ZIP 打包 | 合成实现当前即可 | 手算90分、0/100边界；错列/错ID/错阶段/ZIP正反例 |
| M3 | D1 split、授权历史、available_at、D2 feature builder、缓存 | 通用实现当前即可；真实运行等字典 | 未来扰动、history副本、边界、时间窗、stale、列序、缓存测试 |
| M4 | 固定 CatBoost wrapper、两目标训练/保存/加载、两个DEV场景 | 代码可先写；真实执行要求契约完成 | 固定参数、自检、DEV全部模型/指标/按月误差、B0/B1、无holdout访问 |
| M5 | 冻结候选、四个HOLDOUT预测、统一解封评分、双重训及断网复现 | G0核心正确性通过、代码/特征/依赖冻结 | frozen_manifest；解封账本；全部指标；重训/加载容差；离线运行 |
| M6 | release fit、A/B/C推理（按实际数据）、结果回读、报告、AI技术验收 | 真实契约；G0/G1如实判定 | 原始/最终预测；ZIP；G0/G1状态；完整报告；无自动赛事提交 |

推荐按 M0→M1/M2→M3→M4→M5→M6 提交小任务。M1 的数据缺失不影响 M2、M3 的合成实现及 M4 wrapper 编写。没有真实数据，结束状态必须是“脚手架/合成自检完成，真实基线未训练”。

代码变更和实验变更分开记录。发现正确性 bug 可以修复；如果已经读过 11 月目标，修复后的结果必须标明留出集已消费。分数差但代码正确不构成擅自调优或改门槛的理由。

## 首轮给编码 Agent 的指令

按 AGENTS.md 和冻结的 D0–D3 实施 M0–M3，再实现 M4 的模型接口与合成 smoke test；不要等待用户填写技术决策，不要进行优化。实际数据缺失时写明 BLOCKED_DATA，但完成所有不依赖真实数据的代码与测试。逐任务交付真实执行证据，并由 AI 技术审阅阶段给出结论。暂不创建远程仓库、不推送真实资产、不向赛事平台提交。

## 9. 规则更新与本轮真实性

# 规则问题与来源更正

核对日期：2026-09-06。附件和官网是事实来源；本包的模型、划分和门槛为 AI 设计。

## R1：最后一次 / 最优成绩

附件《高炉铁次预测-1.pdf》第 7 页写“最后一次提交”，第 8 页写复赛/半决赛“最优成绩”。本次重新完整读取官网 HTML，发现官网同样并存两种表述：初赛/复赛/半决赛格式段写最后一次，随后复赛/半决赛段写最优成绩。

更正上一版“官网只写最后一次提交”的不完整描述。没有足够依据确定最终有效版本；问题保持 OPEN_OFFICIAL_CLARIFICATION。禁止据此自动试探排行榜或覆盖最后提交。官网链接为 https://www.aicomp.cn/tracks/tracks-6/4177.html 。

## R2：AI 辅助工具

资料禁止商业闭源在线推理接口/联网模型服务、外部训练数据、外部数据预训练权重及赛事数据泄露。当前文本不足以完整判定所有 AI 辅助编码与第三方数据处理方式。用户的项目授权不是主办方的数据授权。当前按不上传真实生产数据、算法完全离线、Agent 使用代码与独立合成测试数据的边界实施。

## R3：尚缺资料

实际 CSV、data_dictionary.xlsx、官方 baseline 代码/参考分数及后续正式评测细则未提供。不得填写想象的分数、列名、类别数或报告滞后。官网 PDF 读取尝试超时；本次官网核对依据 HTML，附件事实依据已经提供的本地 PDF/README，未声称成功读取官网 PDF。

## 10. 来源

S1：用户上传《高炉铁次预测-1.pdf》，9页。SHA256 `18ee1e4c0217e8d6dab1225045418f8b79a93bd90cd4e958b6da23733dc57146`。

S2：用户上传 README.txt。SHA256 `5baf89b7c07a03b3a96aad964ecf219681e892c6bed8c614aaebb8c51b47f9de`。

S3：赛事官网 HTML，2026-09-06重新读取。用于核对规则，不代表已获取官方群全部新通知。https://www.aicomp.cn/tracks/tracks-6/4177.html

S4：CatBoost 官方回归损失文档，支持 MAE/MAPE 定义与可优化能力。https://catboost.ai/docs/en/concepts/loss-functions-regression

S5：CatBoost 官方通用参数文档，支持本版参数含义、has_time、one_hot_max_size 等行为。https://catboost.ai/docs/en/references/training-parameters/common

S6：CatBoost 官方缺失值文档，支持数值 NaN/Min 处理；类别缺失 token 为本项目工程规则。https://catboost.ai/docs/en/concepts/algorithm-missing-values-processing

S7：CatBoost 官方 fit 文档，说明 eval_set 参与 early stopping/best iteration 选择。https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_fit

软件文档在2026-09-06查阅；查询不等于已经安装/测试该软件，也不据此宣称所用依赖为“最新版本”。
