# Findings

## Baseline evidence

- DEV_SHORT：CatBoost E 0.171456，B0 0.178157，B1 0.178897。
- DEV_LONG：CatBoost E 0.185646，B0 0.176148，B1 0.176101。
- 现象指向时间漂移或特征稳定性问题，而非单纯欠拟合。
- B1 在长窗口更稳，CatBoost 在短窗口更强，值得优先测试 OOF 融合与 B1 anchor 残差模型。

## Repository constraints

- baseline-v0.1-reproducible 是不可变工程基线。
- 2024 年 11 月标签受保护，开发期禁止读取。
- train、validation、prediction 必须共享 as-of feature builder。
- G0 工程状态与 G1 质量状态必须分开。
- run 和失败证据不可覆盖，私有产物不得进入 Git。

## Initial design choice

首轮保持现有特征不变，按 M1 目标级融合、M2 近期样本加权、M3 B1 anchor 残差模型的顺序验证，避免模型、特征和窗口同时变化而无法归因。

## Code organization

- 用户要求用分级小文件夹归类不同预测模型。
- baseline.py 不移动；optimization_v02 下按 M1/M2/M3/M4 分目录，配置和测试镜像分类。

## M1 implementation evidence

- M1 已实现目标级权重、B1 优先 tie-break、键对齐、pooled/逐折 OOF 和安全标签 runner。
- 合成训练 CSV 包含不可解析的 11 月 sentinel，M1 runner 仍通过，证明保护目标未被解析。
- 当前全量测试 70 passed in 9.33s；私有产物守卫 PASS（120 files）。
- iron 目录下未找到任何 local YAML、CSV 或 XLSX，真实 rolling OOF 尚无法运行。

## Locked local baseline

- 系统 /usr/bin/python3.12 缺少 Python.h，无法构建锁定 psutil 5.9.0。
- uv 托管 CPython 3.12.13 包含开发头文件；pytest 需要 pyproject 的 dev extra。
- 锁定环境基线为 50 passed；Phase 1 后为 61 passed。
- Phase 1 新增严格候选 registry、配置合同和不可覆盖 run 初始化器，未修改 baseline.py。

## Existing interfaces

- pyproject 锁定 Python 3.11–3.12、CatBoost 1.2.8、pandas 2.3.3、NumPy 2.2.6 和 pytest 9.0.2。
- DualTargetBaseline 对 frozen 参数做严格相等校验，不能承载 optimization 参数。
- run_development_validation 当前只选择 development folds，并逐折训练、保存 bundle、B0/B1、指标及误差切片。
- MedianControls、select_split、score_predictions 与开发标签保护接口可复用。
- optimization 需要独立配置 validator、候选 registry 和跨折 OOF 聚合；不修改 frozen baseline validator。
- M1 可先实现为纯预测帧与权重选择模块，不依赖 CatBoost 训练，适合首个 TDD 切片。

## Remote branch discovery

- origin/main 与当前 HEAD 均为 7aa982f，没有新增内容。
- 数据集发布在 origin/optimization-v0.2 的 c7b1fb8，而非 main。
- 远端分支还包含另一套 optimization_v0_2 实现；需要与本地未提交 optimization_v02 工作共同保留。
- 28 个本地未跟踪文件与远端树的精确路径交集为空。
- git diff --ignore-space-at-eol --exit-code 返回 0，tracked 修改仅为行尾表现；git diff --summary 为空。
- 当前 HEAD 是 origin/optimization-v0.2 的祖先，可执行快进合并；本地未跟踪内容聚合 SHA-256 为 1b47672da6560786dad1b4617062626d90e30cc0a83ec74c7137b647597f40eb。
- 本地分支已改名为 codex/optimization-v0.2-local，并快进到 c7b1fb8。
- 初赛数据集目录包含 17 个文件；未读取受保护目标内容。
- stash 与恢复后的 28 条路径集合哈希均为 0352137a...，blob 集合哈希均为 ac25fb71...，恢复内容逐字节一致。
- 合并后锁定 Python 3.12 全量测试为 86 passed；私有产物守卫 PASS（161 tracked files）。

## Scheme comparison

- 本地 optimization_v02 设计为 M1 目标级 CatBoost-B1 融合、M2 近期样本加权、M3 B1 anchor 残差、M4 条件性集成。
- 远端 optimization_v0_2 设计使用 E00-E09 候选体系，重点变化包括数据源/历史组件选择、多年龄历史视图、过程变化特征和向 B1 收缩。
- 本地仅 M1 选择器与 OOF 合同已实现；M2/M3/M4 目前主要是注册表、配置与包骨架。
- 初步结论：两套方案在候选定义、算法和实现成熟度上均不一致，暂不满足“完全一致才归并”的条件。
- 本地 M1 从 [0, .25, .5, .75, 1] 为两个目标分别选择 CatBoost 权重，近似并列时偏向 B1；远端 E06 对 E00 与 B1 两目标固定使用 0.75/0.25。
- 本地 M2 是 30/60/120 天半衰期 recency weighting；远端没有该候选，远端 sample_weight 用于等权合成多年龄历史视图。
- 本地 M3 是 leakage-safe B1 anchor 残差学习；远端没有 B1 残差训练候选。
- 远端独有 E01-E05 数据源消融、E07/E08 多年龄历史视图、E09-E11 过程变化特征；本地没有这些算法族。
- 本地选择指标为 3 个单月折的 pooled WMAPE；远端为 5 origin × 1-4 horizon 的 equal-horizon J，并附带 DEV_LONG/SHORT 多门禁。
- 两套方案的共同点是复用 frozen baseline CatBoost、同一 as-of 特征构建器、B0/B1 控制和 2024-11-01 保护边界。
- 本地 M2/M3/M4 包各只有 1 行说明，只有 M1 具备 138 行融合算法和安全标签选择 runner。
- 远端 optimization 包约 2,119 行，覆盖训练、预测、特征消融、历史适配、过程变化、验收和发布。
- 远端结果记录 E09 已通过 G0/G1；本地尚未接入真实 rolling OOF，成熟度与已冻结决策也不相同。
- 最终结论：两套方案不是等价重复实现，不满足用户设定的自动归并条件；任何统一都应作为显式设计整合，而不是目录合并。

## Model-result analysis scope (2026-09-08)

- 本轮只读分析正式运行 `m234-real-oof-20260908-01` 的 OOF、逐折/逐目标/出铁口误差、确认窗口与 as-of 审计。
- 目标是识别时间漂移、目标差异、分组弱点和候选互补性，并提出下一轮预注册实验；不修改当前冻结模型，不访问 2024 年 11 月保护标签。
- 目标分布发生明显位移：tap_iron 月均值 6 月 546.0、7 月 536.8、8 月 487.8，标准差 88.0→106.9→110.6；tap_time_len 月均值 129.1→126.2→122.8，标准差 21.3→23.0→25.7。
- 8 月所有模型均系统性高估：铁量平均偏差 B0/B1/CatBoost/M1/M2/M3 分别约 +53.7/+53.5/+47.6/+50.6/+37.0/+61.2；时长约 +7.2/+7.2/+12.5/+7.2/+5.1/+11.1。
- 跨折 pooled 的第 4 周 E 明显高于第 1–2 周：CatBoost 0.1796 对 0.1424/0.1432；M1 0.1729 对 0.1411/0.1422；B1 0.1739 对 0.1424/0.1434。需进一步拆分月内预测跨度与月份分布位移。
- 出铁口 1 整体较难：M1 pooled E 0.15955，出铁口 2 为 0.14921；M1 对 CatBoost 的改善主要在出铁口 2。
- 各模型最大 10% 样本贡献约 27%–28% 总绝对误差，未显示少数极端样本主导；优先处理系统漂移和分组差异。
- M1 最优权重跨折变化大：tap_iron 的 CatBoost 权重为 6 月 0.75、7 月 0.25、8 月 0.25；tap_time_len 为 1.0、0.25、0.0。当前 pooled 冻结权重 0.5/0.0 不代表逐折稳定最优。
- 留一折权重验证中，基于另外两折选择的 M1 在被留折上均未同时优于 CatBoost 与 B1：6 月 E 0.12653、7 月 0.14940、8 月 0.19339。
- M2 半衰期同样不稳定：逐折 E 最优依次 H60、H30、H120；pooled 铁量最优 H120、时长最优 H30，表明共享半衰期掩盖目标差异。
- 所有主要候选对的 pooled 残差相关均高于 0.95；M1-M2 为铁量 0.9699/时长 0.9607，M1-M3 为 0.9843/0.9576。当前同构候选缺乏 M4 所需互补性。
- 行级绝对误差胜率大多仅接近 50%，改进不是广泛支配；M1 对 CatBoost 胜率为铁量 51.0%、时长 53.9%，时长对 B1 完全相同。
- 月内拆分显示 8 月实际均值继续下降：第 1 至第 4 周铁量均值 513.7→468.2、时长 129.0→117.7，解释了所有静态位置模型持续高估。
- 探索性折前历史控制中，30 天 B0 的 pooled iron/time/E 为 0.160449/0.147273/0.153861，优于当前 M1 E 0.154343；窗口为事后比较结果，不能替代预注册重跑，但应列为下一轮首要候选。
- 30 天 B0 逐折 E 为 0.126077/0.147922/0.188551，在 7 月和 8 月同时优于 CatBoost 与 B1；优势主要来自时长跟随近期水平。
- 探索性的 M1 铁量 + 30 天 B0 时长组合 pooled iron/time/E 为 0.159464/0.147273/0.153369；逐折 E 为 0.125337/0.148426/0.187231，仍在 6 月落后 CatBoost，但 7/8 月优于 CatBoost 与 B1。
- operation 数据中位延迟约 0.47–0.51 小时，burden 约 1.30–1.43 小时；其延迟与 CatBoost 绝对误差的 Spearman 相关接近 0。history 固定-origin 年龄中位约 15 天，但除 7 月轻微相关外，月内误差相关很弱。直接的数据源延迟不是首要解释。
- 按日配对块 bootstrap（诊断性，未纳入冻结门槛）：M1 相对 B0 的 E 差 -0.000387，95% 区间 [-0.001285, 0.000482]，相对 B1 区间也跨 0；相对 CatBoost 为 -0.003469，区间 [-0.005416,-0.001352]。
- 探索性 M1铁量+近期30天时长 相对 B0 的 E 差 -0.001361，95% 区间 [-0.002469,-0.000211]，相对 B1/CatBoost 区间也低于 0；该估计没有校正事后窗口/组合选择偏差，只能作为新折验证的优先假设。
- 远端 E09 在不同的 5 origin×4 horizon 合同下，将 E02 路线加入有符号过程变化特征，J 相对 E00 改善 0.009222，并通过其六项门槛；它只能作为可迁移特征假设，不能与本地三折 OOF 直接比分数。
- 远端结果也表明原始历史年龄特征不稳、历史目标汇总仍有价值；这与本地静态水平漂移、过程源延迟不是主因的诊断方向一致。
- 分位诊断：实际铁量最低四分位的 WMAPE 为 B1 0.4018、CatBoost 0.3962、M2 0.3725；时长最低四分位为 B1 0.3597、CatBoost 0.3776、M2 0.3318。模型普遍向中位数收缩，低产量/短时长状态是主要难点。
- M2 在低值区和 18:00–23:59 时段相对更强，但在中高值区退化，说明近期加权捕捉到部分低位 regime，同时破坏了常态区映射。
- 逐折出铁口诊断显示 8 月 spout 2 的 CatBoost E 0.21249、B1 0.19205、M1 0.19376，是 CatBoost 长期失稳的重要切片；8 月 spout 1 的 M1 反而优于 B1/CatBoost。
- CatBoost 特征重要性约 68% 来自 operation 当前/窗口，约 18% 来自历史目标，历史 count 近乎 0；baseline 跨折特征重要性秩相关约 0.93，M2 降到铁量 0.85、时长 0.82，说明近期加权放大了关联不稳定性。
- operation 变化特征可作为识别低产量、晚班和 spout×月份 regime 的候选信号；不应继续扩大高相关同构模型的集成。
- 折前 30 天相对全历史的中位数位移：铁量在 6/7/8 月为 +6.9/+13.1/+2.6，未预示 8 月实际大幅下降；铁量位置校准的权重跨折不稳定，不应作为首轮主方案。
- 时长位移为 -1/-1/-3，30 天时长调整在所有留一折训练中均选择满权重；目标特异的近期时长基线比共享 M2 衰减更稳定。

## Recommended next optimization phase

- 新阶段应显式命名为 optimization-v0.3-drift，保留 baseline-v0.1 和当前 v0.2 运行证据不变。
- 第一优先候选 C1：铁量沿用当前冻结 M1，时长固定使用折前 30 天全局中位数；30 天窗口预注册，不再搜索 14/60/90。
- 第一优先候选 C2：在冻结 CatBoost 参数下隔离迁移远端 E09 的 E02 路线（去历史 age、保留历史 target/count）与 signed operation level deltas；不同时改变样本权重或模型参数。
- C1/C2 必须在同一 5-origin×H1-H4 时序网格上与 B0/B1/Frozen CatBoost/当前 M1 比较；Sep/Oct 已被开发分析看过，只能算开发证据，不能称独立验证。
- 仅当 C1 和 C2 各自稳定通过后，第二阶段才预注册目标级组合：铁量取 C2，时长取 30 天基线。不得在同一结果上自由搜索组合。
- M2 仅作为低优先级后备：若继续，采用目标独立、带等权下限的软衰减，避免当前 H30 有效样本量骤降与特征重要性不稳；M3 暂停，M4 保持关闭，直到至少两个稳定成员且残差相关低于 0.95。
- 每个候选强制报告 origin、horizon、目标、spout、小时段、实际值四分位、signed bias 和块 bootstrap 区间；验收继续分开报告 G0/G1。
- 2024 年 11 月保护 holdout 只能在方案、候选和阈值冻结后另行授权。

## optimization-v0.3-drift implementation mapping

- 远端 `src/bf_tap/optimization/run.py` 已按 fit_cutoff 对 origin 分组，一次训练后切 H1-H4，并通过 `select_partitions` 同时约束 reference_time 与 label_available_at；可作为统一五起点验证骨架。
- 远端 E09 已完整实现 C2 所需路线：冻结 CatBoost 参数、E02 历史 count/target（去 age）、16 个 operation 列的 3 类有符号差分，共 48 个新增特征。
- 现有 experiment loader 把 optimization_id 写死为 v0.2，并只支持 model/shrink_b1；v0.3 不能直接改写该合同，否则会污染已冻结的 v0.2。
- 现有 run.py 的 acceptance 以 E00 为唯一 learned baseline，并默认所有 candidate 都走 CatBoost/derived shrink；C1 是目标级 control/model 混合，需要独立的 v0.3 candidate 合同或窄适配层。
- C1 的严格因果定义应使用 `reference_time >= fit_cutoff-30d`、`reference_time < fit_cutoff` 且 `label_available_at <= fit_cutoff` 的训练标签；为空时必须失败，不得回退全历史。
- 当前远端诊断只含 month/spout/missing/stale/history-age；v0.3 还需新增 hour bucket、actual quartile、signed bias，并保留每个 unit 的行级误差供日块 bootstrap。
- 远端同一 origin 的 E00/E09 训练可复用现有 feature build；C1 的 M1 铁量应从该 origin 的 E00 和 B1 按已冻结权重 0.5/0.5 计算，时长用 30 天全局中位数。
- v0.3 配置采用完整 canonical equality；任何窗口、权重、候选 ID、Stage 2 前置项、bootstrap 参数或 evidence scope 改动都会失败，不能静默变成搜索。
- C2 逐行别名测试还需要保持 sample_id dtype；候选对齐层现已在字符串键校验后恢复 evaluation ID dtype。
- scenario-pooled 通过 `unit_id::sample_id` 构造评估键，避免相同样本跨 origin/horizon 的重复 ID 被错误当成普通 OOF。
- 日块 bootstrap 先聚合 unit×local-date 的实际值和两模型绝对误差，再成对重采样；这保持 WMAPE 分母与候选/控制配对。
- v0.3 顶层 G1 明确只看 C1/C2 Stage 1 行；即使 M1_FROZEN 或核心 E09 的旧名称通过，也不会绕过 canonical candidate gate。
- C3 预测只有双门通过后才物化；SKIPPED 路径不创建任何 C3 文件，从文件系统层面保留预注册语义。
- pytest 默认 import 模式下，非包测试目录中的同名文件共享顶层模块名；`tests/optimization_v03/__init__.py` 使 v03 使用独立模块命名并消除与 v02 的碰撞。
- optimize-v0.3 CLI 不暴露 suite 或 candidate 参数，用户无法在权威入口中把预注册 all-suite 候选集改成搜索。

## Full-stage E09 release evidence (2026-09-09)

- E09_PROCESS_CHANGE_E02 development training completed with `fit_cutoff=2024-11-01T00:00:00+08:00`; training manifest records `protected_labels_read=false`.
- Prediction manifests are PASS with 335/322/548 rows for test_a/test_b/test_c and `inputs_stable=true`; no predictions were clipped.
- Result hashes: test_a `bf037c118a3ee55cdc95f5964868e312453d87139c4a0de2f8eb55f53cf46ac4`, test_b `eda894da21f868677c84c02c8a7a4cd4a3a1bf71bb7ad924fe697fd9ee4cf1b7`, test_c `6f35df2e6202119326dcbb36266196d74127f14b10b085cd9ce1834db263eec1`.
- ZIP hashes: prelim `89F2822D5571BAEAF928EB238B6B1B2CE2EA50F3BECD80D4B33033FBC0E0292D`, round2 `B609188258FB4E9D13439D0B4209A6DBAC23E98EAEFD46CFCC9A277D5171600D`, semifinal `CC87F31D7A2BC7E786A95CA56C2E092D09A4FCFB152760900ADCBF825AC7872C`.
