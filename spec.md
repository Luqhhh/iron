# optimization-v0.2 模型规格

状态：Approved（2026-09-07）
范围：训练集模型构建与非保护时段验证
排除：修改 baseline-v0.1、读取 2024 年 11 月保护标签、解封最终 holdout

## 1. 问题定义

冻结 baseline 使用同一套 as-of 特征，分别训练两个 CatBoost MAE 回归器，并用全局中位数 B0 和按出铁口中位数 B1 作对照。

现有证据存在时间稳定性差异：

- DEV_SHORT：CatBoost E = 0.171456，优于 B0 0.178157 和 B1 0.178897。
- DEV_LONG：CatBoost E = 0.185646，差于 B0 0.176148 和 B1 0.176101。

首轮核心假设：工艺特征模型可捕获短期偏差，但跨时间稳定性不足；以 B1 稳定先验作锚点并控制旧样本影响，可能保留短期收益并降低漂移风险。

## 2. 目标和非目标

目标：

1. 使用非保护训练标签建立 leakage-safe rolling OOF。
2. 在当前特征集合不变时比较三种模型层优化。
3. 以 E 选择候选，同时限制单目标退化。
4. 产出可重复训练、可 bundle、可离线预测的模型。

非目标：

- 首轮不新增或改变特征窗口。
- 首轮不做大规模超参数搜索。
- 不改变目标、时间语义或 WMAPE 定义。
- 不用 2024 年 11 月标签开发、选择或调权。
- 不替换或重写 baseline 证据。

## 3. 数据和验证合同

- 训练样本只使用保护期前、且在 fold cutoff 已可获得目标值的记录。
- operation、burden、tap history 继续通过现有 as-of builder 读取。
- 2024 年 11 月目标在开发流程中不可读，不得绕过保护接口。
- 候选选择使用按时间递增的 rolling-origin folds。
- 每折训练样本早于验证窗口，训练标签在 cutoff 前可用。
- 所有对照和候选共用相同验证行。
- fold 边界由配置声明并写入运行证据。

主汇总采用 pooled WMAPE：先跨折累计绝对误差分子和真实值绝对值分母，再计算两个目标 WMAPE 与 E。折均值只作稳定性诊断。

## 4. 必须报告的对照

- B0：训练可见标签的全局中位数。
- B1：满足最小样本数时使用出铁口中位数，否则回退 B0。
- Frozen CatBoost：baseline-v0.1 的参数、特征和预测语义。
- Candidate：本规格定义的 optimization 模型。

所有对照在完全相同的 OOF 样本上评估。

## 5. 候选方案

### M1：目标级 CatBoost–B1 融合（推荐先做）

每个目标独立计算 prediction = alpha × CatBoost + (1 - alpha) × B1。alpha 仅由训练期 rolling OOF 选择，使用预注册的小型离散网格；结果近似相同时选择更偏向 B1 的权重。

优点：成本低、解释清晰，可利用短期 CatBoost 与长期 B1 的互补。
风险：固定权重不能适应突发漂移；残差高度相关时收益有限。

### M2：近期样本加权 CatBoost

按训练样本距 fold cutoff 的时长生成非负、单调递减的权重。首轮只比较少量预注册衰减强度和等权对照；除 sample weight 外保持 baseline 参数和特征不变。

优点：直接针对概念漂移并保持 baseline 架构。
风险：有效样本量下降，可能损害稀有出铁口并过拟合单一时间窗。

### M3：B1 anchor 残差 CatBoost

训练 residual = target - B1_anchor，最终输出 B1_anchor + residual_prediction。训练行 anchor 必须由 expanding/as-of 或 OOF 方式生成，不得包含本行目标或未来目标；预测 anchor 只使用预测时点前的可用历史。

优点：显式拆分稳定位置先验和动态工艺偏差。
风险：实现及 bundle 语义更复杂；早期历史不足需一致的回退规则。

暂不进行大规模超参数搜索：在解释 DEV_LONG 退化前，它容易拟合开发窗口，也无法隔离时间权重或数据切分的贡献。

## 6. 推荐实施顺序

1. 生成统一 rolling OOF 表。
2. 检验 CatBoost 与 B1 的残差互补性并实施 M1。
3. 固定特征和参数实施 M2，测量 recency weighting 的净贡献。
4. 在泄漏测试完备后实施 M3。
5. 只有独立候选稳定且残差互补时才评估 M4。

## 7. 验收规则

### G0 工程门禁

- 锁定 Python 3.12 测试路径通过，baseline 测试无回归。
- 保护标签未访问，ledger 无异常。
- 相同输入、配置和 seed 产生一致预测。
- bundle round-trip 后预测在容差内一致。
- run 不覆盖，私有产物不进入 Git。

### G1 模型质量

批准本规格即视为批准以下首轮规则：

1. 候选 pooled rolling OOF E 低于 frozen CatBoost 与 B1。
2. 任一目标相对该目标更好的对照，WMAPE 绝对退化不超过 0.002。
3. 候选在半数以上 folds 优于 frozen CatBoost，最差折无无法解释的显著退化。
4. 候选冻结后运行 DEV_LONG/DEV_SHORT，仅作确认，不继续调权。
5. 首轮不运行保护 holdout；候选、配置和证据冻结后才另行授权。

若主指标改善但目标保护或时间稳定性失败，只能标记为有条件改善，不能判定 G1 PASS。

## 8. 配置与产物

配置至少记录 optimization 版本、候选 ID、fold、两目标参数、weight/anchor/blend 规则、baseline digest、seed 和确定性设置。

每次运行至少保留配置快照、非保护 manifest/digest、折级与 pooled 指标、行级 OOF、误差切片、bundle、语义版本及 G0/G1 状态。产物必须位于 Git 忽略且不可覆盖的 run 目录。

## 9. 必测失败模式

- B1 anchor 含本行或未来目标。
- 验证特征读取 cutoff 后的数据。
- 权重由 DEV_LONG、DEV_SHORT 或保护集反复调优。
- 两目标共用权重却无证据。
- 空折或零分母被静默处理。
- bundle 丢失 anchor、weight 或 blend 语义。
- optimization 覆盖 baseline 配置或 run。
- 只报告 E 而掩盖单目标退化。

## 10. 待批准决策

批准本规格将确认：

1. 首轮只做模型层优化，沿用当前 as-of 特征集合。
2. 候选顺序为 M1、M2、M3。
3. 采用第 7 节 G1 规则。
4. 修改模型代码前建立独立 optimization-v0.2 分支。
5. 首轮不访问 2024 年 11 月保护标签。

## 11. 代码目录合同

baseline.py 保持冻结且不移动。新增模型分级放置：

    src/bf_tap/models/optimization_v02/
      contracts.py
      registry.py
      m1_blend/
      m2_recency/
      m3_residual/
      m4_ensemble/

配置镜像到 configs/optimization/v0.2/，测试镜像到 tests/optimization_v02/。候选只通过公共协议或 registry 被调用，不得导入其他候选的内部实现；M4 仅在误差互补门禁通过后启用。
