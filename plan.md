# optimization-v0.2 实施计划

> **For agentic workers:** 实施时必须逐任务执行测试先行与审查；可使用 superpowers:executing-plans。

**Goal:** 在不接触保护标签的前提下，构建比 frozen CatBoost 与 B1 更准确且更稳定的 optimization-v0.2。

**Architecture:** 复用现有 as-of 特征管线，建立统一 rolling OOF；依次验证 CatBoost–B1 融合、近期样本加权和 B1 anchor 残差模型。所有候选保持独立配置、bundle 语义和不可覆盖证据。

**Tech Stack:** Python 3.12、pandas、NumPy、CatBoost、pytest、YAML。

**Spec:** spec.md

状态：已批准，阶段 1 进行中
基线：baseline-v0.1-reproducible（只读对照，不原地修改）

## 1. 目标

使用训练集中的非保护标签构建 optimization-v0.2 候选模型，降低综合指标 E = (WMAPE_tap_iron + WMAPE_tap_time_len) / 2。分别报告两个目标、各时间折，以及相对 B0、B1、冻结 CatBoost 的表现。

## 2. 硬边界

- 不修改或覆盖 baseline-v0.1 的模型、配置、特征、语义、阈值和证据。
- 优化在独立的 optimization-v0.2 分支和配置命名空间中进行。
- 开发流程不得读取 train_samples.csv 或 tap_history_train.csv 中 2024 年 11 月目标。
- 训练、验证、预测复用同一 as-of builder 和时间可见性语义。
- run 目录只可新建，不可覆盖；失败证据也保留。
- 私有数据、模型、预测、报告、ledger 和提交文件不得进入 Git。
- G0 工程可复现性与 G1 模型质量分开验收。
- commit 与 push 仅在用户明确指令后执行，不以阶段完成自动触发。

## 3. 阶段与检查点

### 阶段 0：设计冻结

- [x] 确定 E 为主指标，两个目标 WMAPE 分别报告。
- [x] 建立 plan.md 和 spec.md。
- [x] 用户批准模型路线和验收规则。

批准前不修改模型实现。

### 阶段 1：隔离和实验入口

- [x] 建立 codex/optimization-v0.2 分支。
- [x] 核对训练、验证、配置、bundle 和证据入口。
- [x] 新增 optimization 配置，不改 baseline 配置。
- [x] 按模型类别建立分级源码、配置和测试目录。
- [x] 每个实验记录源码、配置、数据 manifest、seed 和唯一 run id。
- [x] 保证 run 目录不可覆盖。

### 阶段 2：测试先行

- [x] 测试 M1 优化入口不可访问保护标签。
- [x] 测试 rolling-origin 边界及标签可用性。
- [ ] 测试 OOF/B1 anchor 不含本行或未来标签。
- [x] 测试 M1 融合权重和 pooled WMAPE。
- [ ] 测试 M2 近期样本权重。
- [ ] 测试 bundle round-trip、离线预测和 baseline 无回归。

### 阶段 3：统一 OOF 对照

- [ ] 仅用保护期前可用标签建立 rolling-origin folds。
- [ ] 在相同验证行生成 frozen CatBoost、B0、B1 的 OOF 预测。
- [ ] 保存行级 OOF、折级及 pooled 指标。
- [ ] 按目标、月份、出铁口、缺失和 stale 状态切片。
- [ ] 分析 CatBoost 与 B1 的残差相关性和时间漂移。

### 阶段 4：首轮候选

- [x] M1：完成目标独立融合、OOF 选择和安全标签 runner。
- [ ] M1：接入本地训练数据并生成真实 rolling OOF 指标。
- [ ] M2：只改变 sample weight 的近期样本加权 CatBoost。
- [ ] M3：使用 expanding/as-of 或 OOF anchor 的 B1 残差 CatBoost。
- [ ] M4：仅在稳定候选残差确实互补时进行二次融合。

首轮保持当前 as-of 特征集合不变，以便准确归因。

### 阶段 5：选择和冻结

- [ ] 按 spec.md 的质量和稳定性规则排序。
- [ ] 分别检查 tap_iron、tap_time_len 是否退化。
- [ ] 冻结候选后再跑 DEV_LONG/DEV_SHORT，只作确认，不继续调权。
- [ ] 记录每个候选入选或拒绝理由，冻结配置及语义版本。

### 阶段 6：权威验证

- [ ] 在锁定 Python 3.12 环境运行全量测试。
- [ ] 运行重复训练及预测一致性检查。
- [ ] 确认保护 ledger 无异常。
- [ ] 分开报告 G0、G1，输出实验总结和指标表。

## 4. 模型目录合同

冻结文件 src/bf_tap/models/baseline.py 保持原路径和语义。新增实现按候选模型分级：

- src/bf_tap/models/optimization_v02/
  - contracts.py：候选模型共享协议和类型。
  - registry.py：按 candidate id 解析实现。
  - m1_blend/：CatBoost–B1 融合。
  - m2_recency/：近期样本加权。
  - m3_residual/：B1 anchor 残差模型。
  - m4_ensemble/：满足证据门槛后才启用的候选融合。
- configs/optimization/v0.2/：common.yaml 与各候选配置。
- tests/optimization_v02/：共享合同、OOF 及各候选测试。

候选不得跨目录导入彼此内部实现；共享逻辑只有被至少两个候选使用时才进入公共模块。

---

## 5. 首轮暂不实施

新增特征窗口、趋势/波动特征、目标级特征选择、大规模超参数搜索和异构模型推迟到第二轮。只有首轮误差证据支持时，才更新 spec 后实施。

## 6. 进度

| 日期 | 状态 | 说明 |
|---|---|---|
| 2026-09-07 | Design | 用户同意 E 为主指标；建立计划和规格草案。 |
