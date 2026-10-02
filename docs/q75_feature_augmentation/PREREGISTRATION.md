# 预登记：特征增强（对数/对数比/乘积）与 mixup 筛选（2026-10-02）

## 动机

- 现有证据把"库内加权、校准、日程、损失、选轮、轨迹平均、残差修正"全部关闭；GRANDE 硬树试点也因精度比 1.07–1.09 而拿不到融合权重（残差相关 0.93–0.96 但单独误差过大）。
- 唯一有机制依据、且**在 TabM 时代从未系统测过**的轴是**输入表示**：21 个过程量之间若存在乘性/比值/对数关系（高炉风量—富氧—燃料—压差之间通常是乘性关系），MLP 用有限数据很难学到除法结构。历史 V3.1/V3.3 的"表达式/特征结构"搜索是 CatBoost 时代的，且没有对数比特征。
- 另一条同族的正则化手段是 mixup（在样本间做线性插值），对"数据受限"的时长目标可能等效于增加有效样本。

## 假设

H1：显式对数、对数比与乘积特征能让同一 TabM 配方在**时长**（学习曲线显示仍受数据限制）上取得净改善。
H2：mixup 作为正则化在时长上取得净改善。

## 冻结设计

- 数据/折/骨架：与日程筛选一致（联合 TabM + PLR 0.01、V12 训练设置、冻结折向量）。输入表示是唯一自变量；其余不动。
- 增强特征（全部只由原始 21 列确定性计算，不用标签、不用测试集）：
  - 对数：`log` 于 `air_volume, oxygen, coke_rate, fuel_rate, consumption, pig, air_speed, total_press_diff`，`log1p(humidity)`；
  - 对数比：`log(oxygen/air_volume)`、`log(coal_rate/air_volume)`、`log(coke_rate/air_volume)`、`log(fuel_rate/air_volume)`、`log(air_volume/air_speed)`、`log(hot_air_press/cold_air_press)`、`log(furnace_top_press/total_press_diff)`、`log(consumption/pig)`、`log(hot_air_temp/furnace_top_temp_avg)`、`log((humidity+1)/air_volume)`；
  - 乘积：`air_volume*oxygen`、`air_volume*(coal_rate+coke_rate)`、`air_volume*(humidity+1)`、`hot_air_press*hot_air_temp`。
  - 共 21 + 23 = 44 列；预处理仍是训练折内标准化（原协议）。
- mixup：批次内 `λ~Beta(0.4,0.4)`，`x=λx+(1−λ)x_perm`、`y=λy+(1−λ)y_perm`，其余训练循环不变。
- 臂：`CONTROL`（逐位一致闸门）、`ENG`（增强特征）、`MIXUP`（原特征 + mixup）。
- 单位：`split_seed=42` 的 fold 0 与 fold 1；共 3 臂 × 2 折 = **6 次正式拟合**。
- 升级判据（预先固定）：某臂在两折上**同向**且平均优于 `CONTROL`，则进入 `split_seed∈{42,3407}` × 5 折确认（最多 2 臂 × 10 = 20 次）；否则关闭该臂。
- 预算：筛选 6 次 + 确认最多 20 次 = ≤26 次正式拟合；无全量、无包、无桌面、无上传。
- 资源：单数值线程、worker ≤ 6、峰值内存 1536 MiB/worker；不设时间预算。

## 结果（2026-10-02，筛选即关闭）

`split_seed=42` fold 0/1，3 臂 × 2 折 = 6 次拟合；`CONTROL` 与已记录 V12 预测逐位一致（0.0）。

| 臂 | 铁量均值 | Δ | 时长均值 | Δ | 两折更优 |
| --- | ---: | ---: | ---: | ---: | :--: |
| `CONTROL` | 0.037174 | 0 | 0.038135 | 0 | — |
| `ENG`（21+23 列对数/对数比/乘积） | 0.037505 | +0.000331 | 0.039115 | +0.000979 | 否 |
| `MIXUP`（α=0.4） | 0.046694 | +0.009520 | 0.050227 | +0.012092 | 否 |

两臂在两个目标、两个折上均明显变差，按预登记升级判据**直接关闭**，不进入确认（未消耗阶段 2 预算）。结论限定于该冻结增强特征集与 mixup 参数；"所有特征工程无用"不由此推断，但显式的乘性/比值结构与样本间插值在本数据上明确有害。
