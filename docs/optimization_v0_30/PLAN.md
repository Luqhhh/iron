# optimization-v0.30：双目标 OOB 收益组合与时长森林固定扩容

本阶段是独立优化阶段，审阅基点 `optimization-v0.29-oob-leaf-responses@5e00f63`。两个候选在任何本轮平台反馈前同时冻结，平台预算 2、各一次，默认顺序 A→B，A 的反馈不得改变 B；不生成第三个组合。

- A `V30A_OOB_BOTH_TARGETS`：按 `sample_id` 一一对齐，铁量列逐字符串复制 V29I 完整 CSV 列，时长列逐字符串复制 V29T 完整 CSV 列。不取两份 CSV 的逐列平均，不重新计算 rate/beta，不再次门控，不改变序列化精度。V29I 的铁量已含 CatBoost 完整端点与 OOB 铁量 QRF 各 1/2；V29T 的时长已含旧 V21 后处理，均不得重复施加。
- B `V30B_OOB_TIME_1024`：铁量与 A 完全相同；时长把已认证 V26A 256 棵绝对误差时长森林深复制后 `warm_start=True`、`n_estimators=1024`，对同一 X/y/ID 顺序只调用一次 fit，追加 768 棵树，再对全部 1024 棵树沿用 v0.29 的逐树 OOB 叶响应协议，最后重放原 v0.15 支持度 gate 与 V21 的 25% 收缩。新 gate 不得使用 1024 棵支持度。

注册收益恒等式（同一评分数据、ID、协议且未触发评分下限时）：`S_A = S_V29I + S_V29T - S_V28I`。按显示值 83.2970 + 83.3141 - 83.2936 = 83.3175，是算术推算而非实测，也不是泛化证据。B 相对 A 是本轮扩容主对照；B 高于 V29T 不能把全部收益归给新增树。

## 预算与身份

| 项目 | 开发 | 最终 | 总上限 |
| --- | ---: | ---: | ---: |
| A 新模型/校准 fit | 0 | 0 | 0 |
| B 森林追加 fit 调用 | 6 | 1 | 7 |
| B 新训练树 | 4,608 | 768 | 5,376 |
| B 复用旧树 | 1,536 | 256 | 1,792 |
| B 各模型持有树数合计 | 6,144 | 1,024 | 7,168 |
| B 新 1024 棵 OOB 附件 | 6 | 1 | 7 |
| 预处理/CatBoost/铁量森林/rate/q/LAD 新 fit | 0 | 0 | 0 |
| 新候选包/平台测试 | — | 2/2 | 2/2 |

原 210 列 raw schema、209 数值列 + spout_no、经认证的 213 列 float32 transformed schema、原 v0.15 预处理器、输入值/缺失填充/类别词表/列序/训练行身份与行序、非负 tap_time_len 全部不变。开发沿用 06–11 六个 cutoff，最终沿用 2,754 行与 `2024-12-01 01:44:00+08:00`；不读取测试标签、不扩大训练月份、不移动 cutoff。

扩容前后逐项核验：原 256 棵的 seed/criterion/splitter/children/features/thresholds/impurity/sample counts/values 等完整树状态不变；前 256 份 `estimators_samples_` 与原实际 bootstrap 完全一致；总树数恰为 1024、新树恰为 768；旧源对象、旧磁盘模型及附件摘要不变。禁止 `sklearn.base.clone` 未拟合克隆后从头 fit，禁止参数不匹配时静默从头训练；前缀不一致必须阻断并保留失败。

## OOB 与回归

对全部 1024 棵树使用同一逐树 OOB 规则：仅用本树 bootstrap 未抽中的原训练成员，空 OOB 叶回退该树同一完整叶，一个 OOB 成员合法。先汇总 1024 棵树的统一分布，再取原始时长的较小加权中位数；不平均 256 棵模型各自或每棵树的中位数，不使用 `RandomForestRegressor.predict()` 默认聚合。

前 256 棵的抽样身份、叶成员与回退证书必须与 v0.29 时长附件逐叶一致；旧 full-leaf mapping 复用经验证结果，只为新增 768 棵派生。把新森林限制到旧前 256 棵时，完整原始 QRF 时长、OOB 响应与最终 V29T 时长必须精确复现；这是工程回归，不是第三候选。

离线沿用六 origin 网格 6:4、7:4、8:4、9:3、10:2、11:1，DEV_LONG 复用 July、DEV_SHORT 复用 September；主口径 `macro_origin_mean_wmape`，pooled 仅诊断；共享 calendar-week bootstrap 仍为 1000 次、seed 2026，无效 draw 不补抽。逐 cell 与宏平均核验：A 与 V29T 时长相同且 `ΔE(A,V29T)=0.5·ΔWMAPE_iron`；A 相对 V28I 的增量等于 V29I、V29T 各自增量之和（1e-12）；B 与 A 铁量相同且 `ΔE(B,A)=0.5·ΔWMAPE_time`。

## 交付与停机

冷审计覆盖前缀树、原/新 bootstrap、全部 full/OOB 成员、原响应、gate 来源、全量/反序/分块/子集/单行结果及最终 CSV/ZIP 身份；所有 fit 入口（追加森林、底层树、新 wrapper、原 QRF、预处理器、CatBoost、校准）在冷审计中一律拦截，不做“重训验证”。

工程硬门槛通过且与旧包/另一包非完全相同时又各保留一个事前登记平台名额；离线弱于 V1 或父方案不伪造 PASS，也不自动取消两项实验。桌面写入、替换、公开 push 与平台上传均不自动执行。正式复赛数据改变时新建来源契约，不能把 test_a 成绩外推为 test_b 结果。
