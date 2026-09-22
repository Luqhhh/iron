# V2.1 C2 增量优化

从用户指定的 `994dc4a7` 新建 `round2-v2.1-c2-refinement`。数据为 synthetic_round2_v2，基准为现有双目标 C2，用户回传 96.0537 未独立登录核验。原 ZIP `b44b7cf325ab339dd10ef8adfc563694111fdbf46814d3e20526d76962839aa4`、全量模型、20 个 C2 折模型、对应 C1 和 OOF/折分均核验来源并冻结，不重训原 C2、不重跑数据泄露审计。

阶段配置：`configs/round2_v2_1/experiment.yaml`；入口：`bf_tap_r2.v2_refinement`。split_seeds=42/3407，model_seeds=42/2026/2027。按 ID 恢复已有 folds CSV，不寻找新划分。旧 choose() 保持历史含义，新增 choose_c2() 专用于相对 C2 晋级。

P1 无新增拟合：报告每目标平均/中位残差（预测减真实）、铁口及真实目标五分位组的 WMAPE/绝对误差贡献、500/1000/1500 棵树的训练与验证曲线、C1/C2 残差相关和异号率，并计算固定 50/50 的 B12。树数仅作描述；真实目标分组仅作报告，不用于推理。

P2/P3 固定五个挑战者：B12=C1/C2 等权；B3=C2 训练种子 42/2026/2027 等权，复用 seed42，仅新增 40 个 CV 模型；D4=depth4/3000轮、其余 C2 参数不变，共20个；L15=固定 15 叶 LightGBM、2000轮，共20个；B2L=C2/L15 等权，零额外拟合。禁止后验权重扫描、挑最好单种子、跨折训练内预测混作 OOF、外层早停、目标链、删列及平台偏置扫描。

LightGBM 复用已锁定 4.6.0，平方损失、CPU deterministic、force_col_wise。spout_no 的类别映射仅从训练折建立、随模型保存；未知类别作为缺失类别处理。所有推理复用同一映射，不从测试集学习。类别外推只在合成工程测试中检查，当前 V2 铁口已被训练覆盖。

相对 C2 的目标独立门槛：两组 pooled WMAPE 均改善，至少7/10折改善，每切分每铁口退化不超过0.001。合格者按两组平均 WMAPE 最低选择；差距 <=1e-6 时按预登记静态推理工作量估计、模型成员数、固定顺序打破平局。工作量只是树数×深度估计（L15使用2000×ceil(log2(15))），不是实测延迟，配置中已冻结。每目标最多一个晋级者。

仅对最后选中目标复用现有 sensitivity()：铁口内按 sample_id 配对重采样2000次，C2/候选及两种切分共用抽样索引；同一 ID 不当成独立两条。报告差值区间和分布，不能覆盖重训不确定性或保证隐藏测试收益。

预算：80个真实CV拟合＋最多4个新增全量拟合。失败开始也计数，不通过重开目录恢复预算。原全量 C2 复用；B12晋级只新训C1，B3只新训两个种子，D4/L15/B2L每目标至多一个新模型。新入口专用工程测试含两个小规模合成拟合，单独记录，不挤占真实建模预算。

A包只改铁量，B包只改时长，未改列直接复制原CSV字符串，新预测用 .17g。无晋级则无对应包。C包仅在A/B均有相对96.0537正收益的平台回传后生成。原包永久保留。平台剩余额度未知，已向用户询问；不能按已有两条成绩推算。单批最多A/B/C三次，不上传基准恢复分数。

```bash
UV_CACHE_DIR=/tmp/iron-uv-cache uv run --locked --python 3.12 --extra dev --extra round2 pytest tests/test_round2_v2_refinement.py -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 UV_CACHE_DIR=/tmp/iron-uv-cache uv run --locked --python 3.12 --extra round2 python -u -m bf_tap_r2.v2_refinement run --output local/runs/round2-v2.1/refinement-r1
```

参数/API核对：[LightGBM 4.6.0 官方参数文档](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html)明确 regression 为 L2，并建议 deterministic 配合列式或行式设置；CatBoost 的 ntree_end 参数已在本地锁定1.2.8签名及合成测试核对。依赖不升级。
