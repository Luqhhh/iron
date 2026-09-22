# V2 旧复赛策略重训对比

2026-09-22 用户要求将 V2 M0 包写入桌面 submission，并重训比较旧版复赛使用过的其他策略。桌面标准文件名已更新为 SHA-256 `b4f285db9d4549d4a2ed4536358f1973bd1bdab7f75d6dabc6a76307e909f766` 的 V2 M0 包。

本轮仅重训已有复赛算法：M0 全局中位数、MS 分铁口中位数、M1 Harrell–Davis 中位数、L1 Ridge、L2 中位数线性回归、S1 样条 Ridge、C1/C2 CatBoost MAE/RMSE、S25 固定弱混合、Q2 二阶中位数回归、K1 RBF SVR。原参数与原后处理不变，不迁移初赛时间序列策略。V2 输入无初赛时间字段。

全部路线共用种子 42/3407、五折、按铁口分层和精确特征重复分组的 V2 折分。各折仅在训练折拟合预处理及模型。七个回归器 × 两目标 × 两种子 × 五折 = 140 个 CV 拟合；中位数估计单独处理，S25 复用指定 L2 铁量/L1 时长预测。最终每目标最多一次全量回归拟合，共最多 142 次。无参数扫描、无测试标签、无平台上传。

沿用原门槛：每目标两种子 pooled WMAPE 均改善，至少 7/10 折改善，最差铁口 WMAPE 退化不超过 0.001。符合门槛的候选按两种子平均 WMAPE 最低选择；差距在 1e-6 内按冻结的简洁优先顺序选择。没有合格候选则保留 M0。门槛对比的是 M0，不能解释为跨候选统计显著性。

配置在 `configs/round2_v2/comparison.yaml`，旧模型配置直接引用、不修改。入口：

```bash
UV_CACHE_DIR=/tmp/iron-uv-cache uv run --locked --python 3.12 --extra dev --extra round2 pytest tests/test_round2*.py -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 UV_CACHE_DIR=/tmp/iron-uv-cache uv run --locked --python 3.12 --extra round2 python -u -m bf_tap_r2.v2_compare --output local/runs/round2-v2/comparison-r1
```

新增目录拒绝覆盖、失败证据保留。所有模型、OOF、输入摘要、拟合台账和提交包保存在 Git 外。独立进程将回读全部 CV 模型，检查训练数据/源码/模型身份，重算 OOF、指标、晋级决策并检查乱序一致性；全量模型独立冷推理禁止读取训练文件，结果应逐字节相同。
