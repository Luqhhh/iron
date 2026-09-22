# 复赛 V2 迁移与 M0 重训

2026-09-22 用户授权删除当前工作树旧复赛数据，改用 D 盘提供的 V2 训练和测试包，并明确授权将 V2 数据纳入 Git、提交推送。`复赛_train/` 与 `复赛_test/` 现在仅包含新包文件；旧版缺失于新包的训练字典和说明已删除。初赛数据、不可变标签和历史运行证据保留。

新增独立入口 `bf_tap_r2.v2_release`，不改变旧版模型、特征和验收协议。训练集必须为 2754 个唯一 V2 训练编号，测试集必须为 322 个唯一 V2 测试编号；特征按 ID 一对一连接，模板顺序严格核验，拒绝旧编号、缺失、重复、不一致及非有限数据。

保持之前选定的 M0 全局中位数算法，在 V2 全部公开训练标签上重新计算两个中位数。使用既定 42/3407 两个种子的五折验证，每折仅从本折训练标签计算中位数，并将相同特征的重复样本分组。V2 指标单独记录，不沿用旧版指标，也不声称旧版挑战者比较能证明 V2 的最优模型。

```bash
UV_CACHE_DIR=/tmp/iron-uv-cache uv run --locked --python 3.12 --extra dev --extra round2 pytest tests/test_round2_v2_release.py tests/test_round2_anchor.py tests/test_round2_data.py -q
UV_CACHE_DIR=/tmp/iron-uv-cache uv run --locked --python 3.12 --extra round2 python -m bf_tap_r2.v2_release --output local/runs/round2-v2/m0-release-r1
```

运行目录必须全新，错误写入 FAILED.json 并保留。输入摘要、模型、OOF、验证明细、result.csv 和 ZIP 留在 Git 外。冷推理禁止读取训练文件，重建 CSV 必须逐字节一致。G0 表示工程和 ID 校验，G1 单独报告 V2 OOF；平台成绩须等待真实回传。
