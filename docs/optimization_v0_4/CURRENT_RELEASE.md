# 当前初赛发布：v0.4 R2

2026-09-09 用户回传 **83.0207**，比 E16 82.9918 高 **0.0289**。
成绩未独立核验；关联依据是用户在核验桌面 R2 包替换后回传成绩。
当前登记：`configs/optimization_v0_4/active_release.yaml`。

原始包：`local/runs/optimization-v0.4-r2-challenger-r1/Luqhhh_bf_tap_predict_prelim.zip`。
SHA-256：`e42602d3045e43b4b49dd1e1c104aa8e5c1f29c639ff5f892b3b07434ed9bbdf`。
模型、预测、账本和 ZIP 保存在本机，不进入 Git。

从已保存的 R2 bundle 独立推理（输出路径必须不存在）：

```bash
uv run --locked --python 3.12 python scripts/optimization_v4_cold_predict.py \
  --bundle local/runs/optimization-v0.4-r2-challenger-r1/bundle \
  --data-config local/runs/optimization-v0.4-r2-challenger-r1/cold_data.yaml \
  --stage test_a --output local/predictions/<unique-id>.csv
```

E16 回退仍使用旧归档导出入口，v3_active 的默认值仅代表旧版本归档：

```bash
uv run --locked --python 3.12 python -m bf_tap.optimization.v3_active \
  --release-config configs/optimization_v0_3_r2/active_release.yaml \
  --export-submission --stage test_a --output local/submissions/<unique-id>
```

E16 SHA-256：`1200d4dd8dee6e797aeeba86db1ceb02293c78dd36796d2e8d75156ce77160aa`。
本轮 OPT-11/12/13 已完成。B/C 未作本轮验收，不将初赛反馈外推为后续月份最优。
