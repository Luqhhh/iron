# 冷进程 JSON 月份键修复

12 个时长系数和全部 V1/V7/D1 outer 预测已成功保存；评分标签尚未读取。检查发现冷进程 JSON manifest 的 origins 键为字符串，而模型 inventory 的运行时索引为整数。该表示差异会导致 outer 恢复 KeyError。原冷审计在已完成前三份 H2 replay/reverse 后主动中止，主运行记录 subprocess SIGTERM 失败；根因复现与中止原因保存在 engineering_repair/failure_cause.json，原 failure.json 保留。

修复只在 manifest 读取边界恢复 origins 整数键，另增加无需再次拟合的工程恢复入口和序列化回归测试。未改变时间、预测、系数、训练/特征/模型/预算/门槛。原三个源码文件在改动前按原 manifest SHA 完整归档到同一 run 的 engineering_repair/original_sources/。

恢复入口要求 12 个原成功 fit 与冻结预测摘要完整、原失败存在和干净修复登记提交；将原 manifest 摘要、原源码归档身份、新三个源码摘要与修复提交冻结为 engineering_repair/manifest.json，再追加新账本。原 manifest 不覆盖，所有其他原源码、参数、模型、输入、历史/证据和旧账本仍严格按原摘要校验。身份变化单独记录，不以临时容差通过。

实际恢复命令：

```bash
uv run --locked --python 3.12 pytest tests/test_horizon_calibration.py
uv run --locked --python 3.12 python scripts/optimization_v14_h2_calibration.py \
  --output local/runs/optimization-v0.14-opt30-r1 --resume-engineering
```

恢复仅运行独立冷重载/最优性证书验证与保存预测评分，不调用 P0、ScalarBudget.fit 或 lad_coefficient。CatBoost/校准 fit 保护保持启用；追加赛事 fit 0。原失败和已存系数不删除，不换目录重新拟合。
