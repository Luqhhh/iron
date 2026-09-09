# 当前初赛发布：V1_RATE_STRUCTURAL

用户在 V1 桌面交付及哈希核验后回传 test_a **83.0319**，比 R2 的
83.0207 提高 **0.0112 分**。记录为用户回传，未独立核验平台回执或跨提交评测版本。
V1 作为当前发布候选，登记在 `configs/optimization_v0_8/active_release.yaml`。
桌面保持已交付的 V1；原 R2 配置及原包不改动，继续作为回退。

ZIP：`local/runs/optimization-v0.8-v1-challenger-r1/Luqhhh_bf_tap_predict_prelim.zip`

SHA-256：`fdcbe03e8ea31577bfeed0c45cd0a0013bda42ab88d29db703fad9c2f7e557aa`。

独立推理（父目录须存在，输出文件必须不存在；脚本输出内部预测及审计 JSON，不直接生成赛事 ZIP）：

```bash
mkdir -p local/predictions
.venv/bin/python scripts/optimization_v8_cold_predict.py \
  --bundle local/runs/optimization-v0.8-v1-challenger-r1/bundle \
  --data-config local/runs/optimization-v0.8-v1-challenger-r1/cold_data_repaired.yaml \
  --output local/predictions/<unique-id>.csv
```

R2 回退登记：`configs/optimization_v0_4/active_release.yaml`。
原 ZIP SHA-256：`e42602d3045e43b4b49dd1e1c104aa8e5c1f29c639ff5f892b3b07434ed9bbdf`。

这是较小的总分改善，不能从总分单独判断 iron/time 各自的平台收益，
也不把回溯 H1 改善等同于平台改善。本次仅记录反馈和更新发布登记，
没有新训练、标签读取、预测修改或平台上传；不根据这个分数继续调整系数。
历史 RESULTS.md 和本地 release_validation 中的 R2 incumbent/未上传描述，
对应反馈到来之前的实验交付状态，不回写旧证据。

## v0.9/v0.10 完成后的状态

两轮开发均未晋级，V1 保持活动发布，桌面包与原包 SHA-256 仍一致。
v0.10 完成 12 次 residual fit，221 项测试通过，独立冷进程最大差为 0，
但九项质量门槛全部失败。没有新增待测包或 final 模型。

V1 专用脚本当前只接收 test_a；不能用原 baseline 通用 predict 入口加载这个复合
bundle，也不能把内部 `pred_*` 列 CSV 直接当作赛事 `result.csv` 上传。
现成 ZIP 已完成提交格式验证。新机器需要恢复本地模型和推理配置，Git 不存储模型。

[最新结果](../optimization_v0_10/RESULTS.md) · [文档索引](../INDEX.md)
