# 当前初赛发布：V1_RATE_STRUCTURAL

用户在 V1 桌面交付及哈希核验后回传 test_a **83.0319**，比 R2 的
83.0207 提高 **0.0112 分**。记录为用户回传，未独立核验平台回执或跨提交评测版本。
V1 作为当前发布候选，登记在 `configs/optimization_v0_8/active_release.yaml`。
桌面保持已交付的 V1；原 R2 配置及原包不改动，继续作为回退。

ZIP：`local/runs/optimization-v0.8-v1-challenger-r1/Luqhhh_bf_tap_predict_prelim.zip`

SHA-256：`fdcbe03e8ea31577bfeed0c45cd0a0013bda42ab88d29db703fad9c2f7e557aa`。

独立推理（输出路径必须不存在）：

```bash
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
