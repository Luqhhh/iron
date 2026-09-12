# V11 有限平台探针与 V10 独立推理

起点：`b01ab117589377073710de447114cdb1074db32c`，分支 `platform-probes-r2`。
本阶段获用户实施规格授权，仅恢复已认证最终模型；新模型、预处理、LAD 拟合均为 0。

唯一新候选 `V11_V6I_IRON_QRF_MEAN_TIME` 保留 V10 原铁量 CSV 字段，
时长取原最终 QRF 同一叶子响应分布的加权均值。中位数必须复现原 V8 全精度及六位小数结果。
只有一个新 ZIP；均值若在提交精度下与 V10 完全一致则不封包。
平台反馈未取得时保持 pending，不改历史 G1；V10 83.1951 仍是最高用户回传包。

新增入口从最终 V6I 模型、只读铁量系数和原 V1/R2/E04/rate 恢复铁量；
原 QRF worker 独立恢复森林和预处理器，按 ID 合并时长。原 as-of 特征和来源契约保持不变。
`test_a` 重放须与原 V10 result.csv 字节一致；旧 `test_b` 仅工程预演。
新来源、提前于 cutoff 的样本、训练 ID 重叠、标签路径、隐式拟合或产物覆盖必须拒绝。
反序、分块、子集、单样本与冷进程恢复均须逐样本精确一致。

实际入口：

```bash
uv run --locked --python 3.12 python scripts/platform_probe_r2.py prepare --output local/runs/platform-probes-r2-r1
uv run --locked --python 3.12 python scripts/platform_probe_r2.py predict --stage test_b --bundle BUNDLE_JSON --bundle-sha256 BUNDLE_SHA256 --data-config LABEL_FREE_STAGE_CONFIG --output UNIQUE_INTERNAL_CSV
uv run --locked --python 3.12 pytest
uv run --project workers/qrf_v015 --locked --python 3.12 pytest
```

包、响应、预测、模型、账本和核验报告均留在忽略的 local/。
本轮自动上传、桌面写入为 0。剩余平台次数未知时只完成本地准备；
手动探针须剩余至少两次且已核验 V10 原 ZIP 可恢复。低分时使用原 ZIP 恢复，不能重训。
V2、v0.16 暂停，不追加均值/中位数混合或其他候选。
