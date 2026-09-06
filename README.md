# bf-tap-predict

高炉铁次预测的防泄漏 baseline 实现。`md/` 是原始方案归档；活动配置、契约和状态分别位于 `configs/`、`docs/` 与 `EVIDENCE_STATUS.json`。

模型路线固定为两个 CatBoost MAE 回归器。`competition-timestamp-contract-v1` 是基于赛事公开时间字段的 `ASSUMED` 操作约定，不等同于官方验证过的完整发布时间语义。11 月标签由独立的 `holdout-protection-v1` 硬门禁隔离。

## 本地开发

```bash
uv sync --locked --extra dev --python 3.12
uv run pytest
uv run python scripts/check_no_private_artifacts.py
uv run python -m bf_tap --help
```

## 数据准备

将 `configs/data.example.yaml` 复制为被忽略的 `configs/data.local.yaml`，只填写本机路径。字段时间语义固定在可提交的 `configs/data_contract.yaml`，保护边界固定在 `configs/protection.yaml`；恢复语义不需要复制私人路径配置。

真实数据、模型、运行报告和提交文件均由 `.gitignore` 与 CI allowlist 检查排除。仓库公开不代表赛事数据允许外发。

## 结构审计与 DEV

```bash
uv run python -m bf_tap audit \
  --data-config configs/data.local.yaml \
  --output local/reports/structural-audit-<id>.json

uv run python -m bf_tap validate \
  --data-config configs/data.local.yaml \
  --data-contract configs/data_contract.yaml \
  --protection-policy configs/protection.yaml \
  --config configs/baseline.yaml \
  --feature-config configs/features.yaml \
  --split-config configs/validation.yaml \
  --suite development \
  --output local/runs/<unique-dev-run-id>
```

DEV 命令会在读取标签前校验所有 fold 及独立保护边界。run ID 已存在时直接失败；失败目录和证据不得删除或覆盖。

## 训练与离线推理

先用不跨保护边界的 development 模式验证完整文件链路：

```bash
uv run python -m bf_tap train \
  --data-config configs/data.local.yaml \
  --data-contract configs/data_contract.yaml \
  --protection-policy configs/protection.yaml \
  --config configs/baseline.yaml \
  --feature-config configs/features.yaml \
  --mode development \
  --train-start 2024-03-01T00:00:00+08:00 \
  --fit-cutoff 2024-11-01T00:00:00+08:00 \
  --output local/runs/<unique-train-run-id>

uv run python -m bf_tap predict \
  --bundle local/runs/<train-run-id>/bundle \
  --data-config configs/predict.local.yaml \
  --stage test_a \
  --output local/predictions/<unique-predict-run-id>
```

`predict.local.yaml` 只需要对应阶段样本、operation 和 burden 路径；不需要 `train_samples.csv`。历史授权快照及特征语义随 bundle 保存。`predict` 不调用 `fit`。

正式 `official-release` 训练会读取受保护标签，必须先有冻结 manifest 和显式 `final_training` 访问账本。本轮禁止运行该生命周期。

## 校验和打包

```bash
uv run python -m bf_tap check-submission \
  --data-config configs/predict.local.yaml --stage test_a \
  --path local/predictions/<id>/result.csv

uv run python -m bf_tap pack \
  --data-config configs/predict.local.yaml --stage test_a \
  --team-name <team> --result local/predictions/<id>/result.csv \
  --output-dir local/submissions/<id>
```

`pack` 会自行按目标阶段主表重新检查列、ID、顺序、行数、有限性和非负性，并在压缩后重新读取 ZIP payload 验证摘要；不依赖先执行 `check-submission`。

## 状态解释

- G0：工程正确性与可复现性。
- G1：冻结模型质量；DEV_LONG 已失败，修复工程代码不会改写该历史结果。
- `ASSUMED`：可复现的操作口径，但仍待官方补充业务发布时间依据。
- `VERIFIED`：有可定位的官方材料直接确认。
