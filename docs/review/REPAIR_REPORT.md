# Baseline 工程修复报告

日期：2026-09-06。固定父提交：`3f94a9892bf5997746624672ea512ff0e7067495`。结论：本地锁定环境 G0 工程闭环通过，G1 保持 `FAIL_DEV_LONG`，发布状态为 `BASELINE_REPRODUCIBLE_QUALITY_FAILED`。

本轮没有改变两个 CatBoost 模型的参数、800 轮、MAE、特征窗口、非负裁剪或质量门槛；验收执行阶段没有执行 H1–H4、保护评分、正式最终训练或排行榜提交。旧运行 `dev-baseline-v0.1-contract-v1-r2` 和原报告均保留。修复工程随后以提交 `4d6c2be4420d631e8383b212b6fdab036031bc5b` 推送至 `origin/main`。

## 批次交付

### Repair-1：R1/R2/R3/R4

- 先加入坏输入回归；初次定向收集因待实现 API 导入失败，随后实现并转绿。
- 独立 `holdout-protection-v1` 在标签解析前限制 DEV 到 `2024-11-01T00:00:00+08:00`；fold 顺序、时区、唯一 ID、非空分区和 train/eval ID 交集均为硬校验。
- 新增历史 `reference_time <= tap_end_time <= available_at`、有限目标及同 ID 跨表元信息/目标一致性校验；11 月只允许无目标元数据审计。
- 数字解析区分声明缺失标记与非法文本，拒绝正负无穷；数据路径与语义映射分离，配置按命令严格检查未知、缺失和空值。
- `competition-timestamp-contract-v1` 保持 `ASSUMED`，现记录规则 PDF、字段字典 SHA-256 及 operation/burden/history/target 的字段级依据位置、结论和待确认项。

### Repair-2：R5/R6

- CLI 新增正式 `train`/`predict`；bundle v2 绑定语义与特征配置、字段名/类型、训练和历史身份、代码/环境/锁文件身份及组件摘要，并包含授权历史快照。
- 文件级合成集成测试在删除训练标签文件后启动两个独立预测进程，证明预测不调用 `fit`；同结果再由 `pack` 自检并压缩。
- `pack` 读取阶段主表 ID，自行拒绝错误列、错行数、重复/缺失 ID、非有限或负预测；ZIP 创建后重新读取 payload 并复核摘要。保留拒绝覆盖。

### Repair-3：R7/R8

- run 启动前写 resolved config、环境、代码 commit/tree/dirty/source snapshot、完整输入与 `uv.lock` 摘要；完成前复核输入未变化，最终状态原子写入，相同 run ID 拒绝复用。
- DEV 保存模型/预测/分区/历史授权集合摘要。缓存明确为 `enabled=false`，没有把未接入的缓存函数写成已验证链路。
- CI 的权威 job 使用 Python 3.12 + `uv sync --locked --extra dev`，输出 JUnit/环境并运行不可跳过的模型测试；Python 3.11 作为兼容性 job 单独标注。修复后的远程 CI 尚未执行。
- 添加公开仓库私有资产 allowlist 检查；根 `AGENTS.md` 和 `md/ARCHIVE_NOTICE.md` 明确执行权威位置与保密边界。

### Repair-4：P2、文档与真实复现

- CatBoost、B0、B1 均保存逐月、逐铁口、缺失和陈旧分组；目标指标保存绝对误差分子、目标分母、signed error、signed bias、裁剪数。Top-error 增加 `reference_time`、`spout_no` 与三类来源可用时刻/可见行数。
- 使用最终源码与字段证据契约产生两次干净 DEV run：`local/runs/dev-baseline-v0.1-repair-r5`、`local/runs/dev-baseline-v0.1-repair-r6`。
- 同 bundle 的 test_a 双独立进程输出：`local/predictions/test-a-repair-p5`、`local/predictions/test-a-repair-p6`。路径配置只含测试主表、operation 和 burden；训练标签文件不是推理依赖。

## 实际命令与结果

```bash
uv sync --locked --extra dev
.venv/bin/pytest --junitxml=local/reports/pytest-repair-main.xml -q
.venv/bin/python scripts/check_no_private_artifacts.py
.venv/bin/python scripts/write_environment.py --output local/reports/environment.json
.venv/bin/python -m bf_tap audit --data-config configs/data.local.yaml --output local/reports/structural_audit_repair.json
```

最终测试：48 passed，0 failed，0 skipped；Python 3.12.12。JUnit SHA-256 `35b0716255fefe122068e11f1029a549b36e2477399d56ce74cb9976418e9129`，环境摘要 SHA-256 `5798759f72fe32582d0accf3103cce26439ba0b29ff6f2f30ef692506ceb4992`，`uv.lock` SHA-256 `bd804b855a47234b07ae7ea4baecaa584c479e88fa8564207bab79028450d6dc`。私有资产检查通过。结构审计 7 个 CSV，SHA-256 `aeb9d435a252ce1536e7221043218676f98999598abe64cd9776da9bdb5d35d0`。

两次 DEV 均执行：

```bash
.venv/bin/python -m bf_tap validate \
  --data-config configs/data.local.yaml \
  --data-contract configs/data_contract.yaml \
  --protection-policy configs/protection.yaml \
  --protection-ledger local/manifests/protected_access.json \
  --config configs/baseline.yaml \
  --feature-config configs/features.yaml \
  --split-config configs/validation.yaml \
  --suite development \
  --output local/runs/<dev-baseline-v0.1-repair-r5-or-r6>
```

| run | 执行 | 质量 | walltime | run manifest SHA-256 | metrics SHA-256 |
| --- | --- | --- | ---: | --- | --- |
| repair-r5 | PASS | FAIL | 101.00s | `4b34facf8205afc046bcfa6797a9650368d9c340d58908315566cc1571a11013` | `78d44b5590b9fedd2c543d9ac2058a1a1f56f430905509f5fac578efec1e4f2c` |
| repair-r6 | PASS | FAIL | 99.81s | `9da14363bf6f806025d12a3c71a070a6a678fb64d9aaae51c3e23804f5dc608b` | `0950275c24e920de8ff2a2314ab7b7c758f6e664baf8f67b7387748402401a28` |

两次指标完全一致：

| 场景 | CatBoost E | B0 | B1 | 质量 |
| --- | ---: | ---: | ---: | --- |
| DEV_LONG | 0.185646 | 0.176148 | 0.176101 | FAIL |
| DEV_SHORT | 0.171456 | 0.178157 | 0.178897 | PASS |

DEV_LONG/SHORT 重训 raw 最大绝对差均为 `0.0`，最终 CSV 均字节相同。CatBoost 模型二进制摘要不同，推断为序列化运行元数据差异；不把它写成二进制复现通过。每个 bundle 自身的组件摘要校验均通过。

双独立推理命令：

```bash
.venv/bin/python -m bf_tap predict \
  --bundle local/runs/dev-baseline-v0.1-repair-r5/DEV_SHORT/bundle \
  --data-config configs/predict.local.yaml --stage test_a \
  --output local/predictions/<test-a-repair-p5-or-p6>
```

335 行 raw 最大绝对差 `0.0`，result CSV 字节相同且阶段检查通过。两次 walltime 为 6.92s/7.77s，每样本 0.0207s/0.0232s，峰值 RSS 164,496/168,100 KiB，CatBoost 文件合计 993,048 bytes，结果 SHA-256 均为 `d287bd6e1b096f4e16dfaec4b76b5036aed15b50b1aa4a2f3da3243c5ededf1d`。完整复现报告为 `local/reports/repair_reproducibility_release.json`，SHA-256 `0917405c113f1ab7ce41e78a0e4f72808749d1f18d38cec8d476163f4db5cc4a`。

## 保护状态、语义变化与未解决项

- r5/r6 均记录读取终点为 2024-11-01、输入稳定、保护 ledger absent、`consumed=false`。没有创建 `local/manifests/protected_access.json`。
- H1–H4、保护集分数和正式最终训练：未执行。它们只在候选冻结且确需最终评分后，经显式生命周期账本统一执行。
- operation `clock`、burden `cal_time`、history/target `tap_end_time` 的“可用时刻”仍是 `ASSUMED` 操作口径，不是完整报告可用性的已验证事实。小时窗口边界、变料报告延迟、目标报告延迟仍待官方确认。
- GitHub 上修复后的 CI、PR 必需检查和 main 分支保护：未执行/未配置，不写成本地 G0 证据。
- 真实数据、模型、预测、报告、账本和提交包均只在获授权本地且被 Git 忽略；仓库公开不代表获准外发。
