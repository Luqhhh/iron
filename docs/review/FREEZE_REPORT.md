# baseline-v0.1 工程冻结报告

日期：2026-09-06。起始 main：`c8cb0468800db949e69e2eaea445a92e85217f00`。冻结标签：`baseline-v0.1-reproducible`。

结论：两个剩余 P1 已完成，模型输出没有变化。G0 保持 `PASS_LOCAL_LOCKED_ENVIRONMENT`，G1 保持 `FAIL_DEV_LONG`，发布口径保持 `BASELINE_REPRODUCIBLE_QUALITY_FAILED`。

P1 代码提交为 `e24be809b55e2474ae6e82b273c7a4d8ddce1122`；对应 GitHub Actions run `34038910336` 已完成并成功。冻结标签指向包含本报告和机器状态的最终冻结提交。

## P1-1：公共推理 source 强绑定

bundle 升级为 schema v3，新增独立 `inference_source_contract`，只保存公共 process source 的 SHA-256、字节数、semantic contract SHA-256 和派生 contract ID，不保存本机路径。

当前真实 source contract：`public-process-sources-v1-2773c0f38f89ce32`。

- operation_hourly：SHA-256 `9884e95d986f0f4a9d76df47f74c780a518578ccf622c1c0c49132350f31fd0b`，1,492,853 bytes。
- burden_change：SHA-256 `32113bc49b2ed7c59b3ea876984a5c93db421b8a5e84e88021ca63e04d416c04`，243,066 bytes。

predict 在构建特征前比较实际 source 与 bundle 契约。合成文件级 E2E 已验证：路径改变但内容相同可以通过；同 schema CSV 内容改变会以 `inference source identity mismatch` 拒绝，并要求显式新 source contract。

## P1-2：完整 frozen config digest

`configs/baseline.yaml` 现在声明：

- baseline contract：`798a44e4fb98a1f83d91f0077c32618117b9c25098562a73e6412e792911639c`
- feature contract：`c2fa86c47b4ef906591530cd7fbb368de14bf48f1b1fb3bc4f1dba6bd51e3191`
- semantic contract：`b54c7add8805094245548ac79ff9087d1d7183d68a79441b27a8a96604ad6a25`

baseline 摘要排除三个摘要声明字段自身以避免自引用，其余 mapping 全部参与 canonical JSON SHA-256。回归测试逐项改变 sample derived、operation 字段、operation/burden stale threshold、history flag、semantic pending 和 baseline cardinality，均被启动校验拒绝。

## 实际验证

锁定测试命令：

```bash
uv sync --locked --extra dev --python 3.12
.venv/bin/pytest -q
```

结果：50 passed，0 failed，0 skipped。JUnit SHA-256 `0ff53c32a43185b0bf2348c73e79f9f7fbf486161b07c937067136e7fe1726af`。

真实 DEV 命令：

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
  --output local/runs/dev-baseline-v0.1-frozen-r1
```

执行 `PASS_EXECUTION`，质量 `FAIL`，walltime 100.30s。run manifest SHA-256 `417f1e97f54d981802ff806818ff837c04623f8548850746af8090342650e3ee`；metrics SHA-256 `1fee1898f694bdb010da107f37addd7571acf666da245273c4134721fe90419d`。两个 fold 相对修复前最后证据 r5 的 raw prediction 最大绝对差均为 `0.0`。

使用该 v3 DEV_SHORT bundle 的两个独立 test_a 推理进程均为 335 行，raw 最大绝对差 `0.0`，结果 CSV 字节一致，结果 SHA-256 均为 `d287bd6e1b096f4e16dfaec4b76b5036aed15b50b1aa4a2f3da3243c5ededf1d`。manifest 位于 `local/predictions/test-a-frozen-p1` 和 `local/predictions/test-a-frozen-p2`。

保护状态：读取终点仍为 2024-11-01；ledger absent；`holdout_consumed=false`。H1–H4 未运行。

## 明确延期的 P2

- synthetic protected lifecycle：未实现完整 holdout_scoring/final_training E2E；不使用真实 11 月标签测试。
- 外部 `release_manifest.json`：未实现；bundle.json 仍是 bundle 内部信任根。
- 正式 candidate clean-tree gate：未实现；development 继续允许 dirty 但记录。正式 release 前应要求 clean commit/tree。
- GitHub main branch protection/required checks：仓库平台治理项，未由本次代码变更配置。

标签之后不再原地修改 baseline-v0.1。后续质量工作进入 optimization-v0.2，首要问题是分析 DEV_LONG 上 CatBoost 弱于 per-spout median 的原因。
