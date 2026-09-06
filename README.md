# bf-tap-predict

高炉铁次预测的防泄漏 baseline 实现。设计依据位于 `md/IMPLEMENTATION_PLAN.md`，冻结决策位于 `md/decisions/`。

当前实现阶段：M0–M4。赛事发布表只有一套业务时间字段，项目已冻结可复现的 `competition-timestamp-contract-v1` 映射；真实 DEV 验证使用该映射，11 月标签仍由独立保护门隔离。

## 本地开发

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pip install -e . --no-deps
.venv/bin/pytest
.venv/bin/python -m bf_tap --help
```

真实数据、模型、运行报告和提交文件均由 `.gitignore` 排除。示例配置在 `configs/data.example.yaml`，真实本地映射使用不受 Git 跟踪的 `configs/data.local.yaml`。
