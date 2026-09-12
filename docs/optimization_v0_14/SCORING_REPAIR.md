# 原 v0.8 U1 证据名映射

独立冷审计全部通过、fit 尝试和预测差为 0 后，全网格 scorer 已成功保存 metrics/summary/各 unit errors。随后的旧 J 重建因为原 v0.8 以 U1 而非 V1 命名发布对照而 KeyError。engineering_repair/failure_r1.json 保留该失败。它不影响已计算的分数，也未改变系数或预测。

本次只增加显式 original U1 -> current V1 汇总身份映射、该旧 namespace 的合成测试、恢复已存分数的入口。原工程修复三个源码按其修复 manifest 摘要先归档于 engineering_scoring_repair/original_sources/。新的修复 manifest 绑定原注册 SHA、第一修复 manifest SHA、原/新源码和全部已存评分/冷审计身份；原 manifest/第一修复/失败/评分目录均不覆盖。

```bash
uv run --locked --python 3.12 python scripts/optimization_v14_h2_calibration.py \
  --output local/runs/optimization-v0.14-opt30-r1 --resume-scoring
```

此入口验证已完成冷审计和评分摘要，从冻结原分数继续 bootstrap、门槛及诊断。不再调用 scorer，不重估系数，不重训、不重跑冷模型；追加 fit 0，既有预测/分数数值不变。源码身份变化逐层保留原件，其他模型/数据/配置/预算/门槛仍严格按原摘要验证。
