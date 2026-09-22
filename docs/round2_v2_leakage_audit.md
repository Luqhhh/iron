# V2 泄露与异常审计

用户于 2026-09-22 要求检查训练/测试泄露及异常。仅访问 V2 公开训练标签和测试输入，不读取初赛标签，不读取或推断官方测试真值。

固定检查：文件与模型 SHA 身份、独立 ID 对齐、训练/测试 ID 及匿名后缀交叉、重复特征向量、训练 IQR 标准化 Chebyshev 最近邻（阈值 0/0.01/0.05）、空值与常数、模板标签、单特征 Pearson/Spearman 和精确线性变换、21 特征 KS 分布比较（Bonferroni 0.05/21）、行序/ID 关联、已保存 C2 实际特征名单和训练/验证误差。

字典将 pig 描述为“铁量”，但没有采集时间或因果可用性信息，因此预先固定两项诊断：种子 42 原五折，每折训练标签在铁口内成对打乱（两个目标同时置换），以及训练/验证同时把 pig 设为常数 0。保持 C2 所有参数和其余特征不变，共 20 次诊断拟合。模型与 OOF 保存本地，序列化回读及乱序推理核验。单次置换不是正式显著性检验，不据此挑选新模型或替换提交包。

```bash
UV_CACHE_DIR=/tmp/iron-uv-cache uv run --locked --python 3.12 --extra dev --extra round2 pytest tests/test_round2_v2_leakage_audit.py tests/test_round2_v2_release.py tests/test_round2_v2_compare.py -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 UV_CACHE_DIR=/tmp/iron-uv-cache uv run --locked --python 3.12 --extra round2 python -u -m bf_tap_r2.v2_leakage_audit --output local/runs/round2-v2/leakage-audit-r1
```

检查只能识别可见输入与本地训练流程中的问题。没有官方生成器、原始生产关联键、预测时点、测试真值及后台切分记录，无法证明因果可用性、生成器独立性或排除全部非线性标签代理。
