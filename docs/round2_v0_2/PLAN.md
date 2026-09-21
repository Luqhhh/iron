# 复赛 v0.2 信号审计

用户在首轮70次拟合结束后授权继续。此轮先做零模型拟合诊断，独立于v0.1剩余4次全量拟合额度；不上传平台，不引入初赛任何数据或ID/时间推断，不删除边界标签。

固定配置为`configs/round2_v0_2/signal_audit.yaml`：

1. 核验P0的9个数据文件SHA及上一轮摘要，以标准库CSV按ID字典查表，独立复查pandas连接后的每个特征、铁口和目标。
2. 对21特征×2目标×Pearson/Spearman，在全体及两个铁口分别检查。每个范围内以999次相同的目标联合行置换保留双目标依赖，取当次全部84个绝对相关系数的最大值作零假设参照。三种范围再作Bonferroni校正。
3. 在读出结果前固定7个数值构造：冷风减热风压力、上减下压差、风量乘氧量，以及铁口2指示量分别乘风量/氧量/全压差/pig。按单独探索性检验族报告；不解释为守恒关系，不和原始特征族混为同一个校正结果。
4. 复核原始目标与OOF标签一致，报告OOF预测离散度、目标相关、绝对误差差异及边界/内部样本误差贡献。两个真实目标之间的关系仅作描述，禁止把真实铁量作为测试时长模型输入。

置换概率用`(1 + 更极端次数)/(1 + 置换次数)`，方法背景见[SciPy官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html)。诊断依赖范围内可交换性假设；没有显著相关不等于证明无预测信号。全量训练数据上的诊断属于探索，不是严格OOF评价。若日后据此选特征，必须在训练折内重做选择，并明确额外的开发适应性。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --locked --python 3.12 --extra round2 \
  python -m bf_tap_r2.signal_audit --output local/runs/round2-v0.2/signal-audit-r1
```

输出目录必须不存在，失败保留FAILED.json，不覆盖已有证据。结果、配置副本和数据身份保存在local/。在审计结论形成前不启动新增交互模型、融合扫描或平台提交。
