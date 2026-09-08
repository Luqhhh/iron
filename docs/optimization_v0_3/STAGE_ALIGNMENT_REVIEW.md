# OPT-07A 勘误

旧 H1 与 test_a 相关性分析保留为历史记录，不能继续用它认定统一的开发偏乐观。
同一自然月截止点下：

|名义 cutoff|December test_a|January test_b|February/March test_c|
|---|---|---|---|
|2024-11-01|H2|H3|H4/H5|
|2024-12-01|H1|H2|H3/H4|

实际发布必须用每组件 fit/history/label cutoff 与真实 reference_time 生成
stage_alignment.json，保留 elapsed days 和未覆盖格子。H5 不裁剪成 H4。
December cutoff 是合法全量训练后的名义示例，不是已完成训练或已核验的最终 cutoff。

用户提供的公开六位小数 H2 对照换算近似分数：E00 81.8322、E02 82.4682、
E09 82.7370、E12 82.9552、E16 83.1159。它们不构成新实验；E12 接近平台
82.9543 不证明网格能精确预测平台，更不允许据此反推标签。
