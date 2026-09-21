某钢铁企业高炉铁次预测
复赛数据集指南

一、请先阅读

请不要把初赛代码中的 test_a_samples.csv、test_b_samples.csv、test_c_samples.csv、operation_hourly.csv、burden_change.csv 或 tap_history_train.csv 继续接入复赛建模流程。复赛公开包已经将可用特征整理为样本级特征快照，选手不需要再做时间连接。

二、初赛和复赛的主要变化

初赛数据：
1. 样本主表包含 sample_id、tap_no、spout_no、reference_time。
2. 需要根据 reference_time 对 clock 和 cal_time 做时序对齐。
3. operation_hourly.csv 和 burden_change.csv 是全量时间表，选手需要自行筛选可用记录。
4. 历史铁次实绩通过 tap_history_train.csv 构造历史特征。

复赛数据：
1. 样本编号使用新的 R2S_* 编号，不与初赛 BF4_* 编号对应。
2. 不再提供 reference_time、clock、cal_time 和 tap_no。
3. 不再提供全量时间序列辅助表。
4. train_features.csv 和 test_features.csv 已经是一行一个样本的特征快照。
5. 训练集标签仍在 train/train_samples.csv 中，测试集真实标签不公开。
6. 复赛数据为合成数据，不能解释为新的真实生产批次。

三、复赛目录和文件

train/train_samples.csv
训练样本主表。字段为：
sample_id：训练样本编号。
spout_no：铁口编号，可作为类别特征使用。
tap_iron：训练标签，预测目标一，单位为吨。
tap_time_len：训练标签，预测目标二，单位为分钟。

train/train_features.csv
训练特征表。一行对应一个 sample_id，包含 21 个数值特征。它与 train_samples.csv 通过 sample_id 一对一连接。

test/test_samples.csv
测试样本索引表。一行对应一个测试 sample_id，并提供 spout_no。共 322 条测试样本。

test/test_features.csv
测试特征表。一行对应一个测试 sample_id，字段和 train_features.csv 的特征字段一致。它与 test_samples.csv 通过 sample_id 一对一连接。

test/result_template.csv
结果模板。提交文件必须命名为 result.csv，字段顺序必须为：
sample_id,pred_tap_iron,pred_tap_time_len

四、特征字段

复赛特征字段共 21 个：
air_volume、cold_air_press、hot_air_press、oxygen、hot_air_temp、coal_rate、humidity、gas_rate、furnace_top_press、upper_press_diff、lower_press_diff、total_press_diff、air_press_ratio、furnace_top_temp_avg、air_speed、furnace_throat_temp、pig、all_quality、consumption、fuel_rate、coke_rate。

字段的中文含义、工程单位和数据类型请以 data_dictionary.xlsx 为准。所有特征已经直接绑定到样本，不需要使用时间字段进行二次筛选。

五、提交前检查

1. 结果文件名必须是 result.csv。
2. 必须提交 322 行数据，不含表头时为 322 行。
3. sample_id 必须与 test/test_samples.csv 完全一致，一行不多一行不少。
4. sample_id 不能重复，不能继续使用 BF4_* 编号。
5. 结果字段必须为 sample_id、pred_tap_iron、pred_tap_time_len。
6. 两个预测字段必须是数值，不能包含空值、文字或无穷值。
7. 预测结果建议按照 test_samples.csv 中的 sample_id 顺序输出。

六、关于数据穿越

复赛不再提供 reference_time、clock、cal_time 和全量辅助表，因此不需要进行初赛中的未来数据过滤。test_features.csv 中的每一行已经是该样本可直接使用的合成特征快照。

请勿把初赛公开包中的时间表、初赛测试样本行或初赛样本编号与本复赛数据进行拼接。复赛评分只认本包中的 R2S_* sample_id。

七、常见错误

错误 1：继续读取旧版 test_b_samples.csv。
处理：改为读取 test/test_samples.csv 和 test/test_features.csv。

错误 2：把 test_samples.csv 和 test_features.csv 按行号连接。
处理：始终使用 sample_id 连接，并检查一对一关系。

错误 3：提交旧版 BF4_* sample_id。
处理：直接使用 test/test_samples.csv 中的 R2S_TEST_* sample_id。

错误 4：把 tap_iron 和 tap_time_len 放进测试输入。
处理：这两个字段只在训练集提供，测试集只能输出预测值。

错误 5：把 spout_no 当成时间字段。
处理：spout_no 是铁口类别字段，不承担时间截断功能。

八、最终口径

复赛的最小可用流程是：读取 train/train_samples.csv 和 train/train_features.csv，按 sample_id 合并训练；读取 test/test_samples.csv 和 test/test_features.csv，按 sample_id 合并预测；按照 result_template.csv 的字段生成 result.csv。初赛的时间对齐和全量辅助表处理流程在复赛中不再适用。
