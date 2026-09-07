某钢铁企业高炉铁次预测
对外发布包说明

一、目录用途
本目录为面向参赛者的正式发布数据包，不包含任何测试集真实标签或内部评分文件。

二、目录结构
1. train/
   用于训练模型，包含训练样本主表及配套公共特征源表。
2. test/
   用于测试推理，包含初赛/复赛/半决赛测试样本主表、公共特征源表和结果模板。

三、参赛者可见文件
1. train/train_samples.csv：训练样本主表，包含 sample_id、tap_no、spout_no、reference_time、tap_iron、tap_time_len。
2. test/test_a_samples.csv：初赛测试样本主表，不含标签。
3. test/test_b_samples.csv：复赛测试样本主表，不含标签。
4. test/test_c_samples.csv：半决赛测试样本主表，不含标签。
5. operation_hourly.csv、burden_change.csv、tap_history_train.csv：特征构造所需公共源表。
6. test/result_template.csv：结果提交模板。
7. data_dictionary.xlsx：字段说明和使用口径。

四、提交规则
1. 结果文件固定命名为 result.csv。
2. 结果文件字段固定为 sample_id、pred_tap_iron、pred_tap_time_len。
3. 正式评分仅以 sample_id 对齐。
