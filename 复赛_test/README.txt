高炉铁次预测：复赛合成数据 V2
数据版本：synthetic_round2_v2

本阶段采用合成数据，不代表新的真实生产批次。训练集与测试集使用一致的输入和预测目标定义。请完整替换旧合成版的数据与结果，不混用 R2S_TEST_*、BF4_* 和本版 R2S2_TEST_* 编号。

文件说明
train/train_samples.csv：2,754 条训练样本，包含 sample_id、spout_no、tap_iron、tap_time_len。
train/train_features.csv：2,754 条特征记录，包含 sample_id 和 21 个数值特征。
test/test_samples.csv：322 条测试样本，包含 sample_id、spout_no。
test/test_features.csv：322 条测试特征，列名与训练特征一致。
test/result_template.csv：已填写 322 个 sample_id 的提交模板；两个预测列为空，必须填入模型预测后才能提交。
ROUND2_MIGRATION_GUIDE.txt：初赛到复赛的迁移与操作说明。
data_dictionary.xlsx / dictionary.json：字段说明及机器可读结构。
baseline.py / requirements_baseline.txt：可运行的参考流程。
manifest.json：完整包文件校验清单；独立训练/测试包分别使用 manifest_train.json、manifest_test.json。

开始训练
在数据包根目录执行：
python3 -m pip install -r requirements_baseline.txt
python3 baseline.py --data-dir . --output result.csv

提交方式
将 result.csv 单独放入 ZIP 根目录上传赛事平台。CSV 必须包含 sample_id、pred_tap_iron、pred_tap_time_len 三列，322 个编号与本版测试集完全一致。预测值必须为非负有限数值，无缺失、无重复。

合成数据的样本行号与匿名编号不表示时间顺序。不再需要初赛的小时炉况、变料和历史实绩表的时间连接；本版任务是在给定样本级特征上预测出铁量（吨）与出铁时长（分钟）。

参考代码用于展示完整工作流程，选手可选择其他模型及特征工程方法。样本和标签已更新，应重新训练；旧结果文件不适用于本版。
