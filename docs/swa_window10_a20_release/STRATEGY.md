# 原连续十轮SWA A20平台机制对照

用户2026-10-03接续此前第三优先级。固定原TABM_TIME_SWA_V1，1全量wrapper/selection+refit共2states2optimizer，零新CV/参照拟合。原训练器与历史决定不变。恒LR.001，joint标准化MSE，最后至多10完整epoch参数等权平均，max240/patience25，joint标准化MAE内层选轮，freshrefit至选择epoch。

时长=.8Q75+.2新连续窗口成员，铁量原CSV字段字符串；训练seed42/inner42，单worker/数值Torch线程1/RSS1024MiB，无时间预算。原四seed未过正式门，作为用户平台机制探索，不预测平台。Q75=96.3920、周期A20=96.3899、队友EMA32=96.3857均用户报告未独立核验。

独立新进程核验分区/预处理/选轮/窗口，标签隔离完整逆序chunk37冷推理；ZIP只result.csv，322唯一sample_id/官方模板顺序/有限非负/CRC/铁量原字符串与A20算术。私有模型预测证据ZIP留local，用户自行上传，不桌面、不助手上传、不重试。当前登记未训练，只有占位和完整交付push。
