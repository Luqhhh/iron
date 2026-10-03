# 原连续十轮SWA A20平台机制对照包

接续用户2026-10-03要求，交付此前第三优先级：与已测周期SWA不同的原连续十轮参数平均模型。原四seed负seed/LCB未过正式门及旧决定保持，本次仅用户平台机制探索。本地两开发seed均值+0.0044435780486不预测平台；Q75=96.3920、周期A20=96.3899及队友EMA32=96.3857均用户报告未独立核验。

原V12 joint periodic TabM随机初始化、两目标标准化MSE、AdamW恒LR .001、max240/patience25/min_delta1e-5、joint标准化MAE内层选轮与freshrefit保持。最近至多10个完整epoch的原参数等权平均，初始化不计，梯度用原参数。实际消耗1全量wrapper、selection+freshrefit共2保存state/2optimizer，零新CV/确认/参照拟合。

内层选择epoch=80，全2754行freshrefit同轮数；实际保存平均窗口epochs=[71, 72, 73, 74, 75, 76, 77, 78, 79, 80]，真实平均10个状态。没有强制续训或改写旧配方。

独立新进程禁止拟合与optimizer，核验训练/校准重复组隔离、scaler/词表、真实patience/平局先者/选轮与freshrefit、float64独立窗口重算。另一个新进程禁止训练标签读取，322行完整/逆序/chunk37冷推理通过。实际退出码0，数值和Torch线程1，RSS≤1024MiB，无时间预算。

冷预测最大差6.452284964097998e-06，窗口最大差4.76837158203125e-07；新进程最终包回读算术差0、冷端点差0，无新拟合或模型推理。私有发布程序必要回归神经6/锁定6通过，测试optimizer0，科学源未变，不重复旧全套测试。

时长=.8父Q75+.2新成员；铁量逐行复制Q75原CSV字段字符串，零差异。ZIP仅result.csv，sample_id/pred_tap_iron/pred_tap_time_len，322唯一ID按官方模板顺序，CRC通过，有限非负；无临时裁剪。原父ZIP SHA锚定，OOF不替代322行父提交。用户自行上传，助手不上传，不写桌面。

提交包：local/runs/swa-window10-a20-release-20261003/SWA_WINDOW10_A20/submission.zip
ZIP SHA256：1f3d85cc74cc57bf348d34680449777894c27d776ccdc8ec8a9984854bf048f0
CSV SHA256：b8739308d6d0ab2446cfbbec0ba0b80dabbfc818b9861cf1488de0a6944efffc
父ZIP SHA256：41a046d5ce74e8a9c7c9acb124fa680cabf9e87e4a79edb50d81efb625bde825

全部模型/预测/标签/账本/ZIP私有local；公开仅策略与交付汇总，原证据与失败保持。
