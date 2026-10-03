# Huber SWA A20 提交包交付

2026-10-03，用户明确授权生成。唯一提交文件：local/runs/swa-time-huber-a20-release-20261003/SWA_TIME_HUBER_A20/submission.zip。供用户自行上传，助手未上传，未写桌面。

固定时长=.8Q75+.2Huber fullfit成员；铁量直接保留Q75父包CSV字段原字符串，0差异。父包SHA256：41a046d5ce74e8a9c7c9acb124fa680cabf9e87e4a79edb50d81efb625bde825。Huber2 delta1 loss/.5ironMSE+.5timeHuber2 per head、原V12jointperiodicTabM/恒LR.001/max240/patience25/jointstdMAE选轮/freshrefit/last10完整epochraw均值均保持原冻结配方。独立审计核对模型mechanisms、每epoch损失身份及LR、训练/校准分区、scalers/词表、选轮和窗口。

新增唯一全量wrapper/selection+freshrefit共2states2optimizers，0新CV/确认/参照fit，0旧模型重复训练。内层选择80epoch，freshrefit80epoch，窗口连续71..80。两个环境6项包回归均通过，0工程optimizer。进程实际退出码0。训练峰值RSS657.80078125MiB，低于1024MiB；独立标签隔离新进程完整/逆序/chunk37 cold最大差6.452284964098e-06，窗口float64均值最大差4.76837158203125e-07。

最终独立零fit/零模型推理回读通过：322唯一sample_id、官方模板顺序、三列sample_id/pred_tap_iron/pred_tap_time_len、ZIP仅result.csv、CRC正确、finite nonnegative、iron原字符串0差异、time A20算术最大差0。不使用临时裁剪，所有模型/预测/标签/ledger/ZIP私有local，不入Git。

ZIP SHA256：83df29e84ea08d136c8ba589d22b0baf3d47f48ded09555ee603a5799b97179b。
result.csv SHA256：2284d08cb233433c183b2ac8ace7462bb8fe3b3c95655011aef21a4b7ea83233。
preflight SHA256：a920e3ad9f91773766fc70d15fe8c401392de9c73bcb4c064ef9897ebc243d19。

此前四seed本地平均+0.003683563400696，LCB95+0.001206941572002，机制vs原SWAmean+0.001163007530851；本包没有新CV成绩。Q75平台96.3920为用户回传未经独立核验，本轮平台未测，不推算平台分数。正式本地门通过不等于平台已提升。完整交付结果推送后删除定时检查，不自行上传或另起训练。

## 平台反馈（2026-10-03）

用户回传本包96.3882，未经独立核验；相对Q75用户报96.3920下降0.0038，未达到96.4。确切ZIP SHA83df29e84ea08d136c8ba589d22b0baf3d47f48ded09555ee603a5799b97179b。原四seed本地mean+0.003683563400696、LCB95+0.001206941572002及正式门通过结论保留；这次是本地正/平台负实例，不追溯改实验门槛，不推算误筛概率或固定偏移。降低同A20配方平台优先级，不自动扫描权重或重复封包；不能推广为所有Huber或SWA均无效。当前登记Q75参照保持。
