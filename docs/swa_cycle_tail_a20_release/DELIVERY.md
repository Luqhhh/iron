# SWA周期尾段A20探索包已交付

用户于2026-10-02明确选择一个SWA系列策略生成平台提交包，本次只交付SWA_CYCLE_TAIL_V1_A20。时长为0.8Q75+0.2周期尾段全量member，Q75铁量直接复制CSV原字段字符串，0额外参照拟合、0新CV/确认。用户自行上传；平台分数待回传。

原四seed结果mean+0.002917856711、seed单侧LCB95+0.000342761271，但271828微负，原未正式晋级决定保持。此次为用户授权探索发布，不能用本地增量预测平台成绩或保证96.4+。Q75=96.3920来自用户报告未经独立核验。

原CycleSWARegressor及V12 joint periodic TabM科学源码/训练参数不变，random/inner seed42；两目标标准化MSE、AdamW、前80轮恒LR0.001及之后20轮余弦0.001→0.0001，周期末100..240稀疏参数窗口。内层选择完整240epoch后选100，freshrefit全部2754行到100；此时只有一个可用周期末raw状态，按原min(5,available)规则平均该状态，没有强行增加周期或声称实际平均了5个状态。

实际消耗1 full wrapper、2保存state/2 optimizer（内层selection+freshrefit），失败预约不可重复。独立新进程零fit审计训练/校准分区、重复组隔离、特征/目标scaler、词表、240轮LR及8个周期末选轮、100轮freshrefit和窗口算术。另一个新进程禁止训练数据与标签cache读取，完成322行完整/逆序/chunk37冷推理。实际退出码0，峰值RSS548.859375MiB≤1024MiB。数值和Torch线程均1，无时间预算。

冷预测最大差 3.226142496259854e-06；窗口差0；最终CSV与独立冷全batch重建端点逐值一致。包仅result.csv，列sample_id,pred_tap_iron,pred_tap_time_len，322唯一ID按官方模板顺序，CRC通过、有限非负、铁量字段字符串零差异；未临时裁剪。两个Python环境必要4项封包/预算测试均通过，未重复旧全套、G0或科学fits。

原Q75 ZIP由用户提供目录，SHA与已登记平台包一致；原OOF共享包不含正式测试预测，因此没有用OOF替代平台父包。模型训练/冷审计先独立完成等待原件，然后仅做父包绑定与融合，没有重复训练。

提交包：local/runs/swa-cycle-tail-a20-release-20261002/SWA_CYCLE_TAIL_V1_A20/submission.zip

ZIP SHA256：f423159bd2bae74999230f7c9556389690be704b9bfcad34bd2550177961ab9a
CSV SHA256：5e0934975c716c1b35fc26b3134df2e9d2d1bd61288b8f7c2fec488b293aa231
Q75父ZIP SHA256：41a046d5ce74e8a9c7c9acb124fa680cabf9e87e4a79edb50d81efb625bde825

独立最终复核零拟合、零新推理，原科学源/runtime/data/control/旧fit账本哈希未变。全部模型、预测、标签、ledger、原父包与最终ZIP留local；不写桌面、不由助手上传。
