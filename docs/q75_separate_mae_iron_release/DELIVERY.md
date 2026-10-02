# SEPARATE_MAE_IRON_A20已就绪，平台未测

2026-10-02。持续授权内的全量拟合与封包完成。只更换Q75铁量为`.8*Q75_iron+.2*SEPARATE_MAE_iron`，时长322个字段原字符串不变；未组合Laplace时长、写桌面或上传。

唯一私有ZIP：`local/runs/q75-separate-mae-iron-release-20261002/release-r1/SEPARATE_MAE_IRON_A20/Luqhhh_bf_tap_predict_round2.zip`。

- ZIP SHA256：`a6fc793ae48312a868477b20408534aeaa1ace0e23f62d8138915f847c585539`
- result.csv SHA256：`7f08ee740b261507bd354965f6dd13c37d2f8c7d9127127ffc620610f07f5852`
- 父Q75 ZIP SHA256：`41a046d5ce74e8a9c7c9acb124fa680cabf9e87e4a79edb50d81efb625bde825`

G0通过。20项锁定Python3.12零optimizer检查；科学全量1程序/2optimizer/2状态。inner F=2203行，联合标准化MAE选中117轮、147轮停止，fresh全2754行重训117轮。独立新进程核对两目标模型身份、分区、train-only统计和选轮，原序/反序/chunk37/NumPy前向最大差2.28e−13；禁止训练文件读取的冷预测与保存输出逐位一致。五个进程实际exit0，OS峰值RSS505.969MiB，本批全部PID已退出。

ZIP仅result.csv、CRC通过，322唯一ID逐项匹配官方模板，预测有限非负；时长原字符串差异0。包回读及独立标量A20重算通过。没有权重搜索、裁剪、科学重试、新CV或新工程fit。

G1通过本地四seed正式门及额外配对门，详见[确认结果](../q75_separate_mae_iron/RESULTS.md)。对Q75均值+0.006061154、LCB95+0.003411010；对原单目标MAE均值+0.004274957、LCB95+0.001841159。此候选来自前阶段控制的观察后线索，后续确认另行预登记；不改原共享阶段失败决定。多个切分仍重用同一数据池，平台效果未知。

科学worktree commit `4c31ea9`；release manifest SHA256 `eebfe8762725273fe01a0af8858d68797e28a150b8a95013a8c399aa16e7a0f8`，package-audit `fc030ecd5920dc8b6bddf3643ba6a377924f4f5ebaa86728705ed0a5ce095368`。证据、模型和包只留local。

两个平台名额按用户意见继续保留；本包作为铁量方向参与[分配比较](../platform_slot_review/ALLOCATION.md)。当前平台最佳仍为用户回传96.3920，96.45尚未达到。
