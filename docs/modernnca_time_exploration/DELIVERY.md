# ModernNCA 时长单项平台探索已就绪

2026-10-03。`MODERNNCA_TIME_A20` 已完成全量拟合和独立审计，G0通过。G1是**人工平台探索、未正式晋级、平台未测**；原MSE开发门失败及MAE变体失败记录保持。最佳仍为用户回传、未经独立核验的Q75/Q100=96.3920，96.45尚未达到。

固定配方：`time=.8*Q75_time+.2*ModernNCA_MSE_time`；铁量逐行保留Q75 CSV字段原字符串。没有裁剪、追加后处理或权重搜索。唯一新增ZIP：

`local/runs/modernnca-time-exploration-20261003/release-r1/MODERNNCA_TIME_A20/Luqhhh_bf_tap_predict_round2.zip`

- ZIP SHA256：`dd1b1866eb514fefb026c37ef7239127572a65c75b42877b42f50c5aaa2937c5`
- CSV SHA256：`33ecfba6902b98928d1ee41297840ae0d53e7bce2dad25253e1a88189ddfc065`
- 父包SHA256：`41a046d5ce74e8a9c7c9acb124fa680cabf9e87e4a79edb50d81efb625bde825`

原科学源码来自独立worktree提交`f4b0c9d`，发布执行器冻结于`029d8f7`，1698个源码/输入/证据文件终态重新核对未变。内层2203行训练、551行校准，MAE首次严格最佳选中第11轮，patience25使selector在36轮停止；fresh初始化的2754行refit严格运行11轮。只有两个实际AdamW构造，分别324/121个optimizer step；两估计器/两optimizer开始完成闭合，失败/未完成均0。原配方的240轮上限没有命中。

G0证据：锁定Python3.12的45项定向检查通过；两个保存状态由独立NumPy推理实现核验训练库、预处理、目标尺度、选轮轨迹及fresh轮数。整批冷推理最大差`2.0463630789890885e-12`；逆序/单行最大差`5.684341886080802e-14`，低于原1e-8容差。另一个禁止训练文件读取/初始化的新进程从保存的refit邻居库和无标签查询重算包，算术最大差`4.263256414560601e-13`。CSV与ZIP回读：仅result.csv，CRC通过，322唯一ID官方模板顺序，预测有限非负，铁量原字符串差异0。

训练、原生冷审计、无标签封包三个真实进程exit0，supervisor实际exit0；OS峰值785.414MiB，低于1536MiB。所有自有PID已退出，无监控空转。首次父包地址错误发生在科学运行创建前，0拟合，日志保留于checks-r1；改用同SHA私有父包后checks-r2通过。没有重新执行已消费的科学拟合。

新增计数：1全量训练程序、2估计器、2optimizer、2状态、1包；0CV/确认seed、0桌面写入、0助手上传。平台上传与反馈由用户完成。

该候选本地完整两seed增量仅+0.000669/+0.001317，仍保留原探索分类。此次选择依据是未测的检索机制与完整开发证据，不要求它先达到原0.01幅度门，也不假定历史EMA的平台放大比例会重现。固定六包的无标签比较显示，ModernNCA相对Q75改变量与EMA32/Laplace/EMA_MEAN3的余弦分别0.1541/−0.1169/0.0963；这说明预测变化并非同一方向，**不证明误差独立、正收益或达到96.45**。

私有证据：`release-r1/manifest.json`、`warm.json`、`cold.json`、`package-audit.json`；同级`launch-r1/terminal.json`与`final-reconciliation.json`；无标签有限池预登记与独立标量复核位于`information-review-r1/`。机器状态键为`modernnca_time_manual_exploration_20261003`，当前名额安排见[信息问题](../platform_information_value/ALLOCATION.md)。

2026-10-03后续平台反馈：用户要求写桌面后回传 **96.3707**（未经独立平台核验），绑定上述ZIP及[此次桌面交付](../platform_information_value/DESKTOP_DELIVERY_20261003.md)。相对冻结Q75=96.3920为 **−0.0213**；同时回传的EMA_MEAN3=96.3954成为新最佳。本固定ModernNCA配方从待测队列移除；G0及原开发失败决定不变。预测方向差异没有转化成平台增益，不由此关闭整个检索模型家族。
