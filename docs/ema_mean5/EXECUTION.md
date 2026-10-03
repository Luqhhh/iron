# 五成员扩展执行登记

2026-10-03。按[固定协议](PREREGISTRATION.md)完成[工程准入](ENGINEERING.md)后，两个完整开发split已串行启动。源码提交`0655dbc`，1,482项冻结文件，manifest SHA256为`75b605e9d36883b7a118df7b04f03eec3b538bceb6c5968ca7f4df4ed11e5f11`；准备进程实际exit0。

启动观察确认controller PID14963及seed42/fold0/init3042拟合子进程PID14966当时存活，四项数值线程环境均1，可用内存约12.6GiB。PID仅为启动快照，实际后续状态以私有执行事件和进程为准。首个复用子进程已实际exit0：同外层三个原EMA估计器、六个selector/refit状态的新进程训练分区、选轮、预处理与冷推理核验通过，峰值493.90MiB；未读取新候选局部质量。

私有目录`local/runs/ema-mean5-20261003/development-r1/`。剩余范围仍是完整20个新模型/40次optimizer、全100个新旧冷状态以及两个完整OOF和独立终态审计。G0当前只闭合启动与首个复用单元，G1未测。52个子进程由实际退出事件串行监督，按600秒观察；0自动重试、无时间预算。原最佳mean3=96.3954与平台排程保持，尚无新增待测包。

后续实际完成事件：首个新增模型seed42/fold0/init3042及其独立冷审计均exit0，原生2次optimizer与selector/refit step对账、1,482项冻结依赖未变；逐项证据在`execution/first-new-estimator-audit.json`。这只是一个模型的G0，未读取候选局部质量。等待期间另完成[条件确认缓存清点](CONFIRMATION_CACHE.md)，仅核对原哈希/收据，0新确认拟合，不改变当前科学范围。
