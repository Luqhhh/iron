# V2.7 求解器资源补齐

nonlinear-r1 在第16次拟合（split42/fold3/KS1时长）触及SVR max_iter=100000，fit_status=1并产生收敛警告。按冻结规则停止；原15个已收敛模型、账本、FAILED及配置全部保留，不使用失败模型评价。

仅将剩余KS1求解器资源上限提高至1000000，C=10、epsilon=.01、gamma=scale、tol=1e-5、目标变换和切分不变。已收敛且迭代数低于原上限的15个模型复用，不能将其改写成用新上限重训。新增测试确保仅允许此非绑定资源上限差异，不允许放宽tol、改变C或复用未收敛模型。

新目录 nonlinear-completion-r1 保留旧源码快照，绑定原目录所有文件摘要，复制已完成模型并补齐25个任务。完整阶段应有40个收敛模型、41次真实拟合尝试（含原失败1次）；本次恢复新增25次，重复保存的旧15个账本完成事件不计为新训练。独立进程回读全部40个模型之后才接受结果。

执行：`python -m bf_tap_r2.v2_nonlinear complete-fits --source local/runs/round2-v2.7/nonlinear-r1 --output local/runs/round2-v2.7/nonlinear-completion-r1`。不覆盖旧目录，也不通过降低收敛门槛来产生探索候选。
