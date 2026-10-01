# EMA平均跨度串行开发执行（2026-10-01）

科学源码提交b7167ca08c2f3042bbd26363264b9a5e676f7ce2，最终源码锁定Python3.12完整检查1358通过、0跳过、23个既有警告，原控制20份保存状态冷推理和全部缓存身份通过；详见[工程准入](ENGINEERING.md)。冻结SPEC、网络、选轮、beta候选、.75权重、资源和20估计器/40optimizer范围均未改。

首次uv包装的prepare请求因串行准入把本次uv父进程误判为活动科学任务，在新建manifest及任何拟合前拒绝；原prepare.log保留。改用同一已核验的Python3.12解释器直接入口后准入通过，prepare-direct-r2.log与清单追加保存。这是零科学拟合的启动方式修正，不是科学重试，没有放宽串行检查或改动冻结源。

manifest固定735份科学源/配置/测试依赖、103份直接输入和2份主仓库ignored配置摘要；10套训练/query/inner-fit/inner-validation身份对应原两个完整seed的五折，单worker、四种BLAS线程及torch线程各1，原worker1536MiB和冷预测5e-4门槛保持。清单SHA为5b7b57ae313a3e2409883b74928063abdb22406c25dd88c98111c317445d9269；准入可用内存10473.07MiB。

实际独立观察器PID141630、controller PID141636，启动事件与实际进程存活已核验。观察器每600秒读取单位完成/失败事件，controller真实退出立即记录process-terminal.json；没有时间预算、自动重试或对其他会话运行的操作。其输出由持久执行handle保留，具体handle见状态文件。

批次已于2026-10-01 14:19:42 Asia/Shanghai实际正常退出，controller exit0；20单位和40optimizer全部闭合，无失败。原controller在完整OOF汇总后执行独立新进程审计，40份新状态和20份原控制通过；随后单独复核真实终态、全部账本和冻结身份。完整结果及证据摘要见[RESULTS.md](RESULTS.md)。期间没有读取部分fold结果进行选择。新增确认seed、全2754行拟合、包、桌面写入和助手上传均0。

私有根目录local/runs/ema-average-span-20261001/development-r1保留manifest、activation、各单位状态/预测/metadata及events.jsonl，已追加summary、独立audit、completion、process-terminal、终态复核和结束后读取收据。模型、预测、账本、收据及日志均不入Git。SHORT_SPAN两个完整seed均正，按冻结规则选为确认准备对象；自动分类仍为exploration，没有四seed正式晋级或发布许可。当前平台代表仍为用户回传Q75=96.3920，目标96.45尚未达到；待回传EMA铁量包及DE3替补安排保持。controller和observer均已退出，无运行任务时不继续空转观察。
