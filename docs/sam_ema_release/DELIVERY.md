# SAM＋EMA固定人工探索：交付与暂停

2026-10-02。本轮一个全量程序、两次真实AdamW初始化、两个新原生状态和一个ZIP已完成。**G0通过；G1为平台未测的人工探索，未正式晋级。用户明确“本轮结束后暂停”，本轮交付后暂停优化，不启动下一轮。**

候选 `SAM_EMA_TIME_Q75_EXPLORATION`；本地包 `local/runs/sam-ema-time-exploration-20261002/release-r1/package/Luqhhh_bf_tap_predict_round2.zip`，ZIP SHA256 `407cf7f7ff3902856435134896a728c4a1707cc0943a328954e6a818db29c3d5`。固定公式为 `Q75_time+.75*(full_SAM_EMA_time-old_full_EMA_time)`，铁量字段直接复制Q75父CSV原字符串。没有新权重扫描、CV、确认seed、桌面写入或助手上传。用户自行上传；不推断剩余额度。

[冻结协议](PREREGISTRATION.md)和原训练设置保持。2754行复赛训练，group-safe inner为2203/551行，EMA验证指标选中149轮、174轮停止，fresh全量refit149轮。旧完整EMA不重新拟合；仅使用相应训练分区的预处理及目标尺度。

原生selector/refit、真实optimizer账本、同训练/query身份、独立新进程冷见证和独立包算术均通过；full-batch冷差0，逆序/37行分块最大差6.4959634e−6，低于5e−4。包含322个官方模板顺序的唯一ID、有限非负预测，ZIP仅result.csv、CRC正确；独立算术差0、铁量字段字符串变化0。训练峰值RSS736.254MiB，低于1536MiB；controller与supervisor实际exit0。396份冻结源、567份输入和401份模型来源映射重新哈希一致，所登记进程已退出。

锁定Python3.12的62项定向检查通过、0跳过，未运行全库检查。终态归档曾额外误拒绝两份历史切分表及其五列格式；两次检查失败记录保留，按冻结身份纠正后的零拟合复核实际exit0通过，没有更改源、清单或重跑模型。科学程序及原生/封包审计实际exit0不受这两项多余假设影响。

[完整开发](../sam_ema_time/RESULTS.md)相对Q75的两个切分仍为−.024537241/−.021519181，not_shortlisted及确认失败决定保持；相对匹配SAM均改善。原SAM/EMA平台正方向与这项匹配改善为人工交互探索提供理由，不能将旧平台收益相加或预测新成绩。没有新增四seed正式晋级。

公开状态见EVIDENCE_STATUS.json的 `sam_ema_time_manual_exploration_release_20261002`。原N→V36探针保留未测；DE3＋Q75仍暂不平台测试。最佳仍为用户回传、未独立核验的Q75/Q100=96.3920，距96.45为0.0580，目标未达到。暂停后不空转监控。

发布源提交 `93922b76b45a73a1721213e68b4d7e3b14558707`；manifest SHA256 `5fb0dc594921639f3b43e31acfea28dac65c4bc5e21a453f932d2fc11aa4305b`；actual-main-exit SHA256 `022265e760cc762bb18bd946b05c41aa48ca62955a6aec88fc9508fa723a9487`。模型、预测、原始包、审计及失败记录保留于私有local目录。
