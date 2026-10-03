# EMA嵌套残差完整开发结果

2026-10-03。原[冻结协议](PREREGISTRATION.md)的两个split42/3407、各五fold全部完成。冻结参照仍为Q75=96.3920；运行期间用户回传的新最佳EMA_MEAN3=96.3954不回写本阶段。

| 固定修正器 | split42总分增量 | split3407总分增量 | 原阶段决定 |
| --- | ---: | ---: | --- |
| RIDGE | +0.000290578 | +0.000422265 | 两切分为正，取得另行冻结确认的准入；未正式晋级 |
| GBM | −0.003746747 | −0.005939523 | 两切分均负，本固定配方不确认 |

G0通过：监督执行会话真实exit0，121个子进程全部exit0；40个新基础估计器、80次AdamW构造与79,788次原生step、120个新旧基础状态冷审计、20个头拟合全部对账。577项冻结文件未变，独立进程将评分标签绑定到冻结训练分区、重新拼接每个split的2754行完整OOF，两目标独立标量评分最大差1.04e−14。峰值子进程RSS854.05MiB，122个自有监督/子进程PID在最终检查中均已不存在。0确认seed、全量拟合、新包、桌面写入或助手上传。

G1单独报告：RIDGE原阶段平均增量+0.000356421，现行candidate_tiers为exploration，4/10个fold改善，未达到minimum_improved_folds；GBM为not_shortlisted。两切分准入不等于正式晋级、自动确认拟合或平台排程。原头的训练输入来自T内部EMA交叉拟合，部署应用于完整T模型，训练规模迁移限制保持。本结果不关闭整个残差或树模型家族。

私有证据在`local/runs/ema-nested-residual-20261003/development-r1/`：原`report.json`、`execution/terminal.json`、`execution/independent-terminal-audit.json`和`execution/final-reconciliation.json`。原manifest SHA256为`9bb704d31630756054dc4580a45dcfe6b1c23d253660d6062e867196370a84b2`，报告为`09d7de147b44720c309e376caaf9fd078b82d8a2c289d23e6330165dadb025a0`。额外事件监听器因Python缺少pidfd接口未启动，失败证据保留；它没有触发审计、训练或重试，最终审计由原执行会话的实际成功完成事件触发。

当前参照的适用性已在另行预登记、零拟合的[EMA_MEAN3衔接结果](MEAN3_FOLLOWUP_RESULTS.md)中核对。小幅修正保留为研究候选，本阶段不占平台名额。
