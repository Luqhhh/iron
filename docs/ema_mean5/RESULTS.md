# EMA五成员完整开发结果

2026-10-04闭合。当前参照为用户回传、未独立平台核验的EMA_MEAN3_FULL_Q75=96.3954。固定五个训练seed42/1042/2042/3042/4042等权、只新增后两个成员；原科学协议未变。

G1：split42相对mean3总分增量−0.003917277，split3407为−0.000997846，平均−0.002457561；仅1/10折正。两个完整五fold切分各2754行，均未改善。candidate_tiers为not_shortlisted，both_splits_improve和minimum_improved_folds失败；未获得新确认准入，未正式晋级。停止该固定五成员配方的确认和全量发布，保留全部失败证据；不据此否定三成员已测平台正例或整个集成家族，也不挑选额外成员重新包装。

G0通过：20个新估计器、40次AdamW构造、45,964次原生step；40个新selector/refit状态与60个旧状态在独立进程完成冷核验。52个子程序及原始监督会话实际exit0。独立进程从冻结训练T核对标签与ID、检查原生模型轨迹，再以math.fsum重建五成员和双目标得分，最大差5.68e−14。1,482份冻结文件不变，53个所属PID全部退出；峰值RSS812.02MiB，小于1536MiB门。锁定Python3.12、单数值线程、七项非拟合检查及600秒观察规则均保持。

0新确认seed、全量拟合、提交包、桌面写入或助手上传。此前确认缓存清点和全量三成员冷回放只是条件准备，未激活。本阶段没有更改当前平台最佳、分配新名额或推断剩余额度。

完整私有证据：local/runs/ema-mean5-20261003/development-r1，包含report.json、independent-audit.json和execution/final-reconciliation.json；原始监督会话49397实际exit0，所有报告/清单/终态摘要已登记EVIDENCE_STATUS.json。
