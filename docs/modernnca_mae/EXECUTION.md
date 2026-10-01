# MAE开发批次实际启动

2026-10-02，持续授权、冻结374份源码/协议文件、原Q75及完整MSE对照、锁定Python3.12/CPU环境和独占新输出目录均通过启动核验。controller PID254023，supervisor PID254022；这是启动时身份，不是持续运行或终态证明。实时状态见`modernnca_mae_q75_development_20261002`。

MAE全尺寸人工反向/推理资源探针峰值614.140625MiB，低于1536MiB；0官方数据读取、0optimizer，不能替代完整训练峰值。42项定向测试通过。完整向量的独立标量准备检查差3.469446951953614e-16。

按[冻结协议](PREREGISTRATION.md)执行20配对/80状态/40optimizer，split seed42/3407各完整五折，固定融合权重0.2。原MSE完整同分折缓存作为额外损失对照，未改写原分类或开发门。0确认seed、0全量发布/包/桌面写入/助手上传，无时间预算或自动重试。观察间隔600秒，实际完成立即审计。

清单SHA256：`bda079bd9113ee934b67e7fe059bbf8a98e39ab67924ac0772834997784f8158`。私有准备目录`local/runs/modernnca-mae-readiness-20261002/preparation-r1`；执行目录`local/runs/modernnca-mae-development-20261002/development-r1`，实际终态及观察在同级`launch-r1`。G0完整冷审计/独立评分与G1开发质量尚待实际完成，不以启动准入替代。

模型修改在独立worktree提交78a965890a71ee08b01ce22f539d7f048d320a99，该分支未配置upstream。[源码与定向测试补丁](source-change.patch)在主工作分支公开供审阅；主分支正常推送其已配置upstream，不为独立源分支临时设置发布目标。
