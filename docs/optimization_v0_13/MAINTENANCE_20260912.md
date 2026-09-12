# 2026-09-12 维护观察

本记录补充已观察到的公开发布与通知状态，不改写 2026-09-11 的 v0.12 执行报告、旧 FAIL、模型/配置身份或已绑定完成回执。

## 提交、推送与 CI 的不同时间

| 事件 | 实际证据 | 时间 |
| --- | --- | --- |
| v0.12 最终代码提交 | `62c62cc92efa660674877254c6ae8dc2bceaba83` 的 author/committer time | 2026-09-11 23:17:18+08:00 |
| v0.12 推送 | 先前用户明确要求“提交推送”，Git push 回执创建远端分支 | 2026-09-12；无独立 per-branch push 精确事件时间 |
| 仓库 pushed_at | GitHub 元数据（仓库级字段） | 2026-09-12 09:20:06 UTC，即 17:20:06+08:00 |
| push workflow 创建/开始 | locked-tests run `34685572032`，event=push，HEAD=62c62cc | 2026-09-12 09:20:08 UTC |
| workflow 状态更新时间 | completed/success | 2026-09-12 09:21:40 UTC（状态更新时间，不冒充单独完成事件） |
| 本轮查询 | 本机只读 API 与 Git fetch | 精确查询证据在 local/reports/optimization-v0.13-maintenance-observation-r1/ |

本轮重新 fetch 核对 optimization-v0.12 仍为 62c62cc。旧 README 的未推送、无本阶段 CI 已过时，当前维护摘要改为上表观察；旧 v12 RESULTS 中的“本阶段未推送”继续保留其执行时口径。v0.13 未推送，不能声称本阶段有远端 CI。

## 数据与平台边界

GitHub 仓库查询仍为 public，数据历史处置未完成。本轮只在获授权本地环境工作，私下另存包含 28 refs 的完整 Git bundle 并核验，不修改 visibility，不重写历史，不销毁原证据，不追加公开内容。

本轮自行检索提交回执：查阅 docs/submission_log.md、原平台反馈 JSON、active_release、local 报告/提交/账本，以及 Desktop/Downloads 相关文件名；查询公开网页和报名系统。检索到 V1 用户回传 83.0319 与原 ZIP SHA，但未找到 submission_id、上传时间或有效提交回执。报名系统 https://reg.aicomp.cn/ 返回动态 JavaScript 应用，账户接口需要 bearer 授权；当前无该平台登录连接器或可用浏览器调试会话。**platform_verified=false**，不能因此声明复赛资格已确认。检索范围和访问结果保存在独立 local JSON，不要求用户反推或补造信息。

原 V1 包摘要 `fdcbe03e8ea31577bfeed0c45cd0a0013bda42ab88d29db703fad9c2f7e557aa` 与 R2 回退保留；本轮核验原包，不生成替代包，不覆盖桌面，不上传。

## 官方新通知

[主办方 2026-09-11 复赛通知](https://www.aicomp.cn/notice/notice-3/5248.html) 已于本轮重新访问：初赛仅用于复赛资格，不计入后续阶段；复赛每天最多 5 次，取最高成绩，存在算分延迟；高炉复赛基准线 80。该说明不改变初赛口径，也不等于当前 V1 复赛达标或获得资格。

| 节点 | 官网列示时间（2026 年） |
| --- | --- |
| 报名截止 | 09-20 19:00 |
| 初赛结果截止 | 09-20 20:00 |
| 复赛数据开放 | 09-21 11:00 |
| 复赛结果开始 | 09-22 09:00 |
| 复赛结果截止 | 10-05 20:00 |
| 复赛代码截止 | 10-07 23:59 |

通知未单独标注时区，操作以官方报名系统口径为准。[赛题规则](https://www.aicomp.cn/tracks/tracks-6/4177.html) 的数据使用限制和样本 reference-time 约束仍适用。本轮没有保存赛事逐样本内容到这些公开维护文档。

工程 preview、正式数据 identity、模型 quality 和平台 evidence 分别登记。正式包尚未开放；同名旧 test_b 不代表完成正式复赛验收。复赛接入和后续独立注册要求见 [协议草案](SECOND_ROUND_PROTOCOL.md)。
