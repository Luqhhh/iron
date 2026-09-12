# 2026-09-12 / v0.14 维护观察

本轮 `git fetch origin optimization-v0.13` 得到 2db6d5faae9c8568e01f5d75d145c17fc8404fb8，与执行起点一致。GitHub API 再次核实 [locked-tests run 34687452551](https://github.com/Luqhhh/iron/actions/runs/34687452551) 绑定该 SHA，completed/success；run created_at 为 2026-09-12T10:03:44Z，updated_at 为 10:05:26Z。提交身份、远端观察、CI 时间与本轮查询时间分别保存在 local/runs/optimization-v0.14-opt30-r1/maintenance/，不从 CI 时间推测精确 push 时间。

因此，当前 README/文档索引中 v0.13 的“仅本地、未推送”摘要需要更新。v0.13 冻结结果、维护记录、原 manifest 和完成回执保持原文与原哈希；其中未推送描述属于当时执行记录，随后用户明确授权推送。

仓库 API 当前 visibility 仍为 public，含赛事数据历史的独立处置尚未完成。v0.14 只在本地执行和提交代码；不推送，不修改 visibility，不 force-push，不公开逐样本产物。旧 README 的用户授权文字不能解释为主办方允许公开赛事数据。数据文件不在本阶段改动。

原 V1 ZIP SHA256 仍为 fdcbe03e8ea31577bfeed0c45cd0a0013bda42ab88d29db703fad9c2f7e557aa；原 R2 与发布指针不变。桌面副本未找到是 v0.13 的既有核查结果，本轮不重建模型或复制/覆盖桌面包。

平台回执已在此前按用户要求自行检索；可见证据仍只有用户回传分数，没有账号提交 ID、时间或有效状态。公开页面不能证明个人有效提交；本轮不重问用户，也不从包哈希或分数字段编造回执。platform_verified 保持 false。

[官方 2026-09-11 复赛通知](https://www.aicomp.cn/notice/notice-3/5248.html) 与 [赛题规则](https://www.aicomp.cn/tracks/tracks-6/4177.html) 本轮再次读取：正式复赛数据开放 09-21 11:00，复赛每天最多五次、取最高成绩、评分延迟。通知列示时间以官方账号系统口径为准；不推断初赛统一采用最高成绩。旧 test_b 预演不因 v0.14 历史质量结果升级为正式身份验收。
