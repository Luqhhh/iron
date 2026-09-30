# 本机继续状态与重复实验排除（2026-09-30）

本机分支 codex/round2-today-two-slot-release 已先执行 git pull --ff-only。
用户批准的 RFM 设计和计划来源为 86ea6b3；本机尚未启动其拟合。
此次拉取发现另一机器已按该设计完成实验，因此本机不再启动重复批次。

## 已发布远端证据

- RFM：codex/rfm-kernel-preparation，326ea4f559caf0e9a57954f7cbd557b6a74d87ff。
  [结果](https://github.com/Luqhhh/iron/blob/326ea4f559caf0e9a57954f7cbd557b6a74d87ff/docs/rfm_kernel_preparation/RESULTS.md)。
  远端报告 G0 锁定全套1372通过/8跳过，开发40外层单元、80过程、157求解、77更新，冷审计及独立算术通过。
  FULL_RFM 铁量平均相对当轮参照 -0.025145361，时长 -0.048419481。
  两目标无入围者，确认未执行。时长对固定核的机制优势 +0.033564488 未转化为参照增益。
  这是远端已发布的结果，本机尚未收到私有证据，不能称本机独立复核或复现。
- PTaRL：codex/ptarl-space-calibration，ba5ce299185a3e4ef468c5f4aaf61ae9d6cf0859。
  [阶段准备](https://github.com/Luqhhh/iron/blob/ba5ce299185a3e4ef468c5f4aaf61ae9d6cf0859/docs/ptarl_space_calibration/PHASE_EXECUTION.md)。
  远端报告完整执行工程已测试，正式资源探针与开发未启动；依赖完整 DE3 当前四划分参照。
- 参照核心：codex/incumbent-de3-reference，6e285fa63b4cc4b221fb7bfc142c2e69ccd92fbb。
  [设计](https://github.com/Luqhhh/iron/blob/6e285fa63b4cc4b221fb7bfc142c2e69ccd92fbb/docs/incumbent_de3_reference/DESIGN.md)。
  已有合成验证的原生模型及账本核心；完整阶段冻结、资源准入和覆盖审计仍待实现。

## 当前参照与本机缺项

远端 d96d151 登记用户报告 DE3_IRON_USER_REQUESTED=96.3749；
本机不能将该远端转述冒充本对话新收到的平台回执。后续规格需明确绑定该最新已登记参照，
保留原 V32=96.3727 和 RFM 当轮的冻结比较口径。
DE3 原开发门失败、未派生确认的结论不变，不因平台小幅改善追认晋级。

用户于本次对话明确告知私有缓存尚未共享。本机 local/runs 下未找到
independent-ensemble-checkpoints-20260929、strong-component-regularization、rfm-metric-learning-r1 或 PTaRL 缓存目录。
公开代码可以通过 Git 获取；local/ 是明确忽略的私有证据，不应解除忽略或推送模型。
本机六个主要数值库版本与公开原生 SPEC 相符，但这不替代数据、源、预测与模型身份审计。

当前继续顺序：私有缓存交接及零拟合验证 → 完整当前参照阶段规格/计划 → 冻结准入 → PTaRL。
共享清单见 [PRIVATE_CACHE_HANDOFF.md](../incumbent_de3_reference/PRIVATE_CACHE_HANDOFF.md)。
不以重跑已有开发拟合代替证据交接；缺失身份或哈希时停在参照验证阶段。

## 本次实际工作

只读取公开源/规格/结果及本机目录、运行环境，整理交接文件。
官方标签读取、模型拟合、资源探针、包、桌面写入和平台上传均为0。
没有合并或修改队友分支，没有覆盖原缓存或改变历史冻结门槛。
用户的正常训练监控偏好仍为每30分钟一次、正常保持安静；远端文件中的600秒不自动取代该偏好。
