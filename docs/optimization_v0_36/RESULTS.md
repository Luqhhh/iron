# optimization-v0.36 结果：最后一个初赛槽的完整时长端点等权融合

执行日期：2026-09-20。分支：`optimization-v0.36-single-slot-time-blend`。实施起点：`70eebab3c6f469ff2b9f0dc9baf42d0076de0f06`；冻结实现提交：`14754ae1383859ede851d7ce95fddf77848883d1`。权威私有运行：`local/runs/optimization-v0.36-single-slot-time-blend-r1`。

## 结论

唯一候选 `V36T_V34_V30_EQUAL_BLEND` 已完成，G0 工程状态 **PASS**。铁量逐字符串复制 V34T/V30A 的共同完整铁量；时长在两条完整六位端点之后按整数微单位 1/2 平均，半微单位按 ties-to-even。模型、树、预处理器、校准器和系数拟合均为 0；没有读取 test target，没有生成第二候选、恢复包或额外比例。

用户提供的最新约束为：初赛只剩 1 个平台槽，且没有晋级分数要求；该信息未由 agent 独立核验。V35A 已从旧计划释放，状态为 `UNTESTED_SKIPPED_SLOT_REALLOCATED`，不是平台失败；V35B=83.2778 的历史记录不变。当前最高用户回传仍为 V34T=83.3201，但本轮没有恢复上传预算。

## 来源、查重与固定算术

真实执行端的两份 payload 与登记摘要完全一致：

| 端点 | result.csv SHA-256 | 原 ZIP SHA-256 | 用户回传 |
| --- | --- | --- | ---: |
| V34T | `80a74e687f74181ec962cc8a12328380706c0d6bbbc7d53dd932866a169d6466` | `e3f970fa96cad54e0a6c534cc473d3b19eeb88269cb8c2630411796bfa313924` | 83.3201 |
| V30A | `97b6c3e0c648a0c6454cc9b36f518625487c03d1efd7621746d049457c4aed9a` | `fffcf23b04b3069cd71027682047eea764a2747c2dea48cca91320d67e113986` | 83.3175 |

P0 检索没有发现已完成的同协议 v0.36 实验。两个端点的铁量在 335 行逐字符串相同；新时长全部位于两端之间，且相对两个端点均改变 335/335 行。相对 V34T 的最终时长增量范围为 `+0.074219` 至 `+1.908203` 分钟，中位数 `+0.962891`；相对 V30A 为相反方向。

端点选择明确使用过既有平台反馈，`platform_feedback_used_to_select_endpoints=true`。理想未舍入条件界的显示值估算仍是 83.3188；考虑端点显示舍入及隐藏时长分母后的表达式是 `S_emitted >= 83.31875 - 50*n*(0.5e-6)/D_T`。由于 `D_T` 未知且平台回传未经账号回执核验，这不是分数保证。

## 已消费历史回放

六个历史 cutoff 的两端预测均存在，因此完成了同一 `mean6` 回放。主口径为 `macro_origin_mean_wmape`，变化为候选减参照，负数表示改善：

| 参照 | H1 ΔE | H2 ΔE | H3 ΔE | H4 ΔE | J ΔE | DEV_LONG ΔE | DEV_SHORT ΔE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V34T | +0.00090920 | +0.00119866 | +0.00158934 | +0.00174440 | +0.00136040 | +0.00106602 | +0.00161011 |
| V30A | -0.00109991 | -0.00139655 | -0.00174093 | -0.00189662 | -0.00153350 | -0.00121819 | -0.00189675 |
| V1 | +0.00051441 | +0.00169744 | +0.00332882 | +0.00434642 | +0.00247177 | +0.00077747 | +0.00393446 |
| V21_REPLAY | +0.00013142 | -0.00069245 | -0.00175736 | -0.00085057 | -0.00079224 | -0.00104812 | +0.00033893 |

候选位于两个端点之间：相对 V34T 的 H1–H4、J 和两个 DEV 均退化，相对 V30A 则全部改善。历史暴露上的时长绝对误差抵消量合计 `164.635775` 分钟；六位舍入后的最小逐行抵消为约 `-5.0e-7` 分钟，处于登记的半微单位界内。目标隔离检查覆盖 20 个 cell 与 25 个宏摘要，最大恒等式残差 `2.78e-17`，低于 `1e-12`。

这是已消费历史回放，不是独立 holdout。用户已经明确选择最后一个槽作有条件界的探索，因此 G1 风险如实报告但不取消唯一平台名额；不能把这一决定写成离线胜出。

## 最终包、冷审计与测试

| 项目 | 值 |
| --- | --- |
| 候选 | `V36T_V34_V30_EQUAL_BLEND` |
| 行数 | 335 |
| result.csv SHA-256 | `4b1ee14896dc2a77c19dfa9e6f213f59e9dd43f347b8dc10789fab01a2487577` |
| ZIP SHA-256 | `49b2e39a6019d341261221b40a58f7e773edad471896c35808b9e89551f35e80` |
| 私有路径 | `local/runs/optimization-v0.36-single-slot-time-blend-r1/submission/V36T_V34_V30_EQUAL_BLEND/Luqhhh_bf_tap_predict_prelim.zip` |

ZIP 只含 UTF-8 `result.csv`，三列、335 个唯一 ID、有限非负六位预测。独立子进程从固定源 payload 重建结果并逐字节一致；full/reverse/chunk/subset/single 组合检查通过。当前工具完成的是 payload 级冷审计，`source_model_cold_audit_completed_by_this_tool=false`；两条源模型链的既有 completion/cold 摘要被重新绑定，没有冒充本轮重新执行模型推理。

锁定 Python 3.12.12 根测试为 **557 passed**，其中本轮定向测试 **31 passed**；私有产物守卫通过。状态为 `READY_FOR_ONE_EXPLICIT_PLATFORM_SUBMISSION`。agent 平台上传、桌面写入、恢复上传均为 0。

关键私有证据 SHA-256：

- `manifest.json`: `c1db77a64b606d0d0954aecb1aac46f0f11e79e6d79f19d4470e2df227753ae5`
- `p0_complete.json`: `2e306555adc6191e4a1cc03ea46c30358f5ced1919ef286ae456e8287561578d`
- `offline_assessment.json`: `3d59cd225e292149ea6f502337691ba33775fe2c39cca93d1678f37566050e89`
- `package_frozen.json`: `6b8617d1cf6bcabed5c171cc57ec528fb687a9ee903558a301de7e7bbed31ef1`
- `cold_validation.json`: `6408e8111f87e62e5dc4809fb0d5a616e6148b1f8a59198d3ae9c43462052224`
- `fit_counts.json`: `1255527205a349d09628f3baf14f82b6fa24734c820f5af080227904c960d8b2`
- `completion.json`: `3df9a9d96a3d214f22b6ffc24b16c43870b53df8c63e00a59df7c87cf056fb52`

平台反馈尚未产生。有效反馈高于 83.3201 才更新最高用户回传包；低于或等于时保留 V34T 的历史最高记录，但本轮不得安排恢复上传。最高历史回传、最新提交和账号当前生效条目继续分开登记。

## 桌面替换交付（2026-09-20 追加）

用户随后明确要求删除桌面旧提交包并写入 v0.36。桌面中识别出的 4 份旧赛事 ZIP（V34I、V34T、V35A、V35B）已删除；它们的 private local 冻结原件均保留，可由本地证据恢复。唯一新副本为：

`C:\Users\lqh22\Desktop\Luqhhh_bf_tap_predict_prelim_V36T_V34_V30_EQUAL_BLEND.zip`

桌面 SHA-256 为 `49b2e39a6019d341261221b40a58f7e773edad471896c35808b9e89551f35e80`，与冻结源包一致，且 ZIP 只包含 `result.csv`。交付后桌面匹配 `Luqhhh_bf_tap_predict*.zip` 的文件仅此一份；平台上传仍为 0。私有交付回执 `desktop_delivery_receipt.json` 的 SHA-256 为 `6f992966c8f5aea6224869d4336b922eba013471a2b6f85ee4e90309b228085c`。
