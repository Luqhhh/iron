# DE3铁量＋EMA三成员mean时长交付

> 最新反馈（2026-10-04）：用户回传 **96.3977**，相对EMA_MEAN3=96.3954提高 **0.0023**，成为当前用户回传最佳，距96.4/96.45/96.5分别0.0023/0.0523/0.1023。分数未经独立平台核验；下文平台未测与96.3976条件算术保留为交付时快照。

2026-10-04。用户明确要求“生成提交包，替换旧提交包”。已按[事前固定配方与预算](../../configs/de3_ema_mean3_release/SPEC.json)生成 `DE3_IRON_EMA_MEAN3_TIME`，完成独立包审计及桌面替换。铁量直接复制已测DE3原CSV字段字符串，时长直接复制已测EMA_MEAN3_FULL_Q75原CSV字段字符串，按官方模板顺序组合；无浮点重写、训练、模型预测重算、权重选择或新后处理。

当前桌面提交文件：`C:\Users\lqh22\Desktop\submission-DE3-IRON-EMA-MEAN3-TIME-20261004\Luqhhh_bf_tap_predict_round2.zip`。

- ZIP SHA256：`c51e050b69dc40cc808bfbcd6dc5442db131f83d3b05446ed0eba7c255675e78`。
- CSV SHA256：`649759fdb96fa991ad1a41fd1f64df2b85c121330d6933f1d9944ce6f836462b`。
- 私有原包：`local/runs/de3-ema-mean3-release-20261004/release-r1/DE3_IRON_EMA_MEAN3_TIME/Luqhhh_bf_tap_predict_round2.zip`。

G0：Python3.12.12独立新进程审计通过。生成前冻结两个原包、原包审计、模板、配方、生成/审计代码、依赖锁文件与条件算术证据，共11项文件；生成、包审计及桌面审计均重新核对身份。两个父包已有独立冷推理审计，本次直接复用其已审计CSV，没有执行新模型冷推理。新包仅含result.csv，CRC正确，322个唯一ID按官方模板顺序，预测有限非负；铁量相对DE3、时长相对EMA_MEAN3的字段字符串差异均为0。桌面与私有新包字节完全一致。生成、独立包审计、桌面操作及独立桌面回读均实际exit0。

桌面旧目录 `submission-DE3-IRON-EMA-TIME-Q75-RESERVE-20261003` 已改为新候选目录名，旧ZIP和README原字节移入新目录中的 `旧包归档-DE3-Q75/`。旧包SHA仍为 `86bf20d8cbe938f06b7550a3cf50e28fd100ba94a672c666d0aee576f25bcee3`，原README哈希一致；私有旧提交包和历史证据均保留。用户应选择新目录根部ZIP，归档子目录为旧Q75版本。

G1：用户指定、平台未测的列组合，未正式晋级。当前最佳仍为用户回传、未独立核验的EMA_MEAN3=96.3954。在同一平台测试集、相同可加评分及源包/历史回传正确的条件下，组合算术为 `96.3954+(96.3749−96.3727)=96.3976`，相对当前最佳+0.0022，仍低于96.4；不能登记为实测分数或新最佳。旧DE3＋Q75的历史替补决定保留，本次新组合按用户要求就绪。

计数：0新拟合、0optimizer、0新CV/确认seed、1新提交包、1份新桌面复制、1个旧桌面目录内容归档、0助手上传。不推算剩余额度，不改其他会话的研究或运行状态。

私有审计根为 `local/runs/de3-ema-mean3-release-20261004/release-r1/`：`manifest.json`、`release.json`、`package-audit.json`、`desktop-copy.json`、`desktop-audit.json`、`desktop-operations.jsonl`、`terminal-reconciliation.json`。机器状态键为 `de3_ema_mean3_release_20261004`。

## 96.3977平台反馈

用户紧接本包桌面交付回传 `96.3977`，绑定上述ZIP SHA256 `c51e050b69dc40cc808bfbcd6dc5442db131f83d3b05446ed0eba7c255675e78`。G0：Python3.12.12新进程重核11项冻结文件、原交付审计收据、私有与桌面包SHA/CSV/CRC、322个唯一ID官方顺序、有限非负及两列父包原字段，全部通过。G1：用户指定列组合取得用户回传平台正收益，原未正式晋级身份保持。

相对前最佳EMA_MEAN3=96.3954提高0.0023；比原条件算术96.3976高0.0001，与三项历史报分各自四位小数舍入造成的范围相容，不证明新的跨列协同或改变原可加评分协议。将本包登记为当前最佳并移出待测队列；后续新阶段使用96.3977参照，已冻结的EMA_MEAN3_Q100等阶段保持原参照、包和门槛，不追溯重写。

反馈收据：`local/runs/de3-ema-mean3-release-20261004/platform-feedback-r1/feedback.json`，SHA256 `973ce3068a7d7c7f287420dd4df4bf0b3d22b442b20bbd55f62a7eacce95d0e1`。本次登记0新拟合、0包、0桌面写入、0助手上传，不推断剩余额度。当前分数仍低于96.4/96.45/96.5。
