# SHORT_SPAN 四切分确认结果（2026-10-01）

确认批次正常结束，controller exit0；10个新增warm/cold单位、440份保存状态、原生调用预算和独立终态回读均通过。SHORT_SPAN相对同seed Q75的四个完整切分收益全部为正，单侧seed配对LCB95也为正，**通过本阶段四切分证据门**。原开发自动分类仍为exploration，minimum_improved_folds失败记录保持；新增两个确认seed的两seed分类仅作描述，不能追溯替换原开发决定。

| split seed | 相对Q75本地包分增量 | 完整OOF行数 |
| --- | ---: | ---: |
| 42（复用开发） | +0.002189932 | 2754 |
| 3407（复用开发） | +0.001385540 | 2754 |
| 271828（新增确认） | +0.001164418 | 2754 |
| 314159（新增确认） | +0.001527621 | 2754 |

四seed平均增量 **+0.001566878**；单侧95% Student-t下界 **+0.001047449**，df=3。每seed独立构造 `C=Q75+.75*(SHORT−old_EMA)`，铁量取同seed原V32参照，没有跨seed平均预测向量。正收益fold为开发6/10、确认8/10，只作描述；四个切分重用同一批样本，不能证明平台泛化。

## G0与实际终态

- 新增320条参照管线实例和20个EMA估计器，保存并独立冷审计440份状态。CatBoost fit290、EBM fit60/boost480、MLP fit10、torch Optimizer100，与冻结预算逐项一致；10个单位均无失败，没有自动重试。
- 实际warm/cold最大RSS为1283.45/641.17MiB，低于1536MiB。冷推理完整batch差0，反序/分块最大差6.10352e−5，单行4.20599e−5，低于5e−4；固定组合最大差2.27374e−13，低于1e−9。
- 最终原生.pt审计核验训练内预处理、目标尺度、选轮轨迹、新初始化refit与独立冷预测。终态回读再核验751份源码/配置、1234份实际模型源码、182份输入、四份折向量和1760份witness产物文件；四seed独立算术差不超过3.29e−16。
- 实际完成事件、launch/activation的PID与时间顺序一致，controller/observer实际退出，监控结束；总耗时10781.96秒仅作描述。
- 新全2754行拟合、包、桌面写入、助手上传、平台队列新增均0。原summary的G0待审计字段是审计前快照，原字节保留；最终通过依据为audit、completion-event、process-terminal及独立终态回读。

## 平台判断与下一步

这是稳定的小幅本地改善，不是平台成绩或96.45达标证明。当前平台最高仍为用户回传Q75/Q100=96.3920，未独立核验，距96.45为0.0580；不将本地增量直接加到平台成绩，也不使用历史EMA收益的固定放大倍数。

SHORT全量执行与包生成的单候选准备已完成，下一步若获得具体任务授权，只执行一个原配方全量训练程序、2次Optimizer，生成一个本地包；固定beta=.9801、时长替换权重.75，铁量复制Q75父包原字符串。完整冷推理及独立包回读通过后才交付，桌面写入/助手上传0。四切分通过不自动扩大授权或新增平台队列，不为凑额度安排测试。EMA铁量包仍待用户反馈，DE3+Q75继续仅作替补，ModernNCA仍仅工程准备。

## 证据身份

科学源提交 `0f593434b5e8d5dda4e69c4e7ef7a94cdd6864f4`；私有根 `local/runs/ema-span-confirmation-20261001/confirmation-r1`。协议保持[冻结确认协议](PREREGISTRATION.md)。

- `manifest.json`：`7a81ac382bfd09fde2ff625a86f9a9e7fa5326adda6b9cdc23df62c628571120`。
- `summary.json`：`e54aa79ab3a51464e21731538e4068439cccdf7bbf51f26bb07455d502290c94`。
- `audit.json`：`7c19aa5711b666107772a1f37461f497509e79bf8b4849fb3f4533f6ff529b8a`。
- `completion-event.json`：`2f8c67373a01132e5929346acb9c76261951fdd0fde1a8f315967a6f813ed401`。
- `process-terminal.json`：`f6491d22e5713d4950ab66a84530a42ba54c2982a6f56dc504d24d2a9e4a6521`。
- `terminal-verification-r1.json`：`14c123392e7c18222f62da8bc1211618efb88b93f94cd3ee16d085dea35719b9`。
