# SHORT_SPAN_FULL_Q75 本地交付（2026-10-01）

用户已明确确认固定范围。一个原配方全量训练程序、2次Optimizer、一个本地ZIP已完成；controller实际exit0，状态冷推理、原生.pt审计、独立包回读及终态回读全部通过。新增CV/确认seed/权重探针/桌面写入/助手上传均0，平台分数待用户自行上传回传。

私有文件：`local/runs/ema-short-full-release-20261001/release-r1/package/Luqhhh_bf_tap_predict_round2.zip`。

| 项目 | 结果 |
| --- | --- |
| 时长配方 | Q75_time + .75 × (SHORT_full − old_EMA_full) |
| SHORT beta | .9801；其余网络、训练、选轮及fresh-refit配方不变 |
| 铁量 | 直接复制父Q75 CSV原字段字符串，差异0 |
| 原内层训练 / 全量refit | 2203 / 2754行，选中epoch81，fresh-refit81 |
| 原生调用 | torch Optimizer2；CatBoost/EBM/MLP0 |
| 新保存状态 | selection/refit各1份，独立冷审计通过 |
| ZIP | 仅result.csv，322个唯一ID、官方模板顺序、CRC通过 |
| 数值与回读 | 两列有限非负，独立时长算术差0，CSV/ZIP字节一致 |
| 冷推理差 | 完整batch0，反序3.24798e−6、分块6.49596e−6、单行0，均低于5e−4 |
| 最大worker RSS | 728.72MiB，低于1536MiB |

新全量作用域使用预声明−1/−1身份标记，原初始化和inner seed仍42；这不是新增外层切分。没有裁剪、增设后处理、自动重试或覆盖原包。旧输入、377份源文件、382份模型源码和34份冻结输入的身份回读通过，源提交`989209005096c56fd292212ad835540038490b84`。

G0已通过。[四个完整本地切分](../ema_span_confirmation/RESULTS.md)各自正收益，平均+.001566878、seed单侧LCB95+.001047449；原开发exploration及历史失败决定保持。G1平台收益尚未测得，不能将本地增量直接加到用户回传的当前最高96.3920，也不能据此预计达到96.45。这个包用于检验短EMA平均跨度在平台的收益，未自动新增平台队列；EMA铁量仍待用户反馈，DE3+Q75继续仅作替补。

## 文件与证据摘要

- ZIP SHA256：`c42847af646202b66de615566f04a51c5e50ef3c1d19b664366f02d012370a93`。
- CSV SHA256：`f6cda4d4126e8022aea842b08cde567f0e7eff1ebdf6396ef5b8181f4ba8db47`。
- manifest SHA256：`bdd09d6c6564f46d2df123584676e79cd34dbbc06cac1c42190343601639a22a`。
- native-audit SHA256：`d9da64b80c4a37868e01f1018a64c9f43b54676503712e17ed343c0c57d7f7ba`。
- package-audit SHA256：`89ae04905d9192fd7a5c6d4e630cd292ac59e8584f48ef76cef81b86b2972b05`。
- completion-event SHA256：`29ecaf57cece0f1ff3b0e805ef4c5c3b1c0ab909becf0ea7b28f038edd147ab7`。
- process-terminal SHA256：`a2a3d0c391e1fbd2a5564e8d06154fb857318a6a4b726b9c4dc1206e7c29ebce`。
- terminal-verification SHA256：`965d4deab72960d5a12b043327b81a3d7c6511a06d3de7ac88277410ffa68dae`。

## 用户回传：固定SHORT配方低于Q75

用户按前一条文件列表回传“1.96.3911 2.96.3816”：第1项SHORT_SPAN_FULL_Q75为**96.3911**，相对Q75=96.3920为**−0.0009**，来源为用户回传、未独立核验平台凭证。原ZIP身份、322行官方顺序、CRC和铁量原字符串再次核验通过。当前最好Q75/Q100保持，96.45未达成。

四个本地seed均正、LCB95>0的事实保持；此次固定beta=.9801、替换权重.75及原选轮/refit配方没有平台增益，不推广为所有短跨度或整个EMA家族失败。新增拟合/包/桌面写入/助手上传0，不推断剩余额度。

## 固定模型单列权重的条件上界

以Q75为alpha=0、已测包为alpha=1，若平台为同322行的两列等权全局WMAPE且保留另一列、无裁剪或新后处理，隐藏目标L1总量满足T≥P/(1+U)，其中P为父预测L1总量，U=(100−S_Q75)/50。因此分数的Lipschitz常数至多50D(1+U)/P，D为两包改动列预测的L1距离；对0≤alpha≤1，峰值上界为(S0_hi+S1_hi+L)/2。四位报告分数按±0.0001保守包络计算并向上舍入：**SHORT单列混合≤96.4057；EMA铁量单列混合≤96.4432**，均低于96.45。对alpha≥1，凹分数函数的割线延长也是上界，两个方向都不能超过父包。

这些是条件上界，不是可达到分数或预测；不排除小幅提高，也不覆盖负alpha、新拟合模型或同时调整两列。仅降低这两组固定模型正向单列权重扫描对96.45的优先级，不改变科学晋级门槛。零拟合分析没有读取隐藏目标；完整证据见状态键`platform_feedback_short_ema_iron_20261001`。
