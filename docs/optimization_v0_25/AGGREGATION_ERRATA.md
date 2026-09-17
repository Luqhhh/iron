# optimization-v0.25 聚合口径勘误

v0.25 的模型、逐样本预测、平台回传与冻结 manifest 不受影响。原 `RESULTS.md` 的统一离线结果表使用了正确的 `macro_origin_mean_wmape`：每个 origin 先计算 WMAPE，再对同 horizon 的 origin 等权平均，最后令 `E=(WMAPE_iron+WMAPE_time)/2`。

其后文字列出的四个“分目标 WMAPE 变化”误用了 `exposure_pooled_wmape`，即先把同 horizon 各 origin 的误差分子和标签分母分别合并后再相除。因此，那些数字不能与前表的宏平均 ΔE 直接使用 `ΔE=0.5ΔWMAPE_changed` 对照。

正确的宏平均分目标变化如下：

| 候选 | H1 | H2 | H3 | H4 |
| --- | ---: | ---: | ---: | ---: |
| V25I_HISTORY_CENTERED_RECENCY_IRON / tap_iron | +0.00046081003088 | +0.00073650827557 | +0.00149192306759 | +0.00046096763357 |
| V25T_HISTORY_CENTERED_QRF_TIME / tap_time_len | -0.00207225043700 | -0.00619428825508 | -0.00722287401723 | -0.00224502652848 |

逐 cell 的两个目标使用相同样本 ID、n 和标签分母；候选未改目标的预测保持不变。以未打印舍入值核验，每个 cell、H1–H4、J 和 DEV 的 `ΔE=0.5ΔWMAPE_changed` 误差均不超过 `1e-12`。独立机器可读审计由 v0.26 P0 从 v0.25 已保存的 `canonical_scorecard.csv`、`canonical_summary.csv` 与 `all_errors.csv` 生成；不重训，也不覆盖 v0.25 产物。
