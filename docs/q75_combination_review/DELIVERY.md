# 单一组合探针交付（2026-10-02）

**EMA_Q75_N_TO_V36_P05** 已在本地完成，G0通过；G1平台未测，用户自行上传。平台最佳仍是用户回传、未独立核验的Q75/Q100=96.3920。

时长为`0.25V36+0.25N0048−0.25V7+0.75EMA`，等价于`Q75_time+.2*(A35_time−A60_time)`。铁量逐字段复制Q75原字符串，错配0。0新拟合、0新seed，仅交付这一个组合探针。

四seed增量为+.006085、+.007408、+.005998、+.007178；平均+.006667，seed层LCB95+.005809，19/20折正。该组合回答旧N0048/V36份额在EMA加入后是否仍合适；不把本地增量当平台承诺，也不继续扫描。完整损害分布见[复查结果](RESULTS.md)。

独立新进程只回读固定旧包和新包，以50位Decimal恢复V36/N端点重算，最大差2.8e-14；322唯一ID、官方模板顺序、有限非负、ZIP只含result.csv、CRC和CSV/ZIP字节回读全部通过。这里是确定性交付字段的冷回放，未重新拟合或宣称新增模型复现。

包：`local/runs/q75-combination-review-20261002/release-r1/EMA_Q75_N_TO_V36_P05/Luqhhh_bf_tap_predict_round2.zip`。

ZIP SHA256：`ef3ad6f8a90d31722bb3b4f75d497e620c96af81d3785bd550e255fe22e7a8bf`。

公开脚本为`scripts/q75_combination_probe.py`；冻结原包/模板/报告身份在`configs/q75_combination_review/RELEASE.json`。私有manifest、warm及package-audit保存同目录。0桌面写入、0助手上传；不推断额度，DE3＋Q75仍仅替补。
