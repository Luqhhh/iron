# LAPLACE_FIXED_A20 全量交付

2026-10-02。下一次建议优先测试这个固定包：`0.8*Q75_time+0.2*LAPLACE_FIXED_time`，铁量保持Q75父CSV原字段字符串。完整匹配开发中，简单固定尺度MAE位置臂两次优于学习尺度Laplace；相对真实Q75参照通过四seed门，提供一个值得平台验证的点预测方向。保留Gaussian及EMA_MEAN3包，后续顺序结合下一次回传调整，不要求一次测完全部，不推断剩余额度。

**G1相对Q75本地正式晋级，平台未测。** 四seed增量+.008952557572、+.012171228685、+.008908561702、+.016872094481；均值+.011726110610，单侧seed配对LCB95+.007307128497。相对Gaussian均值+.003321179818，但3/4 seed正、LCB95−.000497374980，没有统计独胜结论。详见[完整确认](../q75_laplace_confirmation/RESULTS.md)。重复切分仍是同批样本，本地增量不换算平台分；平台最佳仍为用户回传、未独立核验的96.3920，96.45尚未实现。

**G0通过。** 原科学训练器不变，inner27001将2754行划2203行F/551行C；selector选择93轮，于123轮停止，fresh全量refit93轮。2次实际optimizer、2个新状态；训练身份、预处理统计/词表、Laplace标签、最佳checkpoint、fresh轮数及冷/反序/37行分块通过，最大差2.84217094304e-14。另一个禁止训练输入读取的新进程对322行test冷推理差0。训练、冷审计、无标签推理、封包和独立包回读5个真实进程exit0，峰值OS RSS503.05859375MiB。

ZIP仅含`result.csv`，322唯一ID按官方模板顺序、有限非负、CRC、固定融合独立标量算术、ZIP/CSV字节一致；铁量字段原字符串差异0。无裁剪；0桌面写入、0助手上传，用户自行测试。

私有交付包：`local/runs/q75-laplace-release-20261002/release-r1/LAPLACE_FIXED_A20/Luqhhh_bf_tap_predict_round2.zip`。

- ZIP SHA256：`8557981aff6c20b096434660dbe3930fe91246600dcbace82916685aea6e764c`
- CSV SHA256：`ecb71e035c7e6f6b7a6fd48fb9d8f10c573d8423b623e6c5a10cebb28b70af5e`
- 发布源码：`a59554b`；模型科学实现逐字节保持24b82a4/4994656阶段。
- Manifest：`6ae9aba3330d98d177942e040cbe0433660d79084684f8a3764192a6d0a3f5eb`
- 实际终态：`c8d1b96c7d292e33c3010f73d5bea8c700a88a797fdf3190b2b319e4ca105ed2`。

本Laplace批次分阶段预算：合成工程4、完整开发40、接续确认20、全量2次optimizer，总计66；额外测试optimizer0，自动重试0，1个新提交包。锁定Python3.12检查分别15/6/9项通过，均拦截optimizer构造且实际尝试0；不声称全仓测试。原Gaussian阶段的准备/恢复记录保持独立，不混入本批成功次数。队友SWA确认未重复，原始结果包仍待接收。

用户随后明确“本轮结束后暂停”。本轮已完成，后续优化暂停，无运行任务或后台监控；现有交付包和证据保留，等待用户恢复指令。

## 2026-10-02 桌面交付（用户后续具体指令）

用户在本交付完成后明确要求“LAPLACE_FIXED_A20 写桌面”。按该指令把已审计通过的同一ZIP复制到 `C:\Users\lqh22\Desktop\submission-LAPLACE-FIXED-A20-20261002\`，并附 `README.txt`；本文上半部分记录的“0桌面写入”保持为当时发布快照，不回改历史。

桌面ZIP与原包逐字节相同，SHA256仍为 `8557981aff6c20b096434660dbe3930fe91246600dcbace82916685aea6e764c`。独立新进程（锁定Python 3.12.12）回读通过：仅 `result.csv`、CRC正确、322唯一ID按官方模板顺序、预测有限非负、铁量原字段与Q75差异0、时长322行均为融合后值。0新拟合、0新提交包、0助手上传、0标签读取。收据：`local/runs/q75-laplace-release-20261002/desktop-delivery-r1/desktop-delivery.json`（SHA256 `a1a35db0e8767cb179f9147901ffcaf9fd3e20cf25794d9761471ce167f44b4c`）与 `desktop-independent-audit.json`（SHA256 `0e3cd5445008877040ece608a6c271ab2bbbe8f214745765d10fee2f15e847ef`），审计脚本 `audit_desktop.py` 同目录，均在私有 `local/`。

平台分数仍未测得，96.45未达；上传与回传由用户执行，两个名额安排见[名额评审](../platform_slot_review/ALLOCATION.md)。
