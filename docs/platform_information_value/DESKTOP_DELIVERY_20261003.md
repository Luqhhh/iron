# ModernNCA 与 EMA_MEAN3 桌面交付

2026-10-03，用户明确要求将这两个候选写桌面。已按原始包字节复制到 Windows 桌面的两个新目录；原科学配方、包、历史决定和测试反馈均保持。

| 候选 | 桌面目录 | ZIP SHA256 |
| --- | --- | --- |
| MODERNNCA_TIME_A20 | `C:\Users\lqh22\Desktop\submission-MODERNNCA-TIME-A20-20261003` | `dd1b1866eb514fefb026c37ef7239127572a65c75b42877b42f50c5aaa2937c5` |
| EMA_MEAN3_FULL_Q75 | `C:\Users\lqh22\Desktop\submission-EMA-MEAN3-20261003` | `016e7e9cb3f750c74509dbb51961c61fea205d0296bb233056da66005f69a396` |

每个目录内的提交文件均为 `Luqhhh_bf_tap_predict_round2.zip`。锁定 Python3.12 的独立新进程回读确认：源/桌面包字节一致，ZIP 仅含 `result.csv`，CRC 正确，322 个唯一 ID 按官方模板顺序，预测有限非负，铁量字段相对 Q75 原 CSV 字符串差异为0。复制进程和独立审计进程实际 exit 均为0。新增2份桌面复制，0拟合、0新生成包、0助手上传。

首次复制前校验误用了不带 `pred_` 的预测列名，在任何桌面写入前 exit1；修正校验字段后使用新交付目录，失败收据保留在 `local/runs/platform-candidates-desktop-20261003/delivery-r1/failure.json`，未修改原包。成功的授权、复制和独立审计收据在同批 `delivery-r2/`。

G0 桌面交付通过；G1 两项均保持人工平台探索、未正式晋级，尚无本次平台分数。用户自行上传并回传，不由桌面复制推断已上传或名额消耗。建议先测 ModernNCA，再根据反馈及 EMA_MEAN3 是否已有成绩决定后续；目标96.45尚未达到，当前最佳仍是用户回传、未经独立核验的96.3920。

同日随后用户明确回传“ModernNCA 96.3707 EMAmean 96.3954”。两项按上述候选名称和ZIP绑定，再次核对原包与桌面SHA及CRC通过。ModernNCA对原Q75为−0.0213；EMA_MEAN3为+0.0034，成为新最佳96.3954，距96.45为0.0546。分数来源是用户回传，未经独立平台核验；原本地质量门和人工探索身份保持。私有追加反馈在 `local/runs/platform-candidates-desktop-20261003/platform-feedback-r1/feedback.json`，未推算剩余额度。
