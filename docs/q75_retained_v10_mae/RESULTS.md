# 原V10 MAE当前参照复核：不新增候选

2026-10-02。原MAE配方在旧V7_TIME参照下两个开发seed有增量，但按冻结排序未获确认；原SmoothL1四seed失败保持。本次只读原MAE与V7 MSE的两个完整split、各五fold预测，固定A20在当前Q75上复核，没有重训。

| V10_MAE_Q75_A20相对端点的总分增量 | split42 | split3407 |
| --- | ---: | ---: |
| Q75 | +0.003989254 | +0.007816522 |
| 已就绪LAPLACE_FIXED_A20 | −0.004963303 | −0.004354707 |
| 同骨干V7_MSE_Q75_A20 | −0.000886796 | −0.001007844 |

两个Q75增量为正，但相对就绪Laplace两个seed均负，相对同骨干MSE两个seed也均负；未过预先登记的后续工程研究门，不增加训练、确认或平台候选。原candidate_tiers对Q75给开发formal分类，但不覆盖本阶段比较门，也不等于四seed正式晋级。

旧MAE相对V7_TIME的优势，不能推断它在当前Q75端点上仍优于原MSE。参考融合不同会改变增量排序，这不是标签冲突或某个模型家族整体失败。本结果也不证明平台排序，但没有形成足以优先消耗两个名额的新理由。

G0限于保留预测的身份与算术：原V10/V7的完整2754行frame digest、两套fold digest、20份MAE/MSE预测哈希、物理source/seed/trial、训练和inner ID、train-only特征均值及已保存选轮元信息均匹配。相关原科学源码与原manifest逐文件同哈希，当前锁定依赖匹配。8项Python3.12检查通过，两个独立进程实际exit0；11086项math.fsum校验最大差6.94e−18，OS峰值RSS407.684MiB。

原缓存没有这批模型checkpoint或逐轮selector trace，本次冷模型状态数0，不能宣称原科学模型独立冷复现、重新观察原生optimizer或验证未保存的每轮选择。训练入口及Optimizer构造被显式禁止。新fit、optimizer、状态、工程、确认、全量、包、桌面、上传均0；所有本批执行已结束，无监控任务。

科学worktree commit `05c3dea`；私有结果`local/runs/q75-retained-v10-mae-20261002/review-r1`。manifest SHA256 `8ab794298584a78e173db583c128b7aecca61a9d92553a9310c72d324dc971fb`；report `0964162ee97682f98d004d308ebb446f5dbff265376f9188b44b7f5591f03a1f`；independent-score `038cb15476b97f294888c7a69e2c3892086e0d4c7bb3e64587c77ffbcf07f17d`。

两个平台名额继续保留。当前首项暂定Laplace时长，第二项优先考虑已完成交付的独立MAE铁量，但具体包仍待首项反馈；队友预热→MAE尚未见新结果。当前平台最佳仍为用户回传96.3920，96.45未达到。
