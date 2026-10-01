# 强组件的组合、选择与重训验证（2026-10-02）

最新平台登记最佳仍为用户回传、未独立核验的Q75/Q100=96.3920，距96.45差0.0580。已完成ModernNCA MSE/MAE和SHORT/EMA铁量平台反馈按原状态保留，不重新列为新候选。

单一[组合权重探针](../q75_combination_review/DELIVERY.md) `EMA_Q75_N_TO_V36_P05` 已通过G0，平台反馈待用户回传；不扩展扫描、不由助手上传。DE3+Q75仍只列替补。

## 融合选轮已闭合

原 `local/runs/ema-fusion-selection-20261002/development-r2` 真实两个进程退出码均0，30次optimizer、40状态冷回读及独立标量评分通过，G0通过。

|完整开发split|COMPONENT相对原Q75|FUSION相对原Q75|FUSION−配对COMPONENT|
|---|---:|---:|---:|
|42|+0.001595301|+0.002496809|+0.000901508|
|3407|+0.002024353|+0.001275261|−0.000749092|

G1：融合准则对配对控制未在两split均正，因此不符合预登记确认条件，0新确认seed、0正式晋级。两臂采用共同校准seed27001，原EMA采用42；两臂相对旧Q75的正增量不能全部归因于融合选轮。这个固定实验的失败不能排除所有组合选轮机制，亦不能追溯改门或扩大轮数。

原report SHA256：`c0d84ce0946145092e2fdade380ea7f69e88f8a1136c1acfcda557ad37e0e3d1`；terminal SHA256：`6febe66a0c7ad629f11bd002baebad9b3efe8c49b9df8243baf2d495bded82cb`。原失败启动development-r1保留。

## 后续独立阶段

按顺序执行[训练随机性配对](initialization/PREREGISTRATION.md)，再执行[精确更新次数对照](steps/PREREGISTRATION.md)。前者固定training seeds42/1042/2042、两完整开发split、BASE/EMA配对、同split内等权平均；后者仍固定原training seed42和选轮，不同时采用新初始化、融合准则、beta或权重。科学源固定在独立worktree；状态和私有产物独立，历史冻结协议不改。

初始化阶段新增40估计器/80optimizer并复用20估计器，共120状态；更新阶段新增10fresh refit/10optimizer并复用10估计器，共30状态。分别闭合G0/G1和真实终态，再按预登记条件决定确认准备；本批没有预授权的新增确认seed、全量拟合或平台包预算，后续符合门槛时仍先另行冻结，不需常规逐项申请。

任务状态以本阶段EVIDENCE_STATUS键 `ema_retraining_initialization_20261002`、`ema_retraining_step_matching_20261002` 和私有terminal为准。运行期间G1未测，不能把启动、冷回读工程成功、多个初始化或两split结果称为四split正式晋级。所有本地收益不外加到96.3920，平台由用户上传并回传。

初始化批次已于2026-10-02冻结并串行启动，源码commit `5faffd3`，manifest `5dd1baee3cfce70cfad9bdda4e0aa050847856d05b00d1588ff99d998374708f`。锁定Python3.12完整测试1481 passed、0 skipped；第一次缺少历史缓存别名的工程测试失败保存在私有engineering目录，0科学拟合。独立service `iron-ema-retraining-validation-r1.service` 固定Restart=no、RuntimeMax=infinity、全部数值线程1，600秒观察并响应真实退出。初始化真实终态和独立审计通过后自动调用步骤阶段的独立冻结与串行程序；任何阶段失败则保留证据并停止后续，不自动科学重试。两个阶段G1仍待完整OOF，0确认/全量/包/上传。
