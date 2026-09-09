# optimization-v0.3-drift 设计规格

日期：2026-09-08
状态：已批准，进入实施
证据范围：公开开发标签且 reference_time 严格早于 2024-11-01；9/10 月已被分析，不称为独立验证。

## 目标与候选

在不改变 baseline-v0.1、CatBoost 参数、目标、损失、基础特征窗口或非负裁剪的前提下，验证两个漂移候选。

- M1_FROZEN 控制：tap_iron = 0.5 E00 + 0.5 B1；tap_time_len = B1。
- C1_RECENT30_TIME：iron 与 M1_FROZEN 相同；time 是每个 fit cutoff 前 30 天全局中位数。
- C2_PROCESS_CHANGE：E09_PROCESS_CHANGE_E02 的逐行等价别名。该路线去除 history age、保留 history count/target，并加入 16 个 operation 列的三类 signed level delta，共 48 列。
- C3_PREREG_COMBINED：仅当 C1/C2 各自通过全部既有门时生成；iron 来自 C2，time 来自 C1。不再训练或调权。

C1 的近期标签必须同时满足 reference_time >= cutoff-30d、reference_time < cutoff、label_available_at <= cutoff。窗口为空即失败，不回退、不搜索其它窗口。本轮不使用样本权重；M2 降级为未来后备，M3 暂停，M4 关闭。

## 边界与验证

复用 configs/optimization_v0_2/validation.yaml 的五个 origin：O202406 H1-H4、O202407 H1-H4、O202408 H1-H3、O202409 H1-H2、O202410 H1，以及 DEV_LONG/DEV_SHORT。开发读取上界固定为 2024-11-01T00:00:00+08:00（exclusive），不读取 11 月目标，不运行保护 holdout。

沿用 configs/optimization_v0_2/acceptance.yaml：

- J 相对 E00 改善至少 0.002；
- 至少 3 个 horizon 改善；
- 任一 horizon 退化不超过 0.002；
- DEV_SHORT 退化不超过 0.002；
- 任一目标 mean WMAPE 退化不超过 0.005；
- DEV_LONG 优于 B0/B1 中更好的控制。

C1/C2 独立判定。G1 仅在至少一个 Stage 1 候选通过时为 PASS；M1 或控制不替代该条件。C3 单独判定，不反向改变 Stage 1 结论。

## 输出合同

每个 evaluation unit 保存 B0、B1、E00、M1、C1、C2 和条件式 C3 的预测、行级误差与指标。顶层必须输出：

- equal-horizon J、H1-H4、scenario-pooled 与 monthly E；
- 两目标 WMAPE；
- origin、horizon、spout、00-05/06-11/12-17/18-23 小时段；
- 两目标各自的实际值 Q1-Q4；
- signed bias 与 signed error sum；
- C1/C2/条件式 C3 相对 E00/B0/B1 的本地日成块配对 bootstrap，2000 次，seed=20260908；
- Stage 2 触发或跳过原因；
- 输入/代码身份、ledger 未变化和 protected_labels_read=false。

scenario-pooled 把 origin×horizon 场景行当评估记录，仅作补充诊断，不替代 J，也不宣称重复样本独立。

## 体系结构和失败语义

新增 src/bf_tap/optimization_v03/ 薄编排层，不修改 src/bf_tap/optimization/ 的 v0.2 合同。它先调用现有 run_optimization_validation，以 suite=all、候选 E00/E09 生成 core_v02/；再按 origin 从同一安全开发标签与核心预测派生控制和候选。C2 必须与核心 E09 最大绝对差为 0。

run 目录用 exist_ok=False。配置不一致、近期窗口为空、预测键不一致、C2 不等价、输入/代码变化或 ledger 变化都写 FAILED 并保留目录。G0/G1 分开。未经用户指令不 commit、不 push。
