# optimization-v0.25 实施登记

本阶段从 `d4342998176293262a030dcb2f6997cc39c871f3` 建立独立分支
`optimization-v0.25-history-centered-targets`。固定两个候选：

- `V25I_HISTORY_CENTERED_RECENCY_IRON`：原 210 列、原 recency60 权重和参数，CatBoost 学习铁量相对合法 last100 历史基准的有符号偏差；V21 时长逐字符串不变。
- `V25T_HISTORY_CENTERED_QRF_TIME`：复用原 v0.15 预处理器，QRF 学习时长相对合法 last100 历史基准的有符号响应；原 QRF 门控和 V21 中位数收缩不变，V21 铁量逐字符串不变。

基准选择固定为同铁口 last100、全炉 last100、零回退；不扫描窗口或计数阈值，不递归写回测试预测。开发使用 June–November 六个 cutoff，最终使用认证 2,754 行。预算为 7 个 centered CatBoost、7 个 signed QRF（1,792 树）、0 个新预处理器、0 个校准拟合和两份候选包。A/B 两份包须在任一平台反馈前同时冻结，平台上传必须另获明确授权。

G0 工程、已消费回溯 G1 和平台反馈分别登记。旧模型、旧 QRF 协议、历史产物和失败证据不改写；所有模型、响应、预测、账本和提交包只保存在 private `local/`。

