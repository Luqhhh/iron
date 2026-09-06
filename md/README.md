# 高炉 baseline — AI 主导实施包 v2

本包替代旧版需要用户先填写核心判断的方案。用户已授权 AI 主导本竞赛；D0–D3 和模型、验证、特征、验收配置已经冻结。

先阅读 IMPLEMENTATION_PLAN.md，然后按 TASKS.md 和 AGENTS.md 执行。

本包是实施设计，不含 src 训练程序、真实赛事 CSV、模型或成绩。当前缺少 CSV 与 data_dictionary.xlsx；数据相关字段为 null 是事实缺口，不是待用户批准。

内容：完整方案；AI 工作约束；任务；D0–D3；5份 YAML 配置；规则问题和数据映射契约；审阅报告模板；本包静态校验结果。

把本包文件复制进待建私有代码仓库；decisions/ 在目标仓库放 docs/decisions/。configs/data.example.yaml 应复制为不被Git跟踪的 data.local.yaml 后由 Agent 根据官方材料填写。不要把未实现的 CLI 示例当成已可运行命令。

此版本不执行远程建仓、付费资源创建或赛事提交，也没有运行真实训练。
