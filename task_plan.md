# optimization-v0.2 工作状态

## Objective

依据 spec.md 和 plan.md，在冻结 baseline-v0.1 的基础上，用非保护训练标签构建并验证 optimization-v0.2 候选模型。

## Current Phase

Phase 14 完成 — E09 全阶段提交物已生成并完成最终核验

## Phases

- [complete] Phase 0: spec 与实施计划审批
- [complete] Phase 1: 隔离分支、配置和实验入口
- [complete] Phase 2: 测试先行
- [complete] Phase 3: rolling OOF 对照
- [complete] Phase 4: M1/M2/M3 候选实现与本地 M4 门控
- [complete] Phase 5: 模型选择和冻结
- [complete] Phase 6: Python 3.12 权威验证与报告
- [complete] Phase 7: 改名本地分支，备份未提交改动，并合并 origin/optimization-v0.2
- [complete] Phase 8: 比较本地 optimization_v02 与远端 optimization_v0_2；结论为不一致，不归并
- [complete] Phase 9: 分析 OOF/确认结果、误差结构与漂移，给出下一轮优化方案
- [complete] Phase 10: 固化 optimization-v0.3-drift 设计与可执行计划
- [complete] Phase 11: 测试先行实现统一验证合同与 C1
- [complete] Phase 12: 测试先行实现 C2 隔离迁移和切片报告
- [complete] Phase 13: Python 3.12 全量验证与五起点真实运行
- [complete] Phase 14: 结果复核、G0/G1 判定、报告与 E09 全阶段发布

## Decisions

- 主指标：E；同时报告两个目标 WMAPE。
- baseline-v0.1 保持不可变。
- 首轮保持现有 as-of 特征集合，先做模型层优化。
- 不读取 2024 年 11 月保护标签。
- 详细设计见 spec.md；实施步骤见 plan.md。
- 新模型按 optimization_v02/m1_blend、m2_recency、m3_residual、m4_ensemble 分级归类。
- commit 与 push 的时点由用户决定；未经明确指令只保留本地改动。
- 当前本地分支改名为 codex/optimization-v0.2-local，再合并远端分支；合并前必须保留可恢复 stash。
- 两套优化方案不完全一致；保留 optimization_v02 与 optimization_v0_2 两套目录，不做有损归并。
- optimization-v0.3-drift 使用独立分支，复用公开开发数据和同一 as-of 构建器；不改 baseline-v0.1、v0.2 模型或既有运行证据。
- C1/C2 先在同一 5-origin×H1-H4 网格独立验证；仅两者均通过时才执行预注册的第二阶段组合。

## Errors Encountered

| 错误 | 尝试 | 处理 |
|---|---:|---|
| unified exec helper setup refresh failed | 2 | 使用经批准的 WSL 只读/补丁命令 |
| 内置 apply_patch 无法跨 UNC 写入 | 4 | 本次仍失败；继续使用 Git zero-context 补丁 |
| WSL 中无 apply_patch 命令 | 1 | 使用 git apply 应用等价新文件补丁 |
| WSL 中无 rg 命令 | 1 | 按仓库约定回退到 grep 完成占位符扫描 |
| 手写 git apply hunk 行数不一致 | 5 | PTY 或计数破坏输入后，改用 zero-context 补丁 |
| 编排隔离环境无 TextEncoder/btoa | 2 | 使用纯 JavaScript UTF-8 与 base64 编码 |
| git hash-object 不支持 -z | 1 | 路径无空格，改用换行分隔并完成 blob 哈希核对 |
| 两路并行只读审查触发账户额度上限 | 1 | 不重试；由主流程基于已读取源码、配置和测试完成比较 |
| Windows 命令行长度上限 | 1 | 改为程序化生成最小 hunk 补丁 |
| 复合并行读取超时 | 1 | 终止只读会话，改用较小的逐文件并行读取 |
| git diff --check 检出 Markdown 行尾空格 | 1 | 清理后重新暂存并复检 |
| 系统 Python 3.12 缺少 Python.h | 1 | 使用 uv 托管 CPython 3.12.13 |
| 托管环境默认未安装 pytest | 1 | 按 pyproject 增加 --extra dev |
| PowerShell 提前解析 shell 表达式 | 2 | 改由编排层传递绝对路径，禁止命令替换 |
| 系统 Python 3.12 缺少 Python.h | 1 | 使用 uv 托管 CPython 3.12.13 |
| 托管环境默认未安装 pytest | 1 | 按 pyproject 增加 --extra dev |
| PowerShell 提前解析 shell 表达式 | 2 | 改由编排层传递绝对路径，后续禁止命令替换 |
| 远端候选配置文件路径猜测错误 | 1 | 先列目录，再读取 experiment/features/acceptance/validation 实际文件 |
| 最终只读核对并行审批超时 | 1 | 拆为单个小命令后重算成功 |
| WSL 无法解析技能脚本 CRLF | 1 | 改用 PowerShell 版本；其旧格式解析显示 10/0，但计划内无 in_progress 阶段 |
| 内置 apply_patch 再次无法读取 UNC 工作区 | 1 | 改用已验证的 Git 零上下文补丁流 |
| PowerShell 提前展开 WSL 循环变量 | 1 | 改用显式文件名，不在 PowerShell 命令字符串中使用 `$f` |
| 并行只读检查审批超时 | 1 | 未重复并行审批；拆成单个只读命令后完成 |
| CLI 首个 context patch hunk 计数错误 | 1 | 改为程序化生成零上下文插入补丁 |
| v02/v03 同名测试模块 collection 冲突 | 1 | 复现确认 pytest 顶层模块名碰撞；为 v03 测试目录增加包边界 |

