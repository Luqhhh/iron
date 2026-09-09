# Progress

## 2026-09-07

- 用户同意以 E 为主指标，并独立报告 tap_iron 与 tap_time_len WMAPE。
- 创建 plan.md 与 spec.md 草案。
- 校验两个文件为 Git 未跟踪新文件，git diff --check 通过。
- plan/spec 占位符扫描通过；没有 TBD、TODO 或省略性实施项。
- 用户已批准 spec.md。
- 用户选择直接在 iron 目录开发，不创建 worktree。
- 已建立并切换到 codex/optimization-v0.2。
- 新模型将按 M1/M2/M3/M4 分级目录组织；尚未修改模型实现。
- 用户明确：commit 与 push 时点由用户决定；当前没有 commit 或 push，规划文件已撤销暂存。
- 已完成 baseline 配置、模型、验证、指标和时间切分接口的第一轮只读映射。
- 锁定 Python 3.12.13 基线测试：50 passed in 10.81s。
- 完成 Phase 1：分级 registry、严格配置、三个滚动折和不可覆盖 run 初始化器。
- Phase 1 targeted：11 passed；全量回归：61 passed in 8.26s。
- 所有改动保持未暂存、未 commit、未 push。
- 完成 OOF pooled/逐折评分和 M1 目标级融合模型及安全选择 runner。
- M1/OOF 集成测试通过；全量测试 70 passed in 9.33s，私有产物守卫 PASS。
- 仓库内没有本地数据配置或 CSV/XLSX；真实 rolling OOF 等待数据路径。
- 已重新 fetch origin/main；main 无新增提交，快进合并返回 Already up to date。
- 已获取 origin/optimization-v0.2 引用，确认 c7b1fb8 包含训练/测试数据和远端优化实现。
- 用户要求将当前分支改名为 codex/optimization-v0.2-local，并安全合并该远端分支；不 commit、不 push。
- 合并预检通过：远端可快进、本地未跟踪路径无碰撞、tracked 差异仅为行尾表现，并记录本地文件聚合哈希。
- 创建 stash@{0}（d78cdaf…），完整保存 tracked 行尾状态及 28 个未跟踪文件。
- 当前分支改名为 codex/optimization-v0.2-local。
- origin/optimization-v0.2 已快进合并至 c7b1fb8，数据和远端优化实现均已到位。
- 28 个本地文件已从 stash 的 untracked tree 精确恢复；路径与 blob 聚合哈希一致，提交区为空，未 commit、未 push。
- 最终验证：86 passed in 11.15s；私有产物守卫 PASS；HEAD 与 origin/optimization-v0.2 同为 c7b1fb8。
- 用户要求分析两套优化方案；完全一致时才归并为远端 optimization-v0.2 的统一目录与分支。
- 已启动本地/远端两路只读审查；结构映射已发现候选命名、算法族和实现成熟度存在实质差异，暂不归并。
- 配置与核心实现核对确认：融合权重策略、M2/M3 算法、远端特有候选、验证网格和验收阈值均不一致。
- 两路并行审查因账户额度上限中断；未产生文件改动，改由主流程完成剩余核对。
- 主流程确认两套方案仅共享冻结 baseline/as-of/保护边界；候选算法、验证合同和实现成熟度均不同。
- Phase 8 比较完成：判定不完全一致。
- 未移动、覆盖或删除 optimization_v02 / optimization_v0_2；未 commit、未 push。

## 2026-09-08

- 用户批准继续实现 M2/M3/M4，使用真实训练集 OOF；M4 仅组合本地 M1/M2/M3，远端 E09 仅作外部参考。
- 完成 M2 30/60/120 天衰减、M3 因果 expanding anchor 残差模型、M4 定量门控与冻结权重。
- 完成受保护开发数据读取、三折 OOF、选择冻结、DEV_LONG/SHORT 确认和无训练标签文件的离线重放。
- 端到端测试覆盖 11 月哨兵标签、不可覆盖运行目录、冻结选择和离线重放。
- 修复选择快照中的整数网格键经 JSON 往返导致摘要不一致的问题；未改变模型或选择规则。
- 锁定 Python 3.12.13 全量测试：95 passed in 33.63s；私有产物守卫 PASS（175 tracked files）。
- 正式真实数据运行已启动：local/optimization_v02_runs/m234-real-oof-20260908-01；运行期间不修改源代码或配置。
- 独立只读代码审查因账户额度中断；主流程完成代码、测试和证据人工复核，未发现需重跑的实现缺陷。
- 正式运行完成：915 条 rolling OOF；G0 PASS，G1 REVIEW_REQUIRED；M1 为唯一通过定量门槛的候选，M2/M3 FAIL，M4 SKIPPED。
- M1 冻结权重：tap_iron=0.5 CatBoost+0.5 B1，tap_time_len=1.0 B1；选择摘要 e2b3fbb451c087d1501714657ce2b0d3548f86cd624dadff63002a560876dc96。
- DEV_LONG/DEV_SHORT 在选择冻结后运行且未调权；结论为有条件改善，不判定 G1 PASS，不运行保护 holdout。
- 真实首折离线 bundle 重放与 OOF 最大绝对差 5.68e-14。
- 结果报告写入 docs/optimization_v02/RESULTS_SUMMARY.md；所有实现仍未暂存、未 commit、未 push。
- 用户要求进一步分析模型结果并给出优化方案；进入只读诊断阶段，不改模型代码、不访问保护标签。
- 首次读取 as-of 审计字段时 PowerShell 提前展开 WSL 循环变量导致失败；改用显式路径继续。
- 远端 E09 结果已复核；候选配置文件路径首次猜测错误，改为先列目录后读取实际文件。
- 完成目标分布、预测跨度、折/出铁口/时段/分位、偏差、残差相关、M1 留一折权重、M2 半衰期、特征重要性和按日块 bootstrap 分析。
- 下一轮建议收敛为 optimization-v0.3-drift：优先测试 M1铁量+30天时长基线，以及本地 E09 隔离迁移；M2 降级、M3/M4 暂停。
- Phase 9 完成；未修改模型代码、配置或冻结运行产物，未访问 2024 年 11 月保护标签。
- 最终从行级 OOF 重算与保存指标最大绝对差为 0；当前 src/configs 摘要与正式运行完全一致，状态仍为 G0 PASS / G1 REVIEW_REQUIRED。
- 技能检查脚本的 WSL 版本因 CRLF 失败；PowerShell 版本执行成功但不识别现有 legacy `[complete]` 记法，显示 10/0，task_plan.md 实际无 in_progress 阶段。
- 用户批准 optimization-v0.3-drift 方案并要求先写 plan、随后启动真实运行。
- 沿用用户既定选择：直接在 iron checkout 工作，不创建 worktree；暂不 commit 或 push。
- 已从 codex/optimization-v0.2-local 创建并切换到 codex/optimization-v0.3-drift；所有 v0.2 未跟踪实现与证据保持原位。
- Phase 10 已启动：映射远端验证器、E09 特征路线、本地 M1 冻结接口与真实数据入口。
- 完成 v0.3 接口映射：统一网格可复用 optimization/run.py 的 origin 分组；C1 需要新增目标级派生候选，C2 可复用 E09 特征路线。
- 识别隔离要求：不能把现有 optimization_id=v0.2 合同原地改成 v0.3，需新增命名空间或专用配置/适配层。
- 写入并自检 `docs/superpowers/specs/2026-09-08-optimization-v03-drift-design.md` 与 `docs/superpowers/plans/2026-09-08-optimization-v03-drift.md`；Stage 2 路由和触发门已在看结果前冻结。
- 计划占位符/格式扫描完成；修正通用测试路径占位，接口、任务依赖和规格覆盖一致。
- Phase 10 完成，进入 Phase 11 TDD；所有改动仍未暂存、未 commit、未 push。
- Task 1 配置 RED：`test_config.py` 因 `bf_tap.optimization_v03` 不存在而按预期 collection error。
- 配置 GREEN：16 passed；冻结 ID、核心候选、30 天窗口、M1 权重、Stage 2 前置项和 bootstrap 参数均受严格合同约束。
- Task 1 候选 RED：`test_candidates.py` 因候选模块不存在而按预期 collection error。
- 首次 GREEN 暴露 C2 alias 的 sample_id dtype 未保持（1 failed, 4 passed）；在键校验后恢复 eval ID dtype。
- Task 1 最终 GREEN：21 passed in 0.79s；未暂存、未 commit、未 push。
- Task 2 RED：`test_diagnostics.py` 因 diagnostics 模块不存在而按预期 collection error。
- Task 2 GREEN：3 passed in 0.80s；覆盖 required slices、scenario-key 去重和 deterministic paired daily-block bootstrap。
- bootstrap 仅作为诊断证据，不进入既有 acceptance 门。
- Task 3 RED：`test_run.py` 因 run 模块不存在而按预期 collection error。
- Task 3 GREEN：7 passed in 1.35s；Stage 2 双门、Stage 1 专属 G1、C2/E09 exact equivalence 和不可覆盖输出已锁定。
- 当前 v0.3 全集：31 passed in 1.29s。
- runner 已接通 suite=all 的 E00/E09 核心、因果控制派生、丰富切片、scenario summary、bootstrap 和双层 G0/G1 清单；尚未执行真实数据。
- Task 4 CLI RED：2 failed，分别为 argparse 不识别 optimize-v0.3 与缺少 runner import。
- 首个 CLI context patch 因 hunk 行数错误未应用；改用程序化 zero-context insertion 后成功。
- CLI GREEN：2 passed in 0.81s。
- 组合回归首次因 v02/v03 的同名测试文件都被导入为顶层模块而 collection error；新增 v03 测试包边界后，20 tests collected，根因修复。
- 两套 optimization 组合回归：78 passed in 33.72s；Phase 12 完成，进入权威验证与真实运行。
- 真实训练前锁定 Python 3.12.13 全量回归：128 passed in 42.97s。
- 私有产物守卫：PASS（190 tracked files）。
- 已启动独立只读代码审查；真实训练等待 Critical/Important 结论，避免昂贵重跑。

## 2026-09-09

- 终端恢复后完成 E09_PROCESS_CHANGE_E02 的全阶段发布训练，训练截止点固定为 2024-11-01，protected_labels_read=false。
- 在新目录 `local/optimization_v02_runs/release-e09-full-20260909-01` 生成 test_a/test_b/test_c 预测，行数分别为 335/322/548。
- 三个 `check-submission` 均 PASS；在 `local/submissions/release-e09-full-20260909-01` 生成 prelim/round2/semifinal 三份 ZIP，ZIP 根目录均仅含 `result.csv`。
- 锁定 Python 3.12.3、pytest 9.0.2 全量测试：137 passed in 34.98s。
- 提交证据已追加至 `docs/submission_log.md`；未覆盖任何既有 run 或 submission。
