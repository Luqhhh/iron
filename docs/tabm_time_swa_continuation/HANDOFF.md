# 给队友：SWA 271828 / 314159 接续确认

用户已授权另行冻结 SWA 接续确认，使用现成 271828/314159，保留原开发与旧协议。
请在已有 SWA 原件的机器上继续。此包是私有实验交接包，包内模型、OOF、报告和
原始绑定应留在 local/；公开更新仅含代码、冻结协议、文档与摘要。

## 包内内容

- `CONFIRMATION_SPEC.json`：新阶段科学范围；首次拟合前再绑定本机运行清单。
- `ORIGINAL_SWA_SPEC.json`、`original_records/`：原7777/12011协议、停止/恢复/开发记录。
- `refs/`：Q75 的42/3407/271828/314159，各2754行、五折，ID字典序排列。
- `evidence/`：98份直接原件，包括原预测、收据、审计、终态与收据直接引用的状态文件。
- `source_snapshot/`：协议冻结的科学源码、依赖锁、保护策略、原SWA训练配置。
- `scripts/verify_swa_continuation_handoff.py`：纯核验与显式ID重排助手。
- `tests/`、`receipts/`：交接相关测试与生产方核验收据。
- `HANDOFF_MANIFEST.json`：所有文件大小/摘要及原绝对路径到包内相对路径的映射。

原始 binding 保持原字节，里面的绝对路径用 manifest 的映射解析；不要编辑原绑定。
这98份证据用于证明现成参照身份；包内没有全部母模型依赖，也没有队友自己的SWA
开发模型、窗口状态、原账本、保护标签或官方CSV。接收方已有SWA原件仍须核验。

## 先验证包和参照

桌面同目录附ZIP SHA256与 `HANDOFF_MANIFEST.sha256`。先校验ZIP，再解压到一个新
私有目录，避免覆盖已有实验；从外部摘要文件复制manifest的64位SHA256到命令中。
在已核验Python3.12环境运行（设置线程必须早于数值库导入）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /path/to/.venv/bin/python scripts/verify_swa_continuation_handoff.py \
  --directory . --expected-manifest-sha256 EXTERNAL_MANIFEST_SHA256
```

验证器检查外部manifest身份、完整文件清单、原协议与冻结源码、98份来源字节、
原实际exit0终态/审计链、20个原预测单位、Q75固定算术和每seed五折覆盖；
不启动科学拟合、不加载pickle、不读取标签值。额外文件会被拒绝，因此验证时不要
在解压目录内创建日志或__pycache__；如从Python导入此助手，设置PYTHONDONTWRITEBYTECODE=1。

程序中调用 `verify_bundle(directory, external_manifest_sha256)` 后，用
`reindex_reference(verified, seed, native_sample_ids, native_folds)` 返回该seed在自身
帧顺序下的时间列和折号。原生折号应由锁定同协议 `make_folds` 生成并按sample_id对齐；
不同样本集合、重复ID、折号不符或未登记seed都会拒绝。禁止把字典序导出直接当原生
行序，或把两个seed向量平均成一个参照。

## 接续执行顺序

1. 在独立分支/worktree保留原科学源码。公开交接分支是
   `codex/tabm-time-swa-continuation`，精确发布提交记在包manifest中。
   核验官方复赛数据摘要与原标签保护规则；先登记访问，再读取获授权标签。
2. 用本机原SWA开发42/3407原件核验freeze、source-recovery、audit、实际控制器终态、
   输入、窗口见证、BASE原source directory/seed/trial身份及追加账本；独立冷回读并重算
   两seed端点增量和机制优势。保留原evaluation，另写新阶段准入收据。若参照时间列与
   本包同seed不一致，停止准入，不替换原开发或重新训练来对齐。
3. 确认原7777/12011确认没有started/reserved拟合、没有竞争任务；原子写独立接续claim、
   新activation和新账本。原历史等待记录保持。合计确认上限40状态/40 optimizer，
   不同时运行原确认切分对。
4. 新建controller，固定新SPEC、本机实际环境/依赖/分区、源码和导入摘要，运行必要
   锁定Python3.12测试；共享editable环境时显式设置PYTHONPATH为当前worktree的src，
   避免导入主仓库模块。旧 `tabm_swa_run.execute/preflight/run_phase` 与旧activate/start
   入口硬编码旧切分或分支，不能直接调用来启动新阶段。
5. 确认科学训练复用冻结的 `tabm_swa_units.train_unit(... phase="confirmation")`，
   两臂均新拟合；设置来自原round2_v12训练配方。按271828再314159、每seed folds0..4
   完成十个单位，共40 optimizer；不重跑开发、不生成新Q75参照。
6. 复用原冷审计的科学检查，在独立新进程核验40模型及窗口状态、分区、选轮、
   预处理、预测和账本。保持 `local/runs/<新阶段>/confirmation-r1` 深度，以适配旧审计
   `d.parents[3]` 的仓库定位；新manifest包含phase/settings/source_hashes等审计所需字段。
   运行每600秒观察；失败保留证据，无自动重试，无时间预算。
7. 完整确认后重算增量、四seed单侧t-LCB95(df3)及四seed机制均值。
   `tabm_swa_run.evaluate`支持显式新seed及development目录参数，复用前先确认输入/文件
   契约满足；其中de3参数未参与计算，时长单目标增量不需要补造铁量OOF。
   原开发evaluation须已独立重算一致。正式门槛为四seed全正、LCB95正、机制均值正。
   结果另写新阶段状态，保留原分类/协议/失败与等待记录。

新阶段身份、固定alpha0.2、十epoch窗口、预算和容差以SPEC为准；本次没有授权
SWA全量拟合、生成平台提交包或助手上传。生产方交接G0通过不代表接收方原开发
冷复现通过，也不代表四seed晋级；当前只有队友公开报告的两seed开发正收益。

## 可直接交给接续助手的任务

请读取本交接包README和CONFIRMATION_SPEC，先纯核验包内四seed Q75参照及本机已有
SWA开发/账本/恢复桥，保留原42/3407开发与7777/12011等待记录。准入闭合后，新建
独立controller与运行清单，只执行271828/314159的BASE/SWA五折确认，0参照拟合、
0开发重跑、最多40 optimizer，单worker/单数值线程、1024MiB、600秒观察、无时间预算、
无自动重试。完成独立冷审计后计算四seed门槛；不要在本次任务全量拟合或封平台包。
