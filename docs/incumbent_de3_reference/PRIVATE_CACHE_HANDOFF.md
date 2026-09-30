# DE3 私有缓存交接清单（2026-09-30）

目的：复用已经完成的开发拟合，补齐后续 PTaRL/DNNR/DANet 的当前参照。
这不是请求重新训练，也不授权发布私有数据到 Git。

## 请队友共享的目录

以下均为队友 iron 仓库根下的相对路径，保持完整目录结构：

1. `local/runs/independent-ensemble-checkpoints-20260929/development-DE3/`
   包括 manifest、summary、audit、全部单元 metadata/完成摘要、selector/refit 保存状态、预测、fit_ledger及审计依赖。
   这是42/3407完整开发缓存，不能用全数据 release 模型或 submission.zip 替代。
2. `local/runs/strong-component-regularization/development-r2/`
   包括 manifest、summary、audit、audit-recovery-provenance，以及被 DE3 复用的 BASE 铁量单元和 reference 单元、保存状态、预测及完成摘要。
   若无法准确裁出依赖，共享整个目录；无需额外拟合。
3. 上述 manifest/source/reference_hashes 指向的 **Git 外私有依赖**，保持相对路径。
   请附 SHA-256 文件清单：相对路径、字节大小、SHA-256；递归追踪 manifest/audit 引用，
   不只打包两张汇总表。公开 src/configs 的历史版本由 Git 取，不要用另一版本文件冒充原始源。

已有冻结外部锚点：

| 文件 | SHA-256 |
|---|---|
| development-DE3/manifest.json | 25713e219cd2da36afedfaa41f5ffa453334353f2ff08aace94f2303618b33c9 |
| development-DE3/summary.json | cd1d4e7740b7e2799e66f29edc4dab60a4cc3801904e434151ef0e1c1d0bd7c8 |
| development-DE3/audit.json | d2e590109c39ff94ca715b1710edc1805ec96f674f433d8c80a012c5ee00b81c |
| development-r2/manifest.json | cc1f3a9bd141c7c4d20d6dcc7cb129baee0bf8db933462a3fc7081dd3463c933 |
| development-r2/summary.json | e223e4389950cd351a24bcaf70ac5236cb97882713a09b6e18de97efb38e4ace |
| development-r2/audit.json | bc14b82bb1442224d1a77b485a8f0ac0064d1387d797ec28a8fe13570369bcb2 |
| development-r2/audit-recovery-provenance.json | 42fb38bd22ebc2c911063744a23efbf2e575736fb3f169eac15ada2629b50d6f |

锚点取自6e285fa的configs/de3_user_release/RELEASE.yaml与
configs/independent_ensemble_checkpoints/SPEC.yaml。
未知文件不能根据命名猜测哈希；以原 manifest/audit 的完整引用为准。

## 传入本机

请先放到一个全新的交接目录，例如：
`\\wsl.localhost\Ubuntu\home\clairvoyant\code\iron\local\incoming\de3-cache-20260930-r1\`

目录内保留从 `local/runs/...` 起的相对路径，附清单。不要直接覆盖本机原始缓存。
确认完整性后才建立只读引用根或逐文件核验副本。绝不将 local/ 从 .gitignore 删除。
文件可私下共享；完整模型/预测不要放入公开仓库。

## 收到后的零拟合核验

先核对上述外部锚点、所有依赖哈希、原源及版本、数据摘要、同seed完整五折、
训练/查询ID与顺序、selector/fresh-refit身份、账本及冷预测。
本机已有的 V7/V12/V5/V8 四seed缓存另按原审计核验；不凭文件名替换。
若原缓存只有42/3407，不称四seed完成，也不跨划分平均预测补列。
如队友已补建 DE3 四seed overlay，则共享完整 overlay、audit、complete.json 和外部摘要，
先核验其20个新估计器/40次优化的源及预算，避免再补建同一批。

若尚无 overlay，拟议缺项仅为铁量原生联合模型的训练种子104729/130363，
划分7777/12011各五折：20估计器、40次优化（内层选轮+新初始化外层重训）。
原J42和42/3407开发成员复用；保持联合标签和原参数，仅取铁量输出用于参照，时长父列不变。
此补齐服务于当前incumbent参照，不改变DE3旧晋级决定。
该正式预算仍待完整阶段规格/计划与准入，不因本交接清单而自动开始。
无全数据拟合、包、桌面写入、上传或新候选选择。
