# TABM_CORRECTION_FUSION_V1 results

G0 passed: original source/reference and complete cold-audited units were hash verified; both methods had identical sample IDs, folds, parent and raw BASE predictions. Each seed covered 2754 unique IDs exactly once. Old fit/access ledgers remained byte unchanged. Production froze before cached label access. Independent new-process scalar arithmetic differed by at most 2.842170943040401e-14. Neural and locked Python 3.12 each passed 5 targeted tests. No new models, inference or optimizers. Peak RSS 106.26953125 MiB.

G1 is post-selection exploration on four previously used splits. No new independent confirmation or formal promotion. Current platform Q75=96.3920 is user reported, unverified. No platform score inferred.

| Candidate | Four-seed mean gain vs Q75 | Minimum | Descriptive LCB95 |
|---|---:|---:|---:|
| BOOT_DELTA10 | +0.001020291325 | +0.000441141382 | +0.000412815873 |
| BOOT_DELTA20 | +0.001192146286 | -0.000462535244 | -0.000336644934 |
| COV_DELTA10 | +0.000516687111 | -0.000500204353 | -0.000409548728 |
| COV_DELTA20 | +0.000412940717 | -0.001351075208 | -0.001339296576 |
| MIX50_A20 | +0.004102669673 | +0.003277209814 | +0.003081174811 |
| BOOT_A20 | +0.004259479180 | +0.002350669349 | +0.001941404281 |
| COV_A20 | +0.002991348251 | +0.002500456449 | +0.002486665179 |

Frozen ranking retained only MIX50_A20: 0.8 Q75 + 0.1 Bootstrap + 0.1 covariance. Its mean is 0.000156809507 lower than existing Bootstrap A20, while minimum seed gain and descriptive LCB are higher. This is evidence of a stability tradeoff, not proof of platform superiority. Difference-injection candidates remain recorded; positive local gains do not establish platform value. Bootstrap and covariance paired correction correlations ranged from 0.315 to 0.504.

Fold and spout descriptions are private and did not affect selection. Private evidence: local/runs/tabm-correction-fusion-v1/{preflight,evaluation,independent-audit,spout-description-r1}.json; vectors, labels and append-only access ledger stay local/. No fullfit, package, upload or desktop write.
