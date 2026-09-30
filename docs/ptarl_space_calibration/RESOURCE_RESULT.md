# PTaRL resource admission declined — 2026-09-30

The declared synthetic full-size resource probe completed all1 paired unit,
6 optimizer starts and2 KMeans calls. All reservations completed, none failed
or incomplete. Learning, cold-inference and memory gates passed; **development
cost failed**. This is a G0 resource result, not an official G1 quality result.

| frozen check | measured result | outcome |
|---|---:|---|
| projected development time |15949.5434s (4.43043h), limit7200s|fail|
| peak worker RSS |961.890625MiB, limit1536MiB|pass|
| available memory |9511.1875MiB, require4871.5625MiB|pass|
| independent cold difference |1.77636e-15, limit1e-8|pass|
| CONTROL held-out synthetic MAE |0.344535946 vs median1.450194325|pass|
| PTARL_AUX held-out synthetic MAE |0.661189906 vs median1.450194325|pass|

The cost formula was independently recomputed from the measured per-role
seconds/update and frozen240-epoch/four-worker ceiling and reproduces the
admission result. The258.748s actual single-pair probe duration is not the
full-development duration. Source841 hashes, original architecture, budgets,
caps and all candidate-quality gates remain unchanged.

Service iron-ptarl-preflight-r1 endedexit1 due to the deliberate rejection.
The scheduled observation at02:44:26UTC confirmed MainPID0/failed. Preserve
`local/ptarl-space-calibration-r1/preflight` including measurement, all models,
ledger, admission and failed record. Admission SHA-256:
`e0b26300ceee5e3b20946ec55756ddcdb5ea9afc33ef3d45d51d2b9f0c1e8380`. Private independent resource receipt:
`local/research/ptarl-current-reference-r1/resource-result.json`
(SHA-256 `7ce7cb6977dfc34a4547ae9ff576b3f6e461cffa47d8ea27ff772a2cfb5ea328`).

No formal PTaRL development/confirmation fit, full-data fit, package, desktop
write or upload occurred. The failed admission prohibits invoking the formal
controller. Do not lower model capacity/epochs, widen the7200s cap or repeat
the probe to seek a favorable measurement. The current DE3 four-seed reference
is complete and remains reusable by the next independent family.

Synthetic AUX being worse than CONTROL is recorded but does not establish
competition-data ranking. Current platform best remains user-reported DE3
96.3749; the three milestones remain unmet. Proceed with DNNR numerical/model
preparation, then DANet, retaining the serial ordering and frozen quality gates.
