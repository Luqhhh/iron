# Four screening diagnostics delivered

The user explicitly requested the four selected diagnostic releases on 2026-09-30.
All four are delivered to `C:\Users\lqh22\Desktop\submission-local-platform-diagnostic-20260930`.
Each numbered subdirectory contains `Luqhhh_bf_tap_predict_round2.zip`; the root README lists identities and the interpretation reference.

Compare method gains against **V32_TIME_A60V7_50 = 96.3727**, the exact common parent and local reference.
Current user-reported best remains **DE3_IRON_USER_REQUESTED = 96.3749**; beating it is a separate comparison.
Scores are user reports, not independently verified receipts. No diagnostic platform score has been returned.

| # | Candidate | Target | Local mean gain | Selected epoch | ZIP SHA-256 |
|---|---|---|---:|---:|---|
| 1 | LOC_DIAG_EMA_TIME | tap_time_len | +0.002168812 | 91 | `da3b0c79aac7275d2084f21d291e632eb376a8680491c458b6d927e56f4d3836` |
| 2 | LOC_DIAG_LMIX_IRON | tap_iron | -0.001437523 | 74 | `a167927ce315d0179b0fe8769282e44ee129737e8ab94af0188e3abb485e8926` |
| 3 | LOC_DIAG_SAM_IRON | tap_iron | -0.003270936 | 90 | `a39f245adf908a9091442f89a9f2406d1b2137092186a9fe2481422a4417fdc2` |
| 4 | LOC_DIAG_SAM_TIME | tap_time_len | -0.016429659 | 109 | `20557f4363377dfb693215baa86a8990cd126e0af3b30eaa33a08b5f3767efe2` |

## G0: passed

Four full-data estimators/eight optimizer runs, zero new CV fits or derived confirmation seeds.
Reused the original full-data V12 iron and V7 time models; both original predictions replay exactly.
All eight saved selection/refit models pass independent train-only preprocessing, target scaling,
epoch-selection and checkpoint audits. Original settings, seeds, epoch policy and mechanism constants are unchanged.

Each package replaces exactly its designated component using `parent + 0.5*(new-old)`.
The other target field strings are byte-identical to V32; numeric replacement readback is exact.
No clipping was needed; negative replacements would have been refused rather than changing the local experiment.

Fresh cold processes prohibit training reads and reproduce warm predictions exactly (maximum difference 0).
Maximum reversed/chunked/singleton raw difference is 2.8087603482163104e-5, within the original 5e-4 tolerance.
Independent package auditors reload each candidate and original native state using the respective unchanged
historical worktree, then recompute the delivered CSV arithmetic within 1e-10. All four ZIPs contain only
result.csv, have valid CRC, and contain 322 unique IDs in template order with finite nonnegative predictions.
Desktop SHA-256 and CSV payload readback passed for all four.

The frozen manifest anchors 1,427 files; every anchored data, source, parent and historical development
artifact remained unchanged. The manifest and all reports, ledgers, models, predictions and packages remain
private under `local/runs/local-platform-diagnostic-20260930/release-r1`.

Locked Python3.12 `uv run --locked --no-sync --python 3.12 pytest -q`: **1315 passed**, 23 existing warnings.
Focused release/component tests: 22 passed. The first new test fixtures used fewer than the mandatory 322 rows
and failed before exercising replacement arithmetic; corrected synthetic fixtures and the full rerun passed.
That initial test failure is retained under local/tmp/local-platform-diagnostic-20260930; no model or scientific
recipe changed. Private-artifact guard passed before implementation commit and push.

## G1: pending platform feedback

These are explicitly requested diagnostic exploration releases, not four-seed promoted candidates.
Original no-finalist decisions, tiers, +0.01 mechanism gates and release rules remain unchanged.
EMA time tests small consistent positives; D-LMIX iron tests a near-zero mean with disagreeing splits;
SAM iron tests small consistent negatives; SAM time tests larger consistent negatives.
A platform result establishes that candidate’s effect on the fixed test set, not an overall false-negative rate.

Implementation/preregistration commit: `e6a5e42`. Public machine status:
`EVIDENCE_STATUS.json -> round2_local_platform_diagnostics_20260930`.
Users upload and return scores by full candidate name; agent uploads **0**.
Earlier chord-search packages and all old evidence remain preserved. This diagnostic batch takes the four
user-selected slots; it is not the older score-seeking upload queue.
