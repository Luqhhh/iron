# V49 G0 admission

Locked Python 3.12 suite: **1289 passed**, 23 existing warnings. Runtime lock and
20 complete current-reference cache units verified; no new reference fits.
Two frozen, full-size 240-epoch synthetic fits completed. These test engineering
and resource feasibility, not actual-data quality or expected platform score.

| Arm | Query MAE | Constant MAE | Seconds | Peak RSS MiB | Auxiliary updates |
|---|---:|---:|---:|---:|---:|
| BASE | 0.1824432824 | 2.6377198601 | 273.287 | 753.58 | 0 |
| TANGOS | 0.1876134716 | 2.6377198601 | 347.687 | 809.42 | 540 |

Conservative four-worker development projection: **3.863190 hours**, below 7.61.
Maximum measured worker RSS 809.42 MiB <1536; available-memory reserve passed.
Fresh-process, reversed and chunked prediction differences: `{'BASE': 0.0, 'TANGOS': 1.7763568394002505e-15}`.
Training-only preprocessing, scale, saved states, epoch traces and exact auxiliary
update counts verified. Regularization active/nonzero; no requirement or claim that
it beats the control on this toy. No synthetic-based tuning.

Independent post-completion review reconstructed synthetic query errors, constant
error, fixed epoch counts, timing projection, source hashes and runtime identity.
Actual-data G1 remains unmeasured. Admit 40 outer units /80 fits; conditional
confirmation remains gated by the frozen specification. No full-data fits,
packages, desktop writes or uploads. Monitor every 600 seconds; pause after V49.

Private report SHA-256: `3d07270cc28e46399045b5e7d7d16f9dbe8156fb5da0aa37dc93ab1169728fe8`.
Private evidence: `local/runs/round2-v49/preflight-r1`.
