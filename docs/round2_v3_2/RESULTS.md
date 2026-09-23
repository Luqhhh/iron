# Round2 V3.2 Strong-Ensemble and Target-Expression Search

Status: **protocol patch completed; six-recipe seed ensemble and 256-item search not yet executed; no package and no upload.**

Base commit: `4b87834186a892a9299abf965973a12d74a5904a`.

## Current references

| reference | definition | local score |
|---|---|---:|
| P0 | AJ3 iron + full B3 time | 96.016655 under S3 protocol; 96.1259 user-reported platform |
| L0 | V3 frozen weighted fusion | 96.118109 under S3 protocol |
| L1-procedure | V3.1 S3 candidate member pool and inner LP procedure | 96.164036 under S3 protocol |

The V3.1 S3 candidate remains stronger than both P0 and L0 on the same
sample/duplicate-group-isolated outer protocol.  New V3.2 candidates must report
delta versus all three references; local gains are not platform predictions.

## Protocol patch

Implemented in `src/bf_tap_r2/v3_1_models.py` under protocol version
`v3.2-index-aware-v1`:

- CatBoost and XGBoost best iteration is treated as a zero-based index;
  a valid index `k` maps to `k+1` boosting rounds;
- index `0` is a valid best index and maps to one round, not to the configured
  maximum;
- LightGBM uses its retained iteration count directly;
- missing/negative best index falls back to the frozen configured round count;
- every early-stopped tree fit now records:
  - `best_iteration_index`
  - `selected_num_boost_round`
  - `actual_num_boost_round`
  - `configured_num_boost_round`.

Regression tests are in `tests/test_round2_v3_2_protocol.py` and cover index 0,
middle/last indices, no valid index, and LightGBM/XGBoost differences.

Synthetic boundary cases from the task book are recorded privately in:

`local/runs/round2-v3.2-ensemble-and-target-search/protocol-patch-r1/protocol_patch_report.json`

The historical V3/V3.1 scores are not retroactively invalidated by this patch.

## Next execution blocks

1. six base recipes x fixed training seeds `[42, 2026, 2027]`;
2. 256 new configurations by the frozen target allocation;
3. paired P0/L0/L1 replays under the corrected training protocol;
4. two-fold coarse screen, then seeds 42/3407 full OOF;
5. one final outer split (suggested seed 9091) for at most two frozen candidates.

No V3.2 package has been generated, no existing prepared ZIP has been modified,
and no platform upload has occurred.
