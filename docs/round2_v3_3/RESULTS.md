# Round2 V3.3 Structural Search

Status: **branch created; full-recipe residual wrapper implemented and tested; original-unit tree selection and 192-item structure search not yet executed; no package and no upload.**

Base commit: `2e1a419ed85483a6a924214240f300f10eba9698`.

## References

| reference | definition | local score |
|---|---|---:|
| P0 | AJ3 iron + full B3 time | 96.016655 under S3 protocol |
| L0 | V3 frozen weighted fusion | 96.118109 under S3 protocol |
| L1 | V3.1 frozen procedure | 96.136206 under outer seed 9091 |
| V3.2 material | V3.2 fusion | 96.140023 under outer seed 9091; delta vs L1 +0.003817 |

V3.2 is retained as fusion material, not promoted to L2. The five prepared V3.1
ZIPs remain unchanged and unauthorized for upload.

## Completed block: full-recipe residual wrapper

New module:

`src/bf_tap_r2/v3_3_residual.py`

The wrapper preserves the complete `base_trial` dictionary, including:

- feature set;
- target transform;
- base loss;
- seed and training protocol.

Base predictions are restored to original units before the residual is built.
Residual labels may be signed.  Ridge and shallow-CatBoost correctors are
supported with a fixed correction coefficient, and all inner residuals come
from group-aware internal folds.

Tests:

`tests/test_round2_v3_3_structure.py`

Coverage includes:

- `alpha=0` prediction equivalence with the independent parent recipe;
- signed residuals in both directions;
- raw and four-feature plus log1p parent semantics;
- rejection of an incomplete base trial.

## Remaining V3.3 blocks

1. original-unit tree-count selection bridge and the 16 paired checks;
2. 64 feature-group deletion/compression items;
3. 64 EBM items (dependency availability must be checked honestly; no silent
   spline substitution);
4. 48 full-recipe residual items using shared base OOF;
5. optional at most 32 conditional items;
6. at most two frozen procedures and one final outer validation seed 12011.

No V3.3 package has been generated, no prepared ZIP has been modified, and no
platform upload has occurred.
