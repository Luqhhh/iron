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


## Original-unit tree selection: 16 paired checks

`src/bf_tap_r2/v3_3_metric.py` implements Mode B:

- fit the inner CatBoost model without early stopping;
- evaluate a fixed tree-count checkpoint grid on the inner validation split
  after inverse-transforming predictions and labels;
- select the tree count by original-unit WMAPE;
- refit on the full outer-training part with that tree count.

Mode A is the inherited library-metric early-stopping path.

Eight strong backbones were compared on split seed 42, folds 0/1, with
`delta = ModeB - ModeA` (negative is better):

| backbone | Mode A pooled WMAPE | Mode B pooled WMAPE | delta | selected trees |
|---|---:|---:|---:|---|
| iron 0018 | 0.0373927 | 0.0374876 | +0.0000949 | 1900, 1900 |
| iron 0012 | 0.0372709 | 0.0373409 | +0.0000699 | 1100, 2500 |
| iron 0019 | 0.0370291 | 0.0372087 | +0.0001796 | 4000, 4000 |
| V3 iron 0107 | 0.0378006 | 0.0377827 | -0.0000179 | 900, 1000 |
| time 0050 | 0.0384378 | 0.0384741 | +0.0000363 | 1000, 1700 |
| time 0031 | 0.0382225 | 0.0382177 | -0.0000049 | 700, 700 |
| time 0126 | 0.0380222 | 0.0380094 | -0.0000127 | 1600, 1200 |
| time 0137 | 0.0379295 | 0.0378971 | -0.0000323 | 1000, 1300 |

The bridge is implemented and identity fields are recorded, but the paired
check shows **no consistent original-unit selection gain**: four backbones
improve slightly, four degrade slightly, and all differences are below 0.00018
WMAPE.  Mode B is therefore not automatically frozen for this batch; it remains
available for interaction with the structural search lines.
