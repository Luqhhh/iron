# V23_HIST: frozen histogram-target experiment — negative result

Branch: `codex/round2-v23-histogram-target`. Reservation: `c7361dd`.
Frozen design: [PREREGISTRATION.md](../round2_v23/PREREGISTRATION.md),
`configs/round2_v23/SPEC.yaml`. This is the histogram experiment, separate
from the other remote branch's V23 capacity probe (`b2ec1bb`) and V24 log/sqrt
probe (`ed5e9a7`). No other branch's decisions or fitted artifacts were replaced.

## Decision

**No finalist, no promotion, no confirmation fits, no full-data fit, no package
and no upload.** All 40 development fits completed without failure.
Both Gaussian candidates have negative incremental gains on both complete
development split seeds. The frozen two-positive-seed and +0.01 mean gates fail.
The >96.4 platform objective remains unachieved; registered platform best stays
the user-reported V12 score 96.3526, without an independently verified receipt.

## G1: incremental quality

Each result is the isolated column `0.8 R23 + 0.2 histogram_member`,
with the other R23 column unchanged. R23 iron is B0/V12 iron; R23 time is
`0.40 V36 + 0.25 V7_member + 0.35 P_LL_T`, the V21 local incumbent.
All five folds are complete at both split seeds. There is no across-seed OOF
averaging, alpha search, checkpoint selection on outer labels, or fallback promotion.
R23 package scores are 96.258405 / 96.261458. These are local evidence.

| target | arm | seed42 gain vs R23 | seed3407 gain vs R23 | mean gain |
|---|---|---:|---:|---:|
| tap_iron | hard | -0.013126097 | -0.013609261 | -0.013367679 |
| tap_iron | gaussian | -0.009047606 | -0.010518413 | -0.009783010 |
| tap_time_len | hard | -0.018519186 | -0.023244358 | -0.020881772 |
| tap_time_len | gaussian | -0.008335581 | -0.005736371 | -0.007035976 |

Gaussian smoothing improves the matched hard control by +0.003584669
(iron) and +0.013845796 (time), but neither gain survives the current incumbent
comparison. The Gaussian fold profile is 0/10 positive for iron and 3/10 for
time; fold-level diagnostics do not replace seed-level gates.

### Historical B0 comparison of the same resulting candidate column

| target | arm | seed42 gain vs B0 | seed3407 gain vs B0 | mean |
|---|---|---:|---:|---:|
| tap_iron | hard | -0.013126097 | -0.013609261 | -0.013367679 |
| tap_iron | gaussian | -0.009047606 | -0.010518413 | -0.009783010 |
| tap_time_len | hard | -0.008088535 | -0.009290349 | -0.008689442 |
| tap_time_len | gaussian | +0.002095070 | +0.008217638 | +0.005156354 |

The time Gaussian column is positive against old B0 (+0.005156354 mean) but
negative against V21 (-0.007035976). Reporting only the old-anchor gain would
mistake already captured improvement for a new increment.
Mean Gaussian package scores are 96.250149 (iron) and 96.252896 (time): the
unchanged 96.25 working gate does not rescue a negative incumbent-relative gain.
This rejects the single frozen 64-bin / sigma=0.75 specification, not all
distributional regression or all possible histogram recipes.

## G0: engineering and budget

- 40 outer fits, 80 optimizer runs (inner selection and fresh outer refit),
  zero failures; elapsed 1251.650 seconds.
- 40 model files and 40 held-out prediction files preserved privately.
- 442256 parameters per model, identical paired initializations (difference 0).
- Synthetic median MAE 0.391812575 versus constant baseline 6.710810110.
- Independent cold inference over all 40 models: maximum difference 0.
- Source/spec/runtime/reference hashes, current fold identities, outer and inner
  train IDs/row counts, train-only supports/statistics, saved settings, sigma,
  parameter count and epoch traces checked. Independent pooled-error arithmetic
  reproduces all gains within 1e-11; independent selection finds no finalist.
- Every worker has torch intra/inter-op threads 1; BLAS/OMP/MKL/NUMEXPR caps 1.
- All arms have zero 240-epoch cap hits. Inner held-out observations outside
  fitted support were recorded, never used to expand training support.

| target / arm | selected epoch range | cap hits | inner outside-support rows, summed |
|---|---:|---:|---:|
| tap_iron-hard | 53–83 | 0 | 1 |
| tap_iron-gaussian | 75–130 | 0 | 1 |
| tap_time_len-hard | 79–107 | 0 | 1 |
| tap_time_len-gaussian | 92–122 | 0 | 1 |

### Checks and CI correction

Fresh locked Python3.12 full training-dependency suite: **1133 passed**,
23 warnings. Clean checkout export with no private caches and no torch:
**952 passed,59 skipped**,3 warnings. The later independent erf-integration
and .75-bandwidth assertions passed in both modes (V23:12 passed;
clean CI:9 passed,3 skipped). Fitted source hashes remained unchanged.
The reservation push itself succeeded; Actions failed two old V6 tests because
private parent ledgers and the V36 summary are absent on GitHub. Fix0981ee6
isolates those unit tests with public/synthetic fixtures and retains private
cache integration tests. [Both CI jobs subsequently passed](https://github.com/Luqhhh/iron/actions/runs/36396761208).

### Review boundary and unused confirmation path

Independent read-only review found no defect invalidating these negative
development results. It identified an **unused confirmation integrity defect**:
`v23_run.py` trusts a development summary's selected target without binding
its hash to the stored audit or independently reselecting, and its confirmation
audit does not compare the stored promotion decision. No confirmation was earned
or run here. **The confirmation path must not be reused automatically.** Any
future confirmation use requires a separately authorized repair, tests, and
new frozen source/preflight identity. Preserving the exact fitted source here
is preservation of research evidence, not approval for future production use.
Additional independent closure verification checked the actual model settings,
fold hashes, inner IDs, sigma and epoch metadata missing from the built-in audit.

## Evidence identities (private artifacts stay outside Git)

- Frozen spec SHA256: `58f56eb796901c988ff2a69b8992a341ac38fcda7082bfd5f2148ee7aafed96b`.
- Development summary SHA256: `60db4a9ed95a06e5a708d1fb46429266166ff0df48db2b3321e0737de52337f6`.
- Built-in audit SHA256: `a432ef475eeba8829764fec2a3644ffd04f6ecd78b30463bf1a7cb4bde89c626`.
- Expanded independent closure SHA256: `17323a1bf7d34dc7b686b9396b9ab2ac0aeb1b10a269b9fa547405dca845655a`.
- Private run: `local/runs/round2-v23-histogram-target/development-r1`.
- Private verification/planning: `local/v23-strategy-20260928/`.
- Reservation specification and earlier failed CI evidence remain unchanged.
- Final publication includes source/tests and this public numerical summary;
  no models, predictions, local reports, ledgers or submission artifacts.
