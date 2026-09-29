# Serial recovery complete: no development finalist

BASE/EMA/SAM finished on 2026-09-29. All 60 formal units (20 per arm)
and six initialization diagnostic units are complete. Independent corrective
audit passed all 132 saved selection/refit models. The frozen development
gate admits no arm to confirmation. Augmentation/representation stays deferred.

## G1: fixed component replacement

Gains below are local isolated package-score points against the frozen
V32_TIME_A60V7_50 reference. The remaining components and the other target
are unchanged. BASE exactly reproduces each original component and has zero gain.

| Target | Arm | Seed 42 | Seed 3407 | Mean gain | Positive folds (descriptive) | Frozen mechanism decision |
|---|---|---:|---:|---:|---:|---|
| tap_iron | EMA | +0.002821451 | +0.002596886 | +0.002709169 | 7/10 | No finalist |
| tap_iron | SAM | -0.006021182 | -0.000520690 | -0.003270936 | 4/10 | No finalist |
| tap_time_len | EMA | +0.001992716 | +0.002344908 | +0.002168812 | 7/10 | No finalist |
| tap_time_len | SAM | -0.012875987 | -0.019983331 | -0.016429659 | 2/10 | No finalist |

EMA improves both complete development seeds on both targets, but its
means (+0.002709 iron/+0.002169 time) are below the preregistered +0.01
score-point floor. The general candidate-tier policy classifies both EMA
entries as `formal`; retain that descriptive classification. It does not
override this experiment's mechanism gate, establish four-seed promotion,
or authorize a package or platform test. SAM is negative on both seeds for
both targets and remains `not_shortlisted`. No threshold was relaxed.

Confirmation command returned `no_development_finalist`, zero new fits.
The derived seeds 271828/314159 were not consumed. No full-data fitting,
package, desktop write or upload occurred. The current platform best remains
user-reported V32_TIME_A60V7_50=96.3727, not independently verified. Local
gains are not platform forecasts.

## G0: recovery and independent audit

Locked Python3.12 --no-sync suite: 1307 passed, 23 existing warnings; focused
component/recovery suite: 18 passed. The initial unpinned full-suite command
failed 18 cold/runtime checks; supplying the four required thread variables
as 1 resolved all failures without a source or model change.

The original development-r1 is preserved. Recovery copied 37 complete units
and ten reference units verbatim, retained the original manifest byte for
byte, and verified all 418 frozen sources, data, runtime, folds and artifact
identities. It fitted only 23 missing formal units and six predeclared
diagnostics, 58 new selection/refit runs. Four interrupted unit directories
remain in the original run: three contain saved selector checkpoints and one
contains only its start event. None was imported. These interrupted attempts
are preserved separately from the 132 completed-model total.

The original frozen auditor failed on selector target means: reconstructing
a DataFrame target array changes NumPy reduction order (observed maximum
difference 2.27373675e-13). Its failure log is preserved. A separately
recorded audit reconstructs the native `y[inner != 0]` array slice. It retains
all other checks and exact tolerances, modifies no frozen source, and makes
zero training calls. Its effective-source/script/original-auditor hashes
are recorded in private audit-recovery-provenance.json.

- Cold models checked: 132.
- Native BASE predictions replayed: 20; maximum difference 0.0.
- Full-batch cold prediction difference: 0.0.
- Maximum reversed/chunked raw prediction difference: 5.62159220863e-05, within the frozen 5e-4 limit.
- Reused-unit byte mismatches: 0; source/data changes: 0.
- Supervisor terminal exit: 0 at 2026-09-29T17:53:47+08:00; systemd MainPID=0, Result=success.
- Recovery workflow wall span: 2640.14s (44.00 minutes).

## Initialization diagnostics

These six seed1042 fits use only the prespecified seed42/fold0 partition.
They are descriptive sensitivity measurements, not independent split-seed
evidence and not inputs to candidate selection, stopping rules or weights.

| Target | Arm | Mean absolute prediction change | Isolated package-score change vs seed42 |
|---|---|---:|---:|
| tap_iron | BASE | 3.942586649 | +0.009748495 |
| tap_iron | EMA | 2.571648222 | +0.009038522 |
| tap_iron | SAM | 3.748838204 | +0.005463205 |
| tap_time_len | BASE | 1.341328209 | -0.000055514 |
| tap_time_len | EMA | 0.793786124 | +0.008643361 |
| tap_time_len | SAM | 1.462196164 | +0.017615278 |

## Evidence identities

All machine reports, models, predictions and logs remain private under
local/runs/strong-component-regularization. Recovery implementation commit
7e7e3ea; durable launch record ff539ce. The successful synthetic admission
and old interrupted/parallel evidence remain untouched.

- manifest.json: `cc1f3a9bd141c7c4d20d6dcc7cb129baee0bf8db933462a3fc7081dd3463c933`.
- summary.json: `e223e4389950cd351a24bcaf70ac5236cb97882713a09b6e18de97efb38e4ace`.
- audit.json: `bc14b82bb1442224d1a77b485a8f0ac0064d1387d797ec28a8fe13570369bcb2`.
- audit-recovery-provenance.json: `42fb38bd22ebc2c911063744a23efbf2e575736fb3f169eac15ada2629b50d6f`.
