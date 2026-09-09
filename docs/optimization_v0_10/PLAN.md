# optimization-v0.10 / OPT-23 causal OOF residual stacking

The user opened v0.10 after v0.9 closed. Register exactly one candidate before
accessing targets or fitting: V4_R2_CAUSAL_RESIDUAL. The reference remains V1,
user-reported test_a 83.0319. Original V1/R2 models and ZIPs are immutable.

Each target gets its own low-capacity CatBoost MAE residual regressor: depth 2,
200 rounds, learning rate .03, L2 20, all remaining parameters explicitly frozen
in experiment.yaml. No parameter search, early stopping, or residual shrinkage
search. Train y - R2_OOF; predict max(0, R2 + residual). Uniform row weights within
each target match its L1 objective. No V1 residual training because applying a
current outer V1 LAD to its own coefficient-training rows is not strict OOF.
No new base or ratio definition, no target ablation candidate after results.

Inputs are the six certified structural features (R2 iron, R2 time, rate, inverse
rate, rate*R2_time-R2_iron, inverse_rate*R2_iron-R2_time), plus spout_no and the
existing 6-hour means of air_volume and total_press_diff. Those two process means
use the original as-of feature builder and timestamp semantics. No target history
value, sample ID, month, horizon, timestamp or cutoff is a meta-model feature.
No learned preprocessing from future rows. Source prediction inputs reject labels.

For each outer cutoff June–November, use only monthly April–prior-month OOF rows
whose labels are available by the outer cutoff. All R2/rate/q train references
must be strictly before their own monthly cutoff and train/history availability
at or before it. Fit and OOF sample IDs are disjoint; q and R2/rate certificates
must agree. Training labels are joined from the verified outer saved history,
never read from evaluation error tables. Process features are constructed for
those historical samples using their original monthly fold and public as-of data.
The entire outer rollout uses the saved base/rate/q models at that outer cutoff.
Verify original prediction and bundle hashes before reusing cached endpoints.

Budget: six origins times two residual fits = 12; zero development R2/rate/q
fits, zero LAD fits. Generate all outer predictions before score-label access.
Evaluate 18 cells plus DEV_LONG/SHORT, target/spout metrics, signed bias and shared
calendar-week bootstrap. Gates match the strict v0.9 V1-relative rules, replacing
its candidate-specific iron identity with a <=.0003 per-horizon iron regression
limit. J must improve >=.0005; H1 time >=.0005; H1 E regression <=.0002; >=4/6
H1 non-worse and >=2/3 recent H1 improve; H2/H3/H4 E/time regression <=.0005;
DEV regression <=.0007. All must pass. The bootstrap is retrospective and not
selection-adjusted; it does not relax a gate or forecast platform improvement.

Only a full pass permits at most two final residual fits and one final inverse-
rate fit (v0.9 deliberately did not train final q), total maximum 15. Reuse final
R2/rate. Final residual training uses only certified monthly OOF, not final-model
in-sample predictions. At most one challenger after independent cold inference,
reverse-order checks and zero inference fits. No automatic platform upload.
Failure closes this registered V4 without another hyperparameter/feature/target
scan. Preserve active V1 and R2 rollback. Development fit evidence and failures
are append-only, each run has a unique directory, manifest and access ledger.

G0 is engineering validity and locked Python 3.12 tests plus independent cold
reload/provenance/order checks; G1 is all fixed gates. Report them separately.
