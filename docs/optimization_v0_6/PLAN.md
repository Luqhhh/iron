# optimization-v0.6: horizon-aware deployment

Base: optimization-v0.5 at 73a0abc27c6f102b6b03c9179e451b2ec28eba73.
The user authorized OPT-17 and conditional OPT-18. The phase freezes new definitions
in configs/optimization_v0_6; v0.5 candidate failures and thresholds remain unchanged.

S1 uses P2 (0.8 original E09 + 0.2 robust E04) for H1–H3 and R2 for H4.
Calendar horizon is local calendar-month difference from the frozen model cutoff,
plus one. References before the cutoff and horizons outside 1–4 are rejected.
For each prediction/component, history remains frozen at that same cutoff; all
process features retain the original as-of implementation. Component predictions
are clipped before blending. Targets never enter the inference feature builder.

OPT-17 reloads three existing components at each of six origins, verifies identities
and exact schemas, reconstructs both endpoints, then routes every sample including
DEV_LONG/SHORT. It scores 18 cells and two DEV folds with no fit. Duplicate sample
IDs are rejected within each unit; cross-origin repetitions retain their origin
identity and must have identical metadata and labels. The objective averages cells
within horizons then averages horizons, rather than stacking repeated samples.

S1 must pass every frozen deployment gate before a new composite is constructed.
The final bundle copies existing full-training O0/OR/HR with their own model,
history, schema and source hashes. Each stage is predicted both in-process and in
a new process using configurations with no training-label paths. The unchanged
v0.4 loader independently checks R2 endpoints. Only test_a gets a challenger ZIP;
B/C receive unscored dry-run CSVs. R2 stays active and its ZIP stays untouched.

OPT-18 starts only after S1 passes and its challenger is ready. For each target and
horizon, choose the smaller equal-origin mean WMAPE using all other origins; ties
choose R2. Evaluate the held origin using that choice. DEV diagnostic routing also
excludes its corresponding origin. Promote only with J improvement >=0.0003 over
S1, non-regressing H1 and target/horizon regression <=0.001. Only after that gate
may a final mapping be fitted by choosing from the full development origin means.
This mapping selection is not model training. No weight/window/seed/calibration
search or additional candidate is allowed.

LOO prevents direct use of the held origin in its routing choice. It does not make
these consumed months independent: real samples recur across origins, and the
S1 hypothesis was proposed after inspecting v0.5. Report all results as retrospective
post-holdout-consumption development, not untouched confirmation.

No platform upload or incumbent overwrite is part of this execution. Current
platform last-vs-best submission behavior must be checked if upload is later
requested; no new platform score is predicted from development loss.
