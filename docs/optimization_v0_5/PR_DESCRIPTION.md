# OPT-14: verified component replay and near-term acceptance

v0.4 retained only blended R2 development predictions, so the individual robust main
and auxiliary contributions could not be recovered from the CSVs. This change loads
verified saved O0/H0/OR/HR bundles without fit, exports full-precision component
predictions, and scores the three pre-registered crossings against R2 with separate
STAGE_A and GENERAL gates. Old v0.4 code and the active release pointer are unchanged.

Real validation covers 18 cells plus two DEV folds, 40 A00/A11 endpoint checks
(maximum difference 1.71e-13), and zero fit attempts. The locked suite passes 150 tests.
Month/spout/target diagnostics, conditional HR contribution and paired-week intervals
are retained locally. Models, row predictions, ledgers and reports are not committed.

All three candidates fail both gates. P2 restores H1 (0.001476 improvement, five of
six origins) but breaches the H4 regression ceiling (0.002752 versus 0.0015).
No challenger or active-package replacement is made. The registered OPT-15 condition
is met; T1/T2 training and OPT-16 are outside this first-delivery change.
