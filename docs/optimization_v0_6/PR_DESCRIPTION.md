# Add zero-fit horizon routing and gated S1 challenger

P2 improves H1–H3 development loss but exceeds the old H4 risk limit. The new S1
pipeline uses P2 for H1–H3 and unchanged R2 for H4, routed solely by reference-time
calendar month relative to the frozen cutoff. It reloads existing components with
schema/history/source identities; no old algorithm, gate or release pointer changes.

The 18-cell plus DEV replay passes the new deployment gate: J improves 0.000819983,
H1 improves 0.001475603 at five of six origins, and H4 is exactly preserved.
The conditional target/horizon leave-one-origin-out router instead worsens J by
0.000868579 against S1 and is rejected. These remain consumed-development results.

Validation: 40 real endpoint checks (max 1.71e-13), 173 locked Python 3.12 tests,
zero attempted/completed fits, label-free final cold-process recovery (max 5.68e-14),
CSV/ZIP readback and B/C dry runs. The test_a S1 challenger is ready locally;
R2 stays active. Models, official data, predictions and package remain local.
No platform score, desktop overwrite or platform upload is included.
