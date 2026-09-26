# V7: learned periodic representations, complete-coverage screening

Frozen before fitting, 2026-09-26. User objective: platform score 96.4;
current reported best A35 = 96.3366. Tomorrow's five alpha packages remain intact.
The user explicitly reconfirmed that external pretrained weights are forbidden.

The new question is whether learned sinusoidal numerical representations add
incremental information to the incumbent's errors. Previous raw/PLE networks
did not test periodic embeddings. This is a new representation test, not a
reopening of residual correction, NODE, retrieval, or in-pool stacking.
Implementation basis: the authors' [numerical embedding package](https://github.com/yandex-research/rtdl-num-embeddings/blob/main/package/README.md).
PLR learns frequencies, applies sine/cosine and a learned linear/ReLU projection.
Two modest initial frequency scales (0.01, 0.1) and two backbones are declared.
Raw-input controls share the same new training protocol; they are not claimed
to reproduce N-0048, whose architecture and training procedure differ.

`configs/round2_v7/SPEC.yaml` freezes all 12 target/recipe combinations, their
cost order, two complete five-fold development splits and the sparse blend grid.
All use only the V2 training snapshot's 21 numeric features and spout identity.
No test inputs, old protected labels, external data or pretrained weights are read.
Scalers, category vocabulary and target normalization are fitted on the inner
training part during epoch selection. Then a fresh network and preprocessor
are fitted on all outer-training rows for that epoch count. Outer-validation
labels never enter training or epoch selection. Duplicate groups stay together.

Primary reference: V36 iron and 0.65 V36 + 0.35 N-0048 time, the exact A35 recipe.
Report secondary gains against Q20 (0.8 V36 + 0.2 N-0048), never claim an
increment over the incumbent by comparing only against V36.
Development uses recorded reference OOF, and is explicitly replay-relative.
Blend weights are chosen from the other split's OOF losses; vectors are never
averaged across split seeds. Repeated splits reuse labels and are not new data.
Confirmation requires same-fold fixed-recipe reference fits, using the existing
V5 cache only if its identity verifies, plus N-0048's matching predictions.

Apply `bf_tap_r2.candidate_tiers` with the frozen policy as a descriptive
development classification; its legacy fold rules do not replace the later V5
promotion contract (four positive split seeds, seed-level paired LCB95 > 0,
fold counts descriptive only). At most one exploration recommendation.
Only candidates positive on both complete development seeds may consume the
two predeclared confirmation seeds, at most two candidates ranked by mean
increment with frozen cost ties. Existing uses of those seeds are disclosed:
they are held out of this round's recipe selection, not globally untouched.
No release is authorized by this round, including below the unchanged 96.25
local working gate. Platform gains cannot be forecast by a fixed local offset.

G0: tests must check training/validation isolation, actual frequency gradients,
deterministic refit and row/chunk invariance; identities include data, folds,
code, spec and installed versions. Run directories and failures are retained;
all outputs remain private. Pin BLAS/OMP/MKL/NUMEXPR and torch to one thread per
worker. G1: no model quality claim until complete-coverage evidence exists.

Pre-execution environment annotation: the root lockfile had omitted the V4.2
optional torch/sympy dependencies already declared in `pyproject.toml`.
`uv lock` reconciles those entries without upgrading any previously locked
package. V7 records and enforces its installed CPU torch/TabM/embedding versions
separately in the specification; root dependencies are not expanded for V7.
