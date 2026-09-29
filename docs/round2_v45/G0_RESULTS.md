# V45 engineering admission

Three platform milestones remain 96.4 / 96.45 / 96.5, against user-reported
96.3727. This report admits the frozen experiment, not a platform candidate.
Protocol e1de45f, implementation a2106ec. No actual-data V45 fit at admission.

Locked Python 3.12 full suite: **1256 passed**, 23 existing warnings. Eight
focused tests cover full-rank/singular PCA, matched bootstraps and tree seeds,
identity-rotation equivalence, sklearn/exported-tree equality, serialized
inference/order/chunks, label guards and audit rejection of modified states.
Runtime and all 20 original reference-cache unit identities verify.

Two prescribed full-shape synthetic forests completed, each 256 CART trees:

| Arm | MAE | Fit/predict/save seconds | Peak RSS MiB | Cold/order/chunk difference |
|---|---:|---:|---:|---:|
| AXIS control | .645789 | 5.6095 | 430.21 | 0 |
| ROTATE candidate | .363817 | 6.3518 | 459.00 | 0 |

The training-median constant has MAE 3.961012. Both arms learn, and rotation
improves this declared oblique synthetic response. This is a mechanism check,
not evidence that the real targets benefit. Cold verification independently
checks every synthetic tree's bootstrap counts/means and every PCA basis.

The same conservative twice-slower-fit projection over 80 forests / four
workers is **.0705753 hours / 4.235 minutes**, below 7.61 hours. Four measured
worker peaks plus 1024 MiB reserve require **2860 MiB**. Current available RAM,
frozen source/spec/runtime and reference identities are checked again in the
append-only private `preflight-r1/admission.json` before development.

G0 passes; G1 remains unmeasured. Admit the fixed 40 outer units, 80 forests,
20480 component-tree fits. No recipe/grid changes, automatic release or upload.
Private root: `local/runs/round2-v45/preflight-r1`; full-test log:
`local/reports/v45-python312-tests-r1.log`. Public evidence excludes fitted
models, predictions, detailed reports and ledgers.
