# V37: bottom-up hard-tree contraction

Platform target remains 96.5; current user-reported B0 is 96.3679. V36's
full-shape equivalence passed, its memory use passed, but its 9.93-hour
projection failed the unchanged 6-hour limit. No official model was fitted.

This separately frozen implementation reorders the same mathematical
contraction. Instead of materializing leaf indicators and then multiplying
by both leaf outputs and tree weights, contract leaf responses bottom-up
through the straight-through binary nodes first. For INSTANCE, contract leaf
gating logits the same way before the unchanged tree softmax/dropout.
All 1024 trees, depth five, parameters, initializations, feature selection,
loss and optimizer settings remain unchanged. V36 and original V27 stay frozen.

Use the original V27 reference for outputs and all input/parameter gradients
in both arms, train/eval, and zero/nonzero logits. Atol 1e-6, rtol 1e-5; full
batch 256 / 24 inputs / 1024 trees / depth five as well as small shapes.
No optimizer in equivalence checks. Exactly two subsequent isolated resource
optimizer runs, one per arm: three warm-up and 20 timed steps, 20 evaluation
forwards, one CPU thread, same original projection and limits (6 hours,
1536 MiB/worker, four workers plus 1024 MiB within available RAM). No retries,
smaller model or relaxed gates following failure. Failed starts consume budget.
Preserve append-only records under local/runs/round2-v37.

This is engineering admission only. Official data fits remain zero until a
full current-B0-relative scientific protocol and training implementation have
been separately frozen, including train-only transforms, inner selection and
fresh outer refits, persistence, complete coverage and existing four-seed
promotion requirements. At most two synthetic learnability optimizers after
resource admission. No packages, full-data models, desktop writes or uploads.
