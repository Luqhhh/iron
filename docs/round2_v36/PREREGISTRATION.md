# V36: equivalent hard-tree routing and resource admission

Goal remains platform 96.5 versus user-reported B0 96.3679. V27 hard-tree
regression was refused at its CPU resource gate, with no official fits and no
quality evidence. The latest other-branch KAN result is also known and is not
repeated here. V36 does not rewrite either historical decision.

The original V27/GRANDE core expands a depth-five path for each of 32 leaves
over every row/tree. The proposed implementation reuses each parent's path
probability for its two children. This preserves the fixed hard-routing model
and straight-through derivatives while avoiding the large leaf-by-depth tensor.
All 1024 trees, depth 5, feature subsets, parameters, initialization, dropout and
GLOBAL/INSTANCE semantics remain unchanged. Keep upstream MIT attribution and
license, with V27 commit 4f52392 as the exact comparison source.

Before any official fit, check both arms in train/eval modes, zero and nonzero
weight logits, all outputs and all parameter/input gradients against the
unchanged reference. Use atol 1e-6 / rtol 1e-5, and verify one active leaf per tree.
Check the actual full 1024 / depth 5 shape as well as small synthetic shapes.

Then run one isolated resource process per arm, with full model/optimizer
shape, batch 256 / features 24, one CPU thread, 3 warmup and 20 measured optimizer
steps, and 20 evaluation forwards. Two optimizer starts total; preserve started
events and failures. Record every timing, maximum RSS and the original V27
projection formula. Retain its gates: <= 6 hours projected 40-unit development,
<= 1536 MiB worker RSS, and four workers plus 1024 MiB margin within available RAM.
No smaller model, fewer trees or relaxed threshold if the optimization fails.

This is an implementation-equivalence/resource experiment. Passing it alone
does not authorize claiming a quality result. A full training-only preprocessing,
inner-selection/outer-refit, cold-persistence and same-B0 validation protocol
must be implemented and frozen before official data fitting. Synthetic
learnability may use at most two optimizers after the resource gate passes.
Original V27 quality recipes/references are not silently reclassified; the
eventual new experiment must explicitly use current B0 and existing promotion
gates. Full-data models, packages, desktop writes and agent uploads: zero.

All local artifacts remain append-only under local/runs/round2-v36. Publish
validated public source/tests and the scientific status, keeping G0 resource
feasibility separate from G1 quality. This is a concrete attempt to make an
unmeasured learner feasible, not a prediction of platform improvement.
