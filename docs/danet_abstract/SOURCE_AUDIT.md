# DANet abstract-layer numerical preparation

DNNR's frozen official round stopped at its numerical cold-audit gate. The
next queued family is DANet. Its grouped feature processing and raw-feature
shortcuts offer a different mechanism to examine; useful incremental gain
against DE3 remains unmeasured.

Primary sources are the [AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/20309)
and [author repository](https://github.com/WhatAShot/DANet/tree/b007c57121ec9082f6ef19ec7465d9df70767c26).
The paper's abstract layer uses global entmax-1.5 masks, gated affine branches,
ghost normalization and additive group fusion. Stored normalization statistics
and masks can be merged into affine weights for inference. Its network repeats
two main-path layers per block and injects raw features through a shortcut.
The reported architecture includes a final regression MLP. This preparation
implements only the abstract-layer equations and compression; it does not
reproduce the full model, optimizer, training protocol or paper benchmarks.

The author's MIT license is preserved at `licenses/DANet-MIT.txt`. Private
clone: `local/research/danet-author-source-r1`, commit
`b007c57121ec9082f6ef19ec7465d9df70767c26`. The source receipt binds six relevant
files including the license, original layer, sparsity mapping, compression,
configuration and training wrapper. No new dependency was installed; preserved
readers and their frozen environments remain unchanged.

The original evaluation layer is row-local. At eight synthetic rows, 21 input
dimensions, five branches and eight outputs, batch versus single-row and
permuted predictions differ zero; author eager versus compressed differs
1.66533e-16. These are measured properties, not assumptions from tensor shape
comments. The independent implementation uses explicit row/group layout and
support-aware entmax autograd. Its independent bisection oracle, finite
differences and NumPy compressed inference are covered by tests.

Single-layer author parity on 16 synthetic training rows and eight query rows,
float64, ghost size four:

| comparison | maximum difference |
|---|---:|
| initial masks/projection | 0 |
| entmax transform | 0 |
| ghost running-statistic state | 0 |
| training forward | 4.44089e-16 |
| evaluation forward | 1.11022e-16 |
| compressed evaluation | 2.22045e-16 |

A 17-row, ghost-size-four test exposed the author's singleton final ghost
batch failure. The independent implementation refuses that batch before any
running-statistic update; it does not silently merge batches or fall back to
evaluation normalization. The valid 16-row parity witness is separate from
that failure; the singleton refusal itself is explicitly tested.

The prospective matched control freezes only the identically initialized mask
logits. All initial projection weights, biases and normalization states match
the learned-mask arm, as do initial outputs and non-mask gradients. The learned
mask has a nonzero synthetic gradient while the frozen control has none.
This isolates a component intervention, not proof of a useful trained network.

Current status is G0 numerical-component preparation only, G1 unmeasured.
There is no full-network estimator, optimizer, calibration, saved-state audit,
full-size cost admission or formal execution. All must be implemented and
checked before official fits. Preserve complete-coverage screening, unchanged
two-development-seed and four-seed paired-LCB gates, the current DE3 reference,
and the prohibition on mixing OOF split seeds. No package or upload occurred.
User-reported DE3=96.3749 remains best;96.4/96.45/96.5 remain unmet.

Locked Python3.12 final-source checks: **1409 passed, 23 existing warnings**;
11 focused component checks passed. Exact source snapshot:825 files. Private
full-suite receipt SHA-256:
`4867b4a51588306b107927d4fcb27527eecb760dd3c570b1069be54f70ce3ea5`.
Private source/parity/singleton evidence stays under `local/research`. The
singleton diagnostic's original counter field records BatchNorm forward
attempts including the failed fifth call, not five successful updates; a
separate interpretation records four completed chunks and preserves that file.
All failed DNNR evidence and frozen source remain intact on its own worktree.
