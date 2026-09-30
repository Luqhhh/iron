# Completing the current incumbent reference

The current registered platform best is user-reported
`DE3_IRON_USER_REQUESTED = 96.3749`. Its delivery changes only iron:

```
maximum(V32_iron + 0.5 * (mean(J42, J104729, J130363) - J42), 0)
```

The time column remains the original V32 parent. Gaps to the user's platform
milestones 96.4 / 96.45 / 96.5 are 0.0251 / 0.0751 / 0.1251. No new platform
score has been obtained by this work.

The existing DE3 development cache covers split seeds 42 and 3407 with all
three training seeds. The native J42 reference already covers 7777 and 12011.
Completing the current reference therefore needs only training seeds 104729
and 130363 at the latter two splits: 20 estimators, each with native inner
epoch selection followed by a fresh outer-training refit (40 optimizers).
These counts are a proposal, not a reserved or consumed official budget.

This completion supports future incumbent-relative PTaRL, DNNR and DANet
evaluation. It does not confirm or promote the historical DE3 experiment.
That experiment's development mean gain 0.003480204913 still failed the
frozen 0.01 mechanism threshold and selected no finalist. Preserve its
original audit, decision and delivered ZIP. No acceptance threshold changes.

## Implemented core

`incumbent_reference_core.py` wraps the unchanged `ComponentRegressor` BASE
implementation. Before each optimizer is created it reserves an exclusive
append-only event. An estimator also needs a separate reservation before its
output directory is created. Failure or process interruption consumes the
reservation; the wrapper rejects a repeated `fit` even after initialization
failure. The hooks consume no numerical RNG and the synthetic test compares
every final parameter and prediction bit for bit with the original model.

`incumbent_reference_ledger.py` derives from the RFM reservation engine on
`codex/ptarl-space-calibration` at ba5ce299185a3e4ef468c5f4aaf61ae9d6cf0859.
Only the allowed budget kinds and task-specific helpers differ. Locks serialize
reservation counts between processes; policy, starts and terminal receipts
have external digests, exclusive writes and fsync. Historical modules are
unchanged.

The zero-fit saved-state auditor binds the external completion digest, exact
settings and fit/query partitions, both model digests, prediction identities,
native train-only statistics, selector epoch and fresh-refit epoch. It reuses
the unchanged native saved-state verifier. Selector targets are sliced from
the original joint target array, preserving its actual reduction layout.
Full-batch cold predictions must be bit-identical; order and chunk checks must
meet both the original absolute 0.0005 bound and the relative 0.000001 bound.
Audits also reconcile the actual estimator and two optimizer receipts.

`incumbent_reference_columns.py` requires all 60 explicitly identified member
units (four split seeds, five folds, three training seeds), ordered held-out IDs
and finite predictions. It applies the delivered endpoint in its exact
arithmetic order. Each average is within one held-out partition. Different
split seeds are never averaged. Parent time vectors are returned as unchanged
copies. Arithmetic alone is not provenance: the future runner must bind and
audit every member first.

## Required before official fits

The phase runner, metadata/source/artifact cache freeze, resource admission,
bounded execution, complete-coverage audit and PTaRL-compatible overlay writer
are not implemented in this core commit. No official reference fit may be
started merely because core tests pass. Native J42 and old DE3 member reuse
must first verify their original source/data/fold/fit/query identities and
external hashes; the new budget must then be frozen before execution.

The subsequent official job will use at most four workers and the standing
600-second monitoring cadence. It will retain all failed evidence and produce
only private reference artifacts. No full-data fit, package, desktop write or
platform upload is part of this completion.

G0 here is synthetic engineering verification. G1 remains unmeasured; a new
reference by itself does not provide a model-quality gain.
