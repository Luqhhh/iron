# PLE-B mechanism defect: new evidence for a separate repaired experiment

The historical V3.6 `ple_tabm` implementation constructs
`PiecewiseLinearEmbeddings(version="B", activation=True)` and supplies raw numeric
features. `NumericPreprocessor.transform_tabm` intentionally bypasses its fitted
standardization for this structure. Historical source, models, predictions,
negative results and release decisions remain unchanged.

The installed `rtdl-num-embeddings==0.0.12` source reveals two distinct issues:

1. Version B initializes the nonlinear projection weights to exactly zero, with
   no bias. With `activation=True`, its contribution is `ReLU(0)`. PyTorch's
   derivative there is zero, so that entire nonlinear branch receives zero
   gradients. Adam/AdamW leave its zero weights at zero, including weight decay.
   The embedding reduces to its linear bypass; the surrounding TabM backbone
   remains nonlinear. This is not a claim that the whole model is linear.
2. That bypass consumes the unstandardized input directly. Large feature units
   therefore enter the first embedding at their raw scale. This is a separate
   conditioning concern, not itself a demonstrated cause of held-out loss.

The [author package documentation](https://github.com/yandex-research/rtdl-num-embeddings/blob/main/package/README.md#piecewise-linear-encoding--embeddings)
recommends version B with `activation=False`. The locally installed source
hash is `48dfc019d447e0cd225deb8b8145b3efa8b42cde9aa314772d6e234753e1cee1`.
No dependencies or weights were downloaded or changed.

## Measured synthetic diagnostic

Reproducible entry point: `scripts/round2_ple_gradient_diagnostic.py`.
Private evidence: `local/reports/ple-b-gradient-diagnostic-r2.json` (the earlier
r1 diagnostic is retained). One hundred synthetic rows, three numeric columns,
eight bins and embedding width eight, three AdamW updates, fixed seed42.
The same inputs and initialization are used for each activation comparison.
Bins and actual inputs always share the same coordinate system.

| Coordinate | Activation | Nonlinear gradient at step0 | Nonzero nonlinear weights after3 updates | Initial embedding RMS |
|---|---|---:|---:|---:|
| Raw | True | 0 | 0 | 1735.2673 |
| Raw | False | 252.6624 | 192 | 1735.2673 |
| Standardized | True | 0 | 0 | 0.281402 |
| Standardized | False | 0.0360542 | 192 | 0.281402 |

All three observed gradients are exactly zero in each `activation=True` case.
Standardizing alone does not restore the nonlinear branch. Disabling activation
restores it in both coordinate systems. The RMS comparison is synthetic scale
conditioning evidence, not an official-data score or a platform forecast.
Official-data model fits0; packages0; uploads0.

## Consequence and next experiment

Historical failures such as N-0060 WMAPE0.1712 remain valid for their recorded
recipe. They do not test a functioning PLE-B nonlinear branch. This direct
mechanism evidence permits a separately preregistered repaired PLE experiment;
it does not retroactively relabel any old candidate or silently repair an
immutable cached model. No change to V3.6 is made in place.

The controlled next question is standardized PLE-B with activation off versus
its standardized, inactive-branch control, under complete five-fold development
coverage and the V7/V9 incremental references. Freeze the candidate pool, training
budget and cost order before official-data fitting. Retain the four-split-seed
promotion rule, unchanged local gate and release restrictions. V12 is a separate
joint-output experiment already running; its evidence is not altered by this
synthetic diagnosis. This document does not itself claim a promoted candidate,
a new full-data fit, or achievement of the96.4 platform goal.
