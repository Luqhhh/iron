# ModernNCA author-core qualification, 2026-09-30

ModernNCA was previously discussed in V27/V28 and the 2026-09-29 open-source
shortlist; it is not a newly discovered method. The frozen keyword inventory
of 202 visible local/origin references, 109 distinct commits, found discussion
in three remote heads but no matched implementation or formal evaluation.
This is limited name-based evidence, not proof that no differently named or
unpublished trial exists. Inventory SHA-256:
`744dbd810b4753324a111a57c88d61893b3d52ab7cc1fcf8d2c9c2822275412b`.

The [ModernNCA paper](https://arxiv.org/abs/2407.03257) and
[author model](https://github.com/LAMDA-Tabular/TALENT/blob/1b973adffa4203c3f62c4949957b1ba3efbe60d3/TALENT/model/models/modernNCA.py)
specify learned embeddings followed by unsquared Euclidean distances and
softmax-weighted training labels. This directly aggregates target values;
it differs from the separate predictive head in TabR and the signed
coefficients of RFM. Those failed related methods give a negative prior,
not a measurement of this exact model. It stays below the already queued
PTaRL job in priority and does not reopen the closed TabR or global RFM trials.

The MIT-licensed author model, adapter, defaults and license were fetched
at commit `1b973adffa4203c3f62c4949957b1ba3efbe60d3`; every Git blob identity
was verified. Model SHA-256:
`02fce6a107998ab7212774507998e77535f630fbf4ee328acf8519ad7c10632f`.
No dependency was installed, no pretrained weight downloaded, and no
competition data entered this qualification. The default author model has
PLR embeddings; those remain unqualified here.

## Measured scope: float64, no-embedding core only

[Qualification script](../../scripts/qualify_modernnca_source.py) executes the
exact pinned author model on fixed artificial arrays. The unused embedding
factory is intercepted and throws if called; this is explicitly not a test
of TALENT embeddings or its training adapter. Independent NumPy arithmetic
reconstructs linear encodings, optional BatchNorm/MLP blocks, unsquared
distances, stable softmax and weighted response values.

Both zero-block and one-block inference match the independent formula to
`3.410605131648481e-13`; reversed and singleton queries differ by at most
`2.2737367544323206e-13`. Inference-label arguments have zero influence.
The fixed linear-core gradient is `93.39813040466771`, against independent
central finite difference `93.39812882558363` (step1e-6). Training neighbor
sampling and batch-prefix masking match independent arithmetic to
`2.2737367544323206e-13`; all four own-label Jacobian entries are exactly0,
while other batch-neighbor labels contribute. Source tampering is rejected
before execution. Optimizer calls and official label reads are both0.

The adapter source excludes current batch indices before entering the core,
which then prepends the batch and masks each self diagonal. Future adaptation
must preserve both exclusions, plus group-duplicate and outer-fold isolation.
This qualification does not prove that a future adapter preserves them.
Predictions are convex combinations of training labels; lack of extrapolation
is a model limitation, not evidence of failure on this dataset.

## Remaining before any official evaluation

PLR embeddings, complete train-only preprocessing, fresh outer refit,
training/epoch protocol, immutable saved states, independent cold inference,
full-size non-time resource admission and a complete phase runner are all
unverified. The script is partial G0 evidence; **G1 is unmeasured**. No formal
ModernNCA fits, package or upload is scheduled. Avoid substituting these
small artificial witnesses for quality evidence.

Any future official candidate must freeze its pool and current DE3 iron /
V32 time reference first, use complete five-fold development at42/3407,
judge isolated incremental blend gain, and earn confirmation at7777/12011
under the unchanged four-seed paired-LCB rule. The controlling instruction
“不要再设置时间预算” means no elapsed-time cap or projected-time rejection.
Preserve historical negative evidence and do not alter running/frozen jobs.

TabPFN was already listed as a pretrained observation item in the older
shortlist, under the no-external-weight boundary. Metadata inspection finds
[public v2 regression weights](https://huggingface.co/Prior-Labs/TabPFN-v2-reg)
and the [official code](https://github.com/PriorLabs/TabPFN); no weights were
downloaded and no competition rows transmitted. Metadata availability neither
authorizes a change to that boundary nor proves local resource compatibility.
Current platform best remains user-reported DE3=96.3749; milestones96.4,
96.45 and96.5 remain unmet. No platform gain follows from this source audit.

## Subsequent default-encoder/partition milestone

The earlier no-embedding result above is historical. The default PLR encoder,
partition model and saved-state independent auditor are now checked; see
[PARTITION_MODEL.md](PARTITION_MODEL.md). Full-phase execution/resource/G1
remain unmeasured; this milestone does not establish optimization gain.
