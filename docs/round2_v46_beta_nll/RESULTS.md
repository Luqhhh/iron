# V46_TIME_BETA_NLL: complete development, no finalist

The frozen paired development batch and independent audits completed successfully.
**G0 passed; G1 did not qualify.** BETA05 improves the latest frozen local reference
on both development seeds, but its mean gain is below0.01, it loses to the matched
NLL control on both seeds, and its mean local score is below96.25. Confirmation
was not started. No promoted candidate, full-data model, package or upload.

## G1: incremental fixed-A20 results

The frozen platform reference is V32_TIME_A60V7_50, user-reported96.3727 and not
independently verified. Its native local components are iron0.5*V36+0.5*V12 joint;
time0.2*V36+0.3*N0048+0.5*V7 periodic. Historical B0 uses the same iron and
time0.325*V36+0.175*N0048+0.5*V7. Neither reference is changed after evaluation.
The tested column is0.8*current time+0.2*member, retaining current iron.

| Arm / split seed | Local package score | Gain vs current R32 | Gain vs historical B0 |
|---|---:|---:|---:|
| NLL /42 |96.245049 |+0.009630 |-0.002925 |
| NLL /3407 |96.243590 |+0.011569 |-0.003914 |
| **NLL mean (ineligible control)** |**96.244320** |**+0.010600** |**-0.003420** |
| BETA05 /42 |96.241411 |+0.005992 |-0.006563 |
| BETA05 /3407 |96.242799 |+0.010778 |-0.004705 |
| **BETA05 mean (only candidate)** |**96.242105** |**+0.008385** |**-0.005634** |

Current-reference local mean96.233720; historical-B0 local mean96.247739.
A better platform reference need not be the highest local reference. Reporting
both comparisons preserves the approved latest-platform-reference experiment;
it does not authorize selecting a different reference after seeing results.

Frozen development conditions:

- Both BETA05 seed gains positive: pass.
- Mean BETA05 gain>=0.01: fail,+0.0083851902.
- BETA05 minus NLL mean gain>0: fail,-0.0022146106.
- Mean development package score>=96.25: fail,96.2421050297.

Mechanism contrasts are-0.0036381645/-0.0007910567. In this frozen architecture,
ordinary NLL outperformed beta0.5 on both complete development splits. The
ordinary control remains ineligible by preregistration; its positive gain does
not promote it retroactively. No second specification or beta search started.
Fold signs are descriptive: BETA05 8/10 positive,NLL9/10. Model-quality evidence
covers two complete split seeds only; no four-seed quality or LCB claim is made.
The four-seed reference-cache verification is an engineering check.

These are local results, not platform predictions. The project objective remains
>96.4; no new platform score was observed in this experiment.

## G0: complete budget and cold-artifact audit

The single development phase completed20 outer fits/40 optimizer starts, with
native inner seed42 fold0 MAE epoch selection followed by a fresh outer refit.
Both arms used identical initialization and their frozen Gaussian architecture;
query labels were removed. Inner/outer training identities, preprocessing,
selected epochs/traces, query coverage, ledgers and prediction byte hashes passed.
Both saved inner and outer models were reloaded in a separate audit process.
Cold/order/chunk maximum difference was2.842170943040401e-14, below1e-8.

A separately implemented arithmetic check reassembled predictions within each
split seed, recomputed fixed-A20 scores and all three failed conditions, and
agreed with the artifact auditor. A final read-only verification rechecked all
frozen inputs and202 audited artifact identities. Confirmation directory and
confirmation phase reservation are absent. No cross-seed OOF mixing.

The four permitted synthetic optimizer starts were consumed once: two resource
and two learnability probes, all passed. Peak worker RSS422.30MiB. Complete-path
projection493.06s included worst-case selector/refit epochs,1.5 multiplier and
300s I/O/audit margin. Actual one-shot controller wall time was53.80s for
formal development plus cold and independent arithmetic audits; this is not
train-fit summed CPU time or a future-runtime guarantee. Workers<=4, numerical
threads1. Synthetic learnability MAE0.51737/0.55266 beat median constant1.96581.

Failed formal fits0, confirmation fits0, new reference fits0, full-data fits0,
packages0, desktop writes0, uploads0. Exclusive experiment-level phase allocation
prevents repeat batches in another directory; failures do not free allocation.
All completed futures are checked for failures before workers are refilled.

## Source recovery and implementation verification

Reservation47436e6 preceded implementation. Initial preflight stopped before
training because two private originals were missing. The user supplied both:
`configs/data.local.yaml` and `configs/predict.local.yaml`, matching the frozen
V8 hashes. A differing historical private script was recovered from an existing
exact-hash local handoff backup; its previous bytes were saved separately.
All original failed reports, models, predictions and ledgers are retained.

Historical committed-source LF/CRLF variants were accepted only after exact
Git-byte identity proof. Data/manifests/ledgers/models/predictions are never
normalized. Recovery also exposed a V46 verifier count error: original V12 audit
covers20 predictions across four seeds, not10 confirmation-only predictions.
The regression fix requires exact20-file identity coverage. Recovery0ccd0d7 and
full runner509bc9e were ordinary pushes on the configured current upstream.

A fresh whole-runner review found two Important issues before formal fits:
experiment-wide phase allocation and failure-group refill ordering. Both received
one RED/GREEN fix pass. No Critical findings or deferred Minors. Final15 targeted
contracts passed without optimizer starts; neural Python3.12.3 full suite1167
passed/23warnings in141.27s; uv.lock Python3.12.13 full suite984 passed/61skipped/
3warnings in59.07s. The locked environment omits optional torch. These were rerun
after the fixes. Only public evidence changed after formal completion, so no
additional fitting tests were needed. Private-artifact checks remain mandatory.

Final specification SHA-256:
`8c0f733328cd69d60298de862144aad16088b8dd22d9d3f3b5cd5368da7d434a`.
Final preflight SHA-256:
`56a54e7c5d72ef8bd6639d01f9a36133b3a7eff0da16fe440983a09105001cf5`.
Development cold audit SHA-256:
`2a3c1f4521de2bb1db38288927f2e5b8b0ad1770af966d23fa418b486052be2a`.
Independent decision SHA-256:
`43eafe1f74b697e38c7ed8efd34bff078e47b288d821c02cf44820d87ce4b7ed`.
The complete integrity identities are in EVIDENCE_STATUS.json; artifacts stay
under local/. The frozen experiment is complete without a finalist. The quiet
30-minute monitor is paused after final evidence publication.
