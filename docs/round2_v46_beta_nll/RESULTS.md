# V46_TIME_BETA_NLL: stopped at the reference-cache prerequisite

## G0: stopped before model training

The approved strategy was reserved and pushed at `47436e6` before implementation.
The reference-only preflight terminated with `stopped_cache_verification_failed`.
The historical V8 confirmation manifest requires two missing private originals:

| Required source | Frozen SHA-256 |
|---|---|
| `configs/data.local.yaml` | `406a77ef04170f1963582d3bc617b3d429048e8517ff2fe2abf70ee0a77d6fee` |
| `configs/predict.local.yaml` | `fa4051a4cea199d3dc67111326c03d92acce2ef5cf0aa60fc8729ec8e70dc870` |

No matching original was found in native local evidence, root configurations or
the archived `md/` configuration tree. Neither file was guessed or regenerated.
The frozen all-four-seed identity check precedes official fits and permits zero
new reference fits, so training does not proceed. This is a cache-provenance
failure, not a resource measurement or a neural-model quality result.

An earlier legacy loader rejection involved 62 Python files whose recorded LF
hashes differed from Windows CRLF checkouts. All62 differences were proved to be
newline-only with matching committed Git source blobs. V46 verifies exact frozen
source byte variants; it never normalizes data, manifests, ledgers, models or
predictions. Existing loaders, source specifications and historical evidence
remain unchanged. This resolved format issue does not remedy the missing private
configuration originals. Diagnostic failures are retained under `local/`.

## Implemented scope and checks

Retained public components: the matched single-Gaussian network, ordinary NLL
and detached-variance beta0.5 objective, native inner selection/fresh-refit
interface, fixed-gate decision helpers, resource projection, and reference-only
cache verification CLI. The formal batch runner, resource/learnability execution,
model cold-inference audit and confirmation execution were not completed after
the prerequisite stop. No trained-model or complete-experiment reproduction
claim is made.

Eight synthetic unit contracts cover analytical objective values, detached
variance gradients, paired initial parameters, training-only unknown-category
handling, exact Git byte identity, complete paired eligibility, resource caps,
four-seed decisions, and evidence-preserving failure behavior. They create no
V46 optimizers. A fresh read-only reviewer found one Important partition-interface
bug: the native inner helper returns a dict, requiring its `fold` vector. A
no-optimizer regression reproduced it; the fix passed all8 tests. No remaining
Critical/Important findings, no deferred Minor findings. Resource execution,
convergence, saved-model invariance and G1 were explicitly not judged because
they were not reached.

Final checks after the review fix: **1160 passed**,23 warnings in113.73s
in the containing neural runtime (Python3.12.3); separate **uv.lock
Python3.12.13:977 passed,61 skipped**,3 warnings in48.44s. The locked environment
does not install optional torch; these checks do not certify a trained V46 model.
Exact stop-report SHA-256: `956a0eaf373aa166d6cd27d9c2c0b24d70ecfe289d08304c4d75c1dc885c21f2`.
The full commands and logs are retained privately; both suites were rerun after
the review correction. No private cache files were removed for testing.

## G1: not evaluated

Development outer fits0, confirmation outer fits0, V46 optimizer starts0,
synthetic resource/learnability optimizer starts0, new reference fits0,
full-data fits0, packages0, desktop writes0, uploads0. Repository unit-test fits
for historical mechanisms are separate from this experiment's budget.
There is no V46 score, delta, tier, finalist or promotion decision.

Latest remote-registered platform reference remains V32_TIME_A60V7_50=96.3727,
user-reported and not independently verified. Own V28_SHARED=96.3671 remains
historical feedback. No result here changes these scores or predicts >96.4.
V33's ineligible GAUSS1 observation remains an unpromoted control.

The frozen round stops here. No retry, second specification, hidden reference
retraining, package or monitor is started. Recovering an exact historical source
identity would need an explicit follow-up, with the failed evidence preserved.

Private stop report: `local/runs/round2-v46-beta-nll/preflight-r1/report.json`.
Reproduction command (always use a new private output directory):

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
.venv/bin/python -m bf_tap_r2.v46_cache \
  --output local/runs/round2-v46-beta-nll/cache-check-NEW
```

The command is reference-only. Exit2 records the failure and schedules no
optimizer; existing output directories are rejected without overwriting evidence.
## Exact-source recovery follow-up (2026-09-29)

The user supplied the two original private configurations; both SHA-256 values
match the frozen manifest exactly. A further private historical source mismatch
was resolved from an existing local handoff backup whose bytes match the frozen
hash. The previous differing script was saved separately before recovery; all
historical predictions, ledgers, manifests and failed reports are retained.

The resumed check exposed a V46 verifier error: V12's original audit records
20 predictions across development and confirmation, rather than 10 confirmation
predictions alone. The fix requires exact filename coverage for all four seeds
for both V7 and V12, in addition to original passed statuses and byte hashes.
A regression failed before implementation and all9 targeted tests then passed.

A new reference-only report passes all-four-seed cache verification, binding
436 file identities. This supersedes the missing-source blocker, while the
initial stop above remains historical evidence. The specification is unchanged.
G0 resource/learnability and G1 remain unmeasured; V46 optimizer starts and new
reference fits are still0. No package or upload is authorized by this recovery.

Recovery verification: neural-runtime full suite1161 passed,23 warnings in171.72s;
uv.lock Python3.12.13 full suite978 passed,61 skipped,3 warnings in72.50s.
Private-artifact guard passed. Restored private configurations stay outside Git.

## G0 completed after recovery

The four frozen synthetic optimizer starts passed. Resource probes used the
2204/551-row path with preprocessing, optimizer, gradients and validation;
peak worker RSS422.30MiB. The frozen projection is493.06s including both
selector/refit worst-case epochs,1.5 multiplier and300s I/O/audit margin.
This is a synthetic projection, not a measured formal-batch completion time.
Saved models' cold/order/chunk maximum difference is0. Learnability MAE is
0.51737(NLL)/0.55266(BETA05), below the training-median constant1.96581.

The complete runner and saved-artifact cold auditor are now implemented.
A fresh reviewer identified two Important issues before formal fits: missing
experiment-wide budget reservations and worker refill before checking all
simultaneous completions for failures. Both were fixed in one RED/GREEN pass.
Exclusive phase reservations now reject repeated allocations even in new output
directories or after failure; all completed futures are checked before refill.
No Critical findings or deferred Minors. The15 targeted contracts pass without
optimizer starts. Final neural-runtime suite1167 passed,23 warnings in141.27s;
uv.lock Python3.12.13 suite984 passed,61 skipped,3 warnings in59.07s.

The final new preflight freezes transitive sources, control scripts, runtime,
specification, data/folds and reference/G0 evidence. Formal fits remain0 at this
checkpoint. Development is the single frozen20 outer/40 optimizer batch;
confirmation remains conditional on complete independently audited eligibility.
All historical failure evidence is retained; no specification change or G0 retry.
