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