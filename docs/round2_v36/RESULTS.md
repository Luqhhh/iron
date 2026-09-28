# V36: equivalent routing resolves memory, but misses the time budget

Goal remains platform **96.5**; current user-reported best **B0 96.3679**,
gap **0.1321**. No new platform score or model-quality result is claimed.
Frozen implementation: `5f3d2a2`. Completed 2026-09-29 Asia/Shanghai.

## G0: equivalence and resource result

The fixed 1024-tree, depth-five model retains all 1,271,808 parameters at the
24-column synthetic probe shape. The only model-core change reuses each
parent path for its two children instead of expanding every leaf's full path.
The copied V27 reference differs only in the license-link path; the original
source and MIT license hashes were independently verified.

All eight full-shape comparisons (two arms, train/eval, zero/nonzero gating
logits) have **zero maximum output, input-gradient and parameter-gradient
difference**. Every tree has exactly one active leaf. Small-shape tests also
pass. This establishes the tested implementation equivalence, not model quality.

| Arm | Training-step p95 seconds | Evaluation p95 seconds | Peak RSS MiB |
|---|---:|---:|---:|
| GLOBAL | 0.576763 | 0.133619 | 873.004 |
| INSTANCE | 0.486830 | 0.153051 | 875.387 |

Each isolated process used one CPU thread, batch 256, three warm-up plus
20 measured optimizer steps and 20 evaluation forwards. Exactly two resource
optimizers were started, each protected by an exclusive pre-start record.
No retries occurred. Measurements include finite-loss/gradient checks.

The unchanged V27 projection gives **35753.675 seconds / 9.9316 hours**:
`(40/4)*1.5*250*((9+7)*max_training_p95 + 2*max_evaluation_p95)`.
This remains above the **6-hour** limit, even though the original V27 estimate
was 50.4701 hours. RSS now passes the **1536 MiB** limit; four workers plus
1024 MiB also fit within the 7763.36 MiB available at audit. Timing comparisons
are measurements on the recorded runtime, not a controlled hardware benchmark.

RSS measures the core, optimizer and synthetic tensors only. Formal
preprocessing/persistence/runner overhead remains unmeasured. This limitation
cannot overturn the time refusal. The projection assumes the maximum epoch
budget and is not elapsed full-training time; early stopping was not evaluated.

Independent audit recomputed p95 values from every timing sample, checked
source/spec/start records, verified equivalence records and original-source
identity, and recomputed the admission decision without calling the admission
helper. The locked Python 3.12 full suite passed **1213 tests**, with 23 existing
warnings. Audit execution against the actual private evidence also passed.

## G1 and budget decision

**Not evaluated.** The resource gate failed, so no official training was
started. Synthetic learnability runs 0; development, confirmation and full-data
fits 0; packages, desktop writes and uploads 0. No smaller model or relaxed
time threshold was substituted. Existing desktop packages are unchanged.

V36 resolves the original memory limitation and substantially reduces measured
compute cost, while leaving a concrete time gap. It does not establish that
hard-tree models cannot improve B0 or that every equivalent implementation
would fail. Any subsequent optimization requires a separately frozen change
and budget; this round's failure and source identities remain intact.

The five-slot scheduling recommendation is unchanged: begin with the existing
50% interior probe, then the 75% probe; do not replace them with an unverified
new model or fill the remaining slots solely to exhaust quota. These probes
are experiments, not promises of 96.5. Calendar rollover alone does not confirm
account activity or remaining quota.

## Private evidence identity

All artifacts are retained under `local/runs/round2-v36/preflight-r1` outside Git.

- Independent audit SHA-256: `78e59f3776eecce72281eaa2d891a979eaab771c83d2c2355b4e4a150113b000`.
- GLOBAL resource SHA-256: `23adb76c0cbe930c294b67f844b89fe017588d1e1c260708a45805a50c9be06f`.
- INSTANCE resource SHA-256: `7ee2ae84ece5019231bb2097246709ee12e4ae9b661e4f8f52188e84394a45e7`.
- Full-shape equivalence records and their start ledgers are bound in the audit.
