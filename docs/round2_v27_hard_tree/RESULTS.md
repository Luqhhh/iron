# V27_HARD_TREE: stopped at the frozen resource gate

Status: approved implementation, G0 resource refusal, no official fits, no G1 quality result.
Reservation: e455af26e3937e31b0f179e4291f352f9ae7221b. The user explicitly confirmed the frozen design.
The original SPEC remains unchanged (SHA-256 6d1a9fd0b37821b7b36402de1c3ccffd3ac885f6a00fb45335a7d0feebc5a59e); its reservation-status field is historical. Current execution status is recorded here and in EVIDENCE_STATUS.json.

## G0 engineering evidence

Implemented the MIT-attributed regression core from pinned GRANDE commit 07f7278b30ab9ebbbdc544e5f9a91f1b5df7fecb and the paired GLOBAL/INSTANCE gating. No full author package, embeddings, external data or pretrained weights were used. The complete upstream license is preserved in GRANDE_LICENSE.txt.

Four behavior tests first failed because the learner was absent, then passed after adaptation; three resource/admission tests followed the same RED-to-GREEN cycle. Full-size (1024 trees, depth5, 24 synthetic input columns, 1,271,808 parameters) initialization matched the pinned author core exactly. Both evaluation and dropout-enabled training comparisons had forward and all-parameter gradient maximum absolute difference **0.0**.

Single-worker CPU probes used batch256, all BLAS/OMP/MKL/NUMEXPR and torch intra/inter threads1, 3 warm-up optimizer steps plus20 timed steps per arm, and20 timed evaluation forwards. Each arm used one fresh optimizer; no optimizer was repeated.

| Arm | Training-step p95 seconds | Evaluation-forward p95 seconds | Peak worker RSS MiB |
|---|---:|---:|---:|
| GLOBAL | 2.445499 | 0.787643 | 1871.027 |
| INSTANCE | 2.929751 | 0.587924 | 1837.258 |

This RSS covers the core, optimizer and intermediates using synthetic tensors. Train-only preprocessing and the formal runner were not implemented or measured. The measured subset already exceeds the frozen1536MiB cap, so adding that work cannot restore resource admission.

Using the frozen conservative formula:
projected_seconds = (40/4)*1.5*250*((ceil(2204/256)+ceil(1764/256))*max_arm_training_p95 + ceil(441/256)*max_arm_evaluation_p95)
= **181692.37271 seconds / 50.4701 hours**.

This is an estimate for the complete40-fit development batch, not elapsed execution time and not test-suite time. It assumes maximum250 inner epochs plus250 fresh outer-refit epochs, four workers, and a1.5 safety multiplier. Early stopping could shorten actual runtime; full fits were never started. The estimate fails the frozen6-hour gate by8.4x. Maximum worker RSS 1871.027MiB also independently fails its1536MiB gate. The four-worker available-memory check passes (4*peak+1024MiB <=8820.703MiB available), which does not override the other two failures.

Independent closure audit verified source/SPEC/author/license hashes, all recorded timing samples and p95 values, parameter counts, the separate projection arithmetic and absence of a V27 formal run directory. Refusal is supported; full G0 did not pass.

## Budget and unexecuted work

Resource-probe optimizer runs: **2** (46 total optimizer steps, including warm-ups). Synthetic80-epoch learnability runs: **0**. Development/confirmation/full-data fits: **0/0/0**. Packages, desktop writes and agent uploads: **0**. No smaller model, retry, second recipe or confirmation run was started.

After refusal, train-only preprocessing/category handling/target scaling, synthetic learnability, persistence and chunk invariance, inner selection/outer refits, reference-cache audit and integrated promotion execution remain untested or unimplemented. Admission arithmetic helpers do not constitute a formal runner. The private probe is one-shot and must not be reused as restartable orchestration: it has completed-result protection but no pre-optimizer start ledger; a crash followed by manual restart could otherwise repeat work. Both actual probes completed successfully, so the recorded budget is unaffected.

## G1 model quality

**Not evaluated.** There is no evidence that INSTANCE improves GLOBAL or the current R27 reference, no candidate score, no historical-B0 increment and no platform forecast. R27 remains the preregistered V12 iron plus V21 local time anchor; reference caches were not loaded for this refused batch. The registered platform best remains user-reported V12=96.3526; >96.4 is unachieved.

The result closes this fixed1024/depth5 CPU recipe on this runtime, not the entire hard-tree model family. No inference about smaller recipes, optimized implementations or other hardware follows.

## Verification and review

Locked Python3.12.13 full environment: **1140 passed,23 warnings,140.21s**. Fresh exported checkout without torch or private caches: **955 passed,63 skipped,3 warnings,62.41s**. Four new model tests skip without torch; three new protocol tests execute there. Warnings are existing LightGBM feature-name, EBM interaction-display and Lightning deprecation warnings. No failures or errors occurred.

One fresh read-only whole-branch review found no Critical/Important blocker to G0 refusal. Reporting qualification (partial RSS coverage) is stated above. Deferred limitation: general start-ledger/crash-restart support for the private one-shot probe; no future reuse is authorized. Reviewer-declined stages are explicitly listed as unexecuted rather than silently treated as complete.

Private records remain under local/v25-strategy-20260928 (the initial unpublished provisional label is preserved). Key summaries:
- Resource decision SHA-256 aa5129950791c5f186a90ea4f3314e24575cee4264f2b8641388dff21b213ce5.
- Author-equivalence record SHA-256 a3d534fdce2bb29ab7b9e24264c8a17a31fc9bce22c3c8bbf5467ce7647bce9e.
- Independent closure audit SHA-256 527ea0948e41f59ae08788627dd9cea49bc4e8820d73f7b0b9001422a36e4d60.
- Full test XML SHA-256 ff36fd7e341cd867f5c575d84fb8c2b128b8b47fb663b01997bd361eaabdcf6f; clean test XML SHA-256 da233d725cbaa565729d000bd27a0254e1c9048e3a6e17d5dc9abed0c8b831c3.

Public implementation, tests, license and numeric summary are published together at closure. No background training was launched, so no V27 monitor was activated; the completed V23 monitor stays paused.
