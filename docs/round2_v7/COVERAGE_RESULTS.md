# V7b execution evidence (2026-09-26)

**G0: real-data control passed, maximum absolute prediction difference 0.**
The original inner-training procedure at seed 42 / fold 2 / epoch 55 reproduces
the frozen N-0048 held-out predictions bit for bit. The independently written
fixed-epoch optimizer loop therefore passed the predeclared 1e-9 control gate
before any candidate refit started. Control metadata and predictions remain in
`local/runs/round2-v7-coverage/development-r1/`.

The ten full-training-fold refits are running with two single-thread workers.
They retain the original architecture, optimizer, preprocessing and cached
inner-selected epochs. Their additional training rows also increase optimizer
steps; this is the declared procedure, not an isolated causal estimate of row
coverage alone. Original models, ZIPs and past decisions are unchanged.

**G1: candidate results pending.** A bit-identical control establishes the
comparison, not an improvement. No four-seed confirmation, full-data fit,
package or upload has occurred for this strategy. The platform target remains
96.4; current user-reported best remains A35 = 96.3366.

Latest locked Python 3.12 test path: **987 passed, 3 warnings**. This includes
the new coverage-loop control and V7 confirmation identity/coverage guards.
The V7 confirmation runner waits for the complete frozen candidate pool,
selects its highest-ranked two-development-seed-positive candidate, and checks
data, code, runtime, cached-reference identity and held-out row masks. Existing
V5 confirmation caches contain time only; an iron winner requires fresh
same-fold baseline fits and is explicitly refused by that cache-only runner.
No time cache can be silently used for an iron candidate.
