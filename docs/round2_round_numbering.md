# Round2 round numbering and reservations

Status: current as of 2026-09-28. This file is the single place that says what
each `V<n>` label means across the live branches, so a merge does not silently
join two different rounds under one number.

## 1. Renumbering applied on 2026-09-28

The teammate branch `codex/round2-v23-histogram-target` owns **V23** for the
histogram-target prediction round. The local line had already closed two rounds
under that number and the next one, so those two were moved:

| local round | was | now | outcome |
|---|---|---|---|
| time-family capacity probe | `V23` | **`V25`** | closed negative, weight exactly `0.0`, no package |
| target-representation probe | `V24` | **`V26`** | closed negative, weight exactly `0.0`, no package |

What moved (public, Git-tracked labels):

| old | new |
|---|---|
| `configs/round2_v23/` | `configs/round2_v25/` |
| `configs/round2_v24/` | `configs/round2_v26/` |
| `docs/round2_v23/` | `docs/round2_v25/` |
| `docs/round2_v24/` | `docs/round2_v26/` |
| `src/bf_tap_r2/v23_capacity.py` | `src/bf_tap_r2/v25_capacity.py` |
| `src/bf_tap_r2/v24_logtarget.py` | `src/bf_tap_r2/v26_logtarget.py` |
| `tests/test_round2_v23_capacity.py` | `tests/test_round2_v25_capacity.py` |
| `tests/test_round2_v24_logtarget.py` | `tests/test_round2_v26_logtarget.py` |
| EVIDENCE key `round2_v23_2026_09_28` | `round2_v25_2026_09_28` |
| EVIDENCE key `round2_v24_2026_09_28` | `round2_v26_2026_09_28` |
| spec versions `round2-v23-…` / `round2-v24-…` | `round2-v25-…` / `round2-v26-…` |

The renumbering adds zero fits, packages or uploads, so the move changes
no model, no prediction and no score. Both closed rounds retain their recorded
20 development fits. The recorded G1 numbers are identical
before and after; only the labels and paths differ.

## 2. What is frozen and must not be renamed

Private evidence keeps its historical names and was not touched:

- `local/runs/round2-v23/dev-r1` (capacity: `screen-both-r1.json`, per-seed
  prediction files and fit ledgers)
- `local/runs/round2-v24/dev-r1` (target representation: `screen.json`, per-seed
  ledgers)
- `local/reports/v23-*.log`, `local/reports/v24-*.log`

Recorded digests still describe the same bytes. The renamed source modules pin
these directories in `OUTPUT_DEFAULT` and in their guard paths on purpose; a
future re-run of a closed round would write to the historical directory, not to a
new `round2-v25/round2-v26` one.

## 3. `V20` is used by two different rounds

| branch | `V20` means |
|---|---|
| local line (`round2-v20-…`) | P-LL time release on the B0 parent; package `V20_B0_PLLT_A325`, ZIP `f14f39df5474c8904639c0768bf8fa678f645f46fb0f987fbb84690a983eef57`, pending upload |
| teammate (`codex/round2-v20-masked-recon`) | masked feature reconstruction; closed negative, no package |

`V20` was **not** renamed: the released package label is frozen while it is
pending upload. Read `V20` as branch-qualified until the package is scored.

## 4. Reservation rule

- Live local numbers after the move: `V17`, `V19`, `V21`, `V22`, `V25`, `V26`;
  teammate-owned: `V20` (masked, closed) and `V23` (histogram target).
- **Current next free local number: `V28`.** `V27` is now reserved below.
  Do not reuse `V18`/`V20`/`V23`/`V24`.
- Before pre-registering a new round, fetch all remotes and compare
  `configs/round2_v*/SPEC.yaml` version strings across `main` and every live
  branch; reserve here first if a collision is possible.
- When the teammate branch is merged, keep its `configs/round2_v23/SPEC.yaml`
  and `docs/round2_v23/PREREGISTRATION.md` as the meaning of `V23` — this file
  is the authority that the local `V23` was moved to `V25`.

## 5. Related records

- `EVIDENCE_STATUS.json -> round2_round_numbering_2026_09_28` (machine record of
  the same mapping).
- `EVIDENCE_STATUS.json -> round2_v25_2026_09_28` and
  `round2_v26_2026_09_28` (the renamed rounds).
- `AGENTS.md -> Round numbering (2026-09-28)`.

## 6. V27 reservation and later feedback (2026-09-28)

`V27` is the authorized deep-kernel experiment on branch
`round2-v27-deep-kernel`, specification `configs/round2_v27/SPEC.yaml`.
Development completed with no confirmation finalist; the current target is
96.5. See `docs/round2_v27/RESULTS.md`. Its public preregistration and
implementation are pushed. The next free local number is **V28**.

The earlier V20 pending-upload wording above is historical. The user has since
reported `V20_B0_PLLT_A325 = 96.3676`; B0 remains preferred at 96.3679.
The package identity and branch-qualified V20 naming remain frozen.
