# Round2 round numbering and reservations

Status: current as of 2026-09-28 after the V30/V31 collision repair. This file is the single place that says what
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
- **Current next free local number: `V32`.** V27/V28 belong to the remote
  hard-tree/edge-spline rounds, V29 is the concurrent information screen,
  V30 is our deep-kernel round, and V31 is our axis-endpoint round.
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

## 6. Current ownership after the naming collision repair

The user requested that local names yield to the remote reservations. A fresh
fetch found both integer and suffixed configuration directories; reservations
must scan `configs/round2_v<N>*/SPEC.yaml`, not only `round2_v<N>/SPEC.yaml`.

| Number | Owner / experiment | Current specification |
|---|---|---|
| V27 | remote `codex/round2-v27-hard-tree-weights` | `configs/round2_v27_hard_tree/SPEC.yaml` |
| V28 | remote `codex/round2-v28-edge-spline-network` | `configs/round2_v28_edge_kan/SPEC.yaml` |
| V29 | concurrent local information screen; preserved | `configs/round2_v29/SPEC.yaml` |
| V30 | local deep kernels, formerly local V27 | `configs/round2_v30/SPEC.yaml` |
| V31 | local iron/time axis endpoints, formerly local V28 | `configs/round2_v31/SPEC.yaml` |

Remote ownership was checked at hard-tree `4f52392`, edge-spline `92b0d71` and
histogram-target `20c0432`. V29 appeared as untracked work in the shared checkout
while this repair was being prepared. Its files were not changed or included in
the naming commit. The initial V29/V30 proposal was abandoned before any file
was moved; the atomic destination check prevented overwriting V29.

The repair branch is `round2-v30-v31-naming-repair`. Published original branches
`round2-v27-deep-kernel` and `round2-v28-axis-endpoints` remain historical snapshots;
no branch history is rewritten or deleted. The new public names do not rename
remote experiments. Next free local number: **V32**.

## 7. Exact public migration and frozen identities

| Public name before | Public name now |
|---|---|
| `configs/round2_v27/`, `docs/round2_v27/` | `configs/round2_v30/`, `docs/round2_v30/` |
| `bf_tap_r2.v27_*`, `scripts/run_v27_*`, `tests/test_round2_v27_*` | `bf_tap_r2.v30_*`, `scripts/run_v30_*`, `tests/test_round2_v30_*` |
| `round2_v27_2026_09_28` evidence key | `round2_v30_2026_09_28` |
| `configs/round2_v28/`, `docs/round2_v28/` | `configs/round2_v31/`, `docs/round2_v31/` |
| `bf_tap_r2.v28_axis_endpoints`, `tests/test_round2_v28_axis_endpoints` | `bf_tap_r2.v31_axis_endpoints`, `tests/test_round2_v31_axis_endpoints` |
| `round2_v28_2026_09_28` evidence key | `round2_v31_2026_09_28` |

Spec version strings, imports, public links and current headings follow the new
names. Algorithms, parameters, scores, decisions and delivered bytes do not
change. This repair performs zero fits, package builds, desktop writes or uploads.

Frozen private paths stay `local/runs/round2-v27/` and
`local/runs/round2-v28/`; all original `v27-*`/`v28-*` logs and ledgers stay in
place. The three delivered package IDs remain `V28_IRON_W100`,
`V28_IRON_W100_TIME_V75`, and `V28_IRON_W100_TIME_V100`, with the same ZIP hashes
and desktop filenames. Interpret those IDs as the **legacy delivery IDs of V31**.
The old V20 package-name exception in section 3 is likewise retained.

V30's original source-bound manifest and audit describe the frozen V27 execution
at commit `d07dc35` (and the unchanged scientific files at repair base `2e28b83`).
A historical audit must use that frozen checkout with its private evidence made
available; renamed current source is not byte-identical to the old source.
Original digests are neither rewritten nor bypassed. V31's original delivery
source is pinned by `2e28b83`. Read the original preregistrations from those
commits when auditing their original hashes. The renamed documents are public
navigation copies, not a new preregistration or permission to rerun a closed round.

See `EVIDENCE_STATUS.json -> round2_naming_repair_2026_09_28` for the migration
map, verification and unchanged private-artifact snapshot.


## 8. Verification of this repair

- Locked Python 3.12, existing affected tests: **13 passed**.
- All **10** renamed Python files have identical syntax trees after reversing
  the declared public-name substitutions. Both specifications are identical
  after excluding their public `version` and `source_plan` fields.
- **474** original private files and desktop ZIPs retain their hashes.
- All **1457** source/evidence hashes in the old deep-kernel manifest match
  the original Git objects at `2e28b83` or unchanged private files.
- G1 is unchanged. This is a naming migration, not a new model reproduction.

## 9. V33 reservation (2026-09-28)

After fetching all remotes and scanning integer and suffixed SPEC paths across
all local/remote branches, V32 is occupied by the mixture-bound diagnostic.
V33 is reserved for continuous mixture-density regression on branch
`round2-v33-mixture-density`. Next free local number: V34 (recheck before use).

## 10. V34 reservation (2026-09-28)

Fresh remote fetch and all-branch integer/suffixed SPEC scan found V33 occupied
and V34 free. V34 is reserved for CRPS natural-gradient trees on
`round2-v34-crps-boosting`. Next free local number: V35; scan again before use.

## 11. V35 reservation (2026-09-28)

Fresh remote/all-branch scan found V34 occupied and V35 free. V35 is the
independent CRPS convergence study, branch `round2-v35-crps-convergence`.
Next free number: V36; recheck all branch reservations before using it.

## 12. V36 reservation (2026-09-28)

After a fresh remote fetch/all-branch scan, V36 is reserved for an equivalent
hard-tree routing implementation and resource check. This does not modify
V27's CPU refusal. Next free local number V37 requires a new reservation scan.
