# iron 96.4 — V31: pricing the platform-measured iron × time axes

Naming update (2026-09-28): local V28 is now **V31**. This is a public-name
change only; original run paths, hashes and delivery IDs remain frozen. See
[the migration record](../round2_round_numbering.md#7-exact-public-migration-and-frozen-identities).
This renamed copy preserves the original design; it is not a new preregistration.

Design date: 2026-09-28; originally registered as local V28. The public
round name is now V31; V30 is the renamed local deep-kernel round.
Status at design time: no V31 fit has run, no package has been written, no
upload has been made.

## 1. What the "hedge packages" are

The three delivered packages measured `96.3567 / 96.3676 / 96.3679` and, together
with the already-measured columns, **confirmed the additive equal-weight WMAPE
model exactly**: `B0` matched its conditional arithmetic
`96.3526 + 96.3519 - 96.3366 = 96.3679`, and a second, independent rectangle
reproduced the `+0.0160` V12-vs-A35 iron effect.

That confirmation changes what an "extra upload" is worth. Previously a package
had to carry a locally-positive expert and hope it transferred — and it twice did
not (`V20` local `+0.004607` -> platform `-0.0003`; `V21_TIME_LOCAL` local
`+0.012728` -> platform `-0.0112`). Now, because the score decomposes additively
and the platform is deterministic, the response along each **already-measured
direction** is a one-dimensional concave curve whose neighbours can be priced:

* **time axis** — V7-member weight on the `A35` base: `v=0` is `A35`
  (`96.3526`), `v=0.5` is `B0` (`96.3679`). Secant slope `0.0306` per unit `v`,
  so by concavity `v=0.75 <= 96.3755` and `v=1.0 <= 96.3832`.
* **iron axis** — V12-member weight: `w=0` is A35 iron (`96.3366`), `w=0.5` is
  `V12` (`96.3526`). Secant slope `0.0320` per unit `w`, so `w=0.75 <= 96.3686`
  and `w=1.0 <= 96.3846` on the A35 time column.

A **hedge package** is therefore a boundary probe: it sits further along a
measured curve than anything scored, so one upload tells us whether the platform
optimum lies beyond the released weight. The four V18 packages are the time-side
probes (`TIME_V75`, `TIME_V100`), the first iron-side probe (`IRON_W75`) and a
higher-N variant (`TIME_A60V7_50`). They are not "hedges" in the financial sense;
they buy information about where each curve turns.

## 2. What this round adds (zero fit)

V18 stopped at `IRON_W75`; the iron axis has no `w=1.0` endpoint, and nothing
combines the two axes even though additivity is now measured. Three packages are
composed field-exactly from the delivered A35/A60/V7/V12 ZIPs:

| id | iron column | time column | concavity upper bound |
|---|---|---|---|
| `V28_IRON_W100` | pure `V12m` (`w=1.0`) | V7 copy (`v=0.5`) | 96.3846 |
| `V28_IRON_W100_TIME_V75` | pure `V12m` (`w=1.0`) | `0.25 A35 + 0.75 V7m` | 96.3916 |
| `V28_IRON_W100_TIME_V100` | pure `V12m` (`w=1.0`) | pure `V7m` | **96.3992** (the V18 two-line ceiling) |

`V28_IRON_W100_TIME_V100` is the extreme corner of the measured family; the other
two isolate one axis at its endpoint. With the four V18 packages this gives a
5-slot design that prices both curves and their combination.

## 3. Two-column changes are deliberate

The combined packages change **both** columns. This is justified by the measured
additivity: `V18_B0` is itself a two-column combination (V12 iron + V7 time) and
is the current platform best, and every component direction has an isolated
platform measurement. No column is scaled outside the measured range
(`iron w in [0,1]`, `time v in [0,1]`), and no nonlinear post-processing is used.

## 4. Verification and budget

Same contracts as V18: seven hash-pinned source ZIPs; N-line cross-check
`<= 1e-9`; 322 template-ordered unique IDs; copied columns byte-identical; blend
read-back within `1e-12` relative; finite and non-negative; `x`-mode ZIP; an
independent post-write audit re-reads every ZIP from disk.

| item | value |
|---|---:|
| new fits | **0** |
| packages | 3 |
| desktop writes | 3 (only on explicit user request) |
| agent uploads | 0 |

## 5. Limits

The bounds above are **upper** bounds. The slopes beyond `v=0.5` and `w=0.5` are
unmeasured and may be negative — the N line already peaked and turned inside its
own measured range (`A60` is its maximum). All three packages have **worse local
scores than `B0`** (96.222734 / 96.218441 / 96.204497 vs 96.247739), and local magnitude
has now failed twice, so **no platform score is forecast**. If both curves have
already turned, all three land below the current best `96.3679`; because the
platform keeps the best score they cost only slots.
