# V11 quantile representation (2026-09-26)

## G0 engineering

All **80/80** candidate outer fits and the one standardized-coordinate control completed without failure. The control reproduces the original V7 predictions exactly (maximum absolute difference0). The locked Python3.12 development suite passed **1020 tests,23 warnings**.

The independent `audit-r1.json` in `local/runs/round2-v11-quantile-representation/development-r1` verifies80 prediction hashes, source/runtime/data/fold/reference identities, inner/full training row IDs, preprocessing means, selected epochs, and direct sparse-grid, fold, seed, LCB and package-score arithmetic to1e-12. It independently reconstructs40 sets of noisy training-only quantile landmarks using the underlying library, and reproduces the candidate tiers and selection. Audit model fits0.

Seven fits reached the240-epoch selection cap: iron raw-normal1, time raw-normal5, time periodic-uniform1. Their selected checkpoints and cap flags are retained; this is a limitation of these fixed-budget recipes, not evidence of convergence. No budget was enlarged after inspecting results.

## G1 development

Each development seed contains all five folds. CURRENT means the fixed V9 iron candidate (not a platform release) or the delivered V7 time blend (platform score unreported). Other-column package scores retain A35; no joint package is implied. All gains below are full-package score points.

| Target | Recipe | A35 mean | CURRENT seed42 | CURRENT seed3407 | CURRENT mean | Positive folds | Alphas | Tier |
|---|---|---:|---:|---:|---:|---:|---|---|
| tap_iron | raw_qnormal | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 0/10 | 0.00/0.00 | not_shortlisted |
| tap_iron | raw_quniform | +0.000621 | +0.000000 | +0.000000 | +0.000000 | 0/10 | 0.00/0.00 | not_shortlisted |
| tap_iron | plr_qnormal | +0.000314 | +0.000000 | +0.000000 | +0.000000 | 0/10 | 0.00/0.00 | not_shortlisted |
| tap_iron | plr_quniform | +0.000643 | +0.000000 | +0.000000 | +0.000000 | 0/10 | 0.00/0.00 | not_shortlisted |
| tap_time_len | raw_qnormal | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 0/10 | 0.00/0.00 | not_shortlisted |
| tap_time_len | raw_quniform | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 0/10 | 0.00/0.00 | not_shortlisted |
| tap_time_len | plr_qnormal | +0.006176 | -0.000256 | +0.000000 | -0.000128 | 2/10 | 0.05/0.00 | not_shortlisted |
| tap_time_len | plr_quniform | +0.012520 | +0.003121 | +0.002316 | +0.002718 | 7/10 | 0.20/0.20 | formal |

Only **time periodic-uniform (`plr_quniform`)** passes the two-complete-positive-seed prerequisite against both A35 and CURRENT. Its mean V7 increment is **+0.002718**, with two-seed LCB95+0.000179 and7/10 positive folds; both held-seed weights are0.20. The selected recipe has one capped fit. Relative to Q20 its mean gain is+0.012180, while its A35 mean is+0.012520. These larger old-parent comparisons do not establish a larger increment over V7.

Its fixed development package score is **96.232293 <96.25**. The absolute working gate remains unchanged and the user’s V7-specific release exception does not apply to V11. No platform forecast follows from this local score.

All four iron directions optimize to zero weight on top of V9. Raw normal/uniform time directions also optimize to zero on top of V7; periodic-normal has one negative split and fails progression. Preserve these negative results without another alpha scan or substituting an older reference.

## Confirmation allocation

`configs/round2_v11/CONFIRMATION.yaml` freezes the sole finalist and allocates ten candidate fits at seeds7777/12011, with eight workers and all numeric thread counts pinned to1. These are previously used split seeds, not new labels. Existing verified same-fold V7/V5 caches supply the references; new baseline fits0. A zero-fit reference preflight passed for92 files. The locked Python3.12 confirmation suite passed **1027 tests,23 warnings**, including seven candidate-pool entrance tests.

The confirmation entrance rejects missing/duplicate candidate pools, incomplete seed/fold coverage, and unearned or changed selection. Promotion still requires all four seeds positive and a positive seed-level paired LCB95; fold counts are descriptive. No other finalist or extra confirmation budget is authorized by this round.

The completed confirmation is reported below. External data and pretrained weights remain forbidden; full-data fits0, packages0, desktop writes0 and uploads0 for V11. The separately delivered V7 package is unchanged. Platform best remains user-reported A35=96.3366;96.4 has not been established.

## Four-seed confirmation: failed

All ten confirmation candidate fits completed without failure or epoch-cap hits. Private run: `local/runs/round2-v11-quantile-representation/confirmation-r1`. No new baseline fits were needed. The independent audit verifies20 candidate prediction files (ten development plus ten confirmation), source/runtime/data/fold/reference identities, training metadata,20 independently reconstructed inner/full quantile statistic sets, and direct sparse-grid/fold/seed/LCB arithmetic to1e-12. Audit model fits0.

| Split seed | V7-relative increment | Held-seed alpha | A35-relative increment |
|---|---:|---:|---:|
| 42 | +0.002670 | 0.10 | +0.010770 |
| 3407 | +0.001803 | 0.10 | +0.010098 |
| 7777 | -0.004099 | 0.20 | -0.000274 |
| 12011 | +0.001287 | 0.10 | +0.004777 |

The V7-relative mean is **+0.000415**, seed-level paired LCB95 **-0.003189**, with only **3/4** positive seeds and11/20 positive folds. Seed7777 loses0.004099, outweighing the small apparent development increment. Confirmation reselects each held-seed weight using the other three seeds, so the development-seed entries differ from the original two-seed table; both records are retained.

Even the A35 comparison fails the all-seeds-positive requirement: mean+0.006343, LCB95+0.000269, but seed7777 is negative. A positive lower bound alone is insufficient. Q20 also fails (mean+0.005646, LCB95-0.001380,3/4 positive seeds). The fold level remains descriptive.

Disposition: **failed four-seed confirmation; no promoted candidate or new package**. The original development formal classification remains historical. There is no extra finalist, weight repair, increased-epoch refit or automatic joint-column blend. This rejects the frozen quantile recipes as a next increment over the present candidates; it does not establish that every quantile-based model is ineffective.

Total V11 budget consumed:80 development candidate fits+1 exact-control fit+10 confirmation candidate fits, each using inner selection plus a fresh outer-partition refit. Full-data fits0, packages0 and agent uploads0. Locked Python3.12 tests:1027 passed,23 warnings. All private predictions, ledgers and audits remain outside Git.

The [delivered V7 package](../round2_v7/DELIVERY.md) remains available for user upload. V7 has no recorded platform result; current platform best is still user-reported A35=96.3366. The96.4 objective is unresolved, and these failed V11 increments provide no platform forecast.
