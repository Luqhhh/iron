# V7 isolated time release, explicitly authorized 2026-09-26

The user requested the V7 submission package in
`C:\Users\lqh22\Desktop\submission`, followed by continued optimization toward
96.4. This explicitly authorizes the V7 candidate's below-96.25 release
exception, one full-data fit and desktop delivery. It does not lower the global
gate, authorize V9/V10 packages, combine targets or authorize agent uploads.
External pretrained weights remain forbidden.

Frozen package **V7_TIME_PLR001_A50** replaces only A35's time column with
`0.5*A35_time + 0.5*V7_tabm_plr001`. The four nested confirmation weights were
all0.5. Iron keeps the exact A35 CSV field strings. The parent ZIP identity and
all execution settings are frozen in `configs/round2_v7/RELEASE.yaml`; the
historical V7 search specification and its zero-release budget remain intact.

Before full-data training, independently reproduce the original time
seed42/fold0 predictions with the original `PeriodicRegressor`: maximum
difference must be0. Save that model and verify identical prediction in a new
process with training reads prohibited. Then apply the **same fit procedure**
to the2754 training rows: inner-only preprocessing and group-safe MAE epoch
selection, followed by fresh initialization/preprocessing and training on the
complete supplied training partition for the selected epoch count. No seed,
epoch, loss, architecture or alpha search is added at release time.

The full model stays private. A cold process reconstructs all322 predictions
using only the saved model and label-free query inputs; compare exact full-batch
predictions and CSV bytes. Reverse, chunk and singleton query checks retain the
existing relative tolerance1e-6. Read back the deterministic ZIP, require only
`result.csv`, template order, unique IDs, finite nonnegative numbers, exact blend
arithmetic and zero iron-string changes. Verify that source/input hashes remain
unchanged throughout. Preserve all evidence if any check fails.

Only after all checks pass, copy the ZIP and a short identity/readme file into
the new `submission/V7_TIME_PLR001_A50` directory, without overwriting anything.
Verify the copied ZIP SHA-256 and archive contents. The original five alpha
packages and the A35 parent remain untouched; the user uploads and returns the
score. Local confirmation is not a forecast of96.4, and the current registered
platform best remains A35=96.3366 until feedback says otherwise.

## Parent-path repair before any fit

The original A35 desktop directory no longer exists. Release-r1 stopped at
parent validation, before its manifest or any model fit; `FAILED.json` and the
failed ledger entry are preserved. The private archived ZIP under
`local/runs/round2-v6-portfolio-r1/packages/V6_PORT_TIME_A35/` was verified to have
the exact frozen parent SHA-256. `RELEASE_R2.yaml` changes only the parent file
location and fresh output-run identity, retaining all model, blend, validation
and delivery settings. The runner accepts an explicit release-spec path so the
original specification and failed directory remain unchanged.

Release-r2 also stopped before any fit: safe YAML loading returned the unquoted
authorization date as a `datetime.date`, which the JSON manifest writer refused.
The incomplete manifest and failure ledger remain preserved. The writer now
serializes that date explicitly; a regression test covers the actual YAML/JSON
boundary. `RELEASE_R3.yaml` uses a fresh output directory, with the same verified
archived parent, model, weight, validation and desktop destination. Total fits
consumed by the two failed preflights:0.
