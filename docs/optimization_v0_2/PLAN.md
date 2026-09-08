# optimization-v0.2 execution plan

Status as of 2026-09-08: OPT-01 through OPT-04 are implemented and executed on the
`optimization-v0.2` branch. OPT-03 passed G0 but neither candidate passed G1;
OPT-04 E09 passed G0 and every G1 gate. OPT-05 and OPT-06 are not complete. The immutable
comparison remains `baseline-v0.1-reproducible`.

This phase preserves the baseline data, timestamp, model-parameter and
post-processing contracts. It introduces a separate strict configuration parser,
candidate registry and `optimize-v0.2` CLI. Development label reads remain bounded
by `2024-11-01T00:00:00+08:00`; protected reports and final training are outside
OPT-01/02.

The first batch is intentionally small:

| ID | Only change from E00 |
| --- | --- |
| E00 | Frozen baseline through the v0.2 entry |
| E01 | Remove all history features |
| E02 | Remove history-age columns |
| E03 | Remove historical target values; retain history age/count state |
| E04 | Retain sample and history only |
| E05_OPERATION | Retain sample and operation only |
| E05_BURDEN | Retain sample and burden only |
| E06 | Shrink E00 toward B1 with separately declared target weights |

Screening uses the unchanged DEV_LONG and DEV_SHORT folds. Promoted candidates use
five frozen origins and 14 origin/horizon cells. A model is fitted once per origin
and predicts every configured horizon without adding post-origin target history.
Grid selection uses equal weight across H1–H4 rather than pooling correlated rows.

Run IDs are caller-selected output directories and must not already exist. Detailed
predictions and error evidence belong under ignored `local/` paths.

```bash
uv run python -m bf_tap optimize-v0.2 \
  --data-config configs/data.local.yaml \
  --suite screening \
  --e00-reference local/runs/<frozen-baseline-run> \
  --output local/runs/<unique-opt-run>
```

For a promoted subset, repeat `--candidate`; E00 is mandatory, and derived
candidates must include their base candidate.

```bash
uv run python -m bf_tap optimize-v0.2 \
  --data-config configs/data.local.yaml \
  --suite all \
  --candidate E00 --candidate E02 --candidate E04 --candidate E06 \
  --e00-reference local/runs/<frozen-baseline-run> \
  --output local/runs/<unique-promoted-run>
```

Acceptance remains a v0.2 development decision. G0 execution and G1 model quality
are always reported separately.

## Current boundary and next phase

Four test_a candidates were uploaded by the user after OPT-02. Their scores are
external, user-reported evidence; the sequence was exploratory and must not be
presented as an independent final test. Adaptive platform replacement stopped after
the time-calibrated 75/25 blend. That derived blend and calibration have local
manifests, but are not yet registered candidates in the full-grid acceptance run.

The OPT-03 implementation is the original frozen-history adaptation:

- synthetic history origins at ages `{0, 7, 30, 60, 90}` days;
- complete masking of unavailable historical result rows and their end/count data;
- grouping every view of one original sample in the same outer/inner partition;
- total view weight of one per original sample;
- auditable `original_sample_id`, `view_id`, `history_origin` and history identity;
- direct comparison with original-history and no-history routes.

The pre-registered model candidates are `E07_FROZEN_E02` and
`E08_FROZEN_E04`. Both use the exact ages and weighting policy above; their only
difference is the already registered E02 versus E04 feature route. The full-gate
run includes E00, E01, E02 and E04 as controls.

The user explicitly requested one exploratory platform probe despite the failed G1
gate. `E07_FROZEN_E02` was frozen with a manifest and packaged without changing
the gate or representing it as accepted. The user reported a platform score of
`82.3610`, below the incumbent `82.7046`.
This is an explicit exception, not a relaxation of the default rule that
future platform candidates must pass the development protocol.

## OPT-04 pre-registration

OPT-04 adds a single fixed process-change feature family derived from the existing
6-hour, 24-hour and latest operation aggregates. For each frozen operation value,
the family contains `latest−mean6h`, `latest−mean24h` and `mean6h−mean24h`.
Candidate E09 adds this family to E02, E10 adds it to E01, and E11 adds the family
as the only process source on E04. The run retains E00/E01/E02/E04 as direct
controls and uses the unchanged screening and full-grid protocol.

Screening promoted E09 and E10; E11 was rejected for excessive DEV_LONG
regression versus E04. In the complete grid, E09 passed all acceptance gates and
E10 failed only the DEV_LONG control comparison. E09 is frozen and packaged for a
single test_a platform check. The user reported `82.8174`, making E09 the current
platform incumbent while leaving its pre-existing G1 decision unchanged.
