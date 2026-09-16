# optimization-v0.24 dual-target burden-lag

Status: preregistered implementation. Base: `a2e18e9adb30581666ac28b3ec954515ee315281`.

This phase freezes two target-isolated candidates before either receives platform feedback:

- `V24I_BURDEN_LAG_RECENCY_IRON` retrains only the recency60 E09 iron component with 30 appended burden-event summaries. It restores the old E04 component, old rate/R2-time direction, and same-cutoff V6I beta. Its time column is the V21 column exactly.
- `V24T_BURDEN_LAG_QRF_TIME` retrains only the frozen-protocol QRF on the same 30 appended columns. It uses the original QRF effective-neighbor gate and original V21 query-anchored median. Its iron column is the V21 column exactly.

The feature set is fixed at five published burden fields, three disjoint event-time windows `(0,6]`, `(6,24]`, and `(24,72]` hours before each reference, and two statistics (`mean`, `valid_count`). Events must already be available at the reference time. Empty means remain NaN; counts are zero; real zeros participate; no fill, interpolation, persistence weighting, or additional statistic is allowed. Published `cal_time` remains the existing ASSUMED event-time/availability contract, localized to Asia/Shanghai when naive.

Development uses the six June–November origins. Finalization uses the certified 2,754 rows and `2024-12-01 01:44:00+08:00` cutoff. The budget is seven CatBoost iron fits, seven QRF forests, seven QRF preprocessors, 1,792 trees, and zero new calibration fits. Both packages are frozen before feedback and retain one explicitly authorized platform-test slot each, in A then B order. This repository does not upload automatically.

G0 engineering, consumed retrospective evidence, and future platform feedback are reported separately. Offline failure does not erase the two preregistered exploration slots, while any source, causal, identity, numerical, or cold-inference failure blocks packaging.

