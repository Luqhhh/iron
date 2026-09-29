# Synthetic admission: runtime budget not passed

Terminal event 2026-09-29T07:29:38.410017+00:00. All six full-shape synthetic fits finished without
fit exceptions. Learnability, exact full-batch cold prediction, memory and frozen
source checks passed. Maximum order/chunk difference1.6134214e-5; worker peak
668.7109375MiB. This is engineering evidence only, not real-data quality evidence.

The frozen conservative runtime estimate is7.928570820616652hours, above the
7.61hour cap by0.318570820616652hours (about19.1minutes,4.19%). This estimate
includes the specified twofold safety factor; it is not elapsed or promised
completion time. Runtime was the only failed admission check.

The supervisor stopped on that failure, before any actual-data development fit.
The60 planned outer units remain pending. No implicit restart, increased budget,
threshold relaxation, replacement specification or package was produced.
Preserve the complete failed preflight and terminal event. The original
BASE/EMA/SAM workflow is separate and continues unchanged.

G0 overall admission:failed (runtime budget). G1:unmeasured; the mechanisms have
not been rejected for predictive performance. Report SHA-256:69656f20918a25127f9c0e308e55d6edc758b1942dfa6d91874ad81205b9a02d.
