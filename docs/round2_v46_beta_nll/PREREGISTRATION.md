# V46_TIME_BETA_NLL: approved time-only paired likelihood experiment

User confirmed the short design on 2026-09-29. Objective remains **>96.4**.
This strategy identity avoids the newly fetched teammate V45 Rotation Forest.
Only this strategy reservation is public now; implementation and results follow.

## Evidence and hypothesis

V33's frozen GAUSS1 time control had B0-relative gains +0.001803/+0.004840
at complete seeds42/3407. It was ineligible and remains unpromoted. Its MDN3
candidate failed. This observation motivates a new experiment, without
reinterpreting that round. No published result forecasts this task's gain.

Ordinary Gaussian NLL can attenuate mean updates when learned variance is
large. [Seitzer et al., ICLR2022](https://arxiv.org/abs/2203.09168) and the
[authors' implementation](https://github.com/martius-lab/beta-nll) motivate
weighting each row's NLL by `stop_gradient(variance**beta)`. Freeze beta=0.5,
with beta=0 ordinary NLL as matched ineligible control. No beta search.

Both arms use V33's single-Gaussian architecture/initialization and training
settings, including the unused logit output. Use the same normalizing constant
in both objectives. Gaussian mean and median coincide. Validate the analytical
loss and gradients independently; paper RMSE/NLL findings do not establish
WMAPE improvements. This does not reopen fixed sample reweighting, histograms,
CRPS boosting, residual correction or mixture-component scans.

## Reference and validation

Use latest remote-registered V32_TIME_A60V7_50 (user-reported96.3727,
not independently verified): iron0.5*V36+0.5*V12 raw joint member;
time0.2*V36+0.3*N0048+0.5*V7 raw periodic member. The iron column equals
the released original V12 iron. Report historical B0 as well. Do not confuse
raw members with previously released blended columns or use a locally stronger
reference merely because it ranks higher out of fold.

Use verified native caches on matching outer training/query rows and existing
V5 fold protocol. Verify source/spec/data/fold/fit/query/refit/epoch identities
and audit hashes for **all four seeds before official fits**. Team-private
V30/V33/V39 caches are absent here and are not substituted. New reference
fits=0; any missing or incompatible cache stops G0 without hidden retraining.
Fixed0.2 isolated blend needs no inner-calibration reference predictions.

Development: two arms*time only*two seeds42/3407*five folds =20 outer fits,
40 optimizers (inner MAE epoch selection then fresh outer refit). Inner split
uses the native group-safe seed42 fold0 convention. Only BETA05 can advance;
both development seed gains must be positive, mean>=0.01, with positive mean
advantage over NLL. Local working gate96.25 is preserved.

Only complete independently audited eligible development enters a **new**
confirmation directory: repeat both arms at7777/12011, maximum20 outer fits,
40 optimizers. Final promotion needs all four candidate seed gains positive,
positive seed-level paired one-sided95% LCB, development score>=96.25 and
positive four-seed mechanism mean over NLL. Disclose mechanism LCB; folds
are descriptive. No cross-seed OOF mixing. Repeated splits measure stability,
not independent new observations.

## Resource, evidence and stopping

After cache verification, synthetic preflight permits at most4 optimizer
starts: two full-path resource probes and two learnability runs. Formal-sized
preprocessing/model/optimizer/validation measurement projects both selector
and fresh-refit max epochs, with1.5 uncertainty factor and300s audit/I/O margin.
Workers<=4, all numeric thread caps1, peak worker<=1GiB, development projection
<=2h. This is a stopping cap, not a measured ETA. Cache/resource/learnability
failure stops the frozen experiment; no shrinking, retries or second spec.

Freeze transitive source/runtime/data/spec/reference hashes, retain inner and
outer transforms, selected epochs/traces, IDs, saved models, predictions and
append-only optimizer/fit ledgers under `local/`. Audit cold order/chunk
invariance to1e-8 and recompute scoring/eligibility independently. Never
replace failed evidence or overwrite run directories. Separate G0 from G1.

Check quietly every30minutes only while a process needs monitoring. Notify
failure/stoppage/completion/action; no continuous polling. Full-data fits,
packages, desktop writes and uploads=0. Users retain platform upload control.
Public changes use ordinary commits/pushes on current configured upstream;
models, predictions, local reports, ledgers, receipts and ZIPs stay outside Git.