# Serial augmentation and representation results

The serial recovery completed on 2026-09-29 at 20:33:47 CST with exit status 0.
All 60 new development units completed; 20 verified native BASE units and ten
references were reused. No development finalist was selected for either target.
The conditional confirmation command exited with no finalist and zero fits.

## G0: execution and saved-state audit passed

The recovered development-r2 retains the original Budget8 manifest verbatim.
All 425 frozen source/configuration/test hashes and data hashes were verified
after completion. All 30 copied complete directories are byte-identical to
their originals. The four original partial directories remain unchanged with
start events only. Both the original runtime-only failed admission and the
explicit eight-hour readmission remain preserved.

The development phase ended at 20:33:14; the unchanged independent auditor
passed at 20:33:42. The complete workflow took approximately 75.57 minutes.
There were no failed units and no automatic retry. The user service exited
successfully with MainPID=0; RemainAfterExit retains its exited status.

| Check | Result |
| --- | ---: |
| Complete development units, including reused BASE | 80/80 |
| New selection and refit runs | 120 |
| Independently audited saved models | 160 |
| Native BASE replays | 20 |
| Native prediction maximum difference | 0 |
| Full-batch cold prediction difference | 0 |
| Maximum order/chunk difference | 0.000056312603 |
| Frozen order/chunk tolerance | 0.0005 |
| Selection and score arithmetic | Passed |
| Locked Python 3.12 tests before launch | 1318 passed, 23 existing warnings |

No model, training, evaluator, gate or frozen source was changed. The README
cleanup during execution is outside the registered training source inventory.
The auditor was used unchanged; no corrective audit or new audit fit was needed.

## G1: no candidate passes the frozen development gate

These are isolated package score differences against the fixed current
reference, averaged over complete five-fold coverage at split seeds 42 and
3407. The other target and all other composition weights remain fixed. They
are local development evidence, not platform forecasts.

| Target | Method | Seed 42 | Seed 3407 | Mean | Positive folds, descriptive |
| --- | --- | ---: | ---: | ---: | ---: |
| tap_iron | D-LMIX | +0.00975515 | -0.01263020 | -0.00143752 | 5/10 |
| tap_iron | R-FIXED | -0.00926203 | -0.01762981 | -0.01344592 | 2/10 |
| tap_iron | R-LEARNED | -0.00455420 | -0.01477476 | -0.00966448 | 1/10 |
| tap_time_len | D-LMIX | -0.01201120 | -0.00720690 | -0.00960905 | 2/10 |
| tap_time_len | R-FIXED | -0.01707860 | -0.02348864 | -0.02028362 | 0/10 |
| tap_time_len | R-LEARNED | -0.01786124 | -0.01842822 | -0.01814473 | 2/10 |

The matched BASE replay has zero gain on both seeds and both targets. Every
new method has a negative mean gain, fails the two-positive-development-seed
rule, falls below the +0.01 mechanism floor and has no positive mean advantage
over BASE. The general candidate-tier classifier independently places all six
target/method candidates in not_shortlisted; formal and exploration lists are
empty. Fold statistics remain descriptive.

The iron D-LMIX result illustrates the complete-coverage rule: its positive
seed-42 result does not repeat at seed 3407. Neither confirmation seed
271828/314159 was consumed. No four-seed promotion claim is made. This batch
closes these exact frozen configurations; it does not establish that every
possible augmentation or representation method fails.

Full-data fits, packages, desktop writes and agent uploads remain zero. The
registered user-reported platform reference remains V32_TIME_A60V7_50=96.3727;
there is no new platform result or released candidate from either serial queue.

## Artifact identities

Private output: local/runs/component-augmentation-representation-budget8/development-r2.
Private supervisor: the same run root under recovery-r2.

| Artifact | SHA-256 |
| --- | --- |
| Original and recovered manifest | 33d7b5e143e887e9b3c6d9db579fcf358d3c566235b7b5bbb6524aa5c01bbbe8 |
| Complete development summary | 3cbdbe05b20a96150c7f6c87b289e42f6c876966163bd3ea75474230b7cfa2c7 |
| Independent audit | fc28b8d856e49855e566838cef6e09cc346acc4a082e46c03ccbacdea7d58e15 |
| Recovery provenance | 90b96c0a131749933ec7a1e354743d8568226d35f39c21f7420d99d85f1a9854 |

Only aggregate results and artifact digests are public. Saved models,
predictions, machine reports, events and original interrupted evidence stay
under ignored local paths.
