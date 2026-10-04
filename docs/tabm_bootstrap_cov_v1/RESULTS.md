# TABM_BOOTSTRAP_COV_V1 results

Online Poisson1 Bootstrap with an actual-fit covariance quadratic loss completed the preregistered development comparison against the original online Bootstrap. The time endpoint is A20 = 0.8 Q75 + 0.2 member within each split. Both development seeds improve on Q75, but both trail the paired online Bootstrap control. The frozen positive mechanism gate was not met; no confirmation was reserved or fitted. Retain the sole positive time exploration candidate at lower priority than original online Bootstrap and prediction fusion, without revising the historical gate or claiming a platform result.

## G0 engineering

Passed: 2 new saved states and 2 optimizer runs, with 4 candidate/control models independently cold audited. The candidate actual-fit covariance and explicit 2x2 precision reconstruction, and both arms' independent PCG64 Poisson/order certificate replay passed. Partition identities, preprocessing, selected epochs, fresh refit, complete/reverse/chunk predictions and budgets passed. Maximum prediction difference 7.516235626781054e-6; peak RSS 436.625 MiB. No new control or reference fits.

## G1 development

Completed with actual exit 0: 10 outer fits, 20 saved states and 20 optimizer runs. Seeds 42 and 3407 each have complete five-fold OOF over 2754 unique IDs. All 40 new/control checkpoints passed independent cold audit; maximum prediction difference 5.6196785408246797e-5, peak RSS 437.53125 MiB. Original controls retain their source directory, phase, unit and split identities. Independent final zero-fit verification recomputed complete OOF, original reference identities, A20 endpoint arithmetic, scalar-summed gains, frozen time-only eligibility, source hashes, budgets and actual exit. No training or cold inference was repeated.

| Time A20 gain relative to Q75 | Seed 42 | Seed 3407 | Mean |
|---|---:|---:|---:|
| Bootstrap + Cov | +0.000254357721 | +0.002692478152 | +0.001473417937 |
| Original online Bootstrap | +0.002350669349 | +0.006934220048 | +0.004642444699 |
| Paired mechanism difference | -0.002096311628 | -0.004241741896 | -0.003169026762 |

Confirmation consumed 0 outer fits, 0 states and 0 optimizers, including 0 reservations. No four-seed LCB is computed and no formal promotion is claimed. Iron is descriptive only: seed gains +0.000809256801 / -0.000256705264, mean +0.000276275769, paired mechanism mean -0.000111040284. Iron cannot trigger confirmation or a second exploration recommendation.

The combination did not preserve the development advantage of online Bootstrap in this recipe. Prediction-level complementarity of Bootstrap and Cov was motivation, not evidence that their joint training benefits would add. This result applies to the frozen composition; it does not invalidate either model family. Local gains, signs and rankings do not predict platform performance. Existing split seeds reuse the same official samples and are not independent new datasets. Q75 platform 96.3920 is user reported and not independently verified; a complete current-package local score is not claimed.

Necessary full tests ran once: neural 1508 passed / 23 warnings; locked Python 3.12 1071 passed / 114 skipped / 3 warnings. Production science was frozen before fitting, with no scientific retry or source repair. The targeted RED log destination issue is preserved in a private observed-output receipt; it was not a scientific training failure.

Private raw models, weights, predictions, labels, ledgers, audits and final-verification receipts remain under local/runs/tabm-bootstrap-cov-v1 and local/bootstrap-cov-20261004. Fullfit 0 / packages 0 / uploads 0 / desktop writes 0. This authorized stage is terminal; the monitor will be removed after complete result publication. No further recipe has been started.
