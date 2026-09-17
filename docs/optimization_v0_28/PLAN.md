# optimization-v0.28 fixed equal blends

This phase implements the two algorithms frozen in the user-approved v0.28 plan.

- `V28I_CB_QRF_EQUAL_BLEND`: average the completed V26A and V27I iron endpoints 50/50; copy V26A time exactly.
- `V28T_L1_L2_QRF_EQUAL_BLEND`: average the completed V26A and V21 time endpoints 50/50; copy V26A iron exactly.

Both averages cover every sample and are performed after each source endpoint has completed its own model and post-processing. Six-decimal endpoint strings are converted to integer micro-units, averaged with ties-to-even, and written with exactly six decimals. No model, preprocessor, calibration, gate, route, or blend weight is fitted.

The implementation base is `eec5c95bef392348c2d7597d64b01a488be1d71c`, the latest v0.27 publication head. It descends from the plan's reviewed `484d627` point and does not change the frozen v0.27 algorithms or artifacts.

The two definitions, their final files, and their packages must be frozen before any new platform feedback. Platform order is A then B, one explicit test each. Uploading, desktop writes, and public push are outside the automatic lifecycle.
