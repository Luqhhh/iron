# V12 isolated iron delivery (2026-09-27)

**V12_IRON_JOINT_PLR001_A50** is delivered to
`C:\Users\lqh22\Desktop\submission\V12_IRON_JOINT_PLR001_A50\Luqhhh_bf_tap_predict_round2.zip`.
ZIP SHA-256:
`a1c205a6722da3976a12e258458b649967c7c25130a2c55d840c5ecb2a1dc669`.
The user explicitly authorized this candidate on2026-09-26; delivery completed
after midnight on2026-09-27 (Asia/Shanghai). Agent uploads0. Platform score is
unreported; the current reported best remains A35=96.3366 and96.4 is unverified.

Only iron is replaced: `0.5*A35_iron+0.5*V12_joint_plr001_iron`. Time retains the
A35 original CSV field strings. Both original development splits chose this
A35-relative weight; no release-time weight search was performed. Joint training
uses both training targets, but its predicted time output is not used in the ZIP.
No V7/V9 full-data model or combined two-target package was generated.

## G0: completed checks

Locked Python3.12 suite **1052 passed,23 warnings**. Release consumed exactly one
replay fit and one full-data joint fit, each retaining inner epoch selection and
fresh full-training-partition refitting. Joint seed42/fold0 reproduced both output
columns exactly, then passed independent cold inference. Full-data training used
all2754 rows, selected85 epochs, and did not hit the240-epoch selection cap.
The model is trained from scratch; no external weights/data were used.

The delivered322-row ZIP passed template ordering and uniqueness, archive CRC
and single-result.csv structure, finite/nonnegative output, exact blend arithmetic,
and **zero time-string mismatches**. Warm/cold full-batch prediction difference0;
reverse/chunk/singleton checks meet the frozen tolerance. The desktop copy hash
matches the private archive. A second read-only audit independently verified the
actual desktop ZIP, cold/warm arrays, original parent/time strings, arithmetic,
source/input hashes and the exact two-fit ledger. A35 and the original V7 desktop
ZIP remain byte-identical to their recorded hashes.

Private run: local/runs/round2-v12-joint-tabm/release-r1. Its verification.json,
delivery.json and independent-delivery-audit-r1.json bind the model, ZIP, replay,
full training and desktop copy evidence. Models, predictions, access/fit ledgers,
private audits, authorization receipt and ZIPs remain outside Git.

## G1: qualified direction, explicit release exception

Nested four-seed gain over A35 iron: mean+0.016088, paired LCB95+0.010939,
4/4 positive seeds and18/20 positive folds. Increment beyond V9 iron:
mean+0.011627, LCB95+0.008416,4/4 positive seeds and16/20 positive folds.
Folds are descriptive; previously used split seeds are not new independent data.
The frozen development package score96.226441 remains below96.25. Release was
explicitly authorized for this one candidate; no threshold or historical
promotion decision was changed. Local gains do not establish platform magnitude
or rank. The user uploads and returns the candidate's score.
