# Release engineering checks (2026-10-01)

The new12focused checks pass, including actual tiny EMA selection/fresh-refit
models compared against the unchanged unwrapped numerical method: predictions,
selected epoch and every saved model parameter are exactly equal. Tests also
pin the two different replacement formulas, unchanged field strings, refusal
of bad/negative endpoints, and preservation/refusal of failed reservations.

Locked Python3.12 --no-sync, all four numerical thread variables=1: complete
suite coverage1337passed cases,0skips,23existing warnings. Execution evidence
is reported precisely rather than hiding fixture-repair failures:

1. First complete invocation:2failed/1326passed/9skipped. The new worktree
   lacked historical private OOF caches; public sources unchanged.
2. Exact copies of existing round2 JSON/CSV/NumPy references were added privately
   (7411files,0models/0newfits). Second complete invocation:1failed/1336passed/
   0skips. The only remaining failure was the missing original V5 submission ZIP.
3. Copying that original byte-identical ZIP and rerunning its failed readback
   case on the same locked path passed1case. No public source/model/test change
   occurred between these checks. The original full-suite failures remain as
   evidence;1337complete-suite cases are now covered by passed results. A third
   repetition of the already-passed1336cases was unnecessary.

Complete-coverage receipt SHA256:
f29bac35fe91fad3262579995229d9951c62a1df84a9573f939301750c3df732.
Second complete invocation receipt:
1bc6de61fcc8d95219da7733d3ee97aa98cbbe030588891652732799d23a727b.
First invocation receipt:
9d4ac0e5e6c2615126ebbb1f3ce323e5844283f835b2978ca8c4593ae4e184a8.
Private reference-copy receipt:
2b12c441562962b541599fb9972207c478cded5531b0b3d47da83d481aaccc68.
Original V5 ZIP:
5ed99b8014fb1650b88b6ae57cc3ed4dac378a20831eab5efc800de91ab2c982.

This prepares the exploration wrapper only. Original PTaRL scientific sources,
exact-source full-suite manifest, non-time resource requirements and scientific
execution recipe remain immutable. Complete original development cold audit
and independent arithmetic are checked again before a new full-training run.
No official new fit/package has yet been executed at this engineering record.
