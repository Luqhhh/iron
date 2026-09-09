# Complete v0.5 component diagnostics and bounded follow-up

R2 improved the equal-horizon objective but regressed near-term development loss.
This change restores verified original/R2 component predictions without fitting,
evaluates the three fixed crossings, then conditionally trains the two registered
component changes with an append-only 24-target-fit budget. Separate STAGE_A and
GENERAL gates retain the original thresholds and report target/month/spout detail.

All five candidates fail both gates. P2 recovers H1 but exceeds the H4 risk limit;
T1/T2 also breach registered limits. The phase closes with R2 retained and no
OPT-16 final fit or challenger package. Old source, evidence and release pointer
remain unchanged. Conditional mixed-bundle release code rejects failed gates;
its successful release path was not exercised with real data.

Validation: 40 real endpoint checks (maximum difference 1.71e-13), zero OPT-14 fits,
24 completed OPT-15 target fits, zero inference fit attempts and 161 locked Python
3.12 tests. Models, row predictions, ledgers and run reports remain local. No Git
push, platform upload or desktop replacement is included.

B/C label-free cold-process checks passed with the incumbent R2: test_b 322 rows
(all H2), test_c 548 rows (303 H3 and 245 H4). Report:
`local/runs/optimization-v0.5-stage-check-r1/stage_mapping.json`. No labels were
scored and no B/C ZIP was created; these checks make no platform-quality claim.
