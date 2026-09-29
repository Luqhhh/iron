# E development resume after scoped metadata repair

Recovery-r2's complete synthetic/resource/cold admission passed at23:23:55 CST
on2026-09-29. Its development attempt failed before any official fit or run
directory creation. Resolving the shared local/ symlink made the physical
admission path a sibling of the worktree, so relative_to(worktree) was invalid.

private_reference now stores the logical path under the same bounded private
run root, while validating its resolved location. It handles ordinary and shared
local directories and rejects unrelated paths. Only one new helper and one
manifest metadata expression differ from the passed-admission runner. An AST
comparison strips that specific path metadata change and verifies the rest of
the original runner is identical, including training, scoring, seed, weight,
partition and gate computations. Regression checks reject numerical/model
changes through this source bridge.

The new recovery-r3 workflow preserves the successful original admission and
every previous failure. It verifies the original report/manifest/source identities
against source commit81c74cc and binds all28 original admission artifacts. Only
the declared path-routing and test source delta is permitted. Original DE3's
90 completed artifact identities remain verified. Runtime, spec and resource
gates are unchanged; currently available memory is checked again.

A fresh subprocess cold-verifies the original selected states, native epoch
arithmetic and full refits. No synthetic fitting is repeated. The new admission
bridge report explicitly identifies the original report and SHA256, original
training provenance, source delta and zero new training runs. Its original
metrics/costs are historical measurements; current source hashes attest safe
reuse, rather than asserting that old models were fitted with later code.

Only after that zero-fit revalidation does complete E development resume in a
fresh directory, followed by independent audit and confirmation only for earned
finalists. Actual600-second observations and terminal events drive supervision.
No automatic retry, DE3 repetition, recipe expansion, full-data fit, new package,
desktop write or platform upload. The old ETA must be measured from the actual
new development start; synthetic admission is already complete.

Validation:1347 full locked Python3.12 tests passed,23 existing warnings;
40 focused tests passed. Both ordinary/shared-private-root references roundtrip
to the original artifact, and unrelated paths are rejected. Source-delta tests
reject changed candidate computation or package weight. The zero-fit check
verifies all28 original admission artifacts and source identity. Full log:
local/runs/independent-ensemble-checkpoints-20260929/engineering-r3/pytest-python312.log.

## Real-data development started

Implementationb88f6ca was ordinary-pushed before launch. Original r2 service
terminal state is retained underengineering-r3/. The named transient service
started the explicit r3 workflow23:57:40 CST on2026-09-29, initialMainPID313912
active/running, writing a new E-recovery-r3-workflow-output.log. The zero-fit
source/artifact/cold admission bridge ended23:57:47 with exit0 and0 new fits.
The original successful admission report and its provenance remain unchanged.

E development started23:57:47, created its manifest, and completed the10 reused
outer reference tasks. Actual official-data A-only inner-reference fitting
started23:57:52 for split42/fold0. There are10 legal inner-reference units
before the20 candidate units. This establishes real G1 execution, not a quality
or promotion result. Source/config are frozen; subsequent observations occur
only at actual600-second wait events or stage completion/failure events.
