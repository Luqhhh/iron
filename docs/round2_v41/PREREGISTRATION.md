# V41: continuous mixture bound from all measured scores

The objective remains platform 96.5. This is zero-fit mathematical review,
not candidate selection or permission to release packages. The observations
are frozen in SPEC.yaml from the named V40 source commit. They are user reports,
not verified receipts. The concurrent V39 interior analysis and our completed
V39 hard-tree study have distinct branch identities; neither is renamed.

Assume the documented fixed-row additive WMAPE metric and correct package
identities. Work with the **unclamped** score, which is concave and piecewise
linear in fixed prediction mixture weights. Positive observations identify
their unclamped values. Clamp any derived global upper bound at zero afterward.
Four-place scores receive a conservative +/-0.00005 interval; old time scores
lifted by B0 minus the V7 package receive +/-0.00015. Correlations among these
intervals are discarded conservatively. Extra unknown report errors are not
covered; all conclusions remain conditional on the stated assumptions.

At each observed anchor a_i, every supporting gradient g obeys
`g.(a_j-a_i) >= lower_j-upper_i`. Maximizing `g.(x-a_i)` over this polyhedron,
then adding upper_i, bounds the true score at x. A finite maximum supplies
dual nonnegative weights as a directly checkable certificate. Infeasibility
means inconsistent assumptions/data and must stop the analysis; unbounded
anchors cannot supply that upper bound.

For one anchor this upper-envelope function is convex in x. Its maximum over
the entire simplex is bounded by its maximum at the domain vertices. Taking
the smallest such maximum across anchors is a conservative continuous bound,
not a grid estimate and not necessarily the tightest bound. Apply the same
construction to the iron interval; subtract the lower score at w=0.5 to bound
iron improvement, then add to the time upper bound. Do not infer an attainable
score from an upper bound. Equal observations do not prove a plateau or an
interior optimum. No threshold or historical decision changes.

Validation: analytic concave functions in one/two dimensions, an interior
peak with equal measured values, interval enlargement monotonicity, unbounded
and inconsistent systems, and dual-certificate arithmetic. Preserve all local
outputs append-only. Required locked Python 3.12 suite and private-artifact
checks precede publication. No model fits, target reads, packages or uploads.
