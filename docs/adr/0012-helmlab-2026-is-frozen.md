---
status: accepted
date: 2026-11-09
---

# Helmlab 2026 is frozen; the neutral correction is a separate form

## Context and Problem Statement

Color.js and ColorAide implement MetricSpace without the neutral correction. The helmlab package and color-space apply it by default. ΔE helmlab is computed from the form without it.

## Considered Options

* Make the neutral-corrected form normative
* Make the base form normative and give the neutral-corrected form its own identifier

## Decision Outcome

Chosen option: "Make the base form normative and give the neutral-corrected form its own identifier". Helmlab 2026 is the output of helmlab 1.0.0 and never changes. `helmlab-metric` names the base form; the neutral-corrected form is `helmlab-metric-neutral`, described in Appendix B of the specification. New mathematics ships as pre-releases under new identifiers.

### Consequences

* Good, because existing outputs stay valid, and Color.js and ColorAide already implement the normative form.
* Bad, because `hl.metric` returns a different form from `helmlab-metric`, and the documentation has to say so.

## More Information

Until 2026-11-12 the reserved identifier was `helmlab-metric-display`. It was renamed in helmlab#4, because color-space uses the suffix `-display` for display-encoded spaces.
