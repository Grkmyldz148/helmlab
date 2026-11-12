---
status: accepted
date: 2026-11-08
---

# Quality charter for the next core space

## Context and Problem Statement

GenSpace and MetricSpace have measured defects: a piecewise-linear lightness correction with slope jumps, iterative inverses, a neutral-correction table, a lightness jump on a zero-chroma curve, and a selection on in-sample win counts (specification, Section 10).

## Considered Options

* No fixed gates
* Gates derived from the measured defects

## Decision Outcome

Chosen option: "Gates derived from the measured defects". Engineering gates: C² smoothness; closed-form forward and inverse; a neutral axis that is exact by construction; a finite, monotone and invertible domain; round trips within 1e-12; a valid gamut cusp at every hue; Python and JavaScript within 1e-12; speed within a few times that of Oklab; Helmholtz–Kohlrausch in a separate channel; continuity at zero chroma. Scientific gates: no fitting to a benchmark's win count; fit on TRAIN, select on VALIDATION, score LOCKED and BLIND data once (ADR-0011); few parameters, each with a stated purpose.

### Consequences

* Good, because the known defects cannot come back unnoticed.
* Bad, because fewer candidates pass and the work takes longer.

## More Information

The gate on continuity at zero chroma was added on 2026-11-09. A check of lightness steps near black and near white was added on 2026-11-12, after helmlab#5.
