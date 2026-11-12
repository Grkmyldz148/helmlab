---
status: accepted
---

# The behavior reported in helmlab#4 is documented, not changed

## Context and Problem Statement

helmlab#4 (opened 2026-10-19) showed that the neutral correction leaves grays with chroma above Y = 2, and that the paper describes the correction differently from the code.

## Considered Options

* Change the code in a 1.0.x release
* Document the behavior and keep the outputs

## Decision Outcome

Chosen option: "Document the behavior and keep the outputs". The specification lists the behavior as a known limitation and as erratum E-2026-0001; a change belongs to a later definition.

### Consequences

* Good, because implementations that match helmlab 1.0.0 stay conformant.
* Bad, because the gray chroma above Y = 2 stays in Helmlab 2026.

## More Information

The decision was accepted before 2026-11-08; its exact date is not recorded.
