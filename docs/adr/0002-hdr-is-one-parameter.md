---
status: accepted
date: 2026-10-23
---

# HDR is one parameter of the same space

## Context and Problem Statement

HDR content goes above diffuse white. A separate HDR space would double the definitions to port and maintain.

## Considered Options

* A separate HDR space
* The same space with a diffuse-white parameter

## Decision Outcome

Chosen option: "The same space with a diffuse-white parameter", default 203 cd/m² (ITU-R BT.2408), from the next definition on. Helmlab 2026 has no HDR parameter.

### Consequences

* Good, because SDR and HDR use one space.
* Bad, because the space must stay finite, monotone and invertible far above diffuse white (ADR-0009).
