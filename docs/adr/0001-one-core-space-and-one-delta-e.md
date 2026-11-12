---
status: accepted
date: 2026-10-23
---

# One core space and one ΔE on the same coordinates

## Context and Problem Statement

Helmlab 2026 has two spaces: GenSpace for generating colors and MetricSpace for measuring color differences. Web, photo and cinema work could each get a space of its own. Every additional space multiplies the work of porting, testing and maintaining it, and users have to choose between them.

## Considered Options

* One core space per field (web, photo, cinema)
* Two spaces, one for generating and one for measuring (as in Helmlab 2026)
* One core space with one color-difference formula on the same coordinates

## Decision Outcome

Chosen option: "One core space with one color-difference formula on the same coordinates", from the next definition on, because it serves web, photo and cinema with one thing to port and test. CIELAB with CIEDE2000 and Oklab with ΔEOK follow the same pattern. Helmlab 2026 keeps its two spaces unchanged (ADR-0012).

### Consequences

* Good, because a difference is measured on the coordinates that were used to make the colors.
* Bad, because one space has to serve both tasks; ADR-0009 lists the gates it must pass.
