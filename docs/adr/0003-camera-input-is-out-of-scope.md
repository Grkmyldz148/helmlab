---
status: accepted
date: 2026-10-23
---

# Camera input is out of scope

## Context and Problem Statement

Camera pipelines turn sensor data into scene colors (input transforms, raw processing). Helmlab could try to cover that step.

## Considered Options

* Include camera input transforms
* Leave them out

## Decision Outcome

Chosen option: "Leave them out", because they are not the job of a color space. Instead the space stays finite, monotone and invertible for negative input and for input above white, so that a camera pipeline can feed it.

### Consequences

* Good, because the scope stays small.
* Bad, because users need another tool for sensor data.
