---
status: accepted
date: 2026-09-25
---

# Data split: TRAIN, VALIDATION, LOCKED, BLIND

## Context and Problem Statement

Earlier models were tuned and scored on the same data, so their results said little about new data.

## Considered Options

* Use every dataset freely
* Split the data before new models are fitted

## Decision Outcome

Chosen option: "Split the data before new models are fitted", written down on 2026-09-25. TRAIN and VALIDATION data may shape a model. A LOCKED dataset is scored once, after the model's hash is recorded. A BLIND dataset is scored once, after the model is frozen. Using LOCKED or BLIND data for fitting or design voids the model's claim on that data.

### Consequences

* Good, because results on LOCKED and BLIND data say something about new data.
* Bad, because less data is available for fitting, and BLIND data can be used only once.
