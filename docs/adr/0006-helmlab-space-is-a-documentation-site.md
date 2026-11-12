---
status: accepted
date: 2026-11-09
---

# helmlab.space is a documentation site

## Context and Problem Statement

The previous site mixed marketing with performance figures that have since been withdrawn. One person maintains the project.

## Considered Options

* An atlas site like color-space.io
* A general color-library site
* A marketing page
* A documentation site

## Decision Outcome

Chosen option: "A documentation site": the specification, the test vectors, package documentation, one evidence page and a ledger of withdrawn claims, served as static files on Cloudflare Workers. It is the only one of the four that one maintainer can keep current.

### Consequences

* Good, because every figure comes from one evidence file.
* Bad, because there is no interactive playground at launch.
