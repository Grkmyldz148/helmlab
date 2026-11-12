---
status: accepted
date: 2026-11-08
---

# No AI co-author lines

## Context and Problem Statement

The commit that npm 1.0.1 was built from (a2f3a63) ends with an AI co-author line. Tools are not authors of this project.

## Considered Options

* Allow such lines
* Do not allow them

## Decision Outcome

Chosen option: "Do not allow them": commits, pull requests, tags, release notes and public posts carry no AI co-author or "generated with" lines, and git hooks reject them. How AI tools are used is described in GOVERNANCE.md.

### Consequences

* Good, because authorship and responsibility stay with the maintainer.
* Bad, because a2f3a63 stays public; GitHub does not remove non-sensitive data, so the release notes of the next release record that it has the same tree as tag v1.0.1.
