# color-space-routing — an AI skill for picking the right color space

There is no perfect color space (we tried). There IS a right space per task —
this skill teaches your AI assistant the routing table, with measured numbers,
including where each space loses.

## Install

**Claude Code** — copy the folder into your skills directory:

```bash
mkdir -p ~/.claude/skills
cp -r skills/color-space-routing ~/.claude/skills/
```

(or per-project: `.claude/skills/color-space-routing/`). Claude picks it up
automatically whenever a task touches color.

**Cursor / other assistants** — `SKILL.md` is plain markdown: paste it into
`.cursorrules`, an `AGENTS.md`, or your system prompt.

## What's inside

- Task → space routing table (gradients, palettes, ΔE, CVD, HDR, CSS, wide gamut)
- The measured evidence (STRESS on 5 human datasets, ColorBench 90-metric head-to-head)
- Copy-paste recipes (CSS, JS, Python)
- A 9-item pitfall checklist — each entry is a bug we actually hit

Numbers regenerate from the same pipeline that feeds
[helmlab.space/benchmark](https://helmlab.space/benchmark/), so the skill and
the site can't drift apart.
