# ColorQA — measuring whether the skill actually works

10 auto-verifiable color-engineering tasks (gradient hue detours, ΔE misuse,
gray axis, physical-vs-perceptual mixing, hue wrap, LCh distance trap, CVD,
picker, HDR, noticeability). The harness is validated: golden answers pass 10/10.

A/B protocol:
1. `node colorqa.mjs list` → give each prompt to the model in a FRESH context,
   (a) without the skill, (b) with SKILL.md loaded. No other color context.
2. Collect answers as `{taskId: answer}` JSON (code tasks: JS expression string).
3. `node colorqa.mjs verify answers.json` → pass rate per arm.

Report both arms + per-task diffs. Expand the set before publishing numbers
(target: 20+ tasks, multiple models).
