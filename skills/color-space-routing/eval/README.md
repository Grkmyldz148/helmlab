# ColorQA — measuring whether the skill actually works

10 auto-verifiable color-engineering tasks (gradient hue detours, ΔE misuse,
gray axis, physical-vs-perceptual mixing, hue wrap, LCh distance trap, CVD,
picker, HDR, noticeability). The harness is validated: golden answers pass 10/10.

A/B protocol:
1. `node colorqa.mjs list` → give each prompt to the model in a FRESH context,
   (a) without the skill, (b) with SKILL.md loaded. No other color context.
2. Collect answers as `{taskId: answer}` JSON (code tasks: JS expression string).
3. `node colorqa.mjs verify answers.json` → pass rate per arm.

## Results (2026-07-06, 20 tasks × 3 reps, fresh `claude -p` sessions)

| Model | skill-off | skill-on | pkg-independent off → on |
|---|---|---|---|
| claude-haiku-4-5 | 40.0% (24/60) | **93.3%** (56/60) | 54.8% → **97.6%** |
| claude-sonnet-4-6 | 41.7% (25/60) | **88.3%** (53/60) | 52.4% → **92.9%** |

Off-arm failure modes: hallucinated APIs, gamma-space physical mixing,
HSL-as-perceptual, wrong space routing. Remaining on-arm failures are
themselves actionable: sonnet insists on `in oklch` for distant-hue CSS
gradients (form-rule not landing) and once assumed L∈0–100 on GenSpace's
0–1 scale — both now noted for the next skill revision. Raw per-run
answers are committed (`answers_{arm}_{model}_r{rep}.json`); the harness
is validated by golden answers (20/20) before any grading.

**Skill v1.1** (form-rule made imperative + L-scale note, targeting the two
systematic on-arm failures): sonnet's T12 `in oklch` insistence 3/3 → 0/3,
T14 L-scale error 3/3 → 1/3. On-arm totals: haiku 93.3% (unchanged),
sonnet 88.3% → 90.0%. Remaining failures are scattered single-rep misses
(nondeterminism floor), no systematic mode left. Off-arm numbers are
unaffected by skill revisions (no skill in that arm).


## API-QA (companion `helmlab` skill, 2026-07-06)

8 API-usage tasks (exporter space, palette direction, trained metric,
distance contracts, LCh darken…), same protocol, `TASKS=./apiqa.mjs`:

| Model | skill-off | skill-on |
|---|---|---|
| claude-haiku-4-5 | 8.3% (2/24) | **75.0%** (18/24) |
| claude-sonnet-4-6 | 37.5% (9/24) | **91.7%** (22/24) |

Golden answers 8/8 before grading; raw per-run answers committed.


## Split-vs-monolith experiment (2026-07-06): monolith WINS

Tried progressive disclosure (80-line core SKILL.md + references/*.md read
on demand, incl. a new testing.md). On-arm re-run, 3 reps x 2 models:
haiku 93.3% -> 80.0%, sonnet 90.0% -> 81.7%. Headless models answer from
the thin core instead of reading reference files — recipes moving out hurt
code tasks most. **Verdict: the published skill stays a single SKILL.md.**
The testing content will be merged INTO the monolith instead (pending
re-measure). Split variant kept in color-space-routing-split/ as the
negative-finding artifact.
