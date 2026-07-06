// ColorQA A/B runner: solves each task via `claude -p` in a FRESH headless
// session (cwd=/tmp so no project context leaks), with and without SKILL.md.
// Usage: node ab_runner.mjs off|on [model]
import { TASKS } from './colorqa.mjs';
import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';

const arm = process.argv[2];                 // 'off' | 'on'
const model = process.argv[3] || 'claude-haiku-4-5-20251001';
const skill = arm === 'on' ? readFileSync('../SKILL.md', 'utf8') : '';
const answers = {};
for (const t of TASKS) {
  const fmt = t.type === 'code'
    ? 'Reply with ONLY the raw JS expression/function on one line. No markdown fences, no prose, no variable declarations.'
    : 'Reply with the answer only, one short line.';
  const prompt = (skill ? `Reference material you may use:\n\n${skill}\n\n---\n\n` : '')
    + `${t.prompt}\n\n${fmt}`;
  let out = '';
  try {
    out = execFileSync('claude', ['-p', prompt, '--model', model],
      { cwd: '/tmp', encoding: 'utf8', timeout: 180000 }).trim();
  } catch (e) { out = 'ERROR: ' + String(e.message).slice(0, 80); }
  out = out.replace(/^```[a-z]*\n?/,'').replace(/\n?```$/,'').trim();
  answers[t.id] = out;
  console.log(`[${arm}] ${t.id}: ${out.slice(0, 70).replace(/\n/g,' ')}`);
}
writeFileSync(`answers_${arm}.json`, JSON.stringify(answers, null, 1));
console.log(`wrote answers_${arm}.json`);
