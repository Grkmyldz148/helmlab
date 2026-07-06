#!/bin/bash
export TASKS=./apiqa.mjs
export SKILL=../../helmlab/SKILL.md
st(){ for r in 1 2 3; do node ab_runner.mjs $1 $2 api$r >> grid_log.txt 2>&1; done; }
st off claude-haiku-4-5-20251001 & st on claude-haiku-4-5-20251001 &
st off claude-sonnet-4-6 & st on claude-sonnet-4-6 &
wait; echo API GRID DONE
