#!/bin/bash
export SKILL=../../color-space-routing-split/SKILL.md
export REFS=/Volumes/harici_ssd/color-space/helmlab-main-repo/skills/color-space-routing-split/references
st(){ for r in 1 2 3; do node ab_runner.mjs on $1 split$r >> grid_log.txt 2>&1; done; }
st claude-haiku-4-5-20251001 & st claude-sonnet-4-6 &
wait; echo SPLIT DONE
