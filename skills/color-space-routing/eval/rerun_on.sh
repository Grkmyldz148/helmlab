#!/bin/bash
for r in 1 2 3; do node ab_runner.mjs on claude-haiku-4-5-20251001 v2r$r >> grid_log.txt 2>&1; done &
for r in 1 2 3; do node ab_runner.mjs on claude-sonnet-4-6 v2r$r >> grid_log.txt 2>&1; done &
wait; echo V2 DONE
