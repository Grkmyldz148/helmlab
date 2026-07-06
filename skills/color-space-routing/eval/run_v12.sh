#!/bin/bash
st(){ for r in 1 2 3; do node ab_runner.mjs on $1 v12r$r >> grid_log.txt 2>&1; done; }
st claude-haiku-4-5-20251001 & st claude-sonnet-4-6 &
wait; echo V12 DONE
