#!/bin/bash
run_stream() { for r in 1 2 3; do node ab_runner.mjs $1 $2 $r >> grid_log.txt 2>&1; done; }
run_stream off claude-haiku-4-5-20251001 &
run_stream on  claude-haiku-4-5-20251001 &
run_stream off claude-sonnet-4-6 &
run_stream on  claude-sonnet-4-6 &
wait
echo GRID DONE
