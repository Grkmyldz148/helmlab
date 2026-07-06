#!/bin/bash
# Sync skills/ to the public install mirror (Grkmyldz148/color-skills).
# Run after editing any SKILL.md. (CI auto-sync would need a PAT secret —
# GITHUB_TOKEN can't push to a different repo.)
set -e
cd "$(dirname "$0")"
TMP=$(mktemp -d)
git clone -q git@github.com:Grkmyldz148/color-skills.git "$TMP" 2>/dev/null || git clone -q https://github.com/Grkmyldz148/color-skills.git "$TMP"
cp color-space-routing/SKILL.md "$TMP/color-space-routing/"
cp helmlab/SKILL.md "$TMP/helmlab/"
cd "$TMP" && git add -A
git diff --cached --quiet && { echo "already in sync"; exit 0; }
git commit -q -m "sync from helmlab monorepo skills/" && git push -q
echo "synced"
