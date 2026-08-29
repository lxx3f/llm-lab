#!/usr/bin/env bash
# scripts/audit/run_gitignore_coverage.sh
#
# Per-file .gitignore coverage audit for final-audit.
#
# Iterates every file under the sensitive paths and runs
# `git check-ignore --no-index` to confirm coverage.
#
# Reproduction:
#   1. git checkout HEAD~1   # obtain audited tree
#   2. bash scripts/audit/run_gitignore_coverage.sh
#   3. Expected: "Total: N, Ignored: N, Not ignored: 0"
#
# Output: stdout only. Exit 0 on PASS, exit 1 if any file is not ignored.

set -e

total=0
ignored=0
not_ignored=0
not_ignored_list=""

# Collect sensitive paths
sensitive_paths="datasets/tool-calling-d1 datasets/tool-calling-d1-llm datasets/tool-calling-d2 artifacts/ .tmp/"

while IFS= read -r -d '' f; do
  total=$((total + 1))
  if git check-ignore --no-index "$f" >/dev/null 2>&1; then
    ignored=$((ignored + 1))
  else
    not_ignored=$((not_ignored + 1))
    not_ignored_list="${not_ignored_list}${f}\n"
  fi
done < <(find $sensitive_paths -type f -print0 2>/dev/null)

echo "=== Per-file .gitignore audit ==="
echo "Total: $total"
echo "Ignored: $ignored"
echo "Not ignored: $not_ignored"

if [ "$total" -eq 0 ]; then
  echo "ERROR: no sensitive files found (something is wrong)" >&2
  exit 2
fi

if [ "$not_ignored" -ne 0 ]; then
  echo ""
  echo "Files NOT gitignored:"
  printf "$not_ignored_list"
  exit 1
fi

if [ "$ignored" -ne "$total" ]; then
  echo "ERROR: Total ($total) != Ignored ($ignored)" >&2
  exit 3
fi

echo ""
echo "PASS: $ignored/$total (100%, 0 exceptions)"
exit 0