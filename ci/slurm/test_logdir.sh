#!/bin/bash
# test_logdir.sh — new_logdir gives two launches of the same commit in the same second two directories. Run from the
# repository root.
set -euo pipefail
source ci/slurm/logdir.sh
base=$(mktemp -d); trap 'rm -rf "$base"' EXIT
a=$(new_logdir "$base" abcdef12); b=$(new_logdir "$base" abcdef12)
[ -d "$a" ] && [ -d "$b" ] || { echo "FAIL: not created ($a, $b)"; exit 1; }
[ "$a" != "$b" ] || { echo "FAIL: one directory for two launches: $a"; exit 1; }
case $(basename "$a") in [0-9]*-abcdef12-??????) ;; *) echo "FAIL: name $(basename "$a")"; exit 1 ;; esac
echo "ok: two launches, two log directories ($(basename "$a"), $(basename "$b"))"
