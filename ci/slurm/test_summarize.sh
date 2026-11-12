#!/bin/bash
# test_summarize.sh — summarize.sh reads the LAST top-level Test Summary of a package log: the fixture's PoseLib log holds
# its README-example testset's summary (5) before the package total (312 pass, 1 broken, 313). Run from the repository root.
set -euo pipefail
out=$(bash ci/slurm/summarize.sh ci/slurm/fixtures/summarize-two-blocks)
row=$(grep '^| PoseLib |' <<< "$out")
want='| PoseLib | 312 | 0 | 0 | 1 | 313 | 100s | 0 | none |'
[ "$row" = "$want" ] || { echo "FAIL: got  $row"; echo "      want $want"; exit 1; }
echo "ok: summarize.sh reports the package total ($row)"
