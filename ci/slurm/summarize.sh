#!/bin/bash
# summarize.sh <job id | log directory> — one table of a launch of submit_suites.sh: per package the counts of the top-level
# Test Summary (ok, fail, error, broken, total), the wall time and the exit code of Pkg.test, then every failing or erroring test
# with its location and its first informative line, and any test item that timed out. Reads <logdir>/<Package>.log as
# common.sh's run_package writes it; a job id is resolved through ci/slurm/logs/job-<id> (VGTK_SUITE_LOGS for another base).
# Exits 0 when every package's Pkg.test exited 0, else 1 (a package still running, or without a Test Summary, counts as 1).
set -uo pipefail
[ $# -eq 1 ] || { echo "usage: summarize.sh <job id | log directory>"; exit 2; }
if [ -d "$1" ]; then D=$1; else D=${VGTK_SUITE_LOGS:-ci/slurm/logs}/job-$1; fi
[ -d "$D" ] || { echo "no log directory $D"; exit 2; }
source "$(dirname "$0")/packages.sh"            # VGTK_ALL_PACKAGES, the table's order
ORDER=("${VGTK_ALL_PACKAGES[@]}")
LOGS=()
for p in "${ORDER[@]}"; do [ -f "$D/$p.log" ] && LOGS+=("$D/$p.log"); done
for f in "$D"/*.log; do [[ " ${ORDER[*]} " == *" $(basename "$f" .log) "* ]] || LOGS+=("$f"); done
[ ${#LOGS[@]} -gt 0 ] || { echo "no <Package>.log in $D"; exit 2; }

head -n 1 "${LOGS[0]}" | sed -n 's/.*commit=\([0-9a-f]*\).*/commit: \1/p'
echo "logs:   $D"
echo
echo "| Package | ok | fail | error | broken | total | wall | exit | gpu |"
echo "|---|---:|---:|---:|---:|---:|---:|---:|---|"
FAILS=$(mktemp); trap 'rm -f "$FAILS"' EXIT
status=0
for f in "${LOGS[@]}"; do
  sed -E 's/\x1b\[[0-9;]*m//g' "$f" | awk -v pkg="$(basename "$f" .log)" -v fails="$FAILS" '
    # The header of the top-level Test Summary names its columns; the row below it holds one number per named column.
    /^Test Summary:/ && !have { split($0, h, "|"); nh = split(h[2], hdr, " "); getline row; split(row, r, "|"); split(r[2], val, " ");
                                for (k = 1; k <= nh; k++) col[substr(hdr[k], 1, 2)] = val[k]; have = 1 }
    / wall=[0-9]+s exit=/       { w = $0; sub(/.* wall=/, "", w); sub(/s exit=.*/, "", w); e = $0; sub(/.* exit=/, "", e) }
    /gpu=/ && !gpu              { g = $0; sub(/.* gpu=/, "", g); sub(/ LO_TIMING_GATE=.*/, "", g); gpu = g }
    /Error in testset "/        { name = $0; sub(/^[^"]*"/, "", name); sub(/" on worker.*/, "", name) }
    /(Test Failed|Error During Test) at / {
        loc = $0; sub(/.*(Test Failed|Error During Test) at /, "", loc)
        n = $0; if (n ~ /: (Test Failed|Error During Test) at /) { sub(/: (Test Failed|Error During Test) at .*/, "", n); sub(/^ +/, "", n) } else n = name
        kind = ($0 ~ /Test Failed/) ? "FAIL" : "ERROR"; info = ""
        for (k = 0; k < 8 && info == ""; k++) { if ((getline l) <= 0) break
            if (l ~ /^ *(Test threw exception|Got exception outside of a @test|Stacktrace:|\[[0-9]+\]|@ )/ || l ~ /^ *$/) continue
            if (kind == "FAIL" && l ~ /^ *Expression:/) { getline l2; if (l2 ~ /Evaluated:/) l = l "  " l2 }
            if (kind == "ERROR" && l ~ /^ *Expression:/) continue
            info = l }
        gsub(/^ +/, "", info); key = kind loc info
        if (!(key in seen)) { seen[key] = 1; printf "- %s %s: %s — `%s`  (%s)\n", pkg, kind, n, substr(info, 1, 300), loc >> fails } }
    /timed out running test item|Timed out after/ { t = $0; sub(/^ +/, "", t); if (!(t in tseen)) { tseen[t] = 1; printf "- %s TIMEOUT: %s\n", pkg, substr(t, 1, 300) >> fails } }
    END {
        if (!have) { printf "| %s | — | — | — | — | — | %s | %s | %s |\n", pkg, (w == "" ? "running?" : w "s"), (e == "" ? "—" : e), gpu
                     printf "- %s: no Test Summary in the log (crashed, timed out, or still running)\n", pkg >> fails; exit 1 }
        printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n", pkg, col["Pa"] + 0, col["Fa"] + 0, col["Er"] + 0, col["Br"] + 0,
               col["To"] + 0, (w == "" ? "—" : w "s"), (e == "" ? "—" : e), gpu
        exit (e == "0" ? 0 : 1) }' || status=1
done
echo
if [ -s "$FAILS" ]; then echo "Failures and errors:"; cat "$FAILS"; else echo "No failures or errors."; fi
exit $status
