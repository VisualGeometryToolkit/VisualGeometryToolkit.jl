#!/bin/bash
# submit_docs.sh — the six docs builds of the current checkout, the landing gate's docs half: VisualGeometryToolkitCore first
# (one task), then the other five as one array that starts when Core's build has finished, each against this checkout's own
# inventories (docs.sbatch, VGTK_DOCS_LOCAL=1). A failed strict Core build is rebuilt once for its inventory alone, so the
# other five are still checked against today's Core; Core's row keeps the failure. Run from the repository root:
#
#   bash ci/slurm/submit_docs.sh               # prints the two job ids and the log directory
#   bash ci/slurm/submit_docs.sh --summary D   # one line per package of a launch's log directory D: wall time and exit code
#
# The environment: MEM_PER_CPU (MB, default 8480), VGTK_SLURM_PARTITION/ACCOUNT, SBATCH_EXTRA as for submit_suites.sh.
# Exits 0 from --summary when every package's build exited 0.
set -euo pipefail
[ -f ci/slurm/common.sh ] || { echo "run from the repository root"; exit 2; }
if [ "${1:-}" = "--summary" ]; then
  D=${2:?usage: submit_docs.sh --summary <log directory>}; status=0
  echo "| Package | wall | exit |"; echo "|---|---:|---:|"
  source ci/slurm/packages.sh
  for p in "${VGTK_ALL_PACKAGES[@]}"; do
    f=$D/$p.log
    if [ ! -f "$f" ]; then echo "| $p | — | not run |"; status=1; continue; fi
    l=$(grep -E ' docs wall=[0-9]+s exit=' "$f" | tail -n 1 || true)
    if [ -z "$l" ]; then echo "| $p | running? | — |"; status=1; continue; fi
    w=${l#* wall=}; w=${w%% *}; e=${l##*exit=}; echo "| $p | $w | $e |"; [ "$e" = 0 ] || status=1
  done
  for p in "${VGTK_ALL_PACKAGES[@]}"; do
    [ -f "$D/$p.log" ] && grep -h -E '^┌ Error' "$D/$p.log" | sort | uniq -c | sed "s/^/- $p: /" | cut -c1-300
  done
  exit $status
fi
source ci/slurm/packages.sh
COMMIT=$(git rev-parse HEAD)
LOGS=${VGTK_SUITE_LOGS:-ci/slurm/logs}
source ci/slurm/logdir.sh
VGTK_DOCS_LOGDIR=$(new_logdir "$LOGS" "${COMMIT:0:8}-docs"); export VGTK_DOCS_LOGDIR
OPTS=(--parsable --mem-per-cpu="${MEM_PER_CPU:-8480}M" --output="$VGTK_DOCS_LOGDIR/slurm-%A_%a.out")
[ -n "${VGTK_SLURM_PARTITION:-}" ] && OPTS+=(--partition="$VGTK_SLURM_PARTITION")
[ -n "${VGTK_SLURM_ACCOUNT:-}" ] && OPTS+=(--account="$VGTK_SLURM_ACCOUNT")
read -ra EXTRA <<< "${SBATCH_EXTRA:-}"
CORE=${VGTK_ALL_PACKAGES[0]}
REST=("${VGTK_ALL_PACKAGES[@]:1}")
J1=$(VGTK_DOCS_PACKAGES="$CORE" sbatch "${OPTS[@]}" --array=0 ${EXTRA[@]+"${EXTRA[@]}"} ci/slurm/docs.sbatch); J1=${J1%%;*}
J2=$(VGTK_DOCS_PACKAGES="${REST[*]}" sbatch "${OPTS[@]}" --array=0-$(( ${#REST[@]} - 1 )) --dependency=afterany:"$J1" \
       ${EXTRA[@]+"${EXTRA[@]}"} ci/slurm/docs.sbatch); J2=${J2%%;*}
echo "commit   $COMMIT"
echo "logs     $VGTK_DOCS_LOGDIR"
echo "jobs     $J1 ($CORE), $J2 (${REST[*]}, after $J1)"
echo "summary  bash ci/slurm/submit_docs.sh --summary $VGTK_DOCS_LOGDIR"
