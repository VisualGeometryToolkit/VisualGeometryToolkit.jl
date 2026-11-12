#!/bin/bash
# docs.sbatch — one package's docs build (its docs/make.jl) as a Slurm task, CPU only, against this checkout's own builds of
# the packages it links to (VGTK_DOCS_LOCAL=1: a make.jl's InterLinks read <Package>/docs/build/objects.inv where present).
# Submit it through submit_docs.sh, which runs Core first and the other five after it; the package is VGTK_DOCS_PACKAGES'
# entry at the array index. No deployment: the CI variables Documenter's deploydocs reads are cleared.
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --job-name=vgtk-docs
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root (submit_docs.sh does)}"
source ci/slurm/common.sh
: "${VGTK_DOCS_LOGDIR:?set by submit_docs.sh}"
read -ra PKGS <<< "${VGTK_DOCS_PACKAGES:?set by submit_docs.sh}"
PKG=${PKGS[${SLURM_ARRAY_TASK_ID:-0}]}
log=$VGTK_DOCS_LOGDIR/$PKG.log
unset CI GITHUB_ACTIONS GITHUB_TOKEN DOCUMENTER_KEY TRAVIS GITLAB_CI
export VGTK_DOCS_LOCAL=1 JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
t0=$(date +%s)
log_header "$PKG" "$log" docs
# The workspace Manifest is the root's (docs projects are workspace members): resolved once at a time, like the suites.
locked_instantiate . "$log"
"$JULIA" --startup-file=no --project="$PKG/docs" "$PKG/docs/make.jl" >> "$log" 2>&1
rc=$?
# Core's inventory is what the other five link against: a failed strict Core build is rebuilt once with every check a warning
# (VGTK_DOCS_INVENTORY_ONLY=1), so they are still checked against today's Core names; Core's own result stays the failure.
if [ $rc -ne 0 ] && [ "$PKG" = "${VGTK_ALL_PACKAGES[0]}" ]; then
  echo "[$(date +%T)] strict build failed (exit $rc): rebuilding for the inventory only" >> "$log"
  VGTK_DOCS_INVENTORY_ONLY=1 "$JULIA" --startup-file=no --project="$PKG/docs" "$PKG/docs/make.jl" > "$log.inventory" 2>&1
  echo "[$(date +%T)] inventory-only rebuild exit=$? (log $log.inventory)" >> "$log"
fi
log_trailer "$PKG" "$log" "$t0" "$rc" docs
exit $rc
