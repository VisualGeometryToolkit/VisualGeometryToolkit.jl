#!/bin/bash
# suites.sbatch — the package test suites as one Slurm array: task i runs package i of VGTK_SUITE_PACKAGES (instantiate
# <Pkg>/test, then Pkg.test() of <Pkg>) on one GPU and the task's CPUs. Submit it through submit_suites.sh, which sizes the array,
# the CPUs and the memory from the node and fixes the log directory; `submit_suites.sh --local` runs the same body without Slurm.
#
# The #SBATCH lines below are defaults only: submit_suites.sh overrides --cpus-per-task, --mem-per-cpu, --time, --array,
# --output and, when set, --partition and --account on its command line.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --job-name=vgtk-suites
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root (submit_suites.sh does)}"
source ci/slurm/common.sh

: "${VGTK_SUITE_LOGDIR:?set by submit_suites.sh: the log directory of the launch}"
mapfile -t PKGS < <(selected_packages) || exit 2
i=${SLURM_ARRAY_TASK_ID:?an array task: submit_suites.sh, or sbatch --array=0-<n-1>}
[ "$i" -lt "${#PKGS[@]}" ] || { echo "task $i: the launch has ${#PKGS[@]} package(s): ${PKGS[*]}"; exit 2; }
PKG=${PKGS[$i]}
echo "[$(date +%T)] task $i of ${#PKGS[@]}: $PKG  log $VGTK_SUITE_LOGDIR/$PKG.log  gpu=$(gpu_label)"
run_package "$PKG" "$VGTK_SUITE_LOGDIR" "${SLURM_CPUS_PER_TASK:-8}"
rc=$?
echo "[$(date +%T)] $PKG exit=$rc"
exit $rc
