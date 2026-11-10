#!/bin/bash
# submit_suites.sh — the package test suites of the current checkout, one Slurm array task per package (suites.sbatch), or with
# --local the same body one package after another without Slurm. Run from the repository root:
#
#   bash ci/slurm/submit_suites.sh                       # all six packages; prints the job id and the log directory
#   bash ci/slurm/submit_suites.sh Core Estimation       # a subset (short or full names)
#   bash ci/slurm/submit_suites.sh --local [pkgs…]       # no Slurm: sequential, on this machine's CPUs and visible GPUs
#   bash ci/slurm/summarize.sh <job id | log directory>  # the table of a launch
#
# Sizing (the environment; every value overridable):
#   CPUS_PER_TASK   default: the partition's CPUs per node divided by the number of packages (at least 4); --local: nproc
#   MEM_PER_CPU     default: the node's memory per CPU in MB (floor), so the tasks of a full node share its memory evenly
#   TIME_LIMIT      default 03:00:00 per task
#   VGTK_SLURM_PARTITION, VGTK_SLURM_ACCOUNT   --partition / --account (default: none, the site's defaults; Slurm's own
#                   SBATCH_PARTITION and SBATCH_ACCOUNT work as well)
#   SBATCH_EXTRA    further sbatch options, appended last
# The suites' own variables (RETESTITEMS_NWORKERS, LO_TIMING_GATE, VGTK_BENCHMARK_TESTS, JULIA) are read by common.sh; the jobs
# inherit this shell's environment (sbatch's default --export=ALL).
set -euo pipefail
[ -f ci/slurm/common.sh ] || { echo "run from the repository root"; exit 2; }
LOCAL=0; NAMES=()
for a in "$@"; do case $a in --local) LOCAL=1 ;; -h|--help) sed -n '2,22p' "$0"; exit 0 ;; *) NAMES+=("$a") ;; esac; done
source ci/slurm/common.sh
mapfile -t PKGS < <(selected_packages ${NAMES[@]+"${NAMES[@]}"})
[ ${#PKGS[@]} -gt 0 ] || { echo "no package selected"; exit 2; }
export VGTK_SUITE_PACKAGES="${PKGS[*]}"

COMMIT=$(git rev-parse HEAD)
LOGS=${VGTK_SUITE_LOGS:-ci/slurm/logs}
export VGTK_SUITE_LOGDIR=$LOGS/$(date +%Y%m%d-%H%M%S)-${COMMIT:0:8}
mkdir -p "$VGTK_SUITE_LOGDIR"
VGTK_SUITE_LOGDIR=$(cd "$VGTK_SUITE_LOGDIR" && pwd)
echo "commit   $COMMIT ($(git status --porcelain --untracked-files=no | wc -l) modified tracked files)"
echo "packages ${PKGS[*]}"
echo "logs     $VGTK_SUITE_LOGDIR"

if [ $LOCAL = 1 ]; then
  CPUS=${CPUS_PER_TASK:-$(nproc)}
  echo "local    sequential, $CPUS CPUs per package, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset: every GPU>}"
  echo "local" > "$VGTK_SUITE_LOGDIR/JOB_ID"
  status=0
  for p in "${PKGS[@]}"; do
    echo "[$(date +%T)] $p"
    run_package "$p" "$VGTK_SUITE_LOGDIR" "$CPUS" || status=1
    tail -n 1 "$VGTK_SUITE_LOGDIR/$p.log"
  done
  bash ci/slurm/summarize.sh "$VGTK_SUITE_LOGDIR" || true
  exit $status
fi

command -v sbatch > /dev/null || { echo "no sbatch on PATH: run with --local"; exit 2; }
SINFO_P=(); [ -n "${VGTK_SLURM_PARTITION:-}" ] && SINFO_P=(-p "$VGTK_SLURM_PARTITION")
# the partition's largest node: CPUs and memory (MB)
read -r NODE_CPUS NODE_MEM < <(sinfo -h ${SINFO_P[@]+"${SINFO_P[@]}"} -o '%c %m' | tr -d '+' | sort -n | tail -n 1)
N=${#PKGS[@]}
CPUS=${CPUS_PER_TASK:-$(( NODE_CPUS / N > 4 ? NODE_CPUS / N : 4 ))}
MEM=${MEM_PER_CPU:-$(( NODE_MEM / NODE_CPUS ))}
OPTS=(--parsable --array=0-$(( N - 1 )) --cpus-per-task="$CPUS" --mem-per-cpu="${MEM}M" --time="${TIME_LIMIT:-03:00:00}"
      --output="$VGTK_SUITE_LOGDIR/slurm-%A_%a.out")
[ -n "${VGTK_SLURM_PARTITION:-}" ] && OPTS+=(--partition="$VGTK_SLURM_PARTITION")
[ -n "${VGTK_SLURM_ACCOUNT:-}" ] && OPTS+=(--account="$VGTK_SLURM_ACCOUNT")
read -ra EXTRA <<< "${SBATCH_EXTRA:-}"
JOB=$(sbatch "${OPTS[@]}" ${EXTRA[@]+"${EXTRA[@]}"} ci/slurm/suites.sbatch)
JOB=${JOB%%;*}
echo "$JOB" > "$VGTK_SUITE_LOGDIR/JOB_ID"
ln -sfn "$VGTK_SUITE_LOGDIR" "$LOGS/job-$JOB"
echo "sizing   $N task(s) × $CPUS CPUs, ${MEM} MB per CPU, 1 GPU each (node: $NODE_CPUS CPUs, $NODE_MEM MB)"
for i in "${!PKGS[@]}"; do echo "    ${JOB}_$i  ${PKGS[$i]}"; done
echo "job      $JOB"
echo "summary  bash ci/slurm/summarize.sh $JOB"
