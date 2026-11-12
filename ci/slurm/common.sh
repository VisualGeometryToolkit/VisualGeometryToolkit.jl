# common.sh — the body every launcher of this directory shares, sourced from the repository root
# (`source ci/slurm/common.sh`; Slurm copies a batch script into its spool, so suites.sbatch finds this file through
# SLURM_SUBMIT_DIR, the repository root it was submitted from, never through $0).
#
# The environment it reads (all optional):
#   JULIA                  the julia binary (default: julia on PATH)
#   VGTK_SUITE_PACKAGES    the packages of a launch, space-separated, short or full names (default: all six, below)
#   VGTK_SUITE_LOGS        the base of the log directories (default: ci/slurm/logs, git-ignored)
#   RETESTITEMS_NWORKERS   the ReTestItems workers of one package (default: the task's CPUs minus one, at least 1)
#   LO_TIMING_GATE         handed to the suites as set (VisualGeometryToolkitEstimation's LO timing gate: the release machine only)
#   VGTK_BENCHMARK_TESTS   handed to the suites as set (the benchmark test items)
# Every worker runs one Julia thread and one BLAS thread (JULIA_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1): the parallelism is the
# worker pool, sized from the CPUs the task was given.

JULIA=${JULIA:-julia}
command -v "$JULIA" > /dev/null || { echo "no julia binary \"$JULIA\" (set JULIA, or put julia on PATH)"; exit 2; }
[ -f ci/slurm/common.sh ] || { echo "run from the repository root"; exit 2; }

# The packages, in dependency order (packages.sh): the short name a launch may use, and the directory.
source ci/slurm/packages.sh

# package_dir <name>: the package directory of a short or full name (Core, IO, Estimation, Features, Eval, PoseLib)
package_dir() {
  local n=$1
  case $n in
    Core|IO|Estimation|Features|Eval) n=VisualGeometryToolkit$n ;;
  esac
  [ -f "$n/Project.toml" ] && [ -d "$n/test" ] || { echo "unknown package $1 (no $n/Project.toml and $n/test)" >&2; return 2; }
  echo "$n"
}

# selected_packages [names…]: the package directories of a launch, one per line — the names given, else VGTK_SUITE_PACKAGES,
# else all six
selected_packages() {
  local names=("$@") n
  [ ${#names[@]} -gt 0 ] || read -ra names <<< "${VGTK_SUITE_PACKAGES:-${VGTK_ALL_PACKAGES[*]}}"
  for n in "${names[@]}"; do package_dir "$n" || return 2; done
}

# gpu_label: the GPUs this process sees, as the driver names them (index and UUID; under Slurm's device cgroup the index is the
# job-local one, the UUID identifies the card)
gpu_label() {
  command -v nvidia-smi > /dev/null || { echo "none (no nvidia-smi)"; return; }
  [ "${CUDA_VISIBLE_DEVICES-unset}" = "" ] && { echo "none (CUDA_VISIBLE_DEVICES empty)"; return; }
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | paste -sd ';' - || echo "unknown"
}

# sbatch_site_options: the site's sbatch options of every launch, appended to the caller's array OPTS (--partition and
# --account from VGTK_SLURM_PARTITION and VGTK_SLURM_ACCOUNT), and SBATCH_EXTRA's words into the array EXTRA, which the
# caller passes last
sbatch_site_options() {
  [ -n "${VGTK_SLURM_PARTITION:-}" ] && OPTS+=(--partition="$VGTK_SLURM_PARTITION")
  [ -n "${VGTK_SLURM_ACCOUNT:-}" ] && OPTS+=(--account="$VGTK_SLURM_ACCOUNT")
  read -ra EXTRA <<< "${SBATCH_EXTRA:-}"
}

# log_header <pkg> <log> [kind]: the first line of a task's log (truncating it): what ran where; `kind` (docs) after the
# commit. log_trailer <pkg> <log> <t0> <rc> [kind]: its last line, `package=<pkg>[ kind] wall=<s>s exit=<rc>`, which
# summarize.sh and submit_docs.sh --summary read.
log_header() {
  echo "[$(date +%T)] package=$1 commit=$(git rev-parse HEAD)${3:+ $3} host=$(hostname) job=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${SLURM_ARRAY_TASK_ID:-}" > "$2"
}
log_trailer() {
  echo "[$(date +%T)] package=$1${5:+ $5} wall=$(( $(date +%s) - $3 ))s exit=$4" >> "$2"
}

# locked_instantiate <project> <log>: resolve and instantiate <project> into <log>. The workspace has one Manifest at the
# root (untracked): it is resolved first, so a [deps] change since its last resolve (a package that gained a dependency)
# is picked up, then instantiated; the tasks of an array do this one at a time (a lock in the git directory). Returns
# julia's exit code.
locked_instantiate() {
  {
    flock 9
    "$JULIA" --startup-file=no --project="$1" -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
  } 9> "$(git rev-parse --git-dir)/vgtk-suite-instantiate.lock" >> "$2" 2>&1
}

# run_package <dir> <logdir> <cpus>: instantiate <dir>/test, then Pkg.test() of <dir>, into <logdir>/<dir>.log. The first and the
# last lines of the log are this function's: what ran where, and `wall=<s>s exit=<code>`. Returns Pkg.test's exit code.
run_package() {
  local pkg=$1 logdir=$2 cpus=$3 log rc t0
  log=$logdir/$pkg.log
  export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export RETESTITEMS_NWORKERS=${RETESTITEMS_NWORKERS:-$(( cpus > 1 ? cpus - 1 : 1 ))}
  [ -n "${LO_TIMING_GATE:-}" ] && export LO_TIMING_GATE
  [ -n "${VGTK_BENCHMARK_TESTS:-}" ] && export VGTK_BENCHMARK_TESTS
  t0=$(date +%s)
  log_header "$pkg" "$log"
  echo "[$(date +%T)] cpus=$cpus workers=$RETESTITEMS_NWORKERS gpu=$(gpu_label) LO_TIMING_GATE=${LO_TIMING_GATE:-} VGTK_BENCHMARK_TESTS=${VGTK_BENCHMARK_TESTS:-} test_args=${VGTK_SUITE_TEST_ARGS:-}" >> "$log"
  locked_instantiate "$pkg/test" "$log"
  rc=$?
  if [ $rc -eq 0 ]; then
    # VGTK_SUITE_TEST_ARGS (space-separated) reaches the suite as test_args: a ReTestItems runtests.jl reads them as a
    # testitem-name filter, so one package's suite can run as several smaller jobs. Unset: every item.
    VGTK_SUITE_TEST_ARGS="${VGTK_SUITE_TEST_ARGS:-}" "$JULIA" --startup-file=no --project="$pkg" \
      -e 'using Pkg; Pkg.test(; test_args = String.(split(ENV["VGTK_SUITE_TEST_ARGS"])))' >> "$log" 2>&1
    rc=$?
  else
    echo "[$(date +%T)] instantiate of $pkg/test failed (exit $rc): no tests run" >> "$log"
  fi
  log_trailer "$pkg" "$log" "$t0" "$rc"
  return $rc
}
