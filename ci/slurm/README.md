# ci/slurm — the package test suites on Slurm

The six packages' full test suites as one Slurm array: one task per package, each on one GPU and its share of a node's CPUs.
The same body runs without Slurm (`--local`), so a machine without a scheduler runs the identical commands one package after
another.

| File | What it does |
|---|---|
| `submit_suites.sh [--local] [pkgs…]` | submits the array (or a subset of packages) and prints the job id and the log directory; `--local` runs the body sequentially here |
| `suites.sbatch` | one array task: package `SLURM_ARRAY_TASK_ID` of the launch — instantiate `<Pkg>/test`, then `Pkg.test()` of `<Pkg>` |
| `common.sh` | the body both share: package names, the worker count, the instantiate lock, the log header and footer |
| `summarize.sh <job id \| log dir>` | one table of a launch: per package the counts, wall time, exit code and GPU, then every failure and error |

## Running

From the repository root, on the commit to test (the launch tests the working tree as it is; the first line it prints says how
many tracked files are modified):

```bash
bash ci/slurm/submit_suites.sh                     # all six; prints e.g. "job 6931" and the log directory
bash ci/slurm/submit_suites.sh Core Estimation     # a subset: Core, IO, Estimation, Features, Eval, PoseLib or the full names
squeue -j <job id>                                 # the tasks
bash ci/slurm/summarize.sh <job id>                # the table, once the tasks have ended (a running task shows "running?")

bash ci/slurm/submit_suites.sh --local             # no Slurm: one package after another on this machine
CUDA_VISIBLE_DEVICES=0 bash ci/slurm/submit_suites.sh --local Core
```

Logs go to `ci/slurm/logs/<date>-<time>-<commit>/` (git-ignored; `VGTK_SUITE_LOGS` for another base): `<Package>.log` per package
(the instantiate and `Pkg.test()` output, between a header naming the commit, host, job, CPUs, workers and GPU, and a footer
`wall=<s>s exit=<code>`), `slurm-<job>_<task>.out` per task, and `JOB_ID`. `ci/slurm/logs/job-<id>` links to the directory, so
`summarize.sh` takes either.

## Sizing and the variables

Each task gets `--gres=gpu:1` and `CPUS_PER_TASK` CPUs; the suite runs `RETESTITEMS_NWORKERS` = CPUs − 1 workers, each with one
Julia thread and one BLAS thread (`JULIA_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`): the parallelism is the worker pool.

| Variable | Default | Meaning |
|---|---|---|
| `CPUS_PER_TASK` | the partition's CPUs per node ÷ the number of packages (at least 4); `--local`: `nproc` | CPUs of one task |
| `MEM_PER_CPU` | the node's memory ÷ its CPUs, in MB | `--mem-per-cpu` |
| `TIME_LIMIT` | `03:00:00` | `--time` of one task |
| `VGTK_SLURM_PARTITION` | none (the site's default) | `--partition`; Slurm's own `SBATCH_PARTITION` works too |
| `VGTK_SLURM_ACCOUNT` | none | `--account`; or `SBATCH_ACCOUNT` |
| `SBATCH_EXTRA` | none | further sbatch options, appended last |
| `RETESTITEMS_NWORKERS` | CPUs − 1 | the ReTestItems workers of one package |
| `LO_TIMING_GATE` | unset | handed to the suites (VisualGeometryToolkitEstimation's LO timing gate; the release machine only) |
| `VGTK_BENCHMARK_TESTS` | unset | handed to the suites (the benchmark test items) |
| `JULIA` | `julia` on `PATH` | the Julia binary |
| `VGTK_SUITE_PACKAGES` | all six | the packages, when none are named on the command line |

The jobs inherit the submitting shell's environment (sbatch's default `--export=ALL`), so a variable set for the submission
reaches every task.

## Notes

- **One Manifest.** The workspace resolves into one root `Manifest.toml`, so the tasks of a launch instantiate one at a time
  (an `flock` on a file in the git directory); the suites themselves run in parallel.
- **The GPU a task saw.** Under Slurm's device cgroup a task sees its card as index 0; the log header records the card's UUID as
  well (`nvidia-smi --query-gpu=index,uuid`), which tells the cards apart.
- **Known failures before the 0.9 registration.** Each package's README fresh-environment test resolves the registered stack and
  fails until the release it names is registered; `summarize.sh` lists it like any other failure.
- **Without a GPU.** `--local` with `CUDA_VISIBLE_DEVICES=` runs the CPU items; GPU items record a skip without a device.
