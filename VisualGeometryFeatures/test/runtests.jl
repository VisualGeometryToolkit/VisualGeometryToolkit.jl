using ReTestItems, VisualGeometryFeatures

# Test items live in test/*_tests.jl; shared fixtures in test/vgf_testsetup.jl.
# Most items are small and single-threaded, so the suite fans out over many
# one-thread workers sized from the machine: on 32 threads, 16 workers took
# 42 s against 71 s serially and 44 s with 8. (BlobBoards runs CPU_THREADS ÷ 4
# workers × 4 threads because its heavy items multithread.)
const NWORKERS = clamp(Sys.CPU_THREADS ÷ 2, 1, 16)

runtests(VisualGeometryFeatures; nworkers = NWORKERS, nworker_threads = 1, testitem_timeout = 1800)
