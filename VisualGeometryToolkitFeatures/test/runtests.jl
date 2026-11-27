using ReTestItems, VisualGeometryToolkitFeatures
import VisualGeometryToolkitFeatures as VgtkFeatures

# Test items live in test/*_tests.jl; shared fixtures in test/vgf_testsetup.jl.
# Most items are small and single-threaded, so the suite fans out over many
# one-thread workers sized from the machine: on 32 threads, 16 workers took
# 42 s against 71 s serially and 44 s with 8. (BlobBoards runs CPU_THREADS ÷ 4
# workers × 4 threads because its heavy items multithread.)
const NWORKERS = clamp(Sys.CPU_THREADS ÷ 2, 1, 16)

# ONNX Runtime's CUDA path installs its CUDA 12 runtime, its cuDNN (CUDA_Runtime_jll's and CUDNN_jll's artifacts) and its
# CUDA build on first use, about 2.4 GB. Four items probe it (`onnx_cuda_status`, onnx_runtime_tests.jl); on a fresh depot
# their workers raced the same downloads past the item timeout. Where a CUDA 12 driver is, the three are installed here,
# once, before the items fan out (`_package_cuda_source`, the probe's own installs: no library loaded, no device context
# made); the items' probes then find them. Without a driver nothing is installed, as the probe installs nothing there.
let driver = VgtkFeatures._cuda_driver()
    driver isa VersionNumber && driver >= v"12" && VgtkFeatures._package_cuda_source()
end

runtests(VisualGeometryToolkitFeatures; nworkers = NWORKERS, nworker_threads = 1, testitem_timeout = 1800)
