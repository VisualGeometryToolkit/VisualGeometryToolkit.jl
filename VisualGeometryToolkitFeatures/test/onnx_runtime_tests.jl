# ONNX Runtime sessions (src/onnx_runtime.jl): the execution provider's resolution (:auto falls back to the CPU with a
# warning naming why; :cuda refuses by name), the thread pool reaching the session, and, on a CUDA device, LoMa on CUDA
# against LoMa on the CPU. The first two run everywhere on ONNXRunTime's own tiny test model, no LoMa download.

@testitem "execution_provider: :auto falls back to the CPU with a warning naming why, :cuda refuses by name" begin
    using VisualGeometryToolkitFeatures: OnnxCudaStatus, onnx_cuda_status
    import VisualGeometryToolkitFeatures as VgtkFeatures
    resolve = VgtkFeatures.resolve_execution_provider
    # a machine without the GPU path, simulated: the status a probe there returns
    nogpu() = OnnxCudaStatus(false, "no CUDA device is visible (CUDA_VISIBLE_DEVICES=)", nothing, nothing, nothing)
    @test (@test_logs (:warn, r"running on the CPU: no CUDA device is visible") resolve(:auto, nogpu)) === :cpu
    err = try resolve(:cuda, nogpu); nothing catch e; e end
    @test err isa ErrorException && occursin("execution_provider = :cuda", err.msg) && occursin("no CUDA device is visible", err.msg)
    @test (@test_logs resolve(:cpu, () -> error("the CPU never probes CUDA"))) === :cpu
    @test_throws ArgumentError resolve(:tensorrt, nogpu)
    # with the GPU path, :auto and :cuda take it, silently
    gpu() = OnnxCudaStatus(true, "", v"13.0", v"12.9", v"9.28.0")
    @test (@test_logs resolve(:auto, gpu)) === :cuda
    @test (@test_logs resolve(:cuda, gpu)) === :cuda
    # and the bf16 exports refuse a provider that resolved to the CPU, before any download
    @test_throws ArgumentError VgtkFeatures.LoMaMatcher(:b128; bf16 = true, execution_provider = :cpu)
    # an already-loaded library of the same soname serves when at least as new as ours (its version from the file name)
    @test VgtkFeatures._library_version("/a/lib/libcurand.so.10.4.2.55") > VgtkFeatures._library_version("/b/libcurand.so.10.3.10.19")
    @test VgtkFeatures._library_version("/b/libcurand.so.10.3.10.19") == v"10.3.10"
    @test VgtkFeatures._library_version("/c/libcudart.so") == v"0"
    # this machine's own probe answers, available or with a reason
    s = onnx_cuda_status()
    @test s.available ? isempty(s.reason) && s.cuda_runtime.major == 12 : !isempty(s.reason)
end

@testitem "execution_provider: the thread pool is the caller's (Threads.nthreads() by default) and is recorded" begin
    import VisualGeometryToolkitFeatures as VgtkFeatures
    model = VgtkFeatures.ORT.testdatapath("matmul.onnx")
    if !Sys.islinux()
        @test_skip "threads are counted in /proc/self/task (Linux)"
    else
        tasks() = length(readdir("/proc/self/task"))
        VgtkFeatures._onnx_session(model, :cpu, 1)               # ONNX Runtime's process-wide environment, created once
        # an intra-op pool of n runs on the caller and n - 1 threads of its own; 1 adds none; no inter-op pool
        for n in (1, 3, 5)
            before = tasks()
            s = VgtkFeatures._onnx_session(model, :cpu, n)
            @test tasks() - before == n - 1
            @test s.execution_provider === :cpu
            out = s(Dict("input1" => ones(Float32, 2, 3), "input2" => ones(Float32, 3, 4)))
            @test only(values(out)) == fill(3.0f0, 2, 4)
        end
        @test_throws ArgumentError VgtkFeatures._onnx_session(model, :cpu, 0)
        # ONNX Runtime's own default sizes the pool from the machine: what the explicit size replaces
        before = tasks(); VgtkFeatures.ORT.load_inference(model)
        @test tasks() - before > Threads.nthreads() - 1 || Sys.CPU_THREADS <= Threads.nthreads()
    end
    p = VgtkFeatures._onnx_provenance(:cpu, 3)
    @test p.execution_provider === :cpu && p.threads == 3 && startswith(p.onnxruntime, "1.")
end

# ONNX Runtime's CUDA build: conda-forge's files pinned by sha256 and their merged tree by Artifacts.toml. Nothing here
# downloads or runs it (the probe in the first item and the device item below do, where a device is visible).
@testitem "ONNX Runtime's CUDA build: conda-forge's files pinned, merged into one tree, checked" begin
    using Base.BinaryPlatforms: Platform
    using Artifacts: artifact_hash, load_artifacts_toml
    using VisualGeometryToolkitCore: sha256_hex
    import VisualGeometryToolkitFeatures as VgtkFeatures
    toml = VgtkFeatures._ARTIFACTS_TOML
    entry = only(load_artifacts_toml(toml)["onnxruntime_cuda"])
    @test entry["lazy"] && (entry["arch"], entry["os"], entry["libc"]) == ("x86_64", "linux", "glibc")
    @test !haskey(entry, "download")                         # installed by the package, not by Pkg
    @test artifact_hash("onnxruntime_cuda", toml; platform = Platform("x86_64", "linux")) !== nothing
    @test artifact_hash("onnxruntime_cuda", toml; platform = Platform("aarch64", "macos")) === nothing
    files = VgtkFeatures._ORT_CUDA_FILES
    @test startswith(first(first(files)), "onnxruntime-cpp-1.31.1-") && endswith(first(first(files)), "_cuda129")
    @test all(f -> occursin(r"^[0-9a-f]{64}$", f[2]), files) && allunique(first.(files))
    # a download whose sha256 differs is refused and leaves nothing behind (a local file:// source)
    mktempdir() do dir
        src = joinpath(dir, "src.bin"); write(src, "not the pinned bytes")
        dst = joinpath(dir, "dst.bin")
        @test_throws r"sha256 .* expected 0{64}" VgtkFeatures._download_verified("file://$src", "0"^64, dst)
        @test !ispath(dst)
        digest = open(sha256_hex, src)
        @test VgtkFeatures._download_verified("file://$src", digest, dst) == dst && read(dst, String) == "not the pinned bytes"
    end
    # packages merge directory by directory; two holding one file is an error
    mktempdir() do dir
        a, b, out = mkpath(joinpath(dir, "a", "lib")), mkpath(joinpath(dir, "b", "lib")), mkpath(joinpath(dir, "out"))
        write(joinpath(a, "liba.so.1"), "a"); symlink("liba.so.1", joinpath(a, "liba.so"))
        write(joinpath(b, "libb.so.1"), "b"); mkpath(joinpath(dir, "b", "include"))
        VgtkFeatures._merge_tree!(joinpath(dir, "a"), out)
        VgtkFeatures._merge_tree!(joinpath(dir, "b"), out)
        @test sort(readdir(joinpath(out, "lib"))) == ["liba.so", "liba.so.1", "libb.so.1"]
        @test islink(joinpath(out, "lib", "liba.so")) && isdir(joinpath(out, "include"))
        c = mkpath(joinpath(dir, "c", "lib")); write(joinpath(c, "libb.so.1"), "c")
        @test_throws r"two conda packages hold" VgtkFeatures._merge_tree!(joinpath(dir, "c"), out)
    end
end

# LoMa on CUDA against LoMa on the CPU, on a device (skipped without one: CI). The pair: a textured 784² frame (the graph's
# input, no resampling) and the same scene shifted by 9 pixels, ASIFT-detected; the same detections on both providers.
@testitem "LoMa on CUDA describes and matches as on the CPU (skipped without a device)" begin
    using Random: Xoshiro
    import VisualGeometryToolkitFeatures as VgtkFeatures
    using VisualGeometryToolkitFeatures: LoMaDescriptor, LoMaMatcher, loma_provenance, onnx_cuda_status
    status = onnx_cuda_status()
    if !status.available
        @test_skip "ONNX Runtime's CUDA provider: $(status.reason)"
    else
        rng = Xoshiro(3)
        base = rand(rng, Float32, 820, 820)
        for _ in 1:3
            base .= (base .+ circshift(base, (1, 0)) .+ circshift(base, (0, 1))) ./ 3
        end
        a = base[1:784, 1:784]; b = base[10:793, 1:784]
        img(x) = permutedims(repeat(x, 1, 1, 3), (3, 1, 2))
        dets(x) = VgtkFeatures.asift_detect(x; tilts = [(1.0, 0.0)])
        da, db = dets(a), dets(b)
        sz = (784, 784)
        function run(; kw...)
            d = LoMaDescriptor(:b; max_detections = 1024, kw...)
            m = LoMaMatcher(:b128; kw...)
            fa = d(img(a), da); fb = d(img(b), db)
            return fa, fb, m(fa, fb, sz, sz; min_score = 0.0), loma_provenance(d), loma_provenance(m)
        end
        fa_c, fb_c, mc, pdc, pmc = run(execution_provider = :cpu, threads = 4)
        fa_g, fb_g, mg, pdg, pmg = (@test_logs run())             # the default, :auto, takes the device, without a warning
        @test pdc.execution_provider === pmc.execution_provider === :cpu && pdc.threads == pmc.threads == 4
        @test pdg.execution_provider === pmg.execution_provider === :cuda
        @test pdg.cuda_runtime == string(status.cuda_runtime) && pdg.threads == Threads.nthreads()
        @test pdg.onnxruntime == "1.31.1" && occursin("onnxruntime-cpp-1.31.1-", pdg.onnxruntime_build)
        @test pdg.model == "loma_descriptor_dedode_b.onnx" && pmg.model == "loma_matcher_B128.onnx"
        # Descriptors: the same up to the providers' float differences (cuDNN's convolutions and cuBLAS's products sum in
        # another order than MLAS). Measured 2026-11-10 on the RTX 4090: max |ΔD| / max |D| = 2.1e-3 here and
        # 3.2e-3 on two real panorama frames (0.022 of 6.87); the bound 1e-2 is three times the larger.
        D(f) = reduce(hcat, f.descriptor)
        for (c, g) in ((fa_c, fa_g), (fb_c, fb_g))
            @test maximum(abs, D(c) .- D(g)) < 1.0e-2 * maximum(abs, D(c))
        end
        # Matches: the same (i, j) pairs, but a pair whose probability sits at the network's threshold can flip on a float
        # difference. Measured: 1015 of 1015 identical here; on six real panorama frames 1103 of the 1109 pairs either provider
        # matched over their 15 frame pairs (99.5 %; the worst frame pair with more than 20 matches 99.2 %). Bound 97 %.
        pc = Set((x.first, x.second) for x in mc); pg = Set((x.first, x.second) for x in mg)
        @test length(pc) > 100
        @test length(intersect(pc, pg)) >= 0.97 * length(union(pc, pg))
    end
end

@testitem "LoMa on CUDA after ASIFT on CUDA in one process: the detector's plans are released first (skipped without a device)" begin
    using Random: Xoshiro
    using CUDA: CUDA
    import VisualGeometryToolkitFeatures as VgtkFeatures
    using VisualGeometryToolkitFeatures: LoMaDescriptor, LoMaMatcher, onnx_cuda_status, asift_detect
    status = onnx_cuda_status()
    if !(status.available && CUDA.functional())
        @test_skip "ONNX Runtime's CUDA provider or CUDA.jl unavailable"
    else
        rng = Xoshiro(5)
        H, W = 1424, 2128                                  # a panorama frame's green cells: the plan takes most of the device
        base = rand(rng, Float32, H, W)
        for _ in 1:3
            base .= (base .+ circshift(base, (1, 0)) .+ circshift(base, (0, 1))) ./ 3
        end
        d = LoMaDescriptor(:b; execution_provider = :cuda); m = LoMaMatcher(:b128; execution_provider = :cuda)
        img = permutedims(repeat(base, 1, 1, 3), (3, 1, 2))
        feats = map(1:2) do _                              # detect, describe, detect again: the loop of a registration
            dets = asift_detect(base; device = CUDA.CUDABackend())
            @test !isempty(VgtkFeatures._ASIFT_PLAN_CACHE)
            f = d(img, dets)
            @test isempty(VgtkFeatures._ASIFT_PLAN_CACHE)   # released before ONNX Runtime ran
            f
        end
        @test length(m(feats[1], feats[2], (H, W), (H, W))) > 100
    end
end
