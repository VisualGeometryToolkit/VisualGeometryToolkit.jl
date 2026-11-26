# =============================================================================
# The estimation interface (base/solve_interface.jl): the kind law, the declared defaults, the traits, the front
# end (solve, init/solve!/reinit!), the closed forms' records and the device LM in a kernel
# =============================================================================

@testitem "solve interface: the kind law refuses every mismatched pair before any method runs" begin
    using Test, StaticArrays, Rotations
    using VisualGeometryToolkitCore
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: check_kind, thrown_message
    cs = correspondences([SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(1.0, 1.0), SVector(0.0, 1.0)],
                         [SVector(0.1, 0.0), SVector(1.2, 0.1), SVector(1.1, 1.3), SVector(0.0, 0.9)])
    est = EstimationProblem(HomographyMat, cs)
    ref = RefineProblem(abs, 1.0)
    mini = MinimalProblem(HomographyMat, cs, 1:4)
    # the lawful pairs
    @test check_kind(est, DLT()) === nothing
    @test check_kind(ref, LevenbergMarquardt()) === nothing && check_kind(ref, StaticLevenbergMarquardt()) === nothing
    @test check_kind(mini, Homography4pt()) === nothing
    # a refine handed no guess
    @test occursin("LevenbergMarquardt refines a guess", thrown_message(() -> solve(est, LevenbergMarquardt())))
    @test occursin("StaticLevenbergMarquardt refines a guess", thrown_message(() -> check_kind(est, StaticLevenbergMarquardt())))
    # a from-scratch algorithm handed a guess
    @test occursin("DLT estimates from scratch", thrown_message(() -> solve(ref, DLT())))
    @test occursin("Kabsch estimates from scratch", thrown_message(() -> check_kind(ref, Kabsch())))
    # a minimal solver handed a whole data set, or a guess
    @test occursin("Homography4pt is a minimal solver", thrown_message(() -> solve(est, Homography4pt())))
    @test occursin("Homography4pt is a minimal solver", thrown_message(() -> check_kind(ref, Homography4pt())))
    # a minimal sample handed to anything but a minimal solver
    @test occursin("DLT does not solve a minimal sample", thrown_message(() -> solve(mini, DLT())))
    @test occursin("LevenbergMarquardt does not solve a minimal sample", thrown_message(() -> check_kind(mini, LevenbergMarquardt())))
    # a minimal solver of another model
    @test occursin("Fundamental7pt solves a", thrown_message(() -> check_kind(mini, Fundamental7pt())))
    # either side may be the narrower model type
    @test check_kind(MinimalProblem(HomographyMat{Float64}, cs, 1:4), Homography4pt()) === nothing
    # the refusal comes first: no method of the pair runs (DLT has none on a RefineProblem, and the message is the law's)
    @test_throws ArgumentError solve(RefineProblem(abs, 1.0), DLT())
    # the kind readings
    @test VgtkCore.needs_initial_model(LevenbergMarquardt()) && !VgtkCore.needs_initial_model(DLT())
    @test !VgtkCore.decides_inliers(DLT()) && !VgtkCore.decides_inliers(Homography4pt())
    @test VgtkCore.estimator_kind(DLT()) === ClosedFormKind() && VgtkCore.estimator_kind(Homography4pt()) === MinimalKind()
    @test VgtkCore.estimator_kind(LevenbergMarquardt()) === RefineKind() && VgtkCore.estimator_kind(PhotonTransfer()) === FitKind()
end

@testitem "solve interface: every Core algorithm declares its kind; supports_solve_rng" begin
    using Test
    using VisualGeometryToolkitCore
    import VisualGeometryToolkitCore as VgtkCore
    # every algorithm Core defines answers estimator_kind: every Core binding that is a concrete
    # AbstractEstimationAlgorithm (the module and its Cameras submodule walked by name)
    algs = Set{Type}()
    for M in (VgtkCore, VgtkCore.Cameras), n in names(M; all = true)
        isdefined(M, n) || continue
        T = getfield(M, n)
        T isa Type && T <: AbstractEstimationAlgorithm || continue
        S = Base.unwrap_unionall(T)
        S isa DataType && !isabstracttype(S) && push!(algs, T)
    end
    @test length(algs) >= 20
    for T in algs
        @test hasmethod(VgtkCore.estimator_kind, Tuple{T})
    end
    @test all(T -> T in algs, (DLT, Kabsch, ChordalMean, SpectralAveraging, TotalLeastSquares, DirectEllipse, NearestPoint,
        Triangulation, Zhang, LevenbergMarquardt, StaticLevenbergMarquardt, ScaleMAP, PhotonTransfer, MasterDark))
    # randomness is a trait of the PAIR, derived from the kind: no Core algorithm draws
    est = EstimationProblem(Line2D, [rand(VgtkCore.SVector{2, Float64}) for _ in 1:4])
    @test !VgtkCore.supports_solve_rng(est, TotalLeastSquares())
    @test !VgtkCore.supports_solve_rng(RefineProblem(abs, 1.0), LevenbergMarquardt())
    # a find draws; an algorithm randomized without being a find overrides the trait
    struct _Find <: AbstractEstimationAlgorithm end
    VgtkCore.estimator_kind(::_Find) = FindKind()
    struct _RandomFit <: AbstractEstimationAlgorithm end
    VgtkCore.estimator_kind(::_RandomFit) = FitKind()
    VgtkCore.supports_solve_rng(::AbstractEstimationProblem, ::_RandomFit) = true
    @test VgtkCore.supports_solve_rng(est, _Find()) && VgtkCore.decides_inliers(_Find())
    @test VgtkCore.supports_solve_rng(est, _RandomFit()) && !VgtkCore.decides_inliers(_RandomFit())
end

@testitem "solve interface: the declared defaults; solve(problem); the no-default message" begin
    using Test, StaticArrays, Rotations
    using VisualGeometryToolkitCore
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: default_algorithm, thrown_message, Point
    p2 = [SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(1.0, 1.0), SVector(0.0, 1.0), SVector(0.5, 0.3), SVector(0.2, 0.8),
          SVector(0.7, 0.6), SVector(0.9, 0.1)]
    q2 = [SVector(0.1, 0.0), SVector(1.2, 0.1), SVector(1.1, 1.3), SVector(0.0, 0.9), SVector(0.6, 0.4), SVector(0.3, 0.9),
          SVector(0.8, 0.7), SVector(1.0, 0.2)]
    cs = correspondences(p2, q2)
    v3 = correspondences([SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0), SVector(0.0, 0.0, 1.0)],
                         [SVector(0.0, 1.0, 0.0), SVector(-1.0, 0.0, 0.0), SVector(0.0, 0.0, 1.0)])
    Rs = [RotMatrix{3}(RotZ(0.1)), RotMatrix{3}(RotZ(0.2))]
    θ = range(0, 2π; length = 9)[1:8]
    ell = [SVector(3cos(t) + 1, 2sin(t) - 1) for t in θ]
    lines = correspondences([SVector(0.0, 0.0), SVector(1.0, 0.0)], [SVector(0.0, 1.0), SVector(-1.0, 1.0)])
    @test default_algorithm(EstimationProblem(HomographyMat, cs)) === DLT()
    @test default_algorithm(EstimationProblem(FundMat, cs)) === DLT()
    @test default_algorithm(EstimationProblem(RotMatrix{3}, v3)) === Kabsch()
    @test default_algorithm(EstimationProblem(RotMatrix{3}, Rs)) === ChordalMean()
    @test default_algorithm(EstimationProblem(AbsoluteRotations, Dict((1, 2) => Rs[1]))) === SpectralAveraging()
    @test default_algorithm(EstimationProblem(Line2D, p2)) === TotalLeastSquares()
    @test default_algorithm(EstimationProblem(Line2D, withattrs(p2; weights = ones(8)))) === TotalLeastSquares()
    @test default_algorithm(EstimationProblem(Ellipse, ell)) === DirectEllipse()
    @test default_algorithm(EstimationProblem(Point, lines)) === NearestPoint()
    @test default_algorithm(EstimationProblem(Point3, SVector(0.0, 0.0, 1.0) => SVector(0.0, 0.0, 1.0),
                                              EuclideanMap(one(RotMatrix{3, Float64}), SVector(1.0, 0.0, 0.0)))) === Triangulation()
    @test default_algorithm(EstimationProblem(SquaredScale, (; rss = 2.0, dof = 4))) === ScaleMAP(VgtkCore.JeffreysPrior())
    struct _Family <: VgtkCore.AbstractConstraintProblem end       # a family problem: the chart-LM interface's implementer
    # its default is the host LM, declared with its solve (robust/refine.jl)
    @test default_algorithm(RefineProblem(_Family(), 1.0)) === LevenbergMarquardt()
    # a residual closure has no default: the device LM is named, and the error lists it
    closure_msg = thrown_message(() -> solve(RefineProblem(abs, SVector(1.0))))
    @test occursin("has no default algorithm", closure_msg) && occursin("StaticLevenbergMarquardt", closure_msg)
    @test LevenbergMarquardt() === LevenbergMarquardt(ClassDefault(), VgtkCore.MarquardtDamping(), VgtkCore.NativeSolver())
    # the backend: native by default, its damping the backend's (`default_damping`); a backend whose package is not
    # loaded states none, and Core's refines run the native backend only
    @test LevenbergMarquardt(VgtkCore.NativeSolver()) === LevenbergMarquardt()
    @test VgtkCore.default_damping(VgtkCore.NativeSolver()) === VgtkCore.MarquardtDamping()
    @test LevenbergMarquardt(VgtkCore.PoseLibSolver(); damping = VgtkCore.LevenbergDamping()).backend === VgtkCore.PoseLibSolver()
    # a constraint problem refined by another backend runs that backend's `lm_backend_refine`: none here
    @test_throws ArgumentError solve(RefineProblem(_Family(), 1.0),
                                     LevenbergMarquardt(VgtkCore.PoseLibSolver(); damping = VgtkCore.LevenbergDamping()))
    # a backend whose package implements no LM: one ArgumentError, from the constructor's default damping and from a
    # raw-data refine alike, naming the backend and the contract generics its package implements
    struct _NoLmBackend <: VgtkCore.AbstractSolverBackend end
    nolm = thrown_message(() -> LevenbergMarquardt(_NoLmBackend()))
    @test_throws ArgumentError LevenbergMarquardt(_NoLmBackend())
    @test occursin("_NoLmBackend backend has no Levenberg–Marquardt", nolm) && occursin("lm_backend_refine", nolm)
    H₀ = solve(EstimationProblem(HomographyMat, cs), DLT()).u
    @test thrown_message(() -> solve(RefineProblem(cs, H₀),
                                     LevenbergMarquardt(ClassDefault(), VgtkCore.MarquardtDamping(), _NoLmBackend()))) == nolm
    @test default_algorithm(MinimalProblem(HomographyMat, cs, 1:4)) === Homography4pt()
    @test default_algorithm(MinimalProblem(FundMat, cs, 1:7)) === Fundamental7pt()
    @test default_algorithm(MinimalProblem(RotMatrix{3}, v3, 1:2)) === Rotation2pt()
    @test default_algorithm(MinimalProblem(Circle, p2, 1:3)) === Circle3pt()
    @test default_algorithm(MinimalProblem(Ellipse, ell, 1:5)) === Ellipse5pt()
    # solve(problem) is solve(problem, default_algorithm(problem)), the default recorded on the result
    r = solve(EstimationProblem(HomographyMat, cs))
    @test r.u === solve(EstimationProblem(HomographyMat, cs), DLT()).u && r.alg === DLT()
    @test r.prob === EstimationProblem(HomographyMat, cs) || r.prob.f === cs
    @test solve(MinimalProblem(HomographyMat, cs, 1:4)) == solve(MinimalProblem(HomographyMat, cs, 1:4), Homography4pt())
    # a problem without a declared default: an ArgumentError listing the algorithms that solve it
    bearings = correspondences([SVector(0.0, 0.0, 1.0) for _ in 1:8], [SVector(0.0, 0.0, 1.0) for _ in 1:8])
    msg = thrown_message(() -> solve(EstimationProblem(EssentialMat, bearings)))
    @test occursin("has no default algorithm", msg) && occursin("DLT", msg)
    @test occursin("no loaded algorithm solves it", thrown_message(() -> solve(EstimationProblem(Float64, [1.0]))))
    # the native DLT has no essential 8-point
    @test occursin("NativeSolver backend has no 8-point", thrown_message(() -> solve(EstimationProblem(EssentialMat, bearings), DLT())))
    # a pair with no method at all is the pair's MethodError
    struct _Nothing <: AbstractEstimationAlgorithm end
    VgtkCore.estimator_kind(::_Nothing) = FitKind()
    @test_throws MethodError solve(EstimationProblem(Float64, [1.0]), _Nothing())
end

@testitem "solve interface: reinit! with a new algorithm or rng; results alias the cache" begin
    using Test, Random
    using VisualGeometryToolkitCore
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: ReturnCode
    # a configured algorithm that draws: its cache holds a buffer the result aliases (as a RANSAC's `resid`)
    struct _Jitter <: AbstractEstimationAlgorithm
        width::Float64
    end
    struct _Other <: AbstractEstimationAlgorithm end
    VgtkCore.estimator_kind(::Union{_Jitter, _Other}) = FitKind()
    VgtkCore.supports_solve_rng(::EstimationProblem, ::_Jitter) = true
    mutable struct _JitterCache{G}
        data::Vector{Float64}
        resid::Vector{Float64}
        width::Float64
        rng::G
    end
    VgtkCore.__init(prob::EstimationProblem{Float64}, alg::_Jitter; rng = Xoshiro(0)) =
        _JitterCache(prob.f, similar(prob.f), alg.width, rng)
    function VgtkCore.solve!(c::_JitterCache)
        c.resid .= c.data .+ c.width .* randn(c.rng, length(c.data))
        return FitResult(sum(c.resid), ReturnCode.Success, VgtkCore.ClosedForm, SolveStats(), NaN, length(c.data),
                         nothing; resid = c.resid)
    end
    function VgtkCore.reinit!(c::_JitterCache{G}; f, p, alg::_Jitter = _Jitter(c.width), rng::G = c.rng) where {G}
        c.data = f; length(c.resid) == length(f) || resize!(c.resid, length(f))
        c.width = alg.width; c.rng = rng
        return c
    end
    prob = EstimationProblem(Float64, [1.0, 2.0, 6.0])
    c = init(prob, _Jitter(1.0); rng = Xoshiro(1))
    solve!(c)
    # an algorithm swap: the same run as a fresh init with that algorithm (and the same rng state)
    VgtkCore.reinit!(c; alg = _Jitter(0.5), rng = Xoshiro(7))
    r = solve!(c)
    fresh = solve!(init(prob, _Jitter(0.5); rng = Xoshiro(7)))
    @test r.u === fresh.u && r.resid == fresh.resid && r.alg === _Jitter(0.5) && c.alg === _Jitter(0.5)
    # an rng swap alone keeps the algorithm
    VgtkCore.reinit!(c; rng = Xoshiro(3))
    @test solve!(c).u === solve!(init(prob, _Jitter(0.5); rng = Xoshiro(3))).u && c.alg === _Jitter(0.5)
    # another algorithm type is refused before anything changes
    prob_before = c.prob
    @test_throws ArgumentError VgtkCore.reinit!(c; alg = _Other(), f = [9.0])
    @test c.alg === _Jitter(0.5) && c.prob === prob_before && c.cache.data == [1.0, 2.0, 6.0]
    # results alias the cache: the next solve! overwrites a kept result's resid; a caller that keeps one copies it
    r1 = solve!(c); kept = copy(r1.resid)
    VgtkCore.reinit!(c; rng = Xoshiro(11))
    r2 = solve!(c)
    @test r1.resid === r2.resid === c.cache.resid && r1.resid != kept
end

@testitem "solve interface: init, solve!, reinit!; the front end records problem and algorithm" begin
    using Test
    using VisualGeometryToolkitCore
    import VisualGeometryToolkitCore as VgtkCore
    # an algorithm that holds a workspace: __init + solve!, and reinit! on its cache (Core's own algorithms hold
    # none: this is the contract a package's cached algorithm fulfils, e.g. Estimation's Ransac)
    using VisualGeometryToolkitCore: ReturnCode
    struct _Mean <: AbstractEstimationAlgorithm end
    VgtkCore.estimator_kind(::_Mean) = FitKind()
    mutable struct _MeanCache
        data::Vector{Float64}
        buffer::Vector{Float64}
        solves::Int
    end
    VgtkCore.__init(prob::EstimationProblem{Float64}, ::_Mean) = _MeanCache(prob.f, similar(prob.f), 0)
    function VgtkCore.solve!(c::_MeanCache)
        c.buffer .= c.data
        c.solves += 1
        return FitResult(sum(c.buffer) / length(c.buffer), ReturnCode.Success, VgtkCore.ClosedForm,
                         SolveStats(; nsteps = c.solves), NaN, length(c.data), nothing)
    end
    VgtkCore.reinit!(c::_MeanCache; f, p) = (c.data = f; resize!(c.buffer, length(f)); c)   # Core hands over the whole new problem
    prob = EstimationProblem(Float64, [1.0, 2.0, 6.0])
    c = init(prob, _Mean())
    @test c isa VgtkCore.EstimationCache && c.cache isa _MeanCache && c.prob === prob && c.alg === _Mean()
    r1 = solve!(c)
    @test r1.u == 3.0 && r1.stats.nsteps == 1 && r1.prob === prob && r1.alg === _Mean()   # recorded on this path too
    # the generic __solve is solve!(__init(…)), and the front end records the problem and the algorithm
    r = solve(prob, _Mean())
    @test r.u == 3.0 && r.prob === prob && r.alg === _Mean()
    # reinit! keeps the workspace, new data
    buf = c.cache.buffer
    VgtkCore.reinit!(c; f = [4.0, 8.0, 0.0])
    r2 = solve!(c)
    @test r2.u == 4.0 && r2.stats.nsteps == 2 && c.cache.buffer === buf
    @test r2.prob.f == [4.0, 8.0, 0.0] && c.prob.f === r2.prob.f      # the problem it now solves, recorded
    # the owner is handed the whole new problem, `f` and `p`; a keyword reinit! does not take is a MethodError
    @test_throws MethodError VgtkCore.reinit!(c; f = [1.0], rng = nothing)
    # new data of another type is refused BEFORE either side changes: the cache and its recorded problem as they were
    prob_before, data_before = c.prob, c.cache.data
    @test_throws ArgumentError VgtkCore.reinit!(c; f = Float32[1, 2])
    @test c.prob === prob_before && c.cache.data === data_before
    # init checks the kind law too
    @test_throws ArgumentError init(RefineProblem(abs, 1.0), _Mean())
    # a minimal solve returns its FixedModels unrecorded
    @test init isa Function && solve! isa Function
end

@testitem "solve interface: the closed forms' records — success, degenerate, malformed" begin
    using Test, StaticArrays, Rotations, LinearAlgebra
    using VisualGeometryToolkitCore
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: ReturnCode, ClosedForm, NoRule, Point
    p2 = [SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(1.0, 1.0), SVector(0.0, 1.0), SVector(0.5, 0.3)]
    q2 = [SVector(0.1, 0.0), SVector(1.2, 0.1), SVector(1.1, 1.3), SVector(0.0, 0.9), SVector(0.6, 0.4)]
    r = solve(EstimationProblem(HomographyMat, correspondences(p2, q2)), DLT())
    @test r.u isa HomographyMat{Float64} && r.retcode === ReturnCode.Success && r.stop_rule === ClosedForm
    @test r.stats === SolveStats() && isnan(r.rms) && r.n == 5 && r.resid === nothing && r.original === nothing
    @test successful_retcode(r) && VgtkCore.found_model(r)
    # bit for bit the inner-loop form on a fresh buffer and normalization
    cs = correspondences(p2, q2)
    @test r.u === VgtkCore.homography_dlt!(Matrix{Float64}(undef, 10, 9),
                                             VgtkCore.CorrespondenceNormalization(HomographyMat, cs), cs)
    # degenerate: u = nothing, (InitialFailure, NoRule), n = 0 — a typed result either way (points on a line admit
    # no ellipse)
    line = [SVector(Float64(i), 2.0i + 1) for i in 1:8]
    good = [SVector(3cos(t) + 1, 2sin(t) - 1) for t in range(0, 2π; length = 9)[1:8]]
    d = solve(EstimationProblem(Ellipse, line), DirectEllipse())
    e = solve(EstimationProblem(Ellipse, good), DirectEllipse())
    @test d.u === nothing && d.retcode === ReturnCode.InitialFailure && d.stop_rule === NoRule && d.n == 0
    @test e.u isa Ellipse{Float64} && e.n == 8 && typeof(d) == typeof(e)
    @test !successful_retcode(d) && !VgtkCore.found_model(d)
    # malformed input: the kernels' ArgumentErrors, unchanged in type
    @test_throws ArgumentError solve(EstimationProblem(Line2D, p2[1:2]), TotalLeastSquares())
    @test_throws ArgumentError solve(EstimationProblem(Line2D, withattrs(p2; weights = [1.0, -1.0, 1.0, 1.0, 1.0])), TotalLeastSquares())
    @test_throws ArgumentError solve(EstimationProblem(Ellipse, p2[1:4]), DirectEllipse())
    @test_throws ArgumentError solve(EstimationProblem(RotMatrix{3}, correspondences([SVector(1.0, 0.0, 0.0)], [SVector(0.0, 1.0, 0.0)])), Kabsch())
    @test_throws ArgumentError solve(EstimationProblem(RotMatrix{3}, RotMatrix{3, Float64}[]), ChordalMean())
    @test_throws ArgumentError solve(EstimationProblem(Point, correspondences([SVector(0.0, 0.0)], [SVector(1.0, 0.0)])), NearestPoint())
    # the triangulated point and its depths
    pose = EuclideanMap(one(RotMatrix{3, Float64}), SVector(-1.0, 0.0, 0.0))
    f1 = normalize(SVector(0.0, 0.0, 1.0)); f2 = normalize(SVector(-1.0, 0.0, 1.0))
    t = solve(EstimationProblem(Point3, f1 => f2, pose), Triangulation())
    X, λ1, λ2 = VgtkCore.triangulate_bearings(pose, f1, f2)
    @test t.u === X && t.metadata.depths === (λ1, λ2) && t.n == 1 && t.retcode === ReturnCode.Success
    # parallel rays: degenerate, the NaN point kept as a concrete `u`
    fp = normalize(SVector(0.1, -0.05, 1.0))
    along = EuclideanMap(one(RotMatrix{3, Float64}), 0.5 * fp)
    p = solve(EstimationProblem(Point3, fp => fp, along), Triangulation())
    @test p.retcode === ReturnCode.InitialFailure && p.stop_rule === NoRule && p.n == 0 && !successful_retcode(p)
    @test all(isnan, p.u) && all(isnan, p.metadata.depths) && typeof(p) == typeof(t)
    # the inverse-gamma MAP of ŝ², the inner loops' formula bit for bit
    prior = VgtkCore.IGPrior(2.0, 0.5)
    s = solve(EstimationProblem(SquaredScale, (; rss = 3.0, dof = 7)), ScaleMAP(prior))
    @test s.u === VgtkCore._ig_map_scale(prior, 3.0, 7) === (3.0 + 2 * 0.5) / (7 + 2 * 2.0)
    @test s.retcode === ReturnCode.Success && s.n == 0
    # the sensor estimators validate their configuration when built
    @test_throws ArgumentError PhotonTransfer(; significance = 1.5)
    @test_throws ArgumentError MasterDark(; false_defects = 2)
end

@testitem "solve interface: StaticLevenbergMarquardt in a KernelAbstractions kernel, the kernel's bits" begin
    using Test, StaticArrays, LinearAlgebra
    using KernelAbstractions: @kernel, @index, @Const, CPU, synchronize
    using VisualGeometryToolkitCore
    import VisualGeometryToolkitCore as VgtkCore
    struct _Cubic                      # an isbits residual: r(x) = A·(x − c) + 0.1·(x − c).^3
        A::SMatrix{2, 2, Float32, 4}
        c::SVector{2, Float32}
    end
    (f::_Cubic)(x) = (d = x - f.c; (f.A * d + 0.1f0 .* d .^ 3, f.A + Diagonal(0.3f0 .* d .^ 2)))
    @kernel function static_lm_rays!(xs, ns, oks, @Const(cs), stops)
        i = @index(Global)
        f = _Cubic(SMatrix{2, 2, Float32}(2, 0.3, 0.3, 1.5), cs[i])
        r = solve(RefineProblem(f, zero(SVector{2, Float32}); lb = SVector(-2.0f0, -2.0f0), ub = SVector(2.0f0, 2.0f0)),
                  StaticLevenbergMarquardt(); stops)
        @inbounds xs[i] = r.u
        @inbounds ns[i] = r.stats.nsteps
        @inbounds oks[i] = successful_retcode(r)
    end
    cs = [SVector{2, Float32}(0.3 * sin(k), 0.5 * cos(k)) for k in 1:64]
    stops = VgtkCore.StaticLmStops(SVector(1.0f0, 1.0f0); acceptance = 1.0f-8)
    xs = similar(cs); ns = zeros(Int, 64); oks = falses(64)
    static_lm_rays!(CPU(), 16)(xs, ns, oks, cs, stops; ndrange = 64)
    synchronize(CPU())
    for i in 1:64
        f = _Cubic(SMatrix{2, 2, Float32}(2, 0.3, 0.3, 1.5), cs[i])
        k = VgtkCore._static_lm(f, zero(SVector{2, Float32}), SVector(-2.0f0, -2.0f0), SVector(2.0f0, 2.0f0), stops)
        @test xs[i] === k[1] && ns[i] === k[6] && oks[i] === k[2]
    end
    @test all(oks)
    # the result itself is isbits, the problem and algorithm recorded on it
    f = _Cubic(SMatrix{2, 2, Float32}(2, 0.3, 0.3, 1.5), cs[1])
    r = solve(RefineProblem(f, zero(SVector{2, Float32})), StaticLevenbergMarquardt(); stops)
    @test isbits(r) && r.alg === StaticLevenbergMarquardt() && r.prob.f === f
end

@testitem "solve interface: the minimal defaults agree with the solver registry" begin
    using Test, StaticArrays
    using VisualGeometryToolkitCore
    import VisualGeometryToolkitCore as VgtkCore
    # two tables name the minimal solver of a model: the registry (`minimal_solvers`, by element types) and the declared
    # `default_algorithm` of a MinimalProblem (by model type). For every registered pair they must name the same solver.
    for (S, T) in ((SVector{2, Float64}, SVector{2, Float64}), (SVector{3, Float64}, SVector{3, Float64}))
        cs = correspondences(S[], T[])
        for s in VgtkCore.minimal_solvers(S, T)
            M = VgtkCore.model_type(s)
            @test VgtkCore.default_algorithm(MinimalProblem(M, cs, 1:VgtkCore.sample_size(s))) === s
            @test VgtkCore.minimal_solver(M, S, T) === s
        end
    end
end

@testitem "solve interface: the code review's edges — key order, plain poses, no guessed defaults" begin
    using Test, StaticArrays, Rotations, LinearAlgebra
    using VisualGeometryToolkitCore
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: SquaredScale, ScaleMAP, Triangulation, RadialCamera, MinimalProblem, default_algorithm,
        RefineProblem, EstimationProblem, ViewFrame
    # ScaleMAP reads its sums in any order, through the one keyed-context reader
    a = solve(EstimationProblem(SquaredScale, (; rss = 3.2, dof = 10)), ScaleMAP(VgtkCore.JeffreysPrior()))
    b = solve(EstimationProblem(SquaredScale, (; dof = 10, rss = 3.2)), ScaleMAP(VgtkCore.JeffreysPrior()))
    @test a.u === b.u
    @test_throws ArgumentError solve(EstimationProblem(SquaredScale, (; rss = 3.2)), ScaleMAP(VgtkCore.JeffreysPrior()))
    # Triangulation takes a plain (unitless) pose, labelled or not: the kernel's own domain
    pose = EuclideanMap(one(RotMatrix{3, Float64}), SVector(1.0, 0.0, 0.0))
    f1, f2 = normalize(SVector(0.1, 0.0, 1.0)), normalize(SVector(-0.9, 0.0, 1.0))
    @test solve(EstimationProblem(Point3, f1 => f2, pose)).u isa Point3
    labelled = relabel(ViewFrame{1}, ViewFrame{2}, pose)
    @test solve(EstimationProblem(Point3, f1 => f2, labelled)).u === solve(EstimationProblem(Point3, f1 => f2, pose)).u
    # no default where a default would guess: the radial camera's centre, a refine with no LM solve yet
    @test_throws ArgumentError default_algorithm(MinimalProblem(RadialCamera, Pair{SVector{2, Float64}, SVector{2, Float64}}[], 1:5))
    @test_throws ArgumentError default_algorithm(RefineProblem(x -> x, SVector(0.0, 0.0)))
end

# The front end's cache wrapper adds nothing per solve: `reinit!` (a new guess, data, algorithm configuration or run
# setting) and `solve!` on an `EstimationCache` allocate what the owner's own methods do — zero for an owner that
# allocates nothing (a streaming selector's per-pair reuse is pinned at zero downstream).
@testitem "solve interface: EstimationCache's reinit! and solve! allocate nothing of their own" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test, Random
    using VisualGeometryToolkitCore: EstimationProblem, init, solve!, reinit!, FitResult, ReturnCode, ClosedForm, SolveStats
    struct _ZeroAlg <: VgtkCore.AbstractEstimationAlgorithm
        k::Int
    end
    VgtkCore.estimator_kind(::_ZeroAlg) = VgtkCore.FitKind()
    VgtkCore.supports_solve_rng(::EstimationProblem, ::_ZeroAlg) = true
    mutable struct _ZeroCache{R}
        data::Vector{Float64}
        alg::_ZeroAlg
        rng::R
    end
    VgtkCore.__init(prob::EstimationProblem, alg::_ZeroAlg; rng = Random.default_rng()) = _ZeroCache(prob.f, alg, rng)
    VgtkCore.solve!(c::_ZeroCache) = FitResult(c.data[1], ReturnCode.Success, ClosedForm, SolveStats(), NaN, 0, nothing)
    function VgtkCore.reinit!(c::_ZeroCache; f = c.data, p = VgtkCore.NullParameters(), alg = c.alg, rng = c.rng)
        c.data = f; c.alg = alg; c.rng = rng
        return c
    end
    c = init(EstimationProblem(Float64, [1.0, 2.0]), _ZeroAlg(1); rng = Xoshiro(1))
    plain(c) = @allocations reinit!(c)
    with_alg(c, a, rng) = @allocations reinit!(c; alg = a, rng)
    with_data(c, d) = @allocations reinit!(c; f = d)
    run(c) = @allocations solve!(c)
    a, x, d = _ZeroAlg(2), Xoshiro(2), [3.0, 4.0]
    for _ in 1:2
        plain(c); with_alg(c, a, x); with_data(c, d); run(c)
    end
    @test plain(c) == 0
    @test with_alg(c, a, x) == 0
    @test with_data(c, d) == 0
    @test run(c) == 0
end
