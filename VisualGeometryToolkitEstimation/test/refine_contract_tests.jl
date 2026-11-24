# The refine contract (robust-estimation contract §1, §2, §12): a LevenbergMarquardt refine runs
# no scorer and estimates no scale; its algorithm takes `optimization_strategy =
# Triggs{K} | nothing` (an Irls{K} refused) and its solve EXACTLY ONE of `scale` / `width` for a
# robust strategy (neither for least squares); it records the scale it ran with,
# and returns `FitResult{M, RefineMetadata}` with per-datum loss weights.
@testitem "refine_contract" setup = [HomographyScenes, Contamination, ConicScenes, RefineSolve] begin
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, successful_retcode, RefineMetadata
    using VisualGeometryToolkitEstimation: Ransac, LocalOptimization
    using VisualGeometryToolkitCore: retcode_symbol
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using Test, Random, LinearAlgebra, StaticArrays, StructArrays
    using StableRNGs
    using VisualGeometryToolkitCore: Circle, Point2, HomographyMat, weights, weight_kind, noise_scale, scale_source, inlier_mask, ScaleGiven, ScaleUpstream, ScaleNone, IrlsWeights, BinaryWeights, WelschShape, HuberShape, TLSShape, GaussianShape, efficiency_constant, MsacScoring
    using VisualGeometryToolkitEstimation: CircleProblem, HomographyProblem, RansacWorkspace
    using VisualGeometryToolkitCore: Triggs, Irls

    src, dst = homography_pairs(StableRNG(11), H_PX; n_in = 120, n_out = 60, σ = 0.3, source = gaussian_source,
                                target = rng -> randn(rng, 2) .* 400 .+ 300)
    cs = StructArray(src .=> dst)
    found = solve(EstimationProblem(HomographyProblem(cs)), Ransac(); rng = StableRNG(1))
    H0 = found.u
    σ = 0.3
    err(f) = try
        f(); nothing
    catch e
        e
    end

    @testset "the scale input: exactly one of scale and width for a robust strategy" begin
        e = err(() -> solve(RefineProblem(HomographyProblem(cs), H0), LevenbergMarquardt()))                                   # the default Triggs needs one
        @test e isa ArgumentError && occursin("scale", e.msg) && occursin("width", e.msg)
        @test err(() -> solve(RefineProblem(HomographyProblem(cs), H0), LevenbergMarquardt(); scale = σ, width = 1.0)) isa ArgumentError
        @test err(() -> solve(RefineProblem(HomographyProblem(cs), H0), LevenbergMarquardt(); scale = -1.0)) isa ArgumentError
        @test err(() -> solve(RefineProblem(HomographyProblem(cs), H0), LevenbergMarquardt(); scale = NaN)) isa ArgumentError
        @test err(() -> solve(RefineProblem(HomographyProblem(cs), H0), LevenbergMarquardt(); width = 0.0)) isa ArgumentError
        # least squares reads none: one given is refused, not ignored
        @test err(() -> solve(RefineProblem(HomographyProblem(cs), H0), LevenbergMarquardt(; optimization_strategy = nothing); scale = σ)) isa ArgumentError
        @test err(() -> solve(RefineProblem(HomographyProblem(cs), H0), LevenbergMarquardt(; optimization_strategy = nothing); width = 1.0)) isa ArgumentError
        # an upstream result that recorded no scale is refused (noise_scale throws)
        l2 = solve(RefineProblem(HomographyProblem(cs), H0), LevenbergMarquardt(; optimization_strategy = nothing))
        @test err(() -> solve(RefineProblem(HomographyProblem(cs), H0), LevenbergMarquardt(); scale = l2)) isa ArgumentError
        # TLS has no efficiency constant: it needs one stated, or a width
        e = err(() -> solve(RefineProblem(HomographyProblem(cs), H0), LevenbergMarquardt(; optimization_strategy = Triggs{TLSShape}()); scale = σ))
        @test e isa ArgumentError && occursin("efficiency constant", e.msg)
        @test successful_retcode(solve(RefineProblem(HomographyProblem(cs), H0), LevenbergMarquardt(; optimization_strategy = Triggs{TLSShape}()); width = 3σ))
        # a width with a strategy's own constant would ignore the constant
        @test err(() -> solve(RefineProblem(HomographyProblem(cs), H0), LevenbergMarquardt(; optimization_strategy = Triggs{WelschShape}(efficiency_constant = 2.5)); width = 1.0)) isa ArgumentError
    end

    @testset "the refused keywords and strategies, each with the way to say it" begin
        # a refine's run settings are its solve keywords (scale, width, stops): a find's settings, a scorer, a loss given
        # apart from its strategy, an extent — none is one (D5: an unknown solve keyword is a MethodError), on enough
        # data and on too few alike (never a silent :degenerate)
        for (k, v) in ((:scoring, MsacScoring(HomographyProblem; threshold = 3.0)), (:threshold, 3.0), (:outlier_halfwidth, 20.0),
                       (:loss, WelschShape()), (:loss_width, 1.0), (:efficiency_constant, 2.5),
                       (:local_optimization, Triggs{WelschShape}()), (:local_optimization_strategy, Triggs{WelschShape}()),
                       (:robust, true), (:extent, nothing), (:max_trials, 10), (:rng, StableRNG(1)), (:nonsense, 1))
            @test err(() -> lm(HomographyProblem(cs), H0; scale = σ, (; k => v)...)) isa MethodError
            @test err(() -> lm(HomographyProblem(cs[1:3]), H0; scale = σ, (; k => v)...)) isa MethodError
        end
        # a strategy that needs a consensus set names LocalOptimization; least squares is spelled nothing
        e = err(() -> solve(RefineProblem(HomographyProblem(cs), H0), LevenbergMarquardt(; optimization_strategy = VgtkEstimation.TerminalLO(Irls{TLSShape}()))))
        @test e isa ArgumentError && occursin("LocalOptimization", e.msg)
        e = err(() -> solve(RefineProblem(HomographyProblem(cs), H0), LevenbergMarquardt(; optimization_strategy = Irls{GaussianShape}())))
        @test e isa ArgumentError && occursin("optimization_strategy = nothing", e.msg)
        e = err(() -> solve(RefineProblem(HomographyProblem(cs), H0), LevenbergMarquardt(; optimization_strategy = Irls{WelschShape}(loss_width = 1.0))))
        @test e isa ArgumentError && occursin("width = w", e.msg)
        # the too-few path applies the scale-input rule and the strategy refusals too
        for kws in ((;), (; scale = σ, width = 1.0), (; scale = -1.0), (; optimization_strategy = nothing, scale = σ),
                    (; optimization_strategy = Triggs{TLSShape}(), scale = σ),
                    (; optimization_strategy = VgtkEstimation.TerminalLO(Irls{TLSShape}()), scale = σ),
                    (; optimization_strategy = Irls{GaussianShape}(), scale = σ))
            e, e3 = err(() -> lm(HomographyProblem(cs), H0; kws...)), err(() -> lm(HomographyProblem(cs[1:3]), H0; kws...))
            @test e isa ArgumentError && e3 isa ArgumentError && e3.msg == e.msg
        end
        # every family, too few data included: a scorer is no refine's
        @test err(() -> solve(RefineProblem(CircleProblem([Point2(1.0, 0.0)]), Circle(Point2(0.0, 0.0), 1.0)), LevenbergMarquardt(); scale = σ, scoring = MsacScoring(CircleProblem; threshold = 1.0))) isa MethodError
    end

    @testset "the scale record: given, upstream, a width alone, least squares" begin
        c = efficiency_constant(WelschShape, 2)                 # the homography's transfer residual: codimension 2
        rg = solve(RefineProblem(HomographyProblem(cs), H0), LevenbergMarquardt(); scale = σ)
        @test noise_scale(rg) === σ && scale_source(rg) === ScaleGiven
        @test rg.metadata.loss_width ≈ c * σ
        # the hand-off: an upstream result's own scale — the find_* result's (its scorer's ŝ), or a refine's
        ru = solve(RefineProblem(HomographyProblem(cs), H0), LevenbergMarquardt(); scale = found)
        @test scale_source(ru) === ScaleUpstream && noise_scale(ru) === noise_scale(found)
        @test ru.u == solve(RefineProblem(HomographyProblem(cs), H0), LevenbergMarquardt(); scale = noise_scale(found)).u
        rr = solve(RefineProblem(HomographyProblem(cs), H0), LevenbergMarquardt(); scale = rg)
        @test scale_source(rr) === ScaleUpstream && noise_scale(rr) === σ && rr.u == rg.u
        rw = solve(RefineProblem(HomographyProblem(cs), H0), LevenbergMarquardt(); width = c * σ)
        @test scale_source(rw) === ScaleNone && rw.metadata.loss_width ≈ c * σ
        @test_throws ArgumentError noise_scale(rw)
        rl = solve(RefineProblem(HomographyProblem(cs), H0), LevenbergMarquardt(; optimization_strategy = nothing))
        @test scale_source(rl) === ScaleNone && isnan(rl.metadata.loss_width)
        @test_throws ArgumentError noise_scale(rl)
        # a stated efficiency constant is the plain multiple: width = c × scale
        rc = solve(RefineProblem(HomographyProblem(cs), H0), LevenbergMarquardt(; optimization_strategy = Triggs{WelschShape}(efficiency_constant = 2.0)); scale = σ)
        @test rc.metadata.loss_width ≈ 2σ
        # the same loss whether given as c·σ by scale or as width (the scale record aside)
        @test solve(RefineProblem(HomographyProblem(cs), H0), LevenbergMarquardt(; optimization_strategy = Triggs{WelschShape}(efficiency_constant = 2.0)); scale = 1.0).u ==
            solve(RefineProblem(HomographyProblem(cs), H0), LevenbergMarquardt(); width = 2.0).u
    end

    @testset "the result: FitResult{M, RefineMetadata}, type-stable accessors, weights and their kind" begin
        r = @inferred solve(RefineProblem(HomographyProblem(cs), H0), LevenbergMarquardt(); scale = σ)
        @test r isa VgtkCore.FitResult{<:HomographyMat, <:RefineMetadata}
        @test (@inferred noise_scale(r)) isa Float64
        @test (@inferred scale_source(r)) isa VgtkCore.ScaleSource
        @test (@inferred weights(r)) isa Vector{Float64}
        @test (@inferred weight_kind(r)) === IrlsWeights()
        @test isbitstype(typeof(scale_source(r)))
        w = weights(r)
        @test length(w) == length(cs) == length(r.resid) && all(0 .<= w .<= 1)
        @test r.n == count(>(0), w)
        @test r.rms ≈ sqrt(sum(w .* r.resid .^ 2) / sum(w))
        # Welsch: far outliers carry no weight, inliers near one; a loss weight has no inlier cut of its own
        @test count(>(0.5), w) == 120
        @test_throws ArgumentError inlier_mask(r)
        @test count(inlier_mask(r, 0.5)) == 120
        # least squares: lazy all-ones weights, binary, every datum
        rl = @inferred solve(RefineProblem(HomographyProblem(cs), H0), LevenbergMarquardt(; optimization_strategy = nothing))
        @test weight_kind(rl) === BinaryWeights() && all(==(1), weights(rl)) && sizeof(weights(rl)) < 64
        @test rl.n == length(cs) && all(inlier_mask(rl))
        @test (@inferred weight_kind(solve(RefineProblem(HomographyProblem(cs), H0), LevenbergMarquardt(; optimization_strategy = Triggs{HuberShape}()); scale = σ))) === IrlsWeights()
        # a refine's one fixed-scale robust path is ONE robustified LM: an Irls (reweighting cycles) is refused, naming the
        # Triggs of its loss
        e = try solve(RefineProblem(HomographyProblem(cs), H0), LevenbergMarquardt(; optimization_strategy = Irls{HuberShape}()); scale = σ); nothing; catch err; err; end
        @test e isa ArgumentError && occursin("Triggs{HuberShape}", e.msg) && occursin("ONE robustified LM", e.msg)
        @test isconcretetype(typeof(r.metadata))
        # too few data: the guess, unrefined, of the same type
        rf = solve(RefineProblem(HomographyProblem(cs[1:3]), H0), LevenbergMarquardt(); scale = σ)
        @test typeof(rf) === typeof(r) && retcode_symbol(rf) === :degenerate && rf.n == 0 && rf.stats.nsteps == 0
        @test rf.u == H0
    end

    @testset "every strategy recovers the truth from a perturbed guess on contaminated data" begin
        pts, _ = circle_scene(Xoshiro(41), 800)
        g = Circle(Point2(C₀ + SVector(2.0, -1.5)), R₀ + 3.0)
        for (lo, kw) in ((Triggs{WelschShape}(), (; scale = 0.5)), (Triggs{TLSShape}(), (; width = 1.5)))
            cr = lm(CircleProblem(pts), g; optimization_strategy = lo, kw...).u
            @test norm(SVector(cr.center[1], cr.center[2]) - C₀) < 0.1 && abs(cr.r - R₀) < 0.1
        end
        # the chart LM's task-local workspace is reused: the same refine twice is the same model
        @test solve(RefineProblem(CircleProblem(pts), g), LevenbergMarquardt(); scale = 0.5).u == solve(RefineProblem(CircleProblem(pts), g), LevenbergMarquardt(); scale = 0.5).u
    end

    @testset "the RANSAC-family seeded refine is LocalOptimization, with its scorer" begin
        r = solve(RefineProblem(HomographyProblem(cs), H0), LocalOptimization(Irls{TLSShape}(); scoring = MsacScoring(HomographyProblem; threshold = 3.0)))
        @test successful_retcode(r) && count(inlier_mask(r)) >= 115
    end
end

# The contract on EVERY family problem's refine (the 11 RANSAC classes and the camera and bearing forms of the pose):
# the refusals, the default's need of a scale, least squares, the result of a concrete type (checked at run time: some
# classes' problem constructors do not infer, so the call itself is not `@inferred`) with its accessors inferred — on
# too few data for a minimal sample (the refine parses its strategy and scale first, so the too-few path is the
# contract's, cheaply), and hand-eye (whose problem needs 3 poses to exist) on 3.
@testitem "refine_contract: every family problem's refine" setup = [TooFewData, RefineSolve] begin
    using VisualGeometryToolkitCore: RefineMetadata
    using VisualGeometryToolkitCore: retcode_symbol
    import VisualGeometryToolkitCore as VgtkCore
    using Test, LinearAlgebra, StaticArrays
    using Rotations: RotMatrix, RotXYZ
    using StableRNGs: StableRNG
    using VisualGeometryToolkitCore: Point2, Circle, Ellipse, CameraModel, Correspondences, correspondences, HomographyMat, FundMat, EssentialMat, RadialHomography, RadialCamera, Extrinsics3, Vec3, EuclideanMap, MsacScoring, WelschShape, ScaleGiven, ScaleNone, scale_source, weight_kind, IrlsWeights, BinaryWeights
    using VisualGeometryToolkitEstimation: HomographyProblem, FundMatProblem, EssentialProblem, RotationProblem, CircleProblem,
        EllipseProblem, RadialHomographyProblem, RadialCameraProblem, AbsolutePoseProblem, BearingPoseProblem, HandEyeProblem
    using VisualGeometryToolkitCore: Triggs, Irls, solve, RefineProblem, LevenbergMarquardt

    pts, cs = few_points, few_correspondences
    I3 = SMatrix{3, 3, Float64}(I)
    H = HomographyMat{Float64}(Tuple(I3))
    X3 = SMatrix{3, 3, Float64}(0, 0, 0, 0, 0, -1, 0, 1, 0)
    cam = CameraModel(; focal = 800.0, pp = (480.0, 360.0))
    c = Point2(500.0, 400.0)
    Xs = [SA[0.1, 0.2, 2.0], SA[-0.3, 0.1, 2.5]]
    us = [SA[400.0, 300.0], SA[500.0, 350.0]]
    pose = Extrinsics3(RotMatrix{3, Float64}(I3), Vec3{Float64}(0.0, 0.0, 0.0))
    rigid(R, t) = EuclideanMap(RotMatrix{3, Float64}(R), SVector{3, Float64}(t))
    A = [rigid(RotXYZ(0.3i, -0.2i, 0.1i^2), SVector(0.1i, -0.05i, 0.02i)) for i in 1:3]
    Xhe = rigid(RotXYZ(0.1, 0.2, -0.3), SVector(0.05, 0.0, 0.1))
    hecs = [Ai => inv(Xhe) ∘ Ai ∘ Xhe for Ai in A]
    cases = (   # class => (the family problem, the guess)
        :homography => (HomographyProblem(cs(3)), H),
        :fundmat => (FundMatProblem(cs(6)), FundMat{Float64}(Tuple(X3))),
        :essential => (EssentialProblem(cam, cam, cs(4)), EssentialMat{Float64}(Tuple(X3))),
        :rotation => (RotationProblem(cam, cam, cs(1)), one(RotMatrix{3, Float64})),
        :circle => (CircleProblem(pts(2)), Circle(Point2(0.0, 0.0), 1.0)),
        :ellipse => (EllipseProblem(pts(4)), Ellipse(Point2(0.0, 0.0), 2.0, 1.0, 0.0)),
        :radial_homography => (RadialHomographyProblem(cs(4); center = VgtkCore.center(RadialHomography{Float64}(H, 0.0))), RadialHomography{Float64}(H, 0.0)),
        :radial_camera => (RadialCameraProblem(cs(4); center = c), RadialCamera(H, c)),
        :absolute_pose => (AbsolutePoseProblem(cam, Xs .=> us), pose),
        :bearing_pose => (BearingPoseProblem(Correspondences(Xs, [normalize(x) for x in Xs])), pose),
        :hand_eye => (HandEyeProblem(hecs), Xhe),
    )
    err(f) = try
        f(); nothing
    catch e
        e
    end
    refused = (:scoring => MsacScoring(HomographyProblem; threshold = 3.0), :threshold => 3.0, :outlier_halfwidth => 20.0,
               :loss => WelschShape(), :loss_width => 1.0, :efficiency_constant => 2.5, :local_optimization => Triggs{WelschShape}(),
               :local_optimization_strategy => Triggs{WelschShape}(), :robust => true, :extent => nothing,
               :max_trials => 10, :min_trials => 1, :rng => StableRNG(1), :nonsense => 1)
    for (class, (prob, θ)) in cases
        f(; kw...) = lm(prob, θ; kw...)
        # no such solve keyword, the same for every class
        for (k, v) in refused
            @test err(() -> f(; scale = 1.0, (; k => v)...)) isa MethodError
        end
        # the default is Triggs{Welsch}: it needs exactly one of scale and width
        @test err(() -> f()) isa ArgumentError
        @test err(() -> f(; scale = 1.0, width = 1.0)) isa ArgumentError
        @test err(() -> f(; optimization_strategy = nothing, scale = 1.0)) isa ArgumentError
        @test err(() -> f(; optimization_strategy = Irls{VgtkCore.GaussianShape}(), scale = 1.0)) isa ArgumentError
        @test occursin("Triggs{HuberShape}", err(() -> f(; optimization_strategy = Irls{VgtkCore.HuberShape}(), scale = 1.0)).msg)
        # the result: a refine's, of a concrete type, its accessors inferred; least squares records no scale, a robust
        # strategy the given one
        r = f(; scale = 1.0)
        @test isconcretetype(typeof(r)) && r isa VgtkCore.FitResult{<:Any, <:RefineMetadata}
        @test (@inferred scale_source(r)) === ScaleGiven && (@inferred weight_kind(r)) === IrlsWeights()
        @test (@inferred VgtkCore.noise_scale(r)) === 1.0 && (@inferred VgtkCore.weights(r)) isa Vector{Float64}
        l2 = f(; optimization_strategy = nothing)
        @test isconcretetype(typeof(l2)) && (@inferred scale_source(l2)) === ScaleNone
        @test l2 isa VgtkCore.FitResult{<:Any, <:RefineMetadata{Nothing}} && scale_source(l2) === ScaleNone &&
            weight_kind(l2) === BinaryWeights()
        class === :hand_eye || @test retcode_symbol(r) === :degenerate && retcode_symbol(l2) === :degenerate
    end
end

# The refine weights are Core's IrlsWeights, "normalized to a maximum of 1" (robust/weights.jl): the fits' one rule
# ρ′(r²)/ρ′(0) (irls_weights.jl `_irls_weight`), below eps an exact 0. SmoothTLS is the shape whose ρ′(0) is 2: its
# refine weights reached 2 before they shared the fits' rule.
@testitem "refine_contract: Triggs weights are normalized to a maximum of 1" setup = [HomographyScenes] begin
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt
    using VisualGeometryToolkitEstimation: Ransac, HomographyProblem
    using Test, StructArrays, StableRNGs
    using VisualGeometryToolkitCore: SmoothTLSShape, HuberShape, Triggs, weights, weight_kind, IrlsWeights, weight
    
    src, dst = homography_pairs(StableRNG(11), H_PX; n_in = 80, n_out = 0, σ = 0.3, source = gaussian_source)
    cs = StructArray(src .=> dst)
    H0 = solve(EstimationProblem(HomographyProblem(cs)), Ransac(); rng = StableRNG(1)).u
    c = 3.0
    for lo in (Triggs{SmoothTLSShape}(), Triggs{HuberShape}())   # an Irls refine is refused (_refine.jl)
        r = solve(RefineProblem(HomographyProblem(cs), H0), LevenbergMarquardt(; optimization_strategy = lo); width = c)
        w = weights(r)
        @test weight_kind(r) isa IrlsWeights
        @test maximum(w) <= 1.0 && maximum(w) > 0.9                 # inliers near zero residual weigh near 1, never 2
        shape = typeof(lo).parameters[1]()
        @test w == [(x = weight(shape, e^2, c^2) / weight(shape, 0.0, c^2); x < eps() ? 0.0 : x) for e in r.resid]
    end
end
