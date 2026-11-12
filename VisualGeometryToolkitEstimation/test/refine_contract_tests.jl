# The refine contract (robust-estimation contract §1, §2, §12): a refine_* runs
# no scorer and estimates no scale; it takes `optimization_strategy =
# Triggs{K} | Irls{K} | nothing` and EXACTLY ONE of `scale` / `width` for a
# robust strategy (neither for least squares), records the scale it ran with,
# and returns `FitResult{M, RefineMetadata}` with per-datum loss weights.
@testitem "refine_contract" setup = [HomographyScenes, Contamination, ConicScenes] begin
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using Test, Random, LinearAlgebra, StaticArrays, StructArrays
    using StableRNGs
    using VisualGeometryToolkitCore: Circle, Point2, HomographyMat, estimate, retcode, iterations, weights, weight_kind,
        noise_scale, scale_source, inlier_mask, ScaleGiven, ScaleUpstream, ScaleNone, IrlsWeights, BinaryWeights,
        WelschShape, HuberShape, TLSShape, GaussianShape, efficiency_constant, MsacScoring
    using VisualGeometryToolkitEstimation: refine_homography, refine_circle, find_homography, Triggs, Irls,
        RefineMetadata, CircleProblem, HomographyProblem, RansacWorkspace, locally_optimize

    src, dst = homography_pairs(StableRNG(11), H_PX; n_in = 120, n_out = 60, σ = 0.3, source = gaussian_source,
                                target = rng -> randn(rng, 2) .* 400 .+ 300)
    cs = StructArray(src .=> dst)
    found = find_homography(cs; rng = StableRNG(1))
    H0 = estimate(found)
    σ = 0.3
    err(f) = try
        f(); nothing
    catch e
        e
    end

    @testset "the scale input: exactly one of scale and width for a robust strategy" begin
        e = err(() -> refine_homography(H0, cs))                                   # the default Triggs needs one
        @test e isa ArgumentError && occursin("scale", e.msg) && occursin("width", e.msg)
        @test err(() -> refine_homography(H0, cs; scale = σ, width = 1.0)) isa ArgumentError
        @test err(() -> refine_homography(H0, cs; scale = -1.0)) isa ArgumentError
        @test err(() -> refine_homography(H0, cs; scale = NaN)) isa ArgumentError
        @test err(() -> refine_homography(H0, cs; width = 0.0)) isa ArgumentError
        # least squares reads none: one given is refused, not ignored
        @test err(() -> refine_homography(H0, cs; optimization_strategy = nothing, scale = σ)) isa ArgumentError
        @test err(() -> refine_homography(H0, cs; optimization_strategy = nothing, width = 1.0)) isa ArgumentError
        # an upstream result that recorded no scale is refused (noise_scale throws)
        l2 = refine_homography(H0, cs; optimization_strategy = nothing)
        @test err(() -> refine_homography(H0, cs; scale = l2)) isa ArgumentError
        # TLS has no efficiency constant: it needs one stated, or a width
        e = err(() -> refine_homography(H0, cs; optimization_strategy = Triggs{TLSShape}(), scale = σ))
        @test e isa ArgumentError && occursin("efficiency constant", e.msg)
        @test VgtkCore.converged(refine_homography(H0, cs; optimization_strategy = Triggs{TLSShape}(), width = 3σ))
        # a width with a strategy's own constant would ignore the constant
        @test err(() -> refine_homography(H0, cs; optimization_strategy = Triggs{WelschShape}(efficiency_constant = 2.5),
                                          width = 1.0)) isa ArgumentError
    end

    @testset "the refused keywords, each with the way to say it" begin
        for (k, v, says) in ((:scoring, MsacScoring(HomographyProblem; threshold = 3.0), "no scorer"),
                             (:threshold, 3.0, "no scorer"), (:outlier_halfwidth, 20.0, "no scorer"),
                             (:loss, WelschShape(), "optimization_strategy"), (:loss_width, 1.0, "width"),
                             (:efficiency_constant, 2.5, "optimization_strategy"),
                             (:local_optimization, Triggs{WelschShape}(), "give `optimization_strategy`"),
                             (:local_optimization_strategy, Triggs{WelschShape}(), "give `optimization_strategy`"),
                             (:robust, true, "give `optimization_strategy`"),
                             (:extent, nothing, "BoundedProblem"), (:max_trials, 10, "no samples"),
                             (:rng, StableRNG(1), "no samples"), (:nonsense, 1, "optimization_strategy, scale, width"))
            e = err(() -> refine_homography(H0, cs; scale = σ, (; k => v)...))
            @test e isa ArgumentError && occursin("`$k`", e.msg) && occursin(says, e.msg)
            # too few data for a minimal sample: the same refusal, with the same message (never a silent :degenerate)
            e3 = err(() -> refine_homography(H0, cs[1:3]; scale = σ, (; k => v)...))
            @test e3 isa ArgumentError && e3.msg == e.msg
        end
        # a strategy that needs a consensus set names locally_optimize; least squares is spelled nothing
        e = err(() -> refine_homography(H0, cs; optimization_strategy = VgtkEstimation.TerminalLO(Irls{TLSShape}())))
        @test e isa ArgumentError && occursin("locally_optimize", e.msg)
        e = err(() -> refine_homography(H0, cs; optimization_strategy = Irls{GaussianShape}()))
        @test e isa ArgumentError && occursin("optimization_strategy = nothing", e.msg)
        e = err(() -> refine_homography(H0, cs; optimization_strategy = Irls{WelschShape}(loss_width = 1.0)))
        @test e isa ArgumentError && occursin("width = w", e.msg)
        # the too-few path applies the scale-input rule and the strategy refusals too
        for kws in ((;), (; scale = σ, width = 1.0), (; scale = -1.0), (; optimization_strategy = nothing, scale = σ),
                    (; optimization_strategy = Triggs{TLSShape}(), scale = σ),
                    (; optimization_strategy = VgtkEstimation.TerminalLO(Irls{TLSShape}()), scale = σ),
                    (; optimization_strategy = Irls{GaussianShape}(), scale = σ))
            e, e3 = err(() -> refine_homography(H0, cs; kws...)), err(() -> refine_homography(H0, cs[1:3]; kws...))
            @test e isa ArgumentError && e3 isa ArgumentError && e3.msg == e.msg
        end
        # every refine_* entry, the bearing pose and the bundles' classes included: too few data refuses a scorer
        @test err(() -> refine_circle(Circle(Point2(0.0, 0.0), 1.0), [Point2(1.0, 0.0)]; scale = σ,
                                      scoring = MsacScoring(CircleProblem; threshold = 1.0))) isa ArgumentError
    end

    @testset "the scale record: given, upstream, a width alone, least squares" begin
        c = efficiency_constant(WelschShape, 2)                 # the homography's transfer residual: codimension 2
        rg = refine_homography(H0, cs; scale = σ)
        @test noise_scale(rg) === σ && scale_source(rg) === ScaleGiven
        @test rg.loss_width ≈ c * σ
        # the hand-off: an upstream result's own scale — the find_* result's (its scorer's ŝ), or a refine's
        ru = refine_homography(H0, cs; scale = found)
        @test scale_source(ru) === ScaleUpstream && noise_scale(ru) === noise_scale(found)
        @test estimate(ru) == estimate(refine_homography(H0, cs; scale = noise_scale(found)))
        rr = refine_homography(H0, cs; scale = rg)
        @test scale_source(rr) === ScaleUpstream && noise_scale(rr) === σ && estimate(rr) == estimate(rg)
        rw = refine_homography(H0, cs; width = c * σ)
        @test scale_source(rw) === ScaleNone && isnan(rw.scale) && rw.loss_width ≈ c * σ
        @test_throws ArgumentError noise_scale(rw)
        rl = refine_homography(H0, cs; optimization_strategy = nothing)
        @test scale_source(rl) === ScaleNone && isnan(rl.loss_width)
        @test_throws ArgumentError noise_scale(rl)
        # a stated efficiency constant is the plain multiple: width = c × scale
        rc = refine_homography(H0, cs; optimization_strategy = Triggs{WelschShape}(efficiency_constant = 2.0), scale = σ)
        @test rc.loss_width ≈ 2σ
        # the same loss whether given as c·σ by scale or as width (the scale record aside)
        @test estimate(refine_homography(H0, cs; optimization_strategy = Triggs{WelschShape}(efficiency_constant = 2.0), scale = 1.0)) ==
            estimate(refine_homography(H0, cs; width = 2.0))
    end

    @testset "the result: FitResult{M, RefineMetadata}, type-stable accessors, weights and their kind" begin
        r = @inferred refine_homography(H0, cs; scale = σ)
        @test r isa VgtkCore.FitResult{<:HomographyMat, <:RefineMetadata}
        @test (@inferred noise_scale(r)) isa Float64
        @test (@inferred scale_source(r)) isa VgtkCore.ScaleSource
        @test (@inferred weights(r)) isa Vector{Float64}
        @test (@inferred weight_kind(r)) === IrlsWeights()
        @test isbitstype(typeof(r.scale_source))
        w = weights(r)
        @test length(w) == length(cs) == length(r.residuals) && all(0 .<= w .<= 1)
        @test r.n == count(>(0), w)
        @test r.rms ≈ sqrt(sum(w .* r.residuals .^ 2) / sum(w))
        # Welsch: far outliers carry no weight, inliers near one; a loss weight has no inlier cut of its own
        @test count(>(0.5), w) == 120
        @test_throws ArgumentError inlier_mask(r)
        @test count(inlier_mask(r, 0.5)) == 120
        # least squares: lazy all-ones weights, binary, every datum
        rl = @inferred refine_homography(H0, cs; optimization_strategy = nothing)
        @test weight_kind(rl) === BinaryWeights() && all(==(1), weights(rl)) && sizeof(weights(rl)) < 64
        @test rl.n == length(cs) && all(inlier_mask(rl))
        @test (@inferred weight_kind(refine_homography(H0, cs; optimization_strategy = Irls{HuberShape}(), scale = σ))) === IrlsWeights()
        @test isconcretetype(typeof(VgtkCore.metadata(r)))
        # too few data: the guess, unrefined, of the same type
        rf = refine_homography(H0, cs[1:3]; scale = σ)
        @test typeof(rf) === typeof(r) && retcode(rf) === :degenerate && rf.n == 0 && iterations(rf) == 0
        @test estimate(rf) == H0
    end

    @testset "every strategy recovers the truth from a perturbed guess on contaminated data" begin
        pts, _ = circle_scene(Xoshiro(41), 800)
        g = Circle(Point2(C₀ + SVector(2.0, -1.5)), R₀ + 3.0)
        for (lo, kw) in ((Triggs{WelschShape}(), (; scale = 0.5)), (Irls{WelschShape}(), (; scale = 0.5)),
                         (Triggs{TLSShape}(), (; width = 1.5)))
            cr = estimate(refine_circle(g, pts; optimization_strategy = lo, kw...))
            @test norm(SVector(cr.center[1], cr.center[2]) - C₀) < 0.1 && abs(cr.r - R₀) < 0.1
        end
        # a workspace is reused, never required
        ws = RansacWorkspace(CircleProblem, 800)
        @test estimate(refine_circle(g, pts; scale = 0.5, workspace = ws)) == estimate(refine_circle(g, pts; scale = 0.5))
    end

    @testset "the RANSAC-family seeded refine stays locally_optimize, with its scorer" begin
        r = locally_optimize(HomographyProblem(cs), H0; scoring = MsacScoring(HomographyProblem; threshold = 3.0),
                             local_optimization_strategy = Irls{TLSShape}())
        @test VgtkCore.converged(r) && count(inlier_mask(r)) >= 115
    end
end

# The contract on EVERY refine_* class (the 11 RANSAC classes, the rotation's two forms and the camera and bearing
# forms of the pose): the refusals and their messages, the default's need of a scale, least squares, and the result's
# type inferred — on too few data for a minimal sample (the refine parses every keyword first, so the too-few path is
# the contract's, cheaply), and hand-eye (whose problem needs 3 poses to exist) on 3.
@testitem "refine_contract: every refine_* class" setup = [TooFewData] begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test, LinearAlgebra, StaticArrays
    using Rotations: RotMatrix, RotXYZ
    using StableRNGs: StableRNG
    using VisualGeometryToolkitCore: Point2, Circle, Ellipse, CameraModel, Csponds, csponds, HomographyMat, FundMat,
        EssentialMat, RadialHomography, RadialCamera, Extrinsics3, Vec3, EuclideanMap, MsacScoring, WelschShape,
        retcode, estimate, ScaleGiven, ScaleNone, scale_source, weight_kind, IrlsWeights, BinaryWeights
    using VisualGeometryToolkitEstimation: refine_homography, refine_fundmat, refine_essential, refine_rotation,
        refine_circle, refine_ellipse, refine_radial_homography, refine_radial_camera, refine_absolute_pose,
        refine_hand_eye, RefineMetadata, Triggs, Irls, HomographyProblem

    pts, cs = few_points, few_csponds
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
    cases = (
        :homography => (; kw...) -> refine_homography(H, cs(3); kw...),
        :fundmat => (; kw...) -> refine_fundmat(FundMat{Float64}(Tuple(X3)), cs(6); kw...),
        :essential => (; kw...) -> refine_essential(EssentialMat{Float64}(Tuple(X3)), cam, cam, cs(4); kw...),
        :rotation => (; kw...) -> refine_rotation(one(RotMatrix{3, Float64}), cam, cam, cs(1); kw...),
        :rotation_bearings => (; kw...) -> refine_rotation(one(RotMatrix{3, Float64}), Csponds([SA[0.0, 0.0, 1.0]], [SA[0.0, 0.0, 1.0]]); kw...),
        :circle => (; kw...) -> refine_circle(Circle(Point2(0.0, 0.0), 1.0), pts(2); kw...),
        :ellipse => (; kw...) -> refine_ellipse(Ellipse(Point2(0.0, 0.0), 2.0, 1.0, 0.0), pts(4); kw...),
        :radial_homography => (; kw...) -> refine_radial_homography(RadialHomography{Float64}(H, 0.0), cs(4); kw...),
        :radial_camera => (; kw...) -> refine_radial_camera(RadialCamera(H, c), cs(4); kw...),
        :absolute_pose => (; kw...) -> refine_absolute_pose(pose, cam, Xs .=> us; kw...),
        :bearing_pose => (; kw...) -> refine_absolute_pose(pose, Csponds(Xs, [normalize(x) for x in Xs]); kw...),
        :hand_eye => (; kw...) -> refine_hand_eye(Xhe, hecs; kw...),
    )
    err(f) = try
        f(); nothing
    catch e
        e
    end
    refused = (:scoring => MsacScoring(HomographyProblem; threshold = 3.0), :threshold => 3.0, :outlier_halfwidth => 20.0,
               :loss => WelschShape(), :loss_width => 1.0, :efficiency_constant => 2.5,
               Symbol("scale" * "_multiple") => 2.5, :local_optimization => Triggs{WelschShape}(),
               :local_optimization_strategy => Triggs{WelschShape}(), :robust => true, :extent => nothing,
               :max_trials => 10, :min_trials => 1, :rng => StableRNG(1), :nonsense => 1)
    # the messages are the contract's, the same for every class: read them off the homography
    msg(k, v) = err(() -> cases[1][2](; scale = 1.0, (; k => v)...)).msg
    for (class, f) in cases
        for (k, v) in refused
            e = err(() -> f(; scale = 1.0, (; k => v)...))
            @test e isa ArgumentError && e.msg == msg(k, v)
        end
        # the default is Triggs{Welsch}: it needs exactly one of scale and width
        @test err(() -> f()) isa ArgumentError
        @test err(() -> f(; scale = 1.0, width = 1.0)) isa ArgumentError
        @test err(() -> f(; optimization_strategy = nothing, scale = 1.0)) isa ArgumentError
        @test err(() -> f(; optimization_strategy = Irls{VgtkCore.GaussianShape}(), scale = 1.0)) isa ArgumentError
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
        class === :hand_eye || @test retcode(r) === :degenerate && retcode(l2) === :degenerate
    end
    # the retired width multiple names its successor
    @test occursin("give `efficiency_constant`", msg(Symbol("scale" * "_multiple"), 2.5))
end
