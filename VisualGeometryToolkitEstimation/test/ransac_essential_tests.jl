@testitem "ransac_essential" setup = [EpipolarScenes, Cameras] begin
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
using Test
using LinearAlgebra
using StaticArrays
using Random
using StableRNGs
using StructArrays
using Rotations: RotMatrix, MRP, RotY
using VisualGeometryToolkitCore
using VisualGeometryToolkitCore: EssentialManifold5, Vec3   # the 5-dof E refit chart (public)
using VisualGeometryToolkitCore: backproject_with_jacobian, withattrs, Csponds
using VisualGeometryToolkitCore: converged   # VGC accessor (public)
using VisualGeometryToolkitCore: MsacScoring, RansacScoring   # owned and exported by VisualGeometryToolkitCore
using VisualGeometryToolkitEstimation
using VisualGeometryToolkitEstimation: data_size, refine_workspace, refit_kind, NonlinLsq, _lm_state, residual_rows!

# These points are CALIBRATED and moved into the pixels of ECAM (EpipolarScenes):
# `EssentialProblem` needs a camera whose pixels are pixels, so the gate's PIXEL
# tolerance stays meaningful.

# The weighted Sampson cost of E over calibrated points — F at identity intrinsics.
_sampson_cost(E, u₁, u₂, w) =
    sum(w[i] * sampson_distance(FundMat(E, SMatrix{3, 3, Float64}(I)), u₁[i], u₂[i])^2 for i in eachindex(u₁))

@testset "Essential LO (NonlinLsq pose-LM)" begin
    # Known relative pose → true essential matrix (calibrated points, K = I).
    R = RotY(0.2); t = normalize(SA[0.8, 0.1, 0.3])
    u₁, u₂ = calibrated_pairs(StableRNG(7), R, t; n = 80, σ = 0.002)
    n = length(u₁)
    p = ecs(u₁, u₂)

    @test refit_kind(p) == NonlinLsq()

    # Perturbed starting model (perturb the pose, recompose E).
    E_pert = perturbed_essential(R, t, 0.08, SA[0.1, -0.08, 0.05])

    w = ones(n)
    E_ref = VgtkEstimation.refine!(refine_workspace(typeof(p), data_size(p)), p, E_pert, w)

    @test E_ref !== nothing
    @test E_ref isa EssentialMat
    c0 = _sampson_cost(E_pert, u₁, u₂, w)
    c1 = _sampson_cost(E_ref, u₁, u₂, w)
    @test c1 < c0           # pose-LM refit reduced the weighted Sampson cost
    @test c1 < 0.5 * c0     # ...substantially (perturbation pulled back toward truth)
end

@testset "LM rows and scoring residuals are the same function (assembly canary)" begin
    # THE DEFECT THIS EXISTS TO CATCH. During the bearing port `residuals!` was
    # moved to the tangent Sampson while the LM state's rows still computed
    # the image-plane form on the same (now bearing) points. They disagreed by
    # a median ratio of 1.11 and a MAXIMUM of 446x — a different function, not
    # a scale factor — so the optimiser descended something unrelated to what
    # was scored: cost flat-to-rising while reporting `:converged`, and a
    # 3.4-degree seed landing at 26 degrees. Nothing in the suite noticed.
    #
    # Both paths now call ONE implementation (VGC's tangent rows), so agreement
    # is structural. This pins the ASSEMBLY rather than the maths: the LM state
    # copies `M₁`/`M₂` out of the problem, and nothing structural forces those
    # to be the columns the scoring path reads. A shared implementation fed
    # different columns is still two answers. So the state is built by THE
    # production wiring, `_lm_state(p, chart, sqrtw)` — the one method
    # every refit and lift builds its state through — never assembled here by
    # hand, which could only ever check the test's own wiring.
    #
    # TWO DIFFERENT FOCALS, deliberately: a per-view M swap moves the residual
    # by 1.8% — 0.19% at equal focals, inside anything anyone would call noise
    # — while moving the Jacobian by 341%. At one focal this canary could not
    # detect the swap it is here to detect.
    rng = StableRNG(4242)
    camA = CameraModel(800.0px, SA[320.0px, 240.0px])
    camB = CameraModel(1150.0px, SA[300.0px, 260.0px])
    m = 30
    pa = [SA[320 + 220 * (rand(rng) - 0.5), 240 + 170 * (rand(rng) - 0.5)] for _ in 1:m]
    pb = [SA[300 + 220 * (rand(rng) - 0.5), 260 + 170 * (rand(rng) - 0.5)] for _ in 1:m]
    p = EssentialProblem(camA, camB, csponds(Point2{Float64}.(pa), Point2{Float64}.(pb)))

    R = RotY(0.18)
    t = normalize(SA[0.8, 0.1, 0.25])
    θ = EssentialMat(R, t)

    scored = Vector{Float64}(undef, m)
    VgtkEstimation.residuals!(scored, p, θ)

    ch = EssentialManifold5(RotMatrix{3, Float64}(R), Vec3{Float64}(t))
    rows = residual_rows!(Vector{Float64}(undef, m), _lm_state(p, ch, ones(m)), VgtkCore.zero_tangent(ch))

    # Magnitudes, because the rows are SIGNED and the scored residual is not.
    @test maximum(abs.(abs.(rows) .- abs.(scored))) < 1.0e-12
end

@testset "Essential Jacobian (analytic vs ForwardDiff)" begin
    rng = StableRNG(11)
    R₀ = RotMatrix(MRP(0.1, -0.05, 0.2))
    t₀ = normalize(SA[0.5, -0.3, 0.8])
    m = 25
    # BEARINGS with their per-view Jacobians, and TWO DIFFERENT FOCALS on
    # purpose: a swapped M₁/M₂ moves the residual by only 1.8% (0.19% at equal
    # focals — invisible) but the JACOBIAN by 341%. Since this testset checks
    # the analytic Jacobian, equal focals would make the one error it is best
    # placed to catch undetectable.
    camJ1 = CameraModel(800.0px, SA[320.0px, 240.0px])
    camJ2 = CameraModel(1150.0px, SA[300.0px, 260.0px])
    fm1 = [backproject_with_jacobian(camJ1, SA[320 + 200 * randn(rng), 240 + 150 * randn(rng)]) for _ in 1:m]
    fm2 = [backproject_with_jacobian(camJ2, SA[300 + 200 * randn(rng), 260 + 150 * randn(rng)]) for _ in 1:m]
    u₁ = first.(fm1); u₂ = first.(fm2)
    M₁ = last.(fm1); M₂ = last.(fm2)
    sqrtw = sqrt.(rand(rng, m))
    x = SVector{5, Float64}(0.03, -0.02, 0.05, 0.1, -0.05)  # general (nonzero) iterate

    p = EssentialProblem(withattrs(Csponds(u₁, u₂); M₁, M₂))
    ch = EssentialManifold5(RotMatrix{3, Float64}(R₀), Vec3{Float64}(t₀))   # the 5-dof refit chart (VGC)
    @test VgtkCore.check_rows_jacobian(_lm_state(p, ch, sqrtw), x, m; atol = 1.0e-8)
end

@testset "Measurement-map frame labels (VGC EpipolarForm adoption)" begin
    rng = StableRNG(23)
    u₁ = [Point2{Float64}(0.3 * randn(rng), 0.3 * randn(rng)) for _ in 1:12]
    u₂ = [Point2{Float64}(0.3 * randn(rng), 0.3 * randn(rng)) for _ in 1:12]
    cs = StructArray(u₁ .=> u₂)
    # Non-trivial invertible data→bearing measurement maps.
    C1 = SMatrix{3, 3, Float64}([1.3 0.1 0.02; 0.0 0.9 -0.03; 0.0 0.0 1.0])
    C2 = SMatrix{3, 3, Float64}([0.8 0.0 -0.01; 0.05 1.1 0.04; 0.0 0.0 1.0])
    # Labeled lifts: data frame (caller-dependent ⇒ unlabeled) → per-view
    # bearing frame, the essential model's metric frame.
    H1 = HomographyMap{UnlabeledFrame{2}, GnomonicFrame{:src}}(C1)
    H2 = HomographyMap{UnlabeledFrame{2}, GnomonicFrame{:dst}}(C2)

    # A valid essential matrix for the residual comparisons.
    θ = EssentialMat(RotY(0.15), normalize(SA[0.6, -0.2, 0.4]))

    # E TAKES NO MEASUREMENT MAPS ANY MORE, and this asserts the refusal.
    #
    # The maps existed to express K⁻¹, and a map is a MATRIX — so the best it
    # can produce is a point on the z = 1 plane. Measured: the old lift returned
    # z ≡ 1 with norms 1.0012–1.0185 on a narrow field, against exactly 1.0 from
    # a real back-projection, and BOTH satisfy a `StaticVector{3}` signature. The
    # chart survived the lift having only stopped being visible in the type,
    # which is precisely the defect this model class exists to remove: a ray at
    # 90° has no z = 1 representation at all.
    #
    # So the frame-label junction tests that used to live here are gone with the
    # feature. What replaces them is stronger — bearings now come from the
    # camera's own back-projection, where the frame cannot be misdeclared
    # because it is never declared separately from the camera that defines it.
    # NOTE which refusal fires. With 2-vector data the TYPE check rejects first
    # (a MethodError), so supplying maps alongside it would test the wrong thing.
    # The map refusal has to be exercised on data that would otherwise be
    # accepted — genuine bearings with their columns.
    @test_throws MethodError EssentialProblem(cs; src_map = H1, dst_map = H2)
    fm = [backproject_with_jacobian(ECAM, topx(u)) for u in u₁]
    gm = [backproject_with_jacobian(ECAM, topx(u)) for u in u₂]
    cs_ok = withattrs(
        Csponds(first.(fm), first.(gm)); M₁ = last.(fm), M₂ = last.(gm)
    )
    @test EssentialProblem(cs_ok) isa VgtkEstimation.EssentialProblem
    @test_throws ArgumentError EssentialProblem(cs_ok; src_map = C1, dst_map = C2)

    # And a z = 1 chart point offered directly as a 3-vector is refused by the
    # unit-norm check, which is the guard the signature cannot provide.
    chartpts = [Point3{Float64}(u[1], u[2], 1.0) for u in u₁]   # z = 1 chart, not bearings
    Mcol = [backproject_with_jacobian(ECAM, topx(u))[2] for u in u₁]
    cs_chart = withattrs(Csponds(chartpts, chartpts); M₁ = Mcol, M₂ = Mcol)
    @test_throws ArgumentError EssentialProblem(cs_chart)

    # The camera form is the supported path, and it scores finitely.
    p0 = ecs(u₁, u₂)
    @test !VgtkEstimation._has_measurement_maps(p0)
    n = length(u₁)
    r0 = Vector{Float64}(undef, n)
    VgtkEstimation.residuals!(r0, p0, θ)
    @test all(isfinite, r0)

    # H declares per-view PIXEL frames (ImageFrame) at the same boundary; F,
    # like E, takes no maps at all and refuses them at construction.
    cs7 = StructArray(vcat(u₁, u₁[1:2]) .=> vcat(u₂, u₂[1:2]))
    V1 = HomographyMap{UnlabeledFrame{2}, ImageFrame{:src}}(C1)
    V2 = HomographyMap{UnlabeledFrame{2}, ImageFrame{:dst}}(C2)
    @test_throws ArgumentError FundMatProblem(cs7; src_map = V1, dst_map = V2)
    ph = HomographyProblem(cs7; src_map = V1, dst_map = V2)
    @test VgtkEstimation._has_measurement_maps(ph)
    @test_throws MethodError HomographyProblem(cs7; src_map = V2, dst_map = V1)
end

# A two-camera pixel scene: true relative pose, `n_in` noisy inliers and
# `n_out` uniform outliers, in the pixels of `cam₁`/`cam₂`.
function _e_pixel_scene(rng, cam₁, cam₂; n_in = 150, n_out = 60, σ = 0.5, T = Float64)
    R = RotY(0.2); t = normalize(SA[0.8, 0.1, 0.3])
    c₁ = SA[960.0, 540.0]; c₂ = SA[900.0, 560.0]
    a, b = stereo_pixel_pairs(
        rng, cam₁, cam₂, R, t; n = n_in, σ₁ = σ, σ₂ = σ, point = box_point, n_out,
        outlier = rng -> (
            c₁ + SA[800 * (rand(rng) - 0.5), 500 * (rand(rng) - 0.5)],
            c₂ + SA[800 * (rand(rng) - 0.5), 500 * (rand(rng) - 0.5)],
        )
    )
    return Point2{T}.(a), Point2{T}.(b), R, t
end

# The owner's bearings and back-projection Jacobians (the camera form of
# `EssentialProblem`), STORED at other element types: bearings and `M₁` in T₁,
# `M₂` in T₂ (re-normalized in T₁, as a narrower store would).
function _retyped(p; T₁ = Float64, T₂ = T₁)
    cs = p.cs
    return EssentialProblem(
        withattrs(
            Csponds(normalize.(SVector{3, T₁}.(sources(cs))), normalize.(SVector{3, T₁}.(targets(cs))));
            M₁ = SMatrix{3, 2, T₁, 6}.(cs.M₁), M₂ = SMatrix{3, 2, T₂, 6}.(cs.M₂)
        )
    )
end

@testset "Float32 bearings score and fit (T1); mixed Jacobian types refit (T2)" begin
    # A Float32 problem at f = 2000 px scored every true inlier `Inf` — the
    # tangent-Sampson degeneracy test ran in the model's eltype on a gradient
    # of ~1/f — and RANSAC returned `:no_model`. Scoring is now widened to
    # Float64 per point.
    cam1 = CameraModel(2000.0px, SA[960.0px, 540.0px])
    cam2 = CameraModel(2000.0px, SA[900.0px, 560.0px])
    a, b, R, t = _e_pixel_scene(StableRNG(2031), cam1, cam2)
    p32 = _retyped(EssentialProblem(cam1, cam2, csponds(a, b)); T₁ = Float32)
    @test eltype(eltype(sources(p32.cs))) == Float32     # the regime under test
    θ = EssentialMat(R, t)
    r = Vector{Float64}(undef, length(a))
    VgtkEstimation.residuals!(r, p32, θ)
    @test all(isfinite, r[1:150])
    @test count(<(3.0), r[1:150]) > 140              # true inliers score as inliers
    res = VgtkEstimation.ransac(p32, MsacScoring(p32; threshold = 3.0); rng = StableRNG(7))
    @test converged(res)
    @test count(res.inlier_mask[1:150]) > 0.9 * 150

    # T2: the two Jacobian columns of DIFFERENT types (here Float64 and
    # Float32); the LM state took one type parameter for both, a MethodError
    # at the first local optimization. The refit now runs.
    camA = CameraModel(800.0px, SA[960.0px, 540.0px])
    camB = CameraModel(1150.0px, SA[900.0px, 560.0px])
    a2, b2, R2, t2 = _e_pixel_scene(StableRNG(2032), camA, camB; n_out = 0)
    pm = _retyped(EssentialProblem(camA, camB, csponds(a2, b2)); T₂ = Float32)
    @test eltype(pm.cs.M₁) != eltype(pm.cs.M₂)
    θ₀ = EssentialMat(R2, normalize(t2 + SA[0.05, -0.04, 0.03]))
    θ₁ = VgtkEstimation.refine!(refine_workspace(typeof(pm), data_size(pm)), pm, θ₀, ones(length(a2)))
    @test θ₁ !== nothing
end

@testset "the camera form keeps the :score column (PROSAC) (N1)" begin
    rng = StableRNG(2033)
    cam = CameraModel(800.0px, SA[960.0px, 540.0px])
    a, b, _, _ = _e_pixel_scene(rng, cam, cam; n_out = 20)
    s = rand(rng, length(a))
    scored = Csponds(a, b; score = s)
    ph = HomographyProblem(scored)
    pe = EssentialProblem(cam, cam, scored)
    # the same sampler H and F build from the same column
    @test VgtkEstimation.sampler(ph) isa VgtkEstimation.ProsacSampler
    @test VgtkEstimation.sampler(pe) isa VgtkEstimation.ProsacSampler
    @test pe.cs.score == s
    @test VgtkEstimation.sampler(EssentialProblem(cam, cam, Csponds(a, b))) isa
        VgtkEstimation.UniformSampler
    # and a scored pipeline run accepts it
    @test find_essential(cam, cam, scored; rng = StableRNG(8)) !== nothing
end

@testset "non-finite bearings and Jacobians are refused (N2)" begin
    cam = CameraModel(800.0px, SA[320.0px, 240.0px])
    fm = [backproject_with_jacobian(cam, SA[320.0 + 10i, 240.0 - 7i]) for i in 1:8]
    gm = [backproject_with_jacobian(cam, SA[300.0 + 9i, 250.0 - 5i]) for i in 1:8]
    f₁ = first.(fm); f₂ = first.(gm); M₁ = last.(fm); M₂ = last.(gm)
    @test EssentialProblem(withattrs(Csponds(f₁, f₂); M₁, M₂)) isa VgtkEstimation.EssentialProblem
    # A NaN bearing (what a camera returns past a fisheye's fold): `NaN > worst`
    # is false, so the unit-norm test alone let it in.
    f₁n = copy(f₁); f₁n[3] = eltype(f₁)(NaN, NaN, NaN)
    err = try
        EssentialProblem(withattrs(Csponds(f₁n, f₂); M₁, M₂)); nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("index 3", err.msg)
    M₂n = copy(M₂); M₂n[5] = M₂n[5] * NaN
    @test_throws ArgumentError EssentialProblem(withattrs(Csponds(f₁, f₂); M₁, M₂ = M₂n))
end

@testset "sub-detector threshold refused on every entry (V17)" begin
    rng = StableRNG(2034)
    cam = CameraModel(800.0px, SA[960.0px, 540.0px])
    a, b, R, t = _e_pixel_scene(rng, cam, cam)
    p = EssentialProblem(cam, cam, csponds(a, b))
    θ = EssentialMat(R, t)
    lo = VgtkEstimation.NoLocalOptimization()
    # 0.02 is the OLD calibrated default; in pixels it is nothing — refused as
    # the default MSAC's threshold and as a caller's threshold scorer's.
    @test_throws ArgumentError find_essential(cam, cam, csponds(a, b); threshold = 0.02)
    @test_throws ArgumentError find_essential(cam, cam, csponds(a, b); scoring = MsacScoring(p; threshold = 0.02))
    @test_throws ArgumentError VgtkEstimation.ransac(p, MsacScoring(p; threshold = 0.02))
    @test_throws ArgumentError VgtkEstimation.ransac(p, RansacScoring(p; threshold = 0.02))
    @test_throws ArgumentError locally_optimize(p, θ; threshold = 0.02, local_optimization = lo)
    @test_throws ArgumentError locally_optimize(p, θ; scoring = MsacScoring(p; threshold = 0.02), local_optimization = lo)
    err = try
        VgtkEstimation.ransac(p, MsacScoring(p; threshold = 0.02)); nothing
    catch e
        e
    end
    @test err isa ArgumentError && occursin("threshold 0.02 is below", err.msg)
    # A pixel value is accepted on the same entries — the class default (1 px) included.
    @test locally_optimize(p, θ; threshold = 1.0, local_optimization = lo) !== nothing
    @test converged(find_essential(cam, cam, csponds(a, b); rng = StableRNG(2035)))
    # A unit-focal camera's "pixel" IS a calibrated unit, so a calibrated
    # threshold is right there and is not refused.
    pc = EssentialProblem(IDCAM, IDCAM, csponds(calibrated_point.(Ref(cam), a), calibrated_point.(Ref(cam), b)))
    @test locally_optimize(pc, θ; threshold = 0.002, local_optimization = lo) !== nothing
end
end
