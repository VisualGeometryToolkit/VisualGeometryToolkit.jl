# Closed-form rotation algorithms: Kabsch/Wahba and the chordal mean — the two
# algorithms Rotations.jl deliberately lacks (its mean.jl is a commented-out
# TODO). The SO(3) projection they share is Rotations.nearest_rotation,
# tested upstream and not re-tested here.

@testitem "rotation_fits" begin
    using Test
    using Random: MersenneTwister
    using StaticArrays
    using LinearAlgebra: norm, dot, normalize
    using Rotations: RotMatrix, RotationVec, RotXYZ, rotation_angle
    using VisualGeometryToolkitCore: EuclideanMap
    using FiniteDiff: FiniteDiff
    using VisualGeometryToolkitCore: kabsch, chordal_mean, withattrs, Csponds

    @testset "rotation fits" begin
        rng = MersenneTwister(20260806)

        @testset "kabsch recovery (R·a ≈ b, vectors as given)" begin
            Rgt = rand(rng, RotMatrix{3, Float64})
            as = [normalize(randn(rng, SVector{3, Float64})) for _ in 1:40]
            bs = [Rgt * a for a in as]
            _rms(R, as, bs) = sqrt(sum(sum(abs2, b - R * a) for (a, b) in zip(as, bs)) / length(as))
            # 1e-6, not 1e-12: the trace/acos angle formula floors at ~√eps ≈ 5e-8
            R = kabsch(as .=> bs)
            @test R isa RotMatrix{3, Float64}
            @test rotation_angle(RotMatrix{3}(R' * Rgt)) < 1.0e-6
            @test _rms(R, as, bs) < 1.0e-12                          # noise-free pairs
            @test kabsch(Csponds(as, bs)) == R                        # any cspond container

            # noisy directions still recover to ~noise level
            bsn = [normalize(b + 1.0e-3 * randn(rng, SVector{3, Float64})) for b in bs]
            Rn = kabsch(as .=> bsn)
            @test rotation_angle(RotMatrix{3}(Rn' * Rgt)) < 5.0e-3
            @test 1.0e-4 < _rms(Rn, as, bsn) < 1.0e-2

            # NOT centred: a common offset on both sides moves the answer
            o = SVector(3.0, -1.0, 2.0)
            @test rotation_angle(RotMatrix{3}(kabsch((as .+ Ref(o)) .=> (bs .+ Ref(o)))' * Rgt)) > 1.0e-3

            @test_throws ArgumentError kabsch(as[1:1] .=> bs[1:1])
            # a zero weight drops its pair: fewer than 2 of positive weight is refused, not an arbitrary rotation
            w1 = zeros(40); w1[7] = 1.0
            @test_throws ArgumentError kabsch(withattrs(Csponds(as, bs); weights = w1))
            w2 = zeros(40); w2[7] = 1.0; w2[11] = 2.0
            @test rotation_angle(RotMatrix{3}(kabsch(withattrs(Csponds(as, bs); weights = w2))' * Rgt)) < 1.0e-6
            @test_throws MethodError kabsch(as, bs)                   # correspondences only
        end

        @testset "chordal_mean" begin
            Rc = rand(rng, RotMatrix{3, Float64})
            # symmetric perturbations about Rc average back to ~Rc
            Rs = [
                RotMatrix(RotationVec((s * 0.05 .* SVector(1.0, -0.5, 0.3))...)) * Rc
                    for s in (-2, -1, 1, 2)
            ]
            m = chordal_mean(Rs)
            @test m isa RotMatrix{3, Float64}
            @test rotation_angle(RotMatrix{3}(m' * Rc)) < 1.0e-3
            @test chordal_mean(SMatrix{3, 3, Float64}.(Rs)) ≈ m      # plain matrices too
            # mean of one rotation is itself; empty set throws
            @test norm(chordal_mean([Rc]) - Rc) < 1.0e-12
            @test_throws ArgumentError chordal_mean(RotMatrix{3, Float64}[])
        end

        @testset "stationarity conditions: zero at the algorithm, the objective's gradient" begin
            using VisualGeometryToolkitCore: kabsch_stationarity, chordal_mean_stationarity, exp_so3
            as = [randn(rng, SVector{3, Float64}) for _ in 1:12]
            bs = [rand(rng, RotMatrix{3, Float64}) * a + 0.1 * randn(rng, SVector{3, Float64}) for a in as]   # noisy: a nonzero residual
            R = kabsch(as .=> bs)
            @test norm(kabsch_stationarity(R, as .=> bs)) < 1.0e-12 * sum(norm, bs)
            # away from it: the gradient of Σ bᵀ exp([ω]×) R₁ a at ω = 0, by central differences
            R1 = rand(rng, RotMatrix{3, Float64})
            g = kabsch_stationarity(R1, as .=> bs)
            obj1(ω) = sum(dot(b, exp_so3(ω) * R1 * a) for (a, b) in zip(as, bs))
            gfd = FiniteDiff.finite_difference_gradient(ω -> obj1(SVector{3}(ω)), zeros(3))   # exp_so3 takes an SVector
            @test gfd ≈ g rtol = 1.0e-6
            Rs = [RotMatrix(RotationVec((0.1 .* randn(rng, SVector{3, Float64}))...)) * R1 for _ in 1:5]
            @test norm(chordal_mean_stationarity(chordal_mean(Rs), Rs)) < 1.0e-12
            @test norm(chordal_mean_stationarity(R1 * RotMatrix(RotationVec(0.2, 0.0, 0.0)), Rs)) > 1.0e-3
        end
    end

    @testset "mirror (planar two-fold ambiguity)" begin
        using VisualGeometryToolkitCore: mirror
        pose = EuclideanMap(
            RotMatrix(RotXYZ(0.4, 0.1, -0.2)),
            SVector(0.05, -0.02, 2.0); check = false
        )
        c = SVector(0.0, 0.0, 0.0)
        m = mirror(pose, c)
        # same translation; separated by twice the viewing tilt
        @test m.t == pose.t
        n_c = pose.R * SVector(0.0, 0.0, 1.0)
        v̂ = normalize(pose.R * c + pose.t)
        tilt = acos(clamp(abs(dot(n_c, v̂)), 0, 1))
        @test isapprox(rotation_angle(RotMatrix{3}(m.R' * pose.R)), 2 * tilt; atol = 1.0e-8)
        # involution: mirror of the mirror is the original
        mm = mirror(m, c)
        @test rotation_angle(RotMatrix{3}(mm.R' * pose.R)) < 1.0e-8
        # frontal: normal parallel to the ray -> identity reflection
        frontal = EuclideanMap(
            one(RotMatrix{3, Float64}), SVector(0.0, 0.0, 2.0);
            check = false
        )
        mf = mirror(frontal, c)
        @test rotation_angle(RotMatrix{3}(mf.R' * frontal.R)) < 1.0e-10
    end
end

# kabsch with a `weights` attribute column (Wahba's weighted problem), against
# an SVD reference: unit weights are the unweighted fit, a zero weight drops its
# pair, invalid weights are refused.
@testitem "kabsch: a weights column" begin
    using Test, StaticArrays, LinearAlgebra, Random
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: kabsch, exp_so3, mm, Csponds, withattrs

    rng = MersenneTwister(11)
    Rt = exp_so3(SVector(0.2, 0.5, -0.3))
    as = [normalize(SVector(randn(rng), randn(rng), randn(rng))) for _ in 1:30]
    bs = [normalize(Rt * a + 0.01 * SVector(randn(rng), randn(rng), randn(rng))) for a in as]
    w = rand(rng, 30) .+ 0.1
    cs = Csponds(as, bs)
    # the weighted reference: R = U·diag(1, 1, det(UVᵀ))·Vᵀ of Σ wᵢ bᵢ aᵢᵀ
    B = sum(w[i] * (bs[i] * as[i]') for i in 1:30)
    F = svd(Matrix(B))
    Rref = F.U * Diagonal([1.0, 1.0, sign(det(F.U * F.Vt))]) * F.Vt
    @test SMatrix{3, 3}(kabsch(withattrs(cs; weights = w))) ≈ Rref atol = 1.0e-14
    @test kabsch(withattrs(cs; weights = ones(30))) ≈ kabsch(cs) atol = 1.0e-15        # unit weights: the unweighted fit
    # a gross outlier with zero weight is ignored
    bad = withattrs(Csponds(vcat(as, [SVector(1.0, 0.0, 0.0)]), vcat(bs, [SVector(0.0, 0.0, -1.0)])); weights = vcat(ones(30), 0.0))
    @test kabsch(bad) ≈ kabsch(cs) atol = 1.0e-14
    @test !(kabsch(withattrs(bad; weights = ones(31))) ≈ kabsch(cs))
    # invalid weights: the caller's error
    @test_throws ArgumentError kabsch(withattrs(cs; weights = vcat(-1.0, ones(29))))
    @test_throws ArgumentError kabsch(withattrs(cs; weights = vcat(NaN, ones(29))))
    # physical pairs keep their weights
    csm = withattrs(Csponds([a .* mm for a in as], [b .* mm for b in bs]); weights = w)
    @test SMatrix{3, 3}(kabsch(csm)) ≈ Rref atol = 1.0e-14
end
