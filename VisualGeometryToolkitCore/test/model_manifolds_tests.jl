@testitem "model_manifolds" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using LinearAlgebra
    using StaticArrays
    using Rotations
    using ForwardDiff: ForwardDiff
    using FiniteDiff: FiniteDiff
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: LensManifold, SubsetManifold, action_jacobian, retract_jacobian, local_coordinates, recenter, restrict, retract, tangent_size   # public, not exported
    using VisualGeometryToolkitCore: model, transform_deriv_params, skew, parameter_vector, LeftRigidManifold6, transform_covariance
    using VisualGeometryToolkitCore: EssentialManifold6
    using Random: Random

    # =============================================================================
    # The manifold contract past the retraction: `action_jacobian` (the manifold's
    # action on a point or ray, differentiated — generic chain rule vs the closed
    # forms), `local_coordinates` (the inverse retraction of the rigid and lens
    # manifolds), `restrict` (the free/fixed choice as a manifold operation), the
    # lens manifold (`LensManifold`, including DoubleSphere's logit coordinate) and
    # `minimal_manifold` on poses and lenses.
    # =============================================================================

    # ∂(retract(ch, v)(x))/∂v at 0 by ForwardDiff through the retraction —
    # the reference every closed form is pinned against.
    _fd_action(ch, x, act) = ForwardDiff.jacobian(v -> act(VgtkCore.retract(ch, v), x), zero(SVector{tangent_size(ch), Float64}))

    @testset "Model manifolds — action, inverse, restriction, lens" begin
        R0 = RotMatrix{3, Float64}(RotXYZ(0.3, -0.2, 0.5))
        t0 = SVector(1.0, -2.0, 3.0)
        x = SVector(0.4, -0.7, 2.5)

        @testset "rigid manifold: the closed forms agree with the generic chain rule" begin
            ch = RigidManifold6(R0, t0)
            # ∂(R·x + t)/∂[vec R; t] against ForwardDiff over the 12 ambient numbers
            Tp = transform_deriv_params(model(ch), x)
            @test size(Tp) == (3, 12)
            Tfd = ForwardDiff.jacobian(zeros(12)) do a
                R = SMatrix{3, 3}(a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9])
                R * x + SVector(a[10], a[11], a[12])
            end
            @test Tp ≈ Tfd atol = 1.0e-14
            # the override, the generic default, and ForwardDiff through the action
            Jc = action_jacobian(ch, x)
            Jg = transform_deriv_params(model(ch), x) * retract_jacobian(ch)
            Jf = _fd_action(ch, x, (T, y) -> T(y))
            @test size(Jc) == (3, 6)
            @test Jc ≈ Jg atol = 1.0e-12
            @test Jc ≈ Jf atol = 1.0e-12
            @test Jc[:, 4:6] == SMatrix{3, 3, Float64}(I)
            @test Jc[:, 1:3] == -4 * skew(R0 * x)
            # invoke the generic path explicitly (what a manifold with no override gets)
            @test invoke(action_jacobian, Tuple{Manifold, Any}, ch, x) ≈ Jc atol = 1.0e-12
            # at chart coordinates v ≠ 0: the MRP left-Jacobian closed form, the generic
            # chain rule, ForwardDiff through the retraction; a restriction reads it too
            v = SVector(0.1, -0.25, 0.15, 0.4, -0.1, 0.25)
            Jv = action_jacobian(ch, x, v)
            @test Jv ≈ invoke(action_jacobian, Tuple{Manifold, Any, StaticVector}, ch, x, v) atol = 1.0e-12
            @test Jv ≈ ForwardDiff.jacobian(w -> retract(ch, w)(x), v) atol = 1.0e-12
            @test action_jacobian(ch, x, zero(v)) ≈ Jc atol = 1.0e-14
            rc = restrict(ch, SVector(true, false, true, false, false, true))
            @test action_jacobian(rc, x, SVector(v[1], v[3], v[6])) ≈ ForwardDiff.jacobian(w -> retract(rc, w)(x), SVector(v[1], v[3], v[6])) atol = 1.0e-12
        end

        @testset "rigid manifold: inverse retraction, storage types, minimal_manifold" begin
            ch = RigidManifold6(R0, t0)
            v = SVector(0.05, -0.02, 0.03, 0.4, -0.1, 0.25)
            target = retract(ch, v)
            @test local_coordinates(ch, target) ≈ v atol = 1.0e-12
            @test norm(local_coordinates(ch, model(ch))) == 0.0
            far = EuclideanMap(RotMatrix{3, Float64}(RotXYZ(2.0, -1.0, 0.7)), SVector(9.0, 9.0, 9.0))
            vf = local_coordinates(ch, far)
            back = retract(ch, vf)
            @test SMatrix{3, 3, Float64}(back.R) ≈ SMatrix{3, 3, Float64}(far.R) atol = 1.0e-12
            @test back.t ≈ far.t atol = 1.0e-12
            # re-anchoring keeps the anchor's storage types (a solver's variable type)
            chs = RigidManifold6(SMatrix{3, 3, Float64, 9}(R0), SVector{3, Float64}(t0))
            @test typeof(recenter(chs, v)) === typeof(chs)
            @test typeof(recenter(ch, v)) === typeof(ch)
            @test model(recenter(chs, v)).t ≈ target.t
            @test minimal_manifold(EuclideanMap(R0, t0)) isa RigidManifold6
            @test model(minimal_manifold(EuclideanMap(R0, t0))).t == t0
        end

        @testset "restrict: the parent's columns, the fixed coordinates at zero" begin
            ch = RigidManifold6(R0, t0)
            rc = restrict(ch, SVector(true, false, true, false, false, true))       # ω₁, ω₃, τ₃
            @test rc isa SubsetManifold && tangent_size(rc) == 3 && rc.idx == SVector(1, 3, 6)
            @test model(rc) === model(rc.manifold)
            v = SVector(0.02, -0.01, 0.3)
            lifted = SVector(0.02, 0.0, -0.01, 0.0, 0.0, 0.3)
            @test retract(rc, v).t == retract(ch, lifted).t
            @test SMatrix{3, 3, Float64}(retract(rc, v).R) == SMatrix{3, 3, Float64}(retract(ch, lifted).R)
            @test retract_jacobian(rc) == retract_jacobian(ch)[:, SVector(1, 3, 6)]
            @test action_jacobian(rc, x) == action_jacobian(ch, x)[:, SVector(1, 3, 6)]
            @test action_jacobian(rc, x) ≈ _fd_action(rc, x, (T, y) -> T(y)) atol = 1.0e-12
            rc2 = recenter(rc, v)
            @test rc2 isa SubsetManifold && rc2.idx == rc.idx && model(rc2).t == retract(rc, v).t
            @test local_coordinates(rc, retract(rc, v)) ≈ v atol = 1.0e-12
            # by index; the pinned manifold; the guards
            @test restrict(ch, SVector(2, 5)).idx == SVector(2, 5)
            pinned = restrict(ch, ())
            @test tangent_size(pinned) == 0 && model(pinned).t == t0
            @test size(action_jacobian(pinned, x)) == (3, 0)
            @test_throws ArgumentError restrict(ch, SVector(true, false))
            @test_throws ArgumentError restrict(ch, SVector(3, 1))
            @test_throws ArgumentError restrict(ch, SVector(1, 7))
        end

        @testset "lens manifold: the family's coordinates, the target's free coefficients" begin
            P = SVector(0.3, -0.2, 1.0)
            bc = Lens(Perspective(), Additive(Radial((-0.2, 0.05, 0.01)), Tangential(0.001, -0.002)))
            ch = LensManifold(bc)
            @test minimal_manifold(bc) isa LensManifold && model(ch) === bc
            @test tangent_size(ch) == 5                                 # k1–k3, p1, p2
            v = SVector(0.01, -0.02, 0.003, 0.0004, -0.0001)
            @test parameter_vector(retract(ch, v)) ≈ parameter_vector(bc) + v
            @test local_coordinates(ch, retract(ch, v)) ≈ v atol = 1.0e-15
            @test retract_jacobian(ch) == SMatrix{5, 5, Float64}(I)
            # the action on a ray: the lens's own parameter derivative, and ForwardDiff through the retraction
            Ja = action_jacobian(ch, P)
            @test size(Ja) == (2, 5)
            @test Ja ≈ transform_deriv_params(bc, P)
            @test Ja ≈ _fd_action(ch, P, (l, y) -> VgtkCore.project(l, y)) rtol = 1.0e-8
            # restriction by name is the target's complexity
            rc = restrict(ch, (:k1, :k2, :p1, :p2))
            @test rc.idx == SVector(1, 2, 4, 5) && tangent_size(rc) == 4
            @test action_jacobian(rc, P) == Ja[:, SVector(1, 2, 4, 5)]
            l2 = model(recenter(rc, SVector(0.1, 0.0, 0.0, 0.5)))
            c2, c = parameters(l2), parameters(bc)
            @test c2.k1 ≈ c.k1 + 0.1 && c2.p2 ≈ c.p2 + 0.5 && c2.k3 == c.k3
            @test local_coordinates(rc, l2) ≈ SVector(0.1, 0.0, 0.0, 0.5) atol = 1.0e-15
            @test_throws ArgumentError restrict(ch, (:k9,))
            @test_throws ArgumentError local_coordinates(ch, KannalaBrandt(0.0, 0.0, 0.0, 0.0))
            # a parameterless family: the manifold of nothing
            @test tangent_size(LensManifold(Pinhole())) == 0
            @test size(action_jacobian(LensManifold(Pinhole()), P)) == (2, 0)
        end

        @testset "lens manifold: DoubleSphere's ξ through a log, α ∈ [0, 1] mirrored at its ends" begin
            P = SVector(0.5, 0.3, 1.0)
            ds = Lens(DoubleSphere(-0.2, 0.6))
            ch = LensManifold(ds)
            @test tangent_size(ch) == 2
            l = retract(ch, SVector(0.1, 0.3))
            @test parameters(l).xi ≈ -1 + 0.8 * exp(0.1) && parameters(l).alpha == 0.6 + 0.3   # ξ > −1: ξ = −1 + (ξ₀ + 1)·exp(v); α additive
            @test local_coordinates(ch, l) ≈ SVector(0.1, 0.3) rtol = 1.0e-9
            a8 = parameters(retract(ch, SVector(0.0, 8.0))).alpha                                # a large step folds back inside
            @test 0 <= a8 <= 1
            @test a8 ≈ 0.6 atol = 1.0e-12                                                       # 0.6 + 8, period 2
            @test retract_jacobian(ch) ≈ ForwardDiff.jacobian(v -> parameter_vector(retract(ch, v)), zero(SVector{2, Float64})) atol = 1.0e-12
            @test action_jacobian(ch, P) ≈ _fd_action(ch, P, (m, y) -> VgtkCore.project(m, y)) rtol = 1.0e-8
            @test action_jacobian(restrict(ch, (:alpha,)), P) ≈ action_jacobian(ch, P)[:, SVector(2)]
        end

        @testset "lens manifold: a boundary α is a model, anchored on it with slope 1" begin
            P = SVector(0.5, 0.3, 1.0)
            ch0 = LensManifold(Lens(DoubleSphere(0.0, 0.0)))                                  # α = 0, the default
            @test parameters(model(ch0)).alpha === 0.0
            @test retract_jacobian(ch0)[2, 2] == 1                           # α moves at unit rate
            @test local_coordinates(ch0, Lens(DoubleSphere(0.0, 0.0))) == SVector(0.0, 0.0)
            @test parameters(retract(ch0, SVector(0.0, 0.2))).alpha == 0.2
            @test parameters(retract(ch0, SVector(0.0, -0.5))).alpha === 0.5                   # reflected at the closed end
            @test retract_jacobian(ch0, SVector(0.0, -0.5))[2, 2] == -1                        # the mirrored slope past it
            @test all(isfinite, action_jacobian(ch0, P))
            @test parameters(model(LensManifold(Lens(DoubleSphere(0.1, 1.0))))).alpha === 1.0
        end
    end

    # retract_jacobian(ch, v): the retraction differential at coordinates an
    # iterate-in-manifold optimizer holds, not only at the anchor. Every chart,
    # against central differences of its own retraction.
    @testset "retract_jacobian(ch, v) at v ≠ 0, every chart, against central differences" begin
        # central differences, not ForwardDiff: retract_jacobian is itself
        # ForwardDiff through the retraction, and the check must be independent
        cdiff(ch, v) = FiniteDiff.finite_difference_jacobian(w -> VgtkCore.ambient_coordinates(retract(ch, w)), v, Val(:central))
        R0 = RotMatrix{3, Float64}(RotXYZ(0.3, -0.2, 0.5)); t0 = normalize(SVector(1.0, -2.0, 3.0))
        F0 = SMatrix{3, 3, Float64}(1.0e-6, -2.0e-5, 3.0e-3, 4.0e-5, 1.0e-6, -2.0e-2, -1.0e-3, 2.0e-2, 1.0)
        H0 = SMatrix{3, 3, Float64}(1.1, 0.02, 1.0e-4, -0.03, 0.95, 2.0e-4, 12.0, -7.0, 1.0)
        charts = (
            EssentialManifold5(R0, Vec3(t0...)) => SVector(0.05, -0.03, 0.02, 0.1, -0.07),
            VgtkCore.EssentialManifold6(R0, Vec3(t0...)) => SVector(0.05, -0.03, 0.02, 0.1, -0.07, 0.04),
            FundMatManifold7(F0) => SVector(0.02, -0.01, 0.03, -0.02, 0.01, 0.015, 0.05),
            HomographyManifold8(H0) => SVector(0.01, -0.02, 0.005, 0.01, -1.0e-3, 2.0e-3, 0.03, -0.02),
            RigidManifold6(R0, SVector(1.0, -2.0, 3.0)) => SVector(0.05, -0.02, 0.03, 0.4, -0.1, 0.25),
            restrict(EssentialManifold5(R0, Vec3(t0...)), SVector(1, 3, 5)) => SVector(0.05, 0.02, -0.07),
            LensManifold(Lens(DoubleSphere(-0.2, 0.6))) => SVector(0.1, 0.7),
            LeftRigidManifold6(R0, SVector(1.0, -2.0, 3.0)) => SVector(0.05, -0.02, 0.03, 0.4, -0.1, 0.25),
        )
        for (ch, v) in charts
            J = retract_jacobian(ch, v)
            @test J ≈ cdiff(ch, v) rtol = 1.0e-6 atol = 1.0e-9
            @test retract_jacobian(ch, zero(v)) == retract_jacobian(ch)     # the anchor form, bit for bit
            # v ≠ 0 does move it, except on the first-order sl(3) chart (H(v) = H₀(I + Σvᵢgᵢ) is linear in v)
            ch isa HomographyManifold8 || @test !(J ≈ retract_jacobian(ch))
        end
    end

    # The left-perturbation pose chart: the perturbation constraint_with_jacobians(pose, …)
    # differentiates, so a PnP refit's Jacobian and its re-anchoring agree.
    @testset "RigidExpManifold6 and transform_covariance between the rigid charts" begin
        using VisualGeometryToolkitCore: RigidExpManifold6, transform_covariance
        R0 = RotMatrix{3, Float64}(RotXYZ(0.3, -0.2, 0.5)); t0 = SVector(1.0, -2.0, 3.0)
        ch = RigidExpManifold6(R0, t0)
        X = SVector(0.4, -0.7, 2.5)
        v = SVector(0.05, -0.02, 0.03, 0.4, -0.1, 0.25)
        P = retract(ch, v)
        @test P(X) ≈ VgtkCore.exp_so3(SVector(v[1], v[2], v[3])) * (R0 * X) + t0 + SVector(v[4], v[5], v[6]) atol = 1.0e-14
        @test local_coordinates(ch, P) ≈ v atol = 1.0e-13
        # the action Jacobian at the anchor and at v: closed form = generic = ForwardDiff
        @test action_jacobian(ch, X) ≈ transform_deriv_params(model(ch), X) * retract_jacobian(ch) atol = 1.0e-12
        @test action_jacobian(ch, X, v) ≈ ForwardDiff.jacobian(w -> retract(ch, w)(X), v) atol = 1.0e-12
        @test retract_jacobian(ch, v) ≈ ForwardDiff.jacobian(w -> (Q = retract(ch, w); vcat(vec(SMatrix{3, 3}(Q.R)), Q.t)), v) atol = 1.0e-14
        # transport: from the MRP chart, the rotation block scales by 4² at the anchor
        # (the MRP's differential is 4× the rotation vector's), the translation block is
        # untouched; a round trip is the identity (both charts full rank)
        mrp = RigidManifold6(R0, t0)
        Σ = Matrix(Diagonal([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        Σe = transform_covariance(mrp, ch, Σ)
        @test Σe[1:3, 1:3] ≈ 16 * Σ[1:3, 1:3] rtol = 1.0e-12
        @test Σe[4:6, 4:6] ≈ Σ[4:6, 4:6] rtol = 1.0e-12
        @test transform_covariance(ch, mrp, Σe) ≈ Σ rtol = 1.0e-12
        # to the left-perturbation chart: exp(δω)·t ≈ t − [t]×δω is carried by its δω, so
        # δt_left = δt + [t]×δω
        lft = LeftRigidManifold6(R0, t0)
        Σl = transform_covariance(ch, lft, Σ)
        Jel = vcat(hcat(Matrix(1.0I, 3, 3), zeros(3, 3)), hcat(Matrix(skew(t0)), Matrix(1.0I, 3, 3)))   # ∂(δω, δt)_left/∂(δω, δt)_exp
        @test Σl ≈ Jel * Σ * Jel' rtol = 1.0e-12
    end

    @testset "one re-anchoring policy: SO(3) re-projection, the unit-Frobenius homography" begin
        rng = Random.Xoshiro(4)
        R0 = SMatrix{3, 3, Float64}(RotMatrix{3, Float64}(RotXYZ(0.3, -0.2, 0.5))); t0 = SVector(1.0, -2.0, 3.0)
        for ch0 in (RigidManifold6(R0, t0), LeftRigidManifold6(R0, t0),
                    EssentialManifold5(RotMatrix{3, Float64}(R0), Vec3{Float64}(normalize(t0))),
                    EssentialManifold6(RotMatrix{3, Float64}(R0), Vec3{Float64}(normalize(t0))))
            ch = ch0
            for _ in 1:1000
                ch = recenter(ch, SVector{tangent_size(ch)}(1.0e-2 * randn(rng, tangent_size(ch))))
            end
            R = SMatrix{3, 3, Float64}(ch.R)
            @test maximum(abs, R' * R - I) < 1.0e-15        # a multiplied run drifts ~1e-14 without it
        end
        chH = recenter(HomographyManifold8(3 * SMatrix{3, 3, Float64}(R0)), SVector{8}(1.0e-2 * randn(rng, 8)))
        @test norm(chH.H0) ≈ 1 atol = 1.0e-15
    end

    @testset "pose_action_jacobian: the essential charts' pose acting on a point" begin
        # ∂(R(v)·x + t(v))/∂v against ForwardDiff through recenter_pose, at the
        # anchor and away from it, for both translation kinds; at the anchor the
        # rotation columns are RigidManifold6's (the same MRP increment).
        rng = Random.Xoshiro(12)
        R0 = RotMatrix{3, Float64}(RotXYZ(0.3, -0.2, 0.5)); t0 = Vec3{Float64}(normalize(SVector(1.0, -2.0, 3.0)))
        x = SVector(0.4, -1.1, 5.0)
        for ch in (EssentialManifold5(R0, t0), EssentialManifold6(R0, t0))
            D = tangent_size(ch)
            for v in (zero(SVector{D, Float64}), SVector{D}(0.07 * randn(rng, D)))
                Jp = VgtkCore.pose_action_jacobian(ch, x, v)
                @test Jp isa SMatrix{3, D, Float64}
                fd = ForwardDiff.jacobian(w -> ((R, t) = VgtkCore.recenter_pose(ch, w); SMatrix{3, 3}(R) * x + t), v)
                @test Jp ≈ fd atol = 1.0e-13
            end
            @test VgtkCore.pose_action_jacobian(ch, x)[:, SVector(1, 2, 3)] ≈ action_jacobian(RigidManifold6(R0, t0), x)[:, SVector(1, 2, 3)] atol = 1.0e-14
            # the translation columns are tangent to S² at t: the chart keeps ‖t‖ = 1
            @test norm(t0' * VgtkCore.pose_action_jacobian(ch, x)[:, SVector{D - 3}(4:D)]) < 1.0e-15
        end
    end

    @testset "LeftRigidManifold6: the chart of the pose constraint's left perturbation" begin
        R0 = RotMatrix{3, Float64}(RotXYZ(0.3, -0.2, 0.5)); t0 = SVector(1.0, -2.0, 3.0)
        ch = LeftRigidManifold6(R0, t0)
        X = SVector(0.4, -0.7, 2.5)
        # retract: the whole map perturbed on the left
        v = SVector(0.05, -0.02, 0.03, 0.4, -0.1, 0.25)
        P = retract(ch, v)
        ΔR = VgtkCore.exp_so3(SVector(v[1], v[2], v[3]))
        @test P(X) ≈ ΔR * (R0 * X + t0) + SVector(v[4], v[5], v[6]) atol = 1.0e-14
        @test retract(ch, zero(v)).t == t0 && SMatrix{3, 3}(retract(ch, zero(v)).R) == SMatrix{3, 3}(R0)
        # action Jacobian: the closed form, the generic chain rule, ForwardDiff through the retraction
        Jc = action_jacobian(ch, X)
        @test Jc == hcat(skew(-(R0 * X + t0)), SMatrix{3, 3, Float64}(I))
        @test Jc ≈ _fd_action(ch, X, (T, y) -> T(y)) atol = 1.0e-12
        @test Jc ≈ transform_deriv_params(model(ch), X) * retract_jacobian(ch) atol = 1.0e-12
        # ... and it is exactly the pose constraint's G_θ factor
        ray = SVector(0.1, -0.2, 1.0)
        _, _, Gθ = VgtkCore.constraint_with_jacobians(model(ch), X, ray)
        Jπ = VgtkCore.transform_deriv(PerspectiveMap(), R0 * X + t0)
        @test Gθ == Jπ * Jc
        # at chart coordinates v ≠ 0: the closed form, the generic chain rule and
        # ForwardDiff through the retraction agree; v = 0 is the anchor form
        Jv = action_jacobian(ch, X, v)
        @test Jv ≈ invoke(action_jacobian, Tuple{Manifold, Any, StaticVector}, ch, X, v) atol = 1.0e-12
        @test Jv ≈ ForwardDiff.jacobian(w -> retract(ch, w)(X), v) atol = 1.0e-12
        @test action_jacobian(ch, X, zero(v)) ≈ Jc atol = 1.0e-15
        # inverse retraction round trips, near and far
        @test local_coordinates(ch, retract(ch, v)) ≈ v atol = 1.0e-12
        far = EuclideanMap(RotMatrix{3, Float64}(RotXYZ(2.0, -1.0, 0.7)), SVector(9.0, 9.0, 9.0))
        back = retract(ch, local_coordinates(ch, far))
        @test SMatrix{3, 3}(back.R) ≈ SMatrix{3, 3}(far.R) atol = 1.0e-12
        @test back.t ≈ far.t atol = 1.0e-12
        @test local_coordinates(restrict(ch, SVector(4, 5, 6)), retract(ch, SVector(0, 0, 0, 0.4, -0.1, 0.25))) ≈ SVector(0.4, -0.1, 0.25) atol = 1.0e-14
        # recenter re-anchors at the retraction and keeps R on SO(3) over a long run
        @test model(recenter(ch, v)).t ≈ P.t && SMatrix{3, 3}(model(recenter(ch, v)).R) ≈ SMatrix{3, 3}(P.R)
        c = ch
        for k in 1:20_000
            w = 1.0e-3 * SVector(sin(k), cos(2k), sin(3k), cos(k), sin(5k), cos(7k))
            c = recenter(c, w)
        end
        @test EuclideanMap(RotMatrix{3}(model(c).R), model(c).t) isa EuclideanMap   # the checked constructor accepts it
        # covariance transport round trips (full rank, no gauge)
        Σ = Matrix(Diagonal([1.0e-4, 2.0e-4, 3.0e-4, 1.0e-2, 2.0e-2, 3.0e-2]))
        amb = VgtkCore.EuclideanManifold(ch)
        @test transform_covariance(amb, ch, transform_covariance(ch, amb, Σ)) ≈ Σ rtol = 1.0e-10
        # generic in the float type
        ch32 = LeftRigidManifold6(SMatrix{3, 3, Float32}(R0), Float32.(t0))
        P32 = retract(ch32, Float32.(v))
        @test eltype(P32.t) === Float32 && eltype(retract_jacobian(ch32)) === Float32
        @test P32.t ≈ Float32.(P.t) rtol = 1.0f-5
        @test local_coordinates(ch32, P32) ≈ Float32.(v) rtol = 1.0f-4
        @test LeftRigidManifold6(model(ch)).t == t0
    end
end

@testitem "chart_families" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test, LinearAlgebra, StaticArrays, Rotations, Random
    using FiniteDiff: FiniteDiff
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: RigidChart, MRPParam, ExpParam, FreeT, CarriedT, RigidManifold6,
        LeftRigidManifold6, RigidExpManifold6, EssentialManifold, EssentialManifold5, EssentialManifold6,
        HomographyManifold8, FundMatManifold7, tangent_size, zero_tangent, model, retract, recenter,
        retract_jacobian, action_jacobian, local_coordinates, transform_deriv_params, EuclideanMap,
        HomographyMat, FundMat, EssentialMat, Vec3

    rng = Xoshiro(11)
    R0 = rand(rng, RotMatrix{3, Float64}); t0 = SVector{3}(randn(rng, 3))
    x = SVector{3}(randn(rng, 3))
    # the three named pose charts ARE the family's members
    @test RigidManifold6(R0, t0) isa RigidChart{MRPParam, FreeT}
    @test LeftRigidManifold6(R0, t0) isa RigidChart{ExpParam, CarriedT}
    @test RigidExpManifold6(R0, t0) isa RigidChart{ExpParam, FreeT}
    @test EssentialManifold5(R0, normalize(t0)) isa EssentialManifold
    charts = (
        RigidManifold6(R0, t0), RigidManifold6(SMatrix{3, 3}(R0), t0), LeftRigidManifold6(R0, t0),
        RigidExpManifold6(R0, t0), RigidChart{MRPParam, CarriedT, typeof(R0), typeof(t0)}(R0, t0),
        EssentialManifold5(R0, normalize(t0)), EssentialManifold6(R0, normalize(t0)),
        HomographyManifold8(SMatrix{3, 3}(randn(rng, 3, 3))), FundMatManifold7(SMatrix{3, 3}(randn(rng, 3, 3))),
    )
    for ch in charts
        D = tangent_size(ch)
        v = SVector{D}(0.05 .* randn(rng, D))
        # model is the retraction at the anchor coordinates zero_tangent
        @test zero_tangent(ch) == zero(SVector{D})
        m, r0 = model(ch), retract(ch, zero_tangent(ch))
        @test VgtkCore.ambient_coordinates(m) ≈ VgtkCore.ambient_coordinates(r0) atol = 1.0e-15
        # the generic retraction differential against central differences
        f(w) = VgtkCore.ambient_coordinates(retract(ch, w))
        Tfd = FiniteDiff.finite_difference_jacobian(f, v, Val(:central))
        @test retract_jacobian(ch, v) ≈ Tfd rtol = 1.0e-7 atol = 1.0e-8
        ch isa RigidChart || continue
        # the rigid family's closed-form action Jacobians against the generic chain rule
        @test action_jacobian(ch, x) ≈ transform_deriv_params(model(ch), x) * retract_jacobian(ch) rtol = 1.0e-12
        @test action_jacobian(ch, x, v) ≈ transform_deriv_params(retract(ch, v), x) * retract_jacobian(ch, v) rtol = 1.0e-10
        # local_coordinates inverts retract; recenter lands on the retraction
        @test local_coordinates(ch, retract(ch, v)) ≈ v rtol = 1.0e-10
        @test SMatrix{3, 3}(model(recenter(ch, v)).R) ≈ SMatrix{3, 3}(retract(ch, v).R) rtol = 1.0e-12
    end
end

# A manifold whose model is a plain struct implements the protocol from outside:
# the model's `ambient_coordinates` method is all the defaults of
# `retract_jacobian` and `action_jacobian` need.
@testitem "manifold on a struct model" begin
    using Test, StaticArrays
    import VisualGeometryToolkitCore as VgtkCore
    @eval struct _Gain{T <: Real}       # generic in T: the default Jacobian retracts Duals
        g::T
        b::T
    end
    @eval VgtkCore.ambient_coordinates(m::_Gain) = SVector(m.g, m.b)
    @eval struct _GainManifold{T} <: VgtkCore.Manifold
        anchor::_Gain{T}
    end
    @eval VgtkCore.tangent_size(::_GainManifold) = 2
    @eval VgtkCore.retract(ch::_GainManifold, v) = _Gain(ch.anchor.g * exp(v[1]), ch.anchor.b + v[2])
    @eval VgtkCore.recenter(ch::_GainManifold, v) = _GainManifold(VgtkCore.retract(ch, v))
    ch = _GainManifold(_Gain(2.0, 0.5))
    @test VgtkCore.retract_jacobian(ch) ≈ [2.0 0.0; 0.0 1.0]
    @test VgtkCore.retract_jacobian(ch, SVector(log(1.5), 0.0)) ≈ [3.0 0.0; 0.0 1.0]
    @test VgtkCore.model(ch) == _Gain(2.0, 0.5)
end
