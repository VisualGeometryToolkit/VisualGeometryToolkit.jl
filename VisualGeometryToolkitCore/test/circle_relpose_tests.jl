# #21 ACID TEST — the layering must admit a NEW solver family purely
# additively. This file registers a relative-pose-from-imaged-circles solver
# (two views, two coplanar circles, calibrated cameras: transfer +
# circularity → unique pose up to scale) WITHOUT touching a single line of
# Csponds / the sources-targets protocol / the solve verb / the registry
# machinery. Everything below is what a downstream package would write:
#
#   1. a solver struct   `CircleRelPose <: MinimalSolver{Extrinsics3}`
#   2. verb methods      `sample_size` / `max_solutions` / `solve`
#   3. ONE registry line  `minimal_solvers(::Type{<:Ellipse}, ::Type{<:Ellipse})`
#
# The numeric core is the geometric-init path (cone decomposition → shared
# plane normal → frame alignment), exact on noiseless input. The degree-4
# action-matrix template (HC.jl monodromy: 2 circles → unique pose) is
# future work; the LAYERING is what is under test here.

@testitem "circle_relpose" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using VisualGeometryToolkitCore: max_solutions, minimal_solvers, induced_models, minimal_solver
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: sample_size   # public, not exported
    using VisualGeometryToolkitCore: MinimalSolver, HomEllipseMat, EuclideanMap
    using StaticArrays, LinearAlgebra, Random
    using Rotations: RotMatrix, AngleAxis

    # =============================================================================
    # Forward model (synthesis): image conic of a 3D circle, normalized coords
    # =============================================================================
    # Circle on plane {p : n'p = d} with center c (n'c = d) and radius r, camera
    # at the origin; a ray x lies on the viewing cone iff x'Qx = 0 with
    #   Q = d²·I − d·(n c' + c n') + (|c|² − r²)·n n'
    _crp_conic(n, d, c, r) =
        d^2 * SMatrix{3, 3, Float64}(I) - d * (n * c' + c * n') +
        (dot(c, c) - r^2) * (n * n')

    # =============================================================================
    # The additive solver
    # =============================================================================

    struct CircleRelPose <: MinimalSolver{Extrinsics3} end
    VgtkCore.sample_size(::CircleRelPose) = 2
    VgtkCore.max_solutions(::CircleRelPose) = 1

    # --- cone decomposition ------------------------------------------------------

    # Candidate unit plane normals of the supporting plane of a circle imaged as
    # the conic Q (up to scale AND sign). Classical two-fold family from the
    # eigendecomposition; all sign combinations are emitted and filtered by the
    # reconstruction residual + cheirality downstream.
    function _crp_normal_candidates(Q::SMatrix{3, 3, Float64})
        E = eigen(Symmetric(Q))
        λ = E.values; V = E.vectors                    # ascending
        s = count(>(0), λ)
        (s == 1 || s == 2) || return SVector{3, Float64}[]
        if s == 1                                      # flip sign: pair positive
            λ = -λ
        end
        ord = sortperm(λ; rev = true)                  # λ₁ ≥ λ₂ > 0 > λ₃
        λ = λ[ord]
        (λ[1] >= λ[2] > 0 > λ[3]) || return SVector{3, Float64}[]
        v1 = SVector{3, Float64}(V[:, ord[1]]); v3 = SVector{3, Float64}(V[:, ord[3]])
        g = sqrt(max(λ[1] - λ[2], 0.0)); h = sqrt(max(λ[2] - λ[3], 0.0))
        return [
            normalize(g * v1 + h * v3), normalize(g * v1 - h * v3),
            normalize(-g * v1 + h * v3), normalize(-g * v1 - h * v3),
        ]
    end

    # Given (Q, candidate normal n) at plane gauge d = 1, recover center + radius
    # by matching Q's affine structure: for unit u ⊥ n the I-coefficient is
    # a = u'Qu; then b_⊥ = (Qn)_⊥ gives c = −b_⊥/a + n and r² = |c|² − γ/a.
    function _crp_circle_from_normal(Q::SMatrix{3, 3, Float64}, n::SVector{3, Float64})
        u = normalize(cross(n, abs(n[1]) < 0.9 ? SVector(1.0, 0, 0) : SVector(0, 1.0, 0)))
        a = dot(u, Q * u)
        abs(a) < 1.0e-12 && return nothing
        Qs = Q / a
        bperp = Qs * n - n * dot(n, Qs * n)
        c = -bperp + n                                 # c_⊥ = −b_⊥, c∥ = d·n = n
        b = bperp - n                                  # b∥ = −α·d·(c'n) = −1
        γ = dot(n, Qs * n) - 2 * dot(b, n) - 1.0
        r2 = dot(c, c) - γ
        r2 <= 1.0e-12 && return nothing
        Qr = SMatrix{3, 3, Float64}(I) - (n * c' + c * n') + (dot(c, c) - r2) * (n * n')
        res = norm(Qr - Qs) / max(norm(Qs), 1.0e-12)
        return (n = n, c = c, r = sqrt(r2), res = res)
    end

    # Per-view reconstruction from TWO coplanar circle images: candidates per
    # conic, keep front-side (c₃ > 0) residual-clean ones, pick the cross-circle
    # pair with the SAME plane normal (the spurious cone family disagrees between
    # circles — that is the circularity disambiguation), then gauge the view to
    # circle A's radius (r_A = 1).
    function _crp_view(eA::Ellipse, eB::Ellipse)
        QA = SMatrix{3, 3, Float64}(HomEllipseMat(eA))
        QB = SMatrix{3, 3, Float64}(HomEllipseMat(eB))
        ok(x) = x !== nothing && x.res < 1.0e-8 && x.c[3] > 0
        cA = [r for r in (_crp_circle_from_normal(QA, n) for n in _crp_normal_candidates(QA)) if ok(r)]
        cB = [r for r in (_crp_circle_from_normal(QB, n) for n in _crp_normal_candidates(QB)) if ok(r)]
        (isempty(cA) || isempty(cB)) && return nothing
        best = nothing
        for a in cA, b in cB
            Δ = norm(a.n - b.n)
            (best === nothing || Δ < best.Δ) && (best = (a = a, b = b, Δ = Δ))
        end
        best.Δ < 1.0e-6 || return nothing
        s = 1 / best.a.r
        return (n = best.a.n, cA = best.a.c * s, cB = best.b.c * s, rB = best.b.r * s)
    end

    # --- the solve verb ----------------------------------------------------------

    function VgtkCore.solve(::CircleRelPose, cs, idx::AbstractVector{<:Integer})
        fail = FixedModels{1, Extrinsics3}(0, (zero(Extrinsics3),))
        length(idx) == 2 || throw(ArgumentError("CircleRelPose takes a 2-circle sample"))
        u = sources(cs); v = targets(cs)
        @inbounds v1 = _crp_view(u[idx[1]], u[idx[2]])
        @inbounds v2 = _crp_view(v[idx[1]], v[idx[2]])
        (v1 === nothing || v2 === nothing) && return fail
        # In-plane frame from the inter-center direction; z = plane normal.
        δ1 = v1.cB - v1.cA; δ2 = v2.cB - v2.cA
        (norm(δ1) < 1.0e-9 || norm(δ2) < 1.0e-9) && return fail   # concentric
        frame(n, δ) = begin
            x = normalize(δ - n * dot(n, δ)); z = n; y = cross(z, x)
            SMatrix{3, 3, Float64}([x y z])
        end
        F1 = frame(v1.n, δ1); F2 = frame(v2.n, δ2)
        R = F2 * F1'
        t = v2.cA - R * v1.cA
        # Cross-circle consistency in the shared radius gauge validates the pick.
        norm(v2.cB - (R * v1.cB + t)) < 1.0e-6 || return fail
        abs(v1.rB - v2.rB) < 1.0e-6 || return fail
        pose = EuclideanMap(RotMatrix{3, Float64}(R), Vec3{Float64}(t))
        return FixedModels{1, Extrinsics3}(1, (pose,))
    end

    # --- the ONE registry line ---------------------------------------------------

    VgtkCore.minimal_solvers(::Type{<:Ellipse}, ::Type{<:Ellipse}) =
        (CircleRelPose(),)

    # =============================================================================
    # The acid test
    # =============================================================================

    @testset "acid test — circle relative pose is purely additive (#21)" begin
        rng = MersenneTwister(7)

        # ---- synthesize: two coplanar circles seen from two calibrated views
        n1 = normalize(SVector(0.15, -0.2, 1.0))       # plane normal, view 1
        d1 = 6.0                                       # plane distance, view 1
        r = 0.7                                        # circle radius (physical)
        u_ax = normalize(cross(n1, SVector(0.0, 0.0, 1.0)))
        cA1 = d1 * n1 + 1.9 * u_ax                     # centers on the plane
        cB1 = d1 * n1 - 1.4 * cross(n1, u_ax) - 0.8 * u_ax

        θ = 0.35                                       # relative pose (view1 → view2)
        axis = normalize(SVector(0.2, 1.0, 0.1))
        R_true = SMatrix{3, 3, Float64}(RotMatrix(AngleAxis(θ, axis[1], axis[2], axis[3])))
        t_true = SVector(0.6, -0.3, 0.9)

        n2 = R_true * n1
        d2 = d1 + dot(n2, t_true)
        cA2 = R_true * cA1 + t_true
        cB2 = R_true * cB1 + t_true

        to_ellipse(Q) = Ellipse(HomEllipseMat(Tuple(SMatrix{3, 3, Float64}(Q))))
        eA1 = to_ellipse(_crp_conic(n1, d1, cA1, r))
        eB1 = to_ellipse(_crp_conic(n1, d1, cB1, r))
        eA2 = to_ellipse(_crp_conic(n2, d2, cA2, r))
        eB2 = to_ellipse(_crp_conic(n2, d2, cB2, r))

        # ---- the correspondence container needed ZERO new code
        cs = Csponds([eA1, eB1], [eA2, eB2]; score = [1.0, 0.9])
        @test cs isa AbstractVector{<:Cspond{Ellipse{Float64}, Ellipse{Float64}}}

        # ---- registry: one method made the family discoverable
        @test induced_models(Ellipse{Float64}, Ellipse{Float64}) == (Extrinsics3,)
        @test induced_models(cs) == (Extrinsics3,)
        s = minimal_solver(Extrinsics3, Ellipse{Float64}, Ellipse{Float64})
        @test s isa CircleRelPose
        @test sample_size(s) == 2
        @test max_solutions(s) == 1
        @test model_type(s) == Extrinsics3

        # ---- solve honors the single verb contract
        sols = solve(s, cs, 1:2)
        @test sols isa FixedModels{1, Extrinsics3}
        @test length(sols) == 1
        pose = first(iterate(sols))

        # Rotation is exact; translation is recovered in units of circle A's
        # radius (the scale gauge of a monocular circle solver).
        @test norm(SMatrix{3, 3, Float64}(pose.R) - R_true) < 1.0e-6
        @test norm(pose.t - t_true / r) < 1.0e-6

        # ---- degeneracy honors the count-0 protocol (concentric circles)
        eCd = to_ellipse(_crp_conic(n1, d1, cA1, 0.4 * r))
        eCd2 = to_ellipse(_crp_conic(n2, d2, cA2, 0.4 * r))
        bad = solve(s, Csponds([eA1, eCd], [eA2, eCd2]), 1:2)
        @test bad isa FixedModels{1, Extrinsics3}
        @test length(bad) == 0

        # ---- attribute columns ride along like any other family
        @test hasattr(cs, :score)
        @test points(Csponds([eA1, eB1], [eA2, eB2]))[1] isa Pair  # centroid lift works on ellipses too
    end
end
