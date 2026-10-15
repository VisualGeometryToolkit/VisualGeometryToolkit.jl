# =============================================================================
# Loss-shape math (robust/losses.jl): the GNC surrogate facet, the shape-math
# contract, the Geman–Reynolds lift identities, Barron's dial, the z-block, and
# the chart manifolds' retraction/covariance pins. The LO continuation
# schedules that once shared this item live in LocalOptimizationDoneRight.
# =============================================================================
@testitem "loss_shape_math" begin
    using Test
    using VisualGeometryCore
    using VisualGeometryCore: BandAnnealed, ConvexityAnnealed, EssentialManifold6, NoSurrogate, WidthAnnealed, retract_jacobian, anneal, codimension_factor, retract, s2_tangent_basis, tangent_size, target_loss   # public, not exported
    using VisualGeometryCore: gnc_entry_mu0
    using VisualGeometryCore: EuclideanManifold
    # Explicit: CoordinateTransformations ALSO exports `recenter`; the
    # Manifold testset means VGC's chart retraction.
    using VisualGeometryCore: recenter
    # SHAPE MATH: these take an explicit c² and the shapes are fieldless tags,
    # so these are the shapes themselves — no placeholder width needed.
    #
    # `ψ` is the INFLUENCE FUNCTION ρ′(r); `conjugate` is the half-quadratic
    # penalty the Geman–Reynolds literature ALSO calls ψ. Two psis, and the
    # symbol goes to the M-estimation one, which is what the efficiency
    # constants are solved against.
    #
    # All `public` but unexported — `weight`/`cost` are far too generic to
    # export, and GeometryBasics and IntervalSets both export `width`.
    using VisualGeometryCore: ρ, ψ, weight, conjugate, zblock,
        cost, cost_scale, sq_loss_width
    const GM, WEL, CAU = GemanMcclureShape(), WelschShape(), CauchyShape()
    const TLS_, STLS, HUB = TLSShape(), SmoothTLSShape(), HuberShape()
    barron(α) = BarronShape(α)
    using LinearAlgebra: Diagonal, I, dot, norm, normalize, nullspace, pinv, rank, svd
    using Random: MersenneTwister
    using Rotations: Rotations, RotMatrix
    using StaticArrays: SMatrix, SVector
    using FiniteDiff: FiniteDiff

    @testset "GNC surrogate facet" begin
        cbar2 = 2.3
        @testset "TLS closed forms" begin
            for μ in (0.05, 0.5, 1.0, 7.0)
                lo = (μ / (μ + 1.0)) * cbar2
                hi = ((μ + 1.0) / μ) * cbar2
                @test gnc_tls_weight(0.9 * lo, cbar2, μ) == 1.0
                @test gnc_tls_weight(1.1 * hi, cbar2, μ) == 0.0
                # continuity at the band edges
                @test gnc_tls_weight(lo, cbar2, μ) ≈ 1.0 atol = 1.0e-9
                @test isapprox(sqrt(cbar2 * μ * (μ + 1.0)) / sqrt(hi) - μ, 0.0; atol = 1.0e-12)
                # middle band strictly interior
                mid = 0.5 * (lo + hi)
                w = gnc_tls_weight(mid, cbar2, μ)
                @test 0.0 < w < 1.0
            end
            # The TLS TARGET loss is just the shape's own ρ — `gnc_tls_target`
            # was retired because it was `min(x, c̄²)` written a second time.
            @test ρ(TLS_, 0.5 * cbar2, cbar2) == 0.5 * cbar2
            @test ρ(TLS_, 3.0 * cbar2, cbar2) == cbar2
        end
        @testset "GM/Welsch GNC surrogates ARE lift_weight at μ·c̄² (identity pin)" begin
            for μ in (1.0, 2.5, 40.0), x in (0.0, 0.3, 2.3, 11.0)
                @test weight(GM, x, μ * cbar2) ≈ (μ * cbar2 / (x + μ * cbar2))^2
                @test weight(WEL, x, μ * cbar2) == exp(-x / (μ * cbar2))
            end
        end

        # The trait exists to separate three graduation families, and until now
        # only the Welsch branch was ever executed -- every `Gnc{...}` in every
        # suite across the stack is `Gnc{Irls{WelschShape}}`. So `gnc_*` is
        # pinned here against the closed forms of the Symbol vocabulary it
        # replaced, EXACTLY (`==`), for all three.
        @testset "gnc_* dispatch on GncSurrogate == the retired Symbol forms" begin
            # `scale_multiple = 1` so sq_loss_width IS cbar2, which is what the old API took.
            kW = deploy(WEL, cbar2; scale_multiple = 1.0)
            kG = deploy(GM, cbar2; scale_multiple = 1.0)
            kT = deploy(TLS_, cbar2; scale_multiple = 1.0)
            @test gnc_surrogate(WEL) isa WidthAnnealed
            @test gnc_surrogate(GM) isa ConvexityAnnealed
            @test gnc_surrogate(TLS_) isa BandAnnealed
            for μ in (1.0, 2.5, 40.0), x in (0.0, 0.3, 2.3, 11.0)
                @test gnc_weight(kW, x, μ) == weight(WEL, x, μ * cbar2)
                @test gnc_weight(kG, x, μ) == weight(GM, x, μ * cbar2)
                @test gnc_weight(kT, x, μ) == gnc_tls_weight(x, cbar2, μ)
            end
            for x in (0.0, 0.3, 2.3, 11.0)
                # the historical ×½ on the Welsch tracker, and NOT on GM
                @test gnc_target(kW, x) == 0.5 * ρ(WEL, x, cbar2)
                @test gnc_target(kG, x) == ρ(GM, x, cbar2)
                @test gnc_target(kT, x) == ρ(TLS_, x, cbar2)
            end
            for r2max in (0.5 * cbar2, 3.0, 40.0)
                @test gnc_mu0(kW, r2max) == max(r2max / cbar2, 1.0)
                @test gnc_mu0(kG, r2max) == max(2.0 * r2max / cbar2, 1.0)
                @test gnc_mu0(kT, r2max) ==
                    (r2max <= cbar2 / 2 ? 1.0e6 : cbar2 / (2.0 * r2max - cbar2))
            end
            # The entry-breadth policy: a numeric entry m starts the WIDTH family at
            # m× the target width (μ₀ = max(m², 1), m ≤ 1 the static limit); NaN is
            # the r_max seed extreme; a convexity/band family has no width to be a
            # multiple of, so it has no method.
            @test gnc_entry_mu0(kW, 100.0, 4.0) == 16.0
            @test gnc_entry_mu0(kW, 100.0, 0.5) == 1.0
            @test gnc_entry_mu0(kW, 100.0, NaN) == gnc_mu0(kW, 100.0)
            @test_throws MethodError gnc_entry_mu0(kT, 1.0, 4.0)
            @test_throws MethodError gnc_entry_mu0(kG, 1.0, 4.0)
        end

        # A shape with no graduation path must REFUSE a μ, not absorb it: an
        # inert homotopy that reports convergence is the failure this trait
        # replaced. (Regression for the adversarial-review finding.)
        @testset "a NoSurrogate shape refuses to anneal" begin
            kC = deploy(CAU, cbar2; scale_multiple = 1.0)
            @test gnc_surrogate(CAU) isa NoSurrogate
            @test_throws MethodError sq_loss_width(anneal(kC, 10.0))
            @test_throws MethodError lift_weight(anneal(kC, 10.0), 1.0)
            # ...while a graduating shape anneals exactly
            @test sq_loss_width(anneal(deploy(WEL, cbar2; scale_multiple = 1.0), 2.5)) === 2.5 * cbar2
        end

        # =====================================================================
        # Shape-math CONTRACT. `ρ`/`ψ`/`weight` are a public extension point
        # (8caba4f), so the properties a downstream shape must satisfy are
        # pinned HERE rather than in whichever consumer happened to notice.
        # =====================================================================
        @testset "shape math: totality, ψ = r·w, w = dρ/ds" begin
            shapes = (GaussianShape(), WEL, GM, CAU, STLS, TLS_, HUB, barron(-2.0))
            c2 = 1.7
            for sh in shapes
                # TOTALITY: every shape answers all three, for every argument.
                for s in (0.0, 0.05, 0.9, 3.4, 22.0)
                    @test isfinite(ρ(sh, s, c2))
                    @test isfinite(weight(sh, s, c2))
                    @test isfinite(ψ(sh, sqrt(s), c2))
                end
                # ψ IS r·w(r²) — the defining IRLS relation, and the reason ψ
                # is derived once rather than given a closed form per shape.
                for r in (0.0, 0.3, 1.4, 5.0)
                    @test ψ(sh, r, c2) == r * weight(sh, r * r, c2)
                end
            end
            # a non-finite residual is an OUTLIER for smooth TLS: weight 0 (the
            # ifelse clamp; `max` would have propagated the NaN), loss at the depth
            @test weight(STLS, NaN, c2) === 0.0 && weight(STLS, NaN32, 1.7f0) === 0.0f0
            @test ρ(STLS, NaN, c2) == c2
            @test weight(STLS, 2 * c2, c2) === 0.0 && weight(STLS, c2, c2) === 0.0
            # w = dρ/ds by central difference, away from the kinks (TLS jumps
            # at c2; Huber and smooth-TLS bend there).
            for sh in (WEL, GM, CAU, barron(-2.0), barron(-0.5))
                for s in (0.12, 0.8, 2.9, 14.0)
                    fd = FiniteDiff.finite_difference_derivative(x -> ρ(sh, x, c2), s)
                    @test fd ≈ weight(sh, s, c2) rtol = 1.0e-5
                end
            end
        end

        # The Fenchel/Geman–Reynolds duality of the lift, for the whole
        # liftable family (the Barron-only version lives further down).
        @testset "Geman–Reynolds duality over the liftable shapes" begin
            c2 = 1.3
            wgrid = exp10.(range(-7, 0, length = 2001))
            for sh in (WEL, GM, CAU, STLS)
                wmax = lift_wmax(sh)
                @test liftable(sh)
                for s in (0.0, 0.04, 0.7, 2.6, 11.0)
                    ρs = ρ(sh, s, c2)
                    # the envelope never undercuts ρ...
                    for w in (0.01 * wmax, 0.4 * wmax, wmax)
                        @test w * s + conjugate(sh, w, c2) >= ρs - 1.0e-10
                    end
                    # ...and is ATTAINED, at the IRLS weight. The grid is in the
                    # NORMALIZED w̄ ∈ (0,1]; the envelope is in the unnormalized
                    # w = w̄·w_max, and both terms must use it (smooth-TLS has
                    # w_max = 2, so dropping it on the linear term is silent
                    # for the other three and wrong for that one).
                    env = minimum(
                        ((w * wmax) * s + conjugate(sh, w * wmax, c2)) for w in wgrid
                    )
                    @test env ≈ ρs rtol = 5.0e-4 atol = 1.0e-10
                    w★ = weight(sh, s, c2)
                    @test w★ * s + conjugate(sh, w★, c2) ≈ ρs rtol = 1.0e-10 atol = 1.0e-12
                end
            end
        end

        # The scale_multiple/s2 split exists so one can move without the other.
        # Exported verbs with no consumer yet (RVG adopts them next), so they
        # are pinned here rather than left to rot untested.
        @testset "deployment movers: scale_multiple and s2 move independently" begin
            loss = deploy(WEL, 0.37^2; codim = 2)
            @test loss_shape(loss) === WEL
            @test loss.scale_multiple == efficiency_constant(WelschShape, 2) == 3.251
            @test sq_loss_width(loss) == loss.scale_multiple^2 * loss.s2
            # `loss_width` is scale_multiple*sqrt(s2), NEVER sqrt(scale_multiple^2*s2)
            # — the association is the whole reason s2 is stored rather than a
            # linear sigma.
            @test VisualGeometryCore.loss_width(loss) == loss.scale_multiple * sqrt(loss.s2)
            # the old field name is gone, not aliased
            @test !hasproperty(loss, :k)

            re = rescale(loss, 0.5^2)          # new noise, same multiple
            @test re.s2 == 0.25 && re.scale_multiple == loss.scale_multiple && loss_shape(re) === WEL
            rt = retune(loss, 1.6)             # new multiple, same noise
            @test rt.scale_multiple == 1.6 && rt.s2 == loss.s2

            # codimension_factor is the ratio it says it is, and 1 at d = 1
            @test codimension_factor(WelschShape, 1) == 1.0
            @test codimension_factor(WelschShape, 2) ==
                efficiency_constant(WelschShape, 2) / efficiency_constant(WelschShape)
            # past the solved table, the chi lift takes over
            @test efficiency_constant(WelschShape, 9) ≈
                efficiency_constant(WelschShape) * sqrt((9 + 2) / 3)

            # target_loss unwraps the homotopy; an annealed loss answers
            # every width-reading verb through the AnyLoss union.
            a = anneal(loss, 4.0)
            @test target_loss(a) === loss
            @test target_loss(loss) === loss
            @test sq_loss_width(a) == 4.0 * sq_loss_width(loss)
            @test loss_shape(a) === WEL
            @test liftable(a) && bounded(a)
            @test lift_wmax(a) == lift_wmax(WEL)
            @test lift_weight(a, 2.0) == weight(WEL, 2.0, sq_loss_width(a))
            @test lift_rho(a, 2.0) == ρ(WEL, 2.0, sq_loss_width(a))
            @test lift_conjugate(a, 0.3) == conjugate(WEL, 0.3, sq_loss_width(a))
            @test lift_zblock(a, 0.3) == zblock(WEL, 0.3, sq_loss_width(a))
            @test VisualGeometryCore.loss_width(a) == VisualGeometryCore.loss_width(loss) * 2.0
            # rescale/retune are forwarded through the anneal, keeping μ
            @test sq_loss_width(rescale(a, 0.5^2)) == 4.0 * sq_loss_width(rescale(loss, 0.5^2))
        end

        # The Welsch ½ is the SHAPE's cost convention, not the graduation
        # family's — a second width-annealed shape must not inherit it.
        @testset "cost_scale is per shape, and cost applies it" begin
            @test cost_scale(WEL) == 0.5
            @test all(cost_scale(s) == 1.0 for s in (GM, CAU, STLS, TLS_, HUB, GaussianShape()))
            r = [0.3, -1.2, 2.5]
            c2 = 1.5^2
            @test cost(WEL, r, c2, 3) == 0.5 * sum(ρ(WEL, x^2, c2) for x in r)
            @test cost(GM, r, c2, 3) == sum(ρ(GM, x^2, c2) for x in r)
        end
    end

    @testset "cost: one generic sum over ρ, for every shape" begin
        r = [0.3, -1.2, 2.5]
        @test cost(CAU, r, 1.5^2, 3) == sum(ρ(CAU, x^2, 1.5^2) for x in r)
        # TLS used to be REFUSED here, because ρ was scoped to the
        # Geman–Reynolds liftable shapes. 0.29 made ρ total — a loss needs no
        # lift — so summing it is well defined, and the truncation sums to
        # exactly `min(s, c²)` per point.
        @test cost(TLS_, r, 1.5^2, 3) == sum(min(x^2, 1.5^2) for x in r)
        # What TLS still lacks is the LIFT, which is what actually needs the
        # auxiliary variable to exist.
        @test_throws MethodError conjugate(TLS_, 0.5, 1.5^2)
        @test_throws MethodError zblock(TLS_, 0.5, 1.5^2)
    end

    @testset "BarronShape — α-dial anchors + envelope (MANDATORY pre-launch gates)" begin
        ss = (0.0, 1.0e-4, 0.02, 0.3, 1.7, 9.0)
        c2s = (0.4, 1.0, 2.7)
        @testset "anchor identities (convention-mapped c)" begin
            for s in ss, c2 in c2s
                # α = −2 ≡ Geman–McClure at 4c² (EXACT identity)
                @test ρ(barron(-2.0), s, c2) ≈ ρ(GM, s, 4c2) rtol = 1.0e-9 atol = 1.0e-15
                @test weight(barron(-2.0), s, c2) ≈ weight(GM, s, 4c2) rtol = 1.0e-9
                # α = 0 ≡ Cauchy at 2c²
                @test ρ(barron(0.0), s, c2) ≈ ρ(CAU, s, 2c2) rtol = 1.0e-12
                @test weight(barron(0.0), s, c2) ≈ weight(CAU, s, 2c2) rtol = 1.0e-12
                # α → −∞ ≡ Welsch at 2c²
                # −∞ anchor: rtol 1e-3 — catches any wrong convention constant;
                # tighter tolerances hit the 1+x precision floor of the direct
                # power form at s/(c²|α|) ~ 1e-13, not a formula error.
                @test ρ(barron(-1.0e8), s, c2) ≈ ρ(WEL, s, 2c2) rtol = 1.0e-3 atol = 1.0e-15
                @test weight(barron(-1.0e8), s, c2) ≈ weight(WEL, s, 2c2) rtol = 1.0e-5
            end
            for w in (1.0e-6, 0.03, 0.4, 0.95, 1.0), c2 in c2s
                @test conjugate(barron(-2.0), w, c2) ≈ conjugate(GM, w, 4c2) rtol = 1.0e-10 atol = 1.0e-14
                @test conjugate(barron(0.0), w, c2) ≈ conjugate(CAU, w, 2c2) rtol = 1.0e-10 atol = 1.0e-14
                # zblock: α = −2 ≡ GM and α = 0 ≡ Cauchy, the clamp w̄ = 1 included
                for (sh, ref, c2ref) in ((barron(-2.0), GM, 4c2), (barron(0.0), CAU, 2c2))
                    r2a, ha = zblock(sh, w, c2)
                    r2b, hb = zblock(ref, w, c2ref)
                    @test r2a ≈ r2b rtol = 1.0e-10 atol = 1.0e-12
                    @test ha ≈ hb rtol = 1.0e-8 atol = 1.0e-12
                end
            end
            # α → −∞ ≡ Welsch at 2c² in the z-block, the clamp included
            for w in (1.0e-3, 0.03, 0.4, 0.95, 1 - 1.0e-6, prevfloat(1.0), 1.0), c2 in c2s
                r2a, ha = zblock(barron(-1.0e8), w, c2)
                r2b, hb = zblock(WEL, w, 2c2)
                @test r2a ≈ r2b rtol = 1.0e-5 atol = 1.0e-15
                @test ha ≈ hb rtol = 1.0e-5 atol = 1.0e-15
            end
            # half-weight decision consistency at the anchors
            @test decision_width(barron(-2.0), 1.5 * 0.01) * 2 ≈ decision_width(GM, 1.5 * 0.01) rtol = 1.0e-12
            @test decision_width(barron(0.0), 1.5 * 0.01) * sqrt(2) ≈ decision_width(CAU, 1.5 * 0.01) rtol = 1.0e-12
            @test decision_width(barron(-1.0e8), 1.5 * 0.01) * sqrt(2) ≈ decision_width(WEL, 1.5 * 0.01) rtol = 1.0e-4
        end
        @testset "Geman–Reynolds envelope ρ(s) = min_w [w·s + conjugate(w)] ∀ dial α" begin
            for α in (-8.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0), c2 in (0.7, 1.9), s in ss
                shape = barron(α)
                ρ_val = ρ(shape, s, c2)
                wgrid = exp10.(range(-8, 0, length = 3001))
                env = minimum(w * s + conjugate(shape, w, c2) for w in wgrid)
                # grid-min sanity at grid resolution; the EXACT conjugacy proof is
                # the analytic-argmin attainment below (rtol 1e-10)
                @test env ≈ ρ_val rtol = 5.0e-4 atol = 1.0e-10
                @test env >= ρ_val - 1.0e-10                        # `conjugate` is the exact conjugate: the envelope never undercuts
                # analytic argmin is the IRLS weight; envelope attained there
                w★ = weight(shape, s, c2)
                @test w★ * s + conjugate(shape, w★, c2) ≈ ρ_val rtol = 1.0e-10 atol = 1.0e-12
                # zblock inverse-weight map round-trips: s(w(s)) = s
                r2, _ = zblock(shape, clamp(w★, 1.0e-9, 1.0), c2)
                s > 0 && w★ > 1.0e-9 && @test r2 ≈ s rtol = 1.0e-8
            end
        end
        @testset "traits + monotonicity" begin
            @test liftable(barron(-3.0)) && liftable(barron(1.0))
            @test bounded(barron(-0.5)) && !bounded(barron(0.0)) && !bounded(barron(1.0))
            @test_throws ArgumentError barron(2.0)
            for α in (-8.0, -1.0, 0.0, 1.0)
                ws = [weight(barron(α), s, 1.3) for s in 0.0:0.1:20.0]
                @test ws[1] == 1.0 && all(diff(ws) .<= 0)     # w_max = 1, monotone redescent
            end
            # ρ(∞) = 2c²b/|α| for α < 0 (finite retirement price)
            @test ρ(barron(-2.0), 1.0e12, 1.0) ≈ 2 * 4 / 2 rtol = 1.0e-3
        end
    end

    @testset "z-block Hessian at w̄ → 1: the positive limit, no cancellation" begin
        # rinv2 and κ̄² both vanish at w̄ = 1 but hreg → 2κ̄²″(1) > 0, Zach &
        # Bourmaud's Taylor limit 2γ″(1) (ECCV 2018, Eq. 10); the lift's z clamp
        # puts every confident inlier at w̄ = 1 EXACTLY, and every shape takes
        # the limit there.
        c2 = 9.0e-4
        for (sh, lim) in ((WEL, 2c2), (CAU, 2c2), (GM, c2), (barron(0.0), 4c2), (barron(-2.0), 4c2),
                          (barron(-10.0), 4c2), (barron(1.5), 4c2))
            @test zblock(sh, 1.0, c2) == (0.0, zblock(sh, 1.0, c2)[2])
            @test zblock(sh, 1.0, c2)[2] ≈ lim rtol = 1.0e-12
            @test zblock(sh, prevfloat(1.0), c2)[2] ≈ lim rtol = 1.0e-10
        end
        # against the defining ratio w̄·rinv2²/κ̄² in 256-bit arithmetic, through
        # and on both sides of the series cut
        setprecision(BigFloat, 256) do
            reference(sh, w, c2) = (wb = big(w); cb = big(c2); r = zblock(sh, wb, cb)[1]; Float64(wb * r^2 / conjugate(sh, wb, cb)))
            for sh in (WEL, CAU, barron(-2.0), barron(-1.0e6), barron(1.9)),
                    w in (prevfloat(1.0), 1 - 1.0e-12, 1 - 3.0e-8, 1 - 1.0e-5, 1 - 2.0e-3, 1 - 3.0e-3, 0.99, 0.5, 1.0e-3)
                @test zblock(sh, w, c2)[2] ≈ reference(sh, w, c2) rtol = 1.0e-10
            end
            # the series is two-sided: correct on the continuation w̄ > 1 too
            # (it used to be taken for EVERY w̄ > 1, since ε < 0 < cut)
            for sh in (WEL, CAU, barron(-2.0), barron(1.9)),
                    w in (nextfloat(1.0), 1 + 1.0e-5, 1 + 2.0e-3, 1 + 3.0e-3, 1.01, 1.5, 3.0)
                @test zblock(sh, w, c2)[2] ≈ reference(sh, w, c2) rtol = 1.0e-10
            end
        end
        # continuous across the cut, in both precisions
        for sh in (WEL, CAU, barron(-2.0)), T in (Float64, Float32)
            cut = VisualGeometryCore._zblock_series_cut(T)
            below = zblock(sh, 1 - prevfloat(cut), T(c2))[2]
            above = zblock(sh, 1 - nextfloat(cut), T(c2))[2]
            @test below ≈ above rtol = T === Float64 ? 1.0e-11 : 1.0f-4
        end
    end

    # Moved from RobustVisualGeometry's lift tests (2026-09-30): the
    # z-block's first entry is the derivative of the penalty it is the block of,
    # and the half-weight depth solves the equation that defines it — both are
    # this package's shape math, checked here against oracles that do not
    # restate it (a central difference; the normalized IRLS weight).
    @testset "z-block rinv2 = −κ̄′(w̄), and the half-weight decision width, per liftable shape" begin
        c2 = 1.7
        for shape in (WEL, GM, CAU, STLS)
            d = deploy(shape, c2; scale_multiple = 1.0)
            wmax = lift_wmax(shape)
            κ̄(w̄) = lift_conjugate(d, wmax * w̄) / wmax        # the normalized conjugate penalty
            for w̄ in (1.0e-8, 1.0e-6, 1.0e-3, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999)
                rinv2 = lift_zblock(d, w̄)[1]
                # w̄ runs down to 1e-8 and κ̄ lives on w̄ > 0: the absolute floor (FiniteDiff's
                # default is its relative step, 6e-6) keeps w̄ − h positive
                dκ̄ = FiniteDiff.finite_difference_derivative(κ̄, w̄; absstep = 1.0e-10)
                @test rinv2 ≈ -dκ̄ rtol = 1.0e-4 atol = 1.0e-6 * c2
                @test rinv2 >= 0
            end
            @test lift_zblock(d, 1.0)[1] ≈ 0 atol = 1.0e-14      # r²(1) = 0: the inlier limit
            # the normalized IRLS weight at the decision threshold is ½ at the width it returns
            for scale_multiple in (0.8, 1.0, 1.5, 2.5, 3.3), ŝ in (0.003, 1.0)
                decision_threshold = scale_multiple * ŝ
                loss_width = decision_width(shape, decision_threshold)
                @test loss_width > 0
                @test lift_weight(deploy(shape, loss_width^2; scale_multiple = 1.0), decision_threshold^2) / wmax ≈ 0.5 atol = 1.0e-12
            end
        end
        # a hard truncation decides at its own width
        @test decision_width(TLS_, 0.37) == 0.37
    end

    @testset "shape math keeps the argument's precision" begin
        # Float32 in → Float32 out, for EVERY shape and every shape-math verb:
        # a Float64 literal in a closed form (or a Float64-stored Barron α)
        # would silently promote, and a Float32 GPU/scoring consumer could not
        # route through these definitions. Values agree with the Float64 path
        # to Float32 precision.
        every = (GaussianShape(), WEL, GM, CAU, STLS, TLS_, HUB, barron(-2.0), barron(0.0), barron(1.0))
        lifts = (WEL, GM, CAU, STLS, barron(-2.0), barron(0.0), barron(1.0))
        r32 = Float32[0.1, -2.0, 0.7, 5.0]
        for sh in every, s in (0.0f0, 0.3f0, 4.0f0)
            @test ρ(sh, s, 1.5f0) isa Float32
            @test weight(sh, s, 1.5f0) isa Float32
            @test ψ(sh, sqrt(s), 1.5f0) isa Float32
            @test ρ(sh, s, 1.5f0) ≈ Float32(ρ(sh, Float64(s), 1.5)) rtol = 1.0f-5
            @test weight(sh, s, 1.5f0) ≈ Float32(weight(sh, Float64(s), 1.5)) rtol = 1.0f-5
        end
        for sh in every
            @test cost(sh, r32, 1.5f0, 4) isa Float32
            @test cost(sh, r32, 1.5f0, 4) ≈ Float32(cost(sh, Float64.(r32), 1.5, 4)) rtol = 1.0f-5
        end
        for sh in lifts, w in (0.05f0, 0.5f0, 1.0f0)
            @test conjugate(sh, w, 1.5f0) isa Float32
            @test zblock(sh, w, 1.5f0) isa Tuple{Float32, Float32}
            # rtol for the large values near w → 0; atol for conjugate(w_max) = 0
            @test conjugate(sh, w, 1.5f0) ≈ Float32(conjugate(sh, Float64(w), 1.5)) rtol = 1.0f-5 atol = 1.0f-6
        end
    end

    @testset "Manifold trait — retraction, Jacobian pins, covariance transport" begin
        using VisualGeometryCore: model
        rng = MersenneTwister(3)
        Rr = RotMatrix(Rotations.AngleAxis(0.7, normalize(randn(rng, 3))...))
        tr_ = Vec3{Float64}(normalize(randn(rng, 3)))
        S0 = svd(randn(rng, 3, 3))
        Fanchor = SMatrix{3, 3, Float64}(S0.U * Diagonal([1.0, 0.6, 0.0]) * S0.V')   # generic rank-2 F
        charts = Any[
            EssentialManifold5(Rr, tr_), EssentialManifold6(Rr, tr_),
            FundMatManifold7(Fanchor),
            HomographyManifold8(SMatrix{3, 3, Float64, 9}(RotMatrix(Rotations.AngleAxis(0.3, 0, 0, 1))) + SMatrix{3, 3, Float64, 9}(0.1randn(rng, 3, 3))),
        ]
        for ch in charts
            D = tangent_size(ch)
            # retraction consistency: v = 0 is the anchor; recenter(0) preserves the model
            M0 = SMatrix{3, 3, Float64}(model(ch))
            @test SMatrix{3, 3, Float64}(retract(ch, zero(SVector{D, Float64}))) ≈ M0 rtol = 1.0e-14
            ch0 = recenter(ch, zero(SVector{D, Float64}))
            # the homography chart re-anchors at the unit-Frobenius representative of the
            # same model (its first-order retraction lets the scale drift); the others
            # keep their representative
            rep(M) = ch isa HomographyManifold8 ? M / norm(M) : M
            @test SMatrix{3, 3, Float64}(VisualGeometryCore.model(ch0)) ≈ rep(M0) rtol = 1.0e-12
            # recenter == retract at the same coordinates (retraction = evaluation)
            v = 1.0e-2 * SVector{D}(randn(rng, D))
            @test SMatrix{3, 3, Float64}(VisualGeometryCore.model(recenter(ch, v))) ≈
                rep(SMatrix{3, 3, Float64}(retract(ch, v))) rtol = 1.0e-12
            # Jacobian finite-difference pin (central differences)
            T = retract_jacobian(ch)
            fd = FiniteDiff.finite_difference_jacobian(w -> SVector{9}(Tuple(retract(ch, w))), zero(SVector{D, Float64}), Val(:central))
            for q in 1:D
                @test maximum(abs.(T[:, q] .- fd[:, q])) < 1.0e-6 * max(1.0, maximum(abs.(fd[:, q])))
            end
            # covariance transport round-trip, to the ambient chart and back: == P·Σ·P with P the
            # projector onto the chart's non-gauge subspace (identity for minimal
            # charts; rank-deficient charts project their gauge direction out)
            A = randn(rng, D, D); Σ = A * A' + 0.1I
            Σrt = transform_covariance(EuclideanManifold(ch), ch, transform_covariance(ch, EuclideanManifold(ch), Σ))
            P = pinv(T) * T
            @test Σrt ≈ P * Σ * P' rtol = 1.0e-8
            r = rank(T; rtol = 1.0e-10)
            if ch isa EssentialManifold5 || ch isa HomographyManifold8
                @test r == D
                @test Σrt ≈ Σ rtol = 1.0e-8                    # minimal: exact round-trip
            elseif ch isa EssentialManifold6
                @test r == 5                                  # documented t-scale gauge nullity
                # the gauge direction is the radial t direction in the τ block
                g = nullspace(T)[:, 1]
                @test abs(dot(normalize(g[4:6]), Vector(tr_))) ≈ 1 atol = 1.0e-8
                @test maximum(abs.(g[1:3])) < 1.0e-8
            else
                @test r in (6, 7)                             # σ→1 co-rotation gauge only near-essential
            end
            # the ambient covariance is PSD of the same rank
            Σm = transform_covariance(ch, EuclideanManifold(ch), Σ)
            @test isapprox(Σm, Σm'; rtol = 1.0e-12)
            @test rank(Σm; rtol = 1.0e-10) == r
        end
        # the 5-dof basis convention matches its documented invariants
        b1, b2 = s2_tangent_basis(tr_)
        @test abs(dot(b1, tr_)) < 1.0e-14 && abs(dot(b2, tr_)) < 1.0e-14 && abs(dot(b1, b2)) < 1.0e-14
        @test norm(b1) ≈ 1 && norm(b2) ≈ 1
        # FundMatManifold7 anchors on the SVD ratio convention and recovers F₀
        U = SMatrix{3, 3, Float64}(RotMatrix(Rotations.AngleAxis(0.4, 0, 0, 1)))
        V = SMatrix{3, 3, Float64}(RotMatrix(Rotations.AngleAxis(-0.2, 0, 1, 0)))
        Ftrue = U * Diagonal([2.0, 1.2, 0.0]) * V'
        chF = FundMatManifold7(SMatrix{3, 3, Float64}(Ftrue))
        @test SMatrix{3, 3, Float64}(VisualGeometryCore.model(chF)) * 2.0 ≈ Ftrue rtol = 1.0e-10
    end

    @testset "RigidManifold6 — SO(3)⊕R³, full-rank, exact covariance round-trip" begin
        using VisualGeometryCore: model, RigidManifold6, recenter_pose
        rng = MersenneTwister(11)
        R0 = RotMatrix(Rotations.AngleAxis(0.9, normalize(randn(rng, 3))...))
        t0 = Vec3{Float64}(1.3, -2.1, 4.7)            # FREE R³ anchor (metric scale)
        ch = RigidManifold6(R0, t0)
        @test tangent_size(ch) == 6
        # retraction: v = 0 is the anchor; the free-t block is exact translation
        Rz, tz = recenter_pose(ch, zero(SVector{6, Float64}))
        @test SMatrix{3, 3, Float64}(Rz) ≈ SMatrix{3, 3, Float64}(R0) rtol = 1.0e-14
        @test tz == t0
        v = SVector{6, Float64}(0.03, -0.02, 0.01, 0.5, -1.2, 0.7)
        Rv, tv = recenter_pose(ch, v)
        @test tv == t0 + Vec3{Float64}(0.5, -1.2, 0.7)          # translation is FREE, exact
        @test SMatrix{3, 3, Float64}(Rv) ≈ SMatrix{3, 3, Float64}(RotMatrix(Rotations.MRP(v[1], v[2], v[3])) * R0) rtol = 1.0e-14
        # model is the anchor pose; retract == recenter at the same coords
        @test SMatrix{3, 3, Float64}(VisualGeometryCore.model(ch).R) ≈ SMatrix{3, 3, Float64}(R0)
        @test VisualGeometryCore.model(ch).t == t0
        cp = retract(ch, v)
        @test SMatrix{3, 3, Float64}(cp.R) ≈ SMatrix{3, 3, Float64}(Rv) && cp.t == tv
        # FD Jacobian pin on the 12-vector [vec(R); t] retraction (central diff)
        T = retract_jacobian(ch)
        @test size(T) == (12, 6)
        # the chart's own retraction in its ambient coordinates (the MRP left
        # rotation and the free translation checked explicitly above)
        amb(v) = VisualGeometryCore._ambient_coordinates(retract(ch, v))
        fd = FiniteDiff.finite_difference_jacobian(amb, zero(SVector{6, Float64}), Val(:central))
        for q in 1:6
            @test maximum(abs.(T[:, q] .- fd[:, q])) < 1.0e-6 * max(1.0, maximum(abs.(fd[:, q])))
        end
        # FULL RANK 6 — NO gauge null direction (the rigid chart's defining property)
        @test rank(T; rtol = 1.0e-10) == 6
        @test size(nullspace(T; rtol = 1.0e-9), 2) == 0
        # translation block of T is the identity (rows 10:12, cols 4:6)
        @test T[10:12, 4:6] ≈ Matrix(1.0I, 3, 3) atol = 1.0e-12
        @test maximum(abs.(T[10:12, 1:3])) < 1.0e-12    # rotation coords don't move t
        @test maximum(abs.(T[1:9, 4:6])) < 1.0e-12      # translation coords don't move R
        # covariance transport: EXACT round-trip (P = pinv(T)·T = I₆, no gauge)
        A = randn(rng, 6, 6); Σ = A * A' + 0.05I
        Σrt = transform_covariance(EuclideanManifold(ch), ch, transform_covariance(ch, EuclideanManifold(ch), Σ))
        @test Σrt ≈ Σ rtol = 1.0e-9
        Σm = transform_covariance(ch, EuclideanManifold(ch), Σ)
        @test isapprox(Σm, Σm'; rtol = 1.0e-12)
        @test rank(Σm; rtol = 1.0e-10) == 6
        # metric translation covariance survives transport (unlike EssentialManifold6,
        # whose radial-t gauge would project the translation-scale direction out)
        e6 = RigidManifold6(R0, Vec3{Float64}(0.0, 0.0, 0.0))
        Σt = zeros(6, 6); Σt[4, 4] = 1.0              # pure t-x variance
        @test transform_covariance(e6, EuclideanManifold(e6), Σt)[10, 10] ≈ 1.0 atol = 1.0e-12
    end
end

@testitem "conjugate_curvature" begin
    using Test
    using VisualGeometryCore
    using VisualGeometryCore: conjugate, conjugate_curvature, WelschShape, GemanMcclureShape,
        CauchyShape, SmoothTLSShape, BarronShape, liftable
    using FiniteDiff: FiniteDiff

    # w·κ²″(w) against a central difference of the conjugate itself, every
    # liftable shape (Barron at the Cauchy, GM and a generic α), at weights
    # across the lift window. The second derivative is FiniteDiff's central
    # derivative of its central derivative (default steps): measured max relative
    # error over these 32 points 3.1e-6, against 7.8e-6 for its 1-D Hessian and
    # 7.1e-6 for the hand second difference at h = 1e-4·w it replaces.
    c2 = 0.37
    shapes = (WelschShape(), GemanMcclureShape(), CauchyShape(), SmoothTLSShape(),
        BarronShape(0.0), BarronShape(-2.0), BarronShape(-5.3), BarronShape(1.2))
    for shape in shapes
        @test liftable(shape)
        for w in (0.05, 0.3, 0.7, 0.95)
            κ′(x) = FiniteDiff.finite_difference_derivative(y -> conjugate(shape, y, c2), x, Val(:central))
            fd = w * FiniteDiff.finite_difference_derivative(κ′, w, Val(:central))
            @test conjugate_curvature(shape, w, c2) ≈ fd rtol = 1.0e-5
        end
    end
    # Barron's limits: α = 0 is Cauchy at 2c², α = −2 is Geman–McClure at 4c²
    @test conjugate_curvature(BarronShape(0.0), 0.4, c2) ≈ conjugate_curvature(CauchyShape(), 0.4, 2c2)
    @test conjugate_curvature(BarronShape(-2.0), 0.4, c2) ≈ conjugate_curvature(GemanMcclureShape(), 0.4, 4c2)
end
