# view_angle_limit — the honest θ-domain of each lens (projections.jl). The lens
# regression that caps its fit domain with it (`fit_lens`) lives in
# VisualGeometryToolkitEstimation, with its tests.
@testitem "view_angle" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: view_angle_limit, Division, DoubleSphere, Pinhole, KannalaBrandt, projected_radius
    using VisualGeometryToolkitCore: monotone_radius   # public, not exported
    using StaticArrays: SVector
    using LinearAlgebra: norm

    @testset "view_angle_limit — honest domains" begin
        @test view_angle_limit(Pinhole()) < π / 2
        @test view_angle_limit(Division(-0.3302, -0.0765)) > π / 2        # the division lift sees past 90°
        @test view_angle_limit(Lens(DoubleSphere(0.6, 0.55))) > π / 2
        @test view_angle_limit(Lens(Perspective(), Radial((-0.2,)))) <= π / 2
        # Kannala–Brandt: the EXACT profile fold (1 − 0.3θ² = 0), just inside it;
        # no fold: the sphere's edge
        kb = KannalaBrandt(-0.1)
        θf = sqrt(1 / 0.3)
        @test θf * (1 - 2.0e-9) < view_angle_limit(kb) < θf
        @test isfinite(projected_radius(kb, view_angle_limit(kb)))
        @test view_angle_limit(kb; cap = 1.0) == 1.0
        @test view_angle_limit(KannalaBrandt(0.01)) == π - 1.0e-3
    end

    @testset "a Chain is bounded by its radial stage, where project turns NaN" begin
        # the radial factor behind a tilt, and its Additive twin without the tilt:
        # ρ(1 − 0.3ρ² + 0.04ρ⁴) folds at ρ = √2 (output radius ≈ 0.792)
        F(s) = 1 - 0.3s + 0.04s^2
        tilt = Tilt(0.02, -0.01)
        chain = Lens(Perspective(), Chain(tilt, Additive(Radial((-0.3, 0.04)), Tangential(1.0e-3, -5.0e-4))))
        twin = Lens(Perspective(), Additive(Radial((-0.3, 0.04)), Tangential(1.0e-3, -5.0e-4)))
        ray(θ) = SVector(sin(θ), 0.0, cos(θ))
        function first_nan(l)                   # bisected on project itself, azimuth 0
            lo, hi = 0.5, 1.4
            for _ in 1:80
                m = (lo + hi) / 2
                all(isfinite, project(l, ray(m))) ? (lo = m) : (hi = m)
            end
            return lo
        end
        for (l, before) in ((chain, x -> distort_normalized(Lens(Perspective(), tilt), x)), (twin, identity))
            θn = first_nan(l)
            @test view_angle_limit(l) ≈ θn atol = 1.0e-6
            @test all(isfinite, project(l, ray(view_angle_limit(l))))
            # the radial stage's output radius at its input where project turns NaN
            x = ray(θn)
            ρ = norm(before(SVector(x[1] / x[3], x[2] / x[3])))
            @test monotone_radius(l) ≈ ρ * F(ρ^2) atol = 1.0e-6
        end
        @test monotone_radius(chain) == monotone_radius(twin)

        # every other lens reports what it did before the Chain rule
        unchanged = (
            (KannalaBrandt(-0.04, 0.01), 3.1405926535897932, 4.961538434230614),
            (KannalaBrandt(-0.3, 0.0, 0.0, 0.0), 1.054092552334984, 0.7027283689263066),
            (KannalaBrandt(-0.2, 0.02, 0.0, 0.002), 3.1405926535897932, 62.67892968012786),
            (KannalaBrandt(0.05, -0.01), 2.493389370633712, 2.3047390627059623),
            (Lens(Perspective(), Additive(Radial((-0.12, 0.03)), Tangential(1.0e-3, -5.0e-4))), 1.5697963267948967, Inf),
            (twin, 0.9553166171689783, 0.7919595949289332),
            (Lens(Perspective(), Radial((-0.3, 0.1, 0.0))), 1.5697963267948967, Inf),
            (Lens(Perspective(), Tangential(0.0, 0.1)), 1.5697963267948967, Inf),
            (Lens(Equidistant(), Additive(Radial((-0.02, 0.004, -0.0009, 0.0001)), Tangential(0.0006, -0.0004), ThinPrism((0.0002,), (-0.0001,)))), 3.1405926535897932, 4.0081918874704305),
            (Lens(DoubleSphere(-0.2, 0.55)), 2.392403240346484, Inf),
            (Lens(DoubleSphere(-0.2, 0.55), Radial((-0.05, 0.01))), 2.3915406423718952, Inf),
            (Lens(Unified(0.6)), 2.2134872954430507, Inf),
            (Division(-0.1), 3.1415826535897935, 3.162277660166941),
            (Division(-0.3, 0.05), 1.2806865643649843, 1.941358962420542),
            (Lens(Perspective(), FieldOfView(0.9)), 1.5697963267948967, Inf),
        )
        for (l, θ, ρ) in unchanged
            @test view_angle_limit(l) ≈ θ rtol = 1.0e-12
            @test isequal(monotone_radius(l), ρ) || isapprox(monotone_radius(l), ρ; rtol = 1.0e-12)
        end
    end
end
