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
        ray(θ, ψ = 0.0) = SVector(sin(θ) * cos(ψ), sin(θ) * sin(ψ), cos(θ))
        function first_nan(l, ψ = 0.0)          # bisected on project itself
            lo, hi = 0.3, 1.5
            for _ in 1:80
                m = (lo + hi) / 2
                all(isfinite, project(l, ray(m, ψ))) ? (lo = m) : (hi = m)
            end
            return lo
        end
        # the onset along every azimuth: the limit of a tilt-first chain is its minimum
        min_first_nan(l, n) = minimum(first_nan(l, ψ) for ψ in range(0, 2π; length = n + 1)[1:n])
        for (l, before) in ((chain, x -> distort_normalized(Lens(Perspective(), tilt), x)), (twin, identity))
            θn = first_nan(l)
            # the radial stage's output radius at its input where project turns NaN
            x = ray(θn)
            ρ = norm(before(SVector(x[1] / x[3], x[2] / x[3])))
            @test monotone_radius(l) ≈ ρ * F(ρ^2) atol = 1.0e-6
        end
        @test monotone_radius(chain) == monotone_radius(twin)
        # the twin's bound is radially symmetric: the onset at azimuth 0
        @test view_angle_limit(twin) ≈ first_nan(twin) atol = 1.0e-6
        # a tilt before the radial stage: the onset depends on the azimuth (here
        # 0.9487 at azimuth 0), and the limit is the least of them
        @test view_angle_limit(chain) ≈ 0.9404484 atol = 1.0e-6
        @test view_angle_limit(chain) ≈ min_first_nan(chain, 72) atol = 1.0e-4
        @test view_angle_limit(chain) <= min_first_nan(chain, 72)
        strong = Lens(Perspective(), Chain(Tilt(0.15, -0.1), Additive(Radial((-0.3, 0.04)), Tangential(1.0e-3, -5.0e-4))))
        @test first_nan(strong) > 0.88                                  # azimuth 0 overstates it by ≈ 3°
        @test view_angle_limit(strong) ≈ 0.8375568 atol = 1.0e-6
        @test view_angle_limit(strong) ≈ min_first_nan(strong, 72) atol = 1.0e-4
        for ψ in range(0, 2π; length = 49)                               # every ray within it is imaged
            @test all(isfinite, project(strong, ray(view_angle_limit(strong), ψ)))
        end
        # a chain whose radial stage leads: the onset at azimuth 0, as before the sweep
        radial_first = (
            (Lens(Perspective(), Chain(Additive(Radial((-0.3, 0.04)), Tangential(1.0e-3, 0.0)), Tilt(0.01, 0.0))), 0.9553166181245096, 0.7919595949289332),
            (Lens(Perspective(), Chain(Additive(Radial((-0.150994, -0.0501823, 0.023916), (0.187097, 0.0255086, -0.00128661)),
                Tangential(0.00244469, 0.00575199), ThinPrism((-0.00157781, -0.000969799), (-0.000762879, 0.00210571))),
                Tilt(-0.00992587, -0.0062454))), 0.838476342933477, 0.685637031417361),
            (Lens(Equidistant(), Chain(Radial((-0.02, 0.003)), Tilt(0.05, 0.02))), 3.1405926535897932, 3.439526174339641),
        )
        for (l, θ, ρ) in radial_first
            @test view_angle_limit(l) ≈ θ rtol = 1.0e-12
            @test monotone_radius(l) ≈ ρ rtol = 1.0e-12
        end
        @test view_angle_limit(radial_first[1][1]) ≈ first_nan(radial_first[1][1]) atol = 1.0e-6

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
