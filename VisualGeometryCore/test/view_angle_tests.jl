# view_angle_limit — the honest θ-domain of each lens (projections.jl). The lens
# regression that caps its fit domain with it (`fit_lens`) lives in
# RobustVisualGeometry, with its tests.
@testitem "view_angle" begin
    using Test
    using VisualGeometryCore
    using VisualGeometryCore: view_angle_limit, Division, DoubleSphere, Pinhole, KannalaBrandt, projected_radius

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
end
