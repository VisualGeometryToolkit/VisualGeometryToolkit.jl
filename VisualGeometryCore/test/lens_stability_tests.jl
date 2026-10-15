# Every per-point lens path is type-stable and allocates nothing, for every
# kind of lens, in Float64 and Float32 — the property that makes the per-point
# functions the batch kernels (pointwise!) on a CPU and on a GPU.
@testitem "lens_stability" begin
    using Test
    using StaticArrays: SVector
    using VisualGeometryCore
    using VisualGeometryCore: _precision, project_with_jacobians, transform_deriv_params

    lenses = (
        Pinhole(), Lens(Equidistant()), Lens(Stereographic()), Lens(Equisolid()), Lens(Orthographic()),
        Lens(Perspective(), Radial((-0.12, 0.03))),
        Lens(Perspective(), Additive(Radial((-0.12, 0.03, 0.004), (0.02, -0.001, 0.0003)), Tangential(1.0e-3, -5.0e-4), ThinPrism((-5.5e-3, 7.0e-4), (2.2e-3, -1.2e-3)))),
        Lens(Perspective(), Chain(Additive(Radial((-0.12, 0.03)), Tangential(1.0e-3, -5.0e-4)), Tilt(0.01, -0.02))),
        Lens(Perspective(), Tangential(1.0e-3, -5.0e-4)), Lens(Perspective(), FieldOfView(0.9)),
        KannalaBrandt(-0.043, 0.011, -0.0083, 0.0015),
        Lens(Equidistant(), Additive(Radial((-0.02, 0.004, -0.0009, 0.0001)), Tangential(0.0006, -0.0004), ThinPrism((0.0002,), (-0.0001,)))),
        Lens(Equidistant(), Chain(Radial((0.1, -0.02, 0.003, 0.0, 0.0, 0.0001)), Additive(Tangential(0.0003, -0.0002), ThinPrism((0.001, 0.0), (-0.0005, 2.0e-5))))),
        Division(-0.31), Division(-0.2, 0.03, -0.004), LiftLens(Radial((0.05,), (-0.25, 0.02))), Scaramuzza(-0.17, -0.006, 0.035),
        Lens(DoubleSphere(-0.2, 0.59)), Lens(Unified(0.8)), Lens(ExtendedUnified(0.6, 1.1)),
        Lens(Unified(0.7), Additive(Radial((-0.1, 0.01)), Tangential(1.0e-3, 0.0))),
    )
    allocs(f, a, b) = (f(a, b); @allocated f(a, b))
    for T in (Float64, Float32), lens0 in lenses
        lens = _precision(T, lens0)
        X = SVector{3, T}(0.3, -0.2, 1.0)
        x = SVector{2, T}(0.21, -0.13)
        u = project(lens, X)
        for (f, a) in ((project, X), (backproject, u), (distort_normalized, x), (undistort_normalized, u),
                (transform_deriv, X), (transform_deriv_params, X), (project_with_jacobians, X))
            @test (@inferred f(lens, a); true)
            @test allocs(f, lens, a) == 0
        end
        @test eltype(project(lens, X)) === T && eltype(backproject(lens, u)) === T
    end
end
