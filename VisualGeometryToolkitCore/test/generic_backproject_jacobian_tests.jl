# backproject_with_jacobian(::CentralGenericModel, u): the bearing of backproject,
# its Jacobian against central differences, tangency, NaN outside the area.

@testitem "CentralGenericModel: backproject_with_jacobian" begin
    using Test, StaticArrays, LinearAlgebra, Random
    using VisualGeometryToolkitCore: CentralGenericModel, CameraModel, KannalaBrandt, px, backproject, backproject_with_jacobian

    src = CameraModel(300.0px, SA[320.0px, 240.0px]; lens = KannalaBrandt(-0.02))
    m = CentralGenericModel(src, (640, 480); grid_size = (20, 16))
    rng = Xoshiro(2)
    for _ in 1:200
        u = SVector(40 + 560rand(rng), 40 + 400rand(rng))
        f, M = backproject_with_jacobian(m, u)
        @test f ≈ backproject(m, u) rtol = 1.0e-14
        h = 1.0e-5
        fd = hcat((backproject(m, u + SVector(h, 0)) - backproject(m, u - SVector(h, 0))) / 2h,
                  (backproject(m, u + SVector(0, h)) - backproject(m, u - SVector(0, h))) / 2h)
        @test M ≈ fd rtol = 1.0e-6
        @test norm(f' * M) < 1.0e-15
    end
    f, M = backproject_with_jacobian(m, SVector(-5.0, 10.0))
    @test all(isnan, f) && all(isnan, M)
end
