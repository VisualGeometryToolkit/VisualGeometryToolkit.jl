@testitem "NoncentralSplineModel: the non-central grid on ChartSpline" setup = [SplineAxis] begin
    using VisualGeometryToolkitCore: thrown_message
    using Test
    using StaticArrays
    using Rotations: RotMatrix, RotXYZ
    using LinearAlgebra: norm, dot
    using Random: MersenneTwister
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: CameraModel, SplineLens, ChartSpline, NoncentralSplineModel, KannalaBrandt, px,
        backproject, backproject_with_jacobian, project, central_model, with_points, with_directions, with_present,
        calibration_matrix, CameraCalibrationMat, LogicalIntrinsics, Ray3, PosedCamera, EuclideanMap, m, mm, ustrip,
        with_unit, has_units, valid_domain, support_trait, LocalSupport, nodes, node_position, support, RotationMap,
        convert_eltype
    using Adapt: adapt
    using GeometryBasics: origin, widths

    # =============================================================================
    # The non-central grid (Schöps et al.) rebuilt on ChartSpline in the normalized
    # plane x = K⁻¹u, against the golden outputs of the old non-central grid model
    # (test/fixtures/spline_lens_golden.jl): the central grid's directions, points
    # 0.01 · direction (plain metres), project at X = origin + 1500 · direction.
    # =============================================================================

    include(joinpath(pkgdir(VisualGeometryToolkitCore), "test", "fixtures", "spline_lens_golden.jl"))
    G = GOLDEN
    src = CameraModel(G.f * px, SA[G.pp[1] * px, G.pp[2] * px]; lens = KannalaBrandt(G.kb...))
    image = Size2(G.image[1], G.image[2])
    grid = (G.grid_size[1], G.grid_size[2])
    cam = CameraModel(SplineLens, src, image; grid_size = grid)
    b = cam.lens.backward
    pts = ChartSpline(b.area, map(d -> 0.01 * d, b.controls))
    nc = NoncentralSplineModel(cam.intrinsics, b, pts)
    pixels = [SVector(G.pixels[1, i], G.pixels[2, i]) for i in axes(G.pixels, 2)]
    col3(A, i) = SVector(A[1, i], A[2, i], A[3, i])

    @testset "backproject: the golden lines, NaN where they are" begin
        worst_o, worst_d, same_nan = 0.0, 0.0, true
        for (i, u) in enumerate(pixels)
            far_edge_probe(G, i) && continue   # the old grid's half-open box: the far-edge item checks these
            r = backproject(nc, u)
            @test r isa Ray3{Float64}
            o, d = col3(G.noncentral.origins, i), col3(G.central.backproject, i)
            if any(isnan, o)
                same_nan &= all(isnan, r.origin) && all(isnan, r.direction)
            else
                same_nan &= all(isfinite, r.origin) && all(isfinite, r.direction)
                worst_o = max(worst_o, norm(SVector{3}(r.origin) - o))
                worst_d = max(worst_d, norm(SVector{3}(r.direction) - d))
            end
        end
        @test same_nan
        @test worst_o <= 1.0e-12
        @test worst_d <= 1.0e-12
    end

    @testset "backproject_with_jacobian: the direction and its pixel Jacobian" begin
        worst, tangent, same_nan = 0.0, 0.0, true
        for (i, u) in enumerate(pixels)
            far_edge_probe(G, i) && continue
            f, M = backproject_with_jacobian(nc, u)
            Mg = SMatrix{3, 2}(G.central.jac_M[:, :, i])
            if any(isnan, Mg)
                same_nan &= all(isnan, f) && all(isnan, M)
            else
                same_nan &= all(isfinite, M)
                worst = max(worst, maximum(abs.(M - Mg)), norm(f - col3(G.central.backproject, i)))
                tangent = max(tangent, maximum(abs.(f' * M)))
            end
        end
        @test same_nan
        @test worst <= 1.0e-12
        @test tangent <= 1.0e-15
    end

    @testset "project: the tangent-plane LM, the golden pixels" begin
        worst, same_nan = 0.0, true
        for i in 1:(G.n_lattice)
            X = col3(G.noncentral.origins, i) + 1500.0 * col3(G.central.backproject, i)
            p = project(nc, X)
            ref = SVector(G.noncentral.project_X[1, i], G.noncentral.project_X[2, i])
            if any(isnan, ref)
                same_nan &= all(isnan, p)
            else
                same_nan &= all(isfinite, p)
                worst = max(worst, norm(p - ref))
            end
        end
        @test same_nan
        @test worst <= 1.0e-9
        @test all(isnan, project(nc, SVector(NaN, 0.0, 1.0)))
        @test all(isnan, project(nc, SVector(0.0, 0.0, 1.0), SVector(NaN, 0.0)))
        # the two-argument form's start (the point's pinhole point where the model images
        # it) gives the area-centre start's pixel to the solve's tolerance, the same mask
        cx = origin(b.area) + widths(b.area) / 2
        Xs = [(r = backproject(nc, SVector(x, y)); SVector{3}(r.origin) + 1500.0 * SVector{3}(r.direction)) for y in 0.0:37.0:999.0 for x in 0.0:37.0:999.0]
        warm = [project(nc, X) for X in Xs]
        cold = [VisualGeometryToolkitCore.Cameras._noncentral_project(nc, X, SVector{2, Float64}(cx[1], cx[2])) for X in Xs]
        @test isnan.(first.(warm)) == isnan.(first.(cold))
        @test maximum(i -> norm(warm[i] - cold[i]), findall(p -> all(isfinite, p), warm)) <= 1.0e-9
    end

    @testset "a perturbed point grid: the projected pixel's line goes through X" begin
        rng = MersenneTwister(11)
        gn = with_points(nc, [SVector(0.4 * randn(rng), 0.4 * randn(rng), 0.4 * randn(rng)) for _ in 1:grid[2], _ in 1:grid[1]])
        worst = 0.0
        for _ in 1:15
            p0 = SVector(60 + 880 * rand(rng), 60 + 880 * rand(rng))
            r = backproject(gn, p0)
            X = SVector{3}(r.origin) + 1500.0 * SVector{3}(r.direction)
            p1 = project(gn, X, p0 + SVector(3.0, -2.0))
            @test all(isfinite, p1)
            r1 = backproject(gn, p1)
            pto = SVector{3}(r1.origin) - X
            worst = max(worst, norm(pto - dot(pto, SVector{3}(r1.direction)) * SVector{3}(r1.direction)))
        end
        @test worst < 1.0e-4                                       # point-to-line distance
    end

    @testset "central_model: the SplineLens camera of the directions" begin
        c = central_model(nc)
        @test c isa CameraModel{<:LogicalIntrinsics, <:SplineLens}
        @test calibration_matrix(c) == calibration_matrix(cam)
        worst, same_nan = 0.0, true
        for (i, u) in enumerate(pixels)
            far_edge_probe(G, i) && continue
            d, ref = backproject(c, u), col3(G.central.backproject, i)
            if any(isnan, ref)
                same_nan &= all(isnan, d)
            else
                same_nan &= all(isfinite, d)
                worst = max(worst, norm(d - ref))
            end
        end
        @test same_nan
        @test worst <= 1.0e-12
        @test valid_domain(nc) == valid_domain(c)
    end

    @testset "constructors: skew, mismatched grids, the source constructor" begin
        skewed = LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(300.0, 0.0, 0.0, 1.0, 300.0, 0.0, 500.0, 500.0, 1.0)))
        @test_throws ArgumentError NoncentralSplineModel(skewed, b, pts)
        @test_throws ArgumentError NoncentralSplineModel(cam.intrinsics, b, ChartSpline(b.area, pts.controls[1:(end - 1), :]))
        moved = ChartSpline(typeof(b.area)(origin(b.area) .+ 0.1, widths(b.area)), pts.controls)
        @test_throws ArgumentError NoncentralSplineModel(cam.intrinsics, b, moved)
        skewed_src = CameraModel(skewed, KannalaBrandt(G.kb...))
        @test_throws ArgumentError NoncentralSplineModel(skewed_src, image; grid_size = grid)

        s = NoncentralSplineModel(src, image; grid_size = grid)
        @test s.directions.controls == b.controls
        @test s.directions.area == b.area
        @test all(iszero, s.points.controls)
        @test calibration_matrix(central_model(s)) == calibration_matrix(cam)
        r = backproject(s, SVector(345.0, 210.0))
        @test r.origin == zero(r.origin)
        @test SVector{3}(r.direction) == backproject(cam, SVector(345.0, 210.0))
    end

    @testset "with_present, with_points, with_directions" begin
        mk = with_present(nc, G.masked.present)
        @test mk.present == G.masked.present
        @test all(isnan, mk.points.controls[1, 1]) && all(isnan, mk.directions.controls[1, 1])
        fin = 0
        for (i, u) in enumerate(pixels)
            r, ref = backproject(mk, u), col3(G.masked.backproject, i)
            if any(isnan, ref)
                @test all(isnan, r.origin) && all(isnan, r.direction)
            else
                fin += 1
                @test norm(SVector{3}(r.direction) - ref) <= 1.0e-12
                @test norm(SVector{3}(r.origin) - col3(G.noncentral.origins, i)) <= 1.0e-12
            end
        end
        @test fin >= 100
        @test with_present(nc, trues(size(b.controls))).present === nothing

        z = with_points(nc, zero(pts.controls))
        @test z.directions === nc.directions && z.intrinsics === nc.intrinsics
        @test all(iszero, z.points.controls) && z.points.area == pts.area
        d2 = with_directions(nc, map(d -> 2 * d, b.controls))
        @test d2.points === nc.points
        @test SVector{3}(backproject(d2, SVector(345.0, 210.0)).direction) ≈ SVector{3}(backproject(nc, SVector(345.0, 210.0)).direction) atol = 1.0e-14
        @test_throws ArgumentError with_points(nc, pts.controls[1:(end - 1), :])
    end

    @testset "precision and device forms: convert_eltype, adapt, mixed-precision splines" begin
        n32 = convert_eltype(Float32, nc)
        @test eltype(eltype(n32.directions.controls)) === Float32 && eltype(eltype(n32.points.controls)) === Float32
        @test n32.present === nc.present
        worst, nan_mismatch = 0.0, 0
        for u in pixels
            r, r32 = backproject(nc, u), backproject(n32, u)
            if all(isfinite, r.direction)
                nan_mismatch += !all(isfinite, r32.direction)
                all(isfinite, r32.direction) &&
                    (worst = max(worst, norm(SVector{3}(r32.direction) - SVector{3}(r.direction)), norm(SVector{3}(r32.origin) - SVector{3}(r.origin))))
            else
                nan_mismatch += all(isfinite, r32.direction)
            end
        end
        @test nan_mismatch == 0
        @test worst <= 1.0e-5
        a = adapt(Array, nc)
        @test a isa typeof(nc)
        @test all(u -> isequal(backproject(a, u), backproject(nc, u)), pixels)
        am = adapt(Array, with_present(nc, G.masked.present))
        @test am.present == G.masked.present
        # the direction Jacobian of a Float32 camera follows `backproject`'s precision —
        # Float32 for a Float32 pixel — and is the Float64 one to 1e-5
        c32 = convert_eltype(Float32, cam)
        worst_j, nan_mismatch_j, wrong_type = 0.0, 0, 0
        for u in pixels
            f, M = backproject_with_jacobian(cam, u)
            f32, M32 = backproject_with_jacobian(c32, SVector{2, Float32}(u))
            wrong_type += !(eltype(f32) === Float32 && eltype(M32) === Float32)
            wrong_type += eltype(first(backproject_with_jacobian(c32, u))) !== eltype(backproject(c32, u))
            nan_mismatch_j += all(isfinite, f) != all(isfinite, f32)
            all(isfinite, f) && all(isfinite, f32) && (worst_j = max(worst_j, norm(f32 - f), maximum(abs.(M32 - M))))
        end
        @test wrong_type == 0
        @test nan_mismatch_j == 0
        @test worst_j <= 1.0e-5
        mixed = NoncentralSplineModel(cam.intrinsics, b, convert_eltype(Float32, pts))
        @test typeof(mixed.points) === typeof(mixed.directions)
        @test eltype(eltype(mixed.points.controls)) === Float64
    end

    @testset "the lattice's support hooks, in pixels" begin
        @test support_trait(nc) isa LocalSupport
        @test nodes(nc) == CartesianIndices(size(b.controls))
        h = 1000 / (grid[1] - 3)
        for I in (CartesianIndex(1, 1), CartesianIndex(5, 7), CartesianIndex(grid[2], grid[1]))
            q = node_position(nc, I)
            @test q ≈ SVector((I[2] - 2) * h, (I[1] - 2) * h) atol = 1.0e-9
            s = support(nc, I)
            @test minimum(s)[1] >= -1.0e-9 && maximum(s)[1] <= 1000 + 1.0e-9
        end
    end

    @testset "a plain camera of the model is refused; the physical one is in metres" begin
        R1 = RotMatrix{3, Float64}(RotXYZ(0.01, -0.02, 0.005))
        t1 = SVector(0.012, -0.004, 0.0021)
        E1u = EuclideanMap(R1, t1 .* m)
        s = thrown_message(() -> PosedCamera(nc, EuclideanMap(R1, t1 .* 1000)))
        @test occursin("t .* m", s) && occursin("with_unit(extrinsics, mm)", s)
        @test_throws ArgumentError PosedCamera(nc, RotationMap(R1))
        Cu = PosedCamera(nc, E1u)
        @test has_units(Cu)
        @test ustrip(m, Cu).extrinsics === ustrip(m, E1u)
        @test has_units(with_unit(ustrip(m, Cu), m))
        @test_throws ArgumentError ustrip(mm, Cu)
        @test_throws ArgumentError with_unit(ustrip(m, Cu), mm)
        # the physical camera projects a point in metres as the plain model does
        Xw = SVector(0.05, -0.02, 1.2) .* m
        @test project(Cu, Xw) == project(nc, ustrip(m, E1u)(SVector(0.05, -0.02, 1.2)))
    end
end
