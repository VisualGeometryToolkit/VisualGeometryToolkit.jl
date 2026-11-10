# A spline lens's presence mask: the nodes it holds. Under the footprint rule
# `node_mask(cam, region)` every pixel of the region reads present nodes only,
# so the masked camera evaluates the region bit for bit as the full one, and
# outside the fully present windows it answers as outside its area (NaN).
@testitem "spline present nodes" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: KannalaBrandt, WholePlane, SplineLens, ChartSpline, NoncentralSplineModel, window,
        present_nodes, with_present, convert_eltype, backproject_with_jacobian
    using VisualGeometryToolkitCore.Cameras: _normalized_point
    using StaticArrays
    using LinearAlgebra: normalize

    W, H = 1280, 960
    fish = CameraModel(; focal = 250.0, pp = (640.0, 480.0), lens = KannalaBrandt(-4 / (3π^2)))
    disc = Ellipse(Point2(640.0, 480.0), 250π / 3, 250π / 3, 0.0)            # inside the frame
    cut = Ellipse(Point2(640.0, 480.0), 700.0, 420.0, 0.0)                    # cut by the frame's left and right
    m = CameraModel(SplineLens, fish, (W, H); grid_size = (24, 19), init = :quasi_interpolate)
    region(dom) = valid_domain(m) ∩ dom
    masked(dom) = with_present(m, node_mask(m, region(dom)))                  # NaN in the absent slots: any read shows
    imaged(c, p) = all(isfinite, backproject(c, p))
    raw(c, p) = c.lens.backward(_normalized_point(c.intrinsics, p))           # the backward spline, continued
    pixels = [SVector(x, y) for x in 0.25:9.5:(W - 0.25), y in 0.25:9.5:(H - 0.25)]

    @testset "footprint presence: a disc, an ellipse cut by the frame, the whole plane" begin
        for (dom, some_absent) in ((disc, true), (cut, true), (WholePlane(), false))
            mm = masked(dom)
            P = present_nodes(mm)
            @test size(P) == size(m.lens.backward.controls) && P == node_mask(m, region(dom))
            @test any(.!P) == some_absent
            @test (mm.lens.present === nothing) == !some_absent        # every node present is stored as `nothing`
            # every pixel of the region reads present nodes only: the masked camera evaluates it bit for bit, with NaN
            # in the absent slots
            R = region(dom)
            inside = filter(p -> Point2(p...) ∈ R, pixels)
            @test !isempty(inside)
            @test all(p -> imaged(mm, p), inside)
            @test all(p -> backproject(mm, p) == backproject(m, p), inside)
            @test all(p -> backproject_with_jacobian(mm, p) == backproject_with_jacobian(m, p), inside)
            @test all(p -> raw(mm, p) == raw(m, p), inside)
            # the absent nodes are never free
            @test node_mask(mm, WholePlane()) == P
            @test Set(free_nodes(mm, WholePlane())) == Set(findall(P))
        end
    end

    @testset "outside the fully present windows: no value, never a throw" begin
        mm = masked(disc)
        out = filter(p -> !imaged(mm, p), pixels)
        @test !isempty(out) && all(p -> imaged(m, p), out)
        @test all(p -> all(isnan, backproject(mm, p)), out)
        @test all(p -> all(isnan, backproject_with_jacobian(mm, p)[1]) && all(isnan, backproject_with_jacobian(mm, p)[2]), out)
        @test all(p -> all(isnan, unproject(mm, p, 1.0)), out)
        # a ray landing on such a pixel projects to no pixel; one inside the region to the full camera's pixel
        X_out = backproject(m, out[1])
        @test project(m, [X_out], [out[1]]; iters = 100).ok[1] && !project(mm, [X_out], [out[1]]; iters = 100).ok[1]
        @test all(isnan, project(mm, X_out))
        X_in = backproject(m, SVector(640.0, 480.0))
        @test project(mm, X_in) == project(m, X_in)
        # the batched forms agree with the scalar ones
        pts = [out[1:5]; [SVector(640.0, 480.0), SVector(600.0, 420.0)]]
        rays = backproject(mm, pts)
        @test all(i -> isequal(rays[i], backproject(mm, pts[i])), eachindex(pts))
        Xs = [backproject(m, p) for p in pts]
        b = project(mm, Xs, pts; iters = 100)
        @test all(i -> b.ok[i] == all(isfinite, project(mm, Xs[i])), eachindex(pts))
        @test count(b.ok) == 2
    end

    @testset "a window carries its block of the mask" begin
        mm = masked(disc)
        w = window(mm, 1:12, 1:12)
        @test present_nodes(w) == present_nodes(mm)[1:12, 1:12]
        corner = filter(p -> p[1] < 600 && p[2] < 450, pixels)
        @test all(p -> imaged(mm, p) || !imaged(w, p), corner)                    # an absent node: not imaged
        @test all(p -> !imaged(w, p) || isapprox(backproject(w, p), backproject(mm, p); atol = 1.0e-12), corner)
        @test any(p -> !imaged(mm, p), corner) && any(p -> imaged(w, p), corner)
    end

    @testset "the mask travels; it is checked" begin
        mm = masked(disc)
        @test present_nodes(convert_eltype(Float32, mm)) == present_nodes(mm)
        b = m.lens.backward
        @test SplineLens(b; present = trues(size(b.controls))).present === nothing
        @test_throws DimensionMismatch SplineLens(b; present = trues(3, 3))
        @test_throws ArgumentError SplineLens(b; present = falses(size(b.controls)))
        P = present_nodes(mm)
        nc = NoncentralSplineModel(m.intrinsics, b, ChartSpline(b.area, zero(b.controls)); present = P)
        @test present_nodes(nc) == P && present_nodes(convert_eltype(Float32, nc)) == P
        out = first(filter(p -> !imaged(mm, p), pixels))
        @test all(isnan, backproject(nc, out).direction) && all(isnan, backproject(nc, out).origin)
        @test all(isnan, backproject_with_jacobian(nc, out)[1])
        c = SVector(640.0, 480.0)
        @test backproject(nc, c).direction == backproject(mm, c)
        @test all(isnan, project(nc, backproject(m, out), out))      # seeded there: the window there is absent
    end

    @testset "GPU: device == host with a mask (availability-gated)" begin
        cuda = try
            mod = Base.require(Main, :CUDA)
            (mod !== nothing && Base.invokelatest(getproperty(mod, :functional))) ? mod : nothing
        catch
            nothing
        end
        if cuda === nothing
            @test_skip cuda !== nothing      # no functional GPU here: recorded as Broken, not as a success
        else
            CuArray = getproperty(cuda, :CuArray)
            mm = masked(disc)
            host = backproject(mm, pixels[1:2000])
            dev = Base.invokelatest(Array, Base.invokelatest(backproject, mm, Base.invokelatest(CuArray, pixels[1:2000])))
            @test isequal(dev, host)
            Xs = [normalize(SVector(p[1] - 640.0, p[2] - 480.0, 250.0)) for p in pixels[1:2000]]
            hb = project(mm, Xs, pixels[1:2000]; iters = 50)
            db = Base.invokelatest(project, mm, Base.invokelatest(CuArray, Xs), Base.invokelatest(CuArray, pixels[1:2000]); iters = 50)
            @test Base.invokelatest(Array, db.ok) == hb.ok
            # the camera's device copy holds the mask
            dm = Base.invokelatest(VisualGeometryToolkitCore.Adapt.adapt, CuArray, mm)
            @test Base.invokelatest(Array, dm.lens.present) == Array{Bool}(mm.lens.present)
        end
    end
end
