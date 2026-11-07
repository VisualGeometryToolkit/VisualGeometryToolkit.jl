# A grid model's presence mask: the nodes it holds. Under the footprint rule
# `node_mask(m, imaged_region(m))` every imaged pixel reads present nodes only,
# so the masked model evaluates the region bit for bit as the full one, and
# outside the complete-window domain it answers as outside the calibrated area.
@testitem "generic_present_nodes" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: KannalaBrandt, WholePlane, interpolated_direction, interpolated_point, spline_support,
        spline_weights, window, present_nodes, in_present_domain, convert_eltype, backproject_with_jacobian
    using StaticArrays
    using LinearAlgebra: normalize

    W, H = 1280, 960
    fish = CameraModel(; focal = 250.0, pp = (640.0, 480.0), lens = KannalaBrandt(-4 / (3π^2)))
    disc = Ellipse(Point2(640.0, 480.0), 250π / 3, 250π / 3, 0.0)            # inside the frame
    cut = Ellipse(Point2(640.0, 480.0), 700.0, 420.0, 0.0)                    # cut by the frame's left and right
    full(domain) = CentralGenericModel(fish, (W, H); grid_size = (24, 19), init = :quasi_interpolate, domain)
    masked(m; grid = m.grid) = CentralGenericModel(W, H, m.calib_min_x, m.calib_min_y, m.calib_max_x, m.calib_max_y, grid;
                                                   unit_controls = false, m.nominal, m.domain,
                                                   present = node_mask(m, imaged_region(m)))
    pixels = [SVector(x, y) for x in 0.25:9.5:(W - 0.25), y in 0.25:9.5:(H - 0.25)]
    # absent slots set to NaN: any read of one shows in the result
    nan_absent(m) = (g = copy(m.grid); g[.!node_mask(m, imaged_region(m))] .= Ref(Vec3{Float64}(NaN, NaN, NaN)); g)

    @testset "footprint presence: a disc, an ellipse cut by the frame, the whole plane" begin
        for (dom, some_absent) in ((disc, true), (cut, true), (WholePlane(), false))
            m = full(dom)
            mm = masked(m)
            P = present_nodes(mm)
            @test size(P) == size(m.grid) && P == node_mask(m, imaged_region(m))
            @test any(.!P) == some_absent
            @test (mm.present === nothing) == !some_absent        # every node present is stored as `nothing`
            # every imaged pixel reads present nodes only: the masked model evaluates it bit for bit, even with NaN in the
            # absent slots
            mn = masked(m; grid = nan_absent(m))
            R = imaged_region(m)
            inside = filter(p -> Point2(p...) ∈ R, pixels)
            @test !isempty(inside)
            @test all(p -> in_present_domain(mn, p), inside)
            @test all(p -> backproject(mn, p) == backproject(m, p), inside)
            @test all(p -> backproject_with_jacobian(mn, p) == backproject_with_jacobian(m, p), inside)
            @test all(p -> interpolated_direction(mn, p) == interpolated_direction(m, p), inside)
            @test all(p -> spline_support(mn, p) == spline_support(m, p), inside)
            # the absent nodes are never free
            @test node_mask(mm, WholePlane()) == P
            @test Set(free_nodes(mm, WholePlane())) == Set(findall(P))
        end
    end

    @testset "outside the complete-window domain: no value, never a throw" begin
        m = full(disc)
        mm = masked(m)
        out = filter(p -> !in_present_domain(mm, p), pixels)
        @test !isempty(out) && all(p -> in_present_domain(m, p), out)
        @test all(p -> all(isnan, backproject(mm, p)), out)
        @test all(p -> all(isnan, backproject_with_jacobian(mm, p)[1]) && all(isnan, backproject_with_jacobian(mm, p)[2]), out)
        @test all(p -> all(isnan, interpolated_direction(mm, p)), out)
        @test all(p -> spline_support(mm, p) === nothing, out)
        @test all(p -> unproject(mm, p, 1.0) === nothing, out)
        ix, iy = spline_support(m, out[1])
        @test_throws ArgumentError spline_weights(mm, ix, iy, out[1])
        # a ray landing on such a pixel projects to nothing; one inside the region to the full model's pixel
        X_out = backproject(m, out[1])
        @test project(m, X_out, out[1]) !== nothing && project(mm, X_out, out[1]) === nothing
        X_in = backproject(m, SVector(640.0, 480.0))
        @test project(mm, X_in) == project(m, X_in)
        # the batched forms agree with the scalar ones
        pts = [out[1:5]; [SVector(640.0, 480.0), SVector(600.0, 420.0)]]
        rays = backproject(mm, pts)
        @test all(i -> isequal(rays[i], backproject(mm, pts[i])), eachindex(pts))
        Xs = [backproject(m, p) for p in pts]
        b = project(mm, Xs, pts; iters = 100)
        @test all(i -> b.ok[i] == (project(mm, Xs[i], pts[i]) !== nothing), eachindex(pts))
        @test count(b.ok) == 2
    end

    @testset "a window carries its block of the mask" begin
        m = full(disc)
        mm = masked(m; grid = nan_absent(m))
        w = window(mm, 1:12, 1:12)
        @test present_nodes(w) == present_nodes(mm)[1:12, 1:12]
        corner = filter(p -> p[1] < 600 && p[2] < 450, pixels)
        @test all(p -> in_present_domain(mm, p) || all(isnan, interpolated_direction(w, p)), corner)   # an absent node: not imaged
        @test all(p -> (v = interpolated_direction(w, p); all(isnan, v) || v == interpolated_direction(mm, p)), corner)
        @test any(p -> !in_present_domain(mm, p), corner) && any(p -> all(isfinite, interpolated_direction(w, p)), corner)
    end

    @testset "the mask travels; it is checked" begin
        m = full(disc)
        mm = masked(m)
        @test present_nodes(convert_eltype(Float32, mm)) == present_nodes(mm)
        @test CentralGenericModel(W, H, 0, 0, W - 1, H - 1, m.grid; present = trues(size(m.grid))).present === nothing
        @test_throws DimensionMismatch CentralGenericModel(W, H, 0, 0, W - 1, H - 1, m.grid; present = trues(3, 3))
        @test_throws ArgumentError CentralGenericModel(W, H, 0, 0, W - 1, H - 1, m.grid; present = falses(size(m.grid)))
        P = present_nodes(mm)
        nc = NoncentralGenericModel(W, H, 0, 0, W - 1, H - 1, m.grid, fill(Vec3{Float64}(0, 0, 0), size(m.grid));
                                    unit_controls = false, present = P)
        @test present_nodes(nc) == P && present_nodes(convert_eltype(Float32, nc)) == P
        out = first(filter(p -> !in_present_domain(mm, p), pixels))
        @test !in_present_domain(nc, out) && all(isnan, backproject(nc, out).direction) && all(isnan, backproject(nc, out).origin)
        @test all(isnan, interpolated_point(nc, out)) && all(isnan, backproject_with_jacobian(nc, out)[1])
        c = SVector(640.0, 480.0)
        @test backproject(nc, c).direction == backproject(mm, c)
        @test project(nc, backproject(m, out), out) === nothing      # seeded there: a line through c has two directions
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
            m = full(disc)
            mm = masked(m)
            host = backproject(mm, pixels[1:2000])
            dev = Base.invokelatest(Array, Base.invokelatest(backproject, mm, Base.invokelatest(CuArray, pixels[1:2000])))
            @test isequal(dev, host)
            Xs = [normalize(SVector(p[1] - 640.0, p[2] - 480.0, 250.0)) for p in pixels[1:2000]]
            hb = project(mm, Xs, pixels[1:2000]; iters = 50)
            db = Base.invokelatest(project, mm, Base.invokelatest(CuArray, Xs), Base.invokelatest(CuArray, pixels[1:2000]); iters = 50)
            @test Base.invokelatest(Array, db.ok) == hb.ok
            # the model's device copy holds the mask
            dm = Base.invokelatest(VisualGeometryToolkitCore.Adapt.adapt, CuArray, mm)
            @test Base.invokelatest(Array, dm.present) == Array{Bool}(mm.present)
        end
    end
end
