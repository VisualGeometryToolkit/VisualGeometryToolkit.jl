# Supports on the sensor (cameras/support.jl): which of a model's parameters a
# camera can see. A global model is all free; a lattice's node is free when its
# footprint meets the imaged region `Rect(extent) ∩ image_circle`.
@testitem "support" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: KannalaBrandt, Pinhole, WholePlane, RegionIntersection, SupportTrait,
        GlobalSupport, LocalSupport, support_trait, nodes, support, node_position, cubic_bspline_footprint, CONTRACT_GENERICS,
        Division, AbstractLensModel, spline_support, window_indices
    using StaticArrays
    using LinearAlgebra: normalize, dot

    cam(f, l) = CameraModel(; focal = f, pp = (640.0, 480.0), lens = l)
    fold = KannalaBrandt(-4 / (3π^2))                      # a 180° fisheye, r(90°) = π/3

    @testset "imaged_region: the extent ∩ the image circle" begin
        bc = BoundedCamera(cam(250.0, fold), Size2(1280, 960))
        R = imaged_region(bc)
        @test R isa RegionIntersection
        @test R.a == Rect(0, 0, 1280, 960) && R.b == image_circle(bc)
        @test Point2(640.0, 480.0) ∈ R && Point2(5.0, 5.0) ∉ R
        @test imaged_region(BoundedCamera(cam(800.0, Pinhole()), Size2(1280, 960))) === Rect(0, 0, 1280, 960)
    end

    @testset "the trait and its contract" begin
        @test support_trait(fold) === GlobalSupport() && support_trait(typeof(fold)) === GlobalSupport()
        @test support_trait(cam(800.0, Pinhole())) === GlobalSupport()
        @test support_trait(CentralGenericModel) === LocalSupport()
        @test support_trait(NoncentralGenericModel) === LocalSupport()
        @test all(f -> any(g -> g === f, CONTRACT_GENERICS), (support_trait, nodes, support, node_position))
        @test node_position(fold, 1) === nothing
        # a global model: every parameter is free, one support covering everything
        R = imaged_region(BoundedCamera(cam(250.0, fold), Size2(1280, 960)))
        @test free_nodes(fold, R) === Colon()
        @test node_mask(fold, R) === Colon()
        θ = collect(1.0:5.0)
        @test θ[free_nodes(fold, R)] == θ[node_mask(fold, R)] == θ          # one answer, usable as an index
        # the uniform cubic B-spline footprint: center ± 2 cells
        @test cubic_bspline_footprint((100.0, 50.0), (10.0, 5.0)) == Rect2(80.0, 40.0, 40.0, 20.0)
    end

    @testset "a circular fisheye's grid model: nodes frozen where the footprint misses the circle" begin
        g = CentralGenericModel(cam(250.0, fold), (1280, 960); grid_size = (20, 16))
        R = imaged_region(BoundedCamera(cam(250.0, fold), Size2(1280, 960)))
        @test nodes(g) == CartesianIndices((16, 20))
        m = node_mask(g, R)
        @test size(m) == size(g.grid) == (16, 20)
        @test sum(m) == 108                                    # of 320
        @test free_nodes(g, R) == [i for i in nodes(g) if m[i]]
        # frozen exactly where the footprint misses the circle: brute-force sampling of each footprint
        q = image_circle(BoundedCamera(cam(250.0, fold), Size2(1280, 960)))
        for i in nodes(g)
            s = support(g, i)
            pts = [Point2(x, y) for x in range(minimum(s)[1], maximum(s)[1]; length = 31), y in range(minimum(s)[2], maximum(s)[2]; length = 31)]
            m[i] || @test !any(p -> p ∈ q, pts)
            m[i] && @test any(p -> p ∈ q, pts)
        end
        # the node sits at the centre of its footprint (interior nodes)
        i = CartesianIndex(8, 10)
        @test node_position(g, i) ≈ Point2((minimum(support(g, i)) .+ maximum(support(g, i))) ./ 2...)
        # the noncentral model has the same lattice
        gn = NoncentralGenericModel(cam(250.0, fold), (1280, 960); grid_size = (20, 16))
        @test node_mask(gn, R) == m
        # a rectilinear lens images the whole extent: no node is frozen
        gr = CentralGenericModel(cam(800.0, Pinhole()), (1280, 960); grid_size = (20, 16))
        @test all(node_mask(gr, imaged_region(BoundedCamera(cam(800.0, Pinhole()), Size2(1280, 960)))))
    end

    @testset "the sampler clamps nodes inside the lit field" begin
        # every node of the circular fisheye's grid backprojects finitely, outside the circle too
        g = CentralGenericModel(cam(250.0, fold), (1280, 960); grid_size = (20, 16))
        @test all(v -> all(isfinite, v), g.grid)
        # the margin is a keyword (0.97 by default; 1 would sample the corner nodes AT the fold, which rings)
        g1 = CentralGenericModel(cam(250.0, fold), (1280, 960); grid_size = (20, 16), sampling_margin = 0.9)
        @test g1.grid != g.grid
    end

    @testset "a lift that never folds: sampled past 90°, finite and accurate" begin
        src = cam(300.0, Division(-0.3))                      # images the whole plane: no clamp
        @test image_circle(src) === WholePlane()
        g = CentralGenericModel(src, (1280, 960); grid_size = (20, 16))
        @test all(v -> all(isfinite, v), g.grid)
        worst = 0.0                                           # angular error × f, in px
        for x in 5:20:1275, y in 5:20:955
            d0 = normalize(SVector{3, Float64}(backproject(src, SVector(x + 0.0, y + 0.0))))
            d1 = normalize(SVector{3, Float64}(backproject(g, SVector(x + 0.0, y + 0.0))))
            worst = max(worst, acos(clamp(dot(d0, d1), -1, 1)) * 300)
        end
        @test worst < 0.2                                     # measured 0.13 px
    end

    @testset "a lens whose circle is undefined: the sampler names itself" begin
        struct BlindLens <: AbstractLensModel end             # images nothing
        VisualGeometryToolkitCore.project(::BlindLens, X::StaticVector{3}) = SVector(NaN, NaN)
        e = try
            CentralGenericModel(cam(300.0, BlindLens()), (640, 480); grid_size = (8, 6)); nothing
        catch err
            err
        end
        @test e isa ArgumentError
        @test occursin("generic model", e.msg) && occursin("BlindLens", e.msg) && occursin("sampler", e.msg)
    end

    @testset "a node's support is the spline window's footprint" begin
        # every pixel's 16 window controls have the pixel in their support, and no
        # node outside the window has the pixel strictly inside its support —
        # on the default calibrated area and on a non-default one
        for g in (
                CentralGenericModel(cam(250.0, fold), (1280, 960); grid_size = (20, 16)),
                CentralGenericModel(cam(250.0, fold), (1280, 960); grid_size = (9, 7), calib_area = (100, 50, 1099, 899)),
            )
            missing_from_footprint = 0
            outside_window = 0
            for x in 0.5:37.3:1279.0, y in 0.5:29.1:959.0
                w = spline_support(g, SVector(x, y))
                w === nothing && continue
                win = Set(window_indices(g, w...))
                for I in nodes(g)
                    s = support(g, I)
                    closed = minimum(s)[1] <= x <= maximum(s)[1] && minimum(s)[2] <= y <= maximum(s)[2]
                    strict = minimum(s)[1] < x < maximum(s)[1] && minimum(s)[2] < y < maximum(s)[2]
                    (I in win && !closed) && (missing_from_footprint += 1)
                    (strict && !(I in win)) && (outside_window += 1)
                end
            end
            @test missing_from_footprint == 0
            @test outside_window == 0
        end
        # the clip matters on a smaller calibrated area: no support reaches outside it
        g = CentralGenericModel(cam(250.0, fold), (1280, 960); grid_size = (9, 7), calib_area = (100, 50, 1099, 899))
        @test all(I -> minimum(support(g, I))[1] >= 100 && maximum(support(g, I))[1] <= 1100, nodes(g))
    end
end

@testitem "support: the boundary rule — dependent nodes outside the imaged region" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: KannalaBrandt, Pinhole, BoundaryRule, extend_controls, boundary_rule_jacobian, nodes,
        spline_support, spline_weights, interpolated_direction
    using StaticArrays, ForwardDiff
    using Random: Xoshiro

    cam(f, l) = CameraModel(; focal = f, pp = (640.0, 480.0), lens = l)
    fold = KannalaBrandt(-4 / (3π^2))                      # a 180° fisheye: its image circle lies inside the sensor
    g = CentralGenericModel(cam(250.0, fold), (1280, 960); grid_size = (20, 16))
    R = imaged_region(BoundedCamera(cam(250.0, fold), Size2(1280, 960)))
    rule = BoundaryRule(g, R)
    @test rule.interior == node_mask(g, R) && count(rule.interior) == 108
    @test sort(vcat(findall(rule.interior), rule.dependent)) == vec(collect(nodes(g)))
    @test all(tm -> sum(last, tm) ≈ 1 && all(k -> rule.interior[first(k)], tm), rule.terms)
    J = boundary_rule_jacobian(rule)
    @test size(J) == (320, 108) && all(sum(J; dims = 2) .≈ 1)
    # zero second difference across the border: an affine correction of the lattice is extended exactly
    ref = g.grid
    aff = [ref[i] + SVector(0.01 * i[1] - 0.02 * i[2], 0.003 * i[2], 0.002 * i[1] + 0.001) for i in nodes(g)]
    @test extend_controls(rule, aff, ref) ≈ aff
    @test extend_controls(rule, ref, ref) == ref

    # a refined interior: the region's pixels read interior nodes only — whatever the dependent nodes hold —
    # and the interior weights sum to one at every imaged pixel (partition of unity in the last cells)
    rng = Xoshiro(3)
    θ = [rule.interior[i] ? ref[i] + 0.02 * (SVector{3}(rand(rng, 3)) .- 0.5) : ref[i] for i in nodes(g)]
    full = extend_controls(rule, θ, ref)
    junk = [rule.interior[i] ? θ[i] : SVector(5.0, -3.0, 1.0) for i in nodes(g)]
    gm(grid) = CentralGenericModel(g.width, g.height, g.calib_min_x, g.calib_min_y, g.calib_max_x, g.calib_max_y, grid; unit_controls = false)
    mf, mj = gm(full), gm(junk)
    q = image_circle(BoundedCamera(cam(250.0, fold), Size2(1280, 960)))
    rim = [SVector{2, Float64}(q.center + 0.9999 * (SVector{2, Float64}(boundary_point(q, t)) - q.center)) for t in range(0, 2π; length = 721)]
    imaged = vcat([SVector(x, y) for x in 0.5:9:1279.5, y in 0.5:9:959.5 if Point2(x, y) ∈ R], filter(p -> Point2(p...) ∈ R, rim))
    @test length(imaged) > 2000 + 700                                      # a raster and the circle's rim
    @test all(p -> interpolated_direction(mf, p) == interpolated_direction(mj, p), imaged)
    for p in imaged
        ix, iy = spline_support(g, p)
        w, _, _ = spline_weights(g, ix, iy, p)
        @test sum(w[r, c] for r in 1:4, c in 1:4 if rule.interior[iy - 3 + r, ix - 3 + c]) ≈ 1 atol = 1.0e-12
    end

    # the Jacobian w.r.t. the interior chains through the rule: ForwardDiff through extend_controls at pixels past the circle
    inner = findall(rule.interior)
    outside = [SVector(30.0, 40.0), SVector(1250.0, 900.0), SVector(640.0, 12.0)]
    @test all(p -> Point2(p...) ∉ R, outside)
    for p in outside, comp in 1:3
        ix, iy = spline_support(g, p)
        w, _, _ = spline_weights(g, ix, iy, p)
        val(t) = (c = extend_controls(rule, (z = [x[comp] for x in θ]; zz = Matrix{eltype(t)}(z); zz[inner] .= t; zz), [x[comp] for x in ref]);
                  sum(w[r, c2] * c[iy - 3 + r, ix - 3 + c2] for r in 1:4, c2 in 1:4))
        wfull = zeros(320); L = LinearIndices(rule.interior)
        for r in 1:4, c2 in 1:4
            wfull[L[iy - 3 + r, ix - 3 + c2]] += w[r, c2]
        end
        @test ForwardDiff.gradient(val, [θ[k][comp] for k in inner]) ≈ J' * wfull rtol = 1.0e-12
    end
    # the region must meet the lattice
    @test_throws ArgumentError BoundaryRule(falses(4, 4))
end
