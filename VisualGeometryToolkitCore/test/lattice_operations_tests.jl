# Exact lattice operations on a SplineLens camera (cameras/lattice_operations.jl): the
# subdivision by integers is the same function (against a literal Boehm knot insertion),
# the extension keeps the source's nodes bit for bit and continues them by the bending
# energy's minimiser, and both carry the presence mask, valid_domain and the marks.

@testsetup module LatticeOps
    import VisualGeometryToolkitCore as VgtkCore
    using StaticArrays: SVector, SMatrix
    using LinearAlgebra: norm, cross, dot
    using Random: AbstractRNG
    export random_camera, raw, angle_between, boehm_refine, pixels_in, second_differences, ring_of

    # A SplineLens camera of `gx × gy` random direction controls (near ẑ, a smooth trend plus
    # noise) over a calibrated area whose cells are not square, with a supplied K.
    function random_camera(rng::AbstractRNG, gx, gy; present = nothing)
        K = VgtkCore.LogicalIntrinsics(VgtkCore.CameraCalibrationMat(SMatrix{3, 3, Float64}(612.0, 0, 0, 0, 587.0, 0, 503.5, 391.25, 1)))
        C = [SVector(0.9 * (ix - gx / 2) / gx + 0.05 * randn(rng), 0.8 * (iy - gy / 2) / gy + 0.05 * randn(rng), 1 + 0.05 * randn(rng))
             for iy in 1:gy, ix in 1:gx]
        return VgtkCore.CameraModel(VgtkCore.SplineLens, C, VgtkCore.Size2(1000, 760);
            calib_area = VgtkCore.Rect(37, 21, 901, 703), intrinsics = K, present)
    end

    # The backward spline at a pixel, continued past its area (its raw 3-vector).
    raw(c, p) = c.lens.backward(VgtkCore.normalized_point(c.intrinsics, p))
    angle_between(a, b) = atan(norm(cross(a, b)), dot(a, b))
    function pixels_in(rng, r, n)
        o = VgtkCore.GeometryBasics.origin(r); w = VgtkCore.GeometryBasics.widths(r)
        return [SVector(o[1] + rand(rng) * w[1], o[2] + rand(rng) * w[2]) for _ in 1:n]
    end

    # A literal Boehm (1980) knot insertion: the cubic spline of `c` on the uniform knots −3 … n
    # (node j at j − 2), padded by three zero coefficients each side, the knots s + i/m inserted
    # one at a time into every span of [−3, n]; the fine coefficients of the bases on the knots
    # (j′ − 4)/m … j′/m, j′ = 1 … m(n − 3) + 3.
    function boehm_refine(c::AbstractVector{<:Real}, m::Integer)
        n, p = length(c), 3
        t = collect(Float64, -6:(n + 3))
        d = [zeros(3); Float64.(c); zeros(3)]
        for s in -3:(n - 1), i in 1:(m - 1)
            x = s + i / m
            k = findlast(<=(x), t)
            new = Vector{Float64}(undef, length(d) + 1)
            for j in eachindex(new)
                if j <= k - p
                    new[j] = d[j]
                elseif j <= k
                    α = (x - t[j]) / (t[j + p] - t[j])
                    new[j] = (1 - α) * d[j - 1] + α * d[j]
                else
                    new[j] = d[j - 1]
                end
            end
            insert!(t, k + 1, x); d = new
        end
        return [d[findfirst(τ -> abs(τ - (jf - 4) / m) < 1.0e-9, t)] for jf in 1:(m * (n - 3) + 3)]
    end

    # The central second differences of a control lattice `[gy, gx]` at node (i, j): along x
    # (the columns), along y (the rows) and mixed, their largest component.
    function second_differences(C, i, j)
        xx = C[i, j - 1] - 2C[i, j] + C[i, j + 1]
        yy = C[i - 1, j] - 2C[i, j] + C[i + 1, j]
        xy = (C[i + 1, j + 1] - C[i + 1, j - 1] - C[i - 1, j + 1] + C[i - 1, j - 1]) / 4
        return maximum(abs, xx), maximum(abs, yy), maximum(abs, xy)
    end
    # The Chebyshev ring of node (i, j) around the block `rows × cols` (0 inside it).
    ring_of(i, j, rows, cols) = max(first(rows) - i, i - last(rows), first(cols) - j, j - last(cols), 0)
end

@testitem "lattice operations: subdivision is exact" setup = [LatticeOps] begin
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: subdivide_lattice, lattice_spacing, valid_domain, ChartSpline, Rect2
    using StaticArrays: SVector
    using LinearAlgebra: norm
    using Random: MersenneTwister

    rng = MersenneTwister(108)
    @testset "the same function, $(gx)×$(gy), m = $m" for (gx, gy) in ((16, 18), (26, 20)), m in ((1, 1), (2, 3), (3, 2), (5, 4))
        cam = random_camera(rng, gx, gy)
        r = subdivide_lattice(cam, m)
        fine = r.camera
        @test size(fine.lens.backward.controls) == (m[2] * (gy - 3) + 3, m[1] * (gx - 3) + 3)
        @test fine.intrinsics == cam.intrinsics && fine.lens.backward.area == cam.lens.backward.area
        @test all(lattice_spacing(fine) .≈ lattice_spacing(cam) ./ m)
        @test valid_domain(fine) == valid_domain(cam)
        @test !any(r.extrapolated) && size(r.extrapolated) == size(fine.lens.backward.controls)
        dom = valid_domain(cam)
        pts = [pixels_in(rng, dom, 400); [SVector(37.0, 21.0), SVector(938.0, 724.0)]]
        @test maximum(p -> norm(raw(fine, p) - raw(cam, p)) / norm(raw(cam, p)), pts) <= 1.0e-12
        @test maximum(p -> angle_between(VgtkCore.backproject(fine, p), VgtkCore.backproject(cam, p)), pts) <= 1.0e-9
        # the continuation past the area is the same cubic
        past = pixels_in(rng, Rect2(0.0, 0.0, 1000.0, 760.0), 200)
        @test maximum(p -> norm(raw(fine, p) - raw(cam, p)) / norm(raw(cam, p)), past) <= 1.0e-12
        @test m != (1, 1) || fine.lens.backward.controls == cam.lens.backward.controls
    end

    @testset "against a literal Boehm knot insertion" begin
        for (n1, n2, mx, my) in ((7, 9, 3, 2), (5, 11, 5, 4), (6, 6, 2, 3))
            u, v = randn(rng, n1), randn(rng, n2)
            cs = ChartSpline(Rect2(-1.0, 0.5, 2.0, 1.25), [SVector(u[i] * v[j]) for i in 1:n1, j in 1:n2])
            fine = subdivide_lattice(cs, (mx, my))
            @test fine.area == cs.area
            @test maximum(abs, first.(fine.controls) - boehm_refine(u, my) * boehm_refine(v, mx)') <= 1.0e-13
        end
        @test boehm_refine(collect(1.0:6.0), 1) == collect(1.0:6.0)
    end

    @testset "the factors" begin
        cam = random_camera(rng, 8, 7)
        @test subdivide_lattice(cam, 2).camera.lens.backward.controls == subdivide_lattice(cam, (2, 2)).camera.lens.backward.controls
        @test_throws ArgumentError subdivide_lattice(cam, (0, 2))
        @test_throws ArgumentError subdivide_lattice(cam.lens.backward, -1)
        @test_throws DimensionMismatch subdivide_lattice(cam, 2; extrapolated = falses(3, 3))
        # a forward spline is kept: the backward function has not changed
        fw = VgtkCore.SplineLens(cam.lens.backward; forward = ChartSpline(Rect2(-1.0, -1.0, 2.0, 2.0), fill(SVector(0.0, 0.0), 4, 4)))
        @test subdivide_lattice(VgtkCore.CameraModel(cam.intrinsics, fw), 3).camera.lens.forward === fw.forward
    end
end

@testitem "lattice operations: extension" setup = [LatticeOps] begin
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: extend_lattice, lattice_spacing, valid_domain, present_nodes, Rect2, node_position
    using StaticArrays: SVector
    using LinearAlgebra: norm, normalize
    using Random: MersenneTwister

    rng = MersenneTwister(1081)
    cam = random_camera(rng, 16, 18)
    hx, hy = lattice_spacing(cam)
    A = valid_domain(cam)                                   # the calibrated pixel area [37, 938] × [21, 724]
    ao = VgtkCore.GeometryBasics.origin(A); aw = VgtkCore.GeometryBasics.widths(A)
    # 1.7 cells past the left edge, 0.3 past the top, 2.5 past the right, inside at the bottom
    region = Rect2(ao[1] - 1.7hx, ao[2] - 0.3hy, aw[1] + 4.2hx, aw[2] - 2hy)
    e = extend_lattice(cam, region)
    C, C0 = e.camera.lens.backward.controls, cam.lens.backward.controls
    rows, cols = e.offset[1] .+ (1:18), e.offset[2] .+ (1:16)

    @testset "the lattice: its own spacing and phase, the needed nodes" begin
        # a node at cell t (from the area's origin) is needed when (t − 2, t + 2) meets the region
        @test e.offset == (1, 2)                             # top: t > −2.3 → t ≥ −2; left: t > −3.7 → t ≥ −3
        @test size(C) == (18 + 1, 16 + 2 + 3)                # right: t < 13 + 2.5 + 2 → t ≤ 17, the source's last node at 14
        @test all(lattice_spacing(e.camera) .≈ (hx, hy))
        @test e.camera.intrinsics == cam.intrinsics && e.camera.lens.forward === nothing
        @test node_position(e.camera, CartesianIndex(rows[1], cols[1])) ≈ node_position(cam, CartesianIndex(1, 1))
        @test e.extrapolated == .!(BitMatrix([i in rows && j in cols for i in axes(C, 1), j in axes(C, 2)]))
        @test e.camera.lens.present === nothing
        @test all(v -> all(isfinite, v), C)
        # the region lies in the extended camera's domain
        V = valid_domain(e.camera)
        @test VgtkCore.GeometryBasics.origin(V)[1] <= region.origin[1] + 1.0e-9 && VgtkCore.GeometryBasics.origin(V)[2] <= region.origin[2] + 1.0e-9
        @test V.origin[1] + V.widths[1] >= region.origin[1] + region.widths[1] - 1.0e-9
    end

    @testset "inside the source's domain: unchanged; the source nodes bit-identical" begin
        @test C[rows, cols] == C0
        pts = [pixels_in(rng, A, 500); [SVector(37.0, 21.0), SVector(938.0, 724.0), SVector(37.0, 724.0)]]
        @test maximum(p -> norm(raw(e.camera, p) - raw(cam, p)) / norm(raw(cam, p)), pts) <= 1.0e-12
        @test maximum(p -> angle_between(VgtkCore.backproject(e.camera, p), VgtkCore.backproject(cam, p)), pts) <= 1.0e-12
        # a region inside the area, or the area itself, adds nothing: the camera as it was
        for r in (A, Rect2(100.0, 100.0, 50.0, 50.0))
            n = extend_lattice(cam, r)
            @test n.camera === cam && n.offset == (0, 0) && !any(n.extrapolated)
        end
        @test_throws ArgumentError extend_lattice(cam, Rect2(0.0, 0.0, NaN, 1.0))
    end

    @testset "C¹ and C² across the domain's edge (one-sided differences)" begin
        # each side of the edge is one cubic piece, so the 4-point one-sided stencils are exact
        δ = 0.05hx
        f(x, y) = raw(e.camera, SVector(x, y))
        for y in range(ao[2] + 10, ao[2] + aw[2] - 10; length = 7), (x0, s) in ((ao[1], -1), (ao[1] + aw[1], 1))
            fo = [f(x0 + s * k * δ, y) for k in 0:3]             # outside
            fi = [f(x0 - s * k * δ, y) for k in 0:3]             # inside
            d1o = s * (-11fo[1] + 18fo[2] - 9fo[3] + 2fo[4]) / (6δ)
            d1i = -s * (-11fi[1] + 18fi[2] - 9fi[3] + 2fi[4]) / (6δ)
            d2o = (2fo[1] - 5fo[2] + 4fo[3] - fo[4]) / δ^2
            d2i = (2fi[1] - 5fi[2] + 4fi[3] - fi[4]) / δ^2
            @test norm(d1o - d1i) <= 1.0e-6 * norm(d1i)
            @test norm(d2o - d2i) <= 1.0e-5 * max(norm(d2i), 1 / hx^2 * 1.0e-6)
            # the inside derivatives are the source's own
            gi = [raw(cam, SVector(x0 - s * k * δ, y)) for k in 0:3]
            @test norm((2gi[1] - 5gi[2] + 4gi[3] - gi[4]) / δ^2 - d2i) <= 1.0e-6 * max(norm(d2i), 1.0e-12)
        end
    end

    @testset "the curvature does not grow outward, in every direction, corners included" begin
        big = extend_lattice(cam, Rect2(ao[1] - 4.5hx, ao[2] - 4.5hy, aw[1] + 9hx, aw[2] + 9hy))
        B = big.camera.lens.backward.controls
        br, bc = big.offset[1] .+ (1:18), big.offset[2] .+ (1:16)
        R = maximum(ring_of(i, j, br, bc) for i in axes(B, 1), j in axes(B, 2))
        @test R >= 5
        ring_max(r, k) = maximum(second_differences(B, i, j)[k] for i in 2:(size(B, 1) - 1), j in 2:(size(B, 2) - 1) if ring_of(i, j, br, bc) == r)
        corner_max(r, k) = maximum(second_differences(B, i, j)[k] for i in 2:(size(B, 1) - 1), j in 2:(size(B, 2) - 1)
            if ring_of(i, j, br, bc) == r && !(i in br) && !(j in bc))
        for k in 1:3                                       # along x, along y, mixed
            edge = ring_max(0, k)
            m = [ring_max(r, k) for r in 1:(R - 1)]
            @test all(<=(edge * (1 + 1.0e-9)), m)
            @test all(diff(m) .<= 1.0e-12)                 # non-increasing outward
            @test m[end] < m[1]
            c = [corner_max(r, k) for r in 1:(R - 1)]
            @test all(<=(edge * (1 + 1.0e-9)), c)
            @test all(diff(c) .<= 1.0e-12)
        end
    end

    @testset "an affine lattice is continued exactly" begin
        aff(iy, ix) = SVector(0.3 + 0.02ix - 0.01iy, -0.2 + 0.015iy + 0.004ix, 1.0 - 0.003ix + 0.002iy)
        lin = VgtkCore.with_directions(cam, [aff(iy, ix) for iy in 1:18, ix in 1:16])
        el = extend_lattice(lin, region)
        L = el.camera.lens.backward.controls
        @test maximum(norm(L[i, j] - aff(i - el.offset[1], j - el.offset[2])) for i in axes(L, 1), j in axes(L, 2)) <= 1.0e-12
    end
end

@testitem "lattice operations: the presence mask and valid_domain" setup = [LatticeOps] begin
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: extend_lattice, subdivide_lattice, valid_domain, present_nodes, lattice_spacing, Rect2, node_mask
    using StaticArrays: SVector
    using LinearAlgebra: norm
    using Random: MersenneTwister

    rng = MersenneTwister(1082)
    P = trues(20, 26)
    P[1:4, 1:5] .= false; P[18:20, 22:26] .= false; P[10, 1] = false   # two corners and a border node absent
    cam = random_camera(rng, 26, 20; present = P)
    @test present_nodes(cam) == P
    A = valid_domain(cam)
    imaged(c, p) = all(isfinite, VgtkCore.backproject(c, p))
    pts = pixels_in(rng, Rect2(0.0, 0.0, 1000.0, 760.0), 3000)

    @testset "subdivision: a fine node is present when every coarse node it reads is" begin
        for m in ((2, 3), (3, 2), (5, 4))
            fine = subdivide_lattice(cam, m).camera
            @test valid_domain(fine) == A
            @test all(p -> imaged(fine, p) == imaged(cam, p), pts)
            @test maximum(p -> angle_between(VgtkCore.backproject(fine, p), VgtkCore.backproject(cam, p)), filter(p -> imaged(cam, p), pts)) <= 1.0e-9
            F = present_nodes(fine)
            @test any(.!F) && all(i -> all(isfinite, fine.lens.backward.controls[i]) == F[i], eachindex(F))
        end
    end

    @testset "extension: the needed absent nodes filled and marked; the rest stays absent" begin
        hx, hy = lattice_spacing(cam)
        ao = VgtkCore.GeometryBasics.origin(A)
        # past the top-left corner (absent), not reaching the bottom-right one
        region = Rect2(ao[1] - 1.5hx, ao[2] - 1.5hy, 400.0, 300.0)
        e = extend_lattice(cam, region)
        Pe = present_nodes(e.camera)
        rows, cols = e.offset[1] .+ (1:20), e.offset[2] .+ (1:26)
        @test Pe[rows, cols][P] == trues(count(P))                      # the present source nodes stay present
        @test e.camera.lens.backward.controls[rows, cols][P] == cam.lens.backward.controls[P]
        @test Pe[rows, cols][18:20, 22:26] == falses(3, 5)               # absent, not needed: absent
        @test all(i -> Pe[i] == all(isfinite, e.camera.lens.backward.controls[i]), eachindex(Pe))
        # every node whose support meets the region is present, finite and (if not a present source node) marked
        need = node_mask(VgtkCore.with_present(e.camera, trues(size(Pe))), region)
        @test all(Pe[need])
        @test e.extrapolated == (Pe .& .!BitMatrix([i in rows && j in cols && P[i - e.offset[1], j - e.offset[2]] for i in axes(Pe, 1), j in axes(Pe, 2)]))
        @test e.extrapolated[rows, cols][1:4, 1:5] == trues(4, 5)       # the corner filled
        # the region is imaged; where the source images, unchanged
        @test all(p -> imaged(e.camera, p), pixels_in(rng, region, 500))
        src = filter(p -> imaged(cam, p), pts)
        @test all(p -> imaged(e.camera, p), src)
        @test maximum(p -> angle_between(VgtkCore.backproject(e.camera, p), VgtkCore.backproject(cam, p)), src) <= 1.0e-12
        V = valid_domain(e.camera)
        @test V.origin[1] <= region.origin[1] + 1.0e-9 && V.origin[2] <= region.origin[2] + 1.0e-9
    end
end

@testitem "lattice operations: extend, then subdivide" setup = [LatticeOps] begin
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: extend_lattice, subdivide_lattice, valid_domain, lattice_spacing, Rect2, support, nodes
    using StaticArrays: SVector
    using LinearAlgebra: norm
    using Random: MersenneTwister

    rng = MersenneTwister(1083)
    cam = random_camera(rng, 16, 18)
    hx, hy = lattice_spacing(cam)
    A = valid_domain(cam)
    region = Rect2(A.origin[1] - 2.2hx, A.origin[2] - 1.1hy, A.widths[1] + 3.3hx, A.widths[2] + 2.6hy)
    e = extend_lattice(cam, region)
    for m in ((2, 3), (5, 4))
        f = subdivide_lattice(e.camera, m; e.extrapolated)
        pts = pixels_in(rng, A, 400)
        @test maximum(p -> norm(raw(f.camera, p) - raw(cam, p)) / norm(raw(cam, p)), pts) <= 1.0e-12
        @test maximum(p -> angle_between(VgtkCore.backproject(f.camera, p), VgtkCore.backproject(cam, p)), pts) <= 1.0e-9
        @test valid_domain(f.camera) == valid_domain(e.camera)
        # a marked fine node reads an extrapolated node; none of them reaches the source's domain
        @test any(f.extrapolated) && !all(f.extrapolated)
        hf = lattice_spacing(f.camera)
        for i in findall(f.extrapolated)
            c = VgtkCore.node_position(f.camera, i)
            lo = c .- 2 .* hf; hi = c .+ 2 .* hf                         # the open support
            @test lo[1] >= A.origin[1] + A.widths[1] - 1.0e-9 || hi[1] <= A.origin[1] + 1.0e-9 ||
                lo[2] >= A.origin[2] + A.widths[2] - 1.0e-9 || hi[2] <= A.origin[2] + 1.0e-9
        end
    end
end

@testitem "difference_rows: the mixed rows" begin
    using VisualGeometryToolkitCore: difference_rows, bending_rows
    using Random: MersenneTwister
    rng = MersenneTwister(5)
    D0 = difference_rows((6, 7), 2)
    D = difference_rows((6, 7), 2; mixed = true)
    nm = 5 * 6
    @test size(D) == (size(D0, 1) + nm, 42) && D[1:size(D0, 1), :] == D0       # the axis rows unchanged, first
    c = randn(rng, 6, 7)
    Dc = D[(size(D0, 1) + 1):end, :] * vec(c)
    @test Dc ≈ vec([c[i, j] - c[i + 1, j] - c[i, j + 1] + c[i + 1, j + 1] for i in 1:5, j in 1:6])
    # the bending energy annihilates exactly the affine functions
    aff = [0.3 + 0.2i - 0.7j for i in 1:6, j in 1:7]
    @test maximum(abs, D * vec(aff)) <= 1.0e-12
    @test maximum(abs, D * vec([Float64(i * j) for i in 1:6, j in 1:7])) > 0.5
    mask = trues(6, 7); mask[3, 4] = false
    @test size(difference_rows(mask, 2; mixed = true), 1) == size(difference_rows(mask, 2), 1) + nm - 4
    @test_throws ArgumentError difference_rows((6, 7), 3; mixed = true)
end
