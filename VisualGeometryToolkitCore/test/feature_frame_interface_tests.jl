# The FeatureFrame interface (conventions/feature-frames.md): every verb once, PointFrame first-class at
# Point2's cost, the one detection → point observation conversion, and the absent verbs' pointed MethodError.

@testitem "feature frames: the interface, every frame" begin
    using Test, StaticArrays, LinearAlgebra
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: center, frame_dof, transfer, PointShape, IsotropicScale, ShapeFactorScale, frame_scale
    using CoordinateTransformations: LinearMap, AffineMap, PerspectiveMap
    using Rotations: RotMatrix2
    using Statistics: cov, mean
    using Distributions: MvNormal
    using FlexUnits: ustrip

    c = Point2(10.0, 20.0)
    frames = (PointFrame(c), Disc(c, 2.0), OrientedDisc(c, 2.0, 0.3), EllipticFrame(c, @SMatrix [2.0 0.5; 0.5 1.0]),
              AffineFrame(c, @SMatrix [2.0 0.5; -0.5 1.0]))
    @test map(frame_dof, frames) == (2, 3, 4, 5, 6)
    @test map(frame_scale, frames) == (PointShape(), IsotropicScale(), IsotropicScale(), ShapeFactorScale(), ShapeFactorScale())

    @testset "PointFrame costs what a Point2 costs" begin
        @test isbitstype(PointFrame{Float64}) && sizeof(PointFrame{Float64}) == sizeof(Point2{Float64})
        @test isbitstype(PointFrame{Float32}) && sizeof(PointFrame{Float32}) == sizeof(Point2{Float32})
        @test reinterpret(PointFrame{Float64}, [c])[1] == PointFrame(c)          # the same bits
        @test center(PointFrame(c)) === c && PointFrame(center(PointFrame(c))) === PointFrame(c)
    end

    @testset "center: the one detection → point observation conversion" begin
        for f in frames
            N = frame_dof(f)
            A = SMatrix{N, N}(reshape(collect(1.0:(N * N)), N, N))
            Σ = A * A' + I
            u = Uncertain(f, CovMat(Σ))
            o = center(u)
            @test o isa Uncertain{Point2{Float64}} && o.value == c
            @test SMatrix{2, 2}(cov(o)) == Σ[SOneTo(2), SOneTo(2)]                    # the centre's block, exactly
            @test center(Attributed(u, (; label = :a))) == o                            # a detection: its frame's
            @test center(Uncertain(f, CovMat(Diagonal(SVector(ntuple(k -> Float64(k), N)))))) ==
                  Uncertain(c, CovMat(Diagonal(SVector(1.0, 2.0))))                    # the storage kept
            @test center(Uncertain(f, CovMat{N}(σ = 0.5))) == Uncertain(c, CovMat{2}(σ = 0.5))
            # a precision: the marginal's precision, the inverse of the covariance's block
            @test SMatrix{2, 2}(prec(center(Uncertain(f, PrecMat(inv(Σ)))))) ≈ inv(Σ[SOneTo(2), SOneTo(2)]) rtol = 1e-12
            @test Ellipse(u; sigma = 2) == Ellipse(o; sigma = 2)                       # the error ellipse is its centre's
        end
        Λ = PrecMat(@SMatrix [4.0 1.0; 1.0 2.0])
        @test center(Uncertain(PointFrame(c), Λ)) === Uncertain(c, Λ)                  # 2-DOF: its own form, unchanged
        L = relabel(PixelFrame{:colmap}, CovMat(@SMatrix [4.0 1.0; 1.0 2.0]))
        @test center(Uncertain(PointFrame(c), L)).attributes === L                     # its label kept
    end

    @testset "similarity action and transfer" begin
        S = SimilarityMap(2.0, RotMatrix2(0.4), SVector(1.0, -3.0))
        E = EuclideanMap(RotMatrix2(0.4), SVector(1.0, -3.0))
        for f in frames
            @test center(S(f)) ≈ S(SVector(c))                     # the centre moves as a point
            @test typeof(S(f)) == typeof(f) && typeof(E(f)) == typeof(f)
            @test transfer(S, f) == S(f)                           # a similarity keeps the class, exactly
        end
        @test S(PointFrame(c)) == PointFrame(Point2(S(SVector(c))...))
        @test LinearMap(RotMatrix2(0.4))(PointFrame(c)) == SimilarityMap(LinearMap(RotMatrix2(0.4)))(PointFrame(c))
        a = AffineMap(@SMatrix([1.0 0.2; 0.1 0.9]), SVector(3.0, 4.0))
        H = PerspectiveMap() ∘ HomographyMap(@SMatrix [1.0 0.1 3.0; 0.0 1.1 4.0; 1e-3 2e-3 1.0])
        for m in (a, H)
            t = transfer(m, PointFrame(c))
            @test t isa PointFrame{Float64} && center(t) ≈ m(SVector(c))
            d = transfer(m, Disc(c, 2.0))                          # first order: an AffineFrame, the shape by the Jacobian
            @test d isa AffineFrame && center(d) ≈ m(SVector(c))
        end
        @test shape_factor(transfer(a, Disc(c, 2.0))) ≈ a.linear * shape_factor(Disc(c, 2.0))
    end

    @testset "propagate an uncertain PointFrame: the uncertain point's law, the frame kept" begin
        Σ = CovMat(@SMatrix [4.0 1.0; 1.0 2.0])
        u, p = Uncertain(PointFrame(c), Σ), Uncertain(c, Σ)
        S = SimilarityMap(2.0, RotMatrix2(0.4), SVector(1.0, -3.0))
        h = HomographyMap(@SMatrix [1.0 0.1 3.0; 0.0 1.1 4.0; 1e-3 2e-3 1.0])
        for m in (S, EuclideanMap(RotMatrix2(0.4), SVector(1.0, -3.0)), h)
            v, w = propagate(m, u), propagate(m, p)
            @test v.value isa PointFrame && center(v.value) ≈ w.value && cov(v) ≈ cov(w)
        end
    end

    @testset "per-DOF verbs, hoisted; extrema, Rect2, units" begin
        d, o = frames[2], frames[3]
        @test scale(d) == scale(o) == 2.0
        @test shape_factor(d) == @SMatrix([2.0 0.0; 0.0 2.0])
        @test shape_matrix(o) === @SMatrix([4.0 0.0; 0.0 4.0])                  # σ²I exactly, oriented or not
        @test AffineMap(d).linear == shape_factor(d) && AffineMap(o).linear ≈ 2.0 * RotMatrix2(0.3)
        @test extrema(PointFrame(c)) == (c, c) && extrema(d) == (Point2(8.0, 18.0), Point2(12.0, 22.0))
        @test Rect2(PointFrame(c)) == Rect2(10.0, 20.0, 0.0, 0.0) && Rect2(d) == Rect2(8.0, 18.0, 4.0, 4.0)
        @test bbox(PointFrame(c)) isa VisualGeometryToolkitCore.SpatialIndexing.Rect
        # bbox, once on the trait: a point's is the point at any scale; an isotropic frame's c ∓ kσ; a shape frame's its region's box
        R = VisualGeometryToolkitCore.SpatialIndexing.Rect
        @test bbox(PointFrame(c); scale = 3.0) == R((10.0, 20.0), (10.0, 20.0))
        @test bbox(d; scale = 3.0) == bbox(o; scale = 3.0) == R((4.0, 14.0), (16.0, 26.0))
        @test bbox(d; scale = 1.5, snap_to_grid = true) == R((7.0, 17.0), (13.0, 23.0))
        for f in frames[4:5]
            @test bbox(f; scale = 2.0) == R(Rect2(region(f, 2.0)))
        end
        @test bbox(Disc(Point2(10.0px, 20.0px), 2.0px); scale = 3.0) == bbox(d; scale = 3.0)
        # inv, once on the trait: the canonical frame's inverse, in the frame's class
        for f in (d, frames[4], frames[5])
            g = inv(f)
            @test typeof(g) == typeof(f) && AffineMap(g) ≈ inv(AffineMap(f))
        end
        # MvNormal and Circle, one arm for every scaled frame: the 1σ contour's shape matrix; the circle at r·characteristic_scale
        for f in frames[2:5]
            @test cov(MvNormal(f)) ≈ shape_matrix(f) && mean(MvNormal(f)) == c
            @test Circle(f, 2.0) == Circle(c, 2.0 * characteristic_scale(f))
            @test Circle(Attributed(f, (; label = :a)), 2.0) == Circle(f, 2.0)              # a detection's circle is its frame's
        end
        for f in frames
            @test (f * 2.0) / 2.0 == f                                                  # a power of two: exact
            @test center(f * 3.0) == 3.0 * c && f + (1.0, 1.0) == f + SVector(1.0, 1.0)
        end
        @test PointFrame(c) * 2.0 == PointFrame(2.0 * c)
        @test ustrip(px, PointFrame(Point2(10.0px, 20.0px))) == PointFrame(c)
        @test ustrip(px, Disc(Point2(10.0px, 20.0px), 2.0px)) == Disc(c, 2.0)
    end

    @testset "the scaled frames' verbs are absent from a PointFrame, by design" begin
        p = PointFrame(c)
        for call in (() -> region(p, 1.0), () -> AffineMap(p), () -> scale(p), () -> shape_matrix(p), () -> shape_factor(p),
                     () -> characteristic_scale(p), () -> Circle(p, 1.0), () -> Ellipse(p, 1.0), () -> MvNormal(p))
            e = try call(); nothing catch err; err end
            @test e isa MethodError
            @test occursin("PointFrame is the 2-DOF point feature", sprint(showerror, e))
        end
    end
end

@testitem "feature frames: PointFrame at no overhead (allocations, inference)" begin
    using Test, StaticArrays
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: center, transfer
    using Rotations: RotMatrix2
    using CoordinateTransformations: PerspectiveMap

    S = SimilarityMap(2.0, RotMatrix2(0.4), SVector(1.0, -3.0))
    h = HomographyMap(@SMatrix [1.0 0.1 3.0; 0.0 1.1 4.0; 1e-3 2e-3 1.0])
    u = Uncertain(PointFrame(10.0, 20.0), CovMat(@SMatrix [4.0 1.0; 1.0 2.0]))
    d = Attributed(u, (; label = :a))
    f(m, u) = propagate(m, u)
    g(m, p) = transfer(m, p)
    k(d) = center(d)
    p = getfield(u, :value)               # the frame (not `u.value`: Attributed's property forwarding is not the verb measured)
    # measured inside a function of concrete arguments (a warm-up call first), so only the verb's own allocation counts
    allocs(fn, args...) = (fn(args...); @allocated fn(args...))
    hp = PerspectiveMap() ∘ h               # the homography's point action
    for m in (S, h)
        @test allocs(f, m, u) == 0
    end
    for m in (S, hp)
        @test allocs(g, m, p) == 0
    end
    @test allocs(k, d) == 0
    # typed code with no Union, no boxing: concrete, inferred returns
    for (fn, args) in ((f, (S, u)), (f, (h, u)), (g, (S, p)), (g, (hp, p)), (k, (d,)))
        ci, rt = only(code_typed(fn, typeof.(args)))
        @test isconcretetype(rt)
        @test !any(st -> occursin("Core.Box", string(st)), ci.code)
        @test !any(t -> Core.Compiler.widenconst(t) isa Union, ci.ssavaluetypes)    # Union{} (a throw branch) is not a Union
    end
end

@testitem "feature frames: a KernelAbstractions kernel over a StructArray of uncertain PointFrame detections (CPU; GPU when functional)" begin
    using Test, StaticArrays, StructArrays, Adapt
    import KernelAbstractions as KA
    using KernelAbstractions: @kernel, @index
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: center
    using Rotations: RotMatrix2
    import CUDA

    S = SimilarityMap(2.0, RotMatrix2(0.4), SVector(1.0, -3.0))
    n = 1024
    dets = StructArray([Attributed(Uncertain(PointFrame(Float64(i), Float64(2i)), CovMat(@SMatrix [1.0 0.1; 0.1 2.0])), (; weight = 1.0 / i))
                        for i in 1:n])
    @kernel function moved!(out, @Const(dets), S)
        i = @index(Global)
        @inbounds out[i] = center(propagate(S, dets[i].value))      # the hoisted verbs, inside a kernel
    end
    function run(backend)
        d = adapt(backend, dets)
        out = StructArray{eltype(center.(dets[1:1]))}(undef, n)
        out = adapt(backend, out)
        moved!(backend)(out, d, S; ndrange = n); KA.synchronize(backend)
        return adapt(Array, out)
    end
    ref = [center(propagate(S, x.value)) for x in dets]
    cpu = run(KA.CPU())
    @test collect(cpu) == ref
    if CUDA.functional()
        gpu = run(CUDA.CUDABackend())
        # the device may contract a·b + c into one FMA: a few ulps of each entry (rtol 1e-14 ≈ 45 ulps)
        @test all(isapprox.(getfield.(gpu, :value), getfield.(ref, :value); rtol = 1e-14)) &&
              all(isapprox.(getfield.(gpu, :attributes), getfield.(ref, :attributes); rtol = 1e-14))
    else
        @test_skip CUDA.functional()                         # no functional GPU here: recorded as Broken (the GPU tier runs it)
    end
end
