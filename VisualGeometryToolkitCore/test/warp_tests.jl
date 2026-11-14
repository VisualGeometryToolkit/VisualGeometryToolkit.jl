# =============================================================================
# Warp fusion + KernelAbstractions resampler (transforms/warp.jl)
# =============================================================================
#
# Invariants:
#   1. fuse() is semantically a no-op: fused chains evaluate identically to
#      the raw composition, to fp round-off.
#   2. fuse() flattens to the three canonical shapes (affine → one AffineMap;
#      projective → PerspectiveMap ∘ one HomographyMap; nonlinear core →
#      affine ∘ core ∘ affine).
#   3. is_projective recurses BOTH sides of a composition (the predecessor
#      needs_perspective_division only looked at the inner side).
#   4. The KA kernel reproduces a direct sampling loop bit-for-bit on CPU,
#      for float, UInt8, and colorant destinations, single and batched.
#   5. undistort/undistort! round-trips eltype and matches its own math.

@testitem "warp" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using StaticArrays
    using GeometryBasics: Point2
    using CoordinateTransformations
    using CoordinateTransformations: PerspectiveMap
    using ImageCore: Gray, N0f8
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: is_projective   # public, not exported
    using VisualGeometryToolkitCore: RCSwap, _bilinear_sample

    @testset "Warp fusion + KA resampler" begin
        rc = RCSwap()
        inb = Translation(SVector(-0.5, -0.5))
        outb = Translation(SVector(0.5, 0.5))

        ell = Ellipse(Point2(30.0, 40.0), 10.0, 6.0, 0.3)
        circ = Circle(Point2(50.0, 50.0), 20.0)
        H = HomographyMap(SMatrix{3, 3, Float64}(1.0, 0.1, 1.0e-4, -0.05, 1.1, 2.0e-4, 3.0, -2.0, 1.0))

        @testset "fuse: canonical shapes" begin
            bm_aff = image_warp(canonical_map(ell, 32; bound = 1.0), 32)
            @test bm_aff.map isa AffineMap

            bm_proj = image_warp(H, 32)
            @test bm_proj.map isa CoordinateTransformations.ComposedTransformation{
                PerspectiveMap, <:HomographyMap{3},
            }

            bm_lp = image_warp(logpolar_map(circ, 64, 0.02), 64)
            # affine ∘ (log-polar core ∘ affine): exactly one nonlinear link survives
            @test bm_lp.map.t1 isa AffineMap
            @test bm_lp.map.t2.t2 isa AffineMap
        end

        @testset "fuse: semantic no-op" begin
            chains = [
                (
                    image_warp(canonical_map(ell, 32; bound = 1.0), 32),
                    rc ∘ outb ∘ canonical_map(ell, 32; bound = 1.0) ∘ inb ∘ rc, 32,
                ),
                (
                    image_warp(H, 32),
                    rc ∘ outb ∘ (PerspectiveMap() ∘ H) ∘ inb ∘ rc, 32,
                ),
                (
                    image_warp(logpolar_map(circ, 64, 0.02), 64),
                    rc ∘ outb ∘ logpolar_map(circ, 64, 0.02) ∘ inb ∘ rc, 64,
                ),
            ]
            for (bm, ref, n) in chains
                err = maximum(
                    maximum(
                        abs.(
                            bm(SVector(Float64(i), Float64(j))) .-
                                ref(SVector(Float64(i), Float64(j)))
                        )
                    )
                        for i in 1:n, j in 1:n
                )
                @test err < 1.0e-12
            end
        end

        @testset "is_projective: both sides, CT and Base compositions" begin
            @test is_projective(H)
            @test !is_projective(canonical_map(ell, 32; bound = 1.0))
            # HomographyMap on the OUTER side of a nonlinear composition — the
            # inner-only predecessor returned false here.
            lp = logpolar_to_cartesian(0.01)
            @test is_projective(H ∘ lp)
            @test is_projective(lp ∘ H)
            # No double division: a chain that already divides is left alone.
            bm = image_warp(PerspectiveMap() ∘ H, 32)
            @test bm.map isa CoordinateTransformations.ComposedTransformation{
                PerspectiveMap, <:HomographyMap{3},
            }
        end

        @testset "KA kernel ≡ direct loop (CPU)" begin
            img = Float32.(rand(100, 120))
            bm = image_warp(canonical_map(ell, 32; bound = 1.0), 32)

            function ref_warp(img, bm, n, fv)
                out = Matrix{Float32}(undef, n, n)
                Hs, Ws = size(img)
                for j in 1:n, i in 1:n
                    p = bm(SVector(Float64(i), Float64(j)))
                    out[i, j] = _bilinear_sample(img, p[1], p[2], fv, Hs, Ws)
                end
                return out
            end

            out = Matrix{Float32}(undef, 32, 32)
            image_warp!(out, bm, img; fillvalue = 0.5f0)
            # EXACT ON PURPOSE — a canary. Kernel and loop call the same
            # `bm` and `_bilinear_sample` per pixel; iterating differently does
            # not reassociate the arithmetic within a pixel. If it ever fails:
            # MEASURE the margin and loosen to it, never re-run until green.
            @test out == ref_warp(img, bm, 32, 0.5f0)
            @test_throws MethodError image_warp!(out, img, bm)   # map before image

            # Batched kernel matches per-warp single kernel
            warps = [
                image_warp(
                    canonical_map(
                        Ellipse(Point2(30.0 + 5k, 40.0), 10.0, 6.0, 0.3k), 32;
                        bound = 1.0
                    ), 32
                )
                    for k in 1:4
            ]
            out3 = Array{Float32}(undef, 32, 32, 4)
            image_warp_batch_to!(out3, warps, img; fillvalue = 0.5f0)
            for k in 1:4
                # EXACT ON PURPOSE — same canary, batched driver (see above).
                @test out3[:, :, k] == ref_warp(img, warps[k], 32, 0.5f0)
            end

            # UInt8 destination: saturating round of the Float32 samples
            img_u8 = rand(UInt8, 64, 64)
            out_u8 = Matrix{UInt8}(undef, 32, 32)
            image_warp!(out_u8, bm, img_u8; fillvalue = 0.0f0)
            # NOT an exact integer comparison, and not a ±1 grey-level allowance
            # either. Rounding to UInt8 turns a 1-ulp sampling difference at a
            # `x.5` boundary into a whole grey level, so `==` on the integers is
            # a coin flip there; but allowing ±1 would also accept a genuine
            # two-step sampling bug. What the test actually guarantees is that
            # the SAMPLER agrees, so compare the pre-rounding floats and round
            # once, from a single set of samples.
            ref_f = clamp.(ref_warp(Float32.(img_u8), bm, 32, 0.0f0), 0.0f0, 255.0f0)
            # `|out_u8 − ref_f| ≤ 0.5` says exactly "the kernel's integers ARE a
            # rounding of these samples", which is the guarantee. It is immune to
            # which side of a tie an ulp falls on, and it still catches a real
            # sampling error: being one grey level wrong puts the difference
            # above 0.5 and fails.
            @test maximum(abs, Float32.(out_u8) .- ref_f) <= 0.5f0

            # Colorant destination samples in float and clamps back
            img_g = Gray{N0f8}.(rand(64, 64))
            out_g = Matrix{Gray{N0f8}}(undef, 32, 32)
            image_warp!(out_g, bm, img_g)
            @test eltype(out_g) === Gray{N0f8}
        end

        @testset "undistort: eltype round-trip + precision" begin
            cam = CameraModel(;
                focal = 800.0, pp = (60.0, 50.0),
                lens = Lens(Perspective(), Additive(Radial((-0.15, 0.05)), Tangential(0.003, -0.002))),
            )

            img_g = Gray{N0f8}.(rand(100, 120))
            u = undistort(cam, img_g)
            @test eltype(u) === Gray{N0f8}
            @test size(u) == size(img_g)

            img_f = Float32.(rand(100, 120))
            d64 = undistort!(similar(img_f), cam, img_f)
            d32 = undistort!(similar(img_f), Float32, cam, img_f)
            @test_throws MethodError undistort!(similar(img_f), cam, img_f; precision = Float32)   # the type is positional
            @test maximum(abs.(d64 .- d32)) < 1.0e-3   # single-precision warp drift only

            # Float32 chain is fully single-precision (GPU throughput contract):
            # the leading type DISPATCHES the warp-math precision (rand(T,…) style)
            bm32 = image_warp(Float32, distortion_map(cam), 64)
            @test bm32.map.t1 isa AffineMap{SMatrix{2, 2, Float32, 4}, SVector{2, Float32}}
            @test parameter_vector(bm32.map.t2.t1) isa SVector{N, Float32} where {N}

            # Identity lens: undistort is a no-op forwarding
            cam_pin = CameraModel(; focal = 800.0, pp = (60.0, 50.0))
            @test undistort(cam_pin, img_g) === img_g
            @test undistort(nothing, img_g) === img_g   # BlobBoards' no-camera path
            @test_throws MethodError undistort(img_g, cam)            # model before image
            @test_throws MethodError undistort!(similar(img_f), img_f, cam)
        end

        @testset "the coefficient number type is the lens's" begin
            bc = Lens(Perspective(), Radial((-0.1, 0.02)))
            @test bc isa Lens{Perspective, Radial{VgtkCore.Even, 2, 0, Float64}}
            bc32 = Lens(Perspective(), Radial((-0.1f0,)))
            @test eltype(parameter_vector(bc32)) === Float32
            @test bc32(SVector(0.1f0, 0.2f0)) isa Vec2{Float32}
        end
    end

    @testset "FramedImage — rect-form undistort return currency" begin
        m = CameraModel(
            LogicalIntrinsics(
                CameraCalibrationMat(
                    50.0 * VgtkCore.px,
                    [20.0 * VgtkCore.px, 10.0 * VgtkCore.px]
                )
            ),
            VgtkCore.Division(-0.1)
        )
        img = rand(20, 40)
        rect = VgtkCore.Rect(-5.0, -3.0, 50.0, 26.0)
        ci = undistort(m, img, rect; scale = 2, device = VgtkCore.KernelAbstractions.CPU())
        @test ci isa FramedImage
        @test ci.rect === rect
        @test size(ci) == (13, 25)                       # ceil.(widths ./ scale), row-major
        # the pixels are exactly the in-place path's
        dst = zeros(eltype(img), 13, 25)
        VgtkCore.undistort!(dst, m, img, rect)
        @test ci.data == dst
        # same-size form still returns a bare matrix (placement = the frame)
        @test undistort(m, img) isa AbstractMatrix
    end
end

@testitem "log-polar map: its differential and its inverse (ForwardDiff, the composed map, the device)" begin
    using Test, StaticArrays, Random, LinearAlgebra
    using GeometryBasics: Point2, Rect
    import ForwardDiff
    using CoordinateTransformations: transform_deriv
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: LogPolarToCartesian, CartesianToLogPolar, Ellipse, Circle
    rng = Xoshiro(11)
    jac(f, x) = ForwardDiff.jacobian(q -> SVector(f(q)...), SVector(x...))
    t = logpolar_to_cartesian(0.05)
    ti = inv(t)
    @test ti isa CartesianToLogPolar && inv(ti) isa LogPolarToCartesian
    @test isbitstype(typeof(t)) && isbitstype(typeof(ti))
    for _ in 1:50
        p = SVector(2rand(rng) - 1, 1.98rand(rng) - 0.99)            # θ off the 0 / 2π seam
        @test transform_deriv(t, p) ≈ jac(t, p) rtol = 1e-12
        @test ti(t(p)) ≈ p atol = 1e-12
        x = t(p)
        @test transform_deriv(ti, x) ≈ jac(ti, x) rtol = 1e-12
        @test transform_deriv(ti, x) * transform_deriv(t, p) ≈ I atol = 1e-12
        r = norm(x)                                                  # the area element π·L·r²/2
        @test abs(det(transform_deriv(t, p))) ≈ π * -log(0.05) * r^2 / 2 rtol = 1e-12
    end
    # through the composed patch map of a circle and an ellipse: the chain rule and inv ∘ map = id
    for g in (Circle(Point2(310.0, 205.0), 180.0), Ellipse(Point2(300.0, 200.0), 190.0, 150.0, 0.4))
        m = logpolar_map(g, Rect(0, 0, 400, 1200), 0.2)
        mi = inv(m)
        for _ in 1:20
            p = SVector(400rand(rng), 1190rand(rng) + 5)
            @test transform_deriv(m, p) ≈ jac(m, p) rtol = 1e-10
            @test mi(m(p)) ≈ p atol = 1e-9
            @test transform_deriv(mi, m(p)) ≈ jac(mi, m(p)) rtol = 1e-10
        end
    end
    # on the device (isbits, GPU-compilable): both maps and the differential in a broadcast kernel, and the inverse
    # inside Core's warp kernel (m ∘ inv(m) ∘ m resamples as m does)
    lpm = logpolar_map(Circle(Point2(310.0, 205.0), 180.0), Rect(0, 0, 400, 1200), 0.2)
    ps = [SVector(400rand(rng), 1190rand(rng) + 5) for _ in 1:64]
    lp(m, mi, p) = (q = mi(m(p)); J = transform_deriv(m, p); SVector(q[1] - p[1], q[2] - p[2], abs(J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1])))
    host = lp.(Ref(lpm), Ref(inv(lpm)), ps)
    @test maximum(x -> max(abs(x[1]), abs(x[2])), host) < 1e-9
    img = rand(rng, Float32, 600, 700)
    out(math, src) = (bm = image_warp(Float32, math, Rect(0, 0, 400, 1200)); d = similar(src, Float32, 1200, 400);
                      image_warp!(d, bm, src; fillvalue = NaN32); Array(d))
    @test maximum(filter(isfinite, abs.(out(lpm ∘ inv(lpm) ∘ lpm, img) .- out(lpm, img)))) < 1e-4
    cuda = try
        mod = Base.require(Main, :CUDA)
        (mod !== nothing && Base.invokelatest(getproperty(mod, :functional))) ? mod : nothing
    catch
        nothing
    end
    if cuda === nothing
        @test_skip cuda !== nothing      # no functional GPU here: recorded as Broken, not as a success
    else
        CuArray = Base.invokelatest(getproperty, cuda, :CuArray)
        dev = Base.invokelatest(broadcast, lp, Ref(lpm), Ref(inv(lpm)), Base.invokelatest(CuArray, ps))
        @test Base.invokelatest(Array, dev) ≈ host
        @test maximum(filter(isfinite, abs.(Base.invokelatest(out, lpm ∘ inv(lpm) ∘ lpm, Base.invokelatest(CuArray, img)) .- out(lpm, img)))) < 1e-3
    end
end

@testitem "undistort and rectify of a Gray{N0f8} image under the GPU default (availability-gated)" begin
    using Test, StaticArrays, ImageCore, FixedPointNumbers, Random
    using Rotations: RotMatrix, RotXYZ
    import KernelAbstractions as KA
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: CameraModel, Lens, Perspective, Additive, Radial, Tangential, undistort, rectify,
        StereoRig, PosedCamera, EuclideanMap, CameraCalibrationMat, Division, _DEFAULT_DEVICE, reset_default_device!, default_device
    # The fixed-point colorant store is GPU-compilable (the unchecked `%` conversion after
    # the clamp): a host `Gray{N0f8}` image under the default device (the GPU) comes back
    # an `Array{Gray{N0f8}}` within 1 LSB of the host's; a `CuArray` one stays on the device.
    cuda = try
        mod = Base.require(Main, :CUDA)
        (mod !== nothing && Base.invokelatest(getproperty(mod, :functional))) ? mod : nothing
    catch
        nothing
    end
    if cuda === nothing
        @test_skip cuda !== nothing      # no functional GPU here: recorded as Broken, not as a success
    else
        old_device = _DEFAULT_DEVICE[]
        try
            reset_default_device!()
            gpu = Base.invokelatest(default_device)
            @test gpu isa Base.invokelatest(getproperty, cuda, :CUDABackend)
            cam = CameraModel(; focal = 800.0, pp = (60.0, 50.0),
                lens = Lens(Perspective(), Additive(Radial((-0.15, 0.05)), Tangential(0.003, -0.002))))
            img = Gray{N0f8}.(rand(Xoshiro(5), 100, 120))
            lsb(a, b) = maximum(abs.(Int.(reinterpret.(gray.(a))) .- Int.(reinterpret.(gray.(b)))))
            host = undistort(cam, img; device = KA.CPU())
            dev = Base.invokelatest(undistort, cam, img)
            @test dev isa Array{Gray{N0f8}}
            @test lsb(dev, host) <= 1
            dimg = Base.invokelatest(getproperty(cuda, :CuArray), img)
            dd = Base.invokelatest(undistort, cam, dimg)
            @test !(dd isa Array) && eltype(dd) == Gray{N0f8}
            @test lsb(Base.invokelatest(Array, dd), host) <= 1
            Ks = @SMatrix [80.0 0.0 60.0; 0.0 80.0 50.0; 0.0 0.0 1.0]
            m = CameraModel(CameraCalibrationMat(Ks); lens = Division(-0.12))
            X = EuclideanMap(RotMatrix{3, Float64}(RotXYZ(0.0, 0.03, 0.0)), SVector(-40.0, 0.0, 0.0))
            I3 = EuclideanMap(one(RotMatrix{3, Float64}), zeros(SVector{3, Float64}))
            rig = StereoRig(PosedCamera(m, I3), PosedCamera(m, X); ids = (:l, :r))
            rh = rectify(rig, (img, img); device = KA.CPU())
            rd = Base.invokelatest(rectify, rig, (img, img))
            for i in 1:2
                @test rd.images[i].data isa Array{Gray{N0f8}}
                @test lsb(rd.images[i].data, rh.images[i].data) <= 1
            end
        finally
            _DEFAULT_DEVICE[] = old_device
        end
    end
end

@testitem "image_warp! of a multi-plane image, each plane through its own warp (host and device)" begin
    using Test, StaticArrays, Random
    import KernelAbstractions as KA
    using VisualGeometryToolkitCore: image_warp, image_warp!, Rect, LinearMap, Translation
    # Plane p of dst is plane p of src read through warps[p] — exactly what three 2-D warps give —
    # the planes' maps differing by a translation (a Bayer image's R, G and B sit at different
    # points of the cell); a warp count other than the planes' is refused.
    rng = Xoshiro(3)
    src = rand(rng, 20, 24, 3)
    math = LinearMap(SMatrix{2, 2}(0.9, 0.1, -0.05, 1.1))
    warps = ntuple(p -> image_warp(Float64, Translation(SVector(0.25 * (p - 2), -0.25 * (p - 2))) ∘ math, Rect(0, 0, 16, 18)), 3)
    dst = image_warp!(zeros(18, 16, 3), warps, src)
    for p in 1:3
        @test dst[:, :, p] == image_warp!(zeros(18, 16), warps[p], src[:, :, p])
    end
    @test dst[:, :, 1] != dst[:, :, 3]
    @test_throws DimensionMismatch image_warp!(zeros(18, 16, 3), warps[1:2], src)
    @test_throws DimensionMismatch image_warp!(zeros(18, 16, 2), warps, src)
    @test_throws DimensionMismatch image_warp!(zeros(18, 16, 3), warps, src[:, :, 1:2])
    # GPU first: the same call on device arrays runs each plane's warp on the device
    cuda = try
        mod = Base.require(Main, :CUDA)
        (mod !== nothing && Base.invokelatest(getproperty(mod, :functional))) ? mod : nothing
    catch
        nothing
    end
    if cuda === nothing
        @test_skip cuda !== nothing      # no functional GPU here: recorded as Broken, not as a success
    else
        cu = getproperty(cuda, :CuArray)
        d = Base.invokelatest(image_warp!, Base.invokelatest(cu, zeros(18, 16, 3)), warps, Base.invokelatest(cu, src))
        @test !(d isa Array)
        @test Base.invokelatest(Array, d) ≈ dst rtol = 1e-12
    end
end

# Core 0.10.4: `_precision`'s fallback is not silent. A link that holds floats of another precision and has no
# conversion warns once by type; the log-polar maps and intrinsics are kept on purpose (no warning), the log-polar
# coordinates Float64 in a Float32 warp; a link with no floats, and a chain already in T, say nothing.
@testitem "image_warp(Float32, …): a link left in another precision is named, the kept ones are not" begin
    using Test, StaticArrays
    using GeometryBasics: Point2, Rect
    import CoordinateTransformations
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: LogPolarToCartesian, Circle
    struct _Float64Shift <: CoordinateTransformations.Transformation
        dx::Float64
    end
    (t::_Float64Shift)(x) = SVector(x[1] + t.dx, x[2])
    VgtkCore = VisualGeometryToolkitCore
    @test_logs (:warn, r"_Float64Shift link holds floats of another precision") VgtkCore._precision(Float32, _Float64Shift(0.5))
    @test_logs VgtkCore._precision(Float64, _Float64Shift(0.5))                       # already Float64: silent
    @test_logs VgtkCore._precision(Float32, CoordinateTransformations.IdentityTransformation())   # no floats: silent
    lpm = logpolar_map(Circle(Point2(310.0, 205.0), 180.0), Rect(0, 0, 400, 1200), 0.2)
    leaves(x) = x isa CoordinateTransformations.ComposedTransformation ? [leaves(x.t1); leaves(x.t2)] : [x]
    bm = @test_logs image_warp(Float32, lpm, Rect(0, 0, 400, 1200))                   # no warning for the kept link
    lp = filter(x -> x isa LogPolarToCartesian, leaves(bm.map))
    @test !isempty(lp) && all(x -> x.log_inner_radius isa Float64, lp)                # its coordinates stay Float64
    @test image_warp(Float32, lpm, Rect(0, 0, 400, 1200)).map == image_warp(Float64, lpm, Rect(0, 0, 400, 1200)).map ||
          all(x -> x isa LogPolarToCartesian, filter(x -> !(x isa CoordinateTransformations.AffineMap), leaves(bm.map)))
end
