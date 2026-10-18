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
            @test lens_parameters(bm32.map.t2.t1) isa SVector{N, Float32} where {N}

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
            @test eltype(lens_parameters(bc32)) === Float32
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
        ci = undistort(m, img, rect; scale = 2)
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
