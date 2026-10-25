@testitem "imaging_chain" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using VisualGeometryToolkitCore: imaging_chain, FramedStage, FramedPerspective,
        FramedEquidistantProjection, FramedAffine, FramedChain, distort_stage, undistort_stage,
        calibration_stage, RayMap, ray_map, RetinalFrame, EquidistantFrame, DistortedFrame
    using VisualGeometryToolkitCore
    using StaticArrays
    using ForwardDiff: ForwardDiff
    using LinearAlgebra: norm, normalize
    using CoordinateTransformations: ComposedTransformation
    using Rotations: RotMatrix

    # =============================================================================
    # Framed imaging chain (0.10) — typed frame-to-frame stages
    # =============================================================================
    #
    # Covers: the ray-frame projection family (FramedPerspective = gnomonic,
    # delegating to CT's PerspectiveMap; FramedEquidistantProjection =
    # equidistant azimuthal),
    # the deliberate absence of inv on the fiber-collapsing projections, the
    # honest distortion_frame tags per lens model, distort-vs-undistort
    # DIRECTION checking through frame-kind junctions (anonymous view id :_),
    # the frame-collapse rule for identity-in-frame lenses, imaging_chain
    # minting with numerical parity against the frame-free imaging_transform
    # (bit-identical where the math path is shared), anonymous vs rig-minted
    # view ids, and the fused stage of a parametric projection.

    @testset "Framed imaging chain (0.10)" begin
        # --- Fixtures ------------------------------------------------------------
        K_model(lens) = CameraModel(; focal = (620.0, 610.0), pp = (321.5, 242.5), lens)
        pin = K_model(Pinhole())
        bc_lens = Lens(Perspective(), Additive(Radial((-0.28, 0.07, -0.006)), Tangential(0.0009, -0.0004)))
        bc = K_model(bc_lens)
        kb_lens = KannalaBrandt(-0.043, 0.011, -0.0083, 0.0015)
        kb = K_model(kb_lens)
        eqd = K_model(Lens(Equidistant()))
        tpf_lens = Lens(Equidistant(), Additive(Radial((-0.02, 0.004, -0.0009, 0.0001)), Tangential(0.0006, -0.0004), ThinPrism((0.0002,), (-0.0001,))))
        tpf = K_model(tpf_lens)

        # Camera-space test rays (Z > 0, mild to wide angles — within the
        # gnomonic frame's validity, so every stage-1 projection serves them).
        rays = [
            SVector(0.0, 0.0, 1.0),
            SVector(0.1, -0.05, 1.0),
            SVector(-0.35, 0.2, 1.2),
            SVector(0.6, 0.45, 1.0),
            SVector(-0.9, -0.7, 1.1),
            SVector(1.3, 0.2, 1.0),      # θ ≈ 0.92 rad — wide
            SVector(0.02, 1.7, 1.0),     # θ ≈ 1.04 rad — wider
        ]

        @testset "FramedPerspective — gnomonic stage" begin
            p = FramedPerspective()
            @test p isa FramedPerspective{ViewFrame{:_}, GnomonicFrame{:_}}
            @test FramedPerspective(; id = :left) isa
                FramedPerspective{ViewFrame{:left}, GnomonicFrame{:left}}
            # math is the lens contract's Bearing frame projection — bit-identical
            # to the pinhole's `project`, and CT's PerspectiveMap to rounding
            for X in rays
                @test p(X) == project(Pinhole(), X)
                @test p(X) ≈ PerspectiveMap()(X) atol = 1.0e-15
            end
            # inv is absent by design: 3D → 2D fiber collapse has no inverse
            @test_throws MethodError inv(p)
            # its SECTIONS are unproject/backproject: project then unproject at
            # the true depth recovers the point
            for X in rays
                u = imaging_chain(pin)(X)
                @test unproject(pin, u, X[3]) ≈ Point3(X...) atol = 1.0e-9
            end
        end

        @testset "FramedEquidistantProjection — equidistant stage" begin
            a = FramedEquidistantProjection()
            @test a isa FramedEquidistantProjection{ViewFrame{:_}, EquidistantFrame{:_}}
            @test_throws MethodError inv(a)
            # radius equals the polar angle θ; azimuth preserved
            for X in rays
                uv = a(X)
                r = hypot(X[1], X[2])
                @test norm(uv) ≈ atan(r, X[3]) atol = 1.0e-14
                iszero(r) || @test atan(uv[2], uv[1]) ≈ atan(X[2], X[1]) atol = 1.0e-14
            end
            # well-defined past 90° — where the gnomonic frame cannot go
            uv_back = a(SVector(0.5, 0.0, -0.5))
            @test norm(uv_back) ≈ 3π / 4 atol = 1.0e-14
            # the forward pole maps to the origin; the exact backward ray is outside the frame
            @test a(SVector(0.0, 0.0, 2.0)) == Vec2(0.0, 0.0)
            @test all(isnan, a(SVector(0.0, 0.0, -2.0)))
        end

        @testset "distortion_frame — honest per-model tags" begin
            @test distortion_frame(Pinhole()) === GnomonicFrame
            @test distortion_frame(bc_lens) === GnomonicFrame
            @test distortion_frame(Lens(Perspective(), Tangential(0.001, 0.002))) === GnomonicFrame
            @test distortion_frame(Division(-0.2)) === GnomonicFrame
            @test distortion_frame(kb_lens) === EquidistantFrame
            @test distortion_frame(Lens(Equidistant())) === EquidistantFrame
            @test distortion_frame(tpf_lens) === EquidistantFrame
            # a parametric projection: its frame is model-defined — no tag
            @test distortion_frame(Lens(DoubleSphere(0.3, 0.6))) === nothing
            @test distortion_frame(Lens(Unified(0.8), Radial((-0.1,)))) === nothing
        end

        @testset "distort/undistort stages — types, inverses, round-trips" begin
            d = distort_stage(bc_lens)
            @test d isa VgtkCore.FramedDistort{
                GnomonicFrame{:_}, DistortedFrame{:_}, typeof(bc_lens),
            }
            u = undistort_stage(bc_lens)
            @test u isa VgtkCore.FramedUndistort{
                DistortedFrame{:_}, GnomonicFrame{:_}, typeof(bc_lens),
            }
            @test inv(d) === u && inv(u) === d
            # the framed stage wraps the SAME scalar math — bit-identical
            x = SVector(0.21, -0.13)
            @test d(x) == bc_lens(x)
            @test u(d(x)) == undistort_normalized(bc_lens, bc_lens(x))
            @test u(d(x)) ≈ x atol = 1.0e-9

            # KannalaBrandt is honestly tagged on its defining EQUIDISTANT frame
            dkb = distort_stage(kb_lens)
            @test dkb isa VgtkCore.FramedDistort{
                EquidistantFrame{:_}, DistortedFrame{:_}, typeof(kb_lens),
            }
            uv = SVector(0.8, -0.55)                 # θ ≈ 0.97 rad
            θ = norm(uv)
            c = parameters(kb_lens)
            θd = θ * (1 + c.k1 * θ^2 + c.k2 * θ^4 + c.k3 * θ^6 + c.k4 * θ^8)
            @test norm(dkb(uv)) ≈ θd atol = 1.0e-14
            @test undistort_stage(kb_lens)(dkb(uv)) ≈ uv atol = 1.0e-10

            # COLMAP's thin-prism fisheye: the distortion stage is its additive
            # terms on the equidistant frame; round-trip via their inverse
            dtp = distort_stage(tpf_lens)
            @test dtp isa VgtkCore.FramedDistort{EquidistantFrame{:_}, DistortedFrame{:_}}
            @test inv(dtp)(dtp(uv)) ≈ uv atol = 1.0e-8

            # no distortion stage for identity-in-frame lenses (collapse instead)
            @test_throws ArgumentError distort_stage(Pinhole())
            @test_throws ArgumentError distort_stage(Lens(Equidistant()))
            # a parametric projection has no frame split
            @test_throws ArgumentError distort_stage(Lens(DoubleSphere(0.3, 0.6)))
        end

        @testset "direction checking by frame KIND (anonymous id)" begin
            persp = FramedPerspective()
            d = distort_stage(bc_lens)
            u = undistort_stage(bc_lens)
            K = calibration_stage(bc.intrinsics)
            # the honest directions compose
            @test (d ∘ persp) isa FramedChain
            @test (K ∘ d) isa FramedChain
            @test (u ∘ d) isa FramedChain          # Bearing → Distorted → Bearing
            @test (d ∘ u) isa FramedChain          # Distorted → Bearing → Distorted
            @test (u ∘ d)(SVector(0.2, 0.1)) ≈ SVector(0.2, 0.1) atol = 1.0e-9
            # the WRONG directions are junction mismatches — MethodError, with no
            # view identity involved (frame kind does the checking)
            @test_throws MethodError u ∘ persp     # Bearing-frame point into undistort
            @test_throws MethodError d ∘ d         # Distorted point into distort
            @test_throws MethodError u ∘ u
            @test_throws MethodError K ∘ persp     # K consumes DistortedFrame here
            # cross-KIND mismatch: a gnomonic-frame distortion cannot follow the
            # equidistant projection, nor vice versa
            @test_throws MethodError d ∘ FramedEquidistantProjection()
            @test_throws MethodError distort_stage(kb_lens) ∘ persp
            # the FramedChain constructor performs the same junction checks
            @test_throws ArgumentError FramedChain(persp, K)
            @test_throws ArgumentError FramedChain(persp, u)
        end

        @testset "imaging_chain — Pinhole (frame collapse, bit-identical)" begin
            c = imaging_chain(pin)
            @test c isa FramedChain{ViewFrame{:_}, ImageFrame{:_}}
            # collapse: NO distortion stage, no DistortedFrame anywhere — K
            # consumes the GnomonicFrame directly (Bearing ≡ Distorted)
            @test length(c.stages) == 2
            @test c.stages[1] isa FramedPerspective{ViewFrame{:_}, GnomonicFrame{:_}}
            @test c.stages[2] isa FramedAffine{GnomonicFrame{:_}, ImageFrame{:_}}
            legacy = imaging_transform(pin.intrinsics, pin.lens)
            @test legacy isa ComposedTransformation   # frame-free: AffineMap(K) ∘ lens
            for X in rays
                got = c(X)
                want = legacy(X)
                @test got[1] === want[1] && got[2] === want[2]   # bit-identical
            end
        end

        @testset "imaging_transform — K ∘ lens on rays, past 90°" begin
            wide = (
                SVector(0.3, -0.2, 1.0),
                SVector(0.8, 0.6, -0.36),      # ≈ 110°
                SVector(-0.5, 0.1, -0.2),      # ≈ 111°
                SVector(0.05, 0.0, -1.0),      # ≈ 177°
            )
            for m in (kb, eqd, tpf, K_model(Division(-0.3126, -0.0812)), K_model(Lens(DoubleSphere(0.3, 0.6))))
                proj = imaging_transform(m)
                @test proj isa ComposedTransformation{<:AffineMap, typeof(m.lens)}
                for X in wide
                    got, want = proj(X), project(m, X)
                    @test isequal(got[1], want[1]) && isequal(got[2], want[2])   # NaN exactly where the lens images nothing
                end
            end
            @test all(isfinite, imaging_transform(kb)(SVector(0.8, 0.6, -0.36)))
            # a pinhole images nothing behind the camera — no mirrored point
            @test all(isnan, imaging_transform(pin)(SVector(0.1, 0.2, -1.0)))
            # the PnP whitening reads K's 2×2 block off the pinhole composition only
            @test VgtkCore.pnp_pixel_whitening(imaging_transform(pin)) ==
                SMatrix{2, 2}(pin.intrinsics.K[1, 1], pin.intrinsics.K[2, 1], pin.intrinsics.K[1, 2], pin.intrinsics.K[2, 2])
            @test_throws ErrorException VgtkCore.pnp_pixel_whitening(imaging_transform(bc))
        end

        @testset "imaging_chain — a perspective-frame lens (bit-identical)" begin
            c = imaging_chain(bc)
            @test c isa FramedChain{ViewFrame{:_}, ImageFrame{:_}}
            @test length(c.stages) == 3
            @test c.stages[2] isa VgtkCore.FramedDistort{
                GnomonicFrame{:_}, DistortedFrame{:_},
            }
            @test c.stages[3] isa FramedAffine{DistortedFrame{:_}, ImageFrame{:_}}
            legacy = imaging_transform(bc.intrinsics, bc.lens)
            for X in rays
                got = c(X)
                want = legacy(X)
                @test got[1] === want[1] && got[2] === want[2]   # bit-identical
            end
        end

        @testset "imaging_chain — equidistant-frame lenses (numerical parity)" begin
            # The framed route works in the models' defining frame (θ directly
            # from atan(r, Z)) instead of hopping through the gnomonic plane
            # (atan(tan θ)); parity is floating-point round-off, not bit-level.
            for (model, name) in ((kb, :kb), (eqd, :eqd), (tpf, :tpf))
                c = imaging_chain(model)
                @test c isa FramedChain{ViewFrame{:_}, ImageFrame{:_}}
                legacy = imaging_transform(model.intrinsics, model.lens)
                for X in rays
                    got = c(X)
                    want = legacy(X)
                    @test isapprox(got[1], want[1]; atol = 1.0e-10, rtol = 1.0e-12)
                    @test isapprox(got[2], want[2]; atol = 1.0e-10, rtol = 1.0e-12)
                end
            end
            # a pure equidistant lens collapses too: identity in its defining frame
            ce = imaging_chain(eqd)
            @test length(ce.stages) == 2
            @test ce.stages[1] isa FramedEquidistantProjection{ViewFrame{:_}, EquidistantFrame{:_}}
            @test ce.stages[2] isa FramedAffine{EquidistantFrame{:_}, ImageFrame{:_}}
            # KannalaBrandt keeps its distortion stage, on the equidistant frame
            ckb = imaging_chain(kb)
            @test length(ckb.stages) == 3
            @test ckb.stages[1] isa FramedEquidistantProjection
            @test ckb.stages[2] isa VgtkCore.FramedDistort{
                EquidistantFrame{:_}, DistortedFrame{:_},
            }
            # a parametric projection is served by one fused stage
            cds = imaging_chain(bc.intrinsics, Lens(DoubleSphere(0.3, 0.6)))
            @test length(cds.stages) == 2 && cds.stages[1] isa VgtkCore.FramedLensProjection
        end

        @testset "anonymous vs rig-minted view ids" begin
            # a labeled chain wears the requested identity end to end
            cl = imaging_chain(bc; id = :left)
            @test cl isa FramedChain{ViewFrame{:left}, ImageFrame{:left}}
            @test cl.stages[2] isa VgtkCore.FramedDistort{
                GnomonicFrame{:left}, DistortedFrame{:left},
            }
            # identities must MATCH at a junction — :left does not meet :right
            @test_throws MethodError distort_stage(bc_lens; id = :left) ∘
                FramedPerspective(; id = :right)
            # ... and anonymous does not meet a real id (:_ is a label, not a wildcard)
            @test_throws MethodError distort_stage(bc_lens; id = :left) ∘ FramedPerspective()
            @test (distort_stage(bc_lens; id = :left) ∘ FramedPerspective(; id = :left)) isa
                FramedChain{ViewFrame{:left}, DistortedFrame{:left}}
            # rig-derived labeling: mint from the rig's ids
            cam1 = Camera(pin, EuclideanMap(one(RotMatrix{3, Float64}), SVector(0.0, 0.0, 0.0)))
            cam2 = Camera(bc, EuclideanMap(one(RotMatrix{3, Float64}), SVector(120.0, 0.0, 0.0)))
            rig = StereoRig(cam1, cam2; ids = (:src, :tgt))
            a, b = ids(rig)
            c1 = imaging_chain(model(source(rig)); id = a)
            c2 = imaging_chain(model(target(rig)); id = b)
            @test c1 isa FramedChain{ViewFrame{:src}, ImageFrame{:src}}
            @test c2 isa FramedChain{ViewFrame{:tgt}, ImageFrame{:tgt}}
            # numerical behavior is identical under any labeling
            for X in rays
                @test c2(X) == imaging_chain(bc)(X)
            end
        end

        @testset "FramedChain — flattening, application, inv" begin
            persp = FramedPerspective()
            d = distort_stage(bc_lens)
            K = calibration_stage(bc.intrinsics)
            c = (K ∘ d) ∘ persp                      # composition of a chain and a stage
            @test c isa FramedChain{ViewFrame{:_}, ImageFrame{:_}}
            @test length(c.stages) == 3              # flat — no nested chains
            @test !any(s -> s isa FramedChain, c.stages)
            @test c(rays[4]) == imaging_chain(bc)(rays[4])
            # the invertible 2D tail inverts, frames swapped, exact round-trip
            tail = K ∘ d
            @test tail isa FramedChain{GnomonicFrame{:_}, ImageFrame{:_}}
            ti = inv(tail)
            @test ti isa FramedChain{ImageFrame{:_}, GnomonicFrame{:_}}
            x = SVector(0.21, -0.13)
            @test ti(tail(x)) ≈ x atol = 1.0e-9
            # a chain containing the fiber-collapsing projection has no inv
            @test_throws MethodError inv(c)
        end

        @testset "RetinalFrame — the spherical retina names its frames" begin
            # the canonical ray frames are frames of the view's retina, id threaded
            @test GnomonicFrame{:cam1} <: RetinalFrame{:cam1}
            @test EquidistantFrame{:cam1} <: RetinalFrame{:cam1}
            @test GnomonicFrame{:_} <: RetinalFrame{:_}
            # the id is part of the retina: frames of different views are unrelated
            @test !(GnomonicFrame{:cam1} <: RetinalFrame{:cam2})
            # DistortedFrame is defined by the lens map, not the retina; ImageFrame
            # is the pixel frame — neither is a canonical ray frame
            @test !(DistortedFrame{:_} <: RetinalFrame)
            @test !(ImageFrame{:_} <: RetinalFrame)
            # distortion_frame returns retinal-frame kinds for every framed lens
            @test distortion_frame(Lens(Perspective(), Radial((0.1,)))) <: RetinalFrame
            @test distortion_frame(Lens(Equidistant())) <: RetinalFrame
        end
    end

    # =============================================================================
    # RayMap — the lens's sphere self-map (0.11)
    # =============================================================================
    #
    # Covers: the conjugation law π(D(v)) ≈ distort(π(v)), π the frame projection, pinned
    # numerically for a gnomonic-frame lens (OpenCV radial-tangential) and an
    # equidistant-frame lens (KannalaBrandt); azimuth preservation for radial
    # models; the θ-profile of KannalaBrandt's D equal to its defining
    # polynomial θ_d(θ); the honest inv (round-trip on the sphere); identity
    # for identity-in-frame lenses (Pinhole, Lens(Equidistant())); unit-norm
    # output and input-norm irrelevance (D acts on rays); frame typing (id
    # threading, :_ default, junction MethodError); frame-domain honesty
    # (NaN outside the defining frame); and the scoped-out parametric
    # projections.

    @testset "RayMap — the lens's sphere self-map (0.11)" begin
        # --- Fixtures ------------------------------------------------------------
        bc_lens = Lens(Perspective(), Additive(Radial((-0.28, 0.07, -0.006)), Tangential(0.0009, -0.0004)))
        bc_radial = Lens(Perspective(), Radial((-0.28, 0.07, -0.006)))   # no tangential
        kb_lens = KannalaBrandt(-0.043, 0.011, -0.0083, 0.0015)
        tpf_lens = Lens(Equidistant(), Additive(Radial((-0.02, 0.004, -0.0009, 0.0001)), Tangential(0.0006, -0.0004), ThinPrism((0.0002,), (-0.0001,))))

        # frame projections restricted to the sphere (the conjugating sections)
        gnomonic = PerspectiveMap()                       # the gnomonic chart of a ray
        equiframe = FramedEquidistantProjection()

        # forward-hemisphere rays (the gnomonic frame's domain) + wide rays past
        # 90° (equidistant-frame lenses only)
        rays_fwd = [
            SVector(0.0, 0.0, 1.0),
            SVector(0.1, -0.05, 1.0),
            SVector(-0.35, 0.2, 1.2),
            SVector(0.6, 0.45, 1.0),
            SVector(1.3, 0.2, 1.0),
        ]
        rays_wide = [rays_fwd..., SVector(0.5, -0.3, -0.4), SVector(-1.1, 0.9, -0.2)]

        @testset "frame typing — id threading, :_ default, junctions" begin
            D = ray_map(bc_lens)
            @test D isa RayMap{ViewFrame{:_}, GnomonicFrame{:_}, typeof(bc_lens)}
            @test D isa FramedStage{ViewFrame{:_}, ViewFrame{:_}}   # sphere SELF-map
            @test ray_map(kb_lens) isa
                RayMap{ViewFrame{:_}, EquidistantFrame{:_}, typeof(kb_lens)}
            @test ray_map(bc_lens; id = :left) isa
                RayMap{ViewFrame{:left}, GnomonicFrame{:left}, typeof(bc_lens)}
            # the thin-prism fisheye is read in its equidistant frame, where its
            # distortion is the additive terms; the map carries the whole lens
            Dtp = ray_map(tpf_lens)
            @test Dtp isa RayMap{ViewFrame{:_}, EquidistantFrame{:_}}
            @test Dtp.lens === tpf_lens
            # framed composition: D chains through matching ViewFrame junctions
            @test (FramedPerspective() ∘ D) isa FramedChain{ViewFrame{:_}, GnomonicFrame{:_}}
            @test (ray_map(bc_lens; id = :a) ∘ ray_map(bc_lens; id = :a)) isa
                FramedChain{ViewFrame{:a}, ViewFrame{:a}}
            # labeled junction mismatch is a MethodError (:_ is a label, not a wildcard)
            @test_throws MethodError ray_map(bc_lens; id = :left) ∘ ray_map(bc_lens; id = :right)
            @test_throws MethodError FramedPerspective(; id = :right) ∘ ray_map(bc_lens; id = :left)
            @test_throws MethodError FramedPerspective() ∘ ray_map(bc_lens; id = :left)
            # a parametric projection has no frame split — scoped out
            @test_throws ArgumentError ray_map(Lens(DoubleSphere(0.3, 0.6)))
        end

        @testset "conjugation law — radial-tangential (gnomonic frame)" begin
            D = ray_map(bc_lens)
            for X in rays_fwd
                w = D(X)
                @test norm(w) ≈ 1.0 atol = 1.0e-14              # unit representative
                @test gnomonic(w) ≈ bc_lens(gnomonic(X)) atol = 1.0e-12
            end
            # D acts on rays: the input norm is irrelevant
            @test D(2.0 * rays_fwd[4]) ≈ D(rays_fwd[4]) atol = 1.0e-15
            @test D(0.25 * rays_fwd[3]) ≈ D(rays_fwd[3]) atol = 1.0e-15
            # honest inv: round-trip on the sphere
            for X in rays_fwd
                @test inv(D)(D(X)) ≈ normalize(X) atol = 1.0e-9
            end
            @test inv(inv(D)) === D
        end

        @testset "conjugation law — KannalaBrandt (equidistant frame)" begin
            D = ray_map(kb_lens)
            for X in rays_wide
                w = D(X)
                @test norm(w) ≈ 1.0 atol = 1.0e-14
                # conjugation law in the defining frame
                uv = equiframe(X)
                θ = norm(uv)
                c = parameters(kb_lens)
                θd = θ * (1 + c.k1 * θ^2 + c.k2 * θ^4 + c.k3 * θ^6 + c.k4 * θ^8)
                # θ-profile of D IS the defining polynomial θ_d(θ) ...
                @test atan(hypot(w[1], w[2]), w[3]) ≈ θd atol = 1.0e-12
                iszero(θ) && continue                    # on-axis ray: θ_d(0) = 0 checked above
                @test equiframe(w) ≈ (θd / θ) * uv atol = 1.0e-12
                # ... and the azimuth is preserved (radial model)
                @test atan(w[2], w[1]) ≈ atan(X[2], X[1]) atol = 1.0e-12
            end
            # inv round-trip on the sphere, wide rays included
            for X in rays_wide
                @test inv(D)(D(X)) ≈ normalize(X) atol = 1.0e-10
            end
            # a radial-only gnomonic-frame model preserves azimuth too
            Dr = ray_map(bc_radial)
            for X in rays_fwd[2:end]
                w = Dr(X)
                @test atan(w[2], w[1]) ≈ atan(X[2], X[1]) atol = 1.0e-12
            end
        end

        @testset "conjugation law — thin-prism fisheye (additive terms on the equidistant frame)" begin
            D = ray_map(tpf_lens)
            bundle = x -> VgtkCore.Cameras._distort(tpf_lens.distortion, x)
            for X in rays_wide
                w = D(X)
                @test norm(w) ≈ 1.0 atol = 1.0e-14
                @test equiframe(w) ≈ bundle(equiframe(X)) atol = 1.0e-12
            end
            for X in rays_wide
                @test inv(D)(D(X)) ≈ normalize(X) atol = 1.0e-8
            end
        end

        @testset "identity-in-frame lenses — D is the identity (still framed)" begin
            Dp = ray_map(Pinhole())
            @test Dp isa RayMap{ViewFrame{:_}, GnomonicFrame{:_}, Pinhole}
            for X in rays_fwd
                @test Dp(X) ≈ normalize(X) atol = 1.0e-15
            end
            De = ray_map(Lens(Equidistant()))
            @test De isa RayMap{ViewFrame{:_}, EquidistantFrame{:_}, Lens{Equidistant, Nothing}}
            for X in rays_wide          # identity on (almost) the whole sphere
                @test De(X) ≈ normalize(X) atol = 1.0e-15
            end
            # the inverse of the identity is the identity
            @test inv(De)(rays_wide[end]) ≈ normalize(rays_wide[end]) atol = 1.0e-15
        end

        @testset "frame-domain honesty — NaN at the boundary" begin
            # gnomonic frame: forward hemisphere only (Z > 0)
            @test all(isnan, ray_map(bc_lens)(SVector(0.1, 0.0, 0.0)))
            @test all(isnan, ray_map(bc_lens)(SVector(0.1, -0.2, -1.0)))
            @test all(isnan, ray_map(Pinhole())(SVector(0.0, 0.0, -1.0)))
            @test all(isnan, inv(ray_map(bc_lens))(SVector(0.1, 0.0, -1.0)))
            # equidistant frame: θ ∈ [0, π) — only the exact backward ray is out
            @test all(isnan, ray_map(kb_lens)(SVector(0.0, 0.0, -1.0)))
            @test all(isnan, ray_map(Lens(Equidistant()))(SVector(0.0, 0.0, -2.0)))
            @test norm(ray_map(Lens(Equidistant()))(SVector(1.0e-9, 0.0, -1.0))) ≈ 1.0 atol = 1.0e-12
            # a distorted radius at or past π lifts to NaN, never a wrapped ray:
            # this KB profile sends θ ≈ π to r ≈ 24.8
            @test all(isnan, ray_map(kb_lens)(SVector(1.0e-9, 0.0, -1.0)))
            # the NaN carries NaN derivatives, as the lens contract's does
            @test all(isnan, ForwardDiff.jacobian(x -> FramedPerspective()(SVector{3}(x)), [0.0, 0.0, -1.0]))
            @test all(isnan, ForwardDiff.jacobian(x -> ray_map(Pinhole())(SVector{3}(x)), [0.1, 0.0, -1.0]))
        end
    end
end

# One FramedProjection{Src, Dst}; the five named projections are aliases with the
# Dst frame family fixed. Goldens recorded from the five structs it replaced.
@testitem "framed_projection" begin
    using Test, StaticArrays
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: FramedProjection, FramedPerspective, FramedEquidistantProjection,
        FramedStereographicProjection, FramedEquisolidProjection, FramedOrthographicProjection, FramedStage,
        GnomonicFrame, EquidistantFrame, StereographicFrame, EquisolidFrame, OrthographicFrame

    rays = [SVector(0.1, 0.2, 1.0), SVector(-0.5, 0.3, 0.8), SVector(0.0, 0.0, 1.0), SVector(1.0, -1.0, 0.2), SVector(0.3, 0.1, -0.5)]
    golden = Dict(
        FramedPerspective => [[0.1, 0.2], [-0.625, 0.375], [0.0, 0.0], [5.0, -5.0], [NaN, NaN]],
        FramedEquidistantProjection => [
            [0.0983816143377869, 0.1967632286755738], [-0.5400829446948567, 0.32404976681691405], [0.0, 0.0],
            [1.0113795137428017, -1.0113795137428017], [2.4453735148215108, 0.8151245049405036],
        ],
        FramedStereographicProjection => [
            [0.09878030638383933, 0.19756061276767867], [-0.5586749813563722, 0.33520498881382327], [0.0, 0.0],
            [1.22828568570857, -1.22828568570857], [6.549647869859769, 2.183215956619923],
        ],
        FramedEquisolidProjection => [
            [0.0981833530726402, 0.1963667061452804], [-0.531200034824358, 0.31872002089461476], [0.0, 0.0],
            [0.9273467483078803, -0.9273467483078803], [1.8224372852066348, 0.6074790950688783],
        ],
        FramedOrthographicProjection => [
            [0.09759000729485331, 0.19518001458970663], [-0.5050762722761053, 0.3030457633656632], [0.0, 0.0],
            [0.7001400420140049, -0.7001400420140049], [NaN, NaN],
        ],
    )
    families = Dict(
        FramedPerspective => GnomonicFrame, FramedEquidistantProjection => EquidistantFrame,
        FramedStereographicProjection => StereographicFrame, FramedEquisolidProjection => EquisolidFrame,
        FramedOrthographicProjection => OrthographicFrame,
    )
    for (A, g) in golden
        p = A()
        F = families[A]
        @test p isa FramedProjection{ViewFrame{:_}, F{:_}} && p isa A && p isa FramedStage
        @test typeof(A(; id = :left)) === FramedProjection{ViewFrame{:left}, F{:left}}
        @test A{ViewFrame{:x}, F{:x}}() isa FramedProjection{ViewFrame{:x}, F{:x}}
        @test_throws MethodError inv(p)
        for (r, gr) in zip(rays, g)
            @test isequal(collect(p(r)), gr)
        end
    end
    @test FramedPerspective <: FramedProjection
    @test !(FramedPerspective{ViewFrame{:_}, GnomonicFrame{:_}} <: FramedEquisolidProjection)
    @test !(FramedProjection{ViewFrame{:_}, EquisolidFrame{:_}}() isa FramedPerspective)
end
