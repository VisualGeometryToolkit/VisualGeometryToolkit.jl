@testitem "retinal_frames" begin
    using Test
    using VisualGeometryCore: imaging_chain, FramedStage, FramedPerspective,
        FramedEquidistantProjection, FramedAffine, FramedChain, distort_stage,
        FramedStereographicProjection, FramedEquisolidProjection, FramedOrthographicProjection,
        FrameTransition, frame_transition, RayMap, ray_map,
        RetinalFrame, EquidistantFrame, StereographicFrame, EquisolidFrame,
        OrthographicFrame, DistortedFrame, FrameDependence, GeometryDefined, ModelDefined,
        frame_dependence
    using VisualGeometryCore
    using StaticArrays
    using LinearAlgebra: norm, normalize

    # =============================================================================
    # The remaining retinal frames (0.13) — stereographic, equisolid,
    # orthographic — and their pure-projection lens models
    # =============================================================================
    #
    # Covers: the radial profiles and honest domains of the three new frames
    # (with the orthographic FOLD at the equator pinned as NaN, not an
    # alias); the framed stage-1 projections (typing, no inv, pole conventions,
    # scale invariance on ray representatives); the sphere section/lift bridges
    # (round-trips on the sphere, NaN at the frame edges); the generic
    # frame_transition (radial-profile pins for EVERY ordered pair among the
    # five frames, inverse round-trips, id junction checking, the
    # gnomonic↔equidistant pin against the ray-map internals and the lens
    # closed forms — OpenCV's hidden tan(θ)/θ_d line as a named object); and
    # the pure-projection fisheye models (identity in frame: imaging_chain
    # collapse, ray_map framed identity, closed-form inverses with honest
    # convergence flags, legacy-route parity).

    const _Cameras = VisualGeometryCore.Cameras

    # The five radial profiles as plain functions of the polar angle θ.
    _profiles = (
        (GnomonicFrame, tan, "gnomonic"),
        (EquidistantFrame, identity, "equidistant"),
        (StereographicFrame, θ -> 2 * tan(θ / 2), "stereographic"),
        (EquisolidFrame, θ -> 2 * sin(θ / 2), "equisolid"),
        (OrthographicFrame, sin, "orthographic"),
    )

    # Upper edge of each frame's honest θ-domain (open or closed is pinned in
    # the per-frame testsets; this bound is for generating in-domain samples).
    _theta_max = Dict(
        GnomonicFrame => π / 2, EquidistantFrame => Float64(π),
        StereographicFrame => Float64(π), EquisolidFrame => Float64(π),
        OrthographicFrame => π / 2,
    )

    _ray(θ, ψ) = SVector(sin(θ) * cos(ψ), sin(θ) * sin(ψ), cos(θ))

    @testset "Retinal frames (0.13) — the classical fisheye family complete" begin
        @testset "FramedStereographicProjection — conformal stage" begin
            p = FramedStereographicProjection()
            @test p isa FramedStereographicProjection{ViewFrame{:_}, StereographicFrame{:_}}
            @test FramedStereographicProjection(; id = :left) isa
                FramedStereographicProjection{ViewFrame{:left}, StereographicFrame{:left}}
            @test_throws MethodError inv(p)
            for θ in (0.2, 1.0, π / 2, 2.4), ψ in (0.0, 1.1, -2.0)
                uv = p(_ray(θ, ψ))
                @test norm(uv) ≈ 2 * tan(θ / 2) atol = 1.0e-13
                @test atan(uv[2], uv[1]) ≈ atan(sin(ψ), cos(ψ)) atol = 1.0e-13
            end
            # scale invariance on ray representatives
            @test p(3.0 * _ray(1.2, 0.4)) ≈ p(_ray(1.2, 0.4)) atol = 1.0e-13
            # forward pole convention; the antipode (θ = π) is outside the frame
            @test p(SVector(0.0, 0.0, 2.0)) == Vec2(0.0, 0.0)
            @test all(isnan, p(SVector(0.0, 0.0, -1.0)))
        end

        @testset "FramedEquisolidProjection — equal-area stage, whole sphere" begin
            p = FramedEquisolidProjection()
            @test p isa FramedEquisolidProjection{ViewFrame{:_}, EquisolidFrame{:_}}
            @test_throws MethodError inv(p)
            for θ in (0.0, 0.3, π / 2, 2.0, 3.1), ψ in (0.0, 2.2)
                uv = p(_ray(θ, ψ))
                @test norm(uv) ≈ 2 * sin(θ / 2) atol = 1.0e-13
            end
            # bounded radius r ∈ [0, 2]; the antipode is IN the frame at (2, 0)
            @test p(SVector(0.0, 0.0, 1.0)) == Vec2(0.0, 0.0)
            @test p(SVector(0.0, 0.0, -3.0)) == Vec2(2.0, 0.0)   # pole azimuth convention
            @test norm(p(_ray(3.14159, 1.0))) <= 2.0 + 1.0e-12
            @test p(0.5 * _ray(2.5, -0.7)) ≈ p(_ray(2.5, -0.7)) atol = 1.0e-13
        end

        @testset "FramedOrthographicProjection — past the fold is NaN" begin
            p = FramedOrthographicProjection()
            @test p isa FramedOrthographicProjection{ViewFrame{:_}, OrthographicFrame{:_}}
            @test_throws MethodError inv(p)
            for θ in (0.0, 0.4, 1.2), ψ in (0.0, -1.3)
                uv = p(_ray(θ, ψ))
                @test norm(uv) ≈ sin(θ) atol = 1.0e-13
            end
            # the closed boundary θ = π/2 is IN the frame (r = 1)...
            @test norm(p(SVector(0.6, -0.8, 0.0))) ≈ 1.0 atol = 1.0e-13
            # ...but past it the frame FOLDS (sin θ non-injective): backward
            # rays are OUTSIDE, never aliased onto forward radii
            @test all(isnan, p(SVector(0.1, 0.0, -0.05)))
            @test all(isnan, p(_ray(2.0, 0.3)))
            @test all(isnan, p(SVector(0.0, 0.0, -1.0)))
            @test p(4.0 * _ray(0.9, 2.0)) ≈ p(_ray(0.9, 2.0)) atol = 1.0e-13
        end

        @testset "section/lift round-trips on the sphere — all five frames" begin
            for (C, profile, name) in _profiles
                Ci = C{:_}
                for θ in (0.0, 0.3, 1.0, 0.98 * _theta_max[C]), ψ in (0.0, 1.7, -2.4)
                    v = _ray(θ, ψ)
                    x = _Cameras._sphere_section(Ci, v)
                    # the section realizes the radial profile...
                    @test norm(x) ≈ profile(θ) atol = 1.0e-11
                    # ...and the lift inverts it back to the unit ray
                    w = _Cameras._sphere_lift(Ci, x)
                    @test norm(w) ≈ 1.0 atol = 1.0e-12
                    @test w ≈ v atol = 1.0e-10
                end
            end
            # lift domain edges: beyond the bounded frames' radii is no ray
            @test all(isnan, _Cameras._sphere_lift(EquisolidFrame{:_}, SVector(2.001, 0.0)))
            @test all(isnan, _Cameras._sphere_lift(OrthographicFrame{:_}, SVector(1.001, 0.0)))
            # the stereographic lift is total (θ < π for every finite radius)
            @test norm(_Cameras._sphere_lift(StereographicFrame{:_}, SVector(500.0, 0.0))) ≈ 1.0 atol = 1.0e-12
            # the orthographic lift selects the FORWARD sheet (the fold)
            @test _Cameras._sphere_lift(OrthographicFrame{:_}, SVector(0.8, 0.0))[3] > 0
            # section domain edges
            @test all(isnan, _Cameras._sphere_section(StereographicFrame{:_}, SVector(0.0, 0.0, -1.0)))
            @test all(isnan, _Cameras._sphere_section(OrthographicFrame{:_}, SVector(0.1, 0.0, -0.1)))
            # equisolid: the whole sphere, antipode included
            @test _Cameras._sphere_section(EquisolidFrame{:_}, SVector(0.0, 0.0, -1.0)) == Vec2(2.0, 0.0)
            @test _Cameras._sphere_lift(EquisolidFrame{:_}, SVector(2.0, 0.0)) ≈ Vec3(0.0, 0.0, -1.0) atol = 1.0e-12
        end

        @testset "frame_transition — radial pins for every ordered pair" begin
            # θ = 0.6 lies in the honest domain of all five frames
            θ = 0.6
            for (C1, p1, n1) in _profiles, (C2, p2, n2) in _profiles
                t = frame_transition(C1{:_}, C2{:_})
                @test t isa FrameTransition{C1{:_}, C2{:_}}
                for ψ in (0.0, 1.9)
                    x = p1(θ) * SVector(cos(ψ), sin(ψ))
                    y = t(x)
                    @test norm(y) ≈ p2(θ) atol = 1.0e-12       # radial profile pin
                    iszero(norm(y)) || @test atan(y[2], y[1]) ≈ atan(sin(ψ), cos(ψ)) atol = 1.0e-12
                    # inverse round-trip (the reverse pair)
                    @test inv(t)(y) ≈ x atol = 1.0e-10
                end
            end
            # wide-angle pins past 90°, where the gnomonic/orthographic frames
            # cannot go: the three wide frames agree through the sphere
            θw = 2.2
            for (C1, p1, _) in _profiles[2:4], (C2, p2, _) in _profiles[2:4]
                x = p1(θw) * SVector(1.0, 0.0)
                @test frame_transition(C1{:_}, C2{:_})(x)[1] ≈ p2(θw) atol = 1.0e-12
            end
        end

        @testset "frame_transition — frames, junctions, honest edges" begin
            t = frame_transition(EquidistantFrame{:_}, GnomonicFrame{:_})
            @test t isa FramedStage{EquidistantFrame{:_}, GnomonicFrame{:_}}
            @test inv(t) isa FrameTransition{GnomonicFrame{:_}, EquidistantFrame{:_}}
            @test inv(inv(t)) === t
            # the id must match: frames of different views never meet
            @test_throws MethodError frame_transition(EquidistantFrame{:left}, GnomonicFrame{:right})
            @test frame_transition(EquidistantFrame{:left}, GnomonicFrame{:left}) isa
                FrameTransition{EquidistantFrame{:left}, GnomonicFrame{:left}}
            # composes junction-checked with the framed surface
            c = t ∘ FramedEquidistantProjection()
            @test c isa FramedChain{ViewFrame{:_}, GnomonicFrame{:_}}
            @test_throws MethodError t ∘ FramedPerspective()      # Bearing into Equidistant slot
            @test_throws MethodError t ∘ FramedStereographicProjection()
            # honest edges: the junction of domains is NaN, never an alias
            @test all(isnan, t(SVector(π / 2 + 0.01, 0.0)))       # Z ≤ 0 for gnomonic
            @test all(isnan, frame_transition(EquidistantFrame{:_}, OrthographicFrame{:_})(SVector(1.7, 0.0)))
            @test all(isnan, frame_transition(EquisolidFrame{:_}, StereographicFrame{:_})(SVector(2.0, 0.0)))
            @test all(isnan, frame_transition(EquisolidFrame{:_}, GnomonicFrame{:_})(SVector(2.5, 0.0)))
        end

        @testset "the gnomonic↔equidistant pin — OpenCV's tan(θ)/θ_d line, named" begin
            # cv::fisheye::undistortPoints' final `scale = tan(θ)/θ_d` IS this
            # transition; the pure equidistant fisheye's closed forms are its
            # two directions. Pinned against the ray-map machinery's own
            # section/lift internals AND against the lens closed forms.
            te = frame_transition(EquidistantFrame{:_}, GnomonicFrame{:_})
            tb = inv(te)
            for x in (SVector(0.3, -0.2), SVector(1.1, 0.6), SVector(0.0, 0.0))
                @test te(x) ≈ undistort_normalized(Lens(Equidistant()), x) atol = 1.0e-13
                @test te(x) == _Cameras._sphere_section(
                    GnomonicFrame{:_}, _Cameras._sphere_lift(EquidistantFrame{:_}, x)
                )
                θ = norm(x)
                iszero(θ) || @test norm(te(x)) ≈ tan(θ) atol = 1.0e-12
            end
            for x in (SVector(0.4, 0.1), SVector(-0.8, 0.9))
                @test tb(x) ≈ Lens(Equidistant())(x) atol = 1.0e-13
            end
            # every pure-projection fisheye callable IS the gnomonic → frame
            # transition of its defining frame (the frame is the model)
            for (lens, C) in (
                    (Lens(Stereographic()), StereographicFrame),
                    (Lens(Equisolid()), EquisolidFrame),
                    (Lens(Orthographic()), OrthographicFrame),
                )
                tt = frame_transition(GnomonicFrame{:_}, C{:_})
                for x in (SVector(0.5, -0.4), SVector(0.05, 0.02))
                    @test lens(x) ≈ tt(x) atol = 1.0e-12
                    @test undistort_normalized(lens, lens(x)) ≈ inv(tt)(tt(x)) atol = 1.0e-11
                end
            end
        end

        @testset "pure-projection fisheye models — frame tags + collapse" begin
            for (lens, C) in (
                    (Lens(Stereographic()), StereographicFrame),
                    (Lens(Equisolid()), EquisolidFrame),
                    (Lens(Orthographic()), OrthographicFrame),
                )
                @test distortion_frame(lens) === C
                @test C{:_} <: RetinalFrame{:_}
                # identity in frame: no distortion stage — the chain collapses
                @test_throws ArgumentError distort_stage(lens)
                m = CameraModel(; focal = (620.0, 610.0), pp = (321.5, 242.5), lens)
                c = imaging_chain(m)
                @test c isa FramedChain{ViewFrame{:_}, ImageFrame{:_}}
                @test length(c.stages) == 2
                @test c.stages[2] isa FramedAffine{C{:_}, ImageFrame{:_}}
                # numerical parity with the frame-free imaging_transform
                legacy = imaging_transform(m.intrinsics, m.lens)
                for X in (SVector(0.0, 0.0, 1.0), SVector(0.1, -0.05, 1.0), SVector(0.6, 0.45, 1.0))
                    got = c(X)
                    want = legacy(X)
                    @test isapprox(got[1], want[1]; atol = 1.0e-10, rtol = 1.0e-12)
                    @test isapprox(got[2], want[2]; atol = 1.0e-10, rtol = 1.0e-12)
                end
            end
            # labeled minting threads the id end to end
            cl = imaging_chain(
                CameraModel(; focal = (500.0, 500.0), pp = (320.0, 240.0), lens = Lens(Equisolid()));
                id = :left
            )
            @test cl isa FramedChain{ViewFrame{:left}, ImageFrame{:left}}
            @test cl.stages[1] isa FramedEquisolidProjection{ViewFrame{:left}, EquisolidFrame{:left}}
        end

        @testset "pure-projection fisheye models — ray_map is the framed identity" begin
            rays = [SVector(0.0, 0.0, 1.0), SVector(0.3, -0.2, 1.0), SVector(0.9, 0.7, 1.1)]
            for lens in (Lens(Stereographic()), Lens(Equisolid()), Lens(Orthographic()))
                D = ray_map(lens)
                @test D isa RayMap{ViewFrame{:_}, distortion_frame(lens){:_}, typeof(lens)}
                for X in rays
                    @test D(X) ≈ normalize(X) atol = 1.0e-14
                    @test inv(D)(D(X)) ≈ normalize(X) atol = 1.0e-14
                end
            end
            # domain honesty follows each defining frame
            @test all(isnan, ray_map(Lens(Orthographic()))(SVector(0.1, 0.0, -0.2)))
            @test all(isnan, ray_map(Lens(Stereographic()))(SVector(0.0, 0.0, -1.0)))
            # wide rays: fine for stereographic (θ < π), fine for equisolid anywhere
            @test norm(ray_map(Lens(Stereographic()))(SVector(0.5, 0.0, -0.4))) ≈ 1.0 atol = 1.0e-13
            @test ray_map(Lens(Equisolid()))(SVector(0.0, 0.0, -1.0)) ≈ Vec3(0.0, 0.0, -1.0) atol = 1.0e-13
        end

        @testset "perspective views — NaN past the perspective plane" begin
            # the perspective point exists only for θ < π/2 — r_d strictly below
            # the equator's radius (stereographic 2, equisolid √2, orthographic 1).
            # Past it the stereographic and equisolid lenses still image a ray;
            # the orthographic one images nothing.
            for (lens, r_edge, past) in (
                    (Lens(Stereographic()), 2.0, true),
                    (Lens(Equisolid()), sqrt(2.0), true),
                    (Lens(Orthographic()), 1.0, false),
                )
                x_in = SVector(0.9 * r_edge, 0.0)
                x = undistort_normalized(lens, x_in)
                @test lens(x) ≈ x_in atol = 1.0e-10
                x_out = SVector(1.1 * r_edge, 0.0)
                @test all(isnan, undistort_normalized(lens, x_out))
                b = backproject(lens, x_out)
                @test past ? (all(isfinite, b) && b[3] < 0) : all(isnan, b)
            end
            # a batch: NaN-poison through the same view
            out = undistort_normalized!(similar([SVector(0.0, 0.0)], 2), Lens(Orthographic()), [SVector(0.5, 0.0), SVector(1.5, 0.0)])
            @test all(isnan, out[2])
            @test out[1][1] ≈ 0.5 / sqrt(1 - 0.25) atol = 1.0e-12
        end
    end

    # =============================================================================
    # FrameDependence (0.14): what pins a frame tag down — dispatched, not
    # documented. Trait values for every frame tag; frame_transition's
    # ModelDefined refusal; sphere-section availability.
    # =============================================================================

    @testset "frame_dependence: trait values pinned for every frame tag" begin
        # GeometryDefined: fixed by the retina's geometry/conventions alone.
        for C in (
                GnomonicFrame, EquidistantFrame, StereographicFrame,
                EquisolidFrame, OrthographicFrame,
            )
            @test frame_dependence(C) === GeometryDefined()
            @test frame_dependence(C{:cam}) === GeometryDefined()
        end
        @test frame_dependence(ImageFrame) === GeometryDefined()
        @test frame_dependence(ImageFrame{:cam}) === GeometryDefined()
        @test frame_dependence(PixelFrame{:colmap}) === GeometryDefined()
        @test frame_dependence(TargetPlaneFrame) === GeometryDefined()
        # ModelDefined: pinned down by the view's lens parameters.
        @test frame_dependence(DistortedFrame) === ModelDefined()
        @test frame_dependence(DistortedFrame{:cam}) === ModelDefined()
        @test GeometryDefined() isa FrameDependence
        @test ModelDefined() isa FrameDependence
    end

    @testset "frame_transition: ModelDefined endpoints refuse with the reason" begin
        for (a, b) in (
                (DistortedFrame{:_}, GnomonicFrame{:_}),
                (GnomonicFrame{:_}, DistortedFrame{:_}),
                (DistortedFrame{:_}, DistortedFrame{:_}),
            )
            err = try
                frame_transition(a, b)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("ModelDefined", err.msg)
            @test occursin("bring the lens", err.msg)
        end
        # GeometryDefined pairs that fail the same-view rule keep the
        # labeled-junction MethodError.
        @test_throws MethodError frame_transition(GnomonicFrame{:a}, EquidistantFrame{:b})
    end

    @testset "sphere sections/lifts: GeometryDefined only" begin
        S = VisualGeometryCore.Cameras._sphere_section
        L = VisualGeometryCore.Cameras._sphere_lift
        # Every GeometryDefined retinal frame has a section and a lift.
        v = SVector(0.1, -0.2, 0.9)
        for C in (GnomonicFrame, EquidistantFrame, StereographicFrame, EquisolidFrame)
            x = S(C{:_}, v)
            @test L(C{:_}, x) ≈ normalize(v) atol = 1.0e-12
        end
        # The ModelDefined frame refuses — no model-free relation to the sphere.
        err = try
            S(DistortedFrame{:_}, v)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("ModelDefined", err.msg)
        @test_throws ArgumentError L(DistortedFrame{:_}, SVector(0.1, 0.2))
    end
end
