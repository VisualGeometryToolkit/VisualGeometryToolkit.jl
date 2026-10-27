# A Radial factor stores where its imaged branch ends (`u_fold`, `u_end`),
# computed once in the constructor by the sweep `_branch_end` runs; the per-point
# branch test is a comparison against it. Checked here: the stored ends are that
# sweep's, for every lens family of the suite; the comparison decides as the
# per-point sweep it replaced did, except inside the sweep's stall sliver next to
# the end; and the ends are not parameters (names, records, rebuilds, ==, hash).
@testitem "radial_branch_end" begin
    using Test
    using StaticArrays: SVector
    using ForwardDiff
    using VisualGeometryToolkitCore
    const C = VisualGeometryToolkitCore.Cameras

    radials = Any[
        C.radial(KannalaBrandt(-0.1)), C.radial(KannalaBrandt(-0.2, 0.01)), C.radial(KannalaBrandt(-0.3)),
        C.radial(KannalaBrandt(-0.043, 0.011, -0.0083, 0.0015)), C.radial(KannalaBrandt(0.5, -0.4, 0.1, 0.0)),
        Radial((-0.3, 0.04)), Radial((-0.12,)), Radial((0.05,), (-0.5,)), Radial((-0.12, 0.03, 0.004), (0.02, -0.001, 0.0003)),
        Radial((-0.2817, 0.0712, -0.0061)), Radial((-0.150994, -0.0501823, 0.023916), (0.187097, 0.0255086, -0.00128661)),
        Radial((1.40468, -0.00298835, 0.0664061), (-2.03318, 0.387737, 0.0918448)),
        Radial((-2.7067990842834013, 4.003544352914542, 1.5827483616019924), (-2.5409744152370775, 3.5017575444899127, 2.4059525010963547)),
        Division(-0.31).radial, Division(0.12).radial, Division(0.2).radial, Division(-0.31, 0.024).radial,
        Division(-0.2, 0.03, -0.004).radial, Division(11 / 6, -1 / 3, 1 / 30).radial,
        LiftLens(Radial((0.05,), (-0.25, 0.02))).radial, LiftLens(Radial((0.03, -0.002), (-0.2,))).radial,
        LiftLens(Radial((-0.4,), (0.1,))).radial,
        Scaramuzza(-0.17, -0.006, 0.035).radial, Scaramuzza(-0.17, -0.006, 1.0e-17).radial,   # a rounding-artefact lead
        Radial(()), Radial((0.0, 0.0), (0.0,)),
    ]
    append!(radials, [VisualGeometryToolkitCore._precision(Float32, LiftLens(r)).radial for r in radials])
    # the one factor whose stored end sits far short of the per-point sweep's (below)
    artefact32 = VisualGeometryToolkitCore._precision(Float32, Scaramuzza(-0.17, -0.006, 1.0e-17)).radial
    @test count(r -> r isa Radial{<:Any, <:Any, <:Any, Float32}, radials) == length(radials) ÷ 2

    # the per-point test the stored end replaced: a sweep from 0 to u, per factor
    old_branch(c, u) = C._positive_prefix(c, u) >= u
    old_lens(r, u) = old_branch(C._fold_coeffs(r), u) && old_branch(C._pole_coeffs(r), u)
    old_lift(r, u) = old_branch(C._fold_coeffs(r), u)

    @testset "the stored ends are the sweep's: $(typeof(r))" for r in radials
        @test r.u_end === min(C._fold_point(C._fold_coeffs(r), Inf), C._fold_point(C._pole_coeffs(r), Inf))
        @test r.u_fold === C._fold_point(C._fold_coeffs(r), Inf)
        @test r.u_end <= r.u_fold
        @test C._branch_end(r, Inf) === r.u_end && C._branch_end(r, 0.5) === min(r.u_end, typeof(r.u_end)(0.5))
    end

    # The sweep stops at a point it certified positive, within 1e-12 (relative)
    # of the root it stalls at: the stored end is that point, so the comparison
    # rejects the sliver between it and the root, where the per-point sweep to u
    # could still certify u. It never accepts a u the sweep rejected. Measured
    # over this set: every disagreement is such a u, at most 1.0002e-12 past the
    # end in Float64, 3.1e-7 (five ulps) in Float32 — except the Float32
    # Scaramuzza whose rounding-artefact lead folds it at u = 3.2e9, where the
    # long sweep's certified steps stop 1.8% short of the per-point sweep's.
    @testset "one comparison decides as the sweep did, but inside its stall sliver" begin
        wide = Set{Any}()                          # factors whose disagreements reach past 1e-6
        for r in radials
            T = eltype(r.k) === Union{} ? typeof(r.u_end) : eltype(r.k)
            for (end_, new, old) in ((r.u_end, u -> C._on_radial_branch(r, u), u -> old_lens(r, u)),
                                     (r.u_fold, u -> u <= r.u_fold, u -> old_lift(r, u)))
                sliver = T === Float64 ? 1.0e-11 : r === artefact32 ? 0.05 : 1.0e-6
                span = isfinite(end_) ? 3 * Float64(end_) : 50.0
                us = T.(collect(range(0, span, length = 4001)))
                if isfinite(end_)
                    append!(us, [T(Float64(end_) * (1 + s * 10.0^-e)) for e in 3:0.25:17 for s in (-1, 1)])
                    push!(us, end_, prevfloat(end_), nextfloat(end_))
                end
                for u in us
                    new(u) == old(u) && continue
                    @test !new(u) && old(u) && isfinite(end_) && Float64(end_) < Float64(u) <= Float64(end_) * (1 + sliver)
                    Float64(u) > Float64(end_) * (1 + 1.0e-6) && push!(wide, r)
                end
            end
        end
        @test length(wide) == 1 && only(wide) === artefact32
    end

    @testset "a Dual factor's ends are its primal's" begin
        r = Radial((-0.3, 0.04))
        d = Radial((ForwardDiff.Dual(-0.3, 1.0), ForwardDiff.Dual(0.04, 0.0)))
        @test ForwardDiff.value(d.u_end) === r.u_end
        @test ForwardDiff.value(d.u_fold) === r.u_fold
        @test C._primal(d) === r
        # widening casts the ends (no new sweep); the primal of a Float64
        # differentiation of a Float32 lens is that lens at Float64
        l32 = VisualGeometryToolkitCore._precision(Float32, KannalaBrandt(-0.3))
        l64 = VisualGeometryToolkitCore._precision(Float64, l32)
        @test l64.distortion.u_end === Float64(l32.distortion.u_end)
        θd = ForwardDiff.Dual.(Float64.(parameter_vector(l32)), 1.0)
        @test C._primal(C._with_own_parameters(l32, θd)) === l64
        # narrowing recomputes from the rounded coefficients
        @test VisualGeometryToolkitCore._precision(Float32, KannalaBrandt(-0.3)).distortion.u_end ===
            C._fold_point(C._fold_coeffs(Radial((-0.3f0,))), Inf)
    end

    @testset "integer coefficients: no throw, ends rounded down" begin
        @test Radial{C.Even, 1, 0, Int}((1,), ()).u_end == typemax(Int)        # never folds
        r = Radial{C.Even, 1, 0, Int}((-1,), ())                              # ρ(1 − ρ²) folds at u = 1/3
        @test r.u_fold == 0 && r.u_end == 0
        @test Radial((-1,), ()).u_fold ≈ 1 / 3
    end

    # Both precisions, the lift and radial families: the view-angle limit is finite
    # (the stored end, stepped inside by a precision-scaled margin),
    # the lens circle exists where the limit is the lens's own, and the two
    # precisions agree.
    @testset "view_angle_limit and the image circle, Float64 and Float32" begin
        tam = LiftLens(Radial((), (-0.17114550341263696, 0.029918349030169777)))
        fams = Any[Division(0.12), Division(0.2), Division(-0.31), Division(-0.2, 0.03, -0.004), Division(11 / 6, -1 / 3, 1 / 30),
            LiftLens(Radial((0.05,), (-0.25, 0.02))), LiftLens(Radial((-0.4,), (0.1,))), tam, Scaramuzza(-0.17, -0.006, 0.035),
            KannalaBrandt(-0.3), KannalaBrandt(-0.1), Lens(Perspective(), Radial((-0.3, 0.04))), Lens(Perspective(), Radial((-0.12,))),
            Lens(Perspective(), Additive(Radial((-0.3, 0.04)), Tangential(1.0e-3, -5.0e-4)))]
        for l64 in fams
            θ64 = view_angle_limit(l64)
            for T in (Float64, Float32)
                l = VisualGeometryToolkitCore._precision(T, l64)
                θ = view_angle_limit(l)
                @test isfinite(θ) && isapprox(θ, θ64; rtol = T === Float64 ? 1.0e-12 : 1.0e-5)
                # imaged just inside a limit of the lens's own (exactly AT a lift's
                # fold, the ray's root is √eps-conditioned and may land past it)
                θ < C._view_angle_edge(l) && @test isfinite(C.projected_radius(l, θ * (1 - 1.0e-6)))
                ρc = C._lens_circle_radius(l)
                @test (ρc === nothing) == (C._lens_circle_radius(l64) === nothing)
                ρc === nothing || @test isfinite(ρc) && ρc > 0
            end
        end
    end

    # Value and derivative agree on what is imaged at the branch's end: rays
    # across the view-angle limit, a Float32 lens differentiated at Float64 too.
    @testset "the Dual path at the fold edge" begin
        for l64 in (KannalaBrandt(-0.3), Lens(Perspective(), Radial((-0.3, 0.04))), Division(0.2), LiftLens(Radial((0.05,), (-0.25, 0.02)))),
                T in (Float64, Float32)
            l = VisualGeometryToolkitCore._precision(T, l64)
            θl = view_angle_limit(l)
            n_imaged = 0
            for θ in range(θl * (1 - 1.0e-5), θl * (1 + 1.0e-5), length = 401)
                X = SVector(sin(θ), 0.0, cos(θ))
                u = project(l, X)
                u2, JX, Jp = C.project_with_jacobians(l, X)
                Jq = VisualGeometryToolkitCore.transform_deriv_params(l, X)
                @test isnan(u[1]) == isnan(u2[1]) == !all(isfinite, JX) == !all(isfinite, Jp) == !all(isfinite, Jq)
                n_imaged += !isnan(u[1])
            end
            @test 0 < n_imaged < 401                 # the sweep crosses the end
        end
    end

    @testset "the ends are not parameters" begin
        lens = Lens(Perspective(), Additive(Radial((-0.12, 0.03, 0.004), (0.02, -0.001, 0.0003)), Tangential(1.0e-3, -5.0e-4)))
        @test parameter_names(lens) == (:k1, :k2, :k3, :d1, :d2, :d3, :p1, :p2)
        @test length(parameter_vector(lens)) == 8
        @test with_parameters(lens, parameter_vector(lens)) == lens
        @test lens_from_record(lens_record(lens)) == lens
        @test lens_record(lens_from_record(lens_record(lens))) == lens_record(lens)
        @test Radial((-0.3, 0.04)) == Radial((-0.3, 0.04)) && hash(Radial((-0.3, 0.04))) == hash(Radial((-0.3, 0.04)))
        @test Radial((-0.3, 0.04)) != Radial((-0.3, 0.05))
        @test zero(KannalaBrandt{2}).distortion.u_end === Inf
        @test repr(Radial((-0.3,), (0.1,))) == repr(Radial{C.Even, 1, 1, Float64}) * "((-0.3,), (0.1,))"
        @test isbits(Radial((-0.3, 0.04)))
    end
end
