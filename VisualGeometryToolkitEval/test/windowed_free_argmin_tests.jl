# =============================================================================
# windowed_free_argmin_tests.jl — regression gate: the windowed selection (the WindowedBootstrap scheme's solve)
# must be the FREE ARGMIN over every native-grid column inside the standardized window
# [to_native(lo, σₙ), to_native(hi, σₙ)] — NEVER the retired snap onto a coarse efficiency-constant axis.
#
# The snap bug shipped panels tuned on a few candidates while the paper
# described a 200-column free argmin (caught 2026-07-06). This test plants the
# validation minimum at a column that is NOT near any snapped candidate: the
# snap cannot find it, the windowed free argmin must.
# =============================================================================

@testitem "windowed_free_argmin" setup = [TuningFixtures] begin
    import VisualGeometryToolkitEval as VgtkEval
    using Test
    using VisualGeometryToolkitEval.Evaluation: ValidationErrors
    using VisualGeometryToolkitEval.Tuning: EfficiencyConstantWindow, MedianError
    using VisualGeometryToolkitCore: solve, EstimationProblem, ClosedFormKind, estimator_kind, ReturnCode
    using VisualGeometryToolkitEstimation: Hyperparameter

    @testset "the windowed selection is the windowed free argmin" begin
        # loss-width map: native = efficiency_constant·σ with σ = 1 ⇒ window is [1, 3.3].
        scheme = fixture_spec(Dict{String,Any}("type" => "windowed_bootstrap",
            "efficiency_constant_window" => Dict{String,Any}("lo" => 1.0, "hi" => 3.3))).scheme
        @test scheme.efficiency_constant_window == EfficiencyConstantWindow(1.0, 3.3) && scheme.criterion isa MedianError
        @test estimator_kind(scheme) === ClosedFormKind()
        g = collect(range(0.5, 4.0; length = 100))
        loss_width_map(m, σ) = m * σ
        planted = argmin(abs.(g .- 2.335))
        errs = fill(10.0, 4, length(g)); errs[:, planted] .= 0.1
        pl = [(1, j) for j in 1:4]
        select(e, σ2) = solve(EstimationProblem(Hyperparameter, ValidationErrors("t", [e], [fill(σ2, 4)], g),
                                                (; pairs = pl, to_native = loss_width_map)), scheme)

        @test select(errs, 1.0).u == planted

        # Window anchoring must survive: corrupt σ so the window [64, 211] excludes
        # the whole grid — the fallback picks the grid's top edge, NOT the global
        # minimum at `planted`.
        col_c = select(errs, 64.0).u
        @test col_c == length(g)
        @test col_c != planted

        # Minimum OUTSIDE the window must not be reachable at calibration either.
        errs2 = fill(10.0, 4, length(g))
        below = argmin(abs.(g .- 0.6))
        inwin = argmin(abs.(g .- 1.7))
        errs2[:, below] .= 0.05; errs2[:, inwin] .= 0.2
        @test select(errs2, 1.0).u == inwin

        # no finite σ̂² among the drawn pairs: degenerate, no column
        r = select(errs, NaN)
        @test r.u === nothing && r.retcode === ReturnCode.InitialFailure
    end
end
