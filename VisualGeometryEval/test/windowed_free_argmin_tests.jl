# =============================================================================
# windowed_free_argmin_tests.jl — regression gate: WindowedGridMethod tuning must
# be the FREE ARGMIN over every native-grid column inside the standardized
# window [to_native(lo, σₙ), to_native(hi, σₙ)] — NEVER the retired snap onto
# a coarse scale-multiple axis.
#
# The snap bug shipped panels tuned on a few candidates while the paper
# described a 200-column free argmin (caught 2026-07-06). This test plants the
# validation minimum at a column that is NOT near any snapped candidate: the
# snap cannot find it, the windowed free argmin must.
# =============================================================================

@testitem "windowed_free_argmin" begin
    using Test, Random
    using VisualGeometryEval.Tuning: ScaleMultipleWindow
    using VisualGeometryEval.Tuning: MedianError
    const RSDRT = VisualGeometryEval.Evaluation

    @testset "WindowedGridMethod tunes by windowed free argmin" begin
        # loss-width map: native = scale_multiple·σ with σ = 1 ⇒ window is [1, 3.3].
        g = collect(range(0.5, 4.0; length = 100))
        window = ScaleMultipleWindow(1.0, 3.3)
        loss_width_map(m, σ) = m * σ
        planted = argmin(abs.(g .- 2.335))
        errs = fill(10.0, 4, length(g)); errs[:, planted] .= 0.1
        km = RSDRT.WindowedGridMethod("t", [errs], [fill(1.0, 4)], g, window, loss_width_map, MedianError())
        pl = [(1, j) for j in 1:4]

        col = RSDRT.tune(MersenneTwister(1), km, pl, 4)
        @test col == planted

        # Window anchoring must survive: corrupt σ so the window [64, 211] excludes
        # the whole grid — the fallback picks the grid's top edge, NOT the global
        # minimum at `planted`.
        km_corrupt = RSDRT.WindowedGridMethod("t", [errs], [fill(64.0, 4)], g, window, loss_width_map, MedianError())
        col_c = RSDRT.tune(MersenneTwister(1), km_corrupt, pl, 4)
        @test col_c == length(g)
        @test col_c != planted

        # Minimum OUTSIDE the window must not be reachable at calibration either.
        errs2 = fill(10.0, 4, length(g))
        below = argmin(abs.(g .- 0.6))
        inwin = argmin(abs.(g .- 1.7))
        errs2[:, below] .= 0.05; errs2[:, inwin] .= 0.2
        km2 = RSDRT.WindowedGridMethod("t", [errs2], [fill(1.0, 4)], g, window, loss_width_map, MedianError())
        @test RSDRT.tune(MersenneTwister(1), km2, pl, 4) == inwin
    end
end
