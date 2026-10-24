# =============================================================================
# tuning_objective_tests.jl — the tuning objectives. mAA is VisualGeometryToolkitCore's
# `Synthetic.mean_average_accuracy`, the stack's one definition — RANSACScoringDoneRight's
# own `_maa` (the area under the sorted accuracy curve) is deleted.
#
# Moved here from RANSACScoringDoneRight/test/ with the criteria themselves
# (2026-10-03). The move left the file behind in a package whose `src` no longer
# defines `MAA`, `MedianError` or `_tune_objective`, so RANSACScoringDoneRight's
# own suite ran it and failed on `UndefVarError` — ReTestItems discovers every
# `*_tests.jl` under test/, so an orphan runs rather than being ignored.
#
# The criteria that were already here (FiniteMedian, SceneMeanMedian, SceneMeanLowerMedian,
# MeanAverageAccuracy) had NO objective coverage anywhere before this file;
# their `empty`/failure edges are the part that decides an argmin, so they are
# asserted here beside the two that moved.
# =============================================================================

@testitem "tuning_objective" begin
    import VisualGeometryToolkitEval as VgtkEval
    using Test, Statistics
    using VisualGeometryToolkitCore.Synthetic: mean_average_accuracy
    using VisualGeometryToolkitEval.Tuning: MedianError, FiniteMedian, SceneMeanMedian,
        SceneMeanLowerMedian, MeanAverageAccuracy, _tune_objective, tuning_criterion_label

    @testset "tuning objectives: mAA is VGC's" begin
        e = [0.5, 2.0, 7.5, 12.0, 3.25]
        maa = MeanAverageAccuracy
        @test _tune_objective(maa(10.0), e) === -mean_average_accuracy(e; T = 10.0)
        @test _tune_objective(maa(20.0), e) === -mean_average_accuracy(e; T = 20.0)
        # the definition: the mean of max(0, 1 − e/T)
        @test -_tune_objective(maa(10.0), e) ≈ sum(max.(0, 1 .- e ./ 10)) / length(e) rtol = 1e-15
        # the edges the deleted `_maa` got differently: a non-finite error is a
        # FAILURE (it used to poison the sum as NaN), and no errors is NaN (was 0.0)
        @test -_tune_objective(maa(10.0), [0.5, NaN, 12.0]) ≈ 0.95 / 3 rtol = 1e-15
        @test -_tune_objective(maa(10.0), [0.5, Inf, 12.0]) ≈ 0.95 / 3 rtol = 1e-15
        @test isnan(_tune_objective(maa(10.0), Float64[]))
        # the median objective counts a failure as ∞, as before
        @test _tune_objective(MedianError(), [1.0, NaN, 3.0]) == 3.0
    end

    # The two median criteria differ ONLY in what a non-finite error means, and
    # that difference decides argmins: MedianError counts it as an arbitrarily
    # bad outcome, FiniteMedian drops it. A column where half the refits failed
    # is therefore scored very differently by the two.
    @testset "MedianError penalises a failure; FiniteMedian drops it" begin
        e = [1.0, 2.0, NaN, NaN]
        @test _tune_objective(MedianError(), e) == Inf       # median(1, 2, Inf, Inf)
        @test _tune_objective(FiniteMedian(), e) == 1.5      # median(1, 2)
        @test _tune_objective(MedianError(), [1.0, 2.0, 3.0]) == 2.0
        @test _tune_objective(FiniteMedian(), [1.0, 2.0, 3.0]) == 2.0
        # an ALL-failed column must LOSE an argmin, never win it: `empty = Inf`
        @test _tune_objective(FiniteMedian(), [NaN, NaN]) == Inf
        @test _tune_objective(FiniteMedian(), Float64[]) == Inf
    end

    # Scene-level criteria take ONE error vector PER SCENE and weight scenes
    # equally, regardless of pair count. A scene with no finite error is left
    # out rather than poisoning the mean.
    @testset "scene-mean criteria weight scenes, not pairs" begin
        scenes = [[1.0, 1.0, 1.0, 1.0, 1.0], [5.0]]          # 5 pairs vs 1
        @test _tune_objective(SceneMeanMedian(), scenes) == 3.0      # (1 + 5)/2, not 1.67
        @test _tune_objective(SceneMeanLowerMedian(), scenes) == 3.0
        # a scene with no finite value is skipped, not counted as a failure
        @test _tune_objective(SceneMeanMedian(), [[2.0], [NaN], [4.0]]) == 3.0
        @test isnan(_tune_objective(SceneMeanMedian(), [[NaN], [Inf]]))
        @test isnan(_tune_objective(SceneMeanLowerMedian(), [[NaN]]))
    end

    # lower_median is the ⌈n/2⌉-th order statistic: on an even n it returns the
    # LOWER of the two middle values rather than their average, so the objective
    # is always a value that actually occurred. This is the whole difference
    # from SceneMeanMedian, and it only shows on even counts.
    @testset "lower_median returns an observed value on even n" begin
        # n = 4: median averages the two middle values, lower_median takes the
        # lower of them — ⌈4/2⌉ = 2, the 2nd order statistic.
        @test _tune_objective(SceneMeanMedian(), [[1.0, 2.0, 3.0, 4.0]]) == 2.5      # averaged
        @test _tune_objective(SceneMeanLowerMedian(), [[1.0, 2.0, 3.0, 4.0]]) == 2.0 # observed
        # n = 2: ⌈2/2⌉ = 1, so the lower of the pair.
        @test _tune_objective(SceneMeanLowerMedian(), [[1.0, 2.0]]) == 1.0
        @test _tune_objective(SceneMeanMedian(), [[1.0, 2.0, 3.0]]) ==
              _tune_objective(SceneMeanLowerMedian(), [[1.0, 2.0, 3.0]])     # odd n agrees
    end

    # ONE mAA criterion. The retired `MAA(max_error_deg)` was the identical
    # statistic under a field name asserting degrees, which a criterion handed
    # millimetres cannot (BlobBoards' bmvc2026 invariance, configs/tuning_grid.toml).
    # Its `"maa"` tag went with it — an alias would make `_tag`'s reverse lookup
    # ambiguous, since it finds a type's tag by scanning the table.
    @testset "mAA is units-neutral and refuses a non-positive cap" begin
        @test_throws ArgumentError MeanAverageAccuracy(0.0)
        @test_throws ArgumentError MeanAverageAccuracy(-1.0)
        @test MeanAverageAccuracy(10.0).max_error == 10.0
        @test !isdefined(VgtkEval.Tuning, :MAA)
    end

    # Both mAA spellings label as "mAA", and both medians as "median": the label
    # is the DISPLAY name, not the type, so a figure legend does not change when
    # a criterion is swapped for its equivalent.
    @testset "criterion labels" begin
        @test tuning_criterion_label(MeanAverageAccuracy(10.0)) == "mAA"
        @test tuning_criterion_label(MedianError()) == "median"
        @test tuning_criterion_label(FiniteMedian()) == "median"
        @test tuning_criterion_label(SceneMeanMedian()) == "scene_mean_median"
        @test tuning_criterion_label(SceneMeanLowerMedian()) == "scene_mean_lower_median"
    end
end
