# A refine from a refine's keyword set, the suite's one spelling of it: the strategy is the LevenbergMarquardt's field,
# every other keyword the solve's (the scale input, the stops — or a keyword under test, which the solve refuses).
@testsetup module RefineSolve
    import VisualGeometryToolkitCore as VgtkCore
    export lm
    lm(f, θ; optimization_strategy = VgtkCore.ClassDefault(), kw...) =
        VgtkCore.solve(VgtkCore.RefineProblem(f, θ), VgtkCore.LevenbergMarquardt(; optimization_strategy); kw...)
end
