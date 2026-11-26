# =============================================================================
# The termination record: ReturnCode (SciMLBase's names, matched), StopRule,
# the result's record fields, successful_retcode/found_model, worst_retcode (base/return_code.jl;
# the robust-estimation contract, §14). NLLSsolver's decode into it: lm_driver_tests.jl
# =============================================================================

@testitem "return_code" begin
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: FitResult, SolveStats, ReturnCode, successful_retcode, StopRule,
        found_model, worst_retcode, test_termination_record
    using VisualGeometryToolkitCore: NoRule, ClosedForm, CostDecrease, IncrementSize, RejectionCap, ParameterChange,
        NoImprovement, ExactFit, ToleranceMet, LadderConverged, LeftBasin, ShapeGate, IterationCap, TimeCap, NonFinite, Callback,
        TrialBudget, AdaptiveCount
    using Test

    # Every result records the code and the rule in its core.
    result(code, rule; estimate = 1.0) = FitResult(estimate, code, rule, SolveStats(; nsteps = 3), 0.1, 10, nothing)

    @testset "the names are SciMLBase.ReturnCode's (3.58), with its integer values" begin
        # A literal pin of SciMLBase's spelling and values for the codes the stack produces
        # (SciMLBase src/retcodes.jl, its EnumX order from Default = 0); no dependency.
        sciml = (Default = 0, Success = 1, Terminated = 2, MaxIters = 4, InitialFailure = 8, Failure = 10, MaxTime = 15,
                 Stalled = 18, InternalLinearSolveFailed = 20)
        @test Set(Symbol.(instances(ReturnCode.T))) == Set(keys(sciml))
        for (name, value) in pairs(sciml)
            @test Integer(getproperty(ReturnCode, name)) == value
        end
        @test isbitstype(ReturnCode.T) && isbitstype(StopRule)
        @test ReturnCode.T <: Enum{Int32}
        # every value public in its module (a package's qualified ReturnCode.X is a public access) and documented
        for n in (:T, keys(sciml)...)
            @test Base.ispublic(ReturnCode, n)
        end
    end

    @testset "successful_retcode: SciMLBase's rule" begin
        @test successful_retcode(ReturnCode.Success) && successful_retcode(ReturnCode.Terminated)
        for c in (ReturnCode.Default, ReturnCode.MaxIters, ReturnCode.MaxTime, ReturnCode.Stalled, ReturnCode.InitialFailure,
                  ReturnCode.Failure, ReturnCode.InternalLinearSolveFailed)
            @test !successful_retcode(c)
        end
    end

    @testset "StopRule: every rule, LODR's two included" begin
        @test length(instances(StopRule)) == 19
        @test LadderConverged isa StopRule && LeftBasin isa StopRule && ShapeGate isa StopRule
    end

    @testset "the record by field, strict success, found_model" begin
        for (code, rule) in ((ReturnCode.Success, CostDecrease), (ReturnCode.Success, ClosedForm),
                             (ReturnCode.MaxIters, IterationCap), (ReturnCode.MaxIters, TrialBudget),
                             (ReturnCode.Stalled, IncrementSize), (ReturnCode.Failure, NonFinite))
            r = result(code, rule)
            @test r.retcode === code
            @test r.stop_rule === rule
            @test @inferred(successful_retcode(r)) === successful_retcode(code)
            test_termination_record(r)
        end
        # STRICT: a cap is never a success, a closed form is
        @test !successful_retcode(result(ReturnCode.MaxIters, IterationCap))
        @test !successful_retcode(result(ReturnCode.MaxIters, TrialBudget))      # a RANSAC at its trial budget
        @test successful_retcode(result(ReturnCode.Success, ClosedForm))
        # found_model: the old find_* meaning, whatever the code
        @test found_model(result(ReturnCode.MaxIters, TrialBudget))
        @test !found_model(result(ReturnCode.Failure, TrialBudget; estimate = nothing))
        # the record is the core's: every result has one, the lean `nothing` metadata included
        @test fieldtype(FitResult{Float64, Nothing}, :retcode) === ReturnCode.T
        @test fieldtype(FitResult{Float64, Nothing}, :stop_rule) === StopRule
        @test_throws MethodError FitResult(1.0, :converged, NoRule, SolveStats(), 0.0, 3, nothing)     # no Symbol record
    end

    @testset "a result's successful_retcode is its code's, for every code and rule" begin
        for code in instances(ReturnCode.T), rule in instances(StopRule)
            r = result(code, rule)
            @test successful_retcode(r) === successful_retcode(code)
        end
        # inferred, as a Bool, for every code
        @test @inferred(successful_retcode(ReturnCode.Success)) === true
        @test @inferred(successful_retcode(ReturnCode.MaxIters)) === false
        @test Base.return_types(successful_retcode, (ReturnCode.T,)) == [Bool]
    end



    @testset "worst_retcode: the most severe solve, the later on a tie" begin
        ok = (ReturnCode.Success, CostDecrease)
        @test @inferred(worst_retcode(ok)) === ok
        @test @inferred(worst_retcode(ok, (ReturnCode.MaxTime, TimeCap))) === (ReturnCode.MaxTime, TimeCap)
        @test worst_retcode((ReturnCode.MaxIters, IterationCap), ok) === (ReturnCode.MaxIters, IterationCap)
        @test worst_retcode(ok, (ReturnCode.Success, IncrementSize)) === (ReturnCode.Success, IncrementSize)
        @test @inferred(worst_retcode(ok, (ReturnCode.Stalled, IncrementSize), (ReturnCode.Failure, NonFinite), ok)) ===
            (ReturnCode.Failure, NonFinite)
        order = (ReturnCode.Default, ReturnCode.Success, ReturnCode.Terminated, ReturnCode.Stalled, ReturnCode.MaxIters, ReturnCode.MaxTime,
                 ReturnCode.Failure, ReturnCode.InternalLinearSolveFailed, ReturnCode.InitialFailure)
        for i in 1:(length(order) - 1)
            @test worst_retcode((order[i + 1], NoRule), (order[i], NoRule))[1] === order[i + 1]
        end
    end

    @testset "public" begin
        for n in (:ReturnCode, :successful_retcode, :StopRule, :found_model,
                  :worst_retcode, :test_termination_record, :LadderConverged, :SeedKept)
            @test Base.ispublic(VgtkCore, n)
        end
        @test !isdefined(VgtkCore, :return_code)          # the record has no metadata copy
        @test !isdefined(VgtkCore, :stop_rule)            # a criterion's rule is its stop_reason's (C8); the rule a FIELD
    end
end
