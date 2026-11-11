# =============================================================================
# The termination record: ReturnCode (SciMLBase's names, matched), StopRule,
# retcode/stop_rule/converged/found_model, nlls_termination, worst_retcode
# (base/return_code.jl, robust/lm_core.jl; the robust-estimation contract, §14)
# =============================================================================

@testitem "return_code" begin
    import VisualGeometryToolkitCore as VgtkCore
    import NLLSsolver
    using VisualGeometryToolkitCore: FitResult, ReturnCode, successful_retcode, StopRule, retcode, stop_rule,
        converged, found_model, return_code, retcode_symbol, worst_retcode, nlls_termination, test_termination_record
    using VisualGeometryToolkitCore: NoRule, ClosedForm, CostDecrease, IncrementSize, RejectionCap, ParameterChange,
        NoImprovement, ExactFit, ToleranceMet, LadderConverged, LeftBasin, ShapeGate, IterationCap, TimeCap, NonFinite, Callback,
        TrialBudget, AdaptiveCount
    using Test

    # A metadata as a producer will carry it: the code and the rule, its accessor declared beside it.
    struct _TermMeta
        return_code::ReturnCode.T
        stop_rule::StopRule
    end
    VgtkCore.return_code(m::_TermMeta) = m.return_code
    result(code, rule; estimate = 1.0) =
        FitResult(estimate, retcode_symbol(code, rule), 3, 0.1, 10, _TermMeta(code, rule))

    @testset "the names are SciMLBase.ReturnCode's (3.58), with its integer values" begin
        # A literal pin of SciMLBase's spelling and values for the codes the stack produces
        # (SciMLBase src/retcodes.jl, its EnumX order from Default = 0); no dependency.
        sciml = (Success = 1, Terminated = 2, MaxIters = 4, InitialFailure = 8, Failure = 10, MaxTime = 15,
                 Stalled = 18, InternalLinearSolveFailed = 20)
        @test Set(Symbol.(instances(ReturnCode.T))) == Set(keys(sciml))
        for (name, value) in pairs(sciml)
            @test Integer(getproperty(ReturnCode, name)) == value
        end
        @test isbitstype(ReturnCode.T) && isbitstype(StopRule)
        @test ReturnCode.T <: Enum{Int32}
    end

    @testset "successful_retcode: SciMLBase's rule" begin
        @test successful_retcode(ReturnCode.Success) && successful_retcode(ReturnCode.Terminated)
        for c in (ReturnCode.MaxIters, ReturnCode.MaxTime, ReturnCode.Stalled, ReturnCode.InitialFailure,
                  ReturnCode.Failure, ReturnCode.InternalLinearSolveFailed)
            @test !successful_retcode(c)
        end
    end

    @testset "StopRule: every rule, LODR's two included" begin
        @test length(instances(StopRule)) == 18
        @test LadderConverged isa StopRule && LeftBasin isa StopRule && ShapeGate isa StopRule
        @test retcode_symbol(ReturnCode.Failure, ShapeGate) === :solve_failed
    end

    @testset "accessors: inferred, strict converged, found_model" begin
        for (code, rule) in ((ReturnCode.Success, CostDecrease), (ReturnCode.Success, ClosedForm),
                             (ReturnCode.MaxIters, IterationCap), (ReturnCode.MaxIters, TrialBudget),
                             (ReturnCode.Stalled, IncrementSize), (ReturnCode.Failure, NonFinite))
            r = result(code, rule)
            @test @inferred(retcode(r)) === code
            @test @inferred(stop_rule(r)) === rule
            @test @inferred(converged(r)) === successful_retcode(code)
            @test r.converged === converged(r)
            test_termination_record(r)
        end
        # STRICT: a cap is never converged, a closed form is
        @test !converged(result(ReturnCode.MaxIters, IterationCap))
        @test !converged(result(ReturnCode.MaxIters, TrialBudget))      # a RANSAC at its trial budget
        @test converged(result(ReturnCode.Success, ClosedForm))
        # found_model: the old find_* meaning, whatever the code
        @test found_model(result(ReturnCode.MaxIters, TrialBudget))
        @test !found_model(result(ReturnCode.Failure, TrialBudget; estimate = nothing))
        # the transition: a producer without a recorded code answers through the core Symbol
        lean = FitResult(1.0, :converged, 1, 0.0, 3, nothing)
        @test @inferred(retcode(lean)) === :converged && converged(lean)
        @test return_code(nothing) === nothing
        @test !converged(FitResult(1.0, :max_iterations, 5, 0.0, 3, nothing))
        @test_throws ArgumentError stop_rule(lean)
    end

    @testset "retcode_symbol: the transition's one map" begin
        @test retcode_symbol(ReturnCode.Success, CostDecrease) === :converged
        @test retcode_symbol(ReturnCode.Success, ClosedForm) === :solved
        @test retcode_symbol(ReturnCode.Terminated, Callback) === :converged
        @test retcode_symbol(ReturnCode.MaxIters, IterationCap) === :max_iterations
        @test retcode_symbol(ReturnCode.MaxTime, TimeCap) === :max_time
        @test retcode_symbol(ReturnCode.Stalled, IncrementSize) === :stalled
        @test retcode_symbol(ReturnCode.InitialFailure, NoRule) === :degenerate
        @test retcode_symbol(ReturnCode.InitialFailure, NonFinite) === :solve_failed
        @test retcode_symbol(ReturnCode.Failure, NonFinite) === :solve_failed
        @test retcode_symbol(ReturnCode.Failure, TrialBudget) === :no_model
        @test retcode_symbol(ReturnCode.InternalLinearSolveFailed, NonFinite) === :solve_failed
    end

    @testset "worst_retcode: the most severe solve, the later on a tie" begin
        ok = (ReturnCode.Success, CostDecrease)
        @test @inferred(worst_retcode(ok)) === ok
        @test @inferred(worst_retcode(ok, (ReturnCode.MaxTime, TimeCap))) === (ReturnCode.MaxTime, TimeCap)
        @test worst_retcode((ReturnCode.MaxIters, IterationCap), ok) === (ReturnCode.MaxIters, IterationCap)
        @test worst_retcode(ok, (ReturnCode.Success, IncrementSize)) === (ReturnCode.Success, IncrementSize)
        @test @inferred(worst_retcode(ok, (ReturnCode.Stalled, IncrementSize), (ReturnCode.Failure, NonFinite), ok)) ===
            (ReturnCode.Failure, NonFinite)
        order = (ReturnCode.Success, ReturnCode.Terminated, ReturnCode.Stalled, ReturnCode.MaxIters, ReturnCode.MaxTime,
                 ReturnCode.Failure, ReturnCode.InternalLinearSolveFailed, ReturnCode.InitialFailure)
        for i in 1:(length(order) - 1)
            @test worst_retcode((order[i + 1], NoRule), (order[i], NoRule))[1] === order[i + 1]
        end
    end

    @testset "nlls_termination: NLLSsolver's bits, in nlls_stop_code's order" begin
        # a synthetic NLLSsolver result whose only meaningful field is its termination bits
        res(bits) = NLLSsolver.NLLSResult(map(fieldnames(NLLSsolver.NLLSResult), fieldtypes(NLLSsolver.NLLSResult)) do f, F
            f === :termination ? F(bits) : F <: Enum ? first(instances(F)) : zero(F)
        end...)
        t(bits, p = true) = @inferred nlls_termination(res(bits), p)
        for b in (0, 1, 4, 5, 10, 12)
            @test t(1 << b) === (ReturnCode.Failure, NonFinite)
        end
        @test t(1 << 11) === (ReturnCode.InitialFailure, NonFinite)
        @test t((1 << 1) | (1 << 11)) === (ReturnCode.InitialFailure, NonFinite)
        @test t(1 << 2) === (ReturnCode.Success, CostDecrease) && t(1 << 3) === (ReturnCode.Success, CostDecrease)
        @test t(1 << 2, false) === (ReturnCode.Success, CostDecrease)               # no postcondition on a cost stop
        @test t(1 << 6) === (ReturnCode.Success, IncrementSize)
        @test t(1 << 6, false) === (ReturnCode.Stalled, IncrementSize)
        @test t((1 << 2) | (1 << 6), false) === (ReturnCode.Stalled, IncrementSize)      # the step stop first
        @test t((1 << 1) | (1 << 6), false) === (ReturnCode.Stalled, IncrementSize)      # a NaN trial at a step stop
        @test t(1 << 7) === (ReturnCode.Success, RejectionCap)
        @test t(1 << 7, false) === (ReturnCode.Stalled, RejectionCap)
        @test t(1 << 16, false) === (ReturnCode.Terminated, Callback)
        @test t(1 << 9) === (ReturnCode.MaxTime, TimeCap)
        @test t((1 << 8) | (1 << 9)) === (ReturnCode.MaxTime, TimeCap)              # time outranks iterations
        @test t(1 << 8) === (ReturnCode.MaxIters, IterationCap)
        @test t((1 << 8) | (1 << 2)) === (ReturnCode.Success, CostDecrease)         # a tolerance outranks the cap
        @test t((1 << 1) | (1 << 8) | (1 << 9)) === (ReturnCode.Failure, NonFinite)  # failure outranks the caps
        # the postcondition runs only where it decides
        calls = Ref(0)
        probe = () -> (calls[] += 1; false)
        @test nlls_termination(1 << 2, probe) === (ReturnCode.Success, CostDecrease) && calls[] == 0
        @test nlls_termination(1 << 6, probe) === (ReturnCode.Stalled, IncrementSize) && calls[] == 1
        # its retcode_symbol is nlls_retcode's: the transition keeps every Symbol
        J = [1.0 0.0; 0.0 2.0; 1.0 1.0]; y = [1.0, 2.0, 4.0]
        for r in (J * (J \ y) - y, J * ((J \ y) .+ 0.1) - y), bits in (1 << 2, 1 << 6, 1 << 7, 1 << 8, 1 << 9, 1 << 1, (1 << 2) | (1 << 6))
            kw = (; tol = 1.0e-6, ftol = eps())
            code, rule = nlls_termination(res(bits), () -> VgtkCore.is_stationary(J, r; kw...))
            @test retcode_symbol(code, rule) === VgtkCore.nlls_retcode(res(bits), J, r; kw...)
        end
    end

    @testset "public" begin
        for n in (:ReturnCode, :successful_retcode, :StopRule, :return_code, :stop_rule, :found_model,
                  :retcode_symbol, :worst_retcode, :nlls_termination, :test_termination_record, :LadderConverged)
            @test Base.ispublic(VgtkCore, n)
        end
        @test VgtkCore.return_code in VgtkCore.CONTRACT_GENERICS && VgtkCore.stop_rule in VgtkCore.CONTRACT_GENERICS
    end
end
