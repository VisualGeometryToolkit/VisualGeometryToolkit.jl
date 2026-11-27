# The local optimization's front end (Core 0.14, the 1A follow-ups G1–G3): the summary-free chart refine — the
# problem's refit chart and rows, one chart LM at the caller's weights, the model only — with a deployed loss given
# in as the width, and the block-coordinate loop from a seeded state. Each is the internal kernel's arithmetic, bit for
# bit, and a warm call through its cache allocates nothing and infers.
@testitem "LO front end: the summary-free refine, a deployed width, a seeded block-coordinate run" begin
    using Test, LinearAlgebra, StaticArrays
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: RefineProblem, LevenbergMarquardt, Triggs, WelschShape, init, solve, solve!,
        reinit!, EuclideanManifold, deploy, sq_loss_width, ReturnCode, NoRule, StopAfterIteration, thrown_message

    # a circle (cx, cy, ρ) fitted to points: the chart is Euclidean at the model, the rows √w·(‖p − c‖ − ρ)
    struct _Circ <: VgtkCore.AbstractConstraintProblem
        pts::Vector{SVector{2, Float64}}
    end
    struct _CircRows
        pts::Vector{SVector{2, Float64}}
        at::SVector{3, Float64}
        sqrtw::Vector{Float64}
    end
    VgtkCore.model_type(::_Circ) = SVector{3, Float64}
    VgtkCore.data_size(f::_Circ) = length(f.pts)
    VgtkCore.codimension(::_Circ) = 1
    VgtkCore.minimal_sample_size(::_Circ) = 3
    VgtkCore.mahalanobis_distances!(r::AbstractVector, f::_Circ, θ) = (for i in eachindex(f.pts); r[i] = abs(norm(f.pts[i] - θ[SOneTo(2)]) - θ[3]); end; r)
    VgtkCore.model_chart(::_Circ, θ) = EuclideanManifold(SVector{3, Float64}(θ))
    VgtkCore.chart_rows(f::_Circ, ch, sqrtw) = _CircRows(f.pts, VgtkCore.model(ch), sqrtw)
    VgtkCore.chart_model(::_Circ, ch) = VgtkCore.model(ch)
    VgtkCore.chart_size(f::_Circ) = (length(f.pts), Val(3))
    VgtkCore.default_optimization_strategy(::_Circ) = nothing
    function VgtkCore.residual_rows!(r, s::_CircRows, x)
        θ = s.at + x
        for i in eachindex(s.pts)
            r[i] = s.sqrtw[i] * (norm(s.pts[i] - θ[SOneTo(2)]) - θ[3])
        end
        return r
    end
    function VgtkCore.residual_jacobian_rows!(J, s::_CircRows, x)
        θ = s.at + x
        for i in eachindex(s.pts)
            d = s.pts[i] - θ[SOneTo(2)]; u = d / norm(d)
            J[i, 1] = -s.sqrtw[i] * u[1]; J[i, 2] = -s.sqrtw[i] * u[2]; J[i, 3] = -s.sqrtw[i]
        end
        return J
    end
    pts = [SVector(1.5 + 2.5cos(t), -1.8 + 2.5sin(t)) + 0.01 * SVector(sin(3t), cos(5t)) for t in range(0, 2π; length = 25)[1:24]]
    pts[3] = SVector(9.0, 9.0)                                     # one gross outlier
    f = _Circ(pts); θ₀ = SVector(1.0, -2.0, 3.0); w = [i == 3 ? 0.0 : 1.0 for i in eachindex(pts)]

    # G1: the summary-free refine IS the kernel's chart refit (the same chart, rows and weights), the model only
    c = init(RefineProblem(f, zero(SVector{3, Float64})), LevenbergMarquardt(); weights = w, summary = nothing)
    r = solve!(reinit!(c, θ₀; weights = w))
    ref = VgtkCore._chart_refit(VgtkCore._ManifoldLmWorkspace(length(pts), Val(3)), f, VgtkCore.model_chart(f, θ₀),
                                VgtkCore.chart_rows, w)
    @test r.u === ref.u && (r.retcode, r.stop_rule) === (ref.retcode, ref.stop_rule) && r.stats.nsteps == ref.stats.nsteps
    @test isnan(r.rms) && r.n == 0 && r.metadata === nothing && VgtkCore.successful_retcode(r)
    @test r.u ≈ SVector(1.5, -1.8, 2.5) atol = 0.02
    lo_call(c, θ, w) = solve!(reinit!(c, θ; weights = w))
    lo_call(c, θ₀, w); lo_call(c, θ₀, w)
    @test (@allocations lo_call(c, θ₀, w)) == 0
    @inferred lo_call(c, θ₀, w)
    # the default summary is the refine of record (its metadata, its residuals)
    full = solve(RefineProblem(f, θ₀), LevenbergMarquardt())
    @test full.metadata isa VgtkCore.RefineMetadata && full.n == length(pts)
    # a declining refit chart: no model, (InitialFailure, NoRule)
    struct _Declines <: VgtkCore.AbstractConstraintProblem
        inner::_Circ
    end
    for g in (:model_type, :data_size, :codimension, :minimal_sample_size, :chart_size, :default_optimization_strategy)
        @eval VgtkCore.$g(d::_Declines) = VgtkCore.$g(d.inner)
    end
    VgtkCore.refit_chart(::_Declines, _θ, _w) = nothing
    rd = solve!(init(RefineProblem(_Declines(f), θ₀), LevenbergMarquardt(); weights = w, summary = nothing))
    @test rd.u === nothing && (rd.retcode, rd.stop_rule) === (ReturnCode.InitialFailure, NoRule)

    # A4: the summary-free refit reads the problem's own model (`convert`), never a user's guess (`guess_model`)
    struct _NoGuess <: VgtkCore.AbstractConstraintProblem
        inner::_Circ
    end
    for g in (:model_type, :data_size, :codimension, :minimal_sample_size, :chart_size, :default_optimization_strategy)
        @eval VgtkCore.$g(d::_NoGuess) = VgtkCore.$g(d.inner)
    end
    VgtkCore.model_chart(d::_NoGuess, θ) = VgtkCore.model_chart(d.inner, θ)
    VgtkCore.chart_rows(d::_NoGuess, ch, sqrtw) = VgtkCore.chart_rows(d.inner, ch, sqrtw)
    VgtkCore.chart_model(d::_NoGuess, ch) = VgtkCore.chart_model(d.inner, ch)
    VgtkCore.guess_model(::_NoGuess, _u0) = error("a user's guess is not read by the LO refit")
    rn = solve(RefineProblem(_NoGuess(f), θ₀), LevenbergMarquardt(); weights = w, summary = nothing)
    @test rn.u === r.u

    # G2: a deployed loss as the width — used as deployed (its squared width bit for bit), the kernel's Triggs refit
    loss = deploy(WelschShape(), 0.03^2; efficiency_constant = 2.9846)
    alg = LevenbergMarquardt(; optimization_strategy = Triggs{WelschShape}())
    ct = init(RefineProblem(f, θ₀), alg; width = loss, weights = w, summary = nothing)
    @test sq_loss_width(ct.cache.spec.loss) === sq_loss_width(loss)
    rt = solve!(ct)
    reft = VgtkCore._triggs_refit(VgtkCore._ManifoldLmWorkspace(length(pts), Val(3)), WelschShape(), f, θ₀, loss; weights = w)
    @test rt.u === reft.u && (rt.retcode, rt.stop_rule) === (reft.retcode, reft.stop_rule)
    lo_call(ct, θ₀, w); ws_t = VgtkCore._ManifoldLmWorkspace(length(pts), Val(3))
    kernel_t() = VgtkCore._triggs_refit(ws_t, WelschShape(), f, θ₀, loss; weights = w); kernel_t(); kernel_t()
    @test (@allocations lo_call(ct, θ₀, w)) <= (@allocations kernel_t())   # no more than the kernel it replaces
    @test_throws ArgumentError init(RefineProblem(f, θ₀), LevenbergMarquardt(; optimization_strategy = Triggs{VgtkCore.CauchyShape}());
                                    width = loss, summary = nothing)   # another shape's loss

    # ℓ2-Triggs at its deployed (unbounded) width: the robustified LM the local optimization always ran, bit for bit
    lossg = deploy(VgtkCore.GaussianShape(), Inf; efficiency_constant = 1.0)
    cg = init(RefineProblem(f, θ₀), LevenbergMarquardt(; optimization_strategy = Triggs{VgtkCore.GaussianShape}());
              width = lossg, weights = w, summary = nothing)
    rg = solve!(cg)
    refg = VgtkCore._triggs_refit(VgtkCore._ManifoldLmWorkspace(length(pts), Val(3)), VgtkCore.GaussianShape(), f, θ₀, lossg;
                                  weights = w)
    @test rg.u === refg.u && (rg.retcode, rg.stop_rule) === (refg.retcode, refg.stop_rule)
    @test_throws ArgumentError solve(RefineProblem(f, θ₀), LevenbergMarquardt(; optimization_strategy = Triggs{VgtkCore.GaussianShape}());
                                     scale = 0.1)                      # the shape-only spelling stays refused
    # the one-shot solve takes the cache settings init takes (SciML: solve ≡ solve!(init(…)))
    r1 = solve(RefineProblem(f, θ₀), LevenbergMarquardt(); weights = w, summary = nothing)
    @test r1.u === r.u && (r1.retcode, r1.stop_rule) === (r.retcode, r.stop_rule) && r1.metadata === nothing

    # Ruling 1: the block-coordinate state is the run's workspace, never the problem's context. A method that holds its
    # buffers seeds them in its `init_state!` (the one-shot solve builds no cache); a cache owns the state and
    # `reinit!` re-seeds it in place through `init_state!(state, method, prob, θ₀)`: zero allocation, `solve!` inferred
    struct _Mean{S} <: VgtkCore.AbstractBlockCoordinateMethod
        st::S          # the method's own buffers: the state its runs start from
        start::Int     # the per-run seed, the method's field
    end
    VgtkCore.init_state!(m::_Mean, data, θ₀) = (m.st.k[] = m.start; m.st)
    VgtkCore.init_state!(st, m::_Mean, data, θ₀) = (st.k[] = m.start; st)
    VgtkCore.update_model!(st, ::_Mean, data, θ) = (st.k[] += 1; sum(data) / length(data))
    VgtkCore.update_residuals!(st, ::_Mean, data, θ) = nothing
    VgtkCore.block_coordinate_result(st, ::_Mean, data, θ, code, iter, rule) =
        VgtkCore.FitResult(θ, code, rule, VgtkCore.SolveStats(; nsteps = iter), NaN, 0, nothing)
    data = [1.0, 2.0, 6.0]; st = (; k = Ref(0))
    bcd_alg(m) = VgtkCore.BlockCoordinateDescent(m; likelihood = VgtkCore.FrozenNuisance())
    rb = solve(RefineProblem(data, 0.0), bcd_alg(_Mean(st, 0)); stops = StopAfterIteration(4))
    @test rb.u == 3.0 && st.k[] == 4 && rb.stats.nsteps == 4
    bcd(st, data) = solve(RefineProblem(data, 0.0), bcd_alg(_Mean(st, 0)); stops = StopAfterIteration(4))
    bcd(st, data); @test (@allocations bcd(st, data)) == 0
    # the seeded-state context of the earlier interface is refused, one-shot and cache alike
    @test_throws ArgumentError solve(RefineProblem(data, 0.0, st), bcd_alg(_Mean(st, 0)); stops = StopAfterIteration(4))
    @test occursin("reads no context", thrown_message(() -> init(RefineProblem(data, 0.0, st), bcd_alg(_Mean(st, 0));
                                                                  stops = StopAfterIteration(2))))
    cb = init(RefineProblem(data, 0.0), bcd_alg(_Mean((; k = Ref(0)), 0)); stops = StopAfterIteration(2))
    @test solve!(cb).u == 3.0
    @test_throws ArgumentError reinit!(cb, 0.0; p = st)
    r2 = solve!(reinit!(cb, 1.0; alg = bcd_alg(_Mean((; k = Ref(0)), 10))))   # a new seed through the method: the kept state re-seeded
    @test r2.u == 3.0 && r2.stats.nsteps == 2
    reuse(cb, a) = solve!(reinit!(cb, 0.5; alg = a))
    a10 = bcd_alg(_Mean((; k = Ref(0)), 10))
    reuse(cb, a10); @test (@allocations reuse(cb, a10)) == 0
    @test (@inferred solve!(cb)).u == 3.0
end

# The block refine's too-few guard (G5) and its summary's scale record and free-coordinate ν (G6).
@testitem "block refine: the too-few guard, the scale record, ν over the free coordinates" begin
    using Test, StaticArrays
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: RefineProblem, LevenbergMarquardt, solve, init, solve!, reinit!, NLLSsolver, ReturnCode,
        NoRule, ScaleNone
    # a line y = a + b·s, its two coefficients two variables (a SVector{1} each); one row per datum
    struct _Row <: NLLSsolver.AbstractResidual
        s::Float64; y::Float64
    end
    NLLSsolver.ndeps(::_Row) = NLLSsolver.static(2)
    NLLSsolver.nres(::_Row) = NLLSsolver.static(1)
    NLLSsolver.varindices(::_Row) = SVector(1, 2)
    NLLSsolver.getvars(::_Row, vars::Vector) = (vars[1]::SVector{1, Float64}, vars[2]::SVector{1, Float64})
    Base.eltype(::_Row) = Float64
    NLLSsolver.computeresidual(r::_Row, a, b) = SVector(a[1] + b[1] * r.s - r.y)
    struct _Line <: VgtkCore.AbstractBlockProblem
        s::Vector{Float64}; y::Vector{Float64}; hold::Bool
    end
    VgtkCore.default_optimization_strategy(::_Line) = nothing
    VgtkCore.data_size(f::_Line) = length(f.s)
    VgtkCore.minimal_sample_size(::_Line) = 2
    VgtkCore.block_variables(::_Line, u0) = Union{SVector{1, Float64}}[SVector(u0[1]), SVector(u0[2])]
    VgtkCore.residual_blocks(f::_Line, _u0) = [_Row(si, yi) for (si, yi) in zip(f.s, f.y)]
    VgtkCore.fixed_variables(f::_Line) = f.hold ? 2 : nothing
    VgtkCore.block_result(::_Line, _u0, r, summary) =
        VgtkCore.FitResult(Tuple(v[1] for v in r.u), r.retcode, r.stop_rule, r.stats, summary.rms, summary.n, summary)
    s = collect(0.0:1.0:5.0); y = 0.5 .+ 2.0 .* s .+ 0.01 .* sin.(3 .* s)
    r = solve(RefineProblem(_Line(s, y, false), (0.0, 0.0)), LevenbergMarquardt())
    @test r.metadata.ν == length(s) - 2 && r.metadata.loss.strategy === nothing &&
          VgtkCore.scale_record(r.metadata.loss).scale_source === ScaleNone
    rh = solve(RefineProblem(_Line(s, y, true), (0.0, 2.0)), LevenbergMarquardt())
    @test rh.metadata.ν == length(s) - 1 && rh.u[2] == 2.0      # the held coefficient is no unknown
    few = solve(RefineProblem(_Line(s[1:1], y[1:1], false), (0.0, 0.0)), LevenbergMarquardt())
    @test few.u === nothing && (few.retcode, few.stop_rule) === (ReturnCode.InitialFailure, NoRule) && few.n == 1

    # PS N2: a held set that depends on the guess, `fixed_variables(f, u0)` (the guess's third entry holds the slope):
    # every variable stated, the held one declared; the one-shot kept problem and a cache's `reinit!` rebuild on a
    # guess whose held set differs, and keep the LM on one whose held set is the cache's
    struct _GaugeLine <: VgtkCore.AbstractBlockProblem
        s::Vector{Float64}; y::Vector{Float64}
    end
    VgtkCore.default_optimization_strategy(::_GaugeLine) = nothing
    VgtkCore.block_variables(::_GaugeLine, u0) = Union{SVector{1, Float64}}[SVector(u0[1]), SVector(u0[2])]
    VgtkCore.residual_blocks(f::_GaugeLine, _u0) = [_Row(si, yi) for (si, yi) in zip(f.s, f.y)]
    VgtkCore.fixed_variables(::_GaugeLine, u0) = u0[3] ? 2 : nothing
    VgtkCore.block_result(::_GaugeLine, _u0, r, summary) =
        VgtkCore.FitResult(Tuple(v[1] for v in r.u), r.retcode, r.stop_rule, r.stats, summary.rms, summary.n, summary)
    g = _GaugeLine(s, y)
    @test VgtkCore.fixed_variables(_Line(s, y, true), (0.0, 2.0)) == 2   # the guess form defaults to the bundle's own
    held = solve(RefineProblem(g, (0.0, 2.0, true)), LevenbergMarquardt())
    @test held.metadata.ν == length(s) - 1 && held.u[2] == 2.0 && all(held.u .≈ rh.u)
    free = solve(RefineProblem(g, (0.0, 2.0, false)), LevenbergMarquardt())   # the kept problem rebuilt: no hold
    @test free.metadata.ν == length(s) - 2 && all(isapprox.(free.u, r.u; atol = 1.0e-9))
    c = init(RefineProblem(g, (0.0, 2.0, true)), LevenbergMarquardt())
    @test solve!(c).u == held.u
    @test solve!(reinit!(c, (0.1, 2.0, true))).metadata.ν == length(s) - 1    # the same held set: the LM kept
    # another held set makes an LM of another type (here: none held): refused, init a new cache
    @test_throws ArgumentError reinit!(c, (0.0, 2.0, false))
    cf = init(RefineProblem(g, (0.0, 2.0, false)), LevenbergMarquardt())
    @test solve!(cf).u == free.u
end

# A row refine's result carries the stationarity postcondition on demand (G8, the chart-problem refine's form), and a
# chart problem whose dof depends on its guess sizes its LM from `chart_size(f, u0)` at init (G9).
@testitem "row and chart refines: `stationary()` on demand; the chart size at the guess" setup = [CircleLmScene] begin
    using Test, LinearAlgebra, StaticArrays
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: RefineProblem, LevenbergMarquardt, solve, EuclideanManifold
    # a circle (cx, cy, ρ) through points, as rows on the chart x about (1, −2, 3): the shared scene's rows
    # (CircleLmScene) as a row problem, with its analytic Jacobian
    struct _CircleRows <: VgtkCore.AbstractRowProblem{3}
        lm::CircleLm
    end
    VgtkCore.chart_size(f::_CircleRows) = (length(f.lm.pts), Val(3))
    VgtkCore.residual_rows!(r, f::_CircleRows, x) = VgtkCore.residual_rows!(r, f.lm, x)
    function VgtkCore.residual_jacobian_rows!(J, f::_CircleRows, x)
        c = SVector(1.0 + x[1], -2.0 + x[2])
        for (i, p) in enumerate(f.lm.pts)
            u = (p - c) / norm(p - c)
            J[i, 1] = -u[1]; J[i, 2] = -u[2]; J[i, 3] = -1.0
        end
        return J
    end
    pts = circle_points()
    r = solve(RefineProblem(_CircleRows(CircleLm(pts)), zero(SVector{3, Float64})), LevenbergMarquardt())
    @test VgtkCore.successful_retcode(r) && r.metadata.stationary() === true && r.metadata.cost < 1.0e-20
    @test r.metadata.stationary() === VgtkCore.is_stationary(VgtkCore._ManifoldLmWorkspace(length(pts), Val(3)),
                                                             _CircleRows(CircleLm(pts)), r.u, VgtkCore.LmStops())
    # the same circle as a chart problem whose chart is the guess's own size: only `chart_size(f, u0)` is stated
    struct _GuessSized <: VgtkCore.AbstractChartProblem
        pts::Vector{SVector{2, Float64}}
    end
    struct _GuessRows
        f::_CircleRows
        at::SVector{3, Float64}
    end
    VgtkCore.model_type(::_GuessSized) = SVector{3, Float64}
    VgtkCore.chart_size(f::_GuessSized, u0::SVector{N}) where {N} = (length(f.pts), Val(N))
    VgtkCore.model_chart(::_GuessSized, θ) = EuclideanManifold(SVector{3, Float64}(θ))
    VgtkCore.chart_rows(f::_GuessSized, ch, _sqrtw) = _GuessRows(_CircleRows(CircleLm(f.pts)), VgtkCore.model(ch) - SVector(1.0, -2.0, 3.0))
    VgtkCore.chart_model(::_GuessSized, ch) = VgtkCore.model(ch)
    VgtkCore.residual_rows!(r, s::_GuessRows, x) = VgtkCore.residual_rows!(r, s.f, s.at + x)
    VgtkCore.residual_jacobian_rows!(J, s::_GuessRows, x) = VgtkCore.residual_jacobian_rows!(J, s.f, s.at + x)
    rc = solve(RefineProblem(_GuessSized(pts), SVector(1.0, -2.0, 3.0)), LevenbergMarquardt(; optimization_strategy = nothing))
    @test VgtkCore.successful_retcode(rc)
    @test rc.u ≈ SVector(1.5, -1.8, 2.5) atol = 1.0e-8
    @test_throws ArgumentError VgtkCore.chart_size(_GuessSized(pts))   # the one-argument size is never read
end

# A3: the one-shot block refine runs on the task's kept problem of its kind — re-seeded in place on the same structure
# (bit for bit the `init`/`solve!` refine), rebuilt on another, one per task and kind whatever the sizes, its
# allocations per call flat in the size.
@testitem "block refine: the one-shot solve's task-kept problem" begin
    using Test, StaticArrays
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: RefineProblem, LevenbergMarquardt, solve, init, solve!, NLLSsolver, Triggs, WelschShape
    # a line y = a + b·s, one row per datum; `rob` the block's robustifier (NoRobust until `with_robustifier`)
    struct _KRow{K} <: NLLSsolver.AbstractResidual
        s::Float64; y::Float64; w::Float64; rob::K
    end
    NLLSsolver.ndeps(::_KRow) = NLLSsolver.static(2)
    NLLSsolver.nres(::_KRow) = NLLSsolver.static(1)
    NLLSsolver.varindices(::_KRow) = SVector(1, 2)
    NLLSsolver.getvars(::_KRow, vars::Vector) = (vars[1]::SVector{1, Float64}, vars[2]::SVector{1, Float64})
    NLLSsolver.robustkernel(r::_KRow) = r.rob
    Base.eltype(::_KRow) = Float64
    NLLSsolver.computeresidual(r::_KRow, a, b) = SVector(r.w * (a[1] + b[1] * r.s - r.y))
    VgtkCore.with_robustifier(r::_KRow, w, k) = _KRow(r.s, r.y, w, k)
    struct _KLine <: VgtkCore.AbstractBlockProblem
        s::Vector{Float64}; y::Vector{Float64}
    end
    VgtkCore.default_optimization_strategy(::_KLine) = nothing
    VgtkCore.codimension(::_KLine) = 1
    VgtkCore.block_variables(::_KLine, u0) = Union{SVector{1, Float64}}[SVector(u0[1]), SVector(u0[2])]
    VgtkCore.residual_blocks(f::_KLine, _u0) = [_KRow(si, yi, 1.0, NLLSsolver.NoRobust()) for (si, yi) in zip(f.s, f.y)]
    VgtkCore.block_result(::_KLine, _u0, r, summary) =
        VgtkCore.FitResult((r.u[1][1], r.u[2][1]), r.retcode, r.stop_rule, r.stats, summary.rms, summary.n, summary.S)
    line(n, k) = (s = collect(range(0.0, 5.0; length = n)); (s, 0.5 .+ 2.0 .* s .+ 0.01 .* sin.(k .* s)))
    same(a, b) = a.u === b.u && (a.retcode, a.stop_rule, a.stats, a.metadata) === (b.retcode, b.stop_rule, b.stats, b.metadata)
    plain = LevenbergMarquardt()
    robust = LevenbergMarquardt(; optimization_strategy = Triggs{WelschShape}())
    for (alg, kw) in ((plain, (;)), (robust, (; scale = 0.05)))
        fetch(@async begin
            for (n, k) in ((20, 3), (20, 5), (31, 3), (20, 7))   # a re-seed, a rebuild (another size), a rebuild back
                f = _KLine(line(n, k)...)
                one = solve(RefineProblem(f, (0.0, 0.0)), alg; kw...)
                @test same(one, solve!(init(RefineProblem(f, (0.0, 0.0)), alg; kw...)))
                @test one.u[2] ≈ 2.0 atol = 0.05
                kept = VgtkCore.task_kept_blocks(_KLine)
                @test length(kept) == 1 && length(only(kept)) == n && only(kept)[1].s == f.s[1]
            end
        end)
        counts = map((50, 500, 5000)) do n
            f = _KLine(line(n, 3)...)
            g = _KLine(line(n, 4)...)
            solve(RefineProblem(f, (0.0, 0.0)), alg; kw...)
            @allocations solve(RefineProblem(g, (0.0, 0.0)), alg; kw...)
        end
        # flat in n: no allocation per block; each n-sized array (the call's blocks, the summary's rows and squares) is
        # one allocation below Julia's large-array size and two above it
        @test maximum(counts) - minimum(counts) <= 8 && maximum(counts) < 100
    end
    @test isempty(fetch(@async VgtkCore.task_kept_blocks(_KLine)))   # a fresh task keeps none
    # plain and robust refines of one bundle on one task: two kinds, two kept problems, each exact
    fetch(@async begin
        f = _KLine(line(20, 3)...)
        for (alg, kw) in ((plain, (;)), (robust, (; scale = 0.05)), (plain, (;)), (robust, (; scale = 0.05)))
            @test same(solve(RefineProblem(f, (0.0, 0.0)), alg; kw...), solve!(init(RefineProblem(f, (0.0, 0.0)), alg; kw...)))
        end
        @test length(VgtkCore.task_kept_blocks(_KLine)) == 2
    end)
    # a refine started from a running one's callback, on the same task and kind: a problem of its own, both exact
    f = _KLine(line(20, 3)...); g = _KLine(line(20, 5)...)
    inner = Ref{Any}(nothing)
    cb = st -> (inner[] === nothing && (inner[] = solve(RefineProblem(g, (0.0, 0.0)), plain)); false)
    outer = solve(RefineProblem(f, (0.0, 0.0)), plain; callback = cb)
    @test outer.u === solve(RefineProblem(f, (0.0, 0.0)), plain).u
    @test same(inner[], solve!(init(RefineProblem(g, (0.0, 0.0)), plain)))
end

# C17: the block refine tells the bundle the two vectors NLLSsolver evaluates its blocks at (`prepare_evaluation!`),
# before the first evaluation and again before each evaluation at the solution: every block evaluation, in the one-shot
# solve, the cache's solve! and a `max_iter = 0` run, reads one of the two vectors the latest call named, and both are
# read (the trial vector too). The default does nothing, allocates nothing and is inferred.
@testitem "block refine: `prepare_evaluation!` names the vectors every block evaluation reads" begin
    using Test, StaticArrays
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: RefineProblem, LevenbergMarquardt, LmStops, solve, init, solve!, reinit!, NLLSsolver
    const _EVENTS = Any[]     # (:told, variables, varnext) and (:read, v), in order
    struct _ERow <: NLLSsolver.AbstractResidual
        s::Float64; y::Float64
    end
    NLLSsolver.ndeps(::_ERow) = NLLSsolver.static(2)
    NLLSsolver.nres(::_ERow) = NLLSsolver.static(1)
    NLLSsolver.varindices(::_ERow) = SVector(1, 2)
    function NLLSsolver.getvars(::_ERow, vars::Vector)
        push!(_EVENTS, (:read, vars))
        return (vars[1]::SVector{1, Float64}, vars[2]::SVector{1, Float64})
    end
    Base.eltype(::_ERow) = Float64
    NLLSsolver.computeresidual(r::_ERow, a, b) = SVector(a[1] + b[1] * r.s - r.y)
    struct _ELine <: VgtkCore.AbstractBlockProblem
        s::Vector{Float64}; y::Vector{Float64}
    end
    VgtkCore.default_optimization_strategy(::_ELine) = nothing
    VgtkCore.block_variables(::_ELine, u0) = Union{SVector{1, Float64}}[SVector(u0[1]), SVector(u0[2])]
    VgtkCore.residual_blocks(f::_ELine, _u0) = [_ERow(si, yi) for (si, yi) in zip(f.s, f.y)]
    VgtkCore.block_result(::_ELine, _u0, r, summary) = (; u = (r.u[1][1], r.u[2][1]), r.retcode, summary.S)
    VgtkCore.prepare_evaluation!(::_ELine, variables, varnext) = (push!(_EVENTS, (:told, variables, varnext)); nothing)
    # every read after the first call of one of the latest call's two vectors; returns (reads before any call — the
    # problem's construction: NLLSsolver's `addcost!` checks each block's `getvars` once, no evaluation —, calls,
    # reads of variables, reads of varnext)
    function audit(events)
        told = nothing; pre = 0; n = 0; nv = 0; nn = 0
        for e in events
            if e[1] === :told
                told = e; n += 1
            elseif told === nothing
                pre += 1
            else
                v = e[2]
                v === told[2] ? (nv += 1) : v === told[3] ? (nn += 1) : @test(false)
            end
        end
        return pre, n, nv, nn
    end
    s = collect(0.0:1.0:5.0); y = 0.5 .+ 2.0 .* s .+ 0.01 .* sin.(3 .* s)
    f = _ELine(s, y)
    alg = LevenbergMarquardt()
    for (run, built) in ((() -> solve(RefineProblem(f, (0.0, 0.0)), alg), true),         # the one-shot solve (built)
                         (() -> solve(RefineProblem(f, (0.1, 1.0)), alg), false),        # the one-shot solve (re-seeded)
                         (() -> (c = init(RefineProblem(f, (0.0, 0.0)), alg); empty!(_EVENTS); solve!(c)), false))
        empty!(_EVENTS)
        r = run()
        @test r.u[2] ≈ 2.0 atol = 0.05
        pre, n, nv, nn = audit(_EVENTS)
        @test pre == (built ? length(s) : 0)
        @test n >= 2 && nv > 0 && nn > 0                     # told before the run and after it; both vectors read
    end
    # the cache, reinit!'d: the same vectors, told again for the new solve
    c = init(RefineProblem(f, (0.0, 0.0)), alg); solve!(c)
    reinit!(c, (0.2, 1.5)); empty!(_EVENTS)
    @test solve!(c).u[2] ≈ 2.0 atol = 0.05
    pre, n, nv, nn = audit(_EVENTS); @test pre == 0 && n >= 2 && nv > 0 && nn > 0
    # no LM (`max_iter = 0`): told once, before the summary's evaluation at the guess
    empty!(_EVENTS)
    r0 = solve(RefineProblem(f, (0.0, 0.0)), alg; stops = LmStops(; max_iter = 0))
    @test r0.u === (0.0, 0.0)
    pre, n, nv, nn = audit(_EVENTS); @test pre == 0 && n == 1 && nv > 0 && nn == 0
    # the default: nothing, no allocation, inferred (compiled away at the call sites)
    v = Union{SVector{1, Float64}}[SVector(0.0), SVector(0.0)]
    struct _Silent <: VgtkCore.AbstractBlockProblem end
    @test @inferred(VgtkCore.prepare_evaluation!(_Silent(), v, v)) === nothing
    VgtkCore.prepare_evaluation!(_Silent(), v, v)
    @test (@allocated VgtkCore.prepare_evaluation!(_Silent(), v, v)) == 0
    @test Base.return_types(VgtkCore.prepare_evaluation!, (_Silent, typeof(v), typeof(v))) == [Nothing]
    @test VgtkCore.prepare_evaluation! in VgtkCore.CONTRACT_GENERICS && VgtkCore.prepare_evaluation! in VgtkCore.SOLVER_HOOKS
end

# C1: a chart problem whose chart has no coordinates (every parameter held) is not solved: the model at the guess,
# `(Success, ClosedForm)`, no step, its cost there, stationary — one-shot and cache alike.
@testitem "chart refine: a zero-dimensional chart is the guess, closed form" begin
    using Test, LinearAlgebra, StaticArrays
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: RefineProblem, LevenbergMarquardt, solve, init, solve!, ReturnCode, ClosedForm
    # a circle (cx, cy, ρ) held entirely: its rows, the points' distances to it, on a chart of no coordinates
    struct _HeldCircle <: VgtkCore.AbstractChartProblem
        pts::Vector{SVector{2, Float64}}
    end
    struct _HeldRows
        pts::Vector{SVector{2, Float64}}
        θ::SVector{3, Float64}
    end
    struct _HeldChart          # the chart of no coordinates: the held model itself
        θ::SVector{3, Float64}
    end
    VgtkCore.model_type(::_HeldCircle) = SVector{3, Float64}
    VgtkCore.chart_size(f::_HeldCircle) = (length(f.pts), Val(0))
    VgtkCore.model_chart(::_HeldCircle, θ) = _HeldChart(SVector{3, Float64}(θ))
    VgtkCore.chart_rows(f::_HeldCircle, ch, _sqrtw) = _HeldRows(f.pts, ch.θ)
    VgtkCore.chart_model(::_HeldCircle, ch) = ch.θ
    VgtkCore.recenter(ch::_HeldChart, ::SVector{0}) = ch
    VgtkCore.chart_result(::_HeldCircle, _u0, r::VgtkCore.FitResult) = r   # the full record: the cost, the chart point, `stationary`
    function VgtkCore.residual_rows!(r, s::_HeldRows, _x)
        for (i, p) in enumerate(s.pts)
            r[i] = norm(p - SVector(s.θ[1], s.θ[2])) - s.θ[3]
        end
        return r
    end
    pts = [SVector(1.5 + 2.5cos(t), -1.8 + 2.5sin(t)) for t in range(0, 2π; length = 13)[1:12]]
    θ = SVector(1.0, -2.0, 3.0)
    cost = sum(p -> (norm(p - SVector(θ[1], θ[2])) - θ[3])^2, pts)
    for r in (solve(RefineProblem(_HeldCircle(pts), θ), LevenbergMarquardt(; optimization_strategy = nothing)),
              solve!(init(RefineProblem(_HeldCircle(pts), θ), LevenbergMarquardt(; optimization_strategy = nothing))))
        @test r.u === θ && (r.retcode, r.stop_rule) === (ReturnCode.Success, ClosedForm) && r.stats.nsteps == 0
        @test r.metadata.cost ≈ cost rtol = 1.0e-14
        @test r.metadata.stationary() === true && r.metadata.x === SVector{0, Float64}()
    end
end

# The block contract's one statement (`block_structure`): a bundle that states itself once gets its `stated` in the
# summary given to `block_result` (one statement per one-shot solve, `init` and `reinit!`), the refine inferred; the
# default keeps the two generics and adds no summary field.
@testitem "block refine: one statement per solve (`block_structure`), its `stated` the result's" begin
    using Test, StaticArrays
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: RefineProblem, LevenbergMarquardt, solve, init, solve!, reinit!, NLLSsolver
    struct _SRow <: NLLSsolver.AbstractResidual
        s::Float64; y::Float64
    end
    NLLSsolver.ndeps(::_SRow) = NLLSsolver.static(2)
    NLLSsolver.nres(::_SRow) = NLLSsolver.static(1)
    NLLSsolver.varindices(::_SRow) = SVector(1, 2)
    NLLSsolver.getvars(::_SRow, vars::Vector) = (vars[1]::SVector{1, Float64}, vars[2]::SVector{1, Float64})
    Base.eltype(::_SRow) = Float64
    NLLSsolver.computeresidual(r::_SRow, a, b) = SVector(a[1] + b[1] * r.s - r.y)
    const _STATEMENTS = Ref(0)
    struct _StatedLine <: VgtkCore.AbstractBlockProblem
        s::Vector{Float64}; y::Vector{Float64}
    end
    VgtkCore.default_optimization_strategy(::_StatedLine) = nothing
    function VgtkCore.block_structure(f::_StatedLine, u0)
        _STATEMENTS[] += 1
        vars = Union{SVector{1, Float64}}[SVector(u0[1]), SVector(u0[2])]
        return vars, [_SRow(si, yi) for (si, yi) in zip(f.s, f.y)], (; guess = u0, n = length(f.s))
    end
    VgtkCore.block_result(::_StatedLine, _u0, r, summary) =
        VgtkCore.FitResult((r.u[1][1], r.u[2][1]), r.retcode, r.stop_rule, r.stats, summary.rms, summary.stated.n, summary.stated)
    # the same line through the two generics: the default statement, no `stated` field
    struct _PlainLine <: VgtkCore.AbstractBlockProblem
        s::Vector{Float64}; y::Vector{Float64}
    end
    VgtkCore.default_optimization_strategy(::_PlainLine) = nothing
    VgtkCore.block_variables(::_PlainLine, u0) = Union{SVector{1, Float64}}[SVector(u0[1]), SVector(u0[2])]
    VgtkCore.residual_blocks(f::_PlainLine, _u0) = [_SRow(si, yi) for (si, yi) in zip(f.s, f.y)]
    VgtkCore.block_result(::_PlainLine, _u0, r, summary) = (; u = (r.u[1][1], r.u[2][1]), summary)
    s = collect(0.0:1.0:5.0); y = 0.5 .+ 2.0 .* s .+ 0.01 .* sin.(3 .* s)
    p = RefineProblem(_StatedLine(s, y), (0.0, 0.0))
    alg = LevenbergMarquardt()
    _STATEMENTS[] = 0
    r = solve(p, alg)
    @test _STATEMENTS[] == 1 && r.metadata === (; guess = (0.0, 0.0), n = 6) && r.n == 6
    @test isconcretetype(only(Base.return_types(solve, Tuple{typeof(p), typeof(alg)})))
    plain = solve(RefineProblem(_PlainLine(s, y), (0.0, 0.0)), alg)
    @test plain.u === r.u && !haskey(plain.summary, :stated)          # the same solve, no field added
    c = init(p, alg); @test _STATEMENTS[] == 2
    @test solve!(c).u === r.u && _STATEMENTS[] == 2                  # the cache's statement serves its solve
    reinit!(c, (0.1, 1.9)); @test _STATEMENTS[] == 3
    @test solve!(c).metadata.guess === (0.1, 1.9)                    # the re-seeded guess's statement
    reinit!(c, (0.0, 0.0); alg = LevenbergMarquardt()); @test _STATEMENTS[] == 4   # a rebuild states it once too
    @test solve!(c).u === r.u
end
