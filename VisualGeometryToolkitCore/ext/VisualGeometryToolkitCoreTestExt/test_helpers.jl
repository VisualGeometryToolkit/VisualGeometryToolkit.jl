# =============================================================================
# The stack's shared test helpers (stubs and docstrings: src/quality/test_helpers.jl)
# =============================================================================

# The `m` residual rows of `lm` at chart point `θ`, into a fresh buffer of θ's
# element type (so a Dual flows through).
_rows(lm, θ, m::Integer) = residual_rows!(Vector{eltype(θ)}(undef, m), lm, θ)

rows_cost(lm, θ, m::Integer) = sum(abs2, _rows(lm, θ, m))

function check_rows_jacobian(
        lm, θ, m::Integer; atol = 0.0, rtol = 0.0, reference = ForwardDiff.jacobian,
        jacobian::AbstractMatrix = Matrix{Float64}(undef, m, length(θ))
    )
    J = residual_jacobian_rows!(jacobian, lm, θ)
    J_ref = reference(z -> _rows(lm, z, m), θ)
    err = maximum(abs, J - J_ref)
    bound = atol + rtol * max(1.0, maximum(abs, J_ref))
    ok = err <= bound
    ok || @warn "check_rows_jacobian: the analytic rows Jacobian misses its reference" reference err bound
    return ok
end

# A gate as a test: every finding printed, and `@test` that there is none.
function _test_none(found)
    isempty(found) || foreach(v -> println(stderr, v), found)
    Test.@test isempty(found)
    return found
end

test_naming(M::Module) = _test_none(check_naming(pkgdir(M)))
test_duplication(M::Module; kwargs...) = _test_none(check_duplication(M; kwargs...))

function thrown_message(f; shown::Bool = false)
    try
        f()
    catch e
        shown && return sprint(showerror, e)
        e isa ArgumentError ? (return e.msg) : rethrow()
    end
    return ""
end

# A recorded run is the bare run (block_coordinate_descent's `Recorded`).
# Everything a state holds that a run moves (numbers, arrays, best iterates,
# the nested states named in `nested`), not the shared workspaces it points into.
_snap(x::Union{Number, Symbol, Nothing}, nested) = x
_snap(x::AbstractArray{<:Number}, nested) = collect(x)
_snap(x::StaticArray, nested) = Tuple(x)
_snap(x::Attributed{<:Any, @NamedTuple{cost::Float64}}, nested) = (_snap(x.value, nested), x.cost)   # a best iterate
_snap(x, nested) = isbits(x) ? x : nameof(typeof(x)) in nested ?
    Tuple(_snap(getfield(x, i), nested) for i in 1:fieldcount(typeof(x))) : :skipped
_snap_state(st, nested) = Tuple(_snap(getfield(st, i), nested) for i in 1:fieldcount(typeof(st)))
_fingerprint(res, nested) = (_snap(estimate(res), nested), retcode(res), iterations(res))

function test_recorded(make; nested = RECORDED_NESTED_STATES)
    # the full run, bare and recorded
    st, lh, m, p, θ₀, max_iter = make()
    bare = block_coordinate_descent!(st, lh, m, p, θ₀; max_iter)
    bare_state = _snap_state(st, nested)
    st2, lh2, m2, p2, θ₀2, _ = make()
    rec = Recorded(st2)
    Test.@test isequal(_fingerprint(block_coordinate_descent!(rec, lh2, m2, p2, θ₀2; max_iter), nested), _fingerprint(bare, nested))
    Test.@test isequal(_snap_state(st2, nested), bare_state)
    Test.@test rec.reason[] === retcode(bare)
    k_run = iterations(bare)
    k_rec = length(rec.iterates)
    Test.@test k_rec == (retcode(bare) === :solve_failed ? k_run - 1 : k_run)
    Test.@test length(rec.cost) == length(rec.s2) == k_rec
    # iterate for iterate: the bare run truncated after k is the recorded run's k-th entry
    for k in 1:k_rec
        stk, lhk, mk, pk, θk, _ = make()
        resk = block_coordinate_descent!(stk, lhk, mk, pk, θk; max_iter = k)
        Test.@test isequal(current_cost(stk), rec.cost[k])
        Test.@test iterations(resk) == k
        k == k_rec && Test.@test isequal(_snap_state(stk, nested), bare_state)
    end
    return k_rec
end
