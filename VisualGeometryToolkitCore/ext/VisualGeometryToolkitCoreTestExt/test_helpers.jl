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

function test_naming(M::Module)
    found = check_naming(pkgdir(M))
    isempty(found) || foreach(v -> println(stderr, v), found)
    Test.@test isempty(found)
    return found
end
