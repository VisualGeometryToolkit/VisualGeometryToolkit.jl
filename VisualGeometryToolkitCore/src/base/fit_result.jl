# =============================================================================
# FitResult — the one return type for estimator fits
# =============================================================================
#
# Anchored on the ecosystem standards (SciML's `retcode`, lsqnonlin's
# exitflag/resnorm/output): every fit returns the same four-field
# core — the estimate, WHY iteration stopped, how many iterations, and the
# residual summary with RMS first-class — plus solver-specific diagnostics
# in `metadata`.
#
# TYPE STABILITY IS A HARD REQUIREMENT (2026-09-02 design round): no
# Union-typed fields — the metadata VARIETY is carried by the type parameter
# `M`, one CONCRETE type per solver family, selected at COMPILE TIME (supply
# the type, never a runtime flag that switches the return type). Derived
# quantities (covariance) are functions computed on demand, never
# maybe-slots.
#
# PROPERTY FORWARDING (2026-09-03): `r.weights` reaches `r.metadata.weights`
# through a `@generated getproperty` over `fieldnames(M)` — the same
# mechanism `Attributed` uses, type-stable, resolved at compile time. This
# makes FitResult "Attributed with a fixed core" and lets consumer packages
# fold their `Attributed{M, A}` fit results onto it
# without rewriting every `.inlier_mask` at the call sites. What the
# design round actually guards against survives intact: a name that is
# neither a core field nor a metadata field is an ERROR, never a silent
# `nothing`. A metadata field that would shadow a core field is rejected
# at first use.

"""
    FitResult{P, M}

Result of a fit: `estimate::P`, `retcode::Symbol` (`:converged`,
`:max_iterations`, `:degenerate`, …; a closed-form solve reports `:solved`
with `iterations = 0`), `iterations`, `rms` over `n` residuals, and solver-specific `metadata::M` (`nothing` for the lean
default; see [`BundleMetadata`](@ref)).

For a RANSAC the loop count `iterations` is its trials — the hypotheses drawn.

Read a result through its accessors, which are the API: [`estimate`](@ref),
[`retcode`](@ref), [`converged`](@ref), [`iterations`](@ref), [`rms`](@ref),
[`metadata`](@ref), and the metadata's own ([`weights`](@ref), [`dof`](@ref), …).
Field and property names are an implementation detail.

(A property spelling still resolves, so older call sites keep working: the
core fields, the derived `converged`, and metadata fields forwarded from
`metadata`. A property that exists on none of them is an error; a metadata
type with a field named like a core accessor (`estimate`, `retcode`,
`iterations`, `rms`, `n`, `metadata`, `converged`) is rejected at
construction.)
"""
struct FitResult{P, M}
    estimate::P
    retcode::Symbol
    iterations::Int
    rms::Float64
    n::Int
    metadata::M
    function FitResult{P, M}(estimate, retcode, iterations, rms, n, metadata) where {P, M}
        _reject_shadowing(M)
        return new{P, M}(estimate, retcode, iterations, rms, n, metadata)
    end
end
FitResult(estimate::P, retcode::Symbol, iterations::Integer, rms::Real, n::Integer, metadata::M) where {P, M} =
    FitResult{P, M}(estimate, retcode, Int(iterations), Float64(rms), Int(n), metadata)

# The core field names, in field order — the one list both the forwarder
# and the shadowing check derive from. `:converged` is not a field but IS a
# property (derived from `retcode` on access), so it is reserved with them.
const _FITRESULT_CORE = (:estimate, :retcode, :iterations, :rms, :n, :metadata)
const _FITRESULT_PROPS = (_FITRESULT_CORE..., :converged)

# A metadata type whose field would collide with the core is refused when
# the result is BUILT: silently resolving `r.n` to one of two different
# things is exactly the ambiguity this type exists to kill. The common case
# (no nested metadata, no clash) is a constant-folded comparison of field
# names; the message is built only on the cold path.
function _reject_shadowing(::Type{M}) where {M}
    own = fieldnames(M)
    nested = nested_metadata(M)
    (isempty(nested) && isdisjoint(own, _FITRESULT_PROPS)) && return nothing
    msg = _shadowing_message(M, own, nested)
    msg === nothing || error(msg)
    return nothing
end

@noinline function _shadowing_message(M, own, nested)
    clash = intersect(_FITRESULT_PROPS, own)
    isempty(clash) || return "FitResult metadata type $M declares field(s) $clash, which would shadow " *
        "FitResult's own core fields — rename them in the metadata type"
    # Nested names (`nested_metadata`) share the one property namespace with
    # the core, the outer fields and each other.
    seen = Set{Symbol}((_FITRESULT_PROPS..., own...))
    for g in nested
        g in own || return "nested_metadata($M) names $g, which is not a field of $M"
        for f in fieldnames(fieldtype(M, g))
            f in seen && return "FitResult metadata type $M: nested field $g declares $f, which " *
                "would shadow a core field, an outer field or another nested field — " *
                "rename it in $(fieldtype(M, g))"
            push!(seen, f)
        end
    end
    return nothing
end

"""
    nested_metadata(::Type{M}) -> Tuple{Vararg{Symbol}}

The fields of the metadata type `M` that are themselves metadata records whose
fields a [`FitResult`](@ref) should forward too: for each listed field `g`,
`r.f` reaches `r.metadata.g.f` for every field `f` of `g`'s type, beside the
outer fields. Default `()`. A method is added once per metadata type, next to
its definition (`nested_metadata(::Type{<:MyMeta}) = (:calibration,)`).

The listed fields stay reachable under their own name. A nested field named
like a core field of the result, an outer field or another nested field is
refused when the result is built, exactly as an outer one is.
"""
nested_metadata(::Type) = ()


"""
    BundleMetadata(weights, residuals, fixed_frames, dof, loss_width, before, after)

Diagnostics of a bundle refinement: the final per-observation `weights` its
robust loss left and `residuals` (each observation's residual NORM, in the
bundle's order — what an inlier classification of the refined model reads; a
weight is the loss's guide to its optimizer, never inverted back into a
residual), the gauge-fixing `fixed_frames`, the degrees of freedom `dof`, the
robust loss's frozen `loss_width` (`NaN` under plain least squares — a LOSS
WIDTH, not a noise scale), and `before`/`after` residual summaries as concrete
`(; rms, median, p90)` NamedTuples — the FitResult's own `rms` mirrors
`after.rms` first-class.
"""
struct BundleMetadata{W <: AbstractVector{Float64}}
    weights::W
    residuals::Vector{Float64}
    fixed_frames::Vector{Int}
    dof::Int
    loss_width::Float64
    before::@NamedTuple{rms::Float64, median::Float64, p90::Float64}
    after::@NamedTuple{rms::Float64, median::Float64, p90::Float64}
    function BundleMetadata(weights::W, residuals::AbstractVector{<:Real}, fixed_frames::AbstractVector{<:Integer},
                            dof::Integer, loss_width::Real, before, after) where {W <: AbstractVector{Float64}}
        length(residuals) == length(weights) || throw(ArgumentError(
            "BundleMetadata: $(length(residuals)) residuals for $(length(weights)) weights — one each per observation"))
        return new{W}(weights, residuals, fixed_frames, dof, loss_width, before, after)
    end
end

"""
    estimate(r::FitResult) -> P

The fitted value — the model, camera, calibration record, … that the fit
produced. The SciML `sol.u`.
"""
estimate(r::FitResult) = r.estimate

"""
    retcode(r::FitResult) -> Symbol

Why the fit stopped: `:converged`, `:max_iterations`, `:degenerate`,
`:solve_failed`, `:stalled` (an LM that stopped at a non-stationary point,
[`lm_retcode`](@ref)), …; a closed-form solve reports `:solved`. Compare with
[`converged`](@ref) for the boolean.
"""
retcode(r::FitResult) = r.retcode

"""
    iterations(r::FitResult) -> Int

Iterations (or RANSAC trials) the fit ran; `0` for a closed-form solve.
"""
iterations(r::FitResult) = r.iterations

"""
    rms(r::FitResult) -> Float64

Root-mean-square residual over the `n` residuals the fit reports on
(`r.n`) — for a robust fit, the accepted set, so the number is not
outlier-dominated.
"""
rms(r::FitResult) = r.rms

"""
    metadata(r::FitResult{P, M}) -> M

The solver-specific diagnostics; `nothing` for a lean result. Its fields are
also reachable directly on `r` as properties.
"""
metadata(r::FitResult) = r.metadata

"""
    converged(r::FitResult) -> Bool

`retcode(r) === :converged` — the SciML `successful_retcode` convenience.
Also reachable as the property `r.converged`.
"""
converged(r::FitResult) = r.retcode === :converged

# -----------------------------------------------------------------------------
# Property forwarding into metadata — derived from `fieldnames(M)`, so a new
# metadata type gets its accessors for free and nothing is hand-listed.
# -----------------------------------------------------------------------------
@generated function Base.getproperty(r::FitResult{P, M}, s::Symbol) where {P, M}
    core = _FITRESULT_CORE          # shadowing already rejected at construction
    meta = fieldnames(M)
    ex = Expr[]
    for f in core
        push!(ex, :(s === $(QuoteNode(f)) && return getfield(r, $(QuoteNode(f)))))
    end
    # Derived, not stored: the one computed property, so `r.converged` reads
    # like every other diagnostic instead of being the lone function call.
    push!(ex, :(s === :converged && return getfield(r, :retcode) === :converged))
    for f in meta
        push!(ex, :(s === $(QuoteNode(f)) && return getfield(getfield(r, :metadata), $(QuoteNode(f)))))
    end
    # Nested metadata: read through an ordinary (non-generated) walk of
    # `nested_metadata(M)`, so a method a metadata type adds after this
    # generator was defined is seen (a generator cannot see later methods).
    push!(ex, :(return _nested_property(r, nested_metadata(M), s)))
    return Expr(:block, ex...)
end

# Walks the nested metadata fields; with a literal `s` the `hasfield` test
# folds, so the result is the one field's concrete type. No nested field has
# the name: the native field error, never `nothing`.
@inline _nested_property(r::FitResult, ::Tuple{}, s::Symbol) = getfield(r, s)
@inline function _nested_property(r::FitResult, nested::Tuple{Vararg{Symbol}}, s::Symbol)
    inner = getfield(getfield(r, :metadata), first(nested))
    hasfield(typeof(inner), s) && return getfield(inner, s)
    return _nested_property(r, Base.tail(nested), s)
end

function Base.propertynames(r::FitResult{P, M}) where {P, M}
    nested = Symbol[f for g in nested_metadata(M) for f in fieldnames(fieldtype(M, g))]
    return (_FITRESULT_PROPS..., fieldnames(M)..., nested...)
end

# Named accessors for the diagnostics the ecosystem asks for by verb. One
# generic method each — a lean result errors on the missing field (there is
# no `nothing` to return), exactly as before, just derived instead of
# hand-written per metadata type.
"""
    weights(r::FitResult) -> AbstractVector{Float64}

Per-observation weights from the metadata (`r.metadata.weights`). Errors on
a metadata type that carries none — including the lean `nothing`.
"""
weights(r::FitResult) = r.metadata.weights

"""
    dof(r::FitResult) -> Int

Residual degrees of freedom from the metadata (`r.metadata.dof`). Errors on
a metadata type that carries none.
"""
dof(r::FitResult) = r.metadata.dof
