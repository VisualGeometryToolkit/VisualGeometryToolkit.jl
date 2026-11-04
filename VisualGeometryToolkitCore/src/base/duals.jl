# =============================================================================
# ForwardDiff Duals: the package's one set of value and layer helpers
# =============================================================================
#
# Two ways to read a Dual's value, for two kinds of solver:
#
#   _primal(x)               every layer stripped: the plain number under all
#                            the Duals (a texel chosen on it, a lens inverse
#                            solved on it, then ONE Newton step on the Duals)
#   _outer_value(D, x)       only D's outermost layer stripped (an implicit
#                            derivative taken one layer at a time, so nested
#                            Duals — ForwardDiff.hessian — differentiate
#                            exactly to every order)
#
# with `_with_value(D, v, p)` and `_lift_dual(D, v)` putting a value of the next
# layer back into D.
# Allocation-free and branch-free in the types, so they run on the device.
# =============================================================================

@inline _primal(x::Real) = x
@inline _primal(x::ForwardDiff.Dual) = _primal(ForwardDiff.value(x))

@inline _is_dual_type(::Type{T}) where {T} = T !== Union{} && T <: ForwardDiff.Dual
_primal_float(::Type{T}) where {T} = _is_dual_type(T) ? _primal_float(ForwardDiff.valtype(T)) : T

_partials_type(::Type{ForwardDiff.Dual{G, V, N}}) where {G, V, N} = ForwardDiff.Partials{N, V}
# A Dual of D's type with the value `v` (of D's value type) and the partials `p`.
@inline _with_value(::Type{ForwardDiff.Dual{G, V, N}}, v, p::ForwardDiff.Partials{N}) where {G, V, N} =
    ForwardDiff.Dual{G}(convert(V, v), convert(ForwardDiff.Partials{N, V}, p))
# `v` lifted into D with zero partials: under ForwardDiff.hessian both layers
# carry ONE tag, and a next-layer value meeting a D in arithmetic would be read
# as D's layer (perturbation confusion), so the lift is explicit.
@inline _lift_dual(::Type{D}, v) where {D <: ForwardDiff.Dual} = _with_value(D, v, zero(_partials_type(D)))

# Whether the element type E is D's outermost layer: E's outer tag is D's,
# nested in itself as often as in D (ForwardDiff.hessian nests one tag inside
# itself; other tags may sit inside either).
_tag_depth(::Type{ForwardDiff.Dual{G, W, M}}, ::Type{ForwardDiff.Dual{H, V, N}}) where {G, W, M, H, V, N} =
    (H === G) + _tag_depth(ForwardDiff.Dual{G, W, M}, V)
_tag_depth(::Type, ::Type) = 0
_in_layer(::Type{D}, ::Type{E}) where {G, D <: ForwardDiff.Dual{G}, E} =
    E <: ForwardDiff.Dual{G} && _tag_depth(D, E) == _tag_depth(D, D)

# The values of D's outermost layer; anything else (plain, an inner layer) as it is.
@inline _outer_value(::Type{D}, x::Real) where {D <: ForwardDiff.Dual} = _in_layer(D, typeof(x)) ? ForwardDiff.value(x) : x
@inline _outer_value(::Type{D}, v::StaticArray) where {D <: ForwardDiff.Dual} = map(x -> _outer_value(D, x), v)
