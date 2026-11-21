# =============================================================================
# kind_test_helpers.jl — two helpers of the tests of a kind, this package's and its owners' (a kind of another package,
# kind_spec.jl): a body read back as another record that embeds it reads it, and a valid record file edited into one a
# reader is to refuse. Public so a kind's owner tests its kinds with the same two helpers.
# =============================================================================

"""
    nested_roundtrip(T, x; style = VgioStyle(), kw...) -> T

`x` as the body of another record (`{"body": x}`), lowered on `style`, written,
parsed and lifted back as a `T` (`kw` to `JSON.lift`): the value a record of
another kind that embeds `x` reads. A kind's test checks that its body nests:
`nested_roundtrip(AbstractVignetting, v; style) == v`.
"""
nested_roundtrip(::Type{T}, x; style::VgioStyle = VgioStyle(), kw...) where {T} =
    JSON.lift(style, T, JSON.parse(JSON.json((body = x,); style)).body; kw...)

"""
    edited_copy(f, src, dst = joinpath(dirname(src), "edited.json")) -> String

The record file `src` parsed into mutable dictionaries, edited by `f(t)` (a key
removed, a value changed) and written to `dst`, which is returned: a record
a reader is to refuse, made from a valid file —
`@test_throws ArgumentError load(T, edited_copy(t -> delete!(t, "units"), path))`.
"""
function edited_copy(f, src::AbstractString, dst::AbstractString = joinpath(dirname(src), "edited.json"))
    t = JSON.parse(read(src, String); dicttype = Dict{String, Any})
    f(t)
    write(dst, JSON.json(t))
    return String(dst)
end
