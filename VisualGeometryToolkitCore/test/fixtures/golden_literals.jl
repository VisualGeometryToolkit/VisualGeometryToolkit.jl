# The literal writer shared by the golden-fixture generators (the Core generator, at commit f745af7c; make_spline_lens_golden_io.jl
# in IO): a NamedTuple of numbers and arrays to Julia source. Floats go through `repr` (exact round trip, NaN as NaN),
# integers and Bools as themselves; an array of 2 or more dimensions as `reshape([...], dims...)`.
lit(x::Bool) = string(x)
lit(x::Integer) = string(x)
lit(x::Real) = repr(Float64(x))
function lit(A::AbstractArray)
    flat = "[" * join((lit(a) for a in A), ", ") * "]"
    return ndims(A) == 1 ? flat : "reshape($flat, $(join(size(A), ", ")))"
end
lit(t::NamedTuple) = "(; " * join(("$k = $(lit(v))" for (k, v) in pairs(t)), ", ") * ")"

# A 3 × N (or 2 × N) Float64 matrix, one column per item.
col(vs) = reduce(hcat, (Float64.(collect(v)) for v in vs))
