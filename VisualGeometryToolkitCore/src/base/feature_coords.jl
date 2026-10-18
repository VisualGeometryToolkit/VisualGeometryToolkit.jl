# =============================================================================
# The feature-coordinate compute boundary
# =============================================================================
#
# The ONE place a feature's coordinates (centre components, scale, radius) turn
# into plain numbers: plain numbers are returned as they are; quantities must
# all share one dimension, pixel or length, and strip in a named unit — pixels
# in `px`, lengths in `mm` (the stack's length unit for feature geometry). A px
# disc and an mm disc can never meet in a predicate as bare numbers, and a
# quantity never mixes with a bare number.

@inline _strip_coords(xs::NTuple{N, Real}) where {N} = xs
_strip_coords(::Tuple{}) = ()   # both methods below would match the empty tuple
function _strip_coords(xs::NTuple{N, Quantity}) where {N}
    u = _coord_unit(xs[1])
    return map(x -> ustrip(u, x), xs)   # each component's dimension is checked against `u`
end
# The unit a feature coordinate strips in: px for pixels, mm for a length.
function _coord_unit(x::Quantity)
    d = dimension(x)
    return d == dimension(px) ? px :
        d == dimension(mm) ? mm :
        throw(ArgumentError("feature coordinates must be pixels or a length, got dimensions $d"))
end
_strip_coords(xs::Tuple) = throw(
    ArgumentError("feature coordinates mix plain numbers with quantities: $(map(typeof, xs))")
)
