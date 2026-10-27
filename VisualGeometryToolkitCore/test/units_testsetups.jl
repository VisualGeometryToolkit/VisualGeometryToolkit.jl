# Shared helpers of the unit tests. ReTestItems discovers `@testsetup` modules
# in `*_testsetups.jl` files; an item takes one with `setup = [UnitHelpers]`.

@testsetup module UnitHelpers
    using StaticArrays: SVector
    using VisualGeometryToolkitCore: m
    export within_ulp, from_mm

    # Within one unit in the last place, elementwise: FlexUnits' `ustrip(mm, ·)`
    # divides by the unit's scale, so a plain form it returns is within an ulp of
    # the millimetre value.
    within_ulp(a::Number, b::Number) = abs(a - b) <= eps(abs(b))
    within_ulp(a, b) = length(a) == length(b) && all(within_ulp(x, y) for (x, y) in zip(a, b))

    # A millimetre text read as a length, as a file reader builds it: the decimal
    # shift of the text, `parse(Float64, "…e-3") * m`.
    from_mm(x::Real) = parse(Float64, "$(repr(Float64(x)))e-3") * m
    from_mm(v::AbstractVector) = SVector{length(v)}(map(from_mm, v))
end
