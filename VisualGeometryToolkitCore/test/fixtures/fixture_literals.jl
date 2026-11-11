# The Julia literals the fixture generators print (height_surface/generate.jl,
# thin_plate_spline/generate.jl): every number by `repr` (an exact round trip),
# vectors as `[…]`, static vectors as `SVector(…)`, a list of points or rows one
# per line. Included by each generator; it runs in the recording checkout's
# environment, which provides StaticArrays.
lit(x::Float64) = repr(x)
lit(x::Float32) = repr(x)
lit(v::AbstractVector) = "[" * join(lit.(v), ", ") * "]"
lit(v::SVector) = "SVector(" * join(lit.(v), ", ") * ")"
lit(vs::AbstractVector{<:SVector{2}}) = rows(vs)
rows(vs) = "[\n        " * join(lit.(vs), ",\n        ") * ",\n    ]"
