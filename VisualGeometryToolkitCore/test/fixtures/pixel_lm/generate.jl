# Regenerates reference.txt: the reference Levenberg–Marquardt — a static bounded
# `lmsolve(f, J, x0, nothing, lo, hi; maxit)` with Marquardt's scaling (MIT, Copyright (c) 2023
# kamesy), the routine the generic models' two-unknown pixel LM kernel (`Cameras._pixel_lm`)
# derives from — on the 1000 problems of problems.jl. The routine is not a dependency of the
# stack: give the path of the source file that defines it and the module it defines (the source
# the fixture was recorded from is named in this directory's git history):
#   julia --project=VisualGeometryToolkitCore/test generate.jl /path/to/Module.jl ModuleName
# Line 1: the problems' checksum; then one line per problem: px py F (%.17g).
using LinearAlgebra, StaticArrays, Random, Printf
include(ARGS[1])
const REF = getfield(Main, Symbol(ARGS[2]))
include(joinpath(@__DIR__, "problems.jl"))
ps = pixel_lm_problems()
open(joinpath(@__DIR__, "reference.txt"), "w") do io
    println(io, @sprintf("%.17g", pixel_lm_problems_checksum(ps)))
    for q in ps
        p, F, = REF.lmsolve(pixel_res(q.coef...), pixel_jac(q.coef...), SVector(q.x0, q.y0), nothing,
                            SVector(q.minx, q.miny), SVector(q.maxx + 0.999, q.maxy + 0.999); maxit = q.maxit)
        println(io, @sprintf("%.17g %.17g %.17g", p[1], p[2], F))
    end
end
