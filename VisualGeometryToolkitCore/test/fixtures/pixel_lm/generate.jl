# Regenerates reference.txt: FastLevenbergMarquardt 0.1's static bounded `lmsolve` — the
# routine the generic models' two-unknown pixel LM kernel (`Cameras._pixel_lm`) derives from
# — on the 1000 problems of problems.jl. FastLevenbergMarquardt left the stack on 2026-10-27,
# so this runs it from a depot's source (MIT, Copyright (c) 2023 kamesy):
#   julia --project=VisualGeometryToolkitCore/test generate.jl /path/to/FastLevenbergMarquardt/src/FastLevenbergMarquardt.jl
# Line 1: the problems' checksum; then one line per problem: px py F (%.17g).
using LinearAlgebra, StaticArrays, Random, Printf
include(ARGS[1])
const FLM = FastLevenbergMarquardt
include(joinpath(@__DIR__, "problems.jl"))
ps = pixel_lm_problems()
open(joinpath(@__DIR__, "reference.txt"), "w") do io
    println(io, @sprintf("%.17g", pixel_lm_problems_checksum(ps)))
    for q in ps
        p, F, = FLM.lmsolve(pixel_res(q.coef...), pixel_jac(q.coef...), SVector(q.x0, q.y0), nothing,
                            SVector(q.minx, q.miny), SVector(q.maxx + 0.999, q.maxy + 0.999); maxit = q.maxit)
        println(io, @sprintf("%.17g %.17g %.17g", p[1], p[2], F))
    end
end
