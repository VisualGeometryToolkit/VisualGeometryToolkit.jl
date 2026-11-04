# The division camera solve's outputs, bit for bit: f, the coefficients, every
# pose, sigma, ν, the per-view weights and the iteration count of
# `fit_division_camera` on clean, contaminated (one and two coefficients) and
# known-k scenes, and two `division_camera_step`s — recorded against the solve
# before its arrow (Schur) elimination was generalized to per-board blocks of any
# size, so a refactor of the elimination that changes one bit fails here.
#
# Bits are a property of the compiled code, not only of the algorithm: whether
# LLVM fuses a `muladd` depends on the Julia/LLVM version, the CPU, and the
# compilation of the enclosing function. So `===` holds on the Julia VERSION and
# CPU the fixture was recorded on (DIVISION_SOLVE_GOLDEN_JULIA / _CPU); elsewhere
# the item compares at tolerance 1e-12 relative to each recorded vector's largest
# entry (absolute for an all-zero one), and its testset name says which mode ran.
# Relative to the vector, not to each entry: a posterior weight
# of 2e-277 (an underflowing tail) moved by 1e-10 RELATIVE under a change that
# moved nothing else beyond 2e-13 (measured 2026-11-04: the known-k fit with the
# camera's depth loop routed through `_arrow_solve`).
@testitem "division_solve_golden" setup = [DivisionBoards] begin
    using Test
    include(joinpath(@__DIR__, "fixtures", "division_solve_golden_record.jl"))
    include(joinpath(@__DIR__, "fixtures", "division_solve_golden.jl"))
    exact = VERSION == DIVISION_SOLVE_GOLDEN_JULIA && Sys.CPU_NAME == DIVISION_SOLVE_GOLDEN_CPU
    mode = exact ? "bit for bit (=== on Julia $VERSION, $(Sys.CPU_NAME))" :
        "tolerance 1e-12 of each vector's scale (recorded on Julia $DIVISION_SOLVE_GOLDEN_JULIA, $DIVISION_SOLVE_GOLDEN_CPU; running $VERSION, $(Sys.CPU_NAME))"
    @info "division_solve_golden: $mode"
    same(x, y, scale) = exact ? x === y : abs(x - y) <= 1.0e-12 * (scale > 0 ? scale : 1.0)
    @testset "division_solve_golden: $mode" begin
        rec = golden_record()
        @test first.(rec) == first.(DIVISION_SOLVE_GOLDEN)
        for ((name, got), (_, want)) in zip(rec, DIVISION_SOLVE_GOLDEN)
            @test (name, length(got)) == (name, length(want))
            scale = maximum(abs, want; init = 0.0)
            @test (name, all(same(x, y, scale) for (x, y) in zip(got, want))) == (name, true)
        end
    end
end
