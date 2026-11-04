# The division camera solve's outputs, bit for bit: f, the coefficients, every
# pose, sigma, ν, the per-view weights and the iteration count of
# `fit_division_camera` on clean, contaminated (one and two coefficients) and
# known-k scenes, and two `division_camera_step`s — recorded against the solve
# before its arrow (Schur) elimination was generalized to per-board blocks of any
# size, so a refactor of the elimination that changes one bit fails here.
@testitem "division_solve_golden" setup = [DivisionBoards] begin
    using Test
    include(joinpath(@__DIR__, "fixtures", "division_solve_golden_record.jl"))
    include(joinpath(@__DIR__, "fixtures", "division_solve_golden.jl"))
    rec = golden_record()
    @test first.(rec) == first.(DIVISION_SOLVE_GOLDEN)
    for ((name, got), (_, want)) in zip(rec, DIVISION_SOLVE_GOLDEN)
        @test (name, length(got)) == (name, length(want))
        @test (name, all(x === y for (x, y) in zip(got, want))) == (name, true)
    end
end
