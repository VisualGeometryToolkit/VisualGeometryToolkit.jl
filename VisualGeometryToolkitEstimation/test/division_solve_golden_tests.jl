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
# the item compares at tolerance GOLDEN_RTOL = 1e-10 relative to each recorded
# vector's largest entry (absolute for an all-zero one), and its testset name says
# which mode ran. Relative to the vector, not to each entry: a posterior weight
# of 2e-277 (an underflowing tail) moved by 1e-10 RELATIVE under a change that
# moved nothing else beyond 2e-13 (measured 2026-11-04: the known-k fit with the
# camera's depth loop routed through `_arrow_solve`).
#
# GOLDEN_RTOL is the solve's rounding floor with a 10x margin, measured
# 2026-11-10 (Julia 1.13.2, raptorlake) by rerunning the four fits on inputs
# moved by random relative amounts of 1 and of 8 ulps (40 and 20 draws): the
# worst deviation, the two-coefficient `contaminated2.coefficients`, was
# 9.3e-12 of the vector's scale at 1 ulp and 9.2e-12 at 8, so it is the
# floor any change of rounding reaches (an FMA contraction on another CPU is
# one: Emerald Rapids moved `contaminated2.coefficients` and `.state.k` past the
# former 1e-12); `contaminated2.state.k` reached 1.3e-12, every other vector
# under 5.5e-13. A real change still fails by orders of magnitude: one board
# point moved by 1e-6 px moves `contaminated2` by 1.2e-8, and a solve stopped one
# iteration early changes `iterations` (and moves the rest by 1.0e-10).
@testitem "division_solve_golden" setup = [DivisionBoards] begin
    using Test
    include(joinpath(@__DIR__, "fixtures", "division_solve_golden_record.jl"))
    include(joinpath(@__DIR__, "fixtures", "division_solve_golden.jl"))
    GOLDEN_RTOL = 1.0e-10
    exact = VERSION == DIVISION_SOLVE_GOLDEN_JULIA && Sys.CPU_NAME == DIVISION_SOLVE_GOLDEN_CPU
    mode = exact ? "bit for bit (=== on Julia $VERSION, $(Sys.CPU_NAME))" :
        "tolerance $GOLDEN_RTOL of each vector's scale (recorded on Julia $DIVISION_SOLVE_GOLDEN_JULIA, $DIVISION_SOLVE_GOLDEN_CPU; running $VERSION, $(Sys.CPU_NAME))"
    @info "division_solve_golden: $mode"
    same(x, y, scale) = exact ? x === y : abs(x - y) <= GOLDEN_RTOL * (scale > 0 ? scale : 1.0)
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
