# Experiments.contract_violations — the codification contract's rules on
# minimal sources (the paper suite applies it to every driver).

@testitem "contracts" begin
    using Test
    using VisualGeometryEval
    using VisualGeometryEval.Experiments: contract_violations

    @testset "codification contract rules" begin
        dir = mktempdir()
        function violations(name, body)
            p = joinpath(dir, name)
            write(p, "# provenance: declared below\n" * body * "\nrun_stages(\"x\", STAGES)\n")
            contract_violations(p)
        end
        # F1 flags pgfplots grid styling, not a Julia binding named grid
        @test any(startswith("F1: grid="), violations("a.jl", "opts = raw\"grid=major, xmin=0\""))
        @test any(startswith("F1: grid="), violations("b.jl", "opts = raw\"grid = both\""))
        @test !any(startswith("F1: grid="), violations("c.jl", "grid = Dict(name => collect(ES) for name in names)"))
        @test !any(startswith("F1: grid="), violations("d.jl", "k_grid = init_grid(3)"))
        @test any(==("F1: ymajorgrids"), violations("e.jl", "opts = raw\"ymajorgrids=true\""))
        # R3: a codified driver needs a stage runner; support files and workers are exempt
        p = joinpath(dir, "f.jl"); write(p, "# provenance: x\nprintln(1)\n")
        @test any(startswith("R3: no run_stages("), contract_violations(p))
        w = joinpath(dir, "g_worker.jl"); write(w, "# provenance: driven by f.jl\nprintln(1)\n")
        @test !any(startswith("R3"), contract_violations(w))          # workers need no stage runner
        w2 = joinpath(dir, "h_worker.jl"); write(w2, "println(1)\n")
        @test any(==("R3: no provenance block"), contract_violations(w2))   # but still a provenance line
        s = joinpath(dir, "arm_common.jl"); write(s, "println(1)\n")
        @test isempty(contract_violations(s))                          # *common.jl support files are exempt
    end
end
