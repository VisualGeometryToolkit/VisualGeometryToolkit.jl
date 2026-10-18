# =============================================================================
# append_jsonl — one record per line, thread-safe, parseable back
# =============================================================================

@testitem "append_jsonl" setup = [VGIoFixtures] begin
    mktempdir() do dir
        path = joinpath(dir, "log.jsonl")
        append_jsonl(path, (; a = 1, b = "x"))
        append_jsonl(path, Dict("nested" => (; c = [1, 2, 3])))
        Threads.@threads for i in 1:64
            append_jsonl(path, (; i, t = Threads.threadid()))
        end
        lines = readlines(path)
        @test length(lines) == 66
        rows = [JSON3.read(l) for l in lines]
        @test rows[1].a == 1 && rows[1].b == "x"
        @test collect(rows[2].nested.c) == [1, 2, 3]
        @test sort([r.i for r in rows[3:end]]) == collect(1:64)
        # every line is complete JSON — no torn interleaves
        @test all(startswith(l, "{") && endswith(l, "}") for l in lines)
    end
end
