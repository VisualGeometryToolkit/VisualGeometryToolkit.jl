# The `--key=value` CLI lookup (src/cli.jl): a default, or a required
# argument that throws when absent.

@testitem "cli_option" begin
    using Test
    using VisualGeometryEval: cli_option

    args = ["--outdir=/tmp/x", "--np=0", "--flag=", "positional"]
    @test cli_option(args, "outdir", "d") == "/tmp/x"
    @test cli_option(args, "np") == "0"
    @test cli_option(args, "flag") == ""                     # present but empty is present
    @test cli_option(args, "missing", "d") == "d"
    @test cli_option(args, "missing", nothing) === nothing
    @test_throws ErrorException cli_option(args, "missing")
    @test cli_option(["--np=1", "--np=2"], "np") == "1"       # the first occurrence wins
end
