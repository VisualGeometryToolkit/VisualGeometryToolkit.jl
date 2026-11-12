# The one install of a published artifact (src/base/artifacts.jl), hoisted from VisualGeometryToolkitEval's Experiments
# (which re-exports it, and whose suite installs a real local artifact through it) for BlobBoards, which has no Eval
# dependency: an unknown name and a placeholder URL are refused, and the Hugging Face token is read from the
# environment first, then from the file `hf auth login` writes.
@testitem "install_artifact: the one artifact install (refusals, the Hugging Face token)" begin
    using Test
    import VisualGeometryToolkitCore as VgtkCore
    dir = mktempdir()
    toml = joinpath(dir, "Artifacts.toml")
    write(toml, """
        [tier_c]
        git-tree-sha1 = "$("1"^40)"
        lazy = true

            [[tier_c.download]]
            url = "https://zenodo.org/records/ZENODO_RECORD_X/files/c.tar.gz"
            sha256 = "$("0"^64)"
        """)
    err = try VgtkCore.install_artifact("tier_zz", toml); nothing catch e; e end
    @test err isa ErrorException && occursin("unknown artifact 'tier_zz'", err.msg)
    err = try VgtkCore.install_artifact("tier_c", toml); nothing catch e; e end
    @test err isa ErrorException && occursin("not published", err.msg)
    hf_home = mktempdir()
    withenv("HF_TOKEN" => nothing, "HF_HOME" => hf_home) do
        @test VgtkCore.huggingface_token() === nothing
        write(joinpath(hf_home, "token"), "hf_file\n")
        @test VgtkCore.huggingface_token() == "hf_file"
        withenv(() -> (@test VgtkCore.huggingface_token() == "hf_env"), "HF_TOKEN" => " hf_env ")
    end
end
