# quality_raw_decoding_tests.jl — the stack's one RAW path, checked by
# VisualGeometryToolkitCore's `check_raw_decoding`: every RAW → image step lives
# in VisualGeometryToolkitFeatures' raw module, and a use of a decoding library
# or an interpolation anywhere else is a finding.
#
# The synthetic sources that must trip the scan are assembled from fragments,
# so this file spells none of the flagged words.

@testitem "quality: RAW decoding lives in one module" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: check_raw_decoding, RawDecodingViolation

    # A throwaway package directory (not a git checkout: every file is in scope).
    function findings(files::Pair...; name = "Consumer")
        return mktempdir() do dir
            write(joinpath(dir, "Project.toml"), "name = \"$name\"\n")
            for (rel, src) in files
                path = joinpath(dir, rel)
                mkpath(dirname(path))
                write(path, src)
            end
            check_raw_decoding(dir)
        end
    end
    uses(files::Pair...; kw...) = sort!(unique(getfield.(findings(files...; kw...), :use)))

    rp = "raw" * "py"
    dm = "demo" * "saic"
    lr = "lib" * "raw_"
    dc = "dc" * "raw"
    pp = "post" * "process"

    @test uses("src/a.jl" => "f(p) = pyimport(\"$rp\").imread(p)\n") == [:rawpy]
    @test uses("src/a.jl" => "f(h) = @ccall libraw.$(lr)unpack(h::Ptr{Cvoid})::Cint\n") == [:libraw]
    @test uses("scripts/a.py" => "img = r.$(pp)(gamma=(1, 1))\n") == [:postprocess]
    @test uses("scripts/a.sh" => "$(dc)_emu -T x.ARW\n") == [:dcraw]
    @test uses("src/a.jl" => "x = $(dm)_bilinear(m)\n") == [:demosaic]
    @test uses("src/a.jl" => "x = interpolate_bayer(m)\n") == [:cfa_interpolation]
    @test uses("pixi.toml" => "[dependencies]\n$rp = \">=0.25\"\n") == [:rawpy]
    @test uses("src/a.jl" => "g = Gray.(PNGFiles.load(joinpath(dirs.linear, name)))\n") == [:png_raw_output]
    # not a use: a comment, a docstring, prose, and a PNG that is not a decoded RAW output
    @test isempty(findings("src/a.jl" => "# $rp was retired\n\"\"\"\n    f()\n\nNo $dm here.\n\"\"\"\nf() = 1\n#= $dc =#\n"))
    @test uses("src/a.jl" => "# σ² and ∂ in a comment\nĤ = 1\ny = $(dm)(Ĥ)\n") == [:demosaic]   # multi-byte characters
    @test isempty(findings("docs/src/a.md" => "The $dm mode is for figures.\n"))
    @test isempty(findings("src/a.jl" => "img = PNGFiles.load(joinpath(dir, \"logo.png\"))\n"))
    @test uses("src/a.jl" => "f(h) = ccall((:$(lr)unpack, lib), Cint, (Ptr{Cvoid},), h)\n") == [:libraw]
    # the decoder trigger is a dependency, and the mosaic's metadata field a name, not a decoding path
    @test isempty(findings("test/a.jl" => "md = (; decoder = \"test\", $(lr)version = \"\")\n"))
    @test isempty(findings("Project.toml" => "name = \"Consumer\"\n[weakdeps]\nLibRaw_jll = \"d66ac3f3-933a-5ab8-8aa2-9e591b3c5af3\"\n"))
    # the raw module itself is the one place: allow-listed in VisualGeometryToolkitFeatures only
    mod = "src/raw_image.jl" => "f(h) = @ccall libraw.$(lr)dcraw_process(h::Ptr{Cvoid})::Cint  # $dm\n"
    @test isempty(findings(mod; name = "VisualGeometryToolkitFeatures"))
    @test uses(mod) == [:dcraw, :libraw]
    # a finding names the file and the line, and prints the rule
    v = only(findings("src/a.jl" => "a = 1\nb = $(dm)(m)\n"))
    @test (v.file, v.line, v.use) == ("src/a.jl", 2, :demosaic)
    @test occursin("VisualGeometryToolkitFeatures", sprint(show, v))
end
