# quality_naming_tests.jl — the stack's iterative-estimator vocabulary
# (conventions/naming.md), checked by VisualGeometryToolkitCore's
# `check_naming`: this package against itself, and the checker against small
# synthetic packages, one rule at a time.
#
# The synthetic sources that must trip the banned-word and retired-name scans
# are assembled from the checker's own fragments (`_STEM`, `RETIRED_NAMES`),
# so this file, which the checker also reads, spells neither list.

@testitem "quality: naming" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using VisualGeometryToolkitCore
    VgtkCore.test_naming(VisualGeometryToolkitCore)
end

@testitem "quality: the thin-plate radial basis is not a kernel" begin
    # R10 (conventions/naming.md): "kernel" is a GPU, convolution, minimal-solver,
    # null-space or probability-density kernel; the thin-plate spline's
    # U(r) = r² log r is its radial basis.
    using Test
    using VisualGeometryToolkitCore
    root = pkgdir(VisualGeometryToolkitCore)
    pattern = r"kernel\s+`?(U\(r\)\s*=\s*)?r² log r|the kernel's scale"
    hits = String[]
    for dir in (joinpath(root, "src"), joinpath(root, "docs", "src")), (d, _, fs) in walkdir(dir), f in fs
        endswith(f, ".jl") || endswith(f, ".md") || continue
        occursin(pattern, read(joinpath(d, f), String)) && push!(hits, relpath(joinpath(d, f), root))
    end
    @test isempty(hits)
end

@testitem "quality: naming checker" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: check_naming, NamingViolation, naming_scope

    # A throwaway package directory (not a git checkout: every file is in scope).
    function findings(files::Pair...)
        return mktempdir() do dir
            for (rel, src) in files
                path = joinpath(dir, rel)
                mkpath(dirname(path))
                write(path, src)
            end
            check_naming(dir)
        end
    end
    rules(files::Pair...) = sort!(unique(getfield.(findings(files...), :rule)))
    quiet(files::Pair...) = isempty(findings(files...))

    NC = Base.get_extension(VisualGeometryToolkitCore, :VisualGeometryToolkitCoreTestExt)   # the checker (loads with Test)
    # The units rule's allowance (R11) names exactly the two files that must spell what it bans — the rule's own lists
    # and the units module's own tests — and nothing else: a new allowance fails here (owner, 2026-11-26: no exemptions).
    @test Set(keys(NC._UNIT_RULE_ALLOW)) == Set([
        ("VisualGeometryToolkitCore", "ext/VisualGeometryToolkitCoreTestExt/check_naming.jl"),
        ("VisualGeometryToolkitCore", "test/units_tests.jl"),
    ])
    P = NC._STEM                                 # the banned stem, from the checker
    Cap = uppercasefirst(P)
    old = NC.RETIRED_NAMES[1]                    # any retired name

    @testset "R9: an algorithm's tag is its one tag, not the retired function's name" begin
        tag = "fit" * "_scale"                       # a retired function's name, and the tag of an algorithm a package may define
        @test tag in NC.RETIRED_NAMES
        alg = "struct FitScale <: AbstractEstimationAlgorithm\n    block::Int\nend\n"
        # the package defines the algorithm: its TOML table and a reader's string literal spell its tag
        @test quiet("src/alg.jl" => alg, "configs/default.toml" => "[$(tag)]\nsignificance = 0.001\n",
                    "src/read.jl" => "const S = toml_default(D, \"$(tag)\", \"significance\")\n")
        # derived through the tag rule, qualified supertype or minimal solver alike
        @test quiet("src/alg.jl" => "struct FitScale{T} <: VgtkCore.AbstractEstimationAlgorithm end\n",
                    "configs/a.toml" => "[$(tag)]\nx = 1\n")
        # under an intermediate abstract type too (the supertype chain, followed)
        @test quiet("src/alg.jl" => "abstract type SensorAlgorithm <: AbstractEstimationAlgorithm end\nstruct FitScale <: SensorAlgorithm end\n",
                    "configs/a.toml" => "[$(tag)]\nx = 1\n")
        # without the algorithm (neither the package's nor a loaded one) the table is the retired name; and in code it is
        # the retired name, always
        @test rules("configs/default.toml" => "[$(tag)]\nx = 1\n") == [:R9]
        @test rules("src/alg.jl" => alg, "src/a.jl" => "x = $(tag)(frames)\n") == [:R9]
        @test rules("src/alg.jl" => alg, "src/a.jl" => "f(x) = x   # $(tag) of a frame\n") == [:R9]
        # a LOADED algorithm's tag is its tag in any package (C10): Core's `PhotonTransfer` and `NormalCG`, configured by a
        # package that defines neither — a table, a tag value, a reader's string literal; prose and code stay retired
        for loaded in ("photon" * "_transfer", "normal" * "_cg")
            @test loaded in NC.RETIRED_NAMES
            @test quiet("configs/a.toml" => "[$(loaded)]\nx = 1\n", "configs/b.toml" => "[solver]\nalgorithm = \"$(loaded)\"\n",
                        "src/read.jl" => "const S = toml_default(D, \"$(loaded)\", \"x\")\n")
            @test rules("configs/a.toml" => "x = 1   # the $(loaded) solve\n") == [:R9]
            @test rules("src/a.jl" => "x = $(loaded)(N, g)\n") == [:R9]
        end
        # a retired name that no algorithm's tag spells stays retired in a TOML file, beside a loaded tag
        other = "grid_brent" * "_minimum"
        @test other in NC.RETIRED_NAMES
        @test rules("configs/a.toml" => "[$(other)]\nx = 1\n") == [:R9]
        @test rules("src/alg.jl" => alg, "configs/a.toml" => "[$(other)]\nsolver = \"normal" * "_cg\"\n") == [:R9]
    end

    @testset "R9: a foreign package's function of a retired name is its owner's" begin
        tri = "triang" * "ulate"                     # a retired stack name, and DelaunayTriangulation's function
        @test tri in NC.RETIRED_NAMES
        # imported explicitly from a non-stack package (one line, several lines) or qualified by one: its calls are that package's
        @test quiet("src/a.jl" => "using DelaunayTriangulation: $(tri), each_solid_triangle\nt = $(tri)(nodes)\n")
        @test quiet("src/a.jl" => "using DelaunayTriangulation: each_solid_triangle,\n    $(tri)\nt = $(tri)(nodes)\n")
        @test quiet("src/a.jl" => "import DelaunayTriangulation\nt = DelaunayTriangulation.$(tri)(nodes)\n")
        @test quiet("src/a.jl" => "using DelaunayTriangulation\nt = DelaunayTriangulation.$(tri)(nodes)\n")
        # from a stack package, by an alias, or unqualified without a foreign import, it is the retired name
        @test rules("src/a.jl" => "using VisualGeometryToolkitCore: $(tri)\n") == [:R9]
        @test rules("src/a.jl" => "x = VgtkCore.$(tri)(f, g)\n") == [:R9]
        @test rules("src/a.jl" => "x = VisualGeometryToolkitCore.Sub.$(tri)(f, g)\n") == [:R9]   # a stack submodule
        @test rules("src/a.jl" => "using DelaunayTriangulation: each_solid_triangle\nt = $(tri)(nodes)\n") == [:R9]
        # a qualifier excuses a hit only when it is a module the file brings in from a non-stack package: never a field,
        # a stack module, a stack package's alias, or a downstream stack package
        @test rules("src/a.jl" => "y = cfg.$(tri)\n") == [:R9]
        @test rules("src/a.jl" => "using DelaunayTriangulation: $(tri)\nx = VgtkCore.$(tri)(f, g)\n") == [:R9]
        @test rules("src/a.jl" => "import VisualGeometryToolkitCore as VGC\nx = VGC.$(tri)(f, g)\n") == [:R9]
        @test quiet("src/a.jl" => "import DelaunayTriangulation as DT\nt = DT.$(tri)(nodes)\n")
        @test quiet("src/a.jl" => "for pkg in (:A, :B)\n    @eval import \$pkg\nend\n")   # an interpolated import parses
        sc = "start" * "_calibration"
        @test rules("src/a.jl" => "using RadiometricCamera: $(sc)\n") == [:R9]
        @test rules("src/a.jl" => "import RadiometricCamera\nc = RadiometricCamera.$(sc)(g, p)\n") == [:R9]
        # a file importing it from a foreign package does not excuse another file
        @test rules("src/a.jl" => "using DelaunayTriangulation: $(tri)\n", "src/b.jl" => "t = $(tri)(x)\n") == [:R9]
    end

    @testset "R15: the kind words name kinds, never functions" begin
        k = "fit" * "_"; fd = "find" * "_"; rf = "refine" * "_"
        @test rules("src/a.jl" => "$(k)thing(x) = x\n") == [:R15]
        @test rules("src/a.jl" => "function $(fd)thing(x)\n    x\nend\n") == [:R15]
        @test rules("src/a.jl" => "$(rf)thing!(x) = x\n") == [:R15]
        @test rules("src/a.jl" => "export $(fd)thing\n") == [:R15]
        @test rules("src/a.jl" => "public $(rf)thing\n") == [:R15]
        @test rules("ext/E.jl" => "$(k)thing(x) = x\n") == [:R15]
        # an internal step, a test helper, a task verb and a local are not estimators' names
        @test quiet("src/a.jl" => "_$(k)thing(x) = x\n")
        @test quiet("test/a.jl" => "$(k)thing(x) = x\n")
        @test quiet("src/a.jl" => "$(fd)boards(img) = img\n")
        @test quiet("src/a.jl" => "function g(x)\n    $(k)thing = x\n    return $(k)thing\nend\n")
        @test all(v -> !isempty(v), values(NC.TASK_VERBS))   # each task verb states its reason
    end

    @testset "R19: solve is the one verb" begin
        sv = "solve" * "_"
        @test rules("src/a.jl" => "$(sv)block!(x) = x\n") == [:R19]
        @test rules("test/a.jl" => "function g(x)\n    $(sv)count = 1\n    return $(sv)count\nend\n") == [:R19]
        @test rules("scripts/a.py" => "def $(sv)thing(x):\n    return x\n") == [:R19]
        # a value (a quoted symbol), an internal `_` name and a use of an owned name are not bindings of it
        @test quiet("src/a.jl" => "const S = :$(sv)failed\n")
        @test quiet("src/a.jl" => "_$(sv)view(x) = x\n")
    end

    @testset "R20: no alias of an algorithm or a solve verb, no algorithm value in a constant" begin
        @test rules("src/a.jl" => "const MyLM = LevenbergMarquardt\n") == [:R20]
        @test rules("src/a.jl" => "const MyDLT = VgtkCore.DLT\n") == [:R20]
        @test rules("src/a.jl" => "const DEFAULT_LM = LevenbergMarquardt()\n") == [:R20]
        @test rules("src/a.jl" => "const estimate_it = solve\n") == [:R20]
        @test rules("src/a.jl" => "struct Mine <: AbstractEstimationAlgorithm end\nconst Yours = Mine\n") == [:R20]
        @test rules("src/a.jl" => "struct Mine{T} <: AbstractEstimationAlgorithm end\nconst Yours{T} = Mine{T}\n") == [:R20]
        # a constant of another kind, and a type alias of a result, are not aliases of an algorithm
        @test quiet("src/a.jl" => "const N = 3\nconst R{M} = FitResult{M}\nconst G = sqrt\n")
        # one concrete instantiation of a stack type is a second name for one type (owner, 2026-11-24): the package's
        # own type and a loaded stack type alike, bare or applied to all its free parameters
        @test rules("src/a.jl" => "struct Pt{T}\n    x::T\nend\nconst PtF = Pt{Float64}\n") == [:R20]
        @test rules("src/a.jl" => "struct Plain\nend\nconst Other = Plain\n") == [:R20]
        @test rules("src/a.jl" => "const Rigid = VgtkCore.EuclideanMap{3, Float64, RotMatrix{3, Float64, 9}, Vec3{Float64}, " *
                    "UnlabeledFrame{3}, UnlabeledFrame{3}}\n") == [:R20]
        @test rules("src/a.jl" => "const Ext = CameraExtrinsics{Float64, RotMatrix{3, Float64, 9}, Vec3{Float64}}\n") == [:R20]
        # a parametric alias with free variables, a bounded or `where` form, a partial application and a type of
        # another package stay
        @test quiet("src/a.jl" => "struct Pt{T}\n    x::T\nend\nconst PtLine{T} = Pt{T}\nconst PtAny = Pt{<:Real}\n" *
                    "const PtW = Pt{T} where {T}\nconst Partial = VgtkCore.EuclideanMap{3, Float64}\nconst E = Float64\n")
        # a UnionAll application (an alias whose body keeps free variables) and an abstract type are no concrete type
        # a foreign type a stack module merely re-exports is not a stack type (ColorTypes' `Gray` through Core)
        isdefined(VisualGeometryToolkitCore, :Gray) && @test quiet("src/a.jl" => "const PX = Gray{N0f8}\n")
        @test quiet("src/a.jl" => "const P = PlainEuclideanMap{3, Float64}\nconst Q = VgtkCore.PlainEuclideanMap{2, Float64}\n" *
                    "abstract type Abs{T} end\nconst AbsF = Abs{Float64}\nconst AM = VgtkCore.AbstractMatrixMap\n")
        # at run time, a computed alias (`typeof(…)`) of a concrete stack type; a value, a type of another package and a
        # type's own name stay
        M = Module(:R20Probe)
        Core.eval(M, :(import VisualGeometryToolkitCore as VgtkCore))
        Core.eval(M, :(const Bad = typeof(VgtkCore.Lens(VgtkCore.Equidistant()))))
        Core.eval(M, :(const Value = VgtkCore.Lens(VgtkCore.Equidistant())))
        Core.eval(M, :(const Foreign = typeof(1.0)))
        Core.eval(M, :(struct Own end))
        Core.eval(M, :(const Partial = VgtkCore.EuclideanMap{3, Float64}))
        found = NC._concrete_type_bindings(M)
        @test length(found) == 1 && startswith(found[1].text, "Bad = ") && found[1].rule === :R20
        # a concrete instantiation bound under the type's own name is one too (it is not the type itself)
        M2 = Module(:R20Own)
        Core.eval(M2, :(import VisualGeometryToolkitCore as VgtkCore))
        Core.eval(M2, :(const Lens = typeof(VgtkCore.Lens(VgtkCore.Equidistant()))))
        own = NC._concrete_type_bindings(M2)
        @test length(own) == 1 && startswith(own[1].text, "Lens = ")
        # a deprecated binding is never read (an error under `--depwarn=error`, the release gate): the binding walk of
        # both the algorithm-name scan and the run-time check skips it
        M3 = Module(:R20Deprecated)
        Core.eval(M3, :(import VisualGeometryToolkitCore as VgtkCore))
        Core.eval(M3, :(const New = typeof(VgtkCore.Lens(VgtkCore.Equidistant()))))
        Core.eval(M3, :(Base.@deprecate_binding Old New))
        @test Base.isdeprecated(M3, :Old)
        @test :Old ∉ first.(collect(NC._bindings(M3))) && :New in first.(collect(NC._bindings(M3)))
        @test all(f -> !startswith(f.text, "Old"), NC._concrete_type_bindings(M3))
    end

    @testset "R9: a call form ending in `(` matches with its argument" begin
        call = "ran" * "sac("
        @test call in NC.RETIRED_NAMES
        @test rules("docs/src/a.md" => "Run `$(call)problem, scoring)`.\n") == [:R9]
        @test rules("src/a.jl" => "x = $(call)p, s)\n") == [:R9]
        # the word itself, the algorithm and an internal step stay
        @test quiet("docs/src/a.md" => "The ran" * "sac loop; `Ransac(; scoring)`; `RansacConfig`.\n")
        @test quiet("src/a.jl" => "x = _$(call)p, s)\n")
    end

    @testset "R21: a FitResult by field, its contract quantities by function (parsed code)" begin
        # the retired accessors, called on a result
        for f in ("estimate", "iterations", "converged", "retcode", "metadata", "stop_rule")
            @test rules("src/a.jl" => "g(r) = $(f)(r)\n") == [:R21]
            @test rules("src/a.jl" => "g(r) = VgtkCore.$(f)(r)\n") == [:R21]
        end
        @test :R21 in rules("src/a.jl" => "g(T) = nested" * "_metadata(T)\n")
        @test rules("src/a.jl" => "rms(r::FitResult) = r.rms\n") == [:R21]
        @test rules("src/a.jl" => "estimate(r::VgtkCore.FitResult) = r.u\n") == [:R21]
        # the contract fields of the metadata, read outside their contract method
        for c in ("weights", "dof", "scale", "scale_source", "s2", "inlier_mask")
            @test rules("src/a.jl" => "g(r) = r.metadata.$(c)\n") == [:R21]
        end
        # … and inside it: the one place the field is read
        @test quiet("src/a.jl" => "weights(r::FitResult) = r.metadata.weights\ndof(r::FitResult) = r.metadata.dof\n")
        @test quiet("src/a.jl" => "function VgtkCore.noise_scale(r::MyResult)\n    return r.metadata.scale\nend\n")
        # not a retired accessor: fields, locals, other arities, other functions, comments and strings
        @test quiet("src/a.jl" => "g(r) = (r.retcode, r.stop_rule, r.metadata, r.stats.nsteps, r.rms, r.metadata.depths)\n")
        @test quiet("src/a.jl" => "struct R\n    retcode::Int\n    estimate::Float64\nend\n")
        @test quiet("src/a.jl" => "g(estimate, ch, c) = estimate(ch, c)\nh(; estimate = nothing) = estimate\n")
        @test quiet("src/a.jl" => "g(c, m) = stop_rule(c, m)\nh(e) = rms(e)\nk(r) = successful_retcode(r)\n")
        @test quiet("src/a.jl" => "# estimate(r) and r.metadata.weights in a comment\ng() = \"estimate(r)\"\n")
        @test quiet("src/a.jl" => "g(sol) = (sol.iterations, sol.converged)\n")
    end

    @testset "R9: a foreign format's token is accepted in a string literal and a data fixture, never as a name" begin
        tok = first(NC.FOREIGN_FORMAT_TOKENS[findfirst(t -> last(t).accepted_in === :format_literal, NC.FOREIGN_FORMAT_TOKENS)])   # a retired name a stored format spells
        @test tok in NC.RETIRED_NAMES
        @test quiet("src/a.jl" => "is_old(d) = d[\"type\"] == \"$(tok)\"\n")                      # the token a reader parses
        @test quiet("test/fixtures/old.yaml" => "type : $(tok)\n", "test/fixtures/old.json" => "{\"type\": \"$(tok)\"}\n")
        @test rules("src/a.jl" => "f(x::$(tok)) = x\n") == [:R9]                                     # as an identifier
        @test rules("src/a.jl" => "f(x) = x   # a $(tok)\n") == [:R9]                                # in a comment
        @test rules("src/a.jl" => "f(x) = (\"$(tok)\", $(tok))\n") == [:R9]                          # a string beside a name
        @test rules("docs/a.md" => "the $(tok) record\n") == [:R9]                                     # in prose
        @test rules("src/a.jl" => "\"\"\"\n    f(x)\n\nthe $(tok) record\n\"\"\"\nf(x) = x\n") == [:R9]   # in a docstring
        @test rules("test/a_tests.jl" => "@testset \"the $(tok) record\" begin end\n") == [:R9]       # in a test's name
        @test rules("src/a.jl" => "f(x) = \"$(old)\"\n") == [:R9]                                    # any other retired name, in a string
    end

    @testset "R9: a published name (a format token) is accepted in a quoted TOML value and an Artifacts.toml, never a TOML key" begin
        tok = "eth3d_splg_cs" * "ponds"                                                       # a published artifact's name (a format token)
        @test tok in NC.RETIRED_NAMES && NC._is_format_token(tok)
        @test quiet("configs/a.toml" => "input_dir = \"/data/splg/$(tok)\"\n")                     # a quoted value
        @test quiet("configs/a.toml" => "dirs = [\n  \"/data/$(tok)\",\n]\n")                    # an array element on its line
        @test quiet("Artifacts.toml" => "[$(tok)]\ngit-tree-sha1 = \"0\"\n")                       # an Artifacts.toml: its keys are published names
        @test :R9 in rules("configs/a.toml" => "$(tok) = 1\n")                                    # a bare TOML key
        @test :R9 in rules("configs/a.toml" => "\"$(tok)\" = 1\n")                                # a quoted TOML key
        @test :R9 in rules("configs/a.toml" => "[artifacts.$(tok)]\n")                             # a table header
        @test rules("configs/a.toml" => "x = 1   # $(tok)\n") == [:R9]                              # a TOML comment
    end

    @testset "R14: the terminology taxonomy's retired words are no binding; \"measured\" a value only" begin
        w = ("measure" * "ment", "cs" * "pond", "sight" * "ing", "key" * "point")
        for word in w
            @test :R14 in rules("src/a.jl" => "f_$(word)_x(y) = y\n")                              # a function
            @test :R14 in rules("src/a.jl" => "f(x) = (z = :$(word)s; z)\n")                       # a quoted symbol
            @test :R14 in rules("configs/a.toml" => "$(word)_count = 1\n")                          # a TOML key
        end
        @test :R14 in rules("src/a.jl" => "Key" * "Point(x) = x\n") && :R14 in rules("src/a.jl" => "key" * "_point_xy(x) = x\n")   # split across words
        m = "measure" * "d"
        @test :R14 in rules("src/a.jl" => "f(x) = ($(m)_us = x; $(m)_us)\n")                       # a bound name
        @test quiet("src/a.jl" => "f(x) = x === :$(m) ? \"$(m)\" : nothing\n")                    # a provenance value
        @test quiet("src/a.jl" => "f(x) = x   # the $(m) values\n")                                 # prose
    end

    @testset "R9: a foreign library's function is accepted in a C/C++ source, nowhere else" begin
        name = first(NC.FOREIGN_FORMAT_TOKENS[findfirst(t -> last(t).accepted_in === :c_source, NC.FOREIGN_FORMAT_TOKENS)])
        @test all(t -> last(t).accepted_in in (:format_literal, :c_source, :cli_argument), NC.FOREIGN_FORMAT_TOKENS)
        @test all(e -> e in NC._NAMING_EXTENSIONS, NC._C_SOURCE_EXTENSIONS)                      # every C source is scanned
        @test name in NC.RETIRED_NAMES
        @test quiet("wrapper/w.cpp" => "    auto s = poselib::$(name)(p, P, &image, opt);   // every $(name) overload\n", "wrapper/w.hpp" => "int $(name)();\n")
        @test rules("src/a.jl" => "f(x) = \"$(name)\"\n") == [:R9]                                   # a Julia file
        @test rules("docs/a.md" => "the $(name) call\n") == [:R9]                                      # prose
        @test rules("test/fixtures/a.yaml" => "f: $(name)\n") == [:R9]                              # a data fixture: not a format token
    end

    @testset "R9: a foreign program's command-line value is accepted after its flag in a shell script and a TOML string, nowhere else" begin
        val = first(NC.FOREIGN_FORMAT_TOKENS[findfirst(t -> last(t).accepted_in === :cli_argument, NC.FOREIGN_FORMAT_TOKENS)])
        @test val in NC.RETIRED_NAMES
        @test quiet(
            "drivers/run.sh" => "J=\$(sbatch -- bash -c \"\$SCH --images \$D --model $(val) --cells 150 > \$K/run.log\")\n",
            "drivers/eq.sh" => "\$SCH --model=$(val)\n",
            "configs/a.toml" => "provenance = \"the detections (camera_calibration --model $(val) --cells 18)\"\n"
        )
        @test rules("drivers/run.sh" => "MODEL=$(val)\n") == [:R9]                        # a shell value, not a flag's
        @test rules("drivers/run.sh" => "x=1   # --model $(val)\n") == [:R9]            # in a comment
        @test rules("drivers/run.sh" => "\$SCH --model $(val)   # the $(val) model\n") == [:R9]   # a second hit, in prose
        @test rules("drivers/run.sbatch" => "\$SCH --model $(val)\n") == [:R9]          # only a `.sh` script
        @test rules("src/a.jl" => "f(x) = \"--model $(val)\"\n") == [:R9]              # a Julia string
        @test rules("src/a.jl" => "$(val)(x) = x\n") == [:R9]                            # a Julia identifier
        @test rules("configs/a.toml" => "$(val) = 1\n") == [:R9]                         # a TOML key
        @test rules("configs/a.toml" => "model = \"$(val)\"\n") == [:R9]               # a TOML value, not a command line
        @test rules("configs/a.toml" => "cmd = \"run\"   # --model $(val)\n") == [:R9]    # a TOML comment
        @test rules("docs/a.md" => "run with `--model $(val)`\n") == [:R9]               # prose
    end

    @testset "R3: update_ ends in !" begin
        @test rules("src/a.jl" => "update_prior(x) = x\n") == [:R3]
        @test rules("src/a.jl" => "update_model(x) = x\n") == [:R3, :R8]   # also a hook variant
        @test rules("src/a.jl" => "_update_scale(x) = x\n") == [:R3]
        @test quiet("src/a.jl" => "update_model!(x) = x\n_update_scale!(x) = x\n")
    end

    @testset "R4: next_ is pure" begin
        @test rules("src/a.jl" => "next_scale!(x) = x\n") == [:R4]
        @test quiet("src/a.jl" => "next_scale(x) = x\n_next_prior(x) = x\n")
    end

    @testset "R5: no identifier holds `step`, but Manopt's step_solver! and SciML's nsteps" begin
        @test rules("src/a.jl" => "lo_step(x) = x\n") == [:R5]
        @test rules("src/a.jl" => "function f(x)\n    step = 2x\n    return step\nend\n") == [:R5]
        @test rules("src/a.jl" => "g(x; steps = 3) = x\n") == [:R5]
        @test rules("src/a.jl" => "struct NewtonStep end\n") == [:R5]
        @test rules("src/a.jl" => "struct S\n    step::Int\nend\n") == [:R5]
        # no `fit_*` anchors a `<fit>_step` any longer: an algorithm's model update is its `update_model!`
        @test rules(
            "test/fit.jl" => "fit_lo(x) = x\nfit_widget_camera(x) = x\n",
            "src/a.jl" => "lo_step(x) = x\n_division_camera_step!(x) = x\n"
        ) == [:R5]
        @test quiet("src/a.jl" => "update_model!(s, m, p, θ) = θ\nstep_solver!(x) = x\n")
        # SciMLBase's counter, in every binding
        @test quiet("src/a.jl" => "struct SolveStats\n    nsteps::Int\nend\nf(; nsteps = 0) = (n = nsteps; k = :nsteps; n)\n")
        # a local named like a fit update is still a local
        @test rules("test/fit.jl" => "fit_lo(x) = x\n", "src/a.jl" => "lo_step = 1\n") == [:R5]
        # a bound name holds the word even when it is joined to another: only
        # the exact `<fit>_step` form is a fit update
        @test rules("src/a.jl" => "lostep = 1\n") == [:R5]
        @test rules("src/a.jl" => "const MAXSTEPS = 3\n") == [:R5]
        @test rules("src/a.jl" => "g(x; stepsize = 1) = x\n") == [:R5]
        @test rules("src/a.jl" => "struct Footsteps end\n") == [:R5]
        @test rules("test/fit.jl" => "fit_lo(x) = x\n", "src/a.jl" => "lostep(x) = x\n") == [:R5]
        @test rules("test/fit.jl" => "fit_lo(x) = x\n", "src/a.jl" => "lo_steps(x) = x\n") == [:R5]
        # prose is not a binding: the word is not banned there
        @test quiet("src/a.jl" => "x = 1   # a stepsize; a footstep\n", "docs/a.md" => "A stepsize.\n")
        # Manopt.jl's name of the driver's one cycle is the one other function
        # that holds the word — the function only, never a local
        @test :R5 ∉ rules("src/a.jl" => "VisualGeometryToolkitCore.step_solver!(s, l, m, p, θ, i) = θ\n")   # (R22 refuses the reach)
        @test rules("src/a.jl" => "step_solver = 1\n") == [:R5]
    end

    @testset "R6: an increment is <method>_increment" begin
        @test rules("src/a.jl" => "newton_delta_step(x) = x\n") == [:R6]
        @test quiet("src/a.jl" => "_newton_increment(x) = x\n_lm_increment!(x) = x\n")
    end

    @testset "R7: the banned words, in names and in prose" begin
        @test rules("src/a.jl" => "n_$(P)ed = 0\n") == [:R7]
        @test rules("src/a.jl" => "f(x; $(P)es = 1) = x\n") == [:R7]
        @test rules("src/a.jl" => "struct Foo$(Cap)Map end\n") == [:R7]
        @test rules("src/a.jl" => "struct S\n    $(P)ed::Bool\nend\n") == [:R7]
        @test rules("src/a.jl" => "s = :$(P)ed\n") == [:R7]
        @test rules("src/a.jl" => "x = 1   # the second $P\n") == [:R7]
        @test rules("src/a.jl" => "\"\"\"\n    f(x)\n\nA $(P)-through map.\n\"\"\"\nf(x) = x\n") == [:R7]
        @test rules("docs/a.md" => "Tests $(P) on CI.\n") == [:R7]
        @test rules("docs/a.md" => "a multi-$P fit\n") == [:R7]
        @test rules("docs/a.md" => "the $(P)through map\n") == [:R7]
        @test rules("Project.toml" => "# $(Cap)ED\nname = \"A\"\n") == [:R7]
        # in prose, other words that contain the letters are other words
        @test quiet(
            "src/a.jl" => "cache = true   # bypass the compass; a password\n",
            "docs/a.md" => "Bypass the compass with a password.\n",
            "scripts/run.sh" => "# bypass the compass\n"
        )
        # in a bound name the letters are matched as a substring, with no exceptions
        @test rules("src/a.jl" => "bypass_cache = true\n") == [:R7]
        @test rules("src/a.jl" => "struct BypassCache end\n") == [:R7]
        @test rules("src/a.jl" => "f(x; password = \"\") = x\n") == [:R7]
        @test rules("src/a.jl" => "struct S\n    compass::Int\nend\n") == [:R7]
        # a comment after a character literal holding a quote is still prose
        @test rules("src/a.jl" => "c = '\"'   # a $P\n") == [:R7]
    end

    @testset "R7: a random draw is a trial, a deterministic cycle an iteration" begin
        RS, PS, IT, TR = uppercase(NC._RS), uppercase(NC._PS), NC._IT, NC._TR
        # prose: every retired phrase, in any file, case-insensitive
        for p in NC.RETIRED_PHRASES
            @test rules("docs/a.md" => "One $(join(p, " ")) here.\n") == [:R7]
            @test rules("src/a.jl" => "x = 1   # one $(uppercase(join(p, "-")))\n") == [:R7]
        end
        @test rules("src/a.jl" => "\"\"\"\nEach $RS $IT draws a sample.\n\"\"\"\nf(x) = x\n") == [:R7]
        @test rules("Project.toml" => "# the $PS $(IT)s\nname = \"A\"\n") == [:R7]
        # the finding names the phrase and its line
        v = only(findings("docs/a.md" => "x\nfour hypotheses per $IT\n"))
        @test (v.line, v.rule) == (2, :R7) && startswith(v.text, "hypotheses per $IT: ")
        # the words must be adjacent; each alone, or beside other words, is prose
        @test quiet(
            "docs/a.md" => "A $RS $TR; the $RS loop's $IT count; an $IT budget of the LM; a sampler's $IT.\n",
            "src/a.jl" => "x = 1   # the LM's candidate step, one per $IT; $(TR)s of the $RS\n"
        )
        # bound names: a min/max bound spelled with iterations, or a RANSAC/PROSAC
        # word beside an iteration word, in Julia, Python, shell and TOML
        @test rules("src/a.jl" => "f(x; max_$(IT)s = 3) = x\n") == [:R7]
        @test rules("src/a.jl" => "struct C\n    min_$(IT)s::Int\nend\n") == [:R7]
        @test rules("src/a.jl" => "n_$(NC._RS)_iters = 0\n") == [:R7]
        @test rules("src/a.jl" => "struct $(uppercasefirst(NC._PS))$(uppercasefirst(IT)) end\n") == [:R7]
        @test rules("src/a.jl" => "k = :$(NC._RS)_iter\n") == [:R7]
        @test rules("scripts/a.py" => "def f(max_$(IT)s=5):\n    return 1\n") == [:R7]
        @test rules("scripts/run.sh" => "MAX_$(uppercase(IT))S=5\n") == [:R7]
        @test rules("config/a.toml" => "[$(NC._RS)]\nmin_$(IT)s = 100\n") == [:R7]
        # the vocabulary's own names, a deterministic loop's stop reason (a
        # value), and uses of names owned elsewhere stand
        @test quiet(
            "src/a.jl" => """
            f(x; max_iter = 5, iterations = 2, max_trials = 10, min_trials = 1, n_trials = 0) = x
            rc = :max_$(IT)s
            y = opts.max_$(IT)s
            g(; max_$(IT)s = 3)
            """,
            "config/a.toml" => "[$(NC._RS)]\nmax_trials = 100\nmin_trials = 10\n"
        )
    end

    @testset "R8: the hooks of BlockCoordinateDescent, exactly" begin
        @test rules("src/a.jl" => "post_iteration(state, m, p, θ, i) = nothing\n") == [:R8]
        @test rules("src/a.jl" => "VisualGeometryToolkitCore.stop_reason!(c, s, m, p, θ, θo, i) = :continue\n") == [:R8]
        @test rules("src/a.jl" => "VisualGeometryToolkitCore.stop_solver(s, c, m, p, θ, θo, i) = :continue\n") == [:R8]
        @test quiet(
            "src/a.jl" => """
            import VisualGeometryToolkitCore: init_state!, update_model!
            init_state!(m::Int, p, θ) = (; r = zeros(3))
            VisualGeometryToolkitCore.post_iteration!(s, m::Int, p, θ, i) = nothing
            VisualGeometryToolkitCore.stopping_criterion(m::Int) = VisualGeometryToolkitCore.StopNever()
            VisualGeometryToolkitCore.stop_reason(c::Int, s, m, p, θ, θo, i) = :continue
            """
        )
    end

    @testset "R9: retired names, anywhere" begin
        @test rules("src/a.jl" => "x = 1   # formerly $old\n") == [:R9]
        @test rules("src/a.jl" => "y = $old(1)\n") == [:R9]
        @test rules("docs/a.md" => "See `$old`.\n") == [:R9]
        @test quiet("docs/a.md" => "See `$(old)_extra` and `pre$old`.\n")
        # every listed name trips, in code and in prose, wherever it is listed
        for r in NC.RETIRED_NAMES
            @test :R9 in rules("docs/a.md" => "See `$r`.\n")
            @test :R9 in rules("src/a.jl" => "x = 1   # formerly $r\n")
        end
        # the pool-frame successors, and names that merely contain a retired one, stand
        @test quiet(
            "docs/a.md" => "`pool_frame_maps`, `pool_normalization`, `pool_frame_calibrations`, " *
                "`POOL_FRAME_R_MAX`, `pool_frame_maps_unknown`, `UNKNOWN_POOL_FRAME_MAP`, " *
                "`welsch_half_weight_depth`, " *
                "`grid`, `extend`, `extension_offset`, `EfficiencyConstantWindow`, `product_tuning_val_refits`, " *
                "`selection_axis`, `tuning_axis`, `width_parameterization`, `OPERATING_EFFICIENCY_CONSTANT`, " *
                "`stream_product_tuning_cpu`, `budget_units`, `_save_artifact`, `ARTIFACT_FORMATS`, " *
                "`load_tune_config`, `tune_tables`, `cv_polish_efficiency_constant`, `run_sb_fair_deployed`, " *
                "`SB_PRODUCT_TUNING`, `PS_CV`, `audit_product_tuning`, `merge_product_tuning_sidecars`, " *
                "`source_map`, `destination_map`, `_heb_pool_frame_maps`, `pool_frame_sidecar`, " *
                "`<scene>.pool_frame.jld2`, `install_flushing_logger!`.\n"
        )
        # the codified cross-validation's successors stand
        @test quiet(
            "docs/a.md" => "`CrossValidation`, `RegisteredSceneSplit`, `StratifiedSampling`, " *
                "`validation_units`, `test_units`, `sample_units`, " *
                "`cap_per_scene`, `fraction`, `BudgetSampling`, `budget_units`, `budget_rng`, " *
                "`resampling_rng`, `stable_seed`, `registered_split`, `scene_errors`, " *
                "`max_pairs_per_scene`, `paper_e2e_per_scene.tsv`.\n"
        )
        # the paper constants: each retired name trips on its own, and the
        # config-driven successors stand
        @test :R9 in rules("src/a.jl" => "p = " * "GT_KIT" * "TI_ROMA\n")
        @test :R9 in rules("src/a.jl" => "s = " * "SEVEN_SCENES_TRAIN" * "_SPLITS\n")
        @test quiet("docs/a.md" => "`registered_split`, `GT_PATHS`, `WMG_NGX`.\n")
        # the robot-world/hand-eye names stand; the bare robot-world ones trip
        @test :R9 in rules("src/a.jl" => "x = " * "fit_robot" * "_world(m)\n")
        @test quiet(
            "docs/a.md" => "`RobotWorldHandEyeProblem`, " *
                "`RobotWorldHandEyeCalibration`, `RobotWorldHandEyeMetadata`, `HandEyeProblem`.\n"
        )
        # a rig's target member: the retired type trips; its successor, its
        # accessors and the stereo pair's destination stand
        @test :R9 in rules("src/a.jl" => "m = " * "Mount" * "ed(t, X)\n")
        @test quiet("docs/a.md" => "`PosedTarget`, `model(m)`, `pose(m)`, `destination(rig)`, a mounted board.\n")
        # the one placement trait: the retired name trips, its successor stands
        @test :R9 in rules("src/a.jl" => "p = " * "has" * "_extrinsics(c)\n")
        @test quiet("docs/a.md" => "`is_posed(member)`, `extrinsics(m)`.\n")
        # a normalization's sides: the retired one trips, the successors stand
        @test :R9 in rules("src/a.jl" => "D = " * "target" * "_map(N)\n")
        @test quiet("docs/a.md" => "`source_map(N)`, `destination_map(N)`, `inv(destination_map(N)) ∘ M ∘ source_map(N)`.\n")
        # the dissolved evaluation package's pool names trip; their successors,
        # and names that merely contain a retired one, stand
        @test :R9 in rules("src/a.jl" => "x = " * "generate" * "_pool(n, snn)\n")
        @test :R9 in rules("docs/a.md" => "See `" * "AbstractRaw" * "Pair`.\n")
        @test quiet(
            "docs/a.md" => "`PoolPair`, `StereoRawPair`, `write_scene_pool`, " *
                "`generate_pools_x`, `EssentialProblem(pair)`, `pool_frame_maps`, `pool_pixel_scale`.\n"
        )
        # a retired file suffix is matched literally: its `.` is no wildcard
        sfx = "canon" * ".jld2"
        @test rules("scripts/run.sh" => "sidecar=\"\$scene.$sfx\"\n") == [:R9]
        @test quiet("docs/a.md" => "See `$(replace(sfx, "." => "_"))` and `$(replace(sfx, "." => "x"))`.\n")
    end

    @testset "R10: a width quantity is bound by its full name" begin
        # the lists, from the checker (this file spells none of them as a binding)
        ab = "k" * "_op"                     # one multi-letter abbreviation
        @test ab in NC._WIDTH_ABBREVIATIONS
        T_WORD = "ta" * "u"
        abd = replace(first(filter(x -> occursin('_', x), sort!(collect(NC._WIDTH_ABBREVIATIONS)))), "_" => "-")
        L = "k"                              # a letter: a width only where the binding says so
        flag(x) = "--" * x
        # an abbreviation: every binding kind, quoted symbols, dict keys, flags
        @test rules("src/a.jl" => "$ab = 1\n") == [:R10]
        @test rules("src/a.jl" => "f(x, $ab) = x\n") == [:R10]
        @test rules("src/a.jl" => "f(x; $ab = 1) = x\n") == [:R10]
        @test rules("src/a.jl" => "struct S\n    $ab::Float64\nend\n") == [:R10]
        @test rules("src/a.jl" => "s = :$ab\n") == [:R10]
        @test rules("src/a.jl" => "d[\"$ab\"] = 1\n") == [:R10]
        @test rules("src/a.jl" => "x = get(d, :$ab, 0)\n") == [:R10]
        @test rules("src/a.jl" => "d = Dict(\"$ab\" => 1)\n") == [:R10]
        @test rules("scripts/a.py" => "$ab = 1\n") == [:R10]
        @test rules("config/a.toml" => "[tuning]\n$ab = 2.5\n") == [:R10]
        # the word `tau` in any bound name
        @test rules("src/a.jl" => "struct $(uppercasefirst(T_WORD))Window end\n") == [:R10]
        @test rules("src/a.jl" => "polish_$(T_WORD)(x) = x\n") == [:R10]
        # a CLI flag, in any file, `_` or `-` between the words
        @test rules("scripts/run.sh" => "julia run.jl $(flag(L)) 2.5\n") == [:R10]
        @test rules("docs/a.md" => "Run with `$(flag(abd)) 3`.\n") == [:R10]
        @test rules("src/a.jl" => "s = \"$(flag(L))\"\n") == [:R10]
        # a letter: a TOML key
        @test rules("config/a.toml" => "[tuning]\n$L = 2.5\n") == [:R10]
        # a letter bound beside a noise scale (signature, struct, NamedTuple)
        @test rules("src/a.jl" => "w(shape, $L, ŝ) = $L * ŝ\n") == [:R10]
        @test rules("src/a.jl" => "deploy(shape, s2; $L = 1.0) = s2\n") == [:R10]
        @test rules("src/a.jl" => "t = ($L = 1.0, s2 = 2.0)\n") == [:R10]
        @test rules("src/a.jl" => "t = (; a = 1.0, s2 = 2.0)\n") == [:R10]
        @test rules("src/a.jl" => "struct D\n    c::Float64\n    s2::Float64\nend\n") == [:R10]
        # a letter that is a field, type parameter or constructor keyword of a
        # width-carrying type, resolved through the package's declarations
        @test rules("src/a.jl" => "struct HQ <: AbstractOptimizationStrategy\n    $L::Float64\nend\n") == [:R10]
        @test rules(
            "src/t.jl" => "abstract type MyLO <: VisualGeometryToolkitCore.AbstractOptimizationStrategy end\n",
            "src/a.jl" => "struct X{T} <: MyLO\n    a::T\nend\n"
        ) == [:R10]
        @test rules("src/a.jl" => "struct Sel <: SampleSelector\n    τ::Float64\nend\n") == [:R10]
        @test rules(
            "src/a.jl" => "struct HQ <: AbstractOptimizationStrategy\n    efficiency_constant::Float64\nend\nHQ(; c = 1.0) = HQ(c)\n"
        ) == [:R10]
        # must NOT flag: loop indices, math locals, positional letters with no
        # noise scale, a letter that is another quantity's field or keyword or a
        # lens coefficient's dict key, quoted letters, and the full names
        @test quiet(
            "src/a.jl" => """
            function f(a, b, c)
                $L = 0
                for $L in 1:3
                    a += $L
                end
                τ = a / b              # an aspect ratio
                return a + c + τ
            end
            struct RadialHomography{T}
                H::Matrix{T}
                $L::T                    # a division coefficient
            end
            struct Ellipse
                a::Float64
                b::Float64
            end
            ellipse(; a = 20.0, b = 10.0) = Ellipse(a, b)
            lens = Dict("$L" => 1.0e-7, "a" => 2)
            v = lens["$L"]
            s = :$L
            deploy(shape, s2; efficiency_constant = 1.0) = (efficiency_constant = efficiency_constant, s2 = s2)
            struct HQ <: AbstractOptimizationStrategy
                efficiency_constant::Float64
                loss_width::Float64
                outlier_halfwidth::Float64
            end
            taught = 1                 # a word that merely starts with the letters
            """,
            "config/a.toml" => "[tuning]\nefficiency_constant = 2.5\nloss_width = 1.0\noutlier_halfwidth = 3.0\n",
            "scripts/run.sh" => "julia run.jl --efficiency-constant 2.5 --outlier-halfwidth 3 --kernel-free\n",
            "docs/a.md" => "A $L-NN graph, and $(T_WORD) in prose, are prose.\n"
        )
    end

    @testset "R9: the scorer's decision point is decision_boundary (the old names retired)" begin
        TH = "thres" * "hold"
        for old in ("inlier_$(TH)", "has_inlier_$(TH)", "decision_$(TH)")
            @test rules("src/a.jl" => "w = $(old)(sc, buf)\n") == [:R9]
        end
        @test isempty(rules("src/a.jl" => "w = decision_boundary(sc, buf); has_decision_boundary(typeof(sc))\n"))
        # RansacLib's and SupeRANSAC's settings field, set by name in a string
        @test isempty(rules("src/a.jl" => "setproperty!(c, Symbol(\"inlier_$(TH)\"), w)\n"))
    end

    @testset "R13: the one parameter-holding keyword is `fixed`" begin
        FR = "fr" * "ee"     # the spellings, built so this file is no finding of its own
        for kw in (FR, "fix", "fixed_params", "hold")
            @test rules("test/a.jl" => "r = refine_board(cam, P, v; $(kw) = (:k1,))\n") == [:R13]   # a call
            @test rules("test/a.jl" => "fit_x(a; $(kw) = ()) = a\n") == [:R13]                                 # a signature
            @test rules("configs/a.toml" => "[refine_board]\n$(kw) = [\"k1\"]\n") == [:R13]       # a refine table's key
        end
        @test rules("test/a.jl" => "VgtkEstimation.refine_x(a; $(FR) = ())\n") == [:R13]                 # qualified
        @test rules("configs/a.toml" => "[pipeline.lm]\nfix = [\"cx\"]\n") == [:R13]
        @test rules("src/a.jl" => "s = LensSpec(:division; coefficients = 2, $(FR) = (:d1,))\n") == [:R13]       # a LensSpec
        @test rules("configs/a.toml" => "[camera]\ntarget = { lens = \"division\", coefficients = 2, $(FR) = [\"d1\"] }\n") == [:R13]
        @test isempty(rules("src/a.jl" => "LensSpec(family::Symbol; fixed = (), $(FR) = RetiredKeyword(), kw...) = nothing\n"))
        # the keyword itself, its refusal slot and the test of the refusal are no finding
        @test isempty(rules("test/a.jl" => "refine_x(a; fixed = (:k1,))\n"))
        @test isempty(rules("test/a.jl" => "function refine_x(a; fixed = (), $(FR) = RetiredKeyword()) end\n"))
        @test isempty(rules("test/a.jl" => "function refine_x(a; fixed = (), $(FR) = VgtkCore.RetiredKeyword()) end\n"))
        @test isempty(rules("test/a.jl" => "@test_throws ArgumentError refine_x(a; $(FR) = ())\n"))
        # an interpolated qualified callee (`@eval M.$f(…) = …`, a port's generated methods) names nothing: no finding, no error
        @test isempty(rules("src/a.jl" => "for f in (:lo_tag,)\n    @eval VgtkCore.\$f(::Int) = 1\nend\n"))
        @test isempty(rules("src/a.jl" => "@eval VgtkCore.\$(f)(x; $(FR) = ()) = x\n"))
        # the word elsewhere: comments (BlobBoards' "CUDA-free", "free memory", "for free"), a local, a field,
        # an accessor, another function's keyword, another TOML table
        @test isempty(rules("src/a.jl" => "f(x) = x   # CUDA-$(FR): the kernels allocate no $(FR) memory, for $(FR)\n"))
        @test isempty(rules("src/a.jl" => "$(FR) = device_$(FR)_bytes(dev); r.$(FR) == (); $(FR)_names(spec)\n"))
        @test isempty(rules("src/a.jl" => "struct M\n    $(FR)::Tuple\nend\nallocate(n; $(FR) = true)\n"))
        @test isempty(rules("configs/a.toml" => "[slurm]\n$(FR) = 4\nhold = true\n"))
        # a typed keyword with no default is the keyword too
        @test rules("test/a.jl" => "fit_x(a; $(FR)::Tuple) = a\n") == [:R13]
        @test rules("test/a.jl" => "function refine_x(a; fixed = (), $(FR)::Tuple) end\n") == [:R13]
        # a TOML finding is at its line in ITS table, not at the first such key of the file
        v = only(filter(v -> v.rule === :R13, findings("configs/a.toml" => "[slurm]\n$(FR) = 4\n\n[pipeline.lm]\n$(FR) = [\"cx\"]\n")))
        @test v.line == 5
    end

    @testset "R12: the Student-t + uniform scorer has no threshold" begin
        TH = "thres" * "hold"     # the word, spelled so this file is no finding of its own
        @test rules("src/a.jl" => "sc = StudentTUniformScoring(P; $(TH) = 3.0)\n") == [:R12]
        @test rules("src/a.jl" => "sc = StudentTUniformScoring(P, $(TH))\n") == [:R12]
        @test rules("configs/a.toml" => "[p]\nscoring = { type = \"student_t_uniform\", $(TH) = 3.0 }\n") == [:R12]
        @test rules("docs/a.md" => "the $(TH) of the StudentTUniformScoring\n") == [:R12]
        @test rules("src/a.jl" => "f(x) = x   # the student_t_uniform scorer's $(TH)\n") == [:R12]
        # the plural and any case
        @test rules("src/a.jl" => "sc = StudentTUniformScoring(P, $(TH)s)\n") == [:R12]
        @test rules("docs/a.md" => "$(uppercasefirst(TH)) of the StudentTUniformScoring\n") == [:R12]
        @test rules("docs/a.md" => "$(uppercase(TH))S beside StudentTUniformScoring\n") == [:R12]
        # its width by its name, its decision boundary, and a compound name holding the word are no finding
        @test isempty(rules("src/a.jl" => "sc = StudentTUniformScoring(P, outlier_halfwidth)\n"))
        @test isempty(rules("src/a.jl" => "w = decision_boundary(sc::StudentTUniformScoring, buf)\n"))
        @test isempty(rules("src/a.jl" => "w = probe_$(TH)_of(sc::StudentTUniformScoring, buf)\n"))
        # an MSAC width far from the Student-t's name is no finding
        @test isempty(rules("src/a.jl" => "a = MsacScoring(P; $(TH) = 3.0)\n" * "x = 1\n"^4 *
                                         "b = StudentTUniformScoring(P, outlier_halfwidth)\n"))
        # a changelog records what was renamed
        @test isempty(rules("CHANGELOG.md" => "the $(TH) of StudentTUniformScoring is outlier_halfwidth\n"))
    end

    @testset "R11: units in one place" begin
        @test rules("src/a.jl" => "u = uparse(s)\n") == [:R11]
        @test rules("src/a.jl" => "q = VisualGeometryToolkitCore.qparse(s)\n") == [:R11]
        @test rules("src/a.jl" => "ok = haskey(UnitRegistry.UNITS, k)\n") == [:R11]
        @test rules("src/a.jl" => "ok = haskey(FlexUnits.UnitRegistry.UNITS, k)\n") == [:R11]
        @test rules("src/a.jl" => "UnitRegistry.register_unit(\"furlong\" => u)\n") == [:R11]
        @test rules("src/a.jl" => "q2 = convert(Quantity{Float64, Dimensions{FixRat32}}, q)\n") == [:R11]
        @test rules("src/a.jl" => "y = x isa Quantity ? ustrip(px, x) : x\n") == [:R11]
        @test rules("src/a.jl" => "const T = (\"mm\", \"px\")\n") == [:R11]
        @test rules("src/a.jl" => "const D = Dict(\"mm\" => 1, \"inch\" => 2)\n") == [:R11]
        @test rules("src/a.jl" => "const C = ((\"mm\", 1), (\"mm/px\", 2))\n") == [:R11]
        @test rules("src/a.jl" => "f(x) = x * 25.4\n") == [:R11]
        @test rules("src/a.jl" => "f(x) = x * 0.3048\n") == [:R11]
        @test rules("src/a.jl" => "f(x) = x * (127 // 5000)\n") == [:R11]
        # a unit or a registry is built in the units module, from RegistryTools, and nowhere else
        @test rules("src/a.jl" => "const furlong = Units(\"furlong\" => 201.168 * m)\n") == [:R11]
        @test rules("src/a.jl" => "t = AffineTransform{Rational{Int}}(scale = 1 // 1000)\n") == [:R11]
        @test rules("test/a.jl" => "R = PermanentDict{Symbol, U}()\nregistry_defaults!(R)\n") == [:R11]
        @test rules("ext/a.jl" => "FlexUnits.RegistryTools.register_unit!(R, \"px\" => d)\n") == [:R11]
        # a stored quantity is base_quantity(q), never a bare ubase (which keeps 142mm's Rational)
        @test rules("src/a.jl" => "q = ubase(q)\n") == [:R11]
        # a pose or a target is built from quantities, never from stripped numbers
        one_r11(files::Pair...) = map(v -> v.rule, findings(files...)) == [:R11]
        @test one_r11("src/a.jl" => "X = EuclideanMap(R, ustrip.(mm, t))\n")
        @test one_r11("src/a.jl" => "X = EuclideanMap{3}(R, ustrip(m, t))\n")
        @test one_r11("src/a.jl" => "T = PlanarTarget(d, [ustrip.(mm, p) for p in ps], w)\n")
        @test one_r11("test/a.jl" => "T = PointTarget([VisualGeometryToolkitCore.ustrip.(m, p) for p in ps])\n")
        # a file length is converted by read_length and write_length, never by a scale
        @test one_r11("src/a.jl" => "function f(d)\n    u = d.length_unit\n    return d.t .* 0.001\nend\n")
        @test one_r11("src/a.jl" => "f(d) = d[\"length_unit\"] == \"m\" ? d[\"x\"] : d[\"x\"] / 1000\n")
        @test one_r11("src/a.jl" => "f(d, x) = haskey(d, :length_unit) ? 1.0e-3 * x : x\n")
        # must NOT flag: attaching a unit, stripping a whole pose, the IO pair,
        # a scale with no file unit in sight, a test rewriting a fixture
        @test quiet("src/a.jl" => "X = EuclideanMap(R, t .* m)\nY = ustrip(m, X)\nZ = PosedTarget(ustrip(m, tg), ustrip(m, X))\n")
        @test quiet("src/a.jl" => "function f(d)\n    u = parse_unit(d.length_unit)\n    return read_length(d.x, u), write_length(d.q, u)\nend\n")
        @test quiet("src/a.jl" => "g(t_ns) = t_ns / 1.0e9\nh(x) = 1000 * x\n")
        @test quiet("test/a.jl" => "function f(d)\n    d[\"length_unit\"] = \"m\"\n    return d[\"t\"] ./ 1000\nend\n")
        # must NOT flag: a test stating an expected value, prose, the units
        # module's own verbs, one unit name
        @test quiet("test/a.jl" => "@test ustrip(mm, 1.0inch) ≈ 25.4\n")
        @test quiet("src/a.jl" => "# uparse(\"mm\") is the registry's\nu = parse_unit(s)\nq = base_quantity(q)\nv = strip_or_bare(px, x)\n")
        @test quiet("test/a.jl" => "@test ubase(142mm) == 71 // 500 * m\n@test uscale(inch) === 127 // 5000\n")
        @test quiet("src/a.jl" => "d = Dict(\"value\" => 1.0, \"unit\" => \"px\")\n")
        # the units module itself is the rule's home
        @test quiet(
            "Project.toml" => "name = \"VisualGeometryToolkitCore\"\n",
            "src/base/units.jl" => "u = uparse(s)\nk = UnitRegistry.UNITS\nUnitRegistry.register_unit(\"px\" => u)\n",
        )
    end

    @testset "names owned outside the stack keep their spelling" begin
        # uses are not bindings
        @test quiet("src/a.jl" => "r = step(1:3)\nt = Test.$Cap\ns = x.step\n")
        # methods of an external function
        @test quiet("src/a.jl" => "Base.step(r::MyRange) = r.δ\n")
        @test quiet("src/a.jl" => "import Base: step\nstep(r::MyRange) = r.δ\n")
        # defining a stack binding with the token is flagged, qualified or not
        @test rules("src/a.jl" => "step(r::MyRange) = r.δ\n") == [:R5]
        @test rules("src/a.jl" => "VisualGeometryToolkitCore.$(P)_rate(x) = x\n") == [:R7]
        @test rules("src/a.jl" => "module M\nend\nM.lo_step(x) = x\n") == [:R5]
    end

    @testset "bound names in Python, shell and TOML, and quoted symbols" begin
        # Python: def/class names, parameters, assignment/annotation/for targets
        @test rules("scripts/a.py" => "def lo_step(x):\n    return x\n") == [:R5]
        @test rules("scripts/a.py" => "def update_prior(x):\n    return x\n") == [:R3]
        @test rules("scripts/a.py" => "def f(x, max_steps=3):\n    return x\n") == [:R5]
        @test rules("scripts/a.py" => "class LoStep:\n    ...\n") == [:R5]
        @test rules("scripts/a.py" => "zstep = 2\n") == [:R5]
        @test rules("scripts/a.py" => "a, (b, z_step) = 1, (2, 3)\n") == [:R5]
        @test rules("scripts/a.py" => "class A:\n    ztick_step: float = 1.0\n") == [:R5]
        @test rules("scripts/a.py" => "def f(self):\n    self.n_steps = 0\n") == [:R5]
        @test rules("scripts/a.py" => "for i, stepj in enumerate(x):\n    ...\n") == [:R5]
        @test rules("scripts/a.py" => "z = [k for kstep in x]\n") == [:R5]
        @test rules("scripts/a.py" => "n_$(P)ed = 0\n") == [:R7]
        # a multi-line signature reports the parameter's own line
        v = only(findings("scripts/a.py" => "def f(\n    x,\n    zstep=1,\n):\n    return x\n"))
        @test (v.line, v.text, v.rule) == (3, "zstep", :R5)
        # uses, call keywords, strings and comments bind nothing
        @test quiet(
            "scripts/a.py" => """
            x = np.arange(0, 1, step=0.1)
            d = dict(
                stepx=1,
            )
            obj.step_count = 1
            s = 'step_size = 1'   # a step
            def f():
                ...
            """
        )
        # Python's empty-block keyword is the language's, not a binding: only
        # the text scan sees it
        v = only(findings("scripts/a.py" => "def f():\n    $P\n"))
        @test (v.line, v.rule) == (2, :R7) && v.text == P
        # shell and sbatch variables, loop variables and functions
        @test rules("scripts/run.sh" => "MAX_STEPS=3\n") == [:R5]
        @test rules("scripts/run.sh" => "export ZSTEP=2\n") == [:R5]
        @test rules("scripts/run.sh" => "for stepv in 1 2; do echo; done\n") == [:R5]
        @test rules("scripts/run.sh" => "if true; then X_STEP=1; fi\n") == [:R5]
        @test rules("scripts/run.sh" => "run_step() {\n  echo\n}\n") == [:R5]
        @test rules("scripts/job.sbatch" => "#SBATCH -N 1\nlocal update_rate=1\n") == [:R3]
        @test quiet(
            "scripts/run.sh" => "# MAX_STEPS=3\necho \"A_STEP=1\"\nrclone config create x smb user=u\n",
            "scripts/job.sbatch" => "#SBATCH --job-name=step\nX=1\n"
        )
        # TOML keys; a dependency table's keys are other projects' names
        @test rules("config/a.toml" => "[plot]\nzstep = 2\n") == [:R5]
        v = only(findings("config/a.toml" => "name = \"A\"\n\n[plot]\nztick_step = 2\n"))
        @test (v.line, v.text) == (4, "ztick_step")
        @test quiet("Project.toml" => "[deps]\nStepFunctions = \"0\"\n\n[compat]\nStepFunctions = \"1\"\n")
        @test rules("config/a.toml" => "x = \n") == [:parse]
        # quoted Julia symbols: a name the stack spells as a value
        @test rules("src/a.jl" => "k = :zstep\n") == [:R5]
        @test rules("src/a.jl" => "k = :update_prior\n") == [:R3]
        @test rules("test/a.jl" => "fit_lo(x) = x\nk = :lo_step\n") == [:R5]
    end

    @testset "scope" begin
        write_tree(dir, files) = for (f, src) in files
            mkpath(dirname(joinpath(dir, f)))
            write(joinpath(dir, f), src)
        end
        vendored(path; kind = "vendored") =
            "[[$kind]]\npath = \"$path\"\nupstream = \"https://example.org/up\"\nlicense = \"BSD-3-Clause\"\n"
        babbage = "BSD 3-Clause License\n\nCopyright (c) 2019, Charles Babbage\n"
        @test quiet(
            "papers/draft.tex" => "$P\n",
            "papers/refs.bib" => "$P\n",
            "docs/build/index.md" => "$P\n",
            "Manifest.toml" => "# $P\n",
            "notes.txt" => "$P\n"
        )
        # process records: plans, and the release tooling's records at the root
        @test quiet(
            "docs/superpowers/plans/p.md" => "the $P\n",
            ".release/docs-$P.md" => "docs-$P: 0123abc\nthe $P\n",
            ".release/sub/notes.md" => "the $P\n"
        )
        # only the top-level `.release/` is a record
        @test rules("docs/.release/a.md" => "the $P\n") == [:R7]
        @test rules("src/.release/a.jl" => "x = 1   # the $P\n") == [:R7]
        # the code, scripts and docs beside a paper are stack text
        @test rules("papers/demo/run.jl" => "x = 1   # the second $P\n") == [:R7]
        @test rules("paper/README.md" => "a multi-$P fit\n") == [:R7]
        @test rules("papers/demo/job.sbatch" => "# the $P\n") == [:R7]
        # provenance is declared, not guessed: a `vendor` directory is stack text
        @test rules("vendor/lib/x.jl" => "lo_step(x) = x\n") == [:R5]
        # a declared vendored tree is excluded, whole
        mktempdir() do dir
            write_tree(dir, (
                "src/a.jl" => "\n",
                "deps/Up/LICENSE" => babbage,
                "deps/Up/c.h" => "// the $P\n",
                "deps/Up/sub/x.jl" => "lo_step(x) = x\n",
                "VENDORED.toml" => vendored("deps/Up"),
            ))
            s = naming_scope(dir)
            @test s.files == ["VENDORED.toml", "src/a.jl"]
            @test first.(s.excluded) == ["deps/Up"]
            @test occursin("https://example.org/up", last(only(s.excluded)))
            @test isempty(check_naming(dir))
        end
        # a declared path that does not exist, and a malformed entry, are errors
        mktempdir() do dir
            write_tree(dir, ("src/a.jl" => "\n", "VENDORED.toml" => vendored("deps/Gone")))
            @test_throws ArgumentError naming_scope(dir)
            @test_throws ArgumentError check_naming(dir)
            write(joinpath(dir, "VENDORED.toml"), "[[vendored]]\npath = \"src\"\n")
            @test_throws ArgumentError naming_scope(dir)          # lacks upstream, license
            write(joinpath(dir, "VENDORED.toml"), vendored("src/a.jl"; kind = "template"))
            @test_throws ArgumentError naming_scope(dir)          # a template names a license file
        end
        # an undeclared tree whose license names another holder is an error
        mktempdir() do dir
            write_tree(dir, ("src/a.jl" => "\n", "deps/Up/LICENSE" => babbage, "deps/Up/c.h" => "\n"))
            err = try
                naming_scope(dir)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("deps/Up", err.msg) && occursin("Charles Babbage", err.msg)
        end
        # a template's license beside stack text: declared, and its siblings stay in scope
        mktempdir() do dir
            files = (
                "LICENSE" => "MIT License\n\nCopyright (c) 2025-2026 Ada Lovelace\n\nThe above copyright notice …\n",
                "src/a.jl" => "\n",
                "slides/LICENSE" => "MIT License\n\nCopyright (c) 2026 Paolo Favaro\n",
                "slides/plot.py" => "# the $P\n",
                "tools/LICENSE" => "MIT License\n\nCopyright (c) 2026 Ada Lovelace\n",   # the stack's own
                "tools/run.sh" => "\n",
            )
            write_tree(dir, files)
            @test_throws ArgumentError naming_scope(dir)
            write(joinpath(dir, "VENDORED.toml"), vendored("slides/LICENSE"; kind = "template"))
            s = naming_scope(dir)
            @test s.files == ["VENDORED.toml", "slides/plot.py", "src/a.jl", "tools/run.sh"]
            @test isempty(s.excluded)
            @test [v.file for v in check_naming(dir)] == ["slides/plot.py"]
        end
        # with no root license, the holder is the author of the first commit
        git = Sys.which("git")
        if git === nothing
            @test_skip "git is not available"
        else
            mktempdir() do dir
                write_tree(dir, (
                    "src/a.jl" => "\n",
                    "slides/LICENSE" => "Copyright (c) 2026 Grace Hopper\n",
                    "slides/plot.py" => "# the $P\n",
                    "template/LICENSE" => "MIT License\n\nCopyright (c) 2023 eccv24\n",
                    "template/make.py" => "# the $P\n",
                ))
                id = `-c user.name="Grace Hopper" -c user.email=gh@example.org -c commit.gpgsign=false`
                run(pipeline(`$git -C $dir init -q`; stdout = devnull, stderr = devnull))
                run(`$git -C $dir add -A`)
                run(pipeline(`$git -C $dir $id commit -q -m init`; stdout = devnull, stderr = devnull))
                err = try
                    naming_scope(dir)
                    nothing
                catch e
                    e
                end
                @test err isa ArgumentError
                @test occursin("template", err.msg) && occursin("eccv24", err.msg) && occursin("Grace Hopper", err.msg)
                @test !occursin("slides", err.msg)
                write(joinpath(dir, "VENDORED.toml"), vendored("template/LICENSE"; kind = "template"))
                run(`$git -C $dir add -A`)
                s = naming_scope(dir)
                @test s.files == ["VENDORED.toml", "slides/plot.py", "src/a.jl", "template/make.py"]
            end
        end
        @test rules("src/bad.jl" => "function f(\n") == [:parse]
    end

    v = only(findings("src/a.jl" => "\n\nupdate_prior(x) = x\n"))
    @test v isa NamingViolation
    @test (v.file, v.line, v.text, v.rule) == ("src/a.jl", 3, "update_prior", :R3)
    @test occursin("update_prior", sprint(show, v))
end

@testitem "quality: no duplication" begin
    # One implementation per concept (VisualGeometryToolkitCore's check_duplication): names
    # an upstream stack package owns, restated function bodies, copied files and retired
    # patterns; the other stack repositories in VGTK_STACK_PATHS are compared too.
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using VisualGeometryToolkitCore
    VgtkCore.test_duplication(VisualGeometryToolkitCore)
end

# R22 (owner, 2026-11-26): outside Core, user-level code solves through the front end. A fixture package, R22Fixture,
# whose sources reach Core's internals in each way the rule refuses, beside the forms it allows. The internal names are
# assembled from fragments, so this file (which the naming and duplication checkers also read) spells none of them.
@testitem "quality: R22, user-level code solves through the front end" begin
    using Test
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: check_naming
    lk = "_lmsolve" * "_chart!"; lp = "_Lm" * "Problem"; bcd = "_block_coordinate" * "_descent!"; old = "lmsolve" * "_chart!"
    function r22(files::Pair...)
        mktempdir() do dir
            write(joinpath(dir, "Project.toml"), "name = \"R22Fixture\"\n")
            for (rel, src) in files
                mkpath(dirname(joinpath(dir, rel))); write(joinpath(dir, rel), src)
            end
            [(v.file, v.rule) for v in check_naming(dir) if v.rule in (:R22, :R9)]
        end
    end
    imp = "import VisualGeometryToolkitCore as VgtkCore\n"
    # flagged: a `_` reach (qualified, imported, through an alias), a driver, a hook called by user-level code, the old name
    @test r22("src/a.jl" => imp * "f(x) = VgtkCore.$(lk)(x)\n") == [("src/a.jl", :R22)]
    @test r22("src/a.jl" => "using VisualGeometryToolkitCore: $(lp)\n") == [("src/a.jl", :R22)]
    @test r22("src/a.jl" => "import VisualGeometryToolkitCore as V\ng(s) = V.$(bcd)(s)\n") == [("src/a.jl", :R22)]
    @test r22("src/a.jl" => imp * "h(s) = VgtkCore.step_solver!(s)\n") == [("src/a.jl", :R22)]
    @test r22("test/user_tests.jl" => imp * "r = VgtkCore.locally_optimize!(L, lo, p, m, cs, buf, sc)\n") == [("test/user_tests.jl", :R22)]
    @test r22("experiments/run.jl" => "s = score!(buf, sc, p, m)\n") == [("experiments/run.jl", :R22)]
    @test r22("docs/src/guide.md" => "# Guide\n\n```julia\nscore!(buf, sc, p, m)\n```\n") == [("docs/src/guide.md", :R22)]
    @test (("src/a.jl", :R9) in r22("src/a.jl" => "x = $(old)(p)\n"))
    # not flagged: the front end; an implementation and a forward; a marked contract test; a contract guide; a paper
    @test isempty(r22("src/a.jl" => "r = solve(RefineProblem(f, u0), LevenbergMarquardt())\n"))
    @test isempty(r22("src/a.jl" => imp * "VgtkCore.locally_optimize!(L, lo::MyLO, p, m, cs, buf, sc) = VgtkCore.locally_optimize!(L, lo.inner, p, m, cs, buf, sc)\n"))
    @test isempty(r22("experiments/lo.jl" => imp * "VgtkCore.score!(ws, sc::MySc, args...) = VgtkCore.score!(ws, sc.inner, args...)\n"))
    @test isempty(r22("test/contract_tests.jl" => "# R22: contract test — the scoring contract\ns = score!(buf, sc, p, m)\n"))
    @test isempty(r22("test/helper_tests.jl" => "test_scoring_contract(sc, p)\ns = score!(buf, sc, p, m)\n"))
    @test isempty(r22("docs/src/extending.md" => "<!-- R22: contract guide -->\n```julia\nscore!(buf, sc, p, m)\n```\n"))
    @test isempty(r22("accv2026/src/x.jl" => imp * "f(x) = VgtkCore.$(lk)(x)\n"))
    @test isempty(r22("src/a.jl" => "y = project(cam, X)\n"))   # an accessor is no hook
    # Core itself: no R22 finding, and the internal drivers are neither public nor exported
    @test !any(v -> v.rule === :R22, check_naming(pkgdir(VgtkCore)))
    for g in VgtkCore.CORE_INTERNAL_DRIVERS
        @test !Base.ispublic(VgtkCore, nameof(g)) && !(nameof(g) in names(VgtkCore))
    end
    for n in Symbol.((lk, lp, bcd, "_chart" * "_refit", "_triggs" * "_refit", "_ManifoldLm" * "Workspace", "_grid_brent" * "_minimum",
                      "_cgls", "_normal" * "_cg"))
        @test isdefined(VgtkCore, n) && !Base.ispublic(VgtkCore, n)
    end
    @test all(g -> g isa Function, VgtkCore.SOLVER_HOOKS) && VgtkCore.score! in VgtkCore.SOLVER_HOOKS
end
