# quality_duplication_tests.jl — VisualGeometryToolkitCore's `check_duplication`:
# each of its checks finds a planted duplicate, and leaves a tree without one
# alone. (Every package's own tree is checked in its quality item.)
#
# The retired patterns the scan must find are assembled from fragments, so this
# file spells none of them.

@testitem "quality: check_duplication finds planted duplicates" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: check_duplication, DuplicationViolation
    ext = Base.get_extension(VisualGeometryToolkitCore, :VisualGeometryToolkitCoreTestExt)

    # A throwaway repository: a `.git` marker at its root, a package in it.
    function repo(files::Pair...; name = "Consumer")
        dir = mktempdir()
        mkdir(joinpath(dir, ".git"))
        write(joinpath(dir, "Project.toml"), "name = \"$name\"\n")
        for (rel, src) in files
            mkpath(dirname(joinpath(dir, rel)))
            write(joinpath(dir, rel), src)
        end
        return dir
    end
    checks(found) = sort!(unique(getfield.(found, :check)))

    # one algorithm, two names: the same body under other identifiers
    body(f, a, b) = """
        function $f($a, $b; tol = 1.0e-9)
            acc = zero(eltype($a))
            for i in eachindex($a, $b)
                d = $a[i] - $b[i]
                abs(d) > tol || continue
                acc += d^2 / (1 + abs($b[i]))
            end
            return sqrt(acc / max(length($a), 1))
        end
        """
    other = """
        function spread(xs)
            m = sum(xs) / length(xs)
            return [x - m for x in xs if isfinite(x)], maximum(abs, xs; init = 0.0)
        end
        """
    copied = repo("src/a.jl" => body("misfit", "x", "y"), "src/b.jl" => "# a comment of its own\n" * body("residual_norm", "u", "v"))
    v = only(check_duplication(copied; stack = String[]))
    @test (v.check, v.file, v.line) == (:similar, "src/a.jl", 1)
    @test occursin("misfit", v.text) && occursin("src/b.jl:2 residual_norm", v.other)
    @test occursin("keep one", sprint(show, v))
    # distinct bodies, and one body defined once, are left alone
    @test isempty(check_duplication(repo("src/a.jl" => body("misfit", "x", "y"), "src/b.jl" => other); stack = String[]))
    # the thresholds are keywords: a lower floor of tokens or Jaccard finds more, a higher one less
    @test isempty(check_duplication(copied; stack = String[], similarity = ((0.6, 500),)))

    # a whole file copied (its comments differ): a file finding as well as the body's
    filecopy = repo("test/a.jl" => "# one\n" * body("misfit", "x", "y"), "test/b.jl" => "# two\n" * body("misfit", "x", "y"))
    @test checks(check_duplication(filecopy; stack = String[])) == [:file, :similar]

    # against another stack repository: named by its project, absent ones skipped
    upstream = repo("src/m.jl" => body("misfit", "x", "y"); name = "Upstream")
    mine = repo("src/n.jl" => body("my_misfit", "p", "q"))
    v = only(check_duplication(mine; stack = [upstream]))
    @test v.check == :similar && startswith(v.other, "Upstream:src/m.jl:1")
    @test isempty(@test_logs (:info, r"absent") check_duplication(mine; stack = [joinpath(upstream, "gone")]))

    # the retired patterns, each outside its owner, and not in a comment or docstring
    nl = "NLLS" * "Options("; ac = "a" * "cos(clamp((tr("; sr = "0.00" * "31308"; nc = "\\\\" * "new" * "command"
    uses(src; name = "Consumer", file = "src/a.jl") =
        [v.text[2:(findfirst(']', v.text) - 1)] for v in check_duplication(repo(file => src; name); stack = String[])]
    @test uses("o = NLLSsolver.$(nl)maxiters = 3)\n") == ["nlls_options"]
    @test uses("θ = $(ac)R) - 1) / 2, -1, 1))\n") == ["rotation_angle"]
    @test uses("srgb(x) = x <= $sr ? 12.92x : 1.055x^(1 / 2.4) - 0.055\n") == ["srgb_transfer"]
    @test uses("println(io, \"$nc{\\\\\", k, \"}{\", v, \"}\")\n") == ["tex_macro_writer"]
    @test uses("print(\"$nc{x}{1}\")\n"; file = "scripts/n.py") == ["tex_macro_writer"]
    @test isempty(uses("# $(nl)) was retired\n\"\"\"\n    f()\n\nNot $ac here.\n\"\"\"\nf() = 1\n"))
    @test isempty(uses("o = NLLSsolver.$(nl)maxiters = 3)\n"; name = "VisualGeometryToolkitCore", file = "src/robust/lm_core.jl"))

    # exceptions: an entry, with its reason, keeps a finding out; one that matches nothing is a finding
    toml(entries) = "DUPLICATION.toml" => entries
    sim = """
        [[similar]]
        functions = ["src/a.jl::misfit", "src/b.jl::residual_norm"]
        reason = "a planted pair"
        """
    pat = """
        [[pattern]]
        use = "rotation_angle"
        file = "src/c.jl"
        reason = "a planted pattern"
        """
    excused = repo("src/a.jl" => body("misfit", "x", "y"), "src/b.jl" => body("residual_norm", "u", "v"),
        "src/c.jl" => "θ = $(ac)R) - 1) / 2, -1, 1))\n", toml(sim * pat))
    @test isempty(check_duplication(excused; stack = String[]))
    stale = repo("src/a.jl" => body("misfit", "x", "y"), toml(sim))
    v = only(check_duplication(stale; stack = String[]))
    @test v.check == :stale_exception && v.file == "DUPLICATION.toml"
    # an entry naming another repository that was not compared is not stale (CI has no stack)
    elsewhere = replace(sim, "src/b.jl" => "Upstream:src/m.jl")
    @test isempty(check_duplication(repo("src/a.jl" => body("misfit", "x", "y"), toml(elsewhere)); stack = String[]))
    @test_throws ArgumentError check_duplication(repo(toml("[[similar]]\nfunctions = [\"src/a.jl::f\", \"src/b.jl::g\"]\n")); stack = String[])
    @test_throws ArgumentError check_duplication(repo(toml("[[copies]]\nreason = \"x\"\n")); stack = String[])

    # inside a monorepo: the package's own files are checked against its siblings', labelled from the root
    mono = repo("Project.toml" => "[workspace]\nprojects = [\"PkgA\", \"PkgB\"]\n",
        "PkgA/Project.toml" => "name = \"PkgA\"\n", "PkgA/src/a.jl" => body("misfit", "x", "y"),
        "PkgB/Project.toml" => "name = \"PkgB\"\n", "PkgB/test/b.jl" => body("residual_norm", "u", "v"))
    v = only(check_duplication(joinpath(mono, "PkgA"); stack = String[]))
    @test (v.file, startswith(v.other, "PkgB/test/b.jl:1")) == ("PkgA/src/a.jl", true)

    # name ownership: a binding the module owns whose name an upstream package owns (private names too);
    # a method added to the upstream function, and an imported name, are not the module's
    planted = Module(:Planted)
    Core.eval(planted, quote
        _NAMING_RULE_TEXT = 1                                         # Core owns this private name
        using VisualGeometryToolkitCore: VisualGeometryToolkitCore, naming_scope
        VisualGeometryToolkitCore.check_naming(::Val{:planted}) = 1  # a method extension
        own_helper(x) = x
    end)
    found = ext._name_violations(planted, [VisualGeometryToolkitCore], "", Dict{String, Any}[])
    @test [v.text for v in found] == ["_NAMING_RULE_TEXT"]
    @test only(found).other == "owned by VisualGeometryToolkitCore"
    excuse = Dict{String, Any}("module" => "Planted", "name" => "_NAMING_RULE_TEXT", "reason" => "planted")
    @test isempty(ext._name_violations(planted, [VisualGeometryToolkitCore], "", [excuse]))
    stale_name = Dict{String, Any}("module" => "Planted", "name" => "own_helper2", "reason" => "planted")
    @test checks(ext._name_violations(planted, [VisualGeometryToolkitCore], "", [excuse, stale_name])) == [:stale_exception]
    # Core is the bottom of the stack: nothing is upstream of it
    @test isempty(ext._stack_upstream(VisualGeometryToolkitCore))
end
