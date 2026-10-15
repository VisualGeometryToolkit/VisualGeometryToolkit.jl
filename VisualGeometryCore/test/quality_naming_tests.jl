# quality_naming_tests.jl — the stack's iterative-estimator vocabulary
# (conventions/naming.md), checked by VisualGeometryCore's
# `check_naming`: this package against itself, and the checker against small
# synthetic packages, one rule at a time.
#
# The synthetic sources that must trip the banned-word and retired-name scans
# are assembled from the checker's own fragments (`_STEM`, `RETIRED_NAMES`),
# so this file, which the checker also reads, spells neither list.

@testitem "quality: naming" begin
    using Test
    using VisualGeometryCore
    VisualGeometryCore.test_naming(VisualGeometryCore)
end

@testitem "quality: naming checker" begin
    using Test
    using VisualGeometryCore
    using VisualGeometryCore: check_naming, NamingViolation, naming_scope

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

    NC = Base.get_extension(VisualGeometryCore, :VisualGeometryCoreTestExt)   # the checker (loads with Test)
    P = NC._STEM                                 # the banned stem, from the checker
    Cap = uppercasefirst(P)
    old = NC.RETIRED_NAMES[1]                    # any retired name

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

    @testset "R5: step is <fit>_step of an existing fit_<fit>" begin
        @test rules("src/a.jl" => "lo_step(x) = x\n") == [:R5]
        @test rules("src/a.jl" => "function f(x)\n    step = 2x\n    return step\nend\n") == [:R5]
        @test rules("src/a.jl" => "g(x; steps = 3) = x\n") == [:R5]
        @test rules("src/a.jl" => "struct NewtonStep end\n") == [:R5]
        @test rules("src/a.jl" => "struct S\n    step::Int\nend\n") == [:R5]
        # the fit exists (in another file of the package): its model update is named
        @test quiet(
            "src/fit.jl" => "fit_lo(x) = x\nfit_division_camera(x) = x\n",
            "src/a.jl" => "lo_step(x) = x\n_division_camera_step!(x) = x\n"
        )
        # a local named like a fit update is still a local
        @test rules("src/fit.jl" => "fit_lo(x) = x\n", "src/a.jl" => "lo_step = 1\n") == [:R5]
        # a bound name holds the word even when it is joined to another: only
        # the exact `<fit>_step` form is a fit update
        @test rules("src/a.jl" => "lostep = 1\n") == [:R5]
        @test rules("src/a.jl" => "const MAXSTEPS = 3\n") == [:R5]
        @test rules("src/a.jl" => "g(x; stepsize = 1) = x\n") == [:R5]
        @test rules("src/a.jl" => "struct Footsteps end\n") == [:R5]
        @test rules("src/fit.jl" => "fit_lo(x) = x\n", "src/a.jl" => "lostep(x) = x\n") == [:R5]
        @test rules("src/fit.jl" => "fit_lo(x) = x\n", "src/a.jl" => "lo_steps(x) = x\n") == [:R5]
        # prose is not a binding: the word is not banned there
        @test quiet("src/a.jl" => "x = 1   # a stepsize; a footstep\n", "docs/a.md" => "A stepsize.\n")
        # Manopt.jl's name of the driver's one cycle is the one other function
        # that holds the word — the function only, never a local
        @test quiet("src/a.jl" => "VisualGeometryCore.step_solver!(s, l, m, p, θ, i) = θ\n")
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

    @testset "R8: the hooks of block_coordinate_descent!, exactly" begin
        @test rules("src/a.jl" => "post_iteration(state, m, p, θ, i) = nothing\n") == [:R8]
        @test rules("src/a.jl" => "VisualGeometryCore.stop_reason!(c, s, m, p, θ, θo, i) = :continue\n") == [:R8]
        @test rules("src/a.jl" => "VisualGeometryCore.stop_solver(s, c, m, p, θ, θo, i) = :continue\n") == [:R8]
        @test quiet(
            "src/a.jl" => """
            import VisualGeometryCore: init_state!, update_model!
            init_state!(m::Int, p, θ) = (; r = zeros(3))
            VisualGeometryCore.post_iteration!(s, m::Int, p, θ, i) = nothing
            VisualGeometryCore.stopping_criterion(m::Int) = VisualGeometryCore.StopNever()
            VisualGeometryCore.stop_reason(c::Int, s, m, p, θ, θo, i) = :continue
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
                "`grid`, `extend`, `extension_offset`, `ScaleMultipleWindow`, `product_tuning_val_refits`, " *
                "`selection_axis`, `tuning_axis`, `width_parameterization`, `OPERATING_SCALE_MULTIPLE`, " *
                "`stream_product_tuning_cpu`, `budget_units`, `_save_artifact`, `ARTIFACT_FORMATS`, " *
                "`load_tune_config`, `tune_tables`, `cv_polish_scale_multiple`, `run_sb_fair_deployed`, " *
                "`SB_PRODUCT_TUNING`, `PS_CV`, `audit_product_tuning`, `merge_product_tuning_sidecars`, " *
                "`source_map`, `target_map`, `_heb_pool_frame_maps`, `pool_frame_sidecar`, " *
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
            "docs/a.md" => "`fit_robot_world_hand_eye`, `refine_robot_world_hand_eye`, " *
                "`RobotWorldHandEyeCalibration`, `RobotWorldHandEyeMetadata`, `find_hand_eye`.\n"
        )
        # the dissolved evaluation package's pool names trip; their successors,
        # and names that merely contain a retired one, stand
        @test :R9 in rules("src/a.jl" => "x = " * "generate" * "_pool(n, snn)\n")
        @test :R9 in rules("docs/a.md" => "See `" * "AbstractRaw" * "Pair`.\n")
        @test quiet(
            "docs/a.md" => "`draw_hypotheses`, `PoolPair`, `StereoRawPair`, `write_scene_pool`, " *
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
        @test rules("src/a.jl" => "struct HQ <: AbstractLocalOptimization\n    $L::Float64\nend\n") == [:R10]
        @test rules(
            "src/t.jl" => "abstract type MyLO <: VisualGeometryCore.AbstractLocalOptimization end\n",
            "src/a.jl" => "struct X{T} <: MyLO\n    a::T\nend\n"
        ) == [:R10]
        @test rules("src/a.jl" => "struct Sel <: SampleSelector\n    τ::Float64\nend\n") == [:R10]
        @test rules(
            "src/a.jl" => "struct HQ <: AbstractLocalOptimization\n    scale_multiple::Float64\nend\nHQ(; c = 1.0) = HQ(c)\n"
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
            deploy(shape, s2; scale_multiple = 1.0) = (scale_multiple = scale_multiple, s2 = s2)
            struct HQ <: AbstractLocalOptimization
                scale_multiple::Float64
                loss_width::Float64
                outlier_halfwidth::Float64
            end
            taught = 1                 # a word that merely starts with the letters
            """,
            "config/a.toml" => "[tuning]\nscale_multiple = 2.5\nloss_width = 1.0\noutlier_halfwidth = 3.0\n",
            "scripts/run.sh" => "julia run.jl --scale-multiple 2.5 --outlier-halfwidth 3 --kernel-free\n",
            "docs/a.md" => "A $L-NN graph, and $(T_WORD) in prose, are prose.\n"
        )
    end

    @testset "R11: units in one place" begin
        @test rules("src/a.jl" => "u = uparse(s)\n") == [:R11]
        @test rules("src/a.jl" => "q = VisualGeometryCore.qparse(s)\n") == [:R11]
        @test rules("src/a.jl" => "ok = haskey(PixelUnits.UNITS, k)\n") == [:R11]
        @test rules("src/a.jl" => "q2 = convert(VgcQuantity{Float64}, q)\n") == [:R11]
        @test rules("src/a.jl" => "y = x isa Quantity ? ustrip(px, x) : x\n") == [:R11]
        @test rules("src/a.jl" => "const T = (\"mm\", \"px\")\n") == [:R11]
        @test rules("src/a.jl" => "const D = Dict(\"mm\" => 1, \"inch\" => 2)\n") == [:R11]
        @test rules("src/a.jl" => "const C = ((\"mm\", 1), (\"mm/px\", 2))\n") == [:R11]
        @test rules("src/a.jl" => "f(x) = x * 25.4\n") == [:R11]
        # must NOT flag: a test stating an expected value, prose, the units
        # module's own verbs, one unit name
        @test quiet("test/a.jl" => "@test ustrip(mm, 1.0inch) ≈ 25.4\n")
        @test quiet("src/a.jl" => "# uparse(\"mm\") is the registry's\nu = parse_unit(s)\nq = ubase(q)\nv = strip_or_bare(px, x)\n")
        @test quiet("src/a.jl" => "d = Dict(\"value\" => 1.0, \"unit\" => \"px\")\n")
        # the units module itself is the rule's home
        @test quiet(
            "Project.toml" => "name = \"VisualGeometryCore\"\n",
            "src/base/units.jl" => "u = uparse(s)\nk = PixelUnits.UNITS\n",
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
        @test rules("src/a.jl" => "VisualGeometryCore.$(P)_rate(x) = x\n") == [:R7]
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
        @test quiet("src/a.jl" => "fit_lo(x) = x\nk = :lo_step\n")
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
