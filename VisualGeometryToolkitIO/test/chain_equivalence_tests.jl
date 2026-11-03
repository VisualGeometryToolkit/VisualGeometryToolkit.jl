# =============================================================================
# The chain-equivalence gate, VisualGeometryToolkitIO's half: the records and
# values written by the code at hand against the goldens written on the code
# before CoordinateTransformations' `∘` replaced `CalibrationChain` and the lens
# `Chain` (VisualGeometryToolkitCore's test/data/chain_equivalence/).
# =============================================================================
#
# The goldens are never regenerated. What the code may change of them is
# stated in the ADAPTER below and nowhere else; a record or value that moves
# any other way fails here.

@testsetup module ChainEquivalenceIO
    import VisualGeometryToolkitCore
    import VisualGeometryToolkitIO
    const DIR = joinpath(pkgdir(VisualGeometryToolkitCore), "test", "data", "chain_equivalence")
    include(joinpath(DIR, "corpus.jl"))
    using .ChainEquivalenceCorpus
    include(joinpath(DIR, "io_values.jl"))
    using VisualGeometryToolkitCore: CalibrationChain, Chain, Affine

    export DIR, CODE, read_goldens, io_values, compare_value, target_cases

    # THE ADAPTER — the code at hand (corpus.jl's header), plus how a golden
    # record becomes the code's (`migrate(golden, out)`) and the content ids
    # the code changes on purpose (`new_content_ids`, name => id). Every field
    # is the code the goldens were written on: the comparison is the identity,
    # which proves the harness.
    const CODE = (
        affine = Affine,
        compose_calibrations = members -> CalibrationChain(members),
        compose_distortions = stages -> Chain(stages...),
        ct_order = false,
        retired = Set{String}(),
        migrate = (golden, out) -> cp(golden, out; force = true),
        new_content_ids = Dict{String, String}(),
    )
end

@testitem "chain equivalence: IO's records and values reproduce the goldens" setup = [ChainEquivalenceIO] begin
    P = read_goldens(joinpath(DIR, "corpus_params.txt"))
    G = read_goldens(joinpath(DIR, "io_values.txt"))
    golden_records = joinpath(DIR, "records")
    retired_file(rel) = any(n -> rel in ("target_$n.json", "target_$n.uncertain.json", "rig_$n.json"), CODE.retired) ||
        (!isempty(CODE.retired) && rel == "rig_every.json")
    mktempdir() do dir
        written, migrated = joinpath(dir, "written"), joinpath(dir, "migrated")
        actual = Dict{String, Any}(io_values(P, CODE, written))
        # the values
        for (key, g) in G
            name = split(key, '/')[2]
            if startswith(key, "content_id/") && (name in CODE.retired || (name == "every" && !isempty(CODE.retired)))
                @test !haskey(actual, key)
                continue
            end
            @test haskey(actual, key)
            haskey(actual, key) || continue
            e = startswith(key, "content_id/") ? get(CODE.new_content_ids, name, g) : g
            ok, _, why = compare_value(e, actual[key])
            ok || @error "chain equivalence: $key moved: $why"
            @test ok
        end
        @test all(k -> haskey(G, k), keys(actual))
        # the records: every golden file, migrated to the code's version, is
        # byte for byte what the code writes; a retired case's is refused
        n_files = 0
        for (root, _, files) in walkdir(golden_records), f in files
            rel = relpath(joinpath(root, f), golden_records)
            out = joinpath(migrated, rel)
            mkpath(dirname(out))
            if retired_file(rel)
                @test_throws ArgumentError CODE.migrate(joinpath(root, f), out)
                @test !isfile(joinpath(written, rel))
                continue
            end
            CODE.migrate(joinpath(root, f), out)
            same = isfile(joinpath(written, rel)) && read(out) == read(joinpath(written, rel))
            same || @error "chain equivalence: the record $rel moved"
            @test same
            n_files += 1
        end
        n_written = sum(length(fs) for (_, _, fs) in walkdir(written))
        @test n_files == n_written
        @test n_files >= 50
    end
end
