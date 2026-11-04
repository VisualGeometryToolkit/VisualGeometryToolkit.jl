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
    using CoordinateTransformations: AffineMap
    using StaticArrays: SMatrix, SVector

    export DIR, CODE, read_goldens, io_values, compare_value, target_cases

    # THE ADAPTER — the code at hand (corpus.jl's header), plus how a golden
    # record becomes the code's (`migrate(golden, out)`) and the content ids
    # the code changes on purpose (`new_content_ids`, name => id):
    # - targets (compose with CoordinateTransformations, revision 2): the
    #   affine kind is CT's `AffineMap`, a composition CT's `∘` with its
    #   members and covariance in CT's order (`ct_order`); the in-plane
    #   compositions are retired. Target and rig records are v3/v6: a golden
    #   v2/v5 file is migrated by `vgio_migrate` and must be the writer's v3/v6
    #   output byte for byte; a retired case's is refused. Every content id is
    #   the golden's: the identity tree lists a composition's members in
    #   application order whatever order the record lists them in, so no id
    #   moves (`new_content_ids` is empty);
    # - lenses: a distortion's stages in sequence are CT's `∘` (the Lens
    #   `Chain` removed); the camera store is v4, its lens record's "chain"
    #   listing the stages outer first: a golden v3 store is migrated by
    #   `vgio_migrate` and must be the writer's v4 output byte for byte. The
    #   OpenCV and COLMAP exports are unchanged.
    const CODE = (
        affine = (A, b) -> AffineMap(SMatrix{2, 2}(A), SVector{2}(b)),
        compose_calibrations = members -> ∘(reverse(members)...),
        compose_distortions = stages -> ∘(reverse(stages)...),
        ct_order = true,
        lens_ct_order = true,
        retired = Set([
            "R_H", "R_A", "R_A_H", "R_T", "R_T_H", "A_R", "A_R_H", "A_T", "A_T_H", "T_R", "T_R_H", "T_A", "T_A_H",
            "R_A_T", "R_A_T_H", "R_T_A", "R_T_A_H", "A_R_T", "A_R_T_H", "A_T_R", "A_T_R_H", "T_R_A", "T_R_A_H", "T_A_R", "T_A_R_H",
        ]),
        migrate = (golden, out) -> occursin(r"^(target|rig|cameras)_", basename(golden)) ?
            VisualGeometryToolkitIO.vgio_migrate(golden; out) : cp(golden, out; force = true),
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
        n_files = n_refused = 0
        for (root, _, files) in walkdir(golden_records), f in files
            rel = relpath(joinpath(root, f), golden_records)
            out = joinpath(migrated, rel)
            mkpath(dirname(out))
            if retired_file(rel)
                @test_throws ArgumentError CODE.migrate(joinpath(root, f), out)
                @test !isfile(joinpath(written, rel))
                n_refused += 1
                continue
            end
            CODE.migrate(joinpath(root, f), out)
            same = isfile(joinpath(written, rel)) && read(out) == read(joinpath(written, rel))
            same || @error "chain equivalence: the record $rel moved"
            @test same
            n_files += 1
        end
        # a rig whose target is composed keeps its id across the v5 → v6
        # migration (persisted-geometry.md: an identity does not change when
        # the file format does); a rig whose id moves maps the golden id to the new one
        for name in ("A_H", "T_H")
            r = VisualGeometryToolkitIO.vgio_migrate(joinpath(golden_records, "rig_$name.json"); out = joinpath(dir, "rig_$name.ids.json"))
            @test r.old_id == r.new_id == G["content_id/$name"] == VisualGeometryToolkitIO.content_id(r.out)
        end
        for (name, id) in CODE.new_content_ids
            r = VisualGeometryToolkitIO.vgio_migrate(joinpath(golden_records, "rig_$name.json"); out = joinpath(dir, "rig_$name.ids.json"))
            @test r.old_id == G["content_id/$name"] && r.new_id == id
        end
        n_written = sum(length(fs) for (_, _, fs) in walkdir(written))
        @test n_files == n_written
        @test n_files + n_refused == sum(length(fs) for (_, _, fs) in walkdir(golden_records))
        @test n_files >= 30
    end
end
