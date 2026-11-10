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
    using StaticArrays: SMatrix, SVector

    export DIR, CODE, read_goldens, io_values, compare_value, target_cases

    # THE ADAPTER — the code at hand (corpus.jl's header), plus how a golden
    # record becomes the code's (`migrate(golden, out)`) and the content ids
    # the code changes on purpose (`new_content_ids`, name => id):
    # - targets (compose with CoordinateTransformations, revision 2; the
    #   gauge-free print error, 0.5.0): a composition is CT's `∘` with its
    #   members and covariance in CT's order (`ct_order`); the in-plane
    #   compositions are retired; the affine kind is retired (`affine =
    #   nothing`), and with it A and A_H — the code writes neither, and a golden
    #   affine record MIGRATES to its factors EuclideanMap ∘ Scale ∘ Aspect ∘
    #   Skew, the same map (`migrated_affine`: checked by value, and the rig's
    #   reported old id is the golden's); no kind has gauges (`gauges = false`);
    #   the thin-plate kinds are pure bending (`projected`): the code writes the
    #   projected values, so their records and content ids move (set aside), and
    #   a golden spline whose interpolant carries an affine term is REFUSED by
    #   the migration (no exact form: vgio_migrate's rule), A_H with them.
    #   Target and rig records are v3/v6: every other golden v2/v5 file is
    #   migrated by `vgio_migrate` and must be the writer's v3/v6 output byte for
    #   byte; a retired case's is refused. The content id of a rig whose value
    #   did not change is the golden's: the identity tree lists a composition's
    #   members in application order whatever order the record lists them in
    #   (`new_content_ids` is empty);
    # - lenses: a distortion's stages in sequence are CT's `∘` (the Lens
    #   `Chain` removed); the camera store is v4, its lens record's "compose"
    #   listing the stages outer first (the goldens' "chain" listed them in
    #   application order), and a composed calibration's record is
    #   `{"type": "composition", "compose": […]}`: a golden v3 store is migrated by
    #   `vgio_migrate` and must be the writer's v4 output byte for byte. The
    #   COLMAP export is unchanged; the OpenCV export declares its pixel frame
    #   under `pixel_frame` (0.9), where the goldens hold the legacy key: a
    #   golden export, read as it is (the reader accepts the legacy key, below),
    #   becomes the code's by that key alone.
    const CODE = (
        affine = nothing,
        gauges = false,
        projected = Set(["T", "H", "T_H"]),
        compose_calibrations = members -> ∘(reverse(members)...),
        compose_distortions = stages -> ∘(reverse(stages)...),
        composition_tags = false,
        ct_order = true,
        lens_ct_order = true,
        retired = Set([
            "A", "A_H", "R_H", "R_A", "R_A_H", "R_T", "R_T_H", "A_R", "A_R_H", "A_T", "A_T_H", "T_R", "T_R_H", "T_A", "T_A_H",
            "R_A_T", "R_A_T_H", "R_T_A", "R_T_A_H", "A_R_T", "A_R_T_H", "A_T_R", "A_T_R_H", "T_R_A", "T_R_A_H", "T_A_R", "T_A_R_H",
        ]),
        # retired cases whose golden records migrate (to the affine factors) rather than being refused
        migrated_affine = Set(["A"]),
        migrate = (golden, out) -> occursin(r"^(target|rig|cameras)_", basename(golden)) ?
            VisualGeometryToolkitIO.vgio_migrate(golden; out) :
            startswith(basename(golden), "opencv_") ? write(out, replace(read(golden, String), "\"image_origin\":" => "\"pixel_frame\":")) :
            cp(golden, out; force = true),
        new_content_ids = Dict{String, String}(),
    )
end

@testitem "chain equivalence: IO's records and values reproduce the goldens" setup = [ChainEquivalenceIO] begin
    import VisualGeometryToolkitCore as VgtkCore
    using StaticArrays: SMatrix, SVector
    P = read_goldens(joinpath(DIR, "corpus_params.txt"))
    G = read_goldens(joinpath(DIR, "io_values.txt"))
    golden_records = joinpath(DIR, "records")
    case_files(n) = ("target_$n.json", "target_$n.uncertain.json", "rig_$n.json")
    retired_file(rel) = any(n -> rel in case_files(n), CODE.retired) || (!isempty(CODE.retired) && rel == "rig_every.json")
    migrated_file(rel) = any(n -> rel in case_files(n), CODE.migrated_affine)
    projected_file(rel) = any(n -> rel in case_files(n), CODE.projected)
    mktempdir() do dir
        written, migrated = joinpath(dir, "written"), joinpath(dir, "migrated")
        actual = Dict{String, Any}(io_values(P, CODE, written))
        # the values
        n_aside = 0
        for (key, g) in G
            name = split(key, '/')[2]
            if startswith(key, "content_id/") && (name in CODE.retired || (name == "every" && !isempty(CODE.retired)))
                @test !haskey(actual, key)
                continue
            end
            @test haskey(actual, key)
            haskey(actual, key) || continue
            if startswith(key, "content_id/") && name in CODE.projected     # the value moved on purpose (projected)
                n_aside += 1
                continue
            end
            e = startswith(key, "content_id/") ? get(CODE.new_content_ids, name, g) : g
            ok, _, why = compare_value(e, actual[key])
            ok || @error "chain equivalence: $key moved: $why"
            @test ok
        end
        @test n_aside == length(CODE.projected)
        @test all(k -> haskey(G, k), keys(actual))
        # the records: every golden file, migrated to the code's version, is
        # byte for byte what the code writes; a retired case's is refused, an
        # affine one migrates (checked by value below), a projected one is
        # refused (its spline carries an affine term) while the code writes the
        # projected values
        n_files = n_refused = n_affine = 0
        for (root, _, files) in walkdir(golden_records), f in files
            rel = relpath(joinpath(root, f), golden_records)
            out = joinpath(migrated, rel)
            mkpath(dirname(out))
            if migrated_file(rel)
                r = CODE.migrate(joinpath(root, f), out)
                @test VisualGeometryToolkitIO.load(startswith(rel, "rig_") ? VgtkCore.Rig : VgtkCore.AbstractTarget, out) !== nothing
                @test !isfile(joinpath(written, rel))
                n_affine += 1
                continue
            end
            if retired_file(rel) || projected_file(rel)
                e = try CODE.migrate(joinpath(root, f), out); nothing catch e; e end
                @test e isa ArgumentError
                projected_file(rel) && @test occursin("pure bending", e.msg)
                @test projected_file(rel) == isfile(joinpath(written, rel))
                n_refused += 1
                continue
            end
            CODE.migrate(joinpath(root, f), out)
            same = isfile(joinpath(written, rel)) && read(out) == read(joinpath(written, rel))
            same || @error "chain equivalence: the record $rel moved"
            @test same
            n_files += 1
        end
        # a golden OpenCV export, its pixel frame under the legacy key, is read as it is (never rewritten): the
        # camera and poses of the code's export
        legacy = filter(f -> startswith(f, "opencv_"), readdir(golden_records))
        @test length(legacy) == 5
        for f in legacy
            g = joinpath(golden_records, f)
            @test occursin("\"image_origin\": \"opencv\"", read(g, String))
            a = VisualGeometryToolkitIO.load_opencv_calibration(g; file_unit = VgtkCore.mm)
            b = VisualGeometryToolkitIO.load_opencv_calibration(joinpath(written, f); file_unit = VgtkCore.mm)
            @test a.camera == b.camera && isequal(a.poses, b.poses) && a.images == b.images
        end
        # an affine golden is its factors: the same map, and the rig reports the golden (v5) id as its old one
        g = VisualGeometryToolkitIO._raw_tree(joinpath(golden_records, "target_A.json"))
        p = g["target"]["calibration"]["parameters"]
        u = VgtkCore.parse_unit(g["length_unit"])
        A, b = SMatrix{2, 2}(p[1], p[3], p[2], p[4]), SVector(VisualGeometryToolkitIO.read_length(p[5], u), VisualGeometryToolkitIO.read_length(p[6], u))
        c = VgtkCore.calibration(VisualGeometryToolkitIO.load(VgtkCore.AbstractTarget, joinpath(migrated, "target_A.json")))
        for q in (SVector(0.0, 0.0), SVector(0.05, -0.02), SVector(-0.03, 0.11))
            x = q .* VgtkCore.m
            @test all(abs.(c(x) .- (A * x + b)) .< 1.0e-15 * VgtkCore.m)
        end
        r = VisualGeometryToolkitIO.vgio_migrate(joinpath(golden_records, "rig_A.json"); out = joinpath(dir, "rig_A.ids.json"))
        @test r.old_id == G["content_id/A"] && r.new_id == VisualGeometryToolkitIO.content_id(r.out) != r.old_id
        for (name, id) in CODE.new_content_ids
            r = VisualGeometryToolkitIO.vgio_migrate(joinpath(golden_records, "rig_$name.json"); out = joinpath(dir, "rig_$name.ids.json"))
            @test r.old_id == G["content_id/$name"] && r.new_id == id
        end
        n_written = sum(length(fs) for (_, _, fs) in walkdir(written))
        n_projected = count(n -> isfile(joinpath(golden_records, n)), [f for n in CODE.projected for f in case_files(n)])
        @test n_files + n_projected == n_written
        @test n_files + n_refused + n_affine == sum(length(fs) for (_, _, fs) in walkdir(golden_records))
        @test n_affine == 3
        @test n_files >= 15                # the lenses, the camera stores, the exports and the rigid target's records
    end
end

@testitem "chain equivalence: a v5 rig's identity is v5's for every non-affine kind" setup = [ChainEquivalenceIO] begin
    # `_v5_id` reports the current identity as the old one for a v5 rig without
    # an affine kind: that is right only if the current identity tree of the
    # SAME value is the one v5 hashed. The goldens' ids were computed by v5's
    # code; the thin-plate values they hold carry an affine term (the current
    # constructor would project them), so they are rebuilt here as they were
    # (`project = false`) — the identity is of the value, not of its record.
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: Rigid, ThinPlateSpline, HeightField, PlanarTarget, PosedTarget, EuclideanMap, Rig, TargetFrame,
        grid_nodes, with_unit, m
    using StaticArrays: SVector
    using Rotations: QuatRotation
    P = read_goldens(joinpath(DIR, "corpus_params.txt"))
    G = read_goldens(joinpath(DIR, "io_values.txt"))
    nodes = grid_nodes(3; lo = 0.0, hi = 0.1)
    T = with_unit(ThinPlateSpline(nodes, P["param/thin_plate_spline/fx"], P["param/thin_plate_spline/fy"]; project = false), m)
    H = with_unit(HeightField(nodes, P["param/height_field/h"]; project = false), m)
    ppts = ChainEquivalenceIO.ChainEquivalenceCorpus.physical_points()[1:20]
    pose = EuclideanMap(ChainEquivalenceIO.RECORD_POSE_ROTATION, SVector(0.1, -0.05, 0.9) .* m)
    gauge = PosedTarget(PlanarTarget(ppts), EuclideanMap(QuatRotation(1.0, 0.0, 0.0, 0.0, false), SVector(0.0, 0.0, 0.0) .* m))
    for (name, c) in ("R" => Rigid(), "T" => T, "H" => H, "T_H" => H ∘ T)
        rig = Rig{TargetFrame{:gauge}}(Dict(:gauge => gauge, Symbol("t_", name) => PosedTarget(PlanarTarget(ppts; calibration = c), pose)))
        @test VisualGeometryToolkitIO.content_id(rig) == G["content_id/$name"]
    end
    # and the migration of the rigid golden reports it
    mktempdir() do dir
        r = VisualGeometryToolkitIO.vgio_migrate(joinpath(DIR, "records", "rig_R.json"); out = joinpath(dir, "rig_R.json"))
        @test r.old_id == r.new_id == G["content_id/R"]
    end
end
