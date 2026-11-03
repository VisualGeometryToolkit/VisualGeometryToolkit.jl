# =============================================================================
# The chain-equivalence gate: the code at hand against the goldens written on
# the code before CoordinateTransformations' `∘` replaced `CalibrationChain`
# and the lens `Chain` (test/data/chain_equivalence/, generate.jl).
# =============================================================================
#
# The goldens are never regenerated. What the code may change of them is
# stated in the ADAPTER below and nowhere else (corpus.jl's
# `expected_core_value`); a value that moves any other way fails here.

@testsetup module ChainEquivalenceCore
    import VisualGeometryToolkitCore
    const DIR = joinpath(pkgdir(VisualGeometryToolkitCore), "test", "data", "chain_equivalence")
    include(joinpath(DIR, "corpus.jl"))
    using .ChainEquivalenceCorpus
    include(joinpath(DIR, "core_values.jl"))
    using VisualGeometryToolkitCore: calibration_tag, transform_deriv, transform_deriv_params
    using CoordinateTransformations: AffineMap, ComposedTransformation
    using StaticArrays: SMatrix, SVector

    export DIR, CODE, MAX_ULPS, read_goldens, target_values, lens_values, expected_core_value, compare_value, scaled_comparison, target_cases,
        design_points, case_members

    # THE ADAPTER — the code at hand, as corpus.jl's header describes, and what
    # it changes of the goldens on purpose:
    # - targets (compose with CoordinateTransformations, revision 2): the
    #   affine print error is CT's `AffineMap`; a composition is CT's `∘`, its
    #   parameters, names, bounds, Jacobian columns, gauges and regularizer
    #   rows in CT's order (the outer map, the height field, first: `ct_order`);
    #   the parameter Jacobian is CT's `transform_deriv_params`, the point
    #   Jacobian CT's `transform_deriv`; a composition has no tag (its record is
    #   written by structure); and the in-plane compositions are retired;
    # - lenses: a distortion's stages in sequence are CT's `∘` (the Lens
    #   `Chain` removed), its parameters, names, bounds and Jacobian columns in
    #   CT's order, the outer stage first (`lens_ct_order`); the lens record's
    #   "chain" lists the stages outer first (`legacy_lens_record` reverses it
    #   back to the goldens' application order).
    const CODE = (
        affine = (A, b) -> AffineMap(SMatrix{2, 2}(A), SVector{2}(b)),
        compose_calibrations = members -> ∘(reverse(members)...),
        compose_distortions = stages -> ∘(reverse(stages)...),
        jacobian_theta = transform_deriv_params,
        jacobian_x = transform_deriv,
        tag = c -> c isa ComposedTransformation ? nothing : calibration_tag(typeof(c)),
        legacy_lens_record = r -> haskey(r, "chain") ? merge(r, Dict{String, Any}("chain" => reverse(r["chain"]))) : r,
        composition_tags = false,
        ct_order = true,
        lens_ct_order = true,
        # Two in-plane kinds never compose (a Rigid member is an in-plane kind):
        # the only compositions left are A_H and T_H, a height field after an
        # AffineMap or a ThinPlateSpline. A target refuses every one of these.
        retired = Set([
            "R_H", "R_A", "R_A_H", "R_T", "R_T_H", "A_R", "A_R_H", "A_T", "A_T_H", "T_R", "T_R_H", "T_A", "T_A_H",
            "R_A_T", "R_A_T_H", "R_T_A", "R_T_A_H", "A_R_T", "A_R_T_H", "A_T_R", "A_T_R_H", "T_R_A", "T_R_A_H", "T_A_R", "T_A_R_H",
        ]),
    )
end

@testitem "chain equivalence: Core's values reproduce the goldens" setup = [ChainEquivalenceCore] begin
    P = read_goldens(joinpath(DIR, "corpus_params.txt"))
    G = read_goldens(joinpath(DIR, "core_values.txt"))
    actual = Dict{String, Any}([target_values(P, CODE); lens_values(P, CODE)])
    failures = String[]
    worst = Dict{String, Int128}()
    n_compared = n_set_aside = 0
    for (key, g) in G
        name = split(key, '/')[2]
        if startswith(key, "target/") && name in CODE.retired
            n_set_aside += 1
            continue
        end
        e = expected_core_value(key, g, CODE)
        if e === :absent
            haskey(actual, key) && push!(failures, "$key: present, the code has no such value")
            n_set_aside += 1
            continue
        end
        haskey(actual, key) || (push!(failures, "$key: missing"); continue)
        ok, d, why = compare_value(e, actual[key]; scaled = scaled_comparison(key, CODE))
        n_compared += 1
        ok || push!(failures, "$key: $why")
        d > 0 && (worst[key] = d)
    end
    for key in keys(actual)
        haskey(G, key) || push!(failures, "$key: computed, but no golden")
    end
    isempty(failures) || @error "chain equivalence: $(length(failures)) values moved" failures
    isempty(worst) || @info "chain equivalence: values off by ulps (≤ $(MAX_ULPS))" worst
    @test isempty(failures)
    @test n_compared + n_set_aside == length(G)
    @test n_compared >= 250
end

@testitem "chain equivalence: the retired compositions are refused" setup = [ChainEquivalenceCore] begin
    using VisualGeometryToolkitCore: PlanarTarget
    P = read_goldens(joinpath(DIR, "corpus_params.txt"))
    cases = Dict(target_cases(P, CODE))
    for name in CODE.retired
        @test_throws ArgumentError PlanarTarget(design_points(); calibration = cases[name])
    end
    @test all(n -> haskey(cases, n), CODE.retired)
end
