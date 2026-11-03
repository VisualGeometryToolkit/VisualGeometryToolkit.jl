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
    using VisualGeometryToolkitCore: CalibrationChain, Chain, Affine, calibration_jacobian, calibration_tag

    export DIR, CODE, MAX_ULPS, read_goldens, target_values, lens_values, expected_core_value, compare_value, target_cases,
        design_points, case_members

    # THE ADAPTER — the code at hand, as corpus.jl's header describes. Every
    # field is the code the goldens were written on: the comparison is the
    # identity, which proves the harness.
    const CODE = (
        affine = Affine,
        compose_calibrations = members -> CalibrationChain(members),
        compose_distortions = stages -> Chain(stages...),
        jacobian_theta = calibration_jacobian,
        jacobian_x = (c, x) -> VisualGeometryToolkitCore.Cameras._point_jacobian(c, x),
        tag = c -> calibration_tag(typeof(c)),
        legacy_lens_record = identity,
        composition_tags = true,
        ct_order = false,
        retired = Set{String}(),
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
        ok, d, why = compare_value(e, actual[key])
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
    @test n_compared >= 500
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
