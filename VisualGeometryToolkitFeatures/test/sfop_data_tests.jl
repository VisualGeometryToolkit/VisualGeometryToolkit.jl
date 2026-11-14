# Opt-in, real data: detect_sfop → the seeded detect_foerstner on the fisheye chrome photograph DSC06962 (RAW green
# cells, Features' one RAW path), the frame on which the owner found SFOP's wrong detections (2026-11-14): dark spots
# taken for circular points. Skipped unless the paper-data tier is under VGTK_DATA_ROOT
# (dgx01's /raid/data/jpr/rc-vgtk-root). The counts are 0.9.6's (first found by its draft on the desktop CPU,
# 2026-11-14); a missed blob is acceptable, a wrong one is not, so each bound is one that must not rise.

@testitem "detect_sfop on DSC06962 (RAW green cells): no dark blob under :light (real data, opt-in)" begin
    import KernelAbstractions as KA
    using LinearAlgebra: norm
    using LibRaw_jll
    path = joinpath(get(ENV, "VGTK_DATA_ROOT", ""), "paper-data", "geometric-accuracy", "fisheye", "chrome-april", "DSC06962.ARW")
    if !isfile(path)
        @test_skip "DSC06962.ARW is not under VGTK_DATA_ROOT ($(repr(get(ENV, "VGTK_DATA_ROOT", ""))))"
    else
        L = load_raw(path; mode = :green_cells)
        all_ = detect_sfop(L)
        light = detect_sfop(L; polarity = :light)
        # the bright board dots: under :light no dark blob (0.9.4's two dark-spot rows, raw (2283, 1075) and (2646, 1107),
        # are gone)
        @test length(light) < length(all_)
        @test !any(d -> any(norm(2 .* d.center .- t) < 4 for t in ((2283.0, 1075.0), (2646.0, 1107.0))), light)
    end
end

# Recall against 0.9.5: test/data/sfop_recall_095.tsv holds every :circular row that Features 0.9.5's pipeline
# (detect_sfop(L), then the seeded detect_foerstner(Float32, L, seeds), on the CPU) found on a sample of the five FX3
# chrome sets of the geometric-accuracy tier: 12 photographs, those with the most dark-polarity, duplicate and
# sign-not-significant rows, and three whose significantly light rows a draft of 0.9.6 lost (DSC06867, DSC07405,
# DSC07435). Each row carries its seed's σ and z, the seed's scale-normalised Laplacian over its sd under 2 grey
# levels of white noise (positive: dark). Under polarity = :light a 0.9.5 row is found again within 0.5 px, or is
# (i) significantly dark (z above the 0.999 quantile) or (ii) of a sign not significant (|z| at most it): the gate
# needs significant evidence of the requested polarity. A lost row that is significantly LIGHT is a correct dot
# dropped: none may be. Duplicates are not removed (the consumer resolves them): each 0.9.5 pair of rows within 1 px
# whose rows are both found again keeps both, two distinct rows.
@testitem "detect_sfop(:light) → seeded detect_foerstner on the five FX3 chrome sets: no light 0.9.5 row lost, duplicates kept (real data, opt-in)" begin
    using LibRaw_jll
    root = joinpath(get(ENV, "VGTK_DATA_ROOT", ""), "paper-data", "geometric-accuracy")
    old = Dict{Tuple{String, String}, Vector{NTuple{4, Float64}}}()     # (set, photo) => (x, y, seed σ, z)
    for l in Iterators.drop(eachline(joinpath(@__DIR__, "data", "sfop_recall_095.tsv")), 1)
        c = split(l, '\t')
        push!(get!(old, (c[1], c[2]), NTuple{4, Float64}[]), Tuple(parse.(Float64, c[3:6])))
    end
    if !all(((s, p),) -> isfile(joinpath(root, s, p)), keys(old))
        @test_skip "the geometric-accuracy tier is not under VGTK_DATA_ROOT ($(repr(get(ENV, "VGTK_DATA_ROOT", ""))))"
    else
        zq = 3.090232306167813                                       # the standard normal's 0.999 quantile
        d(a, b) = hypot(a[1] - b[1], a[2] - b[2])
        tally = Dict(:kept => 0, :dark => 0, :not_significant => 0, :light => 0)
        pairs = kept_pairs = 0
        for ((s, p), o) in sort(collect(old))
            L = load_raw(joinpath(root, s, p); mode = :green_cells)
            seeds = detect_sfop(L; polarity = :light)                   # the default device: a GPU when there is one
            new = [(r.center[1], r.center[2]) for r in detect_foerstner(Float32, L, seeds) if r.label === :circular]
            found = [any(b -> d(a, b) < 0.5, new) for a in o]
            for (i, a) in enumerate(o)
                why = found[i] ? :kept : a[4] > zq ? :dark : abs(a[4]) <= zq ? :not_significant : :light
                tally[why] += 1
                why === :light && @info "a significantly light 0.9.5 row lost" s p a
            end
            # every 0.9.5 pair within 1 px whose rows are both found again: two new rows near it, not one
            for i in eachindex(o), j in (i + 1):lastindex(o)
                d(o[i], o[j]) < 1 && found[i] && found[j] || continue
                pairs += 1
                kept_pairs += count(b -> d(b, o[i]) < 0.5 || d(b, o[j]) < 0.5, new) >= 2
            end
            @test !any(b -> any(a -> a[4] > zq && d(a, b) < 0.5, o), new)  # no new row on a dark blob of 0.9.5's
        end
        @info "recall against 0.9.5" tally pairs kept_pairs
        @test tally[:light] == 0
        @test tally[:dark] > 0 && pairs > 0 && kept_pairs == pairs       # the sample exercises both
    end
end
