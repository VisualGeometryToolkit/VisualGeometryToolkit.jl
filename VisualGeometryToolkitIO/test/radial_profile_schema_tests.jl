# vgio/radial_profile/v1.0 (radial_profile_schema.jl): Core's BinnedRadialProfile — every chart round trips exact; RC's
# measured flat_equisolid.json through a test-local importer; the table's refusals; another pixel frame moved into colmap.

@testitem "radial_profile: every chart round trips exact; RC's flat_equisolid.json through an importer" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitIO: validate
    using VisualGeometryToolkitCore: BinnedRadialProfile, radial_chart
    profiles = (
        BinnedRadialProfile(:linear, SVector(2120.5, 1416.25), 2551.0, [0.0, 0.25, 0.5, 1.0], [0.9 0.8 0.6; 0.95 0.85 0.7], [100, 300, 900]),
        BinnedRadialProfile(:projective, SVector(320.0, 240.0), 400.0, [0.0, 0.1, 0.3], reshape([1.0, 0.75], 1, 2), [0, 12]),
        BinnedRadialProfile(:equisolid, SVector(2126.8, 1434.9), 1060.3, deg2rad.(0.0:1.0:3.0), [0.026 0.025 0.024], [829, 2499, 4173];
                            focal_length = 933.2),
    )
    mktempdir() do dir
        p = joinpath(dir, "profile.json")
        for v in profiles
            save(p, v)
            @test validate(p) == "radial_profile"
            b = load(BinnedRadialProfile, p)
            @test b == v && typeof(b) == typeof(v) && b.edges == v.edges && b.values == v.values
            d = JSON.parsefile(p)
            @test d["profile"]["chart"]["type"] == String(radial_chart(v)) && haskey(d["profile"]["chart"], "focal_length") == (v.focal_length !== nothing)
            @test d["profile"]["values"] == [collect(r) for r in eachrow(v.values)]   # one row per channel
        end
        # RC's measured flat field (read-only), converted by a test-local importer into the value
        rc = "/raid/data/jpr/rc-vgtk-root/vignetting/flat/fisheye/flat_equisolid.json"
        if isfile(rc)
            r = JSON.parsefile(rc)
            v = BinnedRadialProfile(:equisolid, SVector(Float64(r["circle"][1]), Float64(r["circle"][2])), Float64(r["circle"][3]),
                                    deg2rad.(Float64.(r["edges_deg"])), permutedims(reduce(hcat, [Float64.(c) for c in r["flat"]])),
                                    Int.(r["pixels_per_ring"]); focal_length = Float64(r["f_px"]))
            @test size(v.values) == (3, length(r["edges_deg"]) - 1) && v.values[2, 1] == r["flat"][2][1]
            save(p, v; provenance = (source = basename(r["source"]),))
            b = load(BinnedRadialProfile, p)
            @test b == v
        else
            @test_skip isfile(rc)
        end
    end
end

@testitem "radial_profile: the table's refusals; another pixel frame" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitCore: BinnedRadialProfile
    v = BinnedRadialProfile(:equisolid, SVector(100.0, 50.0), 80.0, [0.0, 0.5, 1.0], [0.9 0.7], [10, 20]; focal_length = 60.0)
    mktempdir() do dir
        p, q = joinpath(dir, "p.json"), joinpath(dir, "q.json")
        save(p, v)
        function refused(f!)
            d = JSON.parsefile(p; dicttype = Dict{String, Any})
            f!(d)
            write(q, JSON.json(d))
            return try load(BinnedRadialProfile, q); false catch e; e isa ArgumentError end
        end
        pr(d) = d["profile"]
        @test refused(d -> (pr(d)["edges"] = [0.0, 1.0, 0.5]))                    # not increasing
        @test refused(d -> (pr(d)["counts"] = [10]))                               # one per bin
        @test refused(d -> (pr(d)["counts"] = [10, -1]))
        @test refused(d -> (pr(d)["values"] = [[0.9]]))                            # one value per bin
        @test refused(d -> delete!(pr(d)["chart"], "focal_length"))                # equisolid states its focal length
        @test refused(d -> (pr(d)["chart"]["type"] = "linear"))                    # …and only it
        @test refused(d -> (pr(d)["chart"]["type"] = "stereographic"; delete!(pr(d)["chart"], "focal_length")))
        @test refused(d -> (pr(d)["type"] = "radial"))
        @test !refused(identity)
        d = JSON.parsefile(p; dicttype = Dict{String, Any})
        d["units"]["pixel"] = "opencv"
        write(q, JSON.json(d))
        @test load(BinnedRadialProfile, q).center == v.center .+ 0.5
    end
end
