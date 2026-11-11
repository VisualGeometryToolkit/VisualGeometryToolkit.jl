# The floor plan's station fan as TikZ (src/render/station_fan.jl), byte for
# byte what CaliBlobs icra2026's `write_floorplan_fan` (src/floorplan_tikz.jl, a
# verbatim copy of BlobBoards bmvc2026's `write_floorplan_tikz` fan) wrote for
# the same cells: the fixtures in data/station_fan/ are its output (2026-11-11,
# PlotUtils 1.5.1, ColorSchemes 3.32.0) for the synthetic cells built here.

@testitem "station_fan" begin
    using Test
    using PlotUtils: cgrad
    using VisualGeometryToolkitEval.Render: write_station_fan
    fixtures = joinpath(@__DIR__, "data", "station_fan")
    # every station at three distances; seven cells without a detection, the
    # errors over three orders of magnitude (each span clamped at both ends)
    cells = let out = NamedTuple[]
        for (i, d) in enumerate((1000.0, 2000.0, 3000.0)), a in -80.0:10.0:80.0
            k = round(Int, a / 10) + 9 + 17 * (i - 1)
            n = k % 7 == 3 ? 0 : 1 + k % 5
            push!(out, (angle = a, dist = d, n = n, rate = n == 0 ? 0.0 : n / 5,
                        med = n == 0 ? NaN : 1.3 * 1.17^k, medr = n == 0 ? NaN : 0.11 * 1.19^k))
        end
        out
    end
    colormap = let g = cgrad(:RdYlGn)
        t -> get(g, t)
    end
    header(m) = "floorplan fan: synthetic / $(m) — generated, edit freely"
    fan(; kw...) = sprint(io -> write_station_fan(colormap, io, cells; kw...))
    fixture(name) = read(joinpath(fixtures, name), String)
    @test fan(; value = :rate, span = (0, 1), comment = header("det")) == fixture("fan_det.tex")
    @test fan(; value = :med, span = (2, 120), logarithmic = true, reverse = true, comment = header("trans")) ==
          fixture("fan_trans.tex")
    @test fan(; value = :medr, span = (0.2, 50), logarithmic = true, reverse = true, comment = header("rot")) ==
          fixture("fan_rot.tex")
    @test fan(; value = :med, span = (5, 60), logarithmic = true, reverse = true, comment = header("trans"),
              ring_labels = false) == fixture("fan_trans_range_nolabels.tex")
    # the file form writes the same bytes
    mktempdir() do dir
        path = joinpath(dir, "fan.tex")
        @test write_station_fan(colormap, path, cells; value = :rate, span = (0, 1), comment = header("det")) == path
        @test read(path, String) == fixture("fan_det.tex")
    end
    @test_throws ArgumentError fan(; value = :med, span = (5, 5))
end
