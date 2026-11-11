# =============================================================================
# The cells of a floorplan fan: one per (rig station, distance), restated by the
# BlobBoards bmvc2026 and CaliBlobs icra2026 papers before it moved here.
# =============================================================================

"""
    station_cells(df; stations = -80:10:80) -> Vector{NamedTuple}

The cells of a floorplan fan from `df`, one row per (frame, marker) attempt with
`rig_distances_bin` (mm), `rig_angles` (rad, the frame's), `error_distances`
(mm, `missing` where nothing was placed) and `abs_error_angles_deg`. The
attempts at one distance are numbered into stations by their rig angle, 10°
apart from the first angle rounded to 10°; per (station, distance) a cell
`(; angle, dist, n, rate, med, medr)`: the station's angle (degrees), the
distance, the placed count, the placed fraction, the median distance error and
the median |angle error| (`NaN` with nothing placed). The ±90° stations are
edge-on (a plane presenting no area) and are dropped; every station angle of
`stations` (degrees) at a distance without an attempt gets a cell `n = 0`, so a
hole in the fan always means no detection, never missing data.
"""
function station_cells(df::AbstractDataFrame; stations::AbstractRange = -80:10:80)
    station = Dict{Tuple{Float64, Int}, Int}()
    for spot in unique(df.rig_distances_bin)
        angs = sort(unique(df.rig_angles[df.rig_distances_bin .== spot]))
        anchor = round(Int, rad2deg(first(angs)) / 10) * 10
        for (k, a) in enumerate(angs)
            station[(a, spot)] = anchor + 10 * (k - 1)
        end
    end
    d = transform(df, [:rig_angles, :rig_distances_bin] => ByRow((a, s) -> station[(a, s)]) => :station)
    med(e) = (v = collect(skipmissing(e)); isempty(v) ? NaN : median(v))
    g = combine(
        groupby(d, [:station, :rig_distances_bin]),
        :error_distances => (e -> count(!ismissing, e)) => :n,
        :error_distances => length => :ntot,
        :error_distances => med => :med,
        :abs_error_angles_deg => (e -> med(abs.(e))) => :medr
    )
    cells = [(angle = Float64(r.station), dist = Float64(r.rig_distances_bin), n = r.n, rate = r.n / r.ntot,
              med = r.med, medr = r.medr) for r in eachrow(g)]
    filter!(c -> abs(c.angle) < 90, cells)
    have = Set((c.angle, c.dist) for c in cells)
    for spot in unique(df.rig_distances_bin), a in Float64.(stations)
        (a, Float64(spot)) in have || push!(cells, (angle = a, dist = Float64(spot), n = 0, rate = 0.0, med = NaN, medr = NaN))
    end
    return cells
end
