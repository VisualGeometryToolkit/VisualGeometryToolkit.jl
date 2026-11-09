# Generator of spline_lens_golden.jl: the outputs of the OLD central and non-central
# grid models (CentralGenericModel, NoncentralGenericModel) and of the fitted
# SplineProjection, recorded before the SplineLens fold-in deletes the old types.
# The SplineLens equivalence tests compare against these values to 1e-12.
#
# Run it ONCE at the old code, in the Estimation test env (Core + Estimation):
#   include(".../VisualGeometryToolkitCore/test/fixtures/make_spline_lens_golden.jl")
# It needs the old types, so it cannot be re-run after the fold-in.
using StaticArrays, LinearAlgebra
using VisualGeometryToolkitCore
using VisualGeometryToolkitCore: CameraModel, CentralGenericModel, NoncentralGenericModel, KannalaBrandt, Vec3, Rect, px,
    SplineProjection, backproject, backproject_with_jacobian, project, undistort, with_present, estimate
using VisualGeometryToolkitEstimation: fit_camera

# --- the literal writer: Float64 through `repr` (exact round trip, NaN as NaN) ---------
lit(x::Real) = x isa Bool ? string(x) : repr(Float64(x))
function lit(A::AbstractArray)
    flat = "[" * join((lit(a) for a in A), ", ") * "]"
    ndims(A) == 1 ? flat : "reshape($flat, $(join(size(A), ", ")))"
end
lit(t::NamedTuple) = "(; " * join(("$k = $(lit(v))" for (k, v) in pairs(t)), ", ") * ")"
lit(s::AbstractString) = repr(String(s))
lit(t::Tuple) = "(" * join((lit(v) for v in t), ", ") * ",)"

col(vs) = reduce(hcat, (Float64.(collect(v)) for v in vs))              # a 3×N / 2×N matrix, one column per item
nanpair(p) = p === nothing ? SVector(NaN, NaN) : SVector{2, Float64}(p)
nanvec3(v) = SVector{3, Float64}(v)

# --- the models, as the Core tests build them -------------------------------------------
W = H = 1000
coeffs = (0.2468, 0.1505, -0.0214, 0.1308)
src = CameraModel(300.0px, SA[500.0px, 500.0px]; lens = KannalaBrandt(coeffs...))
central = CentralGenericModel(src, (W, H); grid_size = (24, 24))
present = trues(size(central.grid))
present[1:3, :] .= false; present[(end - 2):end, :] .= false; present[:, 1:3] .= false; present[:, (end - 2):end] .= false
masked = with_present(central, present)
pts = [0.01 * Vec3{Float64}(d) for d in central.grid]
noncentral = NoncentralGenericModel{Float64}(W, H, 0, 0, W - 1, H - 1, central.grid, pts)

# --- the lattice: 17 × 17 over the calibrated area (x fastest), then 8 pixels outside it --
lattice = [SVector(x, y) for y in range(0.0, H - 1.0; length = 17) for x in range(0.0, W - 1.0; length = 17)]
outside_px = [SVector(-1.0, 500.0), SVector(Float64(W), 500.0), SVector(500.0, -1.0), SVector(500.0, Float64(H)),
              SVector(-20.0, -20.0), SVector(W + 20.0, H + 20.0), SVector(-0.5, 10.0), SVector(W + 0.5, 990.0)]
pixels = vcat(lattice, outside_px)
# the 8 rays outside the domain: far off axis, several azimuths
outside_rays = [SVector(sind(t) * cosd(a), sind(t) * sind(a), cosd(t)) for (t, a) in
                ((100.0, 0.0), (120.0, 45.0), (140.0, 90.0), (160.0, 135.0), (175.0, 180.0), (150.0, 225.0), (130.0, 270.0), (110.0, 315.0))]

# The file is capped at 300 KB, so redundant arrays are asserted equal here and not stored: the Jacobian's value
# f is the backproject column, and the batched backproject is the scalar one bit for bit.
function record_central(m, rays_in; jacobian::Bool)
    bp = [backproject(m, p) for p in pixels]
    jf = [backproject_with_jacobian(m, p) for p in pixels]
    pr_in = [nanpair(project(m, X)) for X in rays_in]
    pr_out = [nanpair(project(m, X)) for X in outside_rays]
    bb = backproject(m, pixels)
    @assert isequal(col(first.(jf)), col(bp)) && isequal(col(bb), col(bp))
    bpj = project(m, rays_in, lattice)
    rec = (;
        backproject = col(bp),                                                       # 3 × 297, NaN outside
        project_in = col(pr_in),                                                     # 2 × 289, NaN where `nothing`
        project_out = col(pr_out),                                                   # 2 × 8
        batch_project_px = col(bpj.px),                                              # 2 × 289
        batch_project_ok = collect(Bool.(bpj.ok)),                                   # 289
    )
    jacobian || return rec
    return (; rec..., jac_M = reshape(reduce(hcat, (vec(Matrix(last(j))) for j in jf)), 3, 2, length(jf)))   # 3 × 2 × 297
end
rays_in = [backproject(central, p) for p in lattice]                                 # 289 unit rays, all inside the area

central_rec = record_central(central, rays_in; jacobian = true)
masked_rec = (; present = collect(present), record_central(masked, rays_in; jacobian = false)...)

# --- the non-central copy: the central grid's directions, points 0.01 · d (metres; not stored) -----------
nc_rays = [backproject(noncentral, p) for p in pixels]
@assert isequal(col(r.direction for r in nc_rays), col(backproject(central, p) for p in pixels))   # directions: the central model's, not stored
nc_X = [SVector{3, Float64}(r.origin) + 1500.0 * SVector{3, Float64}(r.direction) for r in nc_rays[1:length(lattice)]]
noncentral_rec = (;
    origins = col(r.origin for r in nc_rays),                                        # 3 × 297, NaN outside
    # project_X is at X = origin + 1500 · direction (the lattice rays; X not stored)
    project_X = col(nanpair(project(noncentral, X)) for X in nc_X),                  # 2 × 289, NaN where `nothing`
)

# --- undistort: a checker image (cells of 8 px, 1 on odd cell parity) sampled through the
# model's own pixels, nominal Kr, a 32 × 32 rect at the principal point ---------------------
checker = [Float64(isodd(fld(x, 8) + fld(y, 8))) for y in 0:(H - 1), x in 0:(W - 1)]   # [row, col]
Kr = SMatrix{3, 3, Float64}(central.nominal.K)
rect = Rect(484.0, 484.0, 32.0, 32.0)
und = undistort(central, checker, Kr, rect)
undistort_rec = (; Kr = Matrix(Kr), rect = [484.0, 484.0, 32.0, 32.0], cell = 8.0,
                 central = Matrix{Float64}(und.data))

# --- the fitted SplineProjection ----------------------------------------------------------
fit = fit_camera(SplineProjection, central, Rect(0, 0, 1000, 1000); tolerance = 0.01)
sp = estimate(fit)
sp_pts = [SVector(500.0 + ρ * cos(a), 500.0 + ρ * sin(a)) for ρ in (0.0, 120.0, 300.0, 470.0) for a in range(0, 2π; length = 16)]   # 64 pixels
sp_rays = [backproject(central, p) for p in sp_pts]
spline_rec = (;
    area = [sp.spline.area.origin[1], sp.spline.area.origin[2], sp.spline.area.widths[1], sp.spline.area.widths[2]],   # x0, y0, wx, wy of the chart-plane area
    controls = reshape(reduce(hcat, (Float64.(collect(c)) for c in sp.spline.controls)), 2, size(sp.spline.controls)...),   # 2 × 60 × 60
    chart_rotation = Matrix{Float64}(sp.chart.t2.linear),                            # chart = Lens{Equidistant} ∘ LinearMap(this)
    pixels = col(sp_pts), rays = col(sp_rays),                                       # 2 × 64, 3 × 64
    project = col(nanpair(project(sp, X)) for X in sp_rays),                         # 2 × 64
)

golden = (;
    f = 300.0, pp = [500.0, 500.0], kb = collect(coeffs), image = [W, H], grid_size = [24, 24],
    pixels = col(pixels), n_lattice = length(lattice),                               # 2 × 297: 289 lattice (x fastest) then 8 outside
    rays_out = col(outside_rays),                                                    # 3 × 8 (the 289 inside rays are central.backproject[:, 1:n_lattice])
    central = central_rec, masked = masked_rec, noncentral = noncentral_rec,
    undistort = undistort_rec, spline = spline_rec,
)

sha = readchomp(`git -C $(@__DIR__) rev-parse --short HEAD`)
out = joinpath(@__DIR__, "spline_lens_golden.jl")
write(out, "# Generated by make_spline_lens_golden.jl at commit $sha. Do not edit: literal values of the OLD grid models,\n" *
           "# read by the SplineLens equivalence tests via `include`.\nconst GOLDEN = " * lit(golden) * "\n")
filesize(out)
