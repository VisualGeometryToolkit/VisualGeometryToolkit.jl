# =============================================================================
# fit_camera's shared cases — ReTestItems `@testsetup` modules
# =============================================================================

# -----------------------------------------------------------------------------
# The lens-family cases of the regression `fit_camera(L, source, region)`
# replaced: every case of its predecessor's suite but the fixed-focal one (the
# focal is always fitted now) and the inverse direction (removed). The
# predecessor's domain — a cap on the radius about the principal point, or on
# the view angle — is the disc about the principal point of that radius.
# -----------------------------------------------------------------------------
@testsetup module FitCameraLensCases
using StaticArrays: SVector, SMatrix
using VisualGeometryToolkitCore: CameraModel, Division, Lens, LiftLens, Radial, Tangential, Additive, Perspective, Equidistant,
    Even, KannalaBrandt, DoubleSphere, Scaramuzza, Pinhole, Ellipse, projected_radius, radial_part, view_angle_limit,
    monotone_radius, calibration_matrix, lens
export KB4, SC3, DSL, OPENCV3, EQ, mdiv, rcorner, disc, LENS_CASES, RECORDED, agrees, agrees_px

const KB4 = KannalaBrandt{4}
const SC3 = Scaramuzza{3}
const DSL = typeof(Lens(DoubleSphere(0.5, 0.5)))
const OPENCV3 = Lens{Perspective, Radial{Even, 3, 0, Float64}}
const EQ = typeof(Lens(Equidistant()))
const pp = (2120.0, 1416.0)                         # a 4240 × 2832 frame's principal point
const mdiv = CameraModel(; pp, focal = 927.0, lens = Division(-0.3302, -0.0765))   # the hallway-session camera
const rcorner = hypot(2120.0, 1416.0)               # the farthest sensor point from pp
disc(m, R) = Ellipse(SVector(calibration_matrix(m)[1, 3], calibration_matrix(m)[2, 3]), R, R, 0.0)
disc_angle(m, θ) = disc(m, Float64(projected_radius(radial_part(lens(m)), θ)) * calibration_matrix(m)[1, 1])
const mkb = CameraModel(; pp, focal = 600.0, lens = KannalaBrandt(-0.02, 0.004, -0.001, 0.0002))
const msc = CameraModel(; pp, focal = 927.0, lens = Scaramuzza(-0.3302, 0.02, -0.0765))
const rlit = 927.0 * monotone_radius(mdiv.lens) * 0.97
const cam900 = CameraModel(; pp, focal = 900.0, lens = Division(-0.2, -0.08))
const DECENTERED = typeof(Lens(Perspective(), Additive(Radial((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), Tangential(0.0, 0.0))))
# name => (target, source, the predecessor's domain keywords, the region)
const LENS_CASES = [
    "KB4 mkb" => (KB4, mkb, (;), disc_angle(mkb, view_angle_limit(radial_part(mkb.lens)))),
    "KB4 mdiv" => (KB4, mdiv, (; max_radius_px = rcorner), disc(mdiv, rcorner)),
    "SC3 mdiv" => (SC3, mdiv, (; max_radius_px = rcorner), disc(mdiv, rcorner)),
    "KB4 msc" => (KB4, msc, (; max_radius_px = rcorner), disc(msc, rcorner)),
    "DSL rlit" => (DSL, mdiv, (; max_radius_px = rlit), disc(mdiv, rlit)),
    "DSL rcorner" => (DSL, mdiv, (; max_radius_px = rcorner), disc(mdiv, rcorner)),
    "OPENCV3" => (OPENCV3, mdiv, (; max_radius_px = rcorner), disc(mdiv, rcorner)),
    "OPENCV decentered" => (DECENTERED, mdiv, (; max_radius_px = rcorner), disc(mdiv, rcorner)),
    "Pinhole 30°" => (Pinhole, mdiv, (; max_view_angle = deg2rad(30)), disc_angle(mdiv, deg2rad(30))),
    "EQ" => (EQ, mdiv, (; max_radius_px = rcorner), disc(mdiv, rcorner)),
    "Division{2}" => (Division{2}, mdiv, (; max_radius_px = rcorner), disc(mdiv, rcorner)),
    "LiftLens 1,2" => (LiftLens{Radial{Even, 1, 2, Float64}}, mdiv, (; max_radius_px = rcorner), disc(mdiv, rcorner)),
    "Division{3}" => (Division{3}, mdiv, (; max_radius_px = rcorner), disc(mdiv, rcorner)),
    "SC3 Div(-0.2)" => (SC3, CameraModel(; pp, focal = 927.0, lens = Division(-0.2)), (; max_radius_px = rcorner), disc(mdiv, rcorner)),
    "SC3 Div(0.05)" => (SC3, CameraModel(; pp, focal = 927.0, lens = Division(0.05)), (; max_radius_px = rcorner), disc(mdiv, rcorner)),
    "Division" => (Division, mdiv, (; max_radius_px = rcorner), disc(mdiv, rcorner)),
    "Division{3,F64} cam900" => (Division{3, Float64}, cam900, (; max_radius_px = 2500.0), disc(cam900, 2500.0)),
]
# The predecessor's results on those cases (its parameters, its focal rescale and
# its node maximum in px), recorded 2026-11-04 from VisualGeometryToolkitEstimation
# 0.5.0 before its removal.
const RECORDED = Dict(
    "KB4 mkb" => ((-0.020000000000000354, 0.004000000000000143, -0.0010000000000000208, 0.00020000000000000096), 1.0000000000000002, 1.0658141036401503e-12),
    "KB4 mdiv" => ((-0.06515434420238685, 0.010607772078737653, -0.0010396595549055023, 0.00010791092268135981), 1.0232102880402831, 5.09032429470407),
    "SC3 mdiv" => ((-0.33020000000000016, 3.5837149639901175e-16, -0.07650000000000012), 1.0, 4.1167069753100805e-13),
    "KB4 msc" => ((-0.044548031256473, 0.004167072885135951, -0.00010264385610513718, 5.7125892738148095e-5), 1.0198680222426653, 4.576977679176289),
    "DSL rlit" => ((0.9933353328021481, 0.9999999999245603), 2.012298105653286, 1.922522583946138),
    "DSL rcorner" => ((0.8811332875618199, 0.6346864061026177), 1.6368194451496931, 84.45536223088573),
    "OPENCV3" => ((0.0, 0.0, 0.0), 1.0, 925672.3450883676),
    "OPENCV decentered" => ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 1.0, 925672.3450883676),
    "Pinhole 30°" => ((), 0.9360598974446428, 16.994581496565733),
    "EQ" => ((), 0.9404146696930101, 192.42342970532528),
    "Division{2}" => ((-0.3302000000000001, -0.07650000000000004), 1.0000000000000002, 4.1167069753100805e-13),
    "LiftLens 1,2" => ((2.3474209565048993e-16, -0.330199999999999, -0.07650000000000017), 0.9999999999999997, 8.233413950620161e-13),
    "Division{3}" => ((-0.3301999999999996, -0.07650000000000003, 6.3113522045140545e-18), 0.9999999999999998, 4.1167069753100805e-13),
    "SC3 Div(-0.2)" => ((-0.19999999999999676, -2.154635470121594e-15, 4.2834133065935985e-16), 0.9999999999999986, 5.145883719137601e-13),
    "SC3 Div(0.05)" => ((0.04999999999999969, 2.9972532570753634e-16, -6.84701367528853e-17), 1.0000000000000002, 1.6466827901240322e-12),
    "Division" => ((-1.55598834216797,), 1.777876726213889, 320.92474843758316),
    "Division{3,F64} cam900" => ((-0.19999999999999993, -0.07999999999999997, -3.486244641673166e-18), 1.0, 3.9968028886505635e-13),
)
# Within the fit's tolerance: parameters and focal rescale to 1e-6, the node
# maximum within 5 % (or both at rounding, below 1e-9 px).
agrees(p, q) = length(p) == length(q) && all(abs.(collect(Float64, p) .- collect(Float64, q)) .<= 1.0e-6)
agrees_px(a, b) = (a < 1.0e-9 && b < 1.0e-9) || abs(a - b) <= 0.05 * max(a, b)
end

# -----------------------------------------------------------------------------
# PanoramaStitching's synthetic camera: a Kannala–Brandt fisheye (f = 300 px)
# imaging to ~100° off axis, its 40 × 40 grid resampled over a 1000 × 1000
# image, the lit disc of radius 480 px about the centre.
# -----------------------------------------------------------------------------
@testsetup module FitCameraFisheye
using StaticArrays: SA, SVector
using VisualGeometryToolkitCore: CameraModel, CentralGenericModel, KannalaBrandt, Ellipse, Rect, px
export FISHEYE, GRID, DISC, REGION

const FISHEYE = CameraModel(300.0px, SA[500.0px, 500.0px]; lens = KannalaBrandt(-0.02))
const GRID = CentralGenericModel(FISHEYE, (1000, 1000); grid_size = (40, 40))
const DISC = Ellipse(SVector(500.0, 500.0), 480.0, 480.0, 0.0)
const REGION = Rect(0, 0, 1000, 1000) ∩ DISC
end
