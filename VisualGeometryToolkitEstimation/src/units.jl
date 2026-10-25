# =============================================================================
# Units at the boundary — one strip at entry, one reattachment at exit
# =============================================================================
#
# The layering (VisualGeometryToolkitCore `conventions/units.md`): a public
# estimator that takes physical values — poses, 3D points — strips them once at
# entry and reattaches the unit once at exit. Its problems, solvers, LM rows and
# charts take plain numbers (`PlainEuclideanMap`, `StaticVector{N, <:Real}`), so a
# physical value that reaches one is a `MethodError`.
#
# The working unit is the METRE. `ustrip(m, q)` of a base-form quantity is its
# stored value, exactly, so a physical input runs the arithmetic of its plain
# form in metres bit for bit, and the reattachment (`x * m`, scale 1) is exact
# too. Unitful in → unitful out; plain in → plain out, as before. The helpers
# below run once per public call, never inside a loop that RANSAC or an LM
# iterates.
#
# What comes back unitful is the model (a pose, a calibration); the residual
# statistics a result carries — residuals, `rms`, `sigma`, covariances, noise
# scales — are plain numbers, and for a physical input their length is the
# metre (each public docstring says which).

# A 3D point of lengths; a pose of lengths.
const _PhysicalPoint3 = StaticVector{3, <:Quantity}
const _PhysicalPose3 = UnitfulEuclideanMap{3}

# A length supplied beside physical data (a threshold, a characteristic length,
# a noise scale), as plain metres. A plain number there is a unit the caller did
# not state: FlexUnits' `DimensionError`, as the arithmetic would raise.
_metres(::Nothing) = nothing
_metres(q::Quantity) = ustrip(VgtkCore.m, q)
_metres(x::Real) = throw(DimensionError(x, VgtkCore.m))

# A physical value as plain metres.
_plain_metres(E::_PhysicalPose3) = ustrip(VgtkCore.m, E)
_plain_metres(X::_PhysicalPoint3) = map(q -> ustrip(VgtkCore.m, q), X)
_plain_metres(r::HyperRectangle{2, <:Quantity}) = ustrip(VgtkCore.m, r)

# A plain pose read as metres: by construction (the documented attachment,
# `EuclideanMap(R, t .* m)`), its frames kept. The rotation is the plain one,
# unchecked and unchanged.
_in_metres(E::EuclideanMap{3, T, R, <:StaticVector{3, <:Real}, Src, Dst}) where {T, R, Src, Dst} =
    relabel(Src, Dst, EuclideanMap(rotation(E), translation(E) .* VgtkCore.m; check = false))

# A correspondence set with its endpoints mapped: a `Csponds` keeps its
# attribute columns (a PROSAC `score`, `weights`), so the plain set samples and
# weighs exactly as the physical one would.
_map_csponds(fs, ft, cs::VgtkCore.Csponds) =
    VgtkCore.Csponds(map(fs, sources(cs)), map(ft, targets(cs)), VgtkCore.attrs_of(cs))
_map_csponds(fs, ft, cs::AbstractVector{<:Pair}) = [fs(first(c)) => ft(last(c)) for c in cs]

# A RANSAC (or any) result with its model read as metres: everything else as it was.
_in_metres(r::FitResult) = FitResult(_in_metres(estimate(r)), retcode(r), iterations(r), rms(r), r.n, metadata(r))

# The plane-model estimators (homography, radial homography, radial camera, the
# division camera) return a matrix model whose entries carry the unit of the
# board plane, and a matrix cannot declare it. Physical plane points are refused
# with the fix named, rather than read in a unit the model would not record —
# VisualGeometryToolkitCore's ruling for `HomographyMat(camera)`.
_point_has_units(::Type{P}) where {P} = _point_scalar(P) <: Quantity
function _refuse_physical_plane(cs, fname::AbstractString)
    _point_has_units(eltype(sources(cs))) && throw(
        ArgumentError(
            "$fname: the plane points are lengths, and the model it returns cannot record their unit; " *
                "strip them in the unit you want the model in (`ustrip.(mm, X)`) and pass plain numbers"
        )
    )
    return cs
end
