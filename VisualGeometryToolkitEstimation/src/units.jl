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
_metres(x::Union{Real, Quantity}) = VgtkCore.strip_metres(x)

# A physical value as plain metres.
_plain_metres(E::_PhysicalPose3) = ustrip(VgtkCore.m, E)
_plain_metres(X::_PhysicalPoint3) = VgtkCore.strip_metres(X)
_plain_metres(r::HyperRectangle{2, <:Quantity}) = ustrip(VgtkCore.m, r)

# A plain pose read as metres: Core's `with_unit`, the one attachment — the
# rotation (of any parameterization) and the frames kept exactly.
_in_metres(E::EuclideanMap{3, T, R, <:StaticVector{3, <:Real}}) where {T, R} = VgtkCore.with_unit(E, VgtkCore.m)

# A correspondence set with its endpoints mapped: a `Csponds` keeps its
# attribute columns (a PROSAC `score`, `weights`), so the plain set samples and
# weighs exactly as the physical one would.
_map_csponds(fs, ft, cs::VgtkCore.Csponds) =
    VgtkCore.Csponds(map(fs, sources(cs)), map(ft, targets(cs)), VgtkCore.attrs_of(cs))
_map_csponds(fs, ft, cs::AbstractVector{<:Pair}) = [fs(first(c)) => ft(last(c)) for c in cs]

# A result with `f` applied to its estimate, everything else as it was. A
# RANSAC result's estimate is `Union{M, Nothing}` (`RansacEstimate`): `f` maps a
# present model, an absent one stays `nothing`, and the result keeps the union
# over `f`'s image of `M`. Any other fit's estimate is concrete and simply mapped.
_map_estimate(f, r::FitResult) = FitResult(f(estimate(r)), retcode(r), iterations(r), rms(r), r.n, metadata(r))
function _map_estimate(f, r::FitResult{Union{M, Nothing}, R}) where {M, R}
    N = Base.promote_op(f, M)
    E = estimate(r)
    return FitResult{Union{N, Nothing}, R}(E === nothing ? nothing : f(E), retcode(r), iterations(r), rms(r), r.n, metadata(r))
end

# A RANSAC (or any) result with its model read as metres: everything else as it was.
_in_metres(r::FitResult) = _map_estimate(_in_metres, r)

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
                "strip them in the unit the rest of your plain data uses — metres, as the stack's boundaries strip " *
                "(`ustrip.(m, X)`) — and supply plain numbers"
        )
    )
    return cs
end
