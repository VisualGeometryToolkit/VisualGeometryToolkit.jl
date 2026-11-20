# =============================================================================
# Retired names (VisualGeometryToolkitCore 0.12.0): the radiometric types moved
# to VisualRadiometryToolkitCore
# =============================================================================
#
# Radiometry lifts geometry: the Visual Radiometry Toolkit depends on this
# package, never the reverse. A camera's vignetting fields, the measured radial
# profile and the scene's light (the files cameras/vignetting.jl and
# cameras/lighting.jl up to 0.11) are VisualRadiometryToolkitCore's, under the
# same names with the same behaviour. Each name is refused here by name with its
# replacement: no alias, no forwarding, not exported (so `using` both packages
# binds VisualRadiometryToolkitCore's). A name read as a type or a constant
# (`AbstractVignetting` in a signature, `VIGNETTING_TAGS[tag]`) is a function
# here, and fails where it is used.
# =============================================================================

_moved_to_radiometry(name::Symbol) = throw(ArgumentError(
    "VisualGeometryToolkitCore.$name is retired (0.12.0): it moved to VisualRadiometryToolkitCore, " *
    "under the same name — `using VisualRadiometryToolkitCore: $name`"))

"""
    MOVED_NAMES

The names this package retired by moving them to another package: each is a
refusal here (an `ArgumentError` naming its replacement), not a definition. The
duplication check reads it (`check_duplication`'s name ownership): a module
does not own a name it lists in its `MOVED_NAMES`, so the package that took
the name over is not reported for defining it. Any stack package may define
its own.
"""
const MOVED_NAMES = (:AbstractVignetting, :TensorProductVignetting, :RadialVignetting, :LogVignetting, :log_spline, :VIGNETTING_TAGS,
             :RADIAL_CHARTS, :BinnedRadialProfile, :projective_chart, :projective_knots, :radial_chart, :radial_coordinate,
             :field_radius, :with_center, :center_gradient, :pole_regular_basis, :coefficient_jacobian, :active_coefficients,
             :control_points, :with_control_points, :n_control_points, :knot_vectors, :spline_degree, :EnvironmentLight,
             :SphericalHarmonicMap, :radiance, :exit_direction, :radiance_map, :coefficients, :with_coefficients,
             :max_degree, :sh_basis, :sh_count, :sh_index)

for name in MOVED_NAMES
    @eval begin
        @doc $("Retired (VisualGeometryToolkitCore 0.12.0): `VisualRadiometryToolkitCore.$name`.")
        $name(args...; kwargs...) = _moved_to_radiometry($(QuoteNode(name)))
    end
end
