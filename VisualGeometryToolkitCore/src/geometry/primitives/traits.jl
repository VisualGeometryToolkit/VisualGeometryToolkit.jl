# ==============================================================================
# Trait Implementations for Geometric Primitives
# ==============================================================================
#
# This file collects all Holy Trait implementations for primitives in one place.
# Traits are defined in base/traits.jl; implementations are here.
#
# NOTE: This file is included AFTER all primitive types are defined.
# ==============================================================================

# -----------------------------------------------------------------------------
# HasBoundary Traits
# -----------------------------------------------------------------------------
# Note: boundary_point methods are defined in the respective type files

has_boundary(::Ellipse) = HasBoundary()
has_boundary(::Circle) = HasBoundary()
has_boundary(::FeatureFrame) = HasBoundary()
has_boundary(::Rect) = HasBoundary()

# -----------------------------------------------------------------------------
# Hyperplane Traits
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# HasArea Traits
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# HasInterior Traits (for containment tests)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# HasOrientation Traits
# -----------------------------------------------------------------------------
# HasOrientation indicates a WELL-DEFINED, semantically meaningful orientation
# that can be used to map to/from a canonical frame unambiguously.
#
# Ellipse stores θ but this is just a parameterization - the same ellipse shape
# can be represented with different θ values. Mapping back to canonical frame
# is ill-defined because the decomposition R(θ)*Scale(a,b) is not unique.
#
# OrientedDisc/EllipticFrame/AffineFrame have orientation from detection
# (e.g., dominant gradient direction), so canonical frame mapping is well-defined.


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# FrameScale Traits (isotropic vs shape-matrix frames)
# -----------------------------------------------------------------------------

frame_scale(::Disc) = IsotropicScale()
frame_scale(::OrientedDisc) = IsotropicScale()
frame_scale(::EllipticFrame) = ShapeMatrixScale()
frame_scale(::AffineFrame) = ShapeMatrixScale()
