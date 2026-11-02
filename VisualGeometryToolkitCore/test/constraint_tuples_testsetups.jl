# A constraint tuple `(g, G_x, G_θ)` scored through the constraint-problem
# protocol: a problem that IS its list of tuples, with optional per-point noise
# (`measurement_covariance`) and rounding scales (`constraint_rounding_scale`).
# The public whitened residual and rows of a bare tuple are those of a one-point
# problem holding it.

@testsetup module ConstraintTuples
    using VisualGeometryToolkitCore: AbstractConstraintProblem
    import VisualGeometryToolkitCore as VgtkCore
    export CDProblem, whitened_of, rows_of, distance_of

    struct CDProblem{C, S, R} <: AbstractConstraintProblem
        cds::Vector{C}
        Σs::S
        ρs::R
    end
    CDProblem(cds::AbstractVector) = CDProblem(collect(cds), nothing, nothing)
    VgtkCore.data_size(p::CDProblem) = length(p.cds)
    VgtkCore.codimension(p::CDProblem) = length(first(p.cds)[1])
    VgtkCore.minimal_sample_size(::CDProblem) = 2
    VgtkCore.constraint_with_jacobians(p::CDProblem, _model, i::Int) = p.cds[i]
    VgtkCore.measurement_covariance(p::CDProblem{<:Any, <:AbstractVector}, i::Int) = p.Σs[i]
    VgtkCore.constraint_rounding_scale(p::CDProblem{<:Any, <:Any, <:AbstractVector}, _model, i::Int) = p.ρs[i]

    _one(cd, Σ, ρ) = CDProblem([cd], Σ === nothing ? nothing : [Σ], ρ === nothing ? nothing : [ρ])
    whitened_of(cd; Σ = nothing, ρ = nothing) = VgtkCore.whitened_constraint_residual(_one(cd, Σ, ρ), nothing, 1)
    rows_of(cd; Σ = nothing, ρ = nothing) = VgtkCore.whitened_constraint_rows(_one(cd, Σ, ρ), nothing, 1)
    distance_of(cd; kw...) = VgtkCore.whitened_distance(whitened_of(cd; kw...))
end
