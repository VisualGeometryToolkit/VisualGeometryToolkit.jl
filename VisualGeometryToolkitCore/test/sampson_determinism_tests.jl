@testitem "sampson determinism: the scorer's residual is the Jacobian row's, bit for bit" begin
    using Test, StaticArrays, Random, LinearAlgebra
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: sampson_residual_with_jacobian, s2_tangent_basis_matrix, skew
    const VgtkCore = VisualGeometryToolkitCore

    # The residual a scorer reads (`sampson_residual`, the batch rows) and the residual the LM's Jacobian row
    # carries (`sampson_residual_with_jacobian`, `_sampson_jacobian_terms`) are one arithmetic: every product of
    # the constraint and of the whitening is a stated fma chain (base/fused_products.jl), inlined, so no compiled
    # copy contracts differently. Before the pin an essential matrix's two residuals were apart in their last bits
    # on 243 of 300 bearing pairs (Core 0.14, the residual's `dot(d, E s)` contracted in one copy and not the other).
    rng = Xoshiro(11)
    n = 600
    R = SMatrix{3, 3}(exp(skew(SVector(0.1, -0.2, 0.05))))
    E = EssentialMat(Tuple(skew(normalize(SVector(1.0, 0.2, -0.1))) * R))
    f₁ = [normalize(SVector(0.6randn(rng), 0.6randn(rng), 1.0)) for _ in 1:n]
    f₂ = [normalize(R * f + SVector(0.3, 0.05, -0.02) + 1.0e-3 * randn(rng, SVector{3})) for f in f₁]
    M₁ = [s2_tangent_basis_matrix(f) * (1 / 800) for f in f₁]; M₂ = [s2_tangent_basis_matrix(f) * (1 / 650) for f in f₂]
    F = FundMat(SMatrix{3, 3}(randn(rng, 3, 3)))
    H = HomographyMat(Tuple(SMatrix{3, 3}(1.0, 0.01, 1.0e-5, -0.02, 1.0, 2.0e-5, 3.0, -2.0, 1.0)))
    u₁ = [SVector(640 * rand(rng), 480 * rand(rng)) for _ in 1:n]
    u₂ = [u + SVector(5randn(rng), 5randn(rng)) for u in u₁]
    sqrtw = rand(rng, n) .+ 0.1
    D9(N) = SMatrix{9, N}(randn(rng, 9N))
    Σ = SMatrix{4, 4}(Diagonal(SVector(1.5, 0.7, 2.0, 0.9))) + 0.1 * ones(SMatrix{4, 4})
    cases = (
        ("E on bearings", E, (f₁, f₂), i -> (; M₁ = M₁[i], M₂ = M₂[i]), D9(5),
            withattrs(Correspondences(f₁, f₂); M₁, M₂, sqrt_weights = sqrtw)),
        ("F", F, (u₁, u₂), i -> NamedTuple(), D9(7), withattrs(Correspondences(u₁, u₂); sqrt_weights = sqrtw)),
        ("H", H, (u₁, u₂), i -> NamedTuple(), D9(8), withattrs(Correspondences(u₁, u₂); sqrt_weights = sqrtw)),
    )
    for (label, M, (xs₁, xs₂), kw, D, xs) in cases
        @testset "$label" begin
            dg = VgtkCore.constraint_dim(M)
            B = VgtkCore._direction_rows(D)
            s_score = [sampson_residual(M, xs₁[i], xs₂[i]; kw(i)...) for i in 1:n]
            s_jac = [first(sampson_residual_with_jacobian(M, xs₁[i], xs₂[i]; directions = D, kw(i)...)) for i in 1:n]
            @test all(i -> s_score[i] === s_jac[i], 1:n)
            # the batch rows (one kernel per set) against the Jacobian terms: r = √w·s, the same product
            r = zeros(dg * n)
            VgtkCore.weighted_whitened_residuals!(r, M, xs)
            @test all(i -> all(k -> r[dg * (i - 1) + k] === sqrtw[i] * s_jac[i][k], 1:dg), 1:n)
            # at an observation noise Σ: the whitened residual of the rows and of the Jacobian terms
            r_Σ = [first(VgtkCore._element_residual(M, Σ, xs₁[i], xs₂[i]; kw(i)...)) for i in 1:n]
            j_Σ = [first(VgtkCore._sampson_jacobian_terms(M, B, Σ, xs₁[i], xs₂[i]; kw(i)...)) for i in 1:n]
            @test all(i -> r_Σ[i] === j_Σ[i], 1:n)
        end
    end
end

@testitem "sampson determinism: the rows hold no contractible operation and no product call" begin
    using Test, StaticArrays, Random, LinearAlgebra
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: skew
    const VgtkCore = VisualGeometryToolkitCore

    # The typed code of the Sampson rows, followed into every call it keeps: no `muladd_float` (an operation LLVM
    # may fuse, or not, per compiled copy) and no call to a fused product (each inlines, base/fused_products.jl).
    # Base's `fma` keeps its software fallback as a call in its untaken branch; that one is not counted.
    _resolve(x) = x isa GlobalRef ? (isdefined(x.mod, x.name) ? getglobal(x.mod, x.name) : nothing) : x
    function audit!(found, @nospecialize(tt), seen = Set{Any}())
        tt in seen && return found
        push!(seen, tt)
        for (ci, _) in Base.code_typed_by_type(tt; optimize = true)
            ci isa Core.CodeInfo || continue
            for st in ci.code
                if Meta.isexpr(st, :call) && _resolve(st.args[1]) === Core.Intrinsics.muladd_float
                    push!(found, "muladd in $(tt.parameters[1])")
                elseif Meta.isexpr(st, :invoke)
                    mi = st.args[1]; mi isa Core.CodeInstance && (mi = mi.def)
                    f = string(mi.specTypes.parameters[1])
                    (occursin("throw", f) || occursin("fma_emulated", f)) && continue
                    occursin("_fma_", f) && push!(found, "call of $(mi.specTypes)")
                    audit!(found, mi.specTypes, seen)
                end
            end
        end
        return found
    end
    rng = Xoshiro(3)
    f₁ = normalize(SVector(0.1, -0.2, 1.0)); f₂ = normalize(SVector(0.12, -0.18, 1.0))
    M₁ = SMatrix{3, 2}(randn(rng, 3, 2)); M₂ = SMatrix{3, 2}(randn(rng, 3, 2))
    u₁ = SVector(310.0, 220.0); u₂ = SVector(305.0, 231.0)
    E = EssentialMat(Tuple(skew(normalize(SVector(1.0, 0.2, -0.1)))))
    F = FundMat(SMatrix{3, 3}(randn(rng, 3, 3)))
    H = HomographyMat(Tuple(SMatrix{3, 3}(1.0, 0.01, 1.0e-5, -0.02, 1.0, 2.0e-5, 3.0, -2.0, 1.0)))
    Σ = SMatrix{4, 4}(1.0I)
    for (M, xs, x, kw, N) in ((E, withattrs(Correspondences([f₁], [f₂]); M₁ = [M₁], M₂ = [M₂], sqrt_weights = [1.0]), (f₁, f₂), (; M₁, M₂), 5),
                              (F, withattrs(Correspondences([u₁], [u₂]); sqrt_weights = [1.0]), (u₁, u₂), NamedTuple(), 7),
                              (H, withattrs(Correspondences([u₁], [u₂]); sqrt_weights = [1.0]), (u₁, u₂), NamedTuple(), 8))
        B = SMatrix{N, 9}(randn(rng, 9N))
        cols, kws = VgtkCore._observation_columns(xs), VgtkCore.constraint_keywords(M, xs)
        res = Tuple{typeof(VgtkCore._sampson_residual_row!), Vector{Float64}, Int, typeof(M), typeof(cols), typeof(kws),
            Nothing, Vector{Float64}}
        jac = Tuple{typeof(VgtkCore._sampson_jacobian_row!), Matrix{Float64}, Int, typeof(M), typeof(B), typeof(cols),
            typeof(kws), Nothing, Vector{Float64}}
        @test isempty(audit!(String[], res))
        @test isempty(audit!(String[], jac))
        # at an observation noise Σ: the whitened residual and its Jacobian terms
        er(M, Σ, x, kw) = VgtkCore._element_residual(M, Σ, x...; kw...)
        jt(M, B, Σ, x, kw) = VgtkCore._sampson_jacobian_terms(M, B, Σ, x...; kw...)
        @test isempty(audit!(String[], Tuple{typeof(er), typeof(M), typeof(Σ), typeof(x), typeof(kw)}))
        @test isempty(audit!(String[], Tuple{typeof(jt), typeof(M), typeof(B), typeof(Σ), typeof(x), typeof(kw)}))
    end
end
