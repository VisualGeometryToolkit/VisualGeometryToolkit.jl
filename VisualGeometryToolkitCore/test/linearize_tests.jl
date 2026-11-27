# `linearize(f, u0; free, classes)`: the dense Gauss–Newton normal equations of a bundle over a selection of its
# variables, per class of residual blocks (robust/refine.jl). A small bundle of three variables and two block types
# (data rows on all three, penalty rows on one): the accumulation against a reference loop that visits the blocks in
# NLLSsolver's order and adds each block's `JᵀJ`, `Jᵀr` at its columns (bits); the values against the bundle's dense
# Jacobian and its rows (`_nlls_rows_and_jacobian`); the class split, a variable subset, the default selection (the
# variables `fixed_variables` leaves free), `prepare_evaluation!` and the visit order.

@testsetup module LinearizeFixture
    using StaticArrays
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: NLLSsolver
    export LData, LPenalty, LBundle, EVENTS, bundle, reference

    const EVENTS = Any[]      # (:told, variables, varnext), (:read, vector), (:data | :penalty, index), in order
    const V2 = SVector{2, Float64}
    const V1 = SVector{1, Float64}

    # a data block: two rows on the variables a (2), b (1) and c (1), differentiated by NLLSsolver
    struct LData <: NLLSsolver.AbstractResidual
        k::Int; s::Float64; y::Float64
    end
    NLLSsolver.ndeps(::LData) = NLLSsolver.static(3)
    NLLSsolver.nres(::LData) = NLLSsolver.static(2)
    NLLSsolver.varindices(::LData) = SVector(1, 2, 3)
    function NLLSsolver.getvars(::LData, vars::Vector)
        push!(EVENTS, (:read, vars))
        return (vars[1]::V2, vars[2]::V1, vars[3]::V1)
    end
    Base.eltype(::LData) = Float64
    function NLLSsolver.computeresidual(r::LData, a, b, c)
        push!(EVENTS, (:data, r.k))
        s = r.s
        return SVector(a[1] + a[2] * s + b[1] * s^2 + c[1] * s^3 - r.y, sin(a[1] * s) + b[1] * c[1] - 0.1 * r.y)
    end
    # a penalty block: one row on one variable (b or c)
    struct LPenalty <: NLLSsolver.AbstractResidual
        k::Int; v::Int; w::Float64
    end
    NLLSsolver.ndeps(::LPenalty) = NLLSsolver.static(1)
    NLLSsolver.nres(::LPenalty) = NLLSsolver.static(1)
    NLLSsolver.varindices(r::LPenalty) = SVector(r.v)
    function NLLSsolver.getvars(r::LPenalty, vars::Vector)
        push!(EVENTS, (:read, vars))
        return (vars[r.v]::V1,)
    end
    Base.eltype(::LPenalty) = Float64
    function NLLSsolver.computeresidual(r::LPenalty, x)
        push!(EVENTS, (:penalty, r.k))
        return SVector(r.w * (x[1] - 0.25))
    end

    struct LBundle <: VgtkCore.AbstractBlockProblem
        blocks::Vector{Any}
        held::Union{Nothing, Vector{Int}}
    end
    VgtkCore.block_variables(::LBundle, u0) = Any[V2(u0[1], u0[2]), V1(u0[3]), V1(u0[4])]
    VgtkCore.residual_blocks(f::LBundle, _u0) = f.blocks
    VgtkCore.fixed_variables(f::LBundle) = f.held
    # the bundle's record of the vectors it is told: the call, in the event log, and the vectors themselves
    function VgtkCore.prepare_evaluation!(::LBundle, variables, varnext)
        told = (:told, variables, varnext)
        push!(EVENTS, told)
        return nothing
    end

    # the blocks as added: data and penalty interleaved (the visit order groups them by type)
    function bundle(; held = nothing)
        blocks = Any[]
        for k in 1:6
            s = 0.4 * k
            push!(blocks, LData(k, s, 1.0 + 0.5 * s - 0.3 * s^2 + 0.05 * sin(7k)))
            isodd(k) && push!(blocks, LPenalty(k, isodd(k ÷ 2) ? 2 : 3, 0.7 + 0.1k))
        end
        return LBundle(blocks, held)
    end

    # The reference accumulation: the blocks in NLLSsolver's order (by type in their Union's order, then as added),
    # each evaluated over its selected variables (every one: the static flags 2ⁿ − 1; some: their mask, static; none:
    # the residual alone), its JᵀJ and −Jᵀr added at the selected columns (ascending variable index), per class.
    utypes(U::Union) = (utypes(U.a)..., utypes(U.b)...)
    utypes(T::Type) = (T,)
    function reference(f, u0, free, classes)
        variables = Vector{Union{V2, V1}}(VgtkCore.block_variables(f, u0))
        blocks = f.blocks
        U = Union{unique(typeof.(blocks))...}
        order = Any[c for T in utypes(U) for c in blocks if typeof(c) === T]
        off = Dict{Int, Int}(); n = 0
        for v in sort(collect(free))
            off[v] = n; n += Int(NLLSsolver.dynamic(NLLSsolver.nvars(variables[v])))
        end
        out = [(; G = zeros(n, n), h = zeros(n), S = Ref(0.0), nrows = Ref(0)) for _ in classes]
        for c in order
            k = something(findfirst(C -> c isa C, classes), 0)
            k == 0 && continue
            vi = NLLSsolver.varindices(c); vars = NLLSsolver.getvars(c, variables)
            flags = 0; cols = Int[]
            for (j, v) in enumerate(vi)
                haskey(off, v) || continue
                flags |= 1 << (j - 1); append!(cols, off[v] .+ (1:Int(NLLSsolver.dynamic(NLLSsolver.nvars(vars[j])))))
            end
            res, J = flags == 0 ? (NLLSsolver.computeresidual(c, vars...), nothing) :
                     NLLSsolver.computeresjac(NLLSsolver.static(flags), c, vars...)
            a = out[k]
            a.S[] += sum(abs2, res); a.nrows[] += length(res)
            J === nothing && continue
            a.G[cols, cols] .+= J' * J
            a.h[cols] .-= J' * res
        end
        return Tuple((; a.G, a.h, S = a.S[], nrows = a.nrows[]) for a in out)
    end
end

@testitem "linearize: bit for bit the per-block accumulation, and the bundle's Jacobian" setup = [LinearizeFixture] begin
    using Test, StaticArrays, LinearAlgebra
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: linearize, NLLSsolver
    f = bundle(); u0 = (0.9, 0.4, -0.2, 0.1)
    bits(x, y) = size(x) == size(y) && all(isequal.(x, y))
    same(a, b) = bits(a.G, b.G) && bits(a.h, b.h) && a.S === b.S && a.nrows === b.nrows
    for free in ([1, 2, 3], [2, 3], [1], [3], [1, 3]), classes in ((Any,), (LData, LPenalty), (LPenalty,))
        got = linearize(f, u0; free, classes)
        ref = reference(f, u0, free, classes)
        @test length(got) == length(classes)
        @test all(same(g, r) for (g, r) in zip(got, ref))
    end
    # the values: the bundle's dense Jacobian at the guess (every variable free, the rows in block order), its
    # selected columns, the class sums
    variables = Vector{Union{SVector{2, Float64}, SVector{1, Float64}}}(VgtkCore.block_variables(f, u0))
    r, Js = VgtkCore._nlls_rows_and_jacobian(f.blocks, variables)
    J = Matrix(Js)
    (; G, h, S, nrows) = only(linearize(f, u0; free = [1, 2, 3]))
    @test G ≈ J' * J rtol = 1e-12
    @test h ≈ -J' * r rtol = 1e-12
    @test S ≈ sum(abs2, r) rtol = 1e-14
    @test nrows == length(r) == 6 * 2 + 3
    sub = only(linearize(f, u0; free = [3, 1]))           # ascending variable index: a's 2 columns, then c's
    cs = [1, 2, 4]
    @test size(sub.G) == (3, 3)
    @test sub.G ≈ J[:, cs]' * J[:, cs] rtol = 1e-12
    @test sub.h ≈ -J[:, cs]' * r rtol = 1e-12
    @test sub.S ≈ S rtol = 1e-14
    @test sub.nrows == nrows                              # every block counts its rows, selected or not
    # a forward difference of the rows (the Jacobian itself, not only NLLSsolver's)
    function rows_at(u)
        vs = Vector{Union{SVector{2, Float64}, SVector{1, Float64}}}(VgtkCore.block_variables(f, u))
        return first(VgtkCore._nlls_rows_and_jacobian(f.blocks, vs))
    end
    ϵ = 1e-7
    Jfd = reduce(hcat, [(rows_at(ntuple(j -> u0[j] + (j == i) * ϵ, 4)) - rows_at(u0)) / ϵ for i in 1:4])
    @test G ≈ Jfd' * Jfd rtol = 1e-5
    # the class split: the two classes' systems add up to the one class's, and a class outside every block is empty
    d, p = linearize(f, u0; free = [2, 3], classes = (LData, LPenalty))
    one_ = only(linearize(f, u0; free = [2, 3]))
    @test d.G + p.G ≈ one_.G rtol = 1e-14
    @test d.h + p.h ≈ one_.h rtol = 1e-14
    @test d.S + p.S ≈ one_.S rtol = 1e-14
    @test d.nrows == 12 && p.nrows == 3
    @test p.G[1, 2] == 0 && p.G[1, 1] > 0 && p.G[2, 2] > 0   # each penalty reads one variable
    none = only(linearize(f, u0; free = [2, 3], classes = (Missing,)))
    @test iszero(none.G) && iszero(none.h) && none.S === 0.0 && none.nrows === 0
    # an empty selection: no columns, the rows still counted; an index outside the variables refused
    e = only(linearize(f, u0; free = Int[]))
    @test size(e.G) == (0, 0) && e.nrows == nrows
    @test e.S ≈ S rtol = 1e-14
    @test_throws ArgumentError linearize(f, u0; free = [4])
    @test_throws ArgumentError linearize(f, u0; free = [0])
end

@testitem "linearize: the default selection, prepare_evaluation!, NLLSsolver's visit order, inference" setup = [LinearizeFixture] begin
    using Test, StaticArrays
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: linearize, NLLSsolver
    u0 = (0.9, 0.4, -0.2, 0.1)
    # the default `free`: every variable fixed_variables does not hold
    held = bundle(; held = [1])
    @test only(linearize(held, u0)) == only(linearize(bundle(), u0; free = [2, 3]))
    @test only(linearize(bundle(), u0)) == only(linearize(bundle(), u0; free = 1:3))
    @test_throws ArgumentError linearize(bundle(; held = [5]), u0)
    # prepare_evaluation! names one vector (as both) before the first evaluation, and every evaluation reads it
    f = bundle()
    empty!(EVENTS)
    linearize(f, u0; free = [2], classes = (LData, LPenalty))
    told = findall(e -> e[1] === :told, EVENTS)
    @test length(told) == 1
    t = EVENTS[only(told)]
    @test t[2] === t[3] && t[2] isa Vector && length(t[2]) == 3
    reads = [e for e in EVENTS[(only(told) + 1):end] if e[1] === :read]
    @test length(reads) == length(f.blocks) && all(e -> e[2] === t[2], reads)
    @test t[2] !== f.blocks && all(e -> e[1] !== :data && e[1] !== :penalty, EVENTS[1:only(told)])   # no evaluation before it
    # the visit order: by type in the blocks' Union order (NLLSsolver's repository), then as added
    U = Union{unique(typeof.(f.blocks))...}
    types = NLLSsolver.uniontotuple(U)
    evaluated = [e for e in EVENTS if e[1] in (:data, :penalty)]
    kinds = [first(e) for e in evaluated]
    expected = [T === LData ? :data : :penalty for T in types for c in f.blocks if typeof(c) === T]
    @test kinds == expected
    @test [last(e) for e in evaluated if first(e) === :data] == 1:6
    @test [last(e) for e in evaluated if first(e) === :penalty] == [1, 3, 5]
    # a block of no class is not evaluated
    empty!(EVENTS)
    linearize(f, u0; classes = (LPenalty,))
    @test all(e -> e[1] !== :data, EVENTS)
    # inferred
    @test @inferred(linearize(f, u0; free = [2, 3], classes = (LData, LPenalty))) isa
          NTuple{2, @NamedTuple{G::Matrix{Float64}, h::Vector{Float64}, S::Float64, nrows::Int}}
end
