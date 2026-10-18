# `Uncertain` is the alias `Attributed{V, <:Union{CovMat, PrecMat}}`: one
# value-with-attributes type. What the alias keeps (its constructors, `.matrix`,
# cov/prec/geom/centroid, the frame and convert paths) and what it now shares
# with every Attributed value (fields, StructArrays, index forwarding).
@testitem "Uncertain is an Attributed" begin
    using Test, LinearAlgebra
    using StaticArrays: SMatrix, SVector, @SMatrix
    using StructArrays: StructArray
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: Attributed, Uncertain, CovMat, PrecMat, Point2, UnlabeledFrame, ImageFrame,
        cov, prec, geom, centroid, relabel, Ellipse, propagate
    using StaticArrays: SVector

    Σ = CovMat{2, Float64}(4.0, 1.0, 1.0, 9.0)
    Λ = PrecMat{2, Float32}(2.0f0, 0.1f0, 0.1f0, 3.0f0)
    p = Point2(100.0, 200.0)

    # one type: the alias and Attributed coincide, the bound is the statistical forms
    @test Uncertain(p, Σ) === Attributed(p, Σ)
    @test Uncertain{Point2{Float64}, typeof(Σ)} === Attributed{Point2{Float64}, typeof(Σ)}
    @test Attributed(p, Λ) isa Uncertain
    @test !(Attributed(p, 1.0) isa Uncertain)
    @test !(Attributed(p, (; σ = 1.0)) isa Uncertain)
    @test_throws TypeError Uncertain{Point2{Float64}, Float64}

    # the one constructor: a CovMat or a PrecMat, nothing else — a bare matrix or
    # a scalar does not say which form is stored
    u = Uncertain(p, CovMat(@SMatrix [0.1 0.0; 0.0 0.2]))
    @test getfield(u, :attributes) isa CovMat{2, Float64, 4, UnlabeledFrame{2}}
    @test_throws MethodError Uncertain(p, @SMatrix [0.1 0.0; 0.0 0.2])
    @test_throws MethodError Uncertain(p, 0.5)
    @test_throws MethodError Uncertain(p, (; σ = 0.5))
    @test !(Attributed(p, @SMatrix [0.1 0.0; 0.0 0.2]) isa Uncertain)   # a plain Attributed, never uncertain
    @test_throws MethodError cov(Attributed(p, @SMatrix [0.1 0.0; 0.0 0.2]))
    @test_throws TypeError Uncertain{typeof(p), SMatrix{2, 2, Float64, 4}}

    # property access: `.matrix` names the form (and only for a statistical form);
    # the form's storage is never forwarded as if it were a record
    d = Uncertain(p, Λ)
    @test d.matrix === Λ === d.attributes && d.value === p
    @test propertynames(d) == (:value, :attributes, :matrix)
    @test d.data === p.data                                # forwarded to the value, not the matrix
    @test !hasproperty(Attributed(p, 1.0), :matrix)
    @test (Attributed(p, (; matrix = 7)).matrix) == 7      # a record attribute's own field

    # the accessors: the stored form directly, the other by one inversion
    @test cov(Uncertain(p, Σ)) === Σ && prec(Uncertain(p, Σ)) == inv(Σ)
    @test prec(d) === Λ && cov(d) == inv(Λ)
    @test geom(d) === p && geom(Attributed(p, 1.0)) === Attributed(p, 1.0)

    # centroid: identity on the point, recursion on a nested value — Attributed's
    @test centroid(d) === p
    @test centroid(Uncertain(Uncertain(p, Σ), Λ)) === p
    @test centroid(Uncertain(Attributed(p, 1.0), Σ)) === p

    # Attributed's vector forwarding now reaches an uncertain point
    @test d[1] == 100.0 && length(d) == 2 && size(d) == (2,)

    # StructArrays: the columnar decomposition is Attributed's (value, attributes)
    sa = StructArray([Uncertain(Point2(Float64(i), 0.0), Σ) for i in 1:3])
    @test sa.value == [Point2(Float64(i), 0.0) for i in 1:3] && all(==(Σ), sa.attributes)
    @test sa[2] === Uncertain(Point2(2.0, 0.0), Σ)

    # the storage-boundary convert (labels an unlabeled form; containers typed on
    # the historical 3-parameter surface)
    v = Vector{Uncertain{Point2{Float32}, PrecMat{2, Float32, 4}}}(undef, 1)
    v[1] = Uncertain(Point2{Float32}(1, 2), Λ)
    @test geom(v[1]) == Point2{Float32}(1, 2)
    ul = convert(Uncertain{Point2{Float64}, CovMat{2, Float64, 4, ImageFrame{:a}}}, Uncertain(p, Σ))
    @test getfield(ul, :attributes) isa CovMat{2, Float64, 4, ImageFrame{:a}}

    # Ellipse: one method, the same ellipse from either stored form
    eΣ = Ellipse(Uncertain(p, Σ); sigma = 2)
    eΛ = Ellipse(Uncertain(p, PrecMat{2, Float64}(Tuple(inv(SMatrix{2, 2, Float64}(Σ))))); sigma = 2)
    @test eΣ.center == p && eΣ.a ≈ eΛ.a && eΣ.b ≈ eΛ.b

    # the keyword-only σ constructors of the forms (the retired σ shorthand's replacement)
    for s in (0.5, 2.0f0, 3)
        Σs = CovMat{2}(σ = s); Λs = PrecMat{2}(σ = s)
        F = float(typeof(s))
        @test Σs isa CovMat{2, F, 4, UnlabeledFrame{2}} && Λs isa PrecMat{2, F, 4, UnlabeledFrame{2}}
        @test SMatrix{2, 2}(Σs) == F(s)^2 * SMatrix{2, 2, F}(I) && isbits(Σs) && isbits(Λs)
        @test SMatrix{2, 2}(inv(Λs)) ≈ SMatrix{2, 2}(Σs)
        @test SMatrix{2, 2}(cov(Uncertain(p, Λs))) ≈ SMatrix{2, 2}(Σs)
        @test @inferred(CovMat{2}(σ = s)) == Σs && @inferred(PrecMat{2}(σ = s)) == Λs
    end
    @test SMatrix{3, 3}(CovMat{3}(σ = (1.0, 2.0, 3.0))) == SMatrix{3, 3}(Diagonal(SVector(1.0, 4.0, 9.0)))
    @test SMatrix{2, 2}(PrecMat{2}(σ = SVector(0.5, 2.0))) == SMatrix{2, 2}(Diagonal(SVector(4.0, 0.25)))
    @test @inferred(CovMat{3}(σ = SVector(1.0, 2.0, 3.0))) isa CovMat{3, Float64, 9}
    @test_throws MethodError CovMat{3}(σ = (1.0, 2.0))          # the diagonal has N entries
    @test_throws ArgumentError CovMat{2}(σ = 0.0)
    @test_throws ArgumentError PrecMat{2}(σ = -1.0)
    @test_throws DimensionMismatch CovMat{2}(0.5)               # no positional scalar (σ or variance?): StaticArrays refuses it
    @test relabel(ImageFrame{:a}, CovMat{2}(σ = 0.5)) isa CovMat{2, Float64, 4, ImageFrame{:a}}
end
