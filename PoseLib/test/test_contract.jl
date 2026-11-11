# The robust-estimation contract on PoseLib's refiners (owner rulings, 2026-11-11):
# the strategy is `optimization_strategy` (Core's `Irls{K}`, PoseLib's `LeZach()`,
# `nothing` for least squares), the width `scale` or `width`, the result a
# `FitResult` carrying PoseLib's `retcode`/`iterations` and the scale record.
# The renaming changes no number: every PoseLib baseline arm of the stack
# (LocalOptimizationDoneRight's PoseLibLm/PoseLibLZ/PoseLibLmAdaptLossWidth
# arms, accv2026's least-squares polish) refines bit-identically.

import VisualGeometryToolkitCore as VgtkCore
using Test
import TOML
using Random: MersenneTwister
using LinearAlgebra: normalize
using Rotations: AngleAxis, RotMatrix
using StaticArrays: SVector, SMatrix
using PoseLib
using PoseLib: LeZach, PoseLibRefineMetadata
using VisualGeometryToolkitCore: Csponds, withattrs, backproject_with_jacobian, CameraModel, LogicalIntrinsics,
    CameraCalibrationMat, Pinhole, Vec3, TLSShape, HuberShape, CauchyShape, GaussianShape, WelschShape,
    Irls, Triggs, EfficiencyConstant, LossWidth, FitResult, estimate, retcode, iterations, noise_scale, scale_source,
    ScaleGiven, ScaleUpstream, ScaleNone, efficiency_constant

# The geometry of test_refine.jl, one draw per refiner (outliers from rng 99).
function _baseline_data()
    _nrm(Y) = SVector(Y[1] / Y[3], Y[2] / Y[3])
    _proj(K, Y) = _nrm(K * Y)
    rng = MersenneTwister(7)
    n = 200; nout = n ÷ 5
    K1 = SMatrix{3,3}(800.0, 0, 0, 0, 800.0, 0, 320.0, 240.0, 1.0)
    K2 = SMatrix{3,3}(760.0, 0, 0, 0, 760.0, 0, 330.0, 250.0, 1.0)
    R = SMatrix{3,3,Float64,9}(RotMatrix(AngleAxis(0.2, normalize([0.3, 1.0, 0.1])...)))
    t = normalize(SVector(1.0, 0.1, 0.05))
    X = [SVector(4 * randn(rng), 3 * randn(rng), 8.0 + 4 * rand(rng)) for _ in 1:n]
    Rs = SMatrix{3,3,Float64,9}(RotMatrix(AngleAxis(0.01, 0.0, 1.0, 0.0))) * R
    seed = PoseLib.Rigid3(RotMatrix{3}(Rs), Vec3{Float64}(normalize(t + SVector(0.0, 0.02, 0.0))))
    r2 = MersenneTwister(99)
    out2() = SVector(640 * rand(r2), 480 * rand(r2))
    x1 = [_proj(K1, Y) for Y in X]
    x2 = [_proj(K2, R * Y + t) for Y in X]
    x2[1:nout] .= [out2() for _ in 1:nout]
    rel = Csponds(x1, x2)
    tx = SMatrix{3,3}(0.0, t[3], -t[2], -t[3], 0.0, t[1], t[2], -t[1], 0.0)
    F = inv(K2)' * tx * R * inv(K1)
    F0 = F .* (1 .+ 1e-3 .* SMatrix{3,3}(randn(r2, 9)))
    H = K2 * (R + t * SVector(0.0, 0.0, 0.1)') * inv(K1)
    h1 = [out2() for _ in 1:n]
    h2 = [_proj(H, SVector(u[1], u[2], 1.0)) for u in h1]
    h2[1:nout] .= [out2() for _ in 1:nout]
    H0 = H .* (1 .+ 1e-3 .* SMatrix{3,3}(randn(r2, 9)))
    cam1 = CameraModel(LogicalIntrinsics(CameraCalibrationMat(K1)), Pinhole())
    cam2 = CameraModel(LogicalIntrinsics(CameraCalibrationMat(K2)), Pinhole())
    bm1 = backproject_with_jacobian.(Ref(cam1), x1)
    bm2 = backproject_with_jacobian.(Ref(cam2), x2)
    bear = withattrs(Csponds(first.(bm1), first.(bm2)); M₁ = last.(bm1), M₂ = last.(bm2))
    f = 800.0
    xa = [_nrm(R * Y + t) for Y in X]
    xa[1:nout] .= [SVector(randn(r2), randn(r2)) for _ in 1:nout]
    ua = [SVector(f * c[1] + K1[1, 3], f * c[2] + K1[2, 3]) for c in xa]
    abs_cs = Csponds(X, ua)
    return (; seed, K1, K2, rel, F0, H0, hom = Csponds(h1, h2), bear, abs_cs)
end

# The refined models of PoseLib 0.1.9 (VisualGeometryToolkit robust-core 3eabba46), captured with
# the retired API (`loss = <shape or LeZachTruncation()>`, `loss_width = w`, `max_iter = 25`; w = 2 px,
# 40 px for the absolute pose; `loss = GaussianShape()` for least squares) on `_baseline_data()`:
# `vec(R)` then `t` for a pose, `vec(M)` for a matrix. Every value round-trips exactly.
const _BASELINES = Dict(
    (:relpose_cam, :tls) => [0.9816974941998087, 0.024378765315571177, -0.18888013575679422, -0.013505989592662667, 0.9981878707128492, 0.05863926165011883, 0.18996741332907494, -0.05501500307585662, 0.9802477907700807, 0.9940280754023516, 0.09940280754335151, 0.049701403761826156],
    (:relpose_cam, :huber) => [0.9836115945638042, 0.023682076144842395, -0.17873832915499788, -0.013947800008865693, 0.9983600041332971, 0.05552261721024929, 0.17976008988249637, -0.052119683579556306, 0.9823287884761415, 1.0025352440354482, 0.06483370169376906, -0.020171296942220644],
    (:relpose_cam, :cauchy) => [0.981834791107843, 0.024339206782729543, -0.18817025796710238, -0.013998081489344671, 0.9983275166372728, 0.05609120465343379, 0.18922076177004463, -0.052438273598996936, 0.980533390954699, 0.9968784691174608, 0.07113158740263692, 0.04232133456016],
    (:relpose_cam, :lezach) => [0.9816974948882986, 0.024378765346639613, -0.18888013217438296, -0.013505997871386222, 0.9981878730388388, 0.05863922014912039, 0.1899674091825674, -0.05501496085950057, 0.9802477939429859, 0.9940176791208709, 0.0994013043124169, 0.04970080876545099],
    (:relpose_cam, :gaussian) => [0.961403388305855, 0.012086289083788714, -0.2748771481411365, -0.0015611783441186432, 0.9992582729312773, 0.038476833297175314, 0.27513830645004533, -0.036562625652212855, 0.9607091582415653, 0.2713542786889437, -0.011009852934928436, -1.6255446517780916],
    (:relpose_bearing, :tls) => [0.9816974941998086, 0.024378765315571156, -0.18888013575679435, -0.013505989592662658, 0.9981878707128492, 0.058639261650118735, 0.18996741332907507, -0.05501500307585652, 0.9802477907700807, 0.9940280754023516, 0.09940280754335099, 0.04970140376182768],
    (:relpose_bearing, :huber) => [0.9836115945638042, 0.023682076144842384, -0.17873832915499793, -0.013947800008865682, 0.9983600041332971, 0.05552261721024928, 0.17976008988249642, -0.0521196835795563, 0.9823287884761414, 1.0025352440354482, 0.0648337016937691, -0.020171296942220376],
    (:relpose_bearing, :cauchy) => [0.981834791107843, 0.024339206782729536, -0.18817025796710252, -0.013998081489344666, 0.9983275166372728, 0.056091204653433764, 0.18922076177004477, -0.05243827359899691, 0.980533390954699, 0.9968784691174609, 0.07113158740263699, 0.04232133456016027],
    (:relpose_bearing, :lezach) => [0.9816974948882986, 0.024378765346639616, -0.18888013217438307, -0.013505997871386228, 0.9981878730388388, 0.05863922014912035, 0.18996740918256752, -0.05501496085950053, 0.9802477939429859, 0.9940176791208707, 0.09940130431241664, 0.04970080876545099],
    (:relpose_bearing, :gaussian) => [0.961403388305855, 0.012086289083788731, -0.27487714814113656, -0.0015611783441186588, 0.9992582729312773, 0.038476833297175314, 0.2751383064500454, -0.036562625652212855, 0.9607091582415653, 0.27135427868894374, -0.011009852934928534, -1.6255446517780916],
    (:fundamental, :tls) => [6.287691084037984e-7, -7.441458488716962e-6, 0.0034065972715106767, 1.3773559246796325e-6, 1.8548392620061638e-6, -0.024673403006193282, -0.0030528713100065963, 0.026221327725685115, 0.9993413711496554],
    (:fundamental, :huber) => [1.0300294837584763e-6, 1.5929788766707957e-6, 0.0034636409564771063, -9.378343085467389e-6, 2.051353937632008e-6, -0.026487079330943648, -0.0029507159507246223, 0.031246295903328856, 0.9991506930923745],
    (:fundamental, :cauchy) => [5.514499436001787e-7, -9.335789570324979e-6, 0.0034157899119898923, 3.2757579056540332e-6, 1.8995932010047864e-6, -0.02556041180797542, -0.0030383072706382974, 0.0266446571592364, 0.9993079158583072],
    (:fundamental, :lezach) => [6.28768385918696e-7, -7.441479242088734e-6, 0.0034065978573714147, 1.3773614551215545e-6, 1.8548440737697815e-6, -0.02467346904359633, -0.0030528690950645003, 0.02622139748608122, 0.9993413676958439],
    (:fundamental, :gaussian) => [1.542367528548878e-6, 3.0407746272564885e-7, 0.004335975103786612, -1.079623563059488e-5, -8.255860307500567e-7, -0.02993450936578179, -0.003939889081480258, 0.03700753747174615, 0.9988499946244104],
    (:homography, :tls) => [0.8556092176902402, -0.035903386519224756, -0.00023635145006797127, 0.01137009385649648, 0.967631999272636, 7.337708939603558e-5, 269.0824929543417, -8.471170462118716, 1.0442873563917667],
    (:homography, :huber) => [0.8635840624172522, -0.03349189645547404, -0.00022866137733833225, 0.011679161123507907, 0.9702558140796672, 7.381630103539245e-5, 267.83438381250454, -9.419198811560488, 1.0442873563917667],
    (:homography, :cauchy) => [0.855644180014634, -0.03589027447694469, -0.00023631593373729773, 0.011365689850302077, 0.9676399790193599, 7.336955808700655e-5, 269.0770244496204, -8.476421756560512, 1.0442873563917667],
    (:homography, :lezach) => [0.8556092600936015, -0.035903370567273836, -0.00023635140687858422, 0.011370088213313286, 0.9676320087616793, 7.337707980522311e-5, 269.0824863423384, -8.471176834690246, 1.0442873563917667],
    (:homography, :gaussian) => [1.371588330673482, 0.14639586602704777, 0.0002921589529861744, -0.07117202249160473, 1.0829221420691995, -1.3145148488742795e-5, 178.26106386518325, -76.50159790069746, 1.0442873563917667],
    (:absolute_pose, :tls) => [0.9816974941996844, 0.02437876531583465, -0.18888013575740556, -0.013505989592875186, 0.9981878707128401, 0.0586392616502246, 0.1899674133297015, -0.05501500307590478, 0.9802477907699566, 0.9938079899998415, 0.09938079899999411, 0.04969039950000005],
    (:absolute_pose, :huber) => [0.9803781325796739, 0.02590749575070142, -0.195416270621253, -0.013300845964953893, 0.9977606070090196, 0.06555042789798747, 0.19667690422672401, -0.061665004377848484, 0.9785272722713823, 0.9247130157125851, 0.16987169659594167, 0.16749721045557764],
    (:absolute_pose, :cauchy) => [0.9816304782727827, 0.024549512254364955, -0.18920604000406716, -0.013619261780654727, 0.9981736741428944, 0.058854328274349915, 0.19030533317408072, -0.05519635582307704, 0.9801720473821698, 0.990738978641407, 0.1012716629465077, 0.05476132743033755],
    (:absolute_pose, :lezach) => [0.9816974767936978, 0.02437881038618295, -0.18888022040711885, -0.013506020027091243, 0.9981878670991879, 0.05863931615393979, 0.18996750111509883, -0.05501504866969066, 0.9802477711986681, 0.9938071951999562, 0.0993812761929912, 0.049691716876411385],
    (:absolute_pose, :gaussian) => [0.9241528253275649, 0.040279496004520056, -0.37989356093608806, 0.05021472403928005, 0.9729899140302033, 0.22532001394717827, 0.37870837979735583, -0.22730637781883622, 0.8971687542899286, -1.1063231160196145, 2.082008799807366, 3.9140833736434804],
)
_flat(m::PoseLib.Rigid3) = vcat(vec(Matrix(m.R)), collect(m.t))
_flat(m) = vec(Matrix(m))

_baseline_calls(d) = (
    relpose_cam = kw -> refine_relpose(d.seed, d.K1, d.K2, d.rel; max_iter = 25, kw...),
    relpose_bearing = kw -> refine_relpose(d.seed, d.bear; max_iter = 25, kw...),
    fundamental = kw -> refine_fundamental(d.F0, d.rel; max_iter = 25, kw...),
    homography = kw -> refine_homography(d.H0, d.hom; max_iter = 25, kw...),
    absolute_pose = kw -> refine_absolute_pose(d.seed, d.K1, d.abs_cs; max_iter = 25, kw...),
)
const _STRATEGIES = (tls = Irls{TLSShape}(), huber = Irls{HuberShape}(), cauchy = Irls{CauchyShape}(),
                     lezach = LeZach(), gaussian = nothing)
const _CODIM = (relpose_cam = 1, relpose_bearing = 1, fundamental = 1, homography = 2, absolute_pose = 2)

@testset "the robust-estimation contract on PoseLib's refiners" begin
    d = _baseline_data()
    calls = _baseline_calls(d)

    @testset "the baselines refine bit-identically: $name, $l" for name in keys(calls), l in keys(_STRATEGIES)
        lo = _STRATEGIES[l]
        w = name === :absolute_pose ? 40.0 : 2.0
        r = lo === nothing ? calls[name]((; optimization_strategy = lo)) : calls[name]((; optimization_strategy = lo, width = w))
        @test _flat(estimate(r)) == _BASELINES[(name, l)]
        # `scale = σ` runs at efficiency_constant(shape, d)·σ: the same refinement as that width
        if lo isa Irls{<:Union{HuberShape, CauchyShape}}
            σ = 0.75
            c = efficiency_constant(lo isa Irls{HuberShape} ? HuberShape() : CauchyShape(), _CODIM[name])
            rs = calls[name]((; optimization_strategy = lo, scale = σ))
            rw = calls[name]((; optimization_strategy = lo, width = c * σ))
            @test estimate(rs) == estimate(rw)
            @test noise_scale(rs) === σ && scale_source(rs) == ScaleGiven && rs.width === c * σ
            # the strategy's own efficiency constant replaces the shape's
            r3 = calls[name]((; optimization_strategy = Irls{typeof(lo).parameters[1]}(25, EfficiencyConstant(3.0)), scale = σ))
            @test estimate(r3) == estimate(calls[name]((; optimization_strategy = lo, width = 3.0 * σ)))
        end
    end

    H = calls.homography
    @testset "the result: FitResult with retcode, iterations and the scale record" begin
        r = H((; optimization_strategy = Irls{TLSShape}(), width = 2.0))
        @test r isa FitResult{VgtkCore.HomographyMat{Float64}, PoseLibRefineMetadata}
        @test retcode(r) === :converged && 0 < iterations(r) < 25
        @test r.n == length(d.hom) && isnan(r.rms)            # PoseLib reports no residual summary
        @test isnan(r.scale) && scale_source(r) == ScaleNone && r.width === 2.0
        @test_throws ArgumentError noise_scale(r)             # a width is never backed out into a scale
        # the budget used up is :max_iterations
        r1 = H((; optimization_strategy = nothing))
        @test retcode(refine_homography(d.H0, d.hom; max_iter = 1, optimization_strategy = nothing)) === :max_iterations
        @test scale_source(r1) == ScaleNone && isnan(r1.width)
        # an upstream result's scale is handed on (ScaleUpstream)
        rg = H((; optimization_strategy = Irls{HuberShape}(), scale = 0.5))
        ru = H((; optimization_strategy = Irls{HuberShape}(), scale = rg))
        @test noise_scale(ru) === 0.5 && scale_source(ru) == ScaleUpstream && estimate(ru) == estimate(rg)
        # the physical absolute pose keeps the plain refinement's record
        Xm = [Y .* VgtkCore.m for Y in VgtkCore.sources(d.abs_cs)]
        seedm = VgtkCore.EuclideanMap(d.seed.R, d.seed.t .* VgtkCore.m)
        rp = refine_absolute_pose(seedm, d.K1, Csponds(Xm, VgtkCore.targets(d.abs_cs)); max_iter = 25,
                                  optimization_strategy = Irls{TLSShape}(), width = 40.0)
        ra = calls.absolute_pose((; optimization_strategy = Irls{TLSShape}(), width = 40.0))
        @test VgtkCore.ustrip(VgtkCore.m, estimate(rp)).R == estimate(ra).R
        @test (retcode(rp), iterations(rp), rp.width) == (retcode(ra), iterations(ra), ra.width)
    end

    @testset "type-stable" begin
        for lo in (Irls{TLSShape}(), Irls{HuberShape}(), LeZach(), nothing)
            kw = lo === nothing ? (;) : (; width = 2.0)
            @test (@inferred refine_homography(d.H0, d.hom; max_iter = 25, optimization_strategy = lo, kw...)) isa FitResult
            @test (@inferred refine_fundamental(d.F0, d.rel; max_iter = 25, optimization_strategy = lo, kw...)) isa FitResult
            @test (@inferred refine_relpose(d.seed, d.K1, d.K2, d.rel; max_iter = 25, optimization_strategy = lo, kw...)) isa FitResult
            @test (@inferred refine_relpose(d.seed, d.bear; max_iter = 25, optimization_strategy = lo, kw...)) isa FitResult
            @test (@inferred refine_absolute_pose(d.seed, d.K1, d.abs_cs; max_iter = 25, optimization_strategy = lo, kw...)) isa FitResult
        end
        @test (@inferred refine_homography(d.H0, d.hom; max_iter = 25, optimization_strategy = Irls{CauchyShape}(), scale = 1.0)) isa FitResult
        @test (@inferred noise_scale(refine_homography(d.H0, d.hom; max_iter = 25, optimization_strategy = Irls{CauchyShape}(), scale = 1.0))) === 1.0
        @test (@inferred PoseLibRefineMetadata(1.0, ScaleGiven, 2.0)) isa PoseLibRefineMetadata
    end

    msg(f) = try f(); "" catch e; sprint(showerror, e) end
    @testset "refusals" begin
        # the retired keywords, by name
        for call in values(calls)
            @test_throws ArgumentError call((; loss = TLSShape(), loss_width = 2.0))
            @test occursin("optimization_strategy", msg(() -> call((; loss = TLSShape(), loss_width = 2.0))))
            @test occursin("`width", msg(() -> call((; optimization_strategy = Irls{TLSShape}(), loss_width = 2.0))))
            # the strategy is required, like max_iter
            @test occursin("needs `optimization_strategy`", msg(() -> call((; width = 2.0))))
        end
        @test occursin("optimization_strategy", msg(() -> H((; local_optimization = Irls{TLSShape}(), width = 2.0))))
        @test_throws ArgumentError H((; optimization_strategy = Irls{TLSShape}(), width = 2.0, tolerance = 1.0))
        # exactly one of scale and width for a robust strategy; neither for least squares
        @test_throws ArgumentError H((; optimization_strategy = Irls{HuberShape}()))
        @test_throws ArgumentError H((; optimization_strategy = Irls{HuberShape}(), scale = 1.0, width = 2.0))
        @test_throws ArgumentError H((; optimization_strategy = Irls{HuberShape}(), width = -2.0))
        @test_throws ArgumentError H((; optimization_strategy = Irls{HuberShape}(), scale = 0.0))
        @test_throws ArgumentError H((; optimization_strategy = Irls{HuberShape}(), scale = H((; optimization_strategy = nothing))))
        @test_throws ArgumentError H((; optimization_strategy = nothing, width = 2.0))
        @test_throws ArgumentError H((; optimization_strategy = nothing, scale = 1.0))
        # TLS and Le–Zach are truncations: no efficiency constant, so a scale needs the strategy's own
        @test occursin("efficiency constant", msg(() -> H((; optimization_strategy = Irls{TLSShape}(), scale = 1.0))))
        @test_throws ArgumentError H((; optimization_strategy = LeZach(), scale = 1.0))
        # the strategy's own efficiency constant is not overridden by a width; an absolute width is the keyword's
        @test_throws ArgumentError H((; optimization_strategy = Irls{HuberShape}(25, EfficiencyConstant(2.0)), width = 2.0))
        @test_throws ArgumentError H((; optimization_strategy = Irls{HuberShape}(25, LossWidth(2.0)), scale = 1.0))
        # Triggs is not what PoseLib's LM runs; no other shape; least squares is `nothing`
        for lo in (Triggs{TLSShape}(nothing), Triggs{HuberShape}(nothing), Triggs{CauchyShape}(nothing))
            @test occursin("Irls", msg(() -> H((; optimization_strategy = lo, width = 2.0))))
        end
        @test_throws ArgumentError H((; optimization_strategy = Irls{WelschShape}(), width = 2.0))
        @test occursin("nothing", msg(() -> H((; optimization_strategy = Irls{GaussianShape}()))))
        @test_throws ArgumentError H((; optimization_strategy = TLSShape(), width = 2.0))
        # the retired names
        @test !isdefined(PoseLib, Symbol("LeZach" * "Truncation"))
    end

    # LeZach is a member of the stack's one optimization-strategy family: registered through Core's
    # machinery with its TOML tag, and read back from a TOML table to the same strategy.
    @testset "LeZach: one strategy family, its tag round-trips through TOML" begin
        @test LeZach <: VgtkCore.AbstractOptimizationStrategy
        reg = PoseLib.POSELIB_STRATEGY_TAGS
        @test reg isa VgtkCore.TagRegistry && collect(keys(reg)) == ["le_zach"] && reg["le_zach"] === LeZach
        @test VgtkCore.tag_of(reg, LeZach()) == "le_zach" == VgtkCore.lo_tag(LeZach())
        @test VgtkCore.expected_tag(reg, LeZach) == "le_zach"
        # its identity, as the registry row states it
        @test VgtkCore.loss_family(LeZach()) == VgtkCore.loss_family(TLSShape)
        @test VgtkCore.optimization_path(LeZach()) === :irls && VgtkCore.lo_label(LeZach()) == "TLS-LZ"
        @test VgtkCore.lo_efficiency_constant_shape(LeZach) === TLSShape
        @test !VgtkCore.lo_parametric(LeZach) && !VgtkCore.lo_takes_width(LeZach)
        # merged with the core strategies' registry: one family, no clash
        tags = merge(VgtkCore.STRATEGY_TAGS, reg)
        @test sort(collect(keys(tags))) == sort([collect(keys(VgtkCore.STRATEGY_TAGS)); "le_zach"])
        # TOML round trip: written as its tag (and its fields, none), read back to the same strategy,
        # and the strategy read back refines as the one written
        written = merge(Dict{String, Any}("type" => VgtkCore.tag_of(tags, LeZach())), VgtkCore.toml_table(LeZach()))
        text = sprint(TOML.print, Dict("optimization_strategy" => written))
        back = TOML.parse(text)["optimization_strategy"]
        @test back == Dict("type" => "le_zach")
        lo = tags[back["type"]]()
        @test lo === LeZach()
        @test _flat(estimate(H((; optimization_strategy = lo, width = 2.0)))) == _BASELINES[(:homography, :lezach)]
        # a retired or unknown tag is refused by the registry
        @test_throws ArgumentError tags["le_zach_truncation"]
    end
end
