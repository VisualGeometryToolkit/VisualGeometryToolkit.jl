# bundle_loss: the one reading of a bundle's robust keywords (refine_camera_and_poses,
# refine_robot_world_hand_eye, PanoramaStitching's rotation bundle) — the refine contract's
# strategy, exactly one scale input, the loss deployed at the residual's codimension.
@testitem "bundle_loss: a bundle's robust loss from its keywords" begin
    using Test
    using VisualGeometryToolkitEstimation: bundle_loss
    using VisualGeometryToolkitCore: Triggs, Irls, WelschShape, HuberShape, TLSShape, GaussianShape, deploy,
        efficiency_constant, sq_loss_width, noise_scale, scale_source, ScaleGiven, ScaleNone
    # least squares reads no scale
    @test bundle_loss(nothing; scale = nothing, width = nothing, caller = "b") == (; strategy = nothing, input = nothing, loss = nothing)
    @test_throws r"b: plain least squares" bundle_loss(nothing; scale = 1.0, width = nothing, caller = "b")
    @test_throws ArgumentError bundle_loss(nothing; scale = nothing, width = 2.0, caller = "b")
    # a robust strategy at the scale: its shape at c × σ, c Core's efficiency constant at the codimension
    r = bundle_loss(Triggs{WelschShape}(); scale = 1.5, width = nothing, caller = "b")
    @test r.strategy === Triggs{WelschShape}() && noise_scale(r.input) == 1.5 && scale_source(r.input) === ScaleGiven
    @test r.loss == deploy(WelschShape(), 1.5^2; efficiency_constant = efficiency_constant(WelschShape, 2))
    @test bundle_loss(Triggs{WelschShape}(); scale = 1.5, width = nothing, codimension = 1, caller = "b").loss ==
          deploy(WelschShape(), 1.5^2; efficiency_constant = efficiency_constant(WelschShape, 1))
    # the strategy's own constant, the plain multiple
    @test sqrt(sq_loss_width(bundle_loss(Irls{HuberShape}(efficiency_constant = 2.0); scale = 3.0, width = nothing, caller = "b").loss)) ≈ 6.0
    # at a width: the loss width itself; a strategy stating a constant too is refused
    rw = bundle_loss(Triggs{WelschShape}(); scale = nothing, width = 4.0, caller = "b")
    @test sqrt(sq_loss_width(rw.loss)) ≈ 4.0 && scale_source(rw.input) === ScaleNone
    @test scale_source(bundle_loss(nothing; caller = "b").input) === ScaleNone          # least squares: no input
    @test_throws ArgumentError bundle_loss(Triggs{WelschShape}(efficiency_constant = 1.0); scale = nothing, width = 4.0, caller = "b")
    # exactly one scale input; a truncation states its constant; least squares is `nothing`; no other strategy
    @test_throws ArgumentError bundle_loss(Triggs{WelschShape}(); scale = nothing, width = nothing, caller = "b")
    @test_throws ArgumentError bundle_loss(Triggs{WelschShape}(); scale = 1.0, width = 1.0, caller = "b")
    @test_throws ArgumentError bundle_loss(Irls{TLSShape}(); scale = 1.0, width = nothing, caller = "b")
    @test_throws ArgumentError bundle_loss(Irls{GaussianShape}(); scale = 1.0, width = nothing, caller = "b")
    @test_throws ArgumentError bundle_loss(:welsch; scale = 1.0, width = nothing, caller = "b")
    @test_throws r"bundle_loss: plain least squares" bundle_loss(nothing; scale = 1.0)   # the default caller
end
