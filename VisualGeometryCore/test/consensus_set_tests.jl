# ConsensusSet's accessors exist so that downstream packages (RobustVisualGeometry,
# RansacScoringDoneRight) stop reading its FIELDS across a package boundary — the
# coupling that lets an upstream struct change break a consumer silently at runtime
# while its own suite stays green. The accessors are consumed downstream, not inside
# VGC, so these items are what pins them here.
@testitem "consensus_set_accessors" begin
    using Test
    using VisualGeometryCore
    using VisualGeometryCore: ConsensusSet, check_fresh, is_fresh, inlier_count, scale_s2, consensus_score   # public, not exported

    seed = ConsensusSet(; count = 2, s2 = 0.25, epoch = 5)
    @test seed == ConsensusSet(2, 0.25, 5, NaN)
    @test scale_s2(seed) == 0.25
    # The evaluation's score: NaN unless the producer carries it.
    @test isnan(consensus_score(seed))
    @test consensus_score(ConsensusSet(; count = 2, s2 = 0.25, epoch = 5, score = -3.5)) == -3.5

    # The cross-package surface: public, imported by name.
    @test inlier_count(seed) == 2
    @test inlier_count(seed) isa Int

    # The epoch is exposed as a COMPARISON, never as a raw number: the stamp is
    # only meaningful against the producing workspace's current epoch.
    @test is_fresh(seed, 5)
    @test !is_fresh(seed, 6)
    @test !is_fresh(seed, 4)

    # check_fresh is the enforcing form: nothing when fresh, a refusal when stale.
    @test check_fresh(seed, 5) === nothing
    @test_throws ArgumentError check_fresh(seed, 6)

    # A real throw, NOT an @assert — the guarded failure is a silent use of an
    # earlier evaluation's consensus set, so it must survive any optimization level.
    err = try
        check_fresh(seed, 6)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("stale ConsensusSet", err.msg)
    @test occursin("5", err.msg) && occursin("6", err.msg)   # both stamps named

    # No raw epoch reader — reading the stamp without comparing it is the
    # hand-rolled contract the checked accessors replace.
    @test !isdefined(VisualGeometryCore, :partition_epoch)
end
