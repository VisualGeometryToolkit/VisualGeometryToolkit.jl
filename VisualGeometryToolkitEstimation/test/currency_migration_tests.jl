# =============================================================================
# The retired width currencies: every stored value migrated once, at full
# precision, so every deployed loss is bit-identical (the owner's Q7 ruling).
# =============================================================================
#
# Until 2026-11 a Geman–McClure multiple was read as a HALF-WEIGHT depth
# (`decision_width(GM, m)` was deployed) and LocalOptimizationDoneRight's
# BarronHQ multiple as `m/√2`. An efficiency constant is now always the plain
# multiple (`loss_width = c·ŝ`), and each stored value was replaced by the
# Float64 the old path computed — so the deployed width, and every result, is
# the same bit for bit. The table lists every migrated value of the monorepo
# and the ports (switch lists in the robust-contract report).

@testitem "currency migration: the stored values deploy bit-identically" begin
    using VisualGeometryToolkitCore: GemanMcclureShape, WelschShape, decision_width, decision_boundary,
        deploy, sq_loss_width, loss_width

    gm(m) = decision_width(GemanMcclureShape(), m)        # the retired half-weight currency
    barron(m) = m / sqrt(2.0)                              # BarronHQ's retired facet-B currency
    #   (old value, migrated value, the retired conversion, where)
    table = [
        (2.0, 3.1075479480600743, gm, "Estimation fit_linear_tests, ransac_homography_tests"),
        (1.25, 1.9422174675375463, gm, "Estimation estimation_interfaces_tests, bcd_recorded_tests; LODR test_strategies, test_bcd_recorded"),
        (1.5 * sqrt(log(2.0)), 1.9404025301632923, gm, "Estimation lo_timing/bench GM_MULTIPLE; LODR scenes_testsetups, bench"),
        (1.5, 2.3306609610450555, gm, "LODR GM arms at the shared Welsch grid's centre"),
        (1.5, 1.0606601717798212, barron, "LODR BarronHQ (test_strategies, test_config_toml)"),
    ]
    for (old, new, conv, where) in table
        @test new === conv(old)
        for s2 in (0.37^2, 1.0, 4.2)        # the deployed loss: the same Float64 arithmetic as before
            @test sq_loss_width(deploy(GemanMcclureShape(), s2; efficiency_constant = new)) ===
                sq_loss_width(deploy(GemanMcclureShape(), s2; efficiency_constant = conv(old)))
        end
    end
    # the GM arms' map from the shared Welsch grid: a half-weight depth m·√ln2, now composed explicitly
    m = 1.5
    @test gm(decision_boundary(WelschShape(), m)) === decision_width(GemanMcclureShape(), m * sqrt(log(2.0)))
end
