# =============================================================================
# VisualGeometryToolkitCoreTestExt — the stack's naming checker and shared test helpers
# =============================================================================
#
# `check_naming` and `naming_scope` (stubs in src/quality/naming.jl), `check_raw_decoding`
# (src/quality/raw_decoding.jl), `check_duplication` (src/quality/duplication.jl), and the
# helpers every stack package's tests share — `rows_cost`, `check_rows_jacobian`,
# `test_naming`, `test_docs` (stubs in src/quality/test_helpers.jl).
# Every caller is a test, so they load with the Test standard library rather
# than with the package.
# =============================================================================
module VisualGeometryToolkitCoreTestExt

using Test: Test
using TOML: TOML
using ForwardDiff: ForwardDiff
using StaticArrays: StaticArray
using VisualGeometryToolkitCore: VisualGeometryToolkitCore, NamingViolation, RawDecodingViolation, DuplicationViolation, residual_rows!, residual_jacobian_rows!,
    Attributed, Recorded, block_coordinate_descent!, current_cost, estimate, retcode, iterations, RECORDED_NESTED_STATES
import VisualGeometryToolkitCore: check_naming, naming_scope, check_raw_decoding, check_duplication, rows_cost, check_rows_jacobian, test_naming, test_docs, thrown_message, test_duplication,
    test_recorded
import VisualGeometryToolkitCore: test_scoring_contract, test_local_optimization_contract, test_constraint_problem,
    test_manifold_contract

include("VisualGeometryToolkitCoreTestExt/check_naming.jl")
include("VisualGeometryToolkitCoreTestExt/check_raw_decoding.jl")   # the one RAW path (reads naming_scope)
include("VisualGeometryToolkitCoreTestExt/check_duplication.jl")    # one implementation per concept (reads both)
include("VisualGeometryToolkitCoreTestExt/test_helpers.jl")
include("VisualGeometryToolkitCoreTestExt/test_docs.jl")         # the docstring gates every package runs
include("VisualGeometryToolkitCoreTestExt/contracts.jl")

end # module VisualGeometryToolkitCoreTestExt
