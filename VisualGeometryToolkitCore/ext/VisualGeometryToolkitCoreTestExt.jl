# =============================================================================
# VisualGeometryToolkitCoreTestExt — the stack's naming checker and shared test helpers
# =============================================================================
#
# `check_naming` and `naming_scope` (stubs in src/quality/naming.jl), and the
# helpers every stack package's tests share — `rows_cost`, `check_rows_jacobian`,
# `test_naming` (stubs in src/quality/test_helpers.jl).
# Every caller is a test, so they load with the Test standard library rather
# than with the package.
# =============================================================================
module VisualGeometryToolkitCoreTestExt

using Test: Test
using TOML: TOML
using ForwardDiff: ForwardDiff
using VisualGeometryToolkitCore: VisualGeometryToolkitCore, NamingViolation, residual_rows!, residual_jacobian_rows!
import VisualGeometryToolkitCore: check_naming, naming_scope, rows_cost, check_rows_jacobian, test_naming
import VisualGeometryToolkitCore: test_scoring_contract, test_local_optimization_contract, test_constraint_problem

include("VisualGeometryToolkitCoreTestExt/check_naming.jl")
include("VisualGeometryToolkitCoreTestExt/test_helpers.jl")
include("VisualGeometryToolkitCoreTestExt/contracts.jl")

end # module VisualGeometryToolkitCoreTestExt
