# The General/AutoMerge rules that can be checked before registering.
#
#   julia --project=ci ci/registration_lint.jl [--subsequent]
#
# By default the rules of a package's FIRST registration apply to every
# package General does not hold yet: its version is the standard initial
# version 0.1.0 and its name is not within DL distance 2 of another's. A
# package General holds under its UUID is never a first registration, so those
# rules never apply to it. `--subsequent` drops them for every package: CI and
# the release run it, since every stack package has a first registration in
# General merged or filed (a filed one is invisible here until it merges).
#
# Per package directory:
#   - Project.toml: name = directory, a UUID, a version; name rules (ASCII
#     letters/digits, uppercase start, ≥ 5 characters, no "julia", no "jl"
#     suffix, Damerau–Levenshtein distance > 2 from every General name unless
#     the package is in WORKSPACE.toml's [manual_merge])
#   - [compat] for every [deps] and [weakdeps] entry, stdlibs and julia included;
#     each entry parses and is bounded (no "*", no ">="); a compat on a stack
#     package names THAT package's own series — its current breaking series in
#     the workspace ("0.5" for 0.5.x, "1" for 1.x), or a raised floor inside it
#     ("0.5.2") when a package needs a sibling's newer feature (every floor is
#     raised to the gated environment's resolution at a release, the owner's
#     ruling of 2026-11-13, so no workflow tests older floors). Each package is
#     versioned on its own: a breaking change bumps that package alone
#     (2026-11-03; the stack no longer moves as one series)
#   - no [sources], [extras] or [targets] in the registered file (test deps live
#     in test/Project.toml, which [sources]-links the package as path "..")
#   - a LICENSE (OSI: MIT) inside the directory — the registered tarball is
#     the subdir only
#   - the stack dependency allow-list (VisualGeometryToolkitCore's
#     stack_dependency_violations) for the package, test and docs projects
# Workspace:
#   - no tracked Manifest*.toml anywhere
#   - the root Project.toml is not a package (no name/uuid)
#   - codecov.yml is the generated text (VisualGeometryToolkitCore's codecov_config)
#
# Prints one line per finding; exits 1 when there is an error. The
# implementation is VisualGeometryToolkitCore's `stack_ci` (the
# `registration-lint` command; `registration_violations`); MANUAL_MERGE is the
# [manual_merge] table of the root WORKSPACE.toml.

using VisualGeometryToolkitCore: stack_ci

exit(stack_ci(["registration-lint"; ARGS]; root = dirname(@__DIR__)))
