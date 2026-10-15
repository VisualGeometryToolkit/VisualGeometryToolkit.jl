## Signatures: the model leads, then its data, then parameters

**One rule: the model or shape first, then the data it acts on or is measured
against, then parameters.**

| Call | Model or shape | Data | Parameters |
|------|----------------|------|------------|
| `project(camera, X)` | camera | `X` | |
| `sampson_distance(F, u₁, u₂)` | `F` | `u₁`, `u₂` | |
| `distance(plane, p)`, `closest_point(plane, p)` | plane | `p` | |
| `is_in_front(camera, X)` | camera | `X` | |
| `undistort(model, img, rect; scale)` | model | `img` | `rect`, `scale` |
| `boundary_point(ellipse, θ)` | ellipse | | `θ` |
| `focal_length_for_fov(sensor, θ)` | sensor | | `θ` |
| `centroid(ellipse)` | ellipse | | |

The first argument names what the function belongs to, in the math's order
(`P X`, `H x`, `mul!(Y, A, X)`). Measuring data against a model is the same
question whether the model is `F` or a plane, so it takes the same order.

What counts as the model:
- **A given model leads, in task verbs too.** The camera handed to
  `find_absolute_pose(camera, pairs)`, the guess handed to
  `refine_relpose(pose, K1, K2, x1, x2)`: estimated or given, it is applied to the data.
- **A required tool leads; a policy with a default stays last.** A metric, a
  descriptor network and gallery, the method or scorer being evaluated:
  `pairwise_matrix(metric, d1, d2)`, `find_boards(gallery, network, img)`,
  `evaluate_sample_budget(methods, pairs, T)`. But `match(d1, d2, strategy =
  MutualNN())`: a defaulted strategy is a parameter. When several model
  arguments lead, they keep their relative order.

**Mutation follows Julia's style guide exactly** ("Write functions with argument
ordering similar to Julia Base"; the model-first rule never overrides it):
- A function that modifies a Julia object argument ends in `!` — a workspace,
  buffer, state object, or a memo cache it fills (Base's `get!(f, d, k)`).
  Not mutation, per the manual and Base: advancing an RNG (a RANSAC sampler's
  state counts as RNG state), writing to an IO or to files (`write`, `mkpath`,
  `rm` carry no `!`), and a caller-supplied scratch buffer given as a keyword
  whose contents don't matter afterwards (`sort(v; scratch)`). A positional
  scratch buffer becomes such a keyword; an output the caller reads back is
  either returned or the leading argument of a `!` function
  (`score_residuals!(sws, q, scoring, residuals, n)`, `refine!(ws, problem, θ, weights)`).
- Base's order: (1) a function argument, (2) an IO stream — an RNG takes this slot
  (Random: "accept an optional `AbstractRNG` object as first argument"), (3) the
  input(s) being mutated, (4) a type, (5) inputs not mutated, (6) key, (7) value,
  (8) everything else, (9) varargs, (10) keywords. The model-first rule above orders
  the inputs within (5).
- So only a function argument, an IO or an RNG may precede what a `!` call
  mutates; a strategy or method object may not: `map!(f, dst, src)`,
  `rand!(rng, A)`, `draw_sample!(rng, indices, problem)`,
  `undistort!(dst, model, src)`, `block_coordinate_descent!(state, likelihood, method, prob, θ₀)`.
  The in-place and allocating forms agree on the order of the rest.
- **A type argument takes slot (4), positionally** — precision and element types,
  frame tags and problem types alike, never a keyword. In a `!` call it follows the mutated inputs:
  `undistort!(dst, [T = Float64], model, src)`, `rand!(rng, A, S)`. In a
  non-mutating call it leads, after any function or RNG: `zeros(Float32, n)`,
  `rand(rng, Float32, dims)`, `image_warp(Float32, math, rect)`,
  `evaluate_sample_budget(problem_type, methods, pairs)`, `relabel(Src, Dst, m)`
  (like `convert(T, x)`). An optional type is an optional leading positional, as
  in `zeros([T = Float64], dims)`.
- **An RNG leads** (the IO slot): `rand(rng, …)`, `tune!(rng, method, spec, n)`.

**Three exceptions:**
- A method added to a function the package does not own keeps the owner's
  order: `p ∈ ellipse` is Base's `in(item, collection)`.
- A symmetric relation takes either order: `intersects(a, b)`, `iou(a, b)`.
- RobustVisualGeometry's RANSAC family leads with the problem, which carries
  the data, and puts the hypothesis after it: `solve(problem, idx)`,
  `test_model(problem, model, idx)`, `residuals!(r, problem, model)`,
  `locally_optimize(problem, θ₀)`, and wrappers that hold a problem or a
  per-pair context (`refit_imc(lo, ctx, seed)`).

(VisualGeometryCore 0.27.0 replaced seven role-based rules with this one:
ACTION vs QUERY put `sampson_distance(F, u)` and `distance(p, plane)` in
opposite orders, and an image-first rule reversed the lens between the batched
point undistortion and `undistort!`.)

VisualGeometryCore's `src/base/interface_functions.jl` repeats the rule beside
the shared stubs.
