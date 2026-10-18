# Gauge Invariance of the Predictive Constraint Covariance

*For homogeneous models.*

We show that the predictive constraint covariance
``\btSigma_{\bg_i}`` (3), and hence the predictive marginal score
(RansacScoringDoneRight's `StudentTUniformScoring{Predictive}`, a predictive
scorer of VisualGeometryToolkitCore's scoring contract),
is gauge-invariant for homogeneous geometric
models such as the projective homography ``\mathbf{H}``. The proof relies on
the degree-1 homogeneity of the algebraic constraints ``\bg_i`` in the model
parameters ``\theta``, which causes the gauge direction to lie in the null
space of the whitened model Jacobian. The singular 9×9 estimation covariance
shape ``\btSigma_\theta`` may therefore be used directly — no explicit gauge
fixing, pseudoinverse, or reduced parametrization is required.

Notation: ``\partial_{\mathbf{x}} \bg_i`` and ``\partial_\theta \bg_i`` are the
Jacobians of ``\bg_i`` with respect to the measurements and the model.

## 1. Setup and Notation

Consider ``N``
observations ``\{\bx_i\}_{i=1}^N`` and a geometric model
``\theta \in \mathbb{R}^{n_\theta+1}`` defined only up to scale (e.g.
``\theta = \vect(\mathbf{H})`` for a ``3\times3`` homography, with
``n_\theta = 8`` free parameters and one gauge freedom). Each observation
imposes ``d_g`` algebraic constraints

```math
\bg(\bx_i, \theta) = \mathbf{0},
\tag{1}
```

and we write ``\bg_i \coloneqq \bg(\bx_i, \hat\theta)`` for the constraint
evaluated at the estimated model.

For the projective homography (``d_g = 2``, ``n_\theta = 8``,
``\dim\theta = 9``), the two constraints encoding
``\tilde{\bx}' \propto \mathbf{H}\,\tilde{\bx}`` are

```math
g_1 = \mathbf{h}_1^\top\tilde{\bx} - x'\,\mathbf{h}_3^\top\tilde{\bx},
\qquad
g_2 = \mathbf{h}_2^\top\tilde{\bx} - y'\,\mathbf{h}_3^\top\tilde{\bx},
\tag{2}
```

where ``\mathbf{h}_k^\top`` is the ``k``-th row of ``\mathbf{H}``,
``\tilde{\bx} = (x,y,1)^\top`` is the source point in homogeneous coordinates,
and ``(x', y')`` is the target.

**Covariance decomposition.** The observation noise covariance is
``\bSigma_{\bx_i} = \sigma^2\,\btSigma_{\bx_i}`` (known shape, unknown global
scale), and the model estimation covariance is
``\bSigma_\theta = \sigma^2\,\btSigma_\theta``. A first-order expansion of
``\bg_i`` about the true values gives the constraint covariance

```math
\bSigma_{\bg_i}
\;=\;
  \partial_{\bx}\bg_i\,\bSigma_{\bx_i}\,
  \bigl(\partial_{\bx}\bg_i\bigr)^\top
\;+\;
  \partial_\theta\bg_i\,\bSigma_\theta\,
  \bigl(\partial_\theta\bg_i\bigr)^\top,
\tag{3}
```

with covariance shape ``\btSigma_{\bg_i} = \bSigma_{\bg_i}/\sigma^2``. The
shape-whitened residual ``\br_i = \mathbf{L}_i^{-1}\bg_i`` (where
``\mathbf{L}_i\mathbf{L}_i^\top = \btSigma_{\bg_i}``) satisfies
``\br_i \sim \mathcal{N}(\mathbf{0}, \sigma^2\mathbf{I})`` under ``H_0``. When
``\btSigma_{\bx_i} = \mathbf{I}`` (isotropic noise) and
``\btSigma_\theta = \mathbf{0}`` (model certain), ``\|\br_i\|^2`` reduces to
the squared Sampson distance.

## 2. Homogeneity of the Algebraic Constraints

The key structural property is that the constraints (2) are *homogeneous of
degree 1* in ``\theta``:

```math
\bg(\bx_i, \lambda\theta) = \lambda\,\bg(\bx_i, \theta)
\qquad \forall\, \lambda \neq 0.
\tag{4}
```

This holds because ``\bg_i`` is linear in the entries of ``\mathbf{H}``. By
Euler's theorem for homogeneous functions,

```math
\partial_\theta\bg_i\,\theta = \bg_i.
\tag{5}
```

At the true model ``\theta^*``, the constraints vanish (``\bg_i = \mathbf{0}``
for noiseless observations), so

```math
\partial_\theta\bg_i\,\theta^* = \mathbf{0}.
\tag{6}
```

That is, *the true parameter vector ``\theta^*`` lies in the null space of the
model Jacobian* ``\partial_\theta\bg_i``.

!!! note "Remark 1 (Generality)"
    Property (4) holds for any model whose algebraic constraints are linear in
    ``\theta``: the fundamental matrix
    (``\mathbf{x}'^\top\mathbf{F}\,\mathbf{x} = 0``), the essential matrix, the
    absolute conic, and all DLT-type problems. The results of this note apply
    to all such models.

## [3. Gauge Invariance of the Prediction Covariance](@id cov-gauge)

We now show that the constraint covariance shape ``\btSigma_{\bg_i}`` (3) is
invariant under reparametrization ``\theta \to \lambda\theta``.

### 3.1 Scaling of Individual Quantities

Under ``\theta \to \lambda\theta`` (with measurements ``\bx`` fixed):

1. **Constraints:** ``\bg_i \to \lambda\,\bg_i`` (degree-1 homogeneity).

2. **Measurement Jacobian:**
   ``\partial_{\bx}\bg_i \to \lambda\,\partial_{\bx}\bg_i``, since ``\bg_i`` is
   linear in ``\theta`` and the mixed partial ``\partial_{\bx}\bg_i`` inherits
   the factor ``\lambda``.

3. **Model Jacobian:** ``\partial_\theta\bg_i`` is *independent* of
   ``\theta``, since ``\bg_i`` is linear in ``\theta`` (the derivative of a
   linear function is constant).

4. **Projected measurement covariance (Sampson matrix):**

   ```math
   \btSigma_{\bg_i}^{(\bx)} \coloneqq
   \operatorname{propagate}\!\bigl(
     \partial_{\bx}\bg_i,\;\btSigma_{\bx_i}\bigr)
   \;\to\; \lambda^2\, \btSigma_{\bg_i}^{(\bx)}.
   \tag{7}
   ```

5. **Cholesky factor:** ``\mathbf{L}_i \to \lambda\,\mathbf{L}_i`` (from
   ``\mathbf{L}_i\mathbf{L}_i^\top = \btSigma_{\bg_i}^{(\bx)} \to \lambda^2\,\btSigma_{\bg_i}^{(\bx)}``).

6. **Solver Jacobian:** The solver maps measurements to the normalized
   parameter: ``\partial_{\bx_\cS}\theta \to \lambda\, \partial_{\bx_\cS}\theta``,
   since the unnormalized solution scales linearly with ``\lambda``. Hence the
   estimation covariance shape:

   ```math
   \btSigma_\theta
   = \operatorname{propagate}\!\bigl(
     \partial_{\bx_\cS}\theta,\;\btSigma_{\bx_\cS}\bigr)
   \;\to\; \lambda^2\,\btSigma_\theta.
   \tag{8}
   ```

### 3.2 Invariance of the Whitened Residual

The whitened residual transforms as

```math
\br_i = \mathbf{L}_i^{-1}\,\bg_i
\;\to\;
  (\lambda\mathbf{L}_i)^{-1}(\lambda\bg_i)
= \mathbf{L}_i^{-1}\bg_i
= \br_i.
\tag{9}
```

The ``\lambda``-factors cancel: the whitened residual is gauge-invariant.

### 3.3 Invariance of the Constraint Covariance Shape

The model-uncertainty term in (3) transforms as

```math
\partial_\theta\bg_i\,\btSigma_\theta\,
\bigl(\partial_\theta\bg_i\bigr)^\top
\;\to\;
  \partial_\theta\bg_i\,(\lambda^2\btSigma_\theta)\,
  \bigl(\partial_\theta\bg_i\bigr)^\top
= \lambda^2\,
  \partial_\theta\bg_i\,\btSigma_\theta\,
  \bigl(\partial_\theta\bg_i\bigr)^\top.
\tag{10}
```

(Recall ``\partial_\theta\bg_i`` is independent of ``\theta``.) Together with
the measurement term (7):

```math
\bSigma_{\bg_i}
\;=\; \sigma^2\Bigl(
  \underbrace{\partial_{\bx}\bg_i\,\btSigma_{\bx_i}\,
    (\partial_{\bx}\bg_i)^\top}_{\to\;\lambda^2(\cdots)}
  \;+\;
  \underbrace{\partial_\theta\bg_i\,\btSigma_\theta\,
    (\partial_\theta\bg_i)^\top}_{\to\;\lambda^2(\cdots)}
\Bigr)
\;\to\; \lambda^2\,\bSigma_{\bg_i}.
\tag{11}
```

Both terms scale identically as ``\lambda^2``, so the *shape* (i.e.
``\bSigma_{\bg_i}`` up to an overall scalar) is invariant.

!!! note "Proposition 1 (Gauge invariance of the prediction covariance)"
    For any model whose algebraic constraints are homogeneous of degree 1 in
    ``\theta``, the constraint covariance shape ``\btSigma_{\bg_i}`` and the
    shape-whitened residual ``\br_i`` are invariant under the gauge
    transformation ``\theta \to \lambda\theta`` for all ``\lambda \neq 0``.
    Consequently, the marginal score ``S(\theta,\cI)`` and the predictive
    Mahalanobis distance ``q_i = \br_i^\top\btSigma_{\bg_i}^{-1}\br_i`` are
    gauge-invariant.

*Proof.* From (9), ``\br_i`` is invariant. From (11),
``\btSigma_{\bg_i} \to \lambda^2\btSigma_{\bg_i}`` and the Cholesky factor
``\mathbf{L}_i \to \lambda\mathbf{L}_i``. But the whitened residual
``\br_i = \mathbf{L}_i^{-1}\bg_i`` is gauge-invariant (9), and so is
``\|\br_i\|^2``, hence the RSS and the score ``S``.

For the predictive score:
``q_i = \br_i^\top\btSigma_{\bg_i}^{-1}\br_i \to \br_i^\top (\lambda^2\btSigma_{\bg_i})^{-1}\br_i = \lambda^{-2}\,\br_i^\top\btSigma_{\bg_i}^{-1}\br_i``.
But recall that the whitening is by the covariance *shape*
``\btSigma_{\bg_i}``, not the full covariance ``\sigma^2\btSigma_{\bg_i}``.
Both the shape and the residual inherit the same ``\lambda^2`` scaling from the
overall covariance, which factors out identically. Explicitly:

```math
q_i
= \bg_i^\top
  \underbrace{\bigl(\sigma^2\btSigma_{\bg_i}\bigr)^{-1}}_{= \bSigma_{\bg_i}^{-1}}\,
  \bg_i
\;\to\;
  (\lambda\bg_i)^\top
  (\lambda^2\bSigma_{\bg_i})^{-1}
  (\lambda\bg_i)
= q_i. \qquad \square
```

## 4. The Null-Space Argument

Proposition 1 shows invariance via the scaling cancellation. A complementary
perspective explains *why* the singular estimation covariance causes no
difficulty.

!!! note "Proposition 2 (Null-space annihilation)"
    At the true model ``\theta^*``, the gauge direction ``\theta^*`` lies in the
    null space of the whitened model Jacobian
    ``\mathbf{L}_i^{-1}\partial_\theta\bg_i``. Consequently, for any symmetric
    matrix ``\mathbf{A} \in \mathbb{R}^{(n_\theta+1)\times(n_\theta+1)}``, the
    component of ``\mathbf{A}`` along ``\theta^*`` does not contribute to the
    projected form
    ``\bigl(\mathbf{L}_i^{-1}\partial_\theta\bg_i\bigr)\,\mathbf{A}\,\bigl(\mathbf{L}_i^{-1}\partial_\theta\bg_i\bigr)^\top``.

*Proof.* By Euler's theorem (5),
``\partial_\theta\bg_i\,\theta^* = \bg_i\big|_{\theta=\theta^*} = \mathbf{0}``.
Hence ``\mathbf{L}_i^{-1}\partial_\theta\bg_i\,\theta^* = \mathbf{0}``. Now
decompose ``\mathbf{A} = \mathbf{A}_\perp + \alpha\,\theta^*\theta^{*\top}``
where ``\mathbf{A}_\perp`` has no component along ``\theta^*``. Then

```math
\bigl(\mathbf{L}_i^{-1}\partial_\theta\bg_i\bigr)\,\mathbf{A}\,
\bigl(\partial_\theta\bg_i\bigr)^\top\mathbf{L}_i^{-\top}
=
\bigl(\mathbf{L}_i^{-1}\partial_\theta\bg_i\bigr)\,
\mathbf{A}_\perp\,
\bigl(\partial_\theta\bg_i\bigr)^\top\mathbf{L}_i^{-\top},
```

since
``\bigl(\mathbf{L}_i^{-1}\partial_\theta\bg_i\bigr)\,\theta^*\theta^{*\top}\,\bigl(\partial_\theta\bg_i\bigr)^\top\mathbf{L}_i^{-\top} = \mathbf{0}\cdot\mathbf{0}^\top = \mathbf{0}``.
``\square``

!!! note "Corollary 1"
    The estimation covariance shape
    ``\btSigma_\theta = \partial_{\bx_\cS}\theta\,\bigl(\partial_{\bx_\cS}\theta\bigr)^\top``
    is a singular ``(n_\theta{+}1) \times (n_\theta{+}1)`` matrix (with
    ``\theta^*`` as a null eigenvector under Frobenius normalization). This
    singularity is harmless: substituting ``\mathbf{A} = \btSigma_\theta`` in
    Proposition 2, the gauge direction contributes nothing to
    ``\btSigma_{\bg_i}``. Adding arbitrary variance ``\alpha > 0`` along the
    gauge direction — i.e. replacing ``\btSigma_\theta`` by
    ``\btSigma_\theta + \alpha\,\theta^*\theta^{*\top}`` — leaves the
    prediction covariance ``\btSigma_{\bg_i}`` unchanged.

This means:

1. The singular ``\btSigma_\theta`` may be used directly in the quadratic
   form (3) without pseudoinverse or reduced parametrization.
2. No explicit gauge fixing is needed beyond whatever normalization the solver
   uses (e.g. Frobenius normalization ``\|\theta\| = 1`` with sign convention).
3. Whether the gauge direction carries zero, finite, or infinite variance in
   ``\btSigma_\theta``, the result ``\btSigma_{\bg_i}`` is identical.

## 5. Implementation: Two Equivalent Representations of ``\btSigma_\theta``

### 5.1 Gram Inverse (WLS / Local Optimization)

When ``\theta`` is estimated by weighted least squares from an observation set
``\cS`` with block-diagonal weight matrix
``\bW = \operatorname{blockdiag}(\btSigma_{\bg_1}^{-1}, \ldots, \btSigma_{\bg_{|\cS|}}^{-1})``,
the estimation covariance shape is the Gram inverse:

```math
\btSigma_\theta
= \Bigl(\bigl(\partial_\theta\bg_\cS\bigr)^\top
  \bW\, \partial_\theta\bg_\cS\Bigr)^{-1}.
\tag{12}
```

This is the form used during local optimization, where
``\partial_\theta\bg_\cS`` is the stacked design matrix and the SVD of
``\widetilde\bW^{1/2}\partial_\theta\bg = \mathbf{U}\boldsymbol{\Lambda}\mathbf{V}^\top``
yields ``\btSigma_\theta = \mathbf{V}\boldsymbol{\Lambda}^{-2}\mathbf{V}^\top``.

For homogeneous models, the Gram matrix
``\bigl(\partial_\theta\bg_\cS\bigr)^\top\bW\,\partial_\theta\bg_\cS`` is
rank-deficient (rank ``n_\theta``, with null space along ``\theta``), so the
"inverse" (12) is understood as a pseudoinverse or, equivalently, the inverse
restricted to the ``n_\theta``-dimensional subspace orthogonal to ``\theta``.

### 5.2 Solver Jacobian (Minimal Sample)

When ``\theta`` is determined by a minimal solver from ``m`` points with
``d_g m = n_\theta`` measurement coordinates, the solver defines a
differentiable map ``\bx_\cS \mapsto \theta(\bx_\cS)``. Under isotropic noise
(``\btSigma_{\bx_i} = \mathbf{I}``), the estimation covariance shape is

```math
\btSigma_\theta
= \partial_{\bx_\cS}\theta\,
  \bigl(\partial_{\bx_\cS}\theta\bigr)^\top,
\qquad
\partial_{\bx_\cS}\theta
\in \mathbb{R}^{(n_\theta+1) \times 2md}.
\tag{13}
```

For the 4-point homography solver,
``\partial_{\bx_\cS}\theta \in \mathbb{R}^{9 \times 16}`` and
``\btSigma_\theta \in \mathbb{R}^{9 \times 9}`` has rank 8.

**Equivalence.** Both representations yield the same ``\btSigma_{\bg_i}`` (3).
To see this, note that the implicit-function theorem applied to the
``n_\theta`` constraints ``\bg_\cS(\bx_\cS, \theta) = \mathbf{0}`` gives

```math
\partial_{\bx_\cS}\theta
= -\Bigl(\partial_\theta\bg_\cS\Bigr)^{\dagger}
  \partial_{\bx}\bg_\cS,
\tag{14}
```

where ``(\cdot)^\dagger`` denotes the pseudoinverse (needed because the system
has one more column than rank in ``\theta``). Substituting into (13) and using
``\btSigma_{\bx_i} = \mathbf{I}`` recovers the Gram inverse (12) with
``\bW = \mathbf{I}``.

### 5.3 Why Sign Normalization Is Needed in the Solver

Although the prediction covariance is gauge-invariant
([Section 3](@ref cov-gauge)), the solver itself must impose a consistent
normalization to ensure that ``\theta(\bx_\cS)`` is a *continuous,
differentiable* function of the measurements. Without sign normalization, the
solver output is discontinuous when ``\theta_{n_\theta+1}`` (e.g.
``H_{3,3}``) crosses zero: an infinitesimal perturbation flips the sign of the
entire output, making the solver Jacobian ``\partial_{\bx_\cS}\theta``
undefined.

Frobenius normalization (``\|\theta\| = 1``) combined with a sign convention
(e.g. ``H_{3,3} \geq 0``) ensures continuity almost everywhere. The sign
normalization is a solver-level convenience, not a mathematical necessity: any
smooth section of the projective equivalence class suffices, and the downstream
covariance ``\btSigma_{\bg_i}`` does not depend on the choice (Proposition 1).

## 6. Whitened Prediction Covariance: Origin of the ``\mathbf{I}`` Term

The prediction covariance (3) is ``d_g \times d_g`` and in general dense. The
scoring code works entirely in *whitened* coordinates, where the measurement
contribution collapses to a scalar multiple of the identity. This section
derives the whitened form and maps each step to the implementation.

### 6.1 Derivation

Define the Cholesky factor ``\mathbf{L}_i`` by the measurement-only covariance
shape (the Sampson matrix):

```math
\mathbf{L}_i\mathbf{L}_i^\top
= \btSigma_{\bg_i}^{(\bx)}
\coloneqq \partial_{\bx}\bg_i\,\btSigma_{\bx_i}\,
  \bigl(\partial_{\bx}\bg_i\bigr)^\top.
\tag{15}
```

For isotropic noise ``\btSigma_{\bx_i} = \mathbf{I}``, this is
``\btSigma_{\bg_i}^{(\bx)} = \partial_{\bx}\bg_i\,\bigl(\partial_{\bx}\bg_i\bigr)^\top``.

Now whiten both terms of the full covariance shape (3) by pre- and
post-multiplying ``\btSigma_{\bg_i}`` by ``\mathbf{L}_i^{-1}`` and
``\mathbf{L}_i^{-\top}``:

```math
\mathbf{L}_i^{-1}\,\btSigma_{\bg_i}\,\mathbf{L}_i^{-\top}
= \underbrace{
    \mathbf{L}_i^{-1}\,
    \btSigma_{\bg_i}^{(\bx)}\,
    \mathbf{L}_i^{-\top}
  }_{
    = \mathbf{L}_i^{-1}(\mathbf{L}_i\mathbf{L}_i^\top)
      \mathbf{L}_i^{-\top} = \mathbf{I}}
\;+\;
\underbrace{
  \bigl(\mathbf{L}_i^{-1}\partial_\theta\bg_i\bigr)\,
  \btSigma_\theta\,
  \bigl(\mathbf{L}_i^{-1}\partial_\theta\bg_i\bigr)^\top
}_{\text{model-uncertainty in whitened coords}}.
\tag{16}
```

With the full covariance (including ``\sigma^2``), the ``d_g \times d_g``
**prediction covariance** in whitened coordinates is therefore

```math
\boxed{
  \widetilde{\mathbf{V}}_i
  = \sigma^2 \mathbf{I}
  \;+\;
  \bigl(\mathbf{L}_i^{-1}\partial_\theta\bg_i\bigr)\,
  \bSigma_\theta\,
  \bigl(\mathbf{L}_i^{-1}\partial_\theta\bg_i\bigr)^\top.
}
\tag{17}
```

The ``\mathbf{I}`` appears because ``\mathbf{L}_i`` is defined as the Cholesky
factor of precisely the measurement-noise term; whitening maps that term to the
identity by construction.

!!! note "Remark 2 (RANSAC convention)"
    In the scoring code, ``s^2 = 1`` is supplied in place of ``\sigma^2`` and the
    unscaled shape ``\btSigma_\theta`` (instead of
    ``\bSigma_\theta = \sigma^2\btSigma_\theta``) is used, giving
    ``\widetilde{\mathbf{V}}_i = \mathbf{I} + (\mathbf{L}_i^{-1}\partial_\theta\bg_i)\,\btSigma_\theta\,(\mathbf{L}_i^{-1}\partial_\theta\bg_i)^\top``.
    This differs from the properly-scaled form (17) by an overall factor of
    ``\sigma^2``, which cancels in the Mahalanobis distance
    ``q_i = \br_i^\top\widetilde{\mathbf{V}}_i^{-1}\br_i`` since the whitened
    residual ``\br_i`` also absorbs a factor of ``\sigma`` (i.e.
    ``\br_i \sim \mathcal{N}(0, \sigma^2 I)`` so that
    ``\br_i^\top(\sigma^2 \widetilde{\mathbf{V}}_i)^{-1}\br_i = \br_i^\top \widetilde{\mathbf{V}}_i^{-1}\br_i``
    up to the overall ``\sigma^2``).

### 6.2 The `propagate` Primitive

Every covariance computation in the pipeline is an instance of the same
bilinear map:

```math
\operatorname{propagate}(A,\, \bSigma)
\;\coloneqq\; A\,\bSigma\,A^\top,
\tag{18}
```

where ``A`` may be rectangular. This unifies:

- **Measurement → constraint:**
  ``\btSigma_{\bg_i}^{(\bx)} = \operatorname{propagate}\!\bigl(\partial_{\bx}\bg_i,\; \btSigma_{\bx_i}\bigr)``
- **Measurement → parameter:**
  ``\btSigma_\theta = \operatorname{propagate}\!\bigl(\partial_{\bx_\cS}\theta,\; \btSigma_{\bx_\cS}\bigr)``
- **Projective conic transfer:**
  ``C^{*\prime} = \operatorname{propagate}(\mathbf{H},\, C^*)`` (``A`` square)

The function `propagate(A, Σ)` dispatches on the type of ``\bSigma`` for
compile-time specialization:

| ``\bSigma`` type | Computation |
|:--|:--|
| `I` (`UniformScaling{Bool}`) | ``A\,A^\top`` |
| ``\lambda\mathbf{I}`` (`UniformScaling`) | ``\lambda\,(A\,A^\top)`` |
| `Diagonal` | ``A\,D\,A^\top`` (column scaling) |
| General matrix | ``A\,\bSigma\,A^\top`` |

When ``d_g = 1``, the Cholesky reduces to a square root and the whitening to
scalar division; the compiler eliminates this automatically since ``d_g`` is a
compile-time constant.

### 6.3 Connection to Code

Each quantity above is computed by one function of the stack: the problem's
constraint rows are VisualGeometryToolkitEstimation's, `propagate` and `whiten` are
VisualGeometryToolkitCore's, and the predictive score is RansacScoringDoneRight's
(a scorer the loop reaches through `scored_model` and `score!`).

| Math | Function | Returns |
|:--|:--|:--|
| ``(\bg_i,\; \partial_{\bx}\bg_i,\; \partial_\theta\bg_i)`` | [`constraint_with_jacobians`](@ref)`(problem, model, i)` | ``(g,\; G_x,\; G_\theta)``, optionally followed by the point's noise shape ``\btSigma_{\bx_i}`` |
| ``\operatorname{propagate}(A, \bSigma) = A\,\bSigma\,A^\top`` (18) | `propagate(A, Σ)` | ``A\,\bSigma\,A^\top`` |
| ``\br_i = \mathbf{L}_i^{-1}\bg_i``, ``\partial_\theta\bg_{i,w} = \mathbf{L}_i^{-1}\partial_\theta\bg_i`` (15) | `whiten(g, G_x, G_θ, Σₓ)` | ``(\br_i,\; \partial_\theta\bg_{i,w},\; \log\lvert\btSigma_{\bg_i}^{(\bx)}\rvert)`` |
| ``\lVert\br_i\rVert``, the model-certain residual | [`whitened_residual`](@ref VisualGeometryToolkitCore.whitened_residual)`(problem, ctx, i)`, per point in [`residuals!`](@ref) | ``\lVert\br_i\rVert`` (signed when ``d_g = 1``) |
| ``\btSigma_\theta`` on an index set ``\cS``: (13) on a minimal sample, (12) on an overdetermined one | [`solver_covariance`](@ref)`(problem, model, idx)` | ``\btSigma_\theta``, unit scale |
| ``\widetilde{\mathbf{V}}_i = \mathbf{I} + \operatorname{propagate}(\partial_\theta\bg_{i,w},\, \btSigma_\theta)``, ``q_i = \br_i^\top \widetilde{\mathbf{V}}_i^{-1} \br_i`` (17), Remark 2 | RansacScoringDoneRight's `score_sq_residuals!` with a `StudentTUniformScoring{Predictive}` | ``q_i`` and the penalty ``\log\lvert\widetilde{\mathbf{V}}_i\rvert`` per point, then the sweep |

The interface is built around one problem method,
`constraint_with_jacobians(problem, model, i)`, which returns the algebraic
constraint ``\bg_i`` and its Jacobians ``\partial_{\bx}\bg_i``,
``\partial_\theta\bg_i`` at data point ``i``. Every downstream quantity is
derived from it.

**Residuals path (model-certain scoring).** For each observation ``i``,
`residuals!` writes `whitened_residual(problem, ctx, i)`, which by default
whitens ``(\bg_i, \partial_{\bx}\bg_i)`` through `whiten`: the norm of
``\br_i = \mathbf{L}_i^{-1}\bg_i``, whose square is the Mahalanobis distance
``\bg_i^\top\bigl(\btSigma_{\bg_i}^{(\bx)}\bigr)^{-1}\bg_i``. A problem may
override it with a cheaper route to the same quantity.

**Predictive scoring path (model-uncertain).** For a hypothesis with model
uncertainty, the pipeline for observation ``i`` is:

1. `constraint_with_jacobians(problem, model, i)` returns
   ``(\bg_i,\, \partial_{\bx}\bg_i,\, \partial_\theta\bg_i)``.
2. `whiten` forms
   ``\btSigma_{\bg_i}^{(\bx)} = \operatorname{propagate}(\partial_{\bx}\bg_i,\, \btSigma_{\bx_i})``,
   takes its Cholesky factor ``\mathbf{L}_i``, and returns the whitened
   residual ``\br_i = \mathbf{L}_i^{-1}\bg_i`` and whitened model Jacobian
   ``\partial_\theta\bg_{i,w} = \mathbf{L}_i^{-1}\partial_\theta\bg_i``.
3. The predictive score assembles
   ``\widetilde{\mathbf{V}}_i = \mathbf{I} + \operatorname{propagate}(\partial_\theta\bg_{i,w},\, \btSigma_\theta)``
   and evaluates ``q_i = \br_i^\top\widetilde{\mathbf{V}}_i^{-1}\br_i`` and
   ``\log\lvert\widetilde{\mathbf{V}}_i\rvert``. The ``\mathbf{I}`` is correct
   precisely because step 2 whitened by the measurement covariance.

The model covariance ``\btSigma_\theta`` is computed once per hypothesis by
`solver_covariance` from the constraint rows of its index set: on a minimal
sample the system is square and the result is the implicit-function covariance
of the minimal solve (13)–(14); on an overdetermined set (a local
optimization's consensus set) it is the covariance of the whitened-constraint
estimator (12). It is a unit-scale shape — the physical covariance is
``\bSigma_\theta = s^2\btSigma_\theta``, the convention of Remark 2. For the
projective models it is computed in the model's minimal chart and carried to
``\vect(\mathbf{H})`` (or ``\vect`` of ``\mathbf{F}``, ``\mathbf{E}``) through
the chart's differential, so the gauge direction carries zero variance — one
of the equivalent choices of Corollary 1.

## 7. Monte Carlo Validation

The gauge-invariance results were validated numerically for the homography
with five tests:

1. **Solver Jacobian:** ``\partial_{\bx_\cS}\theta`` of the 4-point solver
   matches forward-mode automatic differentiation through the solver to
   relative error ``< 10^{-15}``.
2. **Constraint derivatives and ``\btSigma_{\bg_i}``:** The whitened model
   Jacobian ``\mathbf{L}_i^{-1}\partial_\theta\bg_i`` matches forward-mode
   automatic differentiation to relative error ``< 10^{-15}``, and the
   composed prediction covariance ``\btSigma_{\bg_i}`` agrees to the same
   precision.
3. **Full Criminisi MC** (noisy training + noisy test): 200k trials with
   ``\sigma = 0.5``, ``n_{\mathrm{train}} = 4``. Eigenvalue ratios
   (empirical/analytic) ``\approx 0.98``–``0.99``.
4. **Noiseless test MC** (noisy training, exact test): Isolates the
   model-uncertainty term
   ``\partial_\theta\bg_i\,\btSigma_\theta\,(\partial_\theta\bg_i)^\top``.
   Eigenvalue ratios ``\approx 0.98``–``0.99``.
5. **Predictive score ``\chi^2`` test:** Under the RANSAC convention
   (``s^2 = 1``, unscaled ``\btSigma_\theta``), ``\mathbb{E}[q_i] = \sigma^2 d_g``.
   Under full scaling (``s^2 = \sigma^2``), ``q_i \sim \chi^2(d_g)`` with
   ``\mathbb{E}[q_i] = d_g``, ``\operatorname{Var}(q_i) = 2d_g``. Both
   verified to ``< 5\%`` relative error.

The package's test suite keeps the predictive calibration under test for every
covariance-bearing problem (homography, fundamental and essential matrices,
absolute pose, hand-eye, lines): a held-out point's
``\br_i^\top(\sigma^2\widetilde{\mathbf{V}}_i)^{-1}\br_i`` is gated against
``\chi^2(d_g)`` in `test/ransac_mc_calibration_tests.jl`.

## [References](@id cov-references)

- W. Chojnacki, M. J. Brooks, A. van den Hengel, and D. Gawley. *On the fitting
  of surfaces to data with covariances.* IEEE TPAMI, 22(11):1294–1303, 2000.
- R. Hartley and A. Zisserman. *Multiple View Geometry in Computer Vision.*
  Cambridge University Press, 2nd edition, 2003.
