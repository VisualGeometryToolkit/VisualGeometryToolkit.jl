# Homography Decomposition

*``H \longrightarrow (R, \boldsymbol{t}, \boldsymbol{n})`` (Faugeras–Lustman).*

## [1. The Parametric Form](@id hdecomp-form)

For two pinhole cameras observing a world plane ``\pi`` with unit normal
``\boldsymbol{n}`` (in camera 1's frame) and signed distance ``d > 0`` from
camera 1's center, the plane-induced homography is

```math
H \;=\; K_2 \!\left(R + \tfrac{1}{d}\, \boldsymbol{t} \boldsymbol{n}^{\mathsf T}\right) K_1^{-1},
\tag{1}
```

where ``(R, \boldsymbol{t})`` is the relative pose (camera 1 ``\to`` camera 2)
and ``K_1, K_2`` are intrinsics. Our goal is to invert this relation: given
``H``, recover ``(R, \boldsymbol{t}, \boldsymbol{n})``.

### Calibrated form

Strip out the intrinsics:

```math
\boxed{\;\tilde{H} \;=\; K_2^{-1} H K_1 \;=\; R + \frac{1}{d}\,\boldsymbol{t} \boldsymbol{n}^{\mathsf T}.\;}
\tag{2}
```

All of the decomposition math operates on ``\tilde{H}``. This requires
calibrated cameras; if ``K_1, K_2`` are unknown, the decomposition is not
well-defined.

### Degree-of-freedom count

``\tilde{H}`` is projective, so 8 DOF. On the right-hand side of (2), ``R``
contributes 3, ``\boldsymbol{t}`` contributes 3, ``\boldsymbol{n}`` (unit
vector) contributes 2, and ``d`` contributes 1, for 9 DOF; but the product
``\boldsymbol{t} \boldsymbol{n}^{\mathsf T} / d`` is invariant under
``(\boldsymbol{t}, d) \mapsto (\lambda \boldsymbol{t}, \lambda d)``, absorbing
one DOF and bringing the count to 8. So decomposition is generically
well-posed up to a finite set of sign ambiguities.

## [2. Faugeras–Lustman Algorithm](@id hdecomp-fl)

This is the classical algorithm (also the one OpenCV's
`decomposeHomographyMat` and PoseLib's `motion_from_homography` use).

### Step 1: SVD and middle-singular-value normalization

Compute

```math
\tilde{H} = U\,\Sigma\,V^{\mathsf T}, \qquad
\Sigma = \mathrm{diag}(d_1, d_2, d_3), \qquad
d_1 \ge d_2 \ge d_3 > 0.
\tag{3}
```

Normalize so the middle singular value equals 1:

```math
\tilde{H} \leftarrow \tilde{H} / d_2,
\qquad d_1 \leftarrow d_1/d_2, \quad d_3 \leftarrow d_3/d_2.
\tag{4}
```

After this, ``d'_1 \ge 1 \ge d'_3``.

### Step 2: reduce to the SVD basis

In the coordinate system defined by ``V`` (where the normalized plane normal
aligns with the axes), write

```math
R = U R' V^{\mathsf T},
\qquad \boldsymbol{t} = U \boldsymbol{t}',
\qquad \boldsymbol{n} = V \boldsymbol{n}'.
\tag{5}
```

Substituting into (2) and using ``U^{\mathsf T} U = V^{\mathsf T} V = I``, the
problem becomes: find ``R'``, ``\boldsymbol{t}'``, ``\boldsymbol{n}'`` such
that

```math
\Sigma' = R' + \boldsymbol{t}' \boldsymbol{n}'^{\,\mathsf{T}},
\qquad \Sigma' = \mathrm{diag}(d'_1, 1, d'_3).
\tag{6}
```

### Step 3: closed-form expressions for n′

Under the generic case ``d'_1 > 1 > d'_3``, one can show (Faugeras–Lustman
1988) that the normal has the form

```math
\boldsymbol{n}' = \bigl(x_1,\; 0,\; x_3\bigr)^{\mathsf T},
\tag{7}
```

with

```math
x_1 \;=\; \pm\sqrt{\frac{d'_1{}^{\,2} - 1}{d'_1{}^{\,2} - d'_3{}^{\,2}}},
\qquad
x_3 \;=\; \pm\sqrt{\frac{1 - d'_3{}^{\,2}}{d'_1{}^{\,2} - d'_3{}^{\,2}}}.
\tag{8}
```

The two sign choices ``\epsilon_1, \epsilon_3 \in \{+1, -1\}`` yield **four
candidate solutions**. The middle component is zero because the plane normal
has no component in the direction of the middle singular vector (the direction
in which the scene's out-of-plane depth has no effect).

### Step 4: recover R′ and t′ per candidate

Let

```math
\sin\theta = \frac{(d'_1 - d'_3)\, x_1\, x_3}{1},
\tag{9}
```

```math
\cos\theta = \frac{d'_1\, x_3^{\,2} + d'_3\, x_1^{\,2}}{1}.
\tag{10}
```

Then the rotation in the reduced basis is

```math
R' = \begin{pmatrix}
    \cos\theta & 0 & -\sin\theta \\
    0 & 1 & 0 \\
    \sin\theta & 0 & \cos\theta
\end{pmatrix},
\tag{11}
```

and the translation is

```math
\boldsymbol{t}' = \bigl(d'_1 - d'_3\bigr)\, \bigl(x_1,\; 0,\; -x_3\bigr)^{\mathsf T}.
\tag{12}
```

With the two independent sign choices for ``(x_1, x_3)``, this yields four
distinct ``(R', \boldsymbol{t}', \boldsymbol{n}')`` tuples.

### Step 5: transform back to the original frame

For each candidate,

```math
R = U R' V^{\mathsf T},
\qquad \boldsymbol{t} = U\,\boldsymbol{t}',
\qquad \boldsymbol{n} = V\,\boldsymbol{n}'.
\tag{13}
```

### Step 6: cheirality selection

Of the four candidates, typically only one (occasionally two symmetric ones)
corresponds to a physically valid configuration where both cameras view the
plane from the *same* side. The valid candidates satisfy

```math
\boldsymbol{n}^{\mathsf T} X_i > 0 \qquad
\text{for all correspondences } X_i \text{ on the plane, in cam 1's frame}.
\tag{14}
```

Equivalently, for each candidate, every sampled correspondence must produce a
positive projective scale factor ``\lambda_i`` in
``\mathbf{x}'_i = \lambda_i H \mathbf{x}_i``. The two or so candidates that satisfy
this test are the physically meaningful decompositions.

In the benchmark setting where a ground-truth pose
``(R_{gt}, \boldsymbol{t}_{gt})`` is known, one can instead pick the candidate
minimizing

```math
e_R(R, R_{gt}) + \alpha\, e_T(\boldsymbol{t}, \boldsymbol{t}_{gt})
\tag{15}
```

directly. This is what HEB and Shekhovtsov do when computing pose errors.

## [3. Malis–Vargas Algorithm (OpenCV / PoseLib / Shekhovtsov)](@id hdecomp-mv)

An alternative decomposition, used by `cv2.decomposeHomographyMat` and
PoseLib's `motion_from_homography`, is based on eigendecomposition of
``\tilde{H}^{\mathsf T} \tilde{H}`` rather than direct SVD analysis. It
produces the same four candidate tuples as Faugeras–Lustman but through a
different computational path. Shekhovtsov's PyTorch port follows this
algorithm exactly.

### Step 1: sign normalization and middle-singular-value scaling

If ``\det \tilde{H} < 0``, flip ``\tilde{H} \leftarrow -\tilde{H}``
(homographies are defined up to scalar multiplication, so this is a free choice
for orientation consistency). Then

```math
\tilde{H} = U \Sigma V^{\mathsf T},\qquad
H_2 = \tilde{H} / s_2,
\tag{16}
```

where ``s_2`` is the middle singular value. After this, ``H_2`` has singular
values ``(s_1, 1, s_3)`` with ``s_1 \ge 1 \ge s_3``.

### Step 2: eigendecomposition of H₂ᵀH₂

Compute

```math
H_2^{\mathsf T} H_2 = V \Sigma^2 V^{\mathsf T}, \qquad
\Sigma^2 = \mathrm{diag}(s_1^2, s_2^2, s_3^2).
\tag{17}
```

Shekhovtsov's PyTorch port computes this with `torch.linalg.eigh` (symmetric
eigendecomposition), which returns eigenvalues in ascending order, and
extracts

```math
s_1^2 = \text{evals}[2],\quad s_2^2 = \text{evals}[1],\quad s_3^2 = \text{evals}[0],\qquad
v_k = \text{evecs}[:,k].
\tag{18}
```

### Step 3: construct twin vectors in the v₁–v₃ plane

Define

```math
\boxed{\;u_{1,2} \;=\; \frac{\sqrt{1 - s_3^2}\, v_1 \;\pm\; \sqrt{s_1^2 - 1}\, v_3}{\sqrt{s_1^2 - s_3^2}}\;}
\tag{19}
```

The ``\pm`` sign gives two candidate unit vectors in the ``(v_1, v_3)`` plane.
These play the role of the sign-ambiguous ``x_1, x_3`` in Faugeras–Lustman.

### Step 4: orthonormal bases via cross-products

For each ``u_i \in \{u_1, u_2\}``, build two orthonormal triads:

```math
U_i = \bigl[\, v_2,\quad u_i,\quad v_2 \times u_i\, \bigr],
\tag{20}
```

```math
W_i = \bigl[\, H_2\, v_2,\quad H_2\, u_i,\quad (H_2\, v_2) \times (H_2\, u_i)\, \bigr].
\tag{21}
```

``U_i`` is the source frame; ``W_i`` is the image of ``U_i``'s first two
columns under ``H_2``, with the third column taken as the cross product
(forcing it to be a right-handed orthonormal triad).

### Step 5: rotation via orthogonal Procrustes

Each column of ``U_i`` maps to the corresponding column of ``W_i`` under the
rotation, so

```math
\boxed{\;R_i = W_i\, U_i^{\mathsf T}\;}
\tag{22}
```

is the orthogonal Procrustes solution (since both ``U_i`` and ``W_i`` are
orthonormal, ``W_i U_i^{\mathsf T}`` is itself a rotation).

### Step 6: plane normal and translation

```math
n_i = v_2 \times u_i, \qquad \text{flipped so } (n_i)_3 \ge 0,
\tag{23}
```

```math
t_i = (H_2 - R_i)\, n_i.
\tag{24}
```

The translation follows from the identity
``H_2 = R + t\, n^{\mathsf T} \;\Longrightarrow\; (H_2 - R)\, n = t\, n^{\mathsf T} n = t``
(since ``\|n\|=1``).

### Step 7: four candidate solutions

Because ``(R, t, n) \mapsto (R, -t, -n)`` yields the same ``H`` (see
[Section 5](@ref hdecomp-normal)), each of the two rotations ``R_1, R_2``
produces two valid pairings:

| Solution # | ``R`` | ``t`` | ``n`` |
|:--:|:--:|:--:|:--:|
| 0 | ``R_1`` | ``+t_1`` | ``+n_1`` |
| 1 | ``R_1`` | ``-t_1`` | ``-n_1`` |
| 2 | ``R_2`` | ``+t_2`` | ``+n_2`` |
| 3 | ``R_2`` | ``-t_2`` | ``-n_2`` |

These are the four candidate decompositions returned by
`cv2.decomposeHomographyMat`, PoseLib's `motion_from_homography`, and
Shekhovtsov's port.

### Comparison to Faugeras–Lustman

Both algorithms produce the same four candidates up to numerical precision.
Differences:

- **Faugeras–Lustman**: operates directly on the SVD of ``\tilde{H}``,
  closed-form expressions for ``x_1, x_3``. Requires case analysis for
  degenerate singular-value configurations.
- **Malis–Vargas**: operates on the symmetric matrix
  ``\tilde{H}^{\mathsf T} \tilde{H}`` via `eigh`. Numerically more stable for
  near-symmetric homographies. The Procrustes step ``R = W U^{\mathsf T}``
  handles sign conventions implicitly.

OpenCV chose Malis–Vargas; since PoseLib and Shekhovtsov both adopt the same
algorithm, it is the de-facto standard for reproducibility.

## [4. Degenerate Cases](@id hdecomp-degenerate)

Equation (8) is well-defined only in the generic case ``d'_1 > 1 > d'_3``.
Special cases:

**``d_1 = d_2 = d_3`` (pure rotation).** The homography reduces to
``H = K_2 R K_1^{-1}``, i.e., ``\boldsymbol{t} = \boldsymbol{0}``. The plane
normal is undetermined. Return only the rotation.

**``d_1 = d_2 > d_3`` or ``d_1 > d_2 = d_3``.** Two singular values coincide.
The decomposition is still valid but has a continuous family of solutions along
one axis. Fix one representative by convention; in practice this case is
vanishingly rare for realistic data.

**``d_3 \approx 0``.** The homography is near-rank-deficient, i.e., the minimal
solver or sample was degenerate. Decomposition becomes numerically unreliable
and downstream sample-validity checks (conditioning, cheirality) would have
already rejected the model.

## [5. Role of the Plane Normal](@id hdecomp-normal)

The plane normal ``\boldsymbol{n}`` is an *output* of the decomposition, not an
input. The 4-fold sign ambiguity lives in ``(\boldsymbol{n}, \boldsymbol{t})``:
flipping both ``\boldsymbol{n}`` and ``\boldsymbol{t}`` yields the same ``H``
(same plane seen "from the other side" with opposite pose convention). Physical
validity (cheirality) fixes which of the four algebraic solutions is real.

For pose-error measurement we only need ``(R, \boldsymbol{t})``;
``\boldsymbol{n}`` is a disambiguating byproduct and does not appear in the
final ``(e_R, e_T)`` metric. It becomes useful if downstream reasoning wants to
know *which plane* the homography corresponds to — e.g., to distinguish front
wall from ground plane in a scene with multiple planes.

## [References](@id hdecomp-references)

```@bibliography
Pages = []
Canonical = false

Barath2023
FaugerasLustman1988
MalisVargas2007
```
