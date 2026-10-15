# Homography Cheirality

*Why the homography cheirality check is "all ``\lambda_i`` have the same
sign".*

## [1. Setup](@id cheir-setup)

Consider two pinhole cameras observing a single world plane ``\pi``. Working in
camera 1's coordinate frame, let ``\boldsymbol{n}\in\mathbb{R}^3`` be the
plane's unit normal and ``d>0`` its signed distance from the camera centre, so
the plane equation is

```math
\boldsymbol{n}^{\mathsf{T}} X \;=\; d,
\qquad X \in \pi
\tag{1}
```

with ``d>0`` meaning the plane is *in front of* camera 1.

Let ``(R,\boldsymbol{t})`` be the rigid transformation from camera 1 to camera
2, and ``K_1, K_2`` be the intrinsics. A 3-D point ``X_{\text{cam}_{1}}`` in
camera 1's frame maps to

```math
X_{\text{cam}_{2}} \;=\; R\,X_{\text{cam}_{1}} + \boldsymbol{t}.
\tag{2}
```

Its homogeneous pixel projections are ``x = K_1 X_{\text{cam}_{1}}`` and
``x' = K_2 X_{\text{cam}_{2}}``.

## [2. Plane-Induced Homography](@id cheir-homography)

For any ``X\in\pi`` we have ``\boldsymbol{n}^{\mathsf{T}} X = d``, i.e.
``\boldsymbol{n}^{\mathsf{T}} X / d = 1``. Substitute into the inter-camera
transform:

```math
\begin{aligned}
X_{\text{cam}_{2}}
&= R\,X_{\text{cam}_{1}} + \boldsymbol{t} \\
&= R\,X_{\text{cam}_{1}} + \boldsymbol{t} \cdot
    \frac{\boldsymbol{n}^{\mathsf{T}} X_{\text{cam}_{1}}}{d} \\
&= \left(R + \tfrac{1}{d}\,\boldsymbol{t}\,\boldsymbol{n}^{\mathsf{T}}\right) X_{\text{cam}_{1}}.
\end{aligned}
\tag{3}
```

Adding the intrinsics, the pixel-space homography induced by the plane is

```math
\boxed{\;H \;=\; K_2\!\left(R + \tfrac{1}{d}\,\boldsymbol{t}\,\boldsymbol{n}^{\mathsf{T}}\right) K_1^{-1}.\;}
\tag{4}
```

## [3. Projective Scale Factor λᵢ](@id cheir-lambda)

Write the correspondence in homogeneous image coordinates:

```math
x'_i \;=\; \lambda_i \, H \, x_i.
\tag{5}
```

We show that

```math
\boxed{\;\lambda_i \;=\; \dfrac{Z_{1,i}}{Z_{2,i}}\;}
\tag{6}
```

where ``Z_{1,i}`` and ``Z_{2,i}`` are the depths (i.e. ``z``-components) of the
3-D point ``X_i`` in camera 1's and camera 2's frames respectively.

**Derivation.** Let ``u_i = K_1^{-1} x_i`` and ``u'_i = K_2^{-1} x'_i`` be the
normalized image points. From the projection equations,

```math
u_i = \frac{X_{\text{cam}_{1},i}}{Z_{1,i}},
\qquad
u'_i = \frac{X_{\text{cam}_{2},i}}{Z_{2,i}}.
\tag{7}
```

Multiplying the normalized-frame plane homography by ``X_{\text{cam}_{1},i}``,

```math
\left(R + \tfrac{1}{d}\boldsymbol{t}\boldsymbol{n}^{\mathsf{T}}\right) X_{\text{cam}_{1},i}
\;=\; X_{\text{cam}_{2},i},
\tag{8}
```

and dividing both sides by ``Z_{1,i}``,

```math
\left(R + \tfrac{1}{d}\boldsymbol{t}\boldsymbol{n}^{\mathsf{T}}\right) u_i
\;=\; \frac{X_{\text{cam}_{2},i}}{Z_{1,i}}
\;=\; \frac{Z_{2,i}}{Z_{1,i}} \, u'_i.
\tag{9}
```

Equivalently,

```math
u'_i
\;=\; \frac{Z_{1,i}}{Z_{2,i}}
    \left(R + \tfrac{1}{d}\boldsymbol{t}\boldsymbol{n}^{\mathsf{T}}\right) u_i.
\tag{10}
```

Left-multiplying by ``K_2`` and using ``K_1^{-1} x_i = u_i`` on the right gives
(5) with ``\lambda_i = Z_{1,i}/Z_{2,i}``, as claimed.

## [4. Cheirality](@id cheir-cheirality)

For a physically meaningful correspondence, the 3-D point ``X_i`` must lie in
front of both cameras:

```math
Z_{1,i} > 0 \quad\text{and}\quad Z_{2,i} > 0.
\tag{11}
```

Therefore

```math
\lambda_i \;=\; \frac{Z_{1,i}}{Z_{2,i}} \;>\; 0 \qquad \forall i.
\tag{12}
```

**Invariance under ``H\mapsto -H``.** The homography is a projective object,
defined up to a non-zero global scale. Replacing ``H\mapsto -H`` sends
``\lambda_i\mapsto -\lambda_i`` *simultaneously for every ``i``*, so the
cheirality condition is really

```math
\text{“}\lambda_i \text{ all have the same sign”.}
\tag{13}
```

In practice one may normalize ``H`` so that ``\det H > 0``; after this
normalization the condition becomes the one-sided test

```math
\lambda_i > 0 \quad \forall i.
\tag{14}
```

VisualGeometryCore's `test_cheirality` tests the sign-uniform form (13)
directly, which needs no sign normalization of ``H``.

## [5. When Signs Disagree](@id cheir-mixed)

Mixed-sign ``\lambda_i``'s appear precisely when the 4 correspondences cannot
be explained by a single plane in front of both cameras:

- **Outlier correspondence.** The pair ``(x_i,x'_i)`` does not come from a
  common 3-D point, so the depth-ratio identity (6) does not apply and
  ``\lambda_i`` can take any sign.
- **Bow-tie / non-planar sample.** The 4 points cannot lie on a single plane in
  the shared-visibility half-space; some subset gets placed behind a camera,
  flipping the sign of its ``\lambda``.
- **Near-degenerate sample** (e.g. three collinear points). Numerical error in
  the 4-point solver pushes some ``\lambda_i`` slightly negative.

All three cases are rejected by the same *all-positive-``\lambda``* test.
