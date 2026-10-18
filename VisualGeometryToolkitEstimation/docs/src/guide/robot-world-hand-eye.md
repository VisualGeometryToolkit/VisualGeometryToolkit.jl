# Robot–World Hand–Eye Calibration

*The `fit_robot_world_hand_eye` estimator: staged linear robot–world /
multi-marker hand–eye calibration.*

## [1. Problem](@id rwhe-problem)

The robot–world calibration with ``M`` markers sharing one camera transform:

```math
A_i\, X_m \;=\; Y\, B_{im},
\qquad i = 1,\dots,N, \quad (i,m) \in \mathcal{O} ,
\tag{1}
```

where

- ``A_i = (R_{A_i}, \boldsymbol{t}_{A_i}) \in \mathrm{SE}(3)`` — rig pose in
  the reference frame ``W`` at frame ``i`` (motion capture; treated as
  *exact*, its noise being ``\sim\!100\times`` below detection noise);
- ``B_{im} = (R_{B_{im}}, \boldsymbol{t}_{B_{im}}) \in \mathrm{SE}(3)`` — pose
  of marker ``m`` in the camera at frame ``i`` (PnP detection; the
  *measurements*);
- ``X_m = (R_{X_m}, \boldsymbol{t}_{X_m}) \in \mathrm{SE}(3)`` — marker ``m``
  in the rig (``M`` unknowns);
- ``Y = (R_Y, \boldsymbol{t}_Y) \in \mathrm{SE}(3)`` — camera in the reference
  frame (one shared unknown).

Unlike the classical ratio form ``A_{ij}X = XB_{ij}`` (which eliminates ``Y``
and therefore lives in correlated relative-motion space), ``Y`` is kept as a
parameter and all ``M{+}1`` transforms are estimated *jointly* from the
per-observation poses. With a pose prior (mocap), no RANSAC layer is used.

## [2. Estimator: the Staged Linear Solve](@id rwhe-init)

All data, no sampling:

1. **``R_Y`` by Kabsch.** Left-relative motions
   ``M^{A}_{ij} = A_i A_j^{-1}``, ``M^{B}_{ij} = B_{im} B_{jm}^{-1}``
   conjugate by the shared ``Y`` only, so their angle-scaled axes satisfy
   ``\omega^{B} = R_Y^{\top}\, \omega^{A}``. Pooled over all markers and frame
   pairs with ``\lVert\omega^A\rVert`` above a small-angle gate:
   ``R_Y^\top = \arg\min_R \sum \lVert \omega^{B} - R\,\omega^{A} \rVert^2``
   (one SVD).
2. **``R_{X_m}`` by chordal mean.** From
   ``R_{X_m} = R_{A_i}^{\top} R_Y R_{B_{im}}`` per observation, the chordal
   ``L_2`` mean (SVD projection of the sum onto ``\mathrm{SO}(3)``) per marker.
3. **All translations jointly.** The translation part of (1) is linear in
   ``(\boldsymbol{t}_Y, \boldsymbol{t}_{X_1}, \dots, \boldsymbol{t}_{X_M})``:
   ``\boldsymbol{t}_{B} = R_Y^{\top} R_{A} \boldsymbol{t}_{X_m} - R_Y^{\top} \boldsymbol{t}_Y + R_Y^{\top} \boldsymbol{t}_{A}``;
   one least-squares solve in ``3(M{+}1)`` unknowns — *exactly* optimal in the
   translations given the rotations.

Each stage optimizes a different functional, so the combined estimate is not
the stationary point of any single scalar objective; empirically
([Section 4](@ref rwhe-nlls)) this is a feature, not a defect, under the actual
error structure of the data.

## [3. Diagnostic Residual, Covariance, Observability](@id rwhe-diagnostics)

For reporting and uncertainty, the per-observation *decoupled* ``6``-vector
residual in the tangent at the prediction ``\hat B_{im} = Y^{-1} A_i X_m`` is
evaluated at the estimate:

```math
r_{im} \;=\;
\begin{bmatrix}
  \operatorname{Log}\!\bigl(\hat{R}_{im}^{\top} R_{B_{im}}\bigr)/\sigma_R \\[1.2ex]
  \bigl(\boldsymbol{t}_{B_{im}} - \hat{\boldsymbol{t}}_{im}\bigr)/\sigma_t
\end{bmatrix}
\;\in\; \mathbb{R}^{6},
\tag{2}
```

with ``\operatorname{Log} : \mathrm{SO}(3) \to \mathbb{R}^3`` the
rotation-vector logarithm. When not supplied, ``\sigma_R, \sigma_t`` are
profiled as the ``\operatorname{RMS}`` of the respective blocks of the unscaled
residuals at the estimate.

Because the rotation stages do not minimize (2), the estimate is not a
stationary point of the residual sum of squares (only the translation block
is), and the residual's Gauss–Newton covariance
``\hat s^{2}\,(J^{\top} J)^{-1}`` would understate the scatter. The reported
covariance is instead that of the staged estimate itself, first order in the
observation noise ``\hat s^{2}\operatorname{diag}(\sigma_R^2 I_3,\, \sigma_t^2 I_3)``
of each pose observation, with

```math
\hat s^{2} \;=\; \frac{\sum \lVert r_{im}\rVert^2}{6n - 6(M{+}1)} .
\tag{3}
```

The implicit function theorem on the stages' own stationarity conditions
(the Kabsch rotation, the chordal means, the translation normal equations)
carries that noise through to ``(Y, X_1, \dots, X_M)``. The covariance lives in
the tangent coordinates (rad, translation units) of one rigid-motion chart per
block, in the order ``[\,Y \,|\, X_1 \cdots X_M\,]``. The marginal covariance of
``Y`` is the corresponding ``6{\times}6`` sub-block of the joint covariance,
which accounts for the ``X_m`` coupling.

With ``J = \partial r / \partial \delta`` the Jacobian of (2) over the same
chart, the smallest eigenpair of ``J^\top J`` is an observability diagnostic:
a collapsed eigenvalue names the unconstrained direction (e.g. the yaw
``\oplus`` ``\boldsymbol{t}`` family of single-distance calibration data)
instead of hiding it. When ``J^\top J`` is numerically rank-deficient the
covariance is reported unbounded (``\infty``).

## [4. Why There Is No Pose-Residual NLLS Refinement](@id rwhe-nlls)

A joint NLLS refinement of (2) (Levenberg–Marquardt over the stacked MRP
``\oplus`` translation increment; the MLE under an isotropic, homoscedastic
pose-noise model) was implemented, validated on synthetic data (noiseless
exactness, Monte-Carlo ``\chi^2`` covariance calibration, degeneracy
regression), and then *removed* after evaluation on the motivating dataset:

- In-sample it behaved exactly as theory predicts: cost decreased by
  ``\Delta C \approx 283`` from the staged linear solution, consistent with the
  quadratic model for the observed ``\approx 0.17^\circ`` parameter shift
  (``\approx 17\sigma`` statistical).
- Out-of-sample it *extrapolated worse*: common-mode translation bands across
  the evaluation range grew from ``2.8/4.2/6.7`` mm (staged linear) to
  ``3.3/6.3/10.4`` mm.

This is the signature of model misspecification: the pose-noise model (2) is
isotropic and homoscedastic, while the real PnP errors are neither
(depth-vs-lateral anisotropy; a ``4``-blob sliver counts like a ``1000``-blob
board). The MLE of a misspecified model absorbs the unmodeled systematic along
weakly constrained parameter directions — in-sample optimality, out-of-sample
bias. The staged linear solve, which never optimizes the misspecified
likelihood, is the calibration of record.

The sound upgrade is not a better pose-noise model but a *measurement-level*
objective: minimize the reprojection error of the extracted image features
(blob centroids, tag corners) over ``(Y, X_1,\dots,X_M)``, initialized at the
staged linear solution. Per-feature pixel residuals make heteroscedasticity
automatic (each board contributes as many residuals as it has features) and
remove PnP as an intermediate estimator entirely. This is the objective
`refine_robot_world_hand_eye` minimizes.
