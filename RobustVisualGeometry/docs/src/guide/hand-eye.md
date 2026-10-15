# Rig-to-Pattern Hand–Eye Calibration

*Robust estimation of ``T_{RP}`` from synchronized motion capture and visual
pattern tracking.*

## [1. Problem Setup](@id handeye-setup)

A rigid body (*rig*) is tracked by an external motion-capture system, which
reports the rig pose in a world frame at each time index ``i``. Attached
rigidly to the rig is a calibration pattern (e.g., checkerboard, ChArUco),
observed by a static camera fixed in the world. Per-frame PnP on the pattern
returns the pattern pose in the camera frame. The transform between the rig
frame ``R`` and the pattern frame ``P`` is unknown but constant; recovering it
is the calibration goal.

**Coordinate frames.**

- ``W`` – mocap world frame
- ``R`` – rig frame (mocap-tracked rigid body)
- ``P`` – pattern frame (rigidly attached to ``R``)
- ``C`` – camera frame (fixed in ``W``)

**Per-frame measurements.** At time ``i = 1, \ldots, N``:

```math
A_i \;:=\; T_{WR}^{(i)} \;\in\; \mathrm{SE}(3) \quad \text{(rig in world, from mocap)}
\tag{1}
```

```math
B_i \;:=\; T_{CP}^{(i)} \;\in\; \mathrm{SE}(3) \quad \text{(pattern in camera, from PnP)}
\tag{2}
```

**Unknowns (constant in ``i``).**

```math
X \;:=\; T_{RP} \;\in\; \mathrm{SE}(3) \quad \text{(pattern in rig -- the goal)}
\tag{3}
```

```math
Y \;:=\; T_{WC} \;\in\; \mathrm{SE}(3) \quad \text{(camera in world -- nuisance)}
\tag{4}
```

**Constraint.** The pattern-in-world pose ``T_{WP}`` at time ``i`` may be
expressed via either chain:

```math
T_{WP}^{(i)} \;=\; T_{WR}^{(i)} \, T_{RP}
            \;=\; T_{WC}^{(i)} \, T_{CP}^{(i)}.
```

Since ``T_{WC}`` is constant (``Y``) and ``T_{RP}`` is constant (``X``), we
obtain the classical robot–world / hand–eye constraint

```math
\boxed{\; A_i\, X \;=\; Y\, B_i, \qquad i = 1, \ldots, N \;}
\tag{5}
```

holding simultaneously for all ``i``.

**Parameterization.** Write each pose as ``E = (\mathbf{R}, \mathbf{t})`` with
action ``E(\boldsymbol{x}) = \mathbf{R} \boldsymbol{x} + \mathbf{t}``. Then (5)
decouples into a rotation equation and a translation equation:

```math
\mathbf{R}_{A_i} \mathbf{R}_X = \mathbf{R}_Y \mathbf{R}_{B_i},
\tag{6}
```

```math
\mathbf{R}_{A_i} \mathbf{t}_X + \mathbf{t}_{A_i} = \mathbf{R}_Y \mathbf{t}_{B_i} + \mathbf{t}_Y.
\tag{7}
```

## [2. Identifiability](@id handeye-identifiability)

A single motion pair ``(A_i, B_i)`` is insufficient: with one observation, (5)
constrains the composition ``Y B_i X^{-1}`` but leaves a continuous family of
``(X, Y)`` that satisfy it.

**Differential form.** For two motions ``(A_i, B_i)`` and ``(A_j, B_j)``, the
relative motions ``A_{ij} := A_i^{-1} A_j`` and ``B_{ij} := B_i^{-1} B_j``
satisfy

```math
X^{-1} A_{ij}\, X = Y^{-1} B_{ij}\, Y^{-1}\cdot Y = \cdots
```

The standard result (Tsai & Lenz 1989; Chen 1991) is:

> *The hand–eye problem is solvable iff there exist at least two relative
> motions whose rotation axes are not parallel.*

With a smooth trajectory of ``N \gg 2`` poses, this is satisfied generically,
but a degenerate dataset (e.g., the rig only rotated about gravity) leaves the
component about the common axis unobservable. The largest pairwise angle
between rotation axes of ``\{R_{A_i}\}`` serves as a diagnostic: a value below
a threshold signals this degeneracy.

## [3. Closed-Form Initialization (Shah)](@id handeye-shah)

Shah's 2013 solver yields a closed-form seed for ``(X, Y)`` by separating the
rotation and translation blocks.

### 3.1 Rotation block

Vectorizing (6) using the identity
``\mathrm{vec}\!\left(AXB\right) = (B^\top \otimes A)\,\mathrm{vec}\!\left(X\right)``:

```math
\big(\mathbf{I}_3 \otimes \mathbf{R}_{A_i}\big)\, \mathrm{vec}\!\left(\mathbf{R}_X\right)
\;-\;
\big(\mathbf{R}_{B_i}^{\!\top} \otimes \mathbf{I}_3\big)\, \mathrm{vec}\!\left(\mathbf{R}_Y\right)
\;=\; \boldsymbol{0}.
```

Stacking ``N`` such ``9 \times 18`` blocks gives a system

```math
\mathbf{M}\,\boldsymbol{v} = \boldsymbol{0},
\qquad
\boldsymbol{v} = \begin{bmatrix} \mathrm{vec}\!\left(\mathbf{R}_X\right) \\ \mathrm{vec}\!\left(\mathbf{R}_Y\right) \end{bmatrix} \in \mathbb{R}^{18}.
```

The right singular vector corresponding to the smallest singular value of
``\mathbf{M}`` provides the least-squares solution. Reshape into two
``3\times3`` matrices ``\hat{\mathbf{R}}_X, \hat{\mathbf{R}}_Y``, then project
each to the nearest rotation in ``\mathrm{SO}(3)`` via SVD:

```math
\mathbf{R}_X \;=\; U \,\mathrm{diag}(1, 1, \det(UV^\top))\, V^\top,
\qquad \hat{\mathbf{R}}_X = U\Sigma V^\top.
```

(The diagonal correction enforces ``\det \mathbf{R}_X = +1``, avoiding
accidental reflections.)

### 3.2 Translation block

With ``\mathbf{R}_X, \mathbf{R}_Y`` now fixed, (7) is linear in
``(\mathbf{t}_X, \mathbf{t}_Y)``:

```math
\begin{bmatrix} \mathbf{R}_{A_i} & -\mathbf{I}_3 \end{bmatrix}
\begin{bmatrix} \mathbf{t}_X \\ \mathbf{t}_Y \end{bmatrix}
\;=\;
\mathbf{R}_Y \mathbf{t}_{B_i} - \mathbf{t}_{A_i}.
```

Stacking yields a ``3N \times 6`` overdetermined system; solve in the
least-squares sense (QR or normal equations).

## [4. Stage A: Decoupled SE(3) Refinement](@id handeye-stage-a)

The closed-form seed minimizes algebraic error, which mixes degrees of freedom
unequally and is sensitive to outliers. Stage A refines ``(X, Y)`` by
minimizing a robust pose-pair residual.

### 4.1 Residual definition

Define

```math
\mathbf{R}_{e,i} \;:=\; \mathbf{R}_Y\, \mathbf{R}_{B_i}\, \mathbf{R}_X^{\!\top}\, \mathbf{R}_{A_i}^{\!\top},
\tag{8}
```

```math
\mathbf{t}_{e,i} \;:=\; \big(\mathbf{R}_Y \mathbf{t}_{B_i} + \mathbf{t}_Y\big) - \big(\mathbf{R}_{A_i} \mathbf{t}_X + \mathbf{t}_{A_i}\big).
\tag{9}
```

At the optimum ``\mathbf{R}_{e,i} = \mathbf{I}_3`` and
``\mathbf{t}_{e,i} = \boldsymbol{0}`` for every ``i``. Pack the rotation error
into its axis–angle vector via the inverse exponential map of
``\mathrm{SO}(3)``:

```math
\boldsymbol{\omega}_i \;:=\; \operatorname{Log}\!\big(\mathbf{R}_{e,i}\big) \;\in\; \mathbb{R}^3
```

(implemented as Rotations.jl's `Rotations.params(RotationVec(R))`). The
per-frame stacked residual is

```math
\boldsymbol{r}_i(X, Y) \;=\;
\begin{bmatrix} \boldsymbol{\omega}_i \\ \mathbf{t}_{e,i} \end{bmatrix}
\;\in\; \mathbb{R}^6.
\tag{10}
```

### 4.2 Cost

Let ``\rho : \mathbb{R}_{\geq 0} \to \mathbb{R}_{\geq 0}`` be a robust loss
(Huber, Cauchy, ``\ldots``), ``\hat\sigma`` a robust scale estimate (e.g., MAD
over ``\{\|\boldsymbol{r}_i\|\}``). The Stage A cost is

```math
J_A(X, Y) \;=\; \sum_{i=1}^{N} \rho\!\bigg(\frac{\|\boldsymbol{r}_i(X, Y)\|}{\hat\sigma}\bigg).
\tag{11}
```

### 4.3 Left-perturbation parameterization

To take Gauss–Newton steps on the manifold
``\mathrm{SE}(3)\times\mathrm{SE}(3)`` we parameterize local updates by a
``12``-vector

```math
\boldsymbol{\delta} \;=\; (\boldsymbol{\delta}_{\!\omega X}, \boldsymbol{\delta}_{t X}, \boldsymbol{\delta}_{\!\omega Y}, \boldsymbol{\delta}_{t Y})
\;\in\; \mathbb{R}^{12}
```

with the left-perturbation retraction

```math
X \;\leftarrow\; \Delta X \cdot X,
\qquad
\Delta X \;=\;
\big(\mathbf{R}(\boldsymbol{\delta}_{\!\omega X}),\; \boldsymbol{\delta}_{t X}\big),
\tag{12}
```

and analogously for ``Y``. Here
``\mathbf{R}(\boldsymbol{\omega}) := \operatorname{Exp}_{\mathrm{SO}(3)}[\boldsymbol{\omega}]_{\!\times}``
(implemented as Rotations.jl's `RotMatrix(RotationVec(ω))`). Concretely,

```math
\mathbf{R}_X \mapsto \mathbf{R}(\boldsymbol{\delta}_{\!\omega X})\, \mathbf{R}_X,
\qquad
\mathbf{t}_X \mapsto \mathbf{R}(\boldsymbol{\delta}_{\!\omega X})\, \mathbf{t}_X + \boldsymbol{\delta}_{t X}.
```

### 4.4 Analytical Jacobians

Differentiating (8)–(9) at ``\boldsymbol{\delta} = 0``.

**Rotation residual.** Using
``\mathbf{R}(\boldsymbol{\delta})^{\!\top} \approx \mathbf{I} - [\boldsymbol{\delta}]_{\!\times}``
for small ``\boldsymbol{\delta}`` and the identity
``\mathbf{R}[v]_{\!\times}\mathbf{R}^{\!\top} = [\mathbf{R} v]_{\!\times}`` for
any rotation ``\mathbf{R}``:

```math
\begin{aligned}
\mathbf{R}_{e,i}\,(\boldsymbol{\delta}_{\!\omega X})
&= \mathbf{R}_Y \mathbf{R}_{B_i} \big(\mathbf{R}(\boldsymbol{\delta}_{\!\omega X})\mathbf{R}_X\big)^{\!\top}\mathbf{R}_{A_i}^{\!\top}
\;=\; \mathbf{R}_Y \mathbf{R}_{B_i}\mathbf{R}_X^{\!\top}\big(\mathbf{I} - [\boldsymbol{\delta}_{\!\omega X}]_{\!\times}\big)\mathbf{R}_{A_i}^{\!\top}\\
&= \mathbf{R}_{e,i} - \mathbf{R}_{e,i}\, [\mathbf{R}_{A_i} \boldsymbol{\delta}_{\!\omega X}]_{\!\times}
\;\approx\; \mathbf{R}_{e,i}\,\operatorname{Exp}\![-\mathbf{R}_{A_i}\boldsymbol{\delta}_{\!\omega X}]_{\!\times}.
\end{aligned}
```

For ``\boldsymbol{\delta}_{\!\omega Y}``:

```math
\mathbf{R}_{e,i}\,(\boldsymbol{\delta}_{\!\omega Y})
= \mathbf{R}(\boldsymbol{\delta}_{\!\omega Y}) \mathbf{R}_Y \mathbf{R}_{B_i} \mathbf{R}_X^{\!\top}\mathbf{R}_{A_i}^{\!\top}
= \operatorname{Exp}\![\boldsymbol{\delta}_{\!\omega Y}]_{\!\times} \, \mathbf{R}_{e,i}.
```

Taking ``\operatorname{Log}`` and using the right (resp. left) Jacobians
``J_r, J_l`` of ``\mathrm{SO}(3)``:

```math
\frac{\partial \boldsymbol{\omega}_i}{\partial \boldsymbol{\delta}_{\!\omega X}} = -\,J_r^{-1}(\boldsymbol{\omega}_i)\, \mathbf{R}_{A_i},
\quad
\frac{\partial \boldsymbol{\omega}_i}{\partial \boldsymbol{\delta}_{\!\omega Y}} = +\,J_l^{-1}(\boldsymbol{\omega}_i),
\quad
\frac{\partial \boldsymbol{\omega}_i}{\partial \boldsymbol{\delta}_{t X}} = \mathbf{0},
\quad
\frac{\partial \boldsymbol{\omega}_i}{\partial \boldsymbol{\delta}_{t Y}} = \mathbf{0}.
\tag{13}
```

Near convergence ``\mathbf{R}_{e,i} \to \mathbf{I}``, so
``\boldsymbol{\omega}_i \to \boldsymbol{0}`` and
``J_l^{-1}(\boldsymbol{\omega}_i), J_r^{-1}(\boldsymbol{\omega}_i) \to \mathbf{I}_3``.
In practice we use the near-identity approximation throughout; the iteration
converges quadratically once inside the basin.

**Translation residual.** ``\mathbf{t}_{e,i}`` is independent of rotations
only through ``\mathbf{R}_Y`` acting on ``\mathbf{t}_{B_i}`` and
``\mathbf{R}_{A_i}`` acting on ``\mathbf{t}_X``. Perturbing:

```math
\begin{aligned}
\mathbf{R}_{A_i}\mathbf{t}_X &\mapsto \mathbf{R}_{A_i}\big(\mathbf{R}(\boldsymbol{\delta}_{\!\omega X})\mathbf{t}_X + \boldsymbol{\delta}_{t X}\big)
            \approx \mathbf{R}_{A_i}\mathbf{t}_X - \mathbf{R}_{A_i}[\mathbf{t}_X]_{\!\times}\boldsymbol{\delta}_{\!\omega X}
                                + \mathbf{R}_{A_i}\boldsymbol{\delta}_{t X},\\
\mathbf{R}_Y\mathbf{t}_{B_i} + \mathbf{t}_Y
&\mapsto \mathbf{R}(\boldsymbol{\delta}_{\!\omega Y})\big(\mathbf{R}_Y\mathbf{t}_{B_i} + \mathbf{t}_Y\big) + \boldsymbol{\delta}_{t Y}\\
&\approx \big(\mathbf{R}_Y\mathbf{t}_{B_i}+\mathbf{t}_Y\big) - [\mathbf{R}_Y\mathbf{t}_{B_i}+\mathbf{t}_Y]_{\!\times}\,\boldsymbol{\delta}_{\!\omega Y}
                                + \boldsymbol{\delta}_{t Y}.
\end{aligned}
```

Letting ``\boldsymbol{p}_i := \mathbf{R}_Y \mathbf{t}_{B_i} + \mathbf{t}_Y``
(pattern in world):

```math
\frac{\partial \mathbf{t}_{e,i}}{\partial \boldsymbol{\delta}_{\!\omega X}} = +\mathbf{R}_{A_i}[\mathbf{t}_X]_{\!\times},
\quad
\frac{\partial \mathbf{t}_{e,i}}{\partial \boldsymbol{\delta}_{t X}}      = -\mathbf{R}_{A_i},
\quad
\frac{\partial \mathbf{t}_{e,i}}{\partial \boldsymbol{\delta}_{\!\omega Y}} = -[\boldsymbol{p}_i]_{\!\times},
\quad
\frac{\partial \mathbf{t}_{e,i}}{\partial \boldsymbol{\delta}_{t Y}}       = +\mathbf{I}_3.
\tag{14}
```

**Per-frame Jacobian.** Assembling (13)–(14) into the ``6 \times 12`` block
(rows: ``[\boldsymbol{\omega}_i\,;\,\mathbf{t}_{e,i}]``, columns:
``[\boldsymbol{\delta}_{\!\omega X}, \boldsymbol{\delta}_{t X}, \boldsymbol{\delta}_{\!\omega Y}, \boldsymbol{\delta}_{t Y}]``):

```math
\mathbf{J}_i \;=\;
\begin{bmatrix}
  -\mathbf{R}_{A_i}                  & \mathbf{0}_3   &  +\mathbf{I}_3            & \mathbf{0}_3 \\[2pt]
  +\mathbf{R}_{A_i}[\mathbf{t}_X]_{\!\times}     & -\mathbf{R}_{A_i}      &  -[\boldsymbol{p}_i]_{\!\times} & +\mathbf{I}_3
\end{bmatrix}
\in \mathbb{R}^{6 \times 12}
\tag{15}
```

(omitting the near-identity ``J_l^{-1}, J_r^{-1}`` factors in the rotation
block; they may be retained for principled steps far from the optimum).

### 4.5 IRLS update

At iterate ``(X^{(k)}, Y^{(k)})``:

1. Compute residuals ``\boldsymbol{r}_i`` via (10) and per-residual magnitudes
   ``\eta_i = \|\boldsymbol{r}_i\|``.
2. Update scale: ``\hat\sigma \leftarrow \mathrm{MAD}(\eta_i) / 0.6745``.
3. Per-residual weight:
   ``w_i = \psi(\eta_i / \hat\sigma) / (\eta_i / \hat\sigma)``, where
   ``\psi = \rho'`` is the influence function of the robust loss.
4. Solve the normal equations

   ```math
   \Big(\sum_i w_i\, \mathbf{J}_i^{\!\top} \mathbf{J}_i\Big)\, \boldsymbol{\delta}
   \;=\;
   -\sum_i w_i\, \mathbf{J}_i^{\!\top} \boldsymbol{r}_i.
   ```

5. Retract via (12):
   ``X^{(k+1)} = (\mathbf{R}(\boldsymbol{\delta}_{\!\omega X})\mathbf{R}_X,\; \mathbf{R}(\boldsymbol{\delta}_{\!\omega X})\mathbf{t}_X + \boldsymbol{\delta}_{t X})``,
   and analogously for ``Y``.

Terminate on ``\|\boldsymbol{\delta}\| < \tau`` or after a maximum number of
iterations.

## [5. Stage B: Reprojection Refinement (Optional)](@id handeye-stage-b)

If the caller additionally supplies the 3D pattern points
``\{\boldsymbol{P}_j\}_{j=1}^M \subset \mathbb{R}^3`` (in ``P``-coordinates),
the per-frame pixel observations ``\{\boldsymbol{u}_{ij}\}``, and the camera
intrinsics ``K``, the Stage A solution is refined against pixel reprojection.

### 5.1 Geometry

The pattern point ``\boldsymbol{P}_j`` expressed in the camera frame at time
``i`` is

```math
\boldsymbol{X}_{ij}^{C} \;=\; Y^{-1}\big(A_i(X(\boldsymbol{P}_j))\big)
                \;=\; \mathbf{R}_Y^{\!\top}\bigl(\mathbf{R}_{A_i}(\mathbf{R}_X \boldsymbol{P}_j + \mathbf{t}_X) + \mathbf{t}_{A_i} - \mathbf{t}_Y\bigr).
```

Its predicted projection is ``\pi(\boldsymbol{X}_{ij}^{C})`` where
``\pi([x,y,z]^{\!\top}) = [x/z,\, y/z]^{\!\top}`` is perspective division; the
calibrated residual is

```math
\boldsymbol{g}_{ij}(X, Y) \;=\; \pi\!\big(\boldsymbol{X}_{ij}^{C}\big) \;-\; \pi\!\big(\boldsymbol{ray}_{ij}\big),
\tag{16}
```

where ``\boldsymbol{ray}_{ij}`` is the back-projected pixel observation
(calibration handled inside the ray).

### 5.2 Jacobian via composition

VisualGeometryCore exposes the calibrated reprojection Jacobian of an
``\mathrm{SE}(3)`` pose applied to a 3D point, against a left-perturbation
parameterization (`constraint_with_jacobians(pose::EuclideanMap{3}, X, ray)`):

```math
G_\theta \;=\; J_\pi\,\big[\,-[\boldsymbol{X}_{ij}^{C}]_{\!\times} \;\; \mathbf{I}_3\,\big]
\;\in\; \mathbb{R}^{2\times6}.
```

For our composite pose ``T^{(i)} = Y^{-1} A_i X``, define
``\boldsymbol{Z}_{ij} := A_i(X(\boldsymbol{P}_j))`` (pattern in world, fixed
wrt ``X`` only).

**Sensitivity to ``X``.** A left perturbation ``X \mapsto \Delta X \cdot X``
pushes through ``A_i`` unchanged in form: ``A_i \cdot \Delta X \cdot X``. The
effect on the camera-frame point is

```math
\boldsymbol{X}_{ij}^{C} \;\mapsto\; Y^{-1} A_i \Delta X (X \boldsymbol{P}_j)
\;=\; \mathbf{R}_Y^{\!\top}\mathbf{R}_{A_i}\Big(\mathbf{R}(\boldsymbol{\delta}_{\!\omega X})(\mathbf{R}_X \boldsymbol{P}_j + \mathbf{t}_X) + \boldsymbol{\delta}_{t X}\Big) + \cdots
```

To first order:

```math
\frac{\partial \boldsymbol{X}_{ij}^{C}}{\partial \boldsymbol{\delta}_{\!\omega X}}
  = -\mathbf{R}_Y^{\!\top}\mathbf{R}_{A_i}\,[\mathbf{R}_X \boldsymbol{P}_j + \mathbf{t}_X]_{\!\times},
\qquad
\frac{\partial \boldsymbol{X}_{ij}^{C}}{\partial \boldsymbol{\delta}_{t X}}
  = +\mathbf{R}_Y^{\!\top}\mathbf{R}_{A_i}.
\tag{17}
```

**Sensitivity to ``Y``.** A left perturbation ``Y \mapsto \Delta Y \cdot Y``
implies ``Y^{-1} \mapsto Y^{-1} \Delta Y^{-1}``, so

```math
\boldsymbol{X}_{ij}^{C} \;\mapsto\; \mathbf{R}_Y^{\!\top}\mathbf{R}(-\boldsymbol{\delta}_{\!\omega Y})
      \big(\boldsymbol{Z}_{ij} - \mathbf{t}_Y - \boldsymbol{\delta}_{tY}\big) + O(\boldsymbol{\delta}^2).
```

At ``\boldsymbol{\delta} = 0``,

```math
\frac{\partial \boldsymbol{X}_{ij}^{C}}{\partial \boldsymbol{\delta}_{\!\omega Y}}
  = +\mathbf{R}_Y^{\!\top}\,[\boldsymbol{Z}_{ij} - \mathbf{t}_Y]_{\!\times},
\qquad
\frac{\partial \boldsymbol{X}_{ij}^{C}}{\partial \boldsymbol{\delta}_{t Y}}
  = -\mathbf{R}_Y^{\!\top}.
\tag{18}
```

Chain rule through ``\pi`` gives the per-observation ``2\times12`` Jacobian

```math
\mathbf{H}_{ij} \;=\;
J_\pi(\boldsymbol{X}_{ij}^{C})\;
\Big[\,
   -\mathbf{R}_Y^{\!\top}\mathbf{R}_{A_i}[\mathbf{R}_X\boldsymbol{P}_j+\mathbf{t}_X]_{\!\times}
   \;\; +\mathbf{R}_Y^{\!\top}\mathbf{R}_{A_i}
   \;\; +\mathbf{R}_Y^{\!\top}[\boldsymbol{Z}_{ij}-\mathbf{t}_Y]_{\!\times}
   \;\; -\mathbf{R}_Y^{\!\top}\,
\Big].
\tag{19}
```

### 5.3 Stage B cost and IRLS step

```math
J_B(X, Y) \;=\; \sum_i \sum_j \rho\!\left(\frac{\|\boldsymbol{g}_{ij}(X,Y)\|}{\hat\sigma_{\!px}}\right),
\tag{20}
```

with ``\hat\sigma_{\!px}`` a robust scale estimate over the pixel residuals.
The IRLS update is identical in structure to Stage A but assembles

```math
\mathbf{H} = \sum_{ij} w_{ij}\, \mathbf{H}_{ij}^{\!\top}\mathbf{H}_{ij},
\qquad
\boldsymbol{b} = -\sum_{ij} w_{ij}\, \mathbf{H}_{ij}^{\!\top}\boldsymbol{g}_{ij},
\tag{21}
```

solves ``\mathbf{H}\boldsymbol{\delta} = \boldsymbol{b}``, and retracts via
(12).

## [6. Algorithm Summary](@id handeye-algo)

**Require:** pose pairs ``\{(A_i, B_i)\}_{i=1}^N``; optional pattern points
``\{\boldsymbol{P}_j\}``, pixel observations ``\{\boldsymbol{u}_{ij}\}``,
intrinsics ``K``; loss ``\rho``, scale estimator, tolerance ``\tau``.
**Ensure:** rigid transforms ``X = T_{RP}``, ``Y = T_{WC}``.

1. **(0)** Validate inputs; warn if the rotation-axis spread is below a
   threshold ([Section 2](@ref handeye-identifiability)).
2. **(1)** *Closed-form seed* ([Section 3](@ref handeye-shah)):
   - solve ``\mathbf{M}\boldsymbol{v}=\boldsymbol{0}`` via SVD, project the
     halves to ``\mathrm{SO}(3)`` ``\Rightarrow (\mathbf{R}_X^{(0)}, \mathbf{R}_Y^{(0)})``;
   - solve the ``3N\times6`` linear least squares
     ``\Rightarrow (\mathbf{t}_X^{(0)}, \mathbf{t}_Y^{(0)})``.
3. **(2)** *Stage A: SE(3)* M-estimator IRLS
   ([Section 4](@ref handeye-stage-a)): repeat residuals (10) ``\rightarrow``
   scale ``\rightarrow`` weights ``\rightarrow`` normal equations
   ``\rightarrow`` retract, until ``\|\boldsymbol{\delta}\| < \tau``.
4. **If** pattern points, pixels and intrinsics are supplied: **(3)** *Stage B:
   reprojection* IRLS ([Section 5](@ref handeye-stage-b)): residuals (16)
   ``\rightarrow`` scale ``\rightarrow`` weights ``\rightarrow`` assemble (19)
   ``\rightarrow`` retract, until ``\|\boldsymbol{\delta}\| < \tau_{px}``.
5. **Return** ``(X, Y)``.

## [7. Implemented Pipeline (AX = XB) — Loss Variants](@id handeye-variants)

!!! note "Note"
    Sections [1](@ref handeye-setup)–[6](@ref handeye-algo) derive the
    ``AX = YB`` form with both ``X`` and ``Y`` as unknowns. The implementation
    (`find_hand_eye`, on the RANSAC problem `HandEyeProblem`) eliminates the
    nuisance ``Y`` via the relative-motion ratio

    ```math
    A_{ij} = A_j^{-1} A_i, \qquad B_{ij} = B_j^{-1} B_i,
    \qquad A_{ij}\, X \;=\; X\, B_{ij}.
    ```

The per-pair residual is the decoupled 6-vector

```math
\boldsymbol{g}_i(X) \;=\;
\begin{bmatrix}
  \operatorname{Log}\!\bigl(\mathbf{R}_{A_i}\mathbf{R}_X \mathbf{R}_{B_i}^{\!\top} \mathbf{R}_X^{\!\top}\bigr) \\[2pt]
  \mathbf{R}_{A_i}\mathbf{t}_X + \mathbf{t}_{A_i} - \mathbf{R}_X \mathbf{t}_{B_i} - \mathbf{t}_X
\end{bmatrix}
\;=\;
\begin{bmatrix} \theta_i \, \hat{\boldsymbol{n}}_i \\[2pt] \boldsymbol{t}_{e,i} \end{bmatrix}
\;\in\; \mathbb{R}^6,
\tag{22}
```

where ``\theta_i`` is the bi-invariant ``\mathrm{SO}(3)`` geodesic angle
between ``\mathbf{R}_{A_i}\mathbf{R}_X`` and ``\mathbf{R}_X\mathbf{R}_{B_i}``,
and ``\hat{\boldsymbol{n}}_i`` is the corresponding rotation axis.

All four variants below share the same residual ``\boldsymbol{g}_i(X)`` and the
same RANSAC + IRLS framework (the weighted `refine!` of any IRLS-family local
optimization); they differ only in the per-pair weighting matrix that converts
``\boldsymbol{g}_i`` into a scalar contribution to the cost. The IRLS
weights ``w_i^{(k)} \in [0,1]`` act on top of any of these losses by
multiplying the per-pair ``\ell_i``.

### [7.1 Variant 0 (implemented): single global L](@id handeye-variant-0)

Decoupled ``\mathrm{SO}(3) \times \mathbb{R}^3`` split metric with one global
length constant ``L`` converting radians to length units:

```math
\boxed{\;\ell_i^{\text{(0)}}(X) \;=\; L^2\,\theta_i^2 \;+\; \|\boldsymbol{t}_{e,i}\|^2\;}
\tag{23}
```

Equivalently ``\ell_i = \boldsymbol{g}_i^{\!\top} W \boldsymbol{g}_i`` with the
constant ``6\times6``

```math
W \;=\; \mathrm{diag}(L^2, L^2, L^2,\, 1, 1, 1)
```

applied identically to every pair. Default
``L = \mathrm{median}_i\!\|\boldsymbol{t}_{A_i}\|`` (the typical rig
translation per motion); override via the `char_length` keyword.

*Strengths.* Zero extra inputs; one scalar to set. *Weaknesses.* ``L`` is a
heuristic that depends on the rig's typical motion magnitude; treats all frames
as equally informative.

### [7.2 Variant A: two-scale data-driven Mahalanobis](@id handeye-variant-a)

Replace the single ``L`` with two independent scale estimates,
``\hat\sigma_R`` over the per-pair rotation magnitudes ``\theta_i`` and
``\hat\sigma_t`` over the translation magnitudes
``\|\boldsymbol{t}_{e,i}\|``:

```math
\boxed{\;\ell_i^{\text{(A)}}(X) \;=\; \frac{\theta_i^2}{\hat\sigma_R^2}
                                     \;+\; \frac{\|\boldsymbol{t}_{e,i}\|^2}{\hat\sigma_t^2}\;}
\tag{24}
```

Equivalently ``\ell_i = \boldsymbol{g}_i^{\!\top} W \boldsymbol{g}_i`` with the
constant

```math
W \;=\; \mathrm{diag}\bigl(\hat\sigma_R^{-2}, \hat\sigma_R^{-2}, \hat\sigma_R^{-2},
                            \hat\sigma_t^{-2}, \hat\sigma_t^{-2}, \hat\sigma_t^{-2}\bigr).
```

**EM-like scale update (from block coordinate descent).** Rather than using a
MAD estimator on the residual magnitudes (a one-shot robust statistic, blind to
the framework's own posterior weights), we reuse the weighted scale M-step of
the Student-t / uniform mixture EM for scalar residuals (its scale block is
`update_scale!`), of the form ``\hat{s}^2 = \sum_i w_i r_i^2 / \sum_i w_i``.
Applied componentwise:

```math
\boxed{\;\hat\sigma_R^2 \;=\;
        \frac{\sum_i w_i^{(k)}\,\theta_i^2}{\sum_i w_i^{(k)}},
        \qquad
        \hat\sigma_t^2 \;=\;
        \frac{\sum_i w_i^{(k)}\,\|\boldsymbol{t}_{e,i}\|^2}{\sum_i w_i^{(k)}}\;}
\tag{25}
```

where ``w_i^{(k)}`` are the current Student-t/uniform posterior weights from
the E-step (`soft_posterior_weights!` / `update_weights!`). Each LO iteration
runs:

1. M-step (scale): ``(\hat\sigma_R, \hat\sigma_t)`` via (25).
2. Weighted GN step on ``X`` using ``\ell_i^{\text{(A)}}`` as the per-pair
   contribution.
3. E-step (weights): refresh ``w_i^{(k+1)}`` via the mixture posterior.

This is properly EM-like: the M-step is closed-form, monotonically improves the
expected complete-data log-likelihood, and naturally absorbs the
posterior-IRLS outlier rejection (because ``w_i \to 0`` for outliers, they drop
out of both scale estimates).

*Strengths.* Unit-invariant by construction; self-consistent with the
framework's EM weights (no separate MAD estimate); robust by construction (outliers
drop out via ``w_i \to 0``); adapts to the actual residual distribution.
*Weaknesses.* Still treats all pairs as equally informative within each
component (constant ``W`` at each iteration); ``\hat\sigma_t \to 0`` in
pure-rotation calibration — clamp required.

### [7.3 Variant B: per-pair heteroscedastic Mahalanobis](@id handeye-variant-b)

Pair-specific noise covariance derived from explicit measurement-noise models
for ``A_{ij}`` (mocap, depending on inter-frame motion) and ``B_{ij}`` (PnP,
depending on pattern distance, foreshortening, image-area, etc.). With
``\boldsymbol{x}_i = (A_i, B_i)`` jointly the noisy measurement and
``\Sigma_{x_i} = \mathrm{block\text{-}diag}(\Sigma_{A_i}, \Sigma_{B_i}) \in \mathbb{R}^{12\times12}``:

```math
\boxed{\;\ell_i^{\text{(B)}}(X)
      \;=\; \boldsymbol{g}_i^{\!\top}\, \tilde\Sigma_{g,i}^{-1}\, \boldsymbol{g}_i,
      \qquad
      \tilde\Sigma_{g,i} \;=\; \partial_{\boldsymbol{x}}\boldsymbol{g}_i \, \Sigma_{x_i}\,
                                \partial_{\boldsymbol{x}}\boldsymbol{g}_i^{\!\top}\;}
\tag{26}
```

The framework supports this directly: return a 4-tuple
``(\boldsymbol{g}_i, \partial_{\boldsymbol{x}}\boldsymbol{g}_i, \partial_{\theta}\boldsymbol{g}_i, \Sigma_{x_i})``
from `constraint_with_jacobians`, and the default `residuals!` computes
``\sqrt{\boldsymbol{g}_i^{\!\top}\,\tilde\Sigma_{g,i}^{-1}\,\boldsymbol{g}_i}``.

*Strengths.* Textbook-correct heteroscedastic weighting; a far-away
barely-visible pattern frame is automatically down-weighted relative to a
close-up well-lit one. *Weaknesses.* Requires per-frame noise models for both
mocap and PnP; only pays off when uncertainty actually varies across frames.

### [7.4 Variant C: profile L per IRLS iteration](@id handeye-variant-c)

Single global ``L`` as in Variant 0, but re-estimated at every IRLS iteration
so that rotation and translation contribute equal Fisher information to the
Hessian:

```math
\boxed{\;\ell_i^{\text{(C)}}(X) \;=\; L_k^2\,\theta_i^2 \;+\; \|\boldsymbol{t}_{e,i}\|^2,
      \qquad
      L_k \;=\; \frac{\mathrm{median}_i\|\boldsymbol{t}_{e,i}^{(k)}\|}
                     {\mathrm{median}_i\,\theta_i^{(k)}}\;}
\tag{27}
```

where ``\theta_i^{(k)}`` and ``\boldsymbol{t}_{e,i}^{(k)}`` are evaluated at
the current iterate ``X^{(k)}``. At the optimum, ``L_k`` stabilizes to the ratio
of typical translation to typical rotation residual magnitudes.

*Strengths.* One degree of freedom (vs. two for Variant A); self-tuning;
cheaper than Variant A (one division per iteration vs. two MAD computations).
*Weaknesses.* Couples rotation and translation through a single scalar
(Variant A is strictly more flexible); marginally more sensitive to outliers in
early iterations than the MAD-based Variant A.

### 7.5 Summary table

| Variant | Per-pair weight ``W_i`` | DOF | Needs |
|:--|:--|:--|:--|
| 0 (implemented) | ``\mathrm{diag}(L^2 I_3,\, I_3)``, constant | 1 (``L``) | nothing (auto-default) |
| A | ``\mathrm{diag}(\hat\sigma_R^{-2}I_3,\, \hat\sigma_t^{-2}I_3)``, EM M-step | 2 | residuals + framework weights |
| B | ``\tilde\Sigma_{g,i}^{-1}``, per-pair | many | per-frame noise models |
| C | ``\mathrm{diag}(L_k^2 I_3,\, I_3)``, per-iteration | 1 (``L_k``, auto) | nothing |

The variants form a natural progression of expressive power. The
implementation (`HandEyeProblem`) uses Variant 0.

## [8. Building Blocks](@id handeye-building-blocks)

**From VisualGeometryCore.**

- `EuclideanMap` — composition `∘`, `inv`, application `p(x)`.
- `skew(v)` for ``[\cdot]_{\!\times}``; `translation_angle` for diagnostics;
  `log_so3` / `exp_so3` for the ``\mathrm{SO}(3)`` logarithm and exponential.
- `constraint_with_jacobians(pose::EuclideanMap{3}, X, ray)` returns the
  calibrated ``2\times6`` pose-perturbation Jacobian used in Stage B's
  ``\mathbf{H}_{ij}`` assembly (19).
- `HuberShape`, `CauchyShape`, … for the robust loss ``\rho``.

**From Rotations.jl.**

- ``\operatorname{Log}_{\mathrm{SO}(3)}(\mathbf{R})`` = `params(RotationVec(R))`
  — the axis–angle 3-vector; `rotation_angle` for diagnostics.
- ``\operatorname{Exp}_{\mathfrak{so}(3)}(\boldsymbol{\omega})`` =
  `RotMatrix(RotationVec(ω₁, ω₂, ω₃))` — retraction.

## [References](@id handeye-references)

- H. H. Chen. *A screw motion approach to uniqueness analysis of head-eye
  geometry.* CVPR, pp. 145–151, 1991.
- M. Shah. *Solving the robot-world/hand-eye calibration problem using the
  Kronecker product.* ASME Journal of Mechanisms and Robotics, 5(3):031007,
  2013.
- R. Y. Tsai and R. K. Lenz. *A new technique for fully autonomous and
  efficient 3D robotics hand/eye calibration.* IEEE Transactions on Robotics
  and Automation, 5(3):345–358, 1989.
