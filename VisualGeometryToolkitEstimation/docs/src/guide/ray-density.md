# Ray Density Correction for Posterior Weights

*In scale-free marginal RANSAC* (the Student-t + uniform marginal score,
RANSACScoringDoneRight's `StudentTUniformScoring`).

## [1. Setup](@id raydens-setup)

Consider a geometric model ``\theta`` with ``d_g``-dimensional algebraic
constraint ``\boldsymbol{g}(\boldsymbol{x}, \theta) = 0``, where
``\boldsymbol{x} \in \mathbb{R}^d`` is the measurement. With Gaussian noise
``\boldsymbol{x} \sim \mathcal{N}(\bar{\boldsymbol{x}}, \sigma^2 I_d)``, the
first-order approximation gives:

```math
\boldsymbol{g} \approx G_x (\boldsymbol{x} - \bar{\boldsymbol{x}}) \sim \mathcal{N}(0,\; \sigma^2 \tilde{\Sigma}_g),
\qquad \tilde{\Sigma}_g = G_x G_x^\top.
```

The whitened constraint is
``\boldsymbol{w} = \tilde{\Sigma}_g^{-1/2} \boldsymbol{g} \sim \mathcal{N}(0, \sigma^2 I_{d_g})``,
and the squared Mahalanobis distance is:

```math
q_i = \boldsymbol{g}_i^\top \tilde{\Sigma}_{g_i}^{-1} \boldsymbol{g}_i = \|\boldsymbol{w}_i\|^2,
\qquad \frac{q_i}{\sigma^2} \sim \chi^2(d_g).
```

The scalar Sampson distance is ``r_i = \sqrt{q_i} \geq 0``.

## [2. The Two Density Spaces](@id raydens-spaces)

### 2.1 Whitened-space (d_g-dimensional) density

The density of ``\boldsymbol{w} \in \mathbb{R}^{d_g}`` under the inlier model
is:

```math
p_{\mathrm{in}}^{(d_g)}(\boldsymbol{w} \mid \sigma)
= (2\pi\sigma^2)^{-d_g/2} \exp\!\bigl(-\|\boldsymbol{w}\|^2/(2\sigma^2)\bigr).
```

The outlier density in ``d_g``-dimensional whitened space:

```math
p_{\mathrm{out}}^{(d_g)} = (2a)^{-d_g}.
```

### 2.2 1D ray density (Shekhovtsov, Section 3.2)

Transforming to polar coordinates ``r = \|\boldsymbol{w}\|``, the volume
element is ``d\boldsymbol{w} = S_{d_g} \, r^{d_g - 1} \, dr`` where
``S_{d_g} = 2\pi^{d_g/2}/\Gamma(d_g/2)`` is the surface area of the unit
sphere. The *ray density* (marginal density of the scalar distance ``r``) given
known ``\sigma`` is the ``\chi_{d_g}`` distribution:

```math
p_{\mathrm{in}}^{(1)}(r \mid \sigma)
= \frac{2}{\Gamma(d_g/2)\,(2\sigma^2)^{d_g/2}} \; r^{d_g-1} \exp\!\bigl(-r^2/(2\sigma^2)\bigr).
```

**Predictive ray density after marginalizing ``\sigma^2``.** In the
scale-free marginal framework, ``\sigma^2`` is integrated out under the
Jeffreys prior using the sufficient statistics
``\hat\sigma^2 = \text{RSS}/\nu`` and ``\nu = (k-m)\,d_g`` from the sweep
partition. The predictive density for a new point is a multivariate
Student-``t`` with ``\nu`` degrees of freedom, which in 1D ray-density form
becomes:

```math
p_{\mathrm{in,pred}}^{(1)}(r \mid \nu, \hat\sigma^2)
= \frac{2}{\Gamma(d_g/2)} \cdot
  \frac{\Gamma\!\bigl(\tfrac{\nu+d_g}{2}\bigr)}
       {\Gamma\!\bigl(\tfrac{\nu}{2}\bigr) \,
        (\nu\,\hat\sigma^2)^{d_g/2}}
  \; r^{d_g-1}
  \Bigl(1 + \frac{r^2}{\nu\,\hat\sigma^2}\Bigr)^{\!-(\nu+d_g)/2}.
```

This reduces to the Gaussian ``\chi_{d_g}`` as ``\nu \to \infty`` and captures
the heavier tails from scale uncertainty at finite ``\nu``.

The outlier ray density (from a uniform distribution in ``\mathbb{R}^d``) is
constant:

```math
p_{\mathrm{out}}^{(1)}(r) = \frac{1}{2a}.
```

**Key point:** For ``d_g = 1``, the Jacobian factor is
``r^{d_g-1} = r^0 = 1``, so the two formulations coincide. For ``d_g = 2``, the
ray density has an extra factor of ``r``, which changes the relative weighting
of inliers at different distances.

## [3. Effect on Posterior Weights](@id raydens-weights)

The posterior inlier probability for point ``i`` is:

```math
w_i = \frac{\gamma \, p_{\mathrm{in}}(r_i)}
           {\gamma \, p_{\mathrm{in}}(r_i) + (1-\gamma) \, p_{\mathrm{out}}(r_i)}.
```

### 3.1 Whitened-space (d_g-dimensional) formulation

Both densities are evaluated in ``d_g``-dimensional space:

```math
w_i^{(d_g)} = \frac{\gamma \, p_{\mathrm{in}}^{(d_g)}(r_i)}
                   {\gamma \, p_{\mathrm{in}}^{(d_g)}(r_i)
                    + (1-\gamma) \, p_{\mathrm{out}}^{(d_g)}}
\tag{1}
```

where ``p_{\mathrm{in}}^{(d_g)}`` is evaluated at
``\|\boldsymbol{w}\| = r_i`` and ``p_{\mathrm{out}}^{(d_g)} = (2a)^{-d_g}``.

### 3.2 Correct 1D ray density

Both densities are 1D ray densities:

```math
w_i^{(1)} = \frac{\gamma \, p_{\mathrm{in}}^{(1)}(r_i)}
                 {\gamma \, p_{\mathrm{in}}^{(1)}(r_i)
                  + (1-\gamma) \, p_{\mathrm{out}}^{(1)}}
\tag{2}
```

where
``p_{\mathrm{in}}^{(1)}(r) = S_{d_g} \, r^{d_g-1} \, p_{\mathrm{in}}^{(d_g)}(r)``
and ``p_{\mathrm{out}}^{(1)} = (2a)^{-1}``.

### 3.3 The discrepancy

The ratio of inlier-to-outlier likelihood in the two formulations:

```math
\begin{aligned}
\text{$d_g$-dimensional:} \quad
\Lambda_i^{(d_g)} &= \frac{p_{\mathrm{in}}^{(d_g)}(r_i)}{p_{\mathrm{out}}^{(d_g)}}
= (2a)^{d_g} \, (2\pi\sigma^2)^{-d_g/2} \exp(-r_i^2/(2\sigma^2)), \\[6pt]
\text{Correct:} \quad
\Lambda_i^{(1)} &= \frac{p_{\mathrm{in}}^{(1)}(r_i)}{p_{\mathrm{out}}^{(1)}}
= 2a \cdot S_{d_g} \, r_i^{d_g-1} \, (2\pi\sigma^2)^{-d_g/2} \exp(-r_i^2/(2\sigma^2)).
\end{aligned}
```

Their ratio is:

```math
\frac{\Lambda_i^{(1)}}{\Lambda_i^{(d_g)}}
= \frac{S_{d_g} \, r_i^{d_g-1}}{(2a)^{d_g - 1}}.
```

For ``d_g = 1``: the ratio is ``S_1 \cdot r^0 / 1 = 2``, a constant that
cancels in the posterior sigmoid. **No effect.**

For ``d_g = 2``: the ratio is ``S_2 \cdot r / (2a) = 2\pi r / (2a)``, which is
**proportional to ``r_i``**. This means:

- Points very close to the model (``r_i \ll 1``): the ``d_g``-dimensional
  formulation *overestimates* their posterior ``w_i`` relative to the correct
  ray density formulation.
- Points far from the model (``r_i \gg 1``): the ``d_g``-dimensional
  formulation *underestimates* their posterior.

## [4. Impact](@id raydens-impact)

### 4.1 Model selection (scoring evaluation)

For the sweep-based score ``S(\theta) = \max_k S(k)``, the ray density
correction enters as an additive term ``(d_g - 1) \sum_{i \in I} \log r_i`` in
the inlier log-likelihood after marginalizing ``\sigma^2``. This term
penalizes models whose inliers have very small residuals.

Empirically, this correction *hurts* model selection: the corrected
"Ray-Density" score ranks models worse than the uncorrected sweep. The
``d_g``-dimensional formulation (without the Jacobian) is more discriminative
for ranking models because it directly rewards small residuals.

### 4.2 Local optimization (RANSAC LO)

When the posterior weights ``w_i`` are used as soft IRLS weights for the
weighted DLT refit, the discrepancy means:

- For ``d_g = 1`` (F-matrix): no effect.
- For ``d_g = 2`` (homography): points near the model get too much weight
  (missing ``r_i`` factor), while moderate-distance inliers get too little.
  This biases the refit toward going exactly through nearby points rather
  than balancing all inliers.

The practical impact depends on the inlier noise distribution. With typical
inlier distances of 0.5–2 px, the correction factor ``r_i`` is of order 1 and
the effect is moderate.

### 4.3 Scale estimation

The weighted scale estimate
``\hat\sigma^2 = \sum w_i q_i / (d_g \sum w_i)`` is also affected:
overweighting near-zero residuals pulls ``\hat\sigma^2`` down. The Newton
update for ``\nu`` (degrees of freedom) in the EM M-step similarly uses the
biased weights.

## [5. Summary](@id raydens-summary)

In the ``d_g``-dimensional formulation:

| | ``d_g = 1`` (F-matrix) | ``d_g = 2`` (homography) |
|:--|:--:|:--:|
| Sweep score (model selection) | correct | correct\* |
| Posterior weights (LO) | correct | **biased** (missing ``r_i`` factor) |
| Posterior mixture score | correct | **wrong** (space mismatch) |

\*The sweep score is correct in both the ``d_g``-dimensional and 1D ray
density formulations — they differ by a model-dependent additive term, but
cross-validation over ``a`` compensates. Empirically, the ``d_g``-dimensional
version ranks models better.

**Conclusions:**

1. The sweep score works well for model selection in the
   ``d_g``-dimensional formulation.
2. Posterior weights for ``d_g \geq 2`` are corrected by using the 1D
   *predictive* Student-``t`` ray density ``p_{\mathrm{in,pred}}^{(1)}(r)``
   (which includes the Jacobian ``r^{d_g-1}``) and the 1D outlier density
   ``p_{\mathrm{out}}^{(1)} = (2a)^{-1}``.
3. The predictive ray density mixture gives an alternative soft scoring
   function:

   ```math
   S_{\mathrm{soft}}(\theta) = \sum_{i=1}^n \log\!\Bigl(
     \gamma \, p_{\mathrm{in,pred}}^{(1)}(r_i \mid \nu, \hat\sigma^2)
     + (1-\gamma) \, p_{\mathrm{out}}^{(1)}(r_i)
   \Bigr),
   ```

   where ``\gamma = k^*/n``, ``\hat\sigma^2 = \text{RSS}/\nu``, and
   ``\nu = (k^* - m)\,d_g`` are all derived from the sweep partition. This
   combines the sweep's scale estimation with GaU-style soft weighting.

## [References](@id raydens-references)

- A. Shekhovtsov. *RANSAC Scoring Functions: Analysis and Reality Check.*
  Preprint, 2025.
