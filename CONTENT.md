# Project Content Overview

This repository is a first Python implementation for the semester project described in `project_description.pdf`: characterize a relativistic electron bunch drifting into a longitudinal electric field, build analytic and semi-analytic reference models, and compare them against Python and later OPAL/OPALX simulations.

The implemented code currently covers:

- a 1D macroparticle tracker with a grid Poisson solve,
- a 3D macroparticle tracker using `pyHockneySolver`,
- several analytic / semi-analytic moment models,
- a perturbative Lienard-Wiechert ansatz,
- saved simulation data and selected 3D plots.

## Common Physical Setup

The notebooks use electrons with

```text
m_e = 0.511 MeV/c^2
q_e = -1 e
E_z = -1 MV/m
initial kinetic energy = 1e-2 MeV
total bunch charge = -1 nC
```

The longitudinal geometry is

```text
prior region -> drift -> constant Ez cavity -> posterior region
Z_1 = start of Ez region
Z_2 = end of Ez region
```

The relativistic helper equations used throughout are

$$
\gamma(p) = \sqrt{1 + \frac{\lVert p\rVert^2}{m_e^2}},
\qquad
v(p) = \frac{p}{\gamma(p)m_e}c .
$$

Momenta are stored in `MeV/c`. Plots often show `beta gamma = p / m_e`.

## Implemented Python Solvers

### 1. 1D Macroparticle Tracker

File: `python/1D_solve.ipynb`

Current capabilities:

- samples either a Gaussian or a uniform initial longitudinal distribution,
- tracks `N_PARTICLES = 1000` macroparticles,
- supports either a globally constant external field or a field only in `[Z_1, Z_2]`,
- computes a 1D self-field from a grid charge density,
- advances particles with a staggered momentum / velocity-Verlet style update,
- records mean and RMS of `z` and `p_z`,
- saves results to `python/data/1d_data.csv` or `python/data/1d_data_simple.csv`.

The charge scatter is linear cloud-in-cell:

$$
\rho_j =
\frac{1}{\Delta z}
\sum_i q_{\mathrm{macro}} W_j(z_i),
$$

where each particle contributes to its two nearest grid points. The code uses a unit transverse area, so this is effectively a line-charge model interpreted as charge density per `1 m^2`.

The internal potential is obtained by a sine-transform Poisson solve with zero boundary conditions:

$$
\partial_z^2 \Phi(z) = -\frac{\rho(z)}{\epsilon_0},
\qquad
\widehat{\Phi}_k = \frac{\widehat{\rho}_k}{\epsilon_0 k^2},
\qquad
E_{\mathrm{int}}(z) = -\partial_z \Phi(z).
$$

The total electric field is

$$
E_{\mathrm{tot}}(z) =
E_{\mathrm{int}}(z) + E_{\mathrm{ext}}(z),
$$

with either

$$
E_{\mathrm{ext}}(z)=E_z
$$

or

$$
E_{\mathrm{ext}}(z)=
\begin{cases}
E_z, & Z_1 \le z \le Z_2,\\
0, & \mathrm{otherwise}.
\end{cases}
$$

The particle pusher is

$$
p_{i,n+1/2} = p_{i,n-1/2} + q E_{\mathrm{tot}}(z_{i,n}) c \Delta t,
$$

$$
z_{i,n+1} = z_{i,n} + v(p_{i,n+1/2})\Delta t.
$$

Feasibility:

- Good as a fast longitudinal reference tracker and for debugging analytic models.
- The 1D self-field is only a rough approximation because the transverse bunch size enters only implicitly through the assumed unit area.
- Boundary handling is still fragile: particles outside the grid trigger an assertion.
- Next step: turn this notebook into a reproducible script/module, add unit tests for charge conservation and no-self-field acceleration, and compare against the analytic no-self-field solution before trusting self-field results.

### 2. 3D Macroparticle Tracker

File: `python/3D_solve.ipynb`

Current capabilities:

- samples Gaussian or uniform ellipsoidal 3D particle distributions,
- tracks `N_PARTICLES = 32^3`,
- solves the bunch self-field in an approximate bunch rest frame using `pyHockneySolver.solve_open_poisson_hockney`,
- transforms the self-field back to the lab frame,
- adds the external longitudinal electric field,
- advances particles with a Boris / velocity-Verlet pusher,
- removes particles leaving the model box,
- records mean and RMS of all position and momentum components,
- saves results to `python/data/3d_data.csv` or `python/data/3d_data_simple.csv`.

The rest-frame transform currently uses the mean longitudinal motion of the bunch:

$$
\gamma_z = \sqrt{1 + \left(\frac{\langle p_z\rangle}{m_e}\right)^2},
\qquad
\beta_z = \frac{\langle p_z\rangle}{\gamma_z m_e},
$$

$$
z'_i = \gamma_z(z_i - \langle z\rangle),
\qquad
x'_i=x_i,\quad y'_i=y_i.
$$

The open-boundary Poisson solve returns `E'_int`. The lab-frame field transform is

$$
E_x = \gamma_z E'_x,\quad
E_y = \gamma_z E'_y,\quad
E_z = E'_z,
$$

$$
B_x = -\frac{\beta_z}{c}E_y,\quad
B_y = \frac{\beta_z}{c}E_x,\quad
B_z = 0.
$$

The external field is then added:

$$
\mathbf E_{\mathrm{tot},i}
= \mathbf E_{\mathrm{int},i}
+ (0,0,E_z).
$$

The pusher is the standard electric half-kick, magnetic Boris rotation, electric half-kick:

$$
p^- = p^{n-1/2} + \frac{q\mathbf E\ c\Delta t}{2},
$$

$$
t = \frac{q c^2\Delta t}{2m_e\gamma} \mathbf B,
\qquad
s = \frac{2t}{1+\lVert t\rVert^2},
$$

$$
p' = p^- + p^- \times t,
\qquad
p^+ = p^- + p' \times s,
$$

$$
p^{n+1/2}=p^+ + \frac{q\mathbf E\ c\Delta t}{2},
\qquad
r^{n+1}=r^n+v(p^{n+1/2})\Delta t.
$$

Feasibility:

- This is the closest current Python reference for comparison to OPAL/OPALX space-charge behavior.
- The present rest-frame treatment is a one-frame approximation based on the bunch mean, not a velocity-binned solver.
- The Lorentz transform uses only longitudinal mean motion and ignores simultaneity details beyond the applied `z` contraction.
- Next step: validate the no-self-field limit, then compare one-frame versus manually velocity-binned sub-bunch solves. Add conservation diagnostics and runtime scaling measurements.

## Analytic And Semi-Analytic Models

File: `python/analytic_solve.ipynb`

This notebook loads the saved 1D/3D CSV data and implements several analytic approximations.

### 3. Single-Particle No-Self-Field Solution

Current capability:

- computes `p(t)` and `p(z)` for one reference particle drifting into the external field.

For a particle entering the field at `Z_1`, the implemented time-domain form is

$$
p(t)=
\begin{cases}
p_0, & t<t_1,\\
p_0 + qE_z(t-t_1), & t\ge t_1,
\end{cases}
\qquad
t_1=\frac{Z_1}{v(p_0)}.
$$

The position-domain relation is based on energy gain:

$$
\gamma(z)m_ec^2 =
\gamma_0m_ec^2 + qE_z(z-Z_1),
$$

$$
p(z)=
\frac{1}{c}
\sqrt{
\left(\gamma_0m_ec^2+qE_z(z-Z_1)\right)^2
-m_e^2c^4
}.
$$

Feasibility:

- Exact for one particle without self-fields in a piecewise constant longitudinal field.
- Useful as the first validation target for both trackers.
- Next step: clean up unit conventions between `p(t)` and `p(z)` and include the initial position offset explicitly.

### 4. 1D Lagrange / Distribution Method Without Self-Fields

Current capability:

- models a Gaussian initial bunch by integrating over initial momentum `p_0`,
- uses the Gaussian CDF to determine which longitudinal slice has entered the electric field,
- predicts mean momentum and RMS momentum versus time.

Definitions:

$$
g_0(p_0)=
\frac{1}{\sqrt{2\pi}\sigma_{p0}}
\exp\left[-\frac{(p_0-\mu_{p0})^2}{2\sigma_{p0}^2}\right],
$$

$$
z_{\mathrm{crit}}(t,p_0)=Z_1-v(p_0)t,
\qquad
\zeta=\frac{z_{\mathrm{crit}}-\mu_{z0}}{\sqrt{2}\sigma_{z0}}.
$$

The implemented mean momentum is

$$
\mu_p(t)=\mu_{p0}
+\int g_0(p_0)\frac{qE_z}{v_0}
\left[
(\mu_{z0}-z_{\mathrm{crit}})\frac{1-\operatorname{erf}(\zeta)}{2}
+\frac{\sigma_{z0}}{\sqrt{2\pi}}e^{-\zeta^2}
\right]dp_0.
$$

The second moment is computed with the corresponding closed Gaussian truncation formula and

$$
\sigma_p(t)=\sqrt{\langle p^2\rangle(t)-\mu_p(t)^2}.
$$

Feasibility:

- Feasible for no-self-field longitudinal benchmarking.
- It is an approximation because particles are treated ballistically while deciding field entry.
- Next step: compare directly against `1D_solve.ipynb` with self-fields off and quantify the error near the cavity entrance.

### 5. 1D Moment Model With Self-Fields And Cavity Entrance

Current capability:

- evolves the Gaussian bunch moments
  `(mu_z, mu_p, sigma_z^2, sigma_p^2, sigma_zp)`,
- includes the external field only for the fraction of the Gaussian beyond `Z_1`,
- adds a simple 1D Gaussian self-field closure.

The state equations are

$$
\dot{\mu}_z = v(\mu_p),
$$

$$
\dot{\mu}_p =
qE_zc\frac{1-\operatorname{erf}(\zeta)}{2},
\qquad
\zeta=\frac{Z_1-\mu_z}{\sqrt{2}\sigma_z},
$$

$$
\dot{\sigma}_z^2 =
\frac{2c}{\gamma^3m_e}\sigma_{zp},
$$

$$
\dot{\sigma}_p^2 =
2qE_zc
\frac{\sigma_{zp}}{\sqrt{2\pi}\sigma_z}
e^{-(Z_1-\mu_z)^2/(2\sigma_z^2)}
+2q\kappa_Gc\sigma_{zp},
$$

$$
\dot{\sigma}_{zp} =
\frac{c}{\gamma^3m_e}\sigma_p^2
+qE_zc\frac{\sigma_z}{\sqrt{2\pi}}
e^{-(Z_1-\mu_z)^2/(2\sigma_z^2)}
+q\kappa_Gc\sigma_z^2,
$$

with

$$
\kappa_G =
\frac{Q_{\mathrm{tot}}}
{2\sqrt{\pi}\epsilon_0\sigma_z}.
$$

Feasibility:

- Useful for a compact semi-analytic comparison, especially for bunch length and energy spread.
- The self-field closure is very approximate and still effectively 1D.
- Next step: calibrate or replace the 1D self-field closure using the 3D tracker or an ellipsoidal model.

### 6. Simple 1D Uniform Moment Model

Current capability:

- evolves the same 1D moment state for a uniform distribution in a globally constant external field,
- removes the cavity-entry error-function terms.

The equations reduce to

$$
\dot{\mu}_z=v(\mu_p),
\qquad
\dot{\mu}_p=qE_zc,
$$

$$
\dot{\sigma}_z^2 =
\frac{2c}{\gamma^3m_e}\sigma_{zp},
\qquad
\dot{\sigma}_p^2 =
2q\kappa_Uc\sigma_{zp},
$$

$$
\dot{\sigma}_{zp} =
\frac{c}{\gamma^3m_e}\sigma_p^2
+q\kappa_Uc\sigma_z^2,
$$

with

$$
\kappa_U =
\frac{Q_{\mathrm{tot}}}{\sqrt{12}\epsilon_0\sigma_z}.
$$

Feasibility:

- Good toy model for debugging moment evolution without field-entry discontinuities.
- Not sufficient for the real project geometry unless the global constant-field assumption is intentionally used.
- Next step: use it as a sanity check for `1d_data_simple.csv`, then reintroduce the finite cavity.

### 7. Simple 3D Uniform Ellipsoid Moment Model

Current capability:

- evolves longitudinal and one transverse moment block:
  `(mu_z, mu_p, sigma_z^2, sigma_pz^2, sigma_zpz, sigma_x^2, sigma_px^2, sigma_xpx)`,
- assumes `x` and `y` symmetry,
- uses uniform ellipsoid geometry factors.

The ellipsoid radii are

$$
R_x=R_y=\sqrt{5}\sigma_x,
\qquad
R_z=\sqrt{5}\sigma_z.
$$

With

$$
\xi = \gamma\frac{R_z}{R_x},
$$

the code computes geometry factors `g_x=g_y=(1-g_z)/2`, with separate prolate/oblate formulas for `g_z`. The self-field coefficients are

$$
\kappa_z =
\frac{3Q_{\mathrm{tot}}g_z}
{4\pi\epsilon_0R_xR_yR_z},
$$

$$
\kappa_x =
\frac{3Q_{\mathrm{tot}}g_x}
{4\pi\gamma^2\epsilon_0R_xR_yR_z}.
$$

The longitudinal moment block is

$$
\dot{\mu}_z=v(\mu_p),
\qquad
\dot{\mu}_p=qE_zc,
$$

$$
\dot{\sigma}_z^2 =
\frac{2c}{\gamma^3m_e}\sigma_{zp_z},
\qquad
\dot{\sigma}_{p_z}^2 =
2q\kappa_zc\sigma_{zp_z},
$$

$$
\dot{\sigma}_{zp_z} =
\frac{c}{\gamma^3m_e}\sigma_{p_z}^2
+q\kappa_zc\sigma_z^2.
$$

The transverse block is

$$
\dot{\sigma}_x^2 =
\frac{2c}{\gamma m_e}\sigma_{xp_x},
\qquad
\dot{\sigma}_{p_x}^2 =
2q\kappa_xc\sigma_{xp_x},
$$

$$
\dot{\sigma}_{xp_x} =
\frac{c}{\gamma m_e}\sigma_{p_x}^2
+q\kappa_xc\sigma_x^2.
$$

Feasibility:

- Promising as a semi-analytic bridge between the 1D moment model and the full 3D tracker.
- The uniform ellipsoid assumption is restrictive but physically interpretable.
- Next step: compare against `3d_data_simple.csv`, then add a finite-cavity version and possibly non-symmetric transverse moments.

### 8. 3D Lienard-Wiechert Perturbation Ansatz

Current capability:

- computes exact zeroth-order trajectories in the external field,
- solves retarded-time equations for source particles,
- evaluates Lienard-Wiechert fields,
- integrates first-order position and momentum perturbations due to inter-particle fields.

The zeroth-order trajectory is

$$
p_z^{(0)}(t)=p_{z0}+qE_zct,
$$

$$
\gamma^{(0)}(t)=
\sqrt{1+\left(\frac{p_z^{(0)}(t)}{m_e}\right)^2},
$$

$$
z^{(0)}(t)=z_0+
\frac{m_e}{qE_z}
\left[\gamma^{(0)}(t)-\gamma^{(0)}(0)\right].
$$

The retarded time solves

$$
c(t_{\mathrm{obs}}-t_r)
=
\lVert r_{\mathrm{obs}}-r_{\mathrm{src}}^{(0)}(t_r)\rVert .
$$

The Lienard-Wiechert field is implemented in SI units:

$$
\mathbf E =
\frac{q}{4\pi\epsilon_0}
\left[
\frac{\mathbf n-\boldsymbol\beta}
{\gamma^2(1-\mathbf n\cdot\boldsymbol\beta)^3R^2}
+
\frac{\mathbf n\times[(\mathbf n-\boldsymbol\beta)\times\dot{\boldsymbol\beta}]}
{c(1-\mathbf n\cdot\boldsymbol\beta)^3R}
\right],
$$

$$
\mathbf B = \frac{\mathbf n\times\mathbf E}{c}.
$$

The first-order perturbation integrates

$$
\frac{d\mathbf p^{(1)}}{dt}
=
q\left(\mathbf E_{\mathrm{LW}}
+\mathbf v^{(0)}\times\mathbf B_{\mathrm{LW}}\right),
$$

$$
\dot{x}^{(1)} =
\frac{p_x^{(1)}}{\gamma m_e}c,\quad
\dot{y}^{(1)} =
\frac{p_y^{(1)}}{\gamma m_e}c,\quad
\dot{z}^{(1)} =
\frac{p_z^{(1)}}{\gamma^3m_e}c.
$$

Feasibility:

- Physically detailed and useful as a small-`N` reference for retardation and magnetic effects.
- Computationally expensive: each target/source interaction needs retarded-time root solving and time integration.
- Not feasible as the main production tracker for large bunches.
- Next step: restrict it to small benchmark cases and compare against the 3D Hockney tracker in regimes where retardation should be negligible.

## Current Data Products

Saved CSV outputs:

| File | Shape | Meaning |
|---|---:|---|
| `python/data/1d_data.csv` | `9054 x 5` | 1D finite-cavity / non-simple run: `t, mean(z), rms(z), mean(pz), rms(pz)` |
| `python/data/1d_data_simple.csv` | `7314 x 5` | 1D uniform + constant-field run |
| `python/data/3d_data.csv` | `906 x 13` | 3D non-simple run: time, 3 position means, 3 position RMS values, 3 momentum means, 3 momentum RMS values |
| `python/data/3d_data_simple.csv` | `734 x 13` | 3D uniform + constant-field run |

The saved 3D runs reach approximately `mean(z) = 4 m`, i.e. the end of the electric-field region. Example final values:

```text
1d_data.csv:        t = 1.8104e-08 s, mean(z) = 4.00008 m, rms(z) = 0.489942 m, mean(pz) = 4.04256 MeV/c
1d_data_simple.csv: t = 1.4624e-08 s, mean(z) = 4.00007 m, rms(z) = 0.101025 m, mean(pz) = 4.48572 MeV/c
3d_data.csv:        t = 1.8080e-08 s, mean(z) = 4.00046 m, rms(z) = 0.521389 m, mean(pz) = 4.02763 MeV/c
3d_data_simple.csv: t = 1.4640e-08 s, mean(z) = 4.00460 m, rms(z) = 0.100500 m, mean(pz) = 4.49356 MeV/c
```

Additional saved plots:

- `python/plots/3D-rel/r-vs-t.png`
- `python/plots/3D-rel/p-vs-t.png`
- `python/plots/3D-rel/p-vs-z.png`
- `python/plots/3D-nonrel/r-vs-t.png`
- `python/plots/3D-nonrel/p-vs-t.png`
- `python/plots/3D-nonrel/p-vs-z.png`

## Recommended Next Steps

1. Reproducibility: move shared constants, sampling, field solves, and pushers out of notebooks into importable Python modules.
2. Unit consistency: document the hybrid `MeV/c`, `MV/m`, and SI time conventions and add assertions/tests around every conversion.
3. Baseline validation: run all solvers with self-fields disabled and compare against the single-particle analytic solution.
4. Moment-model validation: compare the simple 1D and simple 3D moment models against `1d_data_simple.csv` and `3d_data_simple.csv`.
5. Physical validation: compare the 3D Hockney one-frame solve against OPAL for the same bunch and cavity setup.
6. Velocity binning: split the bunch into longitudinal momentum or `beta_z` bins, solve each bin in its own approximate rest frame, and compare against the current one-frame 3D implementation.
7. Diagnostics: add charge conservation, lost-particle counts, energy gain, emittance, and bunch-length plots as standard outputs.
8. HPC / scaling preparation: once the physics behavior is validated, measure runtime versus `N_PARTICLES`, grid shape, and number of velocity bins.
