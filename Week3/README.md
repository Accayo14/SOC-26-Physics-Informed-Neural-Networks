# Week 3 — Canonical PDEs: Heat Equation & Wave Equation

> **Goal:** Scale up from ODEs to PDEs. Solve the heat and wave equations using the DeepXDE library, and understand how collocation point count affects accuracy.

---

## Topics Covered

- Heat equation (parabolic): `uₜ = α u_xx` — diffusion over time
- Wave equation (hyperbolic): `uₜₜ = c² u_xx` — wave propagation
- 2D collocation grid — sampling points in `(x, t)` space instead of `x` alone
- Sampling strategies — uniform random, Latin Hypercube Sampling (LHS), regular grid
- DeepXDE library — defining domains, BCs, ICs, and PDEs declaratively without writing a training loop
- Comparing PINN output to analytical ground truth — L² relative error and solution surface plots

---

## Resources

### 💻 Library & Docs

| Resource | Link |
|---|---|
| DeepXDE — forward problem demos (start here) | [deepxde.readthedocs.io](https://deepxde.readthedocs.io/en/stable/demos/pinn_forward.html) |
| DeepXDE GitHub | [github.com/lululxvi/deepxde](https://github.com/lululxvi/deepxde) |

Install DeepXDE:
```bash
pip install deepxde
```

### 📄 Paper

| Paper | Link |
|---|---|
| Lu, Meng, Mao, Karniadakis — "DeepXDE: A deep learning library for solving differential equations" (SIAM Rev. 63, 2021) | [arXiv:1907.04502](https://arxiv.org/abs/1907.04502) |

### 📺 Videos

| Resource | Link |
|---|---|
| Steven Brunton — Physics-Informed Machine Learning playlist (heat/diffusion lectures) | [YouTube](https://www.youtube.com/playlist?list=PLMrJAkhIeNNQromC4WswpU1krLOq5Ro6S) |
| Steven Brunton — "PINNs: Introduction and Applications" | [YouTube](https://www.youtube.com/watch?v=-zrY7P2dVC4) |

### 📖 Review

| Paper | What to read | Link |
|---|---|---|
| Cuomo et al. 2022 | Section 4 (applications) | [arXiv:2201.05624](https://arxiv.org/abs/2201.05624) |

---

## DeepXDE Quickstart

Here is the minimal structure for every DeepXDE PINN:

```python
import deepxde as dde
import numpy as np

# 1. Define the PDE
def heat_pde(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)     # ∂u/∂t
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)     # ∂²u/∂x²
    return dy_t - 0.4 * dy_xx                      # residual

# 2. Define the geometry and time domain
geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# 3. Define ICs and BCs
bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, lambda x: np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial)

# 4. Build the problem and model
data = dde.data.TimePDE(geomtime, heat_pde, [bc, ic], num_domain=2000, num_boundary=200, num_initial=200)
net  = dde.nn.FNN([2] + [64]*4 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

# 5. Train
model.compile("adam", lr=1e-3)
model.train(iterations=10000)
```

---

## Assignment

### Part A — Heat Equation

Solve the 1D heat equation:

```
uₜ = 0.4 u_xx,   x ∈ [0,1],  t ∈ [0,1]
u(0, t) = 0,     u(1, t) = 0         (Dirichlet BCs)
u(x, 0) = sin(πx)                    (Initial condition)
```

Analytical solution: `u(x, t) = exp(−0.4π²t) · sin(πx)`

**Deliverables:**
1. Plot the PINN solution surface `u(x, t)` as a heatmap
2. Plot the analytical solution surface for comparison
3. Plot the absolute error `|u_pinn − u_exact|`
4. Report the final L² relative error

---

### Part B — Collocation Sampling Experiment

Repeat Part A with these collocation counts: `[500, 1000, 2500, 5000, 10000]`

**Deliverable:** A single plot of L² relative error vs number of collocation points. What do you observe?

---

### Part C — Wave Equation

Solve the 1D wave equation:

```
uₜₜ = u_xx,   x ∈ [0,1],  t ∈ [0,1],  c = 1
u(x, 0) = sin(πx),    uₜ(x, 0) = 0
u(0, t) = 0,           u(1, t) = 0
```

**Deliverable:** Plot the solution surface `u(x, t)`. Describe qualitatively how it differs from the heat equation solution.

---

## Key Concept to Take Away

> DeepXDE separates what the problem *is* (geometry, PDE, BCs, ICs) from how to *solve* it (network, optimiser). You only write the physics. The library handles the rest.

Notice how the PINN solution is a **continuous function** over the domain — not a discrete grid. You can query it at any `(x, t)`.

---

## Questions to Think About

1. Why does increasing collocation points improve accuracy, but with diminishing returns?
2. The heat equation solution decays smoothly. The wave equation solution oscillates. How does this difference affect PINN training difficulty?
3. What would happen if you used `relu` instead of `tanh` as the activation function? Why?

---
