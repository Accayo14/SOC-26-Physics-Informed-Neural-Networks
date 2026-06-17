# Week 4 — Burgers' Equation, Boundary Conditions & Classical Solver Comparison

> **Goal:** Reproduce the benchmark PINN result from the founding paper, implement hard boundary conditions, and develop a clear-eyed view of where PINNs win and where they don't.

---

## Topics Covered

- Burgers' equation — why it is the PINN benchmark: nonlinear, develops shocks, has a known analytical solution
- Soft vs hard boundary condition enforcement
- Hard constraints via trial functions — constructing `u(x,t) = φ(x,t) · NN(x,t)` so BCs are satisfied exactly
- Distance function approach to hard BCs (Sukumar & Srivastava 2021)
- Reproducing Raissi et al. 2019 Section 3.1 **from scratch in PyTorch** (no DeepXDE)
- Honest PINNs vs FEM comparison — when each method wins

---

## Resources

### 📄 Papers

| Paper | What to read | Link |
|---|---|---|
| Grossmann, Komorowska, Latz, Schönlieb — "Can PINNs beat the Finite Element Method?" (2023) | Full paper | [arXiv:2302.04107](https://arxiv.org/abs/2302.04107) |
| Sukumar & Srivastava — "Exact imposition of BCs with distance functions in PINNs" (2021) | Sections 1–3 | [arXiv:2104.08426](https://arxiv.org/abs/2104.08426) |
| Raissi et al. 2019 | Section 3.1 (Burgers) | [arXiv:1711.10561](https://arxiv.org/abs/1711.10561) |

### 📖 Review

| Paper | What to read | Link |
|---|---|---|
| Karniadakis et al. — "Physics-informed machine learning" (Nature Reviews Physics 2021) | Sections 1–3 | [doi.org/10.1038/s42254-021-00314-5](https://doi.org/10.1038/s42254-021-00314-5) |

### 💻 Code References

| Resource | Link |
|---|---|
| DeepXDE — Burgers demo | [GitHub](https://github.com/lululxvi/deepxde/blob/master/docs/demos/pinn_forward/burgers.rst) |
| PyTorch reproduction of Raissi 2019 (reference, do not copy) | [GitHub](https://github.com/oscar-rincon/ReScience-PINNs) |

---

## Burgers' Equation Setup

The standard PINN benchmark from Raissi et al. 2019:

```
uₜ + u uₓ = (ν/π) uₓₓ,   x ∈ [−1, 1],  t ∈ [0, 1]

Boundary:  u(−1, t) = 0,  u(1, t) = 0
Initial:   u(x, 0) = −sin(πx)
ν = 0.01/π
```

---

## Soft vs Hard Boundary Conditions

### Soft (what you've been doing)
```python
# Add a boundary loss term to the total loss
u_bc = model(x_boundary)
L_bc = ((u_bc - 0)**2).mean()
L_total = L_pde + lambda_bc * L_bc
```

### Hard (trial function approach)
```python
# Multiply the raw network output by a function that is zero on the boundary
# For u(±1, t) = 0, we can use: φ(x, t) = (1 - x²)
def model_hard(x, t):
    raw = network(torch.cat([x, t], dim=1))
    return (1 - x**2) * raw    # automatically zero at x = ±1
```

Hard BCs remove the boundary loss term entirely. The network *cannot* violate the boundary condition.

---

## Assignment

> Submit a Jupyter notebook + short written answer

---

### Part A — From Scratch (no DeepXDE)

Implement a Burgers' PINN in **pure PyTorch** using the setup above.

Requirements:
- 10,000 interior collocation points (random)
- 200 boundary points
- 100 initial condition points
- 4-layer MLP, 100 neurons per layer, `tanh` activation
- Adam optimiser, 10,000 epochs

Produce:
1. Plot of `u(x, t)` predicted by the PINN
2. Plot of the absolute error `|u_pinn − u_exact|` (use the SciPy exact solution)
3. Training loss curve

---

### Part B — Hard BCs

Modify your Part A implementation to use hard boundary condition enforcement via the trial function `φ(x) = (1 − x²)`.

Produce:
1. Training loss curves for soft vs hard BCs on the same axes
2. Final L² error for both methods in a table
3. One sentence: which is better here, and why might it differ on other problems?

---

### Part C — Written (½ page)

Based on Grossmann et al. 2023 (the "can PINNs beat FEM?" paper), answer:

> In what scenarios would you choose a PINN over a classical finite element solver? Give at least two concrete, specific examples. Be honest about limitations.

---

## Key Concept to Take Away

> Burgers' equation is nonlinear and develops a near-discontinuous shock at `t = 1`. Classical solvers can resolve this perfectly with mesh refinement. PINNs struggle here — the L² error is acceptable but they are not competitive with FEM in this regime.

> **Where PINNs win:** inverse problems, parameter identification, high-dimensional PDEs, data assimilation from sparse sensors. Not forward-solving smooth PDEs that FEM already handles in milliseconds.

---

## Questions to Think About

1. Burgers' equation has a near-shock at `t = 1, x = 0`. Where does your PINN have the highest error?
2. Hard BCs remove one loss term. Does this make training easier or harder? Why?
3. The Grossmann paper says PINNs "rarely beat FEM for forward problems." Does that mean PINNs are useless? What does it mean for how you'd use them in practice?

---

*Next week: We go inside the training and understand why PINNs sometimes fail to converge at all — and three concrete fixes.*
