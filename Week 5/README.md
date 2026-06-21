# Week 5 — Training Pathologies: Why PINNs Fail & How to Fix Them

> **Goal:** Understand the three main reasons PINNs fail to train, and implement two fixes — gradient-norm loss balancing and Fourier feature input embedding — on your Burgers PINN from Week 4.

---

## Topics Covered

- **Spectral bias** — MLPs preferentially learn low-frequency components first; high-frequency solutions converge slowly or not at all
- **Loss imbalance** — the PDE residual, IC loss, and BC loss have competing gradient magnitudes, causing instability
- **Causal violation** — for time-dependent PDEs, the network is asked to satisfy `t = 1` before it has learned `t = 0`
- **Fix 1:** Gradient-norm loss balancing (Wang, Teng, Perdikaris 2021)
- **Fix 2:** Fourier feature input embedding — maps `(x, t)` to `[sin(Bv), cos(Bv)]` to counter spectral bias
- **Fix 3 (read, no implementation):** Causal training — weighting collocation points by their causal order

---

## Resources

### 📄 Papers — read in this order

| Paper | What to read | Link |
|---|---|---|
| Rahaman et al. — "On the Spectral Bias of Neural Networks" (ICML 2019) | Sections 1–3 | [arXiv:1806.08734](https://arxiv.org/abs/1806.08734) |
| Wang, Teng, Perdikaris — "Understanding and mitigating gradient pathologies in PINNs" (SIAM 2021) | Full | [arXiv:2001.04536](https://arxiv.org/abs/2001.04536) |
| Wang, Sankaran, Perdikaris — "Respecting causality is all you need for training PINNs" (2022) | Sections 1–3 | [arXiv:2203.07404](https://arxiv.org/abs/2203.07404) |
| Wang, Sankaran, Wang, Perdikaris — "An Expert's Guide to Training PINNs" (2023) | Skim for techniques | [arXiv:2308.08468](https://arxiv.org/abs/2308.08468) |

### 📝 Blog

| Resource | Link |
|---|---|
| Rajat Vadiraj Dwaraknath — "Understanding the Neural Tangent Kernel" | [rajatvd.github.io/NTK](https://rajatvd.github.io/NTK/) |

### 💻 Code — read, do not reimplement

| Resource | What to look at | Link |
|---|---|---|
| PredictiveIntelligenceLab/jaxpi | `models.py` — see how Fourier features, NTK weighting, and causal weights are implemented | [GitHub](https://github.com/PredictiveIntelligenceLab/jaxpi) |

---

## The Three Failure Modes — Summary

### 1. Spectral Bias
Neural networks learn low-frequency functions first. For high-frequency solutions (like Burgers' near the shock), this means the PDE residual stagnates.

**Fix:** Fourier feature embedding
```python
# Instead of feeding (x, t) directly, embed them
B = torch.randn(d_model // 2, 2) * sigma   # random Fourier matrix
def fourier_embed(xt):
    proj = xt @ B.T                         # (N, d_model//2)
    return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
```

### 2. Loss Imbalance
The PDE loss, BC loss, and IC loss have different gradient magnitudes. One dominates and the others are ignored.

**Fix:** Gradient-norm loss balancing (Wang et al. 2021)
```python
# Compute gradient norms for each loss component
grad_pde = gradient_norm(L_pde, model.parameters())
grad_bc  = gradient_norm(L_bc,  model.parameters())
grad_ic  = gradient_norm(L_ic,  model.parameters())

# Rescale so all components contribute equally
mean_grad = (grad_pde + grad_bc + grad_ic) / 3
lambda_bc = mean_grad / grad_bc
lambda_ic = mean_grad / grad_ic

L_total = L_pde + lambda_bc * L_bc + lambda_ic * L_ic
```

### 3. Causal Violation
The network tries to satisfy `t = 0.9` before `t = 0.1` is converged. This is physically nonsensical.

**Fix:** Causal weighting — weight collocation points by how well earlier times are already satisfied.

---

## Assignment

### Setup

Take your **from-scratch Burgers' PINN from Week 4** (Part A). Do not use DeepXDE.

---

### Task 1 — Implement Fourier Feature Embedding

Modify the network input:
- Instead of feeding `[x, t]` directly (2 features), embed it as `[sin(Bv), cos(Bv)]` where `B` is a (d, 2) random matrix sampled from `N(0, σ²)`.
- Try `σ = 1` and `σ = 10`. Use `d = 128` (giving a 256-dimensional embedding).

---

### Task 2 — Implement Gradient-Norm Loss Balancing

At each training step, compute the mean gradient norm of each loss component and rescale `λ_bc` and `λ_ic` accordingly. Run for 10,000 epochs.

---

### Task 3 — Comparative Experiment

Train three variants on the **exact same Burgers' setup** from Week 4:
1. Vanilla PINN (your Week 4 baseline)
2. Vanilla PINN + Fourier features (σ = 10)
3. Vanilla PINN + gradient-norm loss balancing

For each variant produce:
- Training loss curve
- Final L² relative error

Present results in a **single comparison table** and a **single figure** with all three loss curves.

---

### Task 4 — Analysis (½ page)

Answer:
1. Which fix helped most? Was it what you expected?
2. Look at the gradient norms of `L_pde`, `L_bc`, `L_ic` in your vanilla PINN. Which is largest? What does this tell you about why the PINN trains poorly by default?

---


## Key Concept to Take Away

> The PINN loss is a multi-task learning problem. Like any multi-task problem, the tasks have competing gradients and the naive sum of losses is almost never the right approach. Loss balancing is not a hack — it is essential.

> The "Expert's Guide" (Wang et al. 2023) synthesises all known best practices. Think of it as your PINN training checklist.

---

## Questions to Think About

1. Why does `σ` in the Fourier embedding matter? What happens if it's too small? Too large?
2. The gradient-norm balancing updates `λ` at every step. Does this make training more stable or less stable? What if you updated `λ` only every 100 steps?
3. Causal training was not implemented this week. For which PDEs would it matter most — and why?

---

*Next week: We flip the problem around. Instead of solving for `u` given the equation, we recover unknown parameters in the equation from sparse data.*
