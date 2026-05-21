# Week 1 - PDEs, Neural Network Refresher & Automatic Differentiation

> **Goal:** Understand what a PDE is, refresh your neural network intuition, and learn how to differentiate a neural network with respect to its *inputs* - the one idea that makes PINNs possible.

---

## Topics Covered

- PDE classification - elliptic, parabolic, hyperbolic; reading standard notation (∂u/∂t, ∂²u/∂x²)
- Neural network refresher - MLP architecture, forward pass, loss function, gradient descent
- Automatic differentiation - forward mode vs reverse mode (backpropagation)
- **Key insight:** using `torch.autograd.grad` to differentiate the network output `u(x,t)` with respect to inputs `x` and `t`, not the weights
- PyTorch mechanics - `requires_grad`, `.backward()`, `.grad`, `torch.autograd.grad` for higher-order derivatives

---

## Resources

### Videos

| Resource | Link |
|---|---|---|
| Steven Brunton - Intro to PDEs playlist (Videos 1–3) | [YouTube](https://www.youtube.com/playlist?list=PLMrJAkhIeNNQromC4WswpU1krLOq5Ro6S) |
| Chris Rackauckas - Intro to Scientific ML 2: PINNs (MIT 18.337) | [YouTube](https://www.youtube.com/watch?v=hKHl68Fdpq4) |
| Karpathy - Neural Networks: Zero to Hero, Lecture 1 (micrograd) | [YouTube](https://www.youtube.com/watch?v=VMj-3S1tku0) |

### Blogs / Reading

| Resource | Link |
|---|---|
| Ben Moseley - "So, what is a physics-informed neural network?" | [benmoseley.blog](https://benmoseley.blog/my-research/so-what-is-a-physics-informed-neural-network/) |
| Christopher Olah - "Calculus on Computational Graphs: Backpropagation" | [colah.github.io](https://colah.github.io/posts/2015-08-Backprop/) |

### Coding

| Resource | Link |
|---|---|
| PyTorch Autograd tutorial | [pytorch.org](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) |
| JAX Autodiff Cookbook (read, do not code) | [jax.readthedocs.io](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html) |

### Paper - skim Sections 1-3

| Paper | Link |
|---|---|
| Baydin, Pearlmutter, Radul, Siskind - "Automatic Differentiation in Machine Learning: a Survey" (2018) | [arXiv:1502.05767](https://arxiv.org/abs/1502.05767) |

---