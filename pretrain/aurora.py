"""Aurora optimizer update rule (Tilde Research, 2026).

Aurora replaces Muon's polar(G) with a "leverage-uniform" polar that
iteratively projects onto the intersection of the Stiefel (orthogonal)
and row-oblique (uniform row norms) manifolds. For tall non-square
matrices (e.g. MLP up/gate projections in transformers), this prevents
the "neuron death" pathology where weakly-initialized rows accumulate
small momentum and stay underutilized.

For square matrices Aurora reduces to standard Muon (the row-uniform
constraint has no effect once column-orthogonality is satisfied).

Source: https://github.com/tilde-research/aurora-release
Blog:   https://blog.tilderesearch.com/blog/aurora

Drop-in replacement for muon_update() in muon.py: same call signature,
returns the update tensor (no weight decay or parameter step here —
those are handled by the outer optimizer.step()).
"""
import torch


@torch.no_grad()
def _polar(G: torch.Tensor, steps: int = 12) -> torch.Tensor:
    """Polar factor via simple-quintic Newton-Schulz.

    Coefficients (2, -1.5, 0.5) give p(σ) = 2σ - 1.5σ³ + 0.5σ⁵ with
    σ=1 super-attracting. 12 iterations drives all input singular values
    in (0, √2) to 1 in bf16. Used here as the inner polar call inside
    Aurora's leverage-uniform iteration.
    """
    assert G.ndim >= 2
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 2, -1.5, 0.5
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


@torch.no_grad()
def aurora_update(grad, momentum, beta=0.95, nesterov=True,
                  pp_iterations=2, pp_beta=0.5, eps=1e-7):
    """Aurora update — drop-in replacement for muon_update().

    Returns the update tensor (caller applies weight decay + step).
    Signature matches muon_update so it can be slotted into the same
    optimizer class.
    """
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum.clone()
    if update.ndim == 4:  # conv filters
        update = update.view(len(update), -1)

    m, n = update.size(-2), update.size(-1)
    if m == n:
        # Square: Aurora reduces to standard polar.
        update = _polar(update)
    else:
        # Tall matrices: leverage-uniform polar via diagonal preconditioning.
        # Work on the "tall" orientation (m > n) so target_row_sq = n/m < 1.
        transposed = m < n
        if transposed:
            update = update.mT
            m, n = n, m
        G32 = update.to(torch.float32)
        target_row_sq = n / m
        row_norm = G32.norm(dim=-1, keepdim=True).clamp_(min=eps)
        D = 1.0 / row_norm
        for k in range(pp_iterations):
            U = _polar(D * G32)
            if k < pp_iterations - 1:
                row_sq = U.to(torch.float32).pow(2).sum(dim=-1, keepdim=True).clamp_(min=eps * eps)
                D = D * (target_row_sq / row_sq).pow(pp_beta)
        update = U.mT if transposed else U

    # Spectral aspect-ratio scaling (Muon convention).
    update = update * (max(1, grad.size(-2) / grad.size(-1)) ** 0.5)
    return update
