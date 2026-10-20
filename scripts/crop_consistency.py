"""Training-only peak-coordinate equivariance; no pseudo-labels or audit inputs."""

import torch
from torch.nn import functional as F


def local_centers(logits):
    """Differentiable expectation around detached argmax; matches local decoder."""
    b, k, h, w = logits.shape
    index = logits.flatten(2).detach().argmax(-1)
    y, x = torch.meshgrid(
        torch.arange(h, device=logits.device),
        torch.arange(w, device=logits.device),
        indexing="ij",
    )
    mask = ((x - (index % w)[..., None, None]).abs() <= 2) & (
        (y - (index // w)[..., None, None]).abs() <= 2
    )
    probability = logits.masked_fill(~mask, -torch.inf).flatten(2).softmax(-1)
    grid = torch.stack([x.flatten(), y.flatten()], -1).to(logits.dtype)
    return probability.reshape(b, k, h * w) @ grid


def to_parent(points, transform):
    scale, offset = transform
    size = points.new_tensor([64, 48])
    normal = (points + 0.5) / size * 2 - 1
    return ((normal * scale[:, None] + offset[:, None]) + 1) * size / 2 - 0.5


def equivariance_loss(
    heatmaps_a, heatmaps_b, transform_a, transform_b, target_a, target_b
):
    """Both branches retain supervised CE. Hidden/unknown/out-of-FOV points excluded."""
    xy_a, flags_a = target_a
    xy_b, flags_b = target_b
    size = xy_a.new_tensor([64, 48])
    visible = (flags_a == 1) & (flags_b == 1)
    visible &= ((xy_a >= 0) & (xy_a < size)).all(-1) & (
        (xy_b >= 0) & (xy_b < size)
    ).all(-1)
    if not visible.any():
        return (heatmaps_a.sum() + heatmaps_b.sum()) * 0
    a = to_parent(local_centers(heatmaps_a), transform_a)
    b = to_parent(local_centers(heatmaps_b), transform_b)
    return F.smooth_l1_loss(a[visible], b[visible])
