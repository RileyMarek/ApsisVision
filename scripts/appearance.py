"""Training-only sensor appearance randomization; no claim of calibrated space physics."""

import torch
from torch.nn import functional as F

POLICY = {
    "clean_probability": 0.25,
    "gamma": [0.3, 2.5],
    "gain": [0.35, 2.8],
    "offset": [-0.12, 0.12],
    "blur_kernels": [1, 3, 5, 7],
    "read_noise_std_max": 0.04,
    "shot_noise_std_max": 0.03,
    "quantization_bits": [5, 8],
    "occlusion_probability": 0.3,
    "occlusion_extent_max": 0.25,
}


def occlude(images, xy, flags, rectangles):
    """Rectangles are normalized xyxy. Occluded known points cannot stay visible."""
    if rectangles.shape != (len(images), 4) or xy.shape != (*flags.shape, 2):
        raise ValueError("Occlusion shapes disagree")
    if (
        not torch.isfinite(rectangles).all()
        or (rectangles < 0).any()
        or (rectangles > 1).any()
        or (rectangles[:, 2:] < rectangles[:, :2]).any()
    ):
        raise ValueError("Expected ordered normalized rectangles")
    y, x = torch.meshgrid(
        (torch.arange(images.shape[-2], device=images.device) + 0.5) / images.shape[-2],
        (torch.arange(images.shape[-1], device=images.device) + 0.5) / images.shape[-1],
        indexing="ij",
    )
    low, high = rectangles[:, :2], rectangles[:, 2:]
    mask = (
        (x >= low[:, 0, None, None])
        & (x < high[:, 0, None, None])
        & (y >= low[:, 1, None, None])
        & (y < high[:, 1, None, None])
    )
    normalized = (xy + 0.5) / xy.new_tensor([64, 48])
    hidden = ((normalized >= low[:, None]) & (normalized < high[:, None])).all(-1)
    return images.masked_fill(mask[:, None], 0), torch.where(
        hidden & (flags == 1), torch.zeros_like(flags), flags
    )


def sensor_appearance(images, targets, localizer, generator=None):
    """Labels retain geometry. Only explicit occlusion changes visibility targets."""

    def rand(*shape):
        return torch.rand(shape, device=images.device, generator=generator)

    shape = (len(images), 1, 1, 1)
    clean = rand(*shape) < POLICY["clean_probability"]
    gamma = torch.exp(rand(*shape) * images.new_tensor(2.5 / 0.3).log()) * 0.3
    gain = torch.exp(rand(*shape) * images.new_tensor(2.8 / 0.35).log()) * 0.35
    offset = (rand(*shape) - 0.5) * 0.24
    x = torch.linspace(-1, 1, images.shape[-1], device=images.device)[None, None, None]
    y = torch.linspace(-1, 1, images.shape[-2], device=images.device)[
        None, None, :, None
    ]
    illumination = 1 + (rand(*shape) - 0.5) * 0.6 * x + (rand(*shape) - 0.5) * 0.6 * y
    changed = (images.pow(gamma) * gain * illumination + offset).clamp(0, 1)
    kernels = torch.floor(rand(len(images)) * 4).long()
    for index, kernel in enumerate(POLICY["blur_kernels"]):
        if kernel > 1:
            blurred = F.avg_pool2d(
                changed, kernel, stride=1, padding=kernel // 2, count_include_pad=False
            )
            changed = torch.where(
                (kernels == index)[:, None, None, None], blurred, changed
            )
    noise = torch.randn(images.shape, device=images.device, generator=generator)
    changed = changed + noise * (
        rand(*shape) * 0.04 + rand(*shape) * 0.03 * changed.sqrt()
    )
    levels = 2 ** (5 + torch.floor(rand(*shape) * 4)) - 1
    changed = (changed.clamp(0, 1) * levels).round() / levels
    changed = torch.where(clean, images, changed)
    if localizer:
        xy, flags = targets
        low = rand(len(images), 2) * 0.9
        high = torch.minimum(
            low + 0.05 + rand(len(images), 2) * 0.2, torch.ones_like(low)
        )
        active = (rand(len(images)) < 0.3) & ~clean.flatten()
        high = torch.where(active[:, None], high, low)
        changed, flags = occlude(changed, xy, flags, torch.cat([low, high], dim=1))
        targets = (xy, flags)
    return changed, targets
