"""Optional LPIPS distance between two images (install ``lpips``)."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image


def lpips_alex(
    path_a: Path | str,
    path_b: Path | str,
    device: str | None = None,
) -> float | None:
    """
    Return AlexNet LPIPS between two RGB images, or ``None`` if ``lpips`` is not installed.
    """
    try:
        import lpips as _lpips
    except ImportError:
        return None

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = _lpips.LPIPS(net="alex").to(device)
    loss_fn.eval()

    def load(p):
        im = Image.open(p).convert("RGB")
        import torchvision.transforms as T

        t = T.Compose([T.Resize((256, 256)), T.ToTensor()])
        x = t(im).unsqueeze(0).to(device) * 2.0 - 1.0
        return x

    with torch.no_grad():
        a = load(path_a)
        b = load(path_b)
        d = loss_fn(a, b).item()
    return float(d)
