"""DINOv2 image embedding cosine similarity (identity / structure drift proxy)."""

from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _default_transform():
    return T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


def load_dinov2_vits14(device: str | torch.device | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    try:
        model = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vits14",
            pretrained=True,
            trust_repo=True,
        )
    except TypeError:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
    model.eval()
    model.to(device)
    return model, _default_transform(), device


@torch.no_grad()
def embed_image_path(
    path: Path | str,
    model: torch.nn.Module,
    transform,
    device: torch.device,
) -> torch.Tensor:
    im = Image.open(path).convert("RGB")
    x = transform(im).unsqueeze(0).to(device)
    e = model(x)
    return e / e.norm(dim=-1, keepdim=True)


@torch.no_grad()
def dino_cosine_similarity(
    path_a: Path | str,
    path_b: Path | str,
    model: torch.nn.Module,
    transform,
    device: torch.device,
) -> float:
    ea = embed_image_path(path_a, model, transform, device)
    eb = embed_image_path(path_b, model, transform, device)
    return float((ea * eb).sum(dim=-1).squeeze().item())
