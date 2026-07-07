from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Any

import torch

StreamTensors = dict[str, torch.Tensor]


@dataclass(frozen=True)
class StreamSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    h5_file: str

    @property
    def flat_dim(self) -> int:
        return int(prod(self.shape))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "h5_file": self.h5_file,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StreamSpec":
        return cls(
            name=d["name"],
            shape=tuple(d["shape"]),
            dtype=d["dtype"],
            h5_file=d["h5_file"],
        )


class Backbone(ABC):
    """Text-encoder backbone that produces per-stream embeddings for a prompt.

    Concrete subclasses register themselves with the registry via
    `@register_backbone("name")`. `encode` is used at extraction time,
    `decode` (optional) at inference time to turn reconstructed per-stream
    tensors into a downstream output such as an image.
    """

    name: str = ""
    stream_specs: list[StreamSpec] = []

    def __init__(self, device: str | torch.device = "cuda") -> None:
        self.device = torch.device(device) if isinstance(device, str) else device
        self._loaded = False

    @property
    def flat_dim(self) -> int:
        return sum(s.flat_dim for s in self.stream_specs)

    @abstractmethod
    def load(self) -> None:
        ...

    def unload(self) -> None:
        self._loaded = False

    @abstractmethod
    def encode(self, prompt: str) -> StreamTensors:
        """Return per-stream tensors keyed by StreamSpec.name.

        Each tensor's per-sample shape must match its StreamSpec.shape; the
        leading batch dim is optional (extraction squeezes it before writing).
        """

    def decode(
        self, streams: StreamTensors, output_path: str | Path, **kwargs: Any
    ) -> Any | None:
        """Optional: turn per-stream tensors into a downstream output.

        Backbones without a decode path (e.g. text-only research encoders) may
        leave this as the default no-op that returns None.
        """
        return None
