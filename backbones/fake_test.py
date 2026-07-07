"""Tiny synthetic backbone used only in smoke tests.

Produces deterministic per-prompt embeddings from `hash(prompt)`, with two
streams whose shapes are small enough that a full end-to-end run (extract,
train a few steps, inference reshape) finishes in under a second on CPU.
Not registered when a real backbone is desired -- import this module
explicitly to enable it, e.g. in test scripts.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from backbones.base import Backbone, StreamSpec, StreamTensors
from backbones.registry import register_backbone


@register_backbone("fake_test")
class FakeTestBackbone(Backbone):
    stream_specs = [
        StreamSpec(name="seq", shape=(4, 8), dtype="float32", h5_file="embds.h5"),
        StreamSpec(name="pooled", shape=(6,), dtype="float32", h5_file="embds_pooled.h5"),
    ]

    def load(self) -> None:
        self._loaded = True

    def _seed_for(self, prompt: str) -> int:
        return int.from_bytes(hashlib.sha256(prompt.encode()).digest()[:4], "big")

    def encode(self, prompt: str) -> StreamTensors:
        g = torch.Generator().manual_seed(self._seed_for(prompt))
        return {
            "seq": torch.randn(*self.stream_specs[0].shape, generator=g),
            "pooled": torch.randn(*self.stream_specs[1].shape, generator=g),
        }

    def decode(
        self, streams: StreamTensors, output_path: str | Path, **_: Any
    ) -> None:
        # no-op for tests: this backbone has no downstream artifact
        return None
