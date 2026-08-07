"""Synthetic backbone with SD3.5's exact stream shapes, for CPU smoke tests.

``fake_test`` is too small to exercise ``evaluation/sd3_pack.py``, which hardcodes the
``[333, 4096]`` sequence and ``[2048]`` pooled layout. Anything that packs a prediction
into SD3 conditioning — including the ``prompt_modified_packed`` method added for AUG-01 —
is therefore untestable against ``fake_test``.

This backbone emits the real shapes with deterministic per-prompt noise, so the whole
benchmark path (truncation, normalisation, packing, manifest emission) can be smoke-tested
on a laptop in ``--simulated`` mode. It never loads a diffusion model.

Cost note: one prompt is 333*4096 + 2048 = 1 366 016 float16 values ≈ 2.7 MB on disk.
Keep smoke datasets to a few dozen prompts.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from backbones.base import Backbone, StreamSpec, StreamTensors
from backbones.registry import register_backbone


@register_backbone("fake_sd3")
class FakeSd3Backbone(Backbone):
    stream_specs = [
        StreamSpec(name="seq", shape=(333, 4096), dtype="float16", h5_file="embds.h5"),
        StreamSpec(name="pooled", shape=(2048,), dtype="float16", h5_file="embds_pooled.h5"),
    ]

    def load(self) -> None:
        self._loaded = True

    def _seed_for(self, prompt: str) -> int:
        return int.from_bytes(hashlib.sha256(prompt.encode()).digest()[:4], "big")

    def encode(self, prompt: str) -> StreamTensors:
        g = torch.Generator().manual_seed(self._seed_for(prompt))
        # Scaled down so float16 storage keeps a usable dynamic range, and so the
        # per-coordinate variance ranking that drives top-k truncation is non-degenerate.
        return {
            "seq": torch.randn(333, 4096, generator=g) * 0.5,
            "pooled": torch.randn(2048, generator=g) * 0.5,
        }

    def decode(self, streams: StreamTensors, output_path: str | Path, **_: Any) -> None:
        return None
