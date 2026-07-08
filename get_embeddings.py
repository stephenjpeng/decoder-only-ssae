"""Extract text-encoder embeddings for the SSAE dataset.

Loads the requested backbone from the `backbones` registry, iterates the
prompts, writes one H5 file per stream per prompt, and records a top-level
`manifest.json` that the dataloader + inference layer read to know how to
slice and reshape the flat embedding.
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

import backbones
from backbones import get_backbone, list_backbones
from backbones.base import StreamSpec
from backbones.registry import _REGISTRY as _BACKBONE_REGISTRY


NUMPY_DTYPES = {"float16": np.float16, "float32": np.float32, "bfloat16": np.float32}
TORCH_DTYPES = {"float16": torch.float16, "float32": torch.float32}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", required=True, choices=list_backbones())
    p.add_argument("--prompts", required=True, help="Path to prompts.json")
    p.add_argument(
        "--out",
        required=True,
        help="Output folder; embeddings go under <out>/embds/",
    )
    p.add_argument("--n", type=int, default=None, help="Optional prompt cap")
    p.add_argument(
        "--style",
        default="",
        help="Optional suffix appended to every prompt (kept for parity with the old script)",
    )
    # Backbone constructor overrides -- forwarded only to backbones whose
    # __init__ accepts the matching kwarg (see _filter_backbone_kwargs). Lets
    # you swap the HF checkpoint / context length without touching the code.
    p.add_argument("--model-id", default=None, help="Override the backbone's model id")
    p.add_argument("--max-length", type=int, default=None, help="Tokenizer max length (text backbones)")
    p.add_argument("--hidden-size", type=int, default=None, help="Model hidden size (generic hf_causal_lm)")
    p.add_argument("--dtype", default=None, choices=["float16", "bfloat16", "float32"], help="Torch dtype for the backbone")
    p.add_argument(
        "--prompt-template",
        default=None,
        help="String with a '{prompt}' placeholder wrapped around each prompt before tokenization",
    )
    p.add_argument(
        "--apply-chat-template",
        dest="apply_chat_template",
        action="store_true",
        default=None,
        help="Run the tokenizer's chat template (instruction-tuned models like gemma-2-2b-it default to on)",
    )
    p.add_argument(
        "--no-chat-template",
        dest="apply_chat_template",
        action="store_false",
        help="Disable chat template even on IT presets",
    )
    p.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system message prepended when --apply-chat-template is on",
    )
    return p.parse_args()


def _filter_backbone_kwargs(name: str, candidate: dict) -> dict:
    cls = _BACKBONE_REGISTRY[name]
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    return {k: v for k, v in candidate.items() if v is not None and k in accepted}


def _write_stream(path: Path, tensor: torch.Tensor, spec: StreamSpec) -> None:
    torch_dtype = TORCH_DTYPES.get(spec.dtype, torch.float32)
    data = tensor.to(torch_dtype).flatten().cpu().detach().numpy()
    with h5py.File(path, "w") as f:
        f.create_dataset("vector", data=data)


def _validate_stream(name: str, tensor: torch.Tensor, spec: StreamSpec) -> None:
    if tuple(tensor.shape) != spec.shape:
        raise ValueError(
            f"backbone returned stream {name!r} with shape {tuple(tensor.shape)}, "
            f"expected {spec.shape}"
        )


def main() -> None:
    args = parse_args()

    with open(args.prompts, "r") as f:
        prompts = json.load(f)
    if args.n is not None:
        prompts = prompts[: args.n]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}, backbone: {args.backbone}, n_prompts: {len(prompts)}")

    overrides = _filter_backbone_kwargs(
        args.backbone,
        {
            "model_id": args.model_id,
            "max_length": args.max_length,
            "hidden_size": args.hidden_size,
            "dtype": args.dtype,
            "prompt_template": args.prompt_template,
            "apply_chat_template": args.apply_chat_template,
            "system_prompt": args.system_prompt,
        },
    )
    if overrides:
        print(f"backbone overrides: {overrides}")
    backbone = get_backbone(args.backbone, device=device, **overrides)
    backbone.load()

    out_root = Path(args.out)
    embds_root = out_root / "embds"
    embds_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "backbone": backbone.name,
        "backbone_kwargs": overrides,
        "streams": [s.to_dict() for s in backbone.stream_specs],
        "flat_dim": backbone.flat_dim,
        "n_prompts": len(prompts),
        "style_suffix": args.style,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(embds_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    start = time.time()
    for i, entry in enumerate(prompts):
        prompt_text = entry["prompt"] + args.style
        prompt_dir = embds_root / f"embds_{i}"
        prompt_dir.mkdir(exist_ok=True)

        with open(prompt_dir / "prompts.txt", "w") as f:
            f.write(prompt_text)

        streams = backbone.encode(prompt_text)
        for spec in backbone.stream_specs:
            if spec.name not in streams:
                raise KeyError(
                    f"backbone {backbone.name!r} did not return stream {spec.name!r}"
                )
            tensor = streams[spec.name]
            _validate_stream(spec.name, tensor, spec)
            _write_stream(prompt_dir / spec.h5_file, tensor, spec)

        if i % 10 == 0:
            elapsed = time.time() - start
            print(f"i={i}/{len(prompts)} ({elapsed:.1f}s): {prompt_text}")

    print(f"done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    _ = backbones  # keep import so registrations run
    main()
