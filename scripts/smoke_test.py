"""End-to-end smoke test using the fake_test backbone.

Creates a temp dataset folder, runs extraction, instantiates H5Dataset,
performs a training step on the trainable-inputs decoder, and validates the
inference-time flat -> per-stream reshape round-trip.

Run from repo root:
    .venv/bin/python scripts/smoke_test.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from math import prod
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backbones import get_backbone  # noqa: E402
from trainings.dataloader.dataloader import H5Dataset  # noqa: E402
from trainings.models.model_trainable_inputs import Decoder  # noqa: E402


# Small cartesian product -- must yield >= 16 prompts because a hardcoded
# consistency check in trainings/dataloader/checks.py samples prompt idx 15.
PROPERTIES = {
    "hair": ["A blond girl", "A brunette girl"],
    "eyes": ["with blue eyes", "with brown eyes"],
    "pose": ["looking left", "looking right"],
    "outfit": ["in a red coat", "in a blue coat"],
}

# properties_same.json: False means "never same" across prompts (i.e. varies).
PROPERTIES_SAME = {k: False for k in PROPERTIES}


def _build_prompts() -> list[dict]:
    keys = list(PROPERTIES)
    prompts = []

    def rec(i: int, chosen: list[str]) -> None:
        if i == len(keys):
            prompts.append({"prompt": ", ".join(chosen)})
            return
        for v in PROPERTIES[keys[i]]:
            rec(i + 1, chosen + [v])

    rec(0, [])
    return prompts


def _run_extraction(prompts_path: Path, out_dir: Path) -> None:
    cmd = [
        sys.executable,
        str(REPO / "get_embeddings.py"),
        "--backbone",
        "fake_test",
        "--prompts",
        str(prompts_path),
        "--out",
        str(out_dir),
    ]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO)


def _check_manifest(embds_dir: Path) -> dict:
    manifest_path = embds_dir / "manifest.json"
    assert manifest_path.exists(), f"missing {manifest_path}"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["backbone"] == "fake_test"
    assert manifest["flat_dim"] == 4 * 8 + 6
    print("OK manifest:", manifest)
    return manifest


def _check_dataset(folder_path: str) -> H5Dataset:
    dataset = H5Dataset(
        folder_path=folder_path,
        truncate_n_prompts=None,
        truncate_embds_topk=None,
        normalize=None,
    )
    assert dataset.backbone_name == "fake_test"
    assert dataset.dim_x == 4 * 8 + 6
    n_expected = 1
    for v in PROPERTIES.values():
        n_expected *= len(v)
    assert len(dataset) == n_expected, (len(dataset), n_expected)
    # sanity: get one item, verify shape and concat order via the backbone
    embd, mask = dataset[0]
    assert embd.shape == (4 * 8 + 6,), embd.shape

    # verify per-stream order in the flat vector matches the backbone's encode()
    backbone = get_backbone("fake_test")
    prompt_text = dataset.properties.prompts[0]["prompt"]
    streams = backbone.encode(prompt_text)
    expected = torch.cat([streams["seq"].flatten(), streams["pooled"].flatten()])
    assert torch.allclose(embd, expected, atol=1e-6), (
        "flat concat does not match backbone.encode() order"
    )
    print("OK dataset: dim_x, len, and stream-order concat all correct")
    return dataset


def _one_training_step(dataset: H5Dataset, device: torch.device) -> None:
    # n_repeat must exceed n_pid_never_same for the Decoder invariant
    # n_features_situation < n_features (see trainable_inputs_all_clips.py).
    n_repeat = 10
    n_features = dataset.properties.n_properties * n_repeat
    n_features_situation = (
        dataset.properties.n_properties * dataset.same_id.n_pid_never_same
    )
    decoder = Decoder(
        n_features=n_features,
        n_prompts=len(dataset),
        dim_output=dataset.dim_x,
        n_repeat=n_repeat,
        tid_same=dataset.same_id.tid_same,
        n_features_situation=n_features_situation,
    ).to(device)
    decoder.apply_mask(dataset.mask_reduced.to(device), batch_size=len(dataset))

    ys = torch.stack([dataset[i][0] for i in range(len(dataset))]).to(device)
    optim = torch.optim.Adam(decoder.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()

    loss_before = None
    for step in range(20):
        pred = decoder(batch_size=len(dataset), batch_idx=0)
        loss = loss_fn(pred, ys)
        if step == 0:
            loss_before = loss.item()
        optim.zero_grad()
        loss.backward()
        optim.step()
    loss_after = loss.item()
    print(f"OK training step: loss {loss_before:.4f} -> {loss_after:.4f}")
    assert loss_after < loss_before, "loss did not decrease"


def _check_inference_reshape(dataset: H5Dataset) -> None:
    from inference.abstract import SFDInference  # noqa: F401 -- kept for API check

    # Manually exercise the reshape logic (without instantiating the full
    # SFDInference which requires a saved checkpoint).
    full_embd = dataset[0][0].clone()  # already normalized=None
    streams: dict[str, torch.Tensor] = {}
    offset = 0
    for spec in dataset.stream_specs:
        n = int(prod(spec.shape))
        slice_flat = full_embd[offset : offset + n]
        streams[spec.name] = slice_flat.reshape(1, *spec.shape)
        offset += n
    # invariants
    assert set(streams) == {"seq", "pooled"}
    assert streams["seq"].shape == (1, 4, 8)
    assert streams["pooled"].shape == (1, 6)
    # round-trip: reshape -> flatten -> re-concat should match the original
    round_trip = torch.cat(
        [streams[spec.name].flatten() for spec in dataset.stream_specs]
    )
    assert torch.allclose(full_embd, round_trip, atol=0)
    print("OK inference reshape: round-trip flat<->per-stream exact")


def _check_missing_manifest_error(tmp: Path) -> None:
    bad = tmp / "no_manifest"
    (bad / "embds" / "embds_0").mkdir(parents=True)
    (bad / "properties.json").write_text(json.dumps(PROPERTIES))
    (bad / "prompts.json").write_text(json.dumps(_build_prompts()))
    (bad / "properties_same.json").write_text(json.dumps(PROPERTIES_SAME))
    try:
        H5Dataset(folder_path=str(bad) + "/")
    except FileNotFoundError as e:
        assert "manifest.json" in str(e), e
        print("OK MissingManifestError raised for legacy folder")
        return
    raise AssertionError("expected MissingManifestError")


def main() -> None:
    device = torch.device("cpu")
    with tempfile.TemporaryDirectory(prefix="ssae_smoke_") as tmp_s:
        tmp = Path(tmp_s)
        folder = tmp / "ds"
        folder.mkdir()

        # dataset scaffolding
        (folder / "properties.json").write_text(json.dumps(PROPERTIES, indent=2))
        (folder / "properties_same.json").write_text(json.dumps(PROPERTIES_SAME))
        prompts_path = folder / "prompts.json"
        prompts_path.write_text(json.dumps(_build_prompts(), indent=2))

        # 1) run extraction CLI
        _run_extraction(prompts_path, folder)

        # 2) manifest is present + correct
        _check_manifest(folder / "embds")

        # 3) dataloader reads manifest, discovers streams, concat is correct
        dataset = _check_dataset(str(folder) + "/")

        # 4) training step decreases loss
        _one_training_step(dataset, device)

        # 5) inference reshape is exact round-trip
        _check_inference_reshape(dataset)

        # 6) helpful error for pre-manifest folders
        _check_missing_manifest_error(tmp)

        # 7) text-LM backbones register and expose expected stream shapes
        _check_hf_causal_lm_registration()

    print("\nALL SMOKE CHECKS PASSED")


def _check_hf_causal_lm_registration() -> None:
    """Registry + stream_specs wiring for the text-LM backbones.

    Cheap check that avoids downloading weights: instantiate without calling
    `load()`, confirm stream shapes and flat_dim match constructor args.
    """
    from backbones import get_backbone, list_backbones

    for name in ("hf_causal_lm", "gemma_2_2b_it"):
        assert name in list_backbones(), f"{name} missing from registry"

    generic = get_backbone(
        "hf_causal_lm",
        device="cpu",
        model_id="unused/for-shape-check",
        hidden_size=16,
        max_length=8,
    )
    assert [s.name for s in generic.stream_specs] == ["seq"]
    assert generic.stream_specs[0].shape == (8, 16)
    assert generic.flat_dim == 8 * 16

    gemma = get_backbone("gemma_2_2b_it", device="cpu", max_length=32)
    assert gemma.model_id == "google/gemma-2-2b-it"
    assert gemma.hidden_size == 2304
    assert gemma.stream_specs[0].shape == (32, 2304)

    # model_id override still works via the preset
    gemma_base = get_backbone(
        "gemma_2_2b_it", device="cpu", model_id="google/gemma-2-2b", max_length=32
    )
    assert gemma_base.model_id == "google/gemma-2-2b"

    # prompt-wrapping defaults + overrides
    assert gemma.apply_chat_template is True, "IT preset should default to chat template on"
    assert generic.apply_chat_template is False, "generic hf_causal_lm should default off"

    templated = get_backbone(
        "hf_causal_lm",
        device="cpu",
        model_id="unused",
        hidden_size=16,
        max_length=8,
        prompt_template="describe: {prompt}",
    )
    # _format_prompt without chat template = pure string substitution (no tokenizer needed)
    assert templated._format_prompt("a cat") == "describe: a cat"

    try:
        get_backbone(
            "hf_causal_lm",
            device="cpu",
            model_id="unused",
            hidden_size=16,
            max_length=8,
            prompt_template="no placeholder here",
        )
    except ValueError as e:
        assert "{prompt}" in str(e), e
    else:
        raise AssertionError("expected ValueError for template missing {prompt}")
    print("OK hf_causal_lm + gemma_2_2b_it registration, shapes, and prompt wrapping")


if __name__ == "__main__":
    main()
