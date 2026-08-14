"""End-to-end training smoke tests using simulated data.

Runs `training()` for 1 epoch on each registered decoder variant and checks
that a checkpoint file is produced and the recorded loss is finite. Uses a
synthetic prompt/property fixture (no real embeddings), so the test runs on
CPU in seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from trainable_inputs_all_clips import training


CATEGORIES = {
    "hair": ["A blond girl", "A brunette girl", "A ginger girl"],
    "eyes": ["with blue eyes", "with brown eyes", "with green eyes"],
    "pose": ["sitting", "standing", "walking"],
}

# eyes has a "same" property; hair and pose are never-same. This keeps
# n_features_situation strictly less than n_features for model_trainable_inputs.
PROPERTIES_SAME = {
    "hair": False,
    "eyes": ["with blue eyes"],
    "pose": False,
}


def _build_dataset_folder(root: Path) -> Path:
    folder = root / "dataset"
    folder.mkdir()
    embds_folder = folder / "embds"
    embds_folder.mkdir()
    (embds_folder / "embds_0").mkdir()
    manifest = {
        "backbone": "fake_test",
        "backbone_kwargs": {},
        "streams": [
            {
                "name": "simulated",
                "shape": [32],
                "dtype": "float32",
                "h5_file": "embds.h5",
            }
        ],
    }
    (embds_folder / "manifest.json").write_text(json.dumps(manifest))

    (folder / "properties.json").write_text(json.dumps(CATEGORIES))
    (folder / "properties_same.json").write_text(json.dumps(PROPERTIES_SAME))

    prompts = []
    tid = 0
    for hair in CATEGORIES["hair"]:
        for eyes in CATEGORIES["eyes"]:
            for pose in CATEGORIES["pose"]:
                prompts.append({"id": tid, "prompt": f"{hair}, {eyes}, {pose}"})
                tid += 1
    (folder / "prompts.json").write_text(json.dumps(prompts))

    return folder


def _build_yaml(root: Path, dataset_folder: Path, model_name: str) -> Path:
    params = {
        "training": {
            "model": {
                "model_name": model_name,
                "using_blocs": False,
            },
            "dataloader": {
                "folder_path": str(dataset_folder) + "/",
                "truncate_n_prompts": None,
                "truncate_embds_topk": None,
                "add_property_is_the_same": True,
                "normalize": None,
                "num_workers": 0,
                "simulated": {
                    "simulated": True,
                    "dim_clip_simulated": 32,
                },
            },
            "training": {
                "n_epochs": 1,
                "print_frequency": 1,
                "save_model_frequency": None,
                "plot_frequency": 1,
                "seed": 0,
                "batch_size": 27,
                "lr": 0.001,
                "beta1": 0.9,
                "beta2": 0.999,
                "lr_scheduler": {
                    "lr_scheduler_type": None,
                },
            },
            "sparse_feature_design": {
                "n_repeat": 7,
            },
        }
    }
    yaml_path = root / f"params_{model_name}.yaml"
    yaml_path.write_text(yaml.safe_dump(params))
    return yaml_path


@pytest.mark.parametrize(
    "model_name",
    ["model_trainable_inputs", "model_avg_feature", "model_trainable_input_inv"],
)
def test_training_one_epoch_smoke(
    tmp_path: Path, model_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    dataset_folder = _build_dataset_folder(tmp_path)
    yaml_path = _build_yaml(tmp_path, dataset_folder, model_name)
    output_folder = tmp_path / f"run_{model_name}"

    training(
        output_folder=str(output_folder),
        path_yaml=yaml_path,
        overwrite_output=True,
    )

    checkpoint = output_folder / "model.pt"
    assert checkpoint.exists(), f"checkpoint missing at {checkpoint}"

    log_path = output_folder / "training.log"
    assert log_path.exists()

    # loss must be finite after 1 epoch (parsed from the epoch line in the log)
    log_text = log_path.read_text()
    import math
    import re

    match = re.search(r"training loss: ([0-9.eE+-]+)", log_text)
    assert match is not None, f"no 'training loss' line in log:\n{log_text}"
    loss_value = float(match.group(1))
    assert math.isfinite(loss_value), f"loss not finite: {loss_value}"

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert len(state) > 0
