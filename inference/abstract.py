import json
from pathlib import Path
from typing import List

import torch

from inference.utils import display_table, flatten_dict_of_list, is_in
from trainings.config.config import (
    derive_shapes,
    initialise_instance,
    read_training_params_from_yaml,
)
from trainings.dataloader.dataloader import H5Dataset
from trainings.models.utils import import_model


class SFDInference:
    def __init__(self, folder_path: str | Path, device: str = "cpu"):
        self.folder_path = folder_path
        self.path_yaml = Path(folder_path, "params.yaml")
        self.device = device

        self.get_training_parameters()
        self.initialize_dataset()
        self.initialize_model()

        self.read_prompts()

    def get_training_parameters(self) -> float:
        self.tp, _ = read_training_params_from_yaml(self.path_yaml)

    def initialize_dataset(self):
        self.dataset = initialise_instance(H5Dataset, self.tp)

        self.tp["logger"] = None
        self.tp["device"] = self.device

        derive_shapes(self.tp, self.dataset)

    def initialize_model(self):
        Decoder = import_model(self.tp["model_name"])
        self.decoder = initialise_instance(Decoder, self.tp)
        self.decoder = self.decoder.to(self.device)
        self.reset_model()

        print(self.decoder)

    def reset_model(self):
        self.decoder.load_state_dict(torch.load(Path(self.folder_path, "model.pt")))
        self.decoder.eval()
        self.decoder.apply_mask(
            self.dataset.mask_reduced.to(self.device), self.tp["batch_size"]
        )

    @property
    def n_properties(self) -> int:
        return self.dataset.n_properties

    @property
    def n_categories(self) -> int:
        return self.dataset.n_categories

    @property
    def n_features(self) -> float:
        return self.dataset.n_properties * self.tp["n_repeat"]

    @property
    def dim_output(self) -> float:
        return self.dataset.dim_output

    @property
    def n_prompts(self) -> float:
        return self.dataset.n_prompts

    def search_idx_prompt(
        self, list_properties: List[str], only_first: bool = True
    ) -> tuple[List[int], List[str]] | int:
        # check all the properties exist
        for p in list_properties:
            if p.strip() not in self.properties:
                raise Exception(f"property '{p}' not found.")

        indices = []
        prompts = []
        for idx, prompt in enumerate(self.prompts):
            prompt = prompt["prompt"]
            prompt_split = prompt.split(", ")
            if is_in(list_properties, prompt_split):
                indices.append(idx)
                prompts.append(prompt)

        if not only_first:
            # display
            for idx, prompt in zip(indices, prompts):
                print(f"{idx:6d}   {prompt}")

            return indices, prompts

        return indices[0], prompts[0]

    def read_prompts(self):
        # properties metadata
        with open(Path(self.tp["folder_path"], "properties.json"), "r") as f:
            self.dict_categories_with_properties = json.load(f)

        self.properties = flatten_dict_of_list(self.dict_categories_with_properties)
        self.properties = [p.strip() for p in self.properties]

        # prompts
        with open(Path(self.tp["folder_path"], "prompts.json"), "r") as f:
            self.prompts = json.load(f)

    def display_properties(self):
        # p = 0
        # c = 0
        # for key, value in self.dict_categories_with_properties.items():
        #     print(f"c{c}", key)
        #     for v in value:
        #         print(" " * 5, f"p{p}", v)
        #         p += 1
        #     c += 1

        display_table(self.dict_categories_with_properties)

    def overwrite_full_embedding(
        self, embd_topk: torch.tensor, idx: int
    ) -> dict[str, torch.Tensor]:
        """Given a top-k reconstruction, rebuild the per-stream embeddings.

        The flat vector is split according to `dataset.stream_specs` (written
        by the backbone at extraction time) and each slice is reshaped back to
        its per-sample shape with a leading batch dim.
        """

        full_embd = self.get_full_embedding(idx)

        embd_topk = self.dataset.denormalize(embd_topk)

        if self.dataset.pca_rotation:
            # slot the edited top-k dims into the rotated full vector, then invert
            full_rot = (full_embd - self.dataset.pca_mean) @ self.dataset.pca_components
            full_rot[self.dataset.indices_truncate_embds_topk] = embd_topk
            full_embd = self.dataset.pca_mean + full_rot @ self.dataset.pca_components.T
        else:
            full_embd[self.dataset.indices_truncate_embds_topk] = embd_topk

        streams: dict[str, torch.Tensor] = {}
        offset = 0
        for spec in self.dataset.stream_specs:
            n = spec.flat_dim
            slice_flat = full_embd[offset : offset + n]
            streams[spec.name] = slice_flat.reshape(1, *spec.shape)
            offset += n
        return streams

    def get_backbone(self, **kwargs):
        """Instantiate the backbone the dataset was extracted with.

        Merges the kwargs recorded in the manifest at extraction time with
        any caller overrides -- important for generic backbones like
        `hf_causal_lm` that need `model_id`/`hidden_size` at construction.
        """
        from backbones import get_backbone

        merged = {**self.dataset.backbone_kwargs, **kwargs}
        return get_backbone(self.dataset.backbone_name, **merged)

    def get_full_embedding(self, idx: int) -> torch.tensor:
        indices_truncate_embds_topk = self.dataset.indices_truncate_embds_topk
        normalize = self.dataset.normalize
        pca_rotation = self.dataset.pca_rotation
        self.dataset.indices_truncate_embds_topk = None
        self.dataset.normalize = None
        self.dataset.pca_rotation = False

        full_embd, _ = self.dataset.__getitem__(idx)

        self.dataset.indices_truncate_embds_topk = indices_truncate_embds_topk
        self.dataset.normalize = normalize
        self.dataset.pca_rotation = pca_rotation

        return full_embd
