import json
import os
from enum import Enum
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from backbones.base import StreamSpec
from trainings.config.config import check_yaml_params
from trainings.dataloader.checks import run_checks_dataset
from trainings.dataloader.properties.properties import Properties
from trainings.dataloader.properties.same_id import SameId

MAX_MIN = "MAX_MIN"
MANIFEST_FILENAME = "manifest.json"

TRUNCATE_RANGE = "range"
TRUNCATE_PCA = "pca"
PCA_RESIDUAL = "residual"
PCA_REPLACE = "replace"


class MissingManifestError(FileNotFoundError):
    """Embedding folder was produced by an older extraction script.

    Re-run `python get_embeddings.py --backbone <name> --prompts ... --out ...`
    to regenerate the folder with a manifest.json.
    """


class H5Dataset(Dataset):
    def __init__(
        self,
        folder_path,
        truncate_n_prompts=None,
        truncate_embds_topk=None,
        truncate_embds_method=TRUNCATE_RANGE,
        pca_semantics=PCA_RESIDUAL,
        add_property_is_the_same=False,
        simulated=False,
        dim_clip_simulated=5000,
        logger=None,
        normalize=None,
    ):
        self.folder_path = folder_path
        self.truncate_n_prompts = truncate_n_prompts
        self.logger = logger
        self.log_print = print if self.logger is None else logger.print
        self.truncate_embds_topk = truncate_embds_topk
        self.truncate_embds_method = check_yaml_params(
            truncate_embds_method, possible_values=[TRUNCATE_RANGE, TRUNCATE_PCA]
        )
        self.pca_semantics = check_yaml_params(
            pca_semantics, possible_values=[PCA_RESIDUAL, PCA_REPLACE]
        )
        self.indices_truncate_embds_topk = None
        self.pca_mean = None
        self.pca_components = None
        self.pca_singular_values = None
        self.pca_explained_variance_ratio = None
        self.normalize = None  # we init the dateset with self.normalize = None and overwrite it to its actual value (normalize) at the end of the init

        self.X = None

        # simulated mode
        # for the moment, simulated uses the same mask as the real data (i.e., same number of prompts)
        self.simulated = simulated
        self.dim_clip_simulated = dim_clip_simulated

        # extract files
        self.folder_embds = folder_path + "embds/"

        # backbone-agnostic manifest describes the streams stored per prompt
        manifest_path = os.path.join(self.folder_embds, MANIFEST_FILENAME)
        if not os.path.exists(manifest_path):
            raise MissingManifestError(
                f"No {MANIFEST_FILENAME} at {manifest_path}. "
                f"Re-extract with `python get_embeddings.py --backbone <name> "
                f"--prompts <prompts.json> --out {folder_path}`."
            )
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        self.backbone_name: str = manifest["backbone"]
        self.backbone_kwargs: dict = manifest.get("backbone_kwargs", {})
        self.stream_specs: list[StreamSpec] = [
            StreamSpec.from_dict(s) for s in manifest["streams"]
        ]
        self.log_print(
            f"Loaded manifest: backbone={self.backbone_name}, "
            f"streams={[s.name for s in self.stream_specs]}"
        )

        self.folders = [
            f
            for f in os.listdir(self.folder_embds)
            if os.path.isdir(os.path.join(self.folder_embds, f))
            and f.startswith("embds_")
        ]

        self.folders = sorted(self.folders, key=lambda x: int(x.split("_")[1]))

        # truncate number of prompts
        if self.truncate_n_prompts is not None:
            self.log_print(
                f"Truncating the number of prompts from {len(self.folders)} to {self.truncate_n_prompts}."
            )
            self.folders = self.folders[: self.truncate_n_prompts]

        # per-stream file paths (relative to folder_embds), one list per stream
        self.files_by_stream: dict[str, list[str]] = {
            spec.name: [f + "/" + spec.h5_file for f in self.folders]
            for spec in self.stream_specs
        }
        self.files_prompts = [f + "/prompts.txt" for f in self.folders]

        # get properties
        self.properties = Properties(folder_path=folder_path, logger=logger)

        print("self.properties = ", self.properties.category_to_cid)

        # get same id
        self.same_id = SameId(
            folder_path=folder_path, properties=self.properties, logger=logger
        )

        self.mask_reduced = torch.tensor(
            np.array(list(self.properties.tid_to_rm.values())), dtype=torch.int16
        )

        if self.logger is not None:
            self.logger.log_matrix(
                "mask_reduced.txt", self.mask_reduced.cpu().detach().numpy(), fmt="%d"
            )

        self.log_print("self.mask_reduced = ", self.mask_reduced)
        self.log_print("self.mask_reduced shape = ", self.mask_reduced.shape)

        if self.simulated:
            self.X_simulated = torch.rand(
                self.mask_reduced.shape[0], self.dim_clip_simulated
            )
            self.log_print(
                f"Simulated dataset: created dummy clip values of shape {self.X_simulated.shape}"
            )

        self.log_print(
            f"Initialised dataset with {self.properties.n_properties} properties and {self.mask_reduced.shape[0]} prompts."
        )

        if not self.simulated:
            assert len(self.folders) == self.mask_reduced.shape[0]

        # get dim_x
        vector_0, mask_0 = self.__getitem__(0)
        self.dim_x = int(vector_0.size()[0])
        self.log_print(f"dim_x = {self.dim_x}")

        # truncate_embds_topk
        self.truncate_embds()

        # normalization
        self.normalize = check_yaml_params(normalize, possible_values=[MAX_MIN])
        if self.normalize is not None:
            self.get_min_max_X()

        # run checks
        run_checks_dataset(
            properties=self.properties, same_id=self.same_id, logger=self.logger
        )

    def __len__(self):
        return self.mask_reduced.shape[0]

    def _get_item_real(self, idx):
        # concat streams in manifest order — this order is load-bearing:
        # inference reverses it via `stream_specs`.
        parts = []
        for spec in self.stream_specs:
            file_path = os.path.join(self.folder_embds, self.files_by_stream[spec.name][idx])
            with h5py.File(file_path, "r") as f:
                arr = f["vector"][:]
            parts.append(torch.from_numpy(arr).to(torch.float32).flatten())

        embds = torch.cat(parts) if len(parts) > 1 else parts[0]
        if self.pca_components is not None:
            embds = (embds - self.pca_mean) @ self.pca_components.T
        elif self.indices_truncate_embds_topk is not None:
            embds = embds[self.indices_truncate_embds_topk]

        if self.normalize is not None and self.normalize == MAX_MIN:
            embds = (embds - self.embds_min) / (self.embds_max - self.embds_min + 1e-6)

        return embds, self.mask_reduced[idx, :]

    def __getitem__(self, idx):
        if self.simulated:
            return self._get_item_simulated(idx)
        else:
            return self._get_item_real(idx)

    def _get_item_simulated(self, idx):
        embds = self.X_simulated[idx, :]
        if self.pca_components is not None:
            embds = (embds - self.pca_mean) @ self.pca_components.T
        elif self.indices_truncate_embds_topk is not None:
            embds = embds[self.indices_truncate_embds_topk]
        if self.normalize is not None and self.normalize == MAX_MIN:
            embds = (embds - self.embds_min) / (self.embds_max - self.embds_min + 1e-6)
        return embds, self.mask_reduced[idx]

    def _get_X_simulated(self):
        self.X = self.X_simulated
        return self.X

    def _get_X_real(self):
        self.X = torch.zeros((len(self), self.dim_x))
        for i in range(1, len(self)):
            y = self.__getitem__(i)[0][None, :]
            self.X[i] = y
        self.log_print(f"X stored in memory with a shape of {self.X.shape}")
        return self.X

    def get_X(self):
        normalize_saved = self.normalize
        self.normalize = None
        if self.simulated:
            X = self._get_X_simulated()
        else:
            X = self._get_X_real()
        self.normalize = normalize_saved
        return X

    def build_property_is_the_same(self):
        pass

    def truncate_embds(self):
        if self.truncate_embds_topk is None:
            return

        self.log_print(
            f"Truncating embeddings from {self.dim_x} to "
            f"{self.truncate_embds_topk} via method={self.truncate_embds_method}..."
        )

        if self.truncate_embds_method == TRUNCATE_PCA:
            self._truncate_embds_pca()
        else:
            self._truncate_embds_range()

        self.dim_x = self.truncate_embds_topk
        self.log_print(f"dim_x is now {self.truncate_embds_topk}.")

    def _truncate_embds_range(self):
        K = self.truncate_embds_topk
        file = f"indices_top_{K}.json"
        pca_file = f"pca_top_{K}.npz"
        if os.path.exists(os.path.join(self.folder_path, pca_file)):
            raise RuntimeError(
                f"Range truncation requested but {pca_file} exists at "
                f"{self.folder_path}. Refusing to mix cache formats. "
                f"Remove {pca_file} or switch truncate_embds_method to 'pca'."
            )
        if os.path.exists(os.path.join(self.folder_path, file)):
            self.log_print(f"Found {file}")
            with open(os.path.join(self.folder_path, file), "r") as f:
                self.indices_truncate_embds_topk = json.load(f)
        else:
            self.log_print(f"Did not find {file}. Re-calculating it...")
            self.indices_truncate_embds_topk = self.get_indices_truncate_embds_topk()
            self.log_print(f"Saving {file}.")
            with open(os.path.join(self.folder_path, file), "w") as f:
                json.dump(self.indices_truncate_embds_topk, f)

    def _truncate_embds_pca(self):
        K = self.truncate_embds_topk
        pca_file = f"pca_top_{K}.npz"
        range_file = f"indices_top_{K}.json"
        pca_path = os.path.join(self.folder_path, pca_file)
        if os.path.exists(os.path.join(self.folder_path, range_file)) and not os.path.exists(pca_path):
            raise RuntimeError(
                f"PCA truncation requested but only {range_file} is cached "
                f"at {self.folder_path}. Remove it (or move it aside) to "
                f"regenerate PCA caches, or switch truncate_embds_method to 'range'."
            )
        if os.path.exists(pca_path):
            self.log_print(f"Found {pca_file}")
            data = np.load(pca_path)
            self.pca_mean = torch.tensor(data["mean"], dtype=torch.float32)
            self.pca_components = torch.tensor(data["components"], dtype=torch.float32)
            self.pca_singular_values = torch.tensor(
                data["singular_values"], dtype=torch.float32
            )
            self.pca_explained_variance_ratio = torch.tensor(
                data["explained_variance_ratio"], dtype=torch.float32
            )
        else:
            self.log_print(f"Did not find {pca_file}. Computing PCA projection...")
            self._compute_pca_projection()
            np.savez(
                pca_path,
                mean=self.pca_mean.cpu().numpy(),
                components=self.pca_components.cpu().numpy(),
                singular_values=self.pca_singular_values.cpu().numpy(),
                explained_variance_ratio=self.pca_explained_variance_ratio.cpu().numpy(),
            )
            self.log_print(f"Saved {pca_file}.")

    def _compute_pca_projection(self):
        if self.X is None:
            _ = self.get_X()
        X = self.X.to(torch.float32)
        n = X.shape[0]
        K = self.truncate_embds_topk
        q = min(K, min(X.shape) - 1) if min(X.shape) > 1 else min(K, min(X.shape))
        q = max(1, q)

        mean = X.mean(dim=0)
        X_centered = X - mean
        U, S, V = torch.pca_lowrank(X_centered, q=q, niter=6)

        if q < K:
            pad_v = torch.zeros(V.shape[0], K - q, dtype=V.dtype)
            pad_s = torch.zeros(K - q, dtype=S.dtype)
            V = torch.cat([V, pad_v], dim=1)
            S = torch.cat([S, pad_s], dim=0)

        components = V[:, :K].T.contiguous()
        singular_values = S[:K].contiguous()

        total_var = (X_centered ** 2).sum() / max(n - 1, 1)
        component_var = singular_values ** 2 / max(n - 1, 1)
        explained_variance_ratio = component_var / (total_var + 1e-12)

        self.pca_mean = mean
        self.pca_components = components
        self.pca_singular_values = singular_values
        self.pca_explained_variance_ratio = explained_variance_ratio

    def get_indices_truncate_embds_topk(self):
        if self.X is None:
            _ = self.get_X()

        diff = torch.max(self.X, dim=0)[0] - torch.min(self.X, dim=0)[0]

        diff_argsort = diff.argsort(descending=True)

        self.log_print("top 50 diff_argsort = ", diff_argsort[:50])
        self.log_print("top 50 diff = ", diff[diff_argsort[:50]])

        self.log_print(
            "Highest diff (max - min) in new dataset : ", diff[diff_argsort[0]]
        )
        self.log_print(
            "Lowest diff (max - min) in new dataset : ",
            diff[diff_argsort[self.truncate_embds_topk]],
        )

        return diff_argsort[: self.truncate_embds_topk].tolist()

    def get_min_max_X(self):
        if self.normalize == MAX_MIN:
            self.log_print(f"Setting up {MAX_MIN} normalization")
            if self.truncate_embds_topk is None:
                file_max = "embds_max.json"
                file_min = "embds_min.json"
            elif self.truncate_embds_method == TRUNCATE_PCA:
                file_max = f"embds_max_pca_{self.truncate_embds_topk}.json"
                file_min = f"embds_min_pca_{self.truncate_embds_topk}.json"
            else:
                file_max = f"embds_max_top_{self.truncate_embds_topk}.json"
                file_min = f"embds_min_top_{self.truncate_embds_topk}.json"
            if os.path.exists(os.path.join(self.folder_path, file_max)):
                self.log_print(f"Found {file_max}")
                with open(os.path.join(self.folder_path, file_max), "r") as f:
                    self.embds_max = json.load(f)
                with open(os.path.join(self.folder_path, file_min), "r") as f:
                    self.embds_min = json.load(f)

            else:
                self.log_print(f"Did not find {file_max}. Re-calculating it...")
                self.embds_max, self.embds_min = self._get_min_max_X()

                for file, embds_extreme in zip(
                    [file_max, file_min], [self.embds_max, self.embds_min]
                ):
                    with open(os.path.join(self.folder_path, file), "w") as f:
                        json.dump(embds_extreme.tolist(), f)

            self.embds_max = torch.tensor(self.embds_max, dtype=torch.float32)
            self.embds_min = torch.tensor(self.embds_min, dtype=torch.float32)

    def _get_min_max_X(self):
        if self.X is None:
            _ = self.get_X()

        if self.pca_components is not None:
            X_proj = (self.X - self.pca_mean) @ self.pca_components.T
            return torch.max(X_proj, dim=0)[0], torch.min(X_proj, dim=0)[0]

        if self.indices_truncate_embds_topk is not None:
            return (
                torch.max(self.X[:, self.indices_truncate_embds_topk], dim=0)[0],
                torch.min(self.X[:, self.indices_truncate_embds_topk], dim=0)[0],
            )

        return torch.max(self.X, dim=0)[0], torch.min(self.X, dim=0)[0]

    def denormalize(self, vector):
        if self.normalize == MAX_MIN:
            return vector * (self.embds_max - self.embds_min) + self.embds_min
        else:
            raise Exception("Wrong normalization")
