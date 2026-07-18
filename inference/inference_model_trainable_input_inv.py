from pathlib import Path

import torch

from inference.abstract import SFDInference


class SFDInferenceModelTrainableInputInv(SFDInference):
    def __init__(self, folder_path: str | Path, device: str = "cpu"):
        super().__init__(folder_path=folder_path, device=device)

    def reset_model(self):
        self.decoder.load_state_dict(torch.load(Path(self.folder_path, "model.pt")))
        self.decoder.eval()
        self.decoder.initialize_mask(self.dataset.mask_reduced.to(self.device))

    @property
    def embeddings(self):
        return self.decoder.latent

    @torch.no_grad()
    def get_x(self, idx):
        return (self.decoder.latent_with_mask[idx : idx + 1] @ self.decoder.W)
