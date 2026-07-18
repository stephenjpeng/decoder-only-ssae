import torch

from inference.abstract import SFDInference


class SFDInferenceModelTrainableInputInv(SFDInference):
    @property
    def embeddings(self):
        return self.decoder.latent

    @torch.no_grad()
    def get_x(self, idx):
        return self.decoder.latent_with_mask[idx : idx + 1] @ self.decoder.W
