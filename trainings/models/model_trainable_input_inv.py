import torch
import torch.nn as nn


@torch.no_grad()
def compute_W(X, Y, lambd, I):
    A = X.T @ X
    return torch.linalg.solve(A + lambd * I, X.T @ Y)


class Decoder(nn.Module):
    """Decoder variant that learns only the sparse feature matrix Y.

    W is not a trainable parameter -- it is refit each step via ridge
    regression against the full target embedding matrix. The training loop
    calls ``refit_W(X_full)`` at the start of each epoch to supply targets.
    """

    def __init__(
        self,
        n_prompts,
        n_properties,
        n_repeat,
        dim_output,
        tid_same=None,
        n_features_situation=0,
        lambd=0.0,
        logger=None,
    ):
        super().__init__()

        self.n_prompts = n_prompts
        self.n_properties = n_properties
        self.n_repeat = n_repeat
        self.n_features = n_properties * n_repeat
        self.dim_output = dim_output
        self.lambd = lambd
        self.logger = logger
        self.log_print = print if logger is None else logger.print

        self.register_buffer("I", torch.eye(self.n_features))
        self.latent = nn.Parameter(torch.rand(self.n_prompts, self.n_features))
        self.register_buffer("W", torch.zeros((self.n_features, self.dim_output)))

        self.activation = nn.ReLU()
        self.mask = None

    def apply_mask(self, mask_reduced, batch_size):
        self.mask = torch.repeat_interleave(mask_reduced, self.n_repeat, dim=1)

    @property
    def latent_with_mask(self):
        return self.activation(self.latent * self.mask)

    @torch.no_grad()
    def refit_W(self, X_full):
        """Refit the closed-form W using the current latent state and full X."""
        self.W = compute_W(self.latent_with_mask.detach(), X_full, self.lambd, self.I)

    def forward(self, batch_size, batch_idx):
        if self.mask is None:
            raise RuntimeError("apply_mask must be called before forward")

        start = batch_idx * batch_size
        end = start + batch_size
        return self.latent_with_mask[start:end] @ self.W

    def get_rank_Y(self):
        return torch.linalg.matrix_rank(self.latent)
