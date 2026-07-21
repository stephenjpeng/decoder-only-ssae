import torch
import torch.nn as nn

from trainings.models.mlp import build_head


class Decoder(nn.Module):
    def __init__(
        self,
        n_prompts,
        n_features,
        n_repeat,
        dim_output,
        tid_same,
        n_features_situation=0,
        trainable_input_init_range=1,
        num_layers=1,
        hidden_dims=None,
        logger=None,
    ):
        super(Decoder, self).__init__()

        # Y : Y values with no mask, no same
        # Y_with_same : Y values with no mask, but with same (some rows are repeated until n_features_situation)
        # Y_with_same_and_mask: Y values with mask and same

        self.n_prompts = n_prompts
        self.n_features = n_features
        self.dim_output = dim_output
        self.n_repeat = n_repeat
        self.tid_same = tid_same
        self.n_features_situation = n_features_situation
        self.trainable_input_init_range = trainable_input_init_range
        self.logger = logger

        self.log_print = print if logger is None else logger.print

        self.Y = nn.Parameter(
            self.trainable_input_init_range
            * torch.rand(self.n_prompts, self.n_features, requires_grad=True)
        )
        self.y_same = nn.Parameter(
            self.trainable_input_init_range
            * torch.rand(
                1, self.n_features - self.n_features_situation, requires_grad=True
            )
        )

        if logger is not None:
            logger.log_matrix("Y_init_without_mask.txt", self.Y.detach().numpy())

        self.mask = None

        self.linear = build_head(
            in_features=self.n_features,
            out_features=self.dim_output,
            num_layers=num_layers,
            hidden_dims=hidden_dims,
            activation_factory=nn.ReLU,
        )
        self.activation = nn.ReLU()

        self.log_print(
            f"Initialised decoder with feature space of shape {self.Y.shape}"
        )

    def apply_mask(self, mask_reduced, batch_size):
        # apply the n_repeat to the mask reduced and save the mask
        self.log_print("Applying mask...")

        self.mask = torch.repeat_interleave(mask_reduced, self.n_repeat, dim=1)

        if self.mask.shape != self.Y.shape:
            raise Exception(
                f"mask applied of dim {self.mask.shape} different from feature space dim {self.latent.shape}"
            )

    def forward(self, batch_size, batch_idx):
        # in the forward, we start from Y and always re-add the same feature (Y_with_same)
        # the mask is applied inside the forward (Y_with_same_and_mask)

        if self.mask is None:
            raise Exception("No mask applyied to Y")

        self.Y_with_same_and_mask = self.Y.clone()
        self.Y_with_same_and_mask[self.tid_same, : -self.n_features_situation] = (
            self.y_same
        )
        self.Y_with_same = self.Y_with_same_and_mask.clone().detach()
        self.Y_with_same_and_mask.mul_(self.mask)  # [1500, 260]

        output = self.Y_with_same_and_mask[
            batch_idx * batch_size : (batch_idx + 1) * batch_size, :
        ]
        output = self.activation(output)
        output = self.linear(output)
        return output

    def get_rank_Y(self):
        if self.mask.shape[0] == 0:
            raise Exception("mask not applied")
        rank = torch.linalg.matrix_rank(self.Y)
        return rank

    def shuffle(self, randperm):
        with torch.no_grad():
            self.mask = self.mask[randperm]
            self.Y.data.copy_(self.Y[randperm])
