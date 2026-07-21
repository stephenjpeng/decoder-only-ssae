import torch
import torch.nn as nn

from trainings.models.mlp import build_head


class Decoder(nn.Module):
    def __init__(
        self,
        n_prompts,
        n_properties,
        n_repeat,
        dim_output,
        trainable_input_init_range=1,
        num_layers=1,
        hidden_dims=None,
        logger=None,
        using_blocs=False,
    ):
        super(Decoder, self).__init__()

        self.n_prompts = n_prompts
        self.n_properties = n_properties
        self.n_repeat = n_repeat
        self.n_features = n_properties * n_repeat
        self.dim_output = dim_output
        self.trainable_input_init_range = trainable_input_init_range
        self.logger = logger
        self.using_blocs = using_blocs

        self.log_print = print if logger is None else logger.print

        if self.using_blocs:
            pass
        else:
            self.Y = nn.Embedding(
                num_embeddings=n_properties + 1, embedding_dim=n_repeat, padding_idx=0
            )

        self.mask = None

        activation_factory = lambda: nn.LeakyReLU(negative_slope=0.2)
        self.linear = build_head(
            in_features=self.n_features,
            out_features=self.dim_output,
            num_layers=num_layers,
            hidden_dims=hidden_dims,
            activation_factory=activation_factory,
        )
        self.activation = activation_factory()

        self.log_print("Decoder initialized")

    def forward(self, batch_size, batch_idx):
        if self.using_blocs:
            return self.blocs_multiplication(batch_size, batch_idx)
        else:
            return self.simple_multiplication(batch_size, batch_idx)

    def blocs_multiplication(self, batch_size, batch_idx):
        raise Exception("Not implemented...")

    def simple_multiplication(self, batch_size, batch_idx):
        embds = self.Y(
            self.mask_arange[batch_size * batch_idx : batch_size * (batch_idx + 1), :]
        )
        embds = self.activation(embds)
        embds = embds.reshape(embds.shape[0], -1)
        return self.linear(embds)

    def apply_mask(self, mask_reduced, batch_size):
        self.mask_reduced = mask_reduced

        self.mask_arange = torch.arange(
            1, self.n_properties + 1, device=mask_reduced.device
        )
        self.mask_arange = (
            self.mask_arange.repeat(self.n_prompts, 1) * self.mask_reduced
        )

    def get_rank_Y(self):
        pass


if __name__ == "__main__":
    mask_reduced = torch.tensor([[0, 1], [1, 0], [1, 1]])

    decoder = Decoder(
        n_prompts=3, n_properties=2, n_repeat=5, dim_output=20, using_blocs=True
    )

    print("mask_reduced = ", mask_reduced)

    output = decoder(mask_reduced)

    print("output = ", output)
