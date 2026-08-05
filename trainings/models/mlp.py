from typing import Iterable, List, Optional, Union

import torch
import torch.nn as nn


HiddenDimsSpec = Union[None, int, str, Iterable[int]]


def parse_hidden_dims(value: HiddenDimsSpec, num_layers: int) -> List[int]:
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")

    expected = num_layers - 1

    if expected == 0:
        if value not in (None, "", []):
            raise ValueError(
                f"hidden_dims must be empty when num_layers == 1, got {value!r}"
            )
        return []

    if value is None:
        raise ValueError(
            f"hidden_dims required when num_layers={num_layers} (expected "
            f"{expected} value(s))"
        )

    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        dims = [int(p) for p in parts]
    elif isinstance(value, int):
        dims = [value]
    else:
        dims = [int(v) for v in value]

    if len(dims) == 1 and expected > 1:
        dims = dims * expected

    if len(dims) != expected:
        raise ValueError(
            f"hidden_dims has {len(dims)} value(s) but num_layers={num_layers} "
            f"needs {expected}"
        )

    if any(d <= 0 for d in dims):
        raise ValueError(f"hidden_dims must be positive ints, got {dims}")

    return dims


def build_head(
    in_features: int,
    out_features: int,
    num_layers: int,
    hidden_dims: Optional[List[int]],
    activation_factory,
    *,
    n_blocks: Optional[int] = None,
) -> nn.Module:
    """Decoder head.

    Dense (default): a single MLP over all input features that cross-mixes property blocks.

    Block-diagonal (``n_blocks`` set): ``in_features`` is split into ``n_blocks`` equal
    chunks; each chunk gets an independent MLP of the same depth, and the outputs are
    summed. This ablates cross-property mixing while keeping the same per-block capacity
    and nonlinearity depth &mdash; used to test whether the L2 head's gain comes from
    within-block nonlinearity or from mixing across property blocks.
    """
    dims = parse_hidden_dims(hidden_dims, num_layers)

    if n_blocks is not None and n_blocks > 1:
        if in_features % n_blocks != 0:
            raise ValueError(
                f"block_diagonal head: in_features {in_features} not divisible by n_blocks {n_blocks}"
            )
        return BlockDiagonalHead(
            in_features=in_features,
            out_features=out_features,
            n_blocks=n_blocks,
            hidden_dims=dims,
            activation_factory=activation_factory,
        )

    if num_layers == 1:
        return nn.Linear(in_features, out_features, bias=True)

    layers: list[nn.Module] = []
    prev = in_features
    for h in dims:
        layers.append(nn.Linear(prev, h, bias=True))
        layers.append(activation_factory())
        prev = h
    layers.append(nn.Linear(prev, out_features, bias=True))
    return nn.Sequential(*layers)


class BlockDiagonalHead(nn.Module):
    """Independent per-block MLPs whose outputs are summed.

    Weight shape is equivalent to a dense head with a block-diagonal first layer:
    input dim ``in_features = n_blocks * block_size``, and each block has its own MLP
    ``block_size -> h1 -> h2 -> ... -> out_features``. This preserves within-block
    nonlinear capacity while forbidding cross-block mixing.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_blocks: int,
        hidden_dims: List[int],
        activation_factory,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_blocks = n_blocks
        self.block_size = in_features // n_blocks
        blocks: list[nn.Module] = []
        for _ in range(n_blocks):
            layers: list[nn.Module] = []
            prev = self.block_size
            for h in hidden_dims:
                layers.append(nn.Linear(prev, h, bias=True))
                layers.append(activation_factory())
                prev = h
            layers.append(nn.Linear(prev, out_features, bias=True))
            blocks.append(nn.Sequential(*layers))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features). Reshape to (..., n_blocks, block_size) and process each block.
        prefix_shape = x.shape[:-1]
        x = x.view(*prefix_shape, self.n_blocks, self.block_size)
        out = self.blocks[0](x[..., 0, :])
        for i in range(1, self.n_blocks):
            out = out + self.blocks[i](x[..., i, :])
        return out
