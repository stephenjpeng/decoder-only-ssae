from typing import Iterable, List, Optional, Union

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
) -> nn.Module:
    dims = parse_hidden_dims(hidden_dims, num_layers)

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
