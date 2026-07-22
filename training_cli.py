import argparse

from trainable_inputs_all_clips import training


def _parse_bool_cli(value: str):
    if value is None:
        return None
    v = value.strip().lower()
    if v in ("true", "1", "yes", "y"):
        return True
    if v in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected True/False, got {value!r}")


def _parse_hidden_dims_cli(value: str):
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError(f"invalid hidden_dims: {value!r}")
    try:
        dims = [int(p) for p in parts]
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            f"hidden_dims must be int or comma-separated ints, got {value!r}"
        ) from err
    if any(d <= 0 for d in dims):
        raise argparse.ArgumentTypeError(
            f"hidden_dims must be positive, got {dims}"
        )
    return dims[0] if len(dims) == 1 else dims


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str, help="output folder")
    parser.add_argument(
        "--path_yaml",
        type=str,
        help="path to the config file (yaml)",
        default="trainings/config/params_default.yaml",
    )
    parser.add_argument(
        "--overwrite_output",
        type=bool,
        help="to overwrite the output if already exists",
        default=True,
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=None,
        help="Number of Linear layers in the decoder head (>=1). "
        "Overrides YAML when set.",
    )
    parser.add_argument(
        "--hidden_dims",
        type=_parse_hidden_dims_cli,
        default=None,
        help="Hidden layer widths as an int or comma-separated ints. "
        "Length must be num_layers-1 (or a single int to broadcast). "
        "Overrides YAML when set.",
    )
    parser.add_argument(
        "--pca_rotation",
        type=_parse_bool_cli,
        default=None,
        help="If True, rotate embeddings into a top-k PCA basis before "
        "truncation. Overrides YAML when set.",
    )
    args = parser.parse_args()

    training(
        output_folder=args.output_folder,
        path_yaml=args.path_yaml,
        overwrite_output=args.overwrite_output,
        num_layers=args.num_layers,
        hidden_dims=args.hidden_dims,
        pca_rotation=args.pca_rotation,
    )
