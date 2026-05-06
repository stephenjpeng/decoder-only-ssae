"""CLI for CLIP image–text or text–text similarity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.clip_metrics import clip_image_text_similarity, clip_text_text_similarity


def main() -> None:
    p = argparse.ArgumentParser(description="CLIP cosine similarity utilities.")
    sub = p.add_subparsers(dest="mode", required=True)

    pi = sub.add_parser("image_text", help="CLIP cosine between an image file and a caption.")
    pi.add_argument("--image", type=Path, required=True)
    pi.add_argument("--text", type=str, required=True)
    pi.add_argument("--device", type=str, default=None)

    pt = sub.add_parser("text_text", help="CLIP cosine between two strings.")
    pt.add_argument("--text_a", type=str, required=True)
    pt.add_argument("--text_b", type=str, required=True)
    pt.add_argument("--device", type=str, default=None)

    args = p.parse_args()
    if args.mode == "image_text":
        s = clip_image_text_similarity(args.image, args.text, device=args.device)
    else:
        s = clip_text_text_similarity(args.text_a, args.text_b, device=args.device)
    print(json.dumps({"similarity": s}, indent=2))


if __name__ == "__main__":
    main()
