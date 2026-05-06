"""Vision-language scoring via OpenAI Chat Completions (GPT-4o / GPT-4o-mini with images).

Requires ``OPENAI_API_KEY`` in the environment (same key as for ChatGPT API access).

The model returns structured JSON: how well the image matches the full prompt and each
attribute phrase (0–1 scores).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


def _encode_image_b64(path: Path | str) -> str:
    import base64

    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("ascii")


def _parse_json_loose(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"No JSON object in model output: {text[:500]}")
    return json.loads(m.group())


def openai_judge_image(
    image_path: Path | str,
    *,
    full_prompt: str,
    attribute_phrases: list[str],
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """
    Ask the model to score alignment of the image with the prompt and each attribute.

    Returns a dict with at least ``match_full_prompt`` (float 0–1) and
    ``attribute_scores`` (list of {phrase, score}).
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("Install openai: pip install openai>=1.40") from e

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError(
            "Set OPENAI_API_KEY in the environment to call the OpenAI API "
            "(API keys are managed at https://platform.openai.com/)."
        )

    client = OpenAI(api_key=key, timeout=timeout)
    b64 = _encode_image_b64(image_path)

    attr_lines = "\n".join(f"- {a!r}" for a in attribute_phrases)
    instructions = (
        "You evaluate a single image for text-to-image alignment.\n"
        "Return ONLY a JSON object with this exact schema:\n"
        "{\n"
        '  "match_full_prompt": <number from 0 to 1>,\n'
        '  "attribute_scores": [\n'
        '     {"phrase": <exact string from the list below>, "score": <0 to 1>},\n'
        "     ...\n"
        "  ],\n"
        '  "non_target_preserved": <0 to 1, how well unrelated aspects stay plausible>,\n'
        '  "notes": <short string>\n'
        "}\n\n"
        f"Full target prompt:\n{full_prompt!r}\n\n"
        f"Attribute phrases (score each):\n{attr_lines}\n"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instructions},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"},
                },
            ],
        }
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return _parse_json_loose(raw)
