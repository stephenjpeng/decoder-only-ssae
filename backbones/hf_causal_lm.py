"""Text-only decoder-only LM backbones (Hugging Face `transformers`).

Emits the padded last-hidden-state sequence as a single `seq` stream. There
is no `decode()` path — these backbones are for studying an LM's own
embedding space rather than driving a downstream image generator, so the
inherited no-op `decode()` from `Backbone` is what we want.

`HFCausalLMBackbone` is the generic entry point (requires `model_id` and
`hidden_size` up front so `stream_specs` is known before `load()`). Concrete
presets like `Gemma2_2bItBackbone` fill those in for a specific checkpoint;
they still accept a `model_id` override so callers can swap the checkpoint
without needing to know the hidden size.
"""

from __future__ import annotations

from typing import Any

import torch

from backbones.base import Backbone, StreamSpec, StreamTensors
from backbones.registry import register_backbone


_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@register_backbone("hf_causal_lm")
class HFCausalLMBackbone(Backbone):
    """Generic HF decoder-only LM. Emits padded last hidden states as `seq`.

    Padded positions are zeroed so the flat vector doesn't carry arbitrary
    junk from the model's response to pad tokens.
    """

    def __init__(
        self,
        model_id: str,
        hidden_size: int,
        max_length: int = 128,
        dtype: str = "bfloat16",
        device: str | torch.device = "cuda",
        trust_remote_code: bool = False,
        prompt_template: str = "{prompt}",
        apply_chat_template: bool = False,
        system_prompt: str | None = None,
    ) -> None:
        super().__init__(device=device)
        if dtype not in _DTYPE_MAP:
            raise ValueError(f"dtype must be one of {list(_DTYPE_MAP)}, got {dtype!r}")
        if "{prompt}" not in prompt_template:
            raise ValueError(
                f"prompt_template must contain '{{prompt}}'; got {prompt_template!r}"
            )
        self.model_id = model_id
        self.hidden_size = int(hidden_size)
        self.max_length = int(max_length)
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.prompt_template = prompt_template
        self.apply_chat_template = apply_chat_template
        self.system_prompt = system_prompt

        # stream_specs is a class attr on the ABC; setting it per instance is
        # fine because the shape depends on constructor args.
        self.stream_specs = [
            StreamSpec(
                name="seq",
                shape=(self.max_length, self.hidden_size),
                dtype="float16",
                h5_file="embds.h5",
            ),
        ]

        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        if self._loaded:
            return

        from transformers import AutoModel, AutoTokenizer

        torch_dtype = _DTYPE_MAP[self.dtype]
        # bfloat16/float16 need a CUDA device; fall back to float32 on CPU so
        # the smoke path still works.
        if self.device.type != "cuda" and torch_dtype != torch.float32:
            torch_dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=self.trust_remote_code,
        )
        self.model.to(self.device)
        self.model.eval()

        model_hidden = getattr(self.model.config, "hidden_size", None)
        if model_hidden is not None and model_hidden != self.hidden_size:
            raise ValueError(
                f"backbone configured with hidden_size={self.hidden_size} but "
                f"{self.model_id} has hidden_size={model_hidden}"
            )

        self._loaded = True

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().unload()

    def _format_prompt(self, prompt: str) -> str:
        """Apply the configured wrapping before tokenization.

        `prompt_template` is a simple `{prompt}` string substitution; when
        `apply_chat_template=True` we then run the tokenizer's chat template
        on top of the substituted text as a single user turn (with an
        optional system message), which is how IT/chat-tuned checkpoints
        like `gemma-2-2b-it` expect their input.
        """
        wrapped = self.prompt_template.format(prompt=prompt)
        if not self.apply_chat_template:
            return wrapped

        messages: list[dict[str, str]] = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": wrapped})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def encode(self, prompt: str) -> StreamTensors:
        if not self._loaded:
            self.load()

        text = self._format_prompt(prompt)
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=not self.apply_chat_template,
        ).to(self.device)

        outputs = self.model(**inputs, output_hidden_states=False)
        hidden = outputs.last_hidden_state  # [1, max_length, hidden_size]

        # zero out padded positions so they don't contribute noise downstream
        attn = inputs["attention_mask"].to(hidden.dtype).unsqueeze(-1)
        hidden = hidden * attn

        return {"seq": hidden.squeeze(0).to(torch.float32)}


@register_backbone("gemma_2_2b_it")
class Gemma2_2bItBackbone(HFCausalLMBackbone):
    """`google/gemma-2-2b-it` preset (hidden_size=2304).

    `model_id` is overridable so this preset can also target other
    Gemma-2-2B variants (e.g. the base `google/gemma-2-2b`) without needing
    a separate registration.
    """

    def __init__(
        self,
        model_id: str = "google/gemma-2-2b-it",
        max_length: int = 128,
        dtype: str = "bfloat16",
        device: str | torch.device = "cuda",
        prompt_template: str = "{prompt}",
        apply_chat_template: bool = True,
        system_prompt: str | None = None,
        **_: Any,
    ) -> None:
        super().__init__(
            model_id=model_id,
            hidden_size=2304,
            max_length=max_length,
            dtype=dtype,
            device=device,
            prompt_template=prompt_template,
            apply_chat_template=apply_chat_template,
            system_prompt=system_prompt,
        )
