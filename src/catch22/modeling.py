from __future__ import annotations

from dataclasses import dataclass

from .watermarks import HashWatermarkLogitsProcessor, apply_text_watermark, finalize_generated_text, local_completion


@dataclass
class GeneratedText:
    text: str
    backend: str


class TextGenerator:
    def __init__(
        self,
        model_name: str,
        *,
        load_in_4bit: bool = True,
        local_backend: bool = False,
        seed: int = 1234,
    ):
        self.model_name = model_name
        self.local_backend = local_backend
        self.seed = seed
        self.tokenizer = None
        self.model = None
        if not local_backend:
            self._load_model(load_in_4bit=load_in_4bit)

    def _load_model(self, *, load_in_4bit: bool) -> None:  # pragma: no cover - requires model access.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)

    def generate(
        self,
        prompt: str,
        *,
        method: str,
        sample_index: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> GeneratedText:
        if self.local_backend:
            text = apply_text_watermark(
                local_completion(prompt, method, sample_index),
                method,
                self.seed,
                local_backend=True,
            )
            return GeneratedText(text=text, backend="local")

        import torch
        from transformers import LogitsProcessorList

        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        processor = HashWatermarkLogitsProcessor(method, self.seed + sample_index, self.tokenizer.vocab_size)
        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                logits_processor=LogitsProcessorList([processor]),
                pad_token_id=self.tokenizer.pad_token_id,
            )
        prompt_len = encoded["input_ids"].shape[-1]
        text = self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
        text = finalize_generated_text(text, method, self.seed)
        return GeneratedText(text=text, backend="hf")
