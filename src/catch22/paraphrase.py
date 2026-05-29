from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AttackOutput:
    text: str
    backend: str


def moderate_dipper_fallback(text: str) -> AttackOutput:
    replacements = {
        "response": "answer",
        "addresses": "discusses",
        "concise": "brief",
        "explanation": "account",
        "connects": "links",
        "main": "central",
        "causes": "drivers",
        "practical": "real-world",
        "implication": "consequence",
    }
    tokens = text.split()
    rewritten = [replacements.get(token.strip(".,").lower(), token) for token in tokens]
    return AttackOutput(" ".join(rewritten), "moderate_fallback")


def extreme_paraphrase_fallback(text: str) -> AttackOutput:
    sentences = [part.strip() for part in text.replace("?", ".").split(".") if part.strip()]
    if not sentences:
        return AttackOutput(text, "extreme_fallback")
    compressed = " ".join(sentences[:2])
    return AttackOutput(f"In short, {compressed.lower()}.", "extreme_fallback")


class Seq2SeqParaphraser:
    def __init__(self, model_name: str, *, local_backend: bool = False):
        self.model_name = model_name
        self.local_backend = local_backend
        self.tokenizer = None
        self.model = None
        if not local_backend:
            self._load()

    def _load(self) -> None:  # pragma: no cover - requires model download/runtime.
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

    def apply(self, text: str, *, strength: str) -> AttackOutput:
        if self.local_backend:
            return moderate_dipper_fallback(text) if strength == "moderate" else extreme_paraphrase_fallback(text)

        import torch

        if strength == "moderate":
            prompt = f"paraphrase: {text}"
            max_new_tokens = 320
        else:
            prompt = f"summarize: {text}"
            max_new_tokens = 180
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                temperature=0.9 if strength == "moderate" else 1.1,
                top_p=0.95,
            )
        return AttackOutput(self.tokenizer.decode(output[0], skip_special_tokens=True), self.model_name)
