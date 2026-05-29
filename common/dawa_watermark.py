from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch.nn import functional as F


class Watermark:
    def __init__(
        self,
        device: torch.device,
        watermark_tokenizer,
        watermark_model,
        auxiliary_tokenizer,
        auxiliary_model,
        alpha: float = 0.2,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        max_new_tokens: int = 300,
        min_new_tokens: int = 200,
        key: int = 123,
        temperature: float = 1.0,
        start: int = 5,
        max_prompt_tokens: int = 512,
    ) -> None:
        self.device = device
        self.watermark_tokenizer = watermark_tokenizer
        self.watermark_model = watermark_model
        self.auxiliary_tokenizer = auxiliary_tokenizer
        self.auxiliary_model = auxiliary_model
        self.alpha = alpha
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.key = key
        self.temperature = temperature
        self.start = start
        self.max_prompt_tokens = max_prompt_tokens

    def _calc_banned_ngram_tokens(
        self,
        prev_input_ids: torch.Tensor,
        num_hypos: int,
        no_repeat_ngram_size: int,
        cur_len: int,
    ):
        if cur_len + 1 < no_repeat_ngram_size:
            return [[] for _ in range(num_hypos)]
        generated_ngrams = [{} for _ in range(num_hypos)]
        for idx in range(num_hypos):
            gen_tokens = prev_input_ids[idx].tolist()
            generated_ngram = generated_ngrams[idx]
            for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
                prev_ngram_tuple = tuple(ngram[:-1])
                generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

        def _get_generated_ngrams(hypo_idx: int):
            start_idx = cur_len + 1 - no_repeat_ngram_size
            ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
            return generated_ngrams[hypo_idx].get(ngram_idx, [])

        return [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]

    def _postprocess_next_token_scores(
        self,
        logits: torch.Tensor,
        batch_size: int,
        num_beams: int,
        prev_output_tokens: torch.Tensor,
    ) -> None:
        if self.repetition_penalty != 1.0:
            for i in range(batch_size * num_beams):
                for previous_token in set(prev_output_tokens[i].tolist()):
                    if logits[i, previous_token] < 0:
                        logits[i, previous_token] *= self.repetition_penalty
                    else:
                        logits[i, previous_token] /= self.repetition_penalty

        if prev_output_tokens.size(1) < self.min_new_tokens:
            logits[:, self.watermark_tokenizer.eos_token_id] = -float("inf")

        if self.no_repeat_ngram_size > 0:
            num_batch_hypotheses = batch_size * num_beams
            banned_batch_tokens = self._calc_banned_ngram_tokens(
                prev_output_tokens,
                num_batch_hypotheses,
                self.no_repeat_ngram_size,
                prev_output_tokens.size(1),
            )
            for i, banned_tokens in enumerate(banned_batch_tokens):
                logits[i, banned_tokens] = -float("inf")

    def _top_p(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits / self.temperature, dim=-1)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort >= self.top_p
        probs_sort[mask] = 0.0

        denom = probs_sort.sum(dim=-1, keepdim=True)
        invalid = denom <= 0
        if invalid.any():
            probs_sort[invalid.expand_as(probs_sort)] = 0.0
            probs_sort[invalid.squeeze(-1), 0] = 1.0
            denom = probs_sort.sum(dim=-1, keepdim=True)
        probs_sort = probs_sort / denom
        return torch.zeros_like(logits).scatter_(-1, probs_idx, probs_sort)

    def _stopping_criteria(self, output_ids: torch.Tensor) -> bool:
        return output_ids[0, -1].item() == self.watermark_tokenizer.eos_token_id

    def _get_zeta(self, probs: torch.Tensor):
        return list(range(len(probs[0]) + 1))

    def _generate_u(self, key: int, x: str, size: int) -> torch.Tensor:
        seed_input = f"{key}-{x}".encode("utf-8")
        seed = int(hashlib.sha256(seed_input).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)
        uniforms = torch.tensor(rng.random_sample(size), device=self.device, dtype=torch.float32)
        return uniforms.unsqueeze(0)

    def _gumbel_sampling(self, probs: torch.Tensor, uniforms: torch.Tensor) -> torch.Tensor:
        perturbed_logit = torch.log(uniforms) / probs
        next_id = torch.argmax(perturbed_logit, dim=-1)
        return next_id.unsqueeze(0)

    def _watermarking(
        self,
        probs: torch.Tensor,
        prev_id: str,
    ):
        probs_zeta = torch.minimum(probs, torch.tensor(self.alpha, device=self.device))
        zeta_tilde = torch.sum(torch.clamp(probs - self.alpha, min=0.0), dim=1, keepdim=True)
        probs_zeta = torch.cat((probs_zeta, zeta_tilde), dim=1)

        uniforms = self._generate_u(key=self.key, x=prev_id, size=len(probs_zeta[0]))
        zeta_i = self._gumbel_sampling(probs_zeta, uniforms)

        if zeta_i.item() != self._get_zeta(probs)[-1]:
            return zeta_i, zeta_i

        probs_tilde = torch.clamp(probs - self.alpha, min=0.0)
        denom = probs_tilde.sum()
        if denom <= 0:
            probs_tilde = probs
        else:
            probs_tilde = probs_tilde / denom
        sampled = torch.multinomial(probs_tilde, num_samples=1)
        return sampled, zeta_i

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        encoded = self.watermark_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_prompt_tokens,
        )
        return encoded["input_ids"].to(self.device)

    @torch.inference_mode()
    def generate_watermarked(self, prompt: str) -> str:
        input_ids = self._encode_prompt(prompt)
        output_ids = torch.empty((1, 0), dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids, device=self.device)
        past_key_values = None

        for t in range(self.max_new_tokens):
            if past_key_values is None:
                output = self.watermark_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            else:
                output = self.watermark_model(
                    input_ids=input_ids[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            logits = output.logits[:, -1, :]
            self._postprocess_next_token_scores(logits, batch_size=1, num_beams=1, prev_output_tokens=output_ids)
            probs = self._top_p(logits)

            if t >= self.start:
                prev_id = "-".join(output_ids[0][-self.start :].cpu().numpy().astype(str))
                next_id, _ = self._watermarking(probs, prev_id)
            else:
                next_id = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat((input_ids, next_id), dim=-1)
            output_ids = torch.cat((output_ids, next_id), dim=-1)
            past_key_values = output.past_key_values
            attention_mask = torch.cat((attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))), dim=-1)

            if output_ids.size(1) >= self.min_new_tokens and self._stopping_criteria(output_ids):
                break

        return self.watermark_tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

    @torch.inference_mode()
    def detection(self, text: str) -> float:
        watermark_ids = self.auxiliary_tokenizer.encode(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.device)

        if watermark_ids.size(1) <= self.start:
            return 0.0

        logits = self.auxiliary_model(watermark_ids).logits
        matched = 0
        evaluated = 0

        for t in range(self.start, watermark_ids.size(1)):
            logits_t = logits[:, t - 1, :]
            probs = self._top_p(logits_t)
            probs_zeta = torch.minimum(probs, torch.tensor(self.alpha, device=self.device))
            zeta_tilde = torch.sum(torch.clamp(probs - self.alpha, min=0.0), dim=1, keepdim=True)
            probs_zeta = torch.cat((probs_zeta, zeta_tilde), dim=1)

            prev_id = "-".join(map(str, watermark_ids[0, t - self.start : t].cpu().numpy()))
            uniforms = self._generate_u(key=self.key, x=prev_id, size=probs_zeta.shape[-1])
            zeta_i = self._gumbel_sampling(probs_zeta, uniforms)

            if zeta_i.item() != probs.shape[-1]:
                evaluated += 1
                if zeta_i.item() == watermark_ids[0, t].item():
                    matched += 1

        if evaluated == 0:
            return 0.0
        return matched / evaluated
