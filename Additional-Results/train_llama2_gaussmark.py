"""
Training script for Llama-2-7B with GaussMark watermarking.
FIXED VERSION - Correct key generation and gradient-based detection.

Usage:
    python train_llama2_gaussmark_fixed.py --mode generate
    python train_llama2_gaussmark_fixed.py --mode tune
    python train_llama2_gaussmark_fixed.py --mode evaluate
    python train_llama2_gaussmark_fixed.py --mode finetune
"""

import argparse
import json
import os
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from scipy import stats

from transformers import AutoModelForCausalLM, AutoTokenizer

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_peft_modules():
    """Lazy import for peft modules."""
    try:
        from peft import (
            LoraConfig,
            get_peft_model,
            prepare_model_for_kbit_training,
            TaskType,
        )
        return LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    except ImportError as e:
        logger.warning(f"PEFT not available: {e}")
        return None, None, None, None


def get_bnb_config():
    """Lazy import for BitsAndBytes config."""
    try:
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig
    except ImportError:
        return None


def get_trainer_modules():
    """Lazy import for Trainer and related modules."""
    from transformers import (
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    return TrainingArguments, Trainer, DataCollatorForLanguageModeling


def get_datasets_module():
    """Lazy import for datasets."""
    from datasets import load_dataset
    return load_dataset


@dataclass
class TrainingConfig:
    """Configuration for training Llama-2-7B with GaussMark."""

    # Model
    model_name: str = "meta-llama/Llama-2-7b-hf"
    use_auth_token: bool = True

    # Watermark parameters
    watermark_layer: int = 16
    watermark_component: str = "up_proj"
    sigma: float = 0.01  # FIXED: Default to larger value

    # Detection
    alpha: float = 0.05

    # Generation
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50

    # Training
    output_dir: str = "./llama2_gaussmark_output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_seq_length: int = 1024
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500

    # LoRA config
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Quantization (disabled by default for GaussMark)
    use_4bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False

    # FIXED: Sigma tuning with larger values
    sigma_search_values: List[float] = field(default_factory=lambda: [
        1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1
    ])
    num_tune_samples: int = 50

    # Misc
    seed: int = 42
    device: str = "auto"


class Llama2GaussMarkWatermarker:
    """
    GaussMark implementation for Llama-2-7B.
    FIXED: Correct key generation and gradient-based detection.
    """

    def __init__(self, config: TrainingConfig, enable_lora: bool = False):
        self.config = config
        self.set_seed(config.seed)
        self.lora_enabled = False

        # Load tokenizer
        logger.info(f"Loading tokenizer from {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            token=config.use_auth_token,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Setup quantization config
        quantization_config = None
        if config.use_4bit:
            BitsAndBytesConfig = get_bnb_config()
            if BitsAndBytesConfig is not None:
                logger.info("Setting up 4-bit quantization")
                compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_quant_type=config.bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=config.use_nested_quant,
                )
            else:
                logger.warning("BitsAndBytes not available, skipping 4-bit quantization")
        else:
            logger.info("Using full precision (fp16) - recommended for GaussMark")

        # Load model
        logger.info(f"Loading model from {config.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=quantization_config,
            device_map=config.device,
            trust_remote_code=True,
            token=config.use_auth_token,
            torch_dtype=torch.float16,
        )

        # Setup LoRA if requested
        if enable_lora and config.use_lora:
            self._setup_lora()

        # Get target weight
        self.target_weight = self._get_target_weight()
        self.weight_shape = self.target_weight.shape
        self.weight_dim = self.target_weight.numel()
        self.original_weight = None
        self.current_key = None
        self.is_quantized = False

        # Store original weights
        self._store_original_weight()

        if config.use_4bit and self.is_quantized:
            logger.warning(
                "="*60 + "\n"
                "WARNING: 4-bit quantization with GaussMark may not work correctly.\n"
                "For accurate results, run without --use_4bit.\n"
                + "="*60
            )

        logger.info(f"Watermark target: layer {config.watermark_layer}, {config.watermark_component}")
        logger.info(f"Target weight shape: {self.weight_shape}")
        logger.info(f"Target weight dimension: {self.weight_dim:,} elements")
        logger.info(f"Expected perturbation norm: ||δ|| ≈ σ√d = {config.sigma * np.sqrt(self.weight_dim):.4f}")

    def _setup_lora(self):
        """Setup LoRA for fine-tuning."""
        LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType = get_peft_modules()
        
        if LoraConfig is None:
            logger.warning("PEFT not available, skipping LoRA setup")
            return

        logger.info("Setting up LoRA")
        if self.config.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        self.lora_enabled = True

    def _get_target_weight(self) -> torch.nn.Parameter:
        """Get the target weight for watermarking."""
        layers = None
        
        if self.lora_enabled and hasattr(self.model, 'base_model'):
            try:
                layers = self.model.base_model.model.model.layers
                logger.info("Using LoRA model structure")
            except AttributeError:
                pass
        
        if layers is None:
            try:
                layers = self.model.model.layers
                logger.info("Using standard model structure")
            except AttributeError:
                pass
        
        if layers is None:
            for attr_path in ['model.layers', 'transformer.h', 'gpt_neox.layers']:
                try:
                    obj = self.model
                    for attr in attr_path.split('.'):
                        obj = getattr(obj, attr)
                    layers = obj
                    logger.info(f"Using fallback model structure: {attr_path}")
                    break
                except AttributeError:
                    continue
        
        if layers is None:
            raise RuntimeError(
                f"Could not find transformer layers in model. "
                f"Model type: {type(self.model).__name__}."
            )

        layer = layers[self.config.watermark_layer]

        component_map = {
            "up_proj": layer.mlp.up_proj,
            "down_proj": layer.mlp.down_proj,
            "gate_proj": layer.mlp.gate_proj,
        }

        module = component_map[self.config.watermark_component]

        if hasattr(module, 'base_layer'):
            return module.base_layer.weight
        if hasattr(module, 'weight'):
            return module.weight
        
        raise RuntimeError(f"Could not find weight in module: {type(module).__name__}")

    def _store_original_weight(self):
        """Store a copy of original weights."""
        if hasattr(self.target_weight, 'quant_state'):
            logger.warning("4-bit quantized weights detected. Consider using --no_4bit.")
            try:
                import bitsandbytes as bnb
                self.original_weight = bnb.functional.dequantize_4bit(
                    self.target_weight.data,
                    self.target_weight.quant_state
                ).clone().float()
                self.is_quantized = True
            except Exception as e:
                logger.error(f"Failed to dequantize weights: {e}")
                self.original_weight = self.target_weight.data.clone().float()
                self.is_quantized = False
        else:
            with torch.no_grad():
                self.original_weight = self.target_weight.data.clone().float()
            self.is_quantized = False

    def set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def generate_key(self, seed: Optional[int] = None) -> torch.Tensor:
        """
        Generate a random watermark key.
        
        FIXED: ξ ~ N(0, I) - NO normalization!
        Each element is independently ~ N(0, 1).
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        # DO NOT normalize! This is critical for GaussMark to work.
        key = torch.randn(self.weight_shape, dtype=torch.float32)
        return key

    def apply_watermark(self, key: torch.Tensor):
        """
        Apply watermark to model weights.
        
        θ(ξ) = θ + σξ where ξ ~ N(0, I)
        """
        if getattr(self, 'is_quantized', False):
            logger.warning("Applying watermark to quantized weights - results may be unreliable.")
        
        with torch.no_grad():
            delta = self.config.sigma * key.to(self.target_weight.device)
            if self.target_weight.dtype != torch.float32:
                delta = delta.to(self.target_weight.dtype)
            self.target_weight.data = self.original_weight.to(
                self.target_weight.device
            ).to(self.target_weight.dtype) + delta
        self.current_key = key

    def remove_watermark(self):
        """Remove watermark from model weights."""
        with torch.no_grad():
            self.target_weight.data = self.original_weight.to(
                self.target_weight.device
            ).to(self.target_weight.dtype)
        self.current_key = None

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        key: Optional[torch.Tensor] = None,
        key_seed: Optional[int] = None,
        **kwargs
    ) -> tuple:
        """Generate watermarked text."""
        if key is None:
            key = self.generate_key(key_seed)

        self.apply_watermark(key)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "top_k": kwargs.get("top_k", self.config.top_k),
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        outputs = self.model.generate(**inputs, **gen_kwargs)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        self.remove_watermark()

        return generated_text, key

    def compute_score(
        self,
        text: str,
        key: torch.Tensor,
        prompt: Optional[str] = None,
    ) -> float:
        """
        Compute watermark detection score using gradient correlation.
        
        FIXED: Uses gradient-based detection instead of hidden state comparison.
        
        ψ = ⟨ξ, ∇_θ log p_θ(y|x)⟩ / ||∇_θ log p_θ(y|x)||
        
        Under H0 (no watermark): ψ ~ N(0, 1)
        Under H1 (correct key): ψ has positive mean
        """
        if prompt:
            full_text = prompt + text
            prompt_tokens = self.tokenizer(prompt, return_tensors="pt")
            prompt_length = prompt_tokens.input_ids.shape[1]
        else:
            full_text = text
            prompt_length = 0
        
        inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = inputs.input_ids.to(self.model.device)
        
        # Create labels with prompt tokens masked
        labels = input_ids.clone()
        if prompt_length > 0:
            labels[:, :prompt_length] = -100
        
        # Enable gradient computation for target weight
        self.target_weight.requires_grad_(True)
        
        try:
            outputs = self.model(input_ids, labels=labels)
            loss = outputs.loss
            
            # Compute gradient of log-likelihood (negate loss since loss = -log p)
            (-loss).backward()
            
            grad = self.target_weight.grad
            
            if grad is None:
                logger.warning("Gradient is None - detection may fail")
                return 0.0
            
            # Flatten for inner product
            key_flat = key.flatten().float().to(self.model.device)
            grad_flat = grad.flatten().float()
            
            grad_norm = torch.norm(grad_flat)
            
            if grad_norm < 1e-10:
                return 0.0
            
            # Test statistic: ψ = ⟨ξ, ∇⟩ / ||∇||
            inner_product = torch.dot(key_flat, grad_flat)
            test_stat = inner_product / grad_norm
            
            return test_stat.item()
            
        finally:
            self.target_weight.requires_grad_(False)
            if self.target_weight.grad is not None:
                self.target_weight.grad.zero_()
            # Clear gradients from model
            self.model.zero_grad()

    def detect(
        self,
        text: str,
        key: torch.Tensor,
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Detect watermark in text."""
        score = self.compute_score(text, key, prompt)
        threshold = stats.norm.ppf(1 - self.config.alpha)
        p_value = 1 - stats.norm.cdf(score)

        return {
            "test_statistic": score,
            "threshold": threshold,
            "p_value": p_value,
            "is_watermarked": score > threshold,
            "alpha": self.config.alpha,
        }

    def quick_test(self, num_samples: int = 5) -> Dict[str, Any]:
        """Quick test to verify watermark is working."""
        print("\n" + "="*60)
        print("QUICK WATERMARK TEST")
        print("="*60)
        
        prompts = [
            "What is machine learning?",
            "Explain quantum computing.",
            "How does photosynthesis work?",
            "What causes climate change?",
            "Describe the solar system.",
        ][:num_samples]
        
        results = {"correct_key": [], "wrong_key": [], "baseline": []}
        
        for i, prompt in enumerate(prompts):
            print(f"\nSample {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            key = self.generate_key(seed=i+1000)
            wrong_key = self.generate_key(seed=i+50000)
            
            # Generate watermarked text
            generated, _ = self.generate(prompt, key=key, max_new_tokens=150)
            response = generated[len(prompt):] if generated.startswith(prompt) else generated
            
            # Generate baseline (unwatermarked)
            baseline, _ = self.generate(prompt, key=self.generate_key(seed=i+99999), max_new_tokens=150)
            baseline_response = baseline[len(prompt):] if baseline.startswith(prompt) else baseline
            
            # Actually generate without watermark for true baseline
            self.remove_watermark()
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=150, temperature=0.7, 
                    top_p=0.9, do_sample=True, pad_token_id=self.tokenizer.pad_token_id
                )
            baseline_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            baseline_response = baseline_text[len(prompt):] if baseline_text.startswith(prompt) else baseline_text
            
            # Detect
            if len(response.strip()) > 10:
                score_correct = self.compute_score(response, key, prompt)
                score_wrong = self.compute_score(response, wrong_key, prompt)
                score_baseline = self.compute_score(baseline_response, key, prompt)
                
                results["correct_key"].append(score_correct)
                results["wrong_key"].append(score_wrong)
                results["baseline"].append(score_baseline)
                
                threshold = stats.norm.ppf(1 - self.config.alpha)
                print(f"  Correct key z-score: {score_correct:.3f} ({'✓' if score_correct > threshold else '✗'})")
                print(f"  Wrong key z-score:   {score_wrong:.3f} ({'✓' if score_wrong > threshold else '✗'})")
                print(f"  Baseline z-score:    {score_baseline:.3f} ({'✓' if score_baseline > threshold else '✗'})")
        
        threshold = stats.norm.ppf(1 - self.config.alpha)
        print("\n" + "-"*60)
        print("SUMMARY:")
        if results["correct_key"]:
            print(f"  Correct key - Mean z-score: {np.mean(results['correct_key']):.3f}, TPR: {np.mean([z > threshold for z in results['correct_key']]):.1%}")
            print(f"  Wrong key   - Mean z-score: {np.mean(results['wrong_key']):.3f}, FPR: {np.mean([z > threshold for z in results['wrong_key']]):.1%}")
            print(f"  Baseline    - Mean z-score: {np.mean(results['baseline']):.3f}, FPR: {np.mean([z > threshold for z in results['baseline']]):.1%}")
        print("="*60)
        
        return results


class TextDataset(Dataset):
    """Simple text dataset for fine-tuning."""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 1024):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.encodings["input_ids"][idx].clone(),
        }


def tune_sigma(
    watermarker: Llama2GaussMarkWatermarker,
    prompts: List[str],
    sigma_values: List[float],
) -> Dict[str, Any]:
    """Tune sigma parameter via grid search."""
    results = {}
    original_sigma = watermarker.config.sigma

    for sigma in tqdm(sigma_values, desc="Tuning sigma"):
        watermarker.config.sigma = sigma
        logger.info(f"\nTesting sigma={sigma}, expected ||δ|| ≈ {sigma * np.sqrt(watermarker.weight_dim):.4f}")

        positive_scores = []
        negative_scores = []

        for prompt in tqdm(prompts[:10], desc=f"sigma={sigma}", leave=False):
            generated, key = watermarker.generate(prompt, key_seed=42, max_new_tokens=150)
            response = generated[len(prompt):] if generated.startswith(prompt) else generated

            if len(response.strip()) > 10:
                pos_score = watermarker.compute_score(response, key, prompt)
                positive_scores.append(pos_score)

                wrong_key = watermarker.generate_key(seed=999)
                neg_score = watermarker.compute_score(response, wrong_key, prompt)
                negative_scores.append(neg_score)

        if positive_scores and negative_scores:
            threshold = stats.norm.ppf(1 - watermarker.config.alpha)
            tpr = np.mean([s > threshold for s in positive_scores])
            fpr = np.mean([s > threshold for s in negative_scores])

            results[sigma] = {
                "tpr": tpr,
                "fpr": fpr,
                "mean_positive": np.mean(positive_scores),
                "mean_negative": np.mean(negative_scores),
                "std_positive": np.std(positive_scores),
            }

            logger.info(f"sigma={sigma}: TPR={tpr:.1%}, FPR={fpr:.1%}, Mean z={np.mean(positive_scores):.3f}")

    # Restore original sigma
    watermarker.config.sigma = original_sigma

    # Find best sigma (maximize TPR - FPR)
    if results:
        best_sigma = max(results.keys(), key=lambda s: results[s]["tpr"] - results[s]["fpr"])
    else:
        best_sigma = original_sigma

    print("\n" + "="*60)
    print("SIGMA TUNING RESULTS")
    print("="*60)
    print(f"{'Sigma':<12} {'TPR':<10} {'FPR':<10} {'Mean Z':<10}")
    print("-"*42)
    for sigma in sigma_values:
        if sigma in results:
            r = results[sigma]
            print(f"{sigma:<12.1e} {r['tpr']:<10.1%} {r['fpr']:<10.1%} {r['mean_positive']:<10.3f}")
    print("-"*42)
    print(f"Best sigma: {best_sigma}")
    print("="*60)

    return {
        "results": {str(k): v for k, v in results.items()},
        "best_sigma": best_sigma,
    }


def evaluate_watermark(
    watermarker: Llama2GaussMarkWatermarker,
    prompts: List[str],
) -> Dict[str, Any]:
    """Evaluate watermark detection performance."""
    from sklearn.metrics import roc_auc_score

    positive_stats = []
    negative_stats = []

    for i, prompt in enumerate(tqdm(prompts, desc="Evaluating")):
        key_seed = i + 1000
        generated, key = watermarker.generate(prompt, key_seed=key_seed, max_new_tokens=150)
        response = generated[len(prompt):] if generated.startswith(prompt) else generated

        if len(response.strip()) < 10:
            continue

        pos_score = watermarker.compute_score(response, key, prompt)
        positive_stats.append(pos_score)

        wrong_key = watermarker.generate_key(seed=key_seed + 50000)
        neg_score = watermarker.compute_score(response, wrong_key, prompt)
        negative_stats.append(neg_score)

    threshold = stats.norm.ppf(1 - watermarker.config.alpha)
    tpr = np.mean([s > threshold for s in positive_stats]) if positive_stats else 0
    fpr = np.mean([s > threshold for s in negative_stats]) if negative_stats else 0

    labels = [1] * len(positive_stats) + [0] * len(negative_stats)
    scores = positive_stats + negative_stats
    auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else 0.5

    results = {
        "tpr": tpr,
        "fpr": fpr,
        "auc": auc,
        "mean_positive_stat": np.mean(positive_stats) if positive_stats else 0,
        "std_positive_stat": np.std(positive_stats) if positive_stats else 0,
        "mean_negative_stat": np.mean(negative_stats) if negative_stats else 0,
        "num_samples": len(positive_stats),
        "threshold": threshold,
        "alpha": watermarker.config.alpha,
        "sigma": watermarker.config.sigma,
    }

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Sigma: {watermarker.config.sigma}")
    print(f"Samples: {len(positive_stats)}")
    print(f"\nDetection @ alpha={watermarker.config.alpha}:")
    print(f"  TPR (correct key): {tpr:.1%}")
    print(f"  FPR (wrong key):   {fpr:.1%}")
    print(f"  AUROC:             {auc:.4f}")
    print(f"\nZ-Score Statistics:")
    print(f"  Correct key - Mean: {results['mean_positive_stat']:.3f}, Std: {results['std_positive_stat']:.3f}")
    print(f"  Wrong key   - Mean: {results['mean_negative_stat']:.3f}")
    print("="*60)

    return results


def finetune_model(
    watermarker: Llama2GaussMarkWatermarker,
    train_texts: List[str],
    eval_texts: Optional[List[str]] = None,
):
    """Fine-tune the model using HuggingFace Trainer."""
    TrainingArguments, Trainer, DataCollatorForLanguageModeling = get_trainer_modules()
    
    config = watermarker.config

    if config.use_lora and not watermarker.lora_enabled:
        watermarker._setup_lora()
        watermarker.target_weight = watermarker._get_target_weight()
        watermarker._store_original_weight()

    train_dataset = TextDataset(train_texts, watermarker.tokenizer, config.max_seq_length)
    eval_dataset = TextDataset(eval_texts, watermarker.tokenizer, config.max_seq_length) if eval_texts else None

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=watermarker.tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps if eval_dataset else None,
        evaluation_strategy="steps" if eval_dataset else "no",
        save_total_limit=3,
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=watermarker.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()

    trainer.save_model(os.path.join(config.output_dir, "final"))
    watermarker.tokenizer.save_pretrained(os.path.join(config.output_dir, "final"))
    watermarker._store_original_weight()

    logger.info(f"Model saved to {config.output_dir}/final")

    return trainer


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate Llama-2-7B with GaussMark (FIXED)")
    parser.add_argument("--mode", type=str, default="test",
                       choices=["generate", "test", "tune", "finetune", "evaluate"],
                       help="Operation mode")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf",
                       help="Model name or path")
    parser.add_argument("--sigma", type=float, default=0.01,
                       help="Watermark noise standard deviation")
    parser.add_argument("--layer", type=int, default=16,
                       help="Layer to watermark")
    parser.add_argument("--output_dir", type=str, default="./gaussmark_output",
                       help="Output directory")
    parser.add_argument("--num_samples", type=int, default=50,
                       help="Number of samples for evaluation/tuning")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Prompt for generation mode")
    parser.add_argument("--use_4bit", action="store_true",
                       help="Use 4-bit quantization (not recommended)")
    parser.add_argument("--no_lora", action="store_true",
                       help="Disable LoRA")

    args = parser.parse_args()

    config = TrainingConfig(
        model_name=args.model,
        sigma=args.sigma,
        watermark_layer=args.layer,
        output_dir=args.output_dir,
        use_4bit=args.use_4bit,
        use_lora=not args.no_lora,
    )

    logger.info("Initializing watermarker...")
    enable_lora = (args.mode == "finetune") and config.use_lora
    watermarker = Llama2GaussMarkWatermarker(config, enable_lora=enable_lora)

    if args.mode == "test":
        # Quick test
        watermarker.quick_test(num_samples=5)

    elif args.mode == "generate":
        prompt = args.prompt or "Explain quantum computing in simple terms:"
        logger.info(f"\nGenerating watermarked text for prompt:\n{prompt}\n")

        generated, key = watermarker.generate(prompt, key_seed=42)

        print("\n" + "="*60)
        print("GENERATED TEXT:")
        print("="*60)
        print(generated)
        print("="*60)

        response = generated[len(prompt):] if generated.startswith(prompt) else generated
        result = watermarker.detect(response, key, prompt)

        print("\nWATERMARK DETECTION:")
        print(f"  Test statistic: {result['test_statistic']:.4f}")
        print(f"  P-value: {result['p_value']:.6f}")
        print(f"  Threshold (alpha={config.alpha}): {result['threshold']:.4f}")
        print(f"  Is watermarked: {result['is_watermarked']}")

    elif args.mode == "tune":
        prompts = [
            "Explain the concept of artificial intelligence:",
            "Write a short story about a robot:",
            "Describe how photosynthesis works:",
            "What are the main causes of climate change?",
            "Explain the theory of relativity:",
        ] * (args.num_samples // 5 + 1)
        prompts = prompts[:args.num_samples]

        results = tune_sigma(watermarker, prompts, config.sigma_search_values)

        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "sigma_tuning.json"), "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {args.output_dir}/sigma_tuning.json")

    elif args.mode == "evaluate":
        test_prompts = [
            "Explain machine learning:",
            "What is quantum computing?",
            "Describe the solar system:",
            "How does the internet work?",
            "What is blockchain technology?",
        ] * (args.num_samples // 5 + 1)
        test_prompts = test_prompts[:args.num_samples]

        results = evaluate_watermark(watermarker, test_prompts)

        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "evaluation.json"), "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {args.output_dir}/evaluation.json")

    elif args.mode == "finetune":
        logger.info("Loading training data...")
        load_dataset = get_datasets_module()

        try:
            dataset = load_dataset("tatsu-lab/alpaca", split="train[:1000]")
            train_texts = [
                f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"
                for ex in dataset
            ]
        except Exception as e:
            logger.warning(f"Could not load dataset: {e}")
            train_texts = ["This is a sample training text." * 50] * 100

        finetune_model(watermarker, train_texts)
        print(f"\nFine-tuning complete! Model saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
