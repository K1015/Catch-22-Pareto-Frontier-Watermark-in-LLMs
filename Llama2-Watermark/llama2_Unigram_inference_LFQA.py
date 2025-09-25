"""
Llama 2 7B with UNIGRAM-WATERMARK for LFQA Dataset Processing
Processes 500 samples from LFQA dataset with Unigram watermarking
Based on Zhao et al. ICLR 2024
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
import json
import hashlib
import secrets
import numpy as np
from typing import List, Dict, Optional, Set
import gc
from scipy.stats import norm
from collections import Counter
from tqdm import tqdm
import os

class UnigramWatermarkDetector:
    """Detector for UNIGRAM-WATERMARK with fixed green list"""
    
    def __init__(self, 
                 vocab_size: int,
                 gamma: float = 0.5,
                 watermark_key: Optional[str] = None):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.watermark_key = watermark_key or secrets.token_hex(16)
        self.green_list = self._generate_fixed_green_list()
    
    def _generate_fixed_green_list(self) -> Set[int]:
        """Generate a fixed green list using the watermark key"""
        hash_obj = hashlib.sha256(self.watermark_key.encode())
        seed = int(hash_obj.hexdigest(), 16) % (2**32)
        
        rng = np.random.RandomState(seed)
        green_list_size = int(self.vocab_size * self.gamma)
        vocab_permutation = rng.permutation(self.vocab_size)
        green_list = set(vocab_permutation[:green_list_size])
        
        return green_list
    
    def detect(self, 
               text: str, 
               tokenizer,
               use_unique: bool = False,
               return_stats: bool = True) -> Dict:
        """Detect watermark in text using UNIGRAM method"""
        tokens = tokenizer.encode(text, return_tensors="pt")[0].tolist()
        
        if len(tokens) < 2:
            return {
                "is_watermarked": False,
                "confidence": 0.0,
                "z_score": 0.0,
                "p_value": 1.0,
                "green_fraction": 0.0,
                "num_tokens": len(tokens)
            }
        
        if use_unique:
            tokens_to_check = list(dict.fromkeys(tokens))
        else:
            tokens_to_check = tokens
        
        green_count = sum(1 for token in tokens_to_check if token in self.green_list)
        n = len(tokens_to_check)
        
        expected_green = n * self.gamma
        variance = n * self.gamma * (1 - self.gamma)
        
        if variance > 0:
            z_score = (green_count - expected_green) / np.sqrt(variance)
            p_value = 1 - norm.cdf(z_score)
        else:
            z_score = 0
            p_value = 1.0
        
        # Match the threshold from the table (z-score ~7.9)
        is_watermarked = z_score > 4.0
        
        return {
            "is_watermarked": bool(is_watermarked),
            "confidence": float(1 - p_value),
            "z_score": float(z_score),
            "p_value": float(p_value),
            "green_fraction": float(green_count / n if n > 0 else 0),
            "num_tokens": int(n),
            "green_count": int(green_count)
        }

class Llama2UnigramLFQA:
    """Llama 2 model with UNIGRAM watermarking for LFQA dataset"""
    
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-2-7b-hf",
                 load_in_8bit: bool = False,
                 load_in_4bit: bool = True,
                 gamma: float = 0.5,
                 delta: float = 2.0,
                 watermark_key: Optional[str] = None):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit if not load_in_8bit else False
        
        self.gamma = gamma
        self.delta = delta
        self.watermark_key = watermark_key or secrets.token_hex(16)
        
        print("="*80)
        print("LLAMA 2 7B - UNIGRAM WATERMARK FOR LFQA DATASET")
        print("="*80)
        print(f"Model: {model_name}")
        print(f"Watermark parameters: gamma={gamma}, delta={delta}")
        print(f"Using fixed green list (Unigram method)")
        
        self._load_model()
        self._generate_fixed_green_list()
        
        self.detector = UnigramWatermarkDetector(
            vocab_size=self.tokenizer.vocab_size,
            gamma=self.gamma,
            watermark_key=self.watermark_key
        )
    
    def _load_model(self):
        """Load model and tokenizer"""
        print(f"\nLoading {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
        elif self.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
                bnb_8bit_use_double_quant=True,
                bnb_8bit_quant_type="nf4"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        
        print(f"Model loaded successfully!")
        if torch.cuda.is_available():
            print(f"VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    def _generate_fixed_green_list(self):
        """Generate fixed green list using watermark key"""
        hash_obj = hashlib.sha256(self.watermark_key.encode())
        seed = int(hash_obj.hexdigest(), 16) % (2**32)
        
        rng = np.random.RandomState(seed)
        vocab_size = self.tokenizer.vocab_size
        green_list_size = int(vocab_size * self.gamma)
        
        vocab_permutation = rng.permutation(vocab_size)
        self.green_list_mask = torch.zeros(vocab_size, dtype=torch.bool, device=self.device)
        self.green_list_mask[vocab_permutation[:green_list_size]] = True
        
        print(f"Generated fixed green list with {green_list_size} tokens ({self.gamma*100:.1f}% of vocabulary)")
    
    def _apply_unigram_watermark(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply UNIGRAM watermark to logits"""
        watermarked_logits = logits.clone()
        watermarked_logits[:, self.green_list_mask] += self.delta
        return watermarked_logits
    
    def generate_watermarked(self,
                           prompt: str,
                           max_new_tokens: int = 300,
                           temperature: float = 0.7,
                           top_p: float = 0.9,
                           top_k: int = 50,
                           do_sample: bool = True) -> str:
        """Generate watermarked text using UNIGRAM method"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        input_length = inputs.input_ids.shape[1]
        
        generated_ids = inputs.input_ids.clone()
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(generated_ids)
                logits = outputs.logits[:, -1, :]
            
            watermarked_logits = self._apply_unigram_watermark(logits)
            
            if temperature > 0:
                watermarked_logits = watermarked_logits / temperature
            
            probs = F.softmax(watermarked_logits, dim=-1)
            
            if do_sample and temperature > 0:
                if top_k > 0:
                    top_k_probs, top_k_indices = torch.topk(probs, min(top_k, probs.shape[-1]))
                    probs = torch.zeros_like(probs).scatter_(1, top_k_indices, top_k_probs)
                
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    
                    indices_to_remove = torch.zeros_like(probs, dtype=torch.bool)
                    indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                    probs[indices_to_remove] = 0
                
                probs = probs / probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(watermarked_logits, dim=-1, keepdim=True)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            if next_token.item() == self.tokenizer.eos_token_id or generated_ids.shape[1] >= 1024:
                break
        
        generated_text = self.tokenizer.decode(generated_ids[0][input_length:], skip_special_tokens=True)
        return generated_text
    
    def process_lfqa_dataset(self,
                            input_file: str,
                            output_dir: str,
                            num_samples: int = 500,
                            max_new_tokens: int = 300,
                            batch_save: int = 10):
        """Process LFQA dataset with UNIGRAM watermarking"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Match the naming convention from the original code
        output_file = os.path.join(
            output_dir,
            f"{self.model_name.replace('/', '-')}_unigram_strength_{self.delta}_"
            f"frac_{self.gamma}_len_{max_new_tokens}_num_{num_samples}.jsonl"
        )
        
        print(f"\nLoading dataset from {input_file}")
        if not os.path.exists(input_file):
            print(f"Error: Input file {input_file} not found!")
            print("Creating sample dataset...")
            self._create_sample_dataset(input_file)
        
        with open(input_file, 'r') as f:
            lines = f.read().strip().split('\n')
            data = [json.loads(line) for line in lines if line]
        
        existing_outputs = []
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        existing_outputs = [json.loads(line) for line in content.split('\n')]
                print(f"Found {len(existing_outputs)} existing outputs, resuming...")
            except:
                print("Starting fresh...")
        
        outputs = []
        samples_processed = len(existing_outputs)
        
        print(f"\nProcessing {min(num_samples, len(data))} samples with UNIGRAM watermarking...")
        for idx in tqdm(range(samples_processed, min(num_samples, len(data))), desc="Generating"):
            sample = data[idx]
            
            # Extract prompt and reference
            prompt = sample.get("prefix", sample.get("question", sample.get("prompt", "")))
            
            gold_completion = ""
            if "gold_completion" in sample:
                gold_completion = sample["gold_completion"]
            elif "targets" in sample and sample["targets"]:
                gold_completion = sample["targets"][0] if isinstance(sample["targets"], list) else sample["targets"]
            
            try:
                # Generate watermarked text
                generated_text = self.generate_watermarked(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
                
                # Detect watermark (both standard and unique methods)
                detection_standard = self.detector.detect(generated_text, self.tokenizer, use_unique=False)
                detection_unique = self.detector.detect(generated_text, self.tokenizer, use_unique=True)
                
                # Create output entry matching the format
                output_entry = {
                    "prefix": str(prompt),
                    "gold_completion": str(gold_completion),
                    "gen_completion": [str(generated_text)],
                    "watermark_params": {
                        "gamma": float(self.gamma),
                        "delta": float(self.delta),
                        "method": "unigram_biased"
                    },
                    "detection_stats": {
                        "z_score": float(detection_standard["z_score"]),
                        "p_value": float(detection_standard["p_value"]),
                        "is_watermarked": bool(detection_standard["is_watermarked"]),
                        "green_fraction": float(detection_standard["green_fraction"]),
                        "num_tokens": int(detection_standard["num_tokens"]),
                        # Additional Unigram-specific stats
                        "z_score_unique": float(detection_unique["z_score"]),
                        "is_watermarked_unique": bool(detection_unique["is_watermarked"])
                    }
                }
                
                outputs.append(json.dumps(output_entry))
                
                if len(outputs) >= batch_save:
                    with open(output_file, 'a') as f:
                        f.write('\n'.join(outputs) + '\n')
                    outputs = []
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
            except Exception as e:
                print(f"\nError processing sample {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if outputs:
            with open(output_file, 'a') as f:
                f.write('\n'.join(outputs) + '\n')
        
        print(f"\nCompleted! Output saved to: {output_file}")
        
        return self._compute_summary_stats(output_file)
    
    def _create_sample_dataset(self, filepath):
        """Create a sample dataset for testing"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        sample_data = [
            {"prefix": "What is machine learning and how does it work?", "gold_completion": ""},
            {"prefix": "Explain the causes of climate change.", "gold_completion": ""},
            {"prefix": "How does the immune system work?", "gold_completion": ""},
            {"prefix": "What are the principles of quantum computing?", "gold_completion": ""},
            {"prefix": "Describe the process of photosynthesis.", "gold_completion": ""},
        ]
        
        with open(filepath, 'w') as f:
            for item in sample_data:
                f.write(json.dumps(item) + '\n')
        print(f"Created sample dataset at {filepath}")
    
    def _compute_summary_stats(self, output_file: str) -> Dict:
        """Compute summary statistics matching the paper's metrics"""
        try:
            with open(output_file, 'r') as f:
                content = f.read().strip()
                results = [json.loads(line) for line in content.split('\n') if line]
            
            if not results:
                return {}
            
            z_scores = [r["detection_stats"]["z_score"] for r in results]
            z_scores_unique = [r["detection_stats"]["z_score_unique"] for r in results]
            detected = [r["detection_stats"]["is_watermarked"] for r in results]
            detected_unique = [r["detection_stats"]["is_watermarked_unique"] for r in results]
            
            stats = {
                "total_samples": len(results),
                "detection_rate_standard": float(sum(detected) / len(detected)) if detected else 0.0,
                "detection_rate_unique": float(sum(detected_unique) / len(detected_unique)) if detected_unique else 0.0,
                "mean_z_score": float(np.mean(z_scores)),
                "std_z_score": float(np.std(z_scores)),
                "median_z_score": float(np.median(z_scores)),
                "mean_z_score_unique": float(np.mean(z_scores_unique)),
                "expected_metrics": {
                    "AUROC": "1.000 (based on paper)",
                    "TPR@1%": "1.000 (based on paper)",
                    "F1@1%": "0.995 (based on paper)",
                    "p_score": "0.68 (based on paper)",
                    "z_score_expected": "7.9 (based on paper)"
                }
            }
            
            print("\n" + "="*50)
            print("UNIGRAM WATERMARK - SUMMARY STATISTICS")
            print("="*50)
            for key, value in stats.items():
                if key == "expected_metrics":
                    print("\nExpected metrics from paper:")
                    for k, v in value.items():
                        print(f"  {k}: {v}")
                elif isinstance(value, float):
                    print(f"{key}: {value:.3f}")
                else:
                    print(f"{key}: {value}")
            
            return stats
        except Exception as e:
            print(f"Error computing statistics: {e}")
            return {}

def main():
    """Main execution for LFQA dataset processing with UNIGRAM watermark"""
    
    config = {
        "model_name": "meta-llama/Llama-2-7b-hf",
        "input_file": "./data/LFQA/inputs.jsonl",
        "output_dir": "./data/LFQA/",
        "num_samples": 5,
        "max_new_tokens": 300,
        "gamma": 0.5,
        "delta": 2.0,  # Matching the paper's configuration
        "load_in_4bit": True
    }
    
    # Initialize model with UNIGRAM watermarking
    processor = Llama2UnigramLFQA(
        model_name=config["model_name"],
        load_in_4bit=config["load_in_4bit"],
        gamma=config["gamma"],
        delta=config["delta"]
    )
    
    # Process dataset
    stats = processor.process_lfqa_dataset(
        input_file=config["input_file"],
        output_dir=config["output_dir"],
        num_samples=config["num_samples"],
        max_new_tokens=config["max_new_tokens"]
    )
    
    if stats:
        stats_file = os.path.join(config["output_dir"], "unigram_watermark_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nStatistics saved to {stats_file}")
    
    print("\nUNIGRAM watermarking processing complete!")
    print("Expected performance based on paper: AUROC=1.0, TPR@1%=1.0, z-score~7.9")

if __name__ == "__main__":
    main()