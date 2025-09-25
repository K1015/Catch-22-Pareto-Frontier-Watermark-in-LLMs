"""
Fixed Llama 2 7B with KGW Watermarking for LFQA Dataset
Handles JSON serialization issues with NumPy types
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
import json
import hashlib
import numpy as np
from typing import List, Dict, Optional, Tuple
import gc
from scipy.stats import norm
from tqdm import tqdm
import os

class WatermarkDetector:
    """Detector for KGW watermark based on green list statistics"""
    
    def __init__(self, 
                 vocab_size: int,
                 gamma: float = 0.5,
                 seeding_scheme: str = "simple_1"):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.seeding_scheme = seeding_scheme
    
    def _get_green_list_mask(self, input_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
        """Generate green list mask for given context"""
        prev_token = input_ids[-1].item() if len(input_ids) > 0 else 0
        
        hash_key = f"{prev_token}_{self.seeding_scheme}"
        hash_obj = hashlib.sha256(hash_key.encode())
        seed = int(hash_obj.hexdigest(), 16) % (2**32)
        
        rng = np.random.RandomState(seed)
        
        green_list_size = int(vocab_size * self.gamma)
        vocab_permutation = rng.permutation(vocab_size)
        green_list_mask = torch.zeros(vocab_size, dtype=torch.bool)
        green_list_mask[vocab_permutation[:green_list_size]] = True
        
        return green_list_mask
    
    def detect(self, 
               text: str, 
               tokenizer,
               return_stats: bool = True) -> Dict:
        """Detect watermark in text"""
        tokens = tokenizer.encode(text, return_tensors="pt")[0]
        
        if len(tokens) < 2:
            return {
                "is_watermarked": False,
                "confidence": 0.0,
                "z_score": 0.0,
                "p_value": 1.0,
                "green_fraction": 0.0,
                "num_tokens": len(tokens)
            }
        
        green_count = 0
        total_count = 0
        
        for i in range(1, len(tokens)):
            context = tokens[:i]
            curr_token = tokens[i]
            
            green_mask = self._get_green_list_mask(context, self.vocab_size)
            
            if green_mask[curr_token]:
                green_count += 1
            total_count += 1
        
        expected_green = total_count * self.gamma
        variance = total_count * self.gamma * (1 - self.gamma)
        
        if variance > 0:
            z_score = (green_count - expected_green) / np.sqrt(variance)
            p_value = 1 - norm.cdf(z_score)
        else:
            z_score = 0
            p_value = 1.0
        
        is_watermarked = z_score > 4.0
        
        return {
            "is_watermarked": bool(is_watermarked),  # Convert to Python bool
            "confidence": float(1 - p_value),
            "z_score": float(z_score),
            "p_value": float(p_value),
            "green_fraction": float(green_count / total_count if total_count > 0 else 0),
            "num_tokens": int(total_count),
            "green_count": int(green_count)
        }

class Llama2WatermarkedLFQA:
    """Modified Llama 2 7B model for LFQA dataset processing with watermarking"""
    
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-2-7b-hf",
                 load_in_8bit: bool = False,
                 load_in_4bit: bool = True,
                 gamma: float = 0.5,
                 delta: float = 2.0,
                 seeding_scheme: str = "simple_1"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit if not load_in_8bit else False
        
        self.gamma = gamma
        self.delta = delta
        self.seeding_scheme = seeding_scheme
        
        print("="*80)
        print("LLAMA 2 7B - LFQA DATASET WATERMARKING")
        print("="*80)
        print(f"Model: {model_name}")
        print(f"Watermark parameters: gamma={gamma}, delta={delta}")
        
        self._load_model()
        
        self.detector = WatermarkDetector(
            vocab_size=self.tokenizer.vocab_size,
            gamma=self.gamma,
            seeding_scheme=self.seeding_scheme
        )
    
    def _load_model(self):
        """Load model with configuration"""
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
    
    def _get_green_list_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate green list mask"""
        prev_token = input_ids[0, -1].item() if input_ids.shape[1] > 0 else 0
        
        hash_key = f"{prev_token}_{self.seeding_scheme}"
        hash_obj = hashlib.sha256(hash_key.encode())
        seed = int(hash_obj.hexdigest(), 16) % (2**32)
        
        rng = np.random.RandomState(seed)
        
        vocab_size = self.tokenizer.vocab_size
        green_list_size = int(vocab_size * self.gamma)
        
        vocab_permutation = rng.permutation(vocab_size)
        green_list_mask = torch.zeros(vocab_size, dtype=torch.bool, device=self.device)
        green_list_mask[vocab_permutation[:green_list_size]] = True
        
        return green_list_mask
    
    def _apply_watermark_to_logits(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply watermark to logits"""
        green_list_mask = self._get_green_list_mask(input_ids)
        watermarked_logits = logits.clone()
        watermarked_logits[:, green_list_mask] += self.delta
        return watermarked_logits
    
    def generate_watermarked(self,
                           prompt: str,
                           max_new_tokens: int = 300,
                           temperature: float = 0.7,
                           top_p: float = 0.9,
                           top_k: int = 50,
                           do_sample: bool = True) -> str:
        """Generate watermarked text"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        input_length = inputs.input_ids.shape[1]
        
        generated_ids = inputs.input_ids.clone()
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(generated_ids)
                logits = outputs.logits[:, -1, :]
            
            watermarked_logits = self._apply_watermark_to_logits(logits, generated_ids)
            
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
        """Process LFQA dataset with watermarking"""
        
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(
            output_dir,
            f"{self.model_name.replace('/', '-')}_strength_{self.delta}_"
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
                print("Could not read existing outputs, starting fresh...")
        
        outputs = []
        samples_processed = len(existing_outputs)
        
        print(f"\nProcessing {min(num_samples, len(data))} samples...")
        for idx in tqdm(range(samples_processed, min(num_samples, len(data))), desc="Generating"):
            sample = data[idx]
            
            prompt = sample.get("prefix", sample.get("question", sample.get("prompt", "")))
            
            gold_completion = ""
            if "gold_completion" in sample:
                gold_completion = sample["gold_completion"]
            elif "targets" in sample and sample["targets"]:
                gold_completion = sample["targets"][0] if isinstance(sample["targets"], list) else sample["targets"]
            
            try:
                generated_text = self.generate_watermarked(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
                
                detection = self.detector.detect(generated_text, self.tokenizer)
                
                # Ensure all values are JSON serializable
                output_entry = {
                    "prefix": str(prompt),
                    "gold_completion": str(gold_completion),
                    "gen_completion": [str(generated_text)],
                    "watermark_params": {
                        "gamma": float(self.gamma),
                        "delta": float(self.delta),
                        "method": "kgw_soft"
                    },
                    "detection_stats": {
                        "z_score": float(detection["z_score"]),
                        "p_value": float(detection["p_value"]),
                        "is_watermarked": bool(detection["is_watermarked"]),
                        "green_fraction": float(detection["green_fraction"]),
                        "num_tokens": int(detection["num_tokens"])
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
            {"prefix": "What is machine learning?", "gold_completion": ""},
            {"prefix": "Explain climate change.", "gold_completion": ""},
            {"prefix": "How does photosynthesis work?", "gold_completion": ""},
            {"prefix": "What is quantum computing?", "gold_completion": ""},
            {"prefix": "Describe the solar system.", "gold_completion": ""},
        ]
        
        with open(filepath, 'w') as f:
            for item in sample_data:
                f.write(json.dumps(item) + '\n')
        print(f"Created sample dataset at {filepath}")
    
    def _compute_summary_stats(self, output_file: str) -> Dict:
        """Compute summary statistics"""
        try:
            with open(output_file, 'r') as f:
                content = f.read().strip()
                results = [json.loads(line) for line in content.split('\n') if line]
            
            if not results:
                return {}
            
            z_scores = [r["detection_stats"]["z_score"] for r in results]
            detected = [r["detection_stats"]["is_watermarked"] for r in results]
            
            stats = {
                "total_samples": len(results),
                "detection_rate": float(sum(detected) / len(detected)) if detected else 0.0,
                "mean_z_score": float(np.mean(z_scores)),
                "std_z_score": float(np.std(z_scores)),
                "median_z_score": float(np.median(z_scores)),
                "min_z_score": float(min(z_scores)),
                "max_z_score": float(max(z_scores))
            }
            
            print("\n" + "="*50)
            print("SUMMARY STATISTICS")
            print("="*50)
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.3f}")
                else:
                    print(f"{key}: {value}")
            
            return stats
        except Exception as e:
            print(f"Error computing statistics: {e}")
            return {}

def main():
    """Main execution"""
    
    config = {
        "model_name": "meta-llama/Llama-2-7b-hf",
        "input_file": "./data/LFQA/inputs.jsonl",
        "output_dir": "./data/LFQA/",
        "num_samples": 5,
        "max_new_tokens": 300,
        "gamma": 0.5,
        "delta": 2.0,
        "load_in_4bit": True
    }
    
    processor = Llama2WatermarkedLFQA(
        model_name=config["model_name"],
        load_in_4bit=config["load_in_4bit"],
        gamma=config["gamma"],
        delta=config["delta"]
    )
    
    stats = processor.process_lfqa_dataset(
        input_file=config["input_file"],
        output_dir=config["output_dir"],
        num_samples=config["num_samples"],
        max_new_tokens=config["max_new_tokens"]
    )
    
    if stats:
        stats_file = os.path.join(config["output_dir"], "watermark_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nStatistics saved to {stats_file}")
    
    print("Processing complete!")

if __name__ == "__main__":
    main()