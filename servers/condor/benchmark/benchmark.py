#!/usr/bin/env python
"""
Benchmarking script for Condor models.

This script runs performance and accuracy tests on different Condor model variants
to compare standard vs. Mixture of Experts architectures.
"""

import os
import sys
import time
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from model import (
    CondorConfig,
    CondorTokenizer, 
    CondorForCausalLM
)
from utils.logging_utils import setup_logger, LogPerformance

# Set up logging
logger = setup_logger(
    name="benchmark",
    log_level="INFO",
    console_output=True,
    file_output=True,
    json_format=False
)

# Define benchmarking datasets - simple samples for testing
SAMPLES = [
    {
        "id": "code_completion",
        "input": "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:",
        "expected_contains": ["return fibonacci(n-1) + fibonacci(n-2)", "return fibonacci(n - 1) + fibonacci(n - 2)"]
    },
    {
        "id": "math_reasoning",
        "input": "If a triangle has sides of length 3, 4, and 5, what is its area?",
        "expected_contains": ["6", "area = 6"]
    },
    {
        "id": "summarization",
        "input": "The patient is a 45-year-old male who presented to the emergency department with acute onset of severe chest pain radiating to the left arm. ECG showed ST-segment elevation in leads V1-V4. Cardiac enzymes were elevated. The patient was diagnosed with acute myocardial infarction and underwent emergency cardiac catheterization. Please summarize this case.",
        "expected_contains": ["heart attack", "myocardial infarction", "chest pain"]
    },
    {
        "id": "creative_writing",
        "input": "Write a short poem about artificial intelligence and creativity.",
        "expected_contains": []  # Subjective, no specific expected output
    }
]


class ModelBenchmark:
    """
    Benchmark class for evaluating Condor model variants.
    """
    
    def __init__(
        self, 
        model_size: str = "7b",
        use_moe: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 42
    ):
        """
        Initialize the benchmark with model configuration.
        
        Args:
            model_size: Size of the model ("7b" or "40b")
            use_moe: Whether to use Mixture of Experts
            device: Device to run the model on
            seed: Random seed for reproducibility
        """
        self.model_size = model_size
        self.use_moe = use_moe
        self.device = device
        self.seed = seed
        
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.model_name = f"condor-{model_size}{'-moe' if use_moe else ''}"
        logger.info(f"Initializing benchmark for {self.model_name}")
        
        # Initialize model and tokenizer
        self._init_model_and_tokenizer()
    
    def _init_model_and_tokenizer(self):
        """Initialize the model and tokenizer based on configuration."""
        logger.info(f"Loading tokenizer and model: {self.model_name}")
        
        # Select appropriate configuration
        if self.model_size == "7b":
            if self.use_moe:
                config = CondorConfig.condor_7b_moe_config()
            else:
                config = CondorConfig.condor_7b_config()
        else:  # 40b
            if self.use_moe:
                config = CondorConfig.condor_40b_moe_config()
            else:
                config = CondorConfig.condor_40b_config()
        
        # Load tokenizer
        self.tokenizer = CondorTokenizer.from_pretrained("../model/weights")
        
        # Initialize model
        self.model = CondorForCausalLM(config)
        
        # Move model to device
        if self.device == "cuda" and torch.cuda.is_available():
            dtype = torch.bfloat16 if hasattr(torch, 'bfloat16') else torch.float16
            self.model = self.model.to(self.device).to(dtype)
        else:
            self.model = self.model.to(self.device)
        
        logger.info(f"Model and tokenizer loaded successfully")
    
    def run_inference(
        self, 
        input_text: str, 
        max_new_tokens: int = 100, 
        temperature: float = 0.7,
        num_runs: int = 3
    ) -> Tuple[str, float, List[float]]:
        """
        Run inference on the model with the given input.
        
        Args:
            input_text: Input text to the model
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            num_runs: Number of runs for averaging performance
            
        Returns:
            Tuple containing:
                - Generated text
                - Average inference time
                - List of inference times for each run
        """
        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference multiple times to get average performance
        inference_times = []
        output_text = None
        
        for i in range(num_runs):
            with torch.no_grad():
                # Measure inference time
                with LogPerformance(logger, f"Inference run {i+1}") as perf:
                    outputs = self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"] if "attention_mask" in inputs else None,
                        max_length=inputs["input_ids"].size(1) + max_new_tokens,
                        temperature=temperature,
                        do_sample=(temperature > 0),
                        top_p=0.95,
                        top_k=50,
                        repetition_penalty=1.1,
                        num_return_sequences=1
                    )
                
                # Record inference time
                inference_times.append(perf.end_time - perf.start_time)
            
            # Only decode output on the first run
            if i == 0:
                output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Remove the input text to get only the generated part
                if output_text.startswith(input_text):
                    output_text = output_text[len(input_text):]
        
        # Calculate average inference time
        avg_inference_time = sum(inference_times) / len(inference_times)
        
        return output_text, avg_inference_time, inference_times
    
    def evaluate_accuracy(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the model's accuracy on the provided samples.
        
        Args:
            samples: List of sample inputs with expected outputs
            
        Returns:
            Dictionary with accuracy metrics
        """
        results = []
        
        for sample in samples:
            sample_id = sample["id"]
            input_text = sample["input"]
            expected_contains = sample.get("expected_contains", [])
            
            logger.info(f"Evaluating sample: {sample_id}")
            
            # Run inference
            output_text, avg_time, all_times = self.run_inference(
                input_text=input_text,
                max_new_tokens=150,
                temperature=0.1  # Low temperature for more deterministic outputs
            )
            
            # Check if output contains expected elements
            contains_expected = False
            if expected_contains:
                contains_expected = any(exp in output_text for exp in expected_contains)
            
            # Record results
            result = {
                "sample_id": sample_id,
                "input": input_text,
                "output": output_text,
                "avg_inference_time": avg_time,
                "inference_times": all_times,
                "tokens_per_second": len(self.tokenizer.encode(output_text)) / avg_time if avg_time > 0 else 0,
                "contains_expected": contains_expected if expected_contains else None
            }
            
            results.append(result)
        
        # Calculate overall metrics
        overall_accuracy = sum(1 for r in results if r["contains_expected"] is True) / sum(1 for r in results if r["contains_expected"] is not None)
        avg_tokens_per_second = sum(r["tokens_per_second"] for r in results) / len(results)
        avg_inference_time = sum(r["avg_inference_time"] for r in results) / len(results)
        
        metrics = {
            "model_name": self.model_name,
            "model_size": self.model_size,
            "use_moe": self.use_moe,
            "device": self.device,
            "overall_accuracy": overall_accuracy,
            "avg_tokens_per_second": avg_tokens_per_second,
            "avg_inference_time": avg_inference_time,
            "individual_results": results
        }
        
        return metrics
    
    def run_benchmark(self, samples: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run a full benchmark on the model.
        
        Args:
            samples: List of sample inputs with expected outputs
            
        Returns:
            Dictionary with benchmark results
        """
        if samples is None:
            samples = SAMPLES
        
        logger.info(f"Running benchmark for {self.model_name}")
        
        # Measure model memory usage
        memory_usage = None
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            # Run a small sample to ensure memory is allocated
            self.run_inference("Hello, world!", max_new_tokens=10, num_runs=1)
            memory_usage = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
        
        # Evaluate accuracy and performance
        accuracy_metrics = self.evaluate_accuracy(samples)
        
        # Combine all benchmark results
        benchmark_results = {
            **accuracy_metrics,
            "memory_usage_mb": memory_usage,
            "timestamp": time.time()
        }
        
        return benchmark_results


def compare_models(
    output_file: str = "benchmark_results.json",
    model_sizes: List[str] = ["7b"],
    run_baseline: bool = True,
    run_moe: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Compare different model variants and write results to a file.
    
    Args:
        output_file: File to write benchmark results to
        model_sizes: List of model sizes to benchmark
        run_baseline: Whether to run the baseline model
        run_moe: Whether to run the MoE model
        device: Device to run the models on
    """
    all_results = []
    
    for model_size in model_sizes:
        # Benchmark baseline model if requested
        if run_baseline:
            benchmark = ModelBenchmark(
                model_size=model_size,
                use_moe=False,
                device=device
            )
            baseline_results = benchmark.run_benchmark()
            all_results.append(baseline_results)
            
            logger.info(f"Baseline model ({model_size}) results:")
            logger.info(f"  Accuracy: {baseline_results['overall_accuracy']:.2f}")
            logger.info(f"  Tokens/sec: {baseline_results['avg_tokens_per_second']:.2f}")
            logger.info(f"  Memory usage: {baseline_results.get('memory_usage_mb', 'N/A')} MB")
        
        # Benchmark MoE model if requested
        if run_moe:
            benchmark = ModelBenchmark(
                model_size=model_size,
                use_moe=True,
                device=device
            )
            moe_results = benchmark.run_benchmark()
            all_results.append(moe_results)
            
            logger.info(f"MoE model ({model_size}) results:")
            logger.info(f"  Accuracy: {moe_results['overall_accuracy']:.2f}")
            logger.info(f"  Tokens/sec: {moe_results['avg_tokens_per_second']:.2f}")
            logger.info(f"  Memory usage: {moe_results.get('memory_usage_mb', 'N/A')} MB")
        
        # Compare models if both were benchmarked
        if run_baseline and run_moe:
            accuracy_diff = moe_results['overall_accuracy'] - baseline_results['overall_accuracy']
            speed_ratio = moe_results['avg_tokens_per_second'] / baseline_results['avg_tokens_per_second']
            
            logger.info(f"Comparison ({model_size}):")
            logger.info(f"  Accuracy difference: {accuracy_diff:.2f} ({'+' if accuracy_diff >= 0 else ''}{accuracy_diff*100:.1f}%)")
            logger.info(f"  Speed ratio: {speed_ratio:.2f}x")
    
    # Write results to file
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Benchmark results written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Condor model variants")
    parser.add_argument("--output", type=str, default="results/benchmark_results.json",
                        help="Output file for benchmark results")
    parser.add_argument("--model-sizes", type=str, nargs="+", default=["7b"],
                        choices=["7b", "40b"], help="Model sizes to benchmark")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip benchmarking the baseline model")
    parser.add_argument("--no-moe", action="store_true",
                        help="Skip benchmarking the MoE model")
    parser.add_argument("--cpu", action="store_true",
                        help="Force using CPU even if CUDA is available")
    
    args = parser.parse_args()
    
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    
    compare_models(
        output_file=args.output,
        model_sizes=args.model_sizes,
        run_baseline=not args.no_baseline,
        run_moe=not args.no_moe,
        device=device
    ) 