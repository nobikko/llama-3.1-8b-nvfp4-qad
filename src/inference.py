#!/usr/bin/env python3
"""
vLLM Inference Script for NVFP4 Quantized Models

This script demonstrates how to load and run inference on NVFP4 quantized models
using vLLM. It supports both single-GPU and multi-GPU inference.

Hardware Requirements:
- NVIDIA Blackwell GPU (SM100+) for optimal NVFP4 performance
- RTX 5090, B100, B200, etc.
- On older GPUs, vLLM will fall back to weight-only quantization

Note: If running on machines with SM < 100, vLLM will not run activation
quantization, only weight-only quantization.
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

import torch

# vLLM imports
try:
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import destroy_model_parallel

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not available. Install with: pip install vllm")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference"""

    model_path: str
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = -1
    repetition_penalty: float = 1.0
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 4096
    dtype: str = "auto"
    trust_remote_code: bool = False


class NVFP4Inference:
    """vLLM inference engine for NVFP4 quantized models"""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.llm = None
        self.sampling_params = None

        self._load_model()

    def _load_model(self):
        """Load the NVFP4 quantized model with vLLM"""

        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is required for inference. Install with: pip install vllm"
            )

        logger.info(f"Loading model from: {self.config.model_path}")
        logger.info(f"Tensor parallel size: {self.config.tensor_parallel_size}")
        logger.info(f"GPU memory utilization: {self.config.gpu_memory_utilization}")

        # Check GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU: {gpu_name}")

            # Check compute capability
            major, minor = torch.cuda.get_device_capability(0)
            compute_capability = major * 10 + minor
            logger.info(f"Compute capability: SM{major}{minor}")

            if compute_capability >= 100:
                logger.info(
                    "Blackwell GPU detected - Full NVFP4 acceleration available!"
                )
            else:
                logger.warning(
                    f"SM{major}{minor} detected. "
                    "Falling back to weight-only quantization (no activation quantization)."
                )
        else:
            raise RuntimeError("CUDA is required for inference")

        # Configure quantization
        # vLLM will automatically detect and use NVFP4 if the model is quantized
        quantization_config = None

        # Load model with vLLM
        try:
            self.llm = LLM(
                model=self.config.model_path,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=self.config.max_model_len,
                dtype=self.config.dtype,
                trust_remote_code=self.config.trust_remote_code,
                quantization=quantization_config,
                # Enable optimized CUDA graphs for better performance
                enforce_eager=False,
            )

            logger.info("Model loaded successfully!")

            # Log model info
            if hasattr(self.llm, "llm_engine"):
                model_config = self.llm.llm_engine.model_config
                logger.info(f"Model dtype: {model_config.dtype}")
                logger.info(f"Max model length: {model_config.max_model_len}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

        # Set up sampling parameters
        self.sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
            max_tokens=self.config.max_tokens,
        )

    def generate(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None,
        use_tqdm: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate completions for a list of prompts.

        Args:
            prompts: List of input prompts
            sampling_params: Optional custom sampling parameters
            use_tqdm: Whether to show progress bar

        Returns:
            List of generation results with text and metadata
        """
        if sampling_params is None:
            sampling_params = self.sampling_params

        logger.info(f"Generating completions for {len(prompts)} prompts...")

        start_time = time.time()
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=use_tqdm)
        end_time = time.time()

        results = []
        total_tokens = 0

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            tokens_generated = len(output.outputs[0].token_ids)
            total_tokens += tokens_generated

            results.append(
                {
                    "prompt": prompt,
                    "generated_text": generated_text,
                    "tokens_generated": tokens_generated,
                    "finish_reason": output.outputs[0].finish_reason,
                }
            )

        elapsed_time = end_time - start_time
        tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0

        logger.info(f"Generation complete!")
        logger.info(f"Total time: {elapsed_time:.2f}s")
        logger.info(f"Total tokens generated: {total_tokens}")
        logger.info(f"Tokens per second: {tokens_per_second:.2f}")

        return results

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        sampling_params: Optional[SamplingParams] = None,
    ) -> str:
        """
        Generate a chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: Optional system prompt
            sampling_params: Optional custom sampling parameters

        Returns:
            Generated response text
        """
        # Format messages using chat template if available
        tokenizer = self.llm.get_tokenizer()

        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Manual formatting
            prompt = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt += f"{role}: {content}\n"
            prompt += "assistant:"

        results = self.generate([prompt], sampling_params)
        return results[0]["generated_text"]

    def benchmark(
        self, prompts: List[str], num_iterations: int = 10, warmup_iterations: int = 3
    ) -> Dict[str, float]:
        """
        Benchmark inference performance.

        Args:
            prompts: List of prompts for benchmarking
            num_iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations

        Returns:
            Dictionary with benchmark metrics
        """
        logger.info(f"Starting benchmark with {len(prompts)} prompts...")
        logger.info(f"Warmup iterations: {warmup_iterations}")
        logger.info(f"Benchmark iterations: {num_iterations}")

        # Warmup
        logger.info("Warming up...")
        for _ in range(warmup_iterations):
            self.generate(prompts, use_tqdm=False)

        # Benchmark
        logger.info("Running benchmark...")
        times = []
        total_tokens = []

        for i in range(num_iterations):
            start_time = time.time()
            outputs = self.generate(prompts, use_tqdm=False)
            end_time = time.time()

            elapsed = end_time - start_time
            tokens = sum(o["tokens_generated"] for o in outputs)

            times.append(elapsed)
            total_tokens.append(tokens)

            logger.info(
                f"Iteration {i + 1}/{num_iterations}: {elapsed:.2f}s, {tokens} tokens"
            )

        # Calculate metrics
        avg_time = sum(times) / len(times)
        avg_tokens = sum(total_tokens) / len(total_tokens)
        tokens_per_second = avg_tokens / avg_time if avg_time > 0 else 0

        results = {
            "avg_time_seconds": avg_time,
            "avg_tokens": avg_tokens,
            "tokens_per_second": tokens_per_second,
            "min_time": min(times),
            "max_time": max(times),
            "throughput_prompts_per_second": len(prompts) / avg_time,
        }

        logger.info("\nBenchmark Results:")
        logger.info(f"Average time: {avg_time:.2f}s")
        logger.info(f"Average tokens: {avg_tokens:.1f}")
        logger.info(f"Tokens/second: {tokens_per_second:.2f}")
        logger.info(f"Prompts/second: {results['throughput_prompts_per_second']:.2f}")

        return results

    def cleanup(self):
        """Clean up resources"""
        if self.llm is not None:
            destroy_model_parallel()
            del self.llm
            self.llm = None
            torch.cuda.empty_cache()
            logger.info("Model resources cleaned up")


def interactive_chat(inference: NVFP4Inference, system_prompt: Optional[str] = None):
    """Run interactive chat session"""
    print("\n" + "=" * 80)
    print("Interactive Chat Mode")
    print("Type 'exit' or 'quit' to end the session")
    print("=" * 80 + "\n")

    if system_prompt:
        print(f"System: {system_prompt}\n")

    messages = []

    while True:
        try:
            user_input = input("User: ").strip()

            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            if not user_input:
                continue

            messages.append({"role": "user", "content": user_input})

            response = inference.chat(messages, system_prompt=system_prompt)

            print(f"Assistant: {response}\n")

            messages.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="vLLM Inference for NVFP4 Quantized Models"
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the quantized model"
    )
    parser.add_argument("--prompt", type=str, help="Single prompt for generation")
    parser.add_argument(
        "--prompts-file", type=str, help="File containing prompts (one per line)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=-1, help="Top-k sampling")
    parser.add_argument(
        "--repetition-penalty", type=float, default=1.0, help="Repetition penalty"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size (number of GPUs)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="GPU memory utilization (0.0-1.0)",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=4096, help="Maximum model context length"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive chat mode"
    )
    parser.add_argument("--system-prompt", type=str, help="System prompt for chat mode")
    parser.add_argument(
        "--benchmark", action="store_true", help="Run performance benchmark"
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="Trust remote code"
    )
    parser.add_argument(
        "--output", type=str, help="Output file for results (JSON format)"
    )

    args = parser.parse_args()

    # Create config
    config = InferenceConfig(
        model_path=args.model_path,
        max_tokens=args.max_max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
    )

    # Initialize inference
    inference = NVFP4Inference(config)

    try:
        if args.interactive:
            # Interactive mode
            interactive_chat(inference, args.system_prompt)

        elif args.benchmark:
            # Benchmark mode
            # Use default prompts if none provided
            if args.prompts_file:
                with open(args.prompts_file, "r") as f:
                    prompts = [line.strip() for line in f if line.strip()]
            else:
                prompts = [
                    "Explain quantum computing in simple terms.",
                    "Write a Python function to calculate fibonacci numbers.",
                    "What are the benefits of renewable energy?",
                    "Describe the process of photosynthesis.",
                ]

            results = inference.benchmark(prompts)

            if args.output:
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Benchmark results saved to {args.output}")

        elif args.prompt or args.prompts_file:
            # Single or batch generation
            if args.prompts_file:
                with open(args.prompts_file, "r") as f:
                    prompts = [line.strip() for line in f if line.strip()]
            else:
                prompts = [args.prompt]

            results = inference.generate(prompts)

            # Print results
            for i, result in enumerate(results):
                print(f"\n{'=' * 80}")
                print(f"Prompt {i + 1}:")
                print(result["prompt"])
                print(f"\nResponse:")
                print(result["generated_text"])
                print(f"\nTokens generated: {result['tokens_generated']}")

            if args.output:
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Results saved to {args.output}")

        else:
            print(
                "No action specified. Use --interactive, --prompt, --prompts-file, or --benchmark"
            )
            parser.print_help()

    finally:
        # Cleanup
        inference.cleanup()


if __name__ == "__main__":
    main()
