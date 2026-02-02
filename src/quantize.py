#!/usr/bin/env python3
"""
NVFP4 Quantization Script using LLM Compressor

This script applies NVFP4 quantization to a trained model.
Should be run AFTER QAD training to convert the model to NVFP4 format.

Hardware Requirements:
- NVIDIA Blackwell GPU (SM100+) for full NVFP4 support
- RTX 5090, B100, B200, etc.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# LLM Compressor imports
try:
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    LLMCOMPRESSOR_AVAILABLE = True
except ImportError:
    LLMCOMPRESSOR_AVAILABLE = False
    print(
        "Warning: llm-compressor not available. Install with: pip install llmcompressor"
    )

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def quantize_to_nvfp4(
    model_path: str,
    output_dir: str,
    calibration_dataset: str = "HuggingFaceH4/ultrachat_200k",
    num_calibration_samples: int = 512,
    max_seq_length: int = 2048,
    trust_remote_code: bool = False,
):
    """
    Apply NVFP4 quantization to a model.

    Args:
        model_path: Path to the model to quantize (can be HuggingFace model ID or local path)
        output_dir: Directory to save the quantized model
        calibration_dataset: Dataset for calibration
        num_calibration_samples: Number of samples for calibration
        max_seq_length: Maximum sequence length
        trust_remote_code: Whether to trust remote code
    """

    if not LLMCOMPRESSOR_AVAILABLE:
        raise ImportError(
            "llm-compressor is required for quantization. "
            "Install with: pip install llmcompressor"
        )

    logger.info(f"Loading model from {model_path}")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        trust_remote_code=trust_remote_code,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Model loaded successfully")
    logger.info(
        f"Model device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'single device'}"
    )

    # Prepare calibration dataset
    logger.info(f"Loading calibration dataset: {calibration_dataset}")

    from datasets import load_dataset

    # Load dataset
    ds = load_dataset(
        calibration_dataset, split=f"train_sft[:{num_calibration_samples}]"
    )

    def format_conversation(example):
        """Format conversation into text"""
        messages = example.get("messages", [])
        if not messages:
            return {"text": ""}

        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            except:
                text = "\n".join(
                    [
                        f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                        for msg in messages
                    ]
                )
        else:
            text = "\n".join(
                [
                    f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                    for msg in messages
                ]
            )

        return {"text": text}

    def tokenize_function(examples):
        """Tokenize texts"""
        texts = []
        for messages in examples.get("messages", []):
            if isinstance(messages, list):
                formatted = format_conversation({"messages": messages})["text"]
                texts.append(formatted)
            else:
                texts.append(str(messages))

        return tokenizer(
            texts, truncation=True, max_length=max_seq_length, padding=False
        )

    # Process dataset
    ds = ds.map(tokenize_function, remove_columns=ds.column_names, batched=True)

    logger.info(f"Calibration dataset prepared: {len(ds)} samples")

    # Configure NVFP4 quantization
    logger.info("Configuring NVFP4 quantization...")

    # NVFP4 quantization scheme
    # - Weights: 4-bit float, per-group (size 16) scales
    # - Activations: 4-bit float, per-tensor scales
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="NVFP4",
        ignore=["lm_head"],  # Don't quantize the final layer
    )

    logger.info("Starting quantization with oneshot...")
    logger.info(
        "This may take several minutes depending on the model size and calibration samples."
    )

    # Apply quantization
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=max_seq_length,
        num_calibration_samples=num_calibration_samples,
    )

    # Save quantized model
    logger.info(f"Saving quantized model to {output_dir}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model.save_pretrained(
        output_dir,
        save_compressed=True,  # Save in compressed format
    )
    tokenizer.save_pretrained(output_dir)

    logger.info("Quantization complete!")
    logger.info(f"Quantized model saved to: {output_dir}")

    # Print model info
    logger.info("\nModel Information:")
    logger.info(f"- Original model: {model_path}")
    logger.info(f"- Quantization: NVFP4")
    logger.info(f"- Output directory: {output_dir}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Apply NVFP4 quantization to a model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model (HuggingFace model ID or local path)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for quantized model",
    )
    parser.add_argument(
        "--calibration-dataset",
        type=str,
        default="HuggingFaceH4/ultrachat_200k",
        help="Calibration dataset name",
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=512,
        help="Number of calibration samples",
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=2048, help="Maximum sequence length"
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="Trust remote code"
    )

    args = parser.parse_args()

    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for quantization")

    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA Version: {torch.version.cuda}")

    # Run quantization
    quantize_to_nvfp4(
        model_path=args.model_path,
        output_dir=args.output_dir,
        calibration_dataset=args.calibration_dataset,
        num_calibration_samples=args.num_calibration_samples,
        max_seq_length=args.max_seq_length,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
