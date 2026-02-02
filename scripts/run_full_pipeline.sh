#!/bin/bash
# Full pipeline script for Llama 3.1 8B NVFP4 QAD Quantization
# This script runs the complete pipeline: QAD training -> Quantization -> Testing

set -e  # Exit on error

# Configuration
TEACHER_MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
OUTPUT_BASE="${2:-./models/llama-3.1-8b-nvfp4}"
QAD_OUTPUT="${OUTPUT_BASE}-qad"
QUANTIZED_OUTPUT="${OUTPUT_BASE}"

# Training parameters
NUM_EPOCHS=3
BATCH_SIZE=1
GRAD_ACCUM=8
NUM_SAMPLES=50000

echo "=============================================="
echo "Llama 3.1 8B NVFP4 QAD Pipeline"
echo "=============================================="
echo "Teacher Model: $TEACHER_MODEL"
echo "QAD Output: $QAD_OUTPUT"
echo "Final Output: $QUANTIZED_OUTPUT"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi

if ! python -c "import torch; print(torch.__version__)" &> /dev/null; then
    echo "Error: PyTorch not installed"
    exit 1
fi

if ! python -c "import transformers; print(transformers.__version__)" &> /dev/null; then
    echo "Error: Transformers not installed"
    exit 1
fi

echo "✓ Prerequisites check passed"
echo ""

# Phase 1: QAD Training
echo "=============================================="
echo "Phase 1: QAD Training"
echo "=============================================="
echo "Starting QAD training..."
echo "This will take several hours depending on your hardware."
echo ""

mkdir -p "$QAD_OUTPUT"

# Check if we have multiple GPUs
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Detected $NUM_GPUS GPUs - using distributed training"
    accelerate launch --multi_gpu --num_processes "$NUM_GPUS" src/qad_train.py \
        --config configs/qad_config.yaml \
        --teacher-model "$TEACHER_MODEL" \
        --output-dir "$QAD_OUTPUT" \
        --num-epochs "$NUM_EPOCHS" \
        --batch-size "$BATCH_SIZE"
else
    echo "Single GPU detected - using single GPU training"
    python src/qad_train.py \
        --config configs/qad_config.yaml \
        --teacher-model "$TEACHER_MODEL" \
        --output-dir "$QAD_OUTPUT" \
        --num-epochs "$NUM_EPOCHS" \
        --batch-size "$BATCH_SIZE"
fi

echo ""
echo "✓ QAD training completed"
echo "Model saved to: $QAD_OUTPUT/checkpoint-final"
echo ""

# Phase 2: Quantization
echo "=============================================="
echo "Phase 2: NVFP4 Quantization"
echo "=============================================="
echo "Applying NVFP4 quantization..."
echo ""

python src/quantize.py \
    --model-path "$QAD_OUTPUT/checkpoint-final" \
    --output-dir "$QUANTIZED_OUTPUT" \
    --num-calibration-samples 512 \
    --max-seq-length 2048

echo ""
echo "✓ Quantization completed"
echo "Quantized model saved to: $QUANTIZED_OUTPUT"
echo ""

# Phase 3: Testing
echo "=============================================="
echo "Phase 3: Testing"
echo "=============================================="
echo "Running inference test..."
echo ""

# Quick test
python src/inference.py \
    --model-path "$QUANTIZED_OUTPUT" \
    --prompt "Explain what NVFP4 quantization is and why it's beneficial for large language models." \
    --max-tokens 200

echo ""
echo "Running benchmark..."
echo ""

# Benchmark
python src/inference.py \
    --model-path "$QUANTIZED_OUTPUT" \
    --benchmark \
    --tensor-parallel-size 1

echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo "QAD Model: $QAD_OUTPUT"
echo "Quantized Model: $QUANTIZED_OUTPUT"
echo ""
echo "Next steps:"
echo "1. Test inference: python src/inference.py --model-path $QUANTIZED_OUTPUT --interactive"
echo "2. Upload to HuggingFace: huggingface-cli upload your-model-name $QUANTIZED_OUTPUT"
echo "3. Deploy with vLLM server: python -m vllm.entrypoints.openai.api_server --model $QUANTIZED_OUTPUT"
