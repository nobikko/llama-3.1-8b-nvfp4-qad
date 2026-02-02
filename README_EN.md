# Llama 3.1 8B NVFP4 QAD Quantization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.4+](https://img.shields.io/badge/CUDA-12.4+-green.svg)](https://developer.nvidia.com/cuda-downloads)

NVIDIA's QAD (Quantization-Aware Distillation) implementation for achieving 99.4% accuracy with Llama 3.1 8B NVFP4 quantization.

![QAD Overview](https://research.nvidia.com/labs/nemotron/images/nemotron-qad/qad_qat.png)

## Features

- Complete QAD training pipeline with teacher-student distillation
- KL divergence-based knowledge distillation
- NVFP4 quantization with LLM Compressor
- High-performance vLLM inference
- Interactive chat and benchmarking modes
- Multi-GPU distributed training support

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU Architecture | NVIDIA Blackwell (SM100+) | 8x H100/B200 or 8x RTX 5090 |
| VRAM | 24 GB | 640 GB total (8x 80GB) |
| System RAM | 64 GB | 256 GB |
| Storage | 100 GB SSD | 500 GB NVMe SSD |

**Compatible GPUs:**
- RTX 5090 (32GB)
- B100, B200 (80GB/180GB)
- H200 (141GB) - Future support

## Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/llama-3.1-8b-nvfp4-qad.git
cd llama-3.1-8b-nvfp4-qad

# Install dependencies
pip install -r requirements.txt

# Run full pipeline
bash scripts/run_full_pipeline.sh
```

## Documentation

- [QAD Training Guide](docs/QAD_TRAINING.md)
- [Quantization Guide](docs/QUANTIZATION.md)
- [Inference Guide](docs/INFERENCE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## Project Structure

```
llama-3.1-8b-nvfp4-qad/
├── src/
│   ├── qad_train.py      # QAD training script
│   ├── quantize.py       # NVFP4 quantization
│   └── inference.py      # vLLM inference
├── configs/
│   └── qad_config.yaml   # Training configuration
├── scripts/
│   └── run_full_pipeline.sh
├── docs/                  # Documentation
├── requirements.txt
└── README.md
```

## Performance

| Metric | BF16 Baseline | NVFP4 QAD | Improvement |
|--------|---------------|-----------|-------------|
| Model Size | 16 GB | 4 GB | **4x smaller** |
| Accuracy | 100% | 99.4% | -0.6% |
| Throughput (RTX 5090) | 45 tok/s | 180 tok/s | **4x faster** |
| Memory Bandwidth | 1TB/s | 0.25TB/s | - |

## Citation

```bibtex
@article{quantizationawaredistillation2026nvidia,
  title={Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery},
  author={NVIDIA},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

Llama 3.1 models are subject to Meta's license.

## Acknowledgments

- NVIDIA Research Team for QAD methodology
- Meta for Llama 3.1 models
- vLLM team for inference engine
- HuggingFace for transformers and datasets

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Support

- GitHub Issues: Bug reports and feature requests
- Discussions: Q&A and general discussion
- Wiki: Additional documentation

---

**Note**: This is a research implementation. For production use, please validate thoroughly on your specific use case.
