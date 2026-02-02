#!/usr/bin/env python3
"""Environment validation script for Llama 3.1 8B NVFP4 QAD Project"""

import sys
import subprocess


def check_python_version():
    """Check Python version"""
    print("=" * 60)
    print("1. Python Version Check")
    print("=" * 60)
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor >= 10:
        print("✓ Python version OK (3.10+)")
        return True
    else:
        print("✗ Python 3.10+ required")
        return False


def check_cuda():
    """Check CUDA availability"""
    print("\n" + "=" * 60)
    print("2. CUDA Check")
    print("=" * 60)

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")

            gpu_count = torch.cuda.device_count()
            print(f"Number of GPUs: {gpu_count}")

            blackwell_count = 0
            for i in range(gpu_count):
                print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
                capability = torch.cuda.get_device_capability(i)
                print(f"  Compute Capability: SM{capability[0]}{capability[1]}")
                print(
                    f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB"
                )

                # Blackwell check
                if capability[0] >= 10:
                    print(f"  ✓ Blackwell GPU detected (NVFP4 compatible)")
                    blackwell_count += 1
                else:
                    print(f"  ⚠ Not a Blackwell GPU (NVFP4 limited support)")

            if blackwell_count > 0:
                print(
                    f"\n✓ {blackwell_count} Blackwell GPU(s) available for full NVFP4 acceleration"
                )
            else:
                print(
                    "\n⚠ No Blackwell GPUs found. Will use weight-only quantization fallback."
                )

            return True
        else:
            print("✗ CUDA not available")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False


def check_packages():
    """Check required packages"""
    print("\n" + "=" * 60)
    print("3. Package Check")
    print("=" * 60)

    packages = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "datasets": "Datasets",
        "accelerate": "Accelerate",
        "llmcompressor": "LLM Compressor",
        "vllm": "vLLM",
        "yaml": "PyYAML",
        "numpy": "NumPy",
        "tqdm": "tqdm",
    }

    all_ok = True
    for module, name in packages.items():
        try:
            if module == "yaml":
                mod = __import__(module)
            else:
                mod = __import__(module)
            version = getattr(mod, "__version__", "unknown")
            print(f"✓ {name}: {version}")
        except ImportError:
            print(f"✗ {name}: not installed")
            all_ok = False

    return all_ok


def check_disk_space():
    """Check disk space"""
    print("\n" + "=" * 60)
    print("4. Disk Space Check")
    print("=" * 60)

    import shutil

    total, used, free = shutil.disk_usage(".")

    print(f"Total: {total / 1024**3:.2f} GB")
    print(f"Used: {used / 1024**3:.2f} GB")
    print(f"Free: {free / 1024**3:.2f} GB")

    if free > 500 * 1024**3:  # 500GB
        print("✓ Sufficient disk space (500GB+)")
        return True
    elif free > 100 * 1024**3:  # 100GB
        print("⚠ Limited disk space (100-500GB). May need to clean up during training.")
        return True
    else:
        print("✗ Insufficient disk space (less than 100GB)")
        return False


def check_memory():
    """Check system memory"""
    print("\n" + "=" * 60)
    print("5. Memory Check")
    print("=" * 60)

    try:
        import psutil

        mem = psutil.virtual_memory()
        print(f"Total RAM: {mem.total / 1024**3:.2f} GB")
        print(f"Available: {mem.available / 1024**3:.2f} GB")

        if mem.total > 128 * 1024**3:  # 128GB
            print("✓ Sufficient RAM (128GB+)")
            return True
        elif mem.total > 64 * 1024**3:  # 64GB
            print("⚠ Limited RAM (64-128GB). Consider using swap.")
            return True
        else:
            print("✗ Insufficient RAM (less than 64GB)")
            return False
    except ImportError:
        print("⚠ psutil not installed, skipping memory check")
        return True


def main():
    print("\n" + "=" * 60)
    print("Environment Validation")
    print("Llama 3.1 8B NVFP4 QAD Project")
    print("=" * 60)

    checks = [
        ("Python Version", check_python_version()),
        ("CUDA/PyTorch", check_cuda()),
        ("Required Packages", check_packages()),
        ("Disk Space", check_disk_space()),
        ("System Memory", check_memory()),
    ]

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")

    all_passed = all(passed for _, passed in checks)

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All checks passed! Environment is ready.")
        print("\nNext steps:")
        print("1. Run: python src/qad_train.py --config configs/qad_config.yaml")
        print("2. Or start with full pipeline: bash scripts/run_full_pipeline.sh")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        print("\nSee docs/ENVIRONMENT_SETUP.md for troubleshooting.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
