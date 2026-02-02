# 環境構築ガイド

このドキュメントでは、Llama 3.1 8B NVFP4 QADプロジェクトを実行するための環境構築手順を説明します。

## 目次

- [前提条件](#前提条件)
- [推奨環境](#推奨環境)
- [Windows環境構築](#windows環境構築)
- [Linux環境構築](#linux環境構築)
- [Docker環境](#docker環境)
- [検証スクリプト](#検証スクリプト)
- [トラブルシューティング](#トラブルシューティング)

---

## 前提条件

### ハードウェア要件

| 項目 | 最小要件 | 推奨環境 |
|-----|---------|---------|
| GPU | NVIDIA Blackwell (RTX 5090) | 8x RTX 5090 または 8x B200 |
| VRAM | 24 GB | 640 GB (8x 80GB) |
| CPU | 16コア | 64コア以上 |
| メモリ | 128 GB | 512 GB |
| ストレージ | 500 GB SSD | 2 TB NVMe SSD |

### ソフトウェア要件

- **OS**: Windows 11 または Ubuntu 22.04 LTS
- **Python**: 3.10 以上
- **CUDA**: 12.4 以上
- **NVIDIAドライバー**: 550.54.14 以上

---

## 推奨環境

### CUDA Compute Capability

```bash
# GPUの確認
nvidia-smi

# CUDA Capabilityの確認（Blackwell = SM100以上）
python -c "import torch; print(torch.cuda.get_device_capability())"
# 出力: (10, 0) 以上が必要
```

---

## Windows環境構築

### 1. NVIDIAドライバーのインストール

1. [NVIDIA Driver Download](https://www.nvidia.com/drivers/) から最新ドライバーをダウンロード
2. RTX 5090対応ドライバー（550.54.14以降）をインストール
3. 再起動

### 2. CUDA Toolkitのインストール

```powershell
# CUDA 12.4のインストール
# https://developer.nvidia.com/cuda-12-4-0-download-archive

# インストーラーをダウンロードして実行
# カスタムインストールを選択し、以下をインストール：
# - CUDA Runtime
# - CUDA Development
# - CUDA Documentation

# 環境変数の確認
[System.Environment]::GetEnvironmentVariable("CUDA_PATH", "Machine")
# 出力: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
```

### 3. Pythonのインストール

```powershell
# Python 3.10のインストール
# https://www.python.org/downloads/release/python-31011/

# インストール時に「Add Python to PATH」にチェック

# バージョン確認
python --version
# Python 3.10.11
```

### 4. 仮想環境の作成

```powershell
# プロジェクトディレクトリに移動
cd C:\path\to\llama-3.1-8b-nvfp4-qad

# 仮想環境の作成
python -m venv venv

# 仮想環境の有効化
.\venv\Scripts\activate

# (無効化する場合)
# deactivate
```

### 5. PyTorchのインストール

```powershell
# CUDA 12.4対応のPyTorch
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124

# インストール確認
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
```

### 6. 依存関係のインストール

```powershell
# 基本パッケージ
pip install transformers==4.45.0
pip install datasets==2.14.0
pip install accelerate==0.34.0
pip install peft==0.13.0
pip install sentencepiece protobuf

# QAD・量子化関連
pip install modelopt==0.21.0
pip install llmcompressor==0.5.0
pip install compressed-tensors==0.8.0

# vLLM (Windows非対応の場合はWSL2を使用)
# pip install vllm==0.8.0
```

**注意**: vLLMはWindowsネイティブでは未対応の場合があります。その場合はWSL2を使用してください。

---

## Linux環境構築 (Ubuntu 22.04)

### 1. システム更新

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl wget
```

### 2. NVIDIAドライバーのインストール

```bash
# 既存ドライバーの削除
sudo apt purge nvidia*

# 推奨ドライバーの確認
ubuntu-drivers devices

# 最新ドライバーのインストール
sudo apt install -y nvidia-driver-550

# 再起動
sudo reboot
```

### 3. CUDA Toolkitのインストール

```bash
# CUDA 12.4のインストール
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/

sudo apt update
sudo apt install -y cuda-toolkit-12-4

# 環境変数の設定
echo 'export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc

# 確認
nvcc --version
nvidia-smi
```

### 4. Pythonのインストール

```bash
# Python 3.10のインストール
sudo apt install -y python3.10 python3.10-venv python3.10-dev python3-pip

# 確認
python3.10 --version
```

### 5. 仮想環境の作成

```bash
# プロジェクトディレクトリに移動
cd ~/llama-3.1-8b-nvfp4-qad

# 仮想環境の作成
python3.10 -m venv venv

# 有効化
source venv/bin/activate

# pipのアップグレード
pip install --upgrade pip setuptools wheel
```

### 6. PyTorchのインストール

```bash
# CUDA 12.4対応のPyTorch 2.5.0
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124

# 確認
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### 7. 依存関係のインストール

```bash
# 基本パッケージ
pip install transformers==4.45.0 datasets==2.14.0 accelerate==0.34.0 peft==0.13.0
pip install sentencepiece protobuf

# 量子化関連
pip install llmcompressor==0.5.0 compressed-tensors==0.8.0

# vLLM
pip install vllm==0.8.0

# オプション: Flash Attention (コンパイルに時間がかかります)
pip install flash-attn --no-build-isolation

# 開発ツール
pip install wandb tensorboard pytest black isort
```

### 8. 完全インストール（推奨）

```bash
# requirements.txtを使用
pip install -r requirements.txt
```

---

## Docker環境

### Dockerfile

```dockerfile
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# システムパッケージ
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージ
RUN pip3 install --no-cache-dir \
    torch==2.5.0 \
    torchvision==0.20.0 \
    torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cu124

# ワークディレクトリ
WORKDIR /workspace

# プロジェクトのコピー
COPY . /workspace/llama-3.1-8b-nvfp4-qad

# 依存関係のインストール
RUN pip3 install --no-cache-dir -r /workspace/llama-3.1-8b-nvfp4-qad/requirements.txt

# 環境変数
ENV PYTHONPATH=/workspace/llama-3.1-8b-nvfp4-qad:$PYTHONPATH
ENV CUDA_HOME=/usr/local/cuda

WORKDIR /workspace/llama-3.1-8b-nvfp4-qad

CMD ["/bin/bash"]
```

### Dockerのビルドと実行

```bash
# イメージのビルド
docker build -t llama-nvfp4-qad .

# コンテナの実行（GPUアクセス付き）
docker run --gpus all -it --rm \
    -v $(pwd)/models:/workspace/models \
    -v $(pwd)/outputs:/workspace/outputs \
    llama-nvfp4-qad

# または docker-composeを使用
docker-compose up -d
docker-compose exec llama-nvfp4-qad bash
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  llama-nvfp4-qad:
    build: .
    image: llama-nvfp4-qad:latest
    container_name: llama-nvfp4-qad
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    volumes:
      - ./models:/workspace/models
      - ./outputs:/workspace/outputs
      - ./cache:/workspace/cache
      - ./logs:/workspace/logs
    shm_size: '256gb'
    stdin_open: true
    tty: true
    command: /bin/bash
```

---

## 検証スクリプト

### 環境検証スクリプト

`scripts/check_environment.py`を作成して実行します：

```python
#!/usr/bin/env python3
"""環境検証スクリプト"""

import sys
import subprocess

def check_python_version():
    """Pythonバージョンの確認"""
    print("=" * 60)
    print("1. Python Version Check")
    print("=" * 60)
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 10:
        print("✓ Python version OK")
        return True
    else:
        print("✗ Python 3.10+ required")
        return False

def check_cuda():
    """CUDAの確認"""
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
            
            for i in range(torch.cuda.device_count()):
                print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
                capability = torch.cuda.get_device_capability(i)
                print(f"  Compute Capability: SM{capability[0]}{capability[1]}")
                print(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
                
                # Blackwellチェック
                if capability[0] >= 10:
                    print(f"  ✓ Blackwell GPU detected (NVFP4 compatible)")
                else:
                    print(f"  ⚠ Not a Blackwell GPU (NVFP4 limited support)")
            
            return True
        else:
            print("✗ CUDA not available")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False

def check_packages():
    """必要なパッケージの確認"""
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
    }
    
    all_ok = True
    for module, name in packages.items():
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "unknown")
            print(f"✓ {name}: {version}")
        except ImportError:
            print(f"✗ {name}: not installed")
            all_ok = False
    
    return all_ok

def check_disk_space():
    """ディスク容量の確認"""
    print("\n" + "=" * 60)
    print("4. Disk Space Check")
    print("=" * 60)
    
    import shutil
    total, used, free = shutil.disk_usage(".")
    
    print(f"Total: {total / 1024**3:.2f} GB")
    print(f"Used: {used / 1024**3:.2f} GB")
    print(f"Free: {free / 1024**3:.2f} GB")
    
    if free > 500 * 1024**3:  # 500GB
        print("✓ Sufficient disk space")
        return True
    else:
        print("⚠ Less than 500GB free (recommend 500GB+)")
        return False

def main():
    print("\n" + "=" * 60)
    print("Environment Validation")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version()),
        ("CUDA/PyTorch", check_cuda()),
        ("Required Packages", check_packages()),
        ("Disk Space", check_disk_space()),
    ]
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")
    
    all_passed = all(passed for _, passed in checks)
    
    if all_passed:
        print("\n✓ All checks passed! Ready to start.")
        return 0
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

### 実行方法

```bash
# Windows PowerShell
python scripts/check_environment.py

# Linux/Mac
python3 scripts/check_environment.py
```

---

## トラブルシューティング

### 1. CUDA not available

**症状**: `torch.cuda.is_available()` が `False` を返す

**解決策**:
```bash
# NVIDIAドライバーの確認
nvidia-smi

# PyTorchの再インストール
pip uninstall torch torchvision torchaudio
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu124

# 環境変数の確認（Linux）
echo $LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### 2. Out of Memory during training

**症状**: CUDA Out of Memoryエラー

**解決策**:
```bash
# バッチサイズを減らす
python src/qad_train.py --batch-size 1 --gradient-accumulation-steps 16

# DeepSpeedを使用
pip install deepspeed
accelerate config  # DeepSpeed ZeRO-3を選択
```

### 3. Flash Attentionのインストールエラー

**症状**: Flash Attentionのビルドに失敗

**解決策**:
```bash
# 前提パッケージのインストール
pip install packaging ninja

# 最大ジョブ数を制限
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

### 4. vLLMのインストールエラー（Windows）

**症状**: WindowsでvLLMがインストールできない

**解決策**:
```powershell
# WSL2を使用するか、Windows用のビルドを使用
# または、WSL2環境で実行
wsl --install
wsl
# その後Linuxの手順に従う
```

### 5. HuggingFace認証エラー

**症状**: モデルダウンロード時に認証エラー

**解決策**:
```bash
# HuggingFace CLIのインストール
pip install huggingface-hub

# ログイン
huggingface-cli login
# または
export HF_TOKEN=your_token_here
```

### 6. インポートエラー（llmcompressor）

**症状**: `ImportError: No module named 'llmcompressor'`

**解決策**:
```bash
# 正しいパッケージ名でインストール
pip install llmcompressor

# バージョン確認
python -c "import llmcompressor; print(llmcompressor.__version__)"
```

---

## パフォーマンス最適化

### PyTorch設定

```python
# 最適化設定
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
```

### 環境変数

```bash
# Linux/.bashrcまたはWindowsシステム環境変数
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=16
```

---

## 次のステップ

環境構築が完了したら：

1. [検証スクリプト](#検証スクリプト)を実行
2. [README.md](../README.md)の「使用方法」に従ってトレーニング開始
3. 問題があれば[GitHub Issues](https://github.com/nobikko/llama-3.1-8b-nvfp4-qad/issues)へ
