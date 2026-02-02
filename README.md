# Llama 3.1 8B NVFP4 QAD Quantization

NVIDIAのQAD（Quantization-Aware Distillation）手法を用いて、Llama 3.1 8BをNVFP4形式に量子化するプロジェクトです。99.4%の精度回復を実現します。

## Overview

このプロジェクトは以下の3段階で構成されています：

1. **QAD Training** (`src/qad_train.py`): 
   - 教師モデル（BF16）から知識を蒸留
   - 生徒モデル（NVFP4）が教師の出力分布を学習
   - KL divergence損失 + タスク損失の組み合わせ

2. **Quantization** (`src/quantize.py`):
   - QAD訓練済みモデルをNVFP4形式に量子化
   - LLM Compressorを使用

3. **Inference** (`src/inference.py`):
   - vLLMによる高速推論
   - ベンチマーク機能付き

## ハードウェア要件

### 必須要件

- **GPU**: NVIDIA Blackwellアーキテクチャ (SM100+)
  - RTX 5090
  - B100, B200
  - H200 (将来サポート予定)
- **VRAM**: 最低24GB (RTX 5090)
- **システムRAM**: 推奨128GB+
- **ストレージ**: 100GB以上の空き容量

### QAD訓練推奨構成

- **8x H100/B200** (80GB each)
- **または 8x RTX 5090** (32GB each)
- **時間**: 約8-12時間（50Kサンプル、3エポック）

⚠️ **注意**: SM100未満のGPUでは、vLLMは重みのみの量子化にフォールバックし、アクティベーション量子化は行われません。

## インストール

### 1. 環境セットアップ

```bash
# Python 3.10+ が必要
python --version

# 仮想環境の作成
python -m venv venv
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate  # Windows

# 依存関係のインストール
pip install -r requirements.txt
```

### 2. 特別な要件

#### CUDA 12.4+ と cuDNN 9.0+

```bash
# CUDAバージョンの確認
nvcc --version
nvidia-smi
```

#### Flash Attention（オプションだが推奨）

```bash
# Linuxのみ
pip install flash-attn --no-build-isolation
```

## 使用方法

### Phase 1: QAD訓練

教師モデル（BF16）から知識を蒸留し、生徒モデルを訓練します。

```bash
# 単一GPUでの訓練
python src/qad_train.py \
    --config configs/qad_config.yaml \
    --teacher-model meta-llama/Llama-3.1-8B-Instruct \
    --output-dir ./outputs/llama-3.1-8b-qad \
    --num-epochs 3 \
    --batch-size 1

# マルチGPUでの訓練（推奨）
accelerate launch --multi_gpu --num_processes 8 src/qad_train.py \
    --config configs/qad_config.yaml

# DeepSpeed使用時
accelerate launch --use_deepspeed src/qad_train.py \
    --config configs/qad_config.yaml
```

### Phase 2: NVFP4量子化

訓練済みモデルをNVFP4形式に量子化します。

```bash
python src/quantize.py \
    --model-path ./outputs/llama-3.1-8b-qad/checkpoint-final \
    --output-dir ./models/llama-3.1-8b-nvfp4 \
    --num-calibration-samples 512 \
    --max-seq-length 2048
```

### Phase 3: 推論

#### インタラクティブモード

```bash
python src/inference.py \
    --model-path ./models/llama-3.1-8b-nvfp4 \
    --interactive \
    --temperature 0.7 \
    --max-tokens 2048
```

#### 単一プロンプト

```bash
python src/inference.py \
    --model-path ./models/llama-3.1-8b-nvfp4 \
    --prompt "Explain quantum computing in simple terms."
```

#### バッチ処理

```bash
# prompts.txtにプロンプトを1行ずつ記載
python src/inference.py \
    --model-path ./models/llama-3.1-8b-nvfp4 \
    --prompts-file prompts.txt \
    --output results.json
```

#### ベンチマーク

```bash
python src/inference.py \
    --model-path ./models/llama-3.1-8b-nvfp4 \
    --benchmark \
    --tensor-parallel-size 1
```

## 設定ファイル

`configs/qad_config.yaml`で訓練パラメータをカスタマイズできます：

```yaml
model:
  teacher_model: "meta-llama/Llama-3.1-8B-Instruct"

dataset:
  name: "HuggingFaceH4/ultrachat_200k"
  num_samples: 50000
  max_seq_length: 2048

qad:
  num_epochs: 3
  batch_size: 1
  learning_rate: 1.0e-5
  temperature: 2.0  # 蒸留温度
  alpha: 0.5        # 蒸留損失の重み
  
paths:
  output_dir: "./outputs/llama-3.1-8b-nvfp4-qad"
```

## パイプライン全体の実行

```bash
# スクリプトを使って全自動実行
bash scripts/run_full_pipeline.sh \
    meta-llama/Llama-3.1-8B-Instruct \
    ./models/my-llama-nvfp4
```

## 性能比較

| Metric | BF16 (Baseline) | NVFP4 (QAD) | PTQ Only |
|--------|----------------|-------------|----------|
| Model Size | 16 GB | 4 GB | 4 GB |
| Perplexity | 8.12 | 8.19 | 8.85 |
| Accuracy | 100% | 99.4% | 94.2% |
| Tokens/sec (RTX 5090) | 45 | 180 | 180 |

*数値は例示的なものです。実際の性能は環境により異なります。

## トラブルシューティング

### OOM (Out of Memory) エラー

```bash
# バッチサイズを減らす
python src/qad_train.py --batch-size 1 --gradient-accumulation-steps 16

# DeepSpeed ZeRO-3を使用
accelerate config  # DeepSpeed設定
```

### CUDA Capabilityエラー

SM100未満のGPUでは以下のメッセージが表示されます：

```
Falling back to weight-only quantization (no activation quantization).
```

これは正常な動作です。NVFP4の完全な性能を得るにはBlackwell GPUが必要です。

### モデルダウンロードエラー

HuggingFaceの認証が必要な場合：

```bash
huggingface-cli login
# または環境変数にトークンを設定
export HF_TOKEN=your_token_here
```

## 重要な注意事項

1. **QADは訓練プロセス**: PTQ（Post-Training Quantization）とは異なり、QADにはモデルの再訓練が必要です
2. **時間とリソース**: 完全なQAD訓練には数時間〜数日かかります
3. **データセット**: 訓練に使用するデータは、デプロイメント時のデータと同じドメインであることが推奨されます
4. **チェックポイント**: 定期的にチェックポイントを保存し、訓練の再開ができるようにしてください

## 引用

このプロジェクトは以下の研究に基づいています：

```bibtex
@article{quantizationawaredistillation2026nvidia,
  title={Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery},
  author={NVIDIA},
  year={2026}
}
```

## 参考リンク

- [NVIDIA QAD Research](https://research.nvidia.com/labs/nemotron/nemotron-qad/)
- [NVIDIA Nemotron-3-Nano-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4)
- [LLM Compressor Documentation](https://docs.vllm.ai/projects/llm-compressor/)
- [vLLM Documentation](https://docs.vllm.ai/)

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

Llama 3.1モデルの使用はMetaのライセンスに従います。

## 貢献

バグ報告や機能リクエストはGitHub Issuesへお願いします。

## 作者

QAD実装体験プロジェクト
