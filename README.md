# VQ-PassGPT

A Transformer-based password generation model combining pattern conditioning, vector quantization, and hierarchical decoding.

## Features

- Built on GPT-2 decoder-only architecture
- Integrates vector quantization (VQ) to compress representations
- Supports pattern-guided password generation
- Uses DGT (divide-and-conquer) for efficient decoding
- Easily extendable and trainable via HuggingFace Trainer

## Training
```bash
python train.py --dataset_path data/train.txt
```
  
# 🔐 VQ-PassGPT - DGT Password Generation

This repository implements **VQ-PassGPT**, a Transformer-based password guessing model that incorporates structural pattern guidance, vector quantization, and a divide-and-conquer generation strategy (DGT). This README explains how to generate passwords using the `DGT_gen.py` script.

---

## DGT-based Password Generation

The script `DGT_gen.py` supports large-scale, pattern-conformant password guessing using a trained VQ-PassGPT model and pattern frequency distribution. It uses a recursive divide-and-conquer mechanism to improve coverage and minimize duplication.

---

## Prerequisites

- Python 3.8+
- PyTorch ≥ 1.12
- Transformers ≥ 4.30
- `tokenizer/char_tokenizer.py`
- Trained VQ-PassGPT model (e.g. under `model/last/`)
- Pattern distribution file (`patterns.txt`), tab-separated with two columns: `pattern` and `rate`

---

## Usage

### Basic Command

```bash
python DGT_gen.py \
  --model_path model/last/ \
  --output_path ./generated/ \
  --pattern_path patterns.txt \
  --generate_num 1000000
