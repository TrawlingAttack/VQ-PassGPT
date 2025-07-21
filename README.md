# VQ-PassGPT

A Transformer-based password generation model combining pattern conditioning, vector quantization, and hierarchical decoding.

## Features

- Built on GPT-2 decoder-only architecture
- Integrates vector quantization (VQ) to compress representations
- Supports pattern-guided password generation
- Uses D&C (divide-and-conquer) for efficient decoding
- Easily extendable and trainable via HuggingFace Trainer

## Training
```bash
python train.py --dataset_path data/train.txt
```
  
# 🔐 VQ-PassGPT - D&C Password Generation

This repository implements **VQ-PassGPT**, a Transformer-based password guessing model that incorporates structural pattern guidance, vector quantization, and a divide-and-conquer generation strategy. This README explains how to generate passwords using the `D&C_GEN.py` script.

---

## D&C-based Password Generation

The script `D&C_GEN.py` supports large-scale, pattern-conformant password guessing using a trained VQ-PassGPT model and pattern frequency distribution. It uses a recursive divide-and-conquer mechanism to improve coverage and minimize duplication.

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

### D&C generate

```bash
python D&C_GEN.py \
  --model_path model/last/ \
  --output_path ./generated/ \
  --pattern_path patterns.txt \
  --generate_num 1000000
```
### Normal generate

```bash
python normal-gen.py --output_path gen_
```

