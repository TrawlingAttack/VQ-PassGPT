# VQ-PassGPT

A Transformer-based password generation model combining pattern conditioning, vector quantization, and hierarchical decoding.

## Features

- Built on GPT-2 decoder-only architecture
- Integrates vector quantization (VQ) to compress representations
- Supports pattern-guided password generation
- Uses D&C (divide-and-conquer) for efficient decoding
- Easily extendable and trainable via HuggingFace Trainer
## Environment
```bash
conda create -n vqpassgpt python=3.8.10
conda activate vqpassgpt
pip install -r requirements.txt
```
## Dataset
These files provides links to password leak datasets used in training and evaluating neural password generation models, particularly under trawling and targeted attack scenarios.
We strongly encourage researchers and practitioners to train models using real-world password leaks. Below are curated datasets that have been cleaned and organized for experimentation.

https://github.com/TrawlingAttack/VQ-PassGPT/releases/download/leakpass/leakdataset.zip
https://github.com/brannondorsey/PassGAN/releases/download/data/68_linkedin_found_hash_plain.txt.zip
## Preprocess
Clean Dataset
```bash
python clean_dataset.py --dataset_path train.txt --output_path train_cleaned.txt
```
Concat pattern passwords
```bash
python concat_pattern_password.py --dataset_path train_cleaned.txt --output_path train_pattern.txt
```
## Training
```bash
python train.py --dataset_path train_pattern.txt
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
- Trained VQ-PassGPT model (e.g. under `model\last-step`)
- Pattern distribution file (`patterns.txt`), tab-separated with two columns: `pattern` and `rate`

---

## Usage

### D&C Generation

```bash
python D&C_GEN.py --model_path model\last-step --output_path ./generated/ --pattern_path patterns.txt --generate_num 1000000
```
### Normal Generation

```bash
python normal-gen.py --model_path model\last-step --output_path ./generated/ --batch_size 100 --generate_num 1000000
```

