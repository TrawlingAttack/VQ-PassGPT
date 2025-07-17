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
