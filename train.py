# VQ-PassGPT Training Script with Encoder, Quantization, and Decoder

from tokenizer.char_tokenizer import CharTokenizer
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from model.vq_passgpt import VQPassGPTModel  # Custom model with encoder + quantization + decoder
import time
import argparse
import os
os.environ["WANDB_DISABLED"] = "true"

parser = argparse.ArgumentParser()
# File path parameters
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--vocabfile_path", type=str, default="./tokenizer/vocab.json")
parser.add_argument("--model_path", type=str, default="./model/")
parser.add_argument("--log_path", type=str, default="./log/")
# Environment
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--num_processer", type=int, default=1)
# Model settings
parser.add_argument("--input_size", type=int, default=32)
parser.add_argument("--embed_size", type=int, default=384)
parser.add_argument("--layer_num", type=int, default=12)
parser.add_argument("--head_num", type=int, default=8)
# Training
parser.add_argument("--epoch_num", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--eval_step", type=int, default=2000)
parser.add_argument("--save_step", type=int, default=6000)
parser.add_argument("--early_stop", type=int, default=3)
args = parser.parse_args()

# Tokenizer
print("Loading tokenizer...")
tokenizer = CharTokenizer(vocab_file=args.vocabfile_path,
                          bos_token="<BOS>", eos_token="<EOS>",
                          sep_token="<SEP>", unk_token="<UNK>",
                          pad_token="<PAD>")

# Dataset
print("Loading dataset...")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
dataset = load_dataset('text', data_files=args.dataset_path, num_proc=args.num_processer, split='train')
dataset = dataset.map(lambda e: tokenizer(e['text'], max_len=args.input_size, padding=True), batched=True)

dataset = dataset.train_test_split(test_size=0.125)
eval_dataset = dataset['test']
train_dataset = dataset['train']

# Model Configuration
print("Setting up model config...")
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=args.input_size,
    n_embd=args.embed_size,
    n_layer=args.layer_num,
    n_head=args.head_num,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# Load VQ-PassGPT custom model
print("Initializing VQ-PassGPT model...")
model = VQPassGPTModel(config=config)  # <- This includes encoder, VQ layer, decoder
print(f"Total parameters: {model.num_parameters()}")

# Training Configuration
print("Preparing training arguments...")
training_args = TrainingArguments(
    output_dir=args.model_path,
    overwrite_output_dir=True,
    num_train_epochs=args.epoch_num,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    eval_steps=args.eval_step,
    save_steps=args.save_step,
    save_strategy='steps',
    evaluation_strategy='steps',
    prediction_loss_only=True,
    logging_dir=args.log_path + time.strftime("%Y%m%d-%H%M", time.localtime()),
    seed=args.random_seed,
    metric_for_best_model='eval_loss',
    load_best_model_at_end=True,
    save_total_limit=1,
    report_to='none',
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stop)]
)

print("Starting training...")
trainer.train()

print("Saving final model...")
trainer.save_model(args.model_path + "last-step/")
print(f"Model saved at {args.model_path + 'last-step/'}")
print("Training complete.")
