from tokenizer.char_tokenizer import CharTokenizer
from datasets import load_dataset
from transformers import GPT2Config,Trainer, TrainingArguments, EarlyStoppingCallback, DataCollatorForLanguageModeling
from model.vq_passgpt import VQPassGPTModel
import time
import argparse
import torch
import os
import math
os.environ["WANDB_DISABLED"] = "true"
# ----------------------------
# Argument Parsing
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--vocabfile_path", type=str, default="./tokenizer/vocab.json")
parser.add_argument("--model_path", type=str, default="./model/")
parser.add_argument("--log_path", type=str, default="./log/")
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--num_processer", type=int, default=1)
parser.add_argument("--input_size", type=int, default=32)
parser.add_argument("--embed_size", type=int, default=384)
parser.add_argument("--layer_num", type=int, default=12)
parser.add_argument("--head_num", type=int, default=8)
parser.add_argument("--epoch_num", type=int, default=60)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--eval_step", type=int, default=2000)
parser.add_argument("--save_step", type=int, default=6000)
parser.add_argument("--early_stop", type=int, default=3)
args = parser.parse_args()

# ----------------------------
# Tokenizer
# ----------------------------
print("Loading tokenizer...")
tokenizer = CharTokenizer(vocab_file=args.vocabfile_path, 
                          bos_token="<BOS>",
                          eos_token="<EOS>",
                          sep_token="<SEP>",
                          unk_token="<UNK>",
                          pad_token="<PAD>",
                          )
def load_matching_weights(target_model, pretrained_model):
    pretrained_dict = pretrained_model.state_dict()
    target_dict = target_model.state_dict()
    matched = {k: v for k, v in pretrained_dict.items() if k in target_dict and v.shape == target_dict[k].shape}
    target_dict.update(matched)
    target_model.load_state_dict(target_dict, strict=False)
    return len(matched)

SEP_ID = 1
def validate_ids(ids):
    # ids: List[int]
    if ids[0] != tokenizer.bos_token_id: return False
    if ids[-1] != tokenizer.eos_token_id: return False
    if SEP_ID not in ids: return False
    sep_pos = ids.index(SEP_ID)
    pattern = ids[1:sep_pos]       
    passwd  = ids[sep_pos+1:-1]   
    if not all(5 <= t <= 40 for t in pattern): return False
    if not all(41 <= t <= 134 for t in passwd): return False
    return True
def encode_row(batch):
    texts = batch["text"]  # list[str]
    input_ids_list, attn_list = [], []
    for s in texts:
        enc = tokenizer(s)
        ids = enc["input_ids"]
        if not validate_ids(ids):
            continue
        L = len(ids)
        max_len = args.input_size
        if L > max_len:
            ids = ids[:max_len]
            L = max_len
        pad_id = tokenizer.pad_token_id
        ids = ids + [pad_id]*(max_len-L)
        attn = [1]*L + [0]*(max_len-L)
        input_ids_list.append(ids)
        attn_list.append(attn)
    return {"input_ids": input_ids_list, "attention_mask": attn_list, "labels": input_ids_list}

def compute_metrics(eval_pred):
    loss = eval_pred.loss
    perplexity = math.exp(loss) if loss < 50 else float("inf")
    return {"perplexity": perplexity, "loss": loss}
# ----------------------------
# Dataset
# ----------------------------
print("Loading dataset...")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
dataset = load_dataset("text", data_files=args.dataset_path, num_proc=args.num_processer, split="train")
dataset = dataset.map(encode_row, batched=True, remove_columns=dataset.column_names)
dataset = dataset.train_test_split(test_size=0.125)
eval_dataset = dataset["test"]
train_dataset = dataset["train"]

# Print samples to check
tokenizer_info = [train_dataset[i]["input_ids"] for i in range(5)]
for line in tokenizer_info:
    print(line)

# ----------------------------
# Model Initialization
# ----------------------------
print("Setting up model config...")

print("Setting up model config...")

config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=args.input_size,
    n_ctx=args.input_size,
    n_embd=args.embed_size,
    n_layer=args.layer_num,
    n_head=args.head_num,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=getattr(tokenizer, "pad_token_id", None),
    activation_function="gelu_new",
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,
    scale_attn_by_inverse_layer_idx=False,
    reorder_and_upcast_attn=False,
)
config.bos_token_id = tokenizer.bos_token_id
config.eos_token_id = tokenizer.eos_token_id
config.sep_token_id = tokenizer.sep_token_id
config.pad_token_id = tokenizer.pad_token_id

config.enforce_constraints_train = True

model = VQPassGPTModel(config=config, use_vq=True)

print(f"Total parameters: {model.num_parameters():,}")

# ----------------------------
# Trainer Configuration
# ----------------------------
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
    callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stop)],
    compute_metrics=compute_metrics
)

print("Starting training...")
trainer.train()

# ----------------------------
# Save Final Model
# ----------------------------
print("Saving final model...")
save_path = os.path.join(args.model_path, "last-step")
os.makedirs(save_path, exist_ok=True)
model.config.use_vq = True
model.save_pretrained(save_path, safe_serialization=True)
torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
print(f"Model and weights saved at {save_path}")
