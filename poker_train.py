"""QLoRA-обучение Gemma-3-4b-it на покерном датасете."""

import os
import json

import torch
from dotenv import load_dotenv
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from poker_common import encode_example

MODEL_ID = "google/gemma-3-4b-it"
OUT_DIR = "./poker_lora"
EPOCHS = 3

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN не найден в .env")

# ---------- токенизатор ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ---------- модель в 4 бита ----------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0},
    token=HF_TOKEN,
)
model.config.use_cache = False

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---------- данные ----------
with open("data_train.jsonl", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

features = [
    encode_example(tokenizer, r["board"], r["hand"], r["answer"])
    for r in rows
]
train_ds = Dataset.from_list(features)

print(f"Примеров: {len(train_ds)}, длина первого: {len(features[0]['input_ids'])} токенов")


def collate(batch):
    max_len = max(len(f["input_ids"]) for f in batch)
    input_ids, labels, attention = [], [], []

    for f in batch:
        pad = max_len - len(f["input_ids"])
        input_ids.append(f["input_ids"] + [tokenizer.pad_token_id] * pad)
        labels.append(f["labels"] + [-100] * pad)
        attention.append(f["attention_mask"] + [0] * pad)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
    }


use_bf16 = torch.cuda.is_bf16_supported()

args = TrainingArguments(
    output_dir="./poker_checkpoints",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=EPOCHS,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=5,
    save_strategy="no",
    report_to="none",
    bf16=use_bf16,
    fp16=not use_bf16,
    optim="adamw_torch",
    max_grad_norm=1.0,
    remove_unused_columns=False,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    data_collator=collate,
)

trainer.train()

model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

print(f"\n✅ Адаптер сохранён в {OUT_DIR}")