"""
train_safe.py - Обучение LoRA, устойчивое к версии transformers.
Лишние/неподдерживаемые аргументы отбрасываются автоматически.
Запуск: python train_safe.py
"""
import os
import json
import time
import random
import inspect

import torch
import transformers
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from torch.utils.data import Dataset
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# ================= КОНФИГ =================
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_FILE = "dataset_combined.json"
OUTPUT_DIR = "./lora_poker_safe"
EXISTING_LORA = None          # или "./lora_preflop_stage_3" чтобы дообучить существующий
MAX_SEQ_LEN = 768             # подними/понизь по выводу check_dataset.py
BATCH_SIZE = 1
GRAD_ACCUM = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 2
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
# ==========================================

print("=" * 60)
print(f"transformers: {transformers.__version__}")
print(f"torch: {torch.__version__}")
print("=" * 60)


def filter_kwargs(callable_obj, kwargs):
    """Оставляет только те аргументы, которые реально принимает объект."""
    try:
        sig = inspect.signature(callable_obj.__init__ if inspect.isclass(callable_obj)
                                else callable_obj)
        params = sig.parameters
    except (TypeError, ValueError):
        return kwargs, []
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs, []
    allowed = {k: v for k, v in kwargs.items() if k in params}
    dropped = [k for k in kwargs if k not in params]
    return allowed, dropped


def make_training_args(**kwargs):
    allowed, dropped = filter_kwargs(TrainingArguments, kwargs)
    if dropped:
        print(f"⚠️  TrainingArguments: отброшены неподдерживаемые аргументы: {dropped}")
    return TrainingArguments(**allowed)


if not torch.cuda.is_available():
    raise SystemExit("❌ CUDA недоступна")
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

# ---------- модель ----------
print("\n📦 Загрузка модели...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, trust_remote_code=True, padding_side="right"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model_kwargs = dict(
    token=HF_TOKEN,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
# в v5 аргумент называется dtype, в v4 — torch_dtype
sig = inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
if "dtype" in sig or "kwargs" in str(sig):
    model_kwargs["dtype"] = torch.float16
else:
    model_kwargs["torch_dtype"] = torch.float16
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
except TypeError:
    model_kwargs.pop("dtype", None)
    model_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)

model.config.use_cache = False
model.gradient_checkpointing_enable()
print("✅ Модель загружена")

# ---------- LoRA ----------
if EXISTING_LORA and os.path.exists(EXISTING_LORA):
    print(f"📦 Дообучение существующего LoRA: {EXISTING_LORA}")
    model = PeftModel.from_pretrained(model, EXISTING_LORA, is_trainable=True)
else:
    print("🆕 Новый LoRA")
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

for n, p in model.named_parameters():
    if "lora" not in n.lower():
        p.requires_grad = False
model.print_trainable_parameters()

# ---------- данные ----------
print(f"\n📚 Загрузка {DATASET_FILE}...")
with open(DATASET_FILE, "r", encoding="utf-8") as f:
    raw = json.load(f)
print(f"   Загружено: {len(raw)}")


class PokerDataset(Dataset):
    """Маскирует prompt. Примеры без обучаемых токенов ОТБРАСЫВАЮТСЯ (это лечит nan)."""

    def __init__(self, examples, tokenizer, max_len):
        self.tok = tokenizer
        self.max_len = max_len
        self.items = []
        skipped = 0
        for item in examples:
            rec = self._build(item)
            if rec is None:
                skipped += 1
            else:
                self.items.append(rec)
        self.skipped = skipped

    def _build(self, item):
        tok = self.tok
        user_text = tok.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        )
        user_ids = tok(user_text, add_special_tokens=False)["input_ids"]

        answer_text = tok.apply_chat_template(
            [{"role": "user", "content": item["prompt"]},
             {"role": "assistant", "content": item["completion"]}],
            tokenize=False, add_generation_prompt=False,
        )
        full_ids = tok(answer_text, add_special_tokens=False)["input_ids"]

        # ответ должен влезть ЦЕЛИКОМ, иначе пример вредный
        answer_len = len(full_ids) - len(user_ids)
        if answer_len <= 0:
            return None
        if len(user_ids) + answer_len > self.max_len:
            return None

        labels = [-100] * len(user_ids) + full_ids[len(user_ids):]
        if all(l == -100 for l in labels):
            return None
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


random.Random(42).shuffle(raw)
split = int(len(raw) * 0.9)
train_ds = PokerDataset(raw[:split], tokenizer, MAX_SEQ_LEN)
val_ds = PokerDataset(raw[split:], tokenizer, MAX_SEQ_LEN)

print(f"   Train: {len(train_ds)} (отброшено {train_ds.skipped})")
print(f"   Val:   {len(val_ds)} (отброшено {val_ds.skipped})")
if train_ds.skipped or val_ds.skipped:
    print("   ⚠️  Отброшенные не влезали в MAX_SEQ_LEN — именно они давали loss=nan.")
if len(val_ds) == 0:
    raise SystemExit("❌ Валидация пуста. Увеличь MAX_SEQ_LEN.")

n_labels = sum(1 for l in train_ds[0]["labels"] if l != -100)
print(f"   Обучаемых токенов в первом примере: {n_labels}")

# ---------- аргументы ----------
args = make_training_args(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_steps=50,              # только steps, без ratio
    max_grad_norm=1.0,
    fp16=True,
    gradient_checkpointing=True,
    optim="adamw_torch",
    logging_steps=10,
    logging_strategy="steps",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=False,
    report_to=[],
    remove_unused_columns=False,
    seed=42,
    dataloader_pin_memory=False,
    dataloader_num_workers=0,     # важно для Windows
)

collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt"
)

trainer_kwargs, dropped = filter_kwargs(Trainer, dict(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=val_ds,
    data_collator=collator, tokenizer=tokenizer,
))
if dropped:
    print(f"⚠️  Trainer: отброшены аргументы: {dropped}")
trainer = Trainer(**trainer_kwargs)

# ---------- запуск ----------
print("\n🚀 ОБУЧЕНИЕ")
print(f"   шагов: {len(train_ds) * NUM_EPOCHS // (BATCH_SIZE * GRAD_ACCUM)}")
t0 = time.time()
try:
    torch.cuda.empty_cache()
    result = trainer.train()
    print(f"\n✅ Готово за {(time.time()-t0)/60:.1f} мин")
    print(f"   train_loss: {result.metrics.get('train_loss')}")
    ev = trainer.evaluate()
    print(f"   eval_loss:  {ev.get('eval_loss')}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"💾 Сохранено в {OUTPUT_DIR}")
except KeyboardInterrupt:
    trainer.save_model(OUTPUT_DIR + "_interrupted")
    print(f"\n⏸️  Прервано. Сохранено в {OUTPUT_DIR}_interrupted")