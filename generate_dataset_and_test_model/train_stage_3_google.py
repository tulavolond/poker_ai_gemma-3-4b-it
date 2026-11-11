"""
train_stage1.py - Обучение Этапа 1 (префлоп) на GTX 1070 8GB.
Используем Qwen2.5-1.5B-Instruct + LoRA + float16.

Запуск: python train_stage1.py
"""
import os
import json
import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# ============================================================
# КОНФИГУРАЦИЯ (ОПТИМИЗИРОВАНО ДЛЯ СКОРОСТИ)
# ============================================================
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  
DATASET_FILE = "dataset_preflop_3.json"
OUTPUT_DIR = "./lora_preflop_stage_3"
MAX_SEQ_LEN = 512       # ОПТИМИЗИРОВАНО: 512 токенов полностью хватает для префлопа
BATCH_SIZE = 2          
GRAD_ACCUM = 8          # Эффективный batch = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3

print("=" * 60)
print("  ЭТАП 1: ОБУЧЕНИЕ ПРЕФЛОП (БЫСТРЫЙ ПЕРЕЗАПУСК)")
print("=" * 60)

# Проверка GPU
if not torch.cuda.is_available():
    print("❌ CUDA не доступна! Убедитесь что у вас NVIDIA GPU")
    exit(1)

print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ Память: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print()

# ============================================================
# 1. ЗАГРУЗКА МОДЕЛИ
# ============================================================
print("📦 Загрузка модели (это займёт 1-2 минуты)...")
print(f"   Модель: {MODEL_ID}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    padding_side="right",
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    dtype=torch.float16,   
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

model.gradient_checkpointing_enable()
model.config.use_cache = False  

print("✅ Модель загружена")

# ============================================================
# 2. НАСТРОЙКА LoRA
# ============================================================
print("🔧 Настройка LoRA адаптера...")

lora_config = LoraConfig(
    r=32,              
    lora_alpha=64, 
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

# Замораживаем всё, кроме LoRA
for name, param in model.named_parameters():
    if "lora" not in name.lower():
        param.requires_grad = False

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"✅ Обучаемых параметров: {trainable_params:,} из {total_params:,}")
print(f"   ({trainable_params / total_params * 100:.2f}%)")
print()

# ============================================================
# 3. ЗАГРУЗКА ДАТАСЕТА
# ============================================================
print(f"📚 Загрузка датасета из {DATASET_FILE}...")

if not os.path.exists(DATASET_FILE):
    print(f"❌ Файл {DATASET_FILE} не найден!")
    exit(1)

with open(DATASET_FILE, 'r', encoding='utf-8') as f:
    examples = json.load(f)

print(f"✅ Загружено {len(examples)} примеров")

split_idx = int(len(examples) * 0.9)
train_examples = examples[:split_idx]
val_examples = examples[split_idx:]
print(f"   Train: {len(train_examples)} примеров")
print(f"   Validation: {len(val_examples)} примеров")
print()

# ============================================================
# 4. КЛАСС ДАТАСЕТА (БЕЗ RETURNING TENSORS ВНУТРИ)
# ============================================================
class PokerDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]

        messages = [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["completion"]}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        user_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}],
            tokenize=False,
            add_generation_prompt=True
        )
        user_len = len(self.tokenizer(user_text, add_special_tokens=False)["input_ids"])

        labels = list(input_ids)
        for i in range(len(labels)):
            if i < user_len or attention_mask[i] == 0:
                labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

train_dataset = PokerDataset(train_examples, tokenizer, MAX_SEQ_LEN)
val_dataset = PokerDataset(val_examples, tokenizer, MAX_SEQ_LEN)

# ============================================================
# 5. НАСТРОЙКИ ТРЕНИРОВКИ (УСКОРЕННЫЕ)
# ============================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,   # Экономия памяти при валидации
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_steps=50,               
    fp16=True,                      
    gradient_checkpointing=True,    
    optim="adamw_torch",
    logging_steps=10,
    
    # Считаем и сохраняем строго раз в эпоху — это уберет простои
    eval_strategy="epoch",          
    save_strategy="epoch",          
    load_best_model_at_end=False,   
    
    save_total_limit=1,             
    report_to=[],                   # Вывод логов напрямую в консоль
    remove_unused_columns=True,     
    seed=42,
    dataloader_pin_memory=False    # Оптимально для Windows 10
)

# Сборщик данных эффективно собирает списки в PyTorch тензоры
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    pad_to_multiple_of=None,
    return_tensors="pt",  
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# ============================================================
# 6. ЗАПУСК ОБУЧЕНИЯ
# ============================================================
print("=" * 60)
print("🚀 НАЧИНАЕМ СВЕРХБЫСТРОЕ ОБУЧЕНИЕ")
print("=" * 60)

start_time = time.time()

# Принудительная очистка кэша видеокарты перед стартом
if torch.cuda.is_available():
    torch.cuda.empty_cache()

try:
    train_result = trainer.train()
    
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 60)
    print(f"⏱️  Время: {elapsed / 60:.1f} минут")
    print(f"📉 Train loss: {train_result.metrics['train_loss']:.4f}")

    print("\n📊 Финальная оценка на validation set...")
    eval_result = trainer.evaluate()
    print(f"📉 Eval loss: {eval_result['eval_loss']:.4f}")

    print(f"\n💾 Сохранение LoRA в {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Сохранено успешно!")

except torch.cuda.OutOfMemoryError as e:
    print("\n❌ НЕ ХВАТИЛО ПАМЯТИ GPU. Попробуйте закрыть браузер и другие программы.")
    print(f"Ошибка: {e}")
