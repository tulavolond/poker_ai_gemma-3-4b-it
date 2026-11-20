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

# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # 1.5B, влезает в 8GB
DATASET_FILE = "dataset_preflop.json"
OUTPUT_DIR = "./lora_preflop_stage1"
MAX_SEQ_LEN = 1024      # Префлоп-промпты короткие
BATCH_SIZE = 1          # Маленький, чтобы влезть в память
GRAD_ACCUM = 4          # Эффективный batch = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
SAVE_STEPS = 100

print("=" * 60)
print("  ЭТАП 1: ОБУЧЕНИЕ ПРЕФЛОП")
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
    dtype=torch.float16,   # float16 вместо fp32 = 2x экономии
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

# Экономия памяти через gradient checkpointing
model.gradient_checkpointing_enable()
model.config.use_cache = False  # Отключаем KV-cache во время обучения

print("✅ Модель загружена")

# ============================================================
# 2. НАСТРОЙКА LoRA
# ============================================================
print("🔧 Настройка LoRA адаптера...")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
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
    print("   Запустите сначала: python dataset_preflop.py")
    exit(1)

with open(DATASET_FILE, 'r', encoding='utf-8') as f:
    examples = json.load(f)

print(f"✅ Загружено {len(examples)} примеров")

# Разбиваем на train (90%) и validation (10%)
split_idx = int(len(examples) * 0.9)
train_examples = examples[:split_idx]
val_examples = examples[split_idx:]
print(f"   Train: {len(train_examples)} примеров")
print(f"   Validation: {len(val_examples)} примеров")
print()

# ============================================================
# 4. КЛАСС ДАТАСЕТА
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

        # Формируем диалог: user + assistant
        messages = [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["completion"]}
        ]

        # Применяем шаблон чата Qwen
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Токенизируем
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]

        # Считаем длину user-части (не учитываем в loss)
        user_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}],
            tokenize=False,
            add_generation_prompt=True
        )
        user_len = len(self.tokenizer(user_text, add_special_tokens=False)["input_ids"])
        # print(f"   User часть: ~{user_len} токенов (игнорируется при обучении)")

        # Маскируем user-часть в labels (чтобы модель училась только ответам)
        labels = input_ids.clone()
        labels[:user_len] = -100

        # Маскируем padding в labels
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

train_dataset = PokerDataset(train_examples, tokenizer, MAX_SEQ_LEN)
val_dataset = PokerDataset(val_examples, tokenizer, MAX_SEQ_LEN)

# Тест: первый пример
sample = train_dataset[0]
print(f"📝 Пример токенизации:")
print(f"   Длина: {len(sample['input_ids'])} токенов")
# print(f"   User часть: ~{user_len} токенов (игнорируется при обучении)")
print()

# ============================================================
# 5. НАСТРОЙКИ ТРЕНИРОВКИ
# ============================================================
# training_args = TrainingArguments(
#     output_dir=OUTPUT_DIR,
#     num_train_epochs=NUM_EPOCHS,
#     per_device_train_batch_size=BATCH_SIZE,
#     per_device_eval_batch_size=BATCH_SIZE,
#     gradient_accumulation_steps=GRAD_ACCUM,
#     learning_rate=LEARNING_RATE,
#     weight_decay=0.01,
#     warmup_ratio=0.05,
#     lr_scheduler_type="cosine",
#     fp16=True,                      # float16 обучение
#     gradient_checkpointing=True,    # Экономия памяти
#     optim="adamw_torch",
#     logging_steps=10,
#     save_steps=SAVE_STEPS,
#     save_total_limit=2,             # Храним только 2 последних чекпоинта
#     eval_strategy="steps",
#     eval_steps=SAVE_STEPS,
#     load_best_model_at_end=True,
#     metric_for_best_model="eval_loss",
#     greater_is_better=False,
#     report_to=[],                   # Отключаем WandB/TensorBoard
#     remove_unused_columns=False,
#     seed=42,
#     dataloader_pin_memory=False,    # Важно для Windows
# )

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    
    # --- ИСПРАВЛЕНИЕ: Используем стандартный warmup_steps вместо ratio ---
    lr_scheduler_type="cosine",
    warmup_steps=100,               # 100 шагов на плавный разогрев (измените, если нужно)
    # ---------------------------------------------------------------------

    fp16=True,                      
    gradient_checkpointing=True,    
    optim="adamw_torch",
    logging_steps=10,
    eval_strategy="steps",
    save_strategy="steps",          
    eval_steps=SAVE_STEPS,
    save_steps=SAVE_STEPS,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,             
    report_to="none",               
    remove_unused_columns=True,     
    seed=42,
    dataloader_pin_memory=False    
)



data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
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
print("🚀 НАЧИНАЕМ ОБУЧЕНИЕ")
print("=" * 60)
print(f"Эпох: {NUM_EPOCHS}")
print(f"Batch size: {BATCH_SIZE} × {GRAD_ACCUM} (grad accum) = {BATCH_SIZE * GRAD_ACCUM} эффективный")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Примерно времени: {len(train_dataset) * NUM_EPOCHS / (BATCH_SIZE * GRAD_ACCUM) * 1.5 / 60:.0f} минут")
print()

start_time = time.time()

try:
    train_result = trainer.train()
    
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 60)
    print(f"⏱️  Время: {elapsed / 60:.1f} минут")
    print(f"📉 Train loss: {train_result.metrics['train_loss']:.4f}")

    # Оценка на validation
    print("\n📊 Оценка на validation set...")
    eval_result = trainer.evaluate()
    print(f"📉 Eval loss: {eval_result['eval_loss']:.4f}")

    # ============================================================
    # 7. СОХРАНЕНИЕ
    # ============================================================
    print(f"\n💾 Сохранение LoRA в {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Сохранено!")

    # Сохраняем метаданные
    metadata = {
        "model": MODEL_ID,
        "stage": "preflop",
        "num_examples": len(examples),
        "train_loss": train_result.metrics['train_loss'],
        "eval_loss": eval_result['eval_loss'],
        "training_time_minutes": elapsed / 60,
        "lora_config": {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "target_modules": list(lora_config.target_modules),
        }
    }
    with open(os.path.join(OUTPUT_DIR, "training_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    print()
    print("=" * 60)
    print("🎉 УСПЕХ! Модель обучена.")
    print("=" * 60)
    print(f"📁 Веса LoRA: {OUTPUT_DIR}/")
    print(f"📊 Метрики: {OUTPUT_DIR}/training_metadata.json")
    print()
    print("Следующий шаг: интеграция с сервером (напишу после проверки)")

except torch.cuda.OutOfMemoryError as e:
    print()
    print("=" * 60)
    print("❌ НЕ ХВАТИЛО ПАМЯТИ GPU")
    print("=" * 60)
    print(f"Ошибка: {e}")
    print()
    print("💡 РЕШЕНИЕ: уменьшите параметры в начале файла:")
    print("   MAX_SEQ_LEN = 768   (вместо 1024)")
    print("   BATCH_SIZE = 1")
    print("   Или переключитесь на 4-бит квантизацию")
    print()

except KeyboardInterrupt:
    print()
    print("=" * 60)
    print("⏸️  ОБУЧЕНИЕ ПРЕРВАНО пользователем")
    print("=" * 60)
    print("Сохраняю текущий прогресс...")
    trainer.save_model(OUTPUT_DIR + "_interrupted")
    print(f"✅ Сохранено в {OUTPUT_DIR}_interrupted/")