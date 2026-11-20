"""
train_combined_lora.py - Дообучение LoRA на смешанном датасете (префлоп + флоп).
КЛЮЧЕВОЕ: правильно маскируем prompt в labels, чтобы модель училась только на completion.
"""
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
import os

# ============ КОНФИГУРАЦИЯ ============
# BASE_MODEL = "meta-llama/Llama-2-7b-hf"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EXISTING_LORA = "./lora_preflop_stage_2"  # путь к префлоп-LoRA (если есть)
OUTPUT_DIR = "./poker_combined_lora"
DATASET_FILE = "dataset_combined.json"
MAX_LENGTH = 768

# MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # 1.5B, влезает в 8GB
# DATASET_FILE = "dataset_preflop_2.jsonl"
# OUTPUT_DIR = "./lora_preflop_stage_3"
# MAX_SEQ_LEN = 512      # Префлоп-промпты короткие # Было 2048. Стало 512 Этого полностью хватит для префлоп-комбинаций!
BATCH_SIZE = 1          # Маленький, чтобы влезть в память
GRAD_ACCUM = 8          # Эффективный batch = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
SAVE_STEPS = 200 # # Было 100. Делать проверку и сохранение в два раза реже
# УВЕЛИЧЕНО: больше параметров LoRA
LORA_R = 32             # было 16, теперь 32
LORA_ALPHA = 64         # было 32, теперь 64

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
)

# TRAINING_ARGS = TrainingArguments(
#     output_dir=OUTPUT_DIR,
#     num_train_epochs=3,
#     per_device_train_batch_size=2,
#     gradient_accumulation_steps=8,
#     learning_rate=2e-5,
#     warmup_ratio=0.03,
#     logging_steps=10,
#     save_steps=500,
#     save_total_limit=3,
#     fp16=True,
#     optim="adamw_torch",
#     lr_scheduler_type="cosine",
#     remove_unused_columns=False,
#     report_to="none",
#     dataloader_pin_memory=True,
# )
TRAINING_ARGS = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,   # По 1 примеру на валидацию, чтобы экономить VRAM
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_steps=50,               
    fp16=True,                      
    gradient_checkpointing=True,    
    optim="adamw_torch",
    logging_steps=10,

    # --- ГЛАВНОЕ ИСПРАВЛЕНИЕ СКОРОСТИ ---
    eval_strategy="epoch",          # Проверяем eval_loss строго 1 раз в конце эпохи, а не каждые 100 шагов
    save_strategy="epoch",          # Сохраняем чекпоинт тоже раз в эпоху
    # save_steps=SAVE_STEPS,
    load_best_model_at_end=False,   # Отключаем постоянную перезапись весов в памяти Windows
    # ------------------------------------

    save_total_limit=1,             
    report_to=[],                   # Метрики будут красиво выводиться в консоль
    remove_unused_columns=True,     
    seed=42,
    dataloader_pin_memory=False    
)

def load_dataset(file_path):
    """Загружает датасет из JSON."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if len(data) == 0:
        raise ValueError(f"Датасет {file_path} пустой!")

    print(f"✅ Загружено {len(data)} примеров из {file_path}")
    print(f"   Пример prompt: {data[0]['prompt'][:100]}...")
    print(f"   Пример completion: {data[0]['completion']}")

    return data


def prepare_dataset(data, tokenizer):
    """
    Подготавливает датасет с правильной маскировкой.
    Модель учится ТОЛЬКО на completion, prompt замаскирован через -100.
    """
    def tokenize_fn(example):
        # Токенизируем prompt БЕЗ special tokens в конце
        prompt_enc = tokenizer(
            example["prompt"],
            truncation=True,
            max_length=MAX_LENGTH - 50,
            add_special_tokens=True,
        )
        prompt_ids = prompt_enc["input_ids"]

        # Токенизируем completion + eos
        completion_enc = tokenizer(
            example["completion"],
            truncation=True,
            max_length=50,
            add_special_tokens=False,
        )
        completion_ids = completion_enc["input_ids"] + [tokenizer.eos_token_id]

        # Объединяем
        input_ids = prompt_ids + completion_ids
        attention_mask = [1] * len(input_ids)

        # КЛЮЧЕВОЕ: маскируем prompt в labels
        labels = [-100] * len(prompt_ids) + completion_ids

        # Обрезаем до MAX_LENGTH
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    dataset = Dataset.from_list(data)
    tokenized = dataset.map(
        tokenize_fn,
        remove_columns=dataset.column_names,
        desc="Токенизация"
    )

    # Проверка
    sample = tokenized[0]
    non_masked = sum(1 for l in sample["labels"] if l != -100)
    print(f"✅ Токенизировано {len(tokenized)} примеров")
    print(f"   Токенов в примере: {len(sample['input_ids'])}")
    print(f"   Обучающихся токенов (completion): {non_masked}")
    print(f"   Замаскированных токенов (prompt): {len(sample['labels']) - non_masked}")

    if non_masked == 0:
        raise ValueError("Все labels замаскированы! Модель нечему учиться.")

    return tokenized


def main():
    print("🎯 Загрузка модели и токенизатора...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Загружаем существующий LoRA или создаём новый
    if EXISTING_LORA and os.path.exists(EXISTING_LORA):
        print(f"📦 Загрузка существующего LoRA из {EXISTING_LORA}...")
        model = PeftModel.from_pretrained(model, EXISTING_LORA)
        # ВАЖНО: при дообучении существующего LoRA не создаём новый
    else:
        print("🆕 Создание нового LoRA...")
        model = get_peft_model(model, LORA_CONFIG)
        model.print_trainable_parameters()

    # Загружаем датасет
    print(f"\n📊 Загрузка датасета из {DATASET_FILE}...")
    data = load_dataset(DATASET_FILE)

    # Подготавливаем
    print("\n🔧 Подготовка датасета...")
    train_dataset = prepare_dataset(data, tokenizer)

    # Trainer
    print("\n🚀 Запуск обучения...")
    trainer = Trainer(
        model=model,
        args=TRAINING_ARGS,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Сохраняем
    print(f"\n💾 Сохранение модели в {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n✅ Обучение завершено!")
    print(f"Модель сохранена в: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()