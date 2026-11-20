"""
train_stage_flop_v2.py - Обучение LoRA на флоп-датасете.
ИСПРАВЛЕНО: убран tokenizer из Trainer, добавлена кодировка UTF-8.
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
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EXISTING_LORA = "./lora_preflop_stage_4"  # путь к префлоп-LoRA
OUTPUT_DIR = "./lora_flop_stage_1"
DATASET_FILE = "dataset_combined.json"
MAX_LENGTH = 512

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.15,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
)

TRAINING_ARGS = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    warmup_steps=50,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,
    fp16=True,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    remove_unused_columns=False,
    report_to="none",
)


def load_dataset(file_path):
    """Загружает датасет из JSON с правильной кодировкой."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if len(data) == 0:
        raise ValueError(f"Датасет {file_path} пустой!")

    print(f"✅ Загружено {len(data)} примеров из {file_path}")
    
    # Проверяем уникальность
    unique_completions = len(set(ex["completion"] for ex in data))
    print(f"   Уникальных completion: {unique_completions} ({unique_completions/len(data)*100:.1f}%)")
    
    # Показываем примеры
    print(f"\n   Пример prompt (первые 200 символов):")
    print(f"   {data[0]['prompt'][:200]}...")
    print(f"\n   Пример completion:")
    print(f"   {data[0]['completion']}")

    return data


def prepare_dataset(data, tokenizer):
    """Подготавливает датасет с правильной маскировкой."""
    def tokenize_fn(example):
        # Токенизируем prompt
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

        # Маскируем prompt в labels
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
    print(f"\n✅ Токенизировано {len(tokenized)} примеров")
    print(f"   Токенов в примере: {len(sample['input_ids'])}")
    print(f"   Обучающихся токенов (completion): {non_masked}")
    print(f"   Замаскированных токенов (prompt): {len(sample['labels']) - non_masked}")

    if non_masked == 0:
        raise ValueError("Все labels замаскированы! Модель нечему учиться.")

    # Разделяем на train/val
    split_idx = int(len(tokenized) * 0.9)
    train_dataset = tokenized.select(range(split_idx))
    val_dataset = tokenized.select(range(split_idx, len(tokenized)))
    
    print(f"   Train: {len(train_dataset)} примеров")
    print(f"   Validation: {len(val_dataset)} примеров")

    return train_dataset, val_dataset


def main():
    print("🎯 Загрузка модели и токенизатора...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Загружаем существующий LoRA или создаём новый
    if EXISTING_LORA and os.path.exists(EXISTING_LORA):
        print(f"📦 Загрузка существующего LoRA из {EXISTING_LORA}...")
        model = PeftModel.from_pretrained(model, EXISTING_LORA)
    else:
        print("🆕 Создание нового LoRA...")
        model = get_peft_model(model, LORA_CONFIG)
        model.print_trainable_parameters()

    # Загружаем датасет
    print(f"\n📊 Загрузка датасета из {DATASET_FILE}...")
    data = load_dataset(DATASET_FILE)

    # Подготавливаем
    print("\n🔧 Подготовка датасета...")
    train_dataset, val_dataset = prepare_dataset(data, tokenizer)

    # Trainer - ИСПРАВЛЕНО: убран tokenizer
    print("\n🚀 Запуск обучения...")
    trainer = Trainer(
        model=model,
        args=TRAINING_ARGS,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # tokenizer=tokenizer,  # УБРАНО: этот параметр больше не поддерживается
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