"""
train_stage_flop_v2.py - Обучение LoRA на флоп-датасете.
ИСПРАВЛЕНО: правильный resume + частые чекпоинты.
"""
import json
import time
import os
import glob
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset

# ============ КОНФИГУРАЦИЯ ============
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EXISTING_LORA = "./lora_preflop_stage_4"
OUTPUT_DIR = "./lora_flop_stage_44"
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
    
    # ИСПРАВЛЕНО: сохраняем чекпоинты каждые 100 шагов
    save_strategy="steps",
    save_steps=100,
    save_total_limit=5,  # храним 5 последних
    
    load_best_model_at_end=False,  # отключено для скорости
    fp16=True,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    remove_unused_columns=False,
    report_to="none",
)


def print_gpu_info():
    """Подробная информация о GPU."""
    if not torch.cuda.is_available():
        print("❌ CUDA недоступна!")
        return False

    print("=" * 60)
    print("🖥️  ИНФОРМАЦИЯ О GPU")
    print("=" * 60)
    print(f"   Название: {torch.cuda.get_device_name(0)}")
    print(f"   Архитектура: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")
    print(f"   CUDA: {torch.version.cuda}")

    total_mem = torch.cuda.get_device_properties(0).total_memory
    free_mem, total_mem_now = torch.cuda.mem_get_info(0)
    used_mem = total_mem_now - free_mem

    print(f"   Всего памяти: {total_mem / 1e9:.2f} GB")
    print(f"   Свободно:     {free_mem / 1e9:.2f} GB")
    print(f"   Используется: {used_mem / 1e9:.2f} GB ({used_mem / total_mem * 100:.1f}%)")
    print("=" * 60)
    return True


def find_latest_checkpoint(output_dir):
    """Находит последний чекпоинт."""
    # Ищем в основной папке
    checkpoints = []
    if os.path.exists(output_dir):
        checkpoints.extend(glob.glob(os.path.join(output_dir, "checkpoint-*")))
    
    # Ищем в _interrupted папке
    interrupted_dir = output_dir + "_interrupted"
    if os.path.exists(interrupted_dir):
        # Проверяем, есть ли там adapter_config.json (признак сохранённого LoRA)
        if os.path.exists(os.path.join(interrupted_dir, "adapter_config.json")):
            return interrupted_dir
    
    if not checkpoints:
        return None

    def get_step(path):
        try:
            return int(path.split("-")[-1])
        except ValueError:
            return -1

    checkpoints.sort(key=get_step, reverse=True)
    return checkpoints[0]


def estimate_time(num_train, num_epochs, batch_size, grad_accum, sec_per_step=0.5):
    """Оценка времени обучения."""
    steps_per_epoch = num_train // (batch_size * grad_accum)
    total_steps = steps_per_epoch * num_epochs
    total_seconds = total_steps * sec_per_step
    return total_steps, total_seconds


def format_time(seconds):
    """Форматирует секунды в читаемый вид."""
    if seconds < 60:
        return f"{seconds:.0f} сек"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} мин"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h} ч {m} мин"


def load_dataset(file_path):
    """Загружает датасет из JSON."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if len(data) == 0:
        raise ValueError(f"Датасет {file_path} пустой!")

    print(f"✅ Загружено {len(data)} примеров из {file_path}")

    unique_completions = len(set(ex["completion"] for ex in data))
    pct = unique_completions / len(data) * 100
    print(f"   Уникальных completion: {unique_completions} ({pct:.1f}%)")
    
    if pct < 10:
        print(f"   ⚠️  ВНИМАНИЕ: очень мало уникальных ответов!")
        print(f"   Модель выучит только {unique_completions} шаблонов.")
        print(f"   Рекомендую увеличить датасет или добавить разнообразие.")

    print(f"\n   Пример prompt (первые 200 символов):")
    print(f"   {data[0]['prompt'][:200]}...")
    print(f"\n   Пример completion:")
    print(f"   {data[0]['completion']}")

    return data


def prepare_dataset(data, tokenizer):
    """Подготавливает датасет с правильной маскировкой."""
    def tokenize_fn(example):
        prompt_enc = tokenizer(
            example["prompt"],
            truncation=True,
            max_length=MAX_LENGTH - 50,
            add_special_tokens=True,
        )
        prompt_ids = prompt_enc["input_ids"]

        completion_enc = tokenizer(
            example["completion"],
            truncation=True,
            max_length=50,
            add_special_tokens=False,
        )
        completion_ids = completion_enc["input_ids"] + [tokenizer.eos_token_id]

        input_ids = prompt_ids + completion_ids
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_ids) + completion_ids

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

    sample = tokenized[0]
    non_masked = sum(1 for l in sample["labels"] if l != -100)
    print(f"\n✅ Токенизировано {len(tokenized)} примеров")
    print(f"   Токенов в примере: {len(sample['input_ids'])}")
    print(f"   Обучающихся токенов (completion): {non_masked}")
    print(f"   Замаскированных токенов (prompt): {len(sample['labels']) - non_masked}")

    if non_masked == 0:
        raise ValueError("Все labels замаскированы! Модель нечему учиться.")

    split_idx = int(len(tokenized) * 0.9)
    train_dataset = tokenized.select(range(split_idx))
    val_dataset = tokenized.select(range(split_idx, len(tokenized)))

    print(f"   Train: {len(train_dataset)} примеров")
    print(f"   Validation: {len(val_dataset)} примеров")

    return train_dataset, val_dataset


def main():
    start_time = time.time()

    if not print_gpu_info():
        return

    print("\n🎯 Загрузка модели и токенизатора...")

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

    # Ищем чекпоинт для продолжения
    latest_checkpoint = find_latest_checkpoint(OUTPUT_DIR)
    resume = False

    if latest_checkpoint:
        print(f"\n🔄 Найден чекпоинт: {latest_checkpoint}")
        print(f"   Продолжаем обучение...")
        model = PeftModel.from_pretrained(model, latest_checkpoint, is_trainable=True)
        resume = True
    elif EXISTING_LORA and os.path.exists(EXISTING_LORA):
        print(f"\n📦 Загрузка существующего LoRA из {EXISTING_LORA}...")
        model = PeftModel.from_pretrained(model, EXISTING_LORA)
    else:
        print("\n🆕 Создание нового LoRA...")
        model = get_peft_model(model, LORA_CONFIG)
        model.print_trainable_parameters()

    print(f"\n📊 Загрузка датасета из {DATASET_FILE}...")
    data = load_dataset(DATASET_FILE)

    print("\n🔧 Подготовка датасета...")
    train_dataset, val_dataset = prepare_dataset(data, tokenizer)

    total_steps, est_seconds = estimate_time(
        len(train_dataset),
        TRAINING_ARGS.num_train_epochs,
        TRAINING_ARGS.per_device_train_batch_size,
        TRAINING_ARGS.gradient_accumulation_steps,
        sec_per_step=0.5
    )
    
    if resume:
        # Оцениваем оставшееся время
        step_num = int(latest_checkpoint.split("-")[-1]) if "checkpoint" in latest_checkpoint else 0
        remaining_steps = total_steps - step_num
        est_seconds = remaining_steps * 0.5
        print(f"\n⏱️  ОЦЕНКА ОСТАВШЕГОСЯ ВРЕМЕНИ:")
        print(f"   Осталось шагов: {remaining_steps}")
    else:
        print(f"\n⏱️  ОЦЕНКА ВРЕМЕНИ:")
        print(f"   Всего шагов: {total_steps}")
    
    print(f"   Примерное время: {format_time(est_seconds)}")

    print("\n🚀 Запуск обучения...")
    trainer = Trainer(
        model=model,
        args=TRAINING_ARGS,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    try:
        if resume and "checkpoint" in latest_checkpoint:
            print(f"   🔄 Продолжение с шага {latest_checkpoint.split('-')[-1]}...")
            train_result = trainer.train(resume_from_checkpoint=latest_checkpoint)
        else:
            train_result = trainer.train()

        elapsed = time.time() - start_time

        print()
        print("=" * 60)
        print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print("=" * 60)
        print(f"⏱️  Фактическое время: {elapsed / 60:.1f} минут")
        print(f"📉 Train loss: {train_result.metrics['train_loss']:.4f}")

        print("\n📊 Оценка на validation set...")
        eval_result = trainer.evaluate()
        print(f"📉 Eval loss: {eval_result['eval_loss']:.4f}")

        print(f"\n💾 Сохранение LoRA в {OUTPUT_DIR}...")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"✅ Сохранено!")

        metadata = {
            "model": BASE_MODEL,
            "stage": "flop",
            "num_examples": len(data),
            "train_examples": len(train_dataset),
            "val_examples": len(val_dataset),
            "train_loss": float(train_result.metrics['train_loss']),
            "eval_loss": float(eval_result['eval_loss']),
            "training_time_minutes": elapsed / 60,
            "max_length": MAX_LENGTH,
            "lora_config": {
                "r": LORA_CONFIG.r,
                "lora_alpha": LORA_CONFIG.lora_alpha,
                "lora_dropout": LORA_CONFIG.lora_dropout,
                "target_modules": list(LORA_CONFIG.target_modules),
            },
            "training_args": {
                "learning_rate": TRAINING_ARGS.learning_rate,
                "num_epochs": TRAINING_ARGS.num_train_epochs,
                "batch_size": TRAINING_ARGS.per_device_train_batch_size,
                "grad_accum": TRAINING_ARGS.gradient_accumulation_steps,
            },
            "resumed": resume,
        }
        with open(os.path.join(OUTPUT_DIR, "training_metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print()
        print("=" * 60)
        print("🎉 УСПЕХ! Модель обучена.")
        print("=" * 60)
        print(f"📁 Веса LoRA: {OUTPUT_DIR}/")
        print(f"📊 Метрики: {OUTPUT_DIR}/training_metadata.json")

    except KeyboardInterrupt:
        print()
        print("=" * 60)
        print("⏸️  ОБУЧЕНИЕ ПРЕРВАНО пользователем")
        print("=" * 60)
        print("Сохраняю текущий прогресс...")

        interrupted_dir = OUTPUT_DIR + "_interrupted"
        try:
            trainer.save_model(interrupted_dir)
            tokenizer.save_pretrained(interrupted_dir)
            print(f"✅ Сохранено в {interrupted_dir}/")
            print(f"   Чтобы продолжить: просто запусти скрипт снова")
        except Exception as e:
            print(f"❌ Ошибка сохранения: {e}")

    except torch.cuda.OutOfMemoryError as e:
        print()
        print("=" * 60)
        print("❌ НЕ ХВАТИЛО ПАМЯТИ GPU")
        print("=" * 60)
        print(f"Ошибка: {e}")
        print()
        print("💡 РЕШЕНИЕ:")
        print(f"   MAX_LENGTH = {MAX_LENGTH // 2}")

    except Exception as e:
        print()
        print("=" * 60)
        print("❌ ОШИБКА ОБУЧЕНИЯ")
        print("=" * 60)
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()