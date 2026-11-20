"""
train_stage_flop_v4.py - Обучение LoRA на флоп-датасете.
+ Ctrl+C: ПОЛНЫЙ чекпоинт (веса + оптимизатор + scheduler + state) в *_interrupted
+ авто-resume при следующем запуске (выбирается самый свежий по global_step)
+ подробная информация о GPU
+ оценка времени + живой ETA по фактической скорости
+ сохранение метаданных (реальный конфиг адаптера, а не LORA_CONFIG)
Сначала проверяем, влезают ли токены: check_dataset.py, корректируем и запускаем.
"""
import json
import math
import os
import glob
import time

import torch
import transformers
import peft
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Меньше фрагментации VRAM (частая причина OOM "при свободной памяти"). Работает на Windows.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

# ============ КОНФИГУРАЦИЯ ============
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EXISTING_LORA = "./dataset_2000_v2"          # путь к префлоп-LoRA (для дообучения)
OUTPUT_DIR = "./train_2/dataset_2000_v2"
DATASET_FILE = "./train_2/dataset_2000_v2.json"
MAX_SEQ_LEN = 768

# prompt: max_length = 712 - 250 = 462
# completion: max_length = 250 (+1 на EOS)

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
    num_train_epochs=6,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,     # после проверки памяти можно поднять до 2-4 (коллатор паддит)
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},  # корректно работает с замороженной базой
    learning_rate=2e-5,
    warmup_steps=50,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="steps",
    save_steps=50,                    # полный чекпоинт каждые 50 шагов
    save_total_limit=10,
    load_best_model_at_end=False,
    fp16=True,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    remove_unused_columns=False,
    report_to=[],
    seed=42,                          # фиксируем явно: от сида зависит порядок данных при resume
    dataloader_num_workers=0,         # Windows: воркеры через spawn дорогие, для bs=1 не нужны
)


# ============ УТИЛИТЫ ============
def print_gpu_info():
    """Подробная информация о GPU."""
    if not torch.cuda.is_available():
        print("❌ CUDA недоступна!")
        return False

    print("=" * 60)
    print("🖥️  ИНФОРМАЦИЯ О GPU")
    print("=" * 60)
    print(f"   Название: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"   Архитектура: {props.major}.{props.minor}")
    print(f"   CUDA: {torch.version.cuda}")

    free_mem, total_mem = torch.cuda.mem_get_info(0)
    used_mem = total_mem - free_mem
    print(f"   Всего памяти: {total_mem / 1e9:.2f} GB")
    print(f"   Свободно:     {free_mem / 1e9:.2f} GB")
    print(f"   Используется: {used_mem / 1e9:.2f} GB ({used_mem / total_mem * 100:.1f}%)")
    print("=" * 60)
    return True


def find_latest_checkpoint(output_dir):
    """
    Самый свежий чекпоинт по global_step из trainer_state.json.
    Смотрит и checkpoint-*, и *_interrupted — кто дальше по шагам, тот и выигрывает.
    Папки без trainer_state.json (недописанные) игнорируются.
    Возвращает (путь, шаг) или (None, 0).
    """
    candidates = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    candidates.append(output_dir + "_interrupted")

    best_path, best_step = None, 0
    for path in candidates:
        state_file = os.path.join(path, "trainer_state.json")
        if not os.path.isfile(state_file):
            continue
        try:
            with open(state_file, encoding="utf-8") as f:
                step = int(json.load(f)["global_step"])
        except (OSError, ValueError, KeyError):
            continue  # битый/недописанный стейт — не кандидат
        if step > best_step:
            best_path, best_step = path, step
    return best_path, best_step


def estimate_time(num_train, num_epochs, batch_size, grad_accum, sec_per_step=15.0):
    """Оценка времени. Trainer округляет шаги вверх. 15 с/шаг — порядок для GTX 1070; точный ETA покажет EtaCallback."""
    steps_per_epoch = math.ceil(math.ceil(num_train / batch_size) / grad_accum)
    total_steps = steps_per_epoch * num_epochs
    return total_steps, total_steps * sec_per_step


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


class EtaCallback(TrainerCallback):
    """Живой ETA по фактической скорости (учитывает resume)."""
    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()
        self.s0 = state.global_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:   # печатаем только на train-логах, не на eval
            return
        done = state.global_step - self.s0
        if done > 0 and state.max_steps:
            sec_per_step = (time.time() - self.t0) / done
            left = (state.max_steps - state.global_step) * sec_per_step
            print(f"⏱️ {sec_per_step:.1f} с/шаг, осталось ~{format_time(left)}")


# ============ ЗАГРУЗКА ДАННЫХ ============
def load_dataset(file_path):
    """Загружает датасет из JSON."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if len(data) == 0:
        raise ValueError(f"Датасет {file_path} пустой!")

    print(f"✅ Загружено {len(data)} примеров из {file_path}")
    unique_completions = len(set(ex["completion"] for ex in data))
    print(f"   Уникальных completion: {unique_completions} ({unique_completions / len(data) * 100:.1f}%)")
    print(f"\n   Пример prompt (первые 200 символов):")
    print(f"   {data[0]['prompt'][:200]}...")
    print(f"\n   Пример completion:")
    print(f"   {data[0]['completion']}")
    return data


def prepare_dataset(data, tokenizer):
    """Подготавливает датасет с правильной маскировкой (учим только completion)."""
    prompt_limit = MAX_SEQ_LEN - 250
    truncated_prompts = 0
    truncated_completions = 0

    def tokenize_fn(example):
        nonlocal truncated_prompts, truncated_completions

        # 1. Prompt — оставляем 250 токенов под completion
        prompt_enc = tokenizer(
            example["prompt"],
            truncation=True,
            max_length=prompt_limit,
            add_special_tokens=True,
        )
        prompt_ids = prompt_enc["input_ids"]
        if len(prompt_ids) >= prompt_limit:
            truncated_prompts += 1  # обрезка справа съедает хвост промпта!

        # 2. Completion — до 250 токенов + EOS (без двойного EOS)
        completion_enc = tokenizer(
            example["completion"],
            truncation=True,
            max_length=250,
            add_special_tokens=False,
        )
        completion_ids = completion_enc["input_ids"]
        if len(completion_ids) >= 250:
            truncated_completions += 1
        if completion_ids[-1:] != [tokenizer.eos_token_id]:
            completion_ids = completion_ids + [tokenizer.eos_token_id]

        # 3. Склеиваем, prompt замаскирован
        input_ids = prompt_ids + completion_ids
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_ids) + completion_ids

        # 4. Обрезаем до MAX_SEQ_LEN (на всякий случай)
        return {
            "input_ids": input_ids[:MAX_SEQ_LEN],
            "attention_mask": attention_mask[:MAX_SEQ_LEN],
            "labels": labels[:MAX_SEQ_LEN],
        }

    dataset = Dataset.from_list(data)
    tokenized = dataset.map(
        tokenize_fn,
        remove_columns=dataset.column_names,
        desc="Токенизация",
        load_from_cache_file=False,  # токенизация быстрая, зато счётчики обрезки считаются каждый запуск
    )

    if truncated_prompts:
        print(f"⚠️ ВНИМАНИЕ: {truncated_prompts} prompt обрезаны (хвост потерян)! "
              f"Подними MAX_SEQ_LEN или сократи промпты (см. check_dataset.py)")
    if truncated_completions:
        print(f"⚠️ ВНИМАНИЕ: {truncated_completions} completion обрезаны до 250 токенов!")

    sample = tokenized[0]
    non_masked = sum(1 for l in sample["labels"] if l != -100)
    print(f"\n✅ Токенизировано {len(tokenized)} примеров")
    print(f"   Токенов в примере: {len(sample['input_ids'])}")
    print(f"   Обучающихся токенов (completion): {non_masked}")
    print(f"   Замаскированных токенов (prompt): {len(sample['labels']) - non_masked}")

    if non_masked == 0:
        raise ValueError("Все labels замаскированы! Модель нечему учиться.")
    if non_masked < 100:
        print(f"⚠️ ВНИМАНИЕ: completion всего {non_masked} токенов — модель может не выучить формат")

    # Опционально: перемешать перед сплитом, если хвост файла нерепрезентативен.
    # Меняет состав train/val относительно прошлой стадии — включай осознанно.
    # tokenized = tokenized.shuffle(seed=42)

    split_idx = int(len(tokenized) * 0.9)
    train_dataset = tokenized.select(range(split_idx))
    val_dataset = tokenized.select(range(split_idx, len(tokenized)))
    print(f"   Train: {len(train_dataset)} примеров")
    print(f"   Validation: {len(val_dataset)} примеров")
    return train_dataset, val_dataset


# ============ ГЛАВНАЯ ФУНКЦИЯ ============
def main():
    start_time = time.time()

    # 1. Информация о GPU
    if not print_gpu_info():
        return

    # Ограничиваем GPU память: 80% от 8GB ≈ 6.4GB (если OOM при свободной памяти — подними до 0.9)
    get_memory_gpu = 0.8
    torch.cuda.set_per_process_memory_fraction(get_memory_gpu, 0)
    print(f"✅ GPU память ограничена до {get_memory_gpu * 100:.0f}% "
          f"({torch.cuda.get_device_properties(0).total_memory * get_memory_gpu / 1e9:.1f} GB)")
    print("=" * 60)

    # 2. Модель и токенизатор
    print("\n🎯 Загрузка модели и токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        token=HF_TOKEN,
        dtype=torch.float16,
        device_map={"": 0},   # всё на GPU 0: "auto" мог бы молча выгрузить слои в RAM и замедлить в разы
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False  # несовместим с gradient checkpointing

    # 3. Ищем чекпоинт для продолжения
    latest_checkpoint, done_steps = find_latest_checkpoint(OUTPUT_DIR)
    resume = False

    if latest_checkpoint:
        print(f"\n🔄 Найден чекпоинт: {latest_checkpoint} (шаг {done_steps})")
        model = PeftModel.from_pretrained(model, latest_checkpoint, is_trainable=True)
        resume = True
    elif os.path.isdir(OUTPUT_DIR + "_interrupted"):
        # Старая папка с весами, но без trainer_state.json — берём веса, но шаги с нуля
        print(f"\n⚠️  Найден прерванный чекпоинт без trainer_state.json: {OUTPUT_DIR}_interrupted")
        print(f"   Загружаю веса, но обучение начнётся с шага 0...")
        model = PeftModel.from_pretrained(model, OUTPUT_DIR + "_interrupted", is_trainable=True)
    elif EXISTING_LORA and os.path.exists(EXISTING_LORA):
        print(f"\n📦 Загрузка существующего LoRA из {EXISTING_LORA}...")
        # is_trainable=True ОБЯЗАТЕЛЬНО: иначе адаптер заморожен и backward упадёт
        model = PeftModel.from_pretrained(model, EXISTING_LORA, is_trainable=True)
    else:
        print("\n🆕 Создание нового LoRA...")
        model = get_peft_model(model, LORA_CONFIG)

    # Обучаемые параметры — в fp32 (база остаётся fp16, autocast считает в fp16).
    # В fp16 обновления весом lr=2e-5 округляются в ноль. Если уже fp32 — no-op.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    trainable_dtype = next(p.dtype for p in model.parameters() if p.requires_grad)
    print(f"   dtype обучаемых параметров: {trainable_dtype}")

    # 4. Датасет (всегда, независимо от resume)
    print(f"\n📊 Загрузка датасета из {DATASET_FILE}...")
    data = load_dataset(DATASET_FILE)

    print("\n🔧 Подготовка датасета...")
    train_dataset, val_dataset = prepare_dataset(data, tokenizer)

    # 5. Оценка времени
    total_steps, est_seconds = estimate_time(
        len(train_dataset),
        TRAINING_ARGS.num_train_epochs,
        TRAINING_ARGS.per_device_train_batch_size,
        TRAINING_ARGS.gradient_accumulation_steps,
        sec_per_step=15.0,  # грубая оценка для GTX 1070; точный ETA покажет EtaCallback
    )
    print(f"\n⏱️  ОЦЕНКА ВРЕМЕНИ:")
    print(f"   Всего шагов: {total_steps}")
    if resume:
        est_seconds = max(total_steps - done_steps, 0) * 15.0
        print(f"   Пройдено шагов: {done_steps}, осталось: {max(total_steps - done_steps, 0)}")
    print(f"   Примерное время: {format_time(est_seconds)} (грубо; точный ETA появится после первых логов)")

    # 6. Trainer
    print("\n🚀 Запуск обучения... (Ctrl+C один раз — сохранит полный чекпоинт и выйдет)")
    trainer = Trainer(
        model=model,
        args=TRAINING_ARGS,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        # паддит input_ids/attention_mask и labels (-100); НЕ DataCollatorForLanguageModeling —
        # тот перезаписал бы labels и размаскировал prompt
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100),
        callbacks=[EtaCallback()],
    )

    # 7. Обучение с обработкой прерывания
    try:
        if resume:
            print(f"   🔄 Продолжение из {latest_checkpoint}...")
            train_result = trainer.train(resume_from_checkpoint=latest_checkpoint)
        else:
            train_result = trainer.train()

        elapsed = time.time() - start_time
        train_loss = train_result.metrics.get("train_loss")  # может не быть при resume "в самый конец"

        print()
        print("=" * 60)
        print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print("=" * 60)
        print(f"⏱️  Фактическое время: {elapsed / 60:.1f} минут")
        print(f"📉 Train loss: {train_loss:.4f}" if train_loss is not None else "📉 Train loss: n/a")

        print("\n📊 Оценка на validation set...")
        eval_result = trainer.evaluate()
        print(f"📉 Eval loss: {eval_result['eval_loss']:.4f}")

        print(f"\n💾 Сохранение LoRA в {OUTPUT_DIR}...")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"✅ Сохранено!")

        # Метаданные: реальный конфиг адаптера (при from_pretrained LORA_CONFIG игнорируется!)
        peft_cfg = model.peft_config[model.active_adapter]
        metadata = {
            "model": BASE_MODEL,
            "stage": "flop",
            "dataset_file": DATASET_FILE,
            "num_examples": len(data),
            "train_examples": len(train_dataset),
            "val_examples": len(val_dataset),
            "train_loss": float(train_loss) if train_loss is not None else None,
            "eval_loss": float(eval_result['eval_loss']),
            "training_time_minutes": elapsed / 60,
            "max_length": MAX_SEQ_LEN,
            "lora_config": {
                "r": peft_cfg.r,
                "lora_alpha": peft_cfg.lora_alpha,
                "lora_dropout": peft_cfg.lora_dropout,
                "target_modules": sorted(peft_cfg.target_modules or []),
            },
            "training_args": {
                "learning_rate": TRAINING_ARGS.learning_rate,
                "num_epochs": TRAINING_ARGS.num_train_epochs,
                "batch_size": TRAINING_ARGS.per_device_train_batch_size,
                "grad_accum": TRAINING_ARGS.gradient_accumulation_steps,
                "seed": TRAINING_ARGS.seed,
            },
            "resumed": resume,
            "resume_from": latest_checkpoint if resume else None,
            "existing_lora": EXISTING_LORA,
            "global_step": trainer.state.global_step,
            "versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "peft": peft.__version__,
            },
        }
        with open(os.path.join(OUTPUT_DIR, "training_metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print()
        print("=" * 60)
        print("🎉 УСПЕХ! Модель обучена.")
        print("=" * 60)
        print(f"📁 Веса LoRA: {OUTPUT_DIR}/")
        print(f"📊 Метрики: {OUTPUT_DIR}/training_metadata.json")
        print("ℹ️  Для НОВОЙ стадии смени OUTPUT_DIR (или убери checkpoint-*),")
        print("   иначе следующий запуск 'продолжит' уже завершённое обучение.")

    except KeyboardInterrupt:
        print()
        print("=" * 60)
        print("⏸️  ОБУЧЕНИЕ ПРЕРВАНО пользователем")
        print("=" * 60)
        print("Сохраняю ПОЛНЫЙ чекпоинт (не жми Ctrl+C повторно!)...")

        interrupted_dir = OUTPUT_DIR + "_interrupted"
        try:
            trainer.save_model(interrupted_dir)
            tokenizer.save_pretrained(interrupted_dir)
            if trainer.optimizer is not None:
                torch.save(trainer.optimizer.state_dict(), os.path.join(interrupted_dir, "optimizer.pt"))
            if trainer.lr_scheduler is not None:
                torch.save(trainer.lr_scheduler.state_dict(), os.path.join(interrupted_dir, "scheduler.pt"))
            scaler = getattr(trainer.accelerator, "scaler", None)
            if scaler is not None:
                torch.save(scaler.state_dict(), os.path.join(interrupted_dir, "scaler.pt"))
            # trainer_state.json пишем ПОСЛЕДНИМ: его наличие = признак полного чекпоинта.
            # Если сохранение оборвётся на середине — папка не будет признана валидной.
            trainer.state.save_to_json(os.path.join(interrupted_dir, "trainer_state.json"))
            print(f"✅ Полный чекпоинт сохранён в {interrupted_dir}/ (шаг {trainer.state.global_step})")
            print(f"   Чтобы продолжить: просто запусти скрипт снова — он найдёт его автоматически.")
        except Exception as e:
            print(f"❌ Ошибка сохранения: {e}")
            print(f"  Resume подхватит последний обычный checkpoint-* из {OUTPUT_DIR}/")

    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        print()
        print("=" * 60)
        print("❌ НЕ ХВАТИЛО ПАМЯТИ GPU")
        print("=" * 60)
        print(f"Ошибка: {e}")
        print()
        print("💡 РЕШЕНИЕ:")
        print(f"   MAX_SEQ_LEN = {MAX_SEQ_LEN // 2}   (вместо {MAX_SEQ_LEN})")
        print(f"   Или подними get_memory_gpu до 0.9, если OOM случился при свободной памяти")
        print(f"   BATCH_SIZE = 1 (уже минимальный)")

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