"""
train_stage1.py - Обучение Этапа 1 (префлоп) на GTX 1070 8GB.
ИСПРАВЛЕНО: борьба с переобучением.
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
# КОНФИГУРАЦИЯ - ИСПРАВЛЕНО ПРОТИВ ПЕРЕОБУЧЕНИЯ
# ============================================================
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_FILE = "dataset_combined.json"
OUTPUT_DIR = "./lora_preflop_stage_4"  # новая папка
MAX_SEQ_LEN = 512
BATCH_SIZE = 1
GRAD_ACCUM = 8

# ИСПРАВЛЕНО: уменьшили learning rate в 10 раз
LEARNING_RATE = 2e-5  # было 2e-4
NUM_EPOCHS = 3

# ИСПРАВЛЕНО: уменьшили LoRA
LORA_R = 16           # было 32
LORA_ALPHA = 32       # было 64
LORA_DROPOUT = 0.15   # было 0.05

SAVE_STEPS = 200

print("=" * 60)
print("  ЭТАП 1: ОБУЧЕНИЕ ПРЕФЛОП (ИСПРАВЛЕНО)")
print("=" * 60)

d=json.load(open('dataset_combined.json')) 
print(f'Всего: {len(d)}') 
print(f'Уникальных completion: {len(set(x["completion"] for x in d))}')

if not torch.cuda.is_available():
    print("❌ CUDA не доступна!")
    exit(1)

print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ Память: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print()

# ============================================================
# 1. ЗАГРУЗКА МОДЕЛИ
# ============================================================
print("📦 Загрузка модели...")
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
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

model.gradient_checkpointing_enable()
model.config.use_cache = False

print("✅ Модель загружена")

# ============================================================
# 2. НАСТРОЙКА LoRA - ИСПРАВЛЕНО
# ============================================================
print("🔧 Настройка LoRA адаптера...")

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,  # ИСПРАВЛЕНО: увеличен dropout
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

for name, param in model.named_parameters():
    if "lora" not in name.lower():
        param.requires_grad = False

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"✅ Обучаемых параметров: {trainable_params:,} из {total_params:,}")
print(f"   ({trainable_params / total_params * 100:.2f}%)")
print(f"   LoRA: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
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

# ИСПРАВЛЕНО: проверяем уникальность completion
unique_completions = len(set(ex["completion"] for ex in examples))
print(f"   Уникальных completion: {unique_completions} ({unique_completions/len(examples)*100:.1f}%)")

if unique_completions < len(examples) * 0.3:
    print("⚠️  ВНИМАНИЕ: мало уникальных completion! Модель переобучится.")
    print("   Рекомендую увеличить датасет или добавить аугментацию.")

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

sample = train_dataset[0]
print(f"📝 Пример токенизации:")
print(f"   Длина: {len(sample['input_ids'])} токенов")
print()

# ============================================================
# 5. НАСТРОЙКИ ТРЕНИРОВКИ - ИСПРАВЛЕНО
# ============================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    
    # ИСПРАВЛЕНО: уменьшили learning rate
    learning_rate=LEARNING_RATE,
    
    # ИСПРАВЛЕНО: добавили weight decay
    weight_decay=0.01,
    
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,  # ИСПРАВЛЕНО: больше warmup
    
    fp16=True,
    gradient_checkpointing=True,
    optim="adamw_torch",
    
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    
    # ИСПРАВЛЕНО: ранняя остановка
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    save_total_limit=2,
    report_to=[],
    remove_unused_columns=True,
    seed=42,
    dataloader_pin_memory=False,
)

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
print("🚀 НАЧИНАЕМ ОБУЧЕНИЕ")
print("=" * 60)
print(f"Эпох: {NUM_EPOCHS}")
print(f"Batch size: {BATCH_SIZE} × {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Weight decay: 0.01")
print(f"LoRA dropout: {LORA_DROPOUT}")
print(f"Примерно времени: {len(train_dataset) * NUM_EPOCHS / (BATCH_SIZE * GRAD_ACCUM) * 1.5 / 60:.0f} минут")
print()

start_time = time.time()

try:
    torch.cuda.empty_cache()
    train_result = trainer.train()
    
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 60)
    print(f"⏱️  Время: {elapsed / 60:.1f} минут")
    print(f"📉 Train loss: {train_result.metrics['train_loss']:.4f}")

    print("\n📊 Оценка на validation set...")
    eval_result = trainer.evaluate()
    print(f"📉 Eval loss: {eval_result['eval_loss']:.4f}")

    # Проверяем переобучение
    if train_result.metrics['train_loss'] < 0.01 and eval_result['eval_loss'] > 1.0:
        print("\n⚠️  ВНИМАНИЕ: модель переобучилась!")
        print(f"   Train loss: {train_result.metrics['train_loss']:.4f}")
        print(f"   Eval loss: {eval_result['eval_loss']:.4f}")
        print("   Рекомендую: увеличить датасет или уменьшить epochs")

    print(f"\n💾 Сохранение LoRA в {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Сохранено!")

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
            "lora_dropout": lora_config.lora_dropout,
            "target_modules": list(lora_config.target_modules),
        }
    }
    with open(os.path.join(OUTPUT_DIR, "training_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    print()
    print("=" * 60)
    print("🎉 УСПЕХ!")
    print("=" * 60)
    print(f"📁 Веса LoRA: {OUTPUT_DIR}/")
    print(f"📊 Метрики: {OUTPUT_DIR}/training_metadata.json")

except torch.cuda.OutOfMemoryError as e:
    print()
    print("=" * 60)
    print("❌ НЕ ХВАТИЛО ПАМЯТИ GPU")
    print("=" * 60)
    print(f"Ошибка: {e}")
    print("\n💡 РЕШЕНИЕ: уменьшите MAX_SEQ_LEN до 384")

except KeyboardInterrupt:
    print()
    print("=" * 60)
    print("⏸️  ОБУЧЕНИЕ ПРЕРВАНО")
    print("=" * 60)
    print("Сохраняю прогресс...")
    trainer.save_model(OUTPUT_DIR + "_interrupted")
    print(f"✅ Сохранено в {OUTPUT_DIR}_interrupted/")