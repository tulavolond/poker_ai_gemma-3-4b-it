import os
import torch
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

# Загружаем HF_TOKEN из .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
torch.cuda.set_per_process_memory_fraction(0.8, 0)
# Путь к датасету (рядом со скриптом)
SCRIPT_DIR = Path(__file__).parent
DATASET_PATH = SCRIPT_DIR / "poker_test_v1.jsonl"

print(f"Ищем датасет по пути: {DATASET_PATH}")
if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Файл датасета не найден: {DATASET_PATH}")

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

print("Загрузка токенизатора...")
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Загрузка модели (float16 для GTX 1070)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    token=HF_TOKEN
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("Загрузка датасета...")
dataset = load_dataset(
    "json",
    data_files=str(DATASET_PATH),
    token=HF_TOKEN
)

print(f"Всего примеров в датасете: {len(dataset['train'])}")

# Функция форматирования с обрезкой до 1024 токенов
def formatting_func(examples):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    tokenized = tokenizer(texts, max_length=1024, truncation=True, padding=False)
    texts_truncated = tokenizer.batch_decode(tokenized["input_ids"], skip_special_tokens=False)
    return {"text": texts_truncated}

print("Форматирование датасета через chat_template...")
dataset_formatted = dataset["train"].map(formatting_func, batched=True)

print(f"Пример отформатированного текста:\n{dataset_formatted[0]['text'][:200]}...")

training_args = TrainingArguments(
    output_dir=str(SCRIPT_DIR / "poker_model_v1"),
    num_train_epochs=1,  # Увеличиваем до 20 эпох для маленького датасета
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,  # Уменьшаем до 2, чтобы было больше шагов
    learning_rate=2e-4,
    fp16=True,
    logging_steps=1,  # Логируем КАЖДЫЙ шаг
    save_strategy="epoch",
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    warmup_steps=5,
    lr_scheduler_type="cosine",
    report_to="none",
    disable_tqdm=False  # Включаем tqdm прогресс-бар
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_formatted,
    args=training_args,
)

print("Начало обучения...")
print(f"Всего шагов обучения: {trainer.state.max_steps}")
trainer.train()

print("Сохранение модели...")
save_dir = SCRIPT_DIR / "poker_model_v1_final"
trainer.save_model(str(save_dir))
tokenizer.save_pretrained(str(save_dir))
print(f"Готово! Модель сохранена в {save_dir}")