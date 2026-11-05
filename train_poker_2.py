import os
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
# 1. Загрузка модели и токенизатора (в 4 битах для экономии памяти)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "google/gemma-3-4b-it", # Или gemma-2-4b-it
    max_seq_length = 1024, # Достаточно для покерного промпта
    dtype = torch.bfloat16, # Обязательно bfloat16 для Gemma
    load_in_4bit = True,
    token = "HF_TOKEN",
)

# 2. Добавляем LoRA адаптеры (дообучаем только их, база заморожена)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Ранг матрицы (8, 16 или 32. Чем больше, тем умнее, но медленнее)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = "unsloth", # Экономит память
    random_state = 3407,
)

# 3. Загрузка датасета
# Ваш файл должен называться poker_dataset.jsonl и лежать рядом
dataset = load_dataset("json", data_files="poker_dataset.jsonl", split="train")

# 4. Настройка тренера
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text", # Unsloth сам обработает формат messages
    max_seq_length = 1024,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2, # Увеличивайте, если позволяет VRAM
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 200, # Количество шагов обучения. Для начала хватит 200-500
        learning_rate = 2e-4, # Низкий LR, чтобы не "сломать" базовые знания
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs_poker_gemma",
    ),
)

# 5. Запуск обучения
print("🚀 Начинаем дообучение...")
trainer.train()

# 6. Сохранение только LoRA-адаптеров (весят ~20-50 МБ)
model.save_pretrained("poker_gemma_lora")
tokenizer.save_pretrained("poker_gemma_lora")
print("✅ Обучение завершено, модель сохранена в папку 'poker_gemma_lora'")