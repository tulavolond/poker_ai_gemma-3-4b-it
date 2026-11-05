"""
Одноразовый скрипт: сливает LoRA с базовой моделью.
Запусти один раз, потом используй объединённую модель.
"""
import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_ID = "google/gemma-3-4b-it"
LORA_PATH = "./poker_gemma3_lora"
MERGED_PATH = "./poker_gemma3_merged"

print("Загрузка базовой модели...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="cpu",          # На CPU для мержинга
    token=HF_TOKEN,
    low_cpu_mem_usage=True,
)

print("Подключение LoRA...")
model = PeftModel.from_pretrained(model, LORA_PATH)

print("Мержинг адаптера с моделью...")
model = model.merge_and_unload()

print(f"Сохранение в {MERGED_PATH}...")
model.save_pretrained(MERGED_PATH, safe_serialization=True)

print("Сохранение токенизатора...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
tokenizer.save_pretrained(MERGED_PATH)

print(f"✅ Готово! Объединённая модель сохранена в {MERGED_PATH}")