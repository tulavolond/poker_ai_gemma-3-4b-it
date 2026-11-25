# merge_master.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path

#    python llama.cpp/convert_hf_to_gguf.py ./qwen25-poker-merged --outfile poker-master.gguf --outtype f16

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "./dataset_21_11/qwen25-master-lora" # Путь к обученному адаптеру
OUT_PATH = "qwen25-poker-merged" # Итоговая папка с моделью

print("⏳ Загрузка базовой модели в память...")
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="cpu")
tok = AutoTokenizer.from_pretrained(BASE_MODEL)

print("⏳ Загрузка LoRA и слияние весов (merge_and_unload)...")
model = PeftModel.from_pretrained(base, LORA_PATH)
model = model.merge_and_unload()

print(f"💾 Сохранение итоговой модели в {OUT_PATH}...")
model.save_pretrained(OUT_PATH, safe_serialization=True)
tok.save_pretrained(OUT_PATH)
print("✅ ГОТОВО! У тебя есть одна универсальная покерная модель.")