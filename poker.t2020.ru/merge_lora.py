import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "google/gemma-3-4b-it"  # Или "google/gemma-3-4b" в зависимости от того, что тюнили
lora_folder = "./poker_gemma_lora"
output_folder = "./poker_gemma_lora_merged"

print("Загрузка базовой модели и токенизатора...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16, # или float16
    device_map="cpu" # используем CPU, чтобы не забить VRAM, если нужно — поменяйте на "cuda"
)

print("Загрузка LoRA адаптера...")
model = PeftModel.from_pretrained(base_model, lora_folder)

print("Слияние весов...")
model = model.merge_and_unload()

print(f"Сохранение объединенной модели в {output_folder}...")
model.save_pretrained(output_folder)
tokenizer.save_pretrained(output_folder)
print("Готово!")
