from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Загрузка базовой модели
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    dtype=torch.float16,
    device_map="auto"
)

# Загрузка LoRA адаптера
model = PeftModel.from_pretrained(base_model, "./dataset_21_11/poker_model_v1/checkpoint-12")
tokenizer = AutoTokenizer.from_pretrained("./dataset_21_11/poker_model_v1/checkpoint-12")

# Тестовый промпт
test_prompt = """Ты эксперт по Техасскому Холдему (NLH Cash Game, 100bb стек).
Ты на BTN. Рука: A♠ K♥. UTG рейзит 3bb. Твои действия?"""

inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)