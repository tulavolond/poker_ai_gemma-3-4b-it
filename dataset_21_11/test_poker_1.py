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


messages = [
    {"role": "system", "content": "Ты эксперт по Техасскому Холдему (NLH Cash Game, 100bb стек)."},
    {"role": "user", "content": "Ты на BTN. Рука: A♠ J♥. UTG рейзит 2bb. Твои действия?"}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)


# Тестовые промпты
test_prompts = [
    "Ты эксперт по Техасскому Холдему (NLH Cash Game, 100bb стек).\nТы на BTN. Рука: A♠ J♥. UTG рейзит 3bb. Твои действия?",
    "Ты эксперт по Техасскому Холдему (NLH Cash Game, 100bb стек).\nТы на BB. Рука: 7♣ 8♣. Флоп: 9♣ 2♣ K♦. Пот 5bb. Ты чек, оппонент бет 4bb. Твои действия?",
    "Что такое Блокер?"
]

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n{'='*60}\nТест {i}:\n{prompt}\n{'='*60}")
    
    # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.3,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)