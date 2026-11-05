from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("YiPz/qwen3-4b-pokerbench-grpo")
tokenizer = AutoTokenizer.from_pretrained("YiPz/qwen3-4b-pokerbench-grpo")

messages = [
    {"role": "system", "content": "You are an expert poker player. Analyze the situation and provide your action."},
    {"role": "user", "content": "I have A♠ K♠ on the button, the flop is Q♠ J♠ 2♦. What should I do?"}
]

# 1. Получаем BatchEncoding
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)

# 2. Извлекаем тензор input_ids
# Если inputs - это BatchEncoding (словарь), берём по ключу
if hasattr(inputs, "input_ids"):
    input_ids = inputs.input_ids
elif isinstance(inputs, dict):
    input_ids = inputs["input_ids"]
else:
    input_ids = inputs # На случай, если вернётся тензор

# 3. Передаём чистый тензор в generate
outputs = model.generate(input_ids, max_new_tokens=512, temperature=0.6, top_p=0.95)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
