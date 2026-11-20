# diagnose.py
import torch, os, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
LORA = "./train_2/dataset_2000_v2"

print("1. Проверка файлов LoRA:")
print(f"   Папка существует: {os.path.exists(LORA)}")
if os.path.exists(LORA):
    for f in os.listdir(LORA):
        p = os.path.join(LORA, f)
        print(f"   {f}: {os.path.getsize(p)} байт")

print("\n2. Проверка trainer_state.json:")
ts = os.path.join(LORA, "trainer_state.json")
if os.path.exists(ts):
    with open(ts) as f:
        s = json.load(f)
    print(f"   global_step: {s.get('global_step')}")
    print(f"   epoch: {s.get('epoch')}")
    logs = s.get('log_history', [])
    print(f"   Последние 3 loss: {[e.get('loss') for e in logs[-3:]]}")
else:
    print("   ❌ НЕТ trainer_state.json — обучение не завершилось")

print("\n3. Загрузка модели...")
tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    BASE, dtype=torch.float16, device_map="auto", trust_remote_code=True
)

if os.path.exists(LORA):
    model = PeftModel.from_pretrained(model, LORA)
    print("   ✅ LoRA загружена")
else:
    print("   ❌ LoRA не загружена — будет базовая Qwen")
model.eval()

print("\n4. Тест на ТОЧНОМ prompt из датасета:")
ds_path = "./train_2/dataset_2000_v2.json"
if os.path.exists(ds_path):
    with open(ds_path, encoding='utf-8') as f:
        ds = json.load(f)
    sample = next((x for x in ds if x['stage'] == 'preflop'), ds[0])
    prompt = sample['prompt']
    expected = sample['completion']
    print(f"   Prompt (первые 200): {prompt[:200]}")
    print(f"   Ожидаемый completion: {expected[:200]}")
else:
    prompt = "[СТАДИЯ: ПРЕФЛОП]\n\n=== ТВОИ КАРТЫ ===\nA♥, 5♥\n\n=== КОНТЕКСТ ===\nПозиция: BTN\nСтек: $1000, Банк: $15\nКолл: $10\n\nОтветь СТРОГО:\nАНАЛИЗ: ...\nДЕЙСТВИЕ: ...\nСУММА: ..."
    expected = "?"

messages = [{"role": "user", "content": prompt}]
chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(chat, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=150, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id)

resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"\n   Ответ модели:\n{resp}")
print(f"\n5. ВЕРДИКТ:")
if "Ты профессиональный" in resp or "пара тузов" in resp.lower():
    print("   ❌ LoRA НЕ РАБОТАЕТ — модель отвечает как базовая Qwen")
elif "АНАЛИЗ:" in resp and "ДЕЙСТВИЕ:" in resp:
    print("   ✅ LoRA РАБОТАЕТ — модель отвечает в правильном формате")
else:
    print("   ⚠️ Непонятно — покажите вывод выше")