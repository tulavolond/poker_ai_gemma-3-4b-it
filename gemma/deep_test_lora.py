"""test_lora.py - Проверка обученной LoRA."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
LORA = "./train_2/dataset_2000_v2"

print("Загрузка...")
tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE, dtype=torch.float16, device_map="auto", trust_remote_code=True
)

if not __import__("os").path.exists(LORA):
    print(f"❌ LoRA не найдена: {LORA}")
    exit(1)

model = PeftModel.from_pretrained(model, LORA)
model.eval()
print(f"✅ LoRA загружена из {LORA}")

# Тот же prompt, что в датасете
prompt = """--- PREFLOP ---
PROMPT:
 [СТАДИЯ: ПРЕФЛОП]

=== ИСТОРИЯ РАЗДАЧИ ===
Hero (BTN) OPEN_RAISE, Villain (BB) CALL

=== ТВОИ КАРТЫ ===
Q♣, J♦

=== АНАЛИЗ СИСТЕМЫ ===
Рука: Бродвей (сила 55/100)
Диапазон оппонента: AA,KK,QQ,JJ,TT,99,88,77,66,55,AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,KQs,KJs,KTs,K9s,QJs,QTs,Q9s,JTs,J9s,T9s,98s,87s,76s,65s,AKo,AQo,AJo,ATo,KQo,KJo,QJo,JTo

=== КОНТЕКСТ ===
Позиция: BTN
Стек: $1000, Банк: $15
Колл: $10

Ответь СТРОГО:
АНАЛИЗ: [логика с числами]
ДЕЙСТВИЕ: [FOLD/CALL/RAISE]
СУММА: [число]"""

messages = [{"role": "user", "content": prompt}]
chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(chat, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs, max_new_tokens=150, do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )

response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("\n" + "=" * 60)
print("ОТВЕТ МОДЕЛИ: PREFLOP")
print("=" * 60)
print(response)




# ***********************************--- FLOP ---******************************************
print("=" * 60)
print("=" * 60)
# Тот же prompt, что в датасете
prompt = """--- FLOP ---
PROMPT:
 [СТАДИЯ: FLOP]

=== ИСТОРИЯ РАЗДАЧИ ===
Hero (MP) OPEN_RAISE, Villain (BTN) CALL. Flop: Опп CHECK

=== ТВОИ КАРТЫ ===
A♦, K♦

=== БОРД (3 карт) ===
4♥ 6♦ 2♠

=== АНАЛИЗ СИСТЕМЫ ===
Комбинация: Старшая карта (сила 5/100)
Эквити против диапазона: 51%
Текстура: DRY (безопасный)
Диапазон оппонента: AA,KK,QQ,JJ,TT,99,88,77,AKs,AQs,AJs,ATs,A9s,KQs,KJs,KTs,QJs,QTs,JTs,AKo,AQo,AJo,ATo,KQo

=== ДЕЙСТВИЕ ОППОНЕНТА ===
Опп CHECK

=== КОНТЕКСТ ===
Позиция: MP
Стек: $1000, Банк: $60
Колл: $0

Ответь СТРОГО:
АНАЛИЗ: [логика с числами]
ДЕЙСТВИЕ: [FOLD/CHECK/CALL/BET/RAISE]
СУММА: [число]"""

messages = [{"role": "user", "content": prompt}]
chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(chat, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs, max_new_tokens=150, do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )

response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("\n" + "=" * 60)
print("ОТВЕТ МОДЕЛИ: FLOP")
print("=" * 60)
print(response)



# ***********************************--- TURN ---******************************************
print("=" * 60)
print("=" * 60)
# Тот же prompt, что в датасете
prompt = """--- TURN ---
PROMPT:
 [СТАДИЯ: TURN]

=== ИСТОРИЯ РАЗДАЧИ ===
Hero (MP) OPEN_RAISE, Villain (BTN) CALL. Turn: Опп CHECK

=== ТВОИ КАРТЫ ===
Q♦, 10♦

=== БОРД (4 карт) ===
A♣ 5♣ 4♣ J♦

=== АНАЛИЗ СИСТЕМЫ ===
Комбинация: Старшая карта (сила 5/100)
Эквити против диапазона: 6%
Текстура: Средний
Диапазон оппонента: AA,KK,QQ,JJ,TT,99,88,77,AKs,AQs,AJs,ATs,A9s,KQs,KJs,KTs,QJs,QTs,JTs,AKo,AQo,AJo,ATo,KQo

=== ДЕЙСТВИЕ ОППОНЕНТА ===
Опп CHECK

=== КОНТЕКСТ ===
Позиция: MP
Стек: $1000, Банк: $100
Колл: $0

Ответь СТРОГО:
АНАЛИЗ: [логика с числами]
ДЕЙСТВИЕ: [FOLD/CHECK/CALL/BET/RAISE]
СУММА: [число]"""

messages = [{"role": "user", "content": prompt}]
chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(chat, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs, max_new_tokens=150, do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )

response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("\n" + "=" * 60)
print("ОТВЕТ МОДЕЛИ: TURN")
print("=" * 60)
print(response)




# ***********************************--- RIVER ---******************************************
print("=" * 60)
print("=" * 60)
# Тот же prompt, что в датасете
prompt = """--- RIVER ---
PROMPT:
 [СТАДИЯ: RIVER]

=== ИСТОРИЯ РАЗДАЧИ ===
Hero (BTN) CALL_3BET, Villain (SB) 3-BET. River: Опп CHECK

=== ТВОИ КАРТЫ ===
K♥, J♥

=== БОРД (5 карт) ===
6♣ A♣ A♥ 4♦ 6♥

=== АНАЛИЗ СИСТЕМЫ ===
Комбинация: Две пары (сила 25/100)
Эквити против диапазона: 36%
Текстура: WET (опасный, много дро)
Диапазон оппонента: AA,KK,QQ,JJ,TT,99,AKs,AQs,AJs,KQs,AKo,AQo

=== ДЕЙСТВИЕ ОППОНЕНТА ===
Опп CHECK

=== КОНТЕКСТ ===
Позиция: BTN
Стек: $1000, Банк: $150
Колл: $0

Ответь СТРОГО:
АНАЛИЗ: [логика с числами]
ДЕЙСТВИЕ: [FOLD/CHECK/CALL/BET/RAISE]
СУММА: [число]"""

messages = [{"role": "user", "content": prompt}]
chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(chat, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs, max_new_tokens=150, do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )

response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("\n" + "=" * 60)
print("ОТВЕТ МОДЕЛИ: RIVER")
print("=" * 60)
print(response)

