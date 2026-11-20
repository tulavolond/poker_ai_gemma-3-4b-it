"""
deep_test_lora.py - Прямая проверка LoRA (без сервера, без fix_response).
Показывает, что РЕАЛЬНО отвечает модель.
"""
import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ============================================================
# КОНФИГ
# ============================================================
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "./train_2/dataset_2000_v2"

# ============================================================
# ЗАГРУЗКА
# ============================================================
print("=" * 60)
print("  ПРОВЕРКА LoRA БЕЗ СЕРВЕРА")
print("=" * 60)

if not os.path.exists(LORA_PATH):
    print(f"❌ LoRA не найдена: {LORA_PATH}")
    exit(1)

print(f"📦 Базовая модель: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, dtype=torch.float16, device_map="auto", trust_remote_code=True
)
print(f"🎯 LoRA: {LORA_PATH}")
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()
print("✅ Загружено\n")

# ============================================================
# ТЕСТОВЫЕ ПРОМПТЫ (как в датасете)
# ============================================================
TESTS = [
    {
        "name": "Префлоп AA (должен RAISE)",
        "prompt": """[СТАДИЯ: ПРЕФЛОП]

=== ИСТОРИЯ РАЗДАЧИ ===
Hero (BTN) OPEN_RAISE, Villain (BB) CALL

=== ТВОИ КАРТЫ ===
A♠, A♥

=== АНАЛИЗ СИСТЕМЫ ===
Рука: Премиум-пара (сила 100/100)
Диапазон оппонента: AA,KK,QQ,JJ,TT,99,88,AKs,AQs,AJs,KQs,AKo,AQo

=== КОНТЕКСТ ===
Позиция: BTN
Стек: $1000, Банк: $15
Колл: $10

Ответь СТРОГО:
АНАЛИЗ: [логика с числами]
ДЕЙСТВИЕ: [FOLD/CALL/RAISE]
СУММА: [число]""",
        "expect": "RAISE",
    },
    {
        "name": "Префлоп 72o (должен FOLD)",
        "prompt": """[СТАДИЯ: ПРЕФЛОП]

=== ИСТОРИЯ РАЗДАЧИ ===
Hero (UTG) OPEN_RAISE, Villain (BB) CALL

=== ТВОИ КАРТЫ ===
7♣, 2♦

=== АНАЛИЗ СИСТЕМЫ ===
Рука: Мусор (сила 5/100)
Диапазон оппонента: AA,KK,QQ,JJ,TT,99,88,AKs,AQs,AJs,KQs,AKo,AQo

=== КОНТЕКСТ ===
Позиция: UTG
Стек: $1000, Банк: $15
Колл: $10

Ответь СТРОГО:
АНАЛИЗ: [логика с числами]
ДЕЙСТВИЕ: [FOLD/CALL/RAISE]
СУММА: [число]""",
        "expect": "FOLD",
    },
    {
        "name": "Флоп топ-пара (должен BET)",
        "prompt": """[СТАДИЯ: ФЛОП]

=== ИСТОРИЯ РАЗДАЧИ ===
Hero (BTN) OPEN_RAISE, Villain (BB) CALL. Flop: Опп CHECK

=== ТВОИ КАРТЫ ===
A♠, K♠

=== БОРД (3 карт) ===
K♦ 7♣ 2♥

=== АНАЛИЗ СИСТЕМЫ ===
Комбинация: Пара (сила 65/100)
Эквити против диапазона: 78%
Текстура: DRY (безопасный)
Диапазон оппонента: AA,KK,QQ,JJ,TT,99,88,AKs,AQs,AJs,KQs,AKo,AQo

=== ДЕЙСТВИЕ ОППОНЕНТА ===
Опп CHECK

=== КОНТЕКСТ ===
Позиция: BTN
Стек: $1000, Банк: $60
Колл: $0

Ответь СТРОГО:
АНАЛИЗ: [логика с числами]
ДЕЙСТВИЕ: [FOLD/CHECK/CALL/BET/RAISE]
СУММА: [число]""",
        "expect": "BET",
    },
    {
        "name": "Ривер стрит против чека (должен BET)",
        "prompt": """[СТАДИЯ: РИВЕР]

=== ИСТОРИЯ РАЗДАЧИ ===
Hero (BTN) OPEN_RAISE, Villain (BB) CALL. River: Опп CHECK

=== ТВОИ КАРТЫ ===
Q♥, 10♠

=== БОРД (5 карт) ===
K♦ J♦ 8♥ 9♥ 3♦

=== АНАЛИЗ СИСТЕМЫ ===
Комбинация: Стрит (сила 75/100)
Эквити против диапазона: 85%
Текстура: Средний
Диапазон оппонента: AA,KK,QQ,JJ,TT,99,88,AKs,AQs,AJs,KQs,AKo,AQo

=== ДЕЙСТВИЕ ОППОНЕНТА ===
Опп CHECK

=== КОНТЕКСТ ===
Позиция: BTN
Стек: $1000, Банк: $150
Колл: $0

Ответь СТРОГО:
АНАЛИЗ: [логика с числами]
ДЕЙСТВИЕ: [FOLD/CHECK/CALL/BET/RAISE]
СУММА: [число]""",
        "expect": "BET",
    },
    {
        "name": "Ривер слабая рука против ставки (должен FOLD)",
        "prompt": """[СТАДИЯ: РИВЕР]

=== ИСТОРИЯ РАЗДАЧИ ===
Hero (BTN) OPEN_RAISE, Villain (BB) CALL. River: Опп BET $100

=== ТВОИ КАРТЫ ===
7♣, 2♦

=== БОРД (5 карт) ===
K♦ J♦ 8♥ 9♥ 3♦

=== АНАЛИЗ СИСТЕМЫ ===
Комбинация: Старшая карта (сила 5/100)
Эквити против диапазона: 10%
Текстура: Средний
Диапазон оппонента: AA,KK,QQ,JJ,TT,99,88,AKs,AQs,AJs,KQs,AKo,AQo

=== ДЕЙСТВИЕ ОППОНЕНТА ===
Опп BET $100

=== КОНТЕКСТ ===
Позиция: BTN
Стек: $1000, Банк: $250
Колл: $100

Ответь СТРОГО:
АНАЛИЗ: [логика с числами]
ДЕЙСТВИЕ: [FOLD/CHECK/CALL/BET/RAISE]
СУММА: [число]""",
        "expect": "FOLD",
    },
]

# ============================================================
# ЗАПУСК
# ============================================================
print("=" * 60)
print("  ТЕСТЫ")
print("=" * 60)

passed = 0
total = len(TESTS)

for i, test in enumerate(TESTS, 1):
    print(f"\n{'='*60}")
    print(f"ТЕСТ {i}/{total}: {test['name']}")
    print(f"{'='*60}")

    messages = [{"role": "user", "content": test["prompt"]}]
    chat = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        chat, return_tensors="pt", truncation=True, max_length=620
    ).to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        out[0][input_len:], skip_special_tokens=True
    ).strip()

    # Парсим
    action = None
    amount = 0
    for line in response.split("\n"):
        line = line.strip().upper()
        if line.startswith("ДЕЙСТВИЕ:") or line.startswith("ACTION:"):
            a = line.split(":", 1)[1].strip().split()[0]
            if a in ("FOLD", "CHECK", "CALL", "BET", "RAISE"):
                action = a
        elif line.startswith("СУММА:") or line.startswith("AMOUNT:"):
            try:
                amount = int(float(line.split(":", 1)[1].strip().replace("$", "")))
            except:
                pass

    print(f"ОТВЕТ МОДЕЛИ:\n{response}\n")
    print(f"Ожидалось: {test['expect']}")
    print(f"Получено:  {action}")
    print(f"Сумма:     {amount}")

    if action == test["expect"]:
        print("✅ PASS")
        passed += 1
    elif action is None:
        print("❌ FAIL — модель не дала ДЕЙСТВИЕ (недообучена или формат другой)")
    else:
        print(f"❌ FAIL — ожидалось {test['expect']}, получено {action}")

print(f"\n{'='*60}")
print(f"  ИТОГ: {passed}/{total} тестов пройдено")
print(f"{'='*60}")

if passed == total:
    print("🎉 LoRA РАБОТАЕТ — модель играет!")
elif passed >= total * 0.6:
    print("⚠️ LoRA частично работает — нужно дообучение")
else:
    print("❌ LoRA НЕ РАБОТАЕТ — модель не выучила датасет")
    print("   Причина: MAX_SEQ_LEN=512 обрезал completion до 51 токена")
    print("   Фикс: MAX_SEQ_LEN=620 (уже сделано) + переобучить")