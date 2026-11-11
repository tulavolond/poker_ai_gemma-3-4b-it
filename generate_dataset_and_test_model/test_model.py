"""
test_model.py - Проверка качества обученной модели.
Тестируем на НОВЫХ примерах, которых не было в обучении.

Запуск: python test_model.py
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from analyzer import build_deck, analyze_preflop, classify_preflop_hand
import random

# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# LORA_PATH = "./lora_preflop_stage1"
LORA_PATH = "./lora_preflop_stage2_interrupted"

print("=" * 60)
print("  ТЕСТ ОБУЧЕННОЙ МОДЕЛИ")
print("=" * 60)

# ============================================================
# 1. ЗАГРУЗКА МОДЕЛИ
# ============================================================
print("📦 Загрузка базовой модели...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

print("🔧 Загрузка LoRA адаптера...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

print(f"✅ Модель загружена на {model.device}")
print()

# ============================================================
# 2. ФУНКЦИЯ ИНФЕРЕНСА
# ============================================================
def generate_response(prompt, max_new_tokens=100):
    """Генерирует ответ модели."""
    messages = [{"role": "user", "content": prompt}]
    
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Детерминированный ответ
            temperature=1.0,
            repetition_penalty=1.1,
        )
    
    # Извлекаем только сгенерированную часть
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Чистим от мусора
    response = response.strip()
    for tok in ('<|im_end|>', '<|im_start|>', 'assistant', 'user'):
        response = response.replace(tok, '')
    
    return response.strip()


# ============================================================
# 3. ТЕСТОВЫЕ ПРИМЕРЫ
# ============================================================
print("=" * 60)
print("🧪 ТЕСТ 1: ПРЕМИУМ-РУКИ (должны всегда рейзить)")
print("=" * 60)

# Генерируем AA, KK, AKs
premium_hands = [
    ("A♠", "A♥"),  # AA
    ("K♠", "K♥"),  # KK
    ("A♠", "K♠"),  # AKs
]

for h1_str, h2_str in premium_hands:
    # Создаём карты
    from analyzer import Card, RANK_NAMES
    rank_map = {str(v): k for k, v in RANK_NAMES.items()}
    rank_map.update({str(i): i for i in range(2, 11)})
    
    h1 = Card(rank_map[h1_str[0]], h1_str[1])
    h2 = Card(rank_map[h2_str[0]], h2_str[1])
    
    # Формируем промпт
    analysis = analyze_preflop(
        hero_cards=[h1, h2],
        position="BTN (баттон)",
        action_before="Все сбросили до тебя",
        stack=1000,
        bb=10
    )
    
    print(f"\n🃏 Рука: {h1}, {h2}")
    response = generate_response(analysis["prompt"])
    print(f"🤖 Ответ модели:\n{response}")

print()
print("=" * 60)
print("🧪 ТЕСТ 2: МУСОРНЫЕ РУКИ (должны фолдить)")
print("=" * 60)

# Генерируем мусор: 72o, 83o, 94o
trash_hands = [
    ("7♠", "2♥"),  # 72o
    ("8♠", "3♥"),  # 83o
    ("9♠", "4♥"),  # 94o
]

for h1_str, h2_str in trash_hands:
    h1 = Card(rank_map[h1_str[0]], h1_str[1])
    h2 = Card(rank_map[h2_str[0]], h2_str[1])
    
    analysis = analyze_preflop(
        hero_cards=[h1, h2],
        position="UTG",
        action_before="Все сбросили до тебя",
        stack=1000,
        bb=10
    )
    
    print(f"\n🃏 Рука: {h1}, {h2}")
    response = generate_response(analysis["prompt"])
    print(f"🤖 Ответ модели:\n{response}")

print()
print("=" * 60)
print("🧪 ТЕСТ 3: СРЕДНИЕ РУКИ (зависит от позиции)")
print("=" * 60)

# Средние руки в разных позициях
test_cases = [
    (("J♠", "T♥"), "UTG", "Все сбросили до тебя"),
    (("J♠", "T♥"), "BTN", "Все сбросили до тебя"),
    (("9♠", "8♠"), "BTN", "Оппонент сделал рейз $30"),
]

for (h1_str, h2_str), pos, action in test_cases:
    h1 = Card(rank_map[h1_str[0]], h1_str[1])
    h2 = Card(rank_map[h2_str[0]], h2_str[1])
    
    analysis = analyze_preflop(
        hero_cards=[h1, h2],
        position=pos,
        action_before=action,
        stack=1000,
        bb=10
    )
    
    tier, name, desc = classify_preflop_hand(h1, h2)
    print(f"\n🃏 Рука: {h1}, {h2} (тир {tier}: {name})")
    print(f"📍 Позиция: {pos}, Действие: {action}")
    response = generate_response(analysis["prompt"])
    print(f"🤖 Ответ модели:\n{response}")

print()
print("=" * 60)
print("🧪 ТЕСТ 4: СЛУЧАЙНЫЕ РУКИ (5 примеров)")
print("=" * 60)

random.seed(42)
deck = build_deck()
for i in range(5):
    random.shuffle(deck)
    hero = deck[0:2]
    
    analysis = analyze_preflop(
        hero_cards=hero,
        position="CO",
        action_before="Оппонент сделал рейз $30",
        stack=1000,
        bb=10
    )
    
    tier, name, desc = classify_preflop_hand(hero[0], hero[1])
    print(f"\n🃏 Рука #{i+1}: {hero[0]}, {hero[1]} (тир {tier}: {name})")
    response = generate_response(analysis["prompt"])
    print(f"🤖 Ответ модели:\n{response}")

print()
print("=" * 60)
print("✅ ТЕСТ ЗАВЕРШЁН")
print("=" * 60)