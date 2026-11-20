"""
dataset_flop.py - Генератор датасета для Этапа 2 (флоп).
Создаёт примеры с ПРАВИЛЬНЫМИ решениями на основе эквити и текстуры борда.
"""
import random
import json
from analyzer import (
    build_deck, Card, analyze_postflop, classify_preflop_hand,
    RANKS, SUITS, HAND_NAMES, evaluate_hand
)

POSITIONS = ["UTG", "MP", "CO", "BTN", "SB", "BB"]
DATA_SET_FILE = "dataset_flop_1.json"

# Действия оппонента на флопе
OPPONENT_ACTIONS = [
    "Оппонент прочекал",
    "Оппонент поставил $30 (1/3 банка)",
    "Оппонент поставил $50 (1/2 банка)",
    "Оппонент поставил $80 (2/3 банка)",
    "Оппонент поставил $120 (пот)",
    "Оппонент сделал рейз $150",
]


def get_flop_decision(features, equity, pot_odds, pot, stack, position, opponent_action, bb=10):
    """
    Учитель для флопа. Принимает решение на основе готовых цифр.
    """
    # Извлекаем фичи
    hand_strength = int(features["Сила руки (0-100)"])
    has_pair = features["Пара"] == "Да"
    has_top_pair = features["Топ-пара"] == "Да"
    has_set = features["Сет"] == "Да"
    has_two_pair = features["Две пары"] == "Да"
    has_flush_draw = features["Флеш-дро"] == "Да"
    has_oesd = features["Стрит-дро (OESD)"] == "Да"
    has_gutshot = features["Гатшот"] == "Да"
    overcards = int(features["Оверкарты"])
    texture = features["Текстура борда"]
    
    # Определяем действие оппонента
    opponent_checked = "прочек" in opponent_action.lower()
    opponent_bet = "поставил" in opponent_action.lower()
    opponent_raise = "рейз" in opponent_action.lower()
    
    # Извлекаем размер ставки
    bet_amount = 0
    if opponent_bet or opponent_raise:
        # Парсим сумму из строки
        try:
            bet_amount = int(opponent_action.split("$")[1].split(" ")[0])
        except:
            bet_amount = pot // 2
    
    # ==== МОНСТРЫ (сет+, фулл-хаус) ====
    if has_set or has_two_pair or hand_strength >= 65:
        if opponent_checked:
            if "DRY" in texture:
                return ("BET", int(pot * 0.6), "Монстр на сухом борде — ставка для вэлью")
            else:
                return ("BET", int(pot * 0.7), "Монстр на мокром борде — большая ставка для защиты")
        elif opponent_raise:
            return ("RAISE", int(bet_amount * 2.5), "Монстр против рейза — рейз для максимума")
        else:  # opponent_bet
            return ("RAISE", int(bet_amount * 2.5), "Монстр против ставки — рейз для вэлью")
    
    # ==== СИЛЬНЫЕ РУКИ (топ-пара с хорошим кикером, две пары) ====
    if has_top_pair or (has_pair and overcards >= 1):
        if opponent_checked:
            return ("BET", int(pot * 0.5), "Топ-пара — продолженная ставка")
        elif opponent_bet and equity > pot_odds:
            return ("CALL", bet_amount, "Топ-пара с хорошим эквити — колл")
        elif opponent_raise and equity > 0.5:
            return ("CALL", bet_amount, "Сильная рука против рейза — колл, оцениваем тёрн")
        elif opponent_bet and equity < pot_odds:
            return ("FOLD", 0, "Топ-пара против ставки без оддсов — фолд")
        else:
            return ("CALL", bet_amount, "Сильная рука — колл")
    
    # ==== ДРО (флеш-дро, стрит-дро) ====
    if has_flush_draw or has_oesd:
        if opponent_checked:
            if has_flush_draw and has_oesd:
                return ("BET", int(pot * 0.6), "Комбо-дро — полу-блеф для фолд-эквити")
            else:
                return ("CHECK", 0, "Дро без агрессии — чек, смотрим бесплатную карту")
        elif opponent_bet and equity > pot_odds:
            return ("CALL", bet_amount, "Дро с хорошими оддсами — колл")
        elif has_flush_draw and has_oesd and equity > 0.4:
            return ("RAISE", int(bet_amount * 2), "Комбо-дро — рейз как полу-блеф")
        elif equity < pot_odds:
            return ("FOLD", 0, "Дро без оддсов — фолд")
        else:
            return ("CALL", bet_amount, "Дро — колл, надеемся на улучшение")
    
    # ==== СЛАБЫЕ РУКИ (маленькая пара, оверкарты) ====
    if has_pair or overcards >= 1:
        if opponent_checked:
            if position in ["CO", "BTN"] and random.random() < 0.4:
                return ("BET", int(pot * 0.4), "Слабая рука в позиции — ставка как блеф")
            else:
                return ("CHECK", 0, "Слабая рука — чек, пытаемся улучшить")
        elif opponent_bet and equity > pot_odds:
            return ("CALL", bet_amount, "Слабая рука с оддсами — колл")
        else:
            return ("FOLD", 0, "Слабая рука против ставки — фолд")
    
    # ==== ВОЗДУХ (нет попадания) ====
    if opponent_checked:
        if position in ["CO", "BTN"] and "DRY" in texture and random.random() < 0.5:
            return ("BET", int(pot * 0.5), "Воздух в позиции на сухом борде — блеф")
        else:
            return ("CHECK", 0, "Воздух — чек, сдаёмся")
    else:
        return ("FOLD", 0, "Воздух против ставки — фолд")


def generate_flop_dataset(count=5000, seed=42):
    """Генерирует датасет флоп-решений."""
    random.seed(seed)
    examples = []
    
    for i in range(count):
        deck = build_deck()
        random.shuffle(deck)
        
        # Генерируем руку героя (из диапазона открытия)
        hero = deck[0:2]
        tier, hand_name, _ = classify_preflop_hand(hero[0], hero[1])
        
        # Пропускаем мусорные руки (тир 7)
        if tier == 7:
            continue
        
        # Генерируем флоп
        flop = deck[2:5]
        
        # Генерируем контекст
        position = random.choice(POSITIONS)
        opponent_action = random.choice(OPPONENT_ACTIONS)
        stack = random.choice([500, 750, 1000, 1500, 2000])
        pot = random.choice([30, 50, 80, 120, 150])
        
        # Определяем to_call из действия оппонента
        to_call = 0
        if "поставил" in opponent_action.lower():
            try:
                to_call = int(opponent_action.split("$")[1].split(" ")[0])
            except:
                to_call = pot // 2
        elif "рейз" in opponent_action.lower():
            try:
                to_call = int(opponent_action.split("$")[1].split(" ")[0])
            except:
                to_call = pot
        
        # Анализируем ситуацию
        analysis = analyze_postflop(
            hero_cards=hero,
            board=flop,
            position=position,
            opponent_action=opponent_action,
            stack=stack,
            pot=pot,
            to_call=to_call,
            num_opponents=1,
            simulations=100
        )
        
        # Получаем решение от учителя
        action, amount, reason = get_flop_decision(
            features=analysis["features"],
            equity=analysis["equity"],
            pot_odds=analysis["pot_odds"],
            pot=pot,
            stack=stack,
            position=position,
            opponent_action=opponent_action,
            bb=10
        )
        
        # Формируем completion
        completion = f"""АНАЛИЗ: {reason}
ДЕЙСТВИЕ: {action}
СУММА: {amount}"""
        
        examples.append({
            "prompt": analysis["prompt"],
            "completion": completion,
            "reward": 1.0,
            "stage": "flop",
            "metadata": {
                "tier": tier,
                "position": position,
                "hand": f"{hero[0]}, {hero[1]}",
                "board": f"{flop[0]}, {flop[1]}, {flop[2]}",
                "equity": analysis["equity"],
                "pot_odds": analysis["pot_odds"],
                "hand_strength": analysis["hand_strength"],
                "opponent_action": opponent_action
            }
        })
        
        if (i + 1) % 100 == 0:
            print(f"  Сгенерировано {i + 1}/{count} примеров...")
    
    return examples


def save_dataset(examples, filename=DATA_SET_FILE):
    """Сохраняет датасет на диск."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=1)
    print(f"✅ Датасет сохранён: {filename} ({len(examples)} примеров)")


def show_statistics(examples):
    """Показывает распределение решений."""
    actions = {}
    tiers = {}
    for ex in examples:
        # Извлекаем действие из completion
        action = ex["completion"].split("ДЕЙСТВИЕ: ")[1].split("\n")[0]
        actions[action] = actions.get(action, 0) + 1
        
        tier = ex["metadata"]["tier"]
        tiers[tier] = tiers.get(tier, 0) + 1
    
    print("\n📊 Статистика датасета:")
    print("\nПо действиям:")
    for action, count in sorted(actions.items()):
        print(f"  {action}: {count} ({count/len(examples)*100:.1f}%)")
    
    print("\nПо тирам рук:")
    for tier, count in sorted(tiers.items()):
        print(f"  Тир {tier}: {count}")


# ============ ЗАПУСК ============
if __name__ == "__main__":
    print("🎯 Генерация датасета для Этапа 2 (ФЛОП)...\n")
    
    examples = generate_flop_dataset(count=5000)
    
    save_dataset(examples, DATA_SET_FILE)
    show_statistics(examples)
    
    print("\n" + "=" * 50)
    print("ПРИМЕР ПЕРВОГО ОБУЧАЮЩЕГО ПАРА:")
    print("=" * 50)
    print("\nПРОМПТ:")
    print(examples[0]["prompt"])
    print("\nПРАВИЛЬНЫЙ ОТВЕТ:")
    print(examples[0]["completion"])