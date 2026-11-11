"""
dataset_preflop.py v2 - БАЛАНСИРОВАННЫЙ датасет.
Равномерное распределение тиров + более активный учитель.
"""
import random
import json
from analyzer import (
    build_deck, Card, classify_preflop_hand,
    analyze_preflop, RANKS, SUITS
)

POSITIONS = ["UTG", "MP", "CO", "BTN", "SB", "BB"]

ACTIONS_BEFORE = [
    "Все сбросили до тебя (ты открываешь торговлю)",
    "Оппонент уравнял блайнд (лимп $10)",
    "Оппонент сделал рейз $30",
    "Оппонент сделал большой рейз $50",
]


def get_preflop_decision(tier, position, action_before, stack, bb=10):
    """
    ПРАВИЛЬНОЕ решение на основе покерной стратегии.
    Более активное — меньше фолдов, больше рейзов в позиции.
    """
    early_pos = position in ["UTG", "MP"]
    late_pos = position in ["CO", "BTN"]
    blind_pos = position in ["SB", "BB"]

    facing_raise = "рейз" in action_before.lower()
    facing_big_raise = "$50" in action_before
    someone_limped = "лимп" in action_before.lower()
    opening = "открываешь" in action_before.lower()

    # ==== ТИР 1: Премиум (AA, KK, AKs) ====
    if tier == 1:
        if facing_big_raise:
            return ("RAISE", int(bb * 12), "Премиум против большого рейза — 4-бет для вэлью")
        elif facing_raise:
            return ("RAISE", int(bb * 9), "Премиум против рейза — 3-бет")
        else:
            return ("RAISE", int(bb * 3), "Премиум-рука — открываем рейзом")

    # ==== ТИР 2: Сильные (QQ, JJ, AK, AQ) ====
    if tier == 2:
        if facing_big_raise:
            return ("RAISE", int(bb * 8), "Сильная рука против большого рейза — 3-бет")
        elif facing_raise:
            if late_pos:
                return ("RAISE", int(bb * 8), "Сильная рука в позиции — 3-бет")
            else:
                return ("CALL", int(bb * 3), "Сильная рука без позиции — колл")
        else:
            return ("RAISE", int(bb * 3), "Сильная рука — открываем рейзом")

    # ==== ТИР 3: Играбельные (TT, 99, suited broadway) ====
    if tier == 3:
        if facing_big_raise:
            return ("FOLD", 0, "Играбельная рука против большого рейза — фолд")
        elif facing_raise:
            if late_pos:
                return ("CALL", int(bb * 3), "Играбельная рука в позиции — колл")
            else:
                return ("FOLD", 0, "Играбельная рука без позиции против рейза — фолд")
        elif someone_limped:
            return ("RAISE", int(bb * 4), "Изоляция лимпера с играбельной рукой")
        else:
            return ("RAISE", int(bb * 3), "Играбельная рука — открываем рейзом")

    # ==== ТИР 4: Позиционные (88, 77, suited connectors) ====
    if tier == 4:
        if facing_big_raise:
            return ("FOLD", 0, "Позиционная рука против большого рейза — фолд")
        elif facing_raise:
            if late_pos:
                return ("CALL", int(bb * 3), "Позиционная рука в позиции — колл на сет")
            else:
                return ("FOLD", 0, "Позиционная рука без позиции против рейза — фолд")
        elif someone_limped and late_pos:
            return ("RAISE", int(bb * 4), "Изоляция лимпера из поздней позиции")
        elif late_pos and opening:
            return ("RAISE", int(bb * 2.5), "Позиционная рука из поздней позиции — рейз")
        else:
            return ("FOLD", 0, "Позиционная рука из ранней позиции — фолд")

    # ==== ТИР 5: Спекулятивные (маленькие пары, одномастные) ====
    if tier == 5:
        if facing_big_raise or facing_raise:
            return ("FOLD", 0, "Спекулятивная рука против рейза — фолд")
        elif late_pos and opening:
            return ("RAISE", int(bb * 2.5), "Спекулятивная рука из поздней позиции — мин-рейз")
        elif late_pos and someone_limped:
            return ("CALL", int(bb * 1), "Спекулятивная рука в позиции — дешёвый колл")
        else:
            return ("FOLD", 0, "Спекулятивная рука — фолд")

    # ==== ТИР 6: Слабые ====
    if tier == 6:
        if facing_raise or facing_big_raise:
            return ("FOLD", 0, "Слабая рука против рейза — фолд")
        elif late_pos and opening:
            if random.random() < 0.4:
                return ("RAISE", int(bb * 2.5), "Слабая рука из поздней позиции — стил блайндов")
            else:
                return ("FOLD", 0, "Слабая рука — фолд")
        elif blind_pos and someone_limped:
            return ("CALL", int(bb * 1), "Слабая рука на блайнде — дешёвый колл")
        else:
            return ("FOLD", 0, "Слабая рука — фолд")

    # ==== ТИР 7: Мусор ====
    return ("FOLD", 0, "Мусорная рука — фолд")


def generate_balanced_dataset(count_per_tier=450, seed=42):
    """
    Генерирует БАЛАНСИРОВАННЫЙ датасет.
    По 450 примеров на каждый тир = 3150 всего.
    Тир 1 даём чуть больше (премиум важен).
    """
    random.seed(seed)
    examples = []

    # Распределение: больше примеров для важных тиров
    tier_counts = {
        1: int(count_per_tier * 1.2),  # Премиум — чуть больше
        2: count_per_tier,
        3: count_per_tier,
        4: count_per_tier,
        5: count_per_tier,
        6: count_per_tier,
        7: count_per_tier,
    }

    for target_tier, target_count in tier_counts.items():
        generated = 0
        attempts = 0
        max_attempts = target_count * 20

        while generated < target_count and attempts < max_attempts:
            attempts += 1
            deck = build_deck()
            random.shuffle(deck)
            hero = deck[0:2]

            # Проверяем, что рука нужного тира
            tier, hand_name, hand_desc = classify_preflop_hand(hero[0], hero[1])
            if tier != target_tier:
                continue

            # Генерируем ситуацию
            position = random.choice(POSITIONS)
            action_before = random.choice(ACTIONS_BEFORE)
            stack = random.choice([500, 750, 1000, 1500, 2000])
            bb = 10

            # Получаем решение
            action, amount, reason = get_preflop_decision(
                tier=tier,
                position=position,
                action_before=action_before,
                stack=stack,
                bb=bb
            )

            # Формируем промпт
            analysis = analyze_preflop(
                hero_cards=hero,
                position=position,
                action_before=action_before,
                stack=stack,
                bb=bb
            )

            completion = f"""АНАЛИЗ: {reason}
ДЕЙСТВИЕ: {action}
СУММА: {amount}"""

            examples.append({
                "prompt": analysis["prompt"],
                "completion": completion,
                "reward": 1.0,
                "stage": "preflop",
                "metadata": {
                    "tier": tier,
                    "position": position,
                    "hand": f"{hero[0]}, {hero[1]}",
                    "action_before": action_before
                }
            })

            generated += 1

        print(f"  Тир {target_tier}: сгенерировано {generated}/{target_count}")

    # Перемешиваем датасет
    random.shuffle(examples)
    return examples


def save_dataset(examples, filename="dataset_preflop_3.json"):
    """Сохраняет датасет на диск."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=1)
    print(f"\n✅ Датасет сохранён: {filename} ({len(examples)} примеров)")


if __name__ == "__main__":
    print("🎯 Генерация БАЛАНСИРОВАННОГО датасета для Этапа 1...\n")

    examples = generate_balanced_dataset(count_per_tier=450)

    save_dataset(examples, "dataset_preflop_3.json")

    print("\nТеперь запустите: python analyze_dataset.py")