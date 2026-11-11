"""
dataset_preflop.py - Генератор датасета для Этапа 1 (префлоп).
Создаёт примеры с ПРАВИЛЬНЫМИ решениями на основе покерных чартов.
"""
import random
import json
from analyzer import (
    build_deck, Card, classify_preflop_hand,
    analyze_preflop, RANKS, SUITS
)

POSITIONS = ["UTG", "MP", "CO", "BTN", "SB", "BB"]
DATA_SET_FILE = "dataset_preflop_3.json"
# Упрощённые действия до нас
ACTIONS_BEFORE = [
    "Все сбросили до тебя (ты открываешь торговлю)",
    "Оппонент уравнял блайнд (лимп $10)",
    "Оппонент сделал рейз $30",
    "Оппонент сделал большой рейз $50",
]


def get_preflop_decision(tier, position, action_before, stack, bb=10):
    """
    Возвращает ПРАВИЛЬНОЕ решение на основе покерной стратегии.
    Это "учитель", который знает как надо играть.
    """
    early_pos = position in ["UTG", "MP"]
    late_pos = position in ["CO", "BTN"]
    blind_pos = position in ["SB", "BB"]

    # Определяем, был ли рейз до нас
    facing_raise = "рейз" in action_before.lower()
    facing_big_raise = "$50" in action_before
    someone_limped = "лимп" in action_before.lower()

    # ==== ТИР 1: Премиум ====
    if tier == 1:
        if facing_big_raise:
            return ("RAISE", bb * 12, "Премиум-рука против большого рейза — 4-бет для максимума вэлью")
        elif facing_raise:
            return ("RAISE", bb * 9, "Премиум-рука против рейза — 3-бет для вэлью")
        else:
            return ("RAISE", bb * 3, "Премиум-рука — открываем рейзом")

    # ==== ТИР 2: Сильные ====
    if tier == 2:
        if facing_big_raise:
            if late_pos:
                return ("CALL", bb * 5, "Сильная рука в позиции против большого рейза — колл")
            else:
                return ("CALL", bb * 5, "Сильная рука — колл, не раздуваем банк без позиции")
        elif facing_raise:
            if late_pos or position == "CO":
                return ("RAISE", bb * 8, "Сильная рука в позиции — 3-бет")
            else:
                return ("CALL", bb * 3, "Сильная рука без позиции — колл")
        else:
            return ("RAISE", bb * 3, "Сильная рука — открываем рейзом")

    # ==== ТИР 3: Играбельные ====
    if tier == 3:
        if facing_big_raise:
            return ("FOLD", 0, "Средняя рука против большого рейза — фолд")
        elif facing_raise:
            if late_pos:
                return ("CALL", bb * 3, "Играбельная рука в позиции — колл")
            else:
                return ("FOLD", 0, "Играбельная рука без позиции против рейза — фолд")
        elif someone_limped:
            return ("RAISE", bb * 4, "Изоляция лимпера с играбельной рукой")
        else:
            if early_pos:
                return ("RAISE", bb * 3, "Играбельная рука — открываем рейзом")
            else:
                return ("RAISE", bb * 3, "Играбельная рука — рейз")

    # ==== ТИР 4: Позиционные ====
    if tier == 4:
        if facing_big_raise or facing_raise:
            return ("FOLD", 0, "Позиционная рука против рейза — фолд")
        elif someone_limped and late_pos:
            return ("RAISE", bb * 4, "Изоляция лимпера из поздней позиции")
        else:
            if late_pos:
                return ("RAISE", bb * 2.5, "Позиционная рука из поздней позиции — рейз")
            else:
                return ("FOLD", 0, "Позиционная рука из ранней позиции — фолд")

    # ==== ТИР 5: Спекулятивные ====
    if tier == 5:
        if facing_big_raise:
            return ("FOLD", 0, "Спекулятивная рука против большого рейза — фолд")
        elif facing_raise:
            return ("FOLD", 0, "Спекулятивная рука против рейза — фолд")
        else:
            if late_pos and not someone_limped:
                return ("RAISE", bb * 2.5, "Спекулятивная рука из поздней позиции — минимальный рейз")
            else:
                return ("FOLD", 0, "Спекулятивная рука — фолд, ждём лучших карт")

    # ==== ТИР 6: Слабые ====
    if tier == 6:
        if late_pos and not facing_raise and not facing_big_raise and not someone_limped:
            if random.random() < 0.3:
                return ("RAISE", bb * 2.5, "Слабая рука из поздней позиции — стил блайндов")
            else:
                return ("FOLD", 0, "Слабая рука — фолд")
        else:
            return ("FOLD", 0, "Слабая рука — фолд")

    # ==== ТИР 7: Мусор ====
    return ("FOLD", 0, "Мусорная рука — сбрасываем без раздумий")


def generate_preflop_dataset(count=3000, seed=42):
    """Генерирует датасет префлоп-решений."""
    random.seed(seed)
    examples = []

    for i in range(count):
        deck = build_deck()
        random.shuffle(deck)
        hero = deck[0:2]

        position = random.choice(POSITIONS)
        action_before = random.choice(ACTIONS_BEFORE)
        stack = random.choice([500, 750, 1000, 1500, 2000])
        bb = 10

        # Получаем правильный ответ от "учителя"
        action, amount, reason = get_preflop_decision(
            tier=None,  # будет определён внутри
            position=position,
            action_before=action_before,
            stack=stack,
            bb=bb
        )

        # Определяем тир для этой руки
        tier, hand_name, hand_desc = classify_preflop_hand(hero[0], hero[1])

        # Получаем решение заново с правильным тиром
        action, amount, reason = get_preflop_decision(
            tier=tier,
            position=position,
            action_before=action_before,
            stack=stack,
            bb=bb
        )

        # Формируем промпт через analyzer
        analysis = analyze_preflop(
            hero_cards=hero,
            position=position,
            action_before=action_before,
            stack=stack,
            bb=bb
        )

        # Формируем правильный ответ (completion)
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

        if (i + 1) % 100 == 0:
            print(f"  Сгенерировано {i + 1}/{count} примеров...")

    return examples


# def save_dataset(examples, filename="dataset_preflop_2.json"):
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
    print("🎯 Генерация датасета для Этапа 1 (ПРЕФЛОП)...\n")

    examples = generate_preflop_dataset(count=3000)

    # save_dataset(examples, "dataset_preflop.json")
    save_dataset(examples, DATA_SET_FILE)
    show_statistics(examples)

    print("\n" + "=" * 50)
    print("ПРИМЕР ПЕРВОГО ОБУЧАЮЩЕГО ПАРА:")
    print("=" * 50)
    print("\nПРОМПТ:")
    print(examples[0]["prompt"])
    print("\nПРАВИЛЬНЫЙ ОТВЕТ:")
    print(examples[0]["completion"])