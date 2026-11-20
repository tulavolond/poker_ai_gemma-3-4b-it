"""
dataset_flop_v2.py - Правильный генератор датасета для флопа.
Основан на префлоп-сценариях, диапазонах и текстурах борда.
"""
import random
import json
from collections import Counter
from analyzer import (
    build_deck, Card, analyze_postflop, classify_preflop_hand,
    RANKS, SUITS, HAND_NAMES, evaluate_hand
)

DATA_SET_FILE = "dataset_flop_v2.json"

# ============ ПРЕФЛОП-СЦЕНАРИИ ============
PREFLOP_SCENARIOS = [
    {
        "name": "BTN open vs BB call",
        "hero_position": "BTN",
        "villain_position": "BB",
        "hero_preflop_action": "OPEN_RAISE",
        "villain_preflop_action": "CALL",
        "pot_bb": 6,
        "stack_bb": 100,
        "weight": 0.25
    },
    {
        "name": "CO open vs BTN call",
        "hero_position": "CO",
        "villain_position": "BTN",
        "hero_preflop_action": "OPEN_RAISE",
        "villain_preflop_action": "CALL",
        "pot_bb": 6,
        "stack_bb": 100,
        "weight": 0.15
    },
    {
        "name": "CO open vs BTN 3-bet",
        "hero_position": "CO",
        "villain_position": "BTN",
        "hero_preflop_action": "CALL_3BET",
        "villain_preflop_action": "3-BET",
        "pot_bb": 21,
        "stack_bb": 100,
        "weight": 0.1
    },
    {
        "name": "SB open vs BB call",
        "hero_position": "SB",
        "villain_position": "BB",
        "hero_preflop_action": "OPEN_RAISE",
        "villain_preflop_action": "CALL",
        "pot_bb": 6,
        "stack_bb": 100,
        "weight": 0.15
    },
    {
        "name": "UTG open vs BTN call",
        "hero_position": "UTG",
        "villain_position": "BTN",
        "hero_preflop_action": "OPEN_RAISE",
        "villain_preflop_action": "CALL",
        "pot_bb": 6,
        "stack_bb": 100,
        "weight": 0.15
    },
    {
        "name": "BTN open vs SB 3-bet",
        "hero_position": "BTN",
        "villain_position": "SB",
        "hero_preflop_action": "CALL_3BET",
        "villain_preflop_action": "3-BET",
        "pot_bb": 21,
        "stack_bb": 100,
        "weight": 0.1
    },
    {
        "name": "MP open vs BTN call",
        "hero_position": "MP",
        "villain_position": "BTN",
        "hero_preflop_action": "OPEN_RAISE",
        "villain_preflop_action": "CALL",
        "pot_bb": 6,
        "stack_bb": 100,
        "weight": 0.1
    }
]

# ============ ДИАПАЗОНЫ ============
RANGES = {
    "BTN_OPEN": {
        "pairs": 7,
        "suited": (12, 8, 3),
        "offsuit": (12, 10, 2)
    },
    "CO_OPEN": {
        "pairs": 8,
        "suited": (11, 9, 2),
        "offsuit": (11, 10, 1)
    },
    "SB_OPEN": {
        "pairs": 9,
        "suited": (11, 10, 2),
        "offsuit": (12, 11, 1)
    },
    "UTG_OPEN": {
        "pairs": 10,
        "suited": (12, 11, 2),
        "offsuit": (13, 12, 1)
    },
    "MP_OPEN": {
        "pairs": 9,
        "suited": (12, 10, 2),
        "offsuit": (12, 11, 1)
    },
    "CALL_3BET": {
        "pairs": 10,
        "suited": (13, 11, 2),
        "offsuit": (14, 13, 0)
    },
    "BTN_CALL": {
        "pairs": 8,
        "suited": (11, 9, 3),
        "offsuit": (12, 11, 1)
    },
    "BB_DEFEND": {
        "pairs": 7,
        "suited": (10, 8, 3),
        "offsuit": (11, 10, 2)
    },
    "3BET": {
        "pairs": 11,
        "suited": (14, 12, 1),
        "offsuit": (14, 14, 0)
    }
}


def get_range_by_tier(range_def):
    """Возвращает список рук для данного диапазона."""
    hands = []
    for r1 in range(2, 15):
        for r2 in range(2, r1 + 1):
            if r1 == r2:  # пара
                if r1 >= range_def["pairs"]:
                    hands.append((r1, r2, True))
            else:
                # Одномастные
                if (r1 >= range_def["suited"][0] and
                    r2 >= range_def["suited"][1] and
                    (r1 - r2) <= range_def["suited"][2]):
                    hands.append((r1, r2, True))
                # Разномастные
                if (r1 >= range_def["offsuit"][0] and
                    r2 >= range_def["offsuit"][1] and
                    (r1 - r2) <= range_def["offsuit"][2]):
                    hands.append((r1, r2, False))
    return hands


def sample_hand_from_range(range_name):
    """Сэмплирует руку из диапазона."""
    range_def = RANGES[range_name]
    hands = get_range_by_tier(range_def)
    if not hands:
        return (random.randint(2, 14), random.randint(2, 14), random.choice([True, False]))
    return random.choice(hands)


def make_cards(r1, r2, suited, used_cards):
    """Создаёт две карты, не конфликтующие с used_cards."""
    for _ in range(100):
        suit1 = random.choice(SUITS)
        suit2 = suit1 if suited else random.choice([s for s in SUITS if s != suit1])
        c1 = Card(r1, suit1)
        c2 = Card(r2, suit2)
        if c1 not in used_cards and c2 not in used_cards and c1 != c2:
            return [c1, c2]
    return None


# ============ ГЕНЕРАЦИЯ БОРДА С ТЕКСТУРОЙ ============
def generate_board_with_texture(hero_cards, villain_cards, texture_type):
    """Генерирует борд с нужной текстурой."""
    deck = build_deck()
    used = set(hero_cards) | set(villain_cards)
    available = [c for c in deck if c not in used]

    for _ in range(200):
        random.shuffle(available)
        flop = available[:3]

        flop_ranks = [c.rank for c in flop]
        flop_suits = [c.suit for c in flop]
        suit_counts = Counter(flop_suits)
        max_suit_count = max(suit_counts.values())
        rank_span = max(flop_ranks) - min(flop_ranks)

        if texture_type == "dry":
            if max_suit_count <= 2 and rank_span >= 5:
                return flop
        elif texture_type == "wet":
            if max_suit_count >= 2 or rank_span <= 4:
                return flop
        elif texture_type == "high":
            if any(r >= 12 for r in flop_ranks):
                return flop
        elif texture_type == "low":
            if all(r <= 10 for r in flop_ranks):
                return flop
        else:
            return flop

    return available[:3]


# ============ МОДЕЛИРОВАНИЕ ДЕЙСТВИЙ ОПОННЕНТА ============
def simulate_villain_action(villain_cards, board, pot):
    """Моделирует действие оппонента на флопе."""
    villain_hand = evaluate_hand(villain_cards + board)
    villain_strength = villain_hand[0]

    if villain_strength >= 3:
        if random.random() < 0.7:
            return ("BET", int(pot * 0.66))
        return ("CHECK", 0)
    elif villain_strength >= 1:
        if random.random() < 0.5:
            return ("BET", int(pot * 0.5))
        return ("CHECK", 0)
    else:
        if random.random() < 0.8:
            return ("CHECK", 0)
        return ("BET", int(pot * 0.66))


# ============ ОПРЕДЕЛЕНИЕ ПРАВИЛЬНОГО РЕШЕНИЯ ============
def get_correct_decision(features, equity, pot_odds, pot, stack,
                         villain_action, position):
    """Определяет правильное решение для героя."""
    hand_strength = int(features["Сила руки (0-100)"])
    has_pair = features["Пара"] == "Да"
    has_top_pair = features["Топ-пара"] == "Да"
    has_set = features["Сет"] == "Да"
    has_two_pair = features["Две пары"] == "Да"
    has_flush_draw = features["Флеш-дро"] == "Да"
    has_oesd = features["Стрит-дро (OESD)"] == "Да"
    overcards = int(features["Оверкарты"])

    villain_checked = villain_action[0] == "CHECK"
    villain_bet = villain_action[0] == "BET"
    bet_amount = villain_action[1]

    if has_set or has_two_pair or hand_strength >= 65:
        if villain_checked:
            return ("BET", int(pot * 0.66), "Монстр — ставка для вэлью")
        return ("RAISE", int(bet_amount * 2.5), "Монстр против ставки — рейз")

    if has_top_pair or (has_pair and overcards >= 1):
        if villain_checked:
            return ("BET", int(pot * 0.5), "Топ-пара — продолженная ставка")
        if villain_bet and equity > pot_odds:
            return ("CALL", bet_amount, "Топ-пара с хорошим эквити — колл")
        return ("FOLD", 0, "Топ-пара без оддсов — фолд")

    if has_flush_draw or has_oesd:
        if villain_checked:
            if has_flush_draw and has_oesd:
                return ("BET", int(pot * 0.6), "Комбо-дро — полу-блеф")
            return ("CHECK", 0, "Дро — чек")
        if villain_bet and equity > pot_odds:
            return ("CALL", bet_amount, "Дро с оддсами — колл")
        return ("FOLD", 0, "Дро без оддсов — фолд")

    if has_pair or overcards >= 1:
        if villain_checked:
            return ("CHECK", 0, "Слабая рука — чек")
        return ("FOLD", 0, "Слабая рука против ставки — фолд")

    if villain_checked:
        return ("CHECK", 0, "Воздух — чек")
    return ("FOLD", 0, "Воздух против ставки — фолд")


# ============ ГЛАВНЫЙ ГЕНЕРАТОР ============
def generate_flop_dataset_v2(count=5000, seed=42):
    """Генерирует датасет флоп-решений на основе префлоп-сценариев."""
    random.seed(seed)
    examples = []
    attempts = 0
    max_attempts = count * 10

    scenario_weights = [s["weight"] for s in PREFLOP_SCENARIOS]

    while len(examples) < count and attempts < max_attempts:
        attempts += 1

        scenario = random.choices(PREFLOP_SCENARIOS, weights=scenario_weights)[0]

        # Определяем диапазоны
        hero_range_name = f"{scenario['hero_position']}_OPEN"
        if scenario['hero_preflop_action'] == "CALL_3BET":
            hero_range_name = "CALL_3BET"

        villain_range_name = "CALL_3BET" if scenario['villain_preflop_action'] == "3-BET" else \
                            "BB_DEFEND" if scenario['villain_position'] == "BB" else \
                            f"{scenario['villain_position']}_CALL"

        # Сэмплируем руки
        hero_r1, hero_r2, hero_suited = sample_hand_from_range(hero_range_name)
        villain_r1, villain_r2, villain_suited = sample_hand_from_range(villain_range_name)

        # Создаём карты
        hero_cards = make_cards(hero_r1, hero_r2, hero_suited, set())
        if hero_cards is None:
            continue
        villain_cards = make_cards(villain_r1, villain_r2, villain_suited, set(hero_cards))
        if villain_cards is None:
            continue

        # Генерируем борд
        texture_type = random.choice(["dry", "wet", "high", "low"])
        board = generate_board_with_texture(hero_cards, villain_cards, texture_type)

        # Контекст
        pot = scenario['pot_bb'] * 10
        stack = scenario['stack_bb'] * 10

        # Действие оппонента
        villain_action = simulate_villain_action(villain_cards, board, pot)

        if villain_action[0] == "CHECK":
            opponent_action_str = "Оппонент прочекал"
            to_call = 0
        else:
            opponent_action_str = f"Оппонент поставил ${villain_action[1]}"
            to_call = villain_action[1]

        # Анализ
        try:
            analysis = analyze_postflop(
                hero_cards=hero_cards,
                board=board,
                position=scenario['hero_position'],
                opponent_action=opponent_action_str,
                stack=stack,
                pot=pot,
                to_call=to_call,
                num_opponents=1,
                simulations=100
            )
        except Exception as e:
            continue

        # Правильное решение
        action, amount, reason = get_correct_decision(
            features=analysis["features"],
            equity=analysis["equity"],
            pot_odds=analysis["pot_odds"],
            pot=pot,
            stack=stack,
            villain_action=villain_action,
            position=scenario['hero_position']
        )

        completion = f"""АНАЛИЗ: {reason}
ДЕЙСТВИЕ: {action}
СУММА: {amount}"""

        examples.append({
            "prompt": analysis["prompt"],
            "completion": completion,
            "reward": 1.0,
            "stage": "flop",
            "metadata": {
                "scenario": scenario['name'],
                "hero_position": scenario['hero_position'],
                "villain_position": scenario['villain_position'],
                "hero_preflop_action": scenario['hero_preflop_action'],
                "villain_preflop_action": scenario['villain_preflop_action'],
                "hero_hand": f"{hero_cards[0]}, {hero_cards[1]}",
                "villain_hand": f"{villain_cards[0]}, {villain_cards[1]}",
                "board": f"{board[0]}, {board[1]}, {board[2]}",
                "equity": analysis["equity"],
                "pot_odds": analysis["pot_odds"],
                "hand_strength": analysis["hand_strength"],
                "villain_action": villain_action
            }
        })

        if len(examples) % 100 == 0:
            print(f"  Сгенерировано {len(examples)}/{count} примеров...")

    return examples


def save_dataset(examples, filename=DATA_SET_FILE):
    """Сохраняет датасет на диск."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=1)
    print(f"✅ Датасет сохранён: {filename} ({len(examples)} примеров)")


def show_statistics(examples):
    """Показывает распределение решений."""
    actions = {}
    scenarios = {}
    for ex in examples:
        action = ex["completion"].split("ДЕЙСТВИЕ: ")[1].split("\n")[0]
        actions[action] = actions.get(action, 0) + 1

        scenario = ex["metadata"]["scenario"]
        scenarios[scenario] = scenarios.get(scenario, 0) + 1

    print("\n📊 Статистика датасета:")
    print("\nПо действиям:")
    for action, count in sorted(actions.items()):
        print(f"  {action}: {count} ({count/len(examples)*100:.1f}%)")

    print("\nПо сценариям:")
    for scenario, count in sorted(scenarios.items()):
        print(f"  {scenario}: {count} ({count/len(examples)*100:.1f}%)")


if __name__ == "__main__":
    print("🎯 Генерация датасета для Этапа 2 (ФЛОП) v2...\n")

    examples = generate_flop_dataset_v2(count=5000)

    save_dataset(examples, DATA_SET_FILE)
    show_statistics(examples)

    print("\n" + "=" * 50)
    print("ПРИМЕР ПЕРВОГО ОБУЧАЮЩЕГО ПАРА:")
    print("=" * 50)
    print("\nПРОМПТ:")
    print(examples[0]["prompt"])
    print("\nПРАВИЛЬНЫЙ ОТВЕТ:")
    print(examples[0]["completion"])