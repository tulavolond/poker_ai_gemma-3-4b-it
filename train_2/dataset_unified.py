"""
dataset_unified.py - Единый датасет для всех стадий покера.
Префлоп + Флоп + Тёрн + Ривер с полной информацией о комбинациях.
"""
import random
import json
from collections import Counter
from analyzer import (
    build_deck, Card, analyze_postflop, classify_preflop_hand,
    RANKS, SUITS, HAND_NAMES, evaluate_hand
)
from poker_math import (
    calculate_exact_outs, calculate_improvement_probs_flop,
    calculate_improvement_probs_turn, estimate_opponent_range,
    format_outs, format_probs, format_opponent_range
)
from dataset_flop_v2 import (
    PREFLOP_SCENARIOS, RANGES, sample_hand_from_range,
    make_cards, generate_board_with_texture, simulate_villain_action
)

DATA_SET_FILE = "./train_2/dataset_unified_test.json"

# ============ ПРЕФЛОП-СЦЕНАРИИ (те же, что и раньше) ============
# ... (копируем из dataset_flop_v2.py) ...


def generate_preflop_example():
    """Генерирует пример для префлопа."""
    scenario = random.choice(PREFLOP_SCENARIOS)
    
    hero_range_name = f"{scenario['hero_position']}_OPEN"
    if scenario['hero_preflop_action'] == "CALL_3BET":
        hero_range_name = "CALL_3BET"
    
    villain_range_name = "CALL_3BET" if scenario['villain_preflop_action'] == "3-BET" else \
                        "BB_DEFEND" if scenario['villain_position'] == "BB" else \
                        f"{scenario['villain_position']}_CALL"
    
    hero_r1, hero_r2, hero_suited = sample_hand_from_range(hero_range_name)
    villain_r1, villain_r2, villain_suited = sample_hand_from_range(villain_range_name)
    
    hero_cards = make_cards(hero_r1, hero_r2, hero_suited, set())
    if hero_cards is None:
        return None
    villain_cards = make_cards(villain_r1, villain_r2, villain_suited, set(hero_cards))
    if villain_cards is None:
        return None
    
    # Классифицируем руку
    tier, hand_name, hand_desc = classify_preflop_hand(hero_cards[0], hero_cards[1])
    
    # Оцениваем диапазон оппонента
    opponent_range = estimate_opponent_range(hero_cards, [], villain_range_name)
    
    # Формируем промпт
    prompt = f"""[СТАДИЯ: ПРЕФЛОП]

=== СИТУАЦИЯ: ПРЕФЛОП ===

ТВОИ КАРТЫ: {hero_cards[0]}, {hero_cards[1]}
БОРД: ещё нет карт на столе

— АНАЛИЗ ОТ СИСТЕМЫ —
Стартовая рука: {hand_name} (Тир {tier}/7)
Описание: {hand_desc}
Одномастная: {"Да" if hero_cards[0].suit == hero_cards[1].suit else "Нет"}
Пара: {"Да" if hero_cards[0].rank == hero_cards[1].rank else "Нет"}

ВОЗМОЖНЫЕ КОМБИНАЦИИ ОПОННЕНТА:
{format_opponent_range(opponent_range)}

— КОНТЕКСТ —
Позиция: {scenario['hero_position']}
Стек: ${scenario['stack_bb'] * 10}
Действия до тебя: {scenario['hero_preflop_action']}

Ответь СТРОГО в формате:
АНАЛИЗ: [твоя логика]
МОИ КОМБИНАЦИИ: [стартовая рука + потенциал]
КОМБИНАЦИИ ОПОННЕНТА: [что он может иметь]
ДЕЙСТВИЕ: [FOLD/CALL/RAISE]
СУММА: [число]"""
    
    # Определяем правильное решение
    if tier <= 2:
        action = "RAISE"
        amount = 30
        reason = f"Сильная рука {hand_name} — открываем рейзом"
    elif tier <= 4:
        action = "RAISE"
        amount = 30
        reason = f"Играбельная рука {hand_name} из {scenario['hero_position']} — рейз"
    else:
        action = "FOLD"
        amount = 0
        reason = f"Слабая рука {hand_name} — фолд"
    
    completion = f"""АНАЛИЗ: {reason}
МОИ КОМБИНАЦИИ: {hand_name} → потенциал на стрит/флеш/сет
КОМБИНАЦИИ ОПОННЕНТА: {format_opponent_range(opponent_range)}
ДЕЙСТВИЕ: {action}
СУММА: {amount}"""
    
    return {
        "prompt": prompt,
        "completion": completion,
        "reward": 1.0,
        "stage": "preflop",
        "metadata": {
            "scenario": scenario['name'],
            "hero_hand": f"{hero_cards[0]}, {hero_cards[1]}",
            "tier": tier,
        }
    }


def generate_flop_example():
    """Генерирует пример для флопа."""
    scenario = random.choice(PREFLOP_SCENARIOS)
    
    hero_range_name = f"{scenario['hero_position']}_OPEN"
    if scenario['hero_preflop_action'] == "CALL_3BET":
        hero_range_name = "CALL_3BET"
    
    villain_range_name = "CALL_3BET" if scenario['villain_preflop_action'] == "3-BET" else \
                        "BB_DEFEND" if scenario['villain_position'] == "BB" else \
                        f"{scenario['villain_position']}_CALL"
    
    hero_r1, hero_r2, hero_suited = sample_hand_from_range(hero_range_name)
    villain_r1, villain_r2, villain_suited = sample_hand_from_range(villain_range_name)
    
    hero_cards = make_cards(hero_r1, hero_r2, hero_suited, set())
    if hero_cards is None:
        return None
    villain_cards = make_cards(villain_r1, villain_r2, villain_suited, set(hero_cards))
    if villain_cards is None:
        return None
    
    texture_type = random.choice(["dry", "wet", "high", "low"])
    board = generate_board_with_texture(hero_cards, villain_cards, texture_type)
    
    # Оцениваем руку героя
    hero_hand = evaluate_hand(hero_cards + board)
    hero_hand_name = HAND_NAMES[hero_hand[0]]
    
    # Считаем ауты и вероятности улучшения
    outs = calculate_exact_outs(hero_cards, board)
    probs = calculate_improvement_probs_flop(hero_cards, board)
    
    # Оцениваем диапазон оппонента
    opponent_range = estimate_opponent_range(hero_cards, board, villain_range_name)
    
    # Моделируем действие оппонента
    villain_action = simulate_villain_action(villain_cards, board, 60)
    
    if villain_action[0] == "CHECK":
        opponent_action_str = "Оппонент прочекал"
        to_call = 0
    else:
        opponent_action_str = f"Оппонент поставил ${villain_action[1]}"
        to_call = villain_action[1]
    
    pot = 60
    stack = 1000
    
    # Формируем промпт
    prompt = f"""[СТАДИЯ: ФЛОП]

=== СИТУАЦИЯ: ФЛОП ===

ТВОИ КАРТЫ: {hero_cards[0]}, {hero_cards[1]}
БОРД (3 карт): {board[0]}, {board[1]}, {board[2]}

— АНАЛИЗ ОТ СИСТЕМЫ —
Твоя текущая комбинация: {hero_hand_name}
Сила руки (0-100): {hero_hand[0] * 10 + 5}

ВОЗМОЖНЫЕ УЛУЧШЕНИЯ (ауты):
{format_outs(outs)}

ВЕРОЯТНОСТИ УЛУЧШЕНИЯ (до ривера):
{format_probs(probs)}

ВОЗМОЖНЫЕ КОМБИНАЦИИ ОПОННЕНТА:
{format_opponent_range(opponent_range)}

— ДЕЙСТВИЕ ОППОНЕНТА —
{opponent_action_str}

— КОНТЕКСТ —
Позиция: {scenario['hero_position']}
Стек: ${stack}, Банк: ${pot}
Стек/банк: {stack / pot:.1f}

Ответь СТРОГО в формате:
АНАЛИЗ: [твоя логика]
МОИ КОМБИНАЦИИ: [текущая] → [возможные]
КОМБИНАЦИИ ОПОННЕНТА: [что он может иметь]
ДЕЙСТВИЕ: [FOLD/CHECK/CALL/BET/RAISE]
СУММА: [число]"""
    
    # Определяем правильное решение
    if hero_hand[0] >= 3:  # сет+
        action = "BET" if villain_action[0] == "CHECK" else "RAISE"
        amount = int(pot * 0.66) if action == "BET" else int(villain_action[1] * 2.5)
        reason = f"Монстр {hero_hand_name} — {'ставка' if action == 'BET' else 'рейз'} для вэлью"
    elif hero_hand[0] >= 1:  # пара+
        if villain_action[0] == "CHECK":
            action = "BET"
            amount = int(pot * 0.5)
            reason = f"{hero_hand_name} — продолженная ставка"
        else:
            action = "CALL"
            amount = to_call
            reason = f"{hero_hand_name} — колл"
    elif outs:  # есть ауты
        if villain_action[0] == "CHECK":
            action = "CHECK"
            amount = 0
            reason = f"Дро — чек, смотрим бесплатную карту"
        else:
            action = "CALL"
            amount = to_call
            reason = f"Дро с аутами — колл"
    else:
        if villain_action[0] == "CHECK":
            action = "CHECK"
            amount = 0
            reason = "Воздух — чек"
        else:
            action = "FOLD"
            amount = 0
            reason = "Воздух против ставки — фолд"
    
    completion = f"""АНАЛИЗ: {reason}
МОИ КОМБИНАЦИИ: {hero_hand_name} → {format_probs(probs)}
КОМБИНАЦИИ ОПОННЕНТА: {format_opponent_range(opponent_range)}
ДЕЙСТВИЕ: {action}
СУММА: {amount}"""
    
    return {
        "prompt": prompt,
        "completion": completion,
        "reward": 1.0,
        "stage": "flop",
        "metadata": {
            "scenario": scenario['name'],
            "hero_hand": f"{hero_cards[0]}, {hero_cards[1]}",
            "board": f"{board[0]}, {board[1]}, {board[2]}",
            "hero_combo": hero_hand_name,
        }
    }


def generate_turn_example():
    """Генерирует пример для тёрна."""
    # Аналогично флопу, но борд 4 карты
    # ... (реализация аналогична)
    pass


def generate_river_example():
    """Генерирует пример для ривера."""
    # Аналогично, но борд 5 карт
    # ... (реализация аналогична)
    pass


def generate_unified_dataset(count_per_stage=50, seed=42):
    """Генерирует единый датасет для всех стадий."""
    random.seed(seed)
    examples = []
    
    print(f"Генерация префлоп ({count_per_stage} примеров)...")
    for _ in range(count_per_stage):
        ex = generate_preflop_example()
        if ex:
            examples.append(ex)
    
    print(f"Генерация флоп ({count_per_stage} примеров)...")
    for _ in range(count_per_stage):
        ex = generate_flop_example()
        if ex:
            examples.append(ex)
    
    # TODO: добавить тёрн и ривер
    
    random.shuffle(examples)
    return examples


def save_dataset(examples, filename=DATA_SET_FILE):
    """Сохраняет датасет."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=1)
    print(f"✅ Датасет сохранён: {filename} ({len(examples)} примеров)")


def show_statistics(examples):
    """Показывает статистику."""
    stages = Counter(ex["stage"] for ex in examples)
    print("\n📊 Статистика:")
    for stage, count in stages.items():
        print(f"  {stage}: {count}")


if __name__ == "__main__":
    print("🎯 Генерация пробного единого датасета...\n")
    
    examples = generate_unified_dataset(count_per_stage=2000)
    
    save_dataset(examples, DATA_SET_FILE)
    show_statistics(examples)
    
    print("\n" + "=" * 60)
    print("ПРИМЕРЫ:")
    print("=" * 60)
    
    for stage in ["preflop", "flop"]:
        stage_examples = [ex for ex in examples if ex["stage"] == stage]
        if stage_examples:
            print(f"\n--- {stage.upper()} ---")
            print("\nПРОМПТ:")
            print(stage_examples[0]["prompt"])
            print("\nОТВЕТ:")
            print(stage_examples[0]["completion"])