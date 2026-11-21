"""
deep_dataset_v2.py - Думающий датасет: EV + reasoning + история + ошибки.
"""
import random
import json
from collections import Counter

from deep_analyzer import (
    build_deck, Card, evaluate_hand, HAND_NAMES,
    analyze_board_features, board_texture, classify_preflop_hand,
    RANKS, SUITS, RANK_NAMES
)
from deep_poker_ev import (
    equity_vs_range, best_action_preflop, best_action_postflop
)
from deep_reasoning_builder import (
    build_reasoning, build_error_reasoning, format_hand, format_board
)
from dataset_flop_v2 import (
    PREFLOP_SCENARIOS, RANGES, sample_hand_from_range, make_cards,
    generate_board_with_texture
)

DATA_SET_FILE = "./train_2/dataset_2000_v2.json"


RANGE_STRINGS = {
    "UTG_OPEN": "AA,KK,QQ,JJ,TT,AKs,AQs,AKo",
    "MP_OPEN": "AA,KK,QQ,JJ,TT,99,AKs,AQs,AJs,KQs,AKo,AQo",
    "CO_OPEN": "AA,KK,QQ,JJ,TT,99,88,AKs,AQs,AJs,ATs,KQs,KJs,QJs,AKo,AQo,AJo",
    "BTN_OPEN": "AA,KK,QQ,JJ,TT,99,88,77,66,55,AKs,AQs,AJs,ATs,A9s,A8s,A7s,KQs,KJs,KTs,K9s,QJs,QTs,Q9s,JTs,J9s,T9s,98s,87s,76s,65s,54s,AKo,AQo,AJo,ATo,A9o,KQo,KJo,KTo,QJo,QTo,JTo,T9o",
    "SB_OPEN": "AA,KK,QQ,JJ,TT,99,88,AKs,AQs,AJs,ATs,KQs,KJs,QJs,AKo,AQo,AJo",
    "BB_DEFEND": "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,KQs,KJs,KTs,K9s,K8s,K7s,K6s,K5s,QJs,QTs,Q9s,Q8s,Q7s,JTs,J9s,J8s,T9s,T8s,98s,87s,76s,65s,54s,43s,AKo,AQo,AJo,ATo,A9o,A8o,A7o,A6o,A5o,A4o,A3o,A2o,KQo,KJo,KTo,K9o,K8o,QJo,QTo,Q9o,JTo,J9o,T9o,98o,87o,76o",
    "CALL_3BET": "AA,KK,QQ,JJ,TT,99,AKs,AQs,AJs,KQs,AKo,AQo",
}


def _range_string(range_name):
    return RANGE_STRINGS.get(range_name, RANGE_STRINGS["BTN_OPEN"])


def _pick_villain_range_name(scenario):
    vpos = scenario["villain_position"]
    vaction = scenario["villain_preflop_action"]
    if vaction == "3-BET":
        return "CALL_3BET"
    if vpos == "BB":
        return "BB_DEFEND"
    return f"{vpos}_OPEN"


def _sample_hero_and_villain(scenario):
    hero_range_name = f"{scenario['hero_position']}_OPEN"
    if scenario['hero_preflop_action'] == "CALL_3BET":
        hero_range_name = "CALL_3BET"
    villain_range_name = _pick_villain_range_name(scenario)

    # [FIX] 30% шанс — случайная рука (включая мусор)
    if random.random() < 0.30:
        deck = build_deck()
        random.shuffle(deck)
        hero_cards = deck[:2]
        villain_cards = deck[2:4]
        return hero_cards, villain_cards, villain_range_name

    # 70% — из диапазона (как было)
    hero_r1, hero_r2, hero_suited = sample_hand_from_range(hero_range_name)
    villain_r1, villain_r2, villain_suited = sample_hand_from_range(villain_range_name)
    hero_cards = make_cards(hero_r1, hero_r2, hero_suited, set())
    if hero_cards is None:
        return None, None, villain_range_name
    villain_cards = make_cards(villain_r1, villain_r2, villain_suited, set(hero_cards))
    if villain_cards is None:
        return None, None, villain_range_name
    return hero_cards, villain_cards, villain_range_name


def generate_preflop_example():
    scenario = random.choice(PREFLOP_SCENARIOS)
    hero_cards, villain_cards, villain_range_name = _sample_hero_and_villain(scenario)
    if hero_cards is None:
        return None
    villain_range_str = _range_string(villain_range_name)
    tier, hand_name, hand_desc = classify_preflop_hand(hero_cards[0], hero_cards[1])
    hand_str = format_hand(hero_cards[0], hero_cards[1])
    action, amount, best_ev, evs = best_action_preflop(
        hero_cards, scenario["hero_position"], scenario["hero_preflop_action"],
        stack=scenario["stack_bb"] * 10, pot=15, to_call=10,
        villain_range=villain_range_str
    )
    equity = equity_vs_range(hero_cards, [], villain_range_str, num_simulations=100)
    history = (f"Hero ({scenario['hero_position']}) {scenario['hero_preflop_action']}, "
               f"Villain ({scenario['villain_position']}) {scenario['villain_preflop_action']}")
    preflop_strength = 100 - (tier - 1) * 15
    preflop_strength = 100 - (tier - 1) * 15
    reasoning = build_reasoning(
        street="preflop", hero_cards=hero_cards, board=[],
        hero_hand_name=hand_name, strength=preflop_strength,
        equity=equity, pot=15, to_call=10,
        action=action, amount=amount, evs=evs,
        villain_range=villain_range_str, history=history, texture="N/A"
    )
    prompt = f"""[СТАДИЯ: ПРЕФЛОП]

=== ИСТОРИЯ РАЗДАЧИ ===
{history}

=== ТВОИ КАРТЫ ===
{hero_cards[0]}, {hero_cards[1]}

=== АНАЛИЗ СИСТЕМЫ ===
Рука: {hand_name} (сила {100 - (tier-1)*15}/100)
Диапазон оппонента: {villain_range_str}

=== КОНТЕКСТ ===
Позиция: {scenario['hero_position']}
Стек: ${scenario['stack_bb'] * 10}, Банк: $15
Колл: $10

Ответь СТРОГО:
АНАЛИЗ: [логика с числами]
ДЕЙСТВИЕ: [FOLD/CALL/RAISE]
СУММА: [число]"""
    completion = f"""АНАЛИЗ: {reasoning}
ДЕЙСТВИЕ: {action}
СУММА: {amount}"""
    return {
        "prompt": prompt, "completion": completion, "stage": "preflop",
        "metadata": {"hero_hand": hand_str, "tier": tier, "action": action,
                     "ev": round(best_ev, 4), "evs": evs},
        "reward": 1.0 if best_ev > 0 else 0.5,
    }


def _generate_postflop(street, board_len):
    scenario = random.choice(PREFLOP_SCENARIOS)
    hero_cards, villain_cards, villain_range_name = _sample_hero_and_villain(scenario)
    if hero_cards is None:
        return None
    villain_range_str = _range_string(villain_range_name)
    texture_type = random.choice(["dry", "wet", "high", "low"])
    board = generate_board_with_texture(hero_cards, villain_cards, texture_type)
    used = set(hero_cards) | set(villain_cards) | set(board)
    remaining = [c for c in build_deck() if c not in used]
    random.shuffle(remaining)
    while len(board) < board_len:
        board.append(remaining.pop())
    hero_hand = evaluate_hand(hero_cards + board)
    hero_hand_name = HAND_NAMES[hero_hand[0]]
    villain_hand = evaluate_hand(villain_cards + board)
    v_strength = villain_hand[0] * 10 + 5
    pot = 60 if board_len == 3 else (100 if board_len == 4 else 150)
    if v_strength >= 40:
        villain_bet = int(pot * 0.66)
        villain_action = ("BET", villain_bet)
        to_call = villain_bet
    elif v_strength >= 25 and random.random() < 0.5:
        villain_bet = int(pot * 0.5)
        villain_action = ("BET", villain_bet)
        to_call = villain_bet
    else:
        villain_action = ("CHECK", 0)
        to_call = 0
    action, amount, best_ev, evs = best_action_postflop(
        hero_cards, board, pot, to_call, villain_range_str,
        street=street, stack=1000
    )
    equity = equity_vs_range(hero_cards, board, villain_range_str, num_simulations=200)
    texture = board_texture(board, analyze_board_features(board, hero_cards))
    opp_action_str = f"Опп {villain_action[0]}" + (
        f" ${villain_action[1]}" if villain_action[0] == "BET" else "")
    history = (f"Hero ({scenario['hero_position']}) {scenario['hero_preflop_action']}, "
               f"Villain ({scenario['villain_position']}) {scenario['villain_preflop_action']}. "
               f"{street.capitalize()}: {opp_action_str}")
    postflop_strength = hero_hand[0] * 10 + 5
    reasoning = build_reasoning(
        street=street, hero_cards=hero_cards, board=board,
        hero_hand_name=hero_hand_name, strength=postflop_strength,
        equity=equity, pot=pot, to_call=to_call,
        action=action, amount=amount, evs=evs,
        villain_range=villain_range_str, history=history, texture=texture
    )
    hand_str = format_hand(hero_cards[0], hero_cards[1])
    board_str = format_board(board)
    prompt = f"""[СТАДИЯ: {street.upper()}]

=== ИСТОРИЯ РАЗДАЧИ ===
{history}

=== ТВОИ КАРТЫ ===
{hero_cards[0]}, {hero_cards[1]}

=== БОРД ({len(board)} карт) ===
{board_str}

=== АНАЛИЗ СИСТЕМЫ ===
Комбинация: {hero_hand_name} (сила {hero_hand[0]*10+5}/100)
Эквити против диапазона: {equity*100:.0f}%
Текстура: {texture}
Диапазон оппонента: {villain_range_str}

=== ДЕЙСТВИЕ ОППОНЕНТА ===
{opp_action_str}

=== КОНТЕКСТ ===
Позиция: {scenario['hero_position']}
Стек: $1000, Банк: ${pot}
Колл: ${to_call}

Ответь СТРОГО:
АНАЛИЗ: [логика с числами]
ДЕЙСТВИЕ: [FOLD/CHECK/CALL/BET/RAISE]
СУММА: [число]"""
    completion = f"""АНАЛИЗ: {reasoning}
ДЕЙСТВИЕ: {action}
СУММА: {amount}"""
    return {
        "prompt": prompt, "completion": completion, "stage": street,
        "metadata": {
            "hero_hand": hand_str,
            "board": board_str,
            "hero_combo": hero_hand_name,
            "action": action,
            "ev": round(best_ev, 4),
            "evs": evs,
            "equity": round(equity, 4),
            "pot": pot,
            "to_call": to_call,
            "hero_cards": [str(c) for c in hero_cards],
            "board_cards": [str(c) for c in board],
            "villain_range": villain_range_str,
            "history": history,
        },
        "reward": 1.0 if best_ev > 0 else 0.5,
    }


def generate_flop_example():
    return _generate_postflop("flop", 3)


def generate_turn_example():
    return _generate_postflop("turn", 4)


def generate_river_example():
    return _generate_postflop("river", 5)


def generate_error_example():
    street = random.choice(["flop", "turn", "river"])
    base = _generate_postflop(street, {"flop": 3, "turn": 4, "river": 5}[street])
    if base is None:
        return None
    evs = base["metadata"]["evs"]
    right_action = base["metadata"]["action"]
    wrong_candidates = [a for a in evs if a != right_action and evs[a] > -900]
    if not wrong_candidates:
        return None
    wrong_action = min(wrong_candidates, key=lambda a: evs[a])
    if abs(evs[right_action] - evs[wrong_action]) < 1.0:
        return None
    reasoning = build_error_reasoning(
        street=street,
        hero_hand_str=base["metadata"]["hero_hand"],
        board_str=base["metadata"]["board"],
        hero_hand_name=base["metadata"]["hero_combo"],
        equity=base["metadata"]["equity"],
        pot=base["metadata"]["pot"],
        to_call=base["metadata"]["to_call"],
        wrong_action=wrong_action,
        right_action=right_action,
        evs=evs,
        villain_range=base["metadata"]["villain_range"],
        history=base["metadata"]["history"]
    )
    prompt = base["prompt"].replace(
        "Ответь СТРОГО:",
        f"Ответь СТРОГО (объясни, почему {wrong_action} — ошибка, и что правильно):"
    )
    right_amount = 0
    if right_action == "CALL":
        right_amount = base["metadata"]["to_call"]
    elif right_action == "BET":
        right_amount = int(base["metadata"]["pot"] * 0.66)
    elif right_action == "RAISE":
        right_amount = int(base["metadata"]["to_call"] * 2.5)
    completion = f"""АНАЛИЗ: {reasoning}
ДЕЙСТВИЕ: {right_action}
СУММА: {right_amount}"""
    return {
        "prompt": prompt, "completion": completion, "stage": street,
        "metadata": {**base["metadata"], "is_error": True,
                     "wrong_action": wrong_action},
        "reward": 1.0,
    }


def generate_unified_dataset(count_per_stage=100, seed=42):
    random.seed(seed)
    examples = []
    print(f"Префлоп ({count_per_stage})...")
    for _ in range(count_per_stage):
        ex = generate_preflop_example()
        if ex:
            examples.append(ex)
    print(f"Флоп ({count_per_stage})...")
    for _ in range(count_per_stage):
        ex = generate_flop_example()
        if ex:
            examples.append(ex)
    print(f"Тёрн ({count_per_stage})...")
    for _ in range(count_per_stage):
        ex = generate_turn_example()
        if ex:
            examples.append(ex)
    print(f"Ривер ({count_per_stage})...")
    for _ in range(count_per_stage):
        ex = generate_river_example()
        if ex:
            examples.append(ex)
    error_count = int(len(examples) * 0.2)
    print(f"Ошибки ({error_count})...")
    for _ in range(error_count):
        ex = generate_error_example()
        if ex:
            examples.append(ex)
    random.shuffle(examples)
    return examples


def save_dataset(examples, filename=DATA_SET_FILE):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=1, default=str)
    print(f"✅ Сохранено: {filename} ({len(examples)} примеров)")


def show_statistics(examples):
    stages = Counter(ex["stage"] for ex in examples)
    unique_prompts = len(set(ex["prompt"] for ex in examples))
    unique_completions = len(set(ex["completion"] for ex in examples))
    errors = sum(1 for ex in examples if ex["metadata"].get("is_error"))
    prompt_to_completion = {}
    for ex in examples:
        prompt_to_completion.setdefault(ex["prompt"], set()).add(ex["completion"])
    conflicts = sum(1 for v in prompt_to_completion.values() if len(v) > 1)
    print("\n📊 Статистика:")
    for s, c in stages.items():
        print(f"  {s}: {c}")
    print(f"  Ошибок (с разбором): {errors} ({errors/len(examples)*100:.1f}%)")
    print(f"  Уникальных prompt:     {unique_prompts} ({unique_prompts/len(examples)*100:.1f}%)")
    print(f"  Уникальных completion: {unique_completions} ({unique_completions/len(examples)*100:.1f}%)")
    print(f"  Конфликтов prompt: {conflicts}")


if __name__ == "__main__":
    print("Ключи RANGES из dataset_flop_v2:", list(RANGES.keys()))
    print()
    examples = generate_unified_dataset(count_per_stage=2000)
    save_dataset(examples)
    show_statistics(examples)
    print("\n" + "=" * 60)
    print("ПРИМЕРЫ:")
    for stage in ["preflop", "flop", "turn", "river"]:
        ex = next((e for e in examples if e["stage"] == stage), None)
        if ex:
            print(f"\n--- {stage.upper()} ---")
            print("PROMPT:\n", ex["prompt"][:800])
            print("\nCOMPLETION:\n", ex["completion"])
    print()
    print(f"DATA_SET_FILE: {DATA_SET_FILE}")