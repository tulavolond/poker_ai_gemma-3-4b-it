"""
deep_poker_ev.py - EV-движок для генерации думающего датасета.
Версия с исправленным CALL (цель: 10-15% в датасете).
"""
import random
from collections import Counter
from deep_analyzer import (
    build_deck, Card, evaluate_hand, HAND_NAMES,
    analyze_board_features, board_texture, classify_preflop_hand,
    RANKS, SUITS, RANK_NAMES
)


# ============ ПАРСЕР ДИАПАЗОНОВ ============

def _rank_from_str(ch):
    if ch == "A": return 14
    if ch == "K": return 13
    if ch == "Q": return 12
    if ch == "J": return 11
    if ch == "T": return 10
    return int(ch)


def _expand_range(range_str):
    hands = []
    if not range_str:
        return hands
    for token in range_str.split(","):
        token = token.strip()
        if not token:
            continue
        plus = token.endswith("+")
        if plus:
            token = token[:-1]
        suited = token.endswith("s")
        offsuit = token.endswith("o")
        if suited or offsuit:
            token = token[:-1]
        if len(token) != 2:
            continue
        r1 = _rank_from_str(token[0])
        r2 = _rank_from_str(token[1])
        if plus:
            hi = max(r1, r2)
            lo = min(r1, r2)
            if hi == lo:
                for r in range(hi, 15):
                    hands.append((r, r, False))
            else:
                for kicker in range(lo, hi + 1):
                    hands.append((hi, kicker, suited))
        else:
            if r1 == r2:
                hands.append((r1, r2, False))
            else:
                hands.append((max(r1, r2), min(r1, r2), suited))
    return hands


def _make_cards_from_ranks(r1, r2, suited, used):
    suits = list(SUITS)
    random.shuffle(suits)
    if r1 == r2:
        for s1 in suits:
            for s2 in suits:
                if s1 == s2:
                    continue
                c1 = Card(r1, s1)
                c2 = Card(r2, s2)
                if c1 not in used and c2 not in used:
                    return [c1, c2]
        return None
    else:
        for s1 in suits:
            for s2 in suits:
                if suited:
                    if s1 != s2:
                        continue
                else:
                    if s1 == s2:
                        continue
                c1 = Card(r1, s1)
                c2 = Card(r2, s2)
                if c1 not in used and c2 not in used:
                    return [c1, c2]
        return None


# ============ ЭКВИТИ ============

def equity_vs_range(hero_cards, board, villain_range, num_simulations=200):
    range_hands = _expand_range(villain_range)
    if not range_hands:
        return 0.5
    deck = build_deck()
    used = set(hero_cards) | set(board)
    deck = [c for c in deck if c not in used]
    wins = ties = 0
    valid = 0
    for _ in range(num_simulations):
        r1, r2, suited = random.choice(range_hands)
        villain_cards = _make_cards_from_ranks(r1, r2, suited, used)
        if villain_cards is None:
            continue
        valid += 1
        need = 5 - len(board)
        deck_copy = list(deck)
        random.shuffle(deck_copy)
        final_board = board + deck_copy[:need]
        hero_hand = evaluate_hand(hero_cards + final_board)
        villain_hand = evaluate_hand(villain_cards + final_board)
        if hero_hand > villain_hand:
            wins += 1
        elif hero_hand == villain_hand:
            ties += 1
    if valid == 0:
        return 0.5
    return (wins + ties * 0.5) / valid


# ============ EV-ФУНКЦИИ ============

def calc_ev_fold(pot, to_call):
    return 0.0


def calc_ev_call(equity, pot, to_call):
    if to_call <= 0:
        return equity * pot
    return equity * (pot + to_call) - (1 - equity) * to_call


def calc_ev_bet(equity, pot, bet_size, fold_equity=0.3):
    if bet_size <= 0:
        return 0.0
    ev_fold = fold_equity * pot
    ev_call = (1 - fold_equity) * (equity * (pot + 2 * bet_size) - bet_size)
    return ev_fold + ev_call


def calc_ev_check(equity, pot):
    return equity * pot


def estimate_fold_equity(board, street, hero_strength):
    """[FIX] Уменьшено — реалистичнее."""
    if street == "preflop" or not board:
        return 0.4
    suits = [c.suit for c in board]
    ranks = [c.rank for c in board]
    suit_counts = Counter(suits)
    wet = 0.0
    if max(suit_counts.values()) >= 2:
        wet += 0.2
    if len(set(ranks)) < len(ranks):
        wet += 0.15
    high_cards = sum(1 for r in ranks if r >= 10)
    wet += high_cards * 0.05
    base = 0.35 - wet
    if hero_strength >= 40:
        base -= 0.1
    return max(0.05, min(0.25, base))


# ============ ПРЕФЛОП ============

def best_action_preflop(hero_cards, position, action_before, stack, pot, to_call,
                        villain_range):
    equity = equity_vs_range(hero_cards, [], villain_range, num_simulations=100)
    tier, hand_name, _ = classify_preflop_hand(hero_cards[0], hero_cards[1])
    real_pot = pot + to_call

    evs = {}
    evs["FOLD"] = 0.0
    evs["CALL"] = calc_ev_call(equity, real_pot, to_call)
    evs["RAISE"] = calc_ev_bet(equity, real_pot, 30, fold_equity=0.4)

    if tier >= 6:
        evs["RAISE"] = -999
        evs["CALL"] = -999
    elif tier >= 5 and position in ("UTG", "MP"):
        evs["RAISE"] = -999
        evs["CALL"] = -999

    # [FIX] Средние руки (tier 3-4) — CALL вместо RAISE, если CALL > 0
    if tier in (3, 4) and evs["CALL"] > 0:
        if evs["CALL"] > evs.get("RAISE", -999) * 0.8:
            evs["RAISE"] = -999

    best = max(evs, key=evs.get)
    if best == "RAISE":
        amount = 30
    elif best == "CALL":
        amount = to_call
    else:
        amount = 0
    return best, amount, evs[best], evs


# ============ ПОСТФЛОП ============

def best_action_postflop(hero_cards, board, pot, to_call, villain_range,
                          street, stack):
    """
    [FIX] Полностью переписано.
    CALL должен появляться 10-15% в датасете.
    """
    equity = equity_vs_range(hero_cards, board, villain_range, num_simulations=200)
    hero_hand = evaluate_hand(hero_cards + board)
    strength = hero_hand[0] * 10 + 5
    fe = estimate_fold_equity(board, street, strength)

    evs = {}
    evs["FOLD"] = 0.0

    # 1. Базовые EV
    if to_call == 0:
        evs["CHECK"] = calc_ev_check(equity, pot)
        evs["CALL"] = -999
        evs["BET"] = calc_ev_bet(equity, pot, int(pot * 0.66), fold_equity=fe)
        evs["RAISE"] = -999
    else:
        evs["CHECK"] = -999
        evs["CALL"] = calc_ev_call(equity, pot, to_call)
        evs["RAISE"] = calc_ev_bet(equity, pot, int(to_call * 2.5), fold_equity=fe * 0.8)
        evs["BET"] = -999

    # 2. [FIX] Главное правило: средние руки — CALL, не RAISE
    #    Условие: эквити 20-60%, CALL положительный, ставка не огромная
    if (to_call > 0
            and 0.20 < equity < 0.60
            and evs["CALL"] > 0
            and to_call < pot * 0.8):
        evs["RAISE"] = -999
        evs["FOLD"] = -999
        # CALL остаётся — не трогаем!

    # 3. [FIX] Ловля блефа: маленькая ставка + маргинальная рука
    if (to_call > 0
            and to_call < pot * 0.4
            and 0.15 < equity < 0.25
            and random.random() < 0.4):
        evs["CALL"] = max(evs["CALL"], 1.0)
        evs["RAISE"] = -999
        evs["FOLD"] = -999

    # 4. [FIX] Слабые руки — FOLD на большие ставки (только если CALL < 0)
    if strength < 20 and to_call > pot * 0.5 and evs["CALL"] < 0:
        pass  # CALL уже отрицательный — FOLD выберется

    # 5. [FIX] Ривер: слабые не блефуют
    if street == "river" and strength < 20 and "BET" in evs and evs["BET"] > -999:
        evs["BET"] *= 0.5

    # 6. Выбор best
    best = max(evs, key=evs.get)
    amount = 0
    if best == "CALL":
        amount = to_call
    elif best == "BET":
        amount = int(pot * 0.66)
    elif best == "RAISE":
        amount = int(to_call * 2.5)

    return best, amount, evs[best], evs