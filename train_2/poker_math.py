"""
poker_math.py - Расширенная математика для покера.
Считает точные вероятности улучшения, ауты, диапазон оппонента.
"""
import random
from collections import Counter, defaultdict
from itertools import combinations
from analyzer import build_deck, Card, evaluate_hand, RANKS, SUITS, HAND_NAMES


def calculate_exact_outs(hero_cards, board):
    """
    Считает точные ауты для улучшения руки.
    Возвращает словарь: {тип_улучшения: [список карт-аутов]}
    """
    deck = build_deck()
    used = set(hero_cards) | set(board)
    available = [c for c in deck if c not in used]
    
    current_hand = evaluate_hand(hero_cards + board)
    current_category = current_hand[0]
    
    outs = defaultdict(list)
    
    for card in available:
        # Проверяем, улучшает ли эта карта руку
        new_hand = evaluate_hand(hero_cards + board + [card])
        new_category = new_hand[0]
        
        if new_category > current_category:
            # Определяем тип улучшения
            if new_category == 9:
                outs["Роял-флеш"].append(card)
            elif new_category == 8:
                outs["Стрит-флеш"].append(card)
            elif new_category == 7:
                outs["Каре"].append(card)
            elif new_category == 6:
                outs["Фулл-хаус"].append(card)
            elif new_category == 5:
                outs["Флеш"].append(card)
            elif new_category == 4:
                outs["Стрит"].append(card)
            elif new_category == 3:
                outs["Сет"].append(card)
            elif new_category == 2:
                outs["Две пары"].append(card)
            elif new_category == 1:
                outs["Пара"].append(card)
    
    return dict(outs)


def calculate_improvement_probs_flop(hero_cards, board):
    """
    Точные вероятности улучшения на флопе (до ривера).
    Перебирает все 47*46=2162 комбинации тёрн+ривер.
    """
    if len(board) != 3:
        return {}
    
    deck = build_deck()
    used = set(hero_cards) | set(board)
    available = [c for c in deck if c not in used]
    
    current_hand = evaluate_hand(hero_cards + board)
    current_category = current_hand[0]
    
    improvements = Counter()
    total = 0
    
    # Перебор всех комбинаций тёрн+ривер
    for turn_card in available:
        for river_card in available:
            if turn_card == river_card:
                continue
            
            new_hand = evaluate_hand(hero_cards + board + [turn_card, river_card])
            new_category = new_hand[0]
            
            if new_category > current_category:
                improvements[new_category] += 1
            
            total += 1
    
    # Конвертируем в проценты
    probs = {}
    for cat, count in improvements.items():
        probs[HAND_NAMES[cat]] = count / total * 100
    
    return probs


def calculate_improvement_probs_turn(hero_cards, board):
    """
    Точные вероятности улучшения на тёрне (до ривера).
    Перебирает все 47 карт ривера.
    """
    if len(board) != 4:
        return {}
    
    deck = build_deck()
    used = set(hero_cards) | set(board)
    available = [c for c in deck if c not in used]
    
    current_hand = evaluate_hand(hero_cards + board)
    current_category = current_hand[0]
    
    improvements = Counter()
    total = len(available)
    
    for river_card in available:
        new_hand = evaluate_hand(hero_cards + board + [river_card])
        new_category = new_hand[0]
        
        if new_category > current_category:
            improvements[new_category] += 1
    
    probs = {}
    for cat, count in improvements.items():
        probs[HAND_NAMES[cat]] = count / total * 100
    
    return probs


def estimate_opponent_range(hero_cards, board, villain_range_name, num_samples=100):
    """
    Оценивает возможные комбинации оппонента на основе его диапазона.
    Сэмплирует руки из диапазона и оценивает их на текущем борде.
    """
    from dataset_flop_v2 import RANGES, get_range_by_tier, make_cards
    
    if villain_range_name not in RANGES:
        return {}
    
    range_def = RANGES[villain_range_name]
    possible_hands = get_range_by_tier(range_def)
    
    if not possible_hands:
        return {}
    
    used = set(hero_cards) | set(board)
    combinations_count = Counter()
    
    samples = random.sample(possible_hands, min(num_samples, len(possible_hands)))
    
    for r1, r2, suited in samples:
        villain_cards = make_cards(r1, r2, suited, used)
        if villain_cards is None:
            continue
        
        if len(board) >= 3:
            hand = evaluate_hand(villain_cards + board)
            combinations_count[HAND_NAMES[hand[0]]] += 1
        else:
            # Префлоп — просто считаем стартовые руки
            if r1 == r2:
                combinations_count["Пара"] += 1
            elif suited:
                combinations_count["Одномастные"] += 1
            else:
                combinations_count["Разномастные"] += 1
    
    # Конвертируем в проценты
    total = sum(combinations_count.values())
    if total == 0:
        return {}
    
    return {k: v / total * 100 for k, v in combinations_count.items()}


def format_outs(outs_dict):
    """Форматирует ауты в читаемый вид."""
    if not outs_dict:
        return "Нет улучшений"
    
    parts = []
    for improvement, cards in outs_dict.items():
        parts.append(f"{improvement}: {len(cards)} аут")
    
    return ", ".join(parts)


def format_probs(probs_dict):
    """Форматирует вероятности в читаемый вид."""
    if not probs_dict:
        return "Нет данных"
    
    parts = []
    for hand_name, prob in sorted(probs_dict.items(), key=lambda x: -x[1]):
        if prob > 0.5:  # показываем только вероятности > 0.5%
            parts.append(f"{hand_name}: {prob:.1f}%")
    
    return ", ".join(parts) if parts else "Маловероятно"


def format_opponent_range(range_dict):
    """Форматирует диапазон оппонента."""
    if not range_dict:
        return "Неизвестно"
    
    parts = []
    for hand_name, prob in sorted(range_dict.items(), key=lambda x: -x[1]):
        if prob > 1:  # показываем только > 1%
            parts.append(f"{hand_name}: {prob:.0f}%")
    
    return ", ".join(parts) if parts else "Широкий диапазон"