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

DATA_SET_FILE = "./train_2/dataset_1000.json"

# ============ ШАБЛОНЫ АНАЛИЗА ДЛЯ РАЗНООБРАЗИЯ ============
# ANALYSIS_TEMPLATES = {
#     "preflop": {
#         "strong": [
#             "Сильная рука {hand} — открываем рейзом для вэлью",
#             "{hand} из позиции {pos} — стандартный рейз",
#             "Премиум-рука {hand} — всегда рейзим",
#             "С {hand} играем агрессивно — рейз",
#             "{hand} — отличная рука для открытия",
#             "Сильная стартовая рука — рейз для изоляции",
#             "{hand} из {pos} — рейз для построения банка",
#         ],
#         "medium": [
#             "Играбельная рука {hand} — рейз из {pos}",
#             "{hand} из поздней позиции — можно открыть",
#             "Средняя рука {hand} — рейз для стила",
#             "{hand} — играем на позицию",
#             "С {hand} можно открыть рейзом",
#             "{hand} из {pos} — стандартное открытие",
#         ],
#         "weak": [
#             "Слабая рука {hand} — фолд",
#             "{hand} недостаточно сильна — сбрасываем",
#             "С {hand} лучше подождать лучшую ситуацию",
#             "{hand} — слишком слаба для входа",
#             "Фолд с {hand}, ждём mejores карты",
#         ],
#     },
#     "flop": {
#         "monster": [
#             "Монстр {hand} — ставка для вэлью",
#             "{hand} на флопе — большая ставка для защиты",
#             "Сильное попадание {hand} — бет для максимума",
#             "{hand} — рейз для извлечения вэлью",
#             "Натсовая рука {hand} — агрессивная игра",
#             "{hand} — ставка, не даём бесплатных карт",
#             "С {hand} строим банк — рейз",
#         ],
#         "strong": [
#             "{hand} — продолженная ставка",
#             "Топ-пара — бет для защиты эквити",
#             "{hand} — стандартный контбет",
#             "С {hand} продолжаем агрессию",
#             "{hand} — бет, оппонент может иметь хуже",
#             "Сильная рука — ставка для вэлью",
#         ],
#         "draw": [
#             "Дро с аутами — колл, надеемся на улучшение",
#             "Флеш/стрит-дро — колл с правильными оддсами",
#             "Дро — чек, смотрим бесплатную карту",
#             "Потенциал на улучшение — колл",
#             "Дро с {outs} аутами — математически правильный колл",
#             "С дро продолжаем, эквити позволяет",
#         ],
#         "weak": [
#             "Слабая рука — чек, пытаемся улучшить",
#             "{hand} — чек, нет силы для агрессии",
#             "Мало попаданий — чек",
#             "Слабое попадание — пассивная игра",
#         ],
#         "air": [
#             "Воздух — чек, сдаёмся",
#             "Нет попадания — фолд против ставки",
#             "Воздух против агрессии — фолд",
#             "Ничего не собрали — сдаёмся",
#         ],
#     },
#     "turn": {
#         "made": [
#             "Готовая рука {hand} — вэлью-бет",
#             "{hand} на тёрне — ставка для вэлью",
#             "Сильная рука — бет, оппонент может коллировать",
#             "{hand} — продолжаем агрессию",
#             "С {hand} строим банк на тёрне",
#         ],
#         "draw": [
#             "Дро на тёрне — колл, шансы ~20%",
#             "Флеш/стрит-дро — колл с правильными оддсами",
#             "Дро — чек, одна карта до ривера",
#             "С дро продолжаем, эквити позволяет",
#         ],
#         "missed": [
#             "Дро не закрылось — фолд",
#             "Не попали — сдаёмся",
#             "Воздух на тёрне — фолд",
#         ],
#     },
#     "river": {
#         "strong": [
#             "Сильная рука {hand} — колл или вэлью-рейз",
#             "{hand} на ривере — колл с хорошим эквити",
#             "Финальная сила {hand} — колл",
#             "С {hand} коллируем, эквити позволяет",
#         ],
#         "medium": [
#             "Средняя рука — колл если пот-оддсы позволяют",
#             "{hand} — колл, банк большой",
#             "Средняя сила — колл",
#         ],
#         "weak": [
#             "Слабая рука — фолд",
#             "{hand} недостаточно — фолд",
#             "Нет силы для колла — фолд",
#         ],
#     },
# }

# ============ ШАБЛОНЫ АНАЛИЗА ДЛЯ РАЗНООБРАЗИЯ (РАСШИРЕННАЯ ВЕРСИЯ) ============
ANALYSIS_TEMPLATES = {
    "preflop": {
        "strong": [
            "Сильная рука {hand} — открываем рейзом для вэлью",
            "{hand} из позиции {pos} — стандартный рейз",
            "Премиум-рука {hand} — всегда рейзим",
            "С {hand} играем агрессивно — рейз",
            "{hand} — отличная рука для открытия",
            "Сильная стартовая рука — рейз для изоляции",
            "{hand} из {pos} — рейз для построения банка",
            "{hand} — рука, с которой нельзя пасовать, рейз",
            "С {hand} входим рейзом, не лимпом",
            "{hand} — топ-рука, максимизируем вэлью рейзом",
            "Открываем {hand} рейзом — слишком сильна для фолда",
            "{hand} из {pos} — рейз 2.5–3 BB для контроля банка",
            "С {hand} рейз, готовы играть на большой банк",
            "{hand} — рука для 3-бета, играем агрессивно",
            "{hand} — рейз, изолируем слабые руки оппонентов",
        ],
        "medium": [
            "Играбельная рука {hand} — рейз из {pos}",
            "{hand} из поздней позиции — можно открыть",
            "Средняя рука {hand} — рейз для стила",
            "{hand} — играем на позицию",
            "С {hand} можно открыть рейзом",
            "{hand} из {pos} — стандартное открытие",
            "{hand} — открываем рейзом, но осторожно на 3-бет",
            "С {hand} играем рейз из поздней позиции",
            "{hand} — рейз для кражи блайндов",
            "С {hand} открываемся, если стол не слишком агрессивный",
            "{hand} из {pos} — рейз, но готовы сбросить на агрессию",
            "Средняя рука {hand} — открываем рейзом для инициативы",
            "{hand} — играбельно, рейз из {pos}",
            "С {hand} рейз, но не переоцениваем силу",
        ],
        "weak": [
            "Слабая рука {hand} — фолд",
            "{hand} недостаточно сильна — сбрасываем",
            "С {hand} лучше подождать лучшую ситуацию",
            "{hand} — слишком слаба для входа",
            "Фолд с {hand}, ждём лучшие карты",
            "{hand} — фолд, не тратим деньги на мусор",
            "С {hand} пас, рука не играбельна из {pos}",
            "{hand} — фолд, слишком много доминирующих рук",
            "С {hand} сбрасываем — нет эквити для входа",
            "{hand} — фолд, ждём премиум-руку",
            "Не открываем {hand} — слабая позиция и рука",
            "{hand} — пас, не стоит ввязываться",
        ],
        "premium": [
            "AA/KK — всегда рейз, не медлим",
            "{hand} — премиум, рейз и готовность к 4-бету",
            "С {hand} играем на стек, рейз",
            "{hand} — рука для 3-бета и 4-бета",
            "С {hand} максимизируем вэлью — рейз",
        ],
        "speculative": [
            "{hand} — спекулятивная рука, играем на имплайд-оддсы",
            "С {hand} можно войти, если оддсы позволяют",
            "{hand} — рука для сет-майнинга, колл при хороших оддсах",
            "С {hand} играем только в поздней позиции",
            "{hand} — спекулятивно, но потенциал есть",
        ],
    },
    "flop": {
        "monster": [
            "Монстр {hand} — ставка для вэлью",
            "{hand} на флопе — большая ставка для защиты",
            "Сильное попадание {hand} — бет для максимума",
            "{hand} — рейз для извлечения вэлью",
            "Натсовая рука {hand} — агрессивная игра",
            "{hand} — ставка, не даём бесплатных карт",
            "С {hand} строим банк — рейз",
            "{hand} — натс, бет 75–100% пота",
            "С {hand} играем на стек — рейз",
            "{hand} — монстр, не слоу-плеим на дро-борде",
            "С {hand} бет-бет-бет, максимизируем вэлью",
            "{hand} — топ-рука, ставка для защиты от дро",
            "На флопе {hand} — большая ставка, оппонент может платить",
        ],
        "strong": [
            "{hand} — продолженная ставка",
            "Топ-пара — бет для защиты эквити",
            "{hand} — стандартный контбет",
            "С {hand} продолжаем агрессию",
            "{hand} — бет, оппонент может иметь хуже",
            "Сильная рука — ставка для вэлью",
            "{hand} — топ-пара, бет 50–66% пота",
            "С {hand} контбет, защищаем эквити",
            "{hand} — сильная рука, но осторожно на дро-борде",
            "С {hand} бет для вэлью, оппонент часто коллирует хуже",
            "{hand} — ставка, не даём бесплатно улучшиться",
            "Топ-пара {hand} — бет, контролируем размер банка",
        ],
        "draw": [
            "Дро с аутами — колл, надеемся на улучшение",
            "Флеш/стрит-дро — колл с правильными оддсами",
            "Дро — чек, смотрим бесплатную карту",
            "Потенциал на улучшение — колл",
            "Дро с {outs} аутами — математически правильный колл",
            "С дро продолжаем, эквити позволяет",
            "Флеш-дро — полу-блеф рейз для фолд-эквити",
            "Стрит-дро — колл, оддсы позволяют",
            "Дро с оверкартами — колл, много аутов",
            "Комбо-дро — играем агрессивно, много эквити",
            "С дро {outs} аутов — колл, имплайд-оддсы хорошие",
            "Дро — чек-колл, не строим банк без попадания",
            "Флеш-дро — бет для полу-блефа, есть фолд-эквити",
        ],
        "weak": [
            "Слабая рука — чек, пытаемся улучшить",
            "{hand} — чек, нет силы для агрессии",
            "Мало попаданий — чек",
            "Слабое попадание — пассивная игра",
            "{hand} — чек, контролируем банк",
            "Слабая рука — чек-фолд на агрессию",
            "{hand} — чек, не вкладываем деньги без силы",
            "Слабое попадание — чек, смотрим тёрн бесплатно",
            "{hand} — пассивно, нет эквити для ставки",
        ],
        "air": [
            "Воздух — чек, сдаёмся",
            "Нет попадания — фолд против ставки",
            "Воздух против агрессии — фолд",
            "Ничего не собрали — сдаёмся",
            "Воздух — чек-фолд, не тратим деньги",
            "Нет эквити — фолд на ставку",
            "Воздух — можно полу-блеф, если оппонент слаб",
            "Ничего не собрали — чек, готовы сбросить",
            "Воздух — сдаёмся, ждём лучшую ситуацию",
        ],
        "bluff": [
            "Воздух — полу-блеф бет, есть фолд-эквити",
            "Слабый дро — блеф-бет для давления",
            "Нет попадания — блеф на сухом борде",
            "Воздух — ставка для кражи банка",
        ],
    },
    "turn": {
        "made": [
            "Готовая рука {hand} — вэлью-бет",
            "{hand} на тёрне — ставка для вэлью",
            "Сильная рука — бет, оппонент может коллировать",
            "{hand} — продолжаем агрессию",
            "С {hand} строим банк на тёрне",
            "{hand} — бет 66–75% пота для защиты",
            "С {hand} вэлью-бет, оппонент часто платит",
            "{hand} — сильная рука, но следим за дро",
            "С {hand} бет, не даём бесплатной карты на ривер",
            "{hand} — ставка для максимизации вэлью",
        ],
        "draw": [
            "Дро на тёрне — колл, шансы ~20%",
            "Флеш/стрит-дро — колл с правильными оддсами",
            "Дро — чек, одна карта до ривера",
            "С дро продолжаем, эквити позволяет",
            "Дро на тёрне — колл, имплайд-оддсы хорошие",
            "Флеш-дро — колл, шансы на ривере ~20%",
            "Дро — чек-колл, не строим банк без попадания",
            "Стрит-дро — колл, потенциально большой банк",
        ],
        "missed": [
            "Дро не закрылось — фолд",
            "Не попали — сдаёмся",
            "Воздух на тёрне — фолд",
            "Дро не закрылось — фолд, не тратим деньги",
            "Не попали — чек-фолд на ставку",
            "Воздух — сдаёмся, ривер не поможет",
        ],
        "strong": [
            "{hand} — топ-пара на тёрне, бет для вэлью",
            "С {hand} продолжаем ставку, оппонент слаб",
            "{hand} — сильная рука, бет 50–75% пота",
            "С {hand} вэлью-бет, защищаем эквити",
        ],
    },
    "river": {
        "strong": [
            "Сильная рука {hand} — колл или вэлью-рейз",
            "{hand} на ривере — колл с хорошим эквити",
            "Финальная сила {hand} — колл",
            "С {hand} коллируем, эквити позволяет",
            "{hand} — вэлью-бет на ривере",
            "С {hand} рейз для вэлью, оппонент часто платит",
            "{hand} — сильная рука, колл или рейз",
            "С {hand} максимизируем вэлью на ривере",
        ],
        "medium": [
            "Средняя рука — колл если пот-оддсы позволяют",
            "{hand} — колл, банк большой",
            "Средняя сила — колл",
            "{hand} — колл, но осторожно на большие ставки",
            "Средняя рука — колл, пот-оддсы хорошие",
            "{hand} — колл, не рейзим без натса",
        ],
        "weak": [
            "Слабая рука — фолд",
            "{hand} недостаточно — фолд",
            "Нет силы для колла — фолд",
            "{hand} — фолд, не платим за проигрыш",
            "Слабая рука — фолд на ставку",
            "{hand} — пас, эквити недостаточно",
        ],
        "bluff_catch": [
            "Средняя рука — колл, ловим блеф",
            "{hand} — колл, оппонент может блефовать",
            "С {hand} коллируем, пот-оддсы позволяют",
            "Ловим блеф с {hand} — колл",
        ],
        "bluff": [
            "Воздух — блеф на ривере, оппонент слаб",
            "Нет попадания — блеф-бет для фолд-эквити",
            "Слабый дро — блеф на ривере",
            "Воздух — ставка для кражи банка",
        ],
    },
}

def get_analysis_text(stage, category, **kwargs):
    """Возвращает рандомизированный текст анализа."""
    templates = ANALYSIS_TEMPLATES.get(stage, {}).get(category, ["Решение"])
    template = random.choice(templates)
    return template.format(**kwargs)


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
    
    tier, hand_name, hand_desc = classify_preflop_hand(hero_cards[0], hero_cards[1])
    opponent_range = estimate_opponent_range(hero_cards, [], villain_range_name)
    
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
    
    if tier <= 2:
        action = "RAISE"
        amount = 30
        category = "strong"
    elif tier <= 4:
        action = "RAISE"
        amount = 30
        category = "medium"
    else:
        action = "FOLD"
        amount = 0
        category = "weak"
    
    analysis_text = get_analysis_text("preflop", category, hand=hand_name, pos=scenario['hero_position'])
    
    completion = f"""АНАЛИЗ: {analysis_text}
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
    
    hero_hand = evaluate_hand(hero_cards + board)
    hero_hand_name = HAND_NAMES[hero_hand[0]]
    
    outs = calculate_exact_outs(hero_cards, board)
    probs = calculate_improvement_probs_flop(hero_cards, board)
    opponent_range = estimate_opponent_range(hero_cards, board, villain_range_name)
    
    villain_action = simulate_villain_action(villain_cards, board, 60)
    
    if villain_action[0] == "CHECK":
        opponent_action_str = "Оппонент прочекал"
        to_call = 0
    else:
        opponent_action_str = f"Оппонент поставил ${villain_action[1]}"
        to_call = villain_action[1]
    
    pot = 60
    stack = 1000
    
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
    
    if hero_hand[0] >= 3:
        action = "BET" if villain_action[0] == "CHECK" else "RAISE"
        amount = int(pot * 0.66) if action == "BET" else int(villain_action[1] * 2.5)
        category = "monster"
    elif hero_hand[0] >= 1:
        if villain_action[0] == "CHECK":
            action = "BET"
            amount = int(pot * 0.5)
            category = "strong"
        else:
            action = "CALL"
            amount = to_call
            category = "strong"
    elif outs:
        if villain_action[0] == "CHECK":
            action = "CHECK"
            amount = 0
            category = "draw"
        else:
            action = "CALL"
            amount = to_call
            category = "draw"
    else:
        if villain_action[0] == "CHECK":
            action = "CHECK"
            amount = 0
            category = "air"
        else:
            action = "FOLD"
            amount = 0
            category = "air"
    
    analysis_text = get_analysis_text("flop", category, hand=hero_hand_name, outs=len(outs))
    
    completion = f"""АНАЛИЗ: {analysis_text}
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
    turn_card = random.choice([c for c in build_deck() if c not in set(hero_cards) | set(villain_cards) | set(board)])
    board.append(turn_card)
    
    hero_hand = evaluate_hand(hero_cards + board)
    hero_hand_name = HAND_NAMES[hero_hand[0]]
    
    outs = calculate_exact_outs(hero_cards, board)
    probs = calculate_improvement_probs_turn(hero_cards, board)
    opponent_range = estimate_opponent_range(hero_cards, board, villain_range_name)
    
    villain_action = simulate_villain_action(villain_cards, board, 100)
    
    if villain_action[0] == "CHECK":
        opponent_action_str = "Оппонент прочекал"
        to_call = 0
    else:
        opponent_action_str = f"Оппонент поставил ${villain_action[1]}"
        to_call = villain_action[1]
    
    pot = 100
    stack = 1000
    
    prompt = f"""[СТАДИЯ: ТЁРН]

=== СИТУАЦИЯ: ТЁРН ===

ТВОИ КАРТЫ: {hero_cards[0]}, {hero_cards[1]}
БОРД (4 карт): {board[0]}, {board[1]}, {board[2]}, {board[3]}

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

Ответь СТРОГО в формате:
АНАЛИЗ: [твоя логика]
МОИ КОМБИНАЦИИ: [текущая] → [возможные]
КОМБИНАЦИИ ОПОННЕНТА: [что он может иметь]
ДЕЙСТВИЕ: [FOLD/CHECK/CALL/BET/RAISE]
СУММА: [число]"""
    
    if hero_hand[0] >= 2:
        action = "BET" if villain_action[0] == "CHECK" else "CALL"
        amount = int(pot * 0.66) if action == "BET" else to_call
        category = "made"
    elif outs:
        action = "CALL" if villain_action[0] == "BET" else "CHECK"
        amount = to_call if action == "CALL" else 0
        category = "draw"
    else:
        action = "FOLD" if villain_action[0] == "BET" else "CHECK"
        amount = 0
        category = "missed"
    
    analysis_text = get_analysis_text("turn", category, hand=hero_hand_name)
    
    completion = f"""АНАЛИЗ: {analysis_text}
МОИ КОМБИНАЦИИ: {hero_hand_name} → {format_probs(probs)}
КОМБИНАЦИИ ОПОННЕНТА: {format_opponent_range(opponent_range)}
ДЕЙСТВИЕ: {action}
СУММА: {amount}"""
    
    return {
        "prompt": prompt,
        "completion": completion,
        "reward": 1.0,
        "stage": "turn",
        "metadata": {
            "scenario": scenario['name'],
            "hero_hand": f"{hero_cards[0]}, {hero_cards[1]}",
            "board": f"{board[0]}, {board[1]}, {board[2]}, {board[3]}",
            "hero_combo": hero_hand_name,
        }
    }


def generate_river_example():
    """Генерирует пример для ривера."""
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
    
    # Добавляем тёрн и ривер
    used = set(hero_cards) | set(villain_cards) | set(board)
    remaining = [c for c in build_deck() if c not in used]
    random.shuffle(remaining)
    board.extend(remaining[:2])
    
    hero_hand = evaluate_hand(hero_cards + board)
    hero_hand_name = HAND_NAMES[hero_hand[0]]
    
    opponent_range = estimate_opponent_range(hero_cards, board, villain_range_name)
    
    villain_action = simulate_villain_action(villain_cards, board, 150)
    
    if villain_action[0] == "CHECK":
        opponent_action_str = "Оппонент прочекал"
        to_call = 0
    else:
        opponent_action_str = f"Оппонент поставил ${villain_action[1]}"
        to_call = villain_action[1]
    
    pot = 150
    stack = 1000
    
    # На ривере нет улучшений, только финальная сила
    hero_strength = hero_hand[0]
    
    prompt = f"""[СТАДИЯ: РИВЕР]

=== СИТУАЦИЯ: РИВЕР ===

ТВОИ КАРТЫ: {hero_cards[0]}, {hero_cards[1]}
БОРД (5 карт): {board[0]}, {board[1]}, {board[2]}, {board[3]}, {board[4]}

— АНАЛИЗ ОТ СИСТЕМЫ —
Твоя финальная комбинация: {hero_hand_name}
Сила руки (0-100): {hero_hand[0] * 10 + 5}

ВОЗМОЖНЫЕ КОМБИНАЦИИ ОПОННЕНТА:
{format_opponent_range(opponent_range)}

— ДЕЙСТВИЕ ОППОНЕНТА —
{opponent_action_str}

— КОНТЕКСТ —
Позиция: {scenario['hero_position']}
Стек: ${stack}, Банк: ${pot}

Ответь СТРОГО в формате:
АНАЛИЗ: [твоя логика]
МОИ КОМБИНАЦИИ: [финальная рука]
КОМБИНАЦИИ ОПОННЕНТА: [что он может иметь]
ДЕЙСТВИЕ: [FOLD/CHECK/CALL/BET/RAISE]
СУММА: [число]"""
    
    if hero_strength >= 4:  # стрит+
        action = "CALL" if villain_action[0] == "BET" else "BET"
        amount = to_call if action == "CALL" else int(pot * 0.66)
        category = "strong"
    elif hero_strength >= 2:  # две пары+
        action = "CALL" if villain_action[0] == "BET" else "CHECK"
        amount = to_call if action == "CALL" else 0
        category = "medium"
    else:
        action = "FOLD" if villain_action[0] == "BET" else "CHECK"
        amount = 0
        category = "weak"
    
    analysis_text = get_analysis_text("river", category, hand=hero_hand_name)
    
    completion = f"""АНАЛИЗ: {analysis_text}
МОИ КОМБИНАЦИИ: {hero_hand_name} (финальная)
КОМБИНАЦИИ ОПОННЕНТА: {format_opponent_range(opponent_range)}
ДЕЙСТВИЕ: {action}
СУММА: {amount}"""
    
    return {
        "prompt": prompt,
        "completion": completion,
        "reward": 1.0,
        "stage": "river",
        "metadata": {
            "scenario": scenario['name'],
            "hero_hand": f"{hero_cards[0]}, {hero_cards[1]}",
            "board": f"{board[0]}, {board[1]}, {board[2]}, {board[3]}, {board[4]}",
            "hero_combo": hero_hand_name,
        }
    }


def generate_unified_dataset(count_per_stage=100, seed=42):
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
    
    print(f"Генерация тёрн ({count_per_stage} примеров)...")
    for _ in range(count_per_stage):
        ex = generate_turn_example()
        if ex:
            examples.append(ex)
    
    print(f"Генерация ривер ({count_per_stage} примеров)...")
    for _ in range(count_per_stage):
        ex = generate_river_example()
        if ex:
            examples.append(ex)
    
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
    unique_completions = len(set(ex["completion"] for ex in examples))
    
    print("\n📊 Статистика:")
    for stage, count in stages.items():
        print(f"  {stage}: {count}")
    print(f"  Уникальных completion: {unique_completions} ({unique_completions/len(examples)*100:.1f}%)")


if __name__ == "__main__":
    print("🎯 Генерация единого датасета...\n")
    
    examples = generate_unified_dataset(count_per_stage=1000)
    
    save_dataset(examples, DATA_SET_FILE)
    show_statistics(examples)
    
    print("\n" + "=" * 60)
    print("ПРИМЕРЫ ПО СТАДИЯМ:")
    print("=" * 60)
    
    for stage in ["preflop", "flop", "turn", "river"]:
        stage_examples = [ex for ex in examples if ex["stage"] == stage]
        if stage_examples:
            print(f"\n--- {stage.upper()} ---")
            print("\nПРОМПТ:")
            print(stage_examples[0]["prompt"][:500] + "...")
            print("\nОТВЕТ:")
            print(stage_examples[0]["completion"])