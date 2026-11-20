"""
dataset_unified.py - Единый датасет для всех стадий покера.
Префлоп + Флоп + Тёрн + Ривер с полной информацией о комбинациях.

[FIX 1] Детерминированный выбор шаблонов (хэш по prompt) — убирает конфликтные prompt.
[FIX 2] Конкретные карты в подстановках — повышает уникальность completion.
[FIX 3] build_*_completion() с числами, оддсами и логикой — учит модель рассуждать.
[FIX 4] format_hand() и format_board() — единый формат карт.
[FIX 5] Исправлена опечатка "mejores" -> "лучшие".
"""
import random
import json
import hashlib
from collections import Counter

from analyzer import (
    build_deck, Card, analyze_postflop, classify_preflop_hand,
    RANKS, SUITS, HAND_NAMES, evaluate_hand,
    RANK_NAMES,  # [FIX] добавьте
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

DATA_SET_FILE = "./train_2/dataset_unified_2000.json"

# ============ ШАБЛОНЫ АНАЛИЗА (РАСШИРЕННАЯ ВЕРСИЯ) ============
ANALYSIS_TEMPLATES = {
    "preflop": {
        "strong": [
            "Сильная рука {hand} — открываем рейзом для вэлью",
            "{hand} из позиции {pos} — стандартный рейз",
            "Премиум-рука {hand} — всегда рейзим",
            "С {hand} играем агрессивно — рейз",
            "{hand} — отличная рука для открытия",
            "Сильная стартовая рука {hand} — рейз для изоляции",
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
            "Фолд с {hand}, ждём лучшие карты",  # [FIX 5]
            "{hand} — фолд, не тратим деньги на мусор",
            "С {hand} пас, рука не играбельна из {pos}",
            "{hand} — фолд, слишком много доминирующих рук",
            "С {hand} сбрасываем — нет эквити для входа",
            "{hand} — фолд, ждём премиум-руку",
            "Не открываем {hand} — слабая позиция и рука",
            "{hand} — пас, не стоит ввязываться",
        ],
        "premium": [
            "{hand} — премиум, рейз и готовность к 4-бету",
            "С {hand} играем на стек, рейз",
            "{hand} — рука для 3-бета и 4-бета",
            "С {hand} максимизируем вэлью — рейз",
            "{hand} — всегда рейз, не медлим",
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
            "{hand} — бет для защиты эквити",
            "{hand} — стандартный контбет",
            "С {hand} продолжаем агрессию",
            "{hand} — бет, оппонент может иметь хуже",
            "Сильная рука {hand} — ставка для вэлью",
            "{hand} — топ-пара, бет 50–66% пота",
            "С {hand} контбет, защищаем эквити",
            "{hand} — сильная рука, но осторожно на дро-борде",
            "С {hand} бет для вэлью, оппонент часто коллирует хуже",
            "{hand} — ставка, не даём бесплатно улучшиться",
            "Топ-пара {hand} — бет, контролируем размер банка",
        ],
        "draw": [
            "Дро {hand} с {outs} аутами — колл, надеемся на улучшение",
            "Флеш/стрит-дро {hand} — колл с правильными оддсами",
            "Дро {hand} — чек, смотрим бесплатную карту",
            "Потенциал на улучшение {hand} — колл",
            "Дро с {outs} аутами — математически правильный колл",
            "С дро {hand} продолжаем, эквити позволяет",
            "Флеш-дро {hand} — полу-блеф рейз для фолд-эквити",
            "Стрит-дро {hand} — колл, оддсы позволяют",
            "Дро с оверкартами {hand} — колл, много аутов",
            "Комбо-дро {hand} — играем агрессивно, много эквити",
            "С дро {hand} ({outs} аутов) — колл, имплайд-оддсы хорошие",
            "Дро {hand} — чек-колл, не строим банк без попадания",
            "Флеш-дро {hand} — бет для полу-блефа, есть фолд-эквити",
        ],
        "weak": [
            "Слабая рука {hand} — чек, пытаемся улучшить",
            "{hand} — чек, нет силы для агрессии",
            "Мало попаданий с {hand} — чек",
            "Слабое попадание {hand} — пассивная игра",
            "{hand} — чек, контролируем банк",
            "Слабая рука {hand} — чек-фолд на агрессию",
            "{hand} — чек, не вкладываем деньги без силы",
            "Слабое попадание {hand} — чек, смотрим тёрн бесплатно",
            "{hand} — пассивно, нет эквити для ставки",
        ],
        "air": [
            "Воздух {hand} — чек, сдаёмся",
            "Нет попадания с {hand} — фолд против ставки",
            "Воздух {hand} против агрессии — фолд",
            "Ничего не собрали с {hand} — сдаёмся",
            "Воздух {hand} — чек-фолд, не тратим деньги",
            "Нет эквити с {hand} — фолд на ставку",
            "Воздух {hand} — можно полу-блеф, если оппонент слаб",
            "Ничего не собрали с {hand} — чек, готовы сбросить",
            "Воздух {hand} — сдаёмся, ждём лучшую ситуацию",
        ],
        "bluff": [
            "Воздух {hand} — полу-блеф бет, есть фолд-эквити",
            "Слабый дро {hand} — блеф-бет для давления",
            "Нет попадания с {hand} — блеф на сухом борде",
            "Воздух {hand} — ставка для кражи банка",
        ],
    },
    "turn": {
        "made": [
            "Готовая рука {hand} — вэлью-бет",
            "{hand} на тёрне — ставка для вэлью",
            "Сильная рука {hand} — бет, оппонент может коллировать",
            "{hand} — продолжаем агрессию",
            "С {hand} строим банк на тёрне",
            "{hand} — бет 66–75% пота для защиты",
            "С {hand} вэлью-бет, оппонент часто платит",
            "{hand} — сильная рука, но следим за дро",
            "С {hand} бет, не даём бесплатной карты на ривер",
            "{hand} — ставка для максимизации вэлью",
        ],
        "draw": [
            "Дро {hand} на тёрне — колл, шансы ~20%",
            "Флеш/стрит-дро {hand} — колл с правильными оддсами",
            "Дро {hand} — чек, одна карта до ривера",
            "С дро {hand} продолжаем, эквити позволяет",
            "Дро {hand} на тёрне — колл, имплайд-оддсы хорошие",
            "Флеш-дро {hand} — колл, шансы на ривере ~20%",
            "Дро {hand} — чек-колл, не строим банк без попадания",
            "Стрит-дро {hand} — колл, потенциально большой банк",
        ],
        "missed": [
            "Дро {hand} не закрылось — фолд",
            "Не попали с {hand} — сдаёмся",
            "Воздух {hand} на тёрне — фолд",
            "Дро {hand} не закрылось — фолд, не тратим деньги",
            "Не попали с {hand} — чек-фолд на ставку",
            "Воздух {hand} — сдаёмся, ривер не поможет",
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
            "Средняя рука {hand} — колл если пот-оддсы позволяют",
            "{hand} — колл, банк большой",
            "Средняя сила {hand} — колл",
            "{hand} — колл, но осторожно на большие ставки",
            "Средняя рука {hand} — колл, пот-оддсы хорошие",
            "{hand} — колл, не рейзим без натса",
        ],
        "weak": [
            "Слабая рука {hand} — фолд",
            "{hand} недостаточно — фолд",
            "Нет силы для колла с {hand} — фолд",
            "{hand} — фолд, не платим за проигрыш",
            "Слабая рука {hand} — фолд на ставку",
            "{hand} — пас, эквити недостаточно",
        ],
        "bluff_catch": [
            "Средняя рука {hand} — колл, ловим блеф",
            "{hand} — колл, оппонент может блефовать",
            "С {hand} коллируем, пот-оддсы позволяют",
            "Ловим блеф с {hand} — колл",
        ],
        "bluff": [
            "Воздух {hand} — блеф на ривере, оппонент слаб",
            "Нет попадания с {hand} — блеф-бет для фолд-эквити",
            "Слабый дро {hand} — блеф на ривере",
            "Воздух {hand} — ставка для кражи банка",
        ],
    },
}


# ============ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ============

def _rank_str(rank):
    """[FIX] Числовой rank (2..14) -> строка ('2'..'A')."""
    return RANK_NAMES.get(rank, str(rank))


def format_hand(c1, c2):
    """[FIX] 'A♣K♣' вместо '14♣13♣'."""
    return f"{_rank_str(c1.rank)}{c1.suit}{_rank_str(c2.rank)}{c2.suit}"


def format_board(board):
    """[FIX] 'A♣ K♦ 7♠' вместо '14♣ 13♦ 7♠'."""
    return " ".join(f"{_rank_str(c.rank)}{c.suit}" for c in board)


def get_analysis_text(stage, category, prompt_key="", **kwargs):
    """
    [FIX 1] Детерминированный выбор шаблона по хэшу prompt_key.
    Одинаковый prompt -> одинаковый completion.
    """
    templates = ANALYSIS_TEMPLATES.get(stage, {}).get(category, ["Решение"])
    if prompt_key:
        digest = hashlib.md5(prompt_key.encode("utf-8")).hexdigest()
        idx = int(digest, 16) % len(templates)
    else:
        idx = 0
    template = templates[idx]
    # Безопасная подстановка: отсутствующие ключи остаются как есть
    try:
        return template.format(**kwargs)
    except KeyError:
        return template


def _safe_odds_pct(to_call, pot):
    """Пот-оддсы в процентах."""
    if to_call <= 0 or (pot + to_call) <= 0:
        return 0.0
    return to_call / (pot + to_call) * 100.0


# ============ ГЕНЕРАТОРЫ ПО СТАДИЯМ ============

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

    hand_str = format_hand(hero_cards[0], hero_cards[1])

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

    # Категория и действие
    if tier <= 2:
        action, amount, category = "RAISE", 30, "strong"
    elif tier <= 4:
        action, amount, category = "RAISE", 30, "medium"
    else:
        action, amount, category = "FOLD", 0, "weak"

    # [FIX 2] конкретная рука + [FIX 1] детерминированный шаблон
    analysis_text = get_analysis_text(
        "preflop", category,
        prompt_key=prompt,
        hand=hand_str, pos=scenario['hero_position']
    )

    # [FIX 3] reasoning с числами
    if category == "strong":
        reasoning = (f"{hand_str} — {hand_name}, тир {tier}/7. "
                     f"Из {scenario['hero_position']} это уверенный рейз для вэлью. "
                     f"Открываем 3 BB, готовы к 3-бету.")
    elif category == "medium":
        reasoning = (f"{hand_str} — {hand_name}, тир {tier}/7. "
                     f"Играбельная рука из {scenario['hero_position']}. "
                     f"Рейз для инициативы и кражи блайндов, но осторожно на 3-бет.")
    else:
        reasoning = (f"{hand_str} — {hand_name}, тир {tier}/7. "
                     f"Слишком слабая рука для входа из {scenario['hero_position']}. "
                     f"Фолд — ждём лучшую ситуацию.")

    completion = f"""АНАЛИЗ: {reasoning}
МОИ КОМБИНАЦИИ: {hand_str} — {hand_name} (тир {tier}/7, {"одномастная" if hero_cards[0].suit == hero_cards[1].suit else "разномастная"})
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
            "hero_hand": hand_str,
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

    hand_str = format_hand(hero_cards[0], hero_cards[1])
    board_str = format_board(board)

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

    # Выбор действия
    if hero_hand[0] >= 3:
        action = "BET" if villain_action[0] == "CHECK" else "RAISE"
        amount = int(pot * 0.66) if action == "BET" else int(villain_action[1] * 2.5)
        category = "monster"
    elif hero_hand[0] >= 1:
        if villain_action[0] == "CHECK":
            action, amount, category = "BET", int(pot * 0.5), "strong"
        else:
            action, amount, category = "CALL", to_call, "strong"
    elif outs:
        if villain_action[0] == "CHECK":
            action, amount, category = "CHECK", 0, "draw"
        else:
            action, amount, category = "CALL", to_call, "draw"
    else:
        if villain_action[0] == "CHECK":
            action, amount, category = "CHECK", 0, "air"
        else:
            action, amount, category = "FOLD", 0, "air"

    analysis_text = get_analysis_text(
        "flop", category,
        prompt_key=prompt,
        hand=hero_hand_name, outs=len(outs)
    )

    # [FIX 3] reasoning с числами и оддсами
    strength = hero_hand[0] * 10 + 5
    if category == "monster":
        reasoning = (f"{hero_hand_name} ({hand_str}) на {board_str}. "
                     f"Сила {strength}/100 — монстр. "
                     f"Бетим крупно для вэлью, опп платит с топ-парой и дро.")
    elif category == "strong":
        reasoning = (f"{hero_hand_name} ({hand_str}) на {board_str}. "
                     f"Сила {strength}/100 — топ-пара. "
                     f"Бет 50-66% пота для защиты от дро.")
    elif category == "draw":
        outs_count = len(outs)
        eq = probs.get("turn", 0) + probs.get("river", 0)
        need = _safe_odds_pct(to_call, pot)
        decision = "колл по оддсам" if eq >= need else "фолд по оддсам"
        reasoning = (f"Дро ({hand_str}) на {board_str}. "
                     f"Аутов: {outs_count}. Эквити ~{eq:.1f}%, нужно {need:.1f}% — {decision}.")
    else:
        reasoning = (f"{hero_hand_name} ({hand_str}) на {board_str}. "
                     f"Сила {strength}/100, аутов нет. "
                     f"Чек/фолд — не вкладываем деньги без эквити.")

    completion = f"""АНАЛИЗ: {reasoning}
МОИ КОМБИНАЦИИ: {hero_hand_name} ({hand_str}) на {board_str} → {format_probs(probs)}
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
            "hero_hand": hand_str,
            "board": board_str,
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

    hand_str = format_hand(hero_cards[0], hero_cards[1])
    board_str = format_board(board)

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

    analysis_text = get_analysis_text(
        "turn", category, prompt_key=prompt, hand=hero_hand_name
    )

    # [FIX 3] reasoning с числами
    strength = hero_hand[0] * 10 + 5
    if category == "made":
        reasoning = (f"{hero_hand_name} ({hand_str}) на {board_str}. "
                     f"Сила {strength}/100 — готовая рука. "
                     f"Вэлью-бет 66% пота, опп часто платит.")
    elif category == "draw":
        eq = probs.get("river", 0)
        need = _safe_odds_pct(to_call, pot)
        reasoning = (f"Дро ({hand_str}) на {board_str}. "
                     f"Аутов: {len(outs)}. Эквити ~{eq:.1f}%, нужно {need:.1f}%. "
                     f"{'Колл по оддсам' if eq >= need else 'Чек/фолд по оддсам'}.")
    else:
        reasoning = (f"{hero_hand_name} ({hand_str}) на {board_str}. "
                     f"Сила {strength}/100, дро не закрылось. "
                     f"Фолд на ставку, не тратим деньги.")

    completion = f"""АНАЛИЗ: {reasoning}
МОИ КОМБИНАЦИИ: {hero_hand_name} ({hand_str}) на {board_str} → {format_probs(probs)}
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
            "hero_hand": hand_str,
            "board": board_str,
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
    hero_strength = hero_hand[0]

    hand_str = format_hand(hero_cards[0], hero_cards[1])
    board_str = format_board(board)

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

    if hero_strength >= 4:
        action = "CALL" if villain_action[0] == "BET" else "BET"
        amount = to_call if action == "CALL" else int(pot * 0.66)
        category = "strong"
    elif hero_strength >= 2:
        action = "CALL" if villain_action[0] == "BET" else "CHECK"
        amount = to_call if action == "CALL" else 0
        category = "medium"
    else:
        action = "FOLD" if villain_action[0] == "BET" else "CHECK"
        amount = 0
        category = "weak"

    analysis_text = get_analysis_text(
        "river", category, prompt_key=prompt, hand=hero_hand_name
    )

    # [FIX 3] reasoning с числами и пот-оддсами
    strength = hero_hand[0] * 10 + 5
    need = _safe_odds_pct(to_call, pot)
    if category == "strong":
        reasoning = (f"{hero_hand_name} ({hand_str}) на {board_str}. "
                     f"Сила {strength}/100 — сильная рука. "
                     f"Колл/вэлью-рейз, эквити позволяет.")
    elif category == "medium":
        reasoning = (f"{hero_hand_name} ({hand_str}) на {board_str}. "
                     f"Сила {strength}/100 — средняя рука. "
                     f"Пот-оддсы {need:.1f}% — колл, если банк большой.")
    else:
        reasoning = (f"{hero_hand_name} ({hand_str}) на {board_str}. "
                     f"Сила {strength}/100 — слабая рука. "
                     f"Фолд на ставку, эквити недостаточно.")

    completion = f"""АНАЛИЗ: {reasoning}
МОИ КОМБИНАЦИИ: {hero_hand_name} ({hand_str}) на {board_str} (финальная)
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
            "hero_hand": hand_str,
            "board": board_str,
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
    unique_prompts = len(set(ex["prompt"] for ex in examples))

    # Конфликтные prompt: одинаковый prompt, разные completion
    prompt_to_completions = {}
    for ex in examples:
        prompt_to_completions.setdefault(ex["prompt"], set()).add(ex["completion"])
    conflicts = sum(1 for v in prompt_to_completions.values() if len(v) > 1)

    print("\n📊 Статистика:")
    for stage, count in stages.items():
        print(f"  {stage}: {count}")
    print(f"  Уникальных prompt:     {unique_prompts} ({unique_prompts/len(examples)*100:.1f}%)")
    print(f"  Уникальных completion: {unique_completions} ({unique_completions/len(examples)*100:.1f}%)")
    print(f"  Конфликтных prompt (1 prompt -> N completions): {conflicts}")


if __name__ == "__main__":
    print("🎯 Генерация единого датасета...\n")

    examples = generate_unified_dataset(count_per_stage=2000)

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