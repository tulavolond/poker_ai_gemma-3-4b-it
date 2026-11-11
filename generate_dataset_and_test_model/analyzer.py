"""
analyzer.py - Модуль анализа покерных ситуаций.
Считает ВСЮ математику и формирует структурированный промпт для ИИ.

Модель НЕ считает. Модель получает готовые цифры и принимает решения.
"""
import random
from collections import Counter

# ============ КАРТЫ ============
RANKS = list(range(2, 15))  # 2-14, где J=11, Q=12, K=13, A=14 "J♠", "T♥"
SUITS = ['♠', '♥', '♦', '♣']

RANK_NAMES = {10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'}


class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    def __repr__(self):
        return f"{RANK_NAMES.get(self.rank, self.rank)}{self.suit}"

    def __hash__(self):
        return hash((self.rank, self.suit))

    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit


def build_deck():
    return [Card(r, s) for r in RANKS for s in SUITS]


# ============ ОЦЕНКА КОМБИНАЦИЙ ============
def evaluate_hand(cards):
    """Оценивает 5-карточную руку. Возвращает (категория, ранги)."""
    ranks = sorted([c.rank for c in cards], reverse=True)
    suits = [c.suit for c in cards]
    counts = Counter(ranks)

    is_flush = len(set(suits)) == 1
    unique = sorted(set(ranks), reverse=True)
    is_straight = False
    straight_high = 0

    if len(unique) == 5:
        if unique[0] - unique[4] == 4:
            is_straight = True
            straight_high = unique[0]
        elif unique == [14, 5, 4, 3, 2]:
            is_straight = True
            straight_high = 5

    cnt_sorted = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)

    if is_flush and is_straight:
        cat = 9 if straight_high == 14 else 8
    elif cnt_sorted[0][1] == 4:
        cat = 7
    elif cnt_sorted[0][1] == 3 and cnt_sorted[1][1] == 2:
        cat = 6
    elif is_flush:
        cat = 5
    elif is_straight:
        cat = 4
    elif cnt_sorted[0][1] == 3:
        cat = 3
    elif cnt_sorted[0][1] == 2 and cnt_sorted[1][1] == 2:
        cat = 2
    elif cnt_sorted[0][1] == 2:
        cat = 1
    else:
        cat = 0

    return (cat, ranks)


HAND_NAMES = {
    0: "Старшая карта", 1: "Пара", 2: "Две пары", 3: "Сет",
    4: "Стрит", 5: "Флеш", 6: "Фулл-хаус", 7: "Каре",
    8: "Стрит-флеш", 9: "Роял-флеш"
}

HAND_STRENGTH_BASE = {
    0: 5, 1: 25, 2: 45, 3: 65, 4: 75, 5: 82, 6: 90, 7: 96, 8: 98, 9: 100
}


# ============ MONTE CARLO ЭКВИТИ ============
def calculate_equity(hero_cards, board, num_opponents=1, simulations=100):
    """Считает шанс победы против N оппонентов методом Монте-Карло."""
    deck = build_deck()
    used = set(hero_cards) | set(board)
    deck = [c for c in deck if c not in used]

    wins = 0
    ties = 0

    for _ in range(simulations):
        random.shuffle(deck)
        cards_needed = 5 - len(board)
        final_board = board + deck[:cards_needed]
        idx = cards_needed

        hero_best = evaluate_hand(hero_cards + final_board)

        best_opp = None
        for o in range(num_opponents):
            opp_cards = deck[idx + o * 2: idx + o * 2 + 2]
            opp_hand = evaluate_hand(opp_cards + final_board)
            if best_opp is None or opp_hand > best_opp:
                best_opp = opp_hand

        if hero_best > best_opp:
            wins += 1
        elif hero_best == best_opp:
            ties += 1

    return (wins + ties * 0.5) / simulations


# ============ АНАЛИЗ БОРДА (ДРО, ТЕКСТУРА) ============
def analyze_board_features(board, hero_cards):
    """Находит дро, попадания, текстуру борда."""
    all_cards = hero_cards + board
    suits = [c.suit for c in all_cards]
    hero_ranks = [c.rank for c in hero_cards]
    board_ranks = [c.rank for c in board]

    features = {
        "pair": False,
        "top_pair": False,
        "set": False,
        "two_pair": False,
        "flush_draw": False,
        "oesd": False,       # Open-Ended Straight Draw
        "gutshot": False,    # Inside Straight Draw
        "overcards": 0,
        "flush_possible": False,
        "paired_board": False,
    }

    board_counts = Counter(board_ranks)

    # Попадания
    for r in hero_ranks:
        if board_counts[r] >= 2:
            features["set"] = True
        elif board_counts[r] == 1:
            features["pair"] = True
            if board_ranks and r == max(board_ranks):
                features["top_pair"] = True

    if sum(1 for r in hero_ranks if board_counts[r] >= 1) == 2:
        features["two_pair"] = True

    # Флеш-дро (4 карты одной масти среди всех)
    suit_counts = Counter(suits)
    max_suit = max(suit_counts.values()) if suit_counts else 0
    if max_suit == 4:
        features["flush_draw"] = True
    if max_suit >= 4 and len(board) >= 3:
        features["flush_possible"] = True

    # Стрит-дро
    unique = sorted(set([c.rank for c in all_cards]))
    for i in range(len(unique) - 3):
        if unique[i + 3] - unique[i] == 3:
            features["oesd"] = True
            break
    for i in range(len(unique) - 3):
        if unique[i + 3] - unique[i] == 4:
            features["gutshot"] = True
            break

    # Оверкарты (карты героя выше максимальной карты борда)
    if board_ranks:
        max_board = max(board_ranks)
        features["overcards"] = sum(1 for r in hero_ranks if r > max_board)

    # Парный борд
    if board_ranks and max(board_counts.values()) >= 2:
        features["paired_board"] = True

    return features


def board_texture(board, features):
    """Определяет текстуру борда: dry / wet."""
    if len(board) < 3:
        return "N/A (префлоп)"

    wet_score = 0
    if features["flush_draw"]:
        wet_score += 0.4
    if features["oesd"]:
        wet_score += 0.3
    if features["gutshot"]:
        wet_score += 0.15
    if features["paired_board"]:
        wet_score += 0.1
    high_cards = sum(1 for c in board if c.rank >= 10)
    wet_score += high_cards * 0.05

    if wet_score >= 0.5:
        return "WET (опасный, много дро)"
    elif wet_score >= 0.25:
        return "Средний"
    else:
        return "DRY (безопасный)"


# ============ ОЦЕНКА ПРЕФЛОП-РУКИ ============
def classify_preflop_hand(h1, h2):
    """Классифицирует стартовую руку. Возвращает (тир, имя, описание)."""
    r1, r2 = max(h1.rank, h2.rank), min(h1.rank, h2.rank)
    suited = h1.suit == h2.suit
    gap = r1 - r2
    pair = (r1 == r2)

    # Тир 1: Премиум
    if pair and r1 >= 13:
        return (1, "Премиум-пара", "AA/KK — лучшие руки, всегда 3-бет/4-бет")
    if r1 == 14 and r2 == 13 and suited:
        return (1, "AKs", "Премиум, играем агрессивно")

    # Тир 2: Сильные
    if pair and r1 >= 11:
        return (2, "Высокая пара", "QQ/JJ — сильная рука для рейза")
    if r1 == 14 and r2 >= 12:
        return (2, "Сильный бродвей", "AK/AQ — сильная рука")
    if r1 == 13 and r2 == 12 and suited:
        return (2, "KQs", "Сильная одномастная рука")

    # Тир 3: Играбельные
    if pair and r1 >= 9:
        return (3, "Средняя пара", "TT/99 — играем на сет-майнинг или вэлью")
    if suited and r1 >= 12 and r2 >= 10:
        return (3, "Одномастный бродвей", "Хорошая рука для рейза")
    if r1 == 14 and suited and r2 >= 8:
        return (3, "Axs сильный", "Одномастный туз с хорошим кикером")

    # Тир 4: Позиционные
    if pair and r1 >= 7:
        return (4, "Маленькая пара", "88/77 — сет-майнинг, играем дёшево")
    if suited and gap <= 1 and r1 >= 9:
        return (4, "Одномастный коннектор", "Хороший потенциал стрита/флеша")
    if r1 >= 11 and r2 >= 9:
        return (4, "Бродвей", "Играбельная рука из позиции")

    # Тир 5: Спекулятивные
    if pair:
        return (5, "Низкая пара", "Сет-майнинг, только дешёвый вход")
    if suited and gap <= 2:
        return (5, "Слабый одномастный", "Потенциал, но рискованно")
    if r1 == 14:
        return (5, "Ax", "Тузы с слабым кикером")

    # Тир 6: Слабые
    if suited:
        return (6, "Слабые одномастные", "Только из поздней позиции дёшево")
    if gap <= 3 and r1 >= 8:
        return (6, "Слабая связка", "Слабый потенциал")

    # Тир 7: Мусор
    return (7, "Мусор", "Сбрасывать без раздумий")


# ============ ПОТ-ОДДСЫ ============
def calculate_pot_odds(pot, to_call):
    """Считает пот-оддсы в процентах."""
    if to_call <= 0:
        return 0
    return to_call / (pot + to_call)


# ============ ГИБРИДНЫЙ ПРОМПТ ============
# Единый шаблон + секции под стадию

PROMPT_HEADER = """=== СИТУАЦИЯ: {stage} ===

ТВОИ КАРТЫ: {hero_cards}
{board_line}

— АНАЛИЗ ОТ СИСТЕМЫ (доверяй этим цифрам, они точные) —
{features}

— ДЕЙСТВИЯ ОППОНЕНТОВ —
{opponent_actions}

— КОНТЕКСТ —
Позиция: {position}
Стек: ${stack}, Банк: ${pot}
Стек/банк: {stack_to_pot:.1f}

{stage_instructions}

Ответь СТРОГО в формате:
АНАЛИЗ: [одно предложение с твоей логикой]
ДЕЙСТВИЕ: [FOLD / CHECK / CALL / RAISE]
СУММА: [число]"""


STAGE_INSTRUCTIONS = {
    "preflop": """— ЗАДАЧА: ПРЕФЛОП —
Оцени стартовую руку по тиру и позиции.
Правила:
- Тир 1-2: всегда рейз или 3-бет
- Тир 3-4: рейз из средней/поздней позиции, осторожнее из ранней
- Тир 5-6: только дешёвый вход из поздней позиции
- Тир 7: фолд
Учитывай действия оппонентов до тебя.""",

    "flop": """— ЗАДАЧА: ФЛОП (3 карты на столе) —
Сравни эквити с пот-оддсами.
Правила:
- Сильное попадание (сет/2 пары): вэлью-бет, не давай бесплатных карт
- Топ-пара с эквити > 50%: продолженная ставка или колл
- Флеш-дро или OESD: можно продолжать, шансы ~35%/32%
- Эквити < 30%: фолд
Текстура борда подсказывает опасность.""",

    "turn": """— ЗАДАЧА: ТЁРН (4 карты на столе) —
Одна карта до вскрытия, дро стало дороже.
Правила:
- Готовая сильная комбинация: вэлью-бет
- Дро: ~20% шанс с одной карты, считай пот-оддсы
- Если эквити < пот-оддсов: фолд
- Сильная рука против агрессии: можно рейзить""",

    "river": """— ЗАДАЧА: РИВЕР (5 карт на столе) —
Дро больше нет, только финальная сила руки.
Правила:
- Эквити > 65%: колл или вэлью-рейз
- Эквити 30-65%: колл если пот-оддсы позволяют
- Эквити < 30%: фолд, изредка блеф если история позволяет
- Никогда не коллируй со старшей картой против ставки"""
}


def format_features(features_dict):
    """Форматирует словарь фичей в читаемый список."""
    lines = []
    for key, value in features_dict.items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


# ============ ГЛАВНЫЕ ФУНКЦИИ АНАЛИЗА ============

def analyze_preflop(hero_cards, position, action_before, stack, bb=10, num_opponents=1):
    """Анализ префлоп-ситуации. Возвращает промпт + фичи."""
    h1, h2 = hero_cards
    tier, hand_name, hand_desc = classify_preflop_hand(h1, h2)

    # Примерная сила руки на префлопе
    strength = max(0, 100 - (tier - 1) * 15)

    features_dict = {
        "Стартовая рука": hand_name,
        "Тир руки (1-7)": f"{tier} из 7",
        "Описание": hand_desc,
        "Сила руки (0-100)": f"{strength}",
        "Одномастная": "Да" if h1.suit == h2.suit else "Нет",
        "Пара": "Да" if h1.rank == h2.rank else "Нет",
        "Разрыв (gap)": abs(h1.rank - h2.rank),
    }

    board_line = "БОРД: ещё нет карт на столе"

    prompt = PROMPT_HEADER.format(
        stage="ПРЕФЛОП",
        hero_cards=f"{h1}, {h2}",
        board_line=board_line,
        features=format_features(features_dict),
        opponent_actions=action_before,
        position=position,
        stack=stack,
        pot=bb * 1.5,
        stack_to_pot=stack / (bb * 1.5),
        stage_instructions=STAGE_INSTRUCTIONS["preflop"]
    )

    return {
        "prompt": prompt,
        "features": features_dict,
        "tier": tier,
        "stage": "preflop"
    }


def analyze_postflop(hero_cards, board, position, opponent_action,
                     stack, pot, to_call, num_opponents=1, simulations=100):
    """Анализ постфлоп-ситуации (флоп/тёрн/ривер). Возвращает промпт + фичи."""
    stage_map = {3: "flop", 4: "turn", 5: "river"}
    stage = stage_map.get(len(board), "flop")
    stage_names = {"flop": "ФЛОП", "turn": "ТЁРН", "river": "РИВЕР"}

    # Считаем всё
    best_hand = evaluate_hand(hero_cards + board)
    hand_name = HAND_NAMES[best_hand[0]]
    hand_strength = HAND_STRENGTH_BASE[best_hand[0]]

    equity = calculate_equity(hero_cards, board, num_opponents, simulations)
    board_feats = analyze_board_features(board, hero_cards)
    texture = board_texture(board, board_feats)
    pot_odds = calculate_pot_odds(pot, to_call)

    features_dict = {
        "Твоя комбинация": hand_name,
        "Сила руки (0-100)": f"{hand_strength}",
        "Эквити (шанс победы)": f"{equity * 100:.0f}%",
        "Пот-оддсы": f"{pot_odds * 100:.0f}%",
        "Эквити > Пот-оддсы": "ДА ✓" if equity > pot_odds else "НЕТ ✗",
        "Текстура борда": texture,
        "Пара": "Да" if board_feats["pair"] else "Нет",
        "Топ-пара": "Да" if board_feats["top_pair"] else "Нет",
        "Сет": "Да" if board_feats["set"] else "Нет",
        "Две пары": "Да" if board_feats["two_pair"] else "Нет",
        "Флеш-дро": "Да" if board_feats["flush_draw"] else "Нет",
        "Стрит-дро (OESD)": "Да" if board_feats["oesd"] else "Нет",
        "Гатшот": "Да" if board_feats["gutshot"] else "Нет",
        "Оверкарты": board_feats["overcards"],
    }

    board_line = f"БОРД ({len(board)} карт): {', '.join(str(c) for c in board)}"

    prompt = PROMPT_HEADER.format(
        stage=stage_names[stage],
        hero_cards=f"{hero_cards[0]}, {hero_cards[1]}",
        board_line=board_line,
        features=format_features(features_dict),
        opponent_actions=opponent_action,
        position=position,
        stack=stack,
        pot=pot,
        stack_to_pot=stack / max(pot, 1),
        stage_instructions=STAGE_INSTRUCTIONS[stage]
    )

    return {
        "prompt": prompt,
        "features": features_dict,
        "equity": equity,
        "pot_odds": pot_odds,
        "hand_strength": hand_strength,
        "stage": stage
    }


# ============ ТЕСТ ============
if __name__ == "__main__":
    print("Тест модуля анализа...\n")

    # Тест префлопа
    deck = build_deck()
    random.shuffle(deck)
    hero = deck[0:2]

    result = analyze_preflop(
        hero_cards=hero,
        position="BTN (баттон)",
        action_before="Все сбросили до тебя",
        stack=1000,
        bb=10
    )
    print("=" * 50)
    print("ПРИМЕР ПРОМПТА (ПРЕФЛОП):")
    print("=" * 50)
    print(result["prompt"])
    print(f"\nТир руки: {result['tier']}")

    # Тест флопа
    print("\n" + "=" * 50)
    print("ПРИМЕР ПРОМПТА (ФЛОП):")
    print("=" * 50)

    deck = build_deck()
    random.shuffle(deck)
    hero = deck[0:2]
    flop = deck[2:5]

    result = analyze_postflop(
        hero_cards=hero,
        board=flop,
        position="BTN (баттон)",
        opponent_action="Оппонент поставил $50",
        stack=950,
        pot=120,
        to_call=50,
        num_opponents=1,
        simulations=100
    )
    print(result["prompt"])
    print(f"\nЭквити: {result['equity']:.2%}, Пот-оддсы: {result['pot_odds']:.2%}")