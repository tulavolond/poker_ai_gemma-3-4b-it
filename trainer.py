"""
Monte Carlo симулятор для обучения покерного ИИ.
Генерирует идеальные решения для каждой фазы игры.
"""
import random
from itertools import combinations
from collections import Counter
import math

RANKS = list(range(2, 15))  # 2-14 (J=11, Q=12, K=13, A=14)
SUITS = ['♠', '♥', '♦', '♣']
HAND_NAMES = {
    0: "старшая карта",
    1: "пара",
    2: "две пары",
    3: "сет",
    4: "стрит",
    5: "флеш",
    6: "фулл-хаус",
    7: "каре",
    8: "стрит-флеш",
    9: "роял-флеш"
}


class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
    def __repr__(self):
        names = {11:'J', 12:'Q', 13:'K', 14:'A'}
        return f"{names.get(self.rank, self.rank)}{self.suit}"
    def __hash__(self): return hash((self.rank, self.suit))
    def __eq__(self, other): return self.rank == other.rank and self.suit == other.suit

def build_deck():
    return [Card(r, s) for r in RANKS for s in SUITS]

# ============ ОЦЕНКА 5-КАРТОЧНОЙ РУКИ ============
def evaluate_hand(cards):
    """Возвращает (категория, кикеры). Больше = сильнее."""
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
            straight_high = 5  # колёсо
    
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

# ============ MONTE CARLO EQUITY ============
def calculate_equity(hero_cards, board, num_opponents=1, simulations=500):
    """
    Возвращает процент побед (0.0 - 1.0) против N случайных оппонентов.
    """
    deck = build_deck()
    used = set(hero_cards) | set(board)
    deck = [c for c in deck if c not in used]
    
    wins = 0
    ties = 0
    
    for _ in range(simulations):
        random.shuffle(deck)
        # добиваем борд до 5 карт
        remaining_board = board + deck[:5 - len(board)]
        idx = 5 - len(board)
        
        hero_best = evaluate_hand(hero_cards + remaining_board)
        
        best_opp = None
        for o in range(num_opponents):
            opp_cards = deck[idx + o*2 : idx + o*2 + 2]
            opp_hand = evaluate_hand(opp_cards + remaining_board)
            if best_opp is None or opp_hand > best_opp:
                best_opp = opp_hand
        
        if hero_best > best_opp:
            wins += 1
        elif hero_best == best_opp:
            ties += 1
    
    return (wins + ties * 0.5) / simulations

# ============ ОЦЕНКА ПРЕФЛОП РУКИ ============
def preflop_hand_category(h1, h2):
    """
    Префлоп-классификатор.
    Возвращает: (tier 1-8, имя типа)
    Tier 1: AA, KK, QQ, AKs
    Tier 2: JJ, TT, AKo, AQs
    ...
    Tier 8: разномастный мусор
    """
    r1, r2 = max(h1.rank, h2.rank), min(h1.rank, h2.rank)
    suited = h1.suit == h2.suit
    gap = r1 - r2
    pair = (r1 == r2)
    
    # Tier 1: премиум
    if pair and r1 >= 12: return (1, "премиум-пара")
    if r1 == 14 and r2 == 13 and suited: return (1, "AKs")
    
    # Tier 2: сильные
    if pair and r1 >= 10: return (2, "высокая пара")
    if r1 == 14 and r2 >= 12: return (2, "AX высокий")
    
    # Tier 3: играбельные
    if pair: return (3, "средняя пара")
    if suited and r1 >= 12: return (3, "suited broadway")
    if r1 == 14 and suited: return (3, "Axs")
    
    # Tier 4: позиционные
    if suited and gap <= 2 and r1 >= 8: return (4, "suited connector")
    if r1 >= 11 and r2 >= 10: return (4, "бродвей")
    
    # Tier 5: маргинальные
    if suited: return (5, "одномастные")
    if r1 == 14: return (5, "Ax")
    if gap <= 2 and r1 >= 8: return (5, "связка")
    
    # Tier 6-8: мусор
    if r1 >= 9: return (6, "средние карты")
    if gap <= 3: return (7, "слабая связка")
    return (8, "мусор")

# ============ АНАЛИЗ ФЛОПА ============
def analyze_flop(board, hero_cards):
    """Анализирует флоп: попадания, дро, текстура."""
    all_cards = hero_cards + board
    ranks = [c.rank for c in all_cards]
    suits = [c.suit for c in all_cards]
    
    hero_ranks = [c.rank for c in hero_cards]
    board_ranks = [c.rank for c in board]
    
    result = {
        "pair": False,           # попадание в пару
        "set": False,            # сет (карманная пара + карта на столе)
        "two_pair": False,
        "flush_draw": False,     # 4 карты одной масти
        "oesd": False,           # open-ended straight draw
        "gutshot": False,        # дырявый стрит
        "overcards": 0,          # карты выше борда
        "wet": 0.0,              # текстура (0=dry, 1=wet)
    }
    
    # Попадания
    rank_counts = Counter(ranks)
    board_counts = Counter(board_ranks)
    
    for r in hero_ranks:
        if board_counts[r] >= 2:
            result["set"] = True
        elif board_counts[r] == 1:
            result["pair"] = True
    
    if sum(1 for r in hero_ranks if board_counts[r] >= 1) == 2:
        result["two_pair"] = True
    
    # Дро
    suit_counts = Counter(suits)
    if max(suit_counts.values()) == 4:
        result["flush_draw"] = True
    
    unique = sorted(set(ranks))
    # OESD: 4 подряд идущих ранга
    for i in range(len(unique) - 3):
        if unique[i+3] - unique[i] == 3:
            result["oesd"] = True
            break
    
    # Gutshot
    for i in range(len(unique) - 3):
        if unique[i+3] - unique[i] == 4:
            result["gutshot"] = True
            break
    
    # Оверкарты
    max_board = max(board_ranks)
    result["overcards"] = sum(1 for r in hero_ranks if r > max_board)
    
    # Текстура
    wet = 0
    if result["flush_draw"]: wet += 0.4
    if result["oesd"]: wet += 0.3
    if result["gutshot"]: wet += 0.15
    if len(set(board_ranks)) < 3: wet += 0.1  # парный борд = опасно
    result["wet"] = min(wet, 1.0)
    
    return result

# def generate_training_examples(count=1000):
#     """
#     Генерирует примеры для обучения LoRA.
#     Для каждой ситуации создаёт идеальный ответ.
#     """
#     examples = []
    
#     for _ in range(count):
#         deck = build_deck()
#         random.shuffle(deck)
        
#         # Раздаём карты
#         hero = deck[0:2]
#         villain = deck[2:4]
#         flop = deck[4:7]
#         turn = deck[7:8]
#         river = deck[8:9]
        
#         # ============ ФАЗА 1: ПРЕФЛОП ============
#         tier, name = preflop_hand_category(hero[0], hero[1])
#         preflop_prompt = f"""СТУДИЯ: ПРЕФЛОП
# Твои карты: {hero[0]}, {hero[1]}
# Позиция: BTN (баттон)
# Стек: $1000, Блайнд: $10
# Действие до тебя: рейз $30

# Категория руки: {name} (тир {tier}/8)

# Прими решение: FOLD / CALL / RAISE"""
        
#         # Идеальное решение
#         if tier <= 2:
#             preflop_action = "RAISE 90"
#             preflop_reason = "Премиум-рука, нужно наращивать банк"
#         elif tier <= 4:
#             preflop_action = "CALL"
#             preflop_reason = "Хорошая рука, играем в позиции"
#         elif tier <= 5:
#             preflop_action = "CALL"
#             preflop_reason = "Маргинальная рука, играем дёшево"
#         else:
#             preflop_action = "FOLD 0"
#             preflop_reason = "Слабая рука против рейза — сбрасываем"
        
#         preflop_completion = f"""АНАЛИЗ: {preflop_reason}
# ДЕЙСТВИЕ: {preflop_action.split()[0]}
# СУММА: {preflop_action.split()[1] if len(preflop_action.split()) > 1 else 0}"""
        
#         examples.append({
#             "stage": "preflop",
#             "prompt": preflop_prompt,
#             "completion": preflop_completion,
#             "reward": 1.0
#         })
        
#         # ============ ФАЗА 2: ФЛОП ============
#         flop_analysis = analyze_flop(flop, hero)
#         flop_equity = calculate_equity(hero, flop, num_opponents=1, simulations=200)
        
#         flop_prompt = f"""СТУДИЯ: ФЛОП (3 карты на столе)
# Твои карты: {hero[0]}, {hero[1]}
# Флоп: {flop[0]}, {flop[1]}, {flop[2]}
# Банк: $70, Стек: $970
# Действие: чек от оппонента

# Анализ: {flop_analysis}
# Эквити против 1 оппонента: {flop_equity:.2%}"""
        
#         # Решение на флопе
#         if flop_analysis["set"] or flop_equity > 0.7:
#             action = "RAISE 50"
#             reason = f"Сильное попадание (эквити {flop_equity:.0%}), вэлью-бет"
#         elif flop_analysis["flush_draw"] or flop_analysis["oesd"]:
#             action = "CALL"
#             reason = f"Дро с хорошими шансами, колл с планой усилиться"
#         elif flop_analysis["pair"] and flop_equity > 0.45:
#             action = "CALL"
#             reason = f"Пара с положительным эквити"
#         elif flop_equity < 0.3:
#             action = "FOLD 0"
#             reason = f"Слабое эквити ({flop_equity:.0%}), сбрасываем"
#         else:
#             action = "CHECK 0"
#             reason = "Средняя рука, играем осторожно"
        
#         flop_completion = f"""АНАЛИЗ: {reason}
# ДЕЙСТВИЕ: {action.split()[0]}
# СУММА: {action.split()[1] if len(action.split()) > 1 else 0}"""
        
#         examples.append({
#             "stage": "flop",
#             "prompt": flop_prompt,
#             "completion": flop_completion,
#             "reward": 1.0
#         })
        
#         # ============ ФАЗА 3: ТЁРН ============
#         turn_board = flop + turn
#         turn_analysis = analyze_flop(turn_board, hero)
#         turn_equity = calculate_equity(hero, turn_board, num_opponents=1, simulations=200)
        
#         turn_prompt = f"""СТУДИЯ: ТЁРН (4 карты на столе)
# Твои карты: {hero[0]}, {hero[1]}
# Борд: {', '.join(str(c) for c in turn_board)}
# Банк: $120, Стек: $920
# Оппонент поставил $60

# Анализ: {turn_analysis}
# Эквити: {turn_equity:.2%}
# Pot odds: {60/(120+60):.2%}"""
        
#         # Решение на тёрне
#         pot_odds = 60 / (120 + 60)
#         if turn_equity > 0.6:
#             action = "RAISE 150"
#             reason = f"Сильная рука, наращиваем банк"
#         elif turn_equity > pot_odds:
#             action = "CALL"
#             reason = f"Эквити ({turn_equity:.0%}) > pot odds ({pot_odds:.0%})"
#         elif turn_analysis["flush_draw"] or turn_analysis["oesd"]:
#             action = "CALL"
#             reason = f"Дро, implied odds высокие"
#         else:
#             action = "FOLD 0"
#             reason = f"Эквити ({turn_equity:.0%}) < pot odds ({pot_odds:.0%})"
        
#         turn_completion = f"""АНАЛИЗ: {reason}
# ДЕЙСТВИЕ: {action.split()[0]}
# СУММА: {action.split()[1] if len(action.split()) > 1 else 0}"""
        
#         examples.append({
#             "stage": "turn",
#             "prompt": turn_prompt,
#             "completion": turn_completion,
#             "reward": 1.0
#         })
        
#         # ============ ФАЗА 4: РИВЕР ============
#         river_board = turn_board + river
#         river_equity = calculate_equity(hero, river_board, num_opponents=1, simulations=300)
#         final_eval = evaluate_hand(hero + river_board)
        
#         river_prompt = f"""СТУДИЯ: РИВЕР (5 карт на столе)
# Твои карты: {hero[0]}, {hero[1]}
# Борд: {', '.join(str(c) for c in river_board)}
# Банк: $300, Стек: $770
# Оппонент поставил $200

# Финальная сила: категория {final_eval[0]}/9
# Эквити: {river_equity:.2%}"""
        
#         if river_equity > 0.65:
#             action = "CALL"
#             reason = "Высокое эквити, колл для вэлью"
#         elif river_equity < 0.25:
#             # Шанс на блеф!
#             if random.random() < 0.3:
#                 action = "RAISE 500"
#                 reason = "Блеф с рандомной картой, пытаемся выбить"
#             else:
#                 action = "FOLD 0"
#                 reason = f"Слабое эквити ({river_equity:.0%}), сбрасываем"
#         else:
#             action = "FOLD 0"
#             reason = "Пограничная рука, pot odds не оправдывают"
        
#         river_completion = f"""АНАЛИЗ: {reason}
# ДЕЙСТВИЕ: {action.split()[0]}
# СУММА: {action.split()[1] if len(action.split()) > 1 else 0}"""
        
#         examples.append({
#             "stage": "river",
#             "prompt": river_prompt,
#             "completion": river_completion,
#             "reward": 1.0
#         })
    
#     return examples

# ============ АНАЛИЗ БОРДА ============
def analyze_board(board, hero_cards):
    """Анализирует борд: попадания, дро, текстура."""
    all_cards = hero_cards + board
    suits = [c.suit for c in all_cards]

    hero_ranks = [c.rank for c in hero_cards]
    board_ranks = [c.rank for c in board]

    result = {
        "pair": False,
        "set": False,
        "two_pair": False,
        "flush_draw": False,
        "oesd": False,
        "gutshot": False,
        "overcards": 0,
        "wet": 0.0,
    }

    board_counts = Counter(board_ranks)

    for r in hero_ranks:
        if board_counts[r] >= 2:
            result["set"] = True
        elif board_counts[r] == 1:
            result["pair"] = True

    if sum(1 for r in hero_ranks if board_counts[r] >= 1) == 2:
        result["two_pair"] = True

    suit_counts = Counter(suits)
    if max(suit_counts.values()) >= 4:
        result["flush_draw"] = True

    unique = sorted(set([c.rank for c in all_cards]))
    for i in range(len(unique) - 3):
        if unique[i + 3] - unique[i] == 3:
            result["oesd"] = True
            break

    for i in range(len(unique) - 3):
        if unique[i + 3] - unique[i] == 4:
            result["gutshot"] = True
            break

    if board_ranks:
        max_board = max(board_ranks)
        result["overcards"] = sum(1 for r in hero_ranks if r > max_board)

    wet = 0
    if result["flush_draw"]:
        wet += 0.4
    if result["oesd"]:
        wet += 0.3
    if result["gutshot"]:
        wet += 0.15
    if len(set(board_ranks)) < 3 and board_ranks:
        wet += 0.1
    result["wet"] = min(wet, 1.0)

    return result


# ============ ГЕНЕРАЦИЯ ОБУЧАЮЩИХ ПРИМЕРОВ ============
def generate_training_examples(count=1000):
    """
    Генерирует примеры для обучения LoRA.
    Возвращает список: [{prompt, completion, reward, stage}, ...]
    """
    examples = []
    positions = ["UTG", "MP", "CO", "BTN", "SB", "BB"]

    for i in range(count):
        deck = build_deck()
        random.shuffle(deck)

        hero = deck[0:2]
        flop = deck[4:7]
        turn = deck[7:8]
        river = deck[8:9]

        position = random.choice(positions)
        stack = random.randint(500, 2000)
        bb = 10

        # ============ ФАЗА 1: ПРЕФЛОП ============
        tier, name = preflop_hand_category(hero[0], hero[1])

        # Случайная ситуация: рейз до нас или все сбросили
        action_before = random.choice(["все сбросили", "лимп $10", "рейз $30", "рейз $50"])
        if "рейз" in action_before:
            raise_amount = int(action_before.split("$")[1])
        else:
            raise_amount = bb

        preflop_prompt = f"""СТУДИЯ: ПРЕФЛОП (0 карт на столе)
ТВОИ КАРТЫ: {hero[0]}, {hero[1]}
ПОЗИЦИЯ: {position}
Стек: ${stack}, Блайнд: ${bb}
Действие до тебя: {action_before}
Категория руки: {name} (тир {tier}/8)

Прими решение: FOLD / CALL / RAISE
Ответь СТРОГО:
АНАЛИЗ: [короткое объяснение]
ДЕЙСТВИЕ: [FOLD/CALL/RAISE]
СУММА: [число]"""

        # Идеальное решение
        early_pos = position in ["UTG", "MP", "SB"]
        if tier <= 2:
            if "рейз" in action_before and raise_amount >= 40:
                preflop_action = f"RAISE {raise_amount * 3}"
                reason = f"Премиум {name} против рейза - 3-бет для вэлью"
            else:
                preflop_action = f"RAISE {bb * 3}"
                reason = f"Премиум {name} - наращиваем банк"
        elif tier <= 4:
            if early_pos and "рейз" in action_before:
                preflop_action = "FOLD 0"
                reason = f"Ранняя позиция против рейза - {name} слабоват"
            elif "рейз" in action_before:
                preflop_action = f"CALL {raise_amount}"
                reason = f"{name} в позиции - колл"
            else:
                preflop_action = f"RAISE {bb * 3}"
                reason = f"{name} - открываем рейзом"
        elif tier <= 5:
            if "рейз" in action_before or early_pos:
                preflop_action = "FOLD 0"
                reason = f"Слабый {name} - сбрасываем"
            else:
                preflop_action = f"CALL {bb}"
                reason = f"{name} - дёшево смотрим флоп"
        else:
            preflop_action = "FOLD 0"
            reason = f"Мусор {name} - сбрасываем"

        parts = preflop_action.split()
        preflop_completion = f"""АНАЛИЗ: {reason}
ДЕЙСТВИЕ: {parts[0]}
СУММА: {parts[1]}"""

        examples.append({
            "prompt": preflop_prompt,
            "completion": preflop_completion,
            "reward": 1.0,
            "stage": "preflop"
        })

        # ============ ФАЗА 2: ФЛОП ============
        flop_analysis = analyze_board(flop, hero)
        flop_equity = calculate_equity(hero, flop, num_opponents=1, simulations=150)
        pot_flop = 70
        opp_action_flop = random.choice(["чек", "бет $35", "бет $50"])

        flop_prompt = f"""СТУДИЯ: ФЛОП (3 карты на столе)
ТВОИ КАРТЫ: {hero[0]}, {hero[1]}
ФЛОП: {flop[0]}, {flop[1]}, {flop[2]}
БАНК: ${pot_flop}, СТЕК: ${stack - 30}
Действие оппонента: {opp_action_flop}

Анализ: пара={flop_analysis['pair']}, сет={flop_analysis['set']}, флеш-дро={flop_analysis['flush_draw']}, стрит-дро={flop_analysis['oesd']}
Эквити против 1 оппонента: {flop_equity:.2%}
Текстура: {'wet' if flop_analysis['wet'] > 0.4 else 'dry'}

Прими решение. Ответь СТРОГО:
АНАЛИЗ: [короткое объяснение]
ДЕЙСТВИЕ: [FOLD/CHECK/CALL/RAISE]
СУММА: [число]"""

        if "бет" in opp_action_flop:
            bet_amt = int(opp_action_flop.split("$")[1])
            pot_odds = bet_amt / (pot_flop + bet_amt)
            if flop_analysis["set"] or flop_equity > 0.7:
                action = f"RAISE {bet_amt * 3}"
                reason = f"Сильная рука ({flop_equity:.0%} эквити) - рейз для вэлью"
            elif flop_equity > pot_odds + 0.1:
                action = f"CALL {bet_amt}"
                reason = f"Эквити ({flop_equity:.0%}) > pot odds ({pot_odds:.0%})"
            elif flop_analysis["flush_draw"] or flop_analysis["oesd"]:
                action = f"CALL {bet_amt}"
                reason = f"Дро с implied odds"
            elif flop_equity < 0.25:
                action = "FOLD 0"
                reason = f"Слабое эквити ({flop_equity:.0%}), фолд"
            else:
                action = f"CALL {bet_amt}"
                reason = f"Средняя рука, колл"
        else:
            if flop_analysis["set"] or flop_equity > 0.7:
                action = f"RAISE {int(pot_flop * 0.6)}"
                reason = f"Сильная рука - вэлью-бет"
            elif flop_analysis["flush_draw"] or flop_analysis["oesd"]:
                action = f"RAISE {int(pot_flop * 0.5)}"
                reason = f"Полу-блеф с дро"
            elif flop_equity > 0.5:
                action = f"RAISE {int(pot_flop * 0.5)}"
                reason = f"Вэлью-бет с хорошим эквити"
            else:
                action = "CHECK 0"
                reason = f"Слабая рука ({flop_equity:.0%}), чек"

        parts = action.split()
        flop_completion = f"""АНАЛИЗ: {reason}
ДЕЙСТВИЕ: {parts[0]}
СУММА: {parts[1]}"""

        examples.append({
            "prompt": flop_prompt,
            "completion": flop_completion,
            "reward": 1.0,
            "stage": "flop"
        })

        # ============ ФАЗА 3: ТЁРН ============
        turn_board = flop + turn
        turn_analysis = analyze_board(turn_board, hero)
        turn_equity = calculate_equity(hero, turn_board, num_opponents=1, simulations=150)
        pot_turn = 120
        bet_turn = 60

        turn_prompt = f"""СТУДИЯ: ТЁРН (4 карты на столе)
ТВОИ КАРТЫ: {hero[0]}, {hero[1]}
БОРД: {', '.join(str(c) for c in turn_board)}
БАНК: ${pot_turn}, СТЕК: ${stack - 100}
Оппонент поставил: ${bet_turn}

Анализ: пара={turn_analysis['pair']}, сет={turn_analysis['set']}, флеш-дро={turn_analysis['flush_draw']}, стрит-дро={turn_analysis['oesd']}
Эквити: {turn_equity:.2%}
Pot odds: {bet_turn / (pot_turn + bet_turn):.2%}

Прими решение. Ответь СТРОГО:
АНАЛИЗ: [короткое объяснение]
ДЕЙСТВИЕ: [FOLD/CALL/RAISE]
СУММА: [число]"""

        pot_odds = bet_turn / (pot_turn + bet_turn)
        if turn_equity > 0.65:
            action = f"RAISE {bet_turn * 3}"
            reason = f"Сильная рука ({turn_equity:.0%}), рейз"
        elif turn_equity > pot_odds + 0.05:
            action = f"CALL {bet_turn}"
            reason = f"Эквити ({turn_equity:.0%}) > pot odds ({pot_odds:.0%})"
        elif turn_analysis["flush_draw"] or turn_analysis["oesd"]:
            action = f"CALL {bet_turn}"
            reason = f"Дро с 1 картой до ривера (~20%)"
        elif turn_equity < 0.3:
            action = "FOLD 0"
            reason = f"Эквити ({turn_equity:.0%}) < pot odds, фолд"
        else:
            action = f"CALL {bet_turn}"
            reason = f"Пограничная рука"

        parts = action.split()
        turn_completion = f"""АНАЛИЗ: {reason}
ДЕЙСТВИЕ: {parts[0]}
СУММА: {parts[1]}"""

        examples.append({
            "prompt": turn_prompt,
            "completion": turn_completion,
            "reward": 1.0,
            "stage": "turn"
        })

        # ============ ФАЗА 4: РИВЕР ============
        river_board = turn_board + river
        river_equity = calculate_equity(hero, river_board, num_opponents=1, simulations=200)
        final_eval = evaluate_hand(hero + river_board)
        hand_name = HAND_NAMES.get(final_eval[0], "неизвестно")
        pot_river = 300
        bet_river = 200

        river_prompt = f"""СТУДИЯ: РИВЕР (5 карт на столе)
ТВОИ КАРТЫ: {hero[0]}, {hero[1]}
БОРД: {', '.join(str(c) for c in river_board)}
БАНК: ${pot_river}, СТЕК: ${stack - 300}
Оппонент поставил: ${bet_river}

Твоя рука: {hand_name}
Эквити на вскрытии: {river_equity:.2%}
Pot odds: {bet_river / (pot_river + bet_river):.2%}

Прими решение. Ответь СТРОГО:
АНАЛИЗ: [короткое объяснение]
ДЕЙСТВИЕ: [FOLD/CALL/RAISE]
СУММА: [число]"""

        pot_odds = bet_river / (pot_river + bet_river)
        if river_equity > 0.65:
            if random.random() < 0.4:
                action = f"RAISE {bet_river * 2}"
                reason = f"Сильная рука ({hand_name}), рейз для вэлью"
            else:
                action = f"CALL {bet_river}"
                reason = f"{hand_name} ({river_equity:.0%}), колл"
        elif river_equity < 0.25:
            if random.random() < 0.25:
                action = f"RAISE {pot_river}"
                reason = "Блеф - пытаюсь выбить оппонента"
            else:
                action = "FOLD 0"
                reason = f"Слабая рука ({hand_name}, {river_equity:.0%}), фолд"
        elif river_equity > pot_odds:
            action = f"CALL {bet_river}"
            reason = f"{hand_name}, эквити ({river_equity:.0%}) > pot odds"
        else:
            action = "FOLD 0"
            reason = f"Эквити ({river_equity:.0%}) < pot odds, фолд"

        parts = action.split()
        river_completion = f"""АНАЛИЗ: {reason}
ДЕЙСТВИЕ: {parts[0]}
СУММА: {parts[1]}"""

        examples.append({
            "prompt": river_prompt,
            "completion": river_completion,
            "reward": 1.0,
            "stage": "river"
        })

        if (i + 1) % 50 == 0:
            print(f"  Сгенерировано {i + 1} рук ({(i + 1) * 4} примеров)...")

    return examples


