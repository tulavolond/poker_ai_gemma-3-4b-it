import json
import random
from itertools import combinations
from collections import Counter


# ============================================================
# Карты
# ============================================================

SUITS = ["♠", "♥", "♦", "♣"]
RANKS = list(range(2, 15))

RANK_CHAR = {
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "T",
    11: "J",
    12: "Q",
    13: "K",
    14: "A",
}

CHAR_TO_RANK = {v: k for k, v in RANK_CHAR.items()}

DECK = [(rank, suit) for rank in RANKS for suit in SUITS]


# ============================================================
# Категории комбинаций
# ============================================================

HIGH = 0
PAIR = 1
TWO_PAIR = 2
SET = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
QUADS = 7
STRAIGHT_FLUSH = 8
ROYAL_FLUSH = 9


LABEL_TO_CATEGORY = {
    "Сет": SET,
    "Стрит": STRAIGHT,
    "Флеш": FLUSH,
    "Фулл-хаус": FULL_HOUSE,
    "Каре": QUADS,
    "Стрит-флеш": STRAIGHT_FLUSH,
    "Роял-флеш": ROYAL_FLUSH,
}

CHECK_WORDS = {
    "Сет": ["сет"],
    "Стрит": ["стрит"],
    "Флеш": ["флеш"],
    "Фулл-хаус": ["фулл"],
    "Каре": ["каре"],
    "Стрит-флеш": ["стрит", "флеш"],
    "Роял-флеш": ["роял"],
}


# ============================================================
# Вспомогательные функции
# ============================================================

def card_str(card):
    rank, suit = card
    return f"{RANK_CHAR[rank]}{suit}"


def format_cards(cards):
    return " ".join(card_str(card) for card in cards)


def parse_card(s):
    return CHAR_TO_RANK[s[0]], s[1]


def parse_cards(s):
    return [parse_card(x) for x in s.split()]


def straight_info(ranks):
    """
    Возвращает (is_straight, high_card).
    Учитывает колесо A-2-3-4-5.
    """
    uniq = sorted(set(ranks), reverse=True)

    for i in range(len(uniq) - 4):
        if uniq[i] - uniq[i + 4] == 4:
            return True, uniq[i]

    if {14, 5, 4, 3, 2}.issubset(set(uniq)):
        return True, 5

    return False, None


def category_5(cards):
    """
    Определяет категорию для 5 карт.
    """
    ranks = [card[0] for card in cards]
    suits = [card[1] for card in cards]

    is_flush = len(set(suits)) == 1
    is_straight, high = straight_info(ranks)

    counts = sorted(Counter(ranks).values())

    if is_flush and is_straight:
        if high == 14:
            return ROYAL_FLUSH
        return STRAIGHT_FLUSH

    if counts == [1, 4]:
        return QUADS

    if counts == [2, 3]:
        return FULL_HOUSE

    if is_flush:
        return FLUSH

    if is_straight:
        return STRAIGHT

    if counts == [1, 1, 3]:
        return SET

    if counts == [1, 2, 2]:
        return TWO_PAIR

    if counts == [1, 1, 1, 2]:
        return PAIR

    return HIGH


def best_category(cards):
    """
    Лучшая категория из 7 карт: доска + рука.
    """
    if len(cards) < 5:
        raise ValueError("Нужно минимум 5 карт")

    return max(category_5(combo) for combo in combinations(cards, 5))


# ============================================================
# Генераторы конкретных комбинаций
# ============================================================

def random_straight_ranks(max_start=10):
    """
    Возвращает ранги для стрита.
    Иногда использует колесо A-2-3-4-5.
    max_start=10 разрешает T-J-Q-K-A.
    max_start=9 запрещает роял.
    """
    if random.random() < 0.12:
        return [14, 2, 3, 4, 5]

    start = random.randint(2, max_start)
    return [start + i for i in range(5)]


def build_set(max_attempts=5000):
    """
    Сет: карманная пара в руке + одна карта того же ранга на столе.
    Лучшая комбинация должна остаться именно сетом, а не стритом/флешем/фуллом.
    """
    for _ in range(max_attempts):
        rank = random.choice(RANKS)

        # Три карты одного ранга: две в руке, одна на столе
        used_suits = random.sample(SUITS, 3)

        hand = [
            (rank, used_suits[0]),
            (rank, used_suits[1]),
        ]

        board = [
            (rank, used_suits[2]),
        ]

        used = set(hand + board)

        # Нужны 4 кикера на доске, чтобы всего было 5 карт на столе
        kicker_rank_choices = [r for r in RANKS if r != rank]
        kicker_ranks = random.sample(kicker_rank_choices, 4)

        kickers = []
        for kicker_rank in kicker_ranks:
            available_suits = [s for s in SUITS if (kicker_rank, s) not in used]
            if not available_suits:
                break

            kicker_card = (kicker_rank, random.choice(available_suits))
            kickers.append(kicker_card)
            used.add(kicker_card)

        if len(kickers) != 4:
            continue

        board += kickers

        if len(set(board)) != 5 or len(set(hand)) != 2:
            continue

        all_cards = board + hand

        if best_category(all_cards) == SET:
            random.shuffle(board)
            random.shuffle(hand)
            return board, hand

    raise RuntimeError("Не удалось сгенерировать Сет")


def build_straight(max_attempts=5000):
    """
    Стрит: есть 5 последовательных рангов, но нет флеша/стрит-флеша.
    """
    for _ in range(max_attempts):
        ranks = random_straight_ranks(max_start=10)

        straight_cards = [
            (rank, random.choice(SUITS))
            for rank in ranks
        ]

        if len(set(straight_cards)) != 5:
            continue

        used = set(straight_cards)

        extra_rank_choices = [r for r in RANKS if r not in set(ranks)]
        if len(extra_rank_choices) < 2:
            continue

        extra_ranks = random.sample(extra_rank_choices, 2)
        extras = []

        for extra_rank in extra_ranks:
            available_suits = [s for s in SUITS if (extra_rank, s) not in used]
            if not available_suits:
                break

            extra_card = (extra_rank, random.choice(available_suits))
            extras.append(extra_card)
            used.add(extra_card)

        if len(extras) != 2:
            continue

        all_cards = straight_cards + extras

        if best_category(all_cards) == STRAIGHT:
            hand = random.sample(straight_cards, 2)
            hand_set = set(hand)
            board = [card for card in all_cards if card not in hand_set]

            if len(board) == 5:
                random.shuffle(board)
                random.shuffle(hand)
                return board, hand

    raise RuntimeError("Не удалось сгенерировать Стрит")


def build_flush(max_attempts=5000):
    """
    Флеш: 5 карт одной масти, но не стрит-флеш.
    """
    for _ in range(max_attempts):
        suit = random.choice(SUITS)

        flush_ranks = random.sample(RANKS, 5)

        # Если эти 5 рангов образуют стрит, получится стрит-флеш
        if straight_info(flush_ranks)[0]:
            continue

        flush_cards = [
            (rank, suit)
            for rank in flush_ranks
        ]

        used = set(flush_cards)

        extra_rank_choices = [r for r in RANKS if r not in set(flush_ranks)]
        if len(extra_rank_choices) < 2:
            continue

        extra_ranks = random.sample(extra_rank_choices, 2)
        extras = []

        other_suits = [s for s in SUITS if s != suit]

        for extra_rank in extra_ranks:
            available_suits = [s for s in other_suits if (extra_rank, s) not in used]
            if not available_suits:
                break

            extra_card = (extra_rank, random.choice(available_suits))
            extras.append(extra_card)
            used.add(extra_card)

        if len(extras) != 2:
            continue

        all_cards = flush_cards + extras

        if best_category(all_cards) == FLUSH:
            hand = random.sample(flush_cards, 2)
            hand_set = set(hand)
            board = [card for card in all_cards if card not in hand_set]

            if len(board) == 5:
                random.shuffle(board)
                random.shuffle(hand)
                return board, hand

    raise RuntimeError("Не удалось сгенерировать Флеш")


def build_full_house(max_attempts=5000):
    """
    Фулл-хаус: тройка + пара.
    """
    for _ in range(max_attempts):
        triple_rank, pair_rank = random.sample(RANKS, 2)

        triple_cards = [
            (triple_rank, suit)
            for suit in random.sample(SUITS, 3)
        ]

        pair_cards = [
            (pair_rank, suit)
            for suit in random.sample(SUITS, 2)
        ]

        core_cards = triple_cards + pair_cards
        used = set(core_cards)

        kicker_rank_choices = [
            r for r in RANKS
            if r not in {triple_rank, pair_rank}
        ]

        if len(kicker_rank_choices) < 2:
            continue

        kicker_ranks = random.sample(kicker_rank_choices, 2)
        kickers = []

        for kicker_rank in kicker_ranks:
            available_suits = [s for s in SUITS if (kicker_rank, s) not in used]
            if not available_suits:
                break

            kicker_card = (kicker_rank, random.choice(available_suits))
            kickers.append(kicker_card)
            used.add(kicker_card)

        if len(kickers) != 2:
            continue

        all_cards = core_cards + kickers

        if best_category(all_cards) == FULL_HOUSE:
            hand = random.sample(core_cards, 2)
            hand_set = set(hand)
            board = [card for card in all_cards if card not in hand_set]

            if len(board) == 5:
                random.shuffle(board)
                random.shuffle(hand)
                return board, hand

    raise RuntimeError("Не удалось сгенерировать Фулл-хаус")


def build_quads(max_attempts=5000):
    """
    Каре: четыре карты одного ранга.
    """
    for _ in range(max_attempts):
        rank = random.choice(RANKS)

        quad_cards = [
            (rank, suit)
            for suit in SUITS
        ]

        used = set(quad_cards)

        kicker_rank_choices = [r for r in RANKS if r != rank]
        kicker_ranks = random.sample(kicker_rank_choices, 3)

        kickers = []

        for kicker_rank in kicker_ranks:
            available_suits = [s for s in SUITS if (kicker_rank, s) not in used]
            if not available_suits:
                break

            kicker_card = (kicker_rank, random.choice(available_suits))
            kickers.append(kicker_card)
            used.add(kicker_card)

        if len(kickers) != 3:
            continue

        all_cards = quad_cards + kickers

        if best_category(all_cards) == QUADS:
            hand = random.sample(quad_cards, 2)
            hand_set = set(hand)
            board = [card for card in all_cards if card not in hand_set]

            if len(board) == 5:
                random.shuffle(board)
                random.shuffle(hand)
                return board, hand

    raise RuntimeError("Не удалось сгенерировать Каре")


def build_straight_flush(max_attempts=5000):
    """
    Стрит-флеш, но не роял-флеш.
    """
    for _ in range(max_attempts):
        suit = random.choice(SUITS)

        # Либо колесо A-2-3-4-5, либо стрит-флеш от 2 до 9
        # Без T-J-Q-K-A, чтобы не получить роял
        if random.random() < 0.15:
            ranks = [14, 2, 3, 4, 5]
        else:
            start = random.randint(2, 9)
            ranks = [start + i for i in range(5)]

        straight_flush_cards = [
            (rank, suit)
            for rank in ranks
        ]

        used = set(straight_flush_cards)

        extra_rank_choices = [r for r in RANKS if r not in set(ranks)]
        if len(extra_rank_choices) < 2:
            continue

        extra_ranks = random.sample(extra_rank_choices, 2)
        extras = []

        other_suits = [s for s in SUITS if s != suit]

        for extra_rank in extra_ranks:
            available_suits = [s for s in other_suits if (extra_rank, s) not in used]
            if not available_suits:
                break

            extra_card = (extra_rank, random.choice(available_suits))
            extras.append(extra_card)
            used.add(extra_card)

        if len(extras) != 2:
            continue

        all_cards = straight_flush_cards + extras

        if best_category(all_cards) == STRAIGHT_FLUSH:
            hand = random.sample(straight_flush_cards, 2)
            hand_set = set(hand)
            board = [card for card in all_cards if card not in hand_set]

            if len(board) == 5:
                random.shuffle(board)
                random.shuffle(hand)
                return board, hand

    raise RuntimeError("Не удалось сгенерировать Стрит-флеш")


def build_royal_flush(max_attempts=5000):
    """
    Роял-флеш: T-J-Q-K-A одной масти.
    """
    for _ in range(max_attempts):
        suit = random.choice(SUITS)

        royal_ranks = [14, 13, 12, 11, 10]

        royal_cards = [
            (rank, suit)
            for rank in royal_ranks
        ]

        used = set(royal_cards)

        extra_rank_choices = [r for r in RANKS if r not in set(royal_ranks)]
        if len(extra_rank_choices) < 2:
            continue

        extra_ranks = random.sample(extra_rank_choices, 2)
        extras = []

        other_suits = [s for s in SUITS if s != suit]

        for extra_rank in extra_ranks:
            available_suits = [s for s in other_suits if (extra_rank, s) not in used]
            if not available_suits:
                break

            extra_card = (extra_rank, random.choice(available_suits))
            extras.append(extra_card)
            used.add(extra_card)

        if len(extras) != 2:
            continue

        all_cards = royal_cards + extras

        if best_category(all_cards) == ROYAL_FLUSH:
            hand = random.sample(royal_cards, 2)
            hand_set = set(hand)
            board = [card for card in all_cards if card not in hand_set]

            if len(board) == 5:
                random.shuffle(board)
                random.shuffle(hand)
                return board, hand

    raise RuntimeError("Не удалось сгенерировать Роял-флеш")


# ============================================================
# Сборка датасета
# ============================================================

GENERATORS = [
    (build_set, SET, "Сет"),
    (build_straight, STRAIGHT, "Стрит"),
    (build_flush, FLUSH, "Флеш"),
    (build_full_house, FULL_HOUSE, "Фулл-хаус"),
    (build_quads, QUADS, "Каре"),
    (build_straight_flush, STRAIGHT_FLUSH, "Стрит-флеш"),
    (build_royal_flush, ROYAL_FLUSH, "Роял-флеш"),
]


def make_example(builder, expected_category, label):
    board, hand = builder()
    all_cards = board + hand

    if best_category(all_cards) != expected_category:
        raise RuntimeError(f"Сгенерирован неверный пример для класса {label}")

    return {
        "board": format_cards(board),
        "hand": format_cards(hand),
        "answer": f"Комбинация: {label}",
        "check": CHECK_WORDS[label],
    }


def generate_dataset(per_class=10, seed=42):
    """
    Генерирует по per_class примеров для каждой комбинации.
    Если per_class=30, получится 30 * 7 = 210 примеров.
    """
    if seed is not None:
        random.seed(seed)

    examples = []

    for _ in range(per_class):
        for builder, expected_category, label in GENERATORS:
            examples.append(
                make_example(
                    builder=builder,
                    expected_category=expected_category,
                    label=label,
                )
            )

    random.shuffle(examples)
    return examples


def verify_examples(examples):
    """
    Проверяет, что во всех примерах нет дублей карт
    и фактическая лучшая комбинация совпадает с ответом.
    """
    for i, ex in enumerate(examples):
        board = parse_cards(ex["board"])
        hand = parse_cards(ex["hand"])
        all_cards = board + hand

        if len(set(all_cards)) != 7:
            raise ValueError(f"Пример {i}: найдены дубликаты карт")

        cat = best_category(all_cards)

        label = ex["answer"].replace("Комбинация: ", "").strip()
        expected_cat = LABEL_TO_CATEGORY[label]

        if cat != expected_cat:
            raise ValueError(
                f"Пример {i}: ожидалась комбинация {label}, "
                f"но фактически другая"
            )

    print(f"✅ Проверено {len(examples)} примеров. Все комбинации корректны.")


def save_jsonl(examples, path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"✅ Датасет сохранен: {path}")


def load_jsonl(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


# ============================================================
# Запуск
# ============================================================

if __name__ == "__main__":
    # Можно поменять количество примеров каждого класса
    PER_CLASS = 30

    examples = generate_dataset(
        per_class=PER_CLASS,
        seed=42,
    )

    verify_examples(examples)

    save_jsonl(
        examples,
        path="poker_combinations_dataset.jsonl",
    )

    print("\nПЕРВЫЕ ПРИМЕРЫ:")
    print("=" * 80)

    for ex in examples[:7]:
        print(ex)