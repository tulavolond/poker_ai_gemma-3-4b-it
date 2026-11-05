"""Генерация датасета: случайные раздачи + эталонная комбинация от treys."""

import json
import random
from collections import Counter

from treys import Card, Evaluator

from poker_common import target_text

random.seed(42)

TRAIN_SIZE = 400
TEST_SIZE = 80
MAX_PER_CLASS = 70  # чтобы класс "Пара" не съел весь датасет

RANKS = "23456789TJQKA"
SUITS = "shdc"  # spades, hearts, diamonds, clubs
DECK = [r + s for r in RANKS for s in SUITS]

ev = Evaluator()


def make_example():
    cards = random.sample(DECK, 7)
    board, hand = cards[:5], cards[5:]

    rank = ev.evaluate([Card.new(c) for c in board], [Card.new(c) for c in hand])
    cls = ev.get_rank_class(rank)  # 1 = стрит-флеш ... 9 = старшая карта

    return {
        "board": " ".join(board),
        "hand": " ".join(hand),
        "cls": cls,
        "answer": target_text(cls),
        "_key": tuple(sorted(cards)),
    }


def generate(n, seen):
    out, per_class = [], Counter()
    attempts = 0

    while len(out) < n and attempts < n * 500:
        attempts += 1
        ex = make_example()

        if ex["_key"] in seen:
            continue
        if per_class[ex["cls"]] >= MAX_PER_CLASS:
            continue

        seen.add(ex["_key"])
        per_class[ex["cls"]] += 1
        ex.pop("_key")
        out.append(ex)

    return out


seen = set()
train = generate(TRAIN_SIZE, seen)
test = generate(TEST_SIZE, seen)  # пересечений с train нет

for path, data in [("data_train.jsonl", train), ("data_test.jsonl", test)]:
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{path}: {len(data)} примеров")

print("\nПример строки:")
print(json.dumps(train[0], ensure_ascii=False))
print("\nРаспределение классов в train:", Counter(r["cls"] for r in train))