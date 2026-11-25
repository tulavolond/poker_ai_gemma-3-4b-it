"""
Генератор покерного датасета (ривер) с ТОЧНЫМ эквити.
Полный перебор 990 рук соперника, корректные кикеры, сбалансированные классы.
"""
import json
import random
from collections import Counter
from itertools import combinations

SEED = 42
rng = random.Random(SEED)

RANKS = "23456789TJQKA"
SUITS = "shdc"
DECK = list(range(52))          # карта = int: rank = c >> 2, suit = c & 3

NOM_PL = ["двойки","тройки","четвёрки","пятёрки","шестёрки","семёрки",
          "восьмёрки","девятки","десятки","валеты","дамы","короли","тузы"]
GEN_PL = ["двоек","троек","четвёрок","пятёрок","шестёрок","семёрок",
          "восьмёрок","девяток","десяток","валетов","дам","королей","тузов"]
GEN_SG = ["двойки","тройки","четвёрки","пятёрки","шестёрки","семёрки",
          "восьмёрки","девятки","десятки","валета","дамы","короля","туза"]
CAT_NAMES = ["Старшая карта","Одна пара","Две пары","Тройка / Сет","Стрит",
             "Флеш","Фулл-хаус","Каре","Стрит-флеш"]

# --- Таблица старшей карты стрита для всех 8192 масок рангов ---
STRAIGHT_HIGH = [-1] * (1 << 13)
for _m in range(1 << 13):
    _hi = -1
    for _h in range(12, 3, -1):
        if (_m >> (_h - 4)) & 0b11111 == 0b11111:
            _hi = _h
            break
    if _hi == -1 and (_m & 0b1000000001111) == 0b1000000001111:
        _hi = 3                                    # колесо A-2-3-4-5
    STRAIGHT_HIGH[_m] = _hi


def eval5(cards):
    """cards — отсортированный tuple из 5 int.
    Возвращает кортеж фиксированной длины (категория, t1..t5) для полного сравнения."""
    r = sorted((c >> 2 for c in cards), reverse=True)
    flush = len({c & 3 for c in cards}) == 1
    mask = 0
    for x in r:
        mask |= 1 << x
    sh = STRAIGHT_HIGH[mask]

    groups = sorted(Counter(r).items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    shape = [g[1] for g in groups]

    if flush and sh >= 0:
        return (8, sh, 0, 0, 0, 0)
    if shape[0] == 4:
        return (7, groups[0][0], groups[1][0], 0, 0, 0)
    if shape[0] == 3 and shape[1] == 2:
        return (6, groups[0][0], groups[1][0], 0, 0, 0)
    if flush:
        return (5, r[0], r[1], r[2], r[3], r[4])
    if sh >= 0:
        return (4, sh, 0, 0, 0, 0)
    if shape[0] == 3:
        return (3, groups[0][0], groups[1][0], groups[2][0], 0, 0)
    if shape[0] == 2 and shape[1] == 2:
        return (2, groups[0][0], groups[1][0], groups[2][0], 0, 0)
    if shape[0] == 2:
        return (1, groups[0][0], groups[1][0], groups[2][0], groups[3][0], 0)
    return (0, r[0], r[1], r[2], r[3], r[4])


def eval7(cards):
    """Полный скор лучшей пятёрки из 7 карт (с кикерами!)."""
    return max(eval5(c) for c in combinations(sorted(cards), 5))


def best_five(cards):
    cards = sorted(cards)
    return max(((eval5(c), c) for c in combinations(cards, 5)), key=lambda x: x[0])


def describe(s):
    c = s[0]
    if c == 8: return "Роял-флеш" if s[1] == 12 else f"Стрит-флеш до {GEN_SG[s[1]]}"
    if c == 7: return f"Каре {GEN_PL[s[1]]}"
    if c == 6: return f"Фулл-хаус ({NOM_PL[s[1]]} по {GEN_PL[s[2]]})"
    if c == 5: return f"Флеш до {GEN_SG[s[1]]}"
    if c == 4: return f"Стрит до {GEN_SG[s[1]]}"
    if c == 3: return f"Тройка {GEN_PL[s[1]]}"
    if c == 2: return f"Две пары ({NOM_PL[s[1]]} и {NOM_PL[s[2]]})"
    if c == 1: return f"Пара {GEN_PL[s[1]]}"
    return f"Старшая карта {GEN_SG[s[1]]}"


SUIT_SYM = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}

def visual(cards):
    """Единый формат: ранг + масть. T корректно превращается в 10."""
    out = []
    for c in cards:
        r = RANKS[c >> 2]
        out.append(f"{'10' if r == 'T' else r}{SUIT_SYM[SUITS[c & 3]]}")
    return " ".join(out)


def analyze(board, hand):
    """ТОЧНЫЙ перебор всех C(45,2)=990 рук соперника."""
    board = sorted(board)
    on_board = set(board)
    b4 = list(combinations(board, 4))
    b3 = list(combinations(board, 3))
    board5 = eval5(tuple(board))

    # Предрасчёт "4 карты борда + 1 карманная" — 47 вариантов вместо 990*10
    one_hole = {h: max(eval5(tuple(sorted(c + (h,)))) for c in b4)
                for h in DECK if h not in on_board}

    def score7(h1, h2):
        best = board5
        if one_hole[h1] > best: best = one_hole[h1]
        if one_hole[h2] > best: best = one_hole[h2]
        for c in b3:
            s = eval5(tuple(sorted(c + (h1, h2))))
            if s > best: best = s
        return best

    my = score7(*hand)
    remaining = [c for c in DECK if c not in on_board and c not in hand]

    wins = ties = losses = total = 0
    dist, beats = Counter(), Counter()
    for o1, o2 in combinations(remaining, 2):
        sc = score7(o1, o2)
        total += 1
        dist[sc[0]] += 1
        if my > sc:   wins += 1
        elif my == sc: ties += 1
        else:
            losses += 1
            beats[sc[0]] += 1

    return {
        "my": my, "total": total, "wins": wins, "ties": ties, "losses": losses,
        "equity": round((wins + ties / 2) / total * 100, 1),
        "dist": dist, "beats": beats,
    }


def pct_exact_100(dist, total):
    """Проценты, которые суммируются ровно в 100.0 (метод наибольшего остатка)."""
    raw = {k: dist.get(k, 0) / total * 100 for k in range(9)}
    out = {k: round(v, 1) for k, v in raw.items()}
    diff = round(100.0 - sum(out.values()), 1)
    if abs(diff) >= 0.1:
        k = max(raw, key=lambda k: raw[k])
        out[k] = round(out[k] + diff, 1)
    return out


# ---------------------- САНИТИ-ТЕСТЫ ----------------------
def C(s):
    return RANKS.index(s[0]) * 4 + SUITS.index(s[1])

def H(s):
    return tuple(sorted(C(t) for t in s.split()))

assert eval5(H("As Ks Qs Js Ts")) > eval5(H("Ac Ad Ah As Kd")), "роял должен бить каре"
assert eval5(H("As 2c 3d 4h 5s")) == (4, 3, 0, 0, 0, 0), "колесо = стрит до 5"
assert eval5(H("Ac Ad Ks Kh Qc")) > eval5(H("As Ah Kc Kd Jc")), "кикер двух пар"
assert eval5(H("Ac Ad Ks Qh Jc")) > eval5(H("As Ah Kc Qd Tc")), "кикер пары"
assert eval5(H("5c 5d 5h 2s 2d")) > eval5(H("4c 4d 4h As Ad")), "фулл-хаус по старшинству трипса"
assert eval5(H("6c 6d 6h As Ad")) > eval5(H("6s 6c 6d Ks Kh")), "фулл-хаус по паре"
assert eval5(H("2c 7d 9h Js Ac"))[0] == 0
assert eval7(H("As Ad Ac Ah Kd Kc 2s"))[0] == 7
print("✅ Все тесты оценщика пройдены\n")


# ---------------------- ГЕНЕРАЦИЯ ----------------------
QUOTA = {0: 80, 1: 80, 2: 80, 3: 80, 4: 80, 5: 80, 6: 80, 7: 40, 8: 20}
MAX_DRAWS = 400_000

counts = {k: 0 for k in QUOTA}
seen = set()
dataset = []
draws = 0
TARGET = sum(QUOTA.values())

print(f"Генерация до {TARGET} примеров (полный перебор эквити)...")

while len(dataset) < TARGET and draws < MAX_DRAWS:
    draws += 1
    dealt = rng.sample(DECK, 7)
    board, hand = dealt[:5], dealt[5:]

    my_cat = eval7(board + hand)[0]
    if counts[my_cat] >= QUOTA[my_cat]:
        continue

    key = (tuple(sorted(board)), tuple(sorted(hand)))
    if key in seen:
        continue
    seen.add(key)

    r = analyze(board, hand)
    counts[my_cat] += 1

    # --- реальные факты для CoT ---
    rank_cnt = Counter(c >> 2 for c in board)
    max_rank = max(rank_cnt.values())
    paired_txt = ("не спаренная" if max_rank == 1 else
                  "спаренная (одна пара)" if max_rank == 2 else
                  "спаренная (трипс на борде)" if max_rank == 3 else
                  "спаренная (каре на борде)")

    flush_combos = r["dist"][5] + r["dist"][8]
    straight_combos = r["dist"][4] + r["dist"][8]
    flush_txt = f"возможен ({flush_combos} комбо)" if flush_combos else "невозможен (0 комбо)"
    straight_txt = f"возможен ({straight_combos} комбо)" if straight_combos else "невозможен (0 комбо)"

    my_score, my_five = best_five(board + hand)

    if r["losses"] == 0:
        threats = "ничего — это натс, максимум возможен делёж банка"
    else:
        threats = "; ".join(f"{CAT_NAMES[k]} — {v} комбо"
                            for k, v in sorted(r["beats"].items(), reverse=True))

    pcts = pct_exact_100(r["dist"], r["total"])
    board_v, hand_v = visual(board), visual(hand)

    user_str = f"на столе {board_v}, у меня {hand_v}"

    assistant_str = (
        f"### Ход мыслей:\n"
        f"1. Доска: {board_v}. Карманные карты: {hand_v}.\n"
        f"2. Текстура доски: {paired_txt}. Флеш соперника: {flush_txt}. "
        f"Стрит соперника: {straight_txt}.\n"
        f"3. Моя лучшая пятёрка из 7 карт: {visual(my_five)} → {describe(my_score)}.\n"
        f"4. Меня могут побить: {threats}.\n"
        f"5. Перебор всех рук соперника: C(45,2) = {r['total']} комбинаций.\n"
        f"   Побеждаем: {r['wins']} | Делим банк: {r['ties']} | Проигрываем: {r['losses']}\n"
        f"6. Эквити = ({r['wins']} + {r['ties']}/2) / {r['total']} × 100 = {r['equity']}%\n\n"
        f"### Итог:\n"
        f"Вероятность победы: {r['equity']}%\n\n"
        f"Распределение комбинаций соперника (все {r['total']} комбо):\n"
        + "\n".join(f"* {CAT_NAMES[k]}: [{pcts[k]}%]" for k in range(8, -1, -1))
    )

    dataset.append({
        "instruction": ("Ты — покерный калькулятор. Рассчитай шансы на выигрыш "
                        "и комбинации соперников на Ривере, используя пошаговые рассуждения."),
        "input": user_str,
        "output": assistant_str,
    })

    if len(dataset) % 50 == 0:
        print(f"  {len(dataset)}/{TARGET} (просеяно раздач: {draws})")

rng.shuffle(dataset)

# Hold-out для честной оценки обобщения
eval_size = max(20, len(dataset) // 20)
eval_set, train_set = dataset[:eval_size], dataset[eval_size:]

for name, part in [("poker_train_dataset.json", train_set),
                   ("poker_eval_dataset.json", eval_set)]:
    with open(name, "w", encoding="utf-8") as f:
        json.dump(part, f, ensure_ascii=False, indent=2)

print("\nБаланс комбинаций:", {CAT_NAMES[k]: v for k, v in counts.items()})
print(f"Просеяно раздач: {draws}")
print(f"Train: {len(train_set)} | Eval: {len(eval_set)}")