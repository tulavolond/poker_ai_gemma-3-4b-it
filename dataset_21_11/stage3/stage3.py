# stage3.py — префлоп чарты (RFI - Raise First In)
import json, random
from pathlib import Path
from treys import Card

HERE = Path(__file__).resolve().parent
random.seed(303)

RANKS = "23456789TJQKA"; RV = {r: i+2 for i, r in enumerate(RANKS)}
SUIT = {"♠":"s", "♥":"h", "♦":"d", "♣":"c"}; SR = {v:k for k,v in SUIT.items()}
FULL = [Card.new(r+s) for r in RANKS for s in "shdc"]

def rs(c):
    s = Card.int_to_str(c); return RV[s[0]], s[1]
def disp(c):
    r, s = rs(c); return ("10" if r == 10 else RANKS[r-2]) + SR[s]
def fmt(cs): return " ".join(disp(c) for c in cs)

def categorize(h):
    r1, r2 = sorted([rs(c)[0] for c in h], reverse=True)
    s1, s2 = [rs(c)[1] for c in h]
    suited = (s1 == s2)
    pair = (r1 == r2)
    gap = r1 - r2

    if pair:
        if r1 >= 10: return "Premium"      # TT+
        if r1 >= 6: return "Medium Pairs"  # 99-66
        return "Small Pairs"               # 55-22

    if r1 == 14: # Туз
        if r2 >= 13 and suited: return "Premium" # AKs
        if r2 >= 12: return "Strong"             # AQs, AQo, AKo
        if r2 >= 10 and suited: return "Strong"  # ATs, AJs
        if r2 >= 10: return "Offsuit Broadways"  # ATo, AJo
        if suited: return "Suited Ax"            # A5s, A2s...
        return "Trash"                           # A9o - A2o

    if r1 >= 11 and r2 >= 10: # Бродвеи
        if suited: return "Suited Broadways"     # KQs, QJs, JTs
        return "Offsuit Broadways"               # KQo, QJo, JTo

    if suited and gap == 1 and r1 <= 10: return "Suited Connectors" # T9s ... 54s
    if suited and gap == 2 and r1 <= 10: return "Suited One-Gappers" # T8s ...
    if suited and r1 <= 10: return "Suited Low" # 85s, 74s...

    return "Trash"

# GTO 6-max RFI Chart (упрощенный)
CHART = {
    "Premium": ["UTG", "MP", "CO", "BTN"],
    "Strong": ["UTG", "MP", "CO", "BTN"],
    "Medium Pairs": ["UTG", "MP", "CO", "BTN"],
    "Suited Broadways": ["MP", "CO", "BTN"],
    "Offsuit Broadways": ["CO", "BTN"],
    "Suited Connectors": ["CO", "BTN"],
    "Small Pairs": ["CO", "BTN"],
    "Suited Ax": ["CO", "BTN"],
    "Suited One-Gappers": ["BTN"],
    "Suited Low": ["BTN"],
    "Trash": []
}

REASONS = {
    "Premium": "премиумная рука, которая доминирует над диапазонами оппонентов",
    "Strong": "сильная рука с высоким шоудаун-эквити",
    "Medium Pairs": "средняя пара, которая хорошо играет на сет-майнинг или как оверпара",
    "Suited Broadways": "одномастные бродвеи, которые хорошо попадают во флеши и топ-пары",
    "Offsuit Broadways": "разномастные бродвеи, которые сильны топ-парой, но уязвимы без позиции",
    "Suited Connectors": "одномастные коннекторы, которые нуждаются в позиции для реализации эквити",
    "Small Pairs": "мелкие пары, играемые исключительно на сет-майнинг",
    "Suited Ax": "одномастные тузы, имеющие потенциал на натсовый флеш",
    "Suited One-Gappers": "одномастные гэпперы, имеющие потенциал на стриты и флеши",
    "Suited Low": "мелкие одномастные карты с низким эквити",
    "Trash": "мусорная рука без потенциала"
}

POS_STR = {
    "UTG": "ранняя (самая невыгодная, требуется тайтовый диапазон)",
    "MP": "средняя",
    "CO": "поздняя (Кат-офф)",
    "BTN": "самая поздняя (Баттон, лучшая позиция, можно играть лузово)"
}

SYSTEM3 = (
    "Ты эксперт по Техасскому Холдему (6-max NLH Cash). "
    "Сейчас префлоп, банк не открыт (все сбросили до тебя). Твоё решение: Рейз 2.5bb или Фолд. "
    "Правило: играй тайтово из ранних позиций (UTG) и лузово из поздних (BTN). "
    "Факты (позиция, категория руки) посчитаны движком. "
    "В <thought> кратко объясни, почему эта категория руки подходит или не подходит для данной позиции. "
    "В <action> выдай ровно 'Рейз 2.5bb' или 'Фолд'."
)

def build(pos, h, cat, gold):
    q = (f"NLH 6-max cash. Твоя позиция: {pos}.\n"
         f"Твоя рука: {fmt(h)}.\n"
         f"Факты от движка: Категория руки = {cat}.\n"
         f"Действие до тебя: Все сбросили (Unopened pot).\n"
         f"Твоё решение?")

    if gold == "raise":
        thought = f"{cat} — это {REASONS[cat]}. Позиция {pos} ({POS_STR[pos]}) позволяет открыть эту руку рейзом."
        act = "Рейз 2.5bb"
    else:
        thought = f"{cat} — это {REASONS[cat]}. Позиция {pos} ({POS_STR[pos]}) требует более тайтовой игры, эта рука недостаточно сильна для открытия."
        act = "Фолд"

    ans = f"<thought>\n{thought}\n</thought>\n<action>{act}</action>"
    return {"messages":[{"role":"system","content":SYSTEM3},
                       {"role":"user","content":q},
                       {"role":"assistant","content":ans}],
            "gold":gold, "pos":pos, "cat":cat}

def gen_preflop(need, cap, seen):
    out = []
    for _ in range(cap):
        if not any(v > 0 for v in need.values()): break
        pos = random.choice([p for p in POS_STR if any(need.get((p, c), 0) > 0 for c in CHART)])
        h = random.sample(FULL, 2)
        cat = categorize(h)
        if need.get((pos, cat), 0) <= 0: continue
        
        gold = "raise" if pos in CHART[cat] else "fold"
        ex = build(pos, h, cat, gold)
        key = ex["messages"][1]["content"]
        if key in seen: continue
        
        seen.add(key); need[(pos, cat)] -= 1; out.append(ex)
    return out

seen = set(); sets = {}
POSITIONS = ["UTG", "MP", "CO", "BTN"]
for name, per, cap in [("train", 8, 10000), ("dev", 4, 4000), ("test", 4, 4000)]:
    need = {(p, c): per for p in POSITIONS for c in CHART}
    rows = gen_preflop(need, cap, seen)
    random.shuffle(rows); sets[name] = rows
    print(f"{name}: {len(rows)}")
    with open(HERE/f"{name}_preflop.jsonl","w",encoding="utf-8") as f:
        for ex in rows:
            row = {"messages":ex["messages"]} if name=="train" else ex
            f.write(json.dumps(row, ensure_ascii=False)+"\n")

print("\nПример:\n", sets["train"][0]["messages"][1]["content"], "\n", sets["train"][0]["messages"][2]["content"])