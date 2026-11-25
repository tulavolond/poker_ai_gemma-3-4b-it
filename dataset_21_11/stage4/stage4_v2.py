# stage4_v2.py — Префлоп против рейза (Сбалансированный + Стили розыгрыша)
import json, random
from pathlib import Path
from collections import Counter
from treys import Card

HERE = Path(__file__).resolve().parent
random.seed(404)

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
    suited = (s1 == s2); pair = (r1 == r2); gap = r1 - r2
    if pair:
        if r1 >= 12: return "Premium"
        if r1 >= 10: return "Strong"
        if r1 >= 6: return "Medium Pairs"
        return "Small Pairs"
    if r1 == 14: 
        if r2 >= 13: return "Premium" if suited else "Strong"
        if r2 >= 12 and suited: return "Strong"
        if r2 >= 12: return "Offsuit Broadways"
        if r2 >= 11 and suited: return "Strong"
        if r2 >= 11: return "Offsuit Broadways"
        if r2 >= 10 and suited: return "Suited Broadways"
        if r2 >= 10: return "Offsuit Broadways"
        if suited: return "Suited Ax"
        return "Trash"
    if r1 >= 11 and r2 >= 10: 
        return "Suited Broadways" if suited else "Offsuit Broadways"
    if suited and gap == 1 and r1 <= 10: return "Suited Connectors"
    if suited and r1 <= 10: return "Trash"
    return "Trash"

RULES = {
    "Premium": {"UTG": {"MP":"3","CO":"3","BTN":"3","SB":"3","BB":"3"}, "MP": {"CO":"3","BTN":"3","SB":"3","BB":"3"}, "CO": {"BTN":"3","SB":"3","BB":"3"}, "BTN": {"SB":"3","BB":"3"}},
    "Strong":  {"UTG": {"MP":"C","CO":"C","BTN":"C","SB":"F","BB":"C"}, "MP": {"CO":"C","BTN":"C","SB":"F","BB":"C"}, "CO": {"BTN":"3","SB":"3","BB":"3"}, "BTN": {"SB":"3","BB":"3"}},
    "Medium Pairs": {"UTG": {"MP":"F","CO":"C","BTN":"C","SB":"F","BB":"F"}, "MP": {"CO":"C","BTN":"C","SB":"F","BB":"F"}, "CO": {"BTN":"C","SB":"F","BB":"C"}, "BTN": {"SB":"F","BB":"C"}},
    "Small Pairs": {"UTG": {"MP":"F","CO":"C","BTN":"C","SB":"F","BB":"F"}, "MP": {"CO":"F","BTN":"C","SB":"F","BB":"F"}, "CO": {"BTN":"C","SB":"F","BB":"C"}, "BTN": {"SB":"F","BB":"C"}},
    "Suited Broadways": {"UTG": {"MP":"F","CO":"C","BTN":"C","SB":"F","BB":"C"}, "MP": {"CO":"F","BTN":"C","SB":"F","BB":"C"}, "CO": {"BTN":"C","SB":"F","BB":"C"}, "BTN": {"SB":"F","BB":"C"}},
    "Offsuit Broadways": {"UTG": {"MP":"F","CO":"F","BTN":"F","SB":"F","BB":"F"}, "MP": {"CO":"F","BTN":"F","SB":"F","BB":"F"}, "CO": {"BTN":"C","SB":"F","BB":"F"}, "BTN": {"SB":"F","BB":"F"}},
    "Suited Connectors": {"UTG": {"MP":"F","CO":"F","BTN":"F","SB":"F","BB":"F"}, "MP": {"CO":"F","BTN":"F","SB":"F","BB":"F"}, "CO": {"BTN":"C","SB":"F","BB":"F"}, "BTN": {"SB":"F","BB":"F"}},
    "Suited Ax": {"UTG": {"MP":"F","CO":"F","BTN":"F","SB":"F","BB":"F"}, "MP": {"CO":"F","BTN":"F","SB":"F","BB":"F"}, "CO": {"BTN":"F","SB":"F","BB":"F"}, "BTN": {"SB":"F","BB":"F"}},
    "Trash": {"UTG": {"MP":"F","CO":"F","BTN":"F","SB":"F","BB":"F"}, "MP": {"CO":"F","BTN":"F","SB":"F","BB":"F"}, "CO": {"BTN":"F","SB":"F","BB":"F"}, "BTN": {"SB":"F","BB":"F"}}
}

# СЛОВАРЬ СТИЛЕЙ (Ключ к успеху!)
STYLES = {
    ("Premium", "3"): "Вэлью 3-бет",
    ("Strong", "3"): "Вэлью 3-бет",
    ("Strong", "C"): "Позиционный розыгрыш",
    ("Medium Pairs", "C"): "Сет-майнинг",
    ("Small Pairs", "C"): "Сет-майнинг",
    ("Suited Broadways", "C"): "Позиционный розыгрыш",
    ("Suited Connectors", "C"): "Позиционный розыгрыш",
    ("Offsuit Broadways", "C"): "Позиционный розыгрыш",
}

SYSTEM4 = (
    "Ты эксперт по Техасскому Холдему (6-max NLH Cash). "
    "Сейчас префлоп. Игрок до тебя сделал рейз 2.5bb. "
    "Ты должен решить: Фолд, Колл 2.5bb или 3-бет (ререйз до 9bb). "
    "Правила GTO на основе фактов движка: "
    "1. Если стиль 'Вэлью 3-бет' — всегда делай 3-бет. "
    "2. Если стиль 'Сет-майнинг' или 'Позиционный розыгрыш' — всегда КОЛЛИРУЙ (Колл 2.5bb). Не фолди и не рейзи эти руки. "
    "3. Если стиль 'Недостаточно эквити' — всегда ФОЛДИ. "
    "В <thought> кратко объясни логику на основе категории и стиля. "
    "В <action> выдай ровно 'Фолд', 'Колл 2.5bb' или '3-бет 9bb'."
)

def build(raiser_pos, hero_pos, h, cat, action_code):
    style = STYLES.get((cat, action_code), "Недостаточно эквити")
    q = (f"NLH 6-max cash. Эффективный стек: 100bb.\n"
         f"Действие: Игрок на {raiser_pos} сделал рейз 2.5bb. Все сбросили до тебя.\n"
         f"Твоя позиция: {hero_pos}.\n"
         f"Твоя рука: {fmt(h)}.\n"
         f"Факты от движка: Категория = {cat}, Стиль розыгрыша = {style}.\n"
         f"Твоё решение?")

    if action_code == "3":
        act_str, thought = "3-бет 9bb", f"{cat}. Стиль: {style}. Делаем 3-бет для вэлью."
    elif action_code == "C":
        act_str, thought = "Колл 2.5bb", f"{cat}. Стиль: {style}. Коллируем для реализации эквити."
    else:
        act_str, thought = "Фолд", f"{cat}. Стиль: {style}. Фолдим из-за риска доминирования."

    ans = f"<thought>\n{thought}\n</thought>\n<action>{act_str}</action>"
    gold = "fold" if action_code == "F" else ("call" if action_code == "C" else "raise")
    return {"messages":[{"role":"system","content":SYSTEM4}, {"role":"user","content":q}, {"role":"assistant","content":ans}],
            "gold":gold, "raiser":raiser_pos, "hero":hero_pos, "cat":cat}

templates = []
for cat in RULES:
    for r_pos in RULES[cat]:
        for h_pos in RULES[cat][r_pos]:
            templates.append((cat, r_pos, h_pos, RULES[cat][r_pos][h_pos]))

fold_temps = [t for t in templates if t[3] == "F"]
call_temps = [t for t in templates if t[3] == "C"]
raise_temps = [t for t in templates if t[3] == "3"]

def gen_balanced(temps, target_count, seen):
    out = []
    repeats = max(1, (target_count // len(temps)) + 2)
    for _ in range(repeats):
        for cat, r_pos, h_pos, action_code in temps:
            if len(out) >= target_count: break
            for _ in range(30):
                h = random.sample(FULL, 2)
                if categorize(h) == cat: break
            ex = build(r_pos, h_pos, h, cat, action_code)
            key = ex["messages"][1]["content"]
            if key not in seen:
                seen.add(key); out.append(ex)
        if len(out) >= target_count: break
    random.shuffle(out)
    return out[:target_count]

seen = set(); sets = {}
TARGET = 250 # По 250 на класс -> ~750 всего

for name, target in [("train", TARGET), ("dev", 60), ("test", 60)]:
    rows = []
    rows += gen_balanced(fold_temps, target, seen)
    rows += gen_balanced(call_temps, target, seen)
    rows += gen_balanced(raise_temps, target, seen)
    random.shuffle(rows)
    sets[name] = rows
    print(f"{name}: {len(rows)} | баланс: {dict(Counter(ex['gold'] for ex in rows))}")
    with open(HERE/f"{name}_facing_v2.jsonl","w",encoding="utf-8") as f:
        for ex in rows:
            row = {"messages":ex["messages"]} if name=="train" else ex
            f.write(json.dumps(row, ensure_ascii=False)+"\n")