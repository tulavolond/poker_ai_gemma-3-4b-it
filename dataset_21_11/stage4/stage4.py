# stage4.py — Префлоп против рейза (Facing a Raise / 3-bet pots)
import json, random
from pathlib import Path
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
    suited = (s1 == s2)
    pair = (r1 == r2)
    gap = r1 - r2

    if pair:
        if r1 >= 12: return "Premium"      # QQ+
        if r1 >= 10: return "Strong"       # JJ, TT
        if r1 >= 6: return "Medium Pairs"  # 99-66
        return "Small Pairs"               # 55-22

    if r1 == 14: # Туз
        if r2 >= 13: return "Premium" if suited else "Strong" # AKs / AKo
        if r2 >= 12 and suited: return "Strong"  # AQs
        if r2 >= 12: return "Offsuit Broadways"  # AQo
        if r2 >= 11 and suited: return "Strong"  # AJs
        if r2 >= 11: return "Offsuit Broadways"  # AJo
        if r2 >= 10 and suited: return "Suited Broadways" # ATs
        if r2 >= 10: return "Offsuit Broadways"  # ATo
        if suited: return "Suited Ax"            # A5s, A2s...
        return "Trash"                           # A9o - A2o

    if r1 >= 11 and r2 >= 10: # Бродвеи
        if suited: return "Suited Broadways"     # KQs, QJs, JTs
        return "Offsuit Broadways"               # KQo, QJo, JTo

    if suited and gap == 1 and r1 <= 10: return "Suited Connectors" # T9s ... 54s
    if suited and r1 <= 10: return "Trash" # Остальные одномастные -> Trash против рейза
    
    return "Trash"

# Матрица GTO решений (F = Фолд, C = Колл, 3 = 3-бет)
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

SYSTEM4 = (
    "Ты эксперт по Техасскому Холдему (6-max NLH Cash). "
    "Сейчас префлоп. Игрок до тебя сделал рейз 2.5bb, все остальные сбросили. "
    "Ты должен решить: Фолд, Колл 2.5bb или 3-бет (ререйз до 9bb). "
    "Правила GTO: "
    "1. С премиум-руками (QQ+, AK) всегда делай 3-бет для вэлью. "
    "2. С руками, требующими позиции (мелкие пары, коннекторы), коллируй только если ты в позиции (BTN, CO) и фолди без позиции (SB, BB). "
    "3. Против ранних позиций (UTG, MP) играй тайтово. Против поздних (CO, BTN) расширяй диапазон 3-бета и колла. "
    "Факты (позиции, категория руки) посчитаны движком. "
    "В <thought> кратко объясни логику решения на основе позиции и категории. "
    "В <action> выдай ровно 'Фолд', 'Колл 2.5bb' или '3-бет 9bb'."
)

def build(raiser_pos, hero_pos, h, cat, action_code):
    q = (f"NLH 6-max cash. Эффективный стек: 100bb.\n"
         f"Действие: Игрок на {raiser_pos} сделал рейз 2.5bb. Все сбросили до тебя.\n"
         f"Твоя позиция: {hero_pos}.\n"
         f"Твоя рука: {fmt(h)}.\n"
         f"Факты от движка: Категория руки = {cat}.\n"
         f"Твоё решение?")

    if action_code == "3":
        act_str = "3-бет 9bb"
        thought = f"{cat} — это сильная рука. Против рейза с {raiser_pos} из {hero_pos} мы делаем 3-бет для вэлью и изоляции широкого диапазона."
    elif action_code == "C":
        act_str = "Колл 2.5bb"
        thought = f"{cat} — рука с хорошим потенциалом. Позиция {hero_pos} против {raiser_pos} дает имплайд-оддсы для реализации эквити без риска 3-бета."
    else:
        act_str = "Фолд"
        thought = f"{cat} — недостаточно сильна для 3-бета или колла против рейза с {raiser_pos} из {hero_pos}. Фолдим, чтобы не играть доминируемыми руками."

    ans = f"<thought>\n{thought}\n</thought>\n<action>{act_str}</action>"
    gold = "fold" if action_code == "F" else ("call" if action_code == "C" else "raise")
    
    return {"messages":[{"role":"system","content":SYSTEM4}, {"role":"user","content":q}, {"role":"assistant","content":ans}],
            "gold":gold, "raiser":raiser_pos, "hero":hero_pos, "cat":cat}

def gen_facing(need, cap, seen):
    out = []
    for _ in range(cap):
        if not any(v > 0 for v in need.values()): break
        live = [k for k, v in need.items() if v > 0]
        cat, r_pos, h_pos = random.choice(live)
        h = random.sample(FULL, 2)
        if categorize(h) != cat: continue
        action_code = RULES[cat][r_pos][h_pos]
        ex = build(r_pos, h_pos, h, cat, action_code)
        key = ex["messages"][1]["content"]
        if key in seen: continue
        seen.add(key); need[(cat, r_pos, h_pos)] -= 1; out.append(ex)
    return out

seen = set(); sets = {}
plan = {(cat, r_pos, h_pos): 0 for cat in RULES for r_pos in RULES[cat] for h_pos in RULES[cat][r_pos]}

for name, per, cap in [("train", 6, 10000), ("dev", 2, 4000), ("test", 2, 4000)]:
    need = {k: per for k in plan}
    rows = gen_facing(need, cap, seen)
    random.shuffle(rows); sets[name] = rows
    print(f"{name}: {len(rows)}")
    with open(HERE/f"{name}_facing.jsonl","w",encoding="utf-8") as f:
        for ex in rows:
            row = {"messages":ex["messages"]} if name=="train" else ex
            f.write(json.dumps(row, ensure_ascii=False)+"\n")

print("\nПример:\n", sets["train"][0]["messages"][1]["content"], "\n", sets["train"][0]["messages"][2]["content"])