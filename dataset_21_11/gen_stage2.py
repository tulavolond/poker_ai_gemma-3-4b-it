# gen_stage1.py — ступень 1: лучшая 5 из 7, сравнение рук, сплиты
import itertools, json, random
from treys import Card, Evaluator

random.seed(42)
EV = Evaluator()
RANKS = "23456789TJQKA"; RV = {r: i + 2 for i, r in enumerate(RANKS)}
FULL = [Card.new(r + s) for r in RANKS for s in "shdc"]
SUIT = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
GEN = {14:"тузов",13:"королей",12:"дам",11:"валетов",10:"десяток",9:"девяток",8:"восьмёрок",7:"семёрок",6:"шестёрок",5:"пятёрок",4:"четвёрок",3:"троек",2:"двоек"}
NOM = {14:"тузы",13:"короли",12:"дамы",11:"валеты",10:"десятки",9:"девятки",8:"восьмёрки",7:"семёрки",6:"шестёрки",5:"пятёрки",4:"четвёрки",3:"тройки",2:"двойки"}
NAME = {0:"старшая карта",1:"пара",2:"две пары",3:"сет",4:"стрит",5:"флеш",6:"фулл-хаус",7:"каре",8:"стрит-флеш"}
CH = {v: r for r, v in RV.items()}

def rs_of(c):                      # treys-карта -> (число 2..14, буква, масть)
    s = Card.int_to_str(c); return RV[s[0]], s[0], s[1]
def dr(v): return "10" if CH[v] == "T" else CH[v]
def disp(c):                       # нотация из твоего примера Г: "10♠"
    _, r, s = rs_of(c); return ("10" if r == "T" else r) + SUIT[s]
def fmt(cards): return " ".join(disp(c) for c in cards)

def classify5(five):               # ключ: больше = сильнее
    info = [rs_of(c) for c in five]
    vals = sorted([v for v, _, _ in info], reverse=True)
    suits = {s for _, _, s in info}
    cnt = {}
    for v in vals: cnt[v] = cnt.get(v, 0) + 1
    g = sorted(cnt.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    flush = len(suits) == 1
    u = sorted(set(vals), reverse=True); shi = 0
    if len(u) == 5:
        if u[0] - u[4] == 4: shi = u[0]
        elif u == [14, 5, 4, 3, 2]: shi = 5          # «колесо» A-2-3-4-5
    if flush and shi: return (8, shi)
    if g[0][1] == 4: return (7, g[0][0], g[1][0])
    if g[0][1] == 3 and g[1][1] == 2: return (6, g[0][0], g[1][0])
    if flush: return (5, *vals)
    if shi: return (4, shi)
    if g[0][1] == 3: return (3, g[0][0], *[v for v in vals if v != g[0][0]])
    if g[0][1] == 2 and g[1][1] == 2:
        hi, lo = g[0][0], g[1][0]
        return (2, hi, lo, [v for v in vals if v not in (hi, lo)][0])
    if g[0][1] == 2: return (1, g[0][0], *[v for v in vals if v != g[0][0]])
    return (0, *vals)

def best5(seven):                  # MAX: ключ больше = рука сильнее
    return max(itertools.combinations(seven, 5), key=classify5)

def describe(k):
    c = k[0]
    if c == 8: return "роял-флеш (10-J-Q-K-A одной масти)" if k[1] == 14 else f"стрит-флеш до {dr(k[1])}"
    if c == 7: return f"каре {GEN[k[1]]}, кикер {dr(k[2])}"
    if c == 6: return f"фулл-хаус: тройка {GEN[k[1]]} и пара {GEN[k[2]]}"
    if c == 5: return f"флеш, старшие карты {' '.join(dr(v) for v in k[1:])}"
    if c == 4: return "стрит до 5 (A-2-3-4-5, «колесо»)" if k[1] == 5 else f"стрит до {dr(k[1])}"
    if c == 3: return f"сет {GEN[k[1]]}, кикеры {dr(k[2])}, {dr(k[3])}"
    if c == 2: return f"две пары: {NOM[k[1]]} и {NOM[k[2]]}, кикер {dr(k[3])}"
    if c == 1: return f"пара {GEN[k[1]]}, кикеры {', '.join(dr(v) for v in k[2:])}"
    return f"старшая карта {dr(k[1])} (далее {', '.join(dr(v) for v in k[2:])})"

def compare_text(k1, k2):
    if k1[0] != k2[0]:
        hi, lo = (k1, k2) if k1[0] > k2[0] else (k2, k1)
        return f"Категория «{NAME[hi[0]]}» сильнее категории «{NAME[lo[0]]}»."
    if k1 == k2:
        return "Лучшие пятёрки полностью идентичны по старшинству — банк делится поровну, масти не сравниваются."
    for a, b in zip(k1[1:], k2[1:]):
        if a != b:
            w, l = (a, b) if a > b else (b, a)
            return f"Категории равны («{NAME[k1[0]]}»). Первое различие при сравнении по порядку: {dr(w)} старше {dr(l)}."

# --- самотесты (падают сразу, если логика сломана) ---
_b = [Card.new(x) for x in ["As", "Kd", "8c", "4h", "2s"]]
assert classify5(best5([Card.new("Ah"), Card.new("Qs")] + _b)) == (1, 14, 13, 12, 8)  # твой пример Б
assert classify5(best5([Card.new(x) for x in ["As","2d","3c","4h","5s","9d","9h"]])) == (4, 5)
assert classify5([Card.new(x) for x in ["As","Ks","9s","7s","3s"]]) > classify5([Card.new(x) for x in ["9s","8h","7d","6c","5s"]])

# --- три типа раздач ---
def deal_random():
    d = FULL[:]; random.shuffle(d); return d[:5], d[5:7], d[7:9]
def deal_kicker():                 # у обоих пара одного ранга — решают кикеры
    R = random.choice(range(2, 15))
    rc = [c for c in FULL if rs_of(c)[0] == R]; random.shuffle(rc)
    rest = [c for c in FULL if rs_of(c)[0] != R]; random.shuffle(rest)
    return rest[:5], [rc[0], rest[5]], [rc[1], rest[6]]
def deal_split():                  # одинаковые ранги, разные масти — частые сплиты
    r1, r2 = random.sample(range(2, 15), 2)
    c1 = [c for c in FULL if rs_of(c)[0] == r1]; c2 = [c for c in FULL if rs_of(c)[0] == r2]
    random.shuffle(c1); random.shuffle(c2)
    h1, h2 = [c1[0], c2[0]], [c1[1], c2[1]]
    rest = [c for c in FULL if c not in h1 + h2]; random.shuffle(rest)
    return rest[:5], h1, h2

def make(kind):
    for _ in range(500):
        b, h1, h2 = {"rnd": deal_random, "kick": deal_kicker, "split": deal_split}[kind]()
        f1, f2 = best5(h1 + b), best5(h2 + b)
        k1, k2 = classify5(f1), classify5(f2)
        if kind == "kick" and not (k1[0] == k2[0] == 1 and k1[1] == k2[1] and k1 != k2): continue
        if kind == "split" and k1 != k2: continue
        s1, s2 = EV.evaluate(b, h1), EV.evaluate(b, h2)      # treys: меньше = сильнее
        assert (k1 > k2) == (s1 < s2) and (k1 == k2) == (s1 == s2), "РАСХОЖДЕНИЕ с treys!"
        gold = "1" if s1 < s2 else "2" if s2 < s1 else "split"
        five1 = fmt(sorted(f1, key=lambda c: -rs_of(c)[0])); five2 = fmt(sorted(f2, key=lambda c: -rs_of(c)[0]))
        rule = random.choice(["Правило: комбинация — ровно 5 лучших карт из 7 (2 руки + 5 доски).",
                              "Правило 5 карт: собираю лучшую пятёрку из 7 доступных."])
        q = random.choice([
            f"На столе: {fmt(b)}. У Игрока 1: {fmt(h1)}. У Игрока 2: {fmt(h2)}. Кто выигрывает банк и почему?",
            f"Доска: {fmt(b)}. Рука Игрока 1: {fmt(h1)}, Игрока 2: {fmt(h2)}. Победитель или ничья? Объясни."])
        verdict = (f"Ничья (Split Pot). {compare_text(k1, k2)}" if gold == "split"
                   else f"Победитель: Игрок {gold}. {compare_text(k1, k2)}")
        ans = (f"<thought>\n1. {rule}\n2. Игрок 1: {five1} → {describe(k1)}.\n"
               f"3. Игрок 2: {five2} → {describe(k2)}.\n4. {compare_text(k1, k2)}\n</thought>\n"
               f"<definition>\n{verdict}\n</definition>")
        return {"messages": [
            {"role": "system", "content": "Ты эксперт по Техасскому Холдему. Твои ответы должны строго соответствовать официальным правилам оценки комбинаций."},
            {"role": "user", "content": q}, {"role": "assistant", "content": ans}], "gold": gold}
    raise RuntimeError(f"не удалось сгенерировать тип {kind}")

items, seen = [], set()
# После пилота ожидай заметный рост; если после <80% — меняешь в генераторе квоты (40,32,32) на (150,120,120),
# (epochи те же 3) и получаешь ступень 1 в полном объёме (~25–40 мин обучения).
for kind, n in [("rnd", 150), ("kick", 120), ("split", 120)]:
    while sum(1 for x in items if x.get("kind") == kind) < n:
        ex = make(kind)
        key = ex["messages"][1]["content"]
        if key in seen: continue
        seen.add(key); ex["kind"] = kind; items.append(ex)
random.shuffle(items)
train, test = items[:80], items[80:]
with open("./dataset_21_11/stage2_train.jsonl", "w", encoding="utf-8") as f:
    for ex in train: f.write(json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n")
with open("./dataset_21_11/stage2_eval.jsonl", "w", encoding="utf-8") as f:
    for ex in test: f.write(json.dumps({"messages": ex["messages"], "gold": ex["gold"]}, ensure_ascii=False) + "\n")
from collections import Counter
print("train:", len(train), "| eval:", len(test), "| исходы eval:", dict(Counter(x["gold"] for x in test)))
print("\nПример:\n", train[0]["messages"][1]["content"], "\n", train[0]["messages"][2]["content"])