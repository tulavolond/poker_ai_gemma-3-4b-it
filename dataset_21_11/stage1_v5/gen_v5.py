# gen_v5.py — pip install treys  — генератор со сканом
import itertools, json, random
from collections import Counter
from pathlib import Path
from treys import Card, Evaluator

HERE = Path(__file__).resolve().parent
random.seed(2026)
EV = Evaluator()
RANKS = "23456789TJQKA"; RV = {r: i+2 for i, r in enumerate(RANKS)}
FULL = [Card.new(r+s) for r in RANKS for s in "shdc"]
SR = {"s":"♠","h":"♥","d":"♦","c":"♣"}
RCH = {1:"A", **{i+2: r for i, r in enumerate(RANKS)}}; RCH[10] = "10"
NAME = {0:"старшая карта",1:"пара",2:"две пары",3:"сет",4:"стрит",5:"флеш",6:"фулл-хаус",7:"каре",8:"стрит-флеш"}
POS = {0:["карта1","карта2","карта3","карта4","карта5"],1:["пара","кикер1","кикер2","кикер3"],
       2:["старшая пара","младшая пара","кикер"],3:["сет","кикер1","кикер2"],4:["старшая"],
       5:["карта1","карта2","карта3","карта4","карта5"],6:["тройка","пара"],7:["каре","кикер"],8:["старшая"]}
# SYSTEM = ("Ты эксперт по Техасскому Холдему. Комбинация — ровно 5 лучших карт из 7. "
#           "Старшинство: стрит-флеш > каре > фулл-хаус > флеш > стрит > сет > две пары > пара > старшая карта. "
#           "В <thought> выполни скан: масти, ранги, повторы, пятёрка. В <definition> дай итог.")

SYSTEM = ("Ты эксперт по Техасскому Холдему. Комбинация — ровно 5 лучших карт из 7 (2 руки + 5 доски). "
          "Флеш = 5 карт ОДНОЙ масти; 4 одной масти — не флеш; смешивать масти нельзя. "
          "Стрит = 5 карт ПОДРЯД по рангу, масти любые; 4 подряд — не стрит. "
          "Туз: старший (10-J-Q-K-A) или младший (A-2-3-4-5). Масти между собой не старшируют. "
          "Старшинство: стрит-флеш > каре > фулл-хаус > флеш > стрит > сет > две пары > пара > старшая карта. "
          "В <thought> выполни скан: масти, ранги, повторы, пятёрка. В <definition> дай итог.")

RULES = [
 ("Какие бывают масти и старшируют ли они друг друга?", "Перечислить масти и зафиксировать, что силы у них нет.",
  "Четыре масти: ♠ пики, ♥ черви, ♦ бубны, ♣ трефы. Масти НЕ старшируют друг друга: при равных рангах банк делится поровну."),
 ("Что такое флеш? Хватит ли четырёх карт одной масти?", "Флеш требует ровно пять карт одной масти; четыре — только дро.",
  "Флеш — 5 карт одной масти. 4 карты одной масти — не флеш. Смешивать разные масти в флеш нельзя."),
 ("Можно ли собрать флеш из карт доски и руки вместе?", "Флеш собирается из любых 5 карт среди 7 доступных.",
  "Да. Например, 3 пики на доске + 2 пики в руке = флеш. Важно только, чтобы все 5 карт были одной масти."),
 ("Что такое стрит? Хватит ли четырёх карт подряд?", "Стрит — ровно пять последовательных рангов, масти не важны.",
  "Стрит — 5 карт подряд по рангу, масти любые. 4 карты подряд — не стрит."),
 ("Как туз работает в стрите?", "Туз может быть и старшим, и младшим, но через угол не соединяется.",
  "Туз старший: 10-J-Q-K-A (Бродвей). Туз младший: A-2-3-4-5 (колесо, стрит до 5). K-A-2-3-4 стритом не является."),
 ("Сколько карт в комбинации и откуда они берутся?", "Правило пяти из семи.",
  "Ровно 5 лучших карт из 7 доступных: 2 свои + 5 на доске. Лишние карты не участвуют."),
 ("Что старше: стрит или флеш?", "Сверить по иерархии категорий.",
  "Флеш старше стрита. Полный порядок: стрит-флеш > каре > фулл-хаус > флеш > стрит > сет > две пары > пара > старшая карта."),
 ("Как сравнивать два флеша?", "Флеши сравниваются по картам от старшей к младшей.",
  "Сравниваем старшую карту, при равенстве — вторую, затем третью, четвёртую, пятую. Масть не влияет на силу."),
 ("Как сравнивать два стрита?", "Стрит определяется старшей картой окна.",
  "Стрит до A (Бродвей) — самый старший, стрит до 5 (колесо) — самый младший. Масти роли не играют."),
 ("Что происходит при полностью равных комбинациях?", "Все пять карт совпадают по рангам.",
  "Ничья (Split Pot): банк делится поровну. Масти не сравниваются и победителя не определяют."),
]

def make_five(kind, want):
    for _ in range(40000):
        d = FULL[:]; random.shuffle(d); five = d[:5]
        a = analyze(five)
        has = a['fsuit'] is not None if kind == "flush" else len(a['run']) >= 5
        near = a['maxsuit'] == 4 if kind == "flush" else len(a['run']) == 4
        if want == "pos" and not has: continue
        if want == "near" and (has or not near): continue
        if want == "easy" and (has or near): continue
        srt = sorted(five, key=lambda c: -rs(c)[0])
        if kind == "flush":
            q = f"Пять карт: {fmt(five)}. Это флеш?"
            th = ("масти " + " ".join(f"{SR[s]}{a['cnt'].get(s,0)}" for s in "shdc")
                  + f" → " + ("все 5 одной масти → флеш есть" if has else "не все одной масти → флеша нет"))
            df = f"Флеш: {fmt(srt)}." if has else "Флеша нет."
        else:
            run = a['run']
            rt = f"подряд {len(run)}" + (f" ({'-'.join(dr(v) for v in run)})" if len(run) >= 2 else "")
            q = f"Пять карт: {fmt(five)}. Это стрит? Укажи окно."
            th = f"ранги {' '.join(dr(v) for v in a['ranks'])} → {rt} → " + ("стрит есть" if has else "стрита нет")
            df = f"Стрит до {dr(run[-1])}: {'-'.join(dr(v) for v in run[-5:])}." if has else "Стрита нет."
        return {"messages":[{"role":"system","content":SYSTEM},{"role":"user","content":q},
                            {"role":"assistant","content":f"<thought>\n{th}\n</thought>\n<definition>\n{df}\n</definition>"}],
                "gold":"yes" if has else "no", "task":kind, "bucket":f"five_{kind}_{want}", "profile":"micro"}
    raise RuntimeError(f"five {kind}/{want}")

def rule_examples():
    return [{"messages":[{"role":"system","content":SYSTEM},{"role":"user","content":q},
                         {"role":"assistant","content":f"<thought>\n{t}\n</thought>\n<definition>\n{d}\n</definition>"}],
             "gold":"rule","task":"rule","bucket":"rules","profile":"rules"} for q,t,d in RULES]

def rs(c):
    s = Card.int_to_str(c); return RV[s[0]], s[1]
def dr(v): return RCH[v]
def disp(c):
    v, s = rs(c); return dr(v) + SR[s]
def fmt(cs): return " ".join(disp(c) for c in cs)

def classify5(five):
    vals = sorted([rs(c)[0] for c in five], reverse=True); suits = {rs(c)[1] for c in five}
    g = sorted(Counter(vals).items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    flush = len(suits) == 1; u = sorted(set(vals), reverse=True); shi = 0
    if len(u) == 5:
        if u[0]-u[4] == 4: shi = u[0]
        elif u == [14,5,4,3,2]: shi = 5
    if flush and shi: return (8, shi)
    if g[0][1] == 4: return (7, g[0][0], g[1][0])
    if g[0][1] == 3 and g[1][1] == 2: return (6, g[0][0], g[1][0])
    if flush: return (5, *vals)
    if shi: return (4, shi)
    if g[0][1] == 3: return (3, g[0][0], *[v for v in vals if v != g[0][0]])
    if g[0][1] == 2 and g[1][1] == 2:
        hi, lo = g[0][0], g[1][0]; return (2, hi, lo, [v for v in vals if v not in (hi, lo)][0])
    if g[0][1] == 2: return (1, g[0][0], *[v for v in vals if v != g[0][0]])
    return (0, *vals)

def longest_run(vals):
    u = set(vals)
    if 14 in u: u.add(1)                      # туз может быть и единицей
    best = []; cur = []
    for v in sorted(u):
        cur = cur + [v] if cur and v == cur[-1]+1 else [v]
        if len(cur) >= len(best): best = cur[:]
    return best

def analyze(seven):
    subs = list(itertools.combinations(seven, 5)); keys = [classify5(s) for s in subs]
    bi = max(range(len(keys)), key=lambda i: keys[i])
    cnt = Counter(rs(c)[1] for c in seven); maxsuit = max(cnt.values())
    fsuit = next((s for s, n in cnt.items() if n >= 5), None)
    franks = sorted([rs(c)[0] for c in seven if rs(c)[1] == fsuit], reverse=True)[:5] if fsuit else []
    run = longest_run([rs(c)[0] for c in seven])
    # самопроверка скана независимым перебором всех 21 пятёрки
    assert (maxsuit >= 5) == any(x[0] in (5,8) for x in keys), "flush scan"
    assert (len(run) >= 5) == any(x[0] in (4,8) for x in keys), "straight scan"
    if len(run) >= 5:
        assert run[-1] == max(x[1] for x in keys if x[0] in (4,8)), "straight high"
    groups = sorted([(r,n) for r,n in Counter(rs(c)[0] for c in seven).items() if n >= 2],
                    key=lambda t: (t[1], t[0]), reverse=True)
    return dict(key=keys[bi], best=subs[bi], cnt=cnt, maxsuit=maxsuit, fsuit=fsuit, franks=franks,
                run=run, groups=groups, ranks=sorted([rs(c)[0] for c in seven], reverse=True))

def scan(tag, a):
    st = " ".join(f"{SR[s]}{a['cnt'].get(s,0)}" for s in "shdc")
    fl = f"флеш {SR[a['fsuit']]} ({' '.join(dr(v) for v in a['franks'])})" if a['fsuit'] else "флеш нет"
    rk = " ".join(dr(v) for v in a['ranks']); run = a['run']
    rt = f"подряд {len(run)}" + (f" ({'-'.join(dr(v) for v in run)})" if len(run) >= 2 else "")
    sr = f"стрит до {dr(run[-1])}" if len(run) >= 5 else "стрит нет"
    gr = ", ".join(f"{dr(r)}×{n}" for r, n in a['groups']) or "нет"
    five = " ".join(dr(v) for v in sorted([rs(c)[0] for c in a['best']], reverse=True))
    return f"{tag}: масти {st} → {fl} | ранги {rk} → {rt} → {sr} | повторы {gr} | пятёрка {five} → {NAME[a['key'][0]]}"

def cmp_line(k1, k2):
    if k1[0] != k2[0]:
        w = 1 if k1[0] > k2[0] else 2; hi, lo = (k1, k2) if w == 1 else (k2, k1)
        return f"Сравнение: {NAME[hi[0]]} > {NAME[lo[0]]} → И{w}."
    if k1 == k2: return "Сравнение: все позиции равны → ничья."
    parts = []
    for nm, a, b in zip(POS[k1[0]], k1[1:], k2[1:]):
        if a == b: parts.append(f"{nm} {dr(a)}={dr(b)}")
        else:
            w = 1 if a > b else 2
            parts.append(f"{nm} {dr(max(a,b))}>{dr(min(a,b))} → И{w}"); break
    return "Сравнение: " + " | ".join(parts) + "."

def deal_random():
    d = FULL[:]; random.shuffle(d); return d[:5], d[5:7], d[7:9]
def deal_ff():                                  # оба игрока с флешем одной масти
    s = random.choice("shdc"); sc = [c for c in FULL if rs(c)[1] == s]; random.shuffle(sc)
    ot = [c for c in FULL if rs(c)[1] != s]; random.shuffle(ot)
    return sc[:3] + ot[:2], sc[3:5], sc[5:7]

def ok(kind, a1, a2):
    k1, k2 = a1['key'], a2['key']; c1, c2 = k1[0], k2[0]
    if kind == "nat": return True
    if kind == "diff_cat": return c1 != c2
    if kind == "same_tb": return c1 == c2 and k1 != k2
    if kind == "split": return k1 == k2
    if kind == "straight_vs": return (c1 == 4 and c2 in (0,1,2,3)) or (c2 == 4 and c1 in (0,1,2,3))
    if kind == "flush_flush": return c1 == c2 == 5 and k1 != k2
    if kind == "twopair_tb": return c1 == c2 == 2 and k1[1] == k2[1] and k1[2] == k2[2] and k1 != k2
    if kind == "pair_deep": return c1 == c2 == 1 and k1[1] == k2[1] and k1[2] == k2[2] and k1[3] != k2[3]
    if kind == "wheel": return (c1 == 4 and k1[1] == 5) or (c2 == 4 and k2[1] == 5)
    if kind == "broadway": return (c1 == 4 and k1[1] == 14) or (c2 == 4 and k2[1] == 14)
    if kind == "near_straight": return c1 not in (4,8) and c2 not in (4,8) and (len(a1['run']) == 4 or len(a2['run']) == 4)
    if kind == "near_flush": return c1 not in (5,8) and c2 not in (5,8) and (a1['maxsuit'] == 4 or a2['maxsuit'] == 4)
    raise KeyError(kind)

def make_compare(bucket):
    for _ in range(40000):
        b, h1, h2 = deal_ff() if bucket == "flush_flush" else deal_random()
        a1, a2 = analyze(b+h1), analyze(b+h2)
        if not ok(bucket, a1, a2): continue
        k1, k2 = a1['key'], a2['key']
        s1, s2 = EV.evaluate(b, h1), EV.evaluate(b, h2)      # treys: меньше = сильнее
        assert (k1 > k2) == (s1 < s2) and (k1 == k2) == (s1 == s2), "treys mismatch"
        gold = "1" if s1 < s2 else "2" if s2 < s1 else "split"
        q = random.choice([
            f"На столе: {fmt(b)}. У Игрока 1: {fmt(h1)}. У Игрока 2: {fmt(h2)}. Кто выигрывает банк и почему?",
            f"Доска: {fmt(b)}. Рука Игрока 1: {fmt(h1)}, Игрока 2: {fmt(h2)}. Победитель или ничья? Объясни."])
        verdict = "Ничья (Split Pot)." if gold == "split" else f"Победитель: Игрок {gold}."
        ans = f"<thought>\n{scan('И1',a1)}\n{scan('И2',a2)}\n{cmp_line(k1,k2)}\n</thought>\n<definition>\n{verdict}\n</definition>"
        return {"messages":[{"role":"system","content":SYSTEM},{"role":"user","content":q},
                            {"role":"assistant","content":ans}],
                "gold":gold, "task":"compare", "bucket":bucket,
                "profile":"natural" if bucket == "nat" else "stress", "cat1":k1[0], "cat2":k2[0]}
    raise RuntimeError(f"не сгенерировался bucket {bucket}")

def make_micro(kind, want):
    for _ in range(40000):
        b, h, _ = deal_random(); a = analyze(b+h)
        has = a['fsuit'] is not None if kind == "flush" else len(a['run']) >= 5
        near = a['maxsuit'] == 4 if kind == "flush" else len(a['run']) == 4
        if want == "pos" and not has: continue
        if want == "near" and (has or not near): continue
        if want == "easy" and (has or near): continue
        if kind == "flush":
            q = f"Доска: {fmt(b)}. Рука: {fmt(h)}. Есть ли флеш? Если да — назови пять карт."
            th = ("масти " + " ".join(f"{SR[s]}{a['cnt'].get(s,0)}" for s in "shdc")
                  + f" → максимум {a['maxsuit']} → " + ("флеш есть" if a['fsuit'] else "флеша нет"))
            df = (f"Флеш: {fmt(sorted([c for c in b+h if rs(c)[1]==a['fsuit']], key=lambda c: -rs(c)[0])[:5])}."
                  if a['fsuit'] else "Флеша нет.")
        else:
            run = a['run']
            rt = f"подряд {len(run)}" + (f" ({'-'.join(dr(v) for v in run)})" if len(run) >= 2 else "")
            q = f"Доска: {fmt(b)}. Рука: {fmt(h)}. Есть ли стрит? Укажи окно."
            th = f"ранги {' '.join(dr(v) for v in a['ranks'])} → {rt} → " + ("стрит есть" if len(run) >= 5 else "стрита нет")
            df = f"Стрит до {dr(run[-1])}: {'-'.join(dr(v) for v in run[-5:])}." if len(run) >= 5 else "Стрита нет."
        return {"messages":[{"role":"system","content":SYSTEM},{"role":"user","content":q},
                            {"role":"assistant","content":f"<thought>\n{th}\n</thought>\n<definition>\n{df}\n</definition>"}],
                "gold":"yes" if has else "no", "task":kind, "bucket":f"{kind}_{want}", "profile":"micro"}
    raise RuntimeError(f"не сгенерировалась микро-задача {kind}/{want}")
# Было (сумма 400):
# CMP_TRAIN = [("nat",60),("diff_cat",40),("same_tb",40),("split",40),("straight_vs",50),
#              ("flush_flush",25),("twopair_tb",25),("pair_deep",25),("wheel",20),("broadway",20),
#              ("near_straight",30),("near_flush",25)]
# CMP_TRAIN = [("nat",30),("diff_cat",20),("same_tb",20),("split",20),("straight_vs",25),
#              ("flush_flush",13),("twopair_tb",13),("pair_deep",12),("wheel",10),("broadway",10),
#              ("near_straight",15),("near_flush",12)]   # сумма = 200
MICRO = [("flush","pos",50),("flush","near",25),("flush","easy",25),
         ("straight","pos",50),("straight","near",25),("straight","easy",25)]
STRESS = ["same_tb","split","straight_vs","flush_flush","twopair_tb","pair_deep",
          "wheel","broadway","near_straight","near_flush"]
# TRAIN_PLAN = MICRO + [("compare",k,n) for k,n in CMP_TRAIN]
# EXAM_PLAN = [(k,w,n//5) for k,w,n in MICRO] + [("compare","nat",40)] + [("compare",k,4) for k in STRESS]
# FIVE = [(f"five_{k}", w, 20) for k in ("flush","straight") for w in ("pos","near","easy")]
FIVE = [(f"five_{k}", w, n) for k in ("flush","straight")
        for w, n in (("pos",30),("near",15),("easy",15))]      # 50/50 да/нет

CMP_TRAIN = [("nat",60),("diff_cat",40),("same_tb",40),("split",40),("straight_vs",50),
             ("flush_flush",25),("twopair_tb",25),("pair_deep",25),("wheel",20),("broadway",20),
             ("near_straight",30),("near_flush",25)]            # сумма 400, как в v4
TRAIN_PLAN = MICRO + FIVE + [("compare",k,n) for k,n in CMP_TRAIN]     # 200 + 120 + 400 = 720
EXAM_PLAN  = [(k,w,n//5) for k,w,n in MICRO] + \
             [(f"five_{k}", w, n) for k in ("flush","straight") for w,n in (("pos",4),("near",3),("easy",3))] + \
             [("compare","nat",40)] + [("compare",k,4) for k in STRESS]   # 140
             
def build(plan, seen):
    out = []

    for task, sub, n in plan:
        got = 0
        while got < n:
            # ex = make_compare(sub) if task == "compare" else make_micro(task, sub)
            if task == "compare": ex = make_compare(sub)
            elif task.startswith("five_"): ex = make_five(task[5:], sub)
            else: ex = make_micro(task, sub)

            key = ex["messages"][1]["content"]
            if key in seen: continue
            seen.add(key); out.append(ex); got += 1
            
    random.shuffle(out); return out

seen = set()
for fname in ("dev.jsonl", "test.jsonl"):
    p = HERE / fname
    if p.exists():
        seen |= {json.loads(l)["messages"][1]["content"] for l in open(p, encoding="utf-8")}

train = rule_examples() + build(TRAIN_PLAN, seen)
random.shuffle(train)
with open(HERE / "train.jsonl", "w", encoding="utf-8") as f:
    for ex in train:
        f.write(json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n")

n_cmp = sum(n for _, n in CMP_TRAIN)
print(f"train {len(train)} (азбука 10 + микро 200 + five 120 + compare {n_cmp}) | dev/test не тронуты")

# seen = set()
# train, dev, test = build(TRAIN_PLAN, seen), build(EXAM_PLAN, seen), build(EXAM_PLAN, seen)
# train = rule_examples() + train
# for name, data in [("train",train),("dev",dev),("test",test)]:
#     with open(HERE/f"{name}.jsonl","w",encoding="utf-8") as f:
#         for ex in data:
#             row = {"messages": ex["messages"]} if name == "train" else ex
#             f.write(json.dumps(row, ensure_ascii=False)+"\n")
# print(f"train {len(train)} | dev {len(dev)} | test {len(test)} | пересечений нет (общий seen)")
# print("категории в train:", dict(sorted(Counter(ex.get("cat1") for ex in train if ex["task"]=="compare").items())))
# from transformers import AutoTokenizer
# tk = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
# lens = [len(tk(tk.apply_chat_template(ex["messages"], tokenize=False), add_special_tokens=False)["input_ids"]) for ex in train]
# print(f"токенов: max={max(lens)} avg={sum(lens)//len(lens)}  (MAX_LEN=1024 должен покрывать max)")
# print("\nПример:\n", train[0]["messages"][1]["content"], "\n", train[0]["messages"][2]["content"])

