# v6.py — скан в промпт, модель только применяет правила сравнения
import json, random, re
from collections import Counter
from pathlib import Path
from gen_v5 import (Card, EV, SR, NAME, rs, dr, fmt, analyze, cmp_line,
                    deal_random, deal_ff, ok, CMP_TRAIN)

HERE = Path(__file__).resolve().parent
random.seed(31337)

RANKS = "23456789TJQKA"; RV = {r: i + 2 for i, r in enumerate(RANKS)}
FULL = [Card.new(r + s) for r in RANKS for s in "shdc"]

SYSTEM6 = ("Ты эксперт по Техасскому Холдему. Комбинация — 5 лучших карт из 7. "
           "Старшинство: стрит-флеш > каре > фулл-хаус > флеш > стрит > сет > две пары > пара > старшая карта. "
           "Скан карт УЖЕ посчитан и дан в вопросе: масти, подряд, повторы, пятёрка, категория. "
           "Пятёрка записана в порядке сравнения: сначала группы, затем кикеры через |. "
           "Твоя задача: сравнить категории; при равенстве сравнивать позиции слева направо до первого различия. "
           "Масти между собой не старшируют. При полном равенстве — ничья. "
           "В <thought> запиши сравнение, в <definition> — итог.")

def five_ordered(k, best):
    vals = [rs(c)[0] for c in best]
    if k[0] in (4, 8):                                   # стрит: окно по убыванию, туз младший для колеса
        order = [5,4,3,2,14] if k[1] == 5 else list(range(k[1], k[1]-5, -1))
        return " ".join(dr(v) for v in order)
    if k[0] in (0, 5):                                   # старшая карта / флеш: просто по убыванию
        return " ".join(dr(v) for v in sorted(vals, reverse=True))
    gs = sorted(Counter(vals).items(), key=lambda t: (t[1], t[0]), reverse=True)
    groups = [r for r, n in gs if n >= 2 for _ in range(n)]
    kick = sorted([r for r, n in gs if n == 1], reverse=True)
    s = " ".join(dr(v) for v in groups)
    return s + (" | " + " ".join(dr(v) for v in kick) if kick else "")

def scan6(tag, a):
    st = " ".join(f"{SR[s]}{a['cnt'].get(s,0)}" for s in "shdc")
    fl = f"флеш {SR[a['fsuit']]}" if a['fsuit'] else "флеш нет"
    run = a['run']
    rt = f"подряд {len(run)}" + (f" ({'-'.join(dr(v) for v in run)})" if len(run) >= 2 else "")
    gr = ", ".join(f"{dr(r)}×{n}" for r, n in a['groups']) or "нет"
    cat = NAME[a['key'][0]] + (f" до {dr(a['key'][1])}" if a['key'][0] in (4,8) else "")
    return f"{tag}: масти {st} → {fl} | {rt} | повторы {gr} | пятёрка {five_ordered(a['key'], a['best'])} → {cat}"

def build_item(b, h1, h2, a1, a2, gold, bucket, profile):
    k1, k2 = a1['key'], a2['key']
    q = (f"Доска: {fmt(b)}. Игрок 1: {fmt(h1)}. Игрок 2: {fmt(h2)}.\n"
         f"Скан: {scan6('И1', a1)}\n{scan6('И2', a2)}\n"
         f"Кто выигрывает банк или ничья?")
    verdict = "Ничья (Split Pot)." if gold == "split" else f"Победитель: Игрок {gold}."
    ans = f"<thought>\n{cmp_line(k1,k2)}\n</thought>\n<definition>\n{verdict}\n</definition>"
    return {"messages":[{"role":"system","content":SYSTEM6},{"role":"user","content":q},
                        {"role":"assistant","content":ans}],
            "gold":gold,"task":"compare","bucket":bucket,"profile":profile,"cat1":k1[0],"cat2":k2[0]}

# def make6(bucket):
#     for _ in range(40000):
#         b, h1, h2 = deal_ff() if bucket == "flush_flush" else deal_random()
#         a1, a2 = analyze(b+h1), analyze(b+h2)
#         if not ok(bucket, a1, a2): continue
#         k1, k2 = a1['key'], a2['key']
#         s1, s2 = EV.evaluate(b, h1), EV.evaluate(b, h2)
#         assert (k1>k2)==(s1<s2) and (k1==k2)==(s1==s2), "treys mismatch"
#         gold = "1" if s1<s2 else "2" if s2<s1 else "split"
#         return build_item(b,h1,h2,a1,a2,gold,bucket,"natural" if bucket=="nat" else "stress")
#     raise RuntimeError(bucket)

def make6(bucket):
    for _ in range(40000):
        if bucket == "flush_flush":  b, h1, h2 = deal_ff()
        elif bucket == "quads_full": b, h1, h2 = deal_quads_full()
        else:                        b, h1, h2 = deal_random()
        a1, a2 = analyze(b+h1), analyze(b+h2)
        if not ok6(bucket, a1, a2): continue
        k1, k2 = a1['key'], a2['key']
        s1, s2 = EV.evaluate(b, h1), EV.evaluate(b, h2)
        assert (k1>k2)==(s1<s2) and (k1==k2)==(s1==s2), "treys mismatch"
        gold = "1" if s1<s2 else "2" if s2<s1 else "split"
        return build_item(b,h1,h2,a1,a2,gold,bucket,"natural" if bucket=="nat" else "stress")
    raise RuntimeError(bucket)

def ok6(kind, a1, a2):
    c1, c2 = a1['key'][0], a2['key'][0]
    if kind == "flush_straight": return (c1 == 5 and c2 == 4) or (c1 == 4 and c2 == 5)
    if kind == "quads_full":     return (c1 == 7 and c2 == 6) or (c1 == 6 and c2 == 7)
    if kind == "full_flush":     return (c1 == 6 and c2 == 5) or (c1 == 5 and c2 == 6)
    # if kind == "pair_deep3": return c1==c2==1 and k1[1]==k2[1] and k1[2]==k2[2] and k1[3]==k2[3] and k1[4]!=k2[4]
    # CMP6: full_flush 10 → 25, добавить ("pair_deep3", 20)
    return ok(kind, a1, a2)

def deal_quads_full():          # каре редкое — конструируем, иначе не нагенерируем
    rr = random.choice(range(2, 15))
    ss = random.choice([r for r in range(2, 15) if r != rr])
    rc = [c for c in FULL if rs(c)[0] == rr]; random.shuffle(rc)
    sc = [c for c in FULL if rs(c)[0] == ss]; random.shuffle(sc)
    rest = [c for c in FULL if rs(c)[0] not in (rr, ss)]; random.shuffle(rest)
    return rc[:2] + sc[:2] + rest[:1], [sc[2], rest[1]], rc[2:4]

# --- конвертация существующих экзаменов: те же раздачи, новый формат ---
CARD_RE = re.compile(r"(10|[2-9JQKA])([♠♥♦♣])")
BACK = {"♠":"s","♥":"h","♦":"d","♣":"c"}
def convert(fin, fout):
    rows = [json.loads(l) for l in open(HERE/fin, encoding="utf-8")]
    out = []
    for ex in rows:
        if ex["task"] != "compare": continue
        c = [Card.new(("T" if r=="10" else r)+BACK[s]) for r,s in CARD_RE.findall(ex["messages"][1]["content"])]
        assert len(c) == 9, ex["messages"][1]["content"]
        b, h1, h2 = c[:5], c[5:7], c[7:9]
        a1, a2 = analyze(b+h1), analyze(b+h2)
        assert a1['key'][0]==ex["cat1"] and a2['key'][0]==ex["cat2"], "категории разошлись"
        s1, s2 = EV.evaluate(b,h1), EV.evaluate(b,h2)
        gold = "1" if s1<s2 else "2" if s2<s1 else "split"
        assert gold == ex["gold"], "gold разошёлся"
        out.append(build_item(b,h1,h2,a1,a2,gold,ex["bucket"],ex["profile"]))
    with open(HERE/fout,"w",encoding="utf-8") as f:
        for ex in out: f.write(json.dumps(ex, ensure_ascii=False)+"\n")
    print(f"{fout}: {len(out)} (из {fin}, раздачи те же)")

seen, train = set(), []
# for bucket, n in CMP_TRAIN:
#     got = 0
#     while got < n:
#         ex = make6(bucket); key = ex["messages"][1]["content"]
#         if key in seen: continue
#         seen.add(key); train.append(ex); got += 1
# random.shuffle(train)
# with open(HERE/"train_v6.jsonl","w",encoding="utf-8") as f:
#     for ex in train: f.write(json.dumps({"messages":ex["messages"]}, ensure_ascii=False)+"\n")
# print(f"train_v6: {len(train)}")
# convert("dev.jsonl", "dev_v6.jsonl")
# convert("test.jsonl", "test_v6.jsonl")
# print("\nПример:\n", train[0]["messages"][1]["content"], "\n", train[0]["messages"][2]["content"])

# print("="*80)
CMP6 = CMP_TRAIN + [("flush_straight",20), ("quads_full",15), ("full_flush",10)]   # 445
STRESS6 = ["same_tb","split","straight_vs","flush_flush","twopair_tb","pair_deep",
           "wheel","broadway","near_straight","near_flush","flush_straight","quads_full","full_flush"]

# train-цикл: for bucket, n in CMP6:
if __name__ == "__main__":
    seen |= {json.loads(l)["messages"][1]["content"] for l in open(HERE/"dev_v6.jsonl", encoding="utf-8")}
    test61 = []
    for bucket, n in [("nat",40)] + [(k,4) for k in STRESS6]:
        got = 0
        while got < n:
            ex = make6(bucket); key = ex["messages"][1]["content"]
            if key in seen: continue
            seen.add(key); test61.append(ex); got += 1
    random.shuffle(test61)
    with open(HERE/"test_v61.jsonl","w",encoding="utf-8") as f:
        for ex in test61: f.write(json.dumps(ex, ensure_ascii=False)+"\n")
    print("test_v61:", len(test61))

