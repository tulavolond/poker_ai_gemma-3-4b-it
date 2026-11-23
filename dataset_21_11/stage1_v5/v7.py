# v7.py — полное покрытие пространства сравнений
# python ./dataset_21_11/stage1_v5/v7.py

# python ./dataset_21_11/stage1_v5/train_lora.py      # ~730 примеров × 2 ≈ 30 мин
# python ./dataset_21_11/stage1_v5/eval_fast.py ./dataset_21_11/stage1_v5/qwen25-v7-lora dev_v6.jsonl
# python ./dataset_21_11/stage1_v5/eval_fast.py ./dataset_21_11/stage1_v5/qwen25-v7-lora test_v7.jsonl
import json, random
from pathlib import Path
from gen_v5 import Card, EV, FULL, RANKS, rs, analyze, deal_random, deal_ff, CMP_TRAIN
from v6 import build_item, ok6, deal_quads_full

HERE = Path(__file__).resolve().parent
random.seed(777)
NPOS = {0:5, 1:4, 2:3, 3:3, 4:1, 5:5, 6:2, 7:2, 8:1}
IMPOSSIBLE = {(7,1)}                      # каре разного ранга у обоих — невозможно
HIER = [(1,0),(2,1),(3,2),(4,3),(5,4),(6,5),(7,6),(8,5),(8,6)]

def tb_ok(c, p, k1, k2):
    return k1[0]==k2[0]==c and k1[1:p]==k2[1:p] and k1[p]!=k2[p]

def finish(b,h1,h2,a1,a2,bucket,profile):
    k1,k2 = a1['key'], a2['key']
    s1,s2 = EV.evaluate(b,h1), EV.evaluate(b,h2)
    assert (k1>k2)==(s1<s2) and (k1==k2)==(s1==s2), "treys mismatch"
    gold = "1" if s1<s2 else "2" if s2<s1 else "split"
    return build_item(b,h1,h2,a1,a2,gold,bucket,profile)

# --- конструктивные дилеры для редких клеток ---
def deal_flush_tb():                      # оба флеш, общая доска из 4 мастей
    s = random.choice("shdc")
    sc = [c for c in FULL if rs(c)[1]==s]; random.shuffle(sc)
    ot = [c for c in FULL if rs(c)[1]!=s]; random.shuffle(ot)
    return sc[:4]+ot[:1], [sc[4],ot[1]], [sc[5],ot[2]]

def deal_fh(p):
    R,S = random.sample(range(2,15),2)
    rc = [c for c in FULL if rs(c)[0]==R]; random.shuffle(rc)
    sc = [c for c in FULL if rs(c)[0]==S]; random.shuffle(sc)
    if p == 1:                            # разные тройки
        rest = [c for c in FULL if rs(c)[0] not in (R,S)]; random.shuffle(rest)
        return rc[:2]+sc[:2]+rest[:1], [rc[2],rest[1]], [sc[2],rest[2]]
    T = random.choice([r for r in range(2,15) if r not in (R,S)])   # та же тройка, разные пары
    tc = [c for c in FULL if rs(c)[0]==T]; random.shuffle(tc)
    rest = [c for c in FULL if rs(c)[0] not in (R,S,T)]; random.shuffle(rest)
    return rc[:3]+rest[:2], sc[:2], tc[:2]

def deal_quads_tb():                      # все 4 карты ранга на доске → каре у обоих
    R = random.choice(range(2,15))
    rc = [c for c in FULL if rs(c)[0]==R]
    rest = [c for c in FULL if rs(c)[0]!=R]; random.shuffle(rest)
    return rc+rest[:1], rest[1:3], rest[3:5]

def deal_sf_tb():
    s = random.choice("shdc"); lo = random.choice(range(2,11))
    mk = lambda r: Card.new(RANKS[r-2]+s)
    ot = [c for c in FULL if rs(c)[1]!=s]; random.shuffle(ot)
    low = Card.new("A"+s) if lo == 2 else mk(lo-1)      # колесо A-2-3-4-5
    return [mk(r) for r in range(lo,lo+4)]+ot[:1], [mk(lo+4),ot[1]], [low,ot[2]]

def deal_full_flush():                    # фулл-хаус против флеша
    R = random.choice(range(2,15))
    rc = [c for c in FULL if rs(c)[0]==R]; random.shuffle(rc)
    sp = [c for c in FULL if rs(c)[1]=="s" and rs(c)[0]!=R]; random.shuffle(sp)
    b = rc[:2]+sp[:3]; pr = rs(sp[0])[0]
    pc = [c for c in FULL if rs(c)[0]==pr and c not in b]
    return b, [rc[2],pc[0]], sp[3:5]

def deal_sf_flush():
    s = random.choice("shdc"); lo = random.choice(range(2,11))
    mk = lambda r: Card.new(RANKS[r-2]+s)
    ot = [c for c in FULL if rs(c)[1]!=s]; random.shuffle(ot)
    b = [mk(r) for r in range(lo,lo+3)]+ot[:2]; h1 = [mk(lo+3),mk(lo+4)]
    sc = [c for c in FULL if rs(c)[1]==s and c not in b and c not in h1]; random.shuffle(sc)
    return b, h1, sc[:2]

def deal_sf_fh():
    s = random.choice("shdc"); lo = random.choice(range(2,10))
    mk = lambda r: Card.new(RANKS[r-2]+s)
    rc = [c for c in FULL if rs(c)[0]==lo   and rs(c)[1]!=s]
    tc = [c for c in FULL if rs(c)[0]==lo+1 and rs(c)[1]!=s]
    return [mk(r) for r in range(lo,lo+3)]+rc[:2], [mk(lo+3),mk(lo+4)], tc[:2]

def tb_dealer(c,p):
    if c==5: return deal_flush_tb()
    if c==6: return deal_fh(p)
    if c==7: return deal_quads_tb()
    if c==8: return deal_sf_tb()
    return deal_random()

def make_tb(c,p):
    for _ in range(60000):
        b,h1,h2 = tb_dealer(c,p); a1,a2 = analyze(b+h1), analyze(b+h2)
        if tb_ok(c,p,a1['key'],a2['key']): return finish(b,h1,h2,a1,a2,f"tb{c}_{p}","stress")
    return None

def make_hier(hi,lo):
    for _ in range(60000):
        if   (hi,lo)==(7,6): b,h1,h2 = deal_quads_full()
        elif (hi,lo)==(6,5): b,h1,h2 = deal_full_flush()
        elif (hi,lo)==(8,5): b,h1,h2 = deal_sf_flush()
        elif (hi,lo)==(8,6): b,h1,h2 = deal_sf_fh()
        else:                b,h1,h2 = deal_random()
        a1,a2 = analyze(b+h1), analyze(b+h2)
        if {a1['key'][0], a2['key'][0]} == {hi,lo}:
            return finish(b,h1,h2,a1,a2,f"hier{hi}_{lo}","stress")
    return None

def ok7(kind,a1,a2):
    k1,k2 = a1['key'],a2['key']
    if kind == "pair_deep3":
        return k1[0]==k2[0]==1 and k1[1]==k2[1] and k1[2]==k2[2] and k1[3]==k2[3] and k1[4]!=k2[4]
    return ok6(kind,a1,a2)

def make7(bucket):
    if bucket.startswith("tb"):   c,p = map(int, bucket[2:].split("_")); return make_tb(c,p)
    if bucket.startswith("hier"): h,l = map(int, bucket[4:].split("_")); return make_hier(h,l)
    for _ in range(60000):
        if   bucket=="flush_flush": b,h1,h2 = deal_ff()
        elif bucket=="quads_full":  b,h1,h2 = deal_quads_full()
        elif bucket=="full_flush":  b,h1,h2 = deal_full_flush()
        else:                       b,h1,h2 = deal_random()
        a1,a2 = analyze(b+h1), analyze(b+h2)
        if ok7(bucket,a1,a2): return finish(b,h1,h2,a1,a2,bucket,"natural" if bucket=="nat" else "stress")
    return None

CELLS  = [f"tb{c}_{p}" for c in NPOS for p in range(1,NPOS[c]+1) if (c,p) not in IMPOSSIBLE]
CELLS += [f"hier{h}_{l}" for h,l in HIER]
BASE   = CMP_TRAIN + [("flush_straight",20),("quads_full",15),("full_flush",25),("pair_deep3",20)]
PLAN   = BASE + [(c,8) for c in CELLS]

def build(plan, seen):
    out, missing = [], []
    for bucket,n in plan:
        got = 0
        while got < n:
            ex = make7(bucket)
            if ex is None: missing.append(bucket); break
            key = ex["messages"][1]["content"]
            if key in seen: continue
            seen.add(key); out.append(ex); got += 1
    random.shuffle(out); return out, missing

seen = set()
for f in ("dev_v6.jsonl","test_v61.jsonl"):
    p = HERE/f
    if p.exists(): seen |= {json.loads(l)["messages"][1]["content"] for l in open(p,encoding="utf-8")}
train, miss1 = build(PLAN, seen)
with open(HERE/"train_v7.jsonl","w",encoding="utf-8") as f:
    for ex in train: f.write(json.dumps({"messages":ex["messages"]},ensure_ascii=False)+"\n")

TEST_PLAN = [("nat",40),("split",6)] + [(c,2) for c in CELLS]
test, miss2 = build(TEST_PLAN, seen)
with open(HERE/"test_v7.jsonl","w",encoding="utf-8") as f:
    for ex in test: f.write(json.dumps(ex,ensure_ascii=False)+"\n")

from collections import Counter
cov = Counter(ex["bucket"] for ex in train)
empty = [c for c in CELLS if cov[c]==0]
print(f"train_v7: {len(train)} | test_v7: {len(test)}")
print(f"ПОКРЫТИЕ: {len(CELLS)-len(empty)}/{len(CELLS)} клеток | пустые: {empty}")
print("не удалось сгенерировать:", sorted(set(miss1+miss2)))
print("минимум примеров на клетку:", min(cov[c] for c in CELLS if cov[c]))