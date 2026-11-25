# stage2.py — решения против ставки: движок считает, модель сравнивает
import itertools, json, random
from collections import Counter
from pathlib import Path
from treys import Card, Evaluator

HERE = Path(__file__).resolve().parent
random.seed(20261123)
EV = Evaluator()
RANKS = "23456789TJQKA"; RV = {r: i+2 for i, r in enumerate(RANKS)}
SUIT = {"♠":"s","♥":"h","♦":"d","♣":"c"}; SR = {v:k for k,v in SUIT.items()}
FULL = [Card.new(r+s) for r in RANKS for s in "shdc"]
NAME = {0:"старшая карта",1:"пара",2:"две пары",3:"сет",4:"стрит",5:"флеш",6:"фулл-хаус",7:"каре",8:"стрит-флеш"}

SYSTEM = ("Ты эксперт по Техасскому Холдему без лимита. Факты (комбинация, эквити, пот-оддсы) "
          "уже посчитаны движком и даны в вопросе — не пересчитывай их. "
          "Правило: эквити >= пот-оддсы -> Колл; эквити < пот-оддсы -> Фолд; при точном равенстве -> Колл. "
          "Имплайд-оддсы не учитываем (бонус 0). Рейзы и блефы не рассматриваются. "
          'В <thought> запиши сравнение двух чисел, в <action> — ровно "Фолд" или "Колл {размер}bb".')

def rs(c):
    s = Card.int_to_str(c); return RV[s[0]], s[1]
def disp(c):
    r, s = rs(c); return ("10" if r == 10 else RANKS[r-2]) + SR[s]
def fmt(cs): return " ".join(disp(c) for c in cs)

def classify5(five):
    vals = sorted([rs(c)[0] for c in five], reverse=True); suits = {rs(c)[1] for c in five}
    g = sorted(Counter(vals).items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    flush = len(suits)==1; u = sorted(set(vals), reverse=True); shi = 0
    if len(u)==5:
        if u[0]-u[4]==4: shi = u[0]
        elif u==[14,5,4,3,2]: shi = 5
    if flush and shi: return (8,shi)
    if g[0][1]==4: return (7,g[0][0],g[1][0])
    if g[0][1]==3 and g[1][1]==2: return (6,g[0][0],g[1][0])
    if flush: return (5,*vals)
    if shi: return (4,shi)
    if g[0][1]==3: return (3,g[0][0],*[v for v in vals if v!=g[0][0]])
    if g[0][1]==2 and g[1][1]==2:
        hi,lo = g[0][0],g[1][0]; return (2,hi,lo,[v for v in vals if v not in (hi,lo)][0])
    if g[0][1]==2: return (1,g[0][0],*[v for v in vals if v!=g[0][0]])
    return (0,*vals)
def cat(cards): return NAME[max(classify5(s) for s in itertools.combinations(cards,5))[0]]

def equity(h, v, b):                       # ТОЧНО, полным перебором
    known = set(h+v+b); rest = [x for x in FULL if x not in known]
    win = tie = n = 0
    for run in itertools.combinations(rest, 5-len(b)):
        fb = b + list(run); s1, s2 = EV.evaluate(fb,h), EV.evaluate(fb,v)
        n += 1; win += s1 < s2; tie += s1 == s2
    return (win + tie/2) / n

SIZES = [(0.25,"¼ банка"), (1/3,"⅓ банка"), (0.5,"½ банка"), (2/3,"⅔ банка"), (1.0,"банк"), (1.5,"1.5 банка")]
RELS = ["far_below","below","border","above","far_above"]
POTS = [6.0, 12.0, 24.0]
STREET_RU = {"flop":"флоп","turn":"тёрн","river":"ривер"}
IMPOSSIBLE = {("river",i,r) for i in range(6) for r in ("below","border","above")}

def rel_of(eq, odds):
    d = (eq-odds)*100
    if d < -10: return "far_below"
    if d < -2:  return "below"
    if d <= 2:  return "border"
    if d <= 10: return "above"
    return "far_above"

def deal(street):
    d = FULL[:]; random.shuffle(d)
    return d[:{"flop":3,"turn":4,"river":5}[street]], d[5:7], d[7:9]

def deal_tie():                            # одинаковые ранги, разные масти -> ничья на ривере
    r1, r2 = random.sample(range(2,15), 2)
    c1 = [c for c in FULL if rs(c)[0]==r1]; random.shuffle(c1)
    c2 = [c for c in FULL if rs(c)[0]==r2]; random.shuffle(c2)
    h, v = [c1[0],c2[0]], [c1[1],c2[1]]
    rest = [c for c in FULL if c not in h+v]; random.shuffle(rest)
    return rest[:5], h, v

def build(street, b, h, v, pot, bet, eq, odds, si, rel):
    gold = "call" if eq >= odds else "fold"
    if street == "flop":
        line = f"Effective стек {bet:g}bb. Оппонент идёт олл-ин {bet:g}bb в банк {pot:g}bb. Колл = {bet:g}bb."
    else:
        line = f"Банк {pot:g}bb. Оппонент ставит {bet:g}bb ({SIZES[si][1]}). Колл = {bet:g}bb."
    q = (f"NLH cash. Ты на BB, оппонент на BTN. Улица: {STREET_RU[street]}.\n"
         f"Твоя рука: {fmt(h)}. Доска: {fmt(b)}.\n{line}\n"
         f"Факты (посчитаны движком): комбинация — {cat(h+b)}; эквити = {eq*100:.2f}%; "
         f"пот-оддсы = {odds*100:.2f}%; имплайд-бонус = 0пп.\nТвоё решение?")
    sign = ">=" if gold == "call" else "<"
    why = "колл выгоден" if gold == "call" else "колл невыгоден"
    act = f"Колл {bet:g}bb" if gold == "call" else "Фолд"
    ans = f"<thought>\nЭквити {eq*100:.2f}% {sign} пот-оддсы {odds*100:.2f}% → {why}.\n</thought>\n<action>{act}</action>"
    return {"messages":[{"role":"system","content":SYSTEM},{"role":"user","content":q},
                        {"role":"assistant","content":ans}],
            "gold":gold,"task":"bet","street":street,"bucket":f"{street}_{si}_{rel}",
            "profile":"border" if rel=="border" else "clear","rel":rel}

def gen(street, need, cap, seen):
    out = []
    for _ in range(cap):
        if not any(n > 0 for n in need.values()): break
        live = [i for i in range(6) if any(need.get((i,r),0) > 0 for r in RELS)]
        si = random.choice(live)
        frac, _ = SIZES[si]; pot = random.choice(POTS); bet = pot*frac
        odds = bet/(pot + 2*bet)
        b,h,v = deal_tie() if (street=="river" and random.random()<0.35) else deal(street)
        eq = equity(h,v,b)
        if 0 < abs(eq-odds) < 0.0002: continue      # защита от округления на границе
        r = rel_of(eq,odds)
        if need.get((si,r),0) <= 0: continue
        ex = build(street,b,h,v,pot,bet,eq,odds,si,r)
        key = ex["messages"][1]["content"]
        if key in seen: continue
        seen.add(key); need[(si,r)] -= 1; out.append(ex)
    return out

def plan(per_cell):
    return {(st,i,r): per_cell for st in ("flop","turn","river")
            for i in range(6) for r in RELS if (st,i,r) not in IMPOSSIBLE}

seen = set(); sets = {}
for name, per, cap in [("train",4,6000), ("dev",2,4000), ("test",2,4000)]:
    need = plan(per); rows = []
    for st in ("turn","river","flop"):              # флоп самый дорогой — последним
        sub = {(k[1], k[2]): v for k, v in need.items() if k[0] == st}
        rows += gen(st, sub, cap, seen)
        for (i, r), v in sub.items():
            need[(st, i, r)] = v
    
    random.shuffle(rows); sets[name] = rows
    unfilled = [f"{k[0]}_{k[1]}_{k[2]}" for k,v in need.items() if v > 0]
    print(f"{name}: {len(rows)} | недозаполненные клетки: {unfilled}")
    with open(HERE/f"{name}_bet.jsonl","w",encoding="utf-8") as f:
        for ex in rows:
            row = {"messages":ex["messages"]} if name=="train" else ex
            f.write(json.dumps(row, ensure_ascii=False)+"\n")

c = Counter(ex["gold"] for ex in sets["train"])
cells = Counter(ex["bucket"] for ex in sets["train"])
print(f"баланс классов train: {dict(c)}")
print(f"покрытие: {len(cells)}/72 клетки | минимум на клетку: {min(cells.values())}")
print("\nПример:\n", sets["train"][0]["messages"][1]["content"], "\n", sets["train"][0]["messages"][2]["content"])