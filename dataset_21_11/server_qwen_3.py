# poker_ai.py — один файл: генерация раздач + движок фактов + модель + автопроверка
#
#   python ./dataset_21_11/server_qwen_3.py --adapter D:\py2026\poker2\dataset_21_11\qwen25-decisions-lora
#   python ./dataset_21_11/server_qwen_3.py --adapter ... --n 20     # прогнать 20 раздач подряд и показать статистику
#   python ./dataset_21_11/server_qwen_3.py --no-model               # только движок (мгновенно, проверить логику)

#   python ./dataset_21_11/server_qwen_3.py --adapter D:\py2026\poker2\dataset_21_11\qwen25-decisions-lora --n 30
#   python ./dataset_21_11/server_qwen_3.py --adapter D:\py2026\poker2\dataset_21_11\qwen25-decisions-lora

import argparse, itertools, os, random, re, time
from collections import Counter
from treys import Card, Evaluator

random.seed()
EV = Evaluator()

RANKS = "23456789TJQKA"; RV = {r: i+2 for i, r in enumerate(RANKS)}
SUIT = {"♠":"s","♥":"h","♦":"d","♣":"c"}; SR = {v:k for k,v in SUIT.items()}
FULL = [Card.new(r+s) for r in RANKS for s in "shdc"]
NAME = {0:"старшая карта",1:"пара",2:"две пары",3:"сет",4:"стрит",
        5:"флеш",6:"фулл-хаус",7:"каре",8:"стрит-флеш"}
ORDER = ["UTG","MP","CO","BTN","SB","BB"]
ORDER_POS = {
    "UTG": "ранняя позиция, первое слово на префлопе",
    "MP":  "средняя позиция",
    "CO":  "предпоследняя, справа от баттона",
    "BTN": "дилер, последняя позиция на постфлопе",
    "SB":  "малый блайнд, пол-блайнда вслепую",
    "BB":  "большой блайнд, полный блайнд вслепую",
}
STREET_RU = {"flop":"флоп","turn":"тёрн","river":"ривер"}

# ───────────────────────── базовые функции (как в stage2.py) ─────────────────────────

def rs(c):
    s = Card.int_to_str(c); return RV[s[0]], s[1]

def disp(c):
    r, s = rs(c); return ("10" if r == 10 else RANKS[r-2]) + SR[s]

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
        hi, lo = g[0][0], g[1][0]; return (2, hi, lo, *[v for v in vals if v not in (hi, lo)])
    if g[0][1] == 2: return (1, g[0][0], *[v for v in vals if v != g[0][0]])
    return (0, *vals)

def cat(cards): return NAME[max(classify5(s) for s in itertools.combinations(cards, 5))[0]]

def equity(h, v, b):                       # ТОЧНО, полным перебором — как при обучении
    known = set(h+v+b); rest = [x for x in FULL if x not in known]
    win = tie = n = 0
    for run in itertools.combinations(rest, 5-len(b)):
        fb = b + list(run); s1, s2 = EV.evaluate(fb, h), EV.evaluate(fb, v)
        n += 1; win += s1 < s2; tie += s1 == s2
    return (win + tie/2) / n

# ───────────────────────── категории руки (как в stage3/stage4) ─────────────────────────

def categorize(h, mode):
    r1, r2 = sorted([rs(h[0])[0], rs(h[1])[0]], reverse=True)
    suited = rs(h[0])[1] == rs(h[1])[1]
    gap = r1 - r2

    if r1 == r2:
        if mode == "rfi":
            if r1 >= 10: return "Premium"
            if r1 >= 6:  return "Medium Pairs"
            return "Small Pairs"
        else:
            if r1 >= 12: return "Premium"
            if r1 >= 10: return "Strong"
            if r1 >= 6:  return "Medium Pairs"
            return "Small Pairs"

    if r1 == 14:
        if mode == "rfi":
            if r2 >= 13 and suited: return "Premium"
            if r2 >= 12:            return "Strong"
            if r2 >= 10 and suited: return "Strong"
            if r2 >= 10:            return "Offsuit Broadways"
            if suited:              return "Suited Ax"
            return "Trash"
        else:
            if r2 >= 13:            return "Premium" if suited else "Strong"
            if r2 >= 12 and suited: return "Strong"
            if r2 >= 12:            return "Offsuit Broadways"
            if r2 >= 11 and suited: return "Strong"
            if r2 >= 11:            return "Offsuit Broadways"
            if r2 >= 10 and suited: return "Suited Broadways"
            if r2 >= 10:            return "Offsuit Broadways"
            if suited:              return "Suited Ax"
            return "Trash"

    if r1 >= 11 and r2 >= 10:
        return "Suited Broadways" if suited else "Offsuit Broadways"

    if suited and gap == 1 and r1 <= 10: return "Suited Connectors"
    if mode == "rfi":
        if suited and gap == 2 and r1 <= 10: return "Suited One-Gappers"
        if suited and r1 <= 10:              return "Suited Low"
    return "Trash"

CHART_RFI = {
    "Premium": ["UTG","MP","CO","BTN"], "Strong": ["UTG","MP","CO","BTN"],
    "Medium Pairs": ["UTG","MP","CO","BTN"], "Suited Broadways": ["MP","CO","BTN"],
    "Offsuit Broadways": ["CO","BTN"], "Suited Connectors": ["CO","BTN"],
    "Small Pairs": ["CO","BTN"], "Suited Ax": ["CO","BTN"],
    "Suited One-Gappers": ["BTN"], "Suited Low": ["BTN"], "Trash": [],
}

RULES = {
    "Premium": {"UTG":{"MP":"3","CO":"3","BTN":"3","SB":"3","BB":"3"},"MP":{"CO":"3","BTN":"3","SB":"3","BB":"3"},"CO":{"BTN":"3","SB":"3","BB":"3"},"BTN":{"SB":"3","BB":"3"}},
    "Strong": {"UTG":{"MP":"C","CO":"C","BTN":"C","SB":"F","BB":"C"},"MP":{"CO":"C","BTN":"C","SB":"F","BB":"C"},"CO":{"BTN":"3","SB":"3","BB":"3"},"BTN":{"SB":"3","BB":"3"}},
    "Medium Pairs": {"UTG":{"MP":"F","CO":"C","BTN":"C","SB":"F","BB":"F"},"MP":{"CO":"C","BTN":"C","SB":"F","BB":"F"},"CO":{"BTN":"C","SB":"F","BB":"C"},"BTN":{"SB":"F","BB":"C"}},
    "Small Pairs": {"UTG":{"MP":"F","CO":"C","BTN":"C","SB":"F","BB":"F"},"MP":{"CO":"F","BTN":"C","SB":"F","BB":"F"},"CO":{"BTN":"C","SB":"F","BB":"C"},"BTN":{"SB":"F","BB":"C"}},
    "Suited Broadways": {"UTG":{"MP":"F","CO":"C","BTN":"C","SB":"F","BB":"C"},"MP":{"CO":"F","BTN":"C","SB":"F","BB":"C"},"CO":{"BTN":"C","SB":"F","BB":"C"},"BTN":{"SB":"F","BB":"C"}},
    "Offsuit Broadways": {"UTG":{"MP":"F","CO":"F","BTN":"F","SB":"F","BB":"F"},"MP":{"CO":"F","BTN":"F","SB":"F","BB":"F"},"CO":{"BTN":"C","SB":"F","BB":"F"},"BTN":{"SB":"F","BB":"F"}},
    "Suited Connectors": {"UTG":{"MP":"F","CO":"F","BTN":"F","SB":"F","BB":"F"},"MP":{"CO":"F","BTN":"F","SB":"F","BB":"F"},"CO":{"BTN":"C","SB":"F","BB":"F"},"BTN":{"SB":"F","BB":"F"}},
    "Suited Ax": {"UTG":{"MP":"F","CO":"F","BTN":"F","SB":"F","BB":"F"},"MP":{"CO":"F","BTN":"F","SB":"F","BB":"F"},"CO":{"BTN":"F","SB":"F","BB":"F"},"BTN":{"SB":"F","BB":"F"}},
    "Trash": {"UTG":{"MP":"F","CO":"F","BTN":"F","SB":"F","BB":"F"},"MP":{"CO":"F","BTN":"F","SB":"F","BB":"F"},"CO":{"BTN":"F","SB":"F","BB":"F"},"BTN":{"SB":"F","BB":"F"}},
}

STYLES = {
    ("Premium","3"):"Вэлью 3-бет", ("Strong","3"):"Вэлью 3-бет",
    ("Strong","C"):"Позиционный розыгрыш",
    ("Medium Pairs","C"):"Сет-майнинг", ("Small Pairs","C"):"Сет-майнинг",
    ("Suited Broadways","C"):"Позиционный розыгрыш",
    ("Suited Connectors","C"):"Позиционный розыгрыш",
    ("Offsuit Broadways","C"):"Позиционный розыгрыш",
}

# ───────────────────────── системные промпты (как в датасетах) ─────────────────────────

SYS_BET = ('Ты эксперт по Техасскому Холдему без лимита. Факты (комбинация, эквити, пот-оддсы) '
           'уже посчитаны движком и даны в вопросе — не пересчитывай их. '
           'Правило: эквити >= пот-оддсы -> Колл; эквити < пот-оддсы -> Фолд; при точном равенстве -> Колл. '
           'Имплайд-оддсы не учитываем (бонус 0). Рейзы и блефы не рассматриваются. '
           'В <thought> запиши сравнение двух чисел, в <action> — ровно "Фолд" или "Колл {размер}bb".')

SYS_RFI = ("Ты эксперт по Техасскому Холдему (6-max NLH Cash). Сейчас префлоп, банк не открыт "
           "(все сбросили до тебя). Твоё решение: Рейз 2.5bb или Фолд. Правило: играй тайтово из ранних "
           "позиций (UTG) и лузово из поздних (BTN). Факты (позиция, категория руки) посчитаны движком. "
           "В <thought> кратко объясни, почему эта категория руки подходит или не подходит для данной позиции. "
           "В <action> выдай ровно 'Рейз 2.5bb' или 'Фолд'.")

SYS_FACING = ("Ты эксперт по Техасскому Холдему (6-max NLH Cash). Сейчас префлоп. Игрок до тебя сделал рейз 2.5bb. "
              "Ты должен решить: Фолд, Колл 2.5bb или 3-бет (ререйз до 9bb). Правила GTO на основе фактов движка: "
              "1. Если стиль 'Вэлью 3-бет' — всегда делай 3-бет. "
              "2. Если стиль 'Сет-майнинг' или 'Позиционный розыгрыш' — всегда КОЛЛИРУЙ (Колл 2.5bb). Не фолди и не рейзи. "
              "3. Если стиль 'Недостаточно эквити' — всегда ФОЛДИ. "
              "В <thought> кратко объясни логику на основе категории и стиля. "
              "В <action> выдай ровно 'Фолд', 'Колл 2.5bb' или '3-бет 9bb'.")

SYS_FREE = ("Ты эксперт по Техасскому Холдему (NLH Cash). Ставки перед тобой нет, доступны Чек или Бет. "
            "Факты (комбинация, эквити) посчитаны движком. С сильной рукой или сильным дро — Бет, иначе Чек. "
            "В <thought> краткий анализ, в <action> — ровно 'Чек' или 'Бет {размер}bb'.")

# ───────────────────────── генераторы раздач ─────────────────────────

POTS = [6.0, 12.0, 24.0]
SIZES = [(0.25,"¼ банка"),(1/3,"⅓ банка"),(0.5,"½ банка"),(2/3,"⅔ банка"),(1.0,"банк"),(1.5,"1.5 банка")]

def deal_street(street):
    d = FULL[:]; random.shuffle(d)
    return d[:{"flop":3,"turn":4,"river":5}[street]], d[5:7], d[7:9]

def gen_rfi():
    hero_pos = random.choice(["UTG", "MP", "CO", "BTN"])
    desc = ORDER_POS[hero_pos]
    d = FULL[:]; random.shuffle(d); h = d[:2]
    c = categorize(h, "rfi")
    gold = ("RAISE", 2.5) if hero_pos in CHART_RFI[c] else ("FOLD", 0)
    user = (f"NLH 6-max cash. Твоя позиция: {hero_pos} — {desc}.\n"
            f"Твоя рука: {fmt(h)}.\n"
            f"Факты от движка: Категория руки = {c}.\n"
            f"Действие до тебя: Все сбросили (Unopened pot).\n"
            f"Твоё решение?")
    return dict(mode="rfi", system=SYS_RFI, user=user, gold=gold,
                facts=[("рука", fmt(h)), ("позиция", f"{hero_pos} — {desc}"), ("категория", c)],
                trained=True)

def gen_facing():
    ri = random.randint(0, 3)
    raiser = ORDER[ri]            # имя по индексу, как было
    hero_pos = random.choice(ORDER[ri+1:])
    raiser_desc = ORDER_POS[raiser]       # описание рейзера
    hero_desc = ORDER_POS[hero_pos]       # описание героя
    d = FULL[:]; random.shuffle(d); h = d[:2]
    c = categorize(h, "facing")
    code = RULES[c][raiser][hero_pos]
    style = STYLES.get((c, code), "Недостаточно эквити")
    gold = {"3": ("RAISE", 9.0), "C": ("CALL", 2.5), "F": ("FOLD", 0)}[code]
    user = (f"NLH 6-max cash. Эффективный стек: 100bb.\n"
            f"Действие: Игрок на {raiser} ({raiser_desc}) сделал рейз 2.5bb. "
            f"Все сбросили до тебя.\n"
            f"Твоя позиция: {hero_pos} — {hero_desc}.\n"
            f"Твоя рука: {fmt(h)}.\n"
            f"Факты от движка: Категория = {c}, Стиль розыгрыша = {style}.\n"
            f"Твоё решение?")
    return dict(mode="facing", system=SYS_FACING, user=user, gold=gold,
                facts=[("рука", fmt(h)), ("рейзер", raiser), ("герой", hero_pos),
                       ("категория", c), ("стиль", style)],
                trained=True)

def gen_bet():
    for _ in range(200):
        street = random.choice(["flop","turn","river"])
        si = random.randint(0, 5)
        frac, size_ru = SIZES[si]
        pot = random.choice(POTS); bet = pot * frac
        odds = bet / (pot + 2*bet)
        b, h, v = deal_street(street)
        eq = equity(h, v, b)
        if 0 < abs(eq-odds) < 0.0002: continue
        gold = ("CALL", bet) if eq >= odds else ("FOLD", 0)
        if street == "flop":
            line = f"Effective стек {bet:g}bb. Оппонент идёт олл-ин {bet:g}bb в банк {pot:g}bb. Колл = {bet:g}bb."
        else:
            line = f"Банк {pot:g}bb. Оппонент ставит {bet:g}bb ({size_ru}). Колл = {bet:g}bb."
        user = (f"NLH cash. Ты на BB, оппонент на BTN. Улица: {STREET_RU[street]}.\n"
                f"Твоя рука: {fmt(h)}. Доска: {fmt(b)}.\n{line}\n"
                f"Факты (посчитаны движком): комбинация — {cat(h+b)}; эквити = {eq*100:.2f}%; "
                f"пот-оддсы = {odds*100:.2f}%; имплайд-бонус = 0пп.\nТвоё решение?")
        
        return dict(mode="bet", system=SYS_BET, user=user, gold=gold,
                    facts=[("рука", fmt(h)), ("борд", fmt(b)), ("улица", STREET_RU[street]),
                           ("банк", f"{pot:g}bb"), ("ставка", f"{bet:g}bb"),
                           ("комбинация", cat(h+b)),
                           ("эквити", f"{eq*100:.2f}%"), ("пот-оддсы", f"{odds*100:.2f}%")],
                    trained=True)
    raise RuntimeError("не смог сгенерировать раздачу")

def gen_free():
    street = random.choice(["flop","turn","river"])
    b, h, v = deal_street(street)
    pot = random.choice(POTS)
    eq = equity(h, v, b)
    gold = ("BET", round(pot*0.5, 1)) if eq >= 0.55 else ("CHECK", 0)
    user = (f"NLH cash. Ты на BTN. Улица: {STREET_RU[street]}.\n"
            f"Твоя рука: {fmt(h)}. Доска: {fmt(b)}.\nБанк {pot:g}bb. Ставки перед тобой нет.\n"
            f"Факты (посчитаны движком): комбинация — {cat(h+b)}; эквити = {eq*100:.2f}%.\nТвоё решение?")
    
    return dict(mode="free", system=SYS_FREE, user=user, gold=gold,
                facts=[("рука", fmt(h)), ("борд", fmt(b)), ("улица", STREET_RU[street]),
                       ("комбинация", cat(h+b)), ("эквити", f"{eq*100:.2f}%")],
                trained=False)

GENERATORS = {"r": (gen_rfi, "префлоп: открытие банка"),
              "v": (gen_facing, "префлоп: против рейза"),
              "b": (gen_bet, "постфлоп: против ставки"),
              "c": (gen_free, "постфлоп: чек/бет (вне обучения)")}

def gen_random():
    return random.choices([gen_bet, gen_facing, gen_rfi, gen_free],
                          weights=[40, 25, 25, 10])[0]()

# ───────────────────────── модель ─────────────────────────

def load_model(adapter, model_name):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria

    tok_src = adapter if adapter and os.path.isdir(adapter) else model_name
    tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    kw = dict(device_map={"": 0}, attn_implementation="sdpa", trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, **kw)

    if adapter and os.path.isdir(adapter):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        print(f"[MODEL] адаптер: {adapter}")
    else:
        print("[MODEL] базовая модель, без адаптера")

    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    class StopAtAction(StoppingCriteria):
        def __call__(self, input_ids, scores, **kw):
            tail = tok.decode(input_ids[0][-12:], skip_special_tokens=True)
            return "</action>" in tail or "</definition>" in tail

    stop = StopAtAction()

    def generate(system, user):
        text = tok.apply_chat_template(
            [{"role":"system","content":system},{"role":"user","content":user}],
            tokenize=False, add_generation_prompt=True)
        inp = tok(text, return_tensors="pt").to(model.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=180, do_sample=False,
                                 pad_token_id=tok.eos_token_id, stopping_criteria=[stop])
        ms = round((time.time()-t0)*1000)
        return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True), ms

    return generate

# ───────────────────────── парсер ответа ─────────────────────────

def parse_model(text):
    th = re.search(r"<thought>\s*([\s\S]*?)\s*</thought>", text, re.I)
    ac = (re.search(r"<action>\s*([\s\S]*?)\s*</action>", text, re.I)
          or re.search(r"<definition>\s*([\s\S]*?)\s*</definition>", text, re.I))
    thought = th.group(1).strip() if th else ""
    raw = ac.group(1).strip() if ac else text.strip()[:120]
    a = raw.lower()

    act = None
    if "фолд" in a or "fold" in a:                                        act = "FOLD"
    elif any(k in a for k in ("3-бет","3бет","3bet","рейз","raise")):     act = "RAISE"
    elif "колл" in a or "call" in a:                                     act = "CALL"
    elif "чек" in a or "check" in a:                                     act = "CHECK"
    elif "бет" in a or "bet" in a:                                       act = "BET"

    num = re.search(r"(\d+(?:[.,]\d+)?)", raw.replace("bb","").replace("$",""))
    amount = float(num.group(1).replace(",", ".")) if num else 0.0
    return act, amount, thought, raw

# ───────────────────────── вывод ─────────────────────────

os.system("")  # включает ANSI-цвета в консоли Windows
G, R, Y, C, D = "\033[92m", "\033[91m", "\033[93m", "\033[96m", "\033[2m"
E = "\033[0m"
MODE_RU = {"rfi":"ПРЕФЛОП / открытие", "facing":"ПРЕФЛОП / против рейза",
           "bet":"ПОСТФЛОП / против ставки", "free":"ПОСТФЛОП / чек-бет"}

stats = Counter(); stats_ok = Counter()

def show(sit, res, show_prompt):
    m = sit["mode"]
    print(f"\n{'='*72}\n{C}■ {MODE_RU[m]}{E}"
          + ("" if sit["trained"] else f"  {Y}(вне обучения){E}"))
    for k, v in sit["facts"]:
        print(f"   {k:<12} {v}")
    if show_prompt:
        print(f"\n{D}--- system ---\n{sit['system']}\n--- user ---\n{sit['user']}{E}")

    g_act, g_amt = sit["gold"]
    gold_s = g_act + (f" {g_amt:g}bb" if g_amt else "")

    if res is None:
        print(f"\n   {Y}движок (без модели){E}: {gold_s}")
        return

    raw, ms = res
    act, amount, thought, raw_act = parse_model(raw)
    ok = act == g_act
    stats[m] += 1; stats_ok[m] += ok
    mark = f"{G}✓{E}" if ok else f"{R}✗{E}"
    model_s = (act or "?") + (f" {amount:g}bb" if amount else "")

    print(f"\n   {D}thought:{E} {thought or '(пусто)'}")
    print(f"   {D}модель:{E}    {model_s}   {D}({ms} мс, ответ: «{raw_act}»){E}")
    print(f"   {D}движок:{E}    {gold_s}")
    print(f"   {mark} {'совпадает' if ok else 'РАСХОЖДЕНИЕ'}")

def print_stats():
    if not stats: return
    print(f"\n{C}── статистика ──{E}")
    for m in ("bet","facing","rfi","free"):
        if stats[m]:
            print(f"   {MODE_RU[m]:<28} {stats_ok[m]}/{stats[m]} = {stats_ok[m]/stats[m]*100:.0f}%")
    total, ok = sum(stats.values()), sum(stats_ok.values())
    print(f"   {'ВСЕГО':<28} {ok}/{total} = {ok/total*100:.0f}%")

# ───────────────────────── главный цикл ─────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=0, help="прогнать N раздач без интерактива")
    ap.add_argument("--no-model", action="store_true")
    a = ap.parse_args()

    generate = None if a.no_model else load_model(a.adapter, a.model)

    def run_one(gen_fn=None):
        sit = gen_fn() if gen_fn else gen_random()
        res = None
        if generate is not None:
            try:
                res = generate(sit["system"], sit["user"])
                print(f"res:\n{res}")
            except Exception as e:
                print(f"{R}ошибка модели: {e}{E}")
        show(sit, res, SHOW_PROMPT)

    SHOW_PROMPT = False

    if a.n:
        for _ in range(a.n):
            run_one()
        print_stats()
        return

    print(f"\n{C}Команды:{E} Enter — случайная раздача | r — открытие | v — против рейза | "
          f"b — против ставки | c — чек/бет | t — показать/скрыть промпт | s — статистика | q — выход\n")
    while True:
        try:
            cmd = input(f"\n{Y}>{E} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if cmd == "q": break
        elif cmd == "s": print_stats()
        elif cmd == "t":
            SHOW_PROMPT = not SHOW_PROMPT
            print(f"промпт: {'показывается' if SHOW_PROMPT else 'скрыт'}")
        elif cmd in GENERATORS: run_one(GENERATORS[cmd][0])
        elif cmd == "": run_one()
        else: print("не понял команду")

    print_stats()

if __name__ == "__main__":
    main()