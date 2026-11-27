# server.py — покерный движок + модель + HTTP-сервер (без FastAPI, только stdlib)
# запуск: python ./dataset_21_11/server_qwen.py --adapter D:\py2026\poker2\dataset_21_11\qwen25-decisions-lora
#         python ./dataset_21_11/server_qwen.py --no-model      (только движок, мгновенно, для проверки)
#         python ./dataset_21_11/server_qwen.py --adapter D:\py2026\poker2\dataset_21_11\qwen25-decisions-lora --iters 300     (итераций Monte Carlo)

import argparse, json, random, re, threading, time, itertools
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import os
os.environ["TORCH_LOGS"] = "-all"
import logging
logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)
from treys import Card, Evaluator
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

HERE = Path(__file__).resolve().parent
EV = Evaluator()

RANKS = "23456789TJQKA"
RV = {r: i + 2 for i, r in enumerate(RANKS)}
SU = {"♠": "s", "♥": "h", "♦": "d", "♣": "c", "s": "s", "h": "h", "d": "d", "c": "c"}
SR = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
FULL = [Card.new(r + s) for r in RANKS for s in "shdc"]

NAME_RU = {1: "стрит-флеш", 2: "каре", 3: "фулл-хаус", 4: "флеш",
           5: "стрит", 6: "сет", 7: "две пары", 8: "пара", 9: "старшая карта"}

# ─────────────────────────── КАРТЫ ───────────────────────────

def parse_card(t):
    t = str(t).strip().upper().replace("10", "T")
    m = re.match(r"^([2-9TJQKA])\s*([♠♥♦♣SHDC])$", t)
    if not m:
        raise ValueError(f"не понял карту: {t!r}")
    r = m.group(1)
    sraw = m.group(2)
    s = SU.get(sraw) or SU[sraw.lower()]
    return {"r": RV[r], "s": s, "t": Card.new(r + s),
            "disp": ("10" if r == "T" else r) + SR[s]}

def parse_cards(seq):
    if isinstance(seq, str):
        seq = seq.replace(",", " ").split()
    return [parse_card(x) for x in seq if str(x).strip()]

def fmt(cs):
    return " ".join(c["disp"] for c in cs)

# ─────────────────── КАТЕГОРИИ РУКИ (как в обучении) ───────────────────

def categorize(hero, mode):
    r1, r2 = sorted([hero[0]["r"], hero[1]["r"]], reverse=True)
    suited = hero[0]["s"] == hero[1]["s"]
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

# ─────────────────── ЧАРТЫ (для фактов и локального фоллбека) ───────────────────

CHART_RFI = {
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
    "Trash": [],
}

RULES = {
    "Premium": {"UTG": {"MP":"3","CO":"3","BTN":"3","SB":"3","BB":"3"}, "MP": {"CO":"3","BTN":"3","SB":"3","BB":"3"}, "CO": {"BTN":"3","SB":"3","BB":"3"}, "BTN": {"SB":"3","BB":"3"}},
    "Strong":  {"UTG": {"MP":"C","CO":"C","BTN":"C","SB":"F","BB":"C"}, "MP": {"CO":"C","BTN":"C","SB":"F","BB":"C"}, "CO": {"BTN":"3","SB":"3","BB":"3"}, "BTN": {"SB":"3","BB":"3"}},
    "Medium Pairs": {"UTG": {"MP":"F","CO":"C","BTN":"C","SB":"F","BB":"F"}, "MP": {"CO":"C","BTN":"C","SB":"F","BB":"F"}, "CO": {"BTN":"C","SB":"F","BB":"C"}, "BTN": {"SB":"F","BB":"C"}},
    "Small Pairs": {"UTG": {"MP":"F","CO":"C","BTN":"C","SB":"F","BB":"F"}, "MP": {"CO":"F","BTN":"C","SB":"F","BB":"F"}, "CO": {"BTN":"C","SB":"F","BB":"C"}, "BTN": {"SB":"F","BB":"C"}},
    "Suited Broadways": {"UTG": {"MP":"F","CO":"C","BTN":"C","SB":"F","BB":"C"}, "MP": {"CO":"F","BTN":"C","SB":"F","BB":"C"}, "CO": {"BTN":"C","SB":"F","BB":"C"}, "BTN": {"SB":"F","BB":"C"}},
    "Offsuit Broadways": {"UTG": {"MP":"F","CO":"F","BTN":"F","SB":"F","BB":"F"}, "MP": {"CO":"F","BTN":"F","SB":"F","BB":"F"}, "CO": {"BTN":"C","SB":"F","BB":"F"}, "BTN": {"SB":"F","BB":"F"}},
    "Suited Connectors": {"UTG": {"MP":"F","CO":"F","BTN":"F","SB":"F","BB":"F"}, "MP": {"CO":"F","BTN":"F","SB":"F","BB":"F"}, "CO": {"BTN":"C","SB":"F","BB":"F"}, "BTN": {"SB":"F","BB":"F"}},
    "Suited Ax": {"UTG": {"MP":"F","CO":"F","BTN":"F","SB":"F","BB":"F"}, "MP": {"CO":"F","BTN":"F","SB":"F","BB":"F"}, "CO": {"BTN":"F","SB":"F","BB":"F"}, "BTN": {"SB":"F","BB":"F"}},
    "Trash": {"UTG": {"MP":"F","CO":"F","BTN":"F","SB":"F","BB":"F"}, "MP": {"CO":"F","BTN":"F","SB":"F","BB":"F"}, "CO": {"BTN":"F","SB":"F","BB":"F"}, "BTN": {"SB":"F","BB":"F"}},
}

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

STYLE_BY_CODE = {"3": "RAISE", "C": "CALL", "F": "FOLD"}

# ─────────────────────────── МАТЕМАТИКА ───────────────────────────

def hand_name(hero, board):
    ranks = EV.evaluate([c["t"] for c in board], [c["t"] for c in hero])
    return NAME_RU[EV.get_rank_class(ranks)]

def equity_mc(hero, board, iters=300, opponents=1):
    ht = [c["t"] for c in hero]
    bt = [c["t"] for c in board]
    known = set(ht + bt)
    rest = [c for c in FULL if c not in known]
    need = 5 - len(bt)
    if need < 0:
        return 50.0
    win = tie = 0
    for _ in range(iters):
        s = random.sample(rest, need + 2 * opponents)
        fb = bt + s[:need]
        hs = EV.evaluate(fb, ht)
        best = None
        for k in range(opponents):
            v = s[need + 2 * k: need + 2 * k + 2]
            vs = EV.evaluate(fb, v)
            if best is None or vs < best:
                best = vs
        if hs < best:  win += 1
        elif hs == best: tie += 1
    return round((win + tie / 2) / iters * 100, 2)

def pot_odds(pot, to_call):
    if to_call <= 0:
        return 0.0
    return round(to_call / (pot + 2 * to_call) * 100, 2)

# ─────────────────────────── ПРОМПТЫ (как в датасетах) ───────────────────────────

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

STREET_RU = {"flop": "флоп", "turn": "тёрн", "river": "ривер"}


def build_prompt(st):
    hero = parse_cards(st["hero"])
    board = parse_cards(st.get("board") or [])
    street = st["street"]
    bb = max(1e-9, float(st.get("bb") or 10))
    pot = float(st.get("pot") or 0)
    to_call = float(st.get("toCall") or 0)
    hero_pos = st.get("heroPos") or "BTN"
    raiser = (st.get("raiserPos") or "").strip()
    opponents = max(1, min(3, int(st.get("opponents") or 1)))
    iters = int(st.get("iters") or ARGS.iters)

    facts = {"hero": fmt(hero), "board": fmt(board), "bb": bb}

    # ── ПРЕФЛОП: банк не открыт ──
    if street == "preflop" and not raiser:
        cat = categorize(hero, "rfi")
        facts.update(mode="rfi", category=cat, position=hero_pos)
        system = SYS_RFI
        user = (f"NLH 6-max cash. Твоя позиция: {hero_pos}.\n"
                f"Твоя рука: {fmt(hero)}.\n"
                f"Факты от движка: Категория руки = {cat}.\n"
                f"Действие до тебя: Все сбросили (Unopened pot).\n"
                f"Твоё решение?")
        fb = ("RAISE", 2.5) if hero_pos in CHART_RFI.get(cat, []) else ("FOLD", 0)
        return system, user, facts, fb

    # ── ПРЕФЛОП: против рейза ──
    if street == "preflop":
        cat = categorize(hero, "facing")
        code = RULES.get(cat, {}).get(raiser, {}).get(hero_pos)
        if code is None:
            code = "3" if cat == "Premium" else "F"
        style = STYLES.get((cat, code), "Недостаточно эквити")
        facts.update(mode="facing", category=cat, style=style,
                     raiser=raiser, position=hero_pos, engine_code=code)
        system = SYS_FACING
        user = (f"NLH 6-max cash. Эффективный стек: 100bb.\n"
                f"Действие: Игрок на {raiser} сделал рейз 2.5bb. Все сбросили до тебя.\n"
                f"Твоя позиция: {hero_pos}.\n"
                f"Твоя рука: {fmt(hero)}.\n"
                f"Факты от движка: Категория = {cat}, Стиль розыгрыша = {style}.\n"
                f"Твоё решение?")
        return system, user, facts, (STYLE_BY_CODE[code], 9 if code == "3" else 2.5)

    # ── ПОСТФЛОП ──
    if len(board) < 3:
        raise ValueError("на постфлопе нужно минимум 3 карты борда")
    combo = hand_name(hero, board)
    eq = equity_mc(hero, board, iters, opponents)
    odds = pot_odds(pot, to_call)
    facts.update(combo=combo, equity=eq, pot_odds=odds, opponents=opponents, iters=iters)

    pot_bb, bet_bb = pot / bb, to_call / bb

    if to_call > 0:
        facts.update(mode="bet")
        line = f"Банк {pot_bb:g}bb. Оппонент ставит {bet_bb:g}bb. Колл = {bet_bb:g}bb."
        system = SYS_BET
        user = (f"NLH cash. Ты на {hero_pos}, оппонент на {raiser or 'BTN'}. Улица: {STREET_RU[street]}.\n"
                f"Твоя рука: {fmt(hero)}. Доска: {fmt(board)}.\n{line}\n"
                f"Факты (посчитаны движком): комбинация — {combo}; эквити = {eq:.2f}%; "
                f"пот-оддсы = {odds:.2f}%; имплайд-бонус = 0пп.\nТвоё решение?")
        fb = ("CALL", bet_bb) if eq >= odds else ("FOLD", 0)
    else:
        facts.update(mode="free")
        system = SYS_FREE
        user = (f"NLH cash. Ты на {hero_pos}. Улица: {STREET_RU[street]}.\n"
                f"Твоя рука: {fmt(hero)}. Доска: {fmt(board)}.\n"
                f"Банк {pot_bb:g}bb. Ставки перед тобой нет.\n"
                f"Факты (посчитаны движком): комбинация — {combo}; эквити = {eq:.2f}%.\n"
                f"Твоё решение?")
        fb = ("BET", round(pot_bb * 0.5, 1)) if eq >= 55 else ("CHECK", 0)

    return system, user, facts, fb

# ─────────────────────────── ПАРСЕР ОТВЕТА ───────────────────────────

def parse_model(text):
    th = re.search(r"<thought>\s*([\s\S]*?)\s*</thought>", text, re.I)
    ac = (re.search(r"<action>\s*([\s\S]*?)\s*</action>", text, re.I)
          or re.search(r"<definition>\s*([\s\S]*?)\s*</definition>", text, re.I))
    thought = th.group(1).strip() if th else ""
    raw_act = ac.group(1).strip() if ac else text.strip()[:120]
    a = raw_act.lower()

    act = None
    if "фолд" in a or "fold" in a:                                    act = "FOLD"
    elif any(k in a for k in ("3-бет", "3бет", "3bet", "рейз", "raise")): act = "RAISE"
    elif "колл" in a or "call" in a:                                 act = "CALL"
    elif "чек" in a or "check" in a:                                 act = "CHECK"
    elif "бет" in a or "bet" in a:                                   act = "BET"

    num = re.search(r"(\d+(?:[.,]\d+)?)", raw_act.replace("$", ""))
    amount = float(num.group(1).replace(",", ".")) if num else 0.0
    return act, amount, thought, raw_act


def to_game(act, amount, st, fallback):
    """Приводит ответ модели (bb / русские слова) к легальному действию в долларах."""
    to_call = float(st.get("toCall") or 0)
    bb = max(1e-9, float(st.get("bb") or 10))
    stack = float(st.get("stack") or 0)
    mn = float(st.get("minTo") or (to_call * 2 or bb * 2))
    mx = float(st.get("maxTo") or stack)

    if act is None:
        act, amount = fallback

    if to_call > 0:
        if act == "CHECK": act = "FOLD"
        if act == "BET":   act = "RAISE"
    else:
        if act == "CALL":  act = "CHECK"
        if act == "FOLD":  act = "CHECK"
        if act == "RAISE": act = "BET"

    if act in ("RAISE", "BET"):
        if 0 < amount <= 50:          # модель ответила в bb
            amount *= bb
        amount = round(amount)
        if amount < mn: amount = round(mn)
        if amount > mx: amount = round(mx)
        if amount <= to_call:
            act, amount = ("CALL", to_call) if to_call > 0 else ("CHECK", 0)
        return act, amount

    if act == "CALL":
        return "CALL", to_call
    return act, 0

# ─────────────────────────── МОДЕЛЬ ───────────────────────────

MODEL = None
TOK = None
LOCK = threading.Lock()

def load_model(adapter):
    global MODEL, TOK
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria

    name = ARGS.model
    tok_src = adapter if adapter and Path(adapter).exists() else name
    TOK = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)

    kw = dict(device_map={"": 0}, attn_implementation="sdpa", trust_remote_code=True)
    try:
        MODEL = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, **kw)
    except TypeError:
        MODEL = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, **kw)

    if adapter and Path(adapter).exists():
        from peft import PeftModel
        MODEL = PeftModel.from_pretrained(MODEL, adapter, token=HF_TOKEN)
        print(f"[MODEL] адаптер: {adapter}")
    else:
        print("[MODEL] базовая модель без адаптера")

    MODEL.eval()
    if TOK.pad_token is None:
        TOK.pad_token = TOK.eos_token

    class StopAtAction(StoppingCriteria):
        def __call__(self, input_ids, scores, **kw):
            tail = TOK.decode(input_ids[0][-12:], skip_special_tokens=True)
            return "</action>" in tail or "</definition>" in tail

    STOP = StopAtAction()

    def generate(system, user):
        text = TOK.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
        inp = TOK(text, return_tensors="pt").to(MODEL.device)
        with torch.no_grad():
            out = MODEL.generate(**inp, max_new_tokens=180, do_sample=False,
                                 pad_token_id=TOK.eos_token_id, stopping_criteria=[STOP])
        try:
            inp_tokens = inp["input_ids"].shape[1]
            new_tokens = out.shape[1] - inp["input_ids"].shape[1]
            out_tokens = out.shape[1]
            print(f"Ответ: out_tokens: {out_tokens}, inp_tokens: {inp_tokens}, new_tokens: {new_tokens}")
        except TypeError:
            print("err")
        r=TOK.decode(out[0][inp["input_ids"].shape[1]:])
        print(f"Ответ: TOK.decode(out[0][inp['input_ids'].shape[1]:]:\n{r}")
        print("="*80)
        return TOK.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)

    return generate

GENERATE = None

# ─────────────────────────── HTTP ───────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            f = HERE / "index.html"
            if not f.exists():
                return self._send(404, {"error": "index.html не найден рядом с server.py"})
            return self._send(200, f.read_bytes(), "text/html; charset=utf-8")
        
        if self.path in ("/poker_client.html"):
            f = HERE / "poker_client.html"
            if not f.exists():
                return self._send(404, {"error": "poker_client не найден рядом с server.py"})
            return self._send(200, f.read_bytes(), "text/html; charset=utf-8")
        
        if self.path in ("/index____.html"):
            f = HERE / "index____.html"
            if not f.exists():
                return self._send(404, {"error": "index____ не найден рядом с server.py"})
            return self._send(200, f.read_bytes(), "text/html; charset=utf-8")
        
        if self.path == "/api/health" or "/api/decide/stats":
            return self._send(200, {"ok": True, "model": None if MODEL is None else ARGS.model,
                                    "adapter": ARGS.adapter, "no_model": ARGS.no_model})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        print("ok!")
        if self.path != "/api/decide":
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            st = json.loads(self.rfile.read(n).decode("utf-8"))
            system, user, facts, fallback = build_prompt(st)

            raw, thought, raw_act, act, amount, source = "", "", "", None, 0, "local"
            t0 = time.time()

            if GENERATE is not None:
                try:
                    with LOCK:
                        raw = GENERATE(system, user)
                    act, amount, thought, raw_act = parse_model(raw)
                    if act is not None:
                        source = "neuro"
                except Exception as e:
                    raw = f"ERROR: {e}"

            final_act, final_amt = to_game(act, amount, st, fallback)
            return self._send(200, {
                "ok": True, "mode": facts.get("mode"), "facts": facts,
                "system": system, "user": user,
                "thought": thought, "action_raw": raw_act, "raw": raw,
                "action": final_act, "amount": final_amt,
                "source": source, "ms": round((time.time() - t0) * 1000)
            })
        except Exception as e:
            print("error!")
            return self._send(200, {"ok": False, "error": f"{type(e).__name__}: {e}"})


def main():
    global ARGS, GENERATE
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="D:\\py2026\poker2\\dataset_21_11\\qwen25-decisions-lora", help="папка LoRA-адаптера D:\py2026\poker2\dataset_21_11\qwen25-decisions-lora")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--iters", type=int, default=300, help="итераций Monte Carlo")
    ap.add_argument("--no-model", action="store_true", help="только движок, без нейросети")
    ARGS = ap.parse_args()

    if ARGS.no_model:
        print("[MODEL] отключена (--no-model), работает только движок")
    else:
        import torch
        print(f"[CUDA] {torch.cuda.is_available()} | arch: {torch.cuda.get_arch_list()}")
        GENERATE = load_model(ARGS.adapter)
        print("[MODEL] готова")

    srv = ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler)
    print(f"\n>>> открой в браузере: http://127.0.0.1:{ARGS.port}\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("exit")

if __name__ == "__main__":
    main()