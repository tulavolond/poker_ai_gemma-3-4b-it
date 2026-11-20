"""
deep_server_v4.py — покерный нейро-сервер с online-LoRA.
Поддерживает Gemma-3 (gated, 4-bit, chat-template) и Qwen2.5 (raw-промпт) — профиль выбирается автоматически.
Протокол для deep_poker.html:  POST /  |  GET /stats  |  PUT /   (+ /save /buffer /memory /pretrain)
Запуск:  python deep_server_v4.py
"""
import os, re, json, time, threading, traceback
from urllib.parse import urlparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from transformers import BitsAndBytesConfig
    BNB_OK = True
except Exception:
    BitsAndBytesConfig, BNB_OK = None, False
try:
    from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
    PEFT_OK = True
except Exception:
    PEFT_OK = False
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

def _env(k, d): return os.getenv(k, d)
def _envb(k, d): return str(os.getenv(k, str(d))).lower() in ("1", "true", "yes", "on")
def _envi(k, d): return int(os.getenv(k, str(d)))

# ================= КОНФИГ =================
MODEL_ID   = _env("POKER_MODEL", "google/gemma-3-4b-it")
HF_TOKEN   = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
LORA_PATH  = _env("POKER_LORA", "./gemma/poker_gemma_lora")
DATA_DIR   = _env("POKER_DATA_DIR", "./gemma")
MEMORY_FILE = os.path.join(DATA_DIR, "poker_memory.json")
BUFFER_FILE = os.path.join(DATA_DIR, "training_buffer.json")
STATS_FILE  = os.path.join(DATA_DIR, "training_stats.json")
HOST = _env("POKER_HOST", "0.0.0.0")
PORT = _envi("POKER_PORT", 8899)

def detect_family(mid):
    m = mid.lower()
    if "gemma" in m: return "gemma"
    if "qwen" in m:  return "qwen"
    if "llama" in m or "mistral" in m: return "llama"
    return "other"
FAMILY = detect_family(MODEL_ID)

# профиль под семейство (всё переопределяется через env)
_D = {
 "gemma": dict(chat=True,  instr=True,  sample=True,  quant="4bit", seq=2048, train_seq=1500, new=192),
 "qwen":  dict(chat=False, instr=False, sample=False, quant="none", seq=788,  train_seq=788,  new=150),
 "llama": dict(chat=True,  instr=False, sample=False, quant="4bit", seq=2048, train_seq=1500, new=192),
 "other": dict(chat=True,  instr=False, sample=False, quant="auto", seq=1536, train_seq=1280, new=192),
}[FAMILY]

USE_CHAT   = _envb("POKER_CHAT", _D["chat"])     # ЕДИНЫЙ формат для train и infer
USE_INSTR  = _envb("POKER_INSTR", _D["instr"])
DO_SAMPLE  = _envb("POKER_SAMPLE", _D["sample"])
TEMPERATURE= float(_env("POKER_TEMP", "0.7"))
TOP_P      = float(_env("POKER_TOP_P", "0.9"))
REP_PEN    = float(_env("POKER_REP", "1.15"))
QUANT      = _env("POKER_QUANT", _D["quant"]).lower()
MAX_SEQ_LEN   = _envi("POKER_SEQ", _D["seq"])
TRAIN_SEQ_LEN = _envi("POKER_TRAIN_SEQ", _D["train_seq"])
MAX_NEW       = _envi("POKER_NEW", _D["new"])
MEMORY_LIMIT  = _envi("POKER_BUFFER", 30 if FAMILY == "gemma" else 100)
MAX_BATCH     = _envi("POKER_MAX_BATCH", 64)
LR            = float(_env("POKER_LR", "1e-5"))
TRAIN_LOSSES  = _envb("POKER_TRAIN_LOSSES", False)
GRAD_CHECKPOINT = _envb("POKER_GRAD_CP", True)
CREATE_LORA   = _envb("POKER_CREATE_LORA", True)
FORCE_ADAPTER = _envb("POKER_FORCE_ADAPTER", False)
SELFTEST      = _envb("POKER_SELFTEST", True)

os.makedirs(DATA_DIR, exist_ok=True)

# ================= СОСТОЯНИЕ =================
model = tokenizer = None
LORA_AVAILABLE = False
QUANT_ACTIVE = False
MODEL_DTYPE = AMP_DTYPE = None
STOP_IDS, BAD_IDS = [], []
END_TURN_TOKEN = None
training_buffer, memory_cache = [], []
last_loss = 0.0
stats = {"pretrain_examples": 0, "online_examples": 0, "batches_trained": 0,
         "preflop_queries": 0, "postflop_queries": 0, "post_requests": 0,
         "put_requests": 0, "skipped_losses": 0, "train_errors": 0,
         "bad_format": 0, "rejected_503": 0, "truncated": 0}
DATA_LOCK  = threading.Lock()
MODEL_LOCK = threading.RLock()
TRAIN_EVENT, STOP_EVENT = threading.Event(), threading.Event()
trainer = {"running": False, "last_result": None, "reported": True,
           "started": time.time(), "last_train_ts": 0.0}
pretrain_status = {"running": False, "progress": 0, "message": "Не запущен"}

# ================= PERSISTENCE =================
def _finite(o):
    if isinstance(o, float):
        return 0.0 if (o != o or o in (float("inf"), float("-inf"))) else round(o, 6)
    if isinstance(o, dict): return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list): return [_finite(v) for v in o]
    return o

def jdump(o): return json.dumps(_finite(o), ensure_ascii=False, allow_nan=False)

def _atomic_write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_finite(obj), f, ensure_ascii=False, indent=1, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def save_all_data():
    with DATA_LOCK:
        try:
            _atomic_write(MEMORY_FILE, memory_cache[-1000:])
            _atomic_write(BUFFER_FILE, training_buffer)
            _atomic_write(STATS_FILE, stats)
        except Exception as e:
            print(f"⚠️ Ошибка сохранения: {e}")

def load_all_data():
    global training_buffer, memory_cache
    for path, tgt in ((MEMORY_FILE, "m"), (BUFFER_FILE, "b"), (STATS_FILE, "s")):
        if not os.path.exists(path): continue
        try:
            with open(path, "r", encoding="utf-8") as f: data = json.load(f)
            if tgt == "m" and isinstance(data, list): memory_cache = data; print(f"📂 Память: {len(data)}")
            elif tgt == "b" and isinstance(data, list):
                training_buffer = [x for x in data if isinstance(x, dict) and "prompt" in x]
                print(f"📂 Буфер: {len(training_buffer)}")
            elif tgt == "s" and isinstance(data, dict): stats.update(data); print("📂 Статистика загружена")
        except Exception as e:
            print(f"⚠️ Не читается {path}: {e}")

# ================= УЛИЦА / ФОРМАТ =================
_STAGE_MAP = {"PREFLOP": "preflop", "PRE-FLOP": "preflop", "PRE FLOP": "preflop",
              "ПРЕФЛОП": "preflop", "ПРЕ-ФЛОП": "preflop", "FLOP": "flop", "ФЛОП": "flop",
              "TURN": "turn", "ТЁРН": "turn", "ТЕРН": "turn", "RIVER": "river", "РИВЕР": "river"}

def detect_stage(text):
    p = (text or "").upper()
    m = re.search(r"\[\s*СТАДИЯ\s*:\s*([A-ZА-ЯЁ\- ]+?)\s*\]", p)
    if m and m.group(1).strip() in _STAGE_MAP: return _STAGE_MAP[m.group(1).strip()]
    m = re.search(r"СТА[ДЛ]ИЯ\s*:\s*([A-ZА-ЯЁ\- ]+)", p)   # СТАДИЯ + опечатка СТУДИЯ
    if m:
        for k, v in _STAGE_MAP.items():
            if m.group(1).strip().startswith(k): return v
    for g in (("ПРЕФЛОП", "PRE-FLOP", "PRE FLOP", "PREFLOP"), ("РИВЕР", "RIVER"),
              ("ТЁРН", "ТЕРН", "TURN"), ("ФЛОП", "FLOP")):
        if any(k in p for k in g): return _STAGE_MAP[g[0]]
    return "unknown"

STAGE_INSTRUCTIONS = {
 "preflop": "ВАЖНО: На префлопе оценивай только категорию руки (тир 1-8).\n- Тир 1-2: премиум (AA, KK, AKs) → рейз или 3-бет\n- Тир 3-4: играбельные (JJ, suited connectors) → колл/рейз из позиции\n- Тир 5-6: маргинальные → колл только дёшево\n- Тир 7-8: мусор → фолд",
 "flop": "ВАЖНО: На флопе оценивай попадания и дро:\n- Сет/две пары: рейз для вэлью\n- Пара с эквити > 50%: колл или чек\n- Флеш-дро (~35%) или OESD (~32%): колл с implied odds\n- Эквити < 30%: фолд",
 "turn": "ВАЖНО: На тёрне считай pot odds = ставка / (банк + ставка).\nЕсли эквити > pot odds + 5% → колл. Дро с 1 картой ~20%. Сильная рука → рейз 2-3x.",
 "river": "ВАЖНО: На ривере только вэлью или блеф.\nЭквити > 65% → колл/рейз. Эквити < 25% → фолд. Никогда не коллируй со старшей картой.",
}
FEW_SHOT = "\n\nПРИМЕР ПРАВИЛЬНОГО ОТВЕТА:\nАНАЛИЗ: У меня пара тузов, эквити 78%, вэлью-бет.\nДЕЙСТВИЕ: RAISE\nСУММА: 50\n"

_ACT_MAP = (("FOLD", ("FOLD", "ФОЛД", "ПАС", "СБРОС")), ("CHECK", ("CHECK", "ЧЕК", "ПРОПУСК")),
            ("CALL", ("CALL", "КОЛЛ", "УРАВН")), ("RAISE", ("RAISE", "РЕЙЗ", "БЕТ", "BET", "ОЛЛ", "ALL-IN", "ALLIN")))

def parse_action(a):
    s = str(a or "").upper(); word = "FOLD"
    for canon, keys in _ACT_MAP:
        if any(k in s for k in keys): word = canon; break
    nums = re.findall(r"\d[\d\s]*", s)
    return word, (re.sub(r"\s", "", nums[0]) if nums else "0")

def fix_response(text):
    t = text or ""
    for tok in ("<end_of_turn>", "<start_of_turn>", "<eos>", "<bos>", "<pad>",
                "<|im_end|>", "<|im_start|>"):
        t = t.replace(tok, "\n")
    t = re.sub(r"^\s*(model|assistant|user|system)\s*$", "", t, flags=re.M).strip()
    a = re.search(r"АНАЛИЗ\s*:\s*([^\n]+)", t, re.I)
    d = re.search(r"ДЕЙСТВИЕ\s*:\s*([^\n]+)", t, re.I)
    s = re.search(r"СУММА\s*:\s*\$?\s*(\d[\d\s]*)", t, re.I)
    analysis = (a.group(1).strip() if a else "Не уверен, играю осторожно.")[:220]
    if not d:
        stats["bad_format"] += 1
        return "АНАЛИЗ: Не уверен, играю осторожно.\nДЕЙСТВИЕ: FOLD\nСУММА: 0", "FOLD", 0, analysis
    act, amt = parse_action(d.group(1))
    amount = int(re.sub(r"\s", "", s.group(1))) if s else (int(amt) if amt.isdigit() else 0)
    if act != "RAISE": amount = 0
    return f"АНАЛИЗ: {analysis}\nДЕЙСТВИЕ: {act}\nСУММА: {amount}", act, amount, analysis

def build_body(user_text, stage):
    """ЕДИНАЯ сборка тела промпта — используется и в инференсе, и в обучении."""
    body = (user_text or "").strip()
    if not body.startswith("[СТАДИЯ:"):
        body = f"[СТАДИЯ: {stage.upper()}]\n{body}"
    if USE_INSTR:
        body += "\n\n" + STAGE_INSTRUCTIONS.get(stage, "") + FEW_SHOT
    return body

def render_prompt(body):
    if USE_CHAT:
        return tokenizer.apply_chat_template([{"role": "user", "content": body}],
                                             tokenize=False, add_generation_prompt=True)
    return body

def ensure_bos(text):
    bos = getattr(tokenizer, "bos_token", None)
    if bos and not text.startswith(bos): return bos + text
    return text

def encode_prompt(text, limit):
    """BOS всегда остаётся первым токеном; при переполнении режем СЕРЕДИНУ, хвост сохраняем."""
    ids = tokenizer(ensure_bos(text), add_special_tokens=False)["input_ids"]
    if limit and len(ids) > limit:
        stats["truncated"] += 1
        bos = getattr(tokenizer, "bos_token", None)
        head = ids[:1] if bos else []
        ids = head + ids[len(ids) - (limit - len(head)):]
    return ids

def build_train_prompt(d):
    stage = str(d.get("stage", "flop")).lower()
    extra = ""
    if d.get("min_raise") is not None: extra += f"\nМИН. РЕЙЗ ДО: ${d['min_raise']}"
    if d.get("max_bet") is not None:   extra += f"\nМАКС. СТАВКА: ${d['max_bet']}"
    if d.get("position"):              extra += f"\nПОЗИЦИЯ: {d['position']}"
    if d.get("players"):               extra += f"\nИГРОКОВ В РАЗДАЧЕ: {d['players']}"
    if d.get("history"):               extra += f"\nДЕЙСТВИЯ НА УЛИЦЕ: {d['history']}"
    return (f"[СТАДИЯ: {stage.upper()}]\nСТАДИЯ: {stage.upper()}\n"
            f"ТВОИ КАРТЫ: {d.get('my_cards','?')}\nБОРД: {d.get('board','—')}\n"
            f"БАНК: ${d.get('pot',0)}, СТЕК: ${d.get('stack',0)}\n"
            f"НУЖНО КОЛЛИРОВАТЬ: ${d.get('to_call',0)}{extra}\n\n"
            f"Ответь СТРОГО:\nАНАЛИЗ: [короткое объяснение]\n"
            f"ДЕЙСТВИЕ: [FOLD/CHECK/CALL/RAISE]\nСУММА: [число]")

# ================= ЗАГРУЗКА МОДЕЛИ =================
def adapter_matches(path):
    cfg = os.path.join(path, "adapter_config.json")
    if not os.path.exists(cfg): return True, "нет adapter_config.json"
    try:
        base = json.load(open(cfg, encoding="utf-8")).get("base_model_name_or_path", "")
        norm = lambda s: str(s).lower().rstrip("/").split("/")[-1]
        if norm(base) == norm(MODEL_ID): return True, base
        return False, base
    except Exception as e:
        return True, f"не читается: {e}"

def load_model():
    global model, tokenizer, LORA_AVAILABLE, QUANT_ACTIVE, MODEL_DTYPE, AMP_DTYPE
    global STOP_IDS, BAD_IDS, END_TURN_TOKEN
    cuda = torch.cuda.is_available()
    AMP_DTYPE = (torch.bfloat16 if (cuda and torch.cuda.is_bf16_supported()) else torch.float16) if cuda else torch.float32
    MODEL_DTYPE = AMP_DTYPE
    print("=" * 64)
    print(f"  ЗАГРУЗКА: {MODEL_ID} | семейство={FAMILY} | cuda={cuda}")
    print(f"  формат: chat={USE_CHAT} instr={USE_INSTR} sample={DO_SAMPLE} quant={QUANT}")
    print("=" * 64)
    if not HF_TOKEN and "gemma" in MODEL_ID.lower():
        print("❗ HF_TOKEN не задан, а Gemma — gated-репозиторий. Добавь HF_TOKEN в .env, иначе 401.")
    tk = dict(token=HF_TOKEN) if HF_TOKEN else {}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="right", **tk)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    quant_cfg = None
    q = QUANT
    if q == "auto": q = "4bit" if cuda else "none"
    if q in ("4bit", "8bit") and not cuda:
        print("⚠️ CUDA нет — квантование отключено"); q = "none"
    if q in ("4bit", "8bit") and not BNB_OK:
        print("⚠️ bitsandbytes недоступен — квантование отключено"); q = "none"
    if q == "4bit":
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                       bnb_4bit_compute_dtype=AMP_DTYPE, bnb_4bit_use_double_quant=True)
    elif q == "8bit":
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
    QUANT_ACTIVE = quant_cfg is not None
    kw = dict(device_map=({"": 0} if cuda else "cpu"), low_cpu_mem_usage=True, **tk)
    if quant_cfg is not None: kw["quantization_config"] = quant_cfg
    try:
        try:
            model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=MODEL_DTYPE, **kw)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=MODEL_DTYPE, **kw)
    except ValueError as e:
        print(f"⚠️ AutoModelForCausalLM не подошёл ({e}); пробую AutoModelForImageTextToText")
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=MODEL_DTYPE, **kw)
    model.eval()
    if QUANT_ACTIVE and PEFT_OK:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=GRAD_CHECKPOINT)
    if hasattr(model, "enable_input_require_grads"):
        try: model.enable_input_require_grads()
        except Exception: pass
    if PEFT_OK and os.path.exists(LORA_PATH):
        ok, base = adapter_matches(LORA_PATH)
        if not ok and not FORCE_ADAPTER:
            print(f"❌ LoRA в {LORA_PATH} обучена на ДРУГОЙ базе: {base} != {MODEL_ID}")
            print("   Адаптер НЕ загружен (иначе мусор/краш). Укажи POKER_LORA от этой модели")
            print("   или создай новый: POKER_CREATE_LORA=1 и удали/переименуй старую папку.")
        else:
            try:
                model = PeftModel.from_pretrained(model, LORA_PATH, is_trainable=True)
                LORA_AVAILABLE = True
                print(f"✅ LoRA загружена: {LORA_PATH} (base={base})")
            except Exception as e:
                print(f"⚠️ LoRA не загрузилась: {e}")
    if PEFT_OK and not LORA_AVAILABLE and CREATE_LORA:
        try:
            targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            cfg = LoraConfig(r=16, lora_alpha=32, target_modules=targets, lora_dropout=0.05,
                             bias="none", task_type="CAUSAL_LM")
            model = get_peft_model(model, cfg)
            LORA_AVAILABLE = True
            print("🆕 Создан новый LoRA-адаптер")
        except Exception as e:
            print(f"⚠️ Не удалось создать LoRA: {e}")
    for n, p in model.named_parameters():
        p.requires_grad = ("lora_" in n.lower())
    model.eval()
    eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    unk = tokenizer.unk_token_id
    END_TURN_TOKEN = "<end_of_turn>" if (eot is not None and eot >= 0 and eot != unk) else None
    stop = [tokenizer.eos_token_id]
    if END_TURN_TOKEN: stop.append(eot)
    STOP_IDS = sorted({i for i in stop if i is not None and i >= 0})
    BAD_IDS = sorted({i for i in (tokenizer.pad_token_id, tokenizer.unk_token_id) if i is not None and i >= 0})
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = STOP_IDS      # НЕ затираем <end_of_turn>
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Устройство: {model.device} | dtype={MODEL_DTYPE} | quant={'да' if QUANT_ACTIVE else 'нет'}")
    print(f"🎓 Обучаемых параметров: {n_train/1e6:.2f}M | STOP_IDS={STOP_IDS} | BAD_IDS={BAD_IDS}")
    print(f"📏 seq={MAX_SEQ_LEN} train_seq={TRAIN_SEQ_LEN} | буфер={MEMORY_LIMIT}")
    if cuda: print(f"🧠 Память GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    if n_train == 0: print("❗ Обучаемых параметров нет — online-обучение отключено.")
    return n_train

def trainable_params():
    return [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n.lower()]

def _autocast():
    if torch.cuda.is_available() and AMP_DTYPE in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=AMP_DTYPE)
    from contextlib import nullcontext
    return nullcontext()

def _set_train_mode(on):
    try:
        if on and GRAD_CHECKPOINT and hasattr(model, "gradient_checkpointing_enable"):
            try: model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError: model.gradient_checkpointing_enable()
        elif hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
    except Exception: pass
    if hasattr(model, "config"): model.config.use_cache = not on

@torch.no_grad()
def run_inference(prompt_text, max_tokens=None, stage_hint=None):
    stage = stage_hint or detect_stage(prompt_text)
    if stage not in ("preflop", "flop", "turn", "river"): stage = "flop"
    text = render_prompt(build_body(prompt_text, stage))
    ids = encode_prompt(text, MAX_SEQ_LEN)
    inp = torch.tensor([ids], dtype=torch.long, device=model.device)
    att = torch.ones_like(inp)
    n_new = min(int(max_tokens or MAX_NEW), 512)
    gen_kw = dict(max_new_tokens=n_new, pad_token_id=tokenizer.pad_token_id,
                  eos_token_id=STOP_IDS or tokenizer.eos_token_id,
                  repetition_penalty=REP_PEN, attention_mask=att)
    if DO_SAMPLE: gen_kw.update(do_sample=True, temperature=TEMPERATURE, top_p=TOP_P)
    else:         gen_kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    with MODEL_LOCK:
        _set_train_mode(False)
        with _autocast():
            out = model.generate(inp, **gen_kw)
    gen = out[0][inp.shape[1]:]
    return fix_response(tokenizer.decode(gen, skip_special_tokens=True).strip())

# ================= ОБУЧЕНИЕ =================
def _encode_pair(prompt_body, completion):
    pre = render_prompt(prompt_body)
    tail = (END_TURN_TOKEN + "\n") if END_TURN_TOKEN else (tokenizer.eos_token or "")
    full = pre + completion + tail
    pre_ids = tokenizer(ensure_bos(pre), add_special_tokens=False)["input_ids"]
    ids = tokenizer(ensure_bos(full), add_special_tokens=False)["input_ids"]
    if len(ids) > TRAIN_SEQ_LEN: ids = ids[:TRAIN_SEQ_LEN]
    n = 0
    while n < min(len(ids), len(pre_ids)) and ids[n] == pre_ids[n]: n += 1
    if n == 0: n = min(len(pre_ids), len(ids))
    return ids, n

def train_batch(batch):
    global last_loss
    params = trainable_params()
    if not params:
        return {"status": "error", "error": "нет обучаемых LoRA-параметров"}
    orig = [p.dtype for p in params]
    for p in params:
        p.requires_grad_(True)
        if p.dtype != torch.float32: p.data = p.data.float()
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
    opt.zero_grad(set_to_none=True)
    total, valid, skipped = 0.0, 0, 0
    denom = max(1, len(batch))
    print(f"\n🧠 ОБУЧЕНИЕ: {len(batch)} примеров | chat={USE_CHAT} instr={USE_INSTR} norm=/{denom}")
    try:
        with MODEL_LOCK:
            _set_train_mode(True)
            model.train()
            for i, item in enumerate(batch):
                try:
                    ids, n_prefix = _encode_pair(item["prompt"], item["completion"])
                    if len(ids) - n_prefix < 4: skipped += 1; continue
                    inp = torch.tensor([ids], dtype=torch.long, device=model.device)
                    labels = inp.clone(); labels[:, :n_prefix] = -100
                    with _autocast():
                        out = model(input_ids=inp, attention_mask=torch.ones_like(inp), labels=labels)
                    loss = out.loss
                    if not torch.isfinite(loss): skipped += 1; continue
                    (loss * float(item.get("reward", 1.0)) / denom).backward()
                    total += float(loss.item()); valid += 1
                    del out, loss
                except torch.cuda.OutOfMemoryError:
                    skipped += 1; opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                except Exception as e:
                    skipped += 1; print(f"⚠️ Пропуск {i}: {e}")
            if valid:
                norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                if torch.isfinite(norm):
                    opt.step(); last_loss = total / valid
                    stats["batches_trained"] += 1
                    threading.Thread(target=save_lora_weights, daemon=True).start()
                    print(f"✅ Loss: {last_loss:.4f} | valid={valid} skip={skipped} | grad_norm={float(norm):.3f}")
                else:
                    stats["train_errors"] += 1; print("❌ Нечисловой градиент — шаг пропущен")
            else:
                stats["train_errors"] += 1; print("❌ Нет валидных примеров — шаг пропущен")
    finally:
        opt.zero_grad(set_to_none=True)
        for p, dt in zip(params, orig):
            if p.dtype != dt: p.data = p.data.to(dt)
        model.eval()
        _set_train_mode(False)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {"status": "trained" if valid else "error", "loss": last_loss,
            "valid": valid, "skipped": skipped, "buffer_size": len(training_buffer)}

def save_lora_weights():
    try:
        with MODEL_LOCK:
            os.makedirs(LORA_PATH, exist_ok=True)
            model.save_pretrained(LORA_PATH)
            tokenizer.save_pretrained(LORA_PATH)
        print(f"💾 LoRA сохранены: {LORA_PATH}")
    except Exception as e:
        print(f"⚠️ Ошибка сохранения LoRA: {e}")

def trainer_loop():
    while not STOP_EVENT.is_set():
        TRAIN_EVENT.wait(0.5); TRAIN_EVENT.clear()
        if STOP_EVENT.is_set(): break
        with DATA_LOCK:
            if len(training_buffer) < MEMORY_LIMIT: continue
            batch = training_buffer[:MAX_BATCH]
            del training_buffer[:len(batch)]
        save_all_data()
        trainer["running"] = True; t0 = time.time()
        try: res = train_batch(batch)
        except Exception as e:
            traceback.print_exc(); stats["train_errors"] += 1
            res = {"status": "error", "error": str(e)}
        trainer.update(running=False, last_result=res, reported=False, last_train_ts=time.time())
        print(f"⏱ Батч за {time.time()-t0:.1f}s")
        save_all_data()

def selftest():
    print("\n" + "=" * 64); print("  САМОТЕСТ"); print("=" * 64)
    try:
        t0 = time.time()
        txt, act, amt, _ = run_inference("[СТАДИЯ: PREFLOP]\nСТАДИЯ: ПРЕФЛОП\nТВОИ КАРТЫ: A♠ A♥\n"
                                         "БОРД: —\nБАНК: $30, СТЕК: $970\nНУЖНО КОЛЛИРОВАТЬ: $20\n\n"
                                         "Ответь СТРОГО:\nАНАЛИЗ: [объяснение]\nДЕЙСТВИЕ: [FOLD/CHECK/CALL/RAISE]\nСУММА: [число]")
        print(txt)
        print(f"⏱ {time.time()-t0:.1f}s → {act} {amt}")
        print("✅ Модель отвечает" if act else "❌ Пустой ответ")
    except Exception as e:
        traceback.print_exc(); print(f"❌ Самотест упал: {e}")
    print("=" * 64 + "\n")

# ================= HTTP =================
class ModelHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 300
    server_version = "DeepPoker/4.0"

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Access-Control-Max-Age", "86400")

    def _json(self, code, obj, extra=None):
        data = jdump(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self._cors()
        for k, v in (extra or {}).items(): self.send_header(k, v)
        self.end_headers()
        try: self.wfile.write(data)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError): pass

    def _body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        if n <= 0: return {}
        return json.loads(self.rfile.read(n).decode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(204); self.send_header("Content-Length", "0"); self._cors(); self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/stats", "/health"):
            return self._json(200, {"status": "ok", "model": MODEL_ID, "family": FAMILY,
                "lora_available": LORA_AVAILABLE, "lora_active": LORA_AVAILABLE, "quantized": QUANT_ACTIVE,
                "buffer_size": len(training_buffer), "buffer_limit": MEMORY_LIMIT, "last_loss": last_loss,
                "device": str(model.device) if model else "—", "dtype": str(MODEL_DTYPE),
                "chat_template": USE_CHAT, "instructions": USE_INSTR, "training": trainer["running"],
                "train_losses": TRAIN_LOSSES, "pretrain_status": pretrain_status,
                "uptime_s": int(time.time() - trainer["started"]), "stats": stats})
        if path == "/buffer":
            return self._json(200, {"buffer_size": len(training_buffer), "examples": training_buffer[:5]})
        if path == "/memory":
            return self._json(200, {"memory": memory_cache[-50:]})
        if path == "/save":
            save_all_data(); save_lora_weights()
            return self._json(200, {"status": "saved", "buffer_size": len(training_buffer)})
        if path == "/selftest":
            try:
                txt, act, amt, _ = run_inference("СТАДИЯ: FLOP\nТВОИ КАРТЫ: A♠ K♥\nБОРД: A♦ 7♣ 2♠\n"
                                                 "БАНК: $120, СТЕК: $880\nНУЖНО КОЛЛИРОВАТЬ: $40")
                return self._json(200, {"response": txt, "action": act, "amount": amt})
            except Exception as e:
                return self._json(500, {"error": str(e)})
        if path == "/pretrain":
            return self.handle_pretrain()
        return self._json(404, {"error": "not found"})

    def handle_pretrain(self):
        global pretrain_status
        if pretrain_status["running"]:
            return self._json(200, {"status": "already_running", "message": "Предобучение уже идёт"})
        try:
            from trainer import generate_training_examples
        except Exception as e:
            return self._json(501, {"error": f"trainer.generate_training_examples недоступен: {e}"})
        def run():
            global pretrain_status
            try:
                pretrain_status = {"running": True, "progress": 10, "message": "Генерация примеров..."}
                ex = generate_training_examples(1000)
                pretrain_status["progress"] = 60
                stats["pretrain_examples"] += len(ex)
                with DATA_LOCK: training_buffer.extend(ex)
                save_all_data()
                TRAIN_EVENT.set()
                pretrain_status = {"running": False, "progress": 100,
                                   "message": f"Добавлено {len(ex)} примеров, обучение в фоне"}
            except Exception as e:
                pretrain_status = {"running": False, "progress": -1, "message": f"Ошибка: {e}"}
        threading.Thread(target=run, daemon=True).start()
        return self._json(200, {"status": "started", "message": "Предобучение запущено в фоне"})

    def do_POST(self):
        try:
            body = self._body()
            prompt = body.get("prompt", "")
            if not prompt: return self._json(400, {"error": "empty prompt"})
            if trainer["running"]:
                stats["rejected_503"] += 1
                return self._json(503, {"error": "training in progress", "response": ""}, {"Retry-After": "10"})
            stats["post_requests"] += 1
            stage = detect_stage(prompt); t0 = time.time()
            text, act, amount, analysis = run_inference(prompt, body.get("max_tokens", MAX_NEW), stage)
            dt = time.time() - t0
            stats["preflop_queries" if stage == "preflop" else "postflop_queries"] += 1
            print(f"📥 {stage} | {len(prompt)} симв → {act} {amount} за {dt:.1f}s")
            return self._json(200, {"response": text, "action": act, "amount": amount,
                                    "analysis": analysis, "stage": stage, "elapsed": round(dt, 2)})
        except Exception as e:
            traceback.print_exc(); return self._json(500, {"error": str(e)})

    def do_PUT(self):
        try:
            body = self._body()
            if not body: return self._json(400, {"error": "empty body"})
            hand = {k: body.get(k, d) for k, d in (("stage", "?"), ("my_cards", "?"), ("board", "?"),
                    ("pot", 0), ("to_call", 0), ("stack", 0), ("action_taken", "?"), ("outcome", "?"), ("lesson", ""))}
            for k in ("min_raise", "max_bet", "position", "players", "history"):
                if body.get(k) is not None: hand[k] = body[k]
            with DATA_LOCK:
                memory_cache.append(hand); del memory_cache[:-1000]
            stats["online_examples"] += 1; stats["put_requests"] += 1
            won = "WIN" in str(hand["outcome"]).upper()
            stage = str(hand["stage"]).lower()
            if stage not in ("preflop", "flop", "turn", "river"): stage = detect_stage(build_train_prompt(hand))
            if won or TRAIN_LOSSES:
                act, amount = parse_action(hand["action_taken"])
                completion = f"АНАЛИЗ: {hand['lesson'] or 'Играю по ситуации.'}\nДЕЙСТВИЕ: {act}\nСУММА: {amount}"
                with DATA_LOCK:
                    training_buffer.append({"prompt": build_body(build_train_prompt(hand), stage),
                                            "completion": completion, "reward": 1.0 if won else 0.2,
                                            "stage": stage})
                    size = len(training_buffer)
            else:
                stats["skipped_losses"] += 1; size = len(training_buffer)
            save_all_data()
            print(f"📝 Буфер: {size}/{MEMORY_LIMIT} ({'WIN' if won else 'LOSS'})")
            if size >= MEMORY_LIMIT: TRAIN_EVENT.set()
            resp = {"status": "waiting", "buffer_size": size, "training": trainer["running"]}
            if not trainer["reported"] and trainer["last_result"]:
                r = trainer["last_result"]; trainer["reported"] = True
                resp["status"] = r.get("status", "trained"); resp["loss"] = r.get("loss", last_loss)
            elif trainer["running"]:
                resp["status"] = "training"
            return self._json(200, resp)
        except Exception as e:
            traceback.print_exc(); return self._json(500, {"error": str(e)})

    def log_message(self, fmt, *args):
        if args and "OPTIONS" in str(args[0]): return
        print("🌐 " + (fmt % args if args else fmt))

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

def serve():
    srv = ThreadedHTTPServer((HOST, PORT), ModelHandler)
    threading.Thread(target=trainer_loop, daemon=True, name="Trainer").start()
    print("\n" + "=" * 64); print("🚀 СЕРВЕР ЗАПУЩЕН"); print("=" * 64)
    print(f"📡 http://{HOST}:{PORT}   (в HTML укажи IP этой машины, не localhost, если играешь с другого устройства)")
    print(f"📊 /stats  💾 /save  📥 /buffer  🧪 /selftest  🎓 /pretrain")
    print(f"🧠 {MODEL_ID} | LoRA: {'✅' if LORA_AVAILABLE else '⚠️ нет'} | формат: {'chat' if USE_CHAT else 'raw'}")
    try: srv.serve_forever()
    except KeyboardInterrupt: print("\n🛑 Остановка...")
    finally:
        STOP_EVENT.set(); TRAIN_EVENT.set()
        save_all_data()
        if model is not None and LORA_AVAILABLE: save_lora_weights()
        srv.server_close(); print("✅ Остановлен, данные сохранены")

def main():
    print("\n📂 Загрузка данных...")
    load_all_data()
    load_model()
    if SELFTEST: selftest()
    serve()

if __name__ == "__main__":
    main()